"""The second half of step 2: from the 28 candidates to the set the final model trains on.

The GBM's own job was a reference IC. This is the other half of the same step — deciding which
columns survive — and the spec fixes the procedure in five passes: sanity, Spearman per branch,
hierarchical clustering at 0.2, permutation importance per cluster, ablation. Every one of them
runs on the train period only. Measuring a selection on the test slice is how a set of columns
that looks excellent out of sample and is worthless in production gets chosen.

    uv run python -m tradingvision.selection
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from tradingvision import dataset, features, gbm, split
from tradingvision.dataset import cached

# Share of a column that may be NaN or infinite before the column stops being usable at all. Not
# from the spec, which says "quota di NaN" without a number: a column that is 99% present carries
# its information, one that is 90% present is mostly the imputation rule talking.
MAX_UNUSABLE = 0.01


def sanity(x: pd.DataFrame, max_unusable: float = MAX_UNUSABLE) -> pd.DataFrame:
    """Pass 1 — NaN share, infinite tails and dispersion, one row per column.

    Column by column rather than frame-wide: a boolean mask over 640k x 560 is a third of a
    gigabyte and the reduction is per column anyway.

    An infinite value is counted apart from a NaN because the two say different things. A NaN is
    warm-up the dataset did not trim; an infinity is a division that reached zero — `np.log(c/x)`
    on a bar with no trades — and it is not something a downstream mean or scaler survives.
    """
    rows = {}
    for c in x.columns:
        v = x[c].to_numpy()
        nan, inf = np.isnan(v), np.isinf(v)
        usable = v[~(nan | inf)]
        rows[c] = {
            "nan": nan.mean(),
            "infinite": inf.mean(),
            "std": float(usable.std()) if usable.size else 0.0,
        }
    report = pd.DataFrame(rows).T
    report["keep"] = (report.nan + report.infinite <= max_unusable) & (report["std"] > 0)
    return report


def spearman(x: pd.DataFrame) -> pd.DataFrame:
    """Pearson on the ranks, which is what Spearman is — computed this way so scipy stays out of
    it, the same trick `metrics.by_date` uses."""
    return x.rank().corr()


def branch_columns(columns: list[str], tf: str) -> dict[str, str]:
    """The 28 value-at-t columns of one branch, indicator name -> column name.

    The lags and the window statistics are deliberately left out here: pass 2 asks which
    *indicators* say the same thing, and `log_return_lag1` correlating with `log_return` answers a
    question nobody asked. They come back in pass 4, where the whole group is permuted at once.
    """
    return {c[: -len(tf) - 1]: c for c in columns if c.endswith(f"_{tf}") and gbm.base_feature(c) == c[: -len(tf) - 1]}


def per_branch(train: pd.DataFrame, branches: tuple[str, ...] = dataset.BRANCHES) -> dict[str, pd.DataFrame]:
    """Pass 2 — one Spearman matrix per branch, indexed by indicator so the four are comparable."""
    out = {}
    for tf in branches:
        names = branch_columns(list(train.columns), tf)
        m = spearman(train[list(names.values())])
        out[tf] = m.set_axis(list(names), axis=0).set_axis(list(names), axis=1)
    return out


def mean_abs(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """|rho| averaged over the branches.

    The four matrices share an index, so this is one number per pair of indicators. It is what
    pass 3 clusters on, and the reason is pass 4: the permutation is done on all four branches at
    once, so the surviving set is a set of indicators and not a set of (indicator, branch) pairs.
    A pair that is redundant on 5m and independent on 1h lands in the middle and is kept.
    """
    return sum(m.abs() for m in matrices.values()) / len(matrices)


def redundant(rho: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """The pairs above the threshold, once each, strongest first."""
    upper = rho.where(np.triu(np.ones(rho.shape, dtype=bool), 1)).stack()
    return upper[upper >= threshold].sort_values(ascending=False)


# "la piu semplice e la meno laggata" is the spec's tie-break inside a cluster, and neither half of
# it is derivable from the code — so it is written down. The number is the effective lag in bars of
# the branch's own timeframe: how far back a value is still influenced by. 0 is the current candle
# alone, 24 is the full window N, and the smoothed indicators are counted at more than their window
# because a recursive filter keeps weight on bars from before it. Two readings are worth naming:
# `ema_slope` and `volume_trend` difference over N/4 but do it on statistics of the full window, so
# they belong with their window and not with the sub-window; KAMA and PSAR are adaptive, so their
# lag is not a constant at all and they sit above the plain smoothings. This table is a judgement
# and it decides survivors — it is the first thing to change if a cluster keeps the wrong column.
EFFECTIVE_LAG = {
    0: ("bar_range_pct", "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "close_position_in_bar"),
    1: ("log_return",),
    24: (
        "cum_log_return_window",
        "close_position_in_window",
        "distance_from_window_high_pct",
        "distance_from_window_low_pct",
        "age_of_window_high",
        "age_of_window_low",
        "window_range_pct",
        "realized_volatility",
        "volatility_expansion",
        "log_volume_vs_median",
        "signed_volume",
        "volume_trend",
        "on_balance_volume_zscore",
        "distance_from_vwap_pct",
    ),
    48: (  # N with one recursive smoothing on top
        "average_true_range_pct",
        "distance_from_ema_pct",
        "ema_slope",
        "adx_trend_strength",
        "rsi_centered",
    ),
    60: ("distance_from_kama_pct", "distance_from_psar_pct"),  # adaptive: the lag itself moves with the data
    72: ("tsi_momentum",),  # two smoothings stacked
}
# Ties on the lag break on the spec table's own order, which runs from the raw candle outwards.
SIMPLICITY = {name: (lag, features.COLUMNS.index(name)) for lag, names in EFFECTIVE_LAG.items() for name in names}
CUT = 0.2  # on distance = 1 - |rho|, so |rho| = 0.8


def clusters(rho: pd.DataFrame, cut: float = CUT) -> pd.Series:
    """Pass 3 — hierarchical clustering on 1 - |rho|, cut at `cut`. Indicator -> cluster id.

    Complete linkage, which the spec leaves open. Under single linkage a cluster only needs a
    chain of redundant pairs to hold together, so |rho| 0.85 from a to b and from b to c puts a
    and c in one group at |rho| 0.3 — and on this data every trend and momentum column would
    collapse into a single cluster through such a chain. Complete linkage asks that *every* pair
    inside a cluster clears the threshold, which is what "questo gruppo e ridondante" means.
    """
    d = 1 - rho.to_numpy().astype(float)
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), method="complete"), t=cut, criterion="distance")
    return pd.Series(labels, index=rho.index, name="cluster")


def representative(names: list[str]) -> str:
    """The survivor of one cluster: the least lagged, ties to the spec table's order."""
    return min(names, key=SIMPLICITY.__getitem__)


def survivors(labels: pd.Series) -> pd.Series:
    """One indicator per cluster, in the spec table's order. Cluster id -> the name that stays."""
    chosen = labels.groupby(labels).apply(lambda g: representative(list(g.index)))
    return chosen.sort_values(key=lambda s: s.map(SIMPLICITY.__getitem__))


def _selfcheck() -> None:
    n = 1000
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "good_5m": rng.normal(size=n),
            "constant_5m": np.ones(n),
            "mostly_nan_5m": np.where(np.arange(n) < 10, 1.0, np.nan),
            "some_inf_5m": np.where(np.arange(n) < 5, np.inf, rng.normal(size=n)),
            "many_inf_5m": np.where(np.arange(n) < 500, -np.inf, 1.0),
        }
    )
    report = sanity(x)
    assert report.keep.tolist() == [True, False, False, True, False], report
    # The dispersion of a column with a few infinities is still the dispersion of the rest of it,
    # not NaN — which is the whole reason the mask comes out before the std.
    assert 0.9 < report.loc["some_inf_5m", "std"] < 1.1, report
    assert report.loc["many_inf_5m", ["nan", "infinite"]].tolist() == [0.0, 0.5]

    # Spearman and not Pearson is the point of pass 2: a monotone but strongly curved relation is
    # a perfect 1 on the ranks and visibly less on the levels, and fat-tailed returns are exactly
    # where the two part company.
    a = pd.Series(rng.normal(size=n))
    curved = pd.DataFrame({"a_5m": a, "b_5m": np.exp(3 * a), "c_5m": rng.normal(size=n)})
    rho = spearman(curved)
    assert rho.loc["a_5m", "b_5m"] > 0.999 and curved.corr().loc["a_5m", "b_5m"] < 0.8
    assert abs(rho.loc["a_5m", "c_5m"]) < 0.1

    # Two branches, and the averaging keeps the pair labels aligned rather than re-sorting them.
    frame = curved.join(curved.rename(columns=lambda c: c.replace("_5m", "_1h")))
    mats = per_branch(frame, ("5m", "1h"))
    assert list(mats["1h"].index) == ["a", "b", "c"] and set(mats) == {"5m", "1h"}
    assert redundant(mean_abs(mats)).index.tolist() == [("a", "b")]

    assert set(SIMPLICITY) == set(features.COLUMNS), "every candidate needs a place in the tie-break"

    # Complete linkage, and the reason for it: a-b and b-c are redundant, a-c is not. Single
    # linkage would chain the three into one cluster on the strength of b alone.
    chain = pd.DataFrame(
        [[1.0, 0.85, 0.3], [0.85, 1.0, 0.85], [0.3, 0.85, 1.0]],
        index=["log_return", "candle_body_pct", "rsi_centered"],
        columns=["log_return", "candle_body_pct", "rsi_centered"],
    )
    labels = clusters(chain)
    assert labels.nunique() == 2 and labels.log_return == labels.candle_body_pct
    assert labels.rsi_centered != labels.log_return
    # Of the pair, the candle body survives: no lag at all against log_return's one bar.
    assert survivors(labels).tolist() == ["candle_body_pct", "rsi_centered"]
    assert representative(["tsi_momentum", "close_position_in_window"]) == "close_position_in_window"
    # Same effective lag, so the spec table's order decides — and it lists the body before the wick.
    assert representative(["upper_wick_pct", "candle_body_pct"]) == "candle_body_pct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--test-start", default="2025-06", help="everything before this is the train period")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--cache", type=Path, default=gbm.CACHE)
    ap.add_argument("--threshold", type=float, default=0.8, help="|rho| above which two indicators are redundant")
    ap.add_argument("--cut", type=float, default=CUT, help="clustering cut on distance = 1 - |rho|")
    args = ap.parse_args()

    _selfcheck()
    df = cached(args.cache, start=args.start, stride=args.stride, lags=True)
    train, _ = split.temporal(df, args.test_start)
    print(f"{len(train):,} train rows to {args.test_start}, {len(gbm.columns(train))} columns\n")

    report = sanity(train[gbm.columns(train)])
    dropped = report[~report.keep]
    print(f"pass 1 — sanity: {report.keep.sum()} of {len(report)} columns usable")
    if len(dropped):
        print("\ndropped\n")
        print(dropped.round(6).to_string())

    matrices = per_branch(train[report[report.keep].index.tolist()])
    rho = mean_abs(matrices)
    pairs = redundant(rho, args.threshold)
    print(f"\npass 2 — Spearman per branch: {len(pairs)} pairs at |rho| >= {args.threshold}\n")
    spread = pd.DataFrame({tf: m.abs().stack()[pairs.index] for tf, m in matrices.items()})
    print(spread.assign(mean=pairs).round(3).to_string())

    labels = clusters(rho, args.cut)
    kept = survivors(labels)
    print(f"\npass 3 — clustering at {args.cut}: {len(kept)} of {len(labels)} indicators survive\n")
    grouped = labels.groupby(labels).apply(lambda g: ", ".join(sorted(g.index, key=SIMPLICITY.__getitem__)))
    print(pd.DataFrame({"survivor": kept, "cluster": grouped}).to_string(index=False))


if __name__ == "__main__":
    main()
