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

from tradingvision import gbm, split
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--test-start", default="2025-06", help="everything before this is the train period")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--cache", type=Path, default=gbm.CACHE)
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


if __name__ == "__main__":
    main()
