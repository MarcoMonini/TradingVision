"""Step 1 of the validation sequence: ordinary least squares on the last candle of each branch.

This is not a model, it is a floor and a leakage alarm. A linear map of 112 point-in-time columns
onto the leg position cannot carry real signal, so the expected reading is a Rank IC around zero.
A materially positive one means something in the pipeline is showing the model its own label —
the branch alignment, the pivots, or the purging — and the number to debug is this one, before any
GRU is written.

Single temporal cut rather than walk-forward: the spec keeps the single split for the fast checks
of step 1, and reserves walk-forward for the comparisons between architectures.

    uv run python -m tradingvision.linear --start 2023 --cut 2025-06
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import metrics, normalize, split
from tradingvision.data.binance import STORE
from tradingvision.dataset import build

# The columns that are not model input: the label itself and what purging needs.
META = ["target", "next_pivot"]
CACHE = STORE / "step1.parquet"


def ols(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Least squares with an intercept. `lstsq` and not the normal equations: the branch columns
    are strongly collinear (the same indicator at four horizons) and it degrades gracefully."""
    a = np.column_stack([x.to_numpy(), np.ones(len(x))])
    return np.linalg.lstsq(a, y.to_numpy(), rcond=None)[0]


def predict(x: pd.DataFrame, coef: np.ndarray) -> pd.Series:
    return pd.Series(x.to_numpy() @ coef[:-1] + coef[-1], index=x.index)


def fit_predict(df: pd.DataFrame, cut: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Fit on the purged train side of `cut`, predict on both sides."""
    train, test = split.temporal(df, cut)
    cols = [c for c in df.columns if c not in META]
    stats = normalize.fit(train[cols])  # train only: refitting on the whole frame is leakage
    x_train, x_test = normalize.apply(train[cols], stats), normalize.apply(test[cols], stats)

    coef = ols(x_train, train.target)
    return train, test, predict(x_train, coef), predict(x_test, coef)


def report(train: pd.DataFrame, test: pd.DataFrame, p_train: pd.Series, p_test: pd.Series) -> pd.DataFrame:
    """The four signal metrics on both sides of the cut."""
    rows = {
        "train": metrics.signal(p_train, train.target),
        "test": metrics.signal(p_test, test.target),
    }
    out = pd.DataFrame(rows).T
    out.insert(0, "bars", [len(train), len(test)])
    return out


def run(df: pd.DataFrame, cut: str) -> pd.DataFrame:
    """Fit and report in one call, for the checks that do not need the fit itself."""
    return report(*fit_predict(df, cut))


# Bars to the next pivot, in 5m steps. The last bucket is open: the horizon is unbounded, p99 is
# 202 bars and the maximum measured 754.
HORIZON_BINS = [0, 6, 12, 24, 48, 96, 192, 10**6]


def by_horizon(test: pd.DataFrame, pred: pd.Series, bins: list[int] = HORIZON_BINS) -> pd.DataFrame:
    """Test Rank IC split by how far the next pivot is — the discriminator between the two
    readings of a high step-1 IC.

    A structural overlap between the label and the causal features (the target is a position
    between two extrema, `close_position_in_window` is the same formula on a past window) sits at
    every distance. Residual leakage does not: an alignment or purging fault shows up on the bars
    whose pivot is imminent, because those are the ones whose label is nearly fixed by prices the
    features can already see. A flat profile clears the pipeline; a spike on the near buckets is a
    bug to find.
    """
    horizon = (test.next_pivot - test.index.get_level_values(0)) // pd.Timedelta(minutes=5)
    bucket = pd.cut(horizon, bins, right=False)
    rows = {}
    for b, rows_in in test.groupby(bucket, observed=True).groups.items():
        per_date = metrics.by_date(pred.loc[rows_in], test.target.loc[rows_in], rank=True).dropna()
        rows[str(b)] = {"bars": len(rows_in), "dates": len(per_date), "rank_ic": per_date.mean()}
    return pd.DataFrame(rows).T


def _selfcheck() -> None:
    """The estimator recovers a linear relation it is given — so that a flat Rank IC on the real
    data reads as "no signal" and not as "the fit is broken"."""
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=400, freq="h", tz="UTC"), list("abcde")])
    x = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    df = x.assign(
        target=x.a - 2 * x.b + rng.normal(0, 0.1, len(idx)),
        next_pivot=idx.get_level_values(0),
    )
    m = run(df, "2024-01-10")
    assert m.loc["test", "rank_ic"] > 0.9, m
    # Pure noise as a label: the fit finds nothing out of sample.
    noise = df.assign(target=rng.normal(size=len(idx)))
    assert abs(run(noise, "2024-01-10").loc["test", "rank_ic"]) < 0.1

    # The horizon split keeps every test bar and puts each one in the bucket of its own distance.
    far = df.assign(next_pivot=idx.get_level_values(0) + pd.Timedelta(hours=3))
    _, far_test, _, far_pred = fit_predict(far, "2024-01-10")
    h = by_horizon(far_test, far_pred, bins=[0, 12, 24, 10**6])
    assert h.bars.sum() == len(far_test)
    assert h.index.tolist() == ["[24, 1000000)"], h  # 3h at a 12-bar hour is 36 bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--cut", default="2025-06", help="train before, test after")
    ap.add_argument("--stride", type=int, default=12, help="5m bars between samples, 12 = hourly")
    ap.add_argument("--cache", type=Path, default=CACHE, help="built dataset, reused when present")
    ap.add_argument("--horizon", action="store_true", help="also split the test Rank IC by distance to the next pivot")
    args = ap.parse_args()

    _selfcheck()
    if args.cache.exists():
        df = pd.read_parquet(args.cache)
    else:
        df = build(start=args.start, stride=args.stride)
        df.to_parquet(args.cache)
    print(f"{len(df):,} rows, {df.index.get_level_values(1).nunique()} symbols, cut at {args.cut}\n")
    train, test, p_train, p_test = fit_predict(df, args.cut)  # one fit, both reports
    print(report(train, test, p_train, p_test).round(4).to_string())
    if args.horizon:
        print("\ntest Rank IC by bars to the next pivot\n")
        print(by_horizon(test, p_test).round(4).to_string())


if __name__ == "__main__":
    main()
