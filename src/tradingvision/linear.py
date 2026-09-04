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

LABELS = ["swing_leg_target", "next_pivot"]
CACHE = STORE / "step1.parquet"


def ols(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Least squares with an intercept. `lstsq` and not the normal equations: the branch columns
    are strongly collinear (the same indicator at four horizons) and it degrades gracefully."""
    a = np.column_stack([x.to_numpy(), np.ones(len(x))])
    return np.linalg.lstsq(a, y.to_numpy(), rcond=None)[0]


def predict(x: pd.DataFrame, coef: np.ndarray) -> pd.Series:
    return pd.Series(x.to_numpy() @ coef[:-1] + coef[-1], index=x.index)


def run(df: pd.DataFrame, cut: str) -> pd.DataFrame:
    """Fit on train, report the four signal metrics on both sides of the cut."""
    train, test = split.temporal(df, cut)
    cols = [c for c in df.columns if c not in LABELS]
    stats = normalize.fit(train[cols])  # train only: refitting on the whole frame is leakage
    x_train, x_test = normalize.apply(train[cols], stats), normalize.apply(test[cols], stats)

    coef = ols(x_train, train.swing_leg_target)
    rows = {
        "train": metrics.signal(predict(x_train, coef), train.swing_leg_target),
        "test": metrics.signal(predict(x_test, coef), test.swing_leg_target),
    }
    out = pd.DataFrame(rows).T
    out.insert(0, "bars", [len(train), len(test)])
    return out


def _selfcheck() -> None:
    """The estimator recovers a linear relation it is given — so that a flat Rank IC on the real
    data reads as "no signal" and not as "the fit is broken"."""
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=400, freq="h", tz="UTC"), list("abcde")])
    x = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    df = x.assign(
        swing_leg_target=x.a - 2 * x.b + rng.normal(0, 0.1, len(idx)),
        next_pivot=idx.get_level_values(0),
    )
    m = run(df, "2024-01-10")
    assert m.loc["test", "rank_ic"] > 0.9, m
    # Pure noise as a label: the fit finds nothing out of sample.
    noise = df.assign(swing_leg_target=rng.normal(size=len(idx)))
    assert abs(run(noise, "2024-01-10").loc["test", "rank_ic"]) < 0.1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--cut", default="2025-06", help="train before, test after")
    ap.add_argument("--stride", type=int, default=12, help="5m bars between samples, 12 = hourly")
    ap.add_argument("--cache", type=Path, default=CACHE, help="built dataset, reused when present")
    args = ap.parse_args()

    _selfcheck()
    if args.cache.exists():
        df = pd.read_parquet(args.cache)
    else:
        df = build(start=args.start, stride=args.stride)
        df.to_parquet(args.cache)
    print(f"{len(df):,} rows, {df.index.get_level_values(1).nunique()} symbols, cut at {args.cut}\n")
    print(run(df, args.cut).round(4).to_string())


if __name__ == "__main__":
    main()
