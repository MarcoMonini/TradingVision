"""The experiment that decided the change of label.

`swing_leg_target` scored Rank IC 0.38 out of sample under an OLS on point-in-time features — a
number that reads as "the model knows it is at 0.9 of the leg, so it is time to sell". The
objection is fair: if that reading held, it would be the trading signal itself.

It does not hold, and this is the measurement that shows it. Train the same OLS on the old label,
then score its predictions against the *predictive* label on the same test bars. Both are the
same question asked twice — "how much of the leg is left" — so an actionable 0.38 would survive
the translation.

Measured on 20 symbols, 2023+, cut at 2025-06, 219,983 test bars:

    old model -> old label (retrospective)   rank_ic 0.3821   rank_icir 0.9501
    old model -> new label (predictive)      rank_ic 0.0753   rank_icir 0.1689
    new model -> new label (predictive)      rank_ic 0.1237   rank_icir 0.2802

Four fifths of the signal evaporates, and what is left is beaten by the same model trained on the
predictive label directly. So the old label was both an optimistic measurement and a worse way of
obtaining what it was measuring — but not empty: 0.075 is small, not zero.

    uv run python -m tradingvision.crosscheck
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tradingvision import linear, metrics
from tradingvision.data.binance import STORE
from tradingvision.data.target import remaining_excursion, swing_leg_target
from tradingvision.dataset import build

LABELS = {"retrospective": swing_leg_target, "predictive": remaining_excursion}


def cached(name: str, cache: Path, start: str, stride: int) -> pd.DataFrame:
    """The dataset labelled `name`, built once and reused."""
    if cache.exists():
        return pd.read_parquet(cache)
    df = build(start=start, stride=stride, label=LABELS[name])
    df.to_parquet(cache)
    return df


def compare(old: pd.DataFrame, new: pd.DataFrame, cut: str) -> pd.DataFrame:
    """The three readings, on the test bars the two datasets share."""
    _, test_old, _, pred_old = linear.fit_predict(old, cut)
    _, test_new, _, pred_new = linear.fit_predict(new, cut)
    at = pred_old.index.intersection(test_new.index)
    rows = {
        "old model -> old label": metrics.signal(pred_old[at], test_old.target[at]),
        "old model -> new label": metrics.signal(pred_old[at], test_new.target[at]),
        "new model -> new label": metrics.signal(pred_new[at], test_new.target[at]),
    }
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--cut", default="2025-06")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--old-cache", type=Path, default=STORE / "step1.parquet")
    ap.add_argument("--new-cache", type=Path, default=STORE / "step1_predictive.parquet")
    args = ap.parse_args()

    old = cached("retrospective", args.old_cache, args.start, args.stride)
    new = cached("predictive", args.new_cache, args.start, args.stride)
    # The old cache predates the rename of the label column and still carries the label's own name.
    old = old.rename(columns={"swing_leg_target": "target"})
    print(f"{len(old):,} rows, cut at {args.cut}\n")
    print(compare(old, new, args.cut).round(4).to_string())


if __name__ == "__main__":
    main()
