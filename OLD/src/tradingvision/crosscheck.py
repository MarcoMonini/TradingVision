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


def _selfcheck() -> None:
    """The three readings are what they claim, and the comparison happens on shared bars only.

    The measurement this module exists for is a translation: the same predictions, scored against
    a different label. If `compare` ever scored each model on its own bars the drop from 0.38 to
    0.075 would be an artefact of the two datasets covering different times, not of the labels
    disagreeing — so the intersection is the part worth pinning down.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=300, freq="h", tz="UTC"), list("abcde")])
    x = pd.Series(rng.normal(size=len(idx)), index=idx)
    frame = pd.DataFrame({"x": x, "next_pivot": idx.get_level_values(0)})
    old = frame.assign(target=x + rng.normal(0, 0.2, len(idx)))
    # The predictive label keeps a fifth of the old one and is otherwise unrelated, which is the
    # shape of the real finding: not zero, but most of the agreement gone.
    new = frame.assign(target=0.2 * x + rng.normal(0, 1.0, len(idx)))

    out = compare(old, new, "2024-01-08")
    assert list(out.index) == ["old model -> old label", "old model -> new label", "new model -> new label"]
    assert out.loc["old model -> old label", "rank_ic"] > 0.8
    assert 0.05 < out.loc["old model -> new label", "rank_ic"] < 0.4, out

    # Half the new dataset's test bars removed: the readings move, and nothing raises on the
    # misaligned index — the three rows are still scored on one common set of bars.
    fewer = new[new.index.get_level_values(1).isin(list("abc"))]
    shared = compare(old, fewer, "2024-01-08")
    assert shared.shape == out.shape and not shared.isna().to_numpy().any(), shared


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
    _selfcheck()
    main()
