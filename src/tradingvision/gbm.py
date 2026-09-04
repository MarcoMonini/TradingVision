"""Step 2 of the validation sequence: gradient boosting on the flattened branches.

A GBM is a sequence of decision trees, each fitted to the residual the sum of the previous ones
leaves behind. It is the baseline here for four reasons: on tabular financial data at this size it
beats neural nets more often than not, it trains in minutes on CPU against hours on GPU, it
reports a per-column importance the GRU cannot, and it needs no scaling — so it runs before the
normalisation rule is even settled. `normalize` is deliberately not imported.

What it has to beat. Step 1 measured Rank IC 0.124 with an ordinary least squares on a single
candle, so 0.02-0.03 is the threshold of existence, not of promotion. The number that matters is
not the aggregate anyway: split by distance to the next pivot, step 1's Rank IC is near zero
inside every band and *negative* under 24 bars — the momentum features say "continue" exactly
where the leg is about to turn. A model that lifts the aggregate and leaves that profile alone
has added nothing. `--horizon` is what says which of the two happened.

Walk-forward and not a single cut: the point of this step is a comparison against step 3, and two
numbers without a dispersion cannot be compared. Each fold trains on everything before its
boundary, purged, and tests on the slice that follows; the report is mean and standard deviation
across folds.

    uv run python -m tradingvision.gbm --horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tradingvision import linear, metrics, split
from tradingvision.data.binance import STORE
from tradingvision.dataset import cached

CACHE = STORE / "step2.parquet"

# Spec starting values. Nothing here is tuned: they are fixed, the result is measured, and then
# one parameter moves at a time. `huber` with alpha = 2.1 is the loss the spec settled on, and the
# delta was measured on this label — see `data.target`, it does not survive a change of horizon.
PARAMS = {
    "objective": "huber",
    "alpha": 2.1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # No built-in metric: early stopping runs on Rank IC alone. Stopping on the Huber loss would
    # optimise the level of the prediction, and every metric downstream reads only its ordering.
    "metric": "None",
    "verbosity": -1,
    "force_col_wise": True,
}
ROUNDS = 2000
# In boosting rounds, not epochs. The spec's patience of 10 is a GRU number, and an epoch moves
# the model much further than a tree at a learning rate of 0.05 does.
#
# The stopping round it reports is not a meaningful quantity, and the first run's spread of 6 to
# 76 trees across four folds is not instability worth chasing. Measured on all four, the
# validation Rank IC is flat to within 0.001-0.004 over every round from 6 to 100, so the argmax
# inside that plateau is decided by noise; stopping at the low end instead of the middle costs at
# most 0.0065 of test Rank IC against a fold-to-fold standard deviation of 0.015. Raising the
# patience does not help — on fold 2 the round-6 and round-60 scores differ by 6e-5. Past 100
# rounds the decline is real and consistent on every fold, on validation and test alike, which is
# what early stopping is here to catch.
#
# The reading that matters from that sweep is elsewhere: a *single* tree already scores a test
# Rank IC of 0.150 averaged over the folds, against 0.157 for the whole ensemble and 0.124 for
# step 1's least squares. Boosting adds 0.007 on top of one coarse partition. With 560 columns
# available, that is the shape of the result — not the tree count.
PATIENCE = 50
# Tail of each fold's train period held out for early stopping, purged like every other boundary.
VALID_FRACTION = 0.2


def columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in linear.META]


def _rank_ic(index: pd.MultiIndex, target: pd.Series):
    """A LightGBM `feval` closing over the validation index, because the metric is
    cross-sectional: the raw prediction vector alone does not say which rows share a timestamp."""

    def evaluate(preds: np.ndarray, _: lgb.Dataset) -> tuple[str, float, bool]:
        per_date = metrics.by_date(pd.Series(preds, index=index), target, rank=True).dropna()
        return "rank_ic", per_date.mean(), True  # higher is better

    return evaluate


def fit(train: pd.DataFrame, valid: pd.DataFrame, rounds: int = ROUNDS) -> lgb.Booster:
    """One booster, stopped when the validation Rank IC stops improving."""
    cols = columns(train)
    data = lgb.Dataset(train[cols], train.target)
    return lgb.train(
        PARAMS,
        data,
        rounds,
        valid_sets=[lgb.Dataset(valid[cols], valid.target, reference=data)],
        feval=_rank_ic(valid.index, valid.target),
        callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)],
    )


def fold(train: pd.DataFrame, test: pd.DataFrame, rounds: int = ROUNDS) -> tuple[lgb.Booster, pd.Series]:
    """Fit on one fold's train side and predict its test slice.

    The validation tail is cut out of the train side and purged against its own boundary — an
    unpurged valid would let early stopping pick the round that best reads labels it has seen.
    """
    inner, valid = split.temporal_fraction(train, VALID_FRACTION)
    model = fit(inner, valid, rounds)
    return model, pd.Series(model.predict(test[columns(test)]), index=test.index)


def base_feature(column: str) -> str:
    """`log_return_lag4_15m` -> `log_return`: the indicator behind one flattened column."""
    name = column.rpartition("_")[0]
    for suffix in ("_lag1", "_lag4", "_mean", "_std"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def run(df: pd.DataFrame, start: str, folds: int = 4, rounds: int = ROUNDS) -> dict[str, pd.DataFrame]:
    """Walk-forward over `folds` expanding folds, and everything the step is meant to report."""
    per_fold, gains, tests, preds = {}, [], [], []
    for i, (train, test) in enumerate(split.walk_forward(df, start, folds), 1):
        model, pred = fold(train, test, rounds)
        per_fold[f"fold {i}"] = {
            "train": len(train),
            "test": len(test),
            "trees": model.best_iteration,
            **metrics.signal(pred, test.target),
        }
        gains.append(pd.Series(model.feature_importance("gain"), index=model.feature_name()))
        tests.append(test)
        preds.append(pred)

    out = pd.DataFrame(per_fold).T
    # Mean and standard deviation over the folds — the reason for walk-forward in the first place.
    out.loc["mean"] = out.mean()
    out.loc["std"] = out.iloc[:-1].std()

    test, pred = pd.concat(tests), pd.concat(preds)
    gain = pd.concat(gains, axis=1).mean(axis=1)
    return {
        "folds": out,
        "importance": (gain.groupby(gain.index.map(base_feature)).sum() / gain.sum()).sort_values(ascending=False),
        "horizon": linear.by_horizon(test, pred),
        "market_beta": pd.DataFrame({"vs residual target": market_beta(pred, test.target)}).T,
    }


def market_beta(pred: pd.Series, target: pd.Series) -> dict[str, float]:
    """The same metrics against the target with its cross-sectional mean removed.

    The 20 USDT pairs move as a block, so a Rank IC confirmed on all of them may rest on far fewer
    degrees of freedom than it looks: one market phase repeated twenty times. If this holds up the
    model separates the symbols and the multi-asset input earns its place; if it collapses to zero
    the signal is the phase of the market as a whole, and the baseline to beat is a model trained
    on BTC alone. A diagnostic, not a preprocessing step — nothing downstream uses this residual.
    """
    return metrics.signal(pred, target - target.groupby(level=0).transform("mean"))


def _selfcheck() -> None:
    """The walk-forward wrapper recovers a relation it is given, and stays flat on noise — so a
    weak reading on the real data is the data talking and not the plumbing."""
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=600, freq="h", tz="UTC"), list("abcde")])
    # Named like real columns — every one of them carries its branch suffix, which is what
    # `base_feature` strips.
    x = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a_5m", "b_5m", "c_1h"])
    df = x.assign(
        target=x.a_5m * x.b_5m + rng.normal(0, 0.1, len(idx)),  # an interaction, which trees find and OLS does not
        next_pivot=idx.get_level_values(0),
    )
    out = run(df, "2024-01-15", folds=2, rounds=60)
    assert out["folds"].index.tolist() == ["fold 1", "fold 2", "mean", "std"]
    assert out["folds"].loc["mean", "rank_ic"] > 0.5, out["folds"]
    assert linear.run(df, "2024-01-15").loc["test", "rank_ic"] < 0.1, "OLS cannot see the interaction"

    noise = run(df.assign(target=rng.normal(size=len(idx))), "2024-01-15", folds=2, rounds=60)
    assert abs(noise["folds"].loc["mean", "rank_ic"]) < 0.1, noise["folds"]
    imp = out["importance"]  # ranked by gain, so the indicator the label ignores lands last
    assert np.isclose(imp.sum(), 1.0) and sorted(imp.index) == ["a", "b", "c"] and imp.index[-1] == "c"

    assert base_feature("log_return_lag4_15m") == "log_return"
    assert base_feature("age_of_window_high_std_1h") == "age_of_window_high"
    assert base_feature("ema_slope_5m") == "ema_slope"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--test-start", default="2025-06", help="first fold boundary; the default matches step 1's cut")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--stride", type=int, default=12, help="5m bars between samples, 12 = hourly")
    ap.add_argument("--cache", type=Path, default=CACHE, help="built dataset, reused when present")
    ap.add_argument("--horizon", action="store_true", help="also split the test Rank IC by distance to the next pivot")
    ap.add_argument("--top", type=int, default=15, help="how many indicators of the importance ranking to print")
    args = ap.parse_args()

    _selfcheck()
    # Half an hour to build and 600 MB on disk, so it is reused — with the parameters recorded
    # beside it, because a silently reused file built at another stride is an irreproducible metric.
    df = cached(args.cache, start=args.start, stride=args.stride, lags=True)
    print(f"{len(df):,} rows, {len(columns(df))} columns, {args.folds} folds from {args.test_start}\n")

    out = run(df, args.test_start, args.folds)
    print(out["folds"].round(4).to_string())
    print("\nimportance by indicator, share of total gain\n")
    print(out["importance"].head(args.top).round(4).to_string())
    print("\nmarket beta — the same metrics once the cross-sectional mean is removed\n")
    print(out["market_beta"].round(4).to_string())
    if args.horizon:
        print("\ntest Rank IC by bars to the next pivot, pooled over the folds\n")
        print(out["horizon"].round(4).to_string())


if __name__ == "__main__":
    main()
