"""Step 3 of the validation sequence: one GRU over the 15m branch alone.

The question is narrow and the spec states it: does *the sequence* add signal over the single
candle? Step 2 gave the GBM the lags and the window statistics of 24 steps on four horizons and it
used none of them — permutation puts `ema_slope` at 0.0040 and `age_of_window_high` at 0.0018
against 0.1296 for one static position. Those were exactly the columns the flattening existed to
expose. The alternative reading — the information is there but lives in a dynamic that trees over
flattened columns do not compose — is the second hypothesis, not the first, and this is what puts
it to the test. `nearpivot` already established the prior half: on the band under 24 bars from the
pivot, 80 of 112 raw columns clear a measured noise floor, so the information is in the inputs.

One branch and not four. Step 4 is the multi-branch model, and it only earns its cost if it beats
this one; running the cheap architecture first is what makes that comparison mean anything.

What it has to beat: Rank IC 0.156 and Rank ICIR 0.58, the same four walk-forward folds from
2025-06 the GBM ran, *and* a visible improvement in the bands under 48 bars. The threshold is the
full-set number and not the 0.1531 of the reduced one: step 3 has to beat the best step 2
produced, not the pruned version whose input it inherits. Lifting the aggregate alone is the case
step 1 already declared useless — the GBM did exactly that, moving only the far bands.

The sample rows are step 2's own, read off its cache: same timestamps, same symbols, same stride,
same folds. Anything else and the comparison measures the sampling too.

    uv run python -m tradingvision.gru --horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from tradingvision import linear, metrics, normalize, split
from tradingvision.data.binance import STORE, load
from tradingvision.data.pivots import EXTREMA_WINDOW
from tradingvision.dataset import _shift
from tradingvision.features import SELECTED, features

BRANCH = "15m"
STEPS = 24
CACHE = STORE / "step2.parquet"  # read for its index, target and pivot horizon only
TENSOR = STORE / "step3-15m.npy"

# Spec starting values, table "Iperparametri". None of them is tuned: they are fixed, the result
# is measured, and then one parameter moves at a time.
H = 32
DROPOUT = 0.2  # on the branch output, before the head — with one branch there is no concat
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
BATCH = 512
GRAD_CLIP = 1.0
EPOCHS = 100
PATIENCE = 10  # on validation Rank IC, never on the loss
DELTA = 2.1  # Huber, measured on this label over the train period — see `data.target`
# Huber for the cross-sectional target, measured the same way — the median of |z| over the train
# period, 0.557. Delta lives in the units of the label, so it cannot be carried over or rescaled
# by eye when the label changes.
Z_DELTA = 0.56
VALID_FRACTION = 0.2  # the same tail the GBM holds out, purged against its own boundary
# Not in the spec, which names ReduceLROnPlateau without settling its shape. Half the early
# stopping patience, so the rate gets one chance to help before the run is stopped.
LR_PATIENCE, LR_FACTOR = 5, 0.5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sequences(index: pd.MultiIndex, keep: list[str] = SELECTED, tf: str = BRANCH, steps: int = STEPS) -> np.ndarray:
    """`(len(index), steps, len(keep))` — the last `steps` closed `tf` candles at each 5m row.

    Oldest step first. The alignment is `dataset`'s and cannot be got wrong here either: a branch
    label sits at `label + tf - 5m` on the 5m grid and is forward filled from there, so the 5m bar
    of 10:05 reads the 15m candle closed at 10:00 and never the one still forming. Step `k` back is
    then the same frame shifted `k` of its own bars, which is why this is 24 reindexes and not a
    rolling window: the 5m grid is not the branch's grid.
    """
    out = np.empty((len(index), steps, len(keep)), dtype="float32")
    pos = pd.Series(np.arange(len(index)), index=index)
    for symbol, rows in pos.groupby(level=1):
        when = rows.index.get_level_values(0)
        f = features(load(symbol, tf), EXTREMA_WINDOW)[keep]
        f.index = f.index + _shift(tf)
        at = rows.to_numpy()
        for k in range(steps):
            out[at, steps - 1 - k] = f.shift(k).reindex(when, method="ffill").to_numpy()
    if not np.isfinite(out).all():
        raise ValueError("the window reaches past the start of a symbol's history — widen the warm-up")
    return out


def cached_sequences(path: Path, index: pd.MultiIndex, **params) -> np.ndarray:
    """`sequences(index, **params)` on disk, with a stamp of what produced it — same contract as
    `dataset.cached`, and for the same reason: a reused tensor built at another stride or another
    feature set is a metric nobody can reproduce."""
    stamp = path.with_suffix(".json")
    written = dict(params, rows=len(index), first=str(index[0][0]), last=str(index[-1][0]))
    if path.exists():
        if not stamp.exists():
            raise SystemExit(f"{path} has no {stamp.name} recording how it was built — delete it and rebuild")
        if (was := json.loads(stamp.read_text())) != written:
            raise SystemExit(f"{path} was built with {was}, not {written} — delete it")
        return np.load(path)
    x = sequences(index, **params)
    np.save(path, x)
    stamp.write_text(json.dumps(written, indent=2, sort_keys=True))
    return x


def scaled(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """`normalize`'s transform, fitted on the train rows of this fold and applied to all of them.

    Fitted on the most recent step alone: every step of the window is drawn from the same series,
    so one step is a clean estimate of the column's train-period quartiles, and it keeps the fit
    to the same shape `normalize.fit` takes everywhere else. Per fold and not once for the run —
    a scaler fitted across the whole span reads quantiles of the test period.
    """
    # Named when the width says these are the selected indicators, so a degenerate column comes
    # back from `normalize.fit` with a name and not a position.
    names = SELECTED if x.shape[2] == len(SELECTED) else list(range(x.shape[2]))
    stats = normalize.fit(pd.DataFrame(x[train, -1], columns=names))
    center, scale = stats.center.to_numpy("float32"), stats.scale.to_numpy("float32")
    return (np.clip((x - center) / scale, -normalize.CLIP, normalize.CLIP) * normalize.SCALE).astype("float32")


def cross_section(y: pd.Series) -> pd.Series:
    """`y` standardised inside each timestamp — the level and the scale the metric cannot see.

    Rank IC is computed per date over the symbols, so adding a constant to a whole cross-section
    or multiplying it by a positive number leaves every rank untouched. Measured on the train
    period, that invisible part is 47% of the target's variance, and a pooled Huber spends its
    gradient on it. Removing it does not throw away signal — it throws away the half of the label
    the evaluation is blind to by construction.

    Dates with a single symbol have no dispersion and drop out (3.6% of the train rows); the
    metric already ignores them, needing three symbols to correlate anything.
    """
    g = y.groupby(level=0)
    return ((y - g.transform("mean")) / g.transform("std")).replace([np.inf, -np.inf], np.nan).dropna()


class Net(nn.Module):
    """GRU over the branch, dropout, linear head. No activation on the output.

    The tanh the spec removed used to saturate exactly on the pivots, where the gradient matters
    most; nothing downstream needs a bounded output, since Rank IC is invariant to any monotone
    transform. The head starts at zero bias so the first predictions sit at the centre of the
    target's distribution. GRU and not LSTM: at 24 steps the cell state buys nothing and costs a
    quarter of the parameters.
    """

    def __init__(self, n_features: int, hidden: int = H, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


def predict(model: Net, x: np.ndarray, rows: np.ndarray, index: pd.MultiIndex, batch: int = 4096) -> pd.Series:
    """Predictions for `rows`, on the (timestamp, symbol) index the metrics need."""
    model.eval()
    out = []
    with torch.no_grad():
        for at in np.array_split(rows, max(1, len(rows) // batch)):
            out.append(model(torch.from_numpy(x[at]).to(DEVICE)).cpu().numpy())
    return pd.Series(np.concatenate(out), index=index)


def fit(
    x: np.ndarray,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seed: int = 0,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH,
    delta: float = DELTA,
    quiet: bool = True,
) -> Net:
    """One model, stopped when the validation Rank IC stops improving.

    Model selection and early stopping read Rank IC and never the loss: the Huber anchors the
    scale of the prediction and every metric downstream reads only its ordering, so the round with
    the best loss is not the round with the best signal.
    """
    torch.manual_seed(seed)
    model = Net(x.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)
    loss_fn = nn.HuberLoss(delta=delta)

    at = train.row.to_numpy()
    y = torch.from_numpy(train.target.to_numpy("float32"))
    rng = np.random.default_rng(seed)
    best, best_state, since = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(at)), max(1, len(at) // batch_size)):
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(x[at[batch]]).to(DEVICE)), y[batch].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        pred = predict(model, x, valid.row.to_numpy(), valid.index)
        score = metrics.by_date(pred, valid.target, rank=True).dropna().mean()
        sched.step(score)
        if not quiet:
            print(f"    epoch {epoch + 1:3d}  valid rank ic {score:+.4f}{'  *' if score > best else ''}")
        if score > best:
            best, since = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def fold(
    x: np.ndarray, train: pd.DataFrame, test: pd.DataFrame, seed: int = 0, z_target: bool = False, **kw
) -> pd.Series:
    """Fit on one fold's train side and predict its test slice.

    The validation tail is cut out of the train side and purged against its own boundary, exactly
    as the GBM does it: an unpurged valid lets early stopping pick the epoch that best reads
    labels it has already seen.
    """
    inner, valid = split.temporal_fraction(train, VALID_FRACTION)
    z = scaled(x, inner.row.to_numpy())  # train only, and the inner train at that
    if z_target:
        # Training only. Validation keeps the raw label, so early stopping reads the metric the
        # result is judged on — which the transform leaves unchanged anyway.
        y = cross_section(inner.target)
        inner, kw = inner.loc[y.index].assign(target=y), dict(kw, delta=Z_DELTA)
    return predict(fit(z, inner, valid, seed, **kw), z, test.row.to_numpy(), test.index)


def run(x: np.ndarray, df: pd.DataFrame, start: str, folds: int = 4, seeds: int = 1, **kw) -> dict[str, pd.DataFrame]:
    """Walk-forward over `folds` expanding folds and `seeds` initialisations each."""
    per_fold, per_seed, tests, preds = {}, {}, [], []
    for i, (train, test) in enumerate(split.walk_forward(df, start, folds), 1):
        runs = [fold(x, train, test, seed, **kw) for seed in range(seeds)]
        for seed, pred in enumerate(runs):
            per_seed[(f"fold {i}", seed)] = metrics.signal(pred, test.target)
        pred = pd.concat(runs, axis=1).mean(axis=1) if seeds > 1 else runs[0]
        # Averaged over the seeds, which is what "media +- std sui seed" reports; the ensemble of
        # the predictions would be a different and better model, and not the one being measured.
        per_fold[f"fold {i}"] = {
            "train": len(train),
            "test": len(test),
            **pd.DataFrame([per_seed[(f"fold {i}", s)] for s in range(seeds)]).mean().to_dict(),
        }
        tests.append(test)
        preds.append(pred)

    out = pd.DataFrame(per_fold).T
    out.loc["mean"] = out.mean()
    out.loc["std"] = out.iloc[:-1].std()
    test, pred = pd.concat(tests), pd.concat(preds)
    return {
        "folds": out,
        "seeds": pd.DataFrame(per_seed).T,
        "pred": pred,
        "target": test.target,
        "horizon": linear.by_horizon(test, pred),
        "market_beta": pd.DataFrame({"vs residual target": _market_beta(pred, test.target)}).T,
    }


def smoothed(pred: pd.Series, k: int) -> pd.Series:
    """The last `k` predictions of each symbol, averaged — a low-pass on the output.

    The place a filter belongs. Smoothing the *input* buys nothing: a GRU over 24 steps can
    already learn any linear filter of the window, so a moving average upstream adds only its own
    lag — the reason every smoothed indicator left the selection. Downstream is different: the
    prediction carries the model's estimation noise on top of whatever signal it found, and that
    noise is what an average over neighbouring bars removes. Causal, so it stays honest — the
    value at `t` reads `t-k+1..t` and nothing later.
    """
    if k <= 1:
        return pred
    wide = pred.unstack(level=1).sort_index()
    return wide.rolling(k, min_periods=1).mean().stack().reorder_levels([0, 1]).reindex(pred.index)


def _market_beta(pred: pd.Series, target: pd.Series) -> dict[str, float]:
    """The same metrics against the target with its cross-sectional mean removed — step 2's
    diagnostic, repeated so the two steps are read the same way."""
    return metrics.signal(pred, target - target.groupby(level=0).transform("mean"))


def meta(path: Path = CACHE) -> pd.DataFrame:
    """Step 2's rows: the label, the purging horizon and each row's place in the tensor."""
    df = pd.read_parquet(path, columns=linear.META)
    return df.assign(row=np.arange(len(df)))


def _selfcheck() -> None:
    """The wrapper recovers a relation that lives in the *sequence* and stays flat on noise, so a
    weak reading on the real data is the data talking and not the plumbing."""
    n, steps, f = 5000, 6, 2
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=n // 5, freq="h", tz="UTC"), list("abcde")])
    x = rng.normal(size=(n, steps, f)).astype("float32")
    # The label is a move at the *start* of the window, gone from the last candle: uncorrelated
    # with the last step by construction, so only a model that carries the sequence can see it —
    # which is the whole claim of step 3 over step 1.
    y = x[:, 0, 0] - x[:, 1, 0] + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"target": y, "next_pivot": idx.get_level_values(0), "row": np.arange(n)}, index=idx)

    kw = dict(folds=2, epochs=60, patience=60, batch_size=128)
    out = run(x, df, "2024-01-25", **kw)
    assert out["folds"].index.tolist() == ["fold 1", "fold 2", "mean", "std"]
    assert out["folds"].loc["mean", "rank_ic"] > 0.5, out["folds"]
    blind = linear.run(df.assign(last=x[:, -1, 0]), "2024-01-25")
    assert blind.loc["test", "rank_ic"] < 0.15, "the last step alone must be blind to a drift"

    noise = run(x, df.assign(target=rng.normal(size=n)), "2024-01-25", **kw | dict(epochs=6, patience=6))
    assert abs(noise["folds"].loc["mean", "rank_ic"]) < 0.15, noise["folds"]

    # Scaling reads the train rows only, so a slice normalises to exactly what it does inside the
    # whole — the leakage check `normalize` makes on frames, made here on the tensor.
    train = np.arange(n // 2)
    z = scaled(x, train)
    assert np.allclose(z[:10], scaled(x[: n // 2], train[: n // 2])[:10])
    assert abs(z).max() <= normalize.CLIP * normalize.SCALE + 1e-6
    # Two seeds differ, which is the reason the spec reports mean +- std over them.
    a = fit(x, df.iloc[: n // 2], df.iloc[n // 2 :], seed=0, epochs=1, patience=1)
    b = fit(x, df.iloc[: n // 2], df.iloc[n // 2 :], seed=1, epochs=1, patience=1)
    assert not torch.allclose(a.head.weight, b.head.weight)

    # The filter averages over time inside a symbol and never across symbols, and k=1 is identity.
    one = out["pred"]
    assert smoothed(one, 1).equals(one)
    two = smoothed(one, 2)
    assert two.index.equals(one.index) and two.notna().all()
    first = one.xs("a", level=1).sort_index()
    assert np.isclose(two.xs("a", level=1).sort_index().iloc[1], first.iloc[:2].mean())
    assert np.isclose(two.xs("a", level=1).sort_index().iloc[0], first.iloc[0]), "no value before the first bar"

    # The cross-sectional target: standardised inside each date, and — the reason for the whole
    # change — scoring identically under the metric, which is what makes dropping it free.
    z = cross_section(df.target)
    per_date = z.groupby(level=0)
    assert np.allclose(per_date.mean(), 0, atol=1e-9) and np.allclose(per_date.std(), 1, atol=1e-9)
    assert np.isclose(
        metrics.signal(one, z.loc[one.index])["rank_ic"], metrics.signal(one, df.target.loc[one.index])["rank_ic"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2025-06", help="first fold boundary; the default matches step 2's")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1, help="initialisations per fold; the spec asks 5 in exploration")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--cache", type=Path, default=CACHE, help="step 2's dataset, read for its rows")
    ap.add_argument("--tensor", type=Path, default=TENSOR, help="the built sequences, reused when present")
    ap.add_argument("--horizon", action="store_true", help="also split the test Rank IC by distance to the next pivot")
    ap.add_argument("--verbose", action="store_true", help="print the validation Rank IC each epoch")
    ap.add_argument("--z-target", action="store_true", help="train on the target standardised inside each timestamp")
    ap.add_argument("--smooth", type=int, nargs="*", default=[2, 4, 12], help="output low-pass windows to report")
    args = ap.parse_args()

    _selfcheck()
    df = meta(args.cache)
    x = cached_sequences(args.tensor, df.index, keep=SELECTED, tf=BRANCH, steps=STEPS)
    print(f"{len(df):,} rows, {x.shape[1]} steps x {x.shape[2]} features of the {BRANCH} branch, on {DEVICE}")
    print(f"{args.folds} folds from {args.test_start}, {args.seeds} seed(s) each\n")

    out = run(
        x,
        df,
        args.test_start,
        args.folds,
        args.seeds,
        epochs=args.epochs,
        z_target=args.z_target,
        quiet=not args.verbose,
    )
    print(out["folds"].round(4).to_string())
    if args.seeds > 1:
        print("\nper seed\n")
        print(out["seeds"].round(4).to_string())
    if args.smooth:
        print("\nlow-pass on the prediction — Rank IC of the k-bar average, pooled over the folds\n")
        rows = {k: metrics.signal(smoothed(out["pred"], k), out["target"]) for k in [1] + args.smooth}
        print(pd.DataFrame(rows).T.rename_axis("k").round(4).to_string())
    print("\nmarket beta — the same metrics once the cross-sectional mean is removed\n")
    print(out["market_beta"].round(4).to_string())
    if args.horizon:
        print("\ntest Rank IC by bars to the next pivot, pooled over the folds\n")
        print(out["horizon"].round(4).to_string())


if __name__ == "__main__":
    main()
