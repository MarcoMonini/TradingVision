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

    uv run python -m tradingvision.gru --branches 5m,15m,30m,1h --features all --seeds 5

Step 4 did not earn it. On the same four folds and five seeds, Rank IC and Rank ICIR:

    15m alone            0.1582 +- 0.0150    0.5855 +- 0.0544
    four branches        0.1530 +- 0.0137    0.5649 +- 0.0439
    four, shared weights 0.1537 +- 0.0147    0.5664 +- 0.0496

Per fold the single branch wins all four, against both variants — the same standard by which step 3
was promoted over the GBM, reading the other way. The two step 4 architectures are 0.0007 apart
against 0.0140 between folds, which closes open point 5 as irrelevant rather than as a winner: the
shared encoder costs nothing and buys nothing, because the parameters were never the constraint.
`--branch-of` had already said why. `close_position_in_window` lives on the 15m branch and
permuting the other three costs 0.0027 of 0.1525, so the three extra encoders had nothing to find
and spent their capacity on noise. The bands near the pivot come out slightly worse, not better.

Which leaves the sequence, and not the horizon, as what step 3 bought — and open point 1 as the
problem it did not touch.

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

from tradingvision import dataset, linear, metrics, nearpivot, normalize, split, threshold
from tradingvision.data.binance import STORE, load
from tradingvision.data.pivots import EXTREMA_WINDOW
from tradingvision.data.target import cross_sectional_return, remaining_excursion, swing_leg_target
from tradingvision.features import COLUMNS, SELECTED, features
from tradingvision.oracle import FEE

BRANCH = "15m"  # step 3's single branch, and the default of `--branches`
BRANCHES = dataset.BRANCHES  # step 4: 5m -> 2h, 15m -> 6h, 30m -> 12h, 1h -> 24h
STEPS = 24
CACHE = STORE / "step2.parquet"  # read for its index, target and pivot horizon only
# One file per branch and feature set, `step3-15m-all.npy` and its kin. The prefix says which step
# first wrote it, not which step may read it: the rows, the steps and the alignment are the same
# for both, so step 4 reuses step 3's 15m tensor instead of spending 1.7 GB to rename it.
TENSOR = STORE / "step3.npy"
# One fitted model, for the chart app to draw. Written by `--save`, and never by a walk-forward
# run: a fold's model is a measurement, this is the one trained on the whole train period.
CHECKPOINT = STORE / "gru.pt"
# The full candidate set, kept as an option rather than a rebuild: the selection was made with a
# GBM on the aggregate, and the columns it dropped are not necessarily the ones a recurrent net
# reading the band cannot use. Measured inside 48 bars of the pivot, `tsi_momentum` and
# `rsi_centered` are the two strongest predictors there and both are among the sixteen it dropped.
FEATURES = {"selected": SELECTED, "all": COLUMNS}

# Spec starting values, table "Iperparametri". None of them is tuned: they are fixed, the result
# is measured, and then one parameter moves at a time.
H = 32
DROPOUT = 0.2  # on the branch outputs, before the head — with one branch there is no concat
# Width of the timeframe embedding of the shared encoder. Not in the spec, which names the variant
# without sizing it. Small on purpose: it has to tell four horizons apart and nothing else, and
# every dimension of it is a dimension the head reads instead of the hidden state.
EMBEDDING = 4
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-3
BATCH = 512
GRAD_CLIP = 1.0
EPOCHS = 100
PATIENCE = 10  # on validation Rank IC, never on the loss
DELTA = 2.1  # Huber, measured on the default label over the train period — see `data.target`
# The label to fit. `excursion` is the one the dataset carries and every measured number in the
# spec is taken on; `swing` is the retrospective label it replaced, kept selectable rather than
# deleted because the two are not comparable and reading one against the other is the point.
# Their Rank ICs live on different scales: `crosscheck` puts an OLS at 0.38 on `swing` and 0.12 on
# `excursion`, because most of `swing` is already known at t. A number on one is not a number on
# the other, and nothing here converts between them.
# `cross` is the third and reads the panel rather than one symbol, so `dataset` relabels it
# through its own function; the entry is here so the CLI can name it and this comment can sit next
# to the other two. Its scale is a third one again — a rank-free z inside each timestamp — and, like
# the other pair, no number taken on one converts to another.
LABELS = {"excursion": remaining_excursion, "swing": swing_leg_target, "cross": cross_sectional_return}
# Huber for the cross-sectional target, measured the same way — the median of |z| over the train
# period, 0.557. Delta lives in the units of the label, so it cannot be carried over or rescaled
# by eye when the label changes.
# One per target: the median of |label| over the train period, the criterion that fixed 2.1.
Z_DELTA = {"z": 0.56, "demean": 1.45}
VALID_FRACTION = 0.2  # the same tail the GBM holds out, purged against its own boundary
# Not in the spec, which names ReduceLROnPlateau without settling its shape. Half the early
# stopping patience, so the rate gets one chance to help before the run is stopped.
LR_PATIENCE, LR_FACTOR = 5, 0.5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def window(f: pd.DataFrame, when: pd.DatetimeIndex, steps: int = STEPS) -> np.ndarray:
    """`(len(when), steps, F)` — the last `steps` rows of `f` at or before each `when`, oldest first.

    `f` is already carried onto the 5m grid, so this is `steps` shifts of it read at `when` and
    never a rolling window: the two grids differ, and a branch contributes one row per *its own*
    bar. `method="ffill"` is what makes a row readable only from its own label onwards, which is
    the no-anticipation rule the whole dataset rests on.
    """
    return np.stack([f.shift(k).reindex(when, method="ffill").to_numpy() for k in reversed(range(steps))], axis=1)


def sequences(
    index: pd.MultiIndex,
    keep: list[str] = SELECTED,
    tf: str = BRANCH,
    steps: int = STEPS,
    rank: bool = False,
) -> np.ndarray:
    """`(len(index), steps, len(keep))` — the last `steps` closed `tf` candles at each 5m row.

    Oldest step first. The alignment is `dataset`'s and cannot be got wrong here either: a branch
    label sits at `label + tf - 5m` on the 5m grid and is forward filled from there, so the 5m bar
    of 10:05 reads the 15m candle closed at 10:00 and never the one still forming. Step `k` back is
    then the same frame shifted `k` of its own bars, which is why this is 24 reindexes and not a
    rolling window: the 5m grid is not the branch's grid.

    With `rank`, every column is replaced by its percentile inside its own timestamp before the
    windows are cut — `normalize.cross_rank`, and see it for why the target makes that the natural
    reading of a feature. It has to happen here and not on the tensor: a rank is a statistic over
    the symbols of one bar, and by the time the rows are stacked into `(row, step, feature)` the
    symbols of step `k` are no longer next to each other.
    """
    pos = pd.Series(np.arange(len(index)), index=index)
    frames = {}
    for symbol in pos.index.get_level_values(1).unique():
        f = features(load(symbol, tf), EXTREMA_WINDOW)[keep]
        f.index = f.index + dataset._shift(tf)
        frames[symbol] = f
    if rank:
        frames = ranked(frames)
    out = np.empty((len(index), steps, len(keep)), dtype="float32")
    for symbol, rows in pos.groupby(level=1):
        out[rows.to_numpy()] = window(frames[symbol], rows.index.get_level_values(0), steps)
    if not np.isfinite(out).all():
        raise ValueError("the window reaches past the start of a symbol's history — widen the warm-up")
    return out


def ranked(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The same per-symbol frames with every column replaced by its rank across the symbols.

    Stacked into one panel, ranked per timestamp by `normalize.cross_rank`, and split back. Thin
    timestamps leave the *index* rather than turning into NaN inside it: `window` reads the last
    bar at or before each row, so a missing timestamp is the case it already handles, while a NaN
    left sitting in place would trip the warm-up check in `sequences` and blame the wrong thing.
    """
    panel = pd.concat(frames, names=["symbol"]).swaplevel().sort_index()
    n = panel.groupby(level=0)[panel.columns[0]].transform("size")
    panel = normalize.cross_rank(panel[n >= normalize.MIN_SYMBOLS])
    return {symbol: g.droplevel(1) for symbol, g in panel.groupby(level=1)}


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
        return np.load(path, mmap_mode="r")
    np.save(path, sequences(index, **params))
    stamp.write_text(json.dumps(written, indent=2, sort_keys=True))
    # Mapped and not resident: four branches are 6.9 GB of float32, and only the rows of the batch
    # being read have to be in memory at once.
    return np.load(path, mmap_mode="r")


def stats_of(x: np.ndarray, train: np.ndarray) -> pd.DataFrame:
    """`normalize.fit` on the train rows of this fold, per fold and never once for the run — a
    scaler fitted across the whole span reads quantiles of the test period.

    Fitted on the most recent step alone: every step of the window is drawn from the same series,
    so one step is a clean estimate of the column's train-period quartiles, and it keeps the fit
    to the same shape `normalize.fit` takes everywhere else.
    """
    # Named when the width says these are the selected indicators, so a degenerate column comes
    # back from `normalize.fit` with a name and not a position.
    names = next((c for c in FEATURES.values() if len(c) == x.shape[2]), list(range(x.shape[2])))
    return normalize.fit(pd.DataFrame(x[train, -1], columns=names))


def apply(x: np.ndarray, stats: pd.DataFrame) -> np.ndarray:
    """`normalize`'s transform, on a whole tensor or on one batch of it — the arithmetic is
    elementwise, so which of the two it is makes no difference to the result."""
    center, scale = stats.center.to_numpy("float32"), stats.scale.to_numpy("float32")
    return (np.clip((x - center) / scale, -normalize.CLIP, normalize.CLIP) * normalize.SCALE).astype("float32")


def scaled(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """The whole tensor, scaled. Kept for the checks; `Branches` never calls it."""
    return apply(x, stats_of(x, train))


class Branches:
    """The branch tensors of one fold, scaled when a batch is read rather than up front.

    Four branches at 643k rows, 24 steps and 28 columns are 6.9 GB of float32, and the whole-array
    form of the transform wants that much again for its output. Scaling the 512 rows of a batch
    instead costs an operation that was going to happen anyway, one batch at a time, and the
    tensors stay memory-mapped and shared between folds. The statistics are still fitted once per
    fold on the train rows alone, which is the part that matters for leakage.

    Indexing returns one tensor per branch, on the device, so `model(x[rows])` reads the same at
    every call site as it did with one array.
    """

    def __init__(self, arrays: list[np.ndarray], train: np.ndarray):
        self.arrays = arrays
        self.stats = [stats_of(a, train) for a in arrays]

    def __getitem__(self, rows: np.ndarray) -> list[torch.Tensor]:
        return [torch.from_numpy(apply(a[rows], st)).to(DEVICE) for a, st in zip(self.arrays, self.stats)]

    @property
    def widths(self) -> list[int]:
        return [a.shape[2] for a in self.arrays]


def cross_section(y: pd.Series, mode: str = "z") -> pd.Series:
    """`y` with the level ("demean") or the level and the scale ("z") of its timestamp removed.

    Rank IC is computed per date over the symbols, so adding a constant to a whole cross-section
    or multiplying it by a positive number leaves every rank untouched. Measured on the train
    period, that invisible part is 47% of the target's variance, and a pooled Huber spends its
    gradient on it. Removing it does not throw away signal — it throws away the half of the label
    the evaluation is blind to by construction.

    Dates with a single symbol have no dispersion and drop out under "z" (3.6% of the train rows);
    the metric already ignores them, needing three symbols to correlate anything.

    The two modes are not the same bet. "demean" removes the 47% of variance that is the common
    component, and nothing else. "z" also divides by the dispersion of the date — which is
    invisible to the metric too, but re-weights the training: a date whose symbols barely differ
    has its small, mostly accidental gaps blown up to unit scale, and the Huber then treats them
    as the errors worth fixing. Measured, "z" costs 0.007 of Rank IC against the raw label.
    """
    g = y.groupby(level=0)
    out = y - g.transform("mean")
    if mode == "z":
        out = out / g.transform("std")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


class Net(nn.Module):
    """A GRU per branch, concatenated final states, dropout, linear head. No activation on output.

    The tanh the spec removed used to saturate exactly on the pivots, where the gradient matters
    most; nothing downstream needs a bounded output, since Rank IC is invariant to any monotone
    transform. The head starts at zero bias so the first predictions sit at the centre of the
    target's distribution. GRU and not LSTM: at 24 steps the cell state buys nothing and costs a
    quarter of the parameters.

    With `shared`, one encoder reads all four branches and a timeframe embedding is concatenated
    to each final state — open point 5. The branches carry the same columns with the same meaning
    at four horizons, so four private encoders learn four times what one could, and the embedding
    is what keeps the head able to tell them apart. Same expressiveness on paper, a quarter of the
    parameters, and which of the two wins is a measurement and not an argument.

    Dropout lands on the concatenated states, which is elementwise the same as on each branch
    before the concat, and one call instead of four.
    """

    def __init__(
        self,
        widths: list[int],
        hidden: int = H,
        dropout: float = DROPOUT,
        shared: bool = False,
        embedding: int = EMBEDDING,
    ):
        super().__init__()
        self.shared = shared
        if shared:
            if len(set(widths)) != 1:
                raise ValueError(f"a shared encoder needs one width, not {widths}")
            self.gru = nn.GRU(widths[0], hidden, batch_first=True)
            self.timeframe = nn.Embedding(len(widths), embedding)
            width = len(widths) * (hidden + embedding)
        else:
            self.gru = nn.ModuleList(nn.GRU(f, hidden, batch_first=True) for f in widths)
            width = len(widths) * hidden
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if self.shared:
            states = [
                torch.cat([self.gru(x)[1][-1], self.timeframe.weight[i].expand(len(x), -1)], dim=1)
                for i, x in enumerate(xs)
            ]
        else:
            states = [gru(x)[1][-1] for gru, x in zip(self.gru, xs)]
        return self.head(self.drop(torch.cat(states, dim=1))).squeeze(-1)


def predict(model: Net, x: Branches, rows: np.ndarray, index: pd.MultiIndex, batch: int = 4096) -> pd.Series:
    """Predictions for `rows`, on the (timestamp, symbol) index the metrics need."""
    model.eval()
    out = []
    with torch.no_grad():
        for at in np.array_split(rows, max(1, len(rows) // batch)):
            out.append(model(x[at]).cpu().numpy())
    return pd.Series(np.concatenate(out), index=index)


def fit(
    x: Branches,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seed: int = 0,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH,
    delta: float = DELTA,
    shared: bool = False,
    quiet: bool = True,
) -> Net:
    """One model, stopped when the validation Rank IC stops improving.

    Model selection and early stopping read Rank IC and never the loss: the Huber anchors the
    scale of the prediction and every metric downstream reads only its ordering, so the round with
    the best loss is not the round with the best signal.
    """
    torch.manual_seed(seed)
    model = Net(x.widths, shared=shared).to(DEVICE)
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
            loss = loss_fn(model(x[at[batch]]), y[batch].to(DEVICE))
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
    if best_state is None:
        raise ValueError(f"the validation Rank IC was never a number over {epochs} epochs — {len(valid)} valid rows")
    model.load_state_dict(best_state)
    return model


def fold(
    x: list[np.ndarray],
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 0,
    z_target: str = "off",
    band: int = 0,
    **kw,
) -> pd.Series:
    """Fit on one fold's train side and predict its test slice.

    The validation tail is cut out of the train side and purged against its own boundary, exactly
    as the GBM does it: an unpurged valid lets early stopping pick the epoch that best reads
    labels it has already seen.
    """
    inner, valid = split.temporal_fraction(train, VALID_FRACTION)
    if band:
        # Validation too: early stopping has to read the band the model is being judged on.
        # `nearpivot.near`, the same rows its measurement was taken on. A model fitted on
        # everything optimises the mass — 84% of the rows sit past 48 bars, where the relation has
        # the opposite sign — so restricting the training set is the upper bound of what any
        # regime-conditioned head could reach.
        inner, valid = nearpivot.near(inner, band), nearpivot.near(valid, band)
    z = Branches(x, inner.row.to_numpy())  # fitted on the train rows only, and the inner train at that
    if z_target != "off":
        # Training only. Validation keeps the raw label, so early stopping reads the metric the
        # result is judged on — which the transform leaves unchanged anyway.
        y = cross_section(inner.target, z_target)
        inner, kw = inner.loc[y.index].assign(target=y), dict(kw, delta=Z_DELTA[z_target])
    return predict(fit(z, inner, valid, seed, **kw), z, test.row.to_numpy(), test.index)


def run(
    x: list[np.ndarray], df: pd.DataFrame, start: str, folds: int = 4, seeds: int = 1, **kw
) -> dict[str, pd.DataFrame]:
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


def _market_beta(pred: pd.Series, target: pd.Series) -> dict[str, float]:
    """The same metrics against the target with its cross-sectional mean removed — step 2's
    diagnostic, repeated so the two steps are read the same way."""
    return metrics.signal(pred, target - target.groupby(level=0).transform("mean"))


def save(
    path: Path, model: Net, x: Branches, branches: list[str], keep: list[str], label: str, rank: bool = False
) -> None:
    """The weights and everything needed to feed them: the scaling of the train period, the
    columns, the branches and their widths. A model without its scaler is not a model."""
    torch.save(
        {
            "state": model.state_dict(),
            "stats": x.stats,
            "widths": x.widths,
            "branches": branches,
            "keep": keep,
            "steps": STEPS,
            "shared": model.shared,
            "label": label,
            "rank": rank,
        },
        path,
    )


def restore(path: Path = CHECKPOINT) -> tuple[Net, dict]:
    checkpoint = torch.load(path, weights_only=False)
    model = Net(checkpoint["widths"], shared=checkpoint["shared"]).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


def predict_frame(model: Net, checkpoint: dict, bars: pd.DataFrame) -> pd.Series:
    """The prediction at every bar of `bars`, which must be candles of the model's own branch.

    The chart draws one timeframe, so this takes one branch and says so rather than guessing how
    to line four of them up on a grid the app does not have.

    `shift(1)` is the no-anticipation rule on this grid: at the bar labelled T the candle T is the
    one still forming — labels are open times everywhere in this project — so the last closed
    candle is T-1, exactly as the 5m row of 10:05 reads the 15m candle labelled 09:45. The first
    bars have no window behind them and come back NaN rather than as a number nothing supports.
    """
    if len(checkpoint["branches"]) != 1:
        raise ValueError(f"the chart draws one branch, not {checkpoint['branches']}")
    # A ranked feature is a statement about this symbol against the others trading at that instant,
    # so there is nothing to compute from one series. The same thing is true of the cross-sectional
    # label, and the chart draws that as a heatmap over the panel rather than as a line on one pair.
    if checkpoint.get("rank"):
        raise ValueError("this model reads cross-sectional ranks, which one pair cannot supply")
    f = features(bars, EXTREMA_WINDOW)[checkpoint["keep"]]
    x = apply(window(f.shift(1), bars.index, checkpoint["steps"]), checkpoint["stats"][0])
    ready = np.isfinite(x).all(axis=(1, 2))
    out = pd.Series(np.nan, index=bars.index, name="prediction")
    if ready.any():
        with torch.no_grad():
            out[ready] = model([torch.from_numpy(x[ready]).to(DEVICE)]).cpu().numpy()
    return out


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
    out = run([x], df, "2024-01-25", **kw)
    assert out["folds"].index.tolist() == ["fold 1", "fold 2", "mean", "std"]
    assert out["folds"].loc["mean", "rank_ic"] > 0.5, out["folds"]
    blind = linear.run(df.assign(last=x[:, -1, 0]), "2024-01-25")
    assert blind.loc["test", "rank_ic"] < 0.15, "the last step alone must be blind to a drift"

    noise = run([x], df.assign(target=rng.normal(size=n)), "2024-01-25", **kw | dict(epochs=6, patience=6))
    assert abs(noise["folds"].loc["mean", "rank_ic"]) < 0.15, noise["folds"]

    # Scaling reads the train rows only, so a slice normalises to exactly what it does inside the
    # whole — the leakage check `normalize` makes on frames, made here on the tensor.
    train = np.arange(n // 2)
    z = scaled(x, train)
    assert np.allclose(z[:10], scaled(x[: n // 2], train[: n // 2])[:10])
    assert abs(z).max() <= normalize.CLIP * normalize.SCALE + 1e-6
    # And a batch scaled on the way to the model is the same number as the whole tensor scaled up
    # front — the equality that lets four branches stay memory mapped instead of copied per fold.
    rows = np.array([3, 700, 4999])
    assert np.allclose(Branches([x], train)[rows][0].cpu().numpy(), z[rows])
    # Two seeds differ, which is the reason the spec reports mean +- std over them.
    kwx = dict(epochs=1, patience=1)
    a = fit(Branches([x], train), df.iloc[: n // 2], df.iloc[n // 2 :], seed=0, **kwx)
    b = fit(Branches([x], train), df.iloc[: n // 2], df.iloc[n // 2 :], seed=1, **kwx)
    assert not torch.allclose(a.head.weight, b.head.weight)

    # Step 4. The signal sits on the *middle* branch and the other two are noise of the same
    # shape, so a model that reads only the first branch, or that lets two useless encoders drown
    # the third, fails here. Shared weights, which is the harder of the two: one encoder sees all
    # three and only the timeframe embedding tells the head which is which.
    blank = [rng.normal(size=(n, steps, f)).astype("float32") for _ in range(2)]
    multi = run([blank[0], x, blank[1]], df, "2024-01-25", shared=True, **kw)
    assert multi["folds"].loc["mean", "rank_ic"] > 0.5, multi["folds"]
    # Same expressiveness, roughly a quarter of the parameters — the reason the variant exists.
    widths = [len(COLUMNS)] * len(BRANCHES)  # the real shape: at f=2 the head, shared either way, dominates

    def size(**how: bool) -> int:
        return sum(p.numel() for p in Net(widths, **how).parameters())

    assert size(shared=True) < size() / 3, (size(shared=True), size())

    # The window, which is the one place a bug would invent future information. A branch label
    # sits at `label + tf - 5m` on the 5m grid, so the 15m candle that closes at 10:00 becomes
    # readable at the 5m bar of 09:55 and not one bar earlier.
    labels = pd.date_range("2024-01-01 09:00", periods=8, freq="15min", tz="UTC")
    frame = pd.DataFrame({"v": range(8)}, index=labels + pd.Timedelta("10min"), dtype="float64")
    grid = pd.date_range("2024-01-01 09:50", periods=4, freq="5min", tz="UTC")  # 09:50 09:55 10:00 10:05
    w = window(frame, grid, steps=3)
    assert w.shape == (4, 3, 1)
    assert w[0, -1, 0] == 2, "at 09:50 the last closed 15m candle is the one labelled 09:30"
    assert w[1, -1, 0] == 3, "at 09:55 the candle labelled 09:45 has closed"
    assert w[3, -1, 0] == 3, "at 10:05 the candle labelled 10:00 is still forming"
    assert list(w[1, :, 0]) == [1.0, 2.0, 3.0], "oldest step first"
    # Causal: truncating the frame after the grid leaves every value untouched.
    assert np.array_equal(w, window(frame[frame.index <= grid[-1]], grid, steps=3))

    # The cross-sectional rank round-trip: three symbols on one grid, one of them always the
    # largest, so its rank is the top of every cross-section it appears in.
    grid = pd.date_range("2024", periods=6, freq="h", tz="UTC")
    panels = {
        "a": pd.DataFrame({"v": [3.0] * 6}, index=grid),
        "b": pd.DataFrame({"v": [2.0] * 6}, index=grid),
        "c": pd.DataFrame({"v": [1.0] * 5}, index=grid[:5]),  # one bar short, as a real pair is
    }
    r = ranked(panels)
    assert set(r) == set(panels) and all(f.index.nlevels == 1 for f in r.values())
    # Three symbols: pct ranks 1/3, 2/3, 1 about a mean of 2/3, so the top sits at +1/3.
    assert np.allclose(r["a"].v, 1 / 3) and np.allclose(r["b"].v, 0.0), "top, middle and bottom of three"
    # The last bar has only two symbols, so it leaves the index instead of becoming a NaN in it —
    # which is what lets `window` read the previous bar there rather than propagating a hole.
    assert len(r["a"]) == 5 and grid[-1] not in r["a"].index
    assert r["a"].notna().all().all()

    # The cross-sectional target: standardised inside each date, and — the reason for the whole
    # change — scoring identically under the metric, which is what makes dropping it free.
    z = cross_section(df.target)
    per_date = z.groupby(level=0)
    assert np.allclose(per_date.mean(), 0, atol=1e-9) and np.allclose(per_date.std(), 1, atol=1e-9)
    d = cross_section(df.target, "demean")
    assert np.allclose(d.groupby(level=0).mean(), 0, atol=1e-9) and d.std() > 0.5, "demean keeps the dispersion"
    p = out["pred"]
    assert np.isclose(
        metrics.signal(p, z.loc[p.index])["rank_ic"], metrics.signal(p, df.target.loc[p.index])["rank_ic"]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default="2025-06", help="first fold boundary; the default matches step 2's")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1, help="initialisations per fold; the spec asks 5 in exploration")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--cache", type=Path, default=CACHE, help="step 2's dataset, read for its rows")
    ap.add_argument("--horizon", action="store_true", help="also split the test Rank IC by distance to the next pivot")
    ap.add_argument(
        "--pnl",
        action="store_true",
        help="also price the prediction as a directional threshold rule, fees included; not for `cross`",
    )
    ap.add_argument("--verbose", action="store_true", help="print the validation Rank IC each epoch")
    ap.add_argument("--features", choices=list(FEATURES), default="selected", help="which candidate set to feed")
    ap.add_argument(
        "--label",
        choices=list(LABELS),
        default="excursion",
        help="which target to fit; `cross` is the cross-sectional one, `swing` the retrospective",
    )
    ap.add_argument(
        "--branches", default=BRANCH, help=f"comma separated, from {','.join(BRANCHES)} — step 4 is all four"
    )
    ap.add_argument(
        "--encoder",
        choices=["separate", "shared"],
        default="separate",
        help="one GRU per branch, or one for all four plus a timeframe embedding",
    )
    ap.add_argument(
        "--rank",
        action="store_true",
        help="feed each feature as its percentile across the symbols of its own timestamp",
    )
    ap.add_argument("--band", type=int, default=0, help="train only on rows within this many 5m bars of the pivot")
    ap.add_argument(
        "--z-target",
        choices=["off", "demean", "z"],
        default="off",
        help="train on the target with the level, or the level and the scale, of its timestamp removed",
    )
    ap.add_argument("--smooth", type=int, nargs="*", default=[2, 4, 12], help="output low-pass windows to report")
    ap.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=CHECKPOINT,
        help="fit one model on the whole train period and write it here for the chart app, then stop",
    )
    args = ap.parse_args()
    if args.pnl and args.label == "cross":
        # Refused before anything is trained, rather than printed as a caveat under the table.
        # `threshold.sweep` prices a directional, one-symbol-at-a-time rule with an absolute band.
        # Applied to a cross-sectional prediction it produces a plausible number that means nothing:
        # on the first `--label cross --rank` run it reported +0.51 a year with `long_share` at zero
        # — a short-only book, over twelve months in which the basket fell 51%. That is the market
        # being read back, not the signal. `simulation --pred` prices the same prediction on the
        # dollar-neutral book the label describes, and puts it on the break-even table it has to
        # clear. The predictions are written by every run, `--pnl` or not, so this costs one command
        # and no retraining.
        raise SystemExit(
            "--pnl prices the directional rule, which a cross-sectional prediction has no business "
            "in. Run without it, then:\n"
            "  uv run python -m tradingvision.simulation --pred <the parquet this writes> --smooth 12"
        )

    _selfcheck()
    df = meta(args.cache)
    # The tensor is built on every row step 2 chose, whatever the label does to the frame
    # afterwards. `row` points into it positionally, so a label that drops rows — `cross` drops the
    # last 72h of each symbol — must not be allowed to move what those positions mean, and the
    # cache stays shared between labels instead of forking per target.
    rows = df.index
    delta = DELTA
    if args.label == "cross":
        # Rewrites the purging horizon as well as the target, which is why it is not a `relabel`
        # call: this label reaches a fixed 72h ahead, four times the pivots' typical reach.
        before = len(df)
        df = dataset.relabel_cross(df)
        print(f"label cross: {before - len(df):,} of {before:,} rows dropped for want of a forward return")
    elif args.label != "excursion":
        df = df.assign(target=dataset.relabel(df.index, LABELS[args.label]))
    if args.label != "excursion":
        # Delta lives in the units of the label, so it is measured and never carried over: the
        # median of |target| over the train period, the same criterion that fixed 2.1 for the
        # default one. Train only, for the reason purging exists.
        delta = float(split.temporal(df, args.test_start)[0].target.abs().median())
        print(f"label {args.label}: |target| median {delta:.3f} on train, sd {df.target.std():.3f}")
    keep = FEATURES[args.features]
    branches = args.branches.split(",")
    if unknown := [tf for tf in branches if tf not in BRANCHES]:
        raise SystemExit(f"{unknown} is not a branch — pick from {BRANCHES}")
    # The flag only reaches the cache stamp when it is on, and the ranked tensor lives under its
    # own name. A run without it therefore still matches the tensors already on disk instead of
    # asking for 1.7 GB to be rebuilt to record a False.
    extra = {"rank": True} if args.rank else {}
    x = [
        cached_sequences(
            TENSOR.with_stem(f"{TENSOR.stem}-{tf}-{args.features}{'-rank' if args.rank else ''}"),
            rows,
            keep=keep,
            tf=tf,
            steps=STEPS,
            **extra,
        )
        for tf in branches
    ]
    print(
        f"{len(rows):,} rows ({len(df):,} labelled), {STEPS} steps x {len(keep)} features on "
        f"{'+'.join(branches)}, {args.encoder}, "
        f"{'cross-sectional ranks' if args.rank else 'levels'}, {DEVICE}"
    )

    if args.save:
        # No folds: the walk-forward exists to measure, and what the app draws is one model fitted
        # on everything before the test period, with the scaling that goes with it.
        train, _ = split.temporal(df, args.test_start)
        inner, valid = split.temporal_fraction(train, VALID_FRACTION)
        z = Branches(x, inner.row.to_numpy())
        model = fit(
            z, inner, valid, epochs=args.epochs, delta=delta, shared=args.encoder == "shared", quiet=not args.verbose
        )
        save(args.save, model, z, branches, keep, args.label, args.rank)
        print(f"fitted on {len(inner):,} rows to {args.test_start} and saved to {args.save}")
        return

    print(f"{args.folds} folds from {args.test_start}, {args.seeds} seed(s) each\n")

    out = run(
        x,
        df,
        args.test_start,
        args.folds,
        args.seeds,
        epochs=args.epochs,
        delta=delta,
        z_target=args.z_target,
        band=args.band,
        shared=args.encoder == "shared",
        quiet=not args.verbose,
    )
    # Always written, whatever else the run was asked to print: the predictions are what every
    # economic reading is taken on, and retraining twenty models to look at them again is a waste.
    predictions = (
        STORE / f"pred-{args.label}-{args.features}{'-rank' if args.rank else ''}-{'+'.join(branches)}.parquet"
    )
    out["pred"].rename("pred").to_frame().to_parquet(predictions)
    print(out["folds"].round(4).to_string())
    if args.seeds > 1:
        print("\nper seed\n")
        print(out["seeds"].round(4).to_string())
    if args.smooth:
        print("\nlow-pass on the prediction — Rank IC of the k-bar average, pooled over the folds\n")
        rows = {k: metrics.signal(threshold.smoothed(out["pred"], k), out["target"]) for k in [1] + args.smooth}
        print(pd.DataFrame(rows).T.rename_axis("k").round(4).to_string())
    print("\nmarket beta — the same metrics once the cross-sectional mean is removed\n")
    print(out["market_beta"].round(4).to_string())
    if args.horizon:
        print("\ntest Rank IC by bars to the next pivot, pooled over the folds\n")
        print(out["horizon"].round(4).to_string())
    if args.pnl:
        # The label says which way the trade points: `swing` is -1 at a low, so a low prediction
        # is a buy, while `excursion` is already signed the way the position is.
        sign = -1 if args.label == "swing" else 1
        close = threshold.prices(out["pred"].index)
        print(f"\nthreshold rule on the test folds — {FEE * 100:.2f}% per side, log returns per year\n")
        print(threshold.sweep(out["pred"], close, sign=sign).round(4).to_string())
        held = threshold.buy_and_hold(close)["net_per_year"]
        print(f"\nbuy and hold on the same rows: {held * 100:+.1f}% per year")
    print(f"\npredictions written to {predictions}")


if __name__ == "__main__":
    main()
