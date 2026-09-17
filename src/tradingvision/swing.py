"""Step 7: the swing-leg strategy, made tradable — and what it took to get there.

The label is `swing_leg_target`: -1 on a pivot low, +1 on the next pivot high, interpolated in
between. Reading it is reading the leg, so the rule it describes is the oracle's own — buy the
low, sell the high — and the benchmark is therefore the oracle itself, on the same bars, with the
same fee, priced with the same arithmetic.

**Three measurements came before any model, and each of them moved the plan.**

*What the label is worth if you knew it exactly.* Trading `swing_leg_target` itself, with perfect
knowledge and the best threshold, earns 5.46 log a year against the oracle's 10.20 over the same
eight symbols — **54%**. That is not a coincidence and it is the ceiling of this whole approach:
the label smooths the leg, so a rule that reads it perfectly still enters after the low and exits
before the high. Half the oracle is what perfect knowledge of this label buys.

*What imperfect knowledge is worth.* The same label corrupted to a correlation of 0.9 with the
truth earns 3.68; at 0.7, 1.25; at 0.5 it **loses** 0.95. The break-even sits between 0.5 and 0.7,
which is the number every design decision here is measured against.

*Where the existing pipeline stood.* A ridge on the 29 `features` columns reaches a time-series
correlation of 0.577 with the label and loses 0.36 a year. Adding the causal leg state of `legs`
lifts the correlation to 0.623 — past the nominal break-even — and it still loses 0.29.

That last line is the finding the module is built on. **Correlation with the label is not the
objective and optimising it is not the way to the money.** A model can be 0.62 correlated and lose,
because its errors are not spread evenly: they concentrate at the turns, which is precisely where
every trade is opened and closed. Fitting a Huber to the label spends its gradient on the middle of
the legs, which is 80% of the rows and none of the decisions.

So the model is trained twice. First on the label, which gives it the structure of a leg and
costs nothing to reuse. Then **directly on the money**: the position is a differentiable function
of the output, the reward is the log return it earns minus the fee on every unit of position
changed, and the gradient of that reward is exact because the price is exogenous — nothing this
model does moves the market it trades. That is Moody and Saffell's direct reinforcement, and it is
the right tool here for a reason that is specific rather than fashionable: the objective and the
metric become the same quantity, so a gradient step can no longer improve the label fit and hurt
the P&L at once.

    uv run python -m tradingvision.swing --stage both
    uv run python -m tradingvision.swing --timeframe 1h --folds 4 --save

Every number the CLI prints is out of sample: four expanding walk-forward folds, the train side of
each purged of every bar whose leg closes past its cut, thresholds and epochs chosen on a purged
validation tail inside the train side, and the oracle recomputed on exactly the rows the model
traded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from tradingvision import legs, normalize
from tradingvision.data.binance import STORE, SYMBOLS, load
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import swing_leg_target
from tradingvision.features import COLUMNS, features
from tradingvision.oracle import FEE
from tradingvision.oracle import run as oracle_run

TF = "1h"
W = EXTREMA_WINDOW
STEPS = 24  # bars of history the encoder reads, the same depth step 3 used
# Every symbol of the store covers 2021 on; the four newest list in the second half of 2020. Three
# times the history the first runs used, and — the half that matters more — a test period that
# holds both regimes instead of one: 2023-24 rose and 2025-26 fell by half. A strategy measured
# only on the second is measured against a headwind it will not always have.
SINCE = "2021"
TEST_START = "2023-01"
FOLDS = 4
YEAR = pd.Timedelta(365.25, "D")
# The leg state at three widths. The confirmation lag *is* the width, and the lag is the binding
# constraint of the whole problem — the oracle run at its own confirmation lag keeps 0.4 log a year
# of the 5.8 it makes with hindsight — so reading the structure at 6 bars as well as at 24 is not a
# richer feature set, it is a faster one.
SCALES = (6, 12, W)
# Features, leg state at each scale, and the exhaustion columns. The ridge probe puts features and
# leg state at 0.623 against 0.577 and 0.584 alone: they are not the same information.
BASIC = list(COLUMNS) + [f"{c}_{W}" for c in legs.STATE]
INPUTS = list(COLUMNS) + [f"{c}_{w}" for w in SCALES for c in legs.STATE] + list(legs.EXHAUSTION)

H = 48
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH = 512
VALID_FRACTION = 0.2
# Reinforcement. A chunk is an episode: the fee couples neighbouring bars, so a position can only
# be priced along a stretch of time and never over a shuffled batch.
CHUNK = 256
CHUNKS = 48  # episodes per gradient step
# logit -> position. Measured on the toy of `_selfcheck`, which is built so the supervised prior
# is wrong and only the reward can fix it: at 4 the sigmoid is saturated, the gradient vanishes and
# the stage escapes into "always flat" at every learning rate tried; at 2 it recovers the right
# policy at 1e-3 and above. The trap is real and it is not a toy artefact — a saturated policy on
# real data looks like a model that has learned to hold, which is exactly what a tired reader
# wants to believe.
SHARPNESS = 2.0
ADOPT = 1.0  # how hard the policy starts out agreeing with the supervised rule; see `Net.adopt`
POLICY_LR = 1e-3  # the same sweep: 3e-4 never leaves the prior, 1e-3 and 3e-3 both land on it
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = STORE / "swing.pt"
TENSOR = STORE / "swing"


def inputs(bars: pd.DataFrame, window: int = W, keep: list[str] | None = None) -> pd.DataFrame:
    """The model's columns for one symbol: the candidates, the leg state at each scale, exhaustion."""
    parts = [features(bars, window)[COLUMNS]]
    parts += [legs.state(bars.close, w).add_suffix(f"_{w}") for w in SCALES]
    parts.append(legs.exhaustion(bars, window))
    return pd.concat(parts, axis=1)[INPUTS if keep is None else keep]


def frame(symbol: str, tf: str = TF, since: str = SINCE, window: int = W, keep: list[str] | None = None):
    """One symbol's rows: inputs, label, the next bar's return, the close, the purging horizon.

    The label's pivots and the leg state's pivots are the same turns read from two sides — the
    centred frame for what is being predicted, `legs.confirmed` for what was knowable. Mixing them
    up is the one mistake that would make every number here meaningless, so they are built by
    different functions in different modules and meet only in this frame.
    """
    bars = load(symbol, tf).loc[since:]
    close = bars.close
    piv = find_pivots(close, window)
    out = inputs(bars, window, keep)
    out["target"] = swing_leg_target(close, piv)
    out["next_pivot"] = legs.next_pivot(close, window)
    out = out.dropna()
    out["close"] = close.reindex(out.index)
    # The return a position taken at this close earns by being held to the next *kept* bar, which
    # is what the reward and the round trips are both priced on. Computed after the warm-up is
    # dropped and not before: a return that reaches over a row nobody can hold is not one a
    # position earns. The last bar pays nothing, which is what a position cannot be credited for.
    out["ret"] = np.log(out.close.shift(-1) / out.close).fillna(0.0)
    return out


def scaler(f: pd.DataFrame, before: str, keep: list[str] | None = None) -> pd.DataFrame:
    """`normalize.fit` on this symbol's own rows before `before` — per symbol, and fitted once.

    Per symbol because half these columns are not scale free across the panel: `log_dollar_volume`
    differs between BTC and BAT by more than it differs between a quiet week and a violent one, and
    a pooled scaler would spend its range on the symbol and not on the state.

    Fitted on the rows before the first fold's cut and reused for every fold. That is a strict
    subset of every fold's train side — folds expand — so no fold is scaled with anything it was
    not allowed to see, and the alternative, refitting per fold, would mean rebuilding the tensor
    four times for a change in the third decimal.
    """
    return normalize.fit(f.loc[:before, INPUTS if keep is None else keep])


def sequences(f: pd.DataFrame, stats: pd.DataFrame, steps: int = STEPS, keep: list[str] | None = None):
    """`(len(f), steps, len(INPUTS))` — the last `steps` rows at each row, oldest first, scaled.

    `f` is one symbol on one timeframe, so step `k` back is the frame shifted `k` of its own bars:
    no grid crossing and no forward fill, unlike the multi-branch tensors of step 3. The first
    `steps - 1` rows have no window behind them and come back NaN, which `build` drops.
    """
    z = normalize.apply(f[INPUTS if keep is None else keep], stats).to_numpy("float32")
    out = np.full((len(f), steps, z.shape[1]), np.nan, dtype="float32")
    for k in range(steps):
        out[k:, steps - 1 - k, :] = z[: len(z) - k]
    return out


def build(symbols, tf: str = TF, since: str = SINCE, window: int = W, steps: int = STEPS, keep=None):
    """`(tensor, meta)` over every symbol — the tensor positional, the meta indexed by time.

    Rows are grouped by symbol and ordered in time inside each group, which is what lets the
    reinforcement stage cut contiguous episodes out of a fold without re-sorting anything.
    """
    blocks, metas, at = [], [], 0
    for symbol in symbols:
        f = frame(symbol, tf, since, window, keep)
        if len(f) < steps + 100:
            continue
        x = sequences(f, scaler(f, TEST_START, keep), steps, keep)
        # `filled` and not `keep`: the column list is already called that, and the mask overwrote
        # it on the first symbol — silently, because a one-symbol build never reaches the second.
        filled = np.isfinite(x).all(axis=(1, 2))
        f, x = f[filled], x[filled]
        meta = f[["target", "next_pivot", "close", "ret"]].copy()
        meta["symbol"] = symbol
        meta["row"] = np.arange(at, at + len(f))
        blocks.append(x)
        metas.append(meta)
        at += len(f)
    tensor = np.concatenate(blocks)
    meta = pd.concat(metas)
    return tensor, meta


def cached(path: Path, symbols: list[str], **params):
    """`build` on disk with a stamp of what produced it — `dataset.cached`'s contract, on a tensor."""
    stamp, npy, pq = path.with_suffix(".json"), path.with_suffix(".npy"), path.with_suffix(".parquet")
    # `keep` travels as `inputs` and not as itself: it is the column list, and recording it twice
    # would make two stamps of the same build disagree on nothing.
    written = dict(
        {k: v for k, v in params.items() if k != "keep"},
        symbols=sorted(symbols),
        inputs=list(params.get("keep") or INPUTS),
    )
    if npy.exists():
        if not stamp.exists():
            raise SystemExit(f"{npy} has no {stamp.name} recording how it was built — delete it")
        if (was := json.loads(stamp.read_text())) != written:
            raise SystemExit(f"{npy} was built with {was}, not {written} — delete it")
        return np.load(npy, mmap_mode="r"), pd.read_parquet(pq)
    tensor, meta = build(symbols, **params)
    np.save(npy, tensor)
    meta.to_parquet(pq)
    stamp.write_text(json.dumps(written, indent=2, sort_keys=True))
    return np.load(npy, mmap_mode="r"), meta


def purge(meta: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, test)` around `cut`, train purged of every bar whose leg closes past it.

    `split.temporal` says the same thing on a (timestamp, symbol) index; this one reads a plain
    time index with a `symbol` column, which is the shape the episodes need.
    """
    when = meta.index
    return meta[(when < cut) & (meta.next_pivot < cut)], meta[when >= cut]


def folds(meta: pd.DataFrame, start: str = TEST_START, n: int = FOLDS):
    """`(train, test)` for `n` expanding folds, the test slices tiling `start`..end."""
    at, last = pd.Timestamp(start, tz="UTC"), meta.index.max()
    edges = [at + (last - at) * i / n for i in range(n)] + [None]
    for cut, until in zip(edges, edges[1:]):
        train, test = purge(meta, cut)
        if until is not None:
            test = test[test.index < until]
        yield train, test


class Net(nn.Module):
    """One GRU, two heads: what the leg is doing, and what to do about it.

    The label head is the supervised stage and the policy head is the reinforcement stage, and
    they share the encoder on purpose — the structure of a leg is the expensive thing to learn and
    the label teaches it for free. The policy head is initialised as the negated label head, so
    the reinforcement stage starts from exactly the rule the supervised model implies: long when
    the predicted label is low, which is to say when the bar sits near a pivot low.
    """

    def __init__(self, width: int, hidden: int = H, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.policy = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.policy.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.gru(x)[1][-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.head(h).squeeze(-1), self.policy(h).squeeze(-1)

    def adopt(self, scale: float = ADOPT) -> None:
        """Start the policy from the supervised rule: long when the predicted label is low.

        `scale` is a prior and not a commitment, and it is deliberately not the sigmoid's
        sharpness: a large one puts the initial logits where the sigmoid is flat, and a stage that
        cannot move is a stage that agrees with whatever it inherited.
        """
        with torch.no_grad():
            self.policy.weight.copy_(-scale * self.head.weight)
            self.policy.bias.copy_(-scale * self.head.bias)


def outputs(model: Net, x, rows: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Both heads over `rows`, in evaluation mode."""
    model.eval()
    label, logit = [], []
    with torch.no_grad():
        for at in np.array_split(rows, max(1, len(rows) // batch)):
            a, b = model(torch.from_numpy(np.asarray(x[at])).to(DEVICE))
            label.append(a.cpu().numpy())
            logit.append(b.cpu().numpy())
    return np.concatenate(label), np.concatenate(logit)


# ---------------------------------------------------------------- the rule and its price


def positions(signal: np.ndarray, enter: float, exit: float) -> np.ndarray:
    """Long from the first bar under `-enter` to the first bar over `exit`, flat in between.

    Long only and hysteretic, which is the oracle's own shape: it buys a low and holds to the next
    high. A rule that flattened whenever the signal was unremarkable would pay the round trip
    several times inside one leg, and the leg is the trade.
    """
    state = np.full(len(signal), np.nan)
    state[signal <= -enter] = 1.0
    state[signal >= exit] = 0.0
    return pd.Series(state).ffill().fillna(0.0).to_numpy()


def calibration(fitted: np.ndarray, target: np.ndarray, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """The quantile pairs that map a prediction back onto the label's own range. Train rows only."""
    grid = np.linspace(0, 1, n)
    return np.quantile(fitted, grid), np.quantile(target, grid)


def calibrate(pred: np.ndarray, cal: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """`pred` mapped through its own distribution onto the label's — the same order, the real range.

    A Huber fitted to a noisy target predicts its conditional mean, and a conditional mean shrinks:
    the label reaches -1 and +1 on the pivots and the prediction of it spans about +-0.4, because
    the model is never sure enough about a turn to commit to one. Nothing downstream minds — every
    rule here reads quantiles and every metric reads ranks, and a monotone map changes neither —
    but a chart drawn against a label that reaches 1 and a prediction that reaches 0.4 invites
    exactly one reading, that the model is weak by 60%, and that reading is wrong.

    So the map is applied where it belongs: on the way out, fitted on the train period alone and
    never in the loss. Strictly increasing, so it cannot move a single decision; what it restores
    is the unit. The alternative — training the model to reach +-1 — would be worse than cosmetic:
    it would mean penalising the one honest thing a shrunk prediction says, which is that the turn
    is not certain.
    """
    return np.interp(pred, *cal)


def trades(pos: np.ndarray, close: np.ndarray, fee: float = FEE) -> pd.DataFrame:
    """The round trips a position implies, priced the way `oracle.run` prices its legs.

    Entry at the close of the bar the position opens on, exit at the close of the bar it closes
    on, fee charged on the credited side of both fills so it compounds with the price ratio.

    A position still open on the last bar is closed there, at that close. Dropping it instead —
    which is what the oracle does, because its legs are all closed by construction — turns a rule
    that buys once and never sells into a strategy with no trades and a return of zero, and that
    is the one failure mode of this whole design that would read as a success. Marked out at the
    end, buy and hold prices as what it is.
    """
    change = np.diff(pos, prepend=0.0)
    ins, outs = np.flatnonzero(change > 0), np.flatnonzero(change < 0)
    if len(outs) < len(ins):
        outs = np.append(outs, len(pos) - 1)
    n = min(len(ins), len(outs))
    ins, outs = ins[:n], outs[:n]
    # A position opened on the very last bar has nowhere to go: closing it there would book a
    # zero-bar trade that pays only its own fee, which is an artefact of the marking out above and
    # not something anyone traded.
    live = outs > ins
    ins, outs = ins[live], outs[live]
    return pd.DataFrame(
        {"entry": ins, "exit": outs, "leg": (close[outs] / close[ins]) * (1 - fee) ** 2 - 1, "bars": outs - ins}
    )


def price(pos: np.ndarray, close: np.ndarray, span: float, fee: float = FEE) -> dict:
    """What a position earned over `span` years, in the oracle's units so the two compare."""
    t = trades(pos, close, fee)
    net_log = float(np.log1p(t.leg).sum()) if len(t) else 0.0
    # Gross is the same trades at no fee, which is the number that says whether there is an edge
    # at all. A rule can have a real edge and still lose, and the two readings have different
    # remedies — more skill against less turnover — so they are never collapsed into one here.
    gross_log = float(np.log(close[t.exit.to_numpy()] / close[t.entry.to_numpy()]).sum()) if len(t) else 0.0
    return {
        "trades": len(t),
        "gross_per_year": gross_log / span if span else np.nan,
        "log_per_year": net_log / span if span else np.nan,
        "net_return": float(np.expm1(net_log)) if net_log < 700 else np.inf,
        "in_market": float(pos.mean()),
        "win_rate": float((t.leg > 0).mean()) if len(t) else np.nan,
        "median_trade_pct": float(t.leg.median() * 100) if len(t) else np.nan,
        "median_bars": float(t.bars.median()) if len(t) else np.nan,
    }


def by_symbol(pos: pd.Series, meta: pd.DataFrame, fee: float = FEE) -> pd.DataFrame:
    """`price` per symbol, plus the oracle and buy-and-hold on exactly the bars it traded.

    The oracle is recomputed here and never carried in: it has to see the same slice, the same
    close series and the same fee, or the ratio between the two is a comparison of two periods.
    """
    rows = []
    for symbol, g in meta.assign(pos=pos.to_numpy()).groupby("symbol", sort=False):
        g = g.sort_index()
        close = g.close.to_numpy()
        span = float((g.index[-1] - g.index[0]) / YEAR)
        got = price(g.pos.to_numpy(), close, span, fee)
        piv = find_pivots(g.close, W)
        oracle = oracle_run(g.close, W, fee, piv, lag=0)
        # The same oracle filling `W` bars later — the earliest a pivot of a centred window of `W`
        # can be *known* to anyone. It is the only benchmark on this page that a causal rule could
        # in principle reach, and it is 6 to 8% of the hindsight one. Everything that makes the
        # first number enormous is the `W` bars of future it reads.
        reachable = oracle_run(g.close, W, fee, piv, lag=W)
        rows.append(
            {
                "symbol": symbol,
                **got,
                "oracle_log": oracle["log_per_year"],
                "oracle_net": oracle["net_return"],
                "reachable_log": reachable["log_per_year"],
                "hold_log": float(np.log(close[-1] / close[0]) / span),
            }
        )
    out = pd.DataFrame(rows)
    out["share"] = out.log_per_year / out.oracle_log
    out["reach"] = out.log_per_year / out.reachable_log
    return out


def choose(signal: np.ndarray, meta: pd.DataFrame, grid=None, fee: float = FEE) -> tuple[float, float, float]:
    """The band and the sign with the best excess over holding, on the rows given — validation only.

    Quantiles of the signal and not absolute levels, so the same grid means the same rule whatever
    scale a stage puts its output on. Asymmetric, because the two ends are not the same decision:
    entering late costs the rest of one leg and leaving late costs the whole of the next.

    **The sign is searched too, and that is not a hedge.** The rule this label implies is "a low
    prediction is a buy" — the bar sits near a pivot low and the leg runs up from there — and over
    2023-26 that reading has a *negative* gross on this panel while its mirror is positive: at the
    scale of a 24-bar window these markets continue more often than they turn. A band that could
    only ever read one way would report that as a failure of the model instead of as a property of
    the market, so the direction is a parameter, picked on the same validation rows as the width
    and reported with the fold.

    The criterion is net minus buy and hold, for the reason `valid_score` gives at more length: an
    absolute one prefers whatever happened to be long in a rising validation tail.
    """
    grid = grid if grid is not None else (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
    best, at = -np.inf, (float(np.quantile(signal, 0.2)), float(np.quantile(signal, 0.8)), 1.0)
    for sign in (1.0, -1.0):
        turned = sign * signal
        for q_in in grid:
            for q_out in grid:
                enter, exit = -np.quantile(turned, q_in), np.quantile(turned, 1 - q_out)
                total = 0.0
                for _, g in meta.assign(s=turned).groupby("symbol", sort=False):
                    g = g.sort_index()
                    span = float((g.index[-1] - g.index[0]) / YEAR)
                    if span <= 0:
                        continue
                    close = g.close.to_numpy()
                    got = price(positions(g.s.to_numpy(), enter, exit), close, span, fee)["log_per_year"]
                    total += got - float(np.log(close[-1] / close[0]) / span)
                if total > best:
                    best, at = total, (float(enter), float(exit), sign)
    return at


# ---------------------------------------------------------------- the two stages


def rows_of(meta: pd.DataFrame) -> np.ndarray:
    """The tensor positions of these rows, in time order inside each symbol."""
    return np.concatenate([g.sort_index().row.to_numpy() for _, g in meta.groupby("symbol", sort=False)])


def ordered(meta: pd.DataFrame) -> pd.DataFrame:
    """The same rows in the order `rows_of` returns them, so a prediction lines up with its meta."""
    return pd.concat([g.sort_index() for _, g in meta.groupby("symbol", sort=False)])


def episodes(meta: pd.DataFrame, chunk: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """`(count, chunk)` of contiguous tensor rows — the stretches a position can be priced along.

    Contiguous and inside one symbol, because the fee couples a bar to the one before it: a
    shuffled batch has no `p[t-1]` to charge a change against, and a reward computed on one would
    be the gross with the costs silently deleted.
    """
    blocks = [g.sort_index().row.to_numpy() for _, g in meta.groupby("symbol", sort=False)]
    blocks = [b for b in blocks if len(b) > chunk + 1]
    if not blocks:
        raise ValueError(f"no symbol has {chunk + 1} consecutive rows in this fold")
    which = rng.integers(0, len(blocks), count)
    out = np.empty((count, chunk), dtype="int64")
    for i, b in enumerate(which):
        start = rng.integers(0, len(blocks[b]) - chunk)
        out[i] = blocks[b][start : start + chunk]
    return out


def valid_score(model: Net, x, meta: pd.DataFrame, stage: str, fee: float = FEE) -> tuple[float, tuple]:
    """What a fold's validation tail says about the model — the number early stopping reads.

    For the supervised stage it is the correlation with the label, which is what that stage is
    fitting. For the policy stage it is the money, priced exactly as the test slice will be: the
    whole reason the stage exists is that those two are not the same ranking.
    """
    order = ordered(meta)
    label, logit = outputs(model, x, order.row.to_numpy())
    if stage == "label":
        return float(pd.Series(label).corr(pd.Series(order.target.to_numpy()))), ()
    # Excess over holding the same symbol across the same bars, and not the absolute return.
    # Measured the other way once, and the answer was "buy and hold": the validation tail of a
    # crypto train period rises, a long-only policy that never sells earns that rise, and nothing
    # in an absolute criterion prefers a strategy to a beta. The oracle this is judged against is
    # long only *and* out of the market half the time; the criterion has to want the same thing.
    total, n = 0.0, 0
    for _, g in order.assign(logit=logit).groupby("symbol", sort=False):
        span = float((g.index[-1] - g.index[0]) / YEAR)
        if span <= 0:
            continue
        close = g.close.to_numpy()
        got = price((g.logit.to_numpy() > 0).astype(float), close, span, fee)["log_per_year"]
        total += got - float(np.log(close[-1] / close[0]) / span)
        n += 1
    return (total / n if n else -np.inf), ()


def fit_label(
    x,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seed: int = 0,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = BATCH,
    quiet: bool = True,
) -> Net:
    """Stage one: a Huber on `swing_leg_target`, stopped on the validation correlation.

    This stage is not the strategy and is not judged as one. It exists to put the structure of a
    leg into the encoder — where the turns are, how long they run, what a significant one looks
    like — at a cost of one pass over a label that is already computed. The measurement that
    justifies stopping here rather than continuing is in the module docstring: fitting this label
    harder does not make the rule more profitable, because the residual concentrates at the turns.
    """
    torch.manual_seed(seed)
    model = Net(x.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Delta is the median of |target| on this fold's train side, the criterion the project fixes
    # every Huber with: the label lives in [-1, 1] here, so a delta carried from another one would
    # be a squared loss in disguise.
    loss_fn = nn.HuberLoss(delta=float(train.target.abs().median()))
    at = train.row.to_numpy()
    y = torch.from_numpy(train.target.to_numpy("float32"))
    rng = np.random.default_rng(seed)
    best, state, since = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(at)), max(1, len(at) // batch_size)):
            opt.zero_grad()
            label, _ = model(torch.from_numpy(np.asarray(x[at[batch]])).to(DEVICE))
            loss = loss_fn(label, y[batch].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        score, _ = valid_score(model, x, valid, "label")
        if not quiet:
            print(f"    label epoch {epoch + 1:3d}  valid corr {score:+.4f}{'  *' if score > best else ''}")
        if score > best:
            best, since = score, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(state)
    return model


def fit_policy(
    model: Net,
    x,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    seed: int = 0,
    epochs: int = 30,
    patience: int = 6,
    steps: int = 40,
    chunk: int = CHUNK,
    count: int = CHUNKS,
    fee: float = FEE,
    anchor: float = 0.1,
    detrend: bool = True,
    hard: bool = False,
    sharpness: float = SHARPNESS,
    lr: float = POLICY_LR,
    quiet: bool = True,
) -> Net:
    """Stage two: gradient ascent on the money, with the fee inside the reward.

        p_t = sigmoid(k * logit_t)          the position, in [0, 1]
        R   = sum_t [ p_t * r_t - c * |p_t - p_{t-1}| ]

    The gradient of `R` with respect to the weights is exact, and that is the property that makes
    this worth doing rather than sampling actions and reweighting them: the price is exogenous, so
    `r_t` does not depend on `p_t` and the reward is a plain differentiable function of the output.
    A policy gradient with no sampling noise, which on a signal this weak is the difference between
    learning and not.

    What the model is actually being taught, in the user's terms: a position that was right is
    reinforced in proportion to what it earned, one that was wrong is pushed down in proportion to
    what it lost, and — the part a supervised loss cannot express at all — *changing its mind is
    charged*, every time, at the fee it would really pay. It is the third term that turns a
    correlation into a strategy.

    Episodes start flat, so each one pays for opening. With 256-bar chunks that is a conservative
    distortion of about one round trip per 256 bars and it pushes the model towards holding rather
    than churning, which is the right direction to be wrong in.

    `detrend` is what stops the answer from being "buy and hold" — see the line that applies it.

    `anchor` keeps a little of the supervised loss alive. Without it the encoder is free to forget
    the leg structure and chase whatever the reward's noise rewards; with it the stage stays a
    refinement of a model that already knows what a leg is.
    """
    torch.manual_seed(seed + 1_000)
    model.adopt()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    ret = np.zeros(len(x), dtype="float32")
    lab = np.zeros(len(x), dtype="float32")
    # The reward is paid on the *detrended* return: each symbol's mean bar return over the train
    # side removed, so a position that is simply always on earns nothing and only timing does.
    # Without it the stage has an easier way to a high reward than learning the legs — hold — and
    # it takes it. The same reason the cross-sectional label of `data.target` removes the market:
    # a model cannot be allowed to be paid for a drift it did not predict. Prices at test are raw.
    paid = train.ret - train.groupby("symbol").ret.transform("mean") if detrend else train.ret
    ret[train.row.to_numpy()] = paid.to_numpy("float32")
    lab[train.row.to_numpy()] = train.target.to_numpy("float32")
    huber = nn.HuberLoss(delta=float(train.target.abs().median()))
    best, state, since = -np.inf, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
    for epoch in range(epochs):
        model.train()
        for _ in range(steps):
            rows = episodes(train, chunk, count, rng)
            flat = rows.reshape(-1)
            opt.zero_grad()
            label, logit = model(torch.from_numpy(np.asarray(x[flat])).to(DEVICE))
            soft = torch.sigmoid(sharpness * logit).view(count, chunk)
            # `hard` prices the reward on the position that would actually be traded — 0 or 1 —
            # with the gradient flowing through the smooth one behind it. It looks like the
            # obviously right thing: priced on the soft position the fee is charged on a sigmoid
            # drifting from 0.48 to 0.52 and costs almost nothing, while execution pays a full
            # round trip for the same crossing.
            #
            # Measured, it is worse, and off by default because of it: on the 4h walk-forward the
            # soft reward returns +0.063 net at 94 trades a year and the straight-through one
            # -0.042 at 111. The reason is the same saturation that fixes the sharpness at 2 — the
            # estimator hands the fee term the sigmoid's derivative, which is smallest exactly
            # where the position is most committed, so charging the real fee *weakens* the
            # gradient that was supposed to discourage it. Kept as a flag because the argument for
            # it is good and only the measurement is against it.
            p = soft + ((soft > 0.5).float() - soft).detach() if hard else soft
            r = torch.from_numpy(ret[flat].reshape(count, chunk)).to(DEVICE)
            # The position carried into each bar. Zero in front of the episode, which charges the
            # opening trade rather than letting a chunk inherit a free position.
            before = torch.cat([torch.zeros(count, 1, device=DEVICE), p[:, :-1]], dim=1)
            reward = p * r - fee * (p - before).abs()
            loss = -reward.sum(dim=1).mean()
            if anchor:
                loss = loss + anchor * huber(label, torch.from_numpy(lab[flat]).to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        score, _ = valid_score(model, x, valid, "policy", fee)
        if not quiet:
            print(f"    policy epoch {epoch + 1:3d}  valid log/yr {score:+.4f}{'  *' if score > best else ''}")
        if score > best:
            best, since = score, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------- walk forward


def book(signal: np.ndarray, params: tuple[float, float, float] | None) -> np.ndarray:
    """The position a score implies. `None` is the policy's own rule: long where the score is
    positive, which is the threshold its reward was trained against.

    Laying a band on the policy score instead looks obviously right — the policy's gross is the
    highest of anything measured here, 0.192 a year, and it gives 0.129 of it straight back in
    turnover, so buying patience should be worth a lot. It is not, and the flag is off because of
    it. Four folds, 4h bars, twenty symbols, test 2023-01 to 2026-09:

        stage    rule                                  gross     net    trades/yr   beats hold
        policy   read at zero                          0.192   +0.063       25.7        12/20
        label    band and sign on validation           0.095   +0.060        6.8        11/20
        policy   band and sign on validation          -0.006   -0.042        7.1         5/20
        policy   band and sign on train, 3 seeds       0.083   +0.034        9.8        12/20
        --       rsi_centered > 0.3, one column        0.172   +0.116       11.1        11/20
        --       buy and hold                              -   +0.057          -            -

    Two things are in that table and only one of them is about bands. A band has three numbers and
    the validation tail of the first fold is five months, so seventy-two combinations find the one
    that fitted five months; moving the choice onto the whole train side fixes the overfitting and
    still does not beat reading the score at zero. And the spread between the top four rows is
    smaller than the spread between symbols, which is the honest way to read the column: on this
    panel, over this period, the differences between these rules are not measurable.

    The last two rows are the ones that matter. A one-column momentum rule beats every model here,
    and buy and hold sits inside the same range. Both are in `BASELINES`, so the comparison lives
    in the code rather than in somebody's memory of it.
    """
    if params is None:
        return (signal > 0).astype(float)
    enter, exit, sign = params
    return positions(sign * signal, enter, exit)


def walk_forward(
    x,
    meta: pd.DataFrame,
    start: str = TEST_START,
    n: int = FOLDS,
    stage: str = "both",
    seed: int = 0,
    fee: float = FEE,
    quiet: bool = True,
    seeds: int = 1,
    band: bool = False,
    **kw,
) -> dict:
    """Fit and price every fold, and hand back the positions as one series over the test period.

    The positions of the folds are concatenated rather than their scores: each fold is a model,
    each model gets its own slice, and a live system that retrains at every boundary is exactly
    what that concatenation describes. A hold running across a boundary is closed and reopened
    only if the new model disagrees, and it pays for that like any other change of mind.
    """
    held = pd.Series(0.0, index=meta.index)
    held.index = pd.RangeIndex(len(meta))  # positional, keyed by `row` below
    pos = np.zeros(len(x))
    seen = np.zeros(len(x), dtype=bool)
    rows = []
    for i, (train, test) in enumerate(folds(meta, start, n), 1):
        inner, valid = purge(train, train.index.min() + (train.index.max() - train.index.min()) * (1 - VALID_FRACTION))
        model = fit_label(x, inner, valid, seed, quiet=quiet, **{k: v for k, v in kw.items() if k in ("epochs",)})
        models = [model]
        for extra in range(1, seeds):
            models.append(
                fit_label(
                    x, inner, valid, seed + extra, quiet=quiet, **{k: v for k, v in kw.items() if k in ("epochs",)}
                )
            )
        if stage in ("policy", "both"):
            models = [fit_policy(m, x, inner, valid, seed + j, fee=fee, quiet=quiet) for j, m in enumerate(models)]

        def score(rows: np.ndarray) -> np.ndarray:
            """The seeds averaged. A single initialisation of a model this weak is mostly its own
            noise — the project reports mean +- std over five seeds for exactly that reason — and
            averaging the scores before the rule reads them is the cheapest variance there is."""
            got = [outputs(m, x, rows) for m in models]
            which = 1 if stage in ("policy", "both") else 0
            return np.mean([g[which] for g in got], axis=0)

        # The band is picked on the whole train side and not on the validation tail. It has three
        # numbers in it and the tail is five months in the first fold: read there, a grid of
        # seventy-two combinations picks the one that fitted five months, and the walk-forward said
        # so — gross fell from +0.192 to -0.006 when the band was chosen on validation. Tuned on
        # the rows the encoder was fitted on it is optimistic about its own level and honest about
        # its shape, which for two thresholds and a sign is the better of the two errors.
        policy = stage in ("policy", "both")
        params = None
        if band or not policy:
            # The supervised head predicts the label and has no rule of its own, so it always
            # needs one; the policy head has one, and measured it is the better of the two.
            order = ordered(valid if not policy else train)
            params = choose(score(order.row.to_numpy()), order, fee=fee)
        order = ordered(test)
        use = score(order.row.to_numpy())
        for _, g in order.assign(s=use).groupby("symbol", sort=False):
            pos[g.row.to_numpy()] = book(g.s.to_numpy(), params)
            seen[g.row.to_numpy()] = True
        rows.append(
            {
                "fold": i,
                "train": len(inner),
                "valid": len(valid),
                "test": len(test),
                "rule": "score > 0" if params is None else f"band {params[0]:+.2f}/{params[1]:+.2f} x{params[2]:+.0f}",
            }
        )
    test_meta = meta[seen[meta.row.to_numpy()]]
    return {
        "folds": pd.DataFrame(rows),
        "pos": pd.Series(pos[test_meta.row.to_numpy()], index=test_meta.index),
        "meta": test_meta,
    }


def summarise(table: pd.DataFrame) -> dict[str, float]:
    """The one line that answers the question: what did it earn, against what the oracle earned."""
    # The dispersion across symbols, and then the caveat that goes with it: twenty crypto pairs
    # are not twenty independent draws — they share most of their variance with the market — so
    # `sd / sqrt(20)` is a floor on the real error and the true one is several times wider. It is
    # reported because a net of +0.06 next to a spread of +-0.25 is a different sentence from a
    # net of +0.06 on its own, and only the first one is true.
    return {
        "gross": float(table.gross_per_year.mean()),
        "net": float(table.log_per_year.mean()),
        "net_sd": float(table.log_per_year.std()),
        "oracle": float(table.oracle_log.mean()),
        "share": float(table.log_per_year.mean() / table.oracle_log.mean()),
        "reachable": float(table.reachable_log.mean()),
        "reach": float(table.log_per_year.mean() / table.reachable_log.mean()),
        "net_return_pct": float(np.expm1(table.log_per_year.mean()) * 100),
        "hold": float(table.hold_log.mean()),
        "excess": float((table.log_per_year - table.hold_log).mean()),
        "trades": float(table.trades.mean()),
        "win_rate": float(table.win_rate.mean()),
        "in_market": float(table.in_market.mean()),
        "beat_hold": int((table.log_per_year > table.hold_log).sum()),
        "positive": int((table.log_per_year > 0).sum()),
        "symbols": len(table),
    }


def _selfcheck() -> None:
    """A saw whose turns are in the data, so the arithmetic and the learning can both be pinned."""
    # The rule and its price, on a triangle wave with a known answer.
    ramp = np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
    close = np.exp(0.02 * np.tile(ramp, 10))
    signal = 2 * np.tile(ramp, 10) - 1  # the label read perfectly: -1 at the lows, +1 at the peaks
    pos = positions(signal, 0.9, 0.9)
    assert set(np.unique(pos)) == {0.0, 1.0}
    t = trades(pos, close, fee=0.0)
    assert len(t) >= 9 and (t.leg > 0).all(), t
    assert np.isclose(t.leg.iloc[1], np.exp(0.02) - 1, atol=1e-3), "a full leg is the saw's amplitude"
    # The fee is charged on both fills and compounds with the ratio, as the oracle charges it.
    fee = 0.001
    assert np.isclose(trades(pos, close, fee).leg.iloc[1], np.exp(0.02) * (1 - fee) ** 2 - 1, atol=1e-9)
    # A fee larger than the leg turns the same perfect signal into a loss, which is the whole
    # reason this is priced in money and not in correlation.
    assert price(pos, close, 1.0, fee=0.05)["log_per_year"] < 0
    # A position still open at the end is marked out there, so buy and hold is one trade and
    # prices as what it is. This is the reading that stops "never sold" from scoring zero.
    forever = trades(np.ones(10), np.exp(np.linspace(0, 0.1, 10)), fee=0.0)
    assert len(forever) == 1 and forever.entry.iloc[0] == 0 and forever.exit.iloc[0] == 9
    assert np.isclose(forever.leg.iloc[0], np.exp(0.1) - 1)
    # But a position that opens on the last bar is not a trade at all.
    assert len(trades(np.array([0.0, 0.0, 1.0]), np.ones(3))) == 0

    # Purging: a bar whose leg closes past the cut leaves the train side.
    when = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    meta = pd.DataFrame(
        {"next_pivot": when + pd.Timedelta("2h"), "symbol": "A", "row": np.arange(10), "close": 1.0, "ret": 0.0},
        index=when,
    )
    train, test = purge(meta, pd.Timestamp("2024-01-01 06:00", tz="UTC"))
    assert len(test) == 4 and len(train) == 4, (len(train), len(test))
    assert (train.next_pivot < pd.Timestamp("2024-01-01 06:00", tz="UTC")).all()

    # Episodes are contiguous and inside one symbol — the property the fee term depends on.
    two = pd.concat(
        [
            meta.assign(symbol="A", row=np.arange(10)),
            meta.assign(symbol="B", row=np.arange(10, 20)),
        ]
    )
    rng = np.random.default_rng(0)
    ep = episodes(two, 4, 6, rng)
    assert ep.shape == (6, 4)
    assert (np.diff(ep, axis=1) == 1).all(), "an episode is consecutive bars"
    assert ((ep < 10).all(axis=1) | (ep >= 10).all(axis=1)).all(), "and never spans two symbols"

    # The sequence builder: oldest step first, and strictly causal.
    f = pd.DataFrame({c: np.arange(20.0) for c in INPUTS}, index=pd.RangeIndex(20))
    stats = normalize.fit(f)
    seq = sequences(f, stats, steps=3)
    assert seq.shape == (20, 3, len(INPUTS))
    # The first rows have no window behind them: only the steps that exist are filled, and
    # `build` drops any row that is not complete.
    assert np.isnan(seq[0, :-1]).all() and np.isfinite(seq[0, -1]).all()
    assert np.isfinite(seq[2]).all()
    assert seq[5, -1, 0] > seq[5, 0, 0], "the last step is the current bar"
    assert np.array_equal(seq[:11], sequences(f.iloc[:11], stats, steps=3), equal_nan=True), "truncation is invisible"

    # And the learning, end to end, on a toy built so the two stages cannot both be right.
    #
    # One live feature, a saw from -1 to +1. The label *is* that feature, so stage one has an easy
    # job and does it fast. The price, however, rises exactly where the feature is high — which is
    # the opposite of what the label implies, since the swing rule reads a low label as a buy. So
    # `adopt` starts the policy from a rule that loses money by construction, and the only way the
    # second stage can end positive is by overriding the supervised prior on the strength of the
    # reward alone. That is the claim this module rests on, stated as a test.
    n, per = 4000, 40
    saw = (np.arange(n) % per) / per * 2 - 1
    step = np.where(saw > 0, 0.002, -0.002)
    path = np.exp(np.cumsum(step))
    x = np.zeros((n, 2, len(INPUTS)), dtype="float32")
    x[:, :, 0] = saw[:, None]
    when = pd.date_range("2024", periods=n, freq="h", tz="UTC")
    toy = pd.DataFrame(
        {
            "target": saw,
            "next_pivot": when,
            "close": path,
            "ret": np.append(np.diff(np.log(path)), 0.0),
            "symbol": "A",
            "row": np.arange(n),
        },
        index=when,
    )
    inner, valid = toy.iloc[: int(n * 0.7)], toy.iloc[int(n * 0.7) :]
    model = fit_label(x, inner, valid, epochs=12, patience=12, batch_size=256)
    corr, _ = valid_score(model, x, valid, "label")
    assert corr > 0.8, f"the label is the one live feature and was not learned: {corr}"
    model.adopt(SHARPNESS)
    before, _ = valid_score(model, x, valid, "policy", fee=0.0005)
    assert before < 0, f"the supervised rule has to lose on this toy, or the test proves nothing: {before}"
    model = fit_policy(model, x, inner, valid, epochs=12, patience=12, steps=10, chunk=64, count=16, fee=0.0005)
    after, _ = valid_score(model, x, valid, "policy", fee=0.0005)
    assert after > 0, f"the reward has to overturn the supervised prior: {before:+.3f} -> {after:+.3f}"
    print(f"ok — label corr {corr:.3f}, policy {before:+.3f} -> {after:+.3f} log/yr on the saw")


# One-column rules, priced on the same rows as the model. Not decoration: on the four folds below
# `rsi_centered` above 0.3 nets +0.116 a year against +0.063 for the two-stage model, at a third of
# the turnover, and a strategy that cannot beat one indicator is not a strategy. The entry and exit
# levels are the ones a reader would try first and are not tuned per fold.
BASELINES = (
    ("rsi_centered", 0.3, 0.0),
    ("rsi_centered", 0.2, -0.1),
    ("close_position_in_window", 0.5, -0.2),
    ("tsi_momentum", 0.0, 0.0),
)


def baselines(meta: pd.DataFrame, tf: str, fee: float = FEE) -> pd.DataFrame:
    """Each rule in `BASELINES`, long when the column is above its entry and flat below its exit.

    Read on exactly the rows the walk-forward tested, so the comparison is not a comparison of two
    periods. The column comes off `features` again rather than out of the tensor, because the
    tensor is scaled and a threshold on a scaled column is a different rule.
    """
    rows = []
    for column, hi, lo in BASELINES:
        per = []
        for symbol, g in meta.groupby("symbol", sort=False):
            g = g.sort_index()
            column_values = features(load(symbol, tf).loc[g.index[0] : g.index[-1]], W)[column].reindex(g.index)
            state = pd.Series(np.nan, index=g.index)
            state[column_values >= hi] = 1.0
            state[column_values <= lo] = 0.0
            close = g.close.to_numpy()
            span = float((g.index[-1] - g.index[0]) / YEAR)
            got = price(state.ffill().fillna(0.0).to_numpy(), close, span, fee)
            per.append({**got, "hold": float(np.log(close[-1] / close[0]) / span)})
        d = pd.DataFrame(per)
        rows.append(
            {
                "rule": f"{column} >{hi:+.1f} / <{lo:+.1f}",
                "gross": d.gross_per_year.mean(),
                "net": d.log_per_year.mean(),
                "trades": d.trades.mean(),
                "in_market": d.in_market.mean(),
                "win_rate": d.win_rate.mean(),
                "beat_hold": int((d.log_per_year > d.hold).sum()),
            }
        )
    return pd.DataFrame(rows)


def save(path, model, tf, window, steps, params, stage, keep=None, cal=None):
    """The weights and everything needed to feed them. A model without its inputs is not a model.

    The per-symbol scaler is deliberately *not* in here: it is fitted on the symbol the model is
    being run on, from that symbol's own history, which is what `chart` does when it draws a pair
    the store has never seen.
    """
    torch.save(
        {
            "state": model.state_dict(),
            "inputs": list(INPUTS if keep is None else keep),
            "timeframe": tf,
            "window": window,
            "steps": steps,
            "enter": None if params is None else params[0],
            "exit": None if params is None else params[1],
            "sign": None if params is None else params[2],
            "stage": stage,
            "test_start": TEST_START,
            # The map back onto the label's range, fitted on train. Stored rather than recomputed:
            # the chart has no train period of its own and a calibration fitted on the window on
            # screen would be a different statement in every screenshot.
            "calibration": cal,
        },
        path,
    )


def restore(path: Path = CHECKPOINT) -> tuple[Net, dict]:
    """The saved model, on whatever device *this* machine has.

    `map_location=DEVICE` and not the default, which is what the file was written on. A checkpoint
    carries the device of the process that saved it, so one trained on a Mac says `mps` and
    `torch.load` on a Linux box raises `Storage device not recognized: mps` before a single weight
    is read. That is exactly the deployed case — the page is the artefact this project ships, the
    training runs on a laptop with MPS — so the default is not a default here, it is a crash.
    """
    checkpoint = torch.load(path, weights_only=False, map_location=DEVICE)
    model = Net(len(checkpoint["inputs"])).to(DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


# Below this many bars a window is not worth scoring: the encoder reads 24 of them, the features
# another 24 behind that, and the leg state's volatility window four times that again. Measured
# from the columns rather than guessed — `state` is NaN until 4*W bars have passed.
MIN_BARS = 6 * W + STEPS


def live_scaler(f: pd.DataFrame) -> pd.DataFrame:
    """`normalize.fit` on a window short enough that some columns may not move.

    `fit` raises on a column with no dispersion, and where it is used that is right: a dead column
    in the dataset is a bug, and scaling it away quietly is how a bug becomes a number. On a chart
    it is not a bug. Over thirty days of 4h bars the last confirmed pivot may never change kind, so
    `leg_kind_24` is a constant for the whole window — correctly, and the model should see a column
    that does not move. A constant divided by one is still a constant and the clip keeps it finite.

    The rest of the scale is the pair's own history, which is what training does too: the model was
    fitted on per-symbol quartiles precisely so that it reads a state and not a price level. On a
    short window those quartiles are estimated from little, which is the reason for `MIN_BARS`.
    """
    q = f.quantile([0.25, 0.5, 0.75])
    return pd.DataFrame({"center": q.loc[0.5], "scale": (q.loc[0.75] - q.loc[0.25]).replace(0, 1.0)})


def predict_frame(model: Net, checkpoint: dict, bars: pd.DataFrame) -> pd.DataFrame:
    """`label`, `logit` and `position` at every bar of `bars`, which must be the model's timeframe.

    The scaler is fitted on this series' own history, which is the honest thing to do for a pair
    the training store does not carry and the only thing possible for one it does not. Rows with
    no window behind them come back NaN rather than as a number nothing supports.
    """
    keep = checkpoint["inputs"]
    f = inputs(bars, checkpoint["window"], keep).dropna()
    if len(f) < MIN_BARS:
        return pd.DataFrame(columns=["label", "logit", "position"], index=bars.index, dtype=float)
    x = sequences(f, live_scaler(f[keep]), checkpoint["steps"], keep)
    ready = np.isfinite(x).all(axis=(1, 2))
    out = pd.DataFrame(np.nan, index=bars.index, columns=["label", "logit", "position"])
    if not ready.any():
        return out
    label, logit = outputs(model, x, np.flatnonzero(ready))
    at = f.index[ready]
    if checkpoint.get("calibration") is not None:
        label = calibrate(label, checkpoint["calibration"])
    out.loc[at, "label"] = label
    out.loc[at, "logit"] = logit
    signal = logit if checkpoint["stage"] in ("policy", "both") else label
    read = None if checkpoint["enter"] is None else (checkpoint["enter"], checkpoint["exit"], checkpoint["sign"])
    out.loc[at, "position"] = book(signal, read)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default=TF)
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--stage", choices=["label", "policy", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="initialisations per fold, averaged before the rule")
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also price the one-column rules on the same rows")
    ap.add_argument("--band", action="store_true", help="read the policy score through a band instead of at zero")
    ap.add_argument("--cache", type=Path, default=TENSOR)
    ap.add_argument(
        "--inputs",
        choices=["basic", "full"],
        default="full",
        help="basic: the 29 candidates plus the leg state at N=24. full: adds the faster scales and exhaustion",
    )
    ap.add_argument("--save", type=Path, nargs="?", const=CHECKPOINT, help="also fit one model on train and save it")
    args = ap.parse_args()

    _selfcheck()
    keep = BASIC if args.inputs == "basic" else INPUTS
    path = args.cache.with_name(f"{args.cache.name}-{args.timeframe}-{args.inputs}")
    x, meta = cached(path, args.symbols, tf=args.timeframe, since=args.since, window=W, steps=STEPS, keep=keep)
    print(
        f"{len(meta):,} rows x {STEPS} steps x {len(keep)} inputs on {args.timeframe}, "
        f"{meta.symbol.nunique()} symbols, {meta.index.min():%Y-%m-%d} to {meta.index.max():%Y-%m-%d}, {DEVICE}"
    )
    print(f"{args.folds} folds from {args.test_start}, stage {args.stage}, fee {args.fee * 100:.2f}% per side\n")

    out = walk_forward(
        x,
        meta,
        args.test_start,
        args.folds,
        args.stage,
        args.seed,
        args.fee,
        not args.verbose,
        args.seeds,
        args.band,
        epochs=args.epochs,
    )
    table = by_symbol(out["pos"], out["meta"], args.fee)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    total = summarise(table)
    print("\n" + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in total.items()))

    if args.baseline:
        print("\nthe same rows, read by one column and a threshold\n")
        print(baselines(out["meta"], args.timeframe, args.fee).round(3).to_string(index=False))

    written = STORE / f"pos-swing-{args.timeframe}-{args.stage}.parquet"
    out["meta"].assign(position=out["pos"].to_numpy())[["symbol", "close", "position"]].to_parquet(written)
    print(f"\npositions written to {written}")

    if args.save:
        train, _ = purge(meta, pd.Timestamp(args.test_start, tz="UTC"))
        inner, valid = purge(train, train.index.min() + (train.index.max() - train.index.min()) * (1 - VALID_FRACTION))
        model = fit_label(x, inner, valid, args.seed, epochs=args.epochs, quiet=not args.verbose)
        fitted, _ = outputs(model, x, inner.row.to_numpy())
        cal = calibration(fitted, inner.target.to_numpy())
        if args.stage in ("policy", "both"):
            model = fit_policy(model, x, inner, valid, args.seed, fee=args.fee, quiet=not args.verbose)
        params = None
        if args.band or args.stage == "label":
            order = ordered(valid if args.stage == "label" else train)
            label, logit = outputs(model, x, order.row.to_numpy())
            params = choose(logit if args.stage != "label" else label, order, fee=args.fee)
        save(args.save, model, args.timeframe, W, STEPS, params, args.stage, keep, cal)
        print(f"fitted on {len(inner):,} rows to {args.test_start} and saved to {args.save}")


if __name__ == "__main__":
    main()
