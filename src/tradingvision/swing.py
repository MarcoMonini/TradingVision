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

So the model was trained twice: first on the label, which gives it the structure of a leg, then
**directly on the money** — Moody and Saffell's direct reinforcement, the net P&L of a
differentiable position with the fee inside the reward. The second stage left on 2026-10-04. On
4h bars it netted +0.063 a year against +0.116 for `rsi_centered` alone, on v2 it settled on no
trade in all four folds, and no rule the study trades reads it: `strategy` and `detect` read the
label head. `OLD/README.md` has its numbers and the git tag `archive-policy` the code.

What is left is the first stage — a Huber on the label, stopped on the validation correlation —
and a band on its output, chosen on the validation tail with the fee inside the criterion.

    uv run python -m tradingvision.swing --timeframe 1h --folds 4 --save

    uv run python -m tradingvision.swing --timeframe 15m --window 12 --smoothing 0.5 --inputs reduced \
        --steps 48 --test-start 2025-06 --save data/swing-v2.pt

Swing Leg Position v2: the swing label back, with pivots *and* features at 12 and a time weight of
0.5. It learns the label — rho 0.644 out of sample — and 0.596 of that is `rsi_centered` at 12 on
the same rows; against the market-neutral forward return it reads -0.026 at 4, 12 and 48 bars. The
long-only band nets -0.275 a year on the thirteen tradable pairs against -0.263 for holding, and
the archived policy stage, paid in money, settled on no trade in all four folds. At 12 the median
leg is 2.4% against a 0.5% round trip (Alpaca's, the fee then) and the oracle that fills 12 bars
after each pivot nets +0.11: the fee decides before the model does.

The swing label is the only one this module trains. Two predictive targets ran on the same encoder
and read nothing of the price — `move_balance` alone (Rank IC +0.0015, t 0.27) and beside the swing
label as an auxiliary head (rho -0.027..+0.046 across folds) — and left on 2026-10-03 with the rest
of the predictive work: `OLD/README.md` has the numbers, and the git tag `archive-predictive` the
code that produced them.

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
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from tradingvision import legs, normalize
from tradingvision.data.binance import STORE, SYMBOLS, TRADABLE, ends, load
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import SMOOTHING, swing_leg_target
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


def scales(window: int = W) -> tuple[int, int, int]:
    """The leg state's three widths for a feature window: a quarter of it, half of it, all of it.

    The confirmation lag *is* the width, and the lag is the binding constraint of the whole problem
    — the oracle run at its own confirmation lag keeps 0.4 log a year of the 5.8 it makes with
    hindsight — so reading the structure at 6 bars as well as at 24 is not a richer feature set, it
    is a faster one. The ratios are the design and the numbers follow the window: (6, 12, 24) at
    the calibrated 24, (3, 6, 12) at the 12 of the v2 model, where a fixed (6, 12, W) would have
    put two identical columns side by side.
    """
    return (window // 4, window // 2, window)


def columns(window: int = W) -> list[str]:
    """Features, leg state at each scale, and the exhaustion columns — `INPUTS` at `window`. The
    ridge probe puts features and leg state at 0.623 against 0.577 and 0.584 alone: they are not
    the same information."""
    return list(COLUMNS) + [f"{c}_{w}" for w in scales(window) for c in legs.STATE] + list(legs.EXHAUSTION)


SCALES = scales(W)
BASIC = list(COLUMNS) + [f"{c}_{W}" for c in legs.STATE]
INPUTS = columns(W)
# The input set of the next model: 15 of the 65, cut by hand on 2026-09-27 in seven passes over the
# rank-correlation map of INPUTS on 15m bars, train period only (swing_leg_pipeline.html, lesson 6,
# where every column taken out is listed with the pass that took it). Out: the volume family and
# every column built on volume, all of the leg state but `signed_move`, the distance-from-the-mean
# group whole, the redundant volatility and trend columns. No pair of the fifteen reaches |rho| 0.8;
# the highest is distance_from_window_high_pct / tsi_momentum at 0.65.
#
# Decided, not measured: no label has been read to choose it. Step 2's selection lost 0.004 of Rank
# IC to the full set once it reached a recurrent net, and dropped `rsi_centered`, one of the two
# strongest columns near the pivot — which this cut drops too. An ablation against the full set,
# judged on the forward return, is what keeps or undoes it. A parameter beside INPUTS, not a deletion.
REDUCED = [
    "log_return",
    "realized_volatility",
    "volatility_expansion",
    "upper_wick_pct",
    "lower_wick_pct",
    "distance_from_window_high_pct",
    "distance_from_window_low_pct",
    "distance_from_psar_pct",
    "adx_trend_strength",
    "tsi_momentum",
    "signed_move_6",
    "signed_move_12",
    "signed_move_24",
    "divergence",
    "deceleration",
]


def reduced(window: int = W) -> list[str]:
    """`REDUCED` at another feature window. The three `signed_move` columns are the leg state at
    `scales(window)`, so they move with it by position — the fastest, the middle and the window's
    own — and the cut keeps its meaning; every other column is named without a width."""
    moved = {f"signed_move_{a}": f"signed_move_{b}" for a, b in zip(SCALES, scales(window))}
    return [moved.get(c, c) for c in REDUCED]


# Bumped whenever `build` writes something different for the same arguments, so a cache from the
# code before is refused by its stamp rather than read back. 2: purging on `legs.label_reach`
# instead of the next pivot, and the scaler strictly before the cut instead of through its month.
BUILD = 2

H = 48
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH = 512
VALID_FRACTION = 0.2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = STORE / "swing.pt"
TENSOR = STORE / "swing"


def inputs(bars: pd.DataFrame, window: int = W, keep: list[str] | None = None) -> pd.DataFrame:
    """The model's columns for one symbol: the candidates, the leg state at each scale, exhaustion."""
    parts = [features(bars, window)[COLUMNS]]
    parts += [legs.state(bars.close, w).add_suffix(f"_{w}") for w in scales(window)]
    parts.append(legs.exhaustion(bars, window))
    return pd.concat(parts, axis=1)[columns(window) if keep is None else keep]


def frame(
    symbol: str,
    tf: str = TF,
    since: str = SINCE,
    window: int = W,
    keep: list[str] | None = None,
    smoothing: float = SMOOTHING,
):
    """One symbol's rows: inputs, label, the next bar's return, the close, the purging horizon.

    `window` is both the features' window and the label's pivots, on purpose: the v2 model moves
    the two together (12), where the archived `legsweep` moved the label alone. `smoothing` is the
    label's time weight and nothing else reads it.

    The label's pivots and the leg state's pivots are the same turns read from two sides — the
    centred frame for what is being predicted, `legs.confirmed` for what was knowable. Mixing them
    up is the one mistake that would make every number here meaningless, so they are built by
    different functions in different modules and meet only in this frame.

    The column `next_pivot` keeps its name because `purge` reads it, and holds `legs.label_reach`,
    which is past the next pivot — see there for why.
    """
    bars = load(symbol, tf).loc[since:]
    close = bars.close
    out = inputs(bars, window, keep)
    out["target"] = swing_leg_target(close, find_pivots(close, window), smoothing=smoothing)
    out["next_pivot"] = legs.label_reach(close, window)
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

    Strictly before the cut, by comparison and not by `.loc[:before]`: `before` is a month, and a
    partial-string slice ends at the *end* of that month — so until 2026-09-28 the quartiles were
    fitted through the whole of the first test month.
    """
    return normalize.fit(f.loc[f.index < pd.Timestamp(before, tz="UTC"), INPUTS if keep is None else keep])


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


def windows(z: np.ndarray, steps: int) -> np.ndarray:
    """`(len(z) - steps + 1, steps, width)` over the scaled rows `z`, as a view — nothing is copied.

    Window `i` is `z[i : i + steps]`, oldest first, so it ends at row `i + steps - 1`: the same
    thing `sequences` materialises. Materialised it no longer fits: 96 steps of 15 columns over
    three million 15m rows is 17 GB on disk, against 180 MB for the rows it repeats. A batch indexes
    the view and copies only its own windows. Windows that straddle two symbols exist in the view
    and no `row` points at one — see `build`.
    """
    return sliding_window_view(z, steps, axis=0).transpose(0, 2, 1)


def build(
    symbols,
    tf: str = TF,
    since: str = SINCE,
    window: int = W,
    steps: int = STEPS,
    keep=None,
    before: str = TEST_START,
    smoothing: float = SMOOTHING,
):
    """`(rows, meta, stats)` over every symbol — the scaled rows positional, the meta indexed by
    time, and each symbol's scaler.

    `rows` is one line per bar, not one window per bar: `windows(rows, steps)[meta.row]` is the
    tensor. A symbol's rows are contiguous, so the window of its `k`-th row starts `steps - 1`
    before it and never reaches into the symbol before; its first `steps - 1` rows have no window
    and get no `row`, which is what `sequences` expresses with NaN.

    Rows are grouped by symbol and ordered in time inside each group, so a fold's rows of one
    symbol are one contiguous stretch. `before` is where the scaler stops reading, the first fold's
    cut. `stats` is returned so `save` can ship it: the model was measured on inputs scaled by
    exactly these quartiles.
    """
    blocks, metas, stats, at = [], [], {}, 0
    for symbol in symbols:
        f = frame(symbol, tf, since, window, keep, smoothing)
        if len(f) < steps + 100:
            continue
        cols = INPUTS if keep is None else keep
        stats[symbol] = scaler(f, before, keep)
        z = normalize.apply(f[cols], stats[symbol]).to_numpy("float32")
        if not np.isfinite(z).all():
            raise ValueError(f"{symbol}: a scaled input is not finite, and a window over it would be NaN")
        meta = f[["target", "next_pivot", "close", "ret"]].iloc[steps - 1 :].copy()
        meta["symbol"] = symbol
        meta["row"] = np.arange(at, at + len(meta))
        blocks.append(z)
        metas.append(meta)
        at += len(z)
    return np.concatenate(blocks), pd.concat(metas), pd.concat(stats, names=["symbol", "column"])


def cached(path: Path, symbols: list[str], **params):
    """`build` on disk with a stamp of what produced it — the contract of the archived `dataset.cached`.

    Returns `(windows, meta, stats)`: the rows are stored and the windows are a view over them.
    """
    # Appended and never `with_suffix`: a tag like `-m0.50` has a dot in it, and `with_suffix` would
    # take `.50` for an extension and put every smoothing of the same window under one `-m0`.
    stamp, npy, pq, scales_at = (
        path.with_name(path.name + end) for end in (".json", ".npy", ".parquet", "-scalers.parquet")
    )
    # `keep` travels as `inputs` and not as itself: it is the column list, and recording it twice
    # would make two stamps of the same build disagree on nothing.
    written = dict(
        {k: v for k, v in params.items() if k != "keep"},
        symbols=sorted(symbols),
        inputs=list(params.get("keep") or INPUTS),
        # Where the store ends: a store refetched since would hold other rows under the same arguments.
        store_ends=ends(sorted(symbols)),
        build=BUILD,
    )
    if npy.exists():
        if not stamp.exists():
            raise SystemExit(f"{npy} has no {stamp.name} recording how it was built — delete it")
        if (was := json.loads(stamp.read_text())) != written:
            raise SystemExit(f"{npy} was built with {was}, not {written} — delete it")
        rows = np.load(npy, mmap_mode="r")
        if rows.ndim != 2:
            raise SystemExit(f"{npy} holds windows, not rows — it predates `windows`, delete it")
        return windows(rows, params["steps"]), pd.read_parquet(pq), pd.read_parquet(scales_at)
    rows, meta, stats = build(symbols, **params)
    np.save(npy, rows)
    meta.to_parquet(pq)
    stats.to_parquet(scales_at)
    stamp.write_text(json.dumps(written, indent=2, sort_keys=True))
    return windows(np.load(npy, mmap_mode="r"), params["steps"]), meta, stats


def purge(meta: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, test)` around `cut`, train purged of every bar whose leg closes past it.

    The archived `split.temporal` said the same thing on a (timestamp, symbol) index; this one
    reads a plain time index with a `symbol` column, which is the shape the rows are kept in.
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
    """One GRU and one head: where the bar sits along its leg."""

    def __init__(self, width: int, hidden: int = H, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.gru(x)[1][-1])).squeeze(-1)


def outputs(model: Net, x, rows: np.ndarray, batch: int = 4096) -> np.ndarray:
    """The head over `rows`, in evaluation mode."""
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.from_numpy(np.asarray(x[at])).to(DEVICE)).cpu().numpy()
                for at in np.array_split(rows, max(1, len(rows) // batch))
            ]
        )


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


def by_symbol(pos: pd.Series, meta: pd.DataFrame, fee: float = FEE, window: int = W) -> pd.DataFrame:
    """`price` per symbol, plus the oracle and buy-and-hold on exactly the bars it traded.

    The oracle is recomputed here and never carried in: it has to see the same slice, the same
    close series and the same fee, or the ratio between the two is a comparison of two periods.
    `window` is the label's, so the oracle trades the legs the model was taught.
    """
    rows = []
    for symbol, g in meta.assign(pos=pos.to_numpy()).groupby("symbol", sort=False):
        g = g.sort_index()
        close = g.close.to_numpy()
        span = float((g.index[-1] - g.index[0]) / YEAR)
        got = price(g.pos.to_numpy(), close, span, fee)
        piv = find_pivots(g.close, window)
        oracle = oracle_run(g.close, window, fee, piv, lag=0)
        # The same oracle filling `window` bars later — the earliest a pivot of a centred window
        # can be *known* to anyone. It is the only benchmark on this page that a causal rule could
        # in principle reach, and it is 6 to 8% of the hindsight one at 24. Everything that makes
        # the first number enormous is the `window` bars of future it reads.
        reachable = oracle_run(g.close, window, fee, piv, lag=window)
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
    scale a model puts its output on. Asymmetric, because the two ends are not the same decision:
    entering late costs the rest of one leg and leaving late costs the whole of the next.

    **The sign is searched too, and that is not a hedge.** The rule this label implies is "a low
    prediction is a buy" — the bar sits near a pivot low and the leg runs up from there — and over
    2023-26 that reading has a *negative* gross on this panel while its mirror is positive: at the
    scale of a 24-bar window these markets continue more often than they turn. A band that could
    only ever read one way would report that as a failure of the model instead of as a property of
    the market, so the direction is a parameter, picked on the same validation rows as the width
    and reported with the fold.

    The criterion is net minus buy and hold, and not the absolute return. Measured the other way,
    the answer was "buy and hold": the validation tail of a crypto train period rises, a long-only
    rule that never sells earns that rise, and nothing in an absolute criterion prefers a strategy
    to a beta.
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


# ---------------------------------------------------------------- the fit


def rows_of(meta: pd.DataFrame) -> np.ndarray:
    """The tensor positions of these rows, in time order inside each symbol."""
    return np.concatenate([g.sort_index().row.to_numpy() for _, g in meta.groupby("symbol", sort=False)])


def ordered(meta: pd.DataFrame) -> pd.DataFrame:
    """The same rows in the order `rows_of` returns them, so a prediction lines up with its meta."""
    return pd.concat([g.sort_index() for _, g in meta.groupby("symbol", sort=False)])


def valid_score(model: Net, x, meta: pd.DataFrame) -> float:
    """The correlation with the label on a fold's validation tail — the number early stopping reads."""
    order = ordered(meta)
    return float(pd.Series(outputs(model, x, order.row.to_numpy())).corr(pd.Series(order.target.to_numpy())))


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
    """A Huber on `swing_leg_target`, stopped on the validation correlation.

    This fit is not the strategy and is not judged as one. It puts the structure of a leg into the
    encoder — where the turns are, how long they run, what a significant one looks like — and the
    rules of `strategy` and `detect` read its output. Fitting this label harder does not make a rule
    more profitable, because the residual concentrates at the turns: see the module docstring.
    """
    torch.manual_seed(seed)
    model = Net(x.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Delta is the median of |target| on this fold's train side, the criterion the project fixes
    # every Huber with: the label lives in [-1, 1] here, so a delta carried from another one would
    # be a squared loss in disguise.
    delta = float(train.target.abs().median())
    loss_fn = nn.HuberLoss(delta=delta)
    at = train.row.to_numpy()
    y = torch.from_numpy(train.target.to_numpy("float32"))
    rng = np.random.default_rng(seed)
    best, state, since = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(at)), max(1, len(at) // batch_size)):
            opt.zero_grad()
            label = model(torch.from_numpy(np.asarray(x[at[batch]])).to(DEVICE))
            loss = loss_fn(label, y[batch].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        score = valid_score(model, x, valid)
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


# ---------------------------------------------------------------- walk forward


def book(signal: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
    """The position a score implies through the band and the sign `choose` picked."""
    enter, exit, sign = params
    return positions(sign * signal, enter, exit)


def walk_forward(
    x,
    meta: pd.DataFrame,
    start: str = TEST_START,
    n: int = FOLDS,
    seed: int = 0,
    fee: float = FEE,
    quiet: bool = True,
    seeds: int = 1,
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
    pred = np.zeros(len(x))
    seen = np.zeros(len(x), dtype=bool)
    rows = []
    for i, (train, test) in enumerate(folds(meta, start, n), 1):
        inner, valid = purge(train, train.index.min() + (train.index.max() - train.index.min()) * (1 - VALID_FRACTION))
        epochs = {k: v for k, v in kw.items() if k == "epochs"}
        models = [fit_label(x, inner, valid, seed + extra, quiet=quiet, **epochs) for extra in range(seeds)]

        def score(rows: np.ndarray) -> np.ndarray:
            """The seeds averaged. A single initialisation of a model this weak is mostly its own
            noise — the project reports mean +- std over five seeds for exactly that reason — and
            averaging the scores before the rule reads them is the cheapest variance there is."""
            return np.mean([outputs(m, x, rows) for m in models], axis=0)

        # The band is picked on the validation tail, the rows early stopping did not train on.
        order = ordered(valid)
        params = choose(score(order.row.to_numpy()), order, fee=fee)
        order = ordered(test)
        use = score(order.row.to_numpy())
        pred[order.row.to_numpy()] = use
        for _, g in order.assign(s=use).groupby("symbol", sort=False):
            pos[g.row.to_numpy()] = book(g.s.to_numpy(), params)
            seen[g.row.to_numpy()] = True
        rows.append(
            {
                "fold": i,
                "train": len(inner),
                "valid": len(valid),
                "test": len(test),
                "rule": f"band {params[0]:+.2f}/{params[1]:+.2f} x{params[2]:+.0f}",
            }
        )
    test_meta = meta[seen[meta.row.to_numpy()]]
    return {
        "folds": pd.DataFrame(rows),
        "pos": pd.Series(pos[test_meta.row.to_numpy()], index=test_meta.index),
        "pred": pd.Series(pred[test_meta.row.to_numpy()], index=test_meta.index),
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
    # A misspelt name in the reduced set would be a KeyError at build time at best, a silently
    # narrower model at worst; and INPUTS' order is the tensor's.
    assert REDUCED == [c for c in INPUTS if c in REDUCED] and len(set(REDUCED)) == 15, REDUCED
    # At another window the cut is the same cut: the leg state follows the window, nothing else moves.
    assert reduced() == REDUCED and reduced(12)[10:13] == ["signed_move_3", "signed_move_6", "signed_move_12"]
    assert reduced(12) == [c for c in columns(12) if c in reduced(12)], "the tensor's order is INPUTS' order"
    # Every input at the v2 window is causal as a whole, not only column family by column family:
    # truncating the candles after bar `t` moves no row at or before it.
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, 900))))
    wick = np.abs(rng.normal(0, 0.002, (2, 900)))
    candles = pd.DataFrame(
        {
            "open": close.shift().fillna(100.0),
            "high": close * (1 + wick[0]),
            "low": close * (1 - wick[1]),
            "close": close,
            "volume": rng.uniform(50, 150, 900),
        }
    )
    candles["high"] = candles[["open", "high", "close"]].max(axis=1)
    candles["low"] = candles[["open", "low", "close"]].min(axis=1)
    whole = inputs(candles, 12, reduced(12))
    assert whole.iloc[200:].notna().all().all(), "the v2 inputs never fill"
    for t in (300, 650):
        assert np.allclose(inputs(candles.iloc[: t + 1], 12, reduced(12)), whole.iloc[: t + 1], equal_nan=True), t
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
    # The view `build` stores is the same tensor: window i is the sequence of row i + steps - 1.
    view = windows(normalize.apply(f, stats).to_numpy("float32"), 3)
    assert view.shape == (18, 3, len(INPUTS)) and np.array_equal(view, seq[2:]), "the view is the tensor"

    # And the learning, end to end, on a toy whose label is its one live feature, a saw.
    n, per = 4000, 40
    saw = (np.arange(n) % per) / per * 2 - 1
    path = np.exp(np.cumsum(np.where(saw > 0, 0.002, -0.002)))
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
    corr = valid_score(model, x, valid)
    assert corr > 0.8, f"the label is the one live feature and was not learned: {corr}"
    print(f"ok — label corr {corr:.3f} on the saw")


# One-column rules, priced on the same rows as the model. Not decoration: on the four folds below
# `rsi_centered` above 0.3 nets +0.116 a year against +0.063 for the archived two-stage model, at a
# third of its turnover, and a strategy that cannot beat one indicator is not a strategy. The entry and exit
# levels are the ones a reader would try first and are not tuned per fold.
BASELINES = (
    ("rsi_centered", 0.3, 0.0),
    ("rsi_centered", 0.2, -0.1),
    ("close_position_in_window", 0.5, -0.2),
    ("tsi_momentum", 0.0, 0.0),
)


def baselines(meta: pd.DataFrame, tf: str, fee: float = FEE, window: int = W) -> pd.DataFrame:
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
            column_values = features(load(symbol, tf).loc[g.index[0] : g.index[-1]], window)[column].reindex(g.index)
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


def save(
    path,
    model,
    tf,
    window,
    steps,
    params,
    keep=None,
    cal=None,
    test_start=TEST_START,
    smoothing=SMOOTHING,
    stats=None,
):
    """The weights and everything needed to feed them. A model without its inputs is not a model.

    `stats` is each training symbol's scaler, fitted before `test_start` — the quartiles every
    number of the walk-forward was measured through. Shipped so `predict_frame` feeds a known pair
    the inputs the model was scored on. Before it, the page refitted the scaler on the window on
    screen: thirty days of quartiles in place of four years, re-centred on the local regime, and
    fitted on bars *after* the one being predicted. A pair the store has never seen still gets
    that fallback, and only that one.
    """
    # CPU whatever device trained it: a checkpoint written from `mps` names that device inside the
    # pickle, and unpickling it on a host with no Metal — the deployed Streamlit page — dies before
    # `restore`'s `map_location` is reached, because the storage itself cannot be built.
    torch.save(
        {
            "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "inputs": list(INPUTS if keep is None else keep),
            "timeframe": tf,
            "window": window,
            "steps": steps,
            "enter": params[0],
            "exit": params[1],
            "sign": params[2],
            # Always the swing label now. Written all the same: the checkpoints from before the
            # cleanup carry the name, and the page reads a card whatever wrote it.
            "label": "swing",
            # The label's time weight; `window` above is its pivots as well as the features'.
            "smoothing": smoothing,
            "scalers": None if stats is None else {s: g.droplevel(0) for s, g in stats.groupby(level=0)},
            "test_start": test_start,
            # The map back onto the label's range, fitted on train. Stored rather than recomputed:
            # the chart has no train period of its own and a calibration fitted on the window on
            # screen would be a different statement in every screenshot.
            "calibration": cal,
        },
        path,
    )


def restore(path: Path = CHECKPOINT) -> tuple[Net, dict]:
    # `map_location` is what loads a checkpoint trained on `mps` where there is no Metal; the
    # files written before `save` went device-free still carry the device of every storage.
    checkpoint = torch.load(path, weights_only=False, map_location=DEVICE)
    model = Net(len(checkpoint["inputs"])).to(DEVICE)
    # The checkpoints written before the policy stage left carry its head, untrained under a label
    # stage. Nothing reads it.
    model.load_state_dict({k: v for k, v in checkpoint["state"].items() if not k.startswith("policy.")})
    model.eval()
    return model, checkpoint


def live_scaler(f: pd.DataFrame) -> pd.DataFrame:
    """`normalize.fit` on a window short enough that some columns may not move.

    `fit` raises on a column with no dispersion, and where it is used that is right: a dead column
    in the dataset is a bug, and scaling it away quietly is how a bug becomes a number. On a chart
    it is not a bug. Over thirty days of 4h bars the last confirmed pivot may never change kind, so
    `leg_kind_24` is a constant for the whole window — correctly, and the model should see a column
    that does not move. A constant divided by one is still a constant and the clip keeps it finite.

    The rest of the scale is the pair's own history, which is what training does too: the model was
    fitted on per-symbol quartiles precisely so that it reads a state and not a price level. On a
    short window those quartiles are estimated from little, which is why `predict_frame` asks for
    six windows of rows before it.
    """
    q = f.quantile([0.25, 0.5, 0.75])
    return pd.DataFrame({"center": q.loc[0.5], "scale": (q.loc[0.75] - q.loc[0.25]).replace(0, 1.0)})


def predict_frame(model: Net, checkpoint: dict, bars: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """`label`, `raw` and `position` at every bar of `bars`, which must be the model's timeframe.

    `label` is the head's output mapped onto the label's +-1 by the calibration fitted on train;
    `raw` is the same output before it, the units every rule of `strategy` and `detect` was measured
    in (the walk-forward writes it as `pred`). The map stretches it: 0.40 raw is 0.575 calibrated.

    `symbol` is the store's name for the pair (`BTC`). When the checkpoint carries that
    symbol's training scaler the inputs are scaled by it, exactly as the walk-forward scaled them;
    otherwise the scaler is fitted on this series' own history, the only thing possible for a pair
    the training store does not carry. Rows with no window behind them come back NaN rather than as
    a number nothing supports.
    """
    keep = checkpoint["inputs"]
    steps = checkpoint["steps"]
    f = inputs(bars, checkpoint["window"], keep).dropna()
    out = pd.DataFrame(np.nan, index=bars.index, columns=["label", "raw", "position"])
    trained = (checkpoint.get("scalers") or {}).get(symbol)
    # The live fallback estimates quartiles from the window itself, so it needs enough of one;
    # the shipped scaler needs nothing but a full window of steps.
    if len(f) < (steps if trained is not None else 6 * checkpoint["window"] + steps):
        return out
    z = normalize.apply(f[keep], trained if trained is not None else live_scaler(f[keep])).to_numpy("float32")
    # A view over the rows and not `sequences`: a year of 15m bars at 48 steps is 100 MB copied.
    x = windows(z, steps)
    label = outputs(model, x, np.arange(len(x)))
    at = f.index[steps - 1 :]
    # The rule reads the head's raw output, before the calibration: `choose` picked its band on the
    # raw output, and a band in those units laid on the calibrated line — which spans the label's
    # +-1 and not the head's +-0.4 — is another rule. Until 2026-09-28 a checkpoint was booked on
    # the calibrated line.
    out.loc[at, "position"] = book(label, (checkpoint["enter"], checkpoint["exit"], checkpoint["sign"]))
    out.loc[at, "raw"] = label
    if checkpoint.get("calibration") is not None:
        label = calibrate(label, checkpoint["calibration"])
    out.loc[at, "label"] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default=TF)
    ap.add_argument("--since", default=SINCE)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="initialisations per fold, averaged before the rule")
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also price the one-column rules on the same rows")
    ap.add_argument("--cache", type=Path, default=TENSOR)
    ap.add_argument(
        "--inputs",
        choices=["basic", "full", "reduced"],
        default="full",
        help="basic: the 29 candidates plus the leg state at N=24. full: adds the faster scales and exhaustion. "
        "reduced: the 15 of REDUCED",
    )
    ap.add_argument("--steps", type=int, default=STEPS, help="bars of history the encoder reads")
    ap.add_argument(
        "--window",
        type=int,
        default=W,
        help="the label's pivot window, and the features' window with it — every column that derives from one",
    )
    ap.add_argument("--smoothing", type=float, default=SMOOTHING, help="the swing label's time weight")
    ap.add_argument("--save", type=Path, nargs="?", const=CHECKPOINT, help="also fit one model on train and save it")
    args = ap.parse_args()

    _selfcheck()
    w = args.window
    keep = {
        "basic": list(COLUMNS) + [f"{c}_{w}" for c in legs.STATE],
        "full": columns(w),
        "reduced": reduced(w),
    }[args.inputs]
    # Every choice that changes the tensor is in its name, so two runs never fight over one file.
    # The window and the smoothing only when they move, so the files already on disk keep theirs.
    tag = f"{args.timeframe}-{args.inputs}-s{args.steps}-t{args.test_start}"
    tag += f"-w{w}" if w != W else ""
    tag += f"-m{args.smoothing:.2f}" if args.smoothing != SMOOTHING else ""
    path = args.cache.with_name(f"{args.cache.name}-{tag}")
    x, meta, stats = cached(
        path,
        args.symbols,
        tf=args.timeframe,
        since=args.since,
        window=w,
        steps=args.steps,
        keep=keep,
        before=args.test_start,
        smoothing=args.smoothing,
    )
    print(
        f"{len(meta):,} rows x {args.steps} steps x {len(keep)} inputs on {args.timeframe}, "
        f"{meta.symbol.nunique()} symbols, {meta.index.min():%Y-%m-%d} to {meta.index.max():%Y-%m-%d}, {DEVICE}"
    )
    print(f"{args.folds} folds from {args.test_start}, fee {args.fee * 100:.2f}% per side\n")

    out = walk_forward(
        x,
        meta,
        args.test_start,
        args.folds,
        args.seed,
        args.fee,
        not args.verbose,
        args.seeds,
        epochs=args.epochs,
    )
    table = by_symbol(out["pos"], out["meta"], args.fee, w)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    total = summarise(table)
    print("\n" + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in total.items()))
    # The line to quote: money is read on the pairs the venue lists, not on the ones it only learns from.
    tradable = table[table.symbol.isin(TRADABLE)]
    if 0 < len(tradable) < len(table):
        print(f"\non the {len(tradable)} tradable pairs\n")
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summarise(tradable).items()))

    if args.baseline:
        print("\nthe same rows, read by one column and a threshold\n")
        print(baselines(out["meta"], args.timeframe, args.fee, w).round(3).to_string(index=False))

    # `-label` kept from when a stage was chosen: `strategy.PRED` and the files on disk carry it.
    written = STORE / f"pos-swing-{tag}-label.parquet"
    got = out["meta"].assign(pred=out["pred"].to_numpy(), position=out["pos"].to_numpy())
    got[["symbol", "close", "target", "pred", "position"]].to_parquet(written)
    print(f"\npredictions and positions written to {written}")

    if args.save:
        train, _ = purge(meta, pd.Timestamp(args.test_start, tz="UTC"))
        inner, valid = purge(train, train.index.min() + (train.index.max() - train.index.min()) * (1 - VALID_FRACTION))
        model = fit_label(x, inner, valid, args.seed, epochs=args.epochs, quiet=not args.verbose)
        cal = calibration(outputs(model, x, inner.row.to_numpy()), inner.target.to_numpy())
        order = ordered(valid)
        params = choose(outputs(model, x, order.row.to_numpy()), order, fee=args.fee)
        save(
            args.save,
            model,
            args.timeframe,
            w,
            args.steps,
            params,
            keep,
            cal,
            args.test_start,
            args.smoothing,
            stats,
        )
        print(f"fitted on {len(inner):,} rows to {args.test_start} and saved to {args.save}")


if __name__ == "__main__":
    main()
