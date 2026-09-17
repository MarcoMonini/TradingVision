"""Exits for the always-in rule: a take profit, a stop loss, and what to do after one fires.

`threshold` holds a position from one band touch to the other and never asks the price anything.
That is the rule the spec priced, and section 10 of the handoff says where it bleeds: bucketed by
the size of the move it held through, the rule's hit rate collapses on the loudest fifth of the
holds — and the duration column next to it says why. A hysteresis rule exits only when the
prediction reaches the opposite band, so a hold the market is proving wrong stays open *while the
move runs*, and a hold the market is proving right gets closed by the same move reaching the band.
Move size and hold length are one variable there. The control (`--horizon`, every entry graded on
a fixed number of bars) is flat at every horizon, which is the reading that matters: the signal
does not walk into big moves more often than into small ones. What loses money on them is the
**exit**, and an exit is what this module adds.

Three things are parameterised, because all three are guesses until something measures them.

**Where the barriers go.** A barrier is a log distance from the entry price, fixed at entry and
never moved (a trailing stop is a different rule and is not here). Three ways to name the distance,
and they answer different questions:

    atr:k   k times the Average True Range at the entry bar, as a fraction of price. The unit of
            what this market does anyway, so the same k is the same aggressiveness on a quiet pair
            and a loud one. `3xATR` is the textbook number and has nothing behind it here.
    fee:k   k times the round trip, `2 * FEE` = 0.50%. The unit of what the trade costs, which is
            the only distance with an arithmetic meaning: a take profit at `fee:1` nets exactly
            zero, so k <= 1 is a barrier that cannot pay for itself, and `fee:2` doubles the fee
            before it counts as a win.
    pct:x   a flat x in log units, for when neither of the above is the question.

The two barriers take separate specs on purpose. Symmetric barriers are a bet that the label is
symmetric; `swing_leg_target` is symmetric around zero by construction but the *price* is not, and
a wide take with a tight stop is the shape "let the leg run, cut the ones that start wrong" — the
shape the by-move table points at. Whether it pays is a measurement and not an opinion.

**Which barrier wins when both sit inside one bar.** An OHLC bar does not say the order its high
and its low arrived in, and assuming the take came first is the oldest way to manufacture a
backtest. `--tie stop` is the default and takes the loss; `--tie take` exists to be run beside it,
because the gap between the two readings is the size of the intrabar assumption. A result that
only survives on `--tie take` is a result about the path, not about the rule.

**What happens after a barrier fires.** The user's question, and the reason this module is not
three lines inside `threshold`. Three policies, and the difference between them is the meaning of
"the signal said long and the market disagreed":

    reverse   flip to the other side immediately, at the exit price. The stop is treated as
              information about direction, not just about size.
    opposite  stand flat until the prediction fires on the *other* side. The stop is treated as a
              statement about this trade only, and the rule waits to be told something new.
    rearm     stand flat until the prediction fires again on either side, the stopped one
              included. Re-entering the same direction needs the prediction to fall back inside
              the band first and cross out again — otherwise the rule would buy back into the
              trade it was just stopped out of on the very next bar, at the cost of a round trip.

That "fresh crossing" rule is what `threshold.signals` was split out for. A forward filled position
repeats its last band touch on every bar forever, so it cannot say whether the signal has spoken
*since* the stop; the raw reading can, and every policy here is built on it.

Each barrier carries its own policy. Reversing after a take profit is a mean reversion bet and
reversing after a stop is a momentum one, and there is no reason for the same answer to serve both.

**Where the accounting is exact and where it is not.** An exit fills at the barrier level, or at
the bar's open when the bar gapped through it — never at the level the price never traded at. The
bar an exit happens in therefore pays two pieces, the hold up to the fill and whatever is held
after it, and both are summed into the same row. With no barriers set the whole thing collapses to
`threshold`'s own arithmetic, bit for bit, and `_selfcheck` asserts exactly that — it is the only
guarantee that the numbers below are comparable to the ones the spec already carries. The one
place the bar granularity shows is a reversal entered intrabar: its own barriers are only checked
from the next bar on, because the path inside the bar it was born in is not known.

Fees are `oracle.FEE` per side and the trade table charges a round trip to each hold, which is the
convention `threshold.legs` uses — a flip is the exit of one hold and the entry of the next, so the
pair adds up to the four sides a flip really pays.

    uv run python -m tradingvision.stops --pred data/pred-swing-all-15m.parquet --at 0.5 \
        --sl atr:2 --tp atr:4 --after-stop reverse
    uv run python -m tradingvision.stops --pred data/pred-swing-all-15m.parquet --at 0.5 --grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from tradingvision import threshold
from tradingvision.data.binance import load
from tradingvision.data.pivots import EXTREMA_WINDOW
from tradingvision.oracle import FEE

YEAR = threshold.YEAR
OHLC = ("open", "high", "low", "close")
# The three ways a barrier distance can be named, and what one unit of each is worth. `fee` is the
# round trip and not one side: the round trip is the hurdle a closed trade has to clear, so
# `fee:1` is exactly break-even and the multiplier reads as "how many times its own cost".
KINDS = ("atr", "fee", "pct")
# What the rule does after a barrier fires. Order matters only in that the first is the default in
# the CLI, and `opposite` is the default because it is the one that changes nothing about the
# signal's authority — it only refuses to hold through a move the signal did not predict.
AFTER = ("opposite", "reverse", "rearm")
# The default grid `--grid` sweeps. Stated in ATR because that is the unit that means the same
# thing on twenty pairs of different volatility, and asymmetric pairs are in it because a
# symmetric grid cannot answer the question the by-move table asks.
GRID = ((2.0, 2.0), (3.0, 3.0), (2.0, 4.0), (4.0, 2.0), (1.0, 3.0), (3.0, 1.0))


def parse_barrier(text: str | None) -> tuple[str, float] | None:
    """`"atr:3"` to `("atr", 3.0)`. `None` or `"off"` to `None`, which is the absent barrier."""
    if text is None or text.lower() in ("off", "none", ""):
        return None
    kind, _, size = text.partition(":")
    if kind not in KINDS or not size:
        raise ValueError(f"barrier must be one of {'/'.join(k + ':k' for k in KINDS)}, got {text!r}")
    return kind, float(size)


def atr_pct(bars: pd.DataFrame, window: int = EXTREMA_WINDOW) -> pd.Series:
    """Wilder's Average True Range at every row, as a fraction of that row's close.

    The same column `features` carries as `average_true_range_pct` and computed by the same class,
    so a barrier quoted in ATR is quoted in the unit the model's own inputs are measured in. Per
    symbol, because a true range spans two bars and the panel interleaves twenty of them — a
    single pass over the stacked frame would take the gap from one symbol's last bar to another
    symbol's first as a real move.
    """
    out = pd.Series(np.nan, index=bars.index, name="atr")
    for _, rows in bars.groupby(level=1, sort=False):
        one = rows.droplevel(1)
        atr = AverageTrueRange(one.high, one.low, one.close, window=window).average_true_range()
        out.loc[rows.index] = (atr / one.close).to_numpy()
    return out


def width(spec: tuple[str, float] | None, atr: pd.Series, fee: float = FEE) -> pd.Series:
    """A barrier spec as a log distance at every row; `+inf` where there is no barrier.

    Infinity and not NaN, and not a branch either: a long's take profit sits at
    `entry * exp(+inf)` which nothing ever reaches, and its stop at `entry * exp(-inf)` which is
    zero and nothing ever reaches either. The absent barrier is then the same arithmetic as a
    present one, so the loop below has no "no barrier" path that could behave differently from
    the path the numbers are quoted on.

    A row whose ATR is not strictly positive gets `+inf` for the same reason, and that case is not
    hypothetical: `ta` fills the indicator's warm-up with **zeros** rather than with NaN, so the
    first `window` rows of every symbol carry a width of exactly zero. A zero-width barrier sits
    on the entry price itself and fires on the first bar that moves at all, which would turn the
    warm-up into a stream of instant stops and nothing would say so — the column would look like
    a rule and behave like a bug. `> 0` and not `notna()` is what catches it.
    """
    if spec is None:
        return pd.Series(np.inf, index=atr.index)
    kind, k = spec
    if kind == "atr":
        return (k * atr).where(atr > 0, np.inf)
    if kind == "fee":
        return pd.Series(k * 2 * fee, index=atr.index)
    return pd.Series(k, index=atr.index)


def walk(
    sig: np.ndarray,
    op: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    after_stop: str = "opposite",
    after_take: str = "opposite",
    tie_stop: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """One symbol, bar by bar: the position held, what it earned, what it traded, why it exited.

    The state machine, and the only place in this module that knows about time. Returns
    `(pos, ret, ret_long, traded, holds)`, all indexed like the input:

        pos[i]       the side held from the close of bar i into bar i+1
        ret[i]       the log P&L that side earned over bar i+1, barrier fill included
        ret_long[i]  the part of it earned while long, so the two legs split exactly even when a
                     reversal happens inside the bar
        traded[i]    units of position changed at the close of bar i or inside bar i
        holds        one tuple per completed hold: (entry bar, exit bar, side, entry, exit, why)

    A decision is taken at a close and the barriers of the resulting position are checked from the
    *next* bar on. Checking the entry bar's own range at its close would be reading the high and
    low of a bar the rule only just decided on, which is the future by one bar — the same
    anticipation the alignment rule exists to forbid.

    `barred` is the side a policy refuses to re-enter, and `sticky` says whether a return of the
    prediction inside the band clears it. Under `rearm` and `reverse` it does, which is what makes
    "the signal spoke again" mean a fresh crossing rather than the same touch repeated; under
    `opposite` it does not, so only the other side can be taken however long the prediction stays
    where it was.
    """
    n = len(cl)
    pos, ret, ret_long, traded = (np.zeros(n) for _ in range(4))
    side, entry, entry_i, barred, sticky = 0.0, np.nan, -1, 0.0, False
    holds: list = []

    for i in range(n):
        s = sig[i]
        # A prediction back inside the band is what re-arms a barred side — the crossing that
        # comes after it is a new statement and not the echo of the one that just stopped out.
        if s == 0.0 and not sticky:
            barred = 0.0
        if s != 0.0 and s != side and s != barred:
            if side != 0.0:
                holds.append((entry_i, i, side, entry, cl[i], "signal"))
                traded[i] += 1.0
            traded[i] += 1.0
            side, entry, entry_i, barred, sticky = s, cl[i], i, 0.0, False
        pos[i] = side
        if i + 1 == n:
            break
        j = i + 1
        if side == 0.0:
            continue

        # The two levels, from the entry price and the widths measured at the entry bar. An absent
        # barrier is an infinite width, so its level is unreachable rather than special-cased.
        up = side > 0.0
        take_at = entry * np.exp(tp[entry_i] if up else -tp[entry_i])
        stop_at = entry * np.exp(-sl[entry_i] if up else sl[entry_i])
        if up:
            gapped_stop, gapped_take = op[j] <= stop_at, op[j] >= take_at
            touched_stop, touched_take = lo[j] <= stop_at, hi[j] >= take_at
        else:
            gapped_stop, gapped_take = op[j] >= stop_at, op[j] <= take_at
            touched_stop, touched_take = hi[j] >= stop_at, lo[j] <= take_at
        # A bar that opens past a level never traded at that level, so it fills at the open. Only
        # one of the two can be gapped through — they sit on opposite sides of the entry.
        if gapped_stop:
            why, at = "stop", op[j]
        elif gapped_take:
            why, at = "take", op[j]
        elif touched_stop and touched_take:
            # Both inside one bar and the bar does not say which came first. The default takes the
            # loss; `--tie take` is the other bound, and the two together say how much of a result
            # is an assumption about the path.
            why, at = ("stop", stop_at) if tie_stop else ("take", take_at)
        elif touched_stop:
            why, at = "stop", stop_at
        elif touched_take:
            why, at = "take", take_at
        else:
            ret[i] = side * np.log(cl[j] / cl[i])
            ret_long[i] = ret[i] if up else 0.0
            continue

        # The hold up to the fill, then whatever is held for the rest of the bar. Both land on row
        # i, which is the row that owns the interval; the turnover lands on row j, which is the bar
        # the fills happened in.
        first = side * np.log(at / cl[i])
        holds.append((entry_i, j, side, entry, at, why))
        traded[j] += 1.0
        stopped = side
        if (after_take if why == "take" else after_stop) == "reverse":
            side, entry, entry_i = -stopped, at, j
            traded[j] += 1.0
        else:
            side = 0.0
        barred = stopped
        sticky = (after_take if why == "take" else after_stop) == "opposite"
        rest = side * np.log(cl[j] / at)
        ret[i] = first + rest
        ret_long[i] = (first if up else 0.0) + (rest if side > 0.0 else 0.0)

    if side != 0.0:
        # The hold still open when the series ends. Kept and marked, not dropped: dropping it would
        # quietly delete every trade a long stop is supposed to produce at the end of a window.
        holds.append((entry_i, n - 1, side, entry, cl[n - 1], "open"))
    return pos, ret, ret_long, traded, holds


def run(
    pred: pd.Series,
    bars: pd.DataFrame,
    band: float,
    take: tuple[str, float] | None = None,
    stop: tuple[str, float] | None = None,
    after_stop: str = "opposite",
    after_take: str = "opposite",
    tie_stop: bool = True,
    sign: int = -1,
    window: int = EXTREMA_WINDOW,
    fee: float = FEE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rule over a whole panel: `(held, trades)`.

    `held` is indexed like `pred` and carries `pos`, `ret`, `ret_long`, `traded` and `exit` — the
    last one names the barrier that fired in that bar, so a chart can mark a stop apart from a
    signal flip. `trades` is one row per hold with its own entry and exit prices, which is the
    unit a hit rate has to be counted in for the same reason `threshold.legs` gives: a share of
    profitable *bars* counts one decision dozens of times.
    """
    if after_stop not in AFTER or after_take not in AFTER:
        raise ValueError(f"policy must be one of {AFTER}")
    missing = [c for c in OHLC if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing {missing}")
    sig = threshold.signals(pred, band, sign)
    tp, sl = width(take, atr_pct(bars, window), fee), width(stop, atr_pct(bars, window), fee)

    held = pd.DataFrame(
        {"pos": 0.0, "ret": 0.0, "ret_long": 0.0, "traded": 0.0, "exit": ""},
        index=pred.index,
    )
    rows = []
    for symbol, at in pred.groupby(pred.index.get_level_values(1), sort=False).groups.items():
        take_ = bars.loc[at]
        pos, ret, ret_long, traded, holds = walk(
            sig.loc[at].to_numpy(),
            *(take_[c].to_numpy() for c in OHLC),
            tp.loc[at].to_numpy(),
            sl.loc[at].to_numpy(),
            after_stop,
            after_take,
            tie_stop,
        )
        held.loc[at, ["pos", "ret", "ret_long", "traded"]] = np.column_stack([pos, ret, ret_long, traded])
        when = at.get_level_values(0)
        for entry_i, exit_i, s, entry, out, why in holds:
            if why in ("take", "stop"):
                held.loc[(when[exit_i], symbol), "exit"] = why
            rows.append(
                {
                    "symbol": symbol,
                    "entry_time": when[entry_i],
                    "exit_time": when[exit_i],
                    "side": s,
                    "entry": entry,
                    "exit": out,
                    "bars": exit_i - entry_i,
                    "move": float(np.log(out / entry)),
                    "why": why,
                }
            )
    columns = ["symbol", "entry_time", "exit_time", "side", "entry", "exit", "bars", "move", "why"]
    trades = pd.DataFrame(rows, columns=columns)
    # The round trip charged to each hold, which is `threshold.legs`' convention: a flip is one
    # hold's exit and the next one's entry, so the two of them together carry the four sides.
    trades["net"] = trades.side * trades.move - 2 * fee
    return held, trades.sort_values(["symbol", "entry_time"], ignore_index=True)


def price(held: pd.DataFrame, trades: pd.DataFrame, fee: float = FEE) -> dict:
    """The panel's P&L per year, in the same keys `threshold.pnl` reports plus the exit counts.

    Same keys on purpose: the whole claim of this module is that it is `threshold`'s rule with an
    exit bolted on, and a dict that lines up is what lets the two be printed in one table and read
    down a column. `_selfcheck` asserts the two agree exactly when no barrier is set.
    """
    when = held.index.get_level_values(0)
    symbol = held.index.get_level_values(1)
    span = (when.max() - when.min()) / YEAR
    gross = held.ret.groupby(symbol).sum()
    cost = (held.traded * fee).groupby(symbol).sum()
    longs = held.ret_long.groupby(symbol).sum()
    return {
        "in_market": float((held.pos != 0).mean()),
        "long_share": float((held.pos > 0).mean()),
        "gross_long": float(longs.mean() / span),
        "gross_short": float((gross - longs).mean() / span),
        "trades_per_year": len(trades) / held.index.get_level_values(1).nunique() / span,
        "gross_per_year": float(gross.mean() / span),
        "fees_per_year": float(cost.mean() / span),
        "net_per_year": float((gross.mean() - cost.mean()) / span),
        "win_rate": float((trades.net > 0).mean()) if len(trades) else np.nan,
        "median_trade": float(trades.net.median()) if len(trades) else np.nan,
        "stopped": float((trades.why == "stop").mean()) if len(trades) else np.nan,
        "took_profit": float((trades.why == "take").mean()) if len(trades) else np.nan,
        "median_bars": float(trades.bars.median()) if len(trades) else np.nan,
    }


def pnl(pred: pd.Series, bars: pd.DataFrame, band: float, **kw) -> dict:
    """`run` and `price` in one call, for the rows a table is built from."""
    fee = kw.get("fee", FEE)
    held, trades = run(pred, bars, band, **kw)
    return price(held, trades, fee)


def grid(
    pred: pd.Series,
    bars: pd.DataFrame,
    band: float,
    pairs=GRID,
    kind: str = "atr",
    after_stop: str = "opposite",
    after_take: str = "opposite",
    **kw,
) -> pd.DataFrame:
    """One row per (take, stop) pair, with the no-barrier rule on top as the control.

    The control is the row every other row has to beat, and it is `threshold`'s own number: if no
    barrier improves on it then the exit is not where the rule loses, whatever the by-move table
    suggested. Read the dispersion and not the best row — six pairs on one grid will always show
    a winner, and picking it afterwards is how a tuned number gets manufactured.
    """
    rows = [pnl(pred, bars, band, after_stop=after_stop, after_take=after_take, **kw) | {"tp": np.nan, "sl": np.nan}]
    for tp, sl in pairs:
        rows.append(
            pnl(
                pred,
                bars,
                band,
                take=(kind, tp),
                stop=(kind, sl),
                after_stop=after_stop,
                after_take=after_take,
                **kw,
            )
            | {"tp": tp, "sl": sl}
        )
    out = pd.DataFrame(rows)
    return out.set_index(pd.Index(["none"] + [f"{kind} {tp:g}/{sl:g}" for tp, sl in pairs], name="tp/sl"))[
        [c for c in out.columns if c not in ("tp", "sl")]
    ]


def frames(index: pd.MultiIndex) -> pd.DataFrame:
    """The 5m OHLC at each (timestamp, symbol) row — `threshold.prices` with the other three
    columns, which a barrier needs and a close-to-close rule does not."""
    out = pd.DataFrame(np.nan, index=index, columns=list(OHLC))
    for symbol, rows in pd.Series(np.arange(len(index)), index=index).groupby(level=1):
        one = load(symbol, "5m")[list(OHLC)].reindex(rows.index.get_level_values(0))
        out.iloc[rows.to_numpy()] = one.to_numpy()
    if out.isna().any().any():
        raise ValueError(f"{int(out.isna().any(axis=1).sum())} rows have no 5m bar")
    return out


def _saw(n: int = 400, period: int = 40, amplitude: float = 0.02):
    """The fixture the checks below run on: a triangle wave in price and a prediction that reads
    it exactly, which is `threshold._selfcheck`'s own saw so the two modules are checked against
    the same thing. Returns `(pred, bars)` under the one-symbol index every function here reads."""
    when = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    idx = pd.MultiIndex.from_arrays([when, ["a"] * n], names=["open_time", "symbol"])
    phase = np.arange(n) % period
    ramp = np.where(phase < period // 2, phase, period - phase) / (period // 2)
    close = np.exp(amplitude * ramp)
    pred = pd.Series(2 * ramp - 1, index=idx)
    # High and low straddle the close by a tenth of the leg, so a barrier inside that band is
    # reachable intrabar and one outside it is not — which is what lets a check aim at either.
    # The open is the previous close, which is what makes this a continuous market: an open equal
    # to its own bar's close would gap past a barrier on every trending bar, and every intrabar
    # check below would be testing the gap path instead of the one it means to test.
    bars = pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close * (1 + amplitude / 10),
            "low": close * (1 - amplitude / 10),
            "close": close,
        },
        index=idx,
    )
    return pred, bars


def _selfcheck() -> None:
    pred, bars = _saw()
    close = bars.close

    # Parsing, which is the only place a typo can turn into a silently different rule.
    assert parse_barrier("atr:3") == ("atr", 3.0) and parse_barrier("fee:2.5") == ("fee", 2.5)
    assert parse_barrier(None) is None and parse_barrier("off") is None
    for bad in ("atr", "sigma:3", "3"):
        try:
            parse_barrier(bad)
            raise AssertionError(f"{bad!r} should not parse")
        except ValueError:
            pass

    # The widths. `fee:1` is the round trip and nothing else, `pct` is what it says, and an absent
    # barrier is infinite rather than a branch — which is what keeps one code path in `walk`.
    atr = atr_pct(bars)
    assert np.isclose(width(("fee", 1.0), atr).iloc[0], 2 * FEE)
    assert np.isclose(width(("fee", 3.0), atr).iloc[0], 6 * FEE)
    assert (width(None, atr) == np.inf).all()
    assert np.isclose(width(("pct", 0.05), atr).iloc[-1], 0.05)
    # `ta` fills the ATR warm-up with zeros, not NaN. A zero width is a barrier on the entry price
    # itself, so the warm-up would stop out on its first tick; it gets no barrier instead.
    assert atr.iloc[0] == 0.0 and width(("atr", 3.0), atr).iloc[0] == np.inf
    assert (width(("atr", 2.0), atr) > 0).all(), "no width is ever zero, whatever the indicator says"
    assert np.isclose(width(("atr", 2.0), atr).iloc[-1], 2 * atr.iloc[-1])
    # The column is `features`' own, so a barrier in ATR is in the unit the model's inputs use.
    from tradingvision.features import features as _features

    assert np.allclose(
        atr.droplevel(1).to_numpy(),
        _features(bars.droplevel(1).assign(volume=1.0)).average_true_range_pct.to_numpy(),
        equal_nan=True,
    )

    # No barrier is exactly `threshold`. Not "close to" — the same positions, the same gross, the
    # same fees, the same win rate. Everything else in this module is only comparable to the
    # numbers the spec already carries because this holds.
    held, trades = run(pred, bars, 0.5)
    assert held.pos.equals(threshold.positions(pred, 0.5).rename("pos"))
    theirs = threshold.pnl(pred, close, 0.5)
    mine = price(held, trades)
    for key in ("in_market", "long_share", "gross_long", "gross_short", "gross_per_year", "fees_per_year"):
        assert np.isclose(mine[key], theirs[key]), (key, mine[key], theirs[key])
    for key in ("net_per_year", "trades_per_year", "win_rate", "median_trade"):
        assert np.isclose(mine[key], theirs[key]), (key, mine[key], theirs[key])
    assert (trades.why == "signal").sum() + (trades.why == "open").sum() == len(trades)
    # And the two accountings close on each other: every hold is flat-to-flat or flip-to-flip, so
    # the P&L of the holds is the P&L of the bars.
    assert np.isclose((trades.side * trades.move).sum(), held.ret.sum())
    # The long leg and the short leg add back up to the whole.
    assert np.isclose(mine["gross_long"] + mine["gross_short"], mine["gross_per_year"])

    # A barrier wider than anything the saw reaches is the same rule again — the reachability of a
    # level is what makes a barrier a barrier, not its presence in the arguments.
    far, _ = run(pred, bars, 0.5, take=("pct", 10.0), stop=("pct", 10.0))
    assert far.pos.equals(held.pos) and np.isclose(far.ret.sum(), held.ret.sum())

    # A stop inside the bar's own wick fires on the first bar it can, which on this fixture is the
    # bar right after every entry: the low sits 0.2% under the close and the stop is at 0.1%.
    tight, hit = run(pred, bars, 0.5, stop=("pct", 0.001))
    assert (hit.why == "stop").any() and (hit.why == "stop").mean() > 0.9, hit.why.value_counts()
    assert (tight.exit == "stop").sum() == (hit.why == "stop").sum()
    # Stopped out of every hold, so the rule is flat most of the time — which the always-in rule
    # never is, and is the whole behavioural difference an exit makes.
    assert tight.pos.eq(0.0).mean() > 0.5 and held.pos.eq(0.0).mean() < 0.01
    # Every stopped hold lost its stop distance and no more, before fees.
    stopped = hit[hit.why == "stop"]
    assert np.allclose(stopped.side * stopped.move, -0.001, atol=2e-4), stopped.head()

    # A take profit inside the wick is the mirror image, and it is the one that pays.
    _, won = run(pred, bars, 0.5, take=("pct", 0.001))
    taken = won[won.why == "take"]
    assert len(taken) and np.allclose(taken.side * taken.move, 0.001, atol=2e-4), taken.head()

    # The tie-break, on a fixture where both levels sit inside every bar's range. The default
    # takes the loss; the other reading takes the profit; the gap between them is the assumption.
    pessimist = pnl(pred, bars, 0.5, take=("pct", 0.001), stop=("pct", 0.001), tie_stop=True)
    optimist = pnl(pred, bars, 0.5, take=("pct", 0.001), stop=("pct", 0.001), tie_stop=False)
    assert optimist["gross_per_year"] > pessimist["gross_per_year"], (optimist, pessimist)
    # Every hold the pessimist sees is a stop, because on this fixture both levels are inside
    # every bar. The optimist keeps the stops the take could not reach — the bars at the turns of
    # the saw, which move only one way — so its share is lower and not zero, and that difference
    # is the whole of the intrabar assumption.
    assert pessimist["stopped"] == 1.0 and 0.0 < optimist["stopped"] < pessimist["stopped"]

    # A gap through the level fills at the open and not at the level, which is the difference
    # between a backtest and a wish. One bar opens 5% under its predecessor's close, and the long
    # held into it has a 0.1% stop: the fill is the open, five percent down.
    # A stop wider than the saw's own amplitude, so the gap is the only thing that can fire it and
    # the assertion below is about the gap and not about the fixture.
    base = threshold.positions(pred, 0.5).to_numpy()
    cut = int(np.flatnonzero(base[:-1] > 0)[5]) + 1  # a bar entered long, which the gap runs over
    gapped = bars.copy()
    gapped.iloc[cut, :] = gapped.iloc[cut].to_numpy() * 0.95
    _, out = run(pred, gapped, 0.5, stop=("pct", 0.03))
    at_gap = out[out.exit_time == gapped.index.get_level_values(0)[cut]]
    assert len(at_gap) == 1 and at_gap.why.iloc[0] == "stop"
    assert np.isclose(at_gap.exit.iloc[0], gapped.open.iloc[cut]), (at_gap.exit.iloc[0], gapped.open.iloc[cut])
    assert at_gap.side.iloc[0] * at_gap.move.iloc[0] < -0.04, "a gap costs what it costs, not the stop distance"

    # The three policies, on the same stopped-out fixture. They differ in exactly one thing — what
    # is held after the stop — so the positions have to differ and the trade counts with them.
    shapes = {}
    for policy in AFTER:
        h, t = run(pred, bars, 0.5, stop=("pct", 0.001), after_stop=policy)
        shapes[policy] = (h, t)
    rev, opp, arm = (shapes[p][0] for p in ("reverse", "opposite", "rearm"))
    # `reverse` is never flat after its first entry: it exits into the other side at the same fill.
    assert rev.pos.iloc[1:].ne(0.0).all(), "reverse leaves no flat bar"
    assert opp.pos.eq(0.0).any() and arm.pos.eq(0.0).any()
    # `rearm` re-enters on the next fresh crossing in either direction and `opposite` only on the
    # other side, so `rearm` is in the market at least as often and trades at least as much.
    assert arm.pos.ne(0.0).mean() >= opp.pos.ne(0.0).mean()
    assert len(shapes["rearm"][1]) >= len(shapes["opposite"][1])
    # And the barred side is really barred: under `opposite`, no hold starts on the side the
    # previous one was stopped out of.
    t = shapes["opposite"][1]
    after = t[t.why.shift() == "stop"]
    assert (after.side.to_numpy() == -t.side.shift()[after.index].to_numpy()).all(), after.head()
    # Under `reverse` the hold that follows a stop is on the other side and starts at the fill
    # price of the stop — no bar of slippage between the two, because they are one event.
    t = shapes["reverse"][1]
    follow = t[t.why.shift() == "stop"]
    assert (follow.side.to_numpy() == -t.side.shift()[follow.index].to_numpy()).all()
    assert np.allclose(follow.entry.to_numpy(), t.exit.shift()[follow.index].to_numpy())

    # A policy name that does not exist is an error and not a silent default.
    for bad in ({"after_stop": "flip"}, {"after_take": ""}):
        try:
            run(pred, bars, 0.5, **bad)
            raise AssertionError(f"{bad} should not run")
        except ValueError:
            pass
    # And so is a frame that cannot answer the question a barrier asks.
    try:
        run(pred, bars[["close"]], 0.5)
        raise AssertionError("a close-only frame should not price a barrier")
    except ValueError:
        pass

    # Two symbols, out of phase, are walked independently: the state machine never carries a side
    # from the end of one symbol's rows into the start of another's.
    pair = pd.concat([pred, pred.rename(index={"a": "b"}, level=1)]).sort_index()
    both = pd.concat([bars, bars.rename(index={"a": "b"}, level=1)]).sort_index()
    h, t = run(pair, both, 0.5, stop=("pct", 0.001))
    one = shapes["opposite"][0]
    assert h.xs("a", level=1).pos.equals(one.xs("a", level=1).pos)
    assert h.xs("b", level=1).pos.equals(one.xs("a", level=1).pos)
    assert set(t.symbol.unique()) == {"a", "b"} and len(t) == 2 * len(shapes["opposite"][1])
    # The panel's rate is the per-symbol rate and not the sum of two of them.
    alone = price(*shapes["opposite"])
    assert np.isclose(price(h, t)["gross_per_year"], alone["gross_per_year"])

    # The grid's first row is the control, and it is `threshold`'s number.
    table = grid(pred, bars, 0.5, pairs=((2.0, 2.0),))
    assert len(table) == 2 and table.index[0] == "none"
    assert np.isclose(table.net_per_year.iloc[0], threshold.pnl(pred, close, 0.5)["net_per_year"])

    # A hold's own arithmetic: the exit price is the entry price moved by `move`, and `net` is
    # that signed by the side and charged a round trip.
    _, t = run(pred, bars, 0.5, stop=("atr", 1.0), take=("atr", 3.0))
    assert np.allclose(t.exit, t.entry * np.exp(t.move))
    assert np.allclose(t.net, t.side * t.move - 2 * FEE)
    assert (t.bars >= 0).all() and (t.exit_time >= t.entry_time).all()
    assert t.why.isin(["signal", "stop", "take", "open"]).all()
    # One open hold at most, and it is the last one on its symbol.
    assert (t.why == "open").sum() <= 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="the predictions a gru run wrote")
    ap.add_argument("--at", type=float, default=0.5, help="the band, as a raw number on the prediction")
    ap.add_argument("--tp", default=None, metavar="KIND:K", help="take profit, e.g. atr:3, fee:2, pct:0.02, off")
    ap.add_argument("--sl", default=None, metavar="KIND:K", help="stop loss, same spelling as --tp")
    ap.add_argument("--after-stop", default=AFTER[0], choices=AFTER, help="what to hold after a stop fires")
    ap.add_argument("--after-take", default=AFTER[0], choices=AFTER, help="what to hold after a take fires")
    ap.add_argument(
        "--tie",
        default="stop",
        choices=["stop", "take"],
        help="which barrier wins when both sit inside one bar; the default takes the loss",
    )
    ap.add_argument("--fee", type=float, default=FEE, help="per side; the default is Alpaca taker tier 1")
    ap.add_argument("--sign", type=int, default=-1, choices=[-1, 1], help="-1 for the swing label")
    ap.add_argument("--window", type=int, default=EXTREMA_WINDOW, help="bars of ATR behind a barrier in atr units")
    ap.add_argument("--grid", action="store_true", help="sweep the default take/stop pairs in ATR against no barrier")
    args = ap.parse_args()

    _selfcheck()
    pred = pd.read_parquet(args.pred).iloc[:, 0]
    bars = frames(pred.index)
    shared = dict(
        after_stop=args.after_stop,
        after_take=args.after_take,
        tie_stop=args.tie == "stop",
        sign=args.sign,
        window=args.window,
        fee=args.fee,
    )
    print(f"{len(pred):,} rows, {pred.index.get_level_values(1).nunique()} symbols, fee {args.fee * 100:.2f}% per side")
    print(f"{pred.index.get_level_values(0).min():%Y-%m-%d} to {pred.index.get_level_values(0).max():%Y-%m-%d}")
    print(f"band +/-{args.at}, after a stop: {args.after_stop}, after a take: {args.after_take}, tie: {args.tie}\n")
    if args.grid:
        print(grid(pred, bars, args.at, **shared).round(4).to_string())
    else:
        row = pnl(pred, bars, args.at, take=parse_barrier(args.tp), stop=parse_barrier(args.sl), **shared)
        print(pd.Series(row).round(4).to_string())
    print(f"\nbuy and hold, same rows: {threshold.buy_and_hold(bars.close)['net_per_year'] * 100:.1f}% log per year")


if __name__ == "__main__":
    main()
