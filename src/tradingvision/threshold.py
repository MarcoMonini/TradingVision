"""What the signal is worth as a rule, in money rather than in correlation.

Rank IC answers "does the ordering carry information". It does not answer "does trading on it
pay", and the two come apart in exactly the way that matters here: a signal can rank the twenty
symbols correctly every hour and still lose to fees, and a signal with a mediocre IC can pay if
its errors land where nothing is held. This module asks the second question and nothing else.

The rule is the simplest one the label describes, and deliberately not a tuned strategy. The
label says where the bar sits on its leg — `-1` at a low, `+1` at a high — so:

    pred <= -t     go long, the leg should run up from here
    pred >= +t     go short
    in between     hold whatever is already held

Hysteresis and not a flat band: the swing the label describes *is* the hold from one extreme to
the other, and a rule that flattens whenever the signal is unremarkable would pay the round trip
several times inside a single leg. Costs are `oracle.FEE` per side, charged on every unit of
position changed, so a flip from long to short pays twice.

That rule is "always in": once it has fired once it is long or short at every bar and never
flat, so a flip pays two sides and the short leg is a position and not an absence. `--at` prices
it at raw numbers rather than at quantiles of the output — `--at 0.4` is the -0.4/+0.4 pair spelled
the way a live system would have to commit to it, and the gap to the quantile grid is how far that
constant drifts from the share of bars it was meant to hold. On `pred-swing-all-15m.parquet`,
twenty symbols, 2025-06 to 2026-09, |pred| clears 0.4 on 23.7% of the rows, so the pair sits near
the 0.76 quantile of the grid below it.

One number and not two, on purpose. The label is symmetric around zero by construction, so a rule
that puts its entry and its exit at different distances is asserting an asymmetry the label does
not carry; `sign` already handles which end means long. Two numbers are one edit away if a
measurement ever asks for them, and until then the grid is the honest way to move the band.

Two honesties about the measurement. The rows are step 2's, one per hour per symbol, so this
trades hourly and never inside the hour. And each symbol is one unit, equally weighted, with no
sizing and no risk limit — the question is whether the edge clears the fee, not what a portfolio
would do with it.

    uv run python -m tradingvision.threshold --pred data/pred-swing-all-15m.parquet --at 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision.data.binance import load
from tradingvision.oracle import FEE

# Quantiles of |prediction| the default grid is taken at: a threshold is only readable next to how
# much of the time it leaves a position open, and the scale of the label changes between targets —
# `swing` lives in [-1, 1], `excursion` in sigma of a 24-bar walk.
QUANTILES = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)
YEAR = pd.Timedelta("365D")


def prices(index: pd.MultiIndex) -> pd.Series:
    """The 5m close at each (timestamp, symbol) row."""
    out = pd.Series(np.nan, index=index, name="close")
    for symbol, rows in pd.Series(np.arange(len(index)), index=index).groupby(level=1):
        close = load(symbol, "5m").close
        out.iloc[rows.to_numpy()] = close.reindex(rows.index.get_level_values(0)).to_numpy()
    if out.isna().any():
        raise ValueError(f"{out.isna().sum()} rows have no 5m close")
    return out


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
    # Along each symbol's own rows, not along a shared timestamp grid. The symbols are not in
    # phase — a gap in a series shifts its stride from that point, and 5% of the rows end up on
    # timestamps no other symbol shares — so a rolling mean over an unstacked frame would average
    # a symbol against its own absence and quietly return the input unchanged.
    ordered = pred.sort_index(level=[1, 0])
    return ordered.groupby(level=1).rolling(k, min_periods=1).mean().droplevel(0).reindex(pred.index)


def signals(pred: pd.Series, threshold: float, sign: int = -1) -> pd.Series:
    """What the band says at each row on its own: +1 long, -1 short, 0 in between.

    The instantaneous reading, before any memory of what is held. `positions` is this forward
    filled and is what the always-in rule holds; the raw series is what a rule with an *exit*
    needs, because once a stop has closed a position the question is no longer "what is the
    signal" but "has the signal said anything new since". A forward filled series cannot answer
    that — every bar after a band touch repeats it forever — so the two readings are separated
    here and `stops` reads this one.

    `sign` is the direction the label points. It is -1 for `swing_leg_target`, where a low
    prediction means the bar sits near a low and the leg runs up from there, and +1 for
    `remaining_excursion`, which is already signed the way the trade is.
    """
    out = pd.Series(0.0, index=pred.index)
    out[pred <= -threshold] = -sign
    out[pred >= threshold] = sign
    return out


def positions(pred: pd.Series, threshold: float, sign: int = -1) -> pd.Series:
    """The held position at every row: +1 long, -1 short, 0 before the first signal."""
    signal = signals(pred, threshold, sign).replace(0.0, np.nan)
    # Forward filled inside each symbol, which is what makes it a hold and not a flicker. The rows
    # arrive sorted by timestamp, so a symbol's rows are already in its own chronological order.
    return signal.groupby(level=1).ffill().fillna(0.0)


def on_one(frame: pd.Series | pd.DataFrame, symbol: str = "one") -> pd.Series | pd.DataFrame:
    """The same series or frame under a one-symbol MultiIndex, which is what everything here reads.

    Every function in this module groups by symbol, because the study is a panel of twenty. A
    chart draws one pair. Lifting the series rather than relaxing the grouping is the choice that
    keeps a single code path: the rule drawn on one pair is then bit for bit the rule `--at`
    prices on the panel, down to the fee arithmetic and the warm-up before the first signal, and
    there is no second implementation to drift away from this one.

    A frame as well as a series, because `stops` reads four price columns where this module reads
    one, and a chart that lifted its close with this function and its OHLC by hand would have two
    spellings of the same index to keep in step.
    """
    at = pd.MultiIndex.from_arrays([frame.index, [symbol] * len(frame)], names=["open_time", "symbol"])
    return frame.set_axis(at)


def forward_return(close: pd.Series) -> pd.Series:
    """Log return from each row to that symbol's next row — the return a position earns by being
    held there. NaN on the last row of each symbol, which no position can be paid for."""
    return np.log(close.groupby(level=1).shift(-1) / close)


def legs(pos: pd.Series, r: pd.Series, fee: float = FEE) -> pd.DataFrame:
    """One row per hold: which side it was on, how far the price travelled under it, what it netted.

    The unit a hit rate has to be counted in. A leg holds one position from one signal to the next,
    so a share of profitable *hours* would count the same decision dozens of times and read its
    autocorrelation as a sample.

    `move` is the price's own log return over the hold and carries no position in it, which is what
    makes it the column to group on: "was the rule on the right side of the big moves" is asked by
    bucketing on `move` and reading `net`, and a column that already had the side multiplied in
    would answer a different question — the size of the P&L instead of the size of the move.
    """
    symbol = pos.index.get_level_values(1)
    f = pd.DataFrame({"pos": pos, "r": r})
    f["leg"] = (f.pos != f.pos.groupby(symbol).shift()).groupby(symbol).cumsum()
    held = f[f.pos != 0]
    if not len(held):
        return pd.DataFrame(columns=["side", "move", "bars", "net"], dtype="float64")
    out = held.groupby([held.index.get_level_values(1), held.leg]).agg(
        side=("pos", "first"), move=("r", "sum"), bars=("r", "size")
    )
    # Both sides of the fill, charged once per hold. A flip is the exit of one leg and the entry of
    # the next, so the two legs each carry their own two sides and the pair adds up to the four a
    # flip really pays — the same total `pnl` reaches through turnover.
    out["net"] = out.side * out.move - 2 * fee
    return out


def pnl(pred: pd.Series, close: pd.Series, threshold: float, fee: float = FEE, sign: int = -1) -> dict:
    """One threshold, priced. Log returns throughout, with the fee charged as a log cost too —
    at 0.25% the difference to the exact multiplicative form is in the fifth decimal."""
    span = (pred.index.get_level_values(0).max() - pred.index.get_level_values(0).min()) / YEAR
    pos = positions(pred, threshold, sign)
    r = forward_return(close).fillna(0.0)
    symbol = pos.index.get_level_values(1)
    # Turnover, in units of position: entering costs one side, flipping costs two.
    turnover = pos.groupby(symbol).diff().fillna(pos).abs()
    gross, cost = (pos * r).groupby(symbol).sum(), (turnover * fee).groupby(symbol).sum()

    # Per trade, so the hit rate is a hit rate and not a share of profitable hours: a leg holds a
    # constant position, and the trade is that whole hold.
    per_trade = legs(pos, r, fee).net

    # Split by side, which is the check that separates an edge from a market. Over a period the
    # market spends falling, a rule that is short half the time earns without predicting anything;
    # a signal that reads turning points has to make money on both sides, or say why not.
    gross_by_side = (pos * r).groupby([symbol, np.sign(pos)]).sum().groupby(level=1).mean()
    return {
        "threshold": threshold,
        "in_market": float((pos != 0).mean()),
        "long_share": float((pos > 0).mean()),
        "gross_long": float(gross_by_side.get(1.0, 0.0) / span),
        "gross_short": float(gross_by_side.get(-1.0, 0.0) / span),
        "trades_per_year": len(per_trade) / len(gross) / span,
        "gross_per_year": float(gross.mean() / span),
        "fees_per_year": float(cost.mean() / span),
        "net_per_year": float((gross.mean() - cost.mean()) / span),
        "win_rate": float((per_trade > 0).mean()) if len(per_trade) else np.nan,
        "median_trade": float(per_trade.median()) if len(per_trade) else np.nan,
    }


def sweep(pred: pd.Series, close: pd.Series, quantiles=QUANTILES, fee: float = FEE, sign: int = -1) -> pd.DataFrame:
    """One row per threshold, taken at quantiles of |prediction| so the grid means the same thing
    whatever scale the label lives on. The 0.0 quantile is the always-in rule, which is the
    control: if no threshold beats it, the signal is adding nothing but fees."""
    rows = [pnl(pred, close, float(pred.abs().quantile(q)), fee, sign) for q in quantiles]
    return pd.DataFrame(rows, index=pd.Index(quantiles, name="quantile"))


def by_symbol(pred: pd.Series, close: pd.Series, threshold: float, fee: float = FEE, sign: int = -1) -> pd.DataFrame:
    """The same threshold priced on each pair alone, with what holding that pair did next to it.

    The panel average is the result and this is the dispersion behind it, which is the difference
    between "the rule loses" and "the rule loses on average because two pairs sank it". Read the
    spread and not the best row: twenty draws from a distribution whose per-pair standard error is
    wide will always show a winner, and picking it afterwards is the oldest way to manufacture one.

    `hold` is on that pair's own rows and is not a benchmark the rule has to beat — an always-in
    rule is short half the time, so it is not competing with holding. It is here because a short
    leg that earns on a pair that fell 70% is exposure, and the two columns side by side say which
    of the two the number is.
    """
    rows = {}
    for symbol, at in pred.groupby(pred.index.get_level_values(1)).groups.items():
        rows[symbol] = pnl(pred.loc[at], close.loc[at], threshold, fee, sign) | {
            "hold": buy_and_hold(close.loc[at])["net_per_year"]
        }
    return pd.DataFrame(rows).T.rename_axis("symbol").sort_values("net_per_year", ascending=False)


def by_move(
    pred: pd.Series,
    close: pd.Series,
    threshold: float,
    buckets: int = 5,
    fee: float = FEE,
    sign: int = -1,
    horizon: int = 0,
) -> pd.DataFrame:
    """Every hold grouped by how far the price travelled while the rule held it.

    This is the reading that tests "it works sideways and the big moves run it over". The claim is
    about *conditioning*, so an average over everything cannot answer it: a rule can be right on
    the quiet bars, wrong on the loud ones, and land anywhere in the aggregate depending only on
    how loud the period was.

    `move` carries no position in it, so the buckets are a property of the market and not of the
    rule — bucketing on the P&L instead would sort the trades by their own answer and every table
    would slope. Equal-count buckets, so each row is the same sample size and the win rates are
    comparable down the column. The null is flat: if the rule has no view, being on the right side
    of a big move is a coin toss exactly like being on the right side of a small one, and the win
    rate does not move with the size.

    A falling win rate down the table is the claim confirmed; a flat one says the big moves are not
    where the rule is losing, whatever the equity curve looks like around them.

    `horizon` is the control, and on this rule it is not optional. Read over the hold, the size of
    the move and the length of the hold are the same variable: a hysteresis rule exits when the
    prediction reaches the other band, so a position on the wrong side of a move stays open while
    the move runs and one on the right side is closed by it. That mechanism alone fills the top
    bucket with losers and needs no signal to do it. With `horizon` set, every entry is graded on
    what the price did over that many bars *from the entry*, whatever the rule did next — the exit
    rule is then out of the measurement entirely, and what survives is a statement about the
    signal. Quote the horizon reading; the hold reading is what it has to be checked against.
    """
    when = pred.index.get_level_values(0)
    span = (when.max() - when.min()) / YEAR
    symbols = pred.index.get_level_values(1).nunique()
    pos = positions(pred, threshold, sign)
    if horizon:
        symbol = pos.index.get_level_values(1)
        opened = (pos != pos.groupby(symbol).shift()) & (pos != 0)
        ahead = np.log(close.groupby(level=1).shift(-horizon) / close)
        # The same two fees a round trip pays, so the column means what it means in the other
        # reading. They are a constant here and move no bucket against another.
        held = pd.DataFrame({"side": pos[opened], "move": ahead[opened], "bars": float(horizon)}).dropna()
        held["net"] = held.side * held.move - 2 * fee
    else:
        held = legs(pos, forward_return(close).fillna(0.0), fee)
    if not len(held):
        return pd.DataFrame()
    size = held.move.abs()
    bucket = pd.qcut(size, buckets, labels=False, duplicates="drop")
    out = (
        held.assign(size=size, win=held.net > 0, hit=held.side * held.move > 0)
        .groupby(bucket)
        .agg(
            trades=("net", "size"),
            median_move=("size", "median"),
            max_move=("size", "max"),
            # The confound this table has to print next to its own result. A hysteresis rule exits
            # when the prediction reaches the other band, so a hold on the wrong side of a move
            # stays open while the move runs and a hold on the right side is closed by it. That
            # alone would fill the top bucket with losers and needs no signal to do it, so a
            # duration that climbs down the table is a reason to distrust the win rate beside it.
            median_bars=("bars", "median"),
            long_share=("side", lambda x: float((x > 0).mean())),
            # Direction only, before the fee. It is the column that answers "was the rule on the
            # right side", and it is the one to read down the table: `win_rate` charges a constant
            # 0.50% round trip against a bucket's own move, so the quiet buckets fail it for being
            # quiet and the comparison between rows would be a comparison of move sizes.
            hit_rate=("hit", "mean"),
            win_rate=("win", "mean"),
            mean_net=("net", "mean"),
            total_net=("net", "sum"),
        )
    )
    out["net_per_year"] = out.total_net / symbols / span
    out["win_rate_se"] = np.sqrt(0.25 / out.trades)
    return out.rename_axis("bucket")


def buy_and_hold(close: pd.Series) -> dict:
    """The same accounting for holding every symbol throughout — the other control, and the one a
    long-only crypto rule has to beat to be worth running at all."""
    r = forward_return(close).fillna(0.0)
    span = (close.index.get_level_values(0).max() - close.index.get_level_values(0).min()) / YEAR
    per_symbol = r.groupby(close.index.get_level_values(1)).sum()
    return {"net_per_year": float(per_symbol.mean() / span), "trades_per_year": 1 / span}


def _selfcheck() -> None:
    """A price that is a clean saw and a prediction that reads it perfectly: the rule has to make
    the saw's amplitude minus its fees, and lose once the fee exceeds the leg."""
    n = 400
    when = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    idx = pd.MultiIndex.from_arrays([when, ["a"] * n])
    # A triangle wave of 40 bars: 20 up, 20 down, 2% peak to peak.
    phase = np.arange(n) % 40
    ramp = np.where(phase < 20, phase, 40 - phase) / 20.0
    close = pd.Series(np.exp(0.02 * ramp), index=idx)
    # The label read exactly: -1 at the lows, +1 at the peaks.
    pred = pd.Series(2 * ramp - 1, index=idx)

    out = pnl(pred, close, 0.9)
    # Every completed trade wins; the last one is still open when the series ends, so it is
    # carrying its entry fee and nothing else. 21 trades, 20 of them closed.
    assert out["win_rate"] >= 0.95, out
    assert out["median_trade"] > 0.01, out
    # Ten full up legs of 2% each over a 400-hour span, so the yearly rate is that scaled up.
    span = (when[-1] - when[0]) / YEAR
    assert np.isclose(out["gross_per_year"] * span, 10 * 0.02 * 2, atol=0.02), out
    assert out["fees_per_year"] > 0 and out["net_per_year"] < out["gross_per_year"]

    # A fee larger than the leg turns the same perfect signal into a loss. Nothing about the
    # ordering changed, which is the whole point of measuring this and not the Rank IC.
    assert pnl(pred, close, 0.9, fee=0.05)["net_per_year"] < 0

    # The raw band reading, which is `positions` before the fill: non-zero exactly where the
    # prediction is outside the band, and zero — not the last side — everywhere in between.
    raw = signals(pred, 0.9)
    assert (raw[pred.abs() < 0.9] == 0.0).all() and (raw[pred <= -0.9] == 1.0).all()
    assert positions(pred, 0.9).equals(raw.replace(0.0, np.nan).groupby(level=1).ffill().fillna(0.0))

    # Hysteresis: the position is held through the middle of the leg and not flattened there.
    pos = positions(pred, 0.9)
    assert set(pos.unique()) <= {-1.0, 1.0, 0.0} and (pos.iloc[1:] != 0).all()
    assert (pos.groupby(level=1).diff().abs() > 0).sum() < 25, "one flip per leg, not one per bar"
    # And the sign: at the low of the saw the rule is long.
    assert pos.iloc[np.argmin(ramp[1:]) + 1] == 1.0

    # Two symbols out of phase are held independently, and a symbol's last row pays nothing.
    both = pd.concat([close, close.rename(index={"a": "b"}, level=1).iloc[::-1]]).sort_index()
    assert forward_return(both).groupby(level=1).tail(1).isna().all()

    # A wider band flips less often, and the zero-threshold control — which turns at the middle
    # of every leg rather than at its ends — has to lose to a real threshold. That comparison is
    # the reason the control is in the grid: it holds the same signal and the same fee, so what
    # separates them is only where the rule chooses to act.
    table = sweep(pred, close)
    assert table.trades_per_year.is_monotonic_decreasing, table
    assert table.net_per_year.iloc[-1] > table.net_per_year.iloc[0], table
    # A symmetric flip rule is always in the market once it has fired at all: `in_market` says
    # whether the prediction ever reached the band, not how selective the rule is.
    assert table.in_market.iloc[0] == 1.0
    # On the saw both sides earn, because the signal really does read the turns. It is the
    # asymmetry on real data that means something, so the symmetric case has to hold here.
    # The filter averages over time inside a symbol and never across symbols, and k=1 is identity.
    one = pred
    assert smoothed(one, 1).equals(one)
    two = smoothed(one, 2)
    assert two.index.equals(one.index) and two.notna().all()
    first = one.xs("a", level=1).sort_index()
    assert np.isclose(two.xs("a", level=1).sort_index().iloc[1], first.iloc[:2].mean())
    assert np.isclose(two.xs("a", level=1).sort_index().iloc[0], first.iloc[0]), "no value before the first bar"
    # Out of phase, as the real symbols are: each one is averaged along its own rows, and a
    # timestamp grid it does not share cannot dilute it into a no-op.
    when = pd.date_range("2024", periods=4, freq="h", tz="UTC")
    apart = pd.Series(
        [1.0, 3.0, 10.0, 30.0],
        index=pd.MultiIndex.from_arrays([[when[0], when[1], when[2], when[3]], ["a", "b", "a", "b"]]),
    )
    assert smoothed(apart, 2).tolist() == [1.0, 3.0, 5.5, 16.5]

    both = pnl(pred, close, 0.9)
    assert both["gross_long"] > 0 and both["gross_short"] > 0, both
    assert 0.3 < both["long_share"] < 0.7, both

    # The always-in rule at a raw pair, which is what `--at` prices: long at or below the lower
    # number, short at or above the upper one, and in between it holds what it already had. The
    # third state is the absence of a third state — there is no flat leg to be wrong about.
    at = positions(pred, 0.4)
    assert (at[pred <= -0.4] == 1.0).all() and (at[pred >= 0.4] == -1.0).all()
    assert (at != 0).all(), "always in: the saw fires on its first bar and never stands aside"
    # A raw number decides nothing the grid could not reach: it is the band at whatever quantile of
    # |pred| it lands on, which is why `--at` reports that quantile next to it and why the two
    # spellings never need reconciling. What a raw number adds is that it stops moving.
    wide = pnl(pred, close, 0.4)
    assert wide["threshold"] == 0.4 and wide["in_market"] == 1.0
    # Wider is not always better: a band past the label's own range never fires and holds nothing,
    # which is the one way this rule can report a zero rather than a loss.
    assert positions(pred, 1.5).eq(0.0).all() and pnl(pred, close, 1.5)["in_market"] == 0.0

    # One pair lifted into a one-symbol panel is the same rule on the same rows: that is what
    # lets the chart draw `positions` directly instead of reimplementing the state machine.
    flat = pred.droplevel(1)
    lifted = on_one(flat)
    assert positions(lifted, 0.4).droplevel(1).equals(positions(pred, 0.4).droplevel(1))
    # A frame lifts the same way, which is what keeps one index for the price columns a barrier
    # rule reads and the close this module reads.
    frame = on_one(pd.DataFrame({"close": close.droplevel(1)}))
    assert frame.index.equals(lifted.index) and frame.close.to_numpy().tolist() == close.to_numpy().tolist()
    assert pnl(lifted, on_one(close.droplevel(1)), 0.4) == pnl(pred, close, 0.4)

    # `legs` is the same trades `pnl` counts, with the move kept apart from the side. On the saw
    # every hold spans exactly one leg of the triangle, so the move is the leg and the side is
    # right every time — which is what makes the by-move table flat here and worth reading on real
    # price, where it is not.
    held = legs(positions(pred, 0.9), forward_return(close).fillna(0.0))
    priced = pnl(pred, close, 0.9)
    assert len(held) == 21 and set(held.side.unique()) == {-1.0, 1.0}, held
    assert np.isclose(held.net.median(), priced["median_trade"]), (held.net.median(), priced)
    assert np.isclose((held.net > 0).mean(), priced["win_rate"]), priced
    assert (held.net.iloc[:-1] > 0).all(), "a perfect read is on the right side of every leg"
    # The move is the price's own and the side is the rule's, so on a signal read perfectly the
    # two agree on every leg. That separation is the property the bucketing rests on: `move` is a
    # fact about the market, `net` is what the rule made of it.
    # The last leg is the one still open when the series ends: one bar, no move, its entry fee and
    # nothing else. Every closed one agrees.
    closed = held.iloc[:-1]
    assert (np.sign(closed.move) == closed.side).all(), held

    table = by_move(pred, close, 0.9, buckets=3)
    assert table.trades.sum() == len(held) and table.win_rate.min() > 0.9, table
    # Equal-count buckets, and the sizes really do increase down the table.
    assert table.median_move.is_monotonic_increasing, table
    # The horizon reading takes the exit rule out: every entry graded over the same ten bars,
    # whatever the rule did next. Half a leg of the saw, so a perfect read is on the right side of
    # all of them — and the duration column is the constant it was asked for and not a measurement.
    ahead = by_move(pred, close, 0.9, buckets=3, horizon=10)
    assert ahead.hit_rate.min() > 0.95 and (ahead.median_bars == 10).all(), ahead
    assert ahead.trades.sum() <= table.trades.sum(), "an entry with no ten bars in front of it is dropped"
    # `hit_rate` is direction before the fee and `win_rate` is after it, so the two come apart
    # exactly where the move is smaller than the round trip and nowhere else.
    assert (ahead.hit_rate >= ahead.win_rate).all(), ahead

    # The per-pair table sums back to the panel: one symbol here, so its row *is* the aggregate.
    one = by_symbol(pred, close, 0.9)
    assert len(one) == 1 and np.isclose(one.net_per_year.iloc[0], pnl(pred, close, 0.9)["net_per_year"])
    assert np.isclose(one.hold.iloc[0], buy_and_hold(close)["net_per_year"])

    # The control the always-in rule is read against. The saw ends a twentieth of a leg above where
    # it started, so holding it earns nothing and every cent of the rule's gross above is timing.
    assert abs(buy_and_hold(close)["net_per_year"] * span) < 0.01, buy_and_hold(close)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="the predictions a gru run wrote")
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(QUANTILES),
        help="quantiles of |prediction| to take the thresholds at",
    )
    ap.add_argument("--fee", type=float, default=FEE, help="per side; the default is Alpaca taker tier 1")
    ap.add_argument(
        "--sign",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="-1 for the swing label (low means long), +1 for remaining excursion",
    )
    ap.add_argument(
        "--at",
        type=float,
        nargs="+",
        help="price these raw thresholds too, e.g. --at 0.4 for the -0.4/+0.4 pair",
    )
    ap.add_argument("--by-symbol", action="store_true", help="the first --at threshold priced on each pair alone")
    ap.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=[0],
        metavar="BARS",
        help="grade each entry over this many bars instead of over the hold; 0 is the hold itself",
    )
    ap.add_argument(
        "--by-move",
        type=int,
        default=0,
        metavar="N",
        help="split the first --at threshold's holds into N equal-count buckets of price move",
    )
    args = ap.parse_args()

    _selfcheck()
    pred = pd.read_parquet(args.pred).iloc[:, 0]
    close = prices(pred.index)
    print(f"{len(pred):,} rows, {pred.index.get_level_values(1).nunique()} symbols, fee {args.fee * 100:.2f}% per side")
    print(f"{pred.index.get_level_values(0).min():%Y-%m-%d} to {pred.index.get_level_values(0).max():%Y-%m-%d}\n")
    print(sweep(pred, close, args.quantiles, args.fee, args.sign).round(4).to_string())
    if args.at:
        share = [float((pred.abs() <= t).mean()) for t in args.at]
        print("\nfixed thresholds on the raw prediction, and the quantile of |pred| each one lands on\n")
        rows = pd.DataFrame([pnl(pred, close, t, args.fee, args.sign) for t in args.at])
        print(rows.assign(quantile=share).round(4).to_string(index=False))
    if args.at and args.by_symbol:
        t = args.at[0]
        print(f"\nthe {t:+.2f} pair on each symbol alone, with holding that symbol next to it\n")
        print(by_symbol(pred, close, t, args.fee, args.sign).round(4).to_string())
    if args.at and args.by_move:
        t = args.at[0]
        for h in args.horizon:
            over = f"the {h} bars after each entry" if h else "each hold"
            print(f"\nthe {t:+.2f} pair, {args.by_move} equal-count buckets of the |move| over {over}\n")
            print(by_move(pred, close, t, args.by_move, args.fee, args.sign, h).round(4).to_string())
    print(f"\nbuy and hold, same rows: {buy_and_hold(close)['net_per_year'] * 100:.1f}% log per year")


if __name__ == "__main__":
    main()
