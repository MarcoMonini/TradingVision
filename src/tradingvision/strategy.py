"""A trading strategy on the v2 prediction, built one entry and exit rule at a time.

The prediction is v2's (`swing --timeframe 15m --window 12 --smoothing 0.5 --inputs reduced
--steps 48 --test-start 2025-06`), out of sample on its four walk-forward folds. Each
step adds one rule to the rule before it and has to beat it on the same rows.

**The protocol, fixed before any rule was priced.**

- *The assets* are the three tradable pairs with the highest time-series Rank IC against
  `swing_leg_target` on folds 1-2: ETH 0.703, BTC 0.688, SOL 0.683. BNB is 0.678 but Alpaca does
  not list it. SOL is ahead of AAVE, LTC and UNI by under 0.01, less than one asset moves from one
  fold to the next. The ranking against the label is stable: Spearman 0.90 between the first and
  the second half. Against the 12-bar forward return it is 0.34, so it was not the criterion.
- *Development* is folds 1-2 (2025-06-01 to 2026-01-28): assets, thresholds and every rule
  parameter are chosen there. *Hold-out* is folds 3-4 (2026-01-28 to 2026-09-26) and is read once,
  at the end, on the rules development settled on. `--holdout` is the switch, so reading it is a
  decision someone has to make.
- *Regimes* follow the BTC cycle on the store's daily bars: the bear market opens at the cycle high
  (2025-10-06, 126,200) and the bull market at the cycle low (2026-07-01, 57,800). One calendar for
  every asset, because it is the market's cycle: ETH topped on 2025-08-24 and SOL on 2025-01-19, so
  calendars per asset would make SOL a bear market until June 2026. Development holds about 4 months of
  bull and 4 of bear, the hold-out 5 of bear and only 3 of bull.
- *No fees* for now, so every number is gross. `bp_trade` is the one that decides whether a rule
  survives the fee: the round trip is 20 bp at OKX's taker tier (`oracle.FEE`). It was 50 bp at
  Alpaca's, the figure the numbers below were read against.

**What step 1 found that shapes the rules.** In the bottom 10% of the prediction, where the band
buys, the price falls about 3 bp over the next 12 bars on 13 of 15 pairs. In the top 10%, where it
shorts, the price rises 3-15 bp on 12 of 15. The rank IC against the forward return is negative,
the right direction, on 14 of 15, so the reversion is in the middle of the range and the extremes
continue. That is why entering when the prediction comes back inside the band is the first
variant to try.

**Step 2, the band rule (always in, long under -a, short over +a), development.** Every threshold
from 0.2 to 0.7 grosses between -0.57 and +0.05 log per asset over the 8 months. The best is +1.7
bp a trade at 0.4, against an error of 9.8. The win rate of 60-65% is small wins against losses
twice their size. Split by regime, the long side wins in the bull market and loses in the bear
market at every threshold, and the short side does the opposite, so the market sets the sign and
the signal adds no timing.

**Step 3, re-entry.** It beats the band at every one of nine thresholds, with the same number of
trades: every excursion past the band ends in a re-entry, so the two rules trade the same events
and re-entry is the band delayed by the time the prediction spends outside. At 0.35 / 0.40 it
makes +12.9 / +15.0 bp a trade (error 7.6 / 10.1), +0.48 / +0.42 log, and pays on both sides. The
bull market is still at or under zero (-0.12 / -0.03) and the bear market carries the result
(+0.60 / +0.45). `momentum` is the band's exact mirror and is jagged across thresholds, so it
says nothing reliable.

**Step 4, re-entry on a moving average of 2, 4 or 8 bars** (`--smooth`), thresholds 0.35 to 0.95.
At 0.40, 2 bars give +16.6 bp a trade against +15.0 raw, 4 give +15.8 and 8 give +7.0, with an
error of 10-16. Up to 4 bars it is a wash and at 8 the delay costs more than the noise it removes.
The peak stays at 0.35-0.40 for every average. Above 0.5 every row loses or is noise. From 0.7 up
there are fewer than 60 trades in 8 months and none past 0.85, and always-in turns a rare entry
into a hold of weeks, so the threshold also sets the holding time. Only an exit separates the two.

**Step 5, exits on re-entry** (`--tp`, `--sl`, `--trail`, through `stops.walk` on 15m bars, ATR at
12: 25 bp on BTC, 39 on ETH, 46 on SOL). A take profit hurts at every size that ever fires: at
0.40, 3 ATR takes the trade from +15.0 to +4.0 bp, because the winners' right tail pays for
everything. A stop of 6 ATR, fixed or trailing at 6-10, lifts the total a little (+0.42 to +0.49
at 0.40) and the bull market to about zero. It lowers the bp a trade to 12-14, because a stopped
side re-arms and trades again. Every difference is inside one error. `--path` says why no fixed
horizon helps: the average re-entry trade is flat for 2 hours, peaks at +5 to +8 bp after 6, and
is negative after 24, while the hold to the opposite crossing makes +13 to +15. The signal's own
exit beats any clock.

**Where the gross comes from** (`excess_bp`, the gross minus the asset's mean bar return over the
segment times the bars held and the side). At 0.40 re-entry makes +31.6 bp a trade of timing in
the bear market, +30.7 on longs and +32.5 on shorts, and -1.8 in the bull market. The band rule
makes +5.6 and -3.7. Volatility at entry does not separate the trades. BTC under its 200-day
average does: +43 bp against -3, the same split as the cycle dates, about a month late. But in
development the bull market is fold 1 and the bear market is fold 2. Only 20 of the 856 trades
fall in a fold-2 bull market, so a better second model and a signal that works in bear markets
read the same here.

**Step 6, the filter, and the candidates fixed for the hold-out** (`CANDIDATES`, `--candidates`).
`below` lets the rule trade only while BTC's last daily close is under its 200-day mean, which in
development is 37% of the bars from November 2025. At 0.40 re-entry with the filter makes the same
+0.42 as without it on 315 trades instead of 856: +41.0 bp a trade (error 16.1), profit factor
1.52, drawdown 15% against 29%. At 0.35 it is +38.6 (11.9). A 6-ATR stop on top lowers the bp a
trade to 27-28 and the drawdown by 2 points. These are in-sample for the filter, which was chosen
on the bear half of this period.

**The hold-out, read once on 2026-10-03 at 0.40: every candidate loses, in both regimes.**

| rule | trades | gross | bp a trade (error) | excess bp | drawdown |
|---|---|---|---|---|---|
| band | 622 | -0.52 | -25.2 (13.9) | -24.9 | 43% |
| reentry | 622 | -0.88 | -42.9 (13.9) | -42.7 | 61% |
| reentry + stop 6 ATR | 969 | -0.73 | -23.0 (7.6) | -22.8 | 54% |
| reentry + BTC<200d | 530 | -0.71 | -40.5 (14.9) | -40.2 | 52% |
| reentry + BTC<200d + stop 6 ATR | 815 | -0.57 | -21.4 (8.4) | -21.2 | 47% |

BTC was under its 200-day mean on 84% of the hold-out, so the filter was on for nearly all of it,
and the bear market that development said carried +32 bp of timing gives -39 here. By fold,
re-entry makes -1.8, +32.8, -44.8 and -40.9 bp a trade (errors 13-20), and it loses on all three
assets in the hold-out. The development result was fold 2 alone: the confound step 5 named came out
as the second model, not the regime. Re-entry against the band flips sign with it (+13 in
development, -18 here), so what the price does while the prediction sits past the band is not a
stable property either. The stop only cuts the losses by holding less.

**After the hold-out: what follows a stop** (`--after`, development only). Every stop above used
`rearm`, which re-enters on the next re-entry of either side, the stopped one included. `opposite`
stays flat until the other side's re-entry, so a stop never adds a trade: at 0.40 with a 6-ATR
stop it keeps the 856 trades and makes +26.1 bp a trade (error 7.9), +0.74 log, profit factor
1.34, drawdown 13% against 29%. It is the first rule positive in both regimes of development, +16.0
bp in the bull market and +37.9 in the bear market. A 10-ATR stop gives +24.5, trailing at 10
gives +25.2, and at 0.35 the gain is smaller (+14.4 / +15.2 against +12.9). `reverse`, flipping at
the stop price, is worse than either: +9.0 at 6 ATR on 1,560 trades. The re-entries `rearm`
allows on the stopped side are where it loses. None of this has a clean out-of-sample reading:
the hold-out was spent before it was measured.

Read on the spent hold-out as a check, with the rule fixed first (0.40, 6-ATR stop, `opposite`):
-26.5 bp a trade (error 8.9), -0.55 log, profit factor 0.75, drawdown 43%, -24.6 in the bull market
and -27.5 in the bear market. That is no better than the `rearm` stop's -23.0. By fold it makes
+14.6, +38.3, -32.6 and -20.1, and it loses on all three assets (BTC -21.7, ETH -31.4, SOL -26.7).
With the BTC filter, +37.2 bp in development and -22.8 (9.8) in the hold-out. The policy lifts
fold 1 from -1.8 to +14.6 and does nothing for folds 3 and 4.

**The worst rule, inverted.** With no fee, taking every trade on the other side flips the sign of
every number, so inverting the worst rule of a period wins on that period by construction. The
test is whether the worst stays the worst. The worst rule of development with at least 200 trades
is the band at 0.55 (-0.566 log, -70.6 bp a trade on 285). Its inverse is `momentum` at 0.55, and
on the hold-out, read as a check, it makes +11.6 bp a trade (error 38.6) on 221 trades: -56.8 in
the bull market, +47.4 in the bear market, +123.6 / +20.0 / +58.5 / -45.1 by fold. That is
positive and indistinguishable from zero, under any fee. The worst rule of the hold-out, re-entry
at 0.40, inverted makes +42.9 there by construction and -15.0 in development.

**Hindsight: trading the turns of the prediction** (`--hindsight`, `turns`; it looks ahead and is
not a strategy). Long from every low of the prediction to its next high and short back, with the
pivots found by a centred window of 6, 12, 24 or 48 bars, against the same rule on the turns of
`rsi_centered` at 12 and of the price, which is the oracle. At 12 bars the prediction's turns make
+180.4 bp a trade in development and +164.4 in the hold-out, against +209.6 / +193.3 for the
price's own and +1.7 / -25.2 for the band at 0.40. In total they reach 96-100% of the oracle's
gross, because they turn more often than the price at the same window. It is the first result
of the study that holds in all four folds, and an RSI does nearly as well: +169.7 / +157.5. A
trade on the prediction's turns beats one on the RSI's in every fold at every window, by 4-13 bp at
6, 6-14 at 12, 8-22 at 24 and 16-40 at 48. At 24 and 48 the two make the same number of trades, so
there the comparison is like for like. That is the model's own contribution to where the turns
are, and it repeats. What does not exist is a way to know the turn while it happens. The threshold
rules keep between -25 and +15 bp of the ~165 a turn is worth at 12 bars.

**The same turns, traded at their confirmation** (`--hindsight ... --causal`, `confirmed_turns`).
Acting on each turn `window` bars after it, the first bar a live reader knows it, takes all of it
away. The prediction's confirmed turns make +1.8 / +4.8 / -6.7 / -10.3 bp a trade in development
at 6 / 12 / 24 / 48 bars and -2.4 / -3.3 / +0.8 / +16.1 in the hold-out, win rate 40-47%, with
signs that change from fold to fold. The RSI's and the price's own confirmed turns do the same,
between -10 and +13. By the time a turn of the prediction is confirmed the price has used up the
move, and the prediction confirms its turns no earlier than the price or an RSI does. Its turns
are better placed than an RSI's, and they arrive just as late.

**Between the two: the same turns, acted on 1 to 6 bars late** (`--hindsight ... --delay`,
`delayed`; still the centred turns, so still a diagnostic). It measures the room a reader that
recognises a turn before its confirmation would have. At 12 bars the prediction's turns make
+180.4 / +126.1 / +99.4 / +79.4 / +68.0 / +54.1 / +41.3 bp a trade in development at delay 0 to 6,
and +164.4 / +115.6 / +88.9 / +70.4 / +54.9 / +44.9 / +34.5 in the hold-out, in every fold the same
shape: 70% of the value is left after one bar, 55% after two, 23% after six, 4% after twelve. The
first bar costs the most, because the turn bar's close is the best price the leg has. What is left
depends on the delay as a share of the window: 20-25% at half the window (6 of 12, 12 of 24, 3 of
6) and about 45% at a quarter of it (6 of 24, 12 of 48). Against a 50 bp round trip the turns at
12 bars still pay at four bars late in both periods (+68.0 / +54.9), at 24 bars at six (+105.1 /
+92.1), at 6 bars only at one. The prediction's lead over the RSI's turns is in the turn bar and the
next one or two: at 12 bars it is +10.7 / +6.9 bp at delay 0, gone by delay 3 (-1.4 / -1.4), and at
delay 6 the RSI's turns pay more (+47.6 / +40.7 against +41.3 / +34.5). Only at 48 bars does the
prediction keep it (+178.9 / +172.7 against +165.9 / +160.1). Every number here is an upper bound:
an early reader would also act on turns that never come, and these rows have none.

**The same curve on a price nobody can predict** (`--null N`, `signflip`). Rebuilt from its own
returns with every sign drawn at random, the price keeps its volatility clusters and tails and loses
any direction. Its turns at 12 bars make +216 to +219 bp a trade against +201 on the real price, an
RSI's turns +169 to +172 against +164, and at 24 bars +325 to +328 against +308. The share left at
each delay is the real one within 0.01-0.03, and it follows 1 - sqrt(d / window), the share a
Brownian path keeps after an extreme: 0.71 / 0.59 / 0.42 / 0.29 at d = 1 / 2 / 4 / 6 of 12, against
0.73 / 0.59 / 0.40 / 0.27 measured. The value of a turn, and the speed it runs out at, is the
geometry of the extremes of any noisy path. What a causal reader can get of it is `detect`'s.

**On the chart page.** `play` runs any rule above on any prediction, and `chart.py` draws through
it, so the page and these tables share one code path: rule, threshold or turn window, the mean of
the prediction, inversion, the BTC filter (on Alpaca's daily closes there) and every exit.

    uv run python -m tradingvision.strategy --rules band reentry
    uv run python -m tradingvision.strategy --rules reentry --smooth 1 2 4 8
    uv run python -m tradingvision.strategy --at 0.35 0.4 --path 1 2 4 8 12 24 48 96 192
    uv run python -m tradingvision.strategy --at 0.35 0.4 --tp 0 3 6 10 15 --sl 0 3 6 10 15 [--trail]
    uv run python -m tradingvision.strategy --hindsight 6 12 24 48 --delay 0 1 2 3 4 5 6 12
    uv run python -m tradingvision.strategy --hindsight 12 24 --delay 0 1 2 3 4 6 12 --null 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from tradingvision import legs, stops, threshold
from tradingvision.data.binance import STORE
from tradingvision.data.binance import load as candles
from tradingvision.data.pivots import find_pivots

PRED = STORE / "pos-swing-15m-reduced-swing-s48-t2025-06-w12-m0.50-label.parquet"
ASSETS = ("ETH", "BTC", "SOL")
TEST_START = pd.Timestamp("2025-06", tz="UTC")
FOLDS = 4
TOP = pd.Timestamp("2025-10-06", tz="UTC")  # BTC cycle high, 126,200
BOTTOM = pd.Timestamp("2026-07-01", tz="UTC")  # BTC cycle low, 57,800
BAR = pd.Timedelta("15min")
YEAR = threshold.YEAR
# v2's feature window. A barrier in ATR is then in the unit the model's own ATR column is in.
WINDOW = 12


def load(path: Path = PRED, assets=ASSETS) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """`(pred, close, cut)` on the (open_time, symbol) index `threshold` reads; `cut` opens the hold-out.

    The cut is `swing.folds`' boundary between folds 2 and 3, recomputed from the file. The file
    holds every test row of the walk-forward, so its last timestamp is the one `folds` used.
    """
    d = pd.read_parquet(path)
    cut = TEST_START + (d.index.max() - TEST_START) * 2 / FOLDS
    d = d[d.symbol.isin(assets)].set_index("symbol", append=True).sort_index()
    return d.pred, d.close, cut


def ohlc(index: pd.MultiIndex) -> pd.DataFrame:
    """The 15m bar at each row. `stops.frames` reads the 5m store at the same timestamps, which on a
    15m index would hand back the first five minutes of every bar as if they were all fifteen."""
    parts = []
    for symbol in index.get_level_values(1).unique():
        one = candles(symbol, "15m")[list(stops.OHLC)]
        parts.append(one.set_axis(pd.MultiIndex.from_arrays([one.index, [symbol] * len(one)], names=index.names)))
    out = pd.concat(parts).reindex(index)
    if out.isna().any().any():
        raise ValueError(f"{int(out.isna().any(axis=1).sum())} rows have no 15m bar")
    return out


def crossings(pred: pd.Series, a: float) -> pd.Series:
    """+1 on the bar the prediction crosses back above `-a` from at or below it, -1 on the bar it
    crosses back below `+a`, 0 everywhere else. The re-entry events, before any position is held."""
    prev = pred.groupby(level=1).shift()
    out = pd.Series(0.0, index=pred.index)
    out[(prev <= -a) & (pred > -a)] = 1.0
    out[(prev >= a) & (pred < a)] = -1.0
    return out


def hold(events: pd.Series) -> pd.Series:
    """Each nonzero event held until the next one: the always-in position a stream of signals means."""
    return events.replace(0.0, np.nan).groupby(level=1).ffill().fillna(0.0)


def reentry(pred: pd.Series, a: float) -> pd.Series:
    """Always in, entering when the prediction comes back inside the band rather than when it leaves.

    Each position is held until the opposite crossing. The band rule buys the bar the prediction
    falls through `-a`, which step 1 found is where the price keeps falling; this one buys after
    the fall has stopped.
    """
    return hold(crossings(pred, a))


# Every rule maps (pred, a) to a position. `momentum` is the band rule with the side flipped. Always
# in, it flips on the same bars as the band, so its gross is exactly minus the band's. It is a
# control and not a strategy: it says whether following the extremes beats fading them.
RULES = {
    "band": lambda pred, a: threshold.positions(pred, a),
    "momentum": lambda pred, a: threshold.positions(pred, a, sign=1),
    "reentry": reentry,
}


def label(when: pd.DatetimeIndex, cut: pd.Timestamp) -> pd.DataFrame:
    """The period and the regime of each timestamp."""
    bear = (when >= TOP) & (when < BOTTOM)
    return pd.DataFrame(
        {"period": np.where(when < cut, "dev", "holdout"), "regime": np.where(bear, "bear", "bull")}, index=when
    )


def trades(pos: pd.Series, close: pd.Series) -> pd.DataFrame:
    """One row per hold: symbol, entry time, side, gross log return, bars held.

    Same accounting as `threshold.legs`, plus the entry time, which is what puts a trade in a
    period and a regime. A trade belongs to the segment it was opened in.
    """
    r = threshold.forward_return(close).fillna(0.0)
    f = pd.DataFrame(
        {"pos": pos.to_numpy(), "r": r.to_numpy(), "t": pos.index.get_level_values(0)},
        index=pos.index.get_level_values(1).rename("symbol"),
    )
    f["leg"] = (f.pos != f.groupby(level=0).pos.shift()).groupby(level=0).cumsum()
    held = f[f.pos != 0]
    out = held.groupby([held.index, held.leg]).agg(
        entry=("t", "first"), side=("pos", "first"), move=("r", "sum"), bars=("r", "size")
    )
    out["gross"] = out.side * out.move
    return out.reset_index(level=1, drop=True)


def plain(pos: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """`(pos, pnl, trades)` for a rule with no barrier: each bar earns its close-to-close return."""
    return pos, pos * threshold.forward_return(close).fillna(0.0), trades(pos, close)


def walked(
    sig: pd.Series,
    bars: pd.DataFrame,
    take: tuple[str, float] | None = None,
    stop: tuple[str, float] | None = None,
    after: str = "rearm",
    after_take: str | None = None,
    trail: bool = False,
    tie_stop: bool = True,
    on: pd.Series | None = None,
    atr_window: int = WINDOW,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Any signal through `stops.walk`, with its barriers: `(pos, pnl, trades)`.

    `sig` is +1 for long, -1 for short and 0 for nothing new. `walk` enters on a nonzero signal
    that is not the side already held, so a repeat of the held side is ignored and the other side
    flips. A level signal (the band, +1 for as long as the prediction stays past it) and an event
    signal (a re-entry or a turn, +1 on one bar) both work. With no barrier the result is `hold`
    of the signal, bar for bar.

    After a barrier the policy decides: `rearm` waits for the next signal on either side (a zero
    between two signals re-arms the side just closed), `opposite` only for the other side,
    `reverse` takes the other side at the fill. `after_take` is the policy after a take profit
    and defaults to `after`. The barriers are fixed at the entry bar, filled at the level or at
    the open of a bar that gaps through it, and `tie_stop` takes the loss when one bar touches
    both. `atr_window` is the ATR a barrier in `atr` units is measured in.

    `on` is a filter: the rule trades only on the bars where it is true. Each run of such bars is
    walked on its own, starting flat, so after the filter switches on the rule waits for a fresh
    signal instead of joining a leg halfway; whatever is held when it switches off is closed at
    that bar's close, `why = "filter"`.

    One row per hold in `trades`: entry and exit time, side, entry and exit price, gross log
    return, bars held, and why it closed (`signal`, `stop`, `take`, `filter`, or `open` at the end).
    """
    sig = sig.to_numpy(dtype=float)
    atr = stops.atr_pct(bars, atr_window)
    tp, sl = stops.width(take, atr).to_numpy(), stops.width(stop, atr).to_numpy()
    prices = [bars[c].to_numpy() for c in stops.OHLC]
    index = bars.index
    symbol, when = index.get_level_values(1), index.get_level_values(0)
    live = np.ones(len(index), dtype=bool) if on is None else on.to_numpy(dtype=bool)
    pos, pnl, rows = np.zeros(len(index)), np.zeros(len(index)), []
    for s in symbol.unique():
        i = np.flatnonzero(symbol == s)
        for run in np.split(i, np.flatnonzero(np.diff(live[i].astype(int))) + 1):
            if not live[run[0]]:
                continue
            p, ret, _, _, holds = stops.walk(
                sig[run], *(c[run] for c in prices), tp[run], sl[run], after, after_take or after, tie_stop, trail
            )
            cut_short = run[-1] != i[-1]  # the filter, not the data, ends this run
            if cut_short:
                p[-1] = 0.0
            pos[run], pnl[run], t = p, ret, when[run]
            rows += [
                {
                    "symbol": s,
                    "entry": t[e],
                    "exit_time": t[x],
                    "side": side,
                    "price": px,
                    "exit": out,
                    "gross": side * np.log(out / px),
                    "bars": x - e,
                    "why": "filter" if why == "open" and cut_short else why,
                }
                for e, x, side, px, out, why in holds
            ]
    columns = ["symbol", "entry", "exit_time", "side", "price", "exit", "gross", "bars", "why"]
    held = pd.DataFrame(rows, columns=columns).set_index("symbol")
    return pd.Series(pos, index=index), pd.Series(pnl, index=index), held


def exits(
    pred: pd.Series,
    bars: pd.DataFrame,
    a: float,
    take: tuple[str, float] | None = None,
    stop: tuple[str, float] | None = None,
    after: str = "rearm",
    trail: bool = False,
    tie_stop: bool = True,
    on: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Re-entry with a take profit and a stop loss: `walked` on the re-entry events.

    With no barrier it is `reentry` exactly (`_selfcheck` asserts it bar for bar).
    """
    return walked(crossings(pred, a), bars, take, stop, after, None, trail, tie_stop, on)


def below(index: pd.MultiIndex, symbol: str = "BTC", days: int = 200, daily: pd.Series | None = None) -> pd.Series:
    """True where `symbol`'s last complete daily close is under its `days`-day average.

    The causal stand-in for the bear market. Each bar reads the close of the day before its own,
    so nothing in it is known later than a live reader would know it. On BTC it switches on in
    November 2025, a month after the cycle high, and off in August-September 2026. `daily` is the
    daily close to read instead of the store's, which is how the chart page passes Alpaca's.
    """
    d = candles(symbol, "1D").close if daily is None else daily
    flag = (d < d.rolling(days).mean()).astype(float).where(d.rolling(days).count() == days).shift(1)
    day = index.get_level_values(0).floor("D")
    return pd.Series(flag.reindex(day).to_numpy() == 1.0, index=index)


def stats(bars: pd.DataFrame, held: pd.DataFrame) -> dict:
    """Bar-level P&L for the totals, trade-level for everything counted per trade.

    `bars` holds `pnl` and `r` on the (open_time, symbol) index; totals are the mean across assets,
    which is a portfolio of one equal unit per asset. Drawdown is read on that portfolio.
    """
    when = bars.index.get_level_values(0)
    span = (when.max() - when.min() + BAR) / YEAR
    n = bars.index.get_level_values(1).nunique()
    gross = bars.pnl.groupby(level=1).sum().mean()
    equity = bars.pnl.groupby(level=0).mean().cumsum()
    drawdown = (equity.cummax().clip(lower=0) - equity).max()
    g = held.gross
    wins, losses = g[g > 0], g[g <= 0]
    # What the trade made beyond holding its side through the segment's average bar: the asset's
    # mean bar return over the segment, times the bars held, times the side. A short in a falling
    # market earns the fall without any timing, and this is the part of the gross that is timing.
    drift = held.side * held.bars * bars.r.groupby(level=1).mean().reindex(held.index).to_numpy()
    return {
        "trades": len(held),
        "trades_yr": len(held) / n / span,
        "gross": gross,
        "gross_yr": gross / span,
        "bp_trade": g.mean() * 1e4,
        # Holds of one asset never overlap, so the trades are close to independent draws; the three
        # assets share the market and make this a floor on the real error, not the error itself.
        "se_bp": g.std() / np.sqrt(len(g)) * 1e4,
        "win": (g > 0).mean(),
        "win_bp": wins.mean() * 1e4,
        "loss_bp": losses.mean() * 1e4,
        "pf": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "mdd": 1 - np.exp(-drawdown),
        "in_mkt": (bars.pos != 0).mean(),
        "hours": held.bars.mean() * BAR / pd.Timedelta("1h"),
        "long_bp": g[held.side > 0].mean() * 1e4,
        "short_bp": g[held.side < 0].mean() * 1e4,
        "excess_bp": (g - drift).mean() * 1e4,
        "long_x": (g - drift)[held.side > 0].mean() * 1e4,
        "short_x": (g - drift)[held.side < 0].mean() * 1e4,
        "stopped": (held.why == "stop").mean() if "why" in held else np.nan,
        "taken": (held.why == "take").mean() if "why" in held else np.nan,
        "hold": bars.r.groupby(level=1).sum().mean(),
    }


def report(
    pos: pd.Series, pnl: pd.Series, held: pd.DataFrame, close: pd.Series, cut: pd.Timestamp, period: str = "dev"
) -> pd.DataFrame:
    """One row per regime of `period` and one for the whole of it."""
    bars = pd.DataFrame({"pos": pos, "r": threshold.forward_return(close).fillna(0.0), "pnl": pnl})
    seg = label(bars.index.get_level_values(0), cut)
    held_seg = label(pd.DatetimeIndex(held.entry), cut).set_axis(held.index)
    rows = {}
    for regime in ("bull", "bear", "all"):
        keep = (seg.period == period) & ((seg.regime == regime) | (regime == "all"))
        took = (held_seg.period == period) & ((held_seg.regime == regime) | (regime == "all"))
        if keep.any():
            rows[regime] = stats(bars[keep.to_numpy()], held[took.to_numpy()])
    return pd.DataFrame(rows).T.rename_axis("regime")


SHOWN = ["trades", "gross", "bp_trade", "se_bp", "excess_bp", "win", "pf", "mdd", "hours", "long_bp", "short_bp"]


def line(rep: pd.DataFrame, extra=()) -> dict:
    """The whole period, then the gross and the bp a trade of each regime."""
    return (
        rep.loc["all", SHOWN + list(extra)].to_dict()
        | {f"{r}_gross": rep.loc[r, "gross"] for r in ("bull", "bear") if r in rep.index}
        | {f"{r}_bp": rep.loc[r, "bp_trade"] for r in ("bull", "bear") if r in rep.index}
        | {f"{r}_x": rep.loc[r, "excess_bp"] for r in ("bull", "bear") if r in rep.index}
    )


def invert(pos: pd.Series, pnl: pd.Series, held: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Every trade taken on the other side. With no fee it is the same ledger with the sign flipped."""
    return -pos, -pnl, held.assign(side=-held.side, gross=-held.gross)


def compare(
    pred: pd.Series,
    close: pd.Series,
    cut: pd.Timestamp,
    rules,
    grid,
    period: str = "dev",
    smooth=(1,),
    inverse: bool = False,
) -> pd.DataFrame:
    """One line per (rule, smoothing, threshold).

    `k` is the moving average the rule reads instead of the raw prediction (`threshold.smoothed`,
    the last `k` bars of the same asset, causal); 1 is the raw prediction. `inverse` takes every
    trade of every rule on the other side.
    """
    rows = []
    for k in smooth:
        p = threshold.smoothed(pred, k)
        for name in rules:
            for a in grid:
                got = plain(RULES[name](p, a), close)
                rep = report(*(invert(*got) if inverse else got), close, cut, period)
                rows.append({"rule": name, "k": k, "a": a} | line(rep))
    return pd.DataFrame(rows).set_index(["rule", "k", "a"]).astype(float)


# The rules the hold-out is read on, fixed on 2026-10-03 before it was read: the reference, re-entry,
# and re-entry with each of the two things development pointed at, alone and together.
CANDIDATES = {
    "band": {},
    "reentry": {},
    "reentry + stop 6 ATR": {"stop": ("atr", 6.0)},
    "reentry + BTC<200d": {"filter": True},
    "reentry + BTC<200d + stop 6 ATR": {"filter": True, "stop": ("atr", 6.0)},
}


def candidates(pred: pd.Series, bars: pd.DataFrame, cut: pd.Timestamp, a: float = 0.40, period: str = "dev"):
    """The candidates at one threshold: one line each, and the full regime table of each."""
    on = below(pred.index)
    rows, tables = [], {}
    for name, kw in CANDIDATES.items():
        if name == "band":
            got = plain(threshold.positions(pred, a), bars.close)
        else:
            got = exits(pred, bars, a, stop=kw.get("stop"), on=on if kw.get("filter") else None)
        tables[name] = report(*got, bars.close, cut, period)
        rows.append({"rule": name} | line(tables[name], ["in_mkt", "stopped"]))
    return pd.DataFrame(rows).set_index("rule").astype(float), tables


def barriers(
    pred: pd.Series, bars: pd.DataFrame, cut: pd.Timestamp, grid, takes, stops_, trail=False, period: str = "dev", **kw
) -> pd.DataFrame:
    """One line per (threshold, take, stop), in ATR multiples; 0 is no barrier, so (0, 0) is re-entry."""
    rows = []
    for a in grid:
        for tp in takes:
            for sl in stops_:
                got = exits(pred, bars, a, ("atr", tp) if tp else None, ("atr", sl) if sl else None, trail=trail, **kw)
                rep = report(*got, bars.close, cut, period)
                rows.append({"a": a, "tp": tp, "sl": sl} | line(rep, ["stopped", "taken"]))
    return pd.DataFrame(rows).set_index(["a", "tp", "sl"]).astype(float)


def path(pred: pd.Series, close: pd.Series, cut: pd.Timestamp, a: float, horizons, period: str = "dev") -> pd.DataFrame:
    """The average re-entry trade, `h` bars after it opened, whatever the rule did next.

    Signed by the side, so positive is the trade going the right way. It says where along the hold
    the edge is earned, which is where an exit can keep it: an edge that has all arrived after 12
    bars and then fades asks for a time exit or a take profit, one that keeps growing asks to be
    left alone.
    """
    events = crossings(pred, a)
    seg = label(events.index.get_level_values(0), cut)
    keep = (events != 0).to_numpy() & (seg.period == period).to_numpy()
    side, regime = events[keep], seg.regime.to_numpy()[keep]
    rows = {}
    for h in horizons:
        g = (side * np.log(close.groupby(level=1).shift(-h) / close)[keep]).to_numpy()
        ok = ~np.isnan(g)
        g, s, r = g[ok], side.to_numpy()[ok], regime[ok]
        rows[h] = {
            "hours": h * BAR / pd.Timedelta("1h"),
            "bp": g.mean() * 1e4,
            "se_bp": g.std() / np.sqrt(len(g)) * 1e4,
            "right": (g > 0).mean(),
            "long_bp": g[s > 0].mean() * 1e4,
            "short_bp": g[s < 0].mean() * 1e4,
            "bull_bp": g[r == "bull"].mean() * 1e4,
            "bear_bp": g[r == "bear"].mean() * 1e4,
        }
    return pd.DataFrame(rows).T.rename_axis("bars")


def fold_of(when: pd.DatetimeIndex, path: Path = PRED) -> np.ndarray:
    """The walk-forward fold, 1 to `FOLDS`, each timestamp falls in: `swing.folds`' edges from the file."""
    last = pd.read_parquet(path, columns=["symbol"]).index.max()
    edges = [TEST_START + (last - TEST_START) * i / FOLDS for i in range(1, FOLDS)]
    return np.searchsorted(pd.DatetimeIndex(edges), when, side="right") + 1


# The history no walk-forward of v2 tested on, up to its first test bar: where the plan of
# `false_alarms.html` confirms whatever does not read v2 (an RSI in its place, the residual, the
# liquidity states, the stops, the taker flow, the slow base). v2 trained on it, so nothing that
# reads v2 is confirmed here.
CONFIRM = (pd.Timestamp("2021-01", tz="UTC"), TEST_START)
PERIODS = ("dev", "holdout", "2021")


def edges(period: str = "dev", path: Path = PRED) -> pd.DatetimeIndex:
    """The fold boundaries of a study period, first bar to end: `dev` is v2's folds 1-2, where every
    criterion of the plan is applied; `holdout` its folds 3-4, read in §17 and so a check and never a
    choice; `2021` is `CONFIRM` cut in `FOLDS` equal slices."""
    if period == "2021":
        a, b = CONFIRM
        return pd.DatetimeIndex([a + (b - a) * i / FOLDS for i in range(FOLDS + 1)])
    last = pd.read_parquet(path, columns=["symbol"]).index.max()
    e = [TEST_START + (last - TEST_START) * i / FOLDS for i in range(FOLDS)] + [last + BAR]  # the last bar is in
    return pd.DatetimeIndex({"dev": e[:3], "holdout": e[2:]}[period])


def fold_in(when: pd.DatetimeIndex, period: str = "dev", path: Path = PRED) -> np.ndarray:
    """The fold of each timestamp inside `period`, 0 outside it. v2's periods keep v2's numbers (1-2,
    3-4) and agree with `fold_of`; `2021` numbers its slices 1 to `FOLDS`."""
    e = edges(period, path)
    k = np.searchsorted(e, when, side="right")
    first = 3 if period == "holdout" else 1
    return np.where((when >= e[0]) & (when < e[-1]), k + first - 1, 0)


def rsi(index: pd.MultiIndex, window: int = WINDOW, close: pd.Series | None = None) -> pd.Series:
    """`rsi_centered` at `window` on each asset's 15m close, the column `features` computes, read
    from the whole store so the warm-up is over before the first row. `close` replaces the store
    for a path that is not in it (`signflip`); its first `window` bars are then warm-up, at 0."""
    parts = []
    for symbol in index.get_level_values(1).unique():
        one = candles(symbol, "15m").close if close is None else close.xs(symbol, level=1)
        r = (RSIIndicator(one, window=window).rsi() / 50 - 1).fillna(0.0)
        parts.append(r.set_axis(pd.MultiIndex.from_arrays([r.index, [symbol] * len(r)], names=index.names)))
    return pd.concat(parts).reindex(index)


def turn_events(series: pd.Series, window: int) -> pd.Series:
    """-1 on every high of `series`, +1 on every low, 0 elsewhere, as `find_pivots` places them.

    The pivots are a centred window of `window` bars on each asset, so every one of them is known
    only `window` bars after it happened, and the merge of same-kind runs reads further still.
    """
    out = pd.Series(0.0, index=series.index)
    symbol = series.index.get_level_values(1)
    for s in symbol.unique():
        i = np.flatnonzero(symbol == s)
        one = pd.Series(series.to_numpy()[i], index=series.index.get_level_values(0)[i])
        # `amplitude` is a log of a price ratio, undefined on a signed series and unused here.
        with np.errstate(invalid="ignore", divide="ignore"):
            piv = find_pivots(one, window)
        out.iloc[i[one.index.get_indexer(piv.index)]] = -piv.kind.to_numpy(dtype=float)
    return out


def turns(series: pd.Series, window: int) -> pd.Series:
    """Long from every low of `series` to its next high, short from the high to the next low.

    This rule cannot be traded: it reads `window` bars of future and more (`turn_events`). It
    answers a different question: if the turns of `series` were known exactly, what would trading
    them earn? On the price it is the oracle; on the prediction or on an RSI it is how much of the
    price's turns those turns carry.
    """
    return hold(turn_events(series, window))


def confirmed_events(series: pd.Series, window: int) -> pd.Series:
    """The turns of `series` at the bar a live reader learns them: -1 for a high, +1 for a low.

    A raw extreme (`legs.raw_extrema`, `find_pivots` without its merge) is known at the close of
    the `window`-th bar after it, when nothing within that many bars has gone past it. A twin of
    the same kind is a second event on the same side, which an always-in rule ignores, so the
    merge that `turn_events` reads from the future is not needed here and nothing in the rule is.
    """
    out = pd.Series(0.0, index=series.index)
    symbol = series.index.get_level_values(1)
    for s in symbol.unique():
        i = np.flatnonzero(symbol == s)
        at, kind = legs.raw_extrema(pd.Series(series.to_numpy()[i]), window)
        known = at + window
        ok = known < len(i)
        out.iloc[i[known[ok]]] = -kind[ok].astype(float)
    return out


def confirmed_turns(series: pd.Series, window: int) -> pd.Series:
    """`turns` as a live reader could hold it: each turn acted on at the bar that confirms it."""
    return hold(confirmed_events(series, window))


# What each rule hands `walked`: +1 long, -1 short, 0 nothing new. `a` is the band of the threshold
# rules and `window` the turn window of the turn rules; each reads the one it needs.
SIGNALS = {
    "band": lambda pred, a, window: threshold.signals(pred, a),
    "reentry": lambda pred, a, window: crossings(pred, a),
    "momentum": lambda pred, a, window: -threshold.signals(pred, a),
    "confirmed": lambda pred, a, window: confirmed_events(pred, window),
    "turns": lambda pred, a, window: turn_events(pred, window),
}


def play(
    pred: pd.Series,
    bars: pd.DataFrame,
    rule: str = "band",
    a: float = 0.40,
    window: int = WINDOW,
    smooth: int = 1,
    inverse: bool = False,
    **kw,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Any rule of the study on any prediction: `(pos, pnl, trades)`.

    The rule's `signal` through `walked` with whatever barriers, policy and filter `kw` carries.
    `turns` reads the future and is a diagnostic, never a strategy.
    """
    return walked(signal(pred, rule, a, window, smooth, inverse), bars, **kw)


def signal(
    pred: pd.Series, rule: str = "band", a: float = 0.40, window: int = WINDOW, smooth: int = 1, inverse: bool = False
) -> pd.Series:
    """What `play` hands `walked`: the rule's signal on the prediction's `smooth`-bar mean
    (`threshold.smoothed`), flipped side for side when `inverse`. The chart page calls it and then
    `walked`, so the rule it draws, and the signals it marks, are the rule priced here."""
    sig = SIGNALS[rule](threshold.smoothed(pred, smooth), a, window)
    return -sig if inverse else sig


def hindsight(
    pred: pd.Series, close: pd.Series, cut: pd.Timestamp, windows, causal: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The turns of the prediction, of an RSI and of the price, against the two threshold rules.

    One line per (signal, window) for development and for the hold-out, and the bp a trade by
    fold. The RSI is the control the prediction has to beat: it is 92.5% of what the prediction
    knows about the label, and in hindsight the turns of any oscillator built on the price sit
    close to the price's own. `causal` trades each turn at its confirmation instead
    (`confirmed_turns`), the version that could run live; on the price it is the oracle at its
    detection lag.
    """
    series = {"prediction": pred, "rsi 12": rsi(pred.index), "price": close}
    books = {("band 0.40", 0): plain(threshold.positions(pred, 0.40), close)}
    books[("reentry 0.40", 0)] = plain(reentry(pred, 0.40), close)
    for name, x in series.items():
        for w in windows:
            books[(name, w)] = plain((confirmed_turns if causal else turns)(x, w), close)
    rows, by_fold = [], {}
    for (name, w), got in books.items():
        row = {"signal": name, "window": w}
        for period in ("dev", "holdout"):
            r = report(*got, close, cut, period)
            a = r.loc["all"]
            row |= {
                f"{period}_trades": a.trades,
                f"{period}_gross": a.gross,
                f"{period}_bp": a.bp_trade,
                f"{period}_se": a.se_bp,
                f"{period}_win": a.win,
                f"{period}_pf": a.pf,
                f"{period}_hours": a.hours,
                f"{period}_bull": r.loc["bull", "bp_trade"],
                f"{period}_bear": r.loc["bear", "bp_trade"],
            }
        rows.append(row)
        held = got[2]
        by_fold[(name, w)] = held.gross.groupby(fold_of(pd.DatetimeIndex(held.entry))).mean() * 1e4
    table = pd.DataFrame(rows).set_index(["signal", "window"]).astype(float)
    folds = pd.DataFrame(by_fold).T.rename_axis(["signal", "window"]).add_prefix("fold ")
    return table, folds


def late(events: pd.Series, bars: int) -> pd.Series:
    """`events` acted on `bars` bars after each one, on each asset."""
    return events.groupby(level=1).shift(bars).fillna(0.0)


def delayed(pred: pd.Series, close: pd.Series, cut: pd.Timestamp, windows, delays) -> pd.DataFrame:
    """The hindsight turns of the prediction, an RSI and the price, each acted on `delay` bars late.

    Between the turn itself (`turns`, delay 0) and its confirmation (`confirmed_turns`, delay =
    window) lies every reader that knows a turn some bars after it and before the window closes.
    The table says how fast the turn's value runs out with the delay, which is the room a rule that
    recognises the turn early would have. It still looks ahead: the turns are the centred ones.
    """
    series = {"prediction": pred, "rsi 12": rsi(pred.index), "price": close}
    rows = []
    for name, x in series.items():
        for w in windows:
            events = turn_events(x, w)
            for d in delays:
                pos, pnl, held = plain(hold(late(events, d)), close)
                row = {"signal": name, "window": w, "delay": d}
                for period in ("dev", "holdout"):
                    a = report(pos, pnl, held, close, cut, period).loc["all"]
                    row |= {f"{period}_trades": a.trades, f"{period}_bp": a.bp_trade, f"{period}_se": a.se_bp}
                fold = held.gross.groupby(fold_of(pd.DatetimeIndex(held.entry))).mean() * 1e4
                rows.append(row | fold.rename(lambda k: f"fold {k}").to_dict())
    return pd.DataFrame(rows).set_index(["signal", "window", "delay"]).astype(float)


def signflip(close: pd.Series, seed: int) -> pd.Series:
    """`close` rebuilt from its own 15m returns with every sign drawn at random.

    The sizes stay, and with them the volatility clusters and the tails; any direction, trend or
    reversion goes. No reader can predict this path, so whatever the turns of it are worth is the
    geometry of the extremes of noise.
    """
    rng = np.random.default_rng(seed)
    logp = np.log(close)
    r = logp.groupby(level=1).diff().fillna(0.0)
    walk = (r.abs() * rng.choice([-1.0, 1.0], len(r))).groupby(level=1).cumsum()
    return np.exp(logp.groupby(level=1).transform("first") + walk)


def null_delays(close: pd.Series, windows, delays, seeds) -> pd.DataFrame:
    """`delayed`'s curve on the real price and on sign-randomised ones (`signflip`), all 16 months.

    The turns of the price and of an RSI at 12, bp a trade at each delay. If the curves agree,
    the value of a turn and the speed it runs out at belong to any noisy path, not to the market.
    """
    paths = {"real": close} | {f"random signs #{k}": signflip(close, k) for k in seeds}
    rows = {}
    for path, c in paths.items():
        for name, x in (("price", c), ("rsi 12", rsi(c.index, close=c))):
            for w in windows:
                e = turn_events(x, w)
                rows[(path, name, w)] = [plain(hold(late(e, d)), c)[2].gross.mean() * 1e4 for d in delays]
    return pd.DataFrame(rows, index=list(delays)).T.rename_axis(["path", "turns of", "window"])


def _selfcheck() -> None:
    """Two symbols, a known position path, every number checked by hand; then the wiring to `stops`."""
    t = pd.date_range("2025-09-30", periods=8, freq="1D", tz="UTC")
    idx = pd.MultiIndex.from_product([t, ["A", "B"]], names=["open_time", "symbol"])
    close = pd.Series(np.repeat([100.0, 110, 121, 110, 100, 110, 121, 133.1], 2), index=idx)
    pos = pd.Series(np.repeat([0.0, 1, 1, -1, -1, 1, 1, 1], 2), index=idx)
    held = trades(pos, close)
    # A: long 110 -> 121 -> 110 (2 bars), short 110 -> 100 -> 110 (2 bars), long from 110 to the last bar.
    assert list(held.loc["A"].side) == [1, -1, 1] and list(held.loc["A"].bars) == [2, 2, 3]
    assert np.isclose(held.loc["A"].gross.iloc[0], 0.0) and np.isclose(held.loc["A"].gross.iloc[1], 0.0)
    assert np.isclose(held.loc["A"].gross.iloc[2], np.log(133.1 / 110))
    # Trade gross adds up to bar gross: the two ledgers never disagree.
    r = threshold.forward_return(close).fillna(0.0)
    assert np.isclose(held.gross.sum(), (pos * r).sum())
    # The confirmation period in four equal slices, nothing outside it, the end exclusive.
    when = pd.DatetimeIndex(["2020-12-31", "2021-01-01", "2022-03-01", "2025-05-31", "2025-06-01"], tz="UTC")
    assert list(fold_in(when, "2021")) == [0, 1, 2, 4, 0]
    # The cycle high on 2025-10-06 splits the week; the cut puts the last two days in the hold-out.
    seg = label(t, pd.Timestamp("2026-10-06", tz="UTC"))
    assert list(seg.regime) == ["bull"] * 6 + ["bear"] * 2 and set(seg.period) == {"dev"}
    rep = report(*plain(pos, close), close, pd.Timestamp("2026-01-01", tz="UTC"))
    assert rep.loc["all", "trades"] == 6 and np.isclose(rep.loc["all", "gross"], np.log(133.1 / 110))
    assert rep.loc["bull", "trades"] + rep.loc["bear", "trades"] == 6

    # Re-entry: out through -0.4 at bar 1, back inside at bar 3 -> long; out through +0.4 at bar 5,
    # still out at bar 6, back inside at bar 7 -> short. The band rule is long from bar 1, short from 5.
    t = pd.date_range("2025-06-01", periods=8, freq="15min", tz="UTC")
    one = threshold.on_one(pd.Series([-0.1, -0.5, -0.6, -0.3, 0.2, 0.5, 0.45, 0.1], index=t))
    assert list(crossings(one, 0.4)) == [0, 0, 0, 1, 0, 0, 0, -1]
    assert list(reentry(one, 0.4)) == [0, 0, 0, 1, 1, 1, 1, -1]
    assert list(RULES["band"](one, 0.4)) == [0, 1, 1, 1, 1, -1, -1, -1]
    # Momentum is the band's mirror, bar for bar, and inverting the band's ledger gives momentum's.
    assert (RULES["momentum"](one, 0.4) == -RULES["band"](one, 0.4)).all()
    fwd = threshold.on_one(pd.Series(np.linspace(100, 108, 8), index=t))
    flip, mom = invert(*plain(RULES["band"](one, 0.4), fwd)), plain(RULES["momentum"](one, 0.4), fwd)
    assert np.allclose(flip[1], mom[1]) and np.allclose(flip[2].gross, mom[2].gross)

    # With no barrier, `exits` is `reentry`: same position on every bar, same P&L, same trades.
    pred, bars = stops._saw(n=600)
    pos, pnl, held = exits(pred, bars, 0.4)
    want, want_pnl, want_held = plain(reentry(pred, 0.4), bars.close)
    assert (pos == want).all() and np.allclose(pnl, want_pnl)
    assert len(held) == len(want_held) and np.allclose(held.gross.sum(), want_pnl.sum())
    assert (held.why == "signal").sum() == len(held) - 1  # every hold closes on a flip but the last
    # The saw's wicks reach 0.2% under the close and a long never trades below its entry otherwise,
    # so a stop at 0.05% fires inside the wick and one at 0.4% never does.
    assert (exits(pred, bars, 0.4, stop=("pct", 0.004))[2].why == "signal").sum() == len(held) - 1
    _, _, stopped = exits(pred, bars, 0.4, stop=("pct", 0.0005))
    assert (stopped.why == "stop").any() and (stopped[stopped.why == "stop"].gross < 0).all()
    # A filter that is always on changes nothing; one that switches off in the middle leaves the
    # rule flat there, closes the hold it had, and waits for a fresh re-entry after it.
    every = pd.Series(True, index=pred.index)
    assert (exits(pred, bars, 0.4, on=every)[0] == pos).all()
    gap = pd.Series((np.arange(len(pred)) < 200) | (np.arange(len(pred)) >= 300), index=pred.index)
    gpos, gpnl, gheld = exits(pred, bars, 0.4, on=gap)
    assert (gpos.iloc[199:300] == 0).all() and (gpnl.iloc[199:300] == 0).all()
    assert (gheld.why == "filter").sum() == 1 and (gheld.entry.iloc[-1] > pred.index[300][0])
    first_after = crossings(pred, 0.4).iloc[300:].ne(0).idxmax()
    assert gpos.iloc[300 : pred.index.get_loc(first_after)].eq(0).all()

    # On the saw the prediction reads the price exactly, so its turns are the price's: long from
    # every low to the next high, and trading them is the oracle, which only ever wins.
    assert (turns(pred, 5) == turns(bars.close, 5)).all()
    _, oracle_pnl, oracle = plain(turns(bars.close, 5), bars.close)
    assert (oracle.gross > 0).all() and set(oracle.side) == {-1.0, 1.0}
    # `play` is every rule above through one door: with no barrier it holds what each rule holds.
    for rule, want in (
        ("band", threshold.positions(pred, 0.4)),
        ("reentry", reentry(pred, 0.4)),
        ("momentum", -threshold.positions(pred, 0.4)),
        ("turns", turns(pred, 5)),
        ("confirmed", confirmed_turns(pred, 5)),
    ):
        got, got_pnl, _ = play(pred, bars, rule, 0.4, 5)
        assert (got == want).all(), rule
        assert np.allclose(got_pnl, plain(want, bars.close)[1]), rule
    flipped = play(pred, bars, "reentry", 0.4, inverse=True)
    assert (flipped[0] == -reentry(pred, 0.4)).all() and np.allclose(flipped[1], -play(pred, bars, "reentry", 0.4)[1])
    assert (play(pred, bars, "reentry", 0.4, smooth=3)[0] == reentry(threshold.smoothed(pred, 3), 0.4)).all()
    # The saw has no twins, so the live reader holds the same turns `window` bars later.
    assert (confirmed_turns(pred, 5) == hold(late(turn_events(pred, 5), 5))).all()
    # A sign-randomised path starts where the price does and moves by the same amounts, bar for bar.
    walk = signflip(bars.close, 0)
    size = np.log(bars.close).groupby(level=1).diff().abs().dropna()
    assert np.allclose(np.log(walk).groupby(level=1).diff().abs().dropna(), size)
    assert np.isclose(walk.iloc[0], bars.close.iloc[0]) and not np.allclose(walk, bars.close)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, default=PRED)
    ap.add_argument("--at", type=float, nargs="+", default=list(np.round(np.arange(0.35, 0.951, 0.05), 2)))
    ap.add_argument("--rules", nargs="+", choices=list(RULES), default=list(RULES))
    ap.add_argument("--invert", action="store_true", help="take every trade of the rules on the other side")
    ap.add_argument("--smooth", type=int, nargs="+", default=[1], help="moving averages of the prediction, in bars")
    ap.add_argument("--detail", action="store_true", help="also the full table of every rule, regime by regime")
    ap.add_argument("--path", type=int, nargs="+", metavar="BARS", help="the average re-entry trade at these bars")
    ap.add_argument("--tp", type=float, nargs="+", help="re-entry with take profits at these ATR multiples, 0 = none")
    ap.add_argument("--sl", type=float, nargs="+", default=[0.0], help="and stop losses at these, 0 = none")
    ap.add_argument("--trail", action="store_true", help="the stop trails the hold's best price")
    ap.add_argument("--after", choices=stops.AFTER, default="rearm", help="what follows a barrier")
    ap.add_argument("--tie", choices=["stop", "take"], default="stop", help="a bar that touches both levels")
    ap.add_argument("--filter", action="store_true", help="with --tp: trade only while BTC is under its 200-day mean")
    ap.add_argument("--candidates", action="store_true", help="the rules fixed for the hold-out, at the first --at")
    ap.add_argument("--holdout", action="store_true", help="read the hold-out: once, on the final rules")
    ap.add_argument(
        "--hindsight",
        type=int,
        nargs="+",
        metavar="BARS",
        help="trade the turns of the prediction, an RSI and the price",
    )
    ap.add_argument("--causal", action="store_true", help="with --hindsight: act on each turn when it is confirmed")
    ap.add_argument("--delay", type=int, nargs="+", help="with --hindsight: act on each turn these many bars late")
    ap.add_argument("--null", type=int, metavar="N", help="with --delay: the same on N sign-randomised prices")
    args = ap.parse_args()

    _selfcheck()
    pred, close, cut = load(args.pred)
    period = "holdout" if args.holdout else "dev"
    when = pred.index.get_level_values(0)
    print(f"{', '.join(ASSETS)}; dev {when.min():%Y-%m-%d} to {cut:%Y-%m-%d}, hold-out to {when.max():%Y-%m-%d}")
    print(f"BTC cycle: bear from {TOP:%Y-%m-%d}, bull from {BOTTOM:%Y-%m-%d}; no fees, reading {period}\n")
    pd.set_option("display.width", 250)
    if args.candidates:
        bars = ohlc(pred.index)
        table, full = candidates(pred, bars, cut, args.at[0], period)
        print(table.round(3).to_string())
        for name, rep in full.items():
            print(f"\n{name} at {args.at[0]:.2f}\n")
            print(rep.astype(float).round(3).to_string())
        share = below(pred.index)[label(when, cut).period.eq(period).to_numpy()].mean()
        print(f"\nBTC under its 200-day mean on {share:.0%} of the bars")
    elif args.hindsight and args.delay and args.null:
        t = null_delays(close, args.hindsight, args.delay, range(1, args.null + 1)).sort_index(level=[2, 1, 0])
        print("bp a trade, all 16 months, no fees: real prices against prices with every return's sign at random\n")
        print(t.round(1).to_string())
        share = t.div(t[0], axis=0)
        w = share.index.get_level_values("window").to_numpy()[:, None]
        brownian = pd.DataFrame(1 - np.sqrt(np.minimum(np.array(args.delay) / w, 1)), index=share.index)
        print("\nshare of delay 0 left, and 1 - sqrt(d / w) under it\n")
        print(
            pd.concat({"measured": share, "1 - sqrt(d/w)": brownian.set_axis(share.columns, axis=1)}, axis=1)
            .round(2)
            .to_string()
        )
    elif args.hindsight and args.delay:
        table = delayed(pred, close, cut, args.hindsight, args.delay)
        print("looks ahead: the centred turns, each acted on `delay` bars after it\n")
        print(table.round(1).to_string())
    elif args.hindsight:
        table, folds = hindsight(pred, close, cut, args.hindsight, args.causal)
        print("causal: each turn at its confirmation\n" if args.causal else "looks ahead: not a strategy\n")
        for period in ("dev", "holdout"):
            print(f"{period}\n")
            cols = [c for c in table.columns if c.startswith(period)]
            print(table[cols].rename(columns=lambda c: c.split("_", 1)[1]).round(3).to_string() + "\n")
        print("bp a trade by fold (1-2 development, 3-4 hold-out)\n")
        print(folds.round(1).to_string())
    elif args.path:
        for a in args.at:
            print(f"the average re-entry trade at {a:.2f}, h bars after entry\n")
            print(path(pred, close, cut, a, args.path, period).round(3).to_string() + "\n")
    elif args.tp:
        bars = ohlc(pred.index)
        assert np.allclose(bars.close, close), "the 15m store and the prediction file disagree on the close"
        out = barriers(
            pred,
            bars,
            cut,
            args.at,
            args.tp,
            args.sl,
            args.trail,
            period,
            after=args.after,
            tie_stop=args.tie == "stop",
            on=below(pred.index) if args.filter else None,
        )
        print(out.round(3).to_string())
    else:
        out = compare(pred, close, cut, args.rules, args.at, period, args.smooth, args.invert)
        print(out.round(3).to_string())
    if args.detail:
        for k in args.smooth:
            for name in args.rules:
                for a in args.at:
                    print(f"\n{name}, mean of {k}, at {a:.2f}\n")
                    rep = report(*plain(RULES[name](threshold.smoothed(pred, k), a), close), close, cut, period)
                    print(rep.astype(float).round(3).to_string())


if __name__ == "__main__":
    main()
