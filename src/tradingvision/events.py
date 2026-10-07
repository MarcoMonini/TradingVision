"""Event studies: what the price does after an event, by the state it happened in (routes 3 and 4).

Routes 3 and 4 of the plan in `false_alarms.html` (cards `#liquidita` and `#shock`, HANDOFF §19)
ask the same question of two kinds of event: does the price come back after it, and does it come
back more in a state that information outside the price can name. One engine serves both; route
4 lands on it next.

**The engine.** `respond` reads, for each event (bar t, pair, direction d), the forward log return
against d, -d (log c[t+j] - log c[t]), from the close of the event bar: positive is a reversal, and
nothing after that close is read before the entry. `aggregate` averages by group and fold per
event, with the error clustered on the clock blocks of `metrics.blocked` (`clustered`: deviations
summed inside each block of h bars, then squared). The periods are `strategy.edges`': `dev`, v2's
folds 1-2, where every criterion is applied; `holdout`, folds 3-4, spent in §17 and a check only;
`2021`, 2021-01 to 2025-06 in four folds, where an RSI at 12 stands in for v2. Every cut (the
tails, the terciles) is fixed on dev's events and applied unchanged everywhere else.

**The mean is per event; the mean of blocks reads the future.** `metrics.blocked`'s mean is a mean
of block means, which weights an event by 1/n_b, the events of its whole block. At v2's extremes a
move that keeps going pulls more pairs into their tail later in the same block, so the weight
shrinks exactly the events that continued: fold 1 at 12 bars is +0.3 bp a trade per event, +21.2
as a mean of blocks, and +2.0 weighting by the events of the block so far; on the null paths,
where the pairs are independent, the three agree within their errors. Every table and verdict
reads the per-event mean; the mean of blocks is printed once, labelled, beside "every extreme".

**Route 3: the liquidity premium at v2's extremes** (`--liquidity`). An event is the bar |v2|
enters its 10% tail, one per excursion: |v2| >= 0.519 in the raw units, one cut for the 15 pairs
over dev's bars (0.581 at 5%, 0.474 at 15%, the robustness lines). The direction is the leg's: at a
v2 low the price had fallen 198 bp over the 12 bars before, at a high risen 172. 4,384 and 3,915
events in folds 1-2. The states, all known at the event bar's close, each oriented so that its
third tercile is the illiquid one (`states`): Amihud over a day against its 30-day median;
realised volatility over a day against its 30-day median; the hours of the leg (from `legs.state`'s
pivot) at their usual volume over the volume they traded; the book within 1% under its 30-day
median (BTC, ETH, SOL); US market open / weekday closed / weekend; the equal-weighted basket's
drawdown from its 3-day high. Tercile cuts from dev's events. The cost is the 20 bp round trip
plus the tercile's effective spread, Abdi-Ranaldo (2017) and not Corwin-Schultz (2012): it reads one
bar's close against its own and the next bar's mid-range rather than a two-day high-low ratio, so
it needs no overnight correction in a market that never closes, and the product is averaged before
the square root, so a negative estimate on one pair of bars is not floored to zero one by one. It
recovers 10, 20 and 60 bp on simulated bounces; here it gives 0-12 bp and 0 in most cells: these
pairs' spreads are under a basis point, and at 15 minutes the volatility is the estimator's floor.

Dev, per event, at 12 / 24 / 48 bars: fold 1 +0.3 (5.4) / -9.1 (9.5) / -14.7 (12.6), fold 2 -10.5
(11.3) / +4.4 (12.5) / +23.7 (18.5). The RSI at 12 does the same (-0.6 / -5.5 / -13.3 and -12.0 /
-1.4 / +17.6); its extremes on sign-randomised paths make -0.5 / +0.1 / +0.6 and -1.9 / -1.9 / -3.5,
its 108 tercile cells all within 2.2 errors of zero. By tercile at 48 bars, liquid / middle /
illiquid, fold 1 then fold 2, errors 14-43:

    amihud       -13.6 / -17.2 / -13.3     +34.1 / +19.1 / +21.3
    volatility    -9.7 / -53.0 / +23.0      +8.0 / +33.3 / +27.3
    leg volume   -20.0 / -25.1 /  +3.0      -2.9 / +43.8 / +27.1
    book depth   -13.7 /  -5.3 / -21.9     +63.9 / +14.6 / +28.1
    session       +6.5 / -32.2 /  +3.1     +13.7 / +31.7 / +20.5
    stress       -28.8 / -23.1 / +16.2     +19.8 /  +6.5 / +39.4

Fold 2 reverts in nearly every tercile and fold 1 continues in most: the period sets the sign, not
the state. The criterion (the illiquid tercile reverts in both folds, the terciles in order, the
illiquid one over 20 bp plus its spread, on TRADABLE) passes for none of the 18 (state, horizon)
pairs, for v2 or for the RSI, and the 5% and 15% tails move with the 10% one. The hold-out's
extremes continue: -15.6 (7.1) / -24.2 / -36.0 and -0.5 / -4.4 / -30.8, again 0 of 18.

The confirmation, 2021 (RSI at 12, 15 pairs, 17,600-19,300 events a fold), reverts a little at 12
bars in all four folds, +3.8 / +4.0 / +4.0 / +4.7 (errors 3.4-5.4), and not at 48 (-9.7 / +7.4 /
+5.4 / +1.9). Realised volatility orders the terciles at 12 bars in every fold, Nagel's sign:
-14.2 / -3.5 / +29.7, -8.4 / +4.1 / +16.1, -0.8 / +2.6 / +9.2, -0.6 / +0.8 / +13.6, but the volatile
tercile clears 20 bp only in fold 1, and on dev the same ordering breaks (RSI, fold 2: -12.1 / -0.7
/ -21.8). The leg's volume runs against the orientation fixed before looking: legs that traded
heavily revert (+29.8 / +17.0 / +7.3 / +13.1 at 12 bars) and thin ones continue (-14.7 / -7.9 /
+2.1 / +0.4), Campbell, Grossman and Wang's sign rather than the liquidity reading. Read on the
confirmation period, it is not a choice. 0 of 15 pass (no depth before May 2025).

`crowd`, the pairs that entered their tail in the hour up to an event, separates nothing on dev
(fold 1 at 48 bars, 1 / 2-3 / 4-7 / 8+ pairs: -17.4 / -0.3 / -16.3 / -25.7). Isolated extremes do not
revert; the gap between the two means above is the weight's look-ahead.

**Power** (`--power`, 2,000 studies a cell at dev's measured errors, on TRADABLE). The criterion
fires 0-1.3% of the time with no effect, finds an illiquid tercile reverting by the full cost
(20-28 bp) 7-18% of the time and by twice it 23-78%, best at 12 bars. The card could not see an
effect of the size it asks for; it sees nothing, and the signs that move between folds say the same.

    uv run python -m tradingvision.events --liquidity [--period dev|holdout|2021] [--seeds 0 1 2]
    uv run python -m tradingvision.events --power
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision import legs, strategy
from tradingvision.data import futures
from tradingvision.data.binance import SYMBOLS, TRADABLE
from tradingvision.data.binance import load as candles
from tradingvision.metrics import blocked
from tradingvision.oracle import FEE

BAR = strategy.BAR
DAY = 96  # 15m bars
MONTH = 30 * DAY
LEG = strategy.WINDOW  # v2's pivot window: the legs and the RSI are read at 12 bars
SPOT_RT = 2 * FEE * 1e4  # OKX spot taker round trip, bp

# Route 3. The tail is the share of v2's (or the RSI's) bars most extreme in absolute value, its
# cut taken on development and kept everywhere; 10% is step 1's tail (`strategy`), 5% and 15% the
# robustness lines.
TAIL, TAILS = 0.10, (0.05, 0.10, 0.15)
LIQ_H = (12, 24, 48)
# Every state oriented so that high is illiquid: tercile 1 is the liquid third, 3 the illiquid one.
STATES = ("amihud", "vol", "thin_leg", "thin_book", "session", "stress")
MIN_LEG = 4  # bars: a leg's volume is read over an hour at least

EFFECTS = ("0", "0.5x", "1x", "2x")


# --- the engine --------------------------------------------------------------------------------


def respond(logc: np.ndarray, at: np.ndarray, d: np.ndarray, horizons) -> np.ndarray:
    """bp against the event's direction, `-d * (log c[t+j] - log c[t])`, one row per event and one
    column per horizon `j`: positive is a reversal. The entry is the close of the event bar `t`, the
    first price anyone reacting to the event can trade, and nothing before it is read. NaN where
    `t + j` is past the series."""
    j = np.asarray(horizons)
    idx = at[:, None] + j[None, :]
    ok = idx < len(logc)
    move = np.where(ok, logc[np.where(ok, idx, 0)] - logc[at][:, None], np.nan)
    return -np.asarray(d, dtype=float)[:, None] * move * 1e4


def clustered(x: pd.Series, horizon: pd.Timedelta) -> tuple[float, float]:
    """`(mean, error)` of the events in `x` (indexed by time), each event weighted once and the
    error clustered on `metrics.blocked`'s blocks of the clock: the deviations are summed inside a
    block before they are squared, so events whose forward returns overlap count as one draw."""
    x = x.dropna()
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan
    s = (x - x.mean()).groupby(x.index.floor(horizon)).sum()
    k = len(s)
    return float(x.mean()), float(np.sqrt(k / max(k - 1, 1) * (s**2).sum()) / len(x)) if k > 1 else np.nan


def aggregate(ev: pd.DataFrame, by: list[str], horizons) -> pd.DataFrame:
    """Mean reversal by group and fold, per event with its clustered error, and as a mean of blocks.

    The forward returns of two events less than `h` bars apart overlap, and at v2's extremes events
    cluster: a market-wide move puts most of the 15 pairs in their tail on the same bars. `h` is the
    mean a trade on every event earns, the one every table and verdict reads, with `clustered`'s
    error. `h_cal` is `metrics.blocked`'s mean of block means, printed beside it only to show that
    it cannot be read: weighting an event by 1/n_b counts the events later in its block, and a move
    that keeps going pulls more pairs into their tail after it, so the weight reads the future. At
    v2's extremes on fold 1 it turns +0.3 bp a trade at 12 bars into +21.2; weighting by the events
    of the block so far instead gives +2.0. `spread_bp` is the Abdi-Ranaldo estimate of the group's
    effective spread, from the `ar` column of its events.
    """
    rows = []
    for key, g in ev.groupby([*by, "fold"], observed=True):
        row = dict(zip([*by, "fold"], key)) | {"n": len(g)}
        if "ar" in g:
            row["spread_bp"] = 1e4 * np.sqrt(max(4 * g.ar.mean(), 0.0))
        when = pd.DatetimeIndex(g.when)
        for h in horizons:
            x = pd.Series(g[f"r{h}"].to_numpy(), index=when).dropna()
            row[h], row[f"{h}_se"] = clustered(x, BAR * h)
            row[f"{h}_cal"] = blocked(x, BAR * h)["mean"]
        rows.append(row)
    return pd.DataFrame(rows).set_index([*by, "fold"])


def cells(t: pd.DataFrame, horizons, calendar: bool = False) -> pd.DataFrame:
    """`aggregate`'s table as text, one column per fold and horizon: `mean (error)` in bp per event,
    or the calendar-time mean."""
    out = pd.DataFrame(index=t.index)
    for h in horizons:
        if calendar:
            out[h] = [f"{m:+.1f}" for m in t[f"{h}_cal"]]
            continue
        out[h] = [f"{m:+.1f} ({s:.1f})" if np.isfinite(m) else "" for m, s in zip(t[h], t[f"{h}_se"])]
    out["n"] = t.n
    if "spread_bp" in t:
        out["spread"] = t.spread_bp.round(1)
    wide = out.unstack("fold")
    return wide.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def entries(x: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """`(positions, sides)` of the bars where `x` enters its tail: +1 at or over `cut`, -1 at or under
    `-cut`. One event per excursion, on the bar it begins; the bars it stays out there are not new."""
    side = np.where(x >= cut, 1, np.where(x <= -cut, -1, 0))
    prev = np.r_[0, side[:-1]]
    at = np.flatnonzero((side != 0) & (side != prev))
    return at, side[at]


def _one(c: pd.Series, symbol: str) -> pd.Series:
    """One symbol's series on the (open_time, symbol) index `strategy` reads."""
    return pd.Series(c.to_numpy(), index=pd.MultiIndex.from_arrays([c.index, [symbol] * len(c)]))


def _flip(close: pd.Series, symbol: str, seed: int) -> pd.Series:
    """`strategy.signflip` of one symbol, seeded per symbol: two pairs with the same history length
    would otherwise draw the same signs on the same bars and co-move on the null path."""
    return strategy.signflip(_one(close, symbol), seed * 100 + SYMBOLS.index(symbol)).droplevel(1)


def _rsi(close: pd.Series, symbol: str) -> pd.Series:
    """`strategy.rsi` at 12 on any close of one symbol."""
    one = _one(close, symbol)
    return strategy.rsi(one.index, LEG, one).droplevel(1)


# --- route 3: the states ------------------------------------------------------------------------


def same_hour(qv: pd.Series) -> pd.Series:
    """The typical volume of each bar's hour of day (UTC): the mean bar of that hour over the 30
    days before the bar's own day, so the day in progress never reads itself."""
    day, hour = qv.index.floor("D"), qv.index.hour
    hourly = qv.groupby([day, hour]).mean().unstack()
    typical = hourly.rolling(30, min_periods=10).mean().shift(1)
    return pd.Series(typical.to_numpy()[typical.index.get_indexer(day), hour], index=qv.index)


def session(when: pd.DatetimeIndex) -> np.ndarray:
    """0 while the US stock market is open (9:30-16:00 New York, Monday to Friday, its daylight
    saving followed through the time zone, holidays not taken out), 1 on a weekday outside it, 2 at
    the weekend (New York's Saturday and Sunday). Read at the bar's open."""
    ny = when.tz_convert("America/New_York")
    minute = ny.hour * 60 + ny.minute
    return np.where(ny.dayofweek >= 5, 2, np.where((minute >= 570) & (minute < 960), 0, 1))


def stress(returns: pd.DataFrame) -> pd.Series:
    """The equal-weighted basket's drawdown from its high of the last three days, as a positive
    number. The basket is the mean of the pairs trading at each bar, so a pair joins it when its
    history starts. Three days holds a fall built over one to three."""
    index = returns.mean(axis=1).cumsum()
    return index.rolling(3 * DAY, min_periods=DAY).max() - index


def states(bars: pd.DataFrame, market: pd.Series, depth: pd.Series | None = None) -> pd.DataFrame:
    """The liquidity state at each bar's close, every column oriented so that high is illiquid.

    `amihud`: the mean of |r| / quote volume over the last day, against its own 30-day median.
    `vol`: the day's realised volatility against its 30-day median. `thin_leg`: what the hours of the
    leg in progress usually trade (`same_hour`) over what this leg traded, the leg starting at the
    pivot `legs.state` holds at 12 bars, and an hour at least. `thin_book`: the notional within 1% of
    the price (`futures` `depth1`, a log) under its 30-day median, NaN without futures. `session`:
    `session`. `stress`: the basket's, `stress`. `ar`: the Abdi-Ranaldo product
    (c_k - eta_k)(c_k - eta_k+1), eta the bar's log mid-range, averaged over the day before the bar;
    four times its mean is the squared effective spread, which `aggregate` takes per group.
    Each column reads bars up to the bar's own close and none after it (`_selfcheck`).
    """
    c, n = np.log(bars.close), len(bars)
    r = c.diff()
    qv = bars.quote_volume.where(bars.quote_volume > 0).astype(float)

    def rel(x: pd.Series) -> pd.Series:
        return x / x.rolling(MONTH, min_periods=MONTH // 3).median()

    typical = same_hour(qv).to_numpy()
    age = (legs.state(bars.close, LEG).leg_age * LEG).round().to_numpy()
    i = np.arange(n)
    lo = np.clip(i + 1 - np.maximum(np.nan_to_num(age, nan=0.0), MIN_LEG).astype(int), 0, n)
    first = int(np.argmax(np.isfinite(typical))) if np.isfinite(typical).any() else n
    cq = np.r_[0.0, np.cumsum(qv.fillna(0.0).to_numpy())]
    ct = np.r_[0.0, np.cumsum(np.nan_to_num(typical))]
    with np.errstate(divide="ignore", invalid="ignore"):
        thin = np.where(np.isfinite(age) & (lo >= first), (ct[i + 1] - ct[lo]) / (cq[i + 1] - cq[lo]), np.nan)

    eta = (np.log(bars.high) + np.log(bars.low)) / 2
    product = (c - eta) * (c - eta.shift(-1))  # known at the close of the bar after
    out = pd.DataFrame(
        {
            "amihud": rel((r.abs() / qv).rolling(DAY, min_periods=DAY // 2).mean()),
            "vol": rel(r.rolling(DAY, min_periods=DAY // 2).std()),
            "thin_leg": thin,
            "thin_book": np.nan,
            "session": session(bars.index).astype(float),
            "stress": market.reindex(bars.index).to_numpy(),
            "ar": product.shift(1).rolling(DAY, min_periods=DAY // 2).mean(),
        },
        index=bars.index,
    )
    if depth is not None:
        out["thin_book"] = -(depth - depth.rolling(MONTH, min_periods=MONTH // 3).median()).reindex(bars.index)
    return out.replace([np.inf, -np.inf], np.nan)


def tercile_cuts(ev: pd.DataFrame) -> dict:
    """Every state's two cut points, from the events they are fixed on; `session` has its groups."""
    return {v: np.nanquantile(ev[v], [1 / 3, 2 / 3]) for v in STATES if v != "session"} | {"session": None}


def terciles(ev: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """The events once per state, with their tercile under `cuts` (1 liquid, 2, 3 illiquid); an
    event whose state is missing is left out of that state's rows."""
    parts = []
    for v in STATES:
        k = ev[v].to_numpy()
        t = k + 1 if cuts[v] is None else np.digitize(k, cuts[v]) + 1
        parts.append(ev.assign(state=v, tercile=np.where(np.isfinite(k), t, 0).astype(int)))
    out = pd.concat(parts, ignore_index=True)
    return out[out.tercile > 0]


def liquidity_events(period: str, seeds) -> tuple[pd.DataFrame, dict]:
    """Every extreme of v2 and of the RSI at 12, on the real path and on sign-randomised ones, with
    its states and its reversal at `LIQ_H`, over development and `period`: `(events, tail_cuts)`.

    The tails are cut on development's bars and applied unchanged: v2 on its raw output, one cut for
    the 15 pairs (it is trained across them and its units are shared), the RSI likewise. The null
    paths are `strategy.signflip`'s; the RSI and its extremes are recomputed on them through the real
    RSI's cut, and the states are the real ones at the same bars: sizes, volume and the market are
    kept, direction is gone. v2 cannot be computed on a rebuilt path, so the null is an RSI's.
    """
    pred = strategy.load(assets=SYMBOLS)[0] if period != "2021" else None
    returns, rsis = {}, {}
    for sym in SYMBOLS:
        c = candles(sym, "15m").close
        returns[sym] = np.log(c).diff()
        rsis[sym] = _rsi(c, sym)
    market = stress(pd.DataFrame(returns))
    del returns

    def tail_cuts(x: np.ndarray) -> dict:
        return dict(zip(TAILS, np.quantile(np.abs(x), [1 - q for q in TAILS])))

    def dev(s: pd.Series) -> np.ndarray:
        return s[strategy.fold_in(s.index, "dev") > 0].to_numpy()

    cut = {"rsi 12": tail_cuts(np.concatenate([dev(rsis[s]) for s in SYMBOLS]))}
    if pred is not None:
        cut["v2"] = tail_cuts(np.concatenate([dev(pred.xs(s, level=1)) for s in SYMBOLS]))

    rows = []
    for sym in SYMBOLS:
        bars = candles(sym, "15m")
        st = states(bars, market, futures.load(sym).depth1 if sym in strategy.ASSETS else None)
        keep = (strategy.fold_in(bars.index, "dev") > 0) | (strategy.fold_in(bars.index, period) > 0)
        logc = np.log(bars.close.to_numpy())
        # (family, the series whose tail makes the event, the log close the response is read on, cuts)
        paths = [("rsi 12", rsis.pop(sym).to_numpy(), logc, cut["rsi 12"])]
        if pred is not None:
            paths.append(("v2", pred.xs(sym, level=1).reindex(bars.index).fillna(0.0).to_numpy(), logc, cut["v2"]))
        for s in seeds:
            flipped = _flip(bars.close, sym, s)
            paths.append(
                ("rsi 12, random signs", _rsi(flipped, sym).to_numpy(), np.log(flipped.to_numpy()), cut["rsi 12"])
            )
        for family, x, lc, cuts in paths:
            for tail in TAILS:
                at, side = entries(np.where(keep, x, 0.0), cuts[tail])
                e = st.iloc[at].reset_index(names="when")
                e[[f"r{h}" for h in LIQ_H]] = respond(lc, at, side, LIQ_H)
                rows.append(e.assign(family=family, symbol=sym, share=tail, d=side))
    ev = pd.concat(rows, ignore_index=True)
    when = pd.DatetimeIndex(ev.when)
    ev["dev_fold"], ev["fold"] = strategy.fold_in(when, "dev"), strategy.fold_in(when, period)
    if period == "2021":
        # The book's depth starts in May 2025 with the futures dump: the last weeks of the period
        # would make a tercile of a few hundred events from three pairs. Left out of 2021 entirely.
        ev.loc[ev.fold > 0, "thin_book"] = np.nan
    return ev, cut


def liquidity_verdict(m: pd.DataFrame, cost: pd.Series, h: int) -> bool:
    """Route 3's criterion at horizon `h`, on `m` with rows (fold, tercile): in every fold of `cost`
    the illiquid tercile reverts, the three terciles are in order, and the illiquid one clears
    `cost`, the round trip plus that tercile's spread in the fold."""
    try:
        return all(
            0 < m.loc[(f, 3), h] and m.loc[(f, 1), h] <= m.loc[(f, 2), h] <= m.loc[(f, 3), h] and m.loc[(f, 3), h] > c
            for f, c in cost.items()
        )
    except KeyError:
        return False


def crowd(ev: pd.DataFrame) -> pd.Series:
    """How many of the pairs began an excursion into the tail in the hour up to each event, the
    event's own included: 1 is an extreme of its own, 8 is the market moving. Per family and tail,
    and known at the event bar's close. Not one of the card's states: it asks whether lone extremes
    revert, which would also part the mean per event from the mean of blocks. They do not; the gap
    is the blocks' look-ahead (`aggregate`)."""
    out = pd.Series(0, index=ev.index)
    for _, g in ev.groupby(["family", "share"]):
        w = pd.DatetimeIndex(g.when).asi8
        t = np.sort(w)
        out[g.index] = np.searchsorted(t, w, "right") - np.searchsorted(t, w - 3 * BAR.value, "left")
    return out


CROWDS = ("1", "2-3", "4-7", "8+")


def liquidity_table(ev: pd.DataFrame, family: str, tail: float = TAIL, symbols=SYMBOLS):
    """`(by tercile, every extreme, cuts)` for one family in the period, cut points from its own
    events at the main tail on development."""
    one = ev[(ev.family == family) & ev.symbol.isin(symbols)]
    cuts = tercile_cuts(one[(one.share == TAIL) & (one.dev_fold > 0)])
    here = one[(one.share == tail) & (one.fold > 0)]
    return (
        aggregate(terciles(here, cuts), ["state", "tercile"], LIQ_H),
        aggregate(here.assign(state="all"), ["state"], LIQ_H),
        cuts,
    )


# --- power ----------------------------------------------------------------------------------------


def power(se: pd.DataFrame, mean, criterion, effects, reps: int = 2000, seed: int = 0) -> list[float]:
    """The share of `reps` simulated studies in which `criterion` passes, for each effect.

    `se` holds the measured errors, one row per cell (fold and group) and one column per horizon.
    Each study draws every cell's mean as `mean(cell, horizon, effect)` plus a Gaussian error with
    the cell's measured error, cumulated over the horizons the way a path is, so a cell's 12- and
    48-bar means are correlated (as se_12 / se_48) and not drawn apart. Gaussian, because every cell
    holds tens to thousands of blocks and single events' tails are averaged away.
    """
    rng = np.random.default_rng(seed)
    var = se.to_numpy() ** 2
    step = np.sqrt(np.maximum(np.diff(np.c_[np.zeros(len(se)), var], axis=1), 0.0))
    out = []
    for e in effects:
        mu = np.array([[mean(cell, h, e) for h in se.columns] for cell in se.index])
        draws = mu + np.cumsum(rng.standard_normal((reps, *step.shape)) * step, axis=2)
        out.append(np.mean([criterion(pd.DataFrame(x, index=se.index, columns=se.columns)) for x in draws]))
    return out


# --- self-check ---------------------------------------------------------------------------------


def _selfcheck() -> None:
    """A planted reversal recovered, a null at zero, the states causal, the spread estimator right."""
    rng = np.random.default_rng(0)
    n = 40 * DAY
    when = pd.date_range("2025-06-01", periods=n, freq="15min", tz="UTC")
    # An event every 50 bars, after which the next 12 bars come back 30 bp against a +1 event.
    at = np.arange(DAY, n - 100, 50)
    r = rng.normal(0, 0.001, n)
    for t in at:
        r[t + 1 : t + 13] -= 0.003 / 12
    got = respond(np.cumsum(r), at, np.ones(len(at)), (12, 24))
    t = aggregate(pd.DataFrame({"when": when[at], "fold": 1, "r12": got[:, 0], "r24": got[:, 1]}), [], (12, 24))
    assert all(abs(t.loc[1, h] - 30) < 3 * t.loc[1, f"{h}_se"] for h in (12, 24)), t
    # On a random walk the same events are nothing.
    walk = np.cumsum(rng.normal(0, 0.003, n))
    t = aggregate(
        pd.DataFrame({"when": when[at], "fold": 1, "r12": respond(walk, at, np.ones(len(at)), (12,))[:, 0]}), [], (12,)
    )
    assert abs(t.loc[1, 12]) < 3 * t.loc[1, "12_se"], t
    # The entry is the event bar's close: the event bar's own move is not part of the response.
    assert np.allclose(respond(np.array([0.0, 1.0, 1.0]), np.array([1]), np.array([1.0]), (1,)), 0.0)
    assert np.isnan(respond(np.zeros(3), np.array([2]), np.array([1.0]), (1,))).all()

    # One event per excursion into the tail, on the bar it enters; a change of side is a new one.
    a, s = entries(np.array([0, 0.5, 0.6, 0.2, -0.7, -0.8, 0.9, 0.0]), 0.5)
    assert list(a) == [1, 4, 6] and list(s) == [1, -1, 1]
    # The crowd counts the hour up to an event and nothing after it: three pairs within 30 minutes,
    # then one alone two hours later, then one more family that does not count against the first.
    t0 = pd.Timestamp("2025-06-01", tz="UTC")
    times = [t0, t0 + BAR, t0 + 2 * BAR, t0 + 8 * BAR, t0]
    k = crowd(pd.DataFrame({"when": times, "family": ["a"] * 4 + ["b"], "share": TAIL}))
    assert list(k) == [1, 2, 3, 1, 1], list(k)

    # The states on candles of 30 trades each at a 20 bp bid-ask bounce around a random walk: every
    # column of a bar is the same whether the series stops there or goes on, and the spread comes back.
    half = 0.001
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0004, n * 30))).reshape(n, 30)
    ticks = mid * (1 + half * rng.choice([-1.0, 1.0], mid.shape))
    qv = rng.lognormal(10, 0.5, n)
    bars = pd.DataFrame(
        {"high": ticks.max(axis=1), "low": ticks.min(axis=1), "close": ticks[:, -1], "quote_volume": qv}, index=when
    )

    def market(b: pd.DataFrame) -> pd.Series:
        return stress(pd.DataFrame({"a": np.log(b.close).diff()}))

    full = states(bars, market(bars))
    cut = 30 * DAY + 7
    early = states(bars.iloc[: cut + 1], market(bars.iloc[: cut + 1]))
    assert np.allclose(early.iloc[-1].to_numpy(), full.iloc[cut].to_numpy(), equal_nan=True), "a state read the future"
    assert full.iloc[-DAY:][["amihud", "vol", "thin_leg", "stress", "ar"]].notna().all().all()
    spread = 1e4 * np.sqrt(4 * full.ar.mean())
    assert abs(spread - 2e4 * half) < 4, spread
    # New York's session follows its daylight saving: 13:30 UTC opens it in July, 14:30 in December.
    s = session(
        pd.DatetimeIndex(["2025-07-01 13:30", "2025-07-01 13:15", "2025-12-01 14:30", "2025-12-06 15:00"], tz="UTC")
    )
    assert list(s) == [0, 1, 0, 2], s
    # Terciles: cut points from one set of events, applied unchanged to another.
    cuts = tercile_cuts(pd.DataFrame({v: np.arange(9.0) for v in STATES}))
    other = terciles(pd.DataFrame({v: [0.0, 4.0, 100.0] for v in STATES} | {"session": [0.0, 1.0, 2.0]}), cuts)
    assert all(other[other.state == v].tercile.tolist() == [1, 2, 3] for v in STATES)

    # The criteria and their power: a large effect against small errors passes nearly always, none never.
    folds = pd.Series(20.0, index=[1, 2])
    se = pd.DataFrame(1.0, index=pd.MultiIndex.from_product([folds.index, [1, 2, 3]]), columns=[48])
    rates = power(se, lambda cell, h, e: e * (cell[1] - 1) / 2, lambda m: liquidity_verdict(m, folds, 48), [0, 40], 200)
    assert rates[0] == 0 and rates[1] > 0.95, rates


# --- the runs -------------------------------------------------------------------------------------


def _show(title: str, table) -> None:
    print(f"\n{title}\n")
    print(table.to_string())


def liquidity(period: str, seeds) -> None:
    """Route 3, printed: the extremes by state, by asset, the tail's robustness, the null, the verdict."""
    ev, cut = liquidity_events(period, seeds)
    print(f"route 3, the liquidity premium at the extremes; period {period}")
    for k, c in cut.items():
        print(f"{k}: tails cut on development at |x| >= " + ", ".join(f"{v:.3f} ({q:.0%})" for q, v in c.items()))
    print("bp against the leg (positive = reversal), mean per event (error clustered on h-bar blocks)")
    print("tercile 1 is the liquid third of a state, 3 the illiquid one")
    for family in ("v2", "rsi 12", "rsi 12, random signs"):
        if family not in set(ev.family):
            continue
        table, every, cuts = liquidity_table(ev, family)
        both = pd.concat({"per event": cells(every, LIQ_H), "blocks (look-ahead)": cells(every, LIQ_H, True)})
        _show(
            f"{family}: every extreme, per event (clustered error) and as a mean of blocks (reads the future)",
            both.droplevel(1),
        )
        _show(f"{family}: by tercile of each state, the tercile's Abdi-Ranaldo spread in bp", cells(table, LIQ_H))
        if family == "rsi 12, random signs":
            continue
        print(
            "\ncut points, fixed on development's events: "
            + "; ".join(f"{v} {np.round(c, 3).tolist()}" for v, c in cuts.items() if c is not None)
        )
        one = ev[(ev.family == family) & (ev.share == TAIL) & (ev.fold > 0)]
        k = np.array(CROWDS)[np.digitize(crowd(one), [2, 4, 8])]
        _show(
            f"{family}: by how many pairs entered their tail in the hour up to the event",
            cells(aggregate(one.assign(crowd=k), ["crowd"], LIQ_H), LIQ_H),
        )
        t = terciles(one, cuts)
        a = (
            aggregate(t[t.tercile.isin([1, 3])].assign(fold=0), ["symbol", "state", "tercile"], (48,))[48]
            .droplevel("fold")
            .unstack("tercile")
        )
        by_asset = (a[1].map("{:+.0f}".format) + " / " + a[3].map("{:+.0f}".format)).unstack("state")
        _show(
            f"{family}: by asset, 48 bars, every fold of the period: liquid / illiquid tercile",
            by_asset.assign(tradable=by_asset.index.isin(TRADABLE)),
        )
        rob = [
            liquidity_table(ev, family, tail)[0][48].unstack("fold").rename(columns=lambda f, q=tail: f"{q:.0%} f{f}")
            for tail in TAILS
        ]
        _show(f"{family}: the tail at 5 / 10 / 15%, 48 bars, by fold", pd.concat(rob, axis=1).round(1))
        money = liquidity_table(ev, family, symbols=TRADABLE)[0]
        print(
            f"\n{family}, verdict on TRADABLE: in every fold the illiquid tercile reverts, the terciles are in order,"
        )
        print(f"and the illiquid one clears {SPOT_RT:.0f} bp plus its spread. Terciles 1 / 2 / 3 against the cost:")
        passed, tested = 0, 0
        for v in STATES:
            if v not in money.index.get_level_values("state"):
                continue
            m = money.loc[v].swaplevel().sort_index()
            cost = SPOT_RT + m.xs(3, level="tercile").spread_bp
            for h in LIQ_H:
                ok = liquidity_verdict(m, cost, h)
                passed, tested = passed + ok, tested + 1
                line = "  ".join(
                    f"f{f} {m.loc[(f, 1), h]:+.1f} / {m.loc[(f, 2), h]:+.1f} / {m.loc[(f, 3), h]:+.1f} vs {c:.1f}"
                    for f, c in cost.items()
                )
                print(f"  {v:10s} {h:3d} bars  {'PASS' if ok else 'fail'}  {line}")
        print(f"  {family}: {passed} of {tested} (state, horizon) pass")


def power_tables(reps: int = 2000) -> None:
    """Each card's criterion simulated at development's measured errors: the detection rate at 0,
    half, 1x and 2x the effect it needs; the 0 column is the false-positive rate."""
    ev, _ = liquidity_events("dev", ())
    table = liquidity_table(ev, "v2", symbols=TRADABLE)[0]
    del ev
    rows = {}
    for v in STATES:
        m = table.loc[v].swaplevel().sort_index()
        cost = SPOT_RT + m.xs(3, level="tercile").spread_bp
        need, se = float(cost.max()), m[[f"{h}_se" for h in LIQ_H]].set_axis(list(LIQ_H), axis=1)
        for h in LIQ_H:
            rows[(v, h, round(need, 1))] = power(
                se,
                lambda cell, _, e: e * (cell[1] - 1) / 2,
                lambda x, h=h, c=cost: liquidity_verdict(x, c, h),
                [0, need / 2, need, 2 * need],
                reps,
            )
    title = (
        "route 3, v2 at 10% on TRADABLE: the illiquid tercile at 0 / 0.5 / 1 / 2 x the cost, the liquid at 0, linear"
    )
    _show(title + " (state, bars, cost bp)", pd.DataFrame(rows, index=EFFECTS).T.round(3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--liquidity", action="store_true", help="route 3: v2's extremes by liquidity state")
    ap.add_argument("--power", action="store_true", help="each card's criterion simulated at development's errors")
    ap.add_argument("--period", choices=strategy.PERIODS, default="dev")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="the sign-randomised paths of the null")
    args = ap.parse_args()
    _selfcheck()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    if args.liquidity:
        liquidity(args.period, args.seeds)
    if args.power:
        power_tables()


if __name__ == "__main__":
    main()
