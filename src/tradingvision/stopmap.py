"""Route 8 of `false_alarms.html`: confirmed pivots as a map of stop orders. Nothing pays.

Osler (2003, 2005) saw it in a bank's real orders: stop-losses cluster just beyond recent extremes
and round numbers, and when the price reaches them they become market orders in the same direction,
so the move accelerates (a cascade); take-profits cluster on round numbers and turn the price. In
crypto the liquidations of leveraged positions play the stops' part. Both are still functions of the
price, so the control is the whole study: the same event at levels where no stop sits.

**The levels are the ones a live reader held.** On 15m closes, `legs.confirmed` at 12 bars (v2's) and
24 (`EXTREMA_WINDOW`), never `find_pivots`: at each bar the last confirmed low and high, each from its
`known` bar until the bar its successor (a more extreme twin, or the next pivot of its kind) is known.
The level is the swing's extreme as a chart shows it, the lowest low (highest high) of the 15m bars
from `window` before the pivot to its confirmation: pivots are found on closes, stops sit past the
wick. A break is the first 5m bar after the level is known whose low (high) trades through it, one per
level, so it is a first passage and a stopping time.

**Two signatures.** The cascade is the return in the break's direction over 1, 2 and 4 15m bars from
the break's 5m close. The failed break is a 15m close back on the level's original side within k = 1,
2 or 4 closes (1 is the break's own bar), then the return in the reversal's direction over 12, 24 and
48 bars from that close. Each is read at the pivot and at two placebos. Four levels shifted from the
pivot by -1, -0.5, +0.5 and +1 sigma (positive past it), sigma the 15m return's deviation over the
4 * window bars to the confirmation times sqrt(window), pooled as `shifted`. And a round number, the
first multiple of a 1-2-2.5-5 step past the price at the confirmation, the step a share of the price
fixed per asset on development so its breaks are as many as the pivots' (`fit_step`): 1.3% (BTC) to
3.7% (FIL, NEAR, UNI) at 12, 1.9% to 5.3% at 24, round thousands on BTC and hundreds on ETH. A grid
that breaks as often as the pivots has to be that coarse, because at its confirmation a pivot already
sits a retracement away from the price. A placebo the price has already passed is not a level.

**The null is a price no order can sit on** (`mirror`, `strategy.signflip` on whole 5m bars: open,
high, low and close as offsets from the last close, negated with high and low swapped). Every
stopping time grosses zero on it, so pivot and placebo are both zero there.

**The mean is per trade, not per block** (`per_trade`). `metrics.blocked` on the events averages each
clock block first, and breaks crowd into the blocks where a move goes on, so it under-weighs the
continuations: on the null it put the pivot's 4-bar cascade at -2.0 to -8.0 bp on fold 1 of all five
seeds. The mean here is the sum over the count, every trade alike, and its error the ratio estimator's,
from each block's residual S_b - mean * n_b. Every break is used, never one phase of the clock.
Pivot less placebo has the two errors added in quadrature, on the safe side since they share moves.

**Development, TRADABLE (the card's criterion is read here):** bp a trade, fold 1 / fold 2.

    window 12        pivot              less shifted       less round
    cascade 1 bar    -1.8 / -0.4        -1.4 / +0.1        -1.1 / -1.2      errors 1.3-3.0
    cascade 4 bars   +0.5 / +7.2        +0.9 / +1.3        -0.0 / +4.1      errors 2.8-6.6
    failed k=1, 24   -5.5 / -13.7       -3.9 / -1.9        -3.7 / -17.1     errors 6.9-14.6
    failed k=1, 48   -5.4 / -5.4        -0.7 / -2.9        -3.7 / -8.8      errors 8.3-16.9
    window 24
    cascade 1 bar    -0.0 / +1.2        -0.0 / -0.3        -1.7 / -0.4      errors 1.9-5.1
    cascade 4 bars   +4.2 / +15.4       +3.1 / +1.4        +1.2 / +8.6      errors 3.8-11.5
    failed k=1, 24   -7.9 / -30.3       -1.3 / -16.9       -10.7 / -28.4    errors 9.2-23.9
    failed k=1, 48   -7.9 / -17.6       +2.9 / -25.6       +0.5 / -38.8     errors 11.4-25.6

Each fold holds about 3,500 breaks of a pivot at 12 bars on the thirteen tradable pairs and 1,800 at
24, and 790 to 2,630 failed breaks. There is no cascade: at every horizon and both windows the pivot
is within 3.1 bp of either placebo in fold 1 and 8.6 in fold 2, against errors of 1.7-11.5. A failed
break does not revert, it goes on: the pivot's own "reversal", averaged over the two folds, is
negative in every development row (-3.1 to -19.1 bp), the placebos' less so, and the difference keeps
its sign in a few rows (the largest, window 24, k = 1, 24 bars: -9.1 against the shifted, -19.5
against round numbers) and reaches 20 bp in none. 0 of 24 rows pass. On the five null paths the
cascade's difference at 1 bar in fold 1 is -0.6 to +0.5 bp, and the failed break's at 48 bars spans
-24 to +24, the noise those rows carry.

**The hold-out (folds 3-4) and 2021-2025 (four slices) say the same.** Hold-out: cascade differences
-1.0 to +1.2 bp over the two folds, failed breaks -7.6 to +5.0 against the shifted and -15.6 to +2.9
against round numbers, 0 of 24. 2021: 0 of 24, with the one trace of stops the study found. In its
first slice, 2021-01 to 2022-02, a broken pivot's next 15 minutes ran on +6.6 bp at 12 bars and +8.8
at 24 (errors 1.5 / 2.1) while the placebos sat at -7.6 to +1.5 (all fifteen pairs), so the difference
against the shifted was +8.0 / +11.1 at 1 bar and +9.4 / +16.3 at 4 (2.8 to 4.4 errors). From 2022 it
is +1.4 to +3.4 bp at 24 bars and -0.2 to +1.4 at 12: at 24, 1 and 2 bars it has one sign in all four
slices against both placebos, at 3-5 bp on the period, a quarter of the round trip. The cascade was
real in the leverage of 2021 and is all but gone. The failed break's sign changes from slice to slice
(-17.5 to +7.9 bp a trade at the pivot).

**Power** (`--power`, block bootstrap of each fold's pivot, shifted and round blocks at the measured
sizes, centred, an effect added to the pivot): on development a row fires with no effect 0-3.4% of the
time (0.19 of the 24 rows expected to, 0.10 on the hold-out, 0.00 on 2021's four folds), at half the
round trip 0-11%, at one round trip 32-38% (the threshold itself, so half the draws fall short and
both placebos have to clear it), at two 86-100% on development (the 48-bar failed breaks at 24 the
lowest) and 97-100% on 2021. A 40 bp edge at a pivot would not have been missed; the 2021 cascade, at
a quarter of that, is visible in the tables and out of the criterion's reach by construction.

**Verdict, the card's own branch:** at 15 minutes, on liquid pairs, stops leave no trace a 20 bp
round trip can collect; the route is closed. The liquidation map from open interest (the card's
exploratory variant) was not tried: it needs the futures dumps, which cover sixteen months of three
assets, and the plain map already reads zero on that span.

    uv run python -m tradingvision.stopmap [--window 12 24] [--period dev|holdout|2021] [--seeds 0 1 2]
    uv run python -m tradingvision.stopmap --period dev --power
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision import legs, strategy
from tradingvision.data.binance import SYMBOLS, TRADABLE
from tradingvision.data.binance import load as candles
from tradingvision.metrics import blocked
from tradingvision.oracle import FEE

WINDOWS = (12, 24)  # v2's pivots and EXTREMA_WINDOW
BAR, FIVE = pd.Timedelta("15min"), pd.Timedelta("5min")
CASCADE = (1, 2, 4)  # 15m bars after the break's 5m close
FAIL = (1, 2, 4)  # 15m closes within which a failed break is back on the level's original side
REVERT = (12, 24, 48)  # 15m bars after that close
# The placebos, in units of sigma, positive past the level (below a low, above a high), negative between it
# and the price. Sigma is the noise over the leg's horizon: the 15m log return's standard deviation over the
# 4 * window bars up to the level's confirmation (the estimator `legs.state` reads) times sqrt(window).
SHIFTS = (-1.0, -0.5, 0.5, 1.0)
LEVELS = ("pivot", "-1 sd", "-0.5 sd", "+0.5 sd", "+1 sd", "round")
# Round-number grid steps a fit on development chooses from, as a share of the price: 0.28% to 7.4%.
STEPS = 2.0 ** np.arange(-8.5, -3.5, 0.25)
NICE = np.array([1.0, 2.0, 2.5, 5.0, 10.0])
ROUND_TRIP_BP = 2 * FEE * 1e4
WARMUP = pd.Timedelta("60D")


def fifteen(five: pd.DataFrame) -> pd.DataFrame:
    """15m bars from 5m ones, `binance.load`'s resampling on the four prices."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return five.resample("15min").agg(agg).dropna(subset=["open"])


def mirror(five: pd.DataFrame, seed: int) -> pd.DataFrame:
    """5m bars with each bar's move from the last close given a random sign, `strategy.signflip` on bars.

    The break reads a bar's low and high, which a rebuilt close does not have, so the whole bar is
    flipped: its open, high, low and close as log offsets from the previous close, negated, with the
    high and the low swapping places. Sizes, ranges, wicks, volatility clusters and tails stay; any
    direction goes, and with it any order sitting at a level.
    """
    lp = np.log(five[["open", "high", "low", "close"]].to_numpy(dtype="float64"))
    off = lp - np.r_[lp[0, 0], lp[:-1, 3]][:, None]
    s = np.random.default_rng(seed).choice([-1.0, 1.0], len(lp))
    off = np.where(s[:, None] > 0, off, -off[:, [0, 2, 1, 3]])
    close = lp[0, 0] + np.cumsum(off[:, 3])
    out = np.exp(np.r_[lp[0, 0], close[:-1]][:, None] + off)
    return pd.DataFrame(out, index=five.index, columns=["open", "high", "low", "close"])


def held(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    """Every level a live reader held: the last confirmed low and the last confirmed high, one row each.

    From `legs.confirmed`, never `find_pivots`. At each timeline row the reader holds a low and a high,
    the row's pivot and the one before it; a level is born on the row that makes it the held one, at
    its `known` bar `start`, and dies on the row that replaces it (`end`, the same side's next start),
    which is either a twin more extreme than it or the next pivot of its kind. The level is the swing's
    extreme on the chart, the lowest low (highest high) of the 15m bars from `window` before the pivot to
    its confirmation: pivots are found on closes, and stops sit past what a chart shows, the wick. Nothing
    in those bars traded through it, so the first trade through it after `start` is a first passage.
    """
    line = legs.confirmed(bars.close, window)
    n = len(bars)
    sigma = np.log(bars.close).diff().rolling(4 * window).std().to_numpy() * np.sqrt(window)
    lo = bars.low.rolling(2 * window + 1, min_periods=1).min().to_numpy()
    hi = bars.high.rolling(2 * window + 1, min_periods=1).max().to_numpy()
    parts = []
    for side in (-1, 1):
        h = pd.Series(np.where(line.kind.to_numpy() == side, line.bar.to_numpy(), line.prev_bar.to_numpy()))
        born = np.flatnonzero((h.ne(h.shift()) & h.notna()).to_numpy())
        start = line.known.to_numpy()[born].astype(int)
        parts.append(
            pd.DataFrame(
                {
                    "side": side,
                    "bar": h.to_numpy()[born].astype(int),
                    "start": start,
                    "end": np.r_[start[1:], n],
                    "level": (hi if side > 0 else lo)[start],
                    "sigma": sigma[start],
                    "close": bars.close.to_numpy()[start],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def nice(x: np.ndarray) -> np.ndarray:
    """The 1-2-2.5-5 number nearest `x` in log: what a round number's step looks like at any price."""
    e = 10.0 ** np.floor(np.log10(x))
    c = NICE[None, :] * e[:, None]
    return c[np.arange(len(x)), np.abs(np.log(c / x[:, None])).argmin(axis=1)]


def round_level(lv: pd.DataFrame, step: float) -> np.ndarray:
    """The first round number past the price at each level's start, on a grid of `step` times the price.

    A low's is the largest multiple below the close, a high's the smallest above it, so a round level
    lives through the same window as the pivot it stands in for and is broken the first time the price
    moves that far. The step is a nice number, so the grid is round at every price and keeps its
    relative spacing within a factor of about two across years and assets.
    """
    c = lv.close.to_numpy()
    s = nice(step * c)
    return np.where(lv.side > 0, (np.floor(c / s) + 1) * s, (np.ceil(c / s) - 1) * s)


def prices(lv: pd.DataFrame, step: float) -> np.ndarray:
    """`(levels, LEVELS)`: the pivot, the four placebos and the round number, NaN where not past the close.

    A level the price has already passed at the start has no first passage to read; the pivot always
    qualifies, a placebo between it and the price may not.
    """
    side, w, s = lv.side.to_numpy(), lv.level.to_numpy(), lv.sigma.to_numpy()
    p = np.column_stack([w] + [w * np.exp(side * k * s) for k in SHIFTS] + [round_level(lv, step)])
    return np.where(side[:, None] * (p - lv.close.to_numpy()[:, None]) > 0, p, np.nan)


def first_breaks(five: pd.DataFrame, bars: pd.DataFrame, lv: pd.DataFrame, price: np.ndarray) -> np.ndarray:
    """The 5m bar that first trades through each price while its level is held, -1 if none.

    A level held from the close of 15m bar `start` is tested from the first 5m bar after that close
    to the close of bar `end`, when its successor is known. One break per level, on the intrabar low
    (high): the running extreme of the window is monotone, so a search finds the first bar past
    every column at once.
    """
    edge = np.r_[(bars.index + BAR).asi8, np.iinfo(np.int64).max]
    s = np.searchsorted(five.index.asi8, edge[lv.start.to_numpy()])
    e = np.searchsorted(five.index.asi8, edge[lv.end.to_numpy()])
    up, down, side = five.high.to_numpy(), -five.low.to_numpy(), lv.side.to_numpy()
    out = np.full(price.shape, -1, dtype=np.int64)
    for i in range(len(lv)):
        run = np.maximum.accumulate((up if side[i] > 0 else down)[s[i] : e[i]])
        j = np.searchsorted(run, side[i] * price[i], side="right")
        out[i] = np.where(j < len(run), s[i] + j, -1)
    return out


def fit_step(five: pd.DataFrame, bars: pd.DataFrame, lv: pd.DataFrame, start, end) -> float:
    """The round grid whose breaks are about as many as the pivots', on the levels born in [start, end)."""
    born = bars.index[lv.start]
    sub = lv[(born >= start) & (born < end)]
    p = np.column_stack([prices(sub, STEPS[0])[:, 0]] + [prices(sub, r)[:, -1] for r in STEPS])
    count = (first_breaks(five, bars, sub, p) >= 0).sum(axis=0) + 1
    return float(STEPS[np.abs(np.log(count[1:] / count[0])).argmin()])


def events(five: pd.DataFrame, window: int, step: float | None = None, dev=None) -> tuple[pd.DataFrame, float]:
    """`(events, step)`: every break of every level, with both signatures, on one asset's 5m bars.

    One row per broken level. `side` is the break's direction (-1 a low broken down), `t_break` the
    break's 5m close, `c1`..`c4` the log return in the break's direction from it after 1, 2, 4 15m
    bars. `back` is how many 15m closes after the break's own (0 = that one) the first close back on
    the level's original side came, NaN if not within `FAIL[-1]`; `t_fail` is that close, and
    `r12`..`r48` the return in the reversal's direction from it. `step` is the round grid, fitted on
    the levels born in `dev` (a `(start, end)` pair) when not given.
    """
    bars = fifteen(five)
    lv = held(bars, window)
    if step is None:
        step = fit_step(five, bars, lv, *dev)
    price = prices(lv, step)
    t = first_breaks(five, bars, lv, price)
    i5, i15 = five.index, bars.index
    lc5, lc15, c15 = np.log(five.close.to_numpy()), np.log(bars.close.to_numpy()), bars.close.to_numpy()
    parts = []
    for v, name in enumerate(LEVELS):
        hit = t[:, v] >= 0
        i, d, p = t[hit, v], lv.side.to_numpy()[hit], price[hit, v]
        when = i5[i]
        out = {"side": d.astype("int8"), "level": name, "t_break": when + FIVE}
        for h in CASCADE:
            x = i5.searchsorted(when + h * BAR, side="right") - 1
            out[f"c{h}"] = np.where(when + h * BAR <= i5[-1], d * (lc5[x] - lc5[i]), np.nan)
        b = i15.searchsorted(when, side="right") - 1
        back = np.full(len(i), np.nan)
        for j in range(max(FAIL) - 1, -1, -1):  # the earliest close back wins
            k = np.minimum(b + j, len(c15) - 1)
            back = np.where((b + j < len(c15)) & (d * (c15[k] - p) < 0), j, back)
        f = b + np.nan_to_num(back).astype(int)
        out["back"] = back
        out["t_fail"] = (i15[f] + BAR).where(~np.isnan(back))
        for h in REVERT:
            x = i15.searchsorted(i15[f] + h * BAR, side="right") - 1
            ok = ~np.isnan(back) & (i15[f] + h * BAR <= i15[-1])
            out[f"r{h}"] = np.where(ok, -d * (lc15[x] - lc15[f]), np.nan)
        parts.append(pd.DataFrame(out))
    ev = pd.concat(parts, ignore_index=True)
    rets = [f"c{h}" for h in CASCADE] + [f"r{h}" for h in REVERT] + ["back"]
    return ev.astype({c: "float32" for c in rets}), step


def per_trade(x: pd.Series, horizon: pd.Timedelta) -> dict:
    """The mean of `x` per event, sum over count, with the ratio estimator's error over clock blocks.

    `blocked` on the events themselves averages each block first, so a break sharing its block with nine
    others weighs a tenth of one alone. Breaks crowd where a move goes on, so that mean under-weighs the
    continuations: on sign-randomised paths, where every stopping time grosses zero, it put the pivot's
    4-bar cascade at -2.0 to -8.0 bp on fold 1 of all five seeds. Here every trade weighs alike, which is
    what a trade earns, and the error is `blocked`'s on each block's residual from that mean, S_b - mean
    * n_b, over the mean count: its mean is zero, so the mean returned is the per-trade one.
    """
    g = x.groupby(x.index.floor(horizon))
    total, n = g.sum(), g.size()
    mean = total.sum() / n.sum()
    return blocked((total - mean * n) / n.mean() + mean, horizon)


# (signature, k, h, entry column, return column): the observations every table reads.
SPECS = [("cascade", 0, h, "t_break", f"c{h}") for h in CASCADE] + [
    ("failed", k, h, "t_fail", f"r{h}") for k in FAIL for h in REVERT
]


def stats(ev: pd.DataFrame, period: str, path=strategy.PRED) -> pd.DataFrame:
    """bp a trade, its error over blocks and counts per (universe, path, window, signature, k, h, level, fold).

    `per_trade` on the events' entries, blocked on the clock at the horizon: breaks of fifteen assets in
    the same quarter hour are one market move, and overlapping forward returns share it. `shifted` pools
    the four sigma placebos.
    """
    rows = []
    for sig, k, h, when, col in SPECS:
        e = ev[ev[col].notna() & ((ev.back < k) if sig == "failed" else True)]
        e = e.assign(fold=strategy.fold_in(pd.DatetimeIndex(e[when]), period, path))
        e = e[e.fold > 0]
        for universe, part in (("all", e), ("tradable", e[e.symbol.isin(TRADABLE)])):
            shifted = part[part.level.isin(LEVELS[1:5])].assign(level="shifted")
            for sub in (part, shifted):
                for (p, w, lvl, f), g in sub.groupby(["path", "window", "level", "fold"], observed=True):
                    b = per_trade(pd.Series(g[col].to_numpy("float64") * 1e4, index=pd.DatetimeIndex(g[when])), h * BAR)
                    rows.append(
                        {
                            "universe": universe,
                            "path": p,
                            "window": w,
                            "signature": sig,
                            "k": k,
                            "h": h,
                            "level": lvl,
                            "fold": f,
                            "bp": b["mean"],
                            "se": b["se"],
                            "n": len(g),
                        }
                    )
    return pd.DataFrame(rows).set_index(["universe", "path", "window", "signature", "k", "h", "level", "fold"])


def contrast(st: pd.DataFrame) -> pd.DataFrame:
    """Pivot against the pooled sigma placebos and against round numbers, per fold, with the errors summed
    in quadrature: the two are correlated through common moves, so this error is on the safe side."""
    bp, se = st.bp.unstack("level"), st.se.unstack("level")
    out = pd.DataFrame(
        {"pivot": bp["pivot"], "pivot_se": se["pivot"], "n": st.n.unstack("level")["pivot"]}
        | {"shifted": bp["shifted"], "round": bp["round"]}
    )
    for other in ("shifted", "round"):
        out[f"-{other}"] = bp["pivot"] - bp[other]
        out[f"-{other}_se"] = np.hypot(se["pivot"], se[other])
    return out


def show(c: pd.DataFrame, universe: str, path: str) -> pd.DataFrame:
    """`contrast` on one universe and path as "bp (error)" cells, folds side by side."""
    t = c.xs((universe, path), level=("universe", "path"))
    out = pd.DataFrame({"n": t.n.astype(int)})
    for col in ("pivot", "-shifted", "-round"):
        out[col] = t[col].map("{:+.1f}".format) + " (" + t[f"{col}_se"].map("{:.1f}".format) + ")"
    return out.unstack("fold").swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


def verdict(c: pd.DataFrame) -> pd.DataFrame:
    """The card's criterion on TRADABLE's real path: pivot minus placebo with one sign in every fold of the
    period and its mean over the folds past the round trip, against each placebo."""
    rows = []
    for key, g in c.xs(("tradable", "real"), level=("universe", "path")).groupby(["window", "signature", "k", "h"]):
        row = dict(zip(["window", "signature", "k", "h"], key)) | {"pivot_bp": g["pivot"].mean()}
        for other in ("shifted", "round"):
            d = g[f"-{other}"]
            row[f"-{other}_bp"] = d.mean()
            row[f"-{other}"] = bool((np.sign(d) == np.sign(d.iloc[0])).all() and abs(d.mean()) > ROUND_TRIP_BP)
        rows.append(row)
    t = pd.DataFrame(rows).set_index(["window", "signature", "k", "h"])
    return t.assign(passes=t["-shifted"] & t["-round"])


def power(ev: pd.DataFrame, period: str, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """How often `verdict` fires on one row, by block bootstrap at the store's sizes.

    For each signature, each fold's pivot, pooled-placebo and round-number blocks (the ones `per_trade`
    reads, on TRADABLE's real path, as a sum and a count) are centred, so nothing is there, and redrawn
    with replacement as many as there are; an effect of 0, half, one and two round trips is added to the
    pivot. A draw fires when the
    pivot clears both placebos as `verdict` asks. The rate at 0 is the criterion's false positives on one
    row, the rest its power. Blocks keep the overlap and the common moves.
    """
    rng = np.random.default_rng(seed)
    effects = (0.0, 0.5, 1.0, 2.0)
    rows = []
    real = ev[(ev.path == "real") & ev.symbol.isin(TRADABLE)]
    for sig, k, h, when, col in SPECS:
        e = real[real[col].notna() & ((real.back < k) if sig == "failed" else True)]
        e = e.assign(fold=strategy.fold_in(pd.DatetimeIndex(e[when]), period))
        for w, g in e[e.fold > 0].groupby("window", observed=True):
            shifted, rounded = [], []
            for f in sorted(g.fold.unique()):
                means = []
                for pick in (g.level == "pivot", g.level.isin(LEVELS[1:5]), g.level == "round"):
                    x = g[pick & (g.fold == f)]
                    s = pd.Series(x[col].to_numpy("float64") * 1e4, index=pd.DatetimeIndex(x[when]))
                    b = s.groupby(s.index.floor(h * BAR))
                    total, n = b.sum().to_numpy(), b.size().to_numpy()
                    total = total - n * total.sum() / n.sum()  # nothing there
                    draws = []
                    for _ in range(-(-reps // 250)):
                        i = rng.integers(0, len(n), (250, len(n)))
                        draws.append(total[i].sum(axis=1) / n[i].sum(axis=1))
                    means.append(np.concatenate(draws)[:reps])
                shifted.append(means[0] - means[1])
                rounded.append(means[0] - means[2])
            row = {"window": w, "signature": sig, "k": k, "h": h, "folds": len(shifted)}
            for m in effects:
                fires = np.ones(reps, dtype=bool)
                for d in (np.column_stack(shifted), np.column_stack(rounded)):
                    x = d + m * ROUND_TRIP_BP
                    fires &= (np.sign(x) == np.sign(x[:, :1])).all(axis=1) & (np.abs(x.mean(axis=1)) > ROUND_TRIP_BP)
                row[f"{m:g}x"] = float(fires.mean())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["window", "signature", "k", "h"]).sort_index()


def run(symbols, windows, period: str, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(events, steps)` on every symbol, real path and sign-randomised ones, one symbol in memory at a time.

    Loaded from the earliest of the period and development less `WARMUP` (development is where the round
    grid is fitted, whatever the period) to the store's end. Events are kept where either entry is in
    the period.
    """
    e = strategy.edges(period)
    dev = strategy.edges("dev")
    since = min(e[0], dev[0]) - WARMUP
    frames, steps = [], []
    for sym in symbols:
        before = len(frames)
        five = candles(sym, "5m").loc[since:, ["open", "high", "low", "close"]].astype("float64")
        for w in windows:
            ev, step = events(five, w, dev=(dev[0], dev[-1]))
            paths = [("real", ev)] + [(f"signs #{s}", events(mirror(five, s), w, step)[0]) for s in seeds]
            for name, x in paths:
                inside = (strategy.fold_in(pd.DatetimeIndex(x.t_break), period) > 0) | (
                    strategy.fold_in(pd.DatetimeIndex(x.t_fail), period) > 0
                )
                frames.append(x[inside].assign(symbol=sym, window=w, path=name))
            steps.append({"symbol": sym, "window": w, "step_pct": 100 * step})
        print(f"{sym}: {sum(len(f) for f in frames[before:]):,} events", flush=True)
    ev = pd.concat(frames, ignore_index=True)
    return ev.astype({"symbol": "category", "level": "category", "path": "category"}), pd.DataFrame(steps)


def _selfcheck() -> None:
    """Levels a live reader held, a planted cascade and a planted reversal recovered, and a null at zero."""
    # The live reader's levels, on `legs`' own case: two highs in a row, the second higher. The first is
    # held from its confirmation (bar 11) until the twin's (bar 20), and broken by the twin itself at bar
    # 17 — a break `find_pivots`, which has already merged the first high away, would never see.
    y = [100.0] * 8 + [104.0] + [101.0] * 8 + [106.0] + [100.0] * 8 + [95.0] + [99.0] * 8
    when = pd.date_range("2025-01-01", periods=3 * len(y), freq="5min", tz="UTC")
    flat = pd.DataFrame({c: np.repeat(y, 3) for c in ("open", "high", "low", "close")}, index=when)
    lv = held(fifteen(flat), 3)
    assert lv[["side", "bar", "start", "end"]].values.tolist() == [[-1, 26, 29, 35], [1, 8, 11, 20], [1, 17, 20, 35]]
    assert lv.level.tolist() == [95.0, 104.0, 106.0]
    ev, _ = events(flat, 3, step=0.05)
    first = ev[(ev.level == "pivot")]
    assert len(first) == 1 and first.t_break.iloc[0] == when[3 * 17] + FIVE, first
    # The twin closed above the level, the next bar closed back under it: a failed break at k = 2, not 1.
    assert first.back.iloc[0] == 1 and first.t_fail.iloc[0] == when[3 * 18] + BAR

    # Causality: cutting the 5m series cannot move a level or a break before the cut.
    rng = np.random.default_rng(3)
    n = 9000
    when = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[100.0, c[:-1]]
    wick = np.abs(rng.normal(0, 0.0005, (2, n)))
    walk = pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) * np.exp(wick[0]), "low": np.minimum(o, c) * np.exp(-wick[1]), "close": c},
        index=when,
    )
    full = events(walk, 12, step=0.01)[0]
    cut = when[6000]  # on a 15m boundary: a partial last bar is not a bar
    part = events(walk.iloc[:6000], 12, step=0.01)[0]
    key = ["level", "side", "t_break"]
    a, b = full[full.t_break <= cut][key], part[part.t_break <= cut][key]
    assert len(a) > 50 and a.sort_values(key).values.tolist() == b.sort_values(key).values.tolist()
    lines = legs.confirmed(fifteen(walk).close, 12)
    lv = held(fifteen(walk), 12)
    assert set(zip(lv.bar, lv.start)) <= set(zip(lines.bar, lines.known)), "a level the reader never held"

    # Mirrored bars are bars, with the same moves in size.
    m = mirror(walk, 0)
    body = m[["open", "close"]]
    assert (m.high >= body.max(axis=1) - 1e-9).all() and (m.low <= body.min(axis=1) + 1e-9).all()
    assert np.allclose(np.sort(np.abs(np.diff(np.log(m.close)))), np.sort(np.abs(np.diff(np.log(walk.close)))))

    # The null: on a random walk the pivot's cascade is the placebos', within three errors.
    def diff(ev, col, k=None):
        e = ev[ev[col].notna() & ((ev.back < k) if k else True)]
        p, q = e[col][e.level == "pivot"], e[col][e.level.isin(LEVELS[1:5])]
        return p.mean() - q.mean(), np.hypot(p.sem(), q.sem())

    d, se = diff(full, "c1")
    assert abs(d) < 3 * se, (d, se)

    # Planted: every break of a pivot is followed by a jump of 0.5% in its direction, or every break that
    # closes straight back (k = 1) by one against it. Planted in time order, since a jump moves every
    # level after it: each pass plants the earliest break not yet planted and recomputes.
    def plant(five, failed: bool, size: float = 0.005):
        lp, done = np.log(five), five.index[0]
        while True:
            e = events(np.exp(lp), 12, step=0.01)[0]
            e = e[(e.level == "pivot") & ((e.back == 0) if failed else True)]
            at = e.t_fail if failed else e.t_break
            later = at[at > done]
            if later.empty:
                return np.exp(lp)
            j = later.idxmin()
            done = at[j]
            lp.loc[lp.index >= done] += (-1 if failed else 1) * e.side[j] * size

    cascade = events(plant(walk, False), 12, step=0.01)[0]
    d, se = diff(cascade, "c1")
    assert d > 0.0025 and d > 5 * se, (d, se)
    reversal = events(plant(walk, True), 12, step=0.01)[0]
    d, se = diff(reversal, "r12", k=1)
    assert d > 0.0025 and d > 3 * se, (d, se)
    print("ok — levels, breaks and null as a live reader would have them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, nargs="+", default=list(WINDOWS), help="legs.confirmed windows")
    ap.add_argument("--period", choices=strategy.PERIODS, default="dev")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="sign-randomised paths, the null")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--power", action="store_true", help="the criterion's detection rate, by block bootstrap")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    _selfcheck()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    ev, steps = run(args.symbols, args.window, args.period, args.seeds)
    print("\nround grid step fitted on development, % of price\n")
    print(steps.pivot(index="symbol", columns="window", values="step_pct").round(2).to_string())
    st = stats(ev, args.period)
    c = contrast(st)
    paths = [("all", "real"), ("tradable", "real")] + [("all", p) for p in sorted(set(ev.path)) if p != "real"]
    for universe, path in paths:
        print(f"\n{args.period}, {universe} symbols, {path} path: bp in the signature's direction (blocked error),")
        print("the pivot, and the pivot less the pooled sigma placebos and the round numbers; k = 0 is the cascade\n")
        print(show(c, universe, path).to_string())
    print("\nevery level, bp per fold, real path, all symbols\n")
    detail = st.xs(("all", "real"), level=("universe", "path")).bp.unstack("level")[list(LEVELS) + ["shifted"]]
    print(detail.round(1).to_string())
    v = verdict(c)
    print(f"\ncriterion on TRADABLE, real path: one sign in every {args.period} fold, |mean| > {ROUND_TRIP_BP:g} bp\n")
    print(v.round(1).to_string())
    print(f"\nverdict: {'PASS' if v.passes.any() else 'FAIL'} — {int(v.passes.sum())} of {len(v)} rows pass")
    if args.power:
        pw = power(ev, args.period, args.reps)
        print(f"\npower: share of {args.reps} block-bootstrap draws where the criterion fires on a row, effect in")
        print("round trips added to the pivot (0x is the false-positive rate)\n")
        print(pw.round(3).to_string())
        print(f"\nrows expected to pass with no effect anywhere: {pw['0x'].sum():.2f} of {len(pw)}")


if __name__ == "__main__":
    main()
