"""Recognising a turn of the v2 prediction while it happens: delay, false alarms, and what they earn.

`strategy --hindsight --delay` found that a centred turn of the prediction at 12 bars is worth
+180 bp a trade on the turn bar and +41 six bars later. This module asks how close a reader that
sees only the past can get, and what being close is worth. Same predictions, same three assets,
same folds as `strategy`: the detectors are fitted on development (folds 1-2), and the hold-out is
read with them as a check, since it has been used before.

**The problem is quickest change detection.** A detector is a stopping time T: its alarm uses
nothing after T. A centred turn τ is not one, it needs `window` bars of future. Delay is T - τ, a
false alarm is an alarm raised inside the wrong leg. With a false alarm every ARL bars, the least
delay any detector can have is about log(ARL) / I (Lorden; CUSUM attains it, Moustakides), where I
is the information a bar carries about the change. Inside the prediction's legs at 12 bars, on
development, the increments are +0.035 a bar going up and -0.036 coming down, sd 0.078: I = 0.42
nats a bar. On the price the same legs give +9.2 / -9.9 bp on 31 bp, I = 0.19. Both promise turns
recognised within a few bars.

**Two detectors.** `zigzag` alarms when the series has come back `h` from its extreme since the
last alarm: CUSUM with no drift allowance. `shiryaev` is the Bayesian one: the probability that
the leg has already turned, updated every bar from a prior hazard and the bar's evidence, with an
alarm above a threshold. The hazard is a logistic of the prediction's level and the leg's age, fitted
on development (`fit`), because the prediction is shrunk towards zero and rarely reaches its ends:
at its turns at 12 bars the level is 0.37 at the median, above 0.5 on 27% of them and above 0.75
on 2.4%. The hazard a bar is 4% at a level of 0-0.25, 7% at 0.25-0.4, 10% at 0.4-0.5, 14% at
0.5-0.6, 25% at 0.6-0.7, and under 1% in the first three bars of a leg. The evidence is the bar's
increment, Gaussian before the turn (+0.027, sd 0.077, from mid-leg bars) and after it (-0.049,
sd 0.076, from the first six bars of the next leg).

**Both detect what they promise and earn nothing.** On development against the prediction's own
turns at 12 bars:

    detector             found   mean delay   false alarms a turn   bp a trade dev / hold-out
    zigzag h=0.2          97%     5.1 bars          0.31                  -0.3 / -1.5
    zigzag h=0.3          87%     7.1               0.14                  -0.2 / -4.9
    shiryaev 0.5          79%     4.7 (median 3)    0.35                  -1.0 / +0.1
    shiryaev 0.9          65%     7.1               0.10                  +6.7 / -13.3
    shiryaev 0.99         43%    10.2               0.04                 +17.7 / -23.5
    shiryaev 0.9, flat    79%     8.0               0.10                  +0.2 / -5.6

The level prior trades found turns for delay and does not move the frontier. Every zigzag from h
= 0.05 to 0.8 makes between -1 and +9 bp a trade; every Shiryaev threshold makes about zero, or the
pattern of every rule in `strategy`, positive on folds 1-2 and negative on 3-4. An RSI at 12 through
the same detector does the same. Where the money goes (`--split`, Shiryaev at 0.5, development):
the 2,765 trades opened by a right detection make +30.9 bp each, the 1,234 opened by a false alarm
lose -79.8, held 20 bars against the leg, and the two cancel. At 0.9: +21.9 on 2,284 against -89.0
on 359; the zigzag at 0.2, +19.8 on 3,403 against -79.5 on 1,081.

**Combining the two moves the parts and not the sum** (`--both`): the alarm waits for Shiryaev's
posterior and for a retracement of h from the extreme. At 0.5, h = 0.2 cuts false alarms from 0.35
to 0.20 a turn and even finds more turns (86%); h = 0.4 cuts them to 0.07. Each step makes the right
detections later and worth less (+30.9, +22.6, +16.4 bp a trade) and the false alarms that survive
dearer (-79.8, -100.4, -138.2): one that outlasts a deeper retracement was entered after a deeper
move against the leg, so the leg's return to its course costs more. Over the twelve combinations of
0.5 / 0.7 / 0.9 and h = 0.1-0.4, development makes +0.6 to +6.5 bp a trade and the hold-out -0.8 to
-15.6. A filter on the same information is one more stopping time on it: telling a true turn from a
false one at the alarm takes information about the next leg, which neither detector has.

**Telling a true alarm from a false one is possible, and pays nothing** (`--features`, `at_alarm`,
`separate`). At Shiryaev 0.5, sixteen columns known at the alarm, each signed into the leg the
alarm closes: from the prediction, from the price, the six exhaustion columns at 12 bars, the BTC
filter. The retracement from the leg's extreme separates best (AUC 0.592 on development, 0.600 on
the hold-out), then the price's retracement in ATR (0.580 / 0.603), the leg's move in units of its
noise (0.579 / 0.599) and `stretch` (0.573 / 0.582). A logistic on all of them reaches 0.637 /
0.629, and the share of true alarms runs from 53% in its bottom quintile to 83% in its top on
development, 60% to 86% on the hold-out. No column correlates with the gross of the trade beyond
|0.06| in either period, and up the quintiles the true alarms make less as they become likelier
(+49 to +25 bp on development, +40 to +9 on the hold-out) while the false ones lose more (-68 to
-99, -65 to -102): every quintile makes between -6.6 and +6.5. What says a turn has happened is how
far it has gone, which is how much of it is already spent.

**A stop loss cuts the false alarms and the true ones alike** (`--sl`). At 1 ATR (at 12 bars) it
closes 96% of the false-alarm trades and halves their loss, -79.8 to -41.6 bp, and closes 51% of the
right detections, which fall from +31.0 to +17.4; at 2 ATR -68.3 and +27.0. The book makes +0.6 /
+1.1 bp a trade at 1 ATR and -0.5 / +1.3 at 2, and no stop from 1 to 6 ATR, fixed or trailing,
leaves -1.4 to +1.6. The trailing stop at 1 ATR is positive in all four folds, +0.6 to +2.0, on
trades of two to four bars: a thirtieth of a 50 bp round trip. Right after an alarm the price is as
noisy around a true turn as around a false one, so a level that closes the false ones closes the
true ones too.

**Funding, open interest, positioning, taker flow and the book add nothing at the alarm**
(`--futures`, `data.futures`). Fifteen futures columns at Shiryaev 0.5's alarms, signed into the
leg: on their own a logistic reaches AUC 0.546 on development and 0.525 on the hold-out, with the
sixteen columns above 0.639 / 0.624 against their 0.637 / 0.629 alone, and every quintile still
makes -11 to +6 bp a trade. Against the forward return, unconditionally, two keep their sign on all
four folds: the open interest's 12-bar change signed by the price's 12-bar move, against the 48-bar
return (+0.111 / +0.039 / +0.077 / +0.001 by fold: a move made with new positions keeps going), and
the book's imbalance within 5% over the last hour against the 12-bar return (+0.025 / +0.059 /
+0.006 / +0.042). Both are the size of v2's own IC (-0.031 / -0.039 at 12 bars) and neither is a
strategy: at an IC of 0.05 a trade on the 48-bar return is worth of the order of 10 bp, against a
20-50 bp round trip.

**Open interest behind the move, in depth** (`--oi`, `open_interest`). Signed by the direction of the
last k bars' move, the open interest's k-bar change (in units of its month's dispersion) has a
positive rank IC with the 48-bar forward return on every fold for k = 4, 12 and 24 (k = 24: +0.057 /
+0.125 / +0.057 / +0.045), and the move alone, momentum, has none (-0.048 / -0.010 / +0.016 /
-0.006). Split by sign: after a 24-bar move made with open interest rising, the next 48 bars go its
way by +10.4 / +16.4 / +18.7 / +19.8 bp; after one made with it falling, they come back by -22.0 /
-13.0 / -2.4 / -6.1. New positions behind a move keep it going, a move made by closing positions
reverses. It is the cleanest fold-by-fold sign of the study, and it was read among 64 variants
that included the hold-out: the one development alone picks, k = 4, falls from +0.117 to +0.024
there. As a rule it does not hold: following the move when open interest rose and fading it when
it fell, entering when |z| reaches 0.5 to 2 and out after 48 bars, makes -6.9 to +0.7 bp a trade
on development and +0.8 to +17.2 on the hold-out, with fold 2 negative in all twelve variants
(-6.4 to -41.5). The effect lives in the bulk of the moves, not in the extremes a trigger picks.

**Keeping only signals past a level** (`--gate`, `gate`). A long only where the prediction is at or
under -L, a short only at or over +L, read at the signal's bar or at the extreme of the leg it
closes; a dropped signal either closes the position or is ignored. It cuts the zigzag's 5,288
development trades to 93 at L = 0.5, and the gross a trade stays near zero: closing on a dropped
signal, -3.2 to +4.2 bp on development and -9.2 to +0.9 on the hold-out for both detectors and
every L from 0.1 to 0.5 but one; ignoring it, the always-in shape comes back, positive on folds 1-2 and down to -47 on
the hold-out. The one row positive on all four folds, the zigzag at L = 0.5 read at the signal and
closing, makes +12.5 +/- 14.7 and +19.2 +/- 16.6 on 160 trades, and Shiryaev's detector through the
same gate makes -3.2 / -3.5. Fewer trades pay fewer fees; they do not pay more each. Chosen on
development alone, for the page (`chart.LEVEL`): of h 0.1-0.5 and p 0.5-0.99 by L 0.4 / 0.5 / 0.6,
closing, the best gross a trade with folds 1 and 2 both positive is that zigzag row and Shiryaev 0.5
at L = 0.6, +15.8 on 81 trades and +28.4 on 40 on the hold-out with a 6 ATR stop, every fold
positive and every number under the 50 bp round trip. Ignoring a dropped signal holds whatever side
the last kept one took, for days, and the result is the trend of those days: Shiryaev 0.5 with the
stop makes -25 / +52 / -91 bp on development at L 0.50 / 0.55 / 0.60.

**Open interest as a confirmation does not confirm** (`--confirm`, `confirmation`). Each signal times
open interest behind the last k bars' move, positive where the column says the next bars go the
signal's way. The gross of the trade a signal opens does not rise with it: across its quintiles
the detectors stay between -9 and +7 bp. Keeping only confirmed signals (a rejected one closes)
moves development down and the hold-out up by a few bp, Shiryaev 0.5 at k = 12 and a confirmation
of 1 making -1.6 +/- 7.5 and +12.3 +/- 7.0, with fold 2 negative and fold 4 positive almost
everywhere: the pattern of the open-interest rule itself, the period and not the signal. The
column is too small and too unstable at these horizons to move a signal it is laid on.

**Why: optional stopping.** If the log price is a martingale, E[p_T - p_S | F_S] = 0 for any two
stopping times S <= T, so a trade a causal detector opens and closes has zero expected gross
whatever its delay and false alarms. The hindsight table does not contradict it because τ + d,
d < window, is not a stopping time; τ + window is, and there the turns are worth -10 to +16
(`strategy --causal`).

**And the turn's value is the geometry of noise** (`strategy --hindsight ... --delay ... --null`).
On prices rebuilt from their own returns with every sign drawn at random, volatility clusters and
tails kept and any direction gone, the turns of the price at 12 bars make +216 to +219 bp a trade
against +201 on the real one, an RSI's +169 to +172 against +164, and the share left at each delay
agrees with the real one to 0.01-0.03. The share follows 1 - sqrt(d / window): after an extreme a
path moves away as sigma sqrt(d), lost once at the entry and once at the exit (0.71, 0.59, 0.42, 0.29
at d = 1, 2, 4, 6 of 12 against 0.73, 0.59, 0.40, 0.27 measured), and the price around the
prediction's turns moves 27 bp in one bar, 56 in four and 82 in nine. The I above is inflated by
the same selection; the information that matters is the prediction's correlation with the forward
return, and at rho ~ 0.03 it is I ~ rho^2 / 2, of the order of 1e-4 to 1e-3, against which Lorden's
bound is thousands of bars. The legs v2 was trained on are 12 bars, so 12 is also the most a turn
can be late before the window confirms it: the whole question lives in those 12 bars, and in them
the price behaves as a coin would.

The price's own zigzag at 3-8% retracements is the one row with a positive number, +11 to +63 bp a
trade, all 16 months; against the same zigzag on 20 sign-randomised paths it is within 0.7-1.6
null standard deviations and changes sign between folds.

    uv run python -m tradingvision.detect --zigzag 0.05 0.1 0.15 0.2 0.3 0.5 0.8
    uv run python -m tradingvision.detect --shiryaev 0.5 0.7 0.8 0.9 0.95 0.98 0.99 [--flat]
    uv run python -m tradingvision.detect --shiryaev 0.5 0.9 --zigzag 0.2 --split
    uv run python -m tradingvision.detect --shiryaev 0.5 0.7 0.9 --both 0.1 0.2 0.3 0.4 --split
    uv run python -m tradingvision.detect --features 0.5
    uv run python -m tradingvision.detect --sl 1 2 3 4 6
    uv run python -m tradingvision.detect --futures   # needs `python -m tradingvision.data.futures` first
    uv run python -m tradingvision.detect --oi
    uv run python -m tradingvision.detect --gate 0.1 0.2 0.3 0.4 0.5
    uv run python -m tradingvision.detect --confirm
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from tradingvision import legs, strategy
from tradingvision.data import futures
from tradingvision.data.binance import load as candles
from tradingvision.strategy import fold_of, hold, plain, turn_events

POST = 6  # the first bars of a leg that make the post-turn increment distribution
# `fit` on v2's walk-forward predictions, development folds only (ETH, BTC, SOL, 2025-06-01 to
# 2026-01-28), in the head's raw units. Fixed here so the chart page, which has no store, runs the
# detector the study measured; `main` refits it and refuses to run if the two have drifted apart.
V2_FIT = {
    "beta": np.array([-3.3566, 3.2917, 0.9579, -0.106, -2.131]),
    "m0": 0.0265,
    "s0": 0.0773,
    "m1": -0.0489,
    "s1": 0.0758,
    "base": 0.0505,
}


def zigzag(v: np.ndarray, h: float, trace: bool = False) -> np.ndarray | pd.DataFrame:
    """+1 when `v` has risen `h` above its low since the last alarm, -1 when it fell `h` below its high.

    `trace` returns, bar by bar, the alarm, the leg the detector believed it was in before the bar
    (+1 up, -1 down, 0 not yet known) and how far `v` had come back from that leg's extreme.
    """
    out, leg, back = np.zeros(len(v)), np.zeros(len(v)), np.zeros(len(v))
    side, hi, lo = 0, v[0], v[0]
    for t in range(len(v)):
        hi, lo = max(hi, v[t]), min(lo, v[t])
        leg[t], back[t] = side, (hi - v[t]) if side > 0 else (v[t] - lo) if side < 0 else max(hi - v[t], v[t] - lo)
        if side >= 0 and hi - v[t] >= h:
            out[t], side, lo = -1, -1, v[t]
        elif side <= 0 and v[t] - lo >= h:
            out[t], side, hi = 1, 1, v[t]
    return pd.DataFrame({"alarm": out, "leg": leg, "retrace": back}) if trace else out


def _design(z: np.ndarray, age: np.ndarray) -> np.ndarray:
    z = np.clip(z, -1, 1)
    return np.column_stack([np.ones_like(z), z, z**2, np.log1p(age), (age <= 3).astype(float)])


def _logit(X: np.ndarray, y: np.ndarray, iters: int = 25) -> np.ndarray:
    """Logistic regression by Newton's method: five columns do not need a library."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        b += np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


def _runs(side: pd.Series) -> np.ndarray:
    """Bars since `side` last changed, per asset: 0 on the bar it changes."""
    run = (side != side.groupby(level=1).shift()).groupby(level=1).cumsum()
    return side.groupby([side.index.get_level_values(1), run]).cumcount().to_numpy()


def fit(x: pd.Series, cut: pd.Timestamp, window: int = strategy.WINDOW) -> dict:
    """The prior hazard and the two increment distributions, from development's centred turns of `x`.

    The hazard of bar t is P(the turn was at t-1 | the leg had not turned before), a logistic of the
    level at t-1 in the leg's direction and of the leg's age. Before the turn the increment is read
    on bars past the first `POST` of a leg, after it on those first bars, in the old leg's direction.
    """
    dev = x.index.get_level_values(0) < cut
    e = turn_events(x, window)
    leg = hold(e)
    side = leg.groupby(level=1).shift(2).fillna(0.0)  # the leg bar t was in, before any turn at t-1
    turned = (e.groupby(level=1).shift(1) == -side).to_numpy().astype(float)
    z = (x.groupby(level=1).shift(1) * side).to_numpy()
    ok = dev & (side != 0).to_numpy() & np.isfinite(z)
    beta = _logit(_design(z[ok], _runs(side)[ok].astype(float)), turned[ok])
    u = (x.groupby(level=1).diff() * leg).to_numpy()  # the bar's move in the direction of its leg
    k = _runs(leg)
    good = dev & np.isfinite(u) & (leg != 0).to_numpy()
    pre, post = u[good & (k > POST)], -u[good & (k >= 1) & (k <= POST)]
    return {
        "beta": beta,
        "m0": pre.mean(),
        "s0": pre.std(),
        "m1": post.mean(),
        "s1": post.std(),
        "base": turned[ok].mean(),
    }


def shiryaev(
    v: np.ndarray, p: dict, threshold: float, flat: bool = False, h: float = 0.0, trace: bool = False
) -> np.ndarray | pd.DataFrame:
    """+1 / -1 alarms on one asset: alarm when P(the leg has turned | the bars so far) >= `threshold`.

    Shiryaev's recursion with a hazard that moves: the prior is last bar's posterior plus the
    hazard of a turn at the last bar, the evidence is this bar's increment under the two Gaussians.
    After an alarm the side flips and the leg starts again at age 0. `flat` holds the hazard at its
    mean, which leaves the evidence alone to decide. `h` asks the zigzag's question as well: the
    alarm also waits for the series to have come back `h` from its extreme since the last alarm.
    `trace` returns, bar by bar, the alarm, the leg the detector believed it was in, the posterior,
    the prior hazard and the retracement from the leg's extreme.
    """
    out = np.zeros(len(v))
    rows = np.full((len(v), 4), np.nan)
    side, pi, age, ext = 1.0, 0.0, 0.0, v[0]
    b, m0, s0, m1, s1 = p["beta"], p["m0"], p["s0"], p["m1"], p["s1"]
    for t in range(1, len(v)):
        z = min(max(side * v[t - 1], -1.0), 1.0)
        if flat:
            rho = p["base"]
        else:
            rho = 1 / (1 + math.exp(-(b[0] + b[1] * z + b[2] * z * z + b[3] * math.log1p(age) + b[4] * (age <= 3))))
        prior = pi + (1 - pi) * rho
        u = side * (v[t] - v[t - 1])
        l1 = math.exp(-0.5 * ((u - m1) / s1) ** 2) / s1
        l0 = math.exp(-0.5 * ((u - m0) / s0) ** 2) / s0
        pi = prior * l1 / (prior * l1 + (1 - prior) * l0)
        age += 1
        ext = max(ext, side * v[t])
        rows[t] = side, pi, rho, ext - side * v[t]
        if pi >= threshold and ext - side * v[t] >= h:
            out[t], side, pi, age = -side, -side, 0.0, 0.0
            ext = side * v[t]
    if trace:
        return pd.DataFrame(rows, columns=["leg", "posterior", "hazard", "retrace"]).assign(alarm=out)
    return out


def alarms(x: pd.Series, detector, *args) -> pd.Series:
    """`detector` run on each asset of `x` separately, back on `x`'s index."""
    parts = [
        pd.Series(detector(x.xs(s, level=1).to_numpy(), *args), index=x.xs(s, level=1, drop_level=False).index)
        for s in x.index.get_level_values(1).unique()
    ]
    return pd.concat(parts).reindex(x.index)


def match(found: pd.Series, truth: pd.Series, cut: pd.Timestamp) -> dict:
    """Share of `truth`'s turns found before the next one, their delay in bars, false alarms a turn.

    An alarm belongs to the last true turn at or before it: the turn's first alarm of its kind is its
    detection, one of the other kind is false, a later one of the same kind is a re-entry after a
    false alarm. Development only.
    """
    delays, false, turns = [], 0, 0
    for s in found.index.get_level_values(1).unique():
        a, t = found.xs(s, level=1), truth.xs(s, level=1)
        a, t = a[(a != 0) & (a.index < cut)], t[(t != 0) & (t.index < cut)]
        turns += len(t)
        j = t.index.searchsorted(a.index, side="right") - 1
        seen = set()
        for when, kind, i in zip(a.index, a.to_numpy(), j):
            if i < 0 or t.iloc[i] != kind:
                false += 1
            elif i not in seen:
                seen.add(i)
                delays.append((when - t.index[i]) / pd.Timedelta("15min"))
    d = np.array(delays)
    return {"found": len(d) / turns, "delay": d.mean(), "delay_med": np.median(d), "false": false / turns}


def book(found: pd.Series, close: pd.Series, cut: pd.Timestamp) -> dict:
    """bp a trade of the always-in rule that flips at each alarm, development and hold-out, and by fold."""
    return _periods(plain(hold(found), close)[2], cut)


def _periods(held: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """bp a trade with its error and count on development and on the hold-out, and by fold."""
    when = pd.DatetimeIndex(held.entry)
    row = {}
    for period, keep in (("dev", when < cut), ("holdout", when >= cut)):
        g = held.gross[keep]
        row |= {f"{period}_bp": g.mean() * 1e4, f"{period}_se": g.std() / np.sqrt(len(g)) * 1e4, f"{period}_n": len(g)}
        if "why" in held:
            row[f"{period}_stopped"] = (held.why[keep] == "stop").mean()
    return row | (held.gross.groupby(fold_of(when)).mean() * 1e4).rename(lambda k: f"fold {k}").to_dict()


def kinds(held: pd.DataFrame, truth: pd.Series) -> list[str]:
    """What opened each trade: a detection, a false alarm, or a re-entry after a false alarm."""
    kind, seen = [], set()
    for s, row in held.iterrows():
        t = truth.xs(s, level=1)
        t = t[t != 0]
        i = t.index.searchsorted(row.entry, side="right") - 1
        if i < 0 or t.iloc[i] != row.side:
            kind.append("false alarm")
        else:
            kind.append("re-entry" if (s, i) in seen else "detection")
            seen.add((s, i))
    return kind


def split(held: pd.DataFrame, truth: pd.Series, cut: pd.Timestamp) -> pd.DataFrame:
    """Development's trades by what opened them, with the share a stop closed."""
    held = held[pd.DatetimeIndex(held.entry) < cut]
    g = held.assign(kind=kinds(held, truth), stopped=held.get("why", pd.Series("", index=held.index)) == "stop")
    g = g.groupby("kind")
    return pd.DataFrame(
        {"trades": g.size(), "bp": g.gross.mean() * 1e4, "bars": g.bars.mean(), "stopped": g.stopped.mean()}
    )


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(a true alarm scores above a false one), from the ranks (Mann-Whitney)."""
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def at_alarm(found: pd.Series, x: pd.Series, close: pd.Series, truth: pd.Series, window: int = strategy.WINDOW):
    """One row per alarm: whether it was true, the gross of the trade it opened, and what was known then.

    Every column is signed into the leg the alarm says is over and reads nothing after the alarm
    bar: from the prediction its level, the leg's extreme, the retracement from it, the last move,
    the leg's age; from the price the leg's move in units of its noise, the retracement from the
    extreme in ATR, the volatility, the volume against its day; the six exhaustion columns at the
    model's window; and the BTC filter of `strategy`.
    """
    rows = []
    for sym in x.index.get_level_values(1).unique():
        xv, a, t = x.xs(sym, level=1), found.xs(sym, level=1), truth.xs(sym, level=1)
        t = t[t != 0]
        bars = candles(sym, "15m")
        ex = legs.exhaustion(bars, window).reindex(xv.index)
        sigma = np.log(bars.close).diff().rolling(4 * window).std().reindex(xv.index)
        c = close.xs(sym, level=1)
        high, low = bars.high.reindex(xv.index), bars.low.reindex(xv.index)
        atr = (np.maximum(high, c.shift()) - np.minimum(low, c.shift())).rolling(window).mean() / c
        volume = (bars.volume / bars.volume.rolling(96).mean()).reindex(xv.index)
        below = strategy.below(pd.MultiIndex.from_arrays([xv.index, [sym] * len(xv)], names=x.index.names)).to_numpy()
        v, lc = xv.to_numpy(), np.log(c.to_numpy())
        at = np.flatnonzero(a.to_numpy() != 0)
        for k, i in enumerate(at):
            kind = a.iloc[i]
            side, start = -kind, at[k - 1] if k else 0
            leg_x, leg_c = side * v[start : i + 1], side * lc[start : i + 1]
            j = t.index.searchsorted(xv.index[i], side="right") - 1
            end = at[k + 1] if k + 1 < len(at) else len(v) - 1
            rows.append(
                {
                    "symbol": sym,
                    "when": xv.index[i],
                    "side": side,
                    "true": int(j >= 0 and t.iloc[j] == kind),
                    "gross": kind * (lc[end] - lc[i]),
                    "level": side * v[i - 1],
                    "extreme": leg_x.max(),
                    "retrace": leg_x.max() - side * v[i],
                    "move": side * (v[i] - v[i - 1]),
                    "age": i - start,
                    "leg_move": (leg_c[-1] - leg_c[0]) / (sigma.iloc[i] * np.sqrt(max(i - start, 1))),
                    "price_retrace": (leg_c.max() - leg_c[-1]) / atr.iloc[i],
                    "volatility": sigma.iloc[i],
                    "volume": volume.iloc[i],
                    "btc_below": float(below[i]),
                }
                | {f"ex_{col}": side * ex[col].iloc[i] for col in legs.EXHAUSTION}
            )
    return pd.DataFrame(rows).dropna()


def separate(f: pd.DataFrame, cut: pd.Timestamp) -> tuple[pd.DataFrame, tuple[float, float], pd.DataFrame]:
    """How well each column, and a logistic on all of them, tells true alarms from false ones.

    Each column's direction and the logistic are fitted on development; the AUC is read on both
    periods, and so is the rank correlation with the trade's gross, which is what pays. The
    quintiles of the logistic's score, cut on development, give the share of true alarms and what
    a true and a false alarm each make there.
    """
    dev = (f.when < cut).to_numpy()
    cols = [c for c in f.columns if c not in ("symbol", "when", "side", "true", "gross")]
    y = f.true.to_numpy()
    rows = []
    for col in cols:
        a = _auc(f[col].to_numpy()[dev], y[dev])
        sign = 1 if a >= 0.5 else -1
        rows.append(
            {
                "column": col,
                "auc_dev": max(a, 1 - a),
                "auc_holdout": _auc(sign * f[col].to_numpy()[~dev], y[~dev]),
                "ic_gross_dev": np.corrcoef(f[col][dev].rank(), f.gross[dev])[0, 1],
                "ic_gross_holdout": np.corrcoef(f[col][~dev].rank(), f.gross[~dev])[0, 1],
            }
        )
    one = pd.DataFrame(rows).set_index("column").sort_values("auc_dev", ascending=False)
    z = ((f[cols] - f[cols][dev].mean()) / f[cols][dev].std()).clip(-5, 5).to_numpy()
    X = np.column_stack([np.ones(len(z)), z])
    score = X @ _logit(X[dev], y[dev].astype(float))
    both = (_auc(score[dev], y[dev]), _auc(score[~dev], y[~dev]))
    q = np.searchsorted(np.quantile(score[dev], [0.2, 0.4, 0.6, 0.8]), score) + 1
    frame = f.assign(period=np.where(dev, "dev", "holdout"), q=q)
    keys = ["period", "q"]
    g = frame.groupby(keys)
    by = pd.DataFrame(
        {
            "alarms": g.size(),
            "share_true": g.true.mean(),
            "true_bp": frame[frame.true == 1].groupby(keys).gross.mean() * 1e4,
            "false_bp": frame[frame.true == 0].groupby(keys).gross.mean() * 1e4,
            "all_bp": g.gross.mean() * 1e4,
        }
    )
    return one, both, by


def gate(
    found: pd.Series, x: pd.Series, level: float, where: str = "alarm", close: bool = True
) -> tuple[pd.Series, pd.Series]:
    """`(kept, on)`: a long alarm kept only where `x` is at or under -`level`, a short at or over +`level`.

    `where` reads `x` at the alarm bar (`alarm`) or at the extreme of the leg the alarm closes
    (`extreme`), the lowest since the last alarm for a long and the highest for a short. A rejected
    alarm is dropped and the rule holds what it held (`close=False`), or it closes the position and
    the rule stands flat until the next kept alarm (`close=True`): `on` is off from the bar after
    it to that alarm, which is how `strategy.walked` stands a rule flat, closing at the rejected
    alarm's own close.
    """
    kept, on = pd.Series(0.0, index=found.index), pd.Series(True, index=found.index)
    for sym in found.index.get_level_values(1).unique():
        i = np.flatnonzero(found.index.get_level_values(1) == sym)
        a, v = found.to_numpy()[i], x.to_numpy()[i]
        keep, live = np.zeros(len(i)), np.ones(len(i), dtype=bool)
        off, last = False, 0
        for t in range(len(i)):
            rejected = False
            if a[t] != 0:
                lvl = v[t] if where == "alarm" else (v[last : t + 1].min() if a[t] > 0 else v[last : t + 1].max())
                if (a[t] > 0 and lvl <= -level) or (a[t] < 0 and lvl >= level):
                    keep[t], off = a[t], False
                else:
                    rejected = close
                last = t
            live[t] = not off
            off = off or rejected
        kept.iloc[i], on.iloc[i] = keep, live
    return kept, on


def _zscore(v: pd.Series, n: int = 96 * 30) -> pd.Series:
    """`v` against its own trailing month: the level of a ratio that drifts, made comparable over time."""
    return (v - v.rolling(n, min_periods=n // 3).mean()) / v.rolling(n, min_periods=n // 3).std()


def futures_columns(symbol: str, close: pd.Series) -> pd.DataFrame:
    """`data.futures`' columns as signals at each 15m bar of one symbol, every one causal.

    Funding and the two long/short ratios as levels against their month, the open interest as its
    change over 12 and 48 bars and signed by the price's 12-bar move (open interest rising with the
    move is new positions behind it), the taker flow over 12 bars, the futures-spot basis, the book's
    imbalance within 1, 2 and 5% and its 4-bar mean, the liquidity within 1%.
    """
    f = futures.load(symbol)
    c = close.reindex(f.index)
    oi, basis = np.log(f.oi.where(f.oi > 0)), np.log(f.fut_close / c)
    out = pd.DataFrame(
        {
            "funding": f.funding,
            "funding_z": _zscore(f.funding),
            "oi_12": oi.diff(12),
            "oi_48": oi.diff(48),
            "oi_with_price_12": np.sign(np.log(c).diff(12)) * oi.diff(12),
            "top_ls_z": _zscore(np.log(f.top_ls)),
            "top_ls_12": np.log(f.top_ls).diff(12),
            "ls_z": _zscore(np.log(f.ls)),
            "taker_ls_12": f.taker_ls.rolling(12).mean(),
            "taker_buy_12": f.taker_buy.rolling(12).mean() - 0.5,
            "basis_z": _zscore(basis),
            "basis_12": basis.diff(12),
            "depth1_z": _zscore(f.depth1),
        },
        index=f.index,
    )
    for k in futures.LEVELS:
        out[f"book{k}"] = f[f"book{k}"]
        out[f"book{k}_4"] = f[f"book{k}"].rolling(4).mean()
    return out


def forward_ic(columns: dict, pred: pd.Series, close: pd.Series, horizons=(12, 48)) -> pd.DataFrame:
    """Rank IC of every column with the forward log return, by fold, mean over the assets.

    Sampled every `h` bars so no two returns overlap; v2's prediction and an RSI at 12 are rows too,
    the yardstick a new column has to clear.
    """
    rows = []
    rsi = strategy.rsi(pred.index)
    for sym, cols in columns.items():
        c = np.log(close.xs(sym, level=1))
        x = cols.reindex(c.index).assign(**{"v2 prediction": pred.xs(sym, level=1), "rsi 12": rsi.xs(sym, level=1)})
        fold = fold_of(c.index)
        for h in horizons:
            fwd = c.shift(-h) - c
            for col in x.columns:
                for k in range(1, strategy.FOLDS + 1):
                    m = (np.arange(len(c)) % h == 0) & (fold == k) & x[col].notna().to_numpy() & fwd.notna().to_numpy()
                    ic = np.corrcoef(x[col][m].rank(), fwd[m].rank())[0, 1]
                    rows.append({"column": col, "h": h, "fold": k, "ic": ic})
    ic = pd.DataFrame(rows).groupby(["column", "h", "fold"]).ic.mean().unstack("fold")
    return ic.assign(
        dev=ic[[1, 2]].mean(axis=1), holdout=ic[[3, 4]].mean(axis=1), same_sign=np.sign(ic).nunique(axis=1) == 1
    )


def behind_the_move(symbol: str, close: pd.Series, k: int) -> pd.DataFrame:
    """The last `k` bars' move and the open interest's change over them, at each 15m bar.

    `move` is the sign of the price's `k`-bar change, `oi_z` the open interest's `k`-bar log change
    against its trailing month's dispersion: positive when positions were opened behind the move.
    """
    f = futures.load(symbol)
    c = np.log(close.reindex(f.index))
    doi = np.log(f.oi.where(f.oi > 0)).diff(k)
    return pd.DataFrame(
        {"c": c, "move": np.sign(c.diff(k)), "oi_z": doi / doi.rolling(96 * 30, min_periods=96 * 10).std()}
    )


def open_interest(close: pd.Series, cut: pd.Timestamp, ks=(4, 12, 24), horizons=(12, 24, 48, 96), hold_bars=48):
    """Open interest behind the move, three ways: `(ic, quadrants, rule)`.

    `ic`: rank IC of `move * oi_z`, and of `move` alone (momentum, the control), with the h-bar
    forward return, by fold. `quadrants`: the next `hold_bars` in the move's direction, bp, by
    whether open interest rose or fell with it. `rule`: when flat and |oi_z| >= Z, follow the move if
    open interest rose with it or fade it if it fell, out after `hold_bars`; one position at a time.
    """
    ic_rows, quad_rows, rule_rows = [], [], []
    for k in ks:
        for sym in strategy.ASSETS:
            d = behind_the_move(sym, close.xs(sym, level=1), k)
            d = d[d.index >= strategy.TEST_START]
            fold = fold_of(d.index)
            for h in horizons:
                fwd, take = d.c.shift(-h) - d.c, np.arange(len(d)) % h == 0
                for name, x in (("oi behind the move", d.move * d.oi_z), ("momentum", d.move)):
                    for f in range(1, strategy.FOLDS + 1):
                        m = take & (fold == f) & x.notna().to_numpy() & fwd.notna().to_numpy()
                        ic = np.corrcoef(x[m].rank(), fwd[m].rank())[0, 1]
                        ic_rows.append({"column": name, "k": k, "h": h, "fold": f, "ic": ic})
                if h == hold_bars:
                    m = take & fwd.notna().to_numpy() & d.oi_z.notna().to_numpy()
                    q = pd.DataFrame({"bp": (d.move * fwd)[m] * 1e4, "oi": np.where(d.oi_z[m] > 0, "rose", "fell")})
                    q["fold"] = fold[m]
                    quad_rows += [
                        {"k": k} | r for r in q.groupby(["oi", "fold"]).bp.mean().reset_index().to_dict("records")
                    ]
            cv, mv, zv = d.c.to_numpy(), d.move.to_numpy(), d.oi_z.to_numpy()
            for Z in (0.5, 1.0, 1.5, 2.0):
                i = 0
                while i < len(cv) - hold_bars:
                    if np.isfinite(zv[i]) and abs(zv[i]) >= Z and mv[i] != 0:
                        bp = mv[i] * np.sign(zv[i]) * (cv[i + hold_bars] - cv[i]) * 1e4
                        rule_rows.append({"k": k, "Z": Z, "when": d.index[i], "bp": bp})
                        i += hold_bars
                    else:
                        i += 1
    ic = pd.DataFrame(ic_rows).groupby(["column", "k", "h", "fold"]).ic.mean().unstack("fold")
    ic = ic.assign(dev=ic[[1, 2]].mean(axis=1), holdout=ic[[3, 4]].mean(axis=1))
    quads = pd.DataFrame(quad_rows).groupby(["k", "oi", "fold"]).bp.mean().unstack("fold")
    r = pd.DataFrame(rule_rows)
    r["period"] = np.where(r.when < cut, "dev", "holdout")
    r["fold"] = fold_of(pd.DatetimeIndex(r.when))
    rule = r.groupby(["k", "Z", "period"]).bp.agg(["size", "mean", "sem"]).unstack("period")
    rule.columns = [f"{period}_{stat}" for stat, period in rule.columns]
    rule = rule.join(r.groupby(["k", "Z", "fold"]).bp.mean().unstack("fold").add_prefix("fold "))
    return ic, quads, rule


def confirmation(sig: pd.Series, close: pd.Series, k: int) -> pd.Series:
    """At each signal, the signal's side times open interest behind the last `k` bars' move.

    The column predicts the next bars' direction (a move continues when positions were opened
    behind it, reverses when they were closed), so the product is positive where it agrees with
    the signal: a long at the end of a fall is confirmed when the fall was made closing positions.
    """
    parts = []
    for sym in strategy.ASSETS:
        d = behind_the_move(sym, close.xs(sym, level=1), k)
        col = (d.move * d.oi_z).reindex(close.xs(sym, level=1).index)
        parts.append(pd.Series(col.to_numpy(), index=close.xs(sym, level=1, drop_level=False).index))
    return (sig * pd.concat(parts).reindex(sig.index)).where(sig != 0)


def futures_at_alarm(f: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """`at_alarm`'s rows with the futures columns at the alarm, signed into the leg it closes.

    Over the leg (from the alarm before to this one): the change in open interest and in the top
    traders' ratio, the mean taker flow; at the alarm: funding, the ratios, the basis, the book.
    """
    rows = []
    for r in f.itertuples():
        F, C = futures.load(r.symbol), columns[r.symbol]
        t, start = r.when, r.when - pd.Timedelta(minutes=15 * int(r.age))
        leg = F.loc[start:t]
        rows.append(
            {
                "oi_leg": np.log(F.oi.get(t, np.nan) / F.oi.get(start, np.nan)),
                "oi_4": np.log(F.oi.get(t, np.nan) / F.oi.shift(4).get(t, np.nan)),
                "funding_s": r.side * F.funding.get(t, np.nan),
                "funding_z_s": r.side * C.funding_z.get(t, np.nan),
                "top_ls_s": r.side * C.top_ls_z.get(t, np.nan),
                "top_ls_leg": r.side * np.log(F.top_ls.get(t, np.nan) / F.top_ls.get(start, np.nan)),
                "ls_s": r.side * C.ls_z.get(t, np.nan),
                "taker_leg": r.side * leg.taker_ls.mean(),
                "taker_4": r.side * F.taker_ls.rolling(4).mean().get(t, np.nan),
                "takerbuy_leg": r.side * (leg.taker_buy.mean() - 0.5),
                "basis_s": r.side * C.basis_z.get(t, np.nan),
                "book1_s": r.side * F.book1.get(t, np.nan),
                "book2_s": r.side * F.book2.get(t, np.nan),
                "book5_s": r.side * F.book5.get(t, np.nan),
                "depth1_leg": F.depth1.get(t, np.nan) - leg.depth1.mean(),
            }
        )
    return pd.concat([f.reset_index(drop=True), pd.DataFrame(rows)], axis=1).dropna()


def _selfcheck() -> None:
    """A noiseless saw with known turns: each detector finds every one, at the delay its rule implies."""
    i = np.arange(24 * 6 + 1)
    v = 0.05 * np.abs(i % 24 - 12)  # a top on bar 0, then a turn every 12 bars, 0.05 a bar
    t = pd.date_range("2025-06-01", periods=len(v), freq="15min", tz="UTC")
    x = pd.Series(v, index=pd.MultiIndex.from_arrays([t, ["A"] * len(v)], names=["open_time", "symbol"]))
    truth = pd.Series(0.0, index=x.index)
    turns = list(range(0, len(v) - 1, 12))
    truth.iloc[turns] = [-1.0 if k % 24 == 0 else 1.0 for k in turns]
    cut = t[-1] + pd.Timedelta("1D")
    # A drop of 0.05 a bar clears h = 0.12 on the third bar after the top: delay 3, no false alarm.
    z = alarms(x, zigzag, 0.12)
    m = match(z, truth, cut)
    assert m["found"] == 1.0 and m["delay"] == 3.0 and m["false"] == 0.0, m
    # With the evidence this sharp and a flat hazard, the first bar that moves back is enough.
    p = {"beta": np.zeros(5), "m0": 0.05, "s0": 0.01, "m1": -0.05, "s1": 0.01, "base": 0.05}
    m = match(alarms(x, shiryaev, p, 0.9, True), truth, cut)
    assert m["found"] == 1.0 and m["delay"] == 1.0 and m["false"] == 0.0, m
    # The gate on the leg's extreme: tops at 0.6 keep every short at 0.5, bottoms at 0 reject every
    # long, and a rejected long closes the short at its own close and stands flat to the next short.
    kept, on = gate(z, x, 0.5, "extreme", True)
    assert set(kept[kept != 0]) == {-1.0}
    assert on.iloc[turns[1] + 3] and not on.iloc[turns[1] + 4] and on.iloc[turns[2] + 3]
    assert gate(z, x, 0.5, "extreme", False)[1].all()
    # An alarm on the wrong side is false, a later one on the right side is a re-entry, not a detection.
    noisy = z.copy()
    noisy.iloc[turns[0] + 6] = 1.0
    assert match(noisy, truth, cut)["false"] == 1 / len(turns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zigzag", type=float, nargs="+", metavar="H", help="retracements, in the prediction's units")
    ap.add_argument("--shiryaev", type=float, nargs="+", metavar="P", help="posterior thresholds")
    ap.add_argument("--flat", action="store_true", help="with --shiryaev: also the flat-hazard detector")
    ap.add_argument("--both", type=float, nargs="+", metavar="H", help="with --shiryaev: also wait for a retracement")
    ap.add_argument("--split", action="store_true", help="where each detector's P&L goes, on the prediction")
    ap.add_argument("--features", type=float, metavar="P", help="what tells Shiryaev P's true alarms from false ones")
    ap.add_argument("--sl", type=float, nargs="+", metavar="ATR", help="Shiryaev 0.5's trades with these stops")
    ap.add_argument("--futures", action="store_true", help="funding, open interest, flow and book depth (data.futures)")
    ap.add_argument("--gate", type=float, nargs="+", metavar="L", help="zigzag 0.2 and Shiryaev 0.5 past these levels")
    ap.add_argument("--oi", action="store_true", help="open interest behind the move: IC, quadrants, a 48-bar rule")
    ap.add_argument("--confirm", action="store_true", help="open interest as a confirmation of the rules' signals")
    args = ap.parse_args()

    _selfcheck()
    pred, close, cut = strategy.load()
    series = {"prediction": pred, "rsi 12": strategy.rsi(pred.index)}
    pd.set_option("display.width", 250)
    truth = turn_events(pred, strategy.WINDOW)
    if args.confirm:
        p, bars, rows = fit(pred, cut), strategy.ohlc(pred.index), []
        rules = {
            "shiryaev 0.5": alarms(pred, shiryaev, p, 0.5),
            "zigzag 0.2": alarms(pred, zigzag, 0.2),
            "reentry 0.40": strategy.signal(pred, "reentry", 0.40),
        }
        for k in (4, 12, 24):
            for name, sig in rules.items():
                conf = confirmation(sig, close, k)
                rows.append({"k": k, "signal": name, "keep": "all"} | _periods(plain(hold(sig), close)[2], cut))
                for c in (0.0, 0.5, 1.0):
                    # `gate` keeps a long where x <= -c and a short where x >= c: x = -side * conf.
                    kept, on = gate(sig, (-sig * conf).fillna(0.0), c, "alarm", True)
                    held = strategy.walked(kept, bars, on=on)[2]
                    rows.append({"k": k, "signal": name, "keep": f"confirmed >= {c:g}"} | _periods(held, cut))
        t = pd.DataFrame(rows).set_index(["k", "signal", "keep"]).filter(regex="^(?!.*stopped)")
        print("signals kept only where open interest confirms them, a rejected one closes; bp a trade, no fees\n")
        print(t.round(1).to_string())
        return
    if args.oi:
        ic, quads, rule = open_interest(close, cut)
        print("rank IC with the h-bar forward return, mean of ETH/BTC/SOL\n")
        print(ic.round(3).to_string())
        print("\nthe next 48 bars in the direction of the last k-bar move, bp, by whether open interest rose with it\n")
        print(quads.round(1).to_string())
        print("\nfollow the move when open interest rose, fade it when it fell, out after 48 bars; bp a trade\n")
        print(rule.round(1).to_string())
        return
    if args.gate:
        p, rows, bars = fit(pred, cut), [], strategy.ohlc(pred.index)
        for name, found in (
            ("zigzag 0.2", alarms(pred, zigzag, 0.2)),
            ("shiryaev 0.5", alarms(pred, shiryaev, p, 0.5)),
        ):
            rows.append({"detector": name, "gate": "none"} | book(found, close, cut))
            for where in ("alarm", "extreme"):
                for shut in (True, False):
                    for level in args.gate:
                        kept, on = gate(found, pred, level, where, shut)
                        label = f"{where}, {'close' if shut else 'ignore'} {level:g}"
                        held = strategy.walked(kept, bars, on=on)[2]
                        rows.append({"detector": name, "gate": label} | _periods(held, cut))
        t = pd.DataFrame(rows).set_index(["detector", "gate"]).filter(regex="^(?!.*stopped)")
        print("signals kept only past a level; bp a trade, no fees; n is trades over the period\n")
        print(t.round(1).to_string())
        return
    if args.futures:
        columns = {sym: futures_columns(sym, close.xs(sym, level=1)) for sym in strategy.ASSETS}
        print("rank IC with the forward log return, mean of ETH/BTC/SOL, one sample every h bars\n")
        print(forward_ic(columns, pred, close).round(3).sort_values(["h", "dev"]).to_string())
        f = at_alarm(alarms(pred, shiryaev, fit(pred, cut), 0.5), pred, close, truth)
        g = futures_at_alarm(f, columns)
        new = [c for c in g.columns if c not in f.columns]
        for label, table in (("the futures columns", g[["symbol", "when", "side", "true", "gross"] + new]), ("all", g)):
            one, (dev, holdout), by = separate(table, cut)
            print(f"\nshiryaev 0.5's alarms, {label}: logistic AUC {dev:.3f}, hold-out {holdout:.3f}\n")
            if label != "all":
                print(one.round(3).to_string() + "\n")
            print(by.round(2).to_string())
        return
    if args.features:
        f = at_alarm(alarms(pred, shiryaev, fit(pred, cut), args.features), pred, close, truth)
        one, (dev, holdout), by = separate(f, cut)
        print(f"shiryaev {args.features}: {len(f)} alarms, {f.true.mean():.0%} true\n")
        print(one.round(3).to_string())
        print(f"\nlogistic on every column, fitted on development: AUC {dev:.3f}, hold-out {holdout:.3f}\n")
        print(by.round(2).to_string())
        return
    if args.sl:
        found, bars = alarms(pred, shiryaev, fit(pred, cut), 0.5), strategy.ohlc(pred.index)
        rows, parts = [], {}
        for k in [0.0, *args.sl]:
            for trail in (False, True) if k else (False,):
                label = f"stop {k:g} ATR" + (", trailing" if trail else "") if k else "no stop"
                stop = ("atr", k) if k else None
                held = strategy.walked(found, bars, stop=stop, after="opposite", trail=trail)[2]
                rows.append({"rule": label} | _periods(held, cut))
                parts[label] = split(held, truth, cut)
        print("shiryaev 0.5 on the prediction; after a stop, flat until the next alarm\n")
        print(pd.DataFrame(rows).set_index("rule").round(2).to_string())
        for label, table in parts.items():
            print(f"\n{label}, development, by what opened the trade\n")
            print(table.round(2).to_string())
        return
    rows, splits = [], {}
    for name, x in series.items():
        truth = turn_events(x, strategy.WINDOW)
        runs = [(f"zigzag {h}", alarms(x, zigzag, h)) for h in args.zigzag or []]
        if args.shiryaev:
            p = fit(x, cut)
            if name == "prediction":
                # The page runs `V2_FIT`; if the predictions moved under it, the page and this table differ.
                assert all(np.allclose(p[k], V2_FIT[k], atol=1e-3) for k in V2_FIT), "refit V2_FIT"
            print(name, {k: np.round(v, 4) for k, v in p.items()})
            for flat in (False, True) if args.flat else (False,):
                runs += [
                    (f"shiryaev {th}" + (" flat" if flat else ""), alarms(x, shiryaev, p, th, flat))
                    for th in args.shiryaev
                ]
            runs += [
                (f"shiryaev {th} + zigzag {h}", alarms(x, shiryaev, p, th, False, h))
                for th in args.shiryaev
                for h in args.both or []
            ]
        for label, found in runs:
            rows.append({"signal": name, "detector": label} | match(found, truth, cut) | book(found, close, cut))
            if args.split and name == "prediction":
                splits[label] = split(plain(hold(found), close)[2], truth, cut)
    print("\nagainst each series' own centred turns at 12 bars; found, delay and false alarms on development\n")
    print(pd.DataFrame(rows).set_index(["signal", "detector"]).round(2).to_string())
    for label, table in splits.items():
        print(f"\n{label} on the prediction, development: trades by what opened them\n")
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
