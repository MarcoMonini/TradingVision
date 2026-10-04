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
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from tradingvision import strategy
from tradingvision.strategy import fold_of, hold, plain, turn_events

POST = 6  # the first bars of a leg that make the post-turn increment distribution


def zigzag(v: np.ndarray, h: float) -> np.ndarray:
    """+1 when `v` has risen `h` above its low since the last alarm, -1 when it fell `h` below its high."""
    out = np.zeros(len(v))
    side, hi, lo = 0, v[0], v[0]
    for t in range(len(v)):
        hi, lo = max(hi, v[t]), min(lo, v[t])
        if side >= 0 and hi - v[t] >= h:
            out[t], side, lo = -1, -1, v[t]
        elif side <= 0 and v[t] - lo >= h:
            out[t], side, hi = 1, 1, v[t]
    return out


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


def shiryaev(v: np.ndarray, p: dict, threshold: float, flat: bool = False, h: float = 0.0) -> np.ndarray:
    """+1 / -1 alarms on one asset: alarm when P(the leg has turned | the bars so far) >= `threshold`.

    Shiryaev's recursion with a hazard that moves: the prior is last bar's posterior plus the
    hazard of a turn at the last bar, the evidence is this bar's increment under the two Gaussians.
    After an alarm the side flips and the leg starts again at age 0. `flat` holds the hazard at its
    mean, which leaves the evidence alone to decide. `h` asks the zigzag's question as well: the
    alarm also waits for the series to have come back `h` from its extreme since the last alarm.
    """
    out = np.zeros(len(v))
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
        if pi >= threshold and ext - side * v[t] >= h:
            out[t], side, pi, age = -side, -side, 0.0, 0.0
            ext = side * v[t]
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
    held = plain(hold(found), close)[2]
    when = pd.DatetimeIndex(held.entry)
    row = {}
    for period, keep in (("dev", when < cut), ("holdout", when >= cut)):
        g = held.gross[keep]
        row |= {f"{period}_bp": g.mean() * 1e4, f"{period}_se": g.std() / np.sqrt(len(g)) * 1e4, f"{period}_n": len(g)}
    return row | (held.gross.groupby(fold_of(when)).mean() * 1e4).rename(lambda k: f"fold {k}").to_dict()


def split(found: pd.Series, truth: pd.Series, close: pd.Series, cut: pd.Timestamp) -> pd.DataFrame:
    """Development's trades by what opened them: a detection, a false alarm, a re-entry after one."""
    held = plain(hold(found), close)[2]
    held = held[pd.DatetimeIndex(held.entry) < cut].copy()
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
    g = held.assign(kind=kind).groupby("kind")
    return pd.DataFrame({"trades": g.size(), "bp": g.gross.mean() * 1e4, "bars": g.bars.mean()})


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
    args = ap.parse_args()

    _selfcheck()
    pred, close, cut = strategy.load()
    series = {"prediction": pred, "rsi 12": strategy.rsi(pred.index)}
    pd.set_option("display.width", 250)
    rows, splits = [], {}
    for name, x in series.items():
        truth = turn_events(x, strategy.WINDOW)
        runs = [(f"zigzag {h}", alarms(x, zigzag, h)) for h in args.zigzag or []]
        if args.shiryaev:
            p = fit(x, cut)
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
                splits[label] = split(found, truth, close, cut)
    print("\nagainst each series' own centred turns at 12 bars; found, delay and false alarms on development\n")
    print(pd.DataFrame(rows).set_index(["signal", "detector"]).round(2).to_string())
    for label, table in splits.items():
        print(f"\n{label} on the prediction, development: trades by what opened them\n")
        print(table.round(1).to_string())


if __name__ == "__main__":
    main()
