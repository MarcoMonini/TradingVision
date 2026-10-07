"""Routes 5 and 6 of `false_alarms.html`: a slow 4h base, volatility-managed, and fast signals as its timing.

Neither reads v2 to decide a trade. Route 5 is the rule a strategy has to beat, `rsi_centered` on 4h
bars, priced at OKX's fee and scaled by the inverse of its volatility (Moreira and Muir 2017). Route 6
keeps that rule's decisions and asks whether the weak fast signals the project has measured (open
interest, the book, exhaustion, v2's level) can choose *when* inside the next few hours to execute
them: a trade the base makes anyway pays its fee anyway, so the signal no longer has to pay one.
Every period is `strategy.edges`: `dev` (v2's folds 1-2, where the cards' criteria are read),
`holdout` (folds 3-4, a check) and `2021` (2021-01 to 2025-06, the confirmation of what reads no v2).
Money is read on the 13 `TRADABLE` pairs, each a sleeve of 1/13 of the capital.

**The rule, reproduced first.** Step 7's `rsi_centered > 0.3` ran at `swing`'s default window, 24
(`EXTREMA_WINDOW`; the cache stamp `data/swing-4h-full.json` says 24 and the 20 `STUDY` pairs).
`swing.baselines` on that cache's rows from 2023-01 gives gross 0.182, net +0.126 a year at 0.25%,
11.1 round trips a year, hold +0.057: the spec's step-7 table to the third decimal. The +0.116 that
`OLD/README.md`, `swing`'s comment and HANDOFF quote is not what these rows give. At OKX's 0.10% the
same rows net +0.159.

**Route 5: the rule against hold, per fold** (`--base`; equal-weighted sleeves, annual Sharpe of the
daily P&L with its error from weekly blocks, log a year, max drawdown):

    period   fold   rule Sharpe     hold Sharpe    rule log/yr  hold log/yr  rule mdd  hold mdd
    dev       1    +1.06 (1.81)    +1.51 (1.41)     +0.192       +0.734       8.9%     19.7%
    dev       2    -1.14 (1.03)    -1.45 (1.45)     -0.200       -1.345       8.4%     43.8%
    holdout   3    -3.39 (1.26)    -1.56 (1.33)     -0.483       -1.212      14.7%     33.6%
    holdout   4    +1.88 (1.54)    +2.14 (2.02)     +0.544       +1.074       5.8%     23.7%
    2021      1    +2.97 (1.19)    +1.96 (0.93)     +1.030       +1.479      16.2%     62.1%
    2021      2    -0.10 (1.06)    -0.50 (0.95)     -0.042       -0.771      23.8%     74.4%
    2021      3    +1.41 (0.81)    +1.40 (0.95)     +0.314       +0.649      12.7%     32.0%
    2021      4    +0.98 (0.96)    +0.56 (0.92)     +0.217       +0.145      16.4%     56.1%

The rule is invested 11-24% of the time and trades 19-26 times a year a sleeve (about twelve round
trips). It keeps a fifth to a half of hold's drawdown and its Sharpe is above hold's in five folds
of eight, but every Sharpe here has an error of about one: eight folds of four to thirteen months
cannot tell the two apart.

**Scaling it by 1/sigma-hat does not move it** (w = min(sigma* / sigma-hat, w_max), sigma-hat an
exponential average of 30 days of squared 4h returns, sigma* the median of the year before the
period; traded only when w leaves a band delta around the drifted weight; the fee on |dw|). The
Sharpe difference against the unscaled rule, with its paired error:

    w_max, delta    dev               holdout          2021
    1, 0.10    +0.05 (0.03) / -0.44 (0.21)   +0.13 / +0.12   -0.01 / +0.14 / +0.02 / -0.06
    1, 0.25    +0.02 / -0.31                 +0.10 / +0.04   +0.04 / +0.12 / +0.01 / -0.03
    2, 0.10    +0.13 (0.14) / -0.52 (0.32)   +0.69 / +0.33   -0.12 / +0.47 / +0.09 / -0.02
    2, 0.25    +0.13 / -0.42                 +0.65 / +0.31   -0.05 / +0.45 / +0.11 / +0.02

and scaling hold, the control, moves hold by -0.05 to +0.13 at w_max 1. At w_max 1 the scaled rule
is the rule: on dev the year before was the louder one, the cap binds almost always (the scaled
hold is 0.99 / 0.91 invested) and turnover moves from 24.7 to 24.5 a year. **The card fails on
dev at every w_max and band**: the scaled rule is above the rule in fold 1 only and above hold in
no dev fold. On the hold-out it is above the rule in both folds and above hold in none (w_max 1)
or in fold 4 only (w_max 2); on 2021 above hold in all four folds, as the unscaled rule already is,
and above the rule in two or three.

**What the card can see** (`--power`, `power_base`): a GARCH of t(4) shocks for the portfolio,
always in, a mean `kappa * sd ** gamma` from gamma 2 (risk-return, scaling hurts) to -2 (the mean
falls as volatility rises), w_max 1, band 0.1, 1,000 reps. The true Sharpe difference runs from
-0.02 to +0.03 on dev's two 120-day folds, and the criterion fires in 17-23% of the reps whatever
it is; on 2021's four 403-day folds, -0.05 to +0.07 and 3-11%. Gamma -4 to -8 give +0.05 to
+0.08 on dev (400 reps): without leverage, a weight that can only fall below 1 has little to
manage. The card cannot pass for the right reason at these sizes, and its failure on dev is no
evidence against a small effect either.

**Route 6: timing the base's executions** (`--timing`). Each decision of the unscaled rule (an entry
or an exit at a 4h close) and, as a second set, each trade of the scaled rule at w_max 1, delta 0.1
(on dev almost the same decisions), is executed on 15m bars: at once (the 15m bar closing at the
4h close, `anchor`), at the first bar of the next N where d * score >= b, else at the window's
last bar (`execute`), or at controls that read no score. The gain against at once is d * (log
p_now - log p_exec): what the base's P&L changes by. The window stops short of the base's next
decision. The score is fast columns, each z-scored against its trailing month (`detect._zscore`)
and signed by the sign of its rank IC with the 24- and 48-bar forward return on dev's every bar
(never one phase), then averaged with equal weights and z-scored again (`scores`). On dev the ICs
are all small: book5_4 +0.041 / +0.034, rsi12 -0.024 / -0.013, v2 -0.023 / -0.008, stretch -0.016 /
-0.012, divergence -0.020 / -0.001, oi24 +0.005 / +0.013, the rest under 0.01. The composite reads
the six exhaustion columns at 12 bars, v2's level and on BTC, ETH and SOL open interest behind the
move at 12 and 24 and the book's imbalance within 5% over an hour (`composite`); on 2021 the RSI at
12 replaces v2 and there are no futures. b is chosen on dev: 1 for the composite at every N and in
both sets, the grid's edge.

Three controls, because a regime in which waiting pays rewards any rule that waits: the random bar
of the card; the window's last bar; and `lag`, the delays the same b took on the asset's other
decisions, drawn at random, which keeps how long b waits and drops when. `b - lag`, paired, is the
score's own part. Means are over executions weighted by the size traded, errors from N-bar blocks
on the clock (`tabulate`).

    composite, rule set, b = 1, bp an execution (error)
    period   fold  N=48 vs at once   random   last bar   b - lag        N=96 b - lag    executions
    dev       1      +0.6 (26.5)     -11.8    -14.2     +13.4 (20.4)    +21.6 (20.2)      106
    dev       2     +43.0 (17.5)     +21.7    +43.0     +23.3 (9.7)     +25.3 (9.6)        82
    holdout   3     +60.8 (11.0)     +42.9    +87.8     +20.2 (12.2)    +16.6 (11.2)       92
    holdout   4      -1.0 (20.9)     -25.1    -56.2     +16.3 (12.7)    +25.6 (14.0)      113
    2021      1     +35.2 (23.4)     -10.9    -77.1     +44.5 (15.1)    +45.6 (13.4)      313
    2021      2     +31.9 (16.9)      +9.3     -2.1     +17.4 (8.2)     +23.4 (9.4)       287
    2021      3     +18.0 (28.3)      +6.8    +22.9      +8.4 (10.5)    +10.1 (10.5)      343
    2021      4      -8.1 (20.2)     -14.2    -33.7      +4.1 (9.8)      +0.7 (10.6)      342

At N = 16 the score's part is -3.7 / +4.0 on dev and the gain -7.0 / +5.9. **The card, as written,
passes on dev** for the rule set at N = 48 and 96 and for the scaled set at 48 (at 96 the random bar,
+38 +/- 18 in fold 2, is outside the noise), and fails at N = 16. Read with its power, that pass is
not evidence: fold 1 is +0.6 +/- 26.5, and the same criterion fires in a third of the reps at IC 0.
The hold-out passes at N = 16 only: at 48 and 96 the random bar alone makes +43 and +53 bp in fold
3, past two errors, and fold 4 is -1.0 / -1.6. On 2021 the rule set fails in fold 4 at every N (the
scaled set passes at 16). What against-at-once mostly reads is the cost of waiting, which flips
with the regime (+88 and -56 bp at the last bar of 12 hours on the hold-out's two folds). The
score's own part does not flip: at N = 48 it is positive in all eight folds, +4 to +45 bp, and the
hold-out and 2021, where neither the signs nor b were chosen, pool (inverse variance) to +16 bp,
error 4. At about 24 executions a year a sleeve that is +0.04 log a year, against a base that nets
-0.48 to +1.03 by fold.

Each column alone (rule set, dev, at its own dev b, N = 48): v2 +6.9 / +62.9, divergence +1.7 /
+42.5, rsi12 -8.5 / +57.3, stretch -13.6 / +45.4, open interest at 12 -14.8 / +35.2 and the book
-4.3 / +19.0 on 26 / 24 executions; deceleration and volume_climax negative in both folds.

**What the card can see** (`--power`, `power_timing`): `sim_timing.py`'s construction (a GARCH of
t(4) at 0.4% a bar, an AR(1) signal of half-life 24 read clipped at one sd) at dev's real counts,
106 and 82 executions, b chosen on both folds pooled. At N = 48, "above zero in every fold" fires in
33% of the reps at IC 0, 41% at 0.02 and 56% at 0.05; adding a pooled t of 2, 3.5%, 6.4% and 14%.
The chosen b gains +12.9 bp at IC 0.05 against the document's +8.2 (1.9); +4.1 at IC 0 is the
choice of b among three on the same rows.

**Simplifications, stated.** A sleeve's weight drifts with its asset between trades and the sleeves
are rebalanced to equal weight daily for free. A period starts in the position the rule holds,
without paying to enter it. An execution happens at the close of the bar whose score triggers it,
the project's convention. The random bar averages 32 draws and moves by a few bp with the seed.

    uv run python -m tradingvision.timing --base [--period dev|holdout|2021] [--fee 0.001]
    uv run python -m tradingvision.timing --timing [--period dev|holdout|2021]
    uv run python -m tradingvision.timing --power [--reps 1000]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from tradingvision import detect, legs, strategy
from tradingvision.data.binance import TRADABLE, load
from tradingvision.metrics import blocked, spearman
from tradingvision.oracle import FEE

# ---------------------------------------------------------------- route 5: the slow base

TF = "4h"
PER_DAY = 6  # 4h bars a day
# `rsi_centered` at 24 is the column of step 7: `swing --timeframe 4h --baseline` ran at swing's
# default window, `EXTREMA_WINDOW` = 24, and its stamp (`data/swing-4h-full.json`) says so.
WINDOW = 24
ENTER, EXIT = 0.3, 0.0  # `swing.BASELINES`' first row: long at or above 0.3, flat at or below 0.0
SPAN = 30 * PER_DAY  # sigma-hat: an exponential average of squared 4h returns, span 30 days
# sigma* is the median sigma-hat of each asset over the year before the period's first bar: fixed
# before the period, and the nearest regime to it rather than the median of all history, which holds
# 2017-2018 and 2021. Chosen, not measured. On dev the year before was the louder one, so at w_max 1
# the cap binds almost always: the scaled hold is 0.99 / 0.91 invested by fold.
BEFORE = pd.Timedelta(365, "D")
WMAX = (1.0, 2.0)  # 1 is spot without leverage, the realistic one
DELTA = (0.1, 0.25)  # the no-rebalance band around the held weight
YEAR = pd.Timedelta(365.25, "D")
WEEK = pd.Timedelta("7D")  # the blocks of the daily P&L's error: a trade of the rule lasts days

# ---------------------------------------------------------------- route 6: the timing

FAST = "15m"
BAR = pd.Timedelta("15min")
FAST_WINDOW = 12  # v2's window, the exhaustion columns' and the RSI's
NS = (16, 48, 96)  # the window after a decision, in 15m bars: 4, 12 and 24 hours
BS = (0.5, 0.75, 1.0)  # the score's threshold, in its standard deviations
DRAWS = 32  # random bars per execution, averaged
HORIZONS = (24, 48)  # the forward returns whose rank IC on dev signs each column
FUTURES = strategy.ASSETS  # the three assets `data.futures` holds
SCALED = (1.0, 0.1)  # (w_max, delta) of the scaled rule whose rebalances are the second set
GAINS = [f"b{b:g}" for b in BS] + ["random", "end"] + [f"lag{b:g}" for b in BS]
VS = [f"b{b:g}-lag" for b in BS]  # each b less its own delays at random, execution by execution: the score's part


def variants() -> list[tuple[str, float | None, float | None]]:
    """`(kind, w_max, delta)`: the rule and hold, each unscaled and at every w_max and band."""
    return [(k, None, None) for k in ("rule", "hold")] + [
        (k, w, d) for k in ("rule", "hold") for w in WMAX for d in DELTA
    ]


def name(kind: str, w: float | None, d: float | None) -> str:
    return kind if w is None else f"{kind} w{w:g} d{d:g}"


def rule(close: pd.Series, window: int = WINDOW, enter: float = ENTER, exit: float = EXIT) -> pd.Series:
    """1 from the first close with `rsi_centered` >= `enter` to the first <= `exit`, 0 otherwise: the
    state `swing.baselines` holds, known at the bar's close. NaN warm-up is flat, as there."""
    r = RSIIndicator(close, window=window).rsi() / 50 - 1
    state = pd.Series(np.nan, index=close.index)
    state[r >= enter] = 1.0
    state[r <= exit] = 0.0
    return state.ffill().fillna(0.0)


def sigma(close: pd.Series, span: int = SPAN) -> pd.Series:
    """sigma-hat at each bar: the root of an exponential average of squared log returns, the bar's own
    return included, so it is known at the bar's close and not before."""
    r = np.log(close).diff()
    return np.sqrt((r**2).ewm(span=span, min_periods=span).mean())


def sleeve(R, state, target, delta, fee: float = FEE):
    """`(net, traded, held)` of a sleeve from bar 1 on, every array `(bars, variants)`.

    `R` is each bar's simple return, `(bars,)` or `(bars, variants)`; `state` the rule's 0/1 and
    `target` the weight wanted at each bar's close. The weight held into bar 0's close is `target[0]`,
    as if the strategy were already running: a period starts in the position the rule holds, without
    paying to enter it. Between trades the weight drifts with the price (a half-invested sleeve whose
    asset rises is more than half invested), and a trade happens at a bar's close when the rule changes
    state, always, or when the target leaves `delta` around the drifted weight. The fee is `fee` on
    the notional traded, |dw| of the sleeve's equity, charged at that close.
    """
    R = np.asarray(R, float)
    state, target = np.asarray(state, float), np.asarray(target, float)
    n = len(target)
    h = target[0].copy()
    net, traded, held = (np.zeros((n - 1,) + target.shape[1:]) for _ in range(3))
    for t in range(1, n):
        g = h * R[t]
        h = h * (1 + R[t]) / (1 + g)
        dw = np.where((state[t] != state[t - 1]) | (np.abs(target[t] - h) > delta), target[t] - h, 0.0)
        h = h + dw
        net[t - 1], traded[t - 1], held[t - 1] = (1 + g) * (1 - fee * np.abs(dw)) - 1, dw, h
    return net, traded, held


def slow(symbol: str, period: str = "dev", fee: float = FEE) -> dict:
    """Every variant of the base on one asset's 4h bars inside `period`: per bar, the net simple return,
    the weight traded at its close and the weight held after it, with the bar's fold."""
    close = load(symbol, TF).close
    e = strategy.edges(period)
    state, s = rule(close).to_numpy(), sigma(close)
    star = s[(s.index >= e[0] - BEFORE) & (s.index < e[0])].median()
    fold = strategy.fold_in(close.index, period)
    inside = np.flatnonzero(fold > 0)
    i0, i1 = inside[0], inside[-1] + 1
    cols, states, targets, deltas = [], [], [], []
    for kind, w, d in variants():
        on = state if kind == "rule" else np.ones(len(close))
        cols.append(name(kind, w, d))
        states.append(on)
        targets.append(on if w is None else on * np.minimum(star / s.to_numpy(), w))
        deltas.append(DELTA[0] if d is None else d)
    R = np.expm1(np.log(close).diff().fillna(0.0).to_numpy())
    net, traded, held = sleeve(
        R[i0 - 1 : i1],
        np.column_stack(states)[i0 - 1 : i1],
        np.column_stack(targets)[i0 - 1 : i1],
        np.array(deltas),
        fee,
    )
    index = close.index[i0:i1]
    return {
        "fold": pd.Series(fold[i0:i1], index=index),
        "net": pd.DataFrame(net, index=index, columns=cols),
        "traded": pd.DataFrame(traded, index=index, columns=cols),
        "held": pd.DataFrame(held, index=index, columns=cols),
        "star": star,
    }


def _sharpe(x: pd.Series) -> tuple[float, float]:
    """Annualised mean of a daily series over its sd, with the error from weekly blocks. The mean is
    the sample's: `blocked`'s is a mean of block means, which a partial first or last week moves."""
    sd = x.std()
    return x.mean() / sd * np.sqrt(365.25), blocked(x, WEEK)["se"] / sd * np.sqrt(365.25)


def summarise(books: dict[str, dict], period: str) -> pd.DataFrame:
    """Per fold and variant, on the equal-weighted portfolio of the sleeves (rebalanced across sleeves
    daily, free): Sharpe and its error, the Sharpe difference against the unscaled version and against
    hold with errors, log a year, max drawdown, and per sleeve turnover, trades a year, mean exposure."""
    e = strategy.edges(period)
    folds = sorted({int(f) for b in books.values() for f in b["fold"].unique()})
    rows = []
    for k in folds:
        years = (e[k - folds[0] + 1] - e[k - folds[0]]) / YEAR
        daily, turnover, trades, expo = [], [], [], []
        for b in books.values():
            m = (b["fold"] == k).to_numpy()
            net = np.log1p(b["net"][m])
            daily.append(np.expm1(net.groupby(net.index.floor("D")).sum()))
            turnover.append(b["traded"][m].abs().sum() / years)
            trades.append((b["traded"][m] != 0).sum() / years)
            expo.append(b["held"][m].mean())
        x = sum(daily) / len(daily)
        z = x / x.std()
        for col in x.columns:
            kind = col.split()[0]
            sr, se = _sharpe(x[col])
            # Each series over its own sd, so the mean of the paired difference is the Sharpe difference.
            vs, vh = z[col] - z[kind], z[col] - z["hold"]
            cum = np.log1p(x[col]).cumsum()
            rows.append(
                {
                    "fold": k,
                    "variant": col,
                    "sharpe": sr,
                    "se": se,
                    "d_unscaled": vs.mean() * np.sqrt(365.25),
                    "d_se": blocked(vs, WEEK)["se"] * np.sqrt(365.25),
                    "d_hold": vh.mean() * np.sqrt(365.25),
                    "dh_se": blocked(vh, WEEK)["se"] * np.sqrt(365.25),
                    "log_year": np.log1p(x[col]).sum() / years,
                    "mdd_pct": 100 * (1 - np.exp(-(cum.cummax().clip(lower=0) - cum).max())),
                    "turnover": pd.concat(turnover, axis=1).loc[col].mean(),
                    "trades": pd.concat(trades, axis=1).loc[col].mean(),
                    "exposure": pd.concat(expo, axis=1).loc[col].mean(),
                }
            )
    return pd.DataFrame(rows)


def base(period: str = "dev", fee: float = FEE, symbols=TRADABLE) -> tuple[pd.DataFrame, dict]:
    """Route 5 on `symbols`: the table of `summarise`, and the books it was read from."""
    books = {s: slow(s, period, fee) for s in symbols}
    return summarise(books, period), books


def verdicts(table: pd.DataFrame) -> list[str]:
    """The card's criterion for each scaled rule, and the scaled hold beside it as the control."""
    out = []
    for w in WMAX:
        for d in DELTA:
            col = name("rule", w, d)
            t = table[table.variant == col].set_index("fold")
            h = table[table.variant == name("hold", w, d)].set_index("fold")
            ok = bool(((t.d_unscaled > 0) & (t.d_hold > 0)).all())
            out.append(
                f"{col}: Sharpe above the rule in folds {list(t.index[t.d_unscaled > 0])}, above hold in "
                f"{list(t.index[t.d_hold > 0])} -> {'PASS' if ok else 'FAIL'}; scaling hold moves its Sharpe "
                + " / ".join(f"{v:+.2f}" for v in h.d_unscaled)
                + " against the rule's "
                + " / ".join(f"{v:+.2f}" for v in t.d_unscaled)
            )
    return out


# ---------------------------------------------------------------- route 6: the timing


def fast(symbol: str, pred: pd.Series | None, since: pd.Timestamp) -> pd.DataFrame:
    """One asset's 15m log close and fast columns from `since`, every column z-scored against its own
    trailing month (`detect._zscore`), unsigned. Computed on the whole store and cut after, so every
    window is warm at `since`; v2's level only where v2 was tested, the futures columns only on
    `FUTURES` and where `data.futures` reaches."""
    bars = load(symbol, FAST)
    close = bars.close
    out = legs.exhaustion(bars, FAST_WINDOW).apply(detect._zscore)
    out["rsi12"] = detect._zscore(RSIIndicator(close, window=FAST_WINDOW).rsi() / 50 - 1)
    if pred is not None:
        out["v2"] = detect._zscore(pred).reindex(close.index)
    if symbol in FUTURES:
        for k in (12, 24):
            b = detect.behind_the_move(symbol, close, k)
            out[f"oi{k}"] = detect._zscore(b.move * b.oi_z).reindex(close.index)
        out["book5_4"] = detect._zscore(detect.futures_columns(symbol, close)["book5_4"]).reindex(close.index)
    out["lp"] = np.log(close)
    return out.loc[since:]


def signs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rank IC of every column with the 24- and 48-bar forward log return on dev's rows, the mean over
    the assets that have it, and the sign that orients it: the only thing read off the data."""
    rows = {}
    for sym, f in frames.items():
        dev = strategy.fold_in(f.index, "dev") > 0
        for c in f.columns.drop("lp"):
            for h in HORIZONS:
                ic = spearman(f[c][dev], (f.lp.shift(-h) - f.lp)[dev])
                if np.isfinite(ic):
                    rows.setdefault((c, h), []).append(ic)
    t = pd.Series({k: np.mean(v) for k, v in rows.items()}).unstack()
    t.columns = [f"ic{h}" for h in t.columns]
    t["sign"] = np.where(t.mean(axis=1) >= 0, 1.0, -1.0)
    return t


def scores(f: pd.DataFrame, sign: pd.Series, period: str) -> pd.DataFrame:
    """Each signed column alone, and `score`: the equal-weighted mean of the signed columns of the
    period's composite that the asset has at the bar (`composite`), z-scored again against its
    trailing month so that a threshold is in its standard deviations."""
    cols = [c for c in sign.index if c in f.columns]
    signed = f[cols] * sign[cols]
    keep = [c for c in composite(period) if c in cols]
    return signed.assign(score=detect._zscore(signed[keep].mean(axis=1)))


def composite(period: str) -> list[str]:
    """The composite's columns: exhaustion everywhere; v2's level, open interest behind the move and
    the book where v2 was tested (the futures only on `FUTURES`); the RSI at 12 in v2's place on 2021,
    which reads no v2 and no futures, so that the confirmation is of what the history before v2 had."""
    return list(legs.EXHAUSTION) + (["rsi12"] if period == "2021" else ["v2", "oi12", "oi24", "book5_4"])


def anchor(index: pd.DatetimeIndex, when: pd.DatetimeIndex) -> np.ndarray:
    """The 15m bar that closes at each decision time, or the last one closed before it: the bar whose
    close is the immediate execution, never a bar that closes after the decision is taken."""
    return np.searchsorted(index + BAR, when, side="right") - 1


def execute(lp: np.ndarray, score: np.ndarray, t0: int, n: int, d: float, rng, bs=BS, draws: int = DRAWS):
    """`(gain at each b, at a random bar, at the window's last bar, the bar each b executed on)` of one
    execution against executing at once, gains in log units.

    The window is bars `t0` .. `t0 + n - 1`; `t0` is the decision's own bar, so a score already past
    `b` there executes at once. With `b`, the first bar where `d * score >= b` (a NaN never is), else
    the window's last; the gain is `d * (lp[t0] - lp[tau])`, a cheaper buy or a dearer sale. The two
    controls read no score: the random bar is what waiting costs on average, the last bar what a b the
    score never reaches would earn, a plain delay of the base.
    """
    s = d * score[t0 : t0 + n]
    taus = [hit[0] if len(hit := np.flatnonzero(s >= b)) else n - 1 for b in bs]
    gains = d * (lp[t0] - lp[t0 + np.array(taus)])
    rnd = float(np.mean(d * (lp[t0] - lp[t0 + rng.integers(0, n, draws)])))
    return gains, rnd, d * (lp[t0] - lp[t0 + n - 1]), np.array(taus)


def decisions(book: dict, col: str) -> pd.DataFrame:
    """The trades of one variant of the base: decision time (the 4h bar's close), side and size."""
    dw = book["traded"][col]
    dw = dw[dw != 0]
    return pd.DataFrame({"when": dw.index + pd.Timedelta(TF), "d": np.sign(dw.to_numpy()), "size": dw.abs().to_numpy()})


def executions(books: dict, frames: dict, sign: pd.Series, period: str, seed: int = 0) -> pd.DataFrame:
    """Every decision of the unscaled rule (`set` rule) and of the scaled rule's rebalances (`set`
    scaled), each timed by every score at every N and b, with three controls that read no score: the
    random bar, the last bar, and `lag`, the delays the same b took on the asset's other decisions
    drawn at random. `lag` is the null of the score's information: a regime in which waiting pays
    rewards any b that waits long, and `lag` keeps how long b waits and drops when. One row per
    execution and score; the window stops short of the next decision of the same set, which has to
    find the first one done. Decisions whose score is NaN at their bar are dropped."""
    rng = np.random.default_rng(seed)
    rows = []
    for sym, book in books.items():
        f = frames[sym]
        sc = scores(f, sign, period)
        lp = f.lp.to_numpy()
        for kind, col in (("rule", "rule"), ("scaled", name("rule", *SCALED))):
            dec = decisions(book, col)
            t0 = anchor(f.index, pd.DatetimeIndex(dec.when))
            room = np.append(np.diff(t0), len(lp) - t0[-1]) if len(t0) else t0
            for c in sc.columns:
                s = sc[c].to_numpy()
                live = np.flatnonzero(np.isfinite(s[t0]) & (t0 >= 0))
                for n in NS:
                    m = np.minimum(n, room[live]).astype(int)
                    got = [execute(lp, s, t0[i], m[k], dec.d.iat[i], rng) for k, i in enumerate(live)]
                    taus = np.array([g[3] for g in got]).reshape(len(got), len(BS))
                    for k, (i, (gains, rnd, end, _)) in enumerate(zip(live, got)):
                        pool = np.delete(taus, k, axis=0) if len(taus) > 1 else taus
                        lag = np.minimum(pool[rng.integers(0, len(pool), DRAWS)], m[k] - 1)
                        lag = (dec.d.iat[i] * (lp[t0[i]] - lp[t0[i] + lag])).mean(axis=0)
                        size = dec["size"].iat[i]
                        rows.append([kind, c, n, sym, dec.when.iat[i], size, *(np.r_[gains, rnd, end, lag] * 1e4)])
    out = pd.DataFrame(rows, columns=["set", "score", "N", "symbol", "when", "size", *GAINS])
    out["fold"] = strategy.fold_in(pd.DatetimeIndex(out.when) - pd.Timedelta(TF), period)
    return out


def tabulate(ex: pd.DataFrame, period: str, net: pd.Series) -> pd.DataFrame:
    """Per set, score, N and fold: the mean gain an execution in bp for every b and the random bar;
    executions; and the value a year per sleeve (the gain weighted by the size traded), beside the
    base's net log a year, `net`.

    The mean is over executions weighted by the size traded, which is what the money sees: a
    rebalance of 0.14 of a sleeve after a spike is not an entry, and unweighted three of them (FIL and
    UNI in November 2025, +330 to +1,640 bp) moved the scaled set's fold 2 by 28 bp. Executions also
    cluster: one market move takes the RSI of most pairs across 0.3 within hours, and on dev the
    clustered ones time worse than the isolated ones, so `metrics.blocked` on the executions
    themselves, a mean of block means, read +16 bp where the money made +0.6 (rule, N = 48, b = 1, fold
    1). So the mean is the ratio sum / size and only the error comes from the clock: `blocked` on each
    block of N bars' residual sum (S_b - mean * W_b), scaled back to one execution, the ratio
    estimator's error, in which a cluster counts as one draw.
    """
    e = strategy.edges(period)
    first = 3 if period == "holdout" else 1
    ex = ex.assign(**{v: ex[v[:-4]] - ex[f"lag{v[1:-4]}"] for v in VS})
    rows = []
    for (kind, c, n, k), g in ex.groupby(["set", "score", "N", "fold"]):
        years = (e[k - first + 1] - e[k - first]) / YEAR
        row = {"set": kind, "score": c, "N": n, "fold": k, "n": len(g), "assets": g.symbol.nunique()}
        for col in GAINS + VS:
            x = g["size"] * g[col]
            mean = x.sum() / g["size"].sum()
            block = (x - mean * g["size"]).groupby(pd.DatetimeIndex(g.when).floor(n * BAR)).sum()
            row[col], row[f"{col}_se"] = mean, blocked(block, n * BAR)["se"] * len(block) / g["size"].sum()
            row[f"{col}_year"] = x.sum() / 1e4 / years / g.symbol.nunique()
        row["base_net"] = net.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def chosen(dev: pd.DataFrame) -> pd.Series:
    """The b of each (set, score, N) with the highest gain over dev's executions, pooled and weighted
    by the size traded as in `tabulate`."""
    keys = [dev.set, dev.score, dev.N]
    gain = dev[[f"b{b:g}" for b in BS]].mul(dev["size"], axis=0).groupby(keys).sum()
    return gain.div(dev["size"].groupby(keys).sum(), axis=0).idxmax(axis=1)


def timing(period: str = "dev", fee: float = FEE, symbols=TRADABLE):
    """Route 6: `(table, table on dev, chosen b, signs, executions)`. Dev is always computed, because
    the signs and b are read there and nowhere else."""
    e, d = strategy.edges(period), strategy.edges("dev")
    since = min(e[0], d[0]) - pd.Timedelta("1D")
    pred = strategy.load(assets=symbols)[0]
    have = set(pred.index.get_level_values(1))
    frames = {s: fast(s, pred.xs(s, level=1) if s in have else None, since) for s in symbols}
    del pred
    ic = signs(frames)
    out = {}
    for p in dict.fromkeys(("dev", period)):
        books = {s: slow(s, p, fee) for s in symbols}
        net = summarise(books, p).query("variant == 'rule'").set_index("fold").log_year
        ex = executions(books, frames, ic["sign"], p)
        out[p] = (tabulate(ex, p, net), ex)
    best = chosen(out["dev"][1])
    return out[period][0], out["dev"][0], best, ic, out[period][1]


def timing_verdicts(table: pd.DataFrame, best: pd.Series) -> list[str]:
    """The card's criterion on the composite score, each set and N, at the b dev chose: above
    immediate in every fold, the random bar inside two errors of zero in every fold."""
    out = []
    for (kind, n), g in table[table.score == "score"].groupby(["set", "N"]):
        b = best[(kind, "score", n)]
        up = bool((g[b] > 0).all())
        noise = bool((g.random.abs() < 2 * g.random_se).all())
        own = g[b + "-lag"], g[b + "-lag_se"]
        out.append(
            f"{kind} N={n} b={b[1:]}: gain {' / '.join(f'{v:+.1f}' for v in g[b])} bp by fold, random "
            f"{' / '.join(f'{v:+.1f}' for v in g.random)}, last bar {' / '.join(f'{v:+.1f}' for v in g.end)}, "
            f"less its delays at random {' / '.join(f'{v:+.1f} ({e:.1f})' for v, e in zip(*own))} -> "
            + ("PASS" if up and noise else "FAIL")
            + ("" if noise else " (random bar outside the noise)")
        )
    return out


# ---------------------------------------------------------------- power


def _garch(rng, shape, sigma: float, a: float, b: float, burn: int = 300):
    """A GARCH(1,1) path of t(4) shocks, `(returns, conditional sd)`, each `shape`, after `burn` bars."""
    w = sigma**2 * (1 - a - b)
    s2 = np.full(shape[1:], sigma**2)
    r, sd = np.empty(shape), np.empty(shape)
    for t in range(-burn, shape[0]):
        eps = rng.standard_t(4, shape[1:]) / np.sqrt(2)
        x = np.sqrt(s2) * eps
        if t >= 0:
            r[t], sd[t] = x, np.sqrt(s2)
        s2 = w + a * x**2 + b * s2
    return r, sd


def power_base(days: list[int], reps: int = 1000, seed: int = 7, sigma4: float = 0.012) -> pd.DataFrame:
    """Detection rate of route 5's criterion, Sharpe of the scaled sleeve above the unscaled in every
    fold, at the folds' real lengths. One series stands for the portfolio, always in (so the rule is
    hold and the second half of the criterion is the first), w_max 1, band 0.1, sigma* the median of
    the year before. The 4h returns are a GARCH(0.05, 0.94) of t(4) shocks at `sigma4`, plus a mean
    `kappa * sd_t ** gamma` set for an annual Sharpe of 0.5 when held: gamma 2 is the risk-return world
    where scaling hurts, 0 the Moreira-Muir world where the mean does not rise with the volatility, and
    below 0 the mean falls when the volatility rises, the effect made larger. The true difference is the
    mean over reps of each rep's difference over the whole period."""
    rng = np.random.default_rng(seed)
    warm = (365 + 30) * PER_DAY
    n = warm + sum(days) * PER_DAY
    rows = []
    for gamma in (2.0, 1.0, 0.0, -1.0, -2.0):
        eps, sd = _garch(rng, (n, reps), sigma4, 0.05, 0.94)
        kappa = 0.5 * sigma4 / np.sqrt(365.25 * PER_DAY) / np.mean(sd**gamma)
        r = eps + kappa * sd**gamma
        close = np.exp(np.cumsum(r, axis=0))
        s = pd.DataFrame(close).pipe(lambda c: np.sqrt((np.log(c).diff() ** 2).ewm(span=SPAN, min_periods=SPAN).mean()))
        star = s.iloc[warm - 365 * PER_DAY : warm].median().to_numpy()
        target = np.minimum(star / s.to_numpy(), 1.0)
        R = np.expm1(r)
        ones = np.ones_like(target)
        net = sleeve(R[warm - 1 :], ones[warm - 1 :], target[warm - 1 :], np.full(reps, 0.1))[0]
        held = R[warm:]
        wins, at = np.ones(reps, bool), 0
        for k in days:
            m = slice(at, at + k * PER_DAY)
            xs = (1 + net[m]).reshape(k, PER_DAY, reps).prod(axis=1) - 1
            xh = (1 + held[m]).reshape(k, PER_DAY, reps).prod(axis=1) - 1
            wins &= xs.mean(0) / xs.std(0) > xh.mean(0) / xh.std(0)
            at += k * PER_DAY
        daily = [(1 + v).reshape(-1, PER_DAY, reps).prod(axis=1) - 1 for v in (net, held)]
        sr = [x.mean(0) / x.std(0) * np.sqrt(365.25) for x in daily]
        rows.append(
            {
                "gamma": gamma,
                "true_d_sharpe": np.mean(sr[0] - sr[1]),
                "hold_sharpe": sr[1].mean(),
                "detected": wins.mean(),
            }
        )
    return pd.DataFrame(rows)


# The construction of `sim_timing.py` (false_alarms.html, appendix): 15m bars, a GARCH(0.08, 0.90) of
# t(4) shocks at 0.4% a bar, an AR(1) signal of half-life 24 bars, and a drift that reads it clipped at
# one standard deviation ("bulk"), scaled for a rank IC rho with the 24-bar forward return.
SIM_SIGMA, SIM_H, SIM_HL = 0.004, 24, 24
_BULK = np.clip(np.random.default_rng(0).standard_normal(1_000_000), -1, 1)
BULK_SD = _BULK.std()
BULK_CORR = np.corrcoef(np.random.default_rng(0).standard_normal(1_000_000), _BULK)[0, 1]


def _scale(rho: float, phi: float) -> float:
    """`sim_timing.scale`: k such that corr(m_t, sum of the next H returns) = rho with var(m) = 1."""
    j = np.arange(SIM_H)
    S = (phi**j).sum()
    V = (phi ** np.abs(j[:, None] - j[None, :])).sum()
    return rho * SIM_SIGMA * np.sqrt(SIM_H) / np.sqrt(S**2 - rho**2 * V)


def power_timing(counts: list[int], reps: int = 1000, seed: int = 13) -> pd.DataFrame:
    """Detection rate of route 6's criterion at the real executions per dev fold, `counts`: b chosen
    among `BS` on both folds pooled, then the gain above zero in every fold (the card), and the same
    with a pooled t of 2 as well. Each execution is a fresh window with the signal drawn from its
    stationary law, the direction at random and independent of it, as in the document's simulation,
    so the random bar is zero by construction and not simulated."""
    rng = np.random.default_rng(seed)
    phi = 0.5 ** (1 / SIM_HL)
    total, nmax = sum(counts), max(NS)
    edges = np.cumsum([0] + counts)
    chunk = max(1, 2_000_000 // (nmax * total))  # reps at a time: a few arrays of 2M floats, not 20M
    rows = []
    for rho in (0.0, 0.02, 0.05):
        k = _scale(min(rho / BULK_CORR, 0.99), phi)
        got = {n: [] for n in NS}  # per N, the chosen b's gains, (reps, executions)
        for start in range(0, reps, chunk):
            c = min(chunk, reps - start)
            r = _garch(rng, (nmax, c, total), SIM_SIGMA, 0.08, 0.90)[0]
            m = np.empty((nmax, c, total))
            m[0] = rng.standard_normal((c, total))
            for t in range(1, nmax):
                m[t] = phi * m[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal((c, total))
            r[1:] += k * np.clip(m[:-1], -1, 1) / BULK_SD
            r[0] = 0.0
            p = np.cumsum(r, axis=0)  # p[0] = 0: the price at the decision
            d = rng.choice([-1.0, 1.0], (c, total))
            for n in NS:
                gains = []
                for b in BS:
                    hit = d * m[:n] >= b
                    tau = np.where(hit.any(0), hit.argmax(0), n - 1)
                    gains.append(-d * np.take_along_axis(p, tau[None], 0)[0] * 1e4)
                g = np.stack(gains)  # (b, reps, executions)
                got[n].append(g[g.mean(2).argmax(0), np.arange(c)])
        for n in NS:
            g = np.concatenate(got[n])
            folds = np.stack([g[:, a:z].mean(1) for a, z in zip(edges, edges[1:])])
            every = (folds > 0).all(0)
            t = g.mean(1) / (g.std(1) / np.sqrt(total))
            rows.append(
                {
                    "rho": rho,
                    "N": n,
                    "mean_bp": g.mean(),
                    "se_bp": g.std(1).mean() / np.sqrt(total),
                    "every_fold": every.mean(),
                    "and_t2": (every & (t > 2)).mean(),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- self-check


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    # The sleeve's accounting. Held at 1 with no fee it earns the asset; held at 0.5 and never
    # rebalanced it is half asset, half cash, whatever the path; a trade pays the fee on |dw| alone.
    R = np.expm1(rng.normal(0, 0.01, 500))
    ones = np.ones((500, 1))
    net, traded, held = sleeve(R, ones, ones, np.array([0.1]), 0.0)
    assert np.allclose(net[:, 0], R[1:]) and not traded.any()
    net = sleeve(R, ones, 0.5 * ones, np.array([1.0]), 0.0)[0]
    assert np.isclose(np.prod(1 + net), 0.5 * np.prod(1 + R[1:]) + 0.5)
    state = np.r_[np.zeros(10), np.ones(490)][:, None]
    net, traded, held = sleeve(np.zeros(500), state, 0.8 * state, np.array([0.1]), 0.001)
    assert np.isclose(traded.sum(), 0.8) and np.isclose(np.prod(1 + net), 1 - 0.0008)
    # A band keeps a drifting weight until it leaves it, and an exit always trades.
    up = np.full(500, 0.01)
    traded = sleeve(up, ones, 0.5 * ones, np.array([0.1]), 0.0)[1]
    assert 0 < (traded != 0).sum() < 30 and (traded <= 0).all()
    small = np.array([[0.05], [0.05], [0.0]])
    exits = sleeve(np.zeros(3), np.array([[1.0], [1.0], [0.0]]), small, np.array([0.1]))[1]
    assert exits[-1, 0] == -0.05, "a weight inside the band still closes when the rule exits"

    # sigma-hat and the rule at a bar are known at its close: a future that changes leaves them as they were.
    t = pd.date_range("2024-01-01", periods=3000, freq="4h", tz="UTC")
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(t)))), index=t)
    other = close.copy()
    other.iloc[2000:] *= np.exp(np.cumsum(rng.normal(0.01, 0.03, 1000)))
    assert sigma(close)[:2000].equals(sigma(other)[:2000]) and rule(close)[:2000].equals(rule(other)[:2000])
    assert sigma(close).iloc[2000] != sigma(other).iloc[2000], "the bar's own return is in its sigma-hat"

    # The execution bar: the 15m bar that closes at the 4h close carries the 4h close; a missing bar
    # falls back to the last one closed before the decision, never to one that closes after it.
    t15 = pd.date_range("2024-01-01", periods=4000, freq="15min", tz="UTC")
    c15 = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.003, len(t15)))), index=t15)
    c4 = c15.resample("4h").last()
    when = c4.index + pd.Timedelta("4h")
    at = anchor(t15, when)
    ok = at < len(t15)
    assert np.allclose(c15.to_numpy()[at[ok]], c4.to_numpy()[ok])
    holed = t15.delete(15)  # the bar closing at 04:00 is missing
    assert holed[anchor(holed, when[:1])[0]] + BAR <= when[0]
    # Execution never comes before the decision: a window of one bar gains exactly nothing.
    lp = np.log(c15.to_numpy())
    noise = rng.normal(size=len(lp))
    assert all((execute(lp, noise, i, 1, 1.0, rng)[0] == 0).all() for i in range(100, 200))

    # The null: on a martingale with a score that knows nothing, waiting gains nothing on average,
    # signal or random. Planted: a score that reads the next 8 bars, the gain is positive and large.
    walk = np.cumsum(rng.normal(0, 0.003, 400_000))
    ar = np.zeros(len(walk))
    for i in range(1, len(walk)):
        ar[i] = 0.97 * ar[i - 1] + np.sqrt(1 - 0.97**2) * rng.standard_normal()
    future = np.r_[walk[8:] - walk[:-8], np.zeros(8)] / (0.003 * np.sqrt(8))
    starts = np.arange(1000, len(walk) - 1000, 97)
    sides = rng.choice([-1.0, 1.0], len(starts))
    for score, planted in ((ar, False), (future, True)):
        got = np.array([np.r_[execute(walk, score, i, 48, s, rng)[:3]] for i, s in zip(starts, sides)])
        mean, se = got.mean(0) * 1e4, got.std(0) / np.sqrt(len(got)) * 1e4
        if planted:
            assert (mean[: len(BS)] > 5 * se[: len(BS)]).all() and (np.abs(mean[len(BS) :]) < 4 * se[len(BS) :]).all()
        else:
            assert (np.abs(mean) < 4 * se).all(), (mean, se)

    # The fast score at a bar is known at its close: z-scores and the composite ignore what comes after.
    f = pd.DataFrame(rng.normal(size=(5000, 3)), index=t15[:4000].append(t15[:1000] + pd.Timedelta("100D")))
    f.columns = ["a", "b", "v2"]
    f = f.apply(detect._zscore).assign(lp=0.0)
    sign = pd.Series({"a": 1.0, "b": -1.0, "v2": 1.0})
    g = f.copy()
    g.iloc[3000:, :3] += 5
    assert scores(f, sign, "dev").iloc[:3000].equals(scores(g, sign, "dev").iloc[:3000])
    s = scores(f, sign, "dev")
    assert np.allclose(s.b, -f.b, equal_nan=True) and s.score.notna().sum() > 3000
    print("ok - sleeve, causality, execution bar, null and planted timing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", action="store_true", help="route 5: the 4h rule, scaled and not, against hold")
    ap.add_argument("--timing", action="store_true", help="route 6: the base's executions timed by the fast score")
    ap.add_argument("--power", action="store_true", help="Monte Carlo detection rates at the store's sample sizes")
    ap.add_argument("--period", choices=strategy.PERIODS, default="dev")
    ap.add_argument("--fee", type=float, default=FEE)
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    _selfcheck()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    if args.base:
        table, books = base(args.period, args.fee)
        print(f"route 5, {args.period}, {len(books)} pairs, fee {args.fee * 100:.2f}% a side")
        print("sigma* (4h):", " ".join(f"{s} {b['star'] * 100:.2f}%" for s, b in books.items()))
        print(table.round(3).to_string(index=False))
        print("\n".join(verdicts(table)))
    if args.timing:
        table, dev, best, ic, ex = timing(args.period, args.fee)
        print(f"route 6, {args.period}; signs from dev's rank IC with the forward return")
        print(ic.round(4).to_string())
        print("\nb chosen on dev:")
        print(best.unstack("N").to_string())
        shown = ["set", "score", "N", "fold", "n", "assets"]
        shown += [c for g in GAINS for c in (g, f"{g}_se")]
        comp = table[table.score == "score"]
        print("\nthe composite score, gain against immediate, bp an execution")
        print(comp[shown].round(1).to_string(index=False))
        print("\neach b less its own delays drawn at random (lag), paired: the score's part")
        print(comp[["set", "N", "fold"] + [c for v in VS for c in (v, f"{v}_se")]].round(1).to_string(index=False))
        print("\nvalue a year per sleeve, log, against the base's net")
        print(comp[["set", "N", "fold"] + [f"{g}_year" for g in GAINS] + ["base_net"]].round(4).to_string(index=False))
        print("\neach column alone, at its dev b")
        alone = table[table.score != "score"].copy()
        alone["b"] = [best.get((s, c, n), "b0.75") for s, c, n in zip(alone.set, alone.score, alone.N)]
        alone["gain"] = [r[r.b] for _, r in alone.iterrows()]
        alone["gain_se"] = [r[r.b + "_se"] for _, r in alone.iterrows()]
        print(
            alone[alone.set == "rule"]
            .pivot_table(index=["score", "N", "b"], columns="fold", values=["gain", "gain_se", "n"])
            .round(1)
            .to_string()
        )
        print("\n" + "\n".join(timing_verdicts(table, best)))
    if args.power:
        e = strategy.edges("dev")
        days = [int((b - a) / pd.Timedelta("1D")) for a, b in zip(e[:-1], e[1:])]
        print(f"route 5 power: dev folds of {days} days, {args.reps} reps")
        print(power_base(days, args.reps).round(3).to_string(index=False))
        e = strategy.edges("2021")
        days = [int((b - a) / pd.Timedelta("1D")) for a, b in zip(e[:-1], e[1:])]
        print(f"\nroute 5 power: 2021 folds of {days} days")
        print(power_base(days, args.reps).round(3).to_string(index=False))
        books = {s: slow(s, "dev") for s in TRADABLE}
        for kind, col in (("rule", "rule"), ("scaled", name("rule", *SCALED))):
            counts = [
                int(sum(((b["traded"][col] != 0) & (b["fold"] == k)).sum() for b in books.values())) for k in (1, 2)
            ]
            print(f"\nroute 6 power, {kind} set: {counts} executions in dev's folds, {args.reps} reps")
            print(power_timing(counts, args.reps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
