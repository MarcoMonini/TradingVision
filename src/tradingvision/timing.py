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

**Simplifications, stated.** A sleeve's weight drifts with its asset between trades and the sleeves
are rebalanced to equal weight daily for free. A period starts in the position the rule holds,
without paying to enter it.

    uv run python -m tradingvision.timing --base [--period dev|holdout|2021] [--fee 0.001]
    uv run python -m tradingvision.timing --power [--reps 1000]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from tradingvision import strategy
from tradingvision.data.binance import TRADABLE, load
from tradingvision.metrics import blocked
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
    print("ok - sleeve, sigma-hat and the rule known at the close")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", action="store_true", help="route 5: the 4h rule, scaled and not, against hold")
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
    if args.power:
        e = strategy.edges("dev")
        days = [int((b - a) / pd.Timedelta("1D")) for a, b in zip(e[:-1], e[1:])]
        print(f"route 5 power: dev folds of {days} days, {args.reps} reps")
        print(power_base(days, args.reps).round(3).to_string(index=False))
        e = strategy.edges("2021")
        days = [int((b - a) / pd.Timedelta("1D")) for a, b in zip(e[:-1], e[1:])]
        print(f"\nroute 5 power: 2021 folds of {days} days")
        print(power_base(days, args.reps).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
