"""Route 7: the spot taker flow decomposed into information and liquidity (HANDOFF §19, `false_alarms.html` #taker).

**The question.** Every trade has an aggressor, and the volume takers bought less the volume they
sold is the signed flow. Hasbrouck (1991) splits the flow's price impact into a permanent part, the
information it carries, and a transient one; Llorente, Michaely, Saar and Wang (2002) find that moves
made by informed trades continue and moves made for liquidity revert. The part of a leg the flow's
permanent impact does not explain is, by construction, the part that could come back. The prior was
low: the futures' taker flow along the leg, raw, told true alarms from false ones at AUC 0.53-0.54
(`detect --futures`, three assets, sixteen months).

**The data.** The spot dumps carry `taker_buy_base` and `taker_buy_quote`, which the store keeps
from 2026-10-07 (`binance.TAKER`): the 15 SYMBOLS from their listing (2017-2020) to 2026-10-06. The
signed flow of a 15m bar is f = (2 taker_buy_quote - quote_volume) / quote_volume, in [-1, 1].

**The VAR.** Per asset, 8 lags of (r, f) by least squares, Hasbrouck's ordering: the flow first, so
the return's equation holds the bar's own flow. Fitted per fold on the 365 days before the fold's
first bar and held fixed through it: strictly before what it is applied to, and per fold rather than
per period because the 2021 period is 4.4 years long. The permanent impact theta is the return's
cumulative response to a unit flow innovation at 96 bars; it settles within 1% of that by bar 7-8 at
the median over assets and folds, 22 at the worst. Measured, by fold, the median over the fifteen:
a one-sd flow innovation moves the price for good by 16.2 / 17.0 bp on development (9.9-22.7 over
the assets), 15.6 / 12.5 on the hold-out, 20.9 / 23.5 / 16.9 / 12.6 on 2021; the informed part
theta v explains 12% of the 15m return variance on development, 8-10% on the hold-out, 7-11% on 2021.

**The decomposition.** The informed part of a bar's return is theta times its flow innovation, the
flow less what the VAR expected from the lags. The leg runs from the pivot a live reader held at the
bar (`legs.confirmed` at 12, v2's window) to the bar; M is its log move and I the sum of the informed
parts over it. The transient share is 1 - I/M, **in units of the leg's move**, chosen before looking:
in units of sigma the transient part is share times M/sigma and carries the leg's size, and at an
extreme the size is what v2 and the RSI already read (the extremes keep going, §17); in the move's
units it reads the leg's composition, which is the one thing the flow adds. Measured afterwards on
2021's RSI extremes: share and size rank-correlate 0.14, the size alone has an IC of +0.039, and the
share net of the size keeps +0.019 of its +0.022, positive in all four folds. The raw flow, the mean
f over the leg signed into it, is the context the card asks for.

**Events.** `v2`: the first bar of each excursion of v2's raw output past the asset's 90th
percentile of |v2| on development (v2 has no out-of-sample bar before it, so the threshold reads the
period's input, never its outcome): 4,339 / 3,903 on folds 1-2, 3,286 / 2,772 on 3-4. `rsi 12`: the
same on an RSI at 12, its threshold the 90th percentile over the year before the fold: 5,815 /
5,774 on development, 17,457-21,653 a fold on 2021. `every bar`: context.

**The IC.** Against the leg, positive = reversal. Ranks per asset and fold, the products' sample mean
over the fold's rows, the error from h-bar clock blocks (`pooled`). Not the mean of block means
(`metrics.blocked`): the extremes cluster on the same bars, and that estimator moved v2's
development IC at 24 bars to +0.028 / -0.032 from the +0.001 / +0.003 the rows hold.

**What it found** (all 15, IC +- blocked error; `python -m tradingvision.flow --store data/taker
--period dev holdout 2021`, 42 s):

    transient share         h = 12               h = 24               h = 48
    v2, dev (f1 / f2)       +0.008 / -0.002      +0.001 / +0.003      +0.014 / +0.006       +- 0.019-0.025
    v2, hold-out (f3 / f4)  -0.026 / +0.026      -0.065 / +0.025      -0.032 / +0.023       +- 0.021-0.027
    rsi 12, dev             +0.013 / -0.006      +0.018 / +0.001      +0.002 / +0.013       +- 0.017-0.023
    rsi 12, 2021 (mean)     +0.021 +- 0.005      +0.023 +- 0.005      +0.022 +- 0.006       every fold > 0
    every bar, dev          +0.010 +- 0.005      +0.012 +- 0.007      +0.011 +- 0.009
    every bar, 2021         +0.005 +- 0.002      +0.007 +- 0.003      +0.009 +- 0.003

2021's folds at 48 bars: +0.035 / +0.019 / +0.019 / +0.015. The raw flow is nothing at the events,
-0.027 to +0.049 by fold and -0.018 to +0.030 by period; on every bar it leans to continuation, -0.013
to -0.017 on development, and is zero on 2021. TRADABLE alone reads the same (v2 on development
+0.009 / +0.002 at 48 bars, the RSI on 2021 +0.039 / +0.022 / +0.021 / +0.017).

**The verdict: it fails, on every horizon and both universes.** rho_min = 2 FEE / (0.8 sigma sqrt h),
sigma 0.461% (the fifteen's median 15m dispersion on development): 0.156 / 0.111 / 0.078 at 12 / 24 /
48 bars. At v2's extremes on development the share's IC is +0.010 at best, a tenth of rho_min and
inside its error, and the hold-out flips its sign between folds. What there is, is small and stable
on 2021: at the RSI's extremes the share predicts the reversal in every fold, +0.022 at 48 bars
(t ~ 4), not the leg's size. That is 0.28 rho_min, about 0.022 x 0.8 sigma sqrt 48 = 5.6 bp a trade
against a 20 bp round trip; and v2's extremes on development, where the card decides, do not show it.

**The criterion can see what it asks for, and nothing else** (`--power`, 600 seeded reps a cell,
3 min, Spearman of normal pairs at each fold's events and at its effective size 1 / se^2, which
the blocks cut to 35-65% of the events). At effects of 0 and 0.5 rho_min it passes 0.0% of the time
at every horizon; at 1 rho_min 23-29% (two independent means each about half the time past their
threshold); at 2 rho_min 100%. The 2021 effect, 0.28 rho_min, is one the criterion is built not to
promote: it fails on its size, not on a want of data.

    uv run python -m tradingvision.flow --store data/taker [--period dev holdout 2021] [--symbols ...]
    uv run python -m tradingvision.flow --power
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import legs, metrics, strategy
from tradingvision.data.binance import STORE, SYMBOLS, TAKER, TRADABLE
from tradingvision.data.binance import load as candles
from tradingvision.oracle import FEE

LAGS = 8  # Hasbrouck's VAR, on (return, flow), as the plan states it
FIT = pd.Timedelta("365D")  # each fold's VAR is fitted on the year of bars before the fold's first
MIN_FIT = 96 * 30  # a pair listed less than a month before the fold has no VAR there
IRF = 96  # bars the cumulative response is summed over; it settles long before (see `settled`)
HORIZONS = (12, 24, 48)
EXTREME = 0.9  # the 10% most extreme |v2| or |RSI|
BAR = strategy.BAR
COLUMNS = ("share", "flow")


def bars(symbol: str, store: Path = STORE) -> pd.DataFrame:
    """`close`, the 15m log return `r` and the signed flow `f` of one pair, on a complete 15m grid.

    f = (2 taker_buy_quote - quote_volume) / quote_volume: the share of the bar's volume bought by
    takers minus the share sold, in [-1, 1]. A bar with no trade, or none at all in the store, has
    a flat close, r = 0 and f = 0, so a lag counts bars of clock and not bars of data.
    """
    df = candles(symbol, "15m", store=store)
    if missing := [c for c in TAKER if c not in df]:
        raise SystemExit(f"{store / f'{symbol}USDT-5m.parquet'} has no {missing}: pass --store with a re-download")
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=BAR, name=df.index.name))
    close, qv = df.close.ffill(), df.quote_volume
    f = ((2 * df.taker_buy_quote - qv) / qv).where(qv > 0).fillna(0.0)
    return pd.DataFrame({"close": close, "r": np.log(close).diff().fillna(0.0), "f": f.astype("float64")})


def design(r: np.ndarray, f: np.ndarray, p: int = LAGS) -> np.ndarray:
    """`[1, r_{t-1}..r_{t-p}, f_{t-1}..f_{t-p}]` at each bar t, NaN where a lag reaches before the series."""
    X = np.full((len(r), 2 * p + 1), np.nan)
    X[:, 0] = 1.0
    for i in range(1, p + 1):
        X[i:, i], X[i:, p + i] = r[:-i], f[:-i]
    return X


def response(flow: np.ndarray, ret: np.ndarray, k: int = IRF, p: int = LAGS) -> np.ndarray:
    """The cumulative response of the return to a unit flow innovation at bar 0, bars 0 to `k`.

    Hasbrouck's ordering: the flow moves first, the return answers within the bar through its
    contemporaneous coefficient `ret[0]`, and both then run on their lags. The return's own
    innovation is held at zero, so what accumulates is the flow's impact alone.
    """
    r, f = np.zeros(k + 1 + p), np.zeros(k + 1 + p)
    for t in range(p, p + k + 1):
        lr, lf = r[t - p : t][::-1], f[t - p : t][::-1]
        f[t] = float(t == p) + flow[1 : p + 1] @ lr + flow[p + 1 :] @ lf
        r[t] = ret[0] * f[t] + ret[2 : p + 2] @ lr + ret[p + 2 :] @ lf
    return np.cumsum(r[p:])


def settled(cum: np.ndarray, tol: float = 0.01) -> int:
    """The first bar from which the cumulative response stays within `tol` of where it ends."""
    off = np.abs(cum - cum[-1]) > tol * abs(cum[-1])
    return int(np.flatnonzero(off)[-1] + 1) if off.any() else 0


def fit(X: np.ndarray, r: np.ndarray, f: np.ndarray, rows: np.ndarray) -> dict:
    """The VAR by least squares on `rows`: the flow on the lags, the return on the flow and the lags.

    `theta` is the permanent impact of a unit flow innovation, the cumulative response at `IRF`;
    `info` the share of the return's variance its informed part, theta times the innovation, makes.
    """
    ok = rows & np.isfinite(X).all(axis=1)
    flow = np.linalg.lstsq(X[ok], f[ok], rcond=None)[0]
    ret = np.linalg.lstsq(np.column_stack([f[ok], X[ok]]), r[ok], rcond=None)[0]
    cum = response(flow, ret)
    v = f[ok] - X[ok] @ flow
    return {
        "flow": flow,
        "ret": ret,
        "theta": cum[-1],
        "settled": settled(cum),
        "sd_v": v.std(),
        "info": (cum[-1] * v).var() / r[ok].var(),
        "bars": int(ok.sum()),
    }


def pivots(close: pd.Series, window: int = strategy.WINDOW) -> np.ndarray:
    """At each bar, the bar of the pivot a live reader held (`legs.confirmed`), -1 before the first."""
    line = legs.confirmed(close, window)
    if line.empty:
        return np.full(len(close), -1)
    row = np.searchsorted(line.known.to_numpy(), np.arange(len(close)), side="right") - 1
    return np.where(row >= 0, line.bar.to_numpy()[np.clip(row, 0, None)], -1)


def decompose(d: pd.DataFrame, X: np.ndarray, var: dict, pivot: np.ndarray) -> pd.DataFrame:
    """The leg since the last confirmed pivot at each bar, split into what the flow's permanent impact
    explains and the rest. Reads nothing after the bar: the innovation is the flow less what the
    VAR expected from the lags, and the VAR is fitted before the bars it is applied to.

    `side` is the sign of the leg's move; `share` = 1 - informed / move, the part of the leg the
    informed flow does not explain (0 when it explains it all, above 1 when the flow ran against
    it); `flow` the mean signed flow over the leg, signed into it.
    """
    v = np.nan_to_num(d.f.to_numpy() - X @ var["flow"])  # the first LAGS bars have no innovation
    logc = np.log(d.close.to_numpy())
    informed, flow = np.cumsum(var["theta"] * v), np.cumsum(d.f.to_numpy())
    ok, p = pivot >= 0, np.clip(pivot, 0, None)
    move = np.where(ok, logc - logc[p], np.nan)
    side = np.sign(move)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(move != 0, 1 - (informed - informed[p]) / move, np.nan)
    mean_flow = (flow - flow[p]) / np.maximum(np.arange(len(d)) - p, 1)
    return pd.DataFrame({"side": side, "share": share, "flow": side * mean_flow}, index=d.index)


def starts(x: np.ndarray, threshold: float) -> np.ndarray:
    """The first bar of each excursion of `x` past +-`threshold`: one event per excursion, known at its bar."""
    side = np.where(np.abs(np.nan_to_num(x)) >= threshold, np.sign(np.nan_to_num(x)), 0)
    return (side != 0) & (side != np.r_[0, side[:-1]])


def pooled(x: pd.Series, horizon: pd.Timedelta) -> dict[str, float]:
    """The sample mean of `x`, with its error from non-overlapping clock blocks of `horizon`.

    Not `metrics.blocked`'s mean, which is the mean of the block means: events at extremes cluster
    (fifteen assets reach one on the same bars), and a block holding one event would weigh as much
    as one holding twenty. This is the ratio estimator sum_b S_b / sum_b n_b, its variance from the
    blocks' residuals S_b - mean n_b, so the overlap of forward returns and the market's move shared
    inside a block are still in the error. The two agree when every block holds the same count.
    """
    g = x.groupby(x.index.floor(horizon))
    s, n = g.sum(), g.size()
    mean, b = s.sum() / n.sum(), len(s)
    se = np.sqrt(((s - mean * n) ** 2).sum() * b / (b - 1)) / n.sum() if b > 1 else np.nan
    return {"mean": float(mean), "se": float(se), "blocks": b}


def rank_ic(rows: pd.DataFrame, column: str, h: int) -> pd.DataFrame:
    """Per fold, the rank IC of `column` with the `h`-bar return against the leg.

    Ranks are taken per asset and fold, standardised, and multiplied: the mean of the product over
    an asset's rows is its Spearman, so the fold's IC, the mean over every row (`pooled`), is the
    assets' Spearmans weighted by their rows. Its error comes from `h`-bar clock blocks, which hold
    the overlap of forward returns and the market's move shared by assets on the same bars.
    """
    y = f"y{h}"
    g = rows.dropna(subset=[column, y])
    g = g[g.groupby(["symbol", "fold"])[y].transform("size") >= 5]
    by = [g.symbol, g.fold]
    z = g[[column, y]].groupby(by).rank()
    z = (z - z.groupby(by).transform("mean")) / z.groupby(by).transform("std", ddof=0)
    product = (z[column] * z[y]).to_numpy()
    ok = np.isfinite(product)  # an asset-fold whose column never moves has no rank to speak of
    prod, fold = pd.Series(product[ok], index=pd.DatetimeIndex(g.when.to_numpy()[ok])), g.fold.to_numpy()[ok]
    out = {}
    for k in np.unique(fold):
        b = pooled(prod[fold == k], BAR * h)
        out[int(k)] = {"ic": b["mean"], "se": b["se"], "n": int((fold == k).sum()), "blocks": b["blocks"]}
    return pd.DataFrame(out).T.rename_axis("fold")


def criterion(dev: np.ndarray, confirm: np.ndarray, rho_min: float) -> np.ndarray:
    """The card's test on fold ICs, `dev` (..., 2) and `confirm` (..., 4): the same sign in both
    development folds, their mean past `rho_min` in that sign, and the 2021 folds all of that sign
    with their mean past `rho_min` too. The sign is development's, so the hypothesis (positive,
    reversal) is not assumed."""
    s = np.sign(dev[..., 0])
    held = (np.sign(dev[..., 1]) == s) & (s * dev.mean(axis=-1) > rho_min)
    return held & (np.sign(confirm) == s[..., None]).all(axis=-1) & (s * confirm.mean(axis=-1) > rho_min)


def rho_min(sigma: float, h: int, cost: float = 2 * FEE) -> float:
    """`false_alarms.html` #soglia: the IC a trade held `h` bars needs to pay `cost`, E|z| = 0.8."""
    return cost / (0.8 * sigma * np.sqrt(h))


def study(periods, store: Path = STORE, symbols=SYMBOLS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """`(rows, vars, sigma)`: one row per (bar, kind) the ICs read, the VAR of every asset and fold,
    and each asset's 15m return dispersion over development, which `rho_min` reads.

    Kinds: `v2` at the first bar of each excursion of v2's raw output past its asset's 90th
    percentile of |v2| over development (dev and hold-out only, v2 has no out-of-sample bar
    before); `rsi 12` the same on an RSI at 12, its threshold the 90th percentile over the year
    before each fold, the stand-in for v2 on 2021; `every bar` all bars, context.
    """
    v2 = pd.read_parquet(strategy.PRED, columns=["symbol", "pred"])
    dev = strategy.edges("dev")
    rows, fits, sigma = [], [], {}
    for symbol in symbols:
        d = bars(symbol, store)
        r, f, when = d.r.to_numpy(), d.f.to_numpy(), d.index
        X, pivot, logc = design(r, f), pivots(d.close), np.log(d.close)
        fwd = {h: (logc.shift(-h) - logc).to_numpy() for h in HORIZONS}
        idx = pd.MultiIndex.from_arrays([when, [symbol] * len(d)], names=["open_time", "symbol"])
        rsi = strategy.rsi(idx, close=pd.Series(d.close.to_numpy(), index=idx)).to_numpy()
        pred = v2.pred[v2.symbol == symbol]
        x2 = pred.reindex(when).to_numpy()
        sigma[symbol] = d.r[(when >= dev[0]) & (when < dev[-1])].std()
        thr2 = pred[(pred.index >= dev[0]) & (pred.index < dev[-1])].abs().quantile(EXTREME)
        for period in periods:
            e, fold = strategy.edges(period), strategy.fold_in(when, period)
            first = 3 if period == "holdout" else 1
            for i, a in enumerate(e[:-1]):
                k = i + first
                before = (when >= a - FIT) & (when < a)
                if before.sum() < MIN_FIT or not (fold == k).any():
                    continue
                var = fit(X, r, f, before)
                fits.append(
                    {"period": period, "fold": k, "symbol": symbol}
                    | {n: var[n] for n in var if n not in ("flow", "ret")}
                )
                c = decompose(d, X, var, pivot)
                base = (fold == k) & np.isfinite(c.share.to_numpy())
                kinds = {"every bar": base, "rsi 12": base & starts(rsi, np.quantile(np.abs(rsi[before]), EXTREME))}
                if period != "2021":
                    kinds["v2"] = base & starts(x2, thr2)
                for kind, m in kinds.items():
                    side = c.side.to_numpy()[m]
                    rows.append(
                        pd.DataFrame(
                            {"when": when[m], "symbol": symbol, "period": period, "fold": k, "kind": kind}
                            | {col: c[col].to_numpy()[m] for col in COLUMNS}
                            | {f"y{h}": -side * fwd[h][m] for h in HORIZONS}
                        )
                    )
        print(f"  {symbol}: {len(d):,} bars", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fits), sigma


def table(rows: pd.DataFrame) -> pd.DataFrame:
    """Every rank IC of the study: (period, kind, column, h) by fold, with the period's mean."""
    out = []
    for (period, kind), g in rows.groupby(["period", "kind"], sort=False):
        for col in COLUMNS:
            for h in HORIZONS:
                t = rank_ic(g, col, h)
                row = {"period": period, "kind": kind, "column": col, "h": h}
                for k, x in t.iterrows():
                    row |= {f"ic{k}": x.ic, f"se{k}": x.se, f"n{k}": x.n}
                row |= {"mean": t.ic.mean(), "mean_se": np.sqrt((t.se**2).sum()) / len(t)}
                out.append(row)
    return pd.DataFrame(out).set_index(["period", "kind", "column", "h"])


def show(t: pd.DataFrame) -> str:
    """`table` as `ic +- se (n)` per fold, which is how it is read."""
    folds = sorted({int(c[2:]) for c in t.columns if c.startswith("ic")})
    cells = pd.DataFrame(index=t.index)
    for k in folds:
        cells[f"fold {k}"] = [
            "" if np.isnan(i) else f"{i:+.3f} ± {s:.3f} ({n:,.0f})"
            for i, s, n in zip(t[f"ic{k}"], t[f"se{k}"], t[f"n{k}"])
        ]
    cells["mean"] = [f"{m:+.3f} ± {s:.3f}" for m, s in zip(t["mean"], t.mean_se)]
    return cells.to_string()


def verdict(t: pd.DataFrame, rmin: dict[int, float]) -> list[str]:
    """The card's criterion on the transient share: v2's extremes on development, RSI's on 2021."""
    lines = []
    for h in HORIZONS:
        try:
            dev = np.array([t.loc[("dev", "v2", "share", h), f"ic{k}"] for k in (1, 2)])
        except KeyError:
            return ["verdict needs --period dev"]
        s = np.sign(dev[0])
        dev_ok = bool(np.sign(dev[1]) == s and s * dev.mean() > rmin[h])
        head = f"h={h}: dev {dev[0]:+.3f} / {dev[1]:+.3f} against rho_min {rmin[h]:.3f}"
        if ("2021", "rsi 12", "share", h) not in t.index:
            lines.append(f"{head}: development {'passes' if dev_ok else 'fails'}; 2021 not run")
            continue
        conf = np.array([t.loc[("2021", "rsi 12", "share", h), f"ic{k}"] for k in (1, 2, 3, 4)])
        ok = bool(criterion(dev, conf, rmin[h]))
        lines.append(f"{head}, 2021 {' / '.join(f'{c:+.3f}' for c in conf)}: {'PASSES' if ok else 'fails'}")
    return lines


def _spearman_draws(n: int, rho: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """`reps` sample Spearman correlations of `n` bivariate normal pairs whose population Spearman is `rho`."""
    r = 2 * np.sin(np.pi * rho / 6)  # Pearson of a normal pair with that Spearman
    ranks = np.broadcast_to(np.arange(n, dtype="float64") - (n - 1) / 2, (min(reps, max(1, 2_000_000 // n)), n))
    out = []
    for i in range(0, reps, len(ranks)):  # in chunks of ~2M pairs: 2021's folds hold ~20,000 events
        m = min(len(ranks), reps - i)
        x = rng.standard_normal((m, n))
        y = r * x + np.sqrt(1 - r**2) * rng.standard_normal((m, n))
        rx, ry = np.empty((m, n)), np.empty((m, n))
        np.put_along_axis(rx, x.argsort(axis=1), ranks[:m], axis=1)
        np.put_along_axis(ry, y.argsort(axis=1), ranks[:m], axis=1)
        out.append((rx * ry).sum(axis=1) / (ranks[0] ** 2).sum())
    return np.concatenate(out)


def power(reps: int = 600, seed: int = 0) -> pd.DataFrame:
    """How often `criterion` passes at the store's sample sizes, for effects of 0, 0.5, 1 and 2 rho_min.

    Each fold's IC is the Spearman of independent normal pairs, drawn at two sizes: the events the
    fold held (`EVENTS`), which ignores that events on the same bars share the market's move and
    overlapping forward returns, and the effective size its blocked error implies, 1 / se^2
    (`EFFECTIVE`), which equals the count for independent events and is the honest one.
    """
    rng = np.random.default_rng(seed)
    out = []
    for h in HORIZONS:
        rmin = rho_min(SIGMA, h)
        for mult in (0.0, 0.5, 1.0, 2.0):
            row = {"h": h, "effect": f"{mult:g} rho_min", "ic": mult * rmin}
            for basis, n in (("events", EVENTS), ("effective", EFFECTIVE[h])):
                dev = np.column_stack([_spearman_draws(m, mult * rmin, reps, rng) for m in n["dev"]])
                conf = np.column_stack([_spearman_draws(m, mult * rmin, reps, rng) for m in n["2021"]])
                row[basis] = criterion(dev, conf, rmin).mean()
            out.append(row)
    return pd.DataFrame(out).set_index(["h", "effect"])


# Measured by `main` on the taker store (2026-10-07), all 15 SYMBOLS: v2's events in the two
# development folds and the RSI's in the four of 2021, at h = 48 (the fewest rows), and the
# effective sizes of the transient share's IC at each horizon; the 15m dispersion is the median
# over the assets on development. `power` reads them so it runs without the store.
SIGMA = 0.00461
EVENTS = {"dev": [4339, 3903], "2021": [20792, 18856, 17457, 21653]}
EFFECTIVE = {
    12: {"dev": [2743, 2589], "2021": [10704, 10693, 11117, 12844]},
    24: {"dev": [1935, 2430], "2021": [7928, 8786, 9261, 10442]},
    48: {"dev": [1590, 1701], "2021": [6917, 6710, 7885, 9384]},
}


def _selfcheck() -> None:
    """A simulated pair with a known impact: the VAR finds it, a planted reversion is found, the null
    is zero, and nothing at a bar moves when the bars after it change."""
    rng = np.random.default_rng(0)
    n, lam, kappa = 40_000, 0.004, 0.003
    when = pd.date_range("2024-01-01", periods=n, freq=BAR, tz="UTC")
    v = rng.normal(0, 0.2, n)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.3 * f[t - 1] + v[t]
    news = rng.normal(0, 0.001, n)
    # Permanent lam per unit innovation, plus kappa that is undone the bar after: theta = lam. As a
    # VAR: f = 0.3 f1 + v, r = (lam + kappa) f - (0.3 (lam + kappa) + kappa) f1 + 0.3 kappa f2.
    flow, ret = np.zeros(2 * LAGS + 1), np.zeros(2 * LAGS + 2)
    flow[LAGS + 1], ret[0], ret[LAGS + 2], ret[LAGS + 3] = 0.3, lam + kappa, -0.3 * (lam + kappa) - kappa, 0.3 * kappa
    cum = response(flow, ret)
    assert np.isclose(cum[-1], lam) and np.isclose(cum[0], lam + kappa) and settled(cum) == 1, cum[:4]
    r = lam * v + kappa * (v - np.r_[0, v[:-1]]) + news
    # Estimated, theta is within 1-4% of lam over seeds at this size (the lags of r are nearly
    # collinear with the flow's here, and their noise feeds back through the response).
    X, everything = design(r, f), np.ones(n, dtype=bool)
    assert abs(fit(X, r, f, everything)["theta"] - lam) < 0.1 * lam
    assert abs(fit(X, news, f, everything)["theta"]) < 0.1 * lam, "returns that ignore the flow have no impact"
    rows = when < when[n // 2]
    var = fit(X, r, f, rows)

    # A transient AR(1) in the price that the flow does not explain, 24-bar half-life, about a third
    # of a bar's variance: the leg's unexplained share has to predict its reversal; without it, nothing.
    z = np.zeros(n)
    phi = 0.5 ** (1 / 24)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal(0, 0.002)
    ics = {}
    for name, ret in (("planted", r + np.diff(z, prepend=0.0)), ("null", r)):
        d = pd.DataFrame({"close": 100 * np.exp(np.cumsum(ret)), "r": ret, "f": f}, index=when)
        X = design(ret, f)
        c = decompose(d, X, fit(X, ret, f, rows), pivots(d.close))
        logc = np.log(d.close.to_numpy())
        y = -c.side.to_numpy() * (np.r_[logc[48:], np.full(48, np.nan)] - logc)
        frame = pd.DataFrame({"when": when, "symbol": "X", "fold": 1, "share": c.share, "y48": y})
        ics[name] = rank_ic(frame[~rows & np.isfinite(c.share.to_numpy())], "share", 48).iloc[0]
    # Measured: +0.123 +- 0.027 planted, at 48 bars on 417 blocks; at 12 bars the same world gives
    # +0.036 +- 0.017, a quarter of the deviation having reverted by then.
    assert ics["planted"].ic > 4 * ics["planted"].se, ics
    assert abs(ics["null"].ic) < 3 * ics["null"].se, ics

    # Causality: change every bar after `cut`; nothing at or before it moves.
    cut = n - 500
    r2, f2 = r.copy(), f.copy()
    r2[cut + 1 :], f2[cut + 1 :] = rng.normal(0, 0.01, n - cut - 1), rng.uniform(-1, 1, n - cut - 1)
    a, b = (
        decompose(
            pd.DataFrame({"close": 100 * np.exp(np.cumsum(x)), "r": x, "f": y}, index=when),
            design(x, y),
            var,
            pivots(pd.Series(100 * np.exp(np.cumsum(x)))),
        )
        for x, y in ((r, f), (r2, f2))
    )
    assert a.iloc[: cut + 1].equals(b.iloc[: cut + 1]) and not a.iloc[cut + 1 :].equals(b.iloc[cut + 1 :])

    # `pooled` is the sample mean, not the mean of blocks: four events at 0 in one block, one at 1 and
    # one at 2 in two others are 0.5 and not 1. With the same count in every block they coincide, error too.
    when = pd.DatetimeIndex(["2025-01-01 00:00"] * 4 + ["2025-01-01 12:00", "2025-01-02 00:00"], tz="UTC")
    x, half = pd.Series([0, 0, 0, 0, 1, 2.0], index=when), pd.Timedelta("12h")
    assert np.isclose(pooled(x, half)["mean"], 0.5) and np.isclose(metrics.blocked(x, half)["mean"], 1.0)
    even = pd.Series(rng.normal(size=48), index=pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC"))
    a, b = pooled(even, half), metrics.blocked(even, half)
    assert np.isclose(a["mean"], b["mean"]) and np.isclose(a["se"], b["se"]), (a, b)

    # One event per excursion, at its first bar; the criterion's sign is development's.
    assert starts(np.array([0, 2, 3, 2, 0, -2, -2, 2, np.nan]), 2).tolist() == [0, 1, 0, 0, 0, 1, 0, 1, 0]
    assert criterion(np.array([-0.2, -0.15]), np.array([-0.1, -0.2, -0.3, -0.2]), 0.1)
    assert not criterion(np.array([0.2, -0.15]), np.array([0.1, 0.2, 0.3, 0.2]), 0.1)
    assert not criterion(np.array([0.2, 0.15]), np.array([0.1, 0.2, -0.3, 0.2]), 0.1)
    assert np.isclose(_spearman_draws(200_000, 0.3, 1, rng)[0], 0.3, atol=0.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", nargs="+", default=["dev"], choices=strategy.PERIODS)
    ap.add_argument("--store", type=Path, default=STORE, help="where the 5m files with the taker columns are")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--power", action="store_true", help="the criterion's detection rate, simulated")
    args = ap.parse_args()

    _selfcheck()
    pd.set_option("display.width", 250)
    if args.power:
        print(f"criterion detection rate, {strategy.FOLDS}-fold sizes as measured, sigma {SIGMA:.4f}\n")
        print(power().round(3).to_string())
        return
    rows, fits, sigma = study(args.period, args.store, args.symbols)
    s = float(np.median(list(sigma.values())))
    rmin = {h: rho_min(s, h) for h in HORIZONS}
    g = fits.assign(impact=fits.theta * fits.sd_v * 1e4).groupby(["period", "fold"])
    print("\nthe VAR by fold, over the assets: permanent impact of a one-sd flow innovation (bp), share of")
    print("the return's variance its informed part explains, bars until the response stays within 1%\n")
    print(
        pd.DataFrame(
            {
                "assets": g.size(),
                "impact_bp": g.impact.median(),
                "impact_min": g.impact.min(),
                "impact_max": g.impact.max(),
                "info": g["info"].median(),
                "settled_median": g.settled.median(),
                "settled_max": g.settled.max(),
            }
        )
        .round(3)
        .to_string()
    )
    t = table(rows)
    print("\nrank IC with the forward return against the leg (positive = reversal), ± blocked error (rows)\n")
    print(show(t))
    tradable = table(rows[rows.symbol.isin(TRADABLE) & (rows.kind != "every bar")])
    print("\nthe same at the events, TRADABLE only\n")
    print(show(tradable))
    folds = sorted({int(c[2:]) for c in t.columns if c.startswith("ic")})
    sizes = pd.DataFrame(
        {f"{c} {k}": t[f"n{k}"] if c == "events" else 1 / t[f"se{k}"] ** 2 for k in folds for c in ("events", "eff")}
    )
    print("\nthe transient share's sample sizes: rows, and the effective size its blocked error implies, 1 / se^2\n")
    print(sizes.xs("share", level="column").round(0).to_string())
    print(
        f"\nsigma {s:.5f} (median 15m sd on development); rho_min " + ", ".join(f"{h}: {rmin[h]:.3f}" for h in HORIZONS)
    )
    for universe, tt in (("SYMBOLS", t), ("TRADABLE", tradable)):
        for line in verdict(tt, rmin):
            print(f"verdict, transient share, {universe}, {line}")


if __name__ == "__main__":
    main()
