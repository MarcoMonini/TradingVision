"""Where an indicator says something about the price N candles later, and whether it still does later on.

The decomposer page draws, for one indicator at a time, the mean change from a candle's close to N
candles later by band of the indicator, as an excess over every candle's in standard errors. Looked
at that way over the whole history, every indicator has a coloured corner, and with 29 columns, five
windows, ten bands and eight N some corners are coloured by chance. This module does the same search
once, over everything, with the choice and the reading on different years.

**What is measured.** Every column of `features` at every window of `WINDOWS` (the columns that read
no window, `FIXED`, once), on BTC and ETH, 5m and 15m. Each column is cut in bands at the quantiles
`QS` of the discovery period — 1% and 5% tails, then deciles and quartiles — and the edges are applied
unchanged to the later periods: thresholds are decided on the first years only. For each band and
each N of `HORIZONS`: the mean change in percent from the close to the close N candles later, less
the mean of every candle of the same period at the same N (`excess`), and its standard error over
non-overlapping blocks of N candles on the clock (`metrics.blocked`'s rule: neighbouring candles
share N − 1 candles of their future). A candle whose future crosses into the next period counts in
neither.

**Three periods.** `discovery`, the store's first bar to 2020-12-31, is where zones are chosen;
`confirm`, 2021-01 to 2025-06 (`strategy.CONFIRM`), and `recent`, 2025-06 to the store's end, are
read with nothing changed.

**What a zone is, decided before any reading.** A cell (timeframe, column, window, band, N) passes
discovery when its excess is beyond `T_PICK` standard errors on BTC and on ETH, with the same sign.
Neighbouring windows and N are one result and not many, so per (timeframe, column, band) only the
cell with the largest smaller-of-the-two t is kept. It holds when the confirm period gives the same
sign on both assets at two standard errors or more. A hold is about the sign; the size to compare
with the fee is the excess on confirm, against 0.20% a round trip at OKX (`oracle.FEE` a side).

An excess per candle is not a trade: consecutive candles of one crash all sit in the bottom band and
share their future. `trades` takes one trade an episode, the first candle in the band, held N candles,
and nothing new until it closes; its error is across trades, gross of fees.

**What came out**, measured 2026-10-09 on the store to 2026-10-06, BTC and ETH, 5m and 15m: 35,498
cells, 115 zones past discovery, 43 that hold on confirm, 29 of them beyond 0.20% in excess. They are
four things seen through many columns, not 43 findings.

1. *A price far under its own average bounces* — the bottom 1% of the distance from KAMA, VWAP, EMA,
   PSAR or the window's high, of the EMA's slope, of the body, of the return over the window. The one
   family that holds on every column and both timeframes. 15m, distance from VWAP at window 12, N 6
   (an hour and a half): confirm +0.56% a trade on BTC (± 0.15, 227 trades, 60% win) and +0.51% on ETH
   (± 0.17, 307); the distance from the 6-candle high, N 24: +1.14% (± 0.36, 118) and +0.94% (± 0.38,
   156); 5m, distance from KAMA at 12, N 24: +0.25% and +0.24% (± 0.12). Its weakness is the third
   period: the edges are 2017-2020's, and a 15m candle's deviation has gone from 0.49% to 0.22% on BTC
   (0.67% to 0.33% on ETH), so from 2025-06 the band fires 6 to 140 times a pair with errors as large as
   the means. Edges in units of the period's own volatility are the next thing to try.
2. *The largest bars come before rises* — the top 1% of the bar's range, the wicks, the realised
   volatility, the ATR, at N 96. Confirm +0.6 to +2.9% a trade on 15m, but on 21 to 165 trades with
   errors of 0.4 to 1.8: the volatile periods are the bull runs, and the crash rebounds of family 1.
3. *The close's place in its bar or window, one candle ahead* — the bottom 1-10% of the close in the
   bar or the 6-candle window rises +0.004 to +0.013% the next candle on confirm, at 3 to 8 standard
   errors on tens of thousands of trades: real, a twentieth of the fee or less, and about a quarter of
   what it was in 2017-2020. The microstructure an exchange's spread lives on.
4. *A slower bounce under the fee* — 15m close at the bottom 1% of its 48-candle window, N 6: +0.14% and
   +0.16% a trade (± 0.03, ± 0.05, 1,277 and 865 trades), the most precise of all and under 0.20%.

What does not hold: momentum. The RSI's and the TSI's top 5-1% at N 48-192, the decomposer's lead
(+0.56% at RSI 80-90 on BTC 15m over the whole history), are 4.3 to 6.4 standard errors on 2017-2020
and 0.5 to 3.1 on 2021-2025: positive, and under 2 on at least one of the two assets. The N = 1 reversals of large
bodies and closes at the high are gone after 2020.

BTC and ETH are not two independent confirmations: their crashes are the same days. A zone of family 1
is a lead for a rule — buy the close of a dislocated candle, hold N — to be written before it is
priced on the third period and in paper trading; the execution at the close of a crash candle is
where its spread and slippage are worst.

    uv run python -m tradingvision.zones [--symbols BTC ETH] [--timeframes 5m 15m]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision import features
from tradingvision.data import binance
from tradingvision.oracle import FEE
from tradingvision.strategy import CONFIRM

WINDOWS = (6, 12, 24, 48, 96)  # `features`' N: every window of every column, `short` its quarter
# The columns that read no window: computed at the first and asserted equal at the others in `_selfcheck`.
FIXED = (
    "log_return",
    "candle_body_pct",
    "bar_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position_in_bar",
    "distance_from_psar_pct",
)
QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)  # the band edges, quantiles of the discovery period
HORIZONS = (1, 3, 6, 12, 24, 48, 96, 192)  # N, in candles
PERIODS = ("discovery", "confirm", "recent")
T_PICK = 4.0  # a discovery cell beyond this on both assets: about Bonferroni over ~20,000 cells an asset
T_HOLD = 2.0  # the confirm period's bar, same sign on both assets
ROUND_TRIP = 2 * FEE * 100  # percent


def period_of(when: pd.DatetimeIndex) -> np.ndarray:
    """0 discovery, 1 confirm, 2 recent, by the bar's open."""
    return np.searchsorted(np.array([CONFIRM[0], CONFIRM[1]]), when, side="right")


def band_edges(x: pd.Series, period: np.ndarray) -> np.ndarray:
    """The inner edges of the bands: `QS` of `x` on discovery. A tie repeats an edge and leaves the band between the
    copies empty rather than merging it, so band b is the quantiles `QS[b-1]` to `QS[b]` on every pair alike."""
    return np.nanquantile(x.to_numpy()[period == 0], QS)


def quantiles(band: int) -> str:
    """Band b's name, the quantiles of discovery it spans: `bands` puts a tie in the highest band it reaches."""
    q = (0, *QS, 1)
    return f"{q[band]:.0%}-{q[band + 1]:.0%}"


def bands(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Each candle's band, 0 to len(edges), closed on the left; −1 where `x` is NaN."""
    v = x.to_numpy()
    return np.where(np.isnan(v), -1, np.searchsorted(edges, v, side="right"))


def cells(code: np.ndarray, close: pd.Series, period: np.ndarray, step: pd.Timedelta) -> pd.DataFrame:
    """By period, band and N: the band's mean change to N candles later (`mean`), every candle's (`all`), the
    excess, its blocked standard error and t, and the count. `code` is `bands`' output, `period` `period_of`'s."""
    c = close.to_numpy()
    ns = close.index.asi8
    rows = []
    for n in HORIZONS:
        future = np.full(len(c), np.nan)
        future[:-n] = (c[n:] / c[:-n] - 1) * 100
        same = np.zeros(len(c), bool)
        same[:-n] = period[n:] == period[:-n]
        for p in range(len(PERIODS)):
            have = np.isfinite(future) & same & (period == p)
            if not have.any():
                continue
            every = future[have].mean()
            m = have & (code >= 0)
            k, f = code[m], future[m]
            block = ns[m] // (n * step.value)
            key, inv = np.unique(k * (block.max() + 1) + block, return_inverse=True)
            bmean = np.bincount(inv, f) / np.bincount(inv)
            bband = key // (block.max() + 1)
            nb = np.bincount(bband)
            s1, s2 = np.bincount(bband, bmean), np.bincount(bband, bmean**2)
            with np.errstate(invalid="ignore", divide="ignore"):
                sd = np.sqrt((s2 - s1**2 / nb) / (nb - 1))
                mean = np.bincount(k, f) / np.bincount(k)
            for b in np.flatnonzero(nb):
                rows.append((PERIODS[p], b, n, mean[b], every, sd[b] / np.sqrt(nb[b]), int((k == b).sum())))
    out = pd.DataFrame(rows, columns=["period", "band", "n", "mean", "all", "se", "count"])
    out["excess"] = out["mean"] - out["all"]
    return out.assign(t=out.excess / out.se)


def scan(symbol: str, timeframe: str) -> pd.DataFrame:
    """`cells` for every column at every window on one pair and timeframe, with each band's edges named."""
    df = binance.load(symbol, timeframe)
    period, step = period_of(df.index), pd.Timedelta(timeframe.replace("m", "min"))
    out = []
    for w in WINDOWS:
        f = features.features(df, w)
        for column in features.COLUMNS:
            if column in FIXED and w != WINDOWS[0]:
                continue
            edges = band_edges(f[column], period)
            names = np.r_[-np.inf, edges, np.inf]
            t = cells(bands(f[column], edges), df.close, period, step)
            t["lo"], t["hi"] = names[t.band], names[t.band + 1]
            out.append(t.assign(column=column, window=0 if column in FIXED else w))
        print(f"  {symbol} {timeframe} window {w}", flush=True)
    return pd.concat(out).assign(symbol=symbol, timeframe=timeframe)


def zones(table: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """The criterion of the docstring: discovery cells beyond `T_PICK` on every symbol with one sign, the best per
    (timeframe, column, band), with every period's excess and t per symbol beside them and the verdict."""
    key = ["timeframe", "column", "window", "band", "n"]
    wide = table.pivot_table(index=key, columns=["period", "symbol"], values=["excess", "t", "count"])
    t = wide["t"]["discovery"][symbols]
    sign = np.sign(t).nunique(axis=1).eq(1) & np.sign(t).ne(0).all(axis=1)
    weakest = t.abs().min(axis=1).where(sign)
    picked = weakest[weakest >= T_PICK].rename("pick").reset_index()
    best = picked.loc[picked.groupby(["timeframe", "column", "band"]).pick.idxmax()]
    out = wide.loc[best.set_index(key).index]
    flat = pd.DataFrame(index=out.index)
    for p in PERIODS:
        for s in symbols:
            flat[f"{p} {s} %"] = out[("excess", p, s)] if ("excess", p, s) in out else np.nan
            flat[f"{p} {s} t"] = out[("t", p, s)] if ("t", p, s) in out else np.nan
    btc = table[(table.symbol == symbols[0]) & (table.period == "discovery")].set_index(key)[["lo", "hi"]]
    flat = flat.join(btc.add_prefix(f"{symbols[0]} "))
    flat.insert(0, "quantiles", [quantiles(b) for b in flat.index.get_level_values("band")])
    d = np.sign(flat[[f"discovery {s} t" for s in symbols]].iloc[:, 0])
    c = flat[[f"confirm {s} t" for s in symbols]]
    flat["holds"] = (c.mul(d, axis=0) >= T_HOLD).all(axis=1)
    flat["confirm %"] = flat[[f"confirm {s} %" for s in symbols]].mean(axis=1)
    flat["pays"] = flat["holds"] & (flat["confirm %"].abs() >= ROUND_TRIP)
    return flat.join(best.set_index(key).pick).sort_values("pick", ascending=False)


def trades(hit: np.ndarray, close: pd.Series, n: int, sign: float) -> np.ndarray:
    """The change in percent of one trade an episode: entered at the close of the first candle of `hit` (a boolean
    mask), held `n` candles in the direction `sign`, and no new entry until it is closed. A candle whose future
    runs past the store's end opens none."""
    c = close.to_numpy()
    out, free = [], -1
    for i in np.flatnonzero(hit):
        if i > free and i + n < len(c):
            out.append(sign * (c[i + n] / c[i] - 1) * 100)
            free = i + n
    return np.array(out)


def detail(z: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """For every zone that holds, `trades` per period and symbol: how many, their mean with its error, median and
    the share that gained, in the zone's direction and gross of fees. The band's edges are discovery's."""
    frames, rows = {}, []
    for (tf, column, w, band, n), r in z[z.holds].iterrows():
        sign = np.sign(r[f"discovery {symbols[0]} t"])
        for s in symbols:
            if (s, tf) not in frames:
                frames[(s, tf)] = binance.load(s, tf)
            df = frames[(s, tf)]
            if (s, tf, w) not in frames:
                frames[(s, tf, w)] = features.features(df, w or WINDOWS[0])
            x, period = frames[(s, tf, w)][column], period_of(df.index)
            code = bands(x, band_edges(x, period))
            row = {"timeframe": tf, "column": column, "window": w, "quantiles": quantiles(band), "n": n, "symbol": s}
            for p, name in enumerate(PERIODS):
                g = trades((code == band) & (period == p), df.close, n, sign)
                row |= {
                    f"{name} trades": len(g),
                    f"{name} mean": g.mean() if len(g) else np.nan,
                    f"{name} se": g.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                    f"{name} median": np.median(g) if len(g) else np.nan,
                    f"{name} win": (g > 0).mean() if len(g) else np.nan,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m"])
    args = parser.parse_args()
    table = pd.concat([scan(s, tf) for tf in args.timeframes for s in args.symbols])
    table.to_parquet(binance.STORE / "zones.parquet")
    z = zones(table, args.symbols)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    print(
        f"{len(table) // len(PERIODS):,} cells; {len(z)} zones pass discovery, {int(z.holds.sum())} hold, "
        f"{int(z.pays.sum())} beyond {ROUND_TRIP:.2f}% on confirm"
    )
    print(z.round(3).to_string())
    print("\nOne trade an episode, gross, in the zone's direction:")
    print(detail(z, args.symbols).round(3).to_string())


def _selfcheck() -> None:
    """`FIXED` reads no window, the bands and the blocked error on hand-made series."""
    when = pd.date_range("2020-12-01", periods=5000, freq="15min", tz="UTC")  # across the cut
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(when)))), index=when)
    df = pd.DataFrame({"open": close.shift().bfill(), "close": close, "volume": rng.uniform(1, 2, len(when))})
    df["high"], df["low"] = df[["open", "close"]].max(axis=1) * 1.001, df[["open", "close"]].min(axis=1) * 0.999
    a, b = features.features(df, 6)[list(FIXED)], features.features(df, 48)[list(FIXED)]
    assert a.iloc[100:].equals(b.iloc[100:]), "a FIXED column reads the window"

    period = period_of(when)
    assert period[0] == 0 and period[-1] == 1 and (np.diff(period) >= 0).all()
    x = pd.Series(np.arange(len(when), dtype=float), index=when).where(lambda s: s % 7 > 0)
    edges = band_edges(x, period)
    code = bands(x, edges)
    assert (code[x.isna().to_numpy()] == -1).all() and code.max() == len(edges)
    # A tie at the bottom: 30% zeros fill the band their quantile reaches and leave the ones below empty, so band 4
    # is 25-50% here as on a pair with no tie.
    ties = pd.Series(np.r_[np.zeros(300), np.arange(1.0, 701)], index=when[:1000])
    c = bands(ties, band_edges(ties, np.zeros(1000, int)))
    assert (c[:300] == 4).all() and set(c[300:]) == set(range(4, len(QS) + 1)) and quantiles(4) == "25%-50%"
    # On a close rising 1% a candle every band's mean is the same, so every excess is zero, and the count of a
    # band is the candles that have a future inside their own period.
    up = pd.Series(100 * 1.01 ** np.arange(len(when)), index=when)
    t = cells(code, up, period, pd.Timedelta("15min"))
    assert np.allclose(t.excess, 0, atol=1e-9) and np.allclose(t["mean"][t.n == 3], (1.01**3 - 1) * 100)
    split = int((period == 0).sum())
    assert t[(t.n == 3) & (t.period == "discovery")]["count"].sum() == int(x.iloc[: split - 3].notna().sum())
    # The blocked error on independent noise is the naive one, std / sqrt(n), within sampling error.
    noise = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200_000))),
        index=pd.date_range("2018-01-01", periods=200_000, freq="5min", tz="UTC"),
    )
    one = cells(np.zeros(len(noise), int), noise, period_of(noise.index), pd.Timedelta("5min"))
    r = one[(one.n == 1) & (one.period == "discovery")].iloc[0]
    naive = noise.pct_change().shift(-1)[period_of(noise.index) == 0].std() * 100 / np.sqrt(r["count"])
    assert abs(r.se / naive - 1) < 0.05
    # Trades: one an episode, the next only after the last is closed, none past the end; a short gains on a fall.
    hit = np.zeros(len(up), bool)
    hit[[10, 11, 12, 20, len(up) - 2]] = True
    g = trades(hit, up, 5, 1.0)
    assert len(g) == 2 and np.allclose(g, (1.01**5 - 1) * 100)
    assert np.allclose(trades(hit, up, 5, -1.0), -g)


if __name__ == "__main__":
    main()
