"""Bulk OHLCV history from the Binance public data dumps.

Alpaca's crypto history only starts 2021-01-01 (its own venue went live then) and is holed by the
2023 delistings, so the ML dataset is built from `data.binance.vision` instead: static ZIPs on S3,
no API key, no rate limit, USDT pairs from 2017. Alpaca stays the live/execution feed.

    python -m tradingvision.data.binance                 # update every symbol to yesterday
    python -m tradingvision.data.binance --symbols BTC ETH --interval 15m

Re-running only fetches what is missing: the store keeps one Parquet per (symbol, interval) and
the last partial month is always re-fetched, so an interrupted run heals itself.

**Every bar opens on the interval's grid, and the store refuses one that does not.** The monthly
dumps of February 2018 break this. Binance halted on 2018-02-08 at 00:28:14.789 and reopened on
the 9th around 10:00; its kline engine resumed counting five minutes from the instant it stopped,
so from 09:58:14.789 to 05:58:14.789 on the 10th every bar opens at hh:m3:14.789 or hh:m8:14.789
(the offset is per pair, to the millisecond: 14.789 BTC, 14.800 ETH, 15.787 BNB, 16.812 LTC).
That is 241 rows in each of the four USDT pairs listed then — BTC, ETH, BNB, LTC — and in no other
month or pair of SYMBOLS or STUDY, checked on 2026-10-09. `parse` reads the stamps as they are.

The rows are real trades on the wrong grid, so neither obvious repair holds. *Snapping* each stamp
down to the grid relabels a bar that closes at 10:18:14 as the one that closes at 10:15: the bars
`load` aggregates from them carry 3m14s of the next bar, and only 2 of the 240 floored bars have
the close of the true on-grid bar. *Dropping* loses twenty hours. The **daily** dumps of the 9th
and the 10th hold the same stretch re-binned on the grid by Binance itself — the same 31,061.592
BTC and 395,086 trades over it as the monthly rows — so `update` replaces the off-grid rows with
the daily files of their days, and drops only what a daily file cannot replace. Before this, every
higher timeframe was on its grid anyway (the resample buckets by stamp: no duplicate, no extra row),
but each bucket in those twenty hours held [b + 3m14s, b + tf + 3m14s), a close past the bar's own.

Nothing measured moved: `swing` reads from 2021 (`SINCE`), so no cached tensor holds these rows and
`swing.BUILD` stays. A cache built with `--since` before 2018-02-10 does hold them, and its stamp
cannot tell — the store's last bar did not move — so delete it. `oracle` reads the whole history by
default and saw 80 of 320,000 15m bars per pair a few minutes late.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

# The universe every number measured before 2026-09-27 was taken on: 20 USDT pairs with at least
# 70 months of 5m history and no missing month. History was the only criterion, and it let in a
# meme coin and four pairs too thin to model. Kept under its own name so those numbers stay
# reproducible — pass it as `--symbols` — and never the default again.
STUDY = [
    "BTC", "ETH", "LTC", "ADA", "XRP", "TRX", "LINK", "BAT", "DOGE", "XTZ",
    "BCH", "YFI", "DOT", "SOL", "CRV", "UNI", "AVAX", "SUSHI", "NEAR", "AAVE",
]  # fmt: skip

# The training universe, from 2026-09-27. Measured on 15m bars over the train period, 2023-01 ->
# 2025-05: at least ~15 M$ a day on Binance, fewer than 10% of bars whose close repeats the one
# before (a price nobody traded, or a tick coarse against the price), no meme coin, history from
# 2020 or earlier. Ordered by that volume, 1,745 M$ for BTC to 15 M$ for UNI.
#
# Out of STUDY: DOGE (meme); BAT, YFI, XTZ, SUSHI at 1-5 M$ a day, XTZ and SUSHI with 13-17% flat
# bars; CRV at the edge (15 M$, 6.4% flat); TRX, liquid but another animal — correlation 0.43 with
# BTC, half the volatility, the most tail rows of the label. In: BNB, fifth by volume at 145 M$,
# history from 2017, and FIL at 25 M$. Measured and left out: ATOM, ETC, ICP, HBAR, GRT, ALGO, XLM,
# all 7-14 M$, HBAR and XLM with 14-15% flat bars.
#
# Not every pair here can be traded: see TRADABLE. A model learns from all of them.
SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "AVAX", "ADA", "LTC",
    "LINK", "NEAR", "FIL", "DOT", "BCH", "AAVE", "UNI",
]  # fmt: skip

# The pairs of SYMBOLS that Alpaca serves, checked against its feed on 2026-09-27: Alpaca is the
# chart page's live feed, so these are the pairs it can draw, and a metric that means money is read
# on these alone. The fee is OKX's (`oracle.FEE`), whose listing has not been checked against this.
# BNB and NEAR stay in training; Alpaca lists neither.
TRADABLE = [s for s in SYMBOLS if s not in ("BNB", "NEAR")]

BASE = "https://data.binance.vision/data/spot"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
# Anchored to the repo root, not the working directory: the oracle and the app are launched from
# wherever, and a relative default turns a wrong CWD into FileNotFoundError instead of data.
# Assumes an editable install, which is how this project is set up.
STORE = Path(__file__).resolve().parents[3] / "data"
WORKERS = 8

# Raw dump layout: no header, 12 columns. We keep the 7 that carry information.
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]  # fmt: skip
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]


OHLC = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "trades": "sum",
}  # fmt: skip


def load(symbol: str, timeframe: str = "5m", *, stored: str = "5m", store: Path = STORE) -> pd.DataFrame:
    """Read one symbol at `timeframe`, resampling up from the `stored` files when they differ.

    `load("BTC", "15m")` reads the 5m Parquet and aggregates it. Bars with no trade in the period
    are dropped rather than forward filled: a synthetic bar would be an invented price, and the
    pivot search would treat it as a real level.
    """
    path = store / f"{symbol}USDT-{stored}.parquet"
    df = pd.read_parquet(path)
    if (off := _off(df.index, stored)).any():
        raise SystemExit(
            f"{path} has {off.sum()} bars off the {stored} grid, from {df.index[off][0]} — run "
            f"`python -m tradingvision.data.binance --symbols {symbol}` to replace them (see the module docstring)"
        )
    if timeframe == stored:
        return df
    return df.resample(_rule(timeframe)).agg(OHLC).dropna(subset=["open"])


def _rule(interval: str) -> str:
    return re.sub(r"m$", "min", interval)  # pandas wants "15min", not "15m"


def _off(index: pd.DatetimeIndex, interval: str):
    """The rows whose open time is not on the interval's grid."""
    return index != index.floor(_rule(interval))


def ends(symbols: list[str], interval: str = "5m", store: Path = STORE) -> dict[str, str | None]:
    """The last bar in the store of each symbol, None where it has no file — for the cache stamps.

    A dataset is a function of the store as much as of its arguments. On 2026-09-27 a step2 build
    over fifteen pairs, two of them fetched that day and thirteen last updated on 2026-09-02, came
    out with 24 days at the end whose cross-section was two pairs wide; the rebuild after bringing
    the thirteen up to date had the same arguments, the same universe, and so the same stamp. A
    store that grows changes which rows exist without changing anything a caller passes, so where
    it ends goes into the stamp and a cache built on another store is refused.

    Reads the index alone, a few hundredths of a second per symbol.
    """
    out = {}
    for s in symbols:
        path = store / f"{s}USDT-{interval}.parquet"
        out[s] = str(pd.read_parquet(path, columns=[]).index[-1]) if path.exists() else None
    return out


def _keys(prefix: str) -> list[str]:
    """ZIP names under an S3 prefix. A listing page holds 1000 keys and the daily folder of an old
    pair holds more, so callers must narrow the prefix down to the months they want."""
    xml = requests.get(LISTING, params={"delimiter": "/", "prefix": prefix}, timeout=30).text
    return sorted({k.rsplit("/", 1)[-1] for k in re.findall(r"<Key>([^<]+\.zip)</Key>", xml)})


def _to_utc(open_time: pd.Series) -> pd.DatetimeIndex:
    """Binance switched open_time from milliseconds to microseconds during 2025, without warning
    and without a version marker, so the unit is inferred per file from the magnitude."""
    if open_time.empty:
        return pd.DatetimeIndex([], tz="UTC")
    unit = "us" if open_time.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(open_time, unit=unit, utc=True)


def parse(content: bytes) -> pd.DataFrame:
    """One dump ZIP into OHLCV indexed by UTC timestamp."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLUMNS)
    # Some files ship a header row; coercing then dropping is cheaper than sniffing.
    raw["open_time"] = pd.to_numeric(raw.open_time, errors="coerce")
    raw = raw.dropna(subset=["open_time"])
    return raw.set_index(_to_utc(raw.open_time))[KEEP].astype(
        {"open": "float64", "high": "float64", "low": "float64", "close": "float64",
         "volume": "float32", "quote_volume": "float32", "trades": "int32"}
    )  # fmt: skip


def _fetch(url: str) -> pd.DataFrame | None:
    """One dump file. Returns None when the file is not published (yet)."""
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return parse(r.content)


def urls(symbol: str, interval: str, since: str | None = None) -> list[str]:
    """Every dump needed to reach today: the monthly files, then the daily ones for the months
    published only as days yet. `since` (YYYY-MM) drops everything strictly before it."""
    pair = f"{symbol}USDT"
    months = {k: re.search(r"(\d{4}-\d{2})\.zip$", k) for k in _keys(f"data/spot/monthly/klines/{pair}/{interval}/")}
    months = {k: m.group(1) for k, m in months.items() if m}
    if not months:
        return []

    last = pd.Period(max(months.values()), "M")
    if since:
        months = {k: m for k, m in months.items() if m >= since}

    # Months Binance has not consolidated into a monthly file yet exist only as daily files. One
    # listing per month keeps every request well under the 1000-key page limit.
    tail = pd.period_range(last + 1, pd.Period(pd.Timestamp.utcnow(), "M"), freq="M")
    days = [
        k
        for m in tail
        for k in _keys(f"data/spot/daily/klines/{pair}/{interval}/{pair}-{interval}-{m}-")
        if re.search(r"\d{4}-\d{2}-\d{2}\.zip$", k)
    ]
    return [f"{BASE}/monthly/klines/{pair}/{interval}/{k}" for k in sorted(months)] + [
        f"{BASE}/daily/klines/{pair}/{interval}/{k}" for k in sorted(days)
    ]


def update(symbol: str, interval: str = "5m", store: Path = STORE) -> pd.DataFrame:
    """Bring one symbol's Parquet up to date and return the full series."""
    path = store / f"{symbol}USDT-{interval}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    # Re-fetch the month the store ends in: it was almost certainly still partial when written.
    since = old.index[-1].strftime("%Y-%m") if old is not None and len(old) else None

    todo = urls(symbol, interval, since)
    with ThreadPoolExecutor(WORKERS) as pool:
        parts = [p for p in pool.map(_fetch, todo) if p is not None]
    if not parts and old is None:
        raise RuntimeError(f"no data for {symbol}USDT {interval}")

    df = pd.concat(([old] if old is not None else []) + parts)
    # Off-grid rows give way to the daily dumps of their days, which Binance publishes on the grid.
    # Checked on the whole series, `old` included, so re-running heals a store written before this.
    off = _off(df.index, interval)
    days = sorted(set(df.index[off].strftime("%Y-%m-%d")))
    pair = f"{symbol}USDT"
    with ThreadPoolExecutor(WORKERS) as pool:
        daily = pool.map(_fetch, [f"{BASE}/daily/klines/{pair}/{interval}/{pair}-{interval}-{d}.zip" for d in days])
        df = pd.concat([df[~off], *[p for p in daily if p is not None]])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[~_off(df.index, interval)]  # what no daily file replaced
    store.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _selfcheck() -> None:
    """The grid test on the stamps February 2018 shipped, and `load` refusing a store that has one."""
    import tempfile

    t = pd.DatetimeIndex(["2018-02-09 09:55", "2018-02-09 09:58:14.789", "2018-02-09 10:00"], tz="UTC")
    assert list(_off(t, "5m")) == [False, True, False]
    assert list(_off(t, "15m")) == [True, True, False]
    bars = pd.DataFrame({c: 1.0 for c in OHLC}, index=t)
    with tempfile.TemporaryDirectory() as d:
        bars.to_parquet(Path(d) / "XUSDT-5m.parquet")
        try:
            load("X", store=Path(d))
        except SystemExit as e:
            assert "1 bars off the 5m grid, from 2018-02-09 09:58:14.789" in str(e)
        else:
            raise AssertionError("an off-grid store was read")
        bars.drop(t[1]).to_parquet(Path(d) / "XUSDT-5m.parquet")
        assert len(load("X", "15m", store=Path(d))) == 2


def main() -> None:
    _selfcheck()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    total = 0
    for i, symbol in enumerate(args.symbols, 1):
        df = update(symbol, args.interval, args.store)
        total += len(df)
        gaps = df.index.to_series().diff().value_counts()
        expected = gaps.index[0]
        print(
            f"[{i}/{len(args.symbols)}] {symbol}USDT  {len(df):>9,} bars  "
            f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}  "
            f"{gaps.iloc[1:].sum():>5,} gaps > {expected}"
        )
    print(f"\n{total:,} rows in {args.store.resolve()}")


if __name__ == "__main__":
    main()
