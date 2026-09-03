"""Bulk OHLCV history from the Binance public data dumps.

Alpaca's crypto history only starts 2021-01-01 (its own venue went live then) and is holed by the
2023 delistings, so the ML dataset is built from `data.binance.vision` instead: static ZIPs on S3,
no API key, no rate limit, USDT pairs from 2017. Alpaca stays the live/execution feed.

    python -m tradingvision.data.binance                 # update every symbol to yesterday
    python -m tradingvision.data.binance --symbols BTC ETH --interval 15m

Re-running only fetches what is missing: the store keeps one Parquet per (symbol, interval) and
the last partial month is always re-fetched, so an interrupted run heals itself.
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

# 20 USDT pairs with at least 70 months of 5m history and no missing month.
SYMBOLS = [
    "BTC", "ETH", "LTC", "ADA", "XRP", "TRX", "LINK", "BAT", "DOGE", "XTZ",
    "BCH", "YFI", "DOT", "SOL", "CRV", "UNI", "AVAX", "SUSHI", "NEAR", "AAVE",
]  # fmt: skip

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
    df = pd.read_parquet(store / f"{symbol}USDT-{stored}.parquet")
    if timeframe == stored:
        return df
    rule = re.sub(r"m$", "min", timeframe)  # pandas wants "15min", not "15m"
    return df.resample(rule).agg(OHLC).dropna(subset=["open"])


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
    df = df[~df.index.duplicated(keep="last")].sort_index()
    store.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def main() -> None:
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
