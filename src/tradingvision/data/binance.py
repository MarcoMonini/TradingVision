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
import tempfile
import time
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
RETRIES = 5

# Raw dump layout: no header, 12 columns. We keep the 9 that carry information.
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]  # fmt: skip
# The taker columns are the volume bought by aggressive buyers, in base and in quote: the only
# order-flow signal the spot dumps carry, and what `flow` decomposes (route 7, HANDOFF §19). Kept
# from 2026-10-07; a store written before then has the first 7 only, and `update` refuses to extend
# such a file rather than append rows that have the columns to rows that do not.
#
# No cache stamp changes with them. The one stamped cache, `swing.cached`, holds rows built by
# `features`, `legs` and the label, which read open, high, low, close and volume and nothing else,
# so its rows are the same whether the file has these columns or not; and the re-download that
# brings them in ends later than the store before it, which `ends` already puts in that stamp. The
# re-download's OHLCV was checked equal to the store it replaces on every common bar of the fifteen
# SYMBOLS, 2026-10-07. Nothing else caches a function of the store.
TAKER = ["taker_buy_base", "taker_buy_quote"]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades", *TAKER]


OHLC = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "trades": "sum",
    "taker_buy_base": "sum", "taker_buy_quote": "sum",
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
    # Aggregates the columns the file has: a store written before the taker columns were kept still
    # loads, without them, and a reader that needs them (`flow.bars`) says so instead of meeting NaN.
    return df.resample(rule).agg({k: v for k, v in OHLC.items() if k in df}).dropna(subset=["open"])


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
         "volume": "float32", "quote_volume": "float32", "trades": "int32",
         "taker_buy_base": "float32", "taker_buy_quote": "float32"}
    )  # fmt: skip


def _fetch(url: str) -> pd.DataFrame | None:
    """One dump file. Returns None when the file is not published (yet).

    Retried on a dropped or stalled connection: a full download is a few thousand files, and on
    2026-10-07, at about 50 KB/s a connection, one read timeout among them killed the whole run.
    """
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            break
        except (requests.ConnectionError, requests.Timeout):  # a timeout mid-body is a ConnectionError
            if attempt == RETRIES - 1:
                raise
            time.sleep(2**attempt)
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
    if old is not None and set(KEEP) - set(old.columns):
        # Extending it would leave years of NaN under the taker columns, which a mean over a
        # period then reads as a number. A new store is a full download (`--store` elsewhere).
        raise RuntimeError(f"{path} lacks {sorted(set(KEEP) - set(old.columns))}: download into a fresh --store")
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


def _selfcheck() -> None:
    """A 12-column dump parsed in memory, both time units, the 15m sums, and the refusal to extend
    a file written before the taker columns. A temporary directory stands in for the store."""
    row = "{t},100.0,101.0,99.0,100.5,2.0,{c},200.0,9,0.5,50.0,0"

    def dump(*lines: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("x.csv", "\n".join(lines))
        return buf.getvalue()

    t0 = 1735689600000  # 2025-01-01 00:00 UTC, in ms
    ms = parse(dump(*(row.format(t=t0 + i * 300_000, c=t0 + i * 300_000 + 299_999) for i in range(6))))
    us = parse(dump(*(row.format(t=(t0 + i * 300_000) * 1000, c=0) for i in range(6))))
    assert ms.equals(us) and ms.index[0] == pd.Timestamp("2025-01-01", tz="UTC"), (ms.index[0], us.index[0])
    assert list(ms.columns) == KEEP and (ms.taker_buy_quote == 50.0).all()
    assert ms.taker_buy_base.dtype == ms.taker_buy_quote.dtype == ms.volume.dtype == "float32"
    with tempfile.TemporaryDirectory() as tmp:
        ms.to_parquet(Path(tmp) / "XUSDT-5m.parquet")
        q = load("X", "15m", store=Path(tmp))
        assert len(q) == 2 and (q.taker_buy_quote == 150.0).all() and (q.taker_buy_base == 1.5).all(), q
        assert (q.quote_volume == 600.0).all() and (q.trades == 27).all()
        # A file from before the columns were kept still loads, without them, and is not extended.
        ms.drop(columns=TAKER).to_parquet(Path(tmp) / "OLDUSDT-5m.parquet")
        assert list(load("OLD", "15m", store=Path(tmp)).columns) == KEEP[:7]
        try:
            update("OLD", store=Path(tmp))
        except RuntimeError as e:
            assert "taker_buy_base" in str(e)
        else:
            raise AssertionError("a store without the taker columns was extended")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--store", type=Path, default=STORE)
    args = ap.parse_args()

    _selfcheck()
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
