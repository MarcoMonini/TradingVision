"""Binance USDⓈ-M futures on the 15m grid: funding, open interest, positioning, taker flow, book depth.

The spot candles say what the price did. The futures market says who was positioned for it and how
hard they pushed, and it is where most crypto price discovery happens, so it is the first place to
look for information about the *next* leg rather than the current one. Everything here comes from
the same public dumps as the candles (`data.binance.vision`, no key), from the futures side:

- `fundingRate` (monthly, one row per 8h settlement): the rate longs pay shorts. Known once settled.
- `metrics` (daily, every 5 minutes): open interest in contracts and in USD, the long/short ratio of
  the top traders by position and of all accounts, and the taker buy/sell volume ratio.
- `bookDepth` (daily, a snapshot about every 30 seconds): the cumulative notional resting within
  1, 2, 3, 4 and 5% of the price, bids and asks apart. An aggregated order book, not the full L2.
- `klines` (monthly, 15m): the futures close and the taker buy volume.

Not published for this period: the best bid and ask (`bookTicker`) and the liquidations.

What they are worth on v2's turns is `detect --futures`: nothing at the alarm, and two columns with
a small, stable IC against the forward return (open interest behind the move, the book's
imbalance within 5%).

The chart page draws the open interest on its own bars (`open_interest_rows`, `open_interest`): the
dumps for the history, the REST statistics for the hours not dumped yet, in coins. The store keeps
the value in USD, whose change carries the bar's own return (`open_interest`).

**Every column of bar t uses only rows stamped before t + 15m**, the bar's close, because that is
when the v2 prediction of bar t is known. A row stamped exactly on the close belongs to the next bar.
Funding is carried forward from its settlement; the 5-minute metrics and the depth snapshots are
the last (levels) or the mean (flows, imbalances) of the rows inside the bar.

    uv run python -m tradingvision.data.futures --symbols BTC ETH SOL --start 2025-05-01 --end 2026-09-26
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from tradingvision.data.binance import STORE, WORKERS

BASE = "https://data.binance.vision/data/futures/um"
BAR = "15min"
LEVELS = (1, 2, 5)  # the depth bands kept, in % from the price


def _csv(url: str, timeout: float = 120) -> pd.DataFrame | None:
    """One dump ZIP as a frame, None when it is not published."""
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]))


def _time(col: pd.Series) -> pd.DatetimeIndex:
    """Epoch milliseconds or microseconds (Binance changed unit in 2025), or a datetime string, to UTC."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="us" if col.iloc[0] > 1e14 else "ms", utc=True)
    return pd.to_datetime(col, utc=True)


def _bar(when: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The 15m bar a row stamped `when` is known in: its open time. A row on the close is the next bar's."""
    return when.floor(BAR)


def _fetch(urls: list[str], timeout: float = 120) -> pd.DataFrame:
    with ThreadPoolExecutor(WORKERS) as pool:
        parts = [p for p in pool.map(lambda url: _csv(url, timeout), urls) if p is not None and len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def depth_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Snapshots of `bookDepth` to bars: the bid/ask imbalance of the notional within 1, 2 and 5%,
    averaged over the bar, and the log of the notional resting within 1% on both sides."""
    wide = raw.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    snap = pd.DataFrame(index=_time(pd.Series(wide.index)))
    for k in LEVELS:
        bid, ask = wide[-k].to_numpy(), wide[k].to_numpy()
        snap[f"book{k}"] = (bid - ask) / (bid + ask)
    snap["depth1"] = np.log(wide[-1].to_numpy() + wide[1].to_numpy())
    return snap.groupby(_bar(snap.index)).mean()


def metrics_features(raw: pd.DataFrame) -> pd.DataFrame:
    """The 5-minute `metrics` to bars: open interest and the two long/short ratios as the last
    value inside the bar, the taker buy/sell ratio as the mean of its log."""
    m = raw.set_index(_time(raw.create_time)).sort_index()
    g = m.groupby(_bar(m.index))
    return pd.DataFrame(
        {
            "oi": g.sum_open_interest_value.last().astype(float),
            "top_ls": g.sum_toptrader_long_short_ratio.last().astype(float),
            "ls": g.count_long_short_ratio.last().astype(float),
            "taker_ls": g.sum_taker_long_short_vol_ratio.apply(lambda v: np.log(v.astype(float)).mean()),
        }
    )


def open_interest(raw: pd.DataFrame, index: pd.DatetimeIndex, bar: pd.Timedelta) -> pd.DataFrame:
    """5-minute open interest rows onto the bars of `index`: the last row inside each bar, in coins
    (`coins`) and in USD (`usd`), NaN where a bar holds none.

    A row belongs to the bar whose open is the last at or before its stamp, so a row stamped exactly
    on a close is the next bar's, as in `metrics_features`. Mapped through `index` rather than
    floored on a fixed grid, so the rows land on whatever grid the chart page's bars are on: under a
    floor, daily bars that open at any hour but midnight UTC would miss every row.

    `coins` is the quantity to read. The value in USD is the coins times the price, so its change
    carries the bar's own return: a 2% rise with not one contract opened reads as open interest up
    2%. `metrics_features` keeps the USD value, and every column `detect` and `events` build on it
    (open interest behind the move, the shocks split by its sign) carries that term.
    """
    out = pd.DataFrame(np.nan, index=index, columns=["coins", "usd"])
    if raw.empty or index.empty:
        return out
    t = pd.DatetimeIndex(_time(raw.create_time))
    at = index.searchsorted(t, side="right") - 1
    rows = pd.DataFrame(
        {
            "t": t,
            "at": at,
            "coins": raw.sum_open_interest.astype(float).to_numpy(),
            "usd": raw.sum_open_interest_value.astype(float).to_numpy(),
        }
    )[(at >= 0) & (t < index[-1] + bar)]
    last = rows.sort_values("t").groupby("at")[["coins", "usd"]].last()
    out.iloc[last.index.to_numpy()] = last.to_numpy()
    return out


# The chart page's open interest, fetched for the bars on screen rather than read from the store: the
# page is stateless (Dockerfile). The dumps carry the history, from 2020-09 for BTCUSDT and 2021-12 for
# ETH and SOL, one file a day published the day after; the REST statistics cover the hours since the
# last file. They keep one month, so `REST_DAYS` stays a day inside it, and Binance refuses them from
# the United States (HTTP 451), where the dumps are still served.
REST = "https://fapi.binance.com/futures/data/openInterestHist"
REST_DAYS = 29
REST_PAGE = 500  # the most rows one call returns: 41.7 hours of 5-minute rows
TIMEOUT = 20  # a page waits on this, where `build`'s 120 s waits on a batch download
OI_COLUMNS = ["create_time", "sum_open_interest", "sum_open_interest_value"]


def _rest(pair: str, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """The REST statistics of `pair` from `since` to `until`, as rows in the dumps' columns."""
    since = max(since, until - pd.Timedelta(days=REST_DAYS))
    step = pd.Timedelta(minutes=5) * REST_PAGE
    rows = []
    while since < until:
        r = requests.get(
            REST,
            params={
                "symbol": pair,
                "period": "5m",
                "limit": REST_PAGE,
                "startTime": int(since.timestamp() * 1000),
                "endTime": int(min(since + step, until).timestamp() * 1000),
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        rows += r.json()
        since += step
    return _rest_rows(rows)


def _rest_rows(rows: list[dict]) -> pd.DataFrame:
    """The REST endpoint's JSON as the dumps' columns, the stamp as a UTC datetime."""
    return pd.DataFrame(
        {
            "create_time": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True),
            "sum_open_interest": [float(r["sumOpenInterest"]) for r in rows],
            "sum_open_interest_value": [float(r["sumOpenInterestValue"]) for r in rows],
        }
    )


def open_interest_rows(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Every 5-minute open interest row of `{symbol}USDT` from `start` to `end`, and the hosts that
    did not answer. A pair with no USDⓈ-M perpetual comes back empty with no host named.

    Either host can be unreachable from where the page runs, and each gives what it has without the
    other. The first file is fetched alone: a host blocked from here answers nothing, and the pool
    would wait out the timeout on every file before the error reached this function.
    """
    pair = f"{symbol}USDT"
    days = pd.date_range(start.normalize(), end.normalize(), freq="D").strftime("%Y-%m-%d")
    urls = [f"{BASE}/daily/metrics/{pair}/{pair}-metrics-{d}.zip" for d in days]
    parts, down = [], []
    try:
        parts = [_csv(urls[0], TIMEOUT), _fetch(urls[1:], TIMEOUT)]
    except requests.RequestException:
        down.append("data.binance.vision")
    parts = [p[OI_COLUMNS].assign(create_time=_time(p.create_time)) for p in parts if p is not None and len(p)]
    dumped = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OI_COLUMNS)
    since = dumped.create_time.max() + pd.Timedelta(minutes=5) if len(dumped) else start
    try:
        parts.append(_rest(pair, since, end))
    except requests.HTTPError as e:
        # 400 is a symbol Binance does not list as a perpetual: an answer, not an outage.
        if e.response is None or e.response.status_code != 400:
            down.append("fapi.binance.com")
    except requests.RequestException:
        down.append("fapi.binance.com")
    parts = [p for p in parts if len(p)]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OI_COLUMNS)), down


def build(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Every column on the 15m grid from `start` to `end`, for one symbol."""
    pair = f"{symbol}USDT"
    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
    months = pd.period_range(start, end, freq="M").strftime("%Y-%m")
    grid = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta("1D"), freq=BAR, tz="UTC", inclusive="left")

    k = _fetch([f"{BASE}/monthly/klines/{pair}/15m/{pair}-15m-{m}.zip" for m in months])
    k = k[pd.to_numeric(k.open_time, errors="coerce").notna()]
    k = k.set_index(_time(pd.to_numeric(k.open_time)))
    out = pd.DataFrame(index=grid)
    out["fut_close"] = k.close.astype(float)
    out["taker_buy"] = k.taker_buy_volume.astype(float) / k.volume.astype(float)

    f = _fetch([f"{BASE}/monthly/fundingRate/{pair}/{pair}-fundingRate-{m}.zip" for m in months])
    funding = f.set_index(_time(f.calc_time)).last_funding_rate.astype(float).sort_index()
    out["funding"] = funding.groupby(_bar(funding.index)).last().reindex(grid).ffill()

    out = out.join(metrics_features(_fetch([f"{BASE}/daily/metrics/{pair}/{pair}-metrics-{d}.zip" for d in days])))
    out = out.join(depth_features(_fetch([f"{BASE}/daily/bookDepth/{pair}/{pair}-bookDepth-{d}.zip" for d in days])))
    return out.rename_axis("open_time")


def load(symbol: str) -> pd.DataFrame:
    """The futures columns of one symbol, as `build` stored them."""
    return pd.read_parquet(STORE / f"{symbol}USDT-futures-15m.parquet")


def _selfcheck() -> None:
    """Bar assignment and the imbalance arithmetic on hand-made rows."""
    t = pd.DatetimeIndex(["2025-06-01 00:14:59", "2025-06-01 00:15:00"], tz="UTC")
    assert list(_bar(t)) == list(pd.DatetimeIndex(["2025-06-01 00:00", "2025-06-01 00:15"], tz="UTC"))
    # Two snapshots in the first bar, one in the next: bids heavier by 3:1 within 1%, then balanced.
    rows = []
    for ts, bid, ask in (("2025-06-01 00:00:10", 300.0, 100.0), ("2025-06-01 00:10:00", 300.0, 100.0)) + (
        ("2025-06-01 00:15:00", 200.0, 200.0),
    ):
        for p in (1, 2, 3, 4, 5):
            rows += [(ts, -p, 0.0, bid * p), (ts, p, 0.0, ask * p)]
    d = depth_features(pd.DataFrame(rows, columns=["timestamp", "percentage", "depth", "notional"]))
    assert len(d) == 2 and np.allclose(d.book1, [0.5, 0.0]) and np.allclose(d.book5, [0.5, 0.0])
    assert np.isclose(d.depth1.iloc[0], np.log(400.0))
    # Metrics: the level is the last row inside the bar, never the one stamped on its close.
    m = pd.DataFrame(
        {
            "create_time": ["2025-06-01 00:00:00", "2025-06-01 00:10:00", "2025-06-01 00:15:00"],
            "sum_open_interest_value": [1.0, 2.0, 3.0],
            "sum_toptrader_long_short_ratio": [1.0, 1.0, 1.0],
            "count_long_short_ratio": [1.0, 1.0, 1.0],
            "sum_taker_long_short_vol_ratio": [1.0, np.e, 1.0],
        }
    )
    got = metrics_features(m)
    assert list(got.oi) == [2.0, 3.0] and np.allclose(got.taker_ls, [0.5, 0.0])

    # The page's open interest: the same bar rule on the page's own bars, from either source.
    rest = _rest_rows(
        [
            {"symbol": "BTCUSDT", "sumOpenInterest": "40.0", "sumOpenInterestValue": "4000.0", "timestamp": ms}
            for ms in (pd.Timestamp("2025-06-01 00:15", tz="UTC").value // 10**6,)
        ]
    )
    assert rest.create_time.iloc[0] == pd.Timestamp("2025-06-01 00:15", tz="UTC") and rest.sum_open_interest[0] == 40
    dumped = pd.DataFrame(
        {
            # Before the first bar, twice inside it (one row duplicated, as in BTCUSDT's 2021-03-01 file), past
            # the last bar's close: the two ends are dropped.
            "create_time": ["2025-05-31 23:55:00", "2025-06-01 00:00:00", "2025-06-01 00:10:00"]
            + ["2025-06-01 00:10:00", "2025-06-01 00:30:00"],
            "sum_open_interest": [9.0, 10.0, 20.0, 20.0, 50.0],
            "sum_open_interest_value": [900.0, 1000.0, 2000.0, 2000.0, 5000.0],
        }
    )
    bars = pd.date_range("2025-06-01", periods=2, freq="15min", tz="UTC")
    oi = open_interest(pd.concat([dumped.assign(create_time=_time(dumped.create_time)), rest]), bars, bars.freq)
    assert list(oi.coins) == [20.0, 40.0] and list(oi.usd) == [2000.0, 4000.0], "the last row inside each bar"
    # A bar with no row is a gap, not the previous bar's level carried forward.
    three = open_interest(dumped, pd.date_range("2025-06-01", periods=3, freq="15min", tz="UTC"), bars.freq)
    assert three.coins.iloc[0] == 20.0 and np.isnan(three.coins.iloc[1]) and three.coins.iloc[2] == 50.0
    # Daily bars that open at 05:00 UTC: a row belongs to the bar it falls inside, not to a UTC day.
    days = pd.DatetimeIndex(["2025-06-01 05:00", "2025-06-02 05:00"], tz="UTC")
    d = pd.DataFrame(
        {
            "create_time": ["2025-06-01 04:55:00", "2025-06-02 04:55:00", "2025-06-02 05:00:00", "2025-06-03 05:00:00"],
            "sum_open_interest": [1.0, 2.0, 3.0, 4.0],
            "sum_open_interest_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert list(open_interest(d, days, pd.Timedelta("1D")).coins) == [2.0, 3.0]
    assert open_interest(pd.DataFrame(columns=OI_COLUMNS), days, pd.Timedelta("1D")).coins.isna().all()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default="2026-09-26")
    args = ap.parse_args()
    _selfcheck()
    for symbol in args.symbols:
        df = build(symbol, args.start, args.end)
        df.to_parquet(STORE / f"{symbol}USDT-futures-15m.parquet")
        print(f"{symbol}USDT  {len(df):,} bars, missing per column:")
        print(df.isna().mean().round(4).to_string())


if __name__ == "__main__":
    main()
