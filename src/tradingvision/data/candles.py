"""Crypto candles from Alpaca.

The crypto endpoint is public: `CryptoHistoricalDataClient` works without credentials, so reading
historical data needs no API keys.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

TIMEFRAMES = {
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "DOGE/USD"]

_client = CryptoHistoricalDataClient()


def get_candles(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """OHLCV indexed by UTC timestamp, from the last `days` days up to now.

    Empty DataFrame when Alpaca has no data for the requested pair.
    """
    bars = _client.get_crypto_bars(
        CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAMES[timeframe],
            start=datetime.now(timezone.utc) - timedelta(days=days),
        )
    ).df
    if bars.empty:
        return bars
    # The index is (symbol, timestamp): with a single symbol the first level is noise.
    return bars.droplevel("symbol").sort_index()


def panel(symbols: list[str], timeframe: str, days: int, bar: pd.Timedelta) -> dict[str, pd.DataFrame]:
    """Close and dollar volume of `symbols` as wide frames on one complete grid.

    What a cross-sectional model reads: a rank is a statement about a symbol against the others
    trading at that instant, so it needs the panel and not a series.

    **A bucket with no trade has to be said, not left out.** Alpaca emits no bar for a period in
    which a pair did not trade, so the union index carries a hole for that pair — and a rolling
    window is NaN for ever after a single hole inside it. Measured over sixty days of the five
    pairs on 15m: BTC misses nothing, ETH 76 bars, DOGE 63, LTC 48, SOL 25, and a thirty-day
    composite came back with one usable column out of five. A missing bucket means no trade, which
    is a flat close and a volume of zero, and that is what is written here.

    `ffill` and never `bfill`: the last trade is what a bar with no trade closes at, while a
    leading gap is a pair that did not exist yet and has to stay NaN rather than borrow a later
    price. A pair Alpaca serves nothing for does not join the panel at all.
    """
    close, volume = {}, {}
    for symbol in symbols:
        bars = get_candles(symbol, timeframe, days)
        if not bars.empty:
            close[symbol], volume[symbol] = bars.close, bars.volume
    if not close:
        return {"close": pd.DataFrame(), "dollar_volume": pd.DataFrame()}
    close, volume = pd.DataFrame(close).sort_index(), pd.DataFrame(volume).sort_index()
    grid = pd.date_range(close.index.min(), close.index.max(), freq=bar)
    close = close.reindex(grid).ffill()
    return {"close": close, "dollar_volume": volume.reindex(grid).fillna(0.0) * close}


def _selfcheck() -> None:
    """The gap rule, on a panel built by hand — the fetch itself needs the network."""
    import numpy as np

    bar = pd.Timedelta("15min")
    grid = pd.date_range("2025-01-01", periods=8, freq=bar, tz="UTC")
    full = pd.Series(np.arange(1.0, 9.0), index=grid)
    holed = full.drop(grid[[2, 5]])  # the bars that pair never traded in

    def fake(symbol, timeframe, days):
        s = full if symbol == "A" else holed
        return pd.DataFrame({"close": s, "high": s, "low": s, "open": s, "volume": s * 10})

    global get_candles
    real, get_candles = get_candles, fake
    try:
        p = panel(["A", "B"], "15m", 1, bar)
    finally:
        get_candles = real

    assert list(p["close"].index) == list(grid), "the grid is complete whatever each pair is missing"
    assert p["close"].notna().all().all(), "and no hole survives it"
    # A bar with no trade closes where the last one did, and moves the price by nothing.
    assert p["close"].B.iloc[2] == p["close"].B.iloc[1] and p["close"].B.iloc[5] == p["close"].B.iloc[4]
    assert np.log(p["close"]).diff().B.iloc[2] == 0.0, "no trade is no return"
    # And it carries no volume, which is the other half of the same statement.
    assert p["dollar_volume"].B.iloc[2] == 0.0 and p["dollar_volume"].B.iloc[5] == 0.0
    assert (p["dollar_volume"].A > 0).all()
    # The failure this exists to stop: one hole used to make a rolling window NaN for ever.
    assert p["close"].B.rolling(4).std().iloc[-1] == p["close"].B.rolling(4).std().iloc[-1]  # not NaN

    # A leading gap is a pair that did not exist yet and must not borrow a later price.
    late = full.iloc[3:]

    def fake_late(symbol, timeframe, days):
        s = full if symbol == "A" else late
        return pd.DataFrame({"close": s, "high": s, "low": s, "open": s, "volume": s * 10})

    real, get_candles = get_candles, fake_late
    try:
        q = panel(["A", "B"], "15m", 1, bar)
    finally:
        get_candles = real
    assert q["close"].B.iloc[:3].isna().all(), "history a pair does not have stays missing"
    assert q["close"].B.iloc[3:].notna().all()


if __name__ == "__main__":
    _selfcheck()
    print("ok — panel gaps read as no trade")
