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
