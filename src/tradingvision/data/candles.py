"""Candele crypto da Alpaca.

L'endpoint crypto e' pubblico: `CryptoHistoricalDataClient` senza credenziali funziona, quindi
niente chiavi da configurare per la sola lettura dello storico.
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
    """OHLCV indicizzato per timestamp UTC, dalle ultime `days` giornate a oggi.

    DataFrame vuoto se Alpaca non ha dati per la coppia richiesta.
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
    # L'indice e' (symbol, timestamp): con un simbolo solo il primo livello e' rumore.
    return bars.droplevel("symbol").sort_index()
