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

from tradingvision.data import binance

TIMEFRAMES = {
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

# `TIMEFRAMES` as durations. Spelled out rather than parsed: `pd.Timedelta("15m")` is minutes but
# deprecated, and `pd.date_range(freq="15m")` is *months* — not an ambiguity to leave implicit.
# Here and not in the app, because `complete` below needs it and the app is not the only caller.
BAR = {
    "5m": pd.Timedelta("5min"),
    "15m": pd.Timedelta("15min"),
    "1h": pd.Timedelta("1h"),
    "4h": pd.Timedelta("4h"),
    "1d": pd.Timedelta("1D"),
}

# The study's twenty pairs, quoted in USD because that is what Alpaca serves where Binance serves
# USDT. `data.binance.SYMBOLS` is the same list and the same order, which is the point: a pair on
# the chart should be a pair the numbers in the spec were measured on, not a different universe.
#
# Which of them Alpaca actually lists is not something this file can know — the venue's coverage is
# narrower than Binance's and changes — so a pair it serves nothing for is not an error here. The
# page says so and moves on, and the combo box takes a typed pair as well as a listed one, so the
# list is a starting point rather than a ceiling.
SYMBOLS = [f"{base}/USD" for base in binance.SYMBOLS]

_client = CryptoHistoricalDataClient()


def complete(bars: pd.DataFrame, bar: pd.Timedelta) -> pd.DataFrame:
    """One pair's OHLCV on a gapless grid: a period with no trade gets a bar, not a hole.

    The same decision `panel` takes for the cross-section, made once and in one place. Alpaca emits
    nothing for a bucket in which a pair did not trade, and a series with holes in it is not the
    series any model here was fitted on — the store is Binance's complete 5m grid, so a window of
    24 bars means 24 *bar durations* there and means an unknown stretch of wall clock on a gappy
    index. Twenty-four 15m bars that quietly span nine hours are a different input from the one
    the weights were trained against, and nothing about the frame says so.

    What a no-trade bucket is: the price did not move because nobody moved it, so open, high, low
    and close are all the previous close and the volume is zero. `ffill` and never `bfill` — a
    leading hole is a pair that had not traded yet and has to stay out of the frame rather than
    borrow a later price, which is why the grid starts at the first real bar.

    The filled bars are visible rather than hidden: they carry a volume of exactly zero, which is
    what lets a caller count them and say how much of a window was silence.
    """
    if bars.empty:
        return bars
    grid = pd.date_range(bars.index.min(), bars.index.max(), freq=bar)
    out = bars.reindex(grid)
    out["close"] = out.close.ffill()
    for column in ("open", "high", "low"):
        # Not ffilled from their own column: a bucket with no trade has no range of its own, and
        # carrying the previous bar's high forward would invent a wick nobody printed — and a
        # barrier rule in `stops` reads exactly those highs and lows.
        out[column] = out[column].fillna(out.close)
    out["volume"] = out.volume.fillna(0.0)
    return out.rename_axis(bars.index.name)


def get_candles(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """OHLCV indexed by UTC timestamp, from the last `days` days up to now, on a gapless grid.

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
    return complete(bars.droplevel("symbol").sort_index(), BAR[timeframe])


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

    # --- `complete`, the same rule on one pair, which is what the chart draws -------------------
    one = pd.DataFrame({"open": holed, "high": holed + 1, "low": holed - 1, "close": holed, "volume": holed * 10})
    out = complete(one, bar)
    assert list(out.index) == list(grid) and out.notna().all().all()
    # A bucket with no trade is the previous close on all four prices — no range of its own. That
    # matters beyond tidiness: `stops` reads exactly these highs and lows for its barriers, and a
    # forward filled high would hand it a wick nobody printed and a stop nobody could have been hit
    # by. The bars either side keep the range they really had.
    for k in (2, 5):
        assert out.open.iloc[k] == out.high.iloc[k] == out.low.iloc[k] == out.close.iloc[k]
        assert out.close.iloc[k] == out.close.iloc[k - 1], "no trade is no return"
        assert out.volume.iloc[k] == 0.0, "and no volume"
    assert out.high.iloc[1] == one.high.iloc[1] and out.low.iloc[1] == one.low.iloc[1]
    # Zero volume is the marker a caller counts filled bars by, and a real bar never carries it.
    assert int((out.volume == 0).sum()) == 2
    # A frame with no holes comes back unchanged, so the fill is never a transform in disguise.
    whole = pd.DataFrame({"open": full, "high": full, "low": full, "close": full, "volume": full})
    assert complete(whole, bar).equals(whole.reindex(grid).rename_axis(whole.index.name))
    assert complete(pd.DataFrame(), bar).empty
    # The grid starts at the pair's own first bar: a leading hole is history it does not have.
    assert complete(one.iloc[2:], bar).index[0] == one.index[2]

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
