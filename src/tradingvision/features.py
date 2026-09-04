"""The 28 candidate feature columns of the swing dataset, one row per bar.

All causal: every rolling window looks only backwards from the current bar, never centred. The
asymmetry with the target is deliberate — pivots see the future because they are the label, the
features do not because they are the input.

Every window derives from `N`, which derives from `EXTREMA_WINDOW` on the branch timeframe: an
indicator has to look over the same horizon as the leg the target describes, not over a convention
borrowed from another problem. The per-branch values are still to be measured (see the spec), so
`N` is a plain argument here and defaults to the reference window.

Column names are the spec's 28 candidates spelled out as descriptive identifiers, grouped by the
spec's families; the definitions are unchanged.

These 28 are candidates, not the final set: the spec reduces them by correlation and permutation
importance, expecting 7-12 survivors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import KAMAIndicator, RSIIndicator, TSIIndicator
from ta.trend import ADXIndicator, PSARIndicator
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

from tradingvision.data.pivots import EXTREMA_WINDOW

# Families, in spec order. Drives the chart grouping and documents what each column measures.
FAMILIES: dict[str, tuple[str, ...]] = {
    "price": ("log_return", "cum_log_return_window", "candle_body_pct"),
    "volatility": (
        "bar_range_pct",
        "window_range_pct",
        "average_true_range_pct",
        "realized_volatility",
        "volatility_expansion",
    ),
    "structure": ("upper_wick_pct", "lower_wick_pct", "close_position_in_bar"),
    "position": (
        "close_position_in_window",
        "distance_from_window_high_pct",
        "distance_from_window_low_pct",
        "age_of_window_high",
        "age_of_window_low",
    ),
    "volume": (
        "volume_vs_median",
        "signed_volume",
        "volume_trend",
        "on_balance_volume_zscore",
        "distance_from_vwap_pct",
    ),
    "trend": ("distance_from_kama_pct", "distance_from_ema_pct", "ema_slope", "distance_from_psar_pct"),
    "momentum": ("adx_trend_strength", "tsi_momentum", "rsi_centered"),
}
COLUMNS = [c for cols in FAMILIES.values() for c in cols]

# Chart labels: the identifier is what the dataset carries, this is what a reader sees on a plot.
LABELS = {
    "log_return": "Log Return",
    "cum_log_return_window": "Cumulative Log Return (N)",
    "candle_body_pct": "Candle Body (%)",
    "bar_range_pct": "Bar Range (%)",
    "window_range_pct": "Window Range (%)",
    "average_true_range_pct": "Average True Range (%)",
    "realized_volatility": "Realized Volatility",
    "volatility_expansion": "Volatility Expansion",
    "upper_wick_pct": "Upper Wick (%)",
    "lower_wick_pct": "Lower Wick (%)",
    "close_position_in_bar": "Close Position in Bar",
    "close_position_in_window": "Close Position in Window",
    "distance_from_window_high_pct": "Distance from Window High (%)",
    "distance_from_window_low_pct": "Distance from Window Low (%)",
    "age_of_window_high": "Age of Window High",
    "age_of_window_low": "Age of Window Low",
    "volume_vs_median": "Volume vs Median",
    "signed_volume": "Signed Volume",
    "volume_trend": "Volume Trend",
    "on_balance_volume_zscore": "On-Balance Volume (z-score)",
    "distance_from_vwap_pct": "Distance from VWAP (%)",
    "distance_from_kama_pct": "Distance from KAMA (%)",
    "distance_from_ema_pct": "Distance from EMA (%)",
    "ema_slope": "EMA Slope",
    "distance_from_psar_pct": "Distance from PSAR (%)",
    "adx_trend_strength": "ADX Trend Strength",
    "tsi_momentum": "TSI Momentum",
    "rsi_centered": "RSI (centered)",
}


def _bars_since(s: pd.Series, n: int, *, high: bool) -> pd.Series:
    """Age of the window extreme, in [0, 1]: 0 on the bar that set it, 1 at the far end."""
    pick = np.argmax if high else np.argmin
    return s.rolling(n).apply(lambda a: n - 1 - pick(a), raw=True) / n


def features(df: pd.DataFrame, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The candidate columns for one OHLCV frame, indexed like `df`.

    Leading rows are NaN until every window is filled; the dataset drops them as warm-up.
    """
    o, h, low, c, v = df.open, df.high, df.low, df.close, df.volume
    short = max(n // 4, 2)  # the "N/4" of the spec, floored so a std over it is defined

    ret = np.log(c / c.shift())
    max_h, min_l = h.rolling(n).max(), low.rolling(n).min()
    bar_range = (h - low).replace(0, np.nan)  # a bar with no range has no location inside it
    ema = c.ewm(span=n, adjust=False).mean()
    obv = (v * np.sign(ret)).rolling(n).sum()

    f = {
        "log_return": ret,
        "cum_log_return_window": np.log(c / c.shift(n)),
        "bar_range_pct": (h - low) / c,
        "candle_body_pct": (c - o) / c,
        "upper_wick_pct": (h - np.maximum(o, c)) / c,
        "lower_wick_pct": (np.minimum(o, c) - low) / c,
        "close_position_in_bar": (c - low) / bar_range,
        "close_position_in_window": 2 * (c - min_l) / (max_h - min_l) - 1,
        "distance_from_window_high_pct": np.log(c / max_h),
        "distance_from_window_low_pct": np.log(c / min_l),
        "age_of_window_high": _bars_since(h, n, high=True),
        "age_of_window_low": _bars_since(low, n, high=False),
        "window_range_pct": np.log(max_h / min_l),
        "average_true_range_pct": AverageTrueRange(h, low, c, window=n).average_true_range() / c,
        "realized_volatility": ret.rolling(n).std(),
        "volatility_expansion": ret.rolling(short).std() / ret.rolling(n).std(),
        "volume_vs_median": v / v.rolling(n).median(),
        "volume_trend": np.log(v.rolling(short).mean() / v.rolling(n).mean()),
        # Rolling, not cumulated from the start of the series: a running total is not stationary.
        "on_balance_volume_zscore": (obv - obv.rolling(n).mean()) / obv.rolling(n).std(),
        "distance_from_vwap_pct": np.log(
            c / VolumeWeightedAveragePrice(h, low, c, v, window=n).volume_weighted_average_price()
        ),
        "distance_from_kama_pct": np.log(c / KAMAIndicator(c, window=n).kama()),
        "distance_from_ema_pct": np.log(c / ema),
        "ema_slope": np.log(ema / ema.shift(short)),
        # On a positional index: ta assigns into the PSAR series by position, which a
        # DatetimeIndex turns into a deprecation warning on every bar.
        "distance_from_psar_pct": (
            c - PSARIndicator(*(s.reset_index(drop=True) for s in (h, low, c))).psar().set_axis(c.index)
        )
        / c,
        # Known fixed ranges, so dividing by a constant puts them on the same scale as the % columns
        # with no statistic to estimate and no leakage. The rest need robust scaling fitted on train.
        "adx_trend_strength": ADXIndicator(h, low, c, window=n).adx() / 100,
        "tsi_momentum": TSIIndicator(c, window_slow=n, window_fast=short).tsi() / 100,
        "rsi_centered": RSIIndicator(c, window=n).rsi() / 50 - 1,
    }
    f["signed_volume"] = f["volume_vs_median"] * np.sign(ret)
    return pd.DataFrame(f)[COLUMNS]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=pd.date_range("2024", periods=500, freq="5min")
    )
    df = pd.DataFrame(
        {
            "open": close.shift().fillna(close.iloc[0]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.lognormal(0, 1, 500),
        }
    )
    out = features(df)
    assert list(out.columns) == COLUMNS and len(COLUMNS) == 28, "28 columns, spec order"
    assert list(LABELS) == COLUMNS, "every column needs a chart label"
    tail = out.iloc[EXTREMA_WINDOW * 4 :]
    assert tail.notna().all().all(), f"NaN past warm-up: {tail.columns[tail.isna().any()].tolist()}"
    assert np.isfinite(tail.to_numpy()).all(), "infinities"
    assert tail.close_position_in_bar.between(0, 1).all() and tail.close_position_in_window.between(-1, 1).all()
    assert tail.distance_from_window_high_pct.le(1e-12).all() and tail.distance_from_window_low_pct.ge(-1e-12).all()
    assert tail.age_of_window_high.between(0, 1).all() and tail.adx_trend_strength.between(0, 1).all()
    assert tail.rsi_centered.between(-1, 1).all() and tail.tsi_momentum.between(-1, 1).all()
    # Causality: a feature must not move when a later bar changes.
    cut = 300
    assert np.allclose(features(df.iloc[:cut]).iloc[-1].to_numpy(), out.iloc[cut - 1].to_numpy(), equal_nan=True)
    print(f"ok — {len(COLUMNS)} columns, {len(tail)} rows past warm-up")
