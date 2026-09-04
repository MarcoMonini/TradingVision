"""The 28 candidate feature columns of the swing dataset, one row per bar.

All causal: every rolling window looks only backwards from the current bar, never centred. The
asymmetry with the target is deliberate — pivots see the future because they are the label, the
features do not because they are the input.

Every window derives from `N`, which derives from `EXTREMA_WINDOW` on the branch timeframe: an
indicator has to look over the same horizon as the leg the target describes, not over a convention
borrowed from another problem. The per-branch values are still to be measured (see the spec), so
`N` is a plain argument here and defaults to the reference window.

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
    "price": ("ritorno_pct", "ritorno_cum_N", "body_pct"),
    "volatility": ("range_pct", "range_width_pct", "atr_pct", "vol_realizzata", "vol_ratio"),
    "structure": ("upper_wick_pct", "lower_wick_pct", "close_loc_bar"),
    "position": ("pos_in_range", "dist_max_pct", "dist_min_pct", "bars_since_max", "bars_since_min"),
    "volume": ("volume_rel", "volume_signed", "volume_trend", "obv_zscore", "vwap_dist_pct"),
    "trend": ("kama_dist_pct", "ema_dist_pct", "ema_slope", "psar_dist_pct"),
    "momentum": ("adx", "tsi", "rsi_centrato"),
}
COLUMNS = [c for cols in FAMILIES.values() for c in cols]


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
        "ritorno_pct": ret,
        "ritorno_cum_N": np.log(c / c.shift(n)),
        "range_pct": (h - low) / c,
        "body_pct": (c - o) / c,
        "upper_wick_pct": (h - np.maximum(o, c)) / c,
        "lower_wick_pct": (np.minimum(o, c) - low) / c,
        "close_loc_bar": (c - low) / bar_range,
        "pos_in_range": 2 * (c - min_l) / (max_h - min_l) - 1,
        "dist_max_pct": np.log(c / max_h),
        "dist_min_pct": np.log(c / min_l),
        "bars_since_max": _bars_since(h, n, high=True),
        "bars_since_min": _bars_since(low, n, high=False),
        "range_width_pct": np.log(max_h / min_l),
        "atr_pct": AverageTrueRange(h, low, c, window=n).average_true_range() / c,
        "vol_realizzata": ret.rolling(n).std(),
        "vol_ratio": ret.rolling(short).std() / ret.rolling(n).std(),
        "volume_rel": v / v.rolling(n).median(),
        "volume_trend": np.log(v.rolling(short).mean() / v.rolling(n).mean()),
        # Rolling, not cumulated from the start of the series: a running total is not stationary.
        "obv_zscore": (obv - obv.rolling(n).mean()) / obv.rolling(n).std(),
        "vwap_dist_pct": np.log(c / VolumeWeightedAveragePrice(h, low, c, v, window=n).volume_weighted_average_price()),
        "kama_dist_pct": np.log(c / KAMAIndicator(c, window=n).kama()),
        "ema_dist_pct": np.log(c / ema),
        "ema_slope": np.log(ema / ema.shift(short)),
        # On a positional index: ta assigns into the PSAR series by position, which a
        # DatetimeIndex turns into a deprecation warning on every bar.
        "psar_dist_pct": (c - PSARIndicator(*(s.reset_index(drop=True) for s in (h, low, c))).psar().set_axis(c.index))
        / c,
        # Known fixed ranges, so dividing by a constant puts them on the same scale as the % columns
        # with no statistic to estimate and no leakage. The rest need robust scaling fitted on train.
        "adx": ADXIndicator(h, low, c, window=n).adx() / 100,
        "tsi": TSIIndicator(c, window_slow=n, window_fast=short).tsi() / 100,
        "rsi_centrato": RSIIndicator(c, window=n).rsi() / 50 - 1,
    }
    f["volume_signed"] = f["volume_rel"] * np.sign(ret)
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
    tail = out.iloc[EXTREMA_WINDOW * 4 :]
    assert tail.notna().all().all(), f"NaN past warm-up: {tail.columns[tail.isna().any()].tolist()}"
    assert np.isfinite(tail.to_numpy()).all(), "infinities"
    assert tail.close_loc_bar.between(0, 1).all() and tail.pos_in_range.between(-1, 1).all()
    assert tail.dist_max_pct.le(1e-12).all() and tail.dist_min_pct.ge(-1e-12).all()
    assert tail.bars_since_max.between(0, 1).all() and tail.adx.between(0, 1).all()
    assert tail.rsi_centrato.between(-1, 1).all() and tail.tsi.between(-1, 1).all()
    # Causality: a feature must not move when a later bar changes.
    cut = 300
    assert np.allclose(features(df.iloc[:cut]).iloc[-1].to_numpy(), out.iloc[cut - 1].to_numpy(), equal_nan=True)
    print(f"ok — {len(COLUMNS)} columns, {len(tail)} rows past warm-up")
