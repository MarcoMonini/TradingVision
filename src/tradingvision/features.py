"""The 29 candidate feature columns of the swing dataset, one row per bar.

All causal: every rolling window looks only backwards from the current bar, never centred. The
asymmetry with the target is deliberate — pivots see the future because they are the label, the
features do not because they are the input.

Every window derives from `N`, which derives from `EXTREMA_WINDOW` on the branch timeframe: an
indicator has to look over the same horizon as the leg the target describes, not over a convention
borrowed from another problem. The per-branch values are still to be measured (see the spec), so
`N` is a plain argument here and defaults to the reference window.

Column names are the spec's 28 candidates spelled out as descriptive identifiers, grouped by the
spec's families, plus `log_dollar_volume` — the one column here whose *level* is the information,
added when the label became cross-sectional. The other definitions are unchanged.

These 29 are candidates, not the final set: the spec reduces them by correlation and permutation
importance, and on the cross-sectional label `SELECTED` is down to two.

Every column comes back finite or NaN, never an infinity. NaN is a value the pipeline handles —
`dataset` drops the row — while an infinity is one nothing downstream sees: it passes `dropna`,
survives a quantile fit and reaches the model as an extreme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import KAMAIndicator, RSIIndicator, TSIIndicator
from ta.trend import ADXIndicator
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
        "log_volume_vs_median",
        "log_dollar_volume",
        "signed_volume",
        "volume_trend",
        "on_balance_volume_zscore",
        "distance_from_vwap_pct",
    ),
    "trend": ("distance_from_kama_pct", "distance_from_ema_pct", "ema_slope", "distance_from_psar_pct"),
    "momentum": ("adx_trend_strength", "tsi_momentum", "rsi_centered"),
}
COLUMNS = [c for cols in FAMILIES.values() for c in cols]

# What the cross-sectional label at 72h rewards. Two columns, and that is not a typo.
#
# The twelve that used to be here were selected by `tradingvision.selection` against
# `remaining_excursion`, a label since measured to be *anti*-correlated with tradable money
# (Rank IC -0.033 against the market-neutral forward return). A selection is only ever valid for
# the target it was run against, so that list had to be redone rather than trimmed.
#
# Univariate Rank IC against the new label, measured on the train period alone (pre-2025, fifteen
# symbols) and then confirmed on three walk-forward folds. Three families separate from the rest:
#
#     volatility  vol_24h -0.076, average_true_range_pct -0.073, realized_volatility -0.069
#     liquidity   trades_24h +0.063, log_dollar_volume +0.062
#     reversal    ret_72h -0.049, ret_7d -0.049
#
# and then a cliff: every momentum, trend and position column lands under 0.04, including
# `close_position_in_window`, which was step 2's most important feature by a factor of thirty on
# the old label. The volatility columns are one factor wearing five names (mutual |rho| > 0.9);
# `realized_volatility` is the representative, on the same `n` as everything else here.
#
# Reversal is measured and then dropped, which is the one non-obvious call. It has a real
# univariate signal, but added to the pair below it *lowers* the combined Rank IC from 0.121 to
# 0.103 across the three folds — it is correlated with volatility and contributes noise where it
# does not contribute information.
#
# Two more things this list assumes, both of them measured and neither of them in this file. The
# features have to be ranked inside each timestamp before a model reads them: the label is a
# ranking inside a timestamp, and the same LightGBM on the same columns goes from 0.054 to 0.098
# on that transform alone. And past that, less machinery wins — an equal-weight composite of these
# two ranked columns scores 0.121 +- 0.025 against 0.104 for a fitted tree on all 29. The set is a
# factor pair, not a feature set, and no recurrent model has been re-measured on it yet.
SELECTED = [
    "realized_volatility",
    "log_dollar_volume",
]

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
    "log_volume_vs_median": "Log Volume vs Median",
    "log_dollar_volume": "Log Dollar Volume (N)",
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


# Wilder's Parabolic SAR, ported from `ta.trend.PSARIndicator._run` and asserted equal to it in
# the self-check below. The port is here for one reason: `ta` runs the recursion with pandas
# scalar `.iloc` get and set, which is 60x the cost of the same loop over arrays, and it is the
# whole of this module's runtime — measured on the store, PSAR was 9.4s of `features`' 10.8s and
# the twenty symbols at three timeframes took 304s against 5s. Bit for bit identical on all sixty
# of those series, which is the only condition under which a number already measured stays
# comparable; `ta` stays the definition and the check is what says so.
#
# The `step`/`max_step` defaults are `ta`'s own. There is no window here — Wilder's SAR is
# parameterised by an acceleration and not by a lookback — so this is the one column in the file
# that does not derive its shape from `n`.
PSAR_STEP, PSAR_MAX_STEP = 0.02, 0.20


def _psar(high: pd.Series, low: pd.Series, close: pd.Series, step: float, max_step: float) -> np.ndarray:
    """The SAR level at every bar. Causal: bar `i` reads `i`, `i-1` and `i-2` and nothing later."""
    h, low_, c = (s.to_numpy(dtype="float64") for s in (high, low, close))
    out = c.copy()
    up, acceleration, trend_high, trend_low = True, step, h[0], low_[0]
    for i in range(2, len(c)):
        reversal = False
        if up:
            out[i] = out[i - 1] + acceleration * (trend_high - out[i - 1])
            if low_[i] < out[i]:
                reversal, out[i], trend_low, acceleration = True, trend_high, low_[i], step
            else:
                if h[i] > trend_high:
                    trend_high, acceleration = h[i], min(acceleration + step, max_step)
                # The two bars behind it cap the level, so the stop never sits inside the range
                # the market has just traded through.
                if low_[i - 2] < out[i]:
                    out[i] = low_[i - 2]
                elif low_[i - 1] < out[i]:
                    out[i] = low_[i - 1]
        else:
            out[i] = out[i - 1] - acceleration * (out[i - 1] - trend_low)
            if h[i] > out[i]:
                reversal, out[i], trend_high, acceleration = True, trend_low, h[i], step
            else:
                if low_[i] < trend_low:
                    trend_low, acceleration = low_[i], min(acceleration + step, max_step)
                if h[i - 2] > out[i]:
                    out[i] = h[i - 2]
                elif h[i - 1] > out[i]:
                    out[i] = h[i - 1]
        up = up != reversal  # XOR
    return out


def _bars_since(s: pd.Series, n: int, *, high: bool) -> pd.Series:
    """Age of the window extreme, in [0, 1]: 0 on the bar that set it, 1 at the far end."""
    pick = np.argmax if high else np.argmin
    return s.rolling(n).apply(lambda a: n - 1 - pick(a), raw=True) / n


# The shortest frame this can be asked about. `ta`'s ADX writes `adx[window]` into an array it has
# already trimmed by `window` rows, so it raises IndexError — not a warning, not NaN — on anything
# under `2 * n`, and the chart page reaches that with one click: a day of history on the 4h
# timeframe is six bars. Measured rather than derived from the library's source: 48 bars raise at
# n = 24 and 49 do not.
MIN_BARS = 2 * EXTREMA_WINDOW + 1


def features(df: pd.DataFrame, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The candidate columns for one OHLCV frame, indexed like `df`.

    Leading rows are NaN until every window is filled; the dataset drops them as warm-up.

    A frame too short to fill any window is that same statement taken to its limit, so it comes
    back all NaN rather than raising. The alternative is not a stricter contract, it is a page that
    dies on a slider: every consumer here already treats NaN as "not scorable yet" and draws a gap,
    and an exception out of `ta` two libraries down is a traceback where a short window belongs.
    """
    if len(df) < 2 * n + 1:
        return pd.DataFrame(np.nan, index=df.index, columns=list(COLUMNS))
    o, h, low, c, v = df.open, df.high, df.low, df.close, df.volume
    short = max(n // 4, 2)  # the "N/4" of the spec, floored so a std over it is defined

    ret = np.log(c / c.shift())
    max_h, min_l = h.rolling(n).max(), low.rolling(n).min()
    bar_range = (h - low).replace(0, np.nan)
    ema = c.ewm(span=n, adjust=False).mean()
    obv = (v * np.sign(ret)).rolling(n).sum()
    volume_rel = v / v.rolling(n).median()

    f = {
        "log_return": ret,
        "cum_log_return_window": np.log(c / c.shift(n)),
        "bar_range_pct": (h - low) / c,
        "candle_body_pct": (c - o) / c,
        "upper_wick_pct": (h - np.maximum(o, c)) / c,
        "lower_wick_pct": (np.minimum(o, c) - low) / c,
        # A bar with no range has no location inside it; 0.5 is the neutral answer, and leaving a
        # NaN would punch holes in the middle of the series (40 bars per million on the store).
        "close_position_in_bar": ((c - low) / bar_range).fillna(0.5),
        "close_position_in_window": 2 * (c - min_l) / (max_h - min_l) - 1,
        "distance_from_window_high_pct": np.log(c / max_h),
        "distance_from_window_low_pct": np.log(c / min_l),
        "age_of_window_high": _bars_since(h, n, high=True),
        "age_of_window_low": _bars_since(low, n, high=False),
        "window_range_pct": np.log(max_h / min_l),
        "average_true_range_pct": AverageTrueRange(h, low, c, window=n).average_true_range() / c,
        "realized_volatility": ret.rolling(n).std(),
        "volatility_expansion": ret.rolling(short).std() / ret.rolling(n).std(),
        # In log: the raw ratio is one-sided with a very long right tail (skew 15 on the store),
        # so a quantile scaler has to clip 2.4% of it against 0.02% for the log. The floor only
        # bites on bars with no trade at all, five per symbol in 2023+, where the ratio is 0.
        "log_volume_vs_median": np.log(volume_rel.clip(lower=1e-6)),
        # The one column here whose *level* is the information, and so the one exception to the
        # stationarity rule the rest of this file follows. Every other feature is scale free on
        # purpose, which makes it comparable through time and useless for telling one symbol from
        # another; this one says how big the symbol is, and size is the second of the two factors
        # the cross-sectional label rewards (Rank IC +0.062 on train, +0.084 out of sample, against
        # -0.002 for the self-normalised `log_volume_vs_median` right above it). Nothing downstream
        # is harmed: `normalize` applies the same affine map to every symbol, and a monotone map
        # leaves the ranking inside a timestamp — which is all the label reads — untouched.
        "log_dollar_volume": np.log((v * c).rolling(n).mean()),
        "volume_trend": np.log(v.rolling(short).mean() / v.rolling(n).mean()),
        # Rolling, not cumulated from the start of the series: a running total is not stationary.
        "on_balance_volume_zscore": (obv - obv.rolling(n).mean()) / obv.rolling(n).std(),
        "distance_from_vwap_pct": np.log(
            c / VolumeWeightedAveragePrice(h, low, c, v, window=n).volume_weighted_average_price()
        ),
        "distance_from_kama_pct": np.log(c / KAMAIndicator(c, window=n).kama()),
        "distance_from_ema_pct": np.log(c / ema),
        "ema_slope": np.log(ema / ema.shift(short)),
        "distance_from_psar_pct": (c - _psar(h, low, c, PSAR_STEP, PSAR_MAX_STEP)) / c,
        # Known fixed ranges, so dividing by a constant puts them on the same scale as the % columns
        # with no statistic to estimate and no leakage. The rest need robust scaling fitted on train.
        "adx_trend_strength": ADXIndicator(h, low, c, window=n).adx() / 100,
        "tsi_momentum": TSIIndicator(c, window_slow=n, window_fast=short).tsi() / 100,
        "rsi_centered": RSIIndicator(c, window=n).rsi() / 50 - 1,
    }
    # log1p, not the log above: the magnitude has to stay non-negative or the sign of the column
    # stops meaning the direction of the bar.
    f["signed_volume"] = np.sign(ret) * np.log1p(volume_rel)
    # Infinities out, NaN in — the contract this file owes everything downstream. A bar with no
    # trade at all makes `volume_rel` infinite whenever the window median is zero (`clip` bounds
    # the ratio from below and cannot bound it from above), and `volume_trend` divides two means
    # that can both be zero. Nothing downstream drops an infinity: `dropna` does not see one,
    # `normalize.fit` takes quantiles that survive it, and LightGBM bins it as an extreme. It only
    # disappeared by accident, through the rolling deviation of `dataset.lagged`, which turned one
    # infinite bar into 24 NaN rows and was the mechanism that shifted a symbol's sampling phase.
    # Measured on 2023+: 8,712 zero-volume bars on BAT and 9,269 on YFI, against 14 on BTC.
    return pd.DataFrame(f)[COLUMNS].replace([np.inf, -np.inf], np.nan)


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
    assert list(out.columns) == COLUMNS and len(COLUMNS) == 29, "29 columns, spec order"
    # The PSAR port against `ta`, which is still the definition of the column. Equal and not
    # approximately equal: the recursion is path dependent, so a single bar of drift would carry
    # to the end of the series and every number already measured on it would be a different one.
    from ta.trend import PSARIndicator

    assert np.array_equal(
        _psar(df.high, df.low, df.close, PSAR_STEP, PSAR_MAX_STEP),
        PSARIndicator(*(s.reset_index(drop=True) for s in (df.high, df.low, df.close))).psar().to_numpy(),
    ), "the PSAR port drifted from ta's"
    assert list(LABELS) == COLUMNS, "every column needs a chart label"
    tail = out.iloc[EXTREMA_WINDOW * 4 :]
    assert tail.notna().all().all(), f"NaN past warm-up: {tail.columns[tail.isna().any()].tolist()}"
    assert np.isfinite(tail.to_numpy()).all(), "infinities"
    # A run of bars with no trade at all: the window median goes to zero and the volume ratios
    # blow up. They have to come back NaN, which `dropna` removes, and never as an infinity, which
    # it does not — the silent path that used to reach the model.
    dead = df.copy()
    dead.iloc[200:240, dead.columns.get_loc("volume")] = 0.0
    v = features(dead).iloc[200:260]
    assert np.isfinite(v.to_numpy()[~np.isnan(v.to_numpy())]).all(), "a dead window must not leave an infinity"
    assert v.log_volume_vs_median.isna().any(), "and it has to say so, rather than read as typical"
    assert tail.close_position_in_bar.between(0, 1).all() and tail.close_position_in_window.between(-1, 1).all()
    assert tail.distance_from_window_high_pct.le(1e-12).all() and tail.distance_from_window_low_pct.ge(-1e-12).all()
    assert tail.age_of_window_high.between(0, 1).all() and tail.adx_trend_strength.between(0, 1).all()
    assert tail.rsi_centered.between(-1, 1).all() and tail.tsi_momentum.between(-1, 1).all()
    flat = df.copy()
    flat.loc[flat.index[300], ["open", "high", "low"]] = flat.close.iloc[300]
    assert features(flat).close_position_in_bar.iloc[300] == 0.5, "a zero-range bar has no location"
    # Causality: a feature must not move when a later bar changes.
    cut = 300
    assert np.allclose(features(df.iloc[:cut]).iloc[-1].to_numpy(), out.iloc[cut - 1].to_numpy(), equal_nan=True)
    # A frame too short to fill a window is all NaN and not an exception. `ta`'s ADX raises
    # IndexError under `2 * n` — not a warning, not NaN — and the chart page reaches that with one
    # click: a day of history on the 4h timeframe is six bars. The shape has to survive it, same
    # index and same columns, so every consumer downstream reads warm-up and draws a gap.
    for count in (1, 6, 2 * EXTREMA_WINDOW):
        stub = df.iloc[:count]
        thin = features(stub, EXTREMA_WINDOW)
        assert thin.index.equals(stub.index) and list(thin.columns) == COLUMNS, count
        assert thin.isna().all().all(), count
    # The boundary is where it was measured and not one bar either side: 2n+1 computes, 2n does not.
    assert MIN_BARS == 2 * EXTREMA_WINDOW + 1
    assert features(df.iloc[:MIN_BARS], EXTREMA_WINDOW).notna().any().any()
    print(f"ok — {len(COLUMNS)} columns, {len(tail)} rows past warm-up")
