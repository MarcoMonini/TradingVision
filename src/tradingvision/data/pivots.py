"""Local extrema (pivots) on the close series, per the dataset spec.

A bar is a pivot high if its close is strictly higher than the closes of all `window` bars on
both sides (mirror for pivot low). Same criterion as `scipy.signal.argrelextrema(order=window)`,
computed with rolling windows so scipy stays out of the dependency list.

Extrema are searched on Close, never High/Low: wicks and liquidation spikes produce extrema that
are neither tradable levels nor structural reversals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bars of the 15m reference timeframe. Measured, not assumed: the bare oracle P&L criterion is
# degenerate (it keeps favouring shorter windows until the average leg stops clearing costs, which
# is a cost/volatility ratio and not market structure), so windows are scored on an oracle
# penalised by a detection lag of 1-3 bars. 24 is the optimum of the 16-28 range at lag 3, within
# 1% of the optimum at lag 2, and has the best share of profitable legs. See the HTML spec.
EXTREMA_WINDOW = 24


def _rolling_extreme(s: pd.Series, window: int, *, reverse: bool, high: bool) -> pd.Series:
    """Extreme of the `window` bars on one side of each bar, excluding the bar itself."""
    side = s.iloc[::-1] if reverse else s
    roll = side.rolling(window).max() if high else side.rolling(window).min()
    out = roll.shift(1)
    return out.iloc[::-1] if reverse else out


def find_pivots(close: pd.Series, window: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """Alternating high/low pivots with the amplitude of the leg ending on each one.

    Columns: `kind` (+1 high, -1 low), `close`, `amplitude` (absolute log return of the leg
    closed at that pivot — backward convention, so it is causal at the pivot).
    Returns an empty frame when the series is too short to hold any pivot.
    """
    is_high = (close > _rolling_extreme(close, window, reverse=False, high=True)) & (
        close > _rolling_extreme(close, window, reverse=True, high=True)
    )
    is_low = (close < _rolling_extreme(close, window, reverse=False, high=False)) & (
        close < _rolling_extreme(close, window, reverse=True, high=False)
    )

    kind = pd.Series(0, index=close.index, dtype=int).mask(is_high, 1).mask(is_low, -1)
    piv = kind[kind != 0].to_frame("kind")
    if piv.empty:
        return piv.assign(close=[], amplitude=[])
    piv["close"] = close.reindex(piv.index)

    # Merge: two same-kind pivots in a row are one leg, keep the more extreme of the run.
    run = (piv.kind != piv.kind.shift()).cumsum()
    pick = piv.close * piv.kind  # maximising this picks the highest high / the lowest low
    piv = piv.loc[pick.groupby(run).idxmax()]

    # Leg amplitude, backward convention; the first pivot has no predecessor, so the leg is
    # measured from the first close available and is truncated by construction.
    prev = piv.close.shift().fillna(close.iloc[0])
    piv["amplitude"] = np.abs(np.log(piv.close / prev))
    return piv


if __name__ == "__main__":
    # Triangle wave: peaks every 20 bars, troughs halfway between.
    x = pd.Series(np.abs(np.arange(200) % 20 - 10.0) + 1.0, index=pd.RangeIndex(200))
    p = find_pivots(x, 5)
    assert p.kind.diff().dropna().abs().eq(2).all(), "highs and lows must alternate"
    assert p.index.to_series().diff().dropna().eq(10).all(), "pivots every half period"
    assert find_pivots(x.iloc[:3], 5).empty
    print(f"ok — {len(p)} pivots")
