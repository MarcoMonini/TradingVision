"""One row per 5m bar: the four branches side by side, plus the label and what purging needs.

The whole file rests on one convention — every frame is indexed by the *open* time of its bar, so
a bar labelled `b` on timeframe `tf` closes at `b + tf`. Two bars are simultaneous when their
labels plus their durations match, and that is what aligns the branches:

    15m bar 09:45  closes 10:00  ==  5m bar 09:55  closes 10:00

So a branch column is placed on the 5m grid at `label + tf - 5m` and forward filled from there.
At the 5m bar of 10:05 (closing 10:10) the 15m branch still reads the candle closed at 10:00, not
the one forming until 10:15 — the rule the spec states, and the one thing here that cannot be got
wrong: a single bar of anticipation on the 1h branch would hand the model twelve 5m bars of
future and inflate every metric downstream.

Pivots follow the same rule. They are extrema of the 15m close (the spec's definition of a leg,
unchanged), so a pivot on the 15m bar 09:45 is a price realised at 10:00 and lands on the 5m bar
09:55. The target is then interpolated over 5m bars between two pivots.

The label is `remaining_excursion` and the column is called `target`, so swapping the label again
costs one line here instead of a rename across the pipeline. It is called on the 5m close, which
is what its `horizon` and `lookback` defaults are counted in — see its docstring, the numbers do
not mean what their names suggest on this grid.

`next_pivot` travels with the data because purging is exact, not a fixed embargo: the split drops
every train bar whose next pivot falls beyond the cut, and that horizon is unbounded (p99 is 202
bars, the maximum measured 754).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from tradingvision.data import binance
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import remaining_excursion
from tradingvision.features import features

BRANCHES = ("5m", "15m", "30m", "1h")
PIVOT_TF = "15m"
BASE_TF = "5m"
# Enough 5m bars ahead of `start` for every branch window to be filled: 24 steps of 1h, plus the
# 24 bars the first step looks back over, is 48h — this is a wide margin over that.
WARMUP = pd.Timedelta("10D")


def _shift(tf: str) -> pd.Timedelta:
    """How far a `tf` label sits from the simultaneous 5m label."""
    return pd.Timedelta(tf) - pd.Timedelta(BASE_TF)


def branch(bars: pd.DataFrame, tf: str, index: pd.DatetimeIndex, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The features of one branch, carried onto the 5m `index` with no bar of anticipation."""
    f = features(bars, n)
    f.index = f.index + _shift(tf)
    return f.reindex(index, method="ffill").add_suffix(f"_{tf}")


# The label. It is an argument rather than a hard call so the comparison against the
# retrospective `swing_leg_target` it replaced can be rebuilt from scratch (see `crosscheck`).
LABEL: Callable[..., pd.Series] = remaining_excursion


def symbol_frame(
    symbol: str, start: str | None = None, n: int = EXTREMA_WINDOW, label: Callable[..., pd.Series] = LABEL
) -> pd.DataFrame:
    """Branches, target and purging horizon for one symbol, warm-up rows dropped."""
    bars = binance.load(symbol, BASE_TF)
    if start is not None:
        bars = bars.loc[pd.Timestamp(start, tz="UTC") - WARMUP :]
    idx = bars.index

    pivots = find_pivots(binance.load(symbol, PIVOT_TF).close, n)
    pivots.index = pivots.index + _shift(PIVOT_TF)
    pivots = pivots[(pivots.index >= idx[0]) & (pivots.index <= idx[-1])]
    # Every 15m close is also a 5m close, so a pivot inside the range must land on an existing 5m
    # bar. It always does today (measured: none missing on 2023+), but a gap in the 5m series
    # would silently move the next pivot of the preceding bars one leg further and mislabel them.
    missing = pivots.index.difference(idx)
    if len(missing):
        raise ValueError(f"{len(missing)} pivots have no 5m bar, first at {missing[0]}")

    out = pd.concat([branch(binance.load(symbol, tf), tf, idx, n) for tf in BRANCHES], axis=1)
    out["target"] = label(bars.close, pivots, n)
    # Timestamp of the pivot that closes each bar's leg — the bar itself when it is a pivot.
    out["next_pivot"] = pd.Series(pivots.index, index=pivots.index).reindex(idx, method="bfill")
    return out.dropna()


def build(
    symbols: list[str] | None = None,
    start: str | None = "2023",
    stride: int = 12,
    n: int = EXTREMA_WINDOW,
    label: Callable[..., pd.Series] = LABEL,
) -> pd.DataFrame:
    """Every symbol stacked on a (timestamp, symbol) index.

    `stride` subsamples the 5m grid — 12 keeps one row per hour. The rows are not independent
    anyway (consecutive samples share 23 of 24 steps, spec point 1), and the full grid at 20
    symbols does not fit in memory as a flat frame.
    """
    symbols = binance.SYMBOLS if symbols is None else symbols
    frames = {}
    for s in symbols:
        f = symbol_frame(s, start, n, label)
        if start is not None:
            f = f.loc[pd.Timestamp(start, tz="UTC") :]
        frames[s] = f.iloc[::stride]
    return pd.concat(frames, names=["symbol"]).swaplevel().sort_index()


if __name__ == "__main__":
    import sys

    df = build(["BTC", "ETH"], start="2024", stride=12)
    print(df.shape, df.index.get_level_values(0).min(), "->", df.index.get_level_values(0).max())
    print(df.columns[:2].tolist(), "...", df.columns[-3:].tolist())
    sys.exit(0)
