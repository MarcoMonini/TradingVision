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

A branch normally contributes one row per bar — the value of its features at `t`. That is all a
model that reads the sequence itself needs. A GBM does not read sequences: it takes a flat table,
so the history has to be spelled out as columns or it is invisible. `lagged` does that, and the
step-2 run turns it on: value at `t`, at `t-1`, at `t-4`, plus the mean and the standard deviation
over the extrema window. Without it the comparison against the GRU is dishonest — the GBM would
be judged on a single candle — and the feature importance would rank `age_of_window_high` and
`ema_slope` last for having no room to move.

`next_pivot` travels with the data because purging is exact, not a fixed embargo: the split drops
every train bar whose next pivot falls beyond the cut, and that horizon is unbounded (p99 is 202
bars, the maximum measured 754).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
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


# Which past steps to spell out, in bars of the branch's own timeframe: the previous one and four
# back. Deliberately short — the mean and the deviation already carry the window, and a lag per
# step would be 24x the columns for a model that cannot use most of them.
LAGS = (1, 4)


def lagged(f: pd.DataFrame, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """`f` widened from value-at-t to value, lags and window statistics — 5 columns per input.

    Causal like everything upstream: `shift` and `rolling` look backwards only. float32 because
    the width is the cost here — 28 columns become 140, four branches 560 — and a GBM bins its
    inputs into 255 buckets, so the discarded precision cannot reach the model anyway.
    """
    parts = [f] + [f.shift(k).add_suffix(f"_lag{k}") for k in LAGS]
    window = f.rolling(n)
    parts += [window.mean().add_suffix("_mean"), window.std().add_suffix("_std")]
    return pd.concat(parts, axis=1).astype("float32")


def branch(
    bars: pd.DataFrame, tf: str, index: pd.DatetimeIndex, n: int = EXTREMA_WINDOW, lags: bool = False
) -> pd.DataFrame:
    """The features of one branch, carried onto the 5m `index` with no bar of anticipation."""
    f = features(bars, n)
    if lags:
        f = lagged(f, n)
    f.index = f.index + _shift(tf)
    return f.reindex(index, method="ffill").add_suffix(f"_{tf}")


# The label. It is an argument rather than a hard call so the comparison against the
# retrospective `swing_leg_target` it replaced can be rebuilt from scratch (see `crosscheck`).
LABEL: Callable[..., pd.Series] = remaining_excursion


def pivots_on(symbol: str, idx: pd.DatetimeIndex, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The `PIVOT_TF` pivots of `symbol`, carried onto the 5m grid `idx` and checked to land on it.

    Its own function because the label is not the only thing that needs them: anything recomputing
    a target on rows this module already produced has to use the same pivots, found the same way,
    or it is labelling a different set of legs.
    """
    pivots = find_pivots(binance.load(symbol, PIVOT_TF).close, n)
    pivots.index = pivots.index + _shift(PIVOT_TF)
    pivots = pivots[(pivots.index >= idx[0]) & (pivots.index <= idx[-1])]
    # Every 15m close is also a 5m close, so a pivot inside the range must land on an existing 5m
    # bar. It always does today (measured: none missing on 2023+), but a gap in the 5m series
    # would silently move the next pivot of the preceding bars one leg further and mislabel them.
    missing = pivots.index.difference(idx)
    if len(missing):
        raise ValueError(f"{len(missing)} pivots have no 5m bar, first at {missing[0]}")
    return pivots


def symbol_frame(
    symbol: str,
    start: str | None = None,
    n: int = EXTREMA_WINDOW,
    label: Callable[..., pd.Series] = LABEL,
    lags: bool = False,
) -> pd.DataFrame:
    """Branches, target and purging horizon for one symbol, warm-up rows dropped."""
    bars = binance.load(symbol, BASE_TF)
    if start is not None:
        bars = bars.loc[pd.Timestamp(start, tz="UTC") - WARMUP :]
    idx = bars.index

    pivots = pivots_on(symbol, idx, n)
    out = pd.concat([branch(binance.load(symbol, tf), tf, idx, n, lags) for tf in BRANCHES], axis=1)
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
    lags: bool = False,
) -> pd.DataFrame:
    """Every symbol stacked on a (timestamp, symbol) index.

    `stride` subsamples the 5m grid — 12 keeps one row per hour. The rows are not independent
    anyway (consecutive samples share 23 of 24 steps, spec point 1), and the full grid at 20
    symbols does not fit in memory as a flat frame.
    """
    symbols = binance.SYMBOLS if symbols is None else symbols
    frames = {}
    for s in symbols:
        f = symbol_frame(s, start, n, label, lags)
        if start is not None:
            f = f.loc[pd.Timestamp(start, tz="UTC") :]
        frames[s] = f.iloc[::stride]
    return pd.concat(frames, names=["symbol"]).swaplevel().sort_index()


def cached(path: Path, **params) -> pd.DataFrame:
    """`build(**params)`, materialised next to a JSON stamp of the arguments that produced it.

    A built dataset takes about half an hour and 600 MB, so it is reused across runs — and a
    reused file that was built with a different stride, window or lag setting is the kind of
    mistake that shows up as a metric nobody can reproduce. So the parameters travel with the
    parquet and a mismatch stops the run instead of quietly answering the wrong question.
    """
    stamp = path.with_suffix(".json")
    # `LAGS` travels too: it is not an argument, and changing it changes every column in the file.
    written = dict(params, symbols=sorted(params.get("symbols") or binance.SYMBOLS), lags_at=list(LAGS))
    if path.exists():
        if not stamp.exists():
            raise SystemExit(f"{path} has no {stamp.name} recording how it was built — delete it and rebuild")
        if (was := json.loads(stamp.read_text())) != written:
            raise SystemExit(f"{path} was built with {was}, not {written} — delete it or pass another --cache")
        return pd.read_parquet(path)

    df = build(**params)
    df.to_parquet(path)
    stamp.write_text(json.dumps(written, indent=2, sort_keys=True))
    return df


def _selfcheck() -> None:
    """`lagged` widens without looking forward — the one thing that would quietly invent signal."""
    f = pd.DataFrame(
        {"a": range(100), "b": [x * x for x in range(100)]},
        index=pd.date_range("2024", periods=100, freq="15min", tz="UTC"),
        dtype="float64",
    )
    out = lagged(f, n=4)
    assert list(out.columns) == ["a", "b", "a_lag1", "b_lag1", "a_lag4", "b_lag4", "a_mean", "b_mean", "a_std", "b_std"]
    assert len(out.columns) == 5 * len(f.columns)
    assert out.a.iloc[7] == 7 and out.a_lag1.iloc[7] == 6 and out.a_lag4.iloc[7] == 3
    assert out.a_mean.iloc[7] == 5.5, "mean of 4,5,6,7"
    assert out.a_lag4.iloc[:4].isna().all(), "no value before the fourth bar"
    # Causality: truncating the future leaves every past row untouched.
    assert lagged(f.iloc[:50], n=4).iloc[49].equals(out.iloc[49])

    # The cache refuses a file built with other parameters instead of answering the wrong question.
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.parquet"
        pd.DataFrame({"target": [1.0]}).to_parquet(path)
        for wrong in (dict(stride=12, symbols=["BTC"]),):  # no stamp yet: unreadable provenance
            try:
                cached(path, **wrong)
            except SystemExit:
                break
            raise AssertionError("a cache with no stamp has to stop the run")
        path.with_suffix(".json").write_text(json.dumps(dict(stride=12, symbols=["BTC"], lags_at=list(LAGS))))
        assert len(cached(path, stride=12, symbols=["BTC"])) == 1, "a matching stamp reads the file back"
        for wrong in (dict(stride=6, symbols=["BTC"]), dict(stride=12, symbols=["ETH"])):
            try:
                cached(path, **wrong)
            except SystemExit:
                continue
            raise AssertionError(f"a cache built with other parameters has to stop the run: {wrong}")


if __name__ == "__main__":
    import sys

    _selfcheck()

    df = build(["BTC", "ETH"], start="2024", stride=12)
    print(df.shape, df.index.get_level_values(0).min(), "->", df.index.get_level_values(0).max())
    print(df.columns[:2].tolist(), "...", df.columns[-3:].tolist())
    sys.exit(0)
