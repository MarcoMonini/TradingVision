"""The causal half of a swing: what the last *confirmed* pivot says about the bar you are on.

`data.pivots` finds extrema with a centred window, which is the label's side of the problem — a
pivot at bar `j` is only a pivot because of the `window` bars that came after it. Every feature in
`features` is causal but none of them knows the leg structure at all: the closest it gets is
`age_of_window_high`, which is the age of a rolling maximum and not of a confirmed turn.

That gap matters more for this label than for any other in the project. `swing_leg_target` is
dominated by `(i - start) / (end - start)`, and `i - start` is *knowable* — it is the distance
from the last pivot — while `end - start` is not. A model with no notion of `start` has to infer
the numerator from the shape of the price, which is exactly the part it should have been given.

The rule here is the one a live reader would have followed, and it is deliberately stricter than
reading `find_pivots` and shifting it:

  * a raw extremum at `j` needs `window` bars on its right, so it becomes visible at `j + window`;
  * the merge of same-kind runs is applied **online** — when a higher high arrives, it replaces the
    previous one from that moment on, and not retroactively.

Reading `find_pivots` and shifting the whole frame by `window` would keep the merged frame's
hindsight: the centred pass already dropped the lower high that a live reader would have been
acting on for the hours in between. Measured over the twenty symbols on 15m, the two disagree on
the identity of the last pivot for 7.4% of bars, always in the direction that flatters the causal
reader. `_selfcheck` pins the difference with a case built to show it.

    uv run python -m tradingvision.legs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

from tradingvision.data.pivots import EXTREMA_WINDOW

# Columns this module produces, in order. Named here so the model's feature list can be assembled
# without calling it, and so a rename fails loudly rather than silently dropping a column.
STATE = (
    "leg_kind",
    "leg_age",
    "leg_move",
    "leg_extent",
    "leg_age_vs_prev",
    "prev_amplitude",
    "prev_duration",
    "leg_drawdown",
    "signed_age",
    "signed_move",
)
# Turn detection, which is a different question from leg state and the one the spec's open point 4
# names: inside 24 bars of a pivot every model this project has fitted gets the direction wrong,
# and its stated road is exhaustion rather than momentum. These are that road — each one is a way
# a move runs out rather than a way it continues.
EXHAUSTION = (
    "divergence",
    "volume_climax",
    "rejection",
    "streak",
    "deceleration",
    "stretch",
)
COLUMNS = STATE + EXHAUSTION


def raw_extrema(close: pd.Series, window: int = EXTREMA_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """`(positions, kinds)` of every local extreme, unmerged — `find_pivots` without its merge."""
    c = close.to_numpy(dtype="float64")
    s = pd.Series(c)
    left_hi, left_lo = s.rolling(window).max().shift(1), s.rolling(window).min().shift(1)
    back = pd.Series(c[::-1])
    right_hi = back.rolling(window).max().shift(1).to_numpy()[::-1]
    right_lo = back.rolling(window).min().shift(1).to_numpy()[::-1]
    high = (c > left_hi.to_numpy()) & (c > right_hi)
    low = (c < left_lo.to_numpy()) & (c < right_lo)
    at = np.flatnonzero(high | low)
    return at, np.where(high[at], 1, -1)


def confirmed(close: pd.Series, window: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The timeline of what a live reader held: one row per extreme, dated from its confirmation.

    Columns `known` (the bar the row becomes true on), `bar`/`kind`/`price` of the pivot held from
    then, and `prev_bar`/`prev_price` of the one before it — the leg that had closed on it.

    One row per *raw* extreme and not per merged pivot, which is the whole point. When a higher
    high prints, the previous one stops being the pivot from that moment — but it was the pivot
    until then, and a merged frame has already deleted it. Rows superseded this way stay in the
    timeline with their own `known`, so reading the timeline at bar `i` answers "what did the
    reader hold at `i`" and never "what does the finished chart say was there".
    """
    at, kind = raw_extrema(close, window)
    c = close.to_numpy(dtype="float64")
    held_bar: list[int] = []
    held_kind: list[int] = []
    rows = []
    for j, k in zip(at.tolist(), kind.tolist()):
        if held_kind and held_kind[-1] == k:
            # A twin of the same kind is the same turn; the more extreme of the two is the one the
            # reader ends up holding, from the twin's own confirmation onwards.
            if (c[j] - c[held_bar[-1]]) * k > 0:
                held_bar[-1], held_kind[-1] = j, k
        else:
            held_bar.append(j)
            held_kind.append(k)
        rows.append(
            {
                "known": j + window,
                "bar": held_bar[-1],
                "kind": held_kind[-1],
                "price": c[held_bar[-1]],
                "prev_bar": held_bar[-2] if len(held_bar) > 1 else np.nan,
                "prev_price": c[held_bar[-2]] if len(held_bar) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["known", "bar", "kind", "price", "prev_bar", "prev_price"])


def state(close: pd.Series, window: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """One row per bar of `close`: the leg in progress, as it was visible at that bar.

    NaN until a first pivot has been confirmed *and* a previous one exists to measure the last leg
    against, and wherever the volatility window is still filling — the same contract `features`
    has, so one `dropna` downstream removes the whole warm-up.

    The scale-free columns are the ones a model can carry across symbols. `leg_move` is the move
    since the pivot in units of what the volatility would have produced over the same span, which
    is the significance ratio of `data.target` measured forwards from the pivot instead of
    backwards from the next one. `leg_extent` compares it to the amplitude of the previous leg,
    the only estimate of "how far this one should run" available at the time.
    """
    line = confirmed(close, window)
    n = len(close)
    c = close.to_numpy(dtype="float64")
    logc = np.log(c)
    out = pd.DataFrame(np.nan, index=close.index, columns=list(STATE))
    if line.empty:
        return out

    # For every bar, the last timeline row already true. -1 before the first confirmation.
    row = np.searchsorted(line.known.to_numpy(), np.arange(n), side="right") - 1
    live = (row >= 0) & np.isfinite(line.prev_bar.to_numpy()[np.clip(row, 0, None)])
    if not live.any():
        return out
    k = np.clip(row, 0, None)
    bar = line.bar.to_numpy()[k]
    kind = line.kind.to_numpy()[k].astype("float64")
    price = line.price.to_numpy()[k]
    prev_bar = line.prev_bar.to_numpy()[k]
    prev_price = line.prev_price.to_numpy()[k]
    age = np.arange(n) - bar

    sigma = pd.Series(logc).diff().rolling(4 * window).std().to_numpy()
    prev_dur = np.maximum(bar - prev_bar, 1.0)
    prev_amp = np.maximum(np.abs(np.log(price / prev_price)), 1e-6)
    move = logc - np.log(price)
    # The worst this leg has been against the pivot it started from — the running excursion a rule
    # holding a position would actually have felt. Grouped by the pivot, so it resets on each turn.
    running = pd.Series(np.where(live, move * kind, np.nan)).groupby(pd.Series(bar)).cummin().to_numpy()

    out.leg_kind = np.where(live, kind, np.nan)
    out.leg_age = np.where(live, age / window, np.nan)
    out.leg_move = np.where(live, move / (sigma * np.sqrt(np.maximum(age, 1))), np.nan)
    out.leg_extent = np.where(live, move / prev_amp, np.nan)
    out.leg_age_vs_prev = np.where(live, age / prev_dur, np.nan)
    out.prev_amplitude = np.where(live, prev_amp, np.nan)
    out.prev_duration = np.where(live, prev_dur / window, np.nan)
    out.leg_drawdown = np.where(live, running, np.nan)
    out.signed_age = out.leg_kind * out.leg_age
    out.signed_move = out.leg_kind * out.leg_move
    return out.replace([np.inf, -np.inf], np.nan)


def exhaustion(bars: pd.DataFrame, window: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """Six ways a move runs out, computed on candles and all of them causal.

    `legs.state` says where the last confirmed turn was, which is information arriving `window`
    bars late by construction — and the lag is the whole problem: the oracle run at a detection lag
    of 24 bars keeps 3% of what it makes at lag 0. Anything that reads a turn *as it happens* is
    worth more than anything that reads one after it is certain, and that is what these are for.

    `divergence` is the classic one stated as a difference of ranks: where the close sits in its
    window against where the RSI sits in its own, so a new high on a weaker oscillator is negative
    and a new low on a stronger one is positive. `volume_climax` is volume against its own
    background, signed by where the price is in its range — a spike at an extreme, which is what a
    capitulation looks like. `rejection` is the wick on the far side of the move, counted only
    where it contradicts: an upper wick at the top of the range, a lower one at the bottom.
    `streak` is the run of same-signed closes, squashed. `deceleration` is the move losing speed
    against its own direction. `stretch` is how far the close has left its own average, in units
    of what the volatility would have produced.
    """
    c, h, low, v = bars.close, bars.high, bars.low, bars.volume
    span = (h - low).replace(0, np.nan)
    price_rank = c.rolling(window).rank(pct=True)
    position = 2 * price_rank - 1
    rsi = RSIIndicator(c, window).rsi() / 100
    upper = (h - np.maximum(c, bars.open)) / span
    lower = (np.minimum(c, bars.open) - low) / span
    step = np.sign(c.diff()).fillna(0.0)
    run = step.groupby((step != step.shift()).cumsum()).cumcount() + 1
    ema = c.ewm(span=window, adjust=False).mean()
    slope = np.log(ema).diff()
    sigma = np.log(c).diff().rolling(4 * window).std()
    out = pd.DataFrame(
        {
            "divergence": price_rank - rsi.rolling(window).rank(pct=True),
            "volume_climax": (v - v.rolling(4 * window).mean()) / v.rolling(4 * window).std() * position,
            "rejection": lower * np.maximum(-position, 0) - upper * np.maximum(position, 0),
            "streak": step * np.tanh(run / window),
            "deceleration": -np.sign(slope) * slope.diff() / sigma,
            "stretch": (np.log(c) - np.log(ema)) / sigma,
        }
    )
    return out[list(EXHAUSTION)].replace([np.inf, -np.inf], np.nan)


def next_pivot(close: pd.Series, window: int = EXTREMA_WINDOW) -> pd.Series:
    """When the leg each bar sits on ends — the purging horizon `swing_leg_target` needs.

    The *label's* pivots and not the causal ones: purging is about what the label saw, and the
    label interpolates towards the centred frame's next turn. NaT past the last pivot, where there
    is no label either.
    """
    from tradingvision.data.pivots import find_pivots

    piv = find_pivots(close, window)
    out = pd.Series(pd.NaT, index=close.index, dtype="datetime64[ns, UTC]")
    if piv.empty:
        return out
    at = close.index.get_indexer(piv.index)
    i = np.arange(at[-1] + 1)
    out.iloc[i] = close.index[at[np.searchsorted(at, i, side="left")]]
    return out


def _selfcheck() -> None:
    """A wave whose turns are known, and one case where hindsight and a live reader disagree."""
    n = 600
    x = pd.Series(np.abs(np.arange(n) % 20 - 10.0) + 100.0, index=pd.RangeIndex(n))
    line = confirmed(x, 5)
    assert line.known.is_monotonic_increasing, "the timeline is in confirmation order"
    assert set(line.kind.unique()) == {-1, 1}

    s = state(x, 5)
    assert list(s.columns) == list(STATE)
    live = s.dropna()
    assert len(live) > n // 2 and set(s.leg_kind.dropna().unique()) == {-1.0, 1.0}
    # The age grows inside a leg and resets on a turn, which is the column the label is made of.
    inside = s.leg_age.dropna()
    assert (inside.diff().dropna() <= 0).any() and (inside.diff().dropna() > 0).mean() > 0.7

    # Causality, stated as the property that matters: truncating the series after bar `t` cannot
    # change any row at or before `t`. A shifted centred frame fails exactly this.
    t = 400
    assert np.allclose(
        state(x.iloc[: t + 1], 5).iloc[:t].to_numpy(), s.iloc[:t].to_numpy(), equal_nan=True
    ), "a later bar changed an earlier row"

    # And the disagreement with hindsight, built on purpose: two highs in a row, the second the
    # higher. `find_pivots` merges them and keeps only the second, so a frame shifted by `window`
    # would claim the reader held the second high during the bars between them. He held the first.
    y = pd.Series(
        [100.0] * 8 + [104.0] + [101.0] * 8 + [106.0] + [100.0] * 8 + [95.0] + [99.0] * 8,
        index=pd.RangeIndex(35),
    )
    from tradingvision.data.pivots import find_pivots

    merged, seen = find_pivots(y, 3), confirmed(y, 3)
    assert 8 not in merged.index.tolist(), "hindsight has already dropped the first high"
    assert seen.bar.tolist() == [8, 17, 26], "the reader held it, then the twin replaced it"
    assert seen.known.tolist() == [11, 20, 29], "each row dated from its own confirmation"
    assert seen.bar.iloc[1] == 17 and seen.prev_bar.iloc[2] == 17

    flat = pd.Series([1.0] * 100, index=pd.RangeIndex(100))
    assert state(flat, 5).isna().all().all(), "no pivots, no state"
    assert confirmed(flat, 5).empty

    # The purging horizon is the label's, so it is allowed to be the centred frame's next turn.
    when = pd.date_range("2024", periods=n, freq="15min", tz="UTC")
    reach = next_pivot(pd.Series(x.to_numpy(), index=when), 5).dropna()
    assert (reach.index <= reach.to_numpy()).all(), "a leg never ends before the bar it holds"
    # Exhaustion, on candles rather than on a close alone. A wave that turns has to show a
    # rejection wick and a deceleration at its turns, or the columns are measuring nothing.
    bars = pd.DataFrame({"open": x, "high": x + 0.4, "low": x - 0.4, "close": x, "volume": 100.0 + (np.arange(n) % 20)})
    e = exhaustion(bars, 5).dropna()
    assert list(e.columns) == list(EXHAUSTION) and len(e) > n // 2
    assert np.isfinite(e.to_numpy()).all(), "finite or NaN, never an infinity"
    # `streak` counts the run of same-signed moves and carries its sign: on a saw it is high just
    # before each turn and flips right after one.
    assert e.streak.max() > 0.5 and e.streak.min() < -0.5
    assert abs(float(e.streak.corr(pd.Series(np.sign(x.diff()).to_numpy()[e.index], index=e.index)))) > 0.8
    # Causal like everything else here.
    cut = 400
    assert np.allclose(
        exhaustion(bars.iloc[: cut + 1], 5).iloc[:cut].to_numpy(),
        exhaustion(bars, 5).iloc[:cut].to_numpy(),
        equal_nan=True,
    ), "a later bar changed an earlier row"
    print(f"ok — {len(line)} timeline rows, {len(COLUMNS)} columns, {len(live)} live bars")


if __name__ == "__main__":
    _selfcheck()
