"""`swing_leg_target`: where a bar sits along the leg between two adjacent pivots, in [-1, +1].

The label the model predicts. It is -1 on a pivot low, +1 on a pivot high, and interpolates
linearly in between along a blend of two advancements, both in [0, 1]:

    x = smoothing * time_advance + (1 - smoothing) * price_advance
    target(i) = valori[k] + x * (valori[k+1] - valori[k])

`smoothing` is the spec's `peso_tempo`: the weight of the *time* term, not a filter applied to the
label afterwards (there is no EMA on the target, and if one is ever added it is a separate step).
At the default 0.7 the label advances almost regularly along the leg even when price moves in
bursts — smoother and easier to learn, less reactive to a sudden acceleration. At 1.0 it is a pure
time ramp between pivots; at 0.0 it follows price alone.

The pivot values are not +/-1 flat. A pivot is a local extreme of a 24-bar window, not an extreme
of the leg, so two pivots can sit a few basis points apart with the price swinging far outside
both of them in between — 1.3% of legs, measured over 20 symbols. On those the label ramps from
+1 to -1 across pure noise, which is a false signal, not a weak one. Each pivot is therefore
scaled by the significance of the leg it closes:

    valori[k] = kind[k] * tanh(amplitude[k] / (sigma * sqrt(duration[k])))

The denominator is what a random walk of the same volatility would cover in the time the leg
actually took, so the ratio reads "how much more than chance did this leg move". It was picked by
measurement over the candidates in `tradingvision.reference` — it catches 85% of the broken legs
against 54% for ATR, because a degenerate leg is long and flat and only a duration-aware scale
separates a big move from a slow one. The median leg scores 1.6 and keeps a pivot value of 0.92;
the median broken leg scores 0.41 and is pulled down to 0.39.

Scaling per pivot, not per leg, is what keeps the label continuous: a pivot is shared by the leg
that ends on it and the one that starts there, and it has to be one number. A weak leg therefore
does not sit flat at 0 — it decays from the value of the pivot before it towards ~0, which reads
as "we were at a top, then nothing meaningful happened".

Retrospective by construction: it interpolates towards the *next* pivot, so it sees the future.
That is what a label is. The causal side of the pipeline is the features, plus `amplitude`, which
is measured backwards precisely so it stays usable at the pivot itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots

# Weight of the time term in the blend (`peso_tempo` in the spec). A starting value, tunable.
SMOOTHING = 0.7
# Bars of the reference timeframe used to estimate volatility: four times the extrema window, a
# day of 15m bars — long enough to average out a single leg, short enough to track a regime.
SIGNIFICANCE_LOOKBACK = 4 * EXTREMA_WINDOW


def bar_sigma(close: pd.Series, lookback: int = SIGNIFICANCE_LOOKBACK) -> pd.Series:
    """Standard deviation of the `lookback` most recent log returns: per-bar volatility, causal."""
    return np.log(close).diff().rolling(lookback).std()


def leg_significance(
    close: pd.Series,
    pivots: pd.DataFrame,
    lookback: int = SIGNIFICANCE_LOOKBACK,
) -> pd.Series:
    """`amplitude[k] / (sigma * sqrt(duration[k]))` per pivot: the leg against chance.

    Around 1 the leg is what the volatility would have produced on its own; the panel median is
    1.6 and broken legs sit near 0.4. NaN on the first pivot (no leg before it) and until the
    volatility window fills, which drops the opening legs of a series from the label.
    """
    at = close.index.get_indexer(pivots.index)
    if (at < 0).any():
        raise ValueError("pivots do not belong to this close series")
    duration = pd.Series(at, index=pivots.index).diff()
    return pivots.amplitude / (bar_sigma(close, lookback).to_numpy()[at] * np.sqrt(duration))


def swing_leg_target(
    close: pd.Series,
    pivots: pd.DataFrame | None = None,
    window: int = EXTREMA_WINDOW,
    smoothing: float = SMOOTHING,
    significance: bool = True,
) -> pd.Series:
    """The label for every bar of `close`, NaN where it is undefined.

    `pivots` skips the detection when the caller already has it for this window.

    NaN before the first pivot and after the last one: those bars have no leg to sit on. The tail
    is the interesting half — it is the permanent condition of the current bar in production, the
    very place where the model has to predict and the label cannot be computed.

    `significance=False` restores flat +/-1 pivots, for comparing the two labellings.
    """
    piv = find_pivots(close, window) if pivots is None else pivots
    out = pd.Series(np.nan, index=close.index, name="swing_leg_target")
    if len(piv) < 2:
        return out

    # Positional throughout: the interpolation runs over bars between pivots, which have no pivot
    # label to index by. As in the oracle, a pivot frame built on another slice has to fail loudly
    # rather than read the wrong bar through get_indexer's -1.
    at = close.index.get_indexer(piv.index)
    if (at < 0).any():
        raise ValueError("pivots do not belong to this close series")

    i = np.arange(at[0], at[-1] + 1)
    # The leg each bar belongs to. `side="right"` puts a pivot bar at the start of its own leg
    # (x = 0), and the clip keeps the last pivot as the end of the final leg (x = 1).
    k = np.clip(np.searchsorted(at, i, side="right") - 1, 0, len(at) - 2)
    start, end = at[k], at[k + 1]

    c = close.to_numpy()
    time_advance = (i - start) / (end - start)
    excursion = c[end] - c[start]
    # A leg with zero excursion has no price advancement to measure — two pivots at the same close.
    # Rare, but it would divide by zero, so those bars fall back to the time term alone.
    price_advance = np.divide(c[i] - c[start], excursion, out=time_advance.copy(), where=excursion != 0)
    # The spec states both terms are in [0, 1]. Time is by construction; price is not: a bar inside
    # the leg can trade beyond either pivot without being a pivot itself (that needs `window` bars
    # clear on both sides). Clipping keeps the label monotone in the pivot values and inside
    # [-1, +1] instead of overshooting on a spike.
    x = smoothing * time_advance + (1 - smoothing) * np.clip(price_advance, 0.0, 1.0)

    values = piv.kind.to_numpy().astype(float)  # -1 on a low, +1 on a high
    if significance:
        # tanh with no scale factor: the ratio is already O(1) by construction, so the spec's
        # formula needs no constant of its own. NaN significance propagates into the two legs that
        # touch that pivot, which is the intended behaviour — an unscorable leg has no label.
        values = values * np.tanh(leg_significance(close, piv).to_numpy())
    out.iloc[at[0] : at[-1] + 1] = values[k] + x * (values[k + 1] - values[k])
    return out


def remaining_excursion(
    close: pd.Series,
    pivots: pd.DataFrame | None = None,
    window: int = EXTREMA_WINDOW,
    horizon: int = EXTREMA_WINDOW,
    lookback: int = SIGNIFICANCE_LOOKBACK,
) -> pd.Series:
    """The predictive label: how far the price still has to travel before the leg ends.

        y(i) = log(close[pivot_successivo] / close[i]) / (sigma_i * sqrt(horizon))

    Numerator entirely in the future, denominator entirely in the past. That asymmetry is the
    whole point. `swing_leg_target` is dominated by `(i - inizio) / (fine - inizio)`, whose
    numerator is already known at `i` and whose denominator is nearly constant across the
    cross-section — so a linear model on point-in-time features reproduces it (Rank IC 0.38 out of
    sample, flat at every distance from the pivot, which rules out a pipeline fault). It measures
    where the bar is, not where the price is going. This one cannot be read off the past.

    Sign and magnitude both carry meaning: positive when the leg ends on a high (there is still a
    rise to capture), negative when it ends on a low, zero on the pivot itself. In units of a
    `horizon`-bar random walk at the volatility measured at `i` — so it compares across symbols
    and regimes, and reads as the P&L of holding to the pivot, in units of risk.

    No significance weighting here, and no tanh: a degenerate leg goes nowhere, so it scores near
    zero on its own. The correction `swing_leg_target` needs is built into the quantity.

    Unbounded and fat-tailed, unlike the old label — the Huber delta of 0.4 was measured on that
    distribution and does not carry over; on this one it is 2.1.

    Both defaults are counted in bars *of the series passed in*, and the dataset passes the 5m one
    while the legs are defined on 15m. So `horizon = 24` is the 2h walk the excursion is measured
    against, not the 6h of a 24-bar 15m window, and `lookback = 96` is 8h of volatility rather
    than the day it means on the 15m grid. Neither number is wrong here — a shorter volatility
    window tracks the regime the bar is actually in, and the reference horizon only sets the unit
    — but they are not the constants their own docstrings describe, and delta = 2.1 was measured
    with exactly these. Changing either one means measuring delta again.
    """
    piv = find_pivots(close, window) if pivots is None else pivots
    out = pd.Series(np.nan, index=close.index, name="remaining_excursion")
    if piv.empty:
        return out

    at = close.index.get_indexer(piv.index)
    if (at < 0).any():
        raise ValueError("pivots do not belong to this close series")

    # Bars up to the last confirmed pivot; past it there is no next pivot and no label, which is
    # the permanent condition of the current bar in production. `side="left"` makes a pivot bar
    # its own next pivot, so the label is exactly 0 there.
    i = np.arange(at[-1] + 1)
    nxt = at[np.searchsorted(at, i, side="left")]
    c = close.to_numpy()
    scale = bar_sigma(close, lookback).to_numpy()[i] * np.sqrt(horizon)
    out.iloc[i] = np.log(c[nxt] / c[i]) / scale
    return out


# Bars of the series passed in, like `remaining_excursion`'s own defaults. 288 is 72h on the 15m
# grid the legs live on. Measured over three walk-forward folds on fifteen symbols, with the label
# horizon purged exactly out of each train side: Rank IC 0.0592 +- 0.0216 at 72h against
# 0.0488 +- 0.0199 at 12h. The difference is inside the fold spread, but 72h wins each of the
# three folds separately (0.056 / 0.040 / 0.082 against 0.043 / 0.032 / 0.071) — the same standard
# by which step 3 was promoted over step 2.
#
# The horizon is not only a signal question, and this is the half that decides it. What a rule has
# to clear is roughly `2c / sigma_H`, and `sigma_H` grows with sqrt(h): `simulation` puts the
# break-even Rank IC at 25bp per side at 0.125 for a 12h label and 0.063 for a 72h one, at the same
# threshold. Both terms move the right way at once, which nothing else on the list does.
#
# What it costs: 51 independent cross-sections a year instead of 306, so every number measured on
# this label carries an error bar four times wider, and the naive Rank ICIR over dates overstates
# its own significance by about sqrt(72). Read `simulation`'s blocked error, never the raw ratio.
CROSS_HORIZON = 288
# The metric already refuses a date with fewer than three symbols, so a label computed on two is a
# number no evaluation would read. Same floor, stated once here.
MIN_SYMBOLS = 3


def cross_sectional_return(panel: pd.DataFrame, horizon: int = CROSS_HORIZON) -> pd.DataFrame:
    """The forward log return over `horizon` bars, standardised across the symbols of each row.

        y(i, t) = [ log(close[i, t+h] / close[i, t]) - mean_t ] / sd_t

    One column per symbol, NaN in the last `horizon` rows and wherever the cross-section is too
    thin to standardise.

    **The numerator removes the market.** `swing_leg_target` and `remaining_excursion` both let a
    model be paid for being short through a falling market, and over the twelve months from
    2025-09 that is exactly where their P&L came from — 0.91 of it on the short side against 0.02
    on the long. A model cannot earn a beta the label no longer contains, so what is left is what
    it knows. This is the whole reason the label exists.

    **The denominator is the dispersion of the date and not the symbol's own volatility**, which
    is the opposite of what readability would suggest and is a measured choice. `sd_t` is one
    number per row, so it leaves the ranking *inside* a timestamp untouched: ranking by this label
    is ranking by raw excess return, which is exactly what an equal-weight book earns and exactly
    what `threshold.positions` trades. Dividing by `sigma_i * sqrt(h)` instead ranks by *risk
    adjusted* excess return — a different question, and one the features answer far worse. Three
    walk-forward folds on fifteen symbols:

        horizon   sd_t                sigma_i
        12h       0.0488 +- 0.0199    0.0213 +- 0.0126
        72h       0.0592 +- 0.0216    0.0148 +- 0.0032

    The per-symbol form costs 60-75% of the Rank IC, in every fold at both horizons. It reads
    better on a single-pair chart and it answers a question the rule does not ask; `app.chart`
    draws the cross-section as a heatmap instead, which is the view this quantity actually has.

    Not deadzoned, which was measured and rejected. Clipping the middle to zero -- the obvious way
    to make BUY and SELL separate cleanly -- costs 42% of the Rank IC (0.024 against 0.041 on the
    same rows). Under a squared loss a zeroed row is not a sharper decision, it is a deleted
    gradient: the model spends capacity learning to output 0 and the informative rows that survive
    are the tail, where the label is noisiest. Separation belongs in the rule that reads the
    prediction, never in the label.

    What one unit of it is worth, measured over ten symbols at h = 48 bars of 15m: y = +1 is about
    +1.4% of excess return, and the round trip costs 0.50%, so |y| ~ 0.35 is where a trade stops
    paying for itself. That is the threshold on the *prediction*, which shrinks towards zero, and
    not on the label.

    Purging is exact and fixed here: a row's label ends `horizon` bars later and nowhere else,
    unlike the unbounded reach of `next_pivot` (p99 202 bars, max 754).
    """
    r = np.log(panel.shift(-horizon) / panel)
    enough = r.notna().sum(axis=1) >= MIN_SYMBOLS
    y = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1), axis=0)
    return y.where(enough, np.nan).replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    # Triangle wave: peaks every 20 bars, troughs halfway between. Price is piecewise linear in
    # time, so both advancements agree and the label is a clean ramp.
    x = pd.Series(np.abs(np.arange(600) % 20 - 10.0) + 100.0, index=pd.RangeIndex(600))
    piv = find_pivots(x, 5)
    flat_pivots = swing_leg_target(x, piv, significance=False)

    assert np.allclose(flat_pivots.loc[piv.index], piv.kind), "without weighting pivots carry +/-1"
    assert flat_pivots.dropna().between(-1, 1).all(), "the label never leaves [-1, +1]"
    first, last = x.index.get_indexer(piv.index)[[0, -1]]
    assert flat_pivots.iloc[:first].isna().all(), "no label before the first pivot"
    assert flat_pivots.iloc[last + 1 :].isna().all(), "no label after the last one"
    assert flat_pivots.iloc[first : last + 1].notna().all(), "every bar between two pivots has a label"
    leg = flat_pivots.iloc[first : first + 11].diff().dropna()  # one full leg
    assert (leg > 0).all() or (leg < 0).all(), "the label is monotone along a leg"
    # Halfway along a linear leg both terms read 0.5, so the blend is 0.5 whatever the weight.
    assert abs(flat_pivots.iloc[first + 5]) < 1e-9, "midpoint of a symmetric leg sits at 0"
    for s in (0.0, 1.0):  # the extremes of the blend stay valid labels
        assert np.allclose(swing_leg_target(x, piv, smoothing=s, significance=False).loc[piv.index], piv.kind)

    # Weighted: every swing of the triangle is identical, so every pivot gets the same value, and
    # it keeps the sign of its pivot while staying inside the flat labelling.
    weighted = swing_leg_target(x, piv)
    at_pivots = weighted.loc[piv.index].dropna()
    assert np.allclose(np.sign(at_pivots), piv.kind.reindex(at_pivots.index)), "the sign is the pivot's"
    assert at_pivots.abs().max() < 1, "a weighted pivot never reaches 1"
    # Not exactly equal: in log terms an up leg and a down leg of the same linear ramp differ
    # slightly, which is the correct behaviour and lands well below a thousandth here.
    assert at_pivots.abs().std() < 1e-3, "identical swings must score identically"

    # A leg that goes nowhere is the case the weighting exists for: same pivots, but the last
    # swing collapsed to a tenth of the amplitude. Its closing pivot has to be pulled towards 0.
    damped = x.copy()
    damped.iloc[-30:] = 100.0 + (damped.iloc[-30:] - 100.0) / 10
    dp = find_pivots(damped, 5)
    dw = swing_leg_target(damped, dp).loc[dp.index].dropna()
    # A tenth of the amplitude scores nine times lower, but the value only drops threefold: the
    # tanh is already flat at the top, which is exactly the compression it is there to provide.
    assert abs(dw.iloc[-1]) < abs(at_pivots.iloc[-1]) / 2, "a collapsed leg must lose most of its value"

    # The predictive label on the same wave: zero on every pivot, and shrinking along each leg
    # since the distance left to travel only decreases.
    rem = remaining_excursion(x, piv, lookback=50)
    assert np.allclose(rem.loc[piv.index].dropna(), 0, atol=1e-12), "no distance left at a pivot"
    assert rem.iloc[last + 1 :].isna().all(), "no label past the last pivot"
    up = rem.iloc[last - 9 : last + 1].dropna()  # the leg closing on the final pivot
    assert (up.diff().dropna() * np.sign(up.iloc[0]) < 0).all(), "the excursion left shrinks along a leg"
    assert np.sign(up.iloc[0]) == piv.kind.iloc[-1], "the sign is the direction towards the next pivot"
    # Not each other's mirror image, even on a wave this regular: the old label ramps once from
    # pivot to pivot, the new one resets at every pivot. Different shapes, not opposite signs.
    both = pd.DataFrame({"old": weighted, "new": rem}).dropna()
    assert abs(both.old.corr(both.new)) < 0.5

    # The cross-sectional label. A shared noisy walk plus a per-symbol drift: the noise cancels
    # in the demeaning, so the sign of each column is known in advance.
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.01, 400))
    step = 0.001 * np.arange(400.0)
    panel = pd.DataFrame(
        {"up": np.exp(common + step), "flat": np.exp(common), "down": np.exp(common - step)},
        index=pd.RangeIndex(400),
    )
    y = cross_sectional_return(panel, horizon=10)
    assert np.allclose(y.dropna().mean(axis=1), 0, atol=1e-12), "every row is centred on its date"
    assert np.allclose(y.dropna().std(axis=1), 1), "and scaled by its own dispersion"
    assert (y.up.dropna() > 0).all() and (y.down.dropna() < 0).all(), "the sign is the relative move"
    assert y.iloc[-10:].isna().all().all(), "the last horizon bars have no forward return"
    # Market neutral, stated as the equality it is: a move common to every symbol -- any size, any
    # sign -- leaves every label untouched. This is the property the whole label exists for, and
    # with `sd_t` in the denominator it holds for a random common path and not only a smooth one.
    shocked = panel.mul(np.exp(np.cumsum(rng.normal(0.001, 0.004, 400))), axis=0)
    assert np.allclose(
        cross_sectional_return(shocked, horizon=10).dropna(), y.dropna()
    ), "a common move is not information"
    # A market that moves as one has no dispersion and the label is 0/0. NaN and not 0 — "nothing
    # to rank here" is not the same statement as "this symbol is average", and only the first is true.
    same = pd.DataFrame({c: np.exp(0.002 * np.arange(50.0)) for c in "abc"})
    assert cross_sectional_return(same, horizon=5).isna().all().all()
    # Under three symbols there is no cross-section, and the metric drops the date anyway.
    assert cross_sectional_return(panel[["up", "down"]], horizon=10).isna().all().all()

    flat = pd.Series([1.0] * 600, index=pd.RangeIndex(600))
    assert swing_leg_target(flat, window=5).isna().all(), "no pivots, no label"
    assert remaining_excursion(flat, window=5).isna().all()
    print(
        f"ok — {weighted.notna().sum()} labelled bars, pivot values |{at_pivots.abs().min():.3f}|"
        f"..|{at_pivots.abs().max():.3f}|, collapsed leg {abs(dw.iloc[-1]):.3f}"
    )
