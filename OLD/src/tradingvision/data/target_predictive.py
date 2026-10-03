"""The three predictive labels, archived — each one measured against the price and dropped.

Moved here from `tradingvision.data.target` on 2026-10-03, verbatim. The live module keeps only
`swing_leg_target` and the volatility it is weighted by. Why each of these left is in
`OLD/README.md`; the code runs as it stands at the git tag `archive-predictive`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import SIGNIFICANCE_LOOKBACK, bar_sigma, swing_leg_target

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
# its own significance by about sqrt(72). Measured on the first cross-sectional run: a naive t over
# 10,944 dates reads 31.9, and the same numbers over 153 non-overlapping 72h blocks read 5.1.
# `metrics.signal(..., horizon=...)` reports the blocked error; never read the raw ratio.
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


# Bars ahead the move is looked for, in bars of the series passed in: 48 is 12h on the 15m grid.
# Chosen, not tuned against a model. move_balance_label.html measures the label at 12..192 bars:
# in units of sigma it barely changes (median |y| 0.67-0.76, Spearman with the forward return
# 0.855-0.858 at every horizon); in percent it grows about as sqrt(N), and at 48 the move clears
# the 0.5% round trip on 83% of rows against 66% at 12. What N trades is cost against the number
# of independent windows — 1,763 non-overlapping blocks over the train period at 48.
MOVE_HORIZON = 48


def move_balance(close: pd.Series, horizon: int = MOVE_HORIZON, lookback: int = SIGNIFICANCE_LOOKBACK) -> pd.Series:
    """The rise ahead minus the fall ahead, over the next `horizon` closes, in units of chance.

        rise(t) = max(0, max_j close[t+j] / close[t] - 1)      j = 1..horizon
        fall(t) = max(0, 1 - min_j close[t+j] / close[t])
        y(t)    = (rise - fall) / (sigma_t * sqrt(horizon))

    Positive when the price reaches further above the close than below it, negative the other way;
    over 100 -> 102, 101, 97 the numerator is +2% - 3% = -1%. A side the price never visits counts
    zero, so a window that only falls is its whole fall and not the fall less the smallest decline.
    `sigma_t` is `bar_sigma` over the `lookback` bars up to t, so the denominator is what a walk of
    the volatility of the moment covers in `horizon` bars — past only, like `remaining_excursion`'s.

    It began as the largest move with its sign, `r[argmax |r|]`, and two measurements over twenty
    15m pairs, 2023-01 -> 2025-05, N = 24, replaced both halves of that.

    *The argmax jumps.* Where the rise and the fall ahead are about the same size, a fraction of a
    bar moves the label from one side to the other: 11.2% of bars changed sign from the bar before,
    by a median 1.97% against a median one-bar move of 0.19%. The distribution had two humps at
    about +-1.6% and a hollow between them, only 0.8% of rows under 0.25%, which is where a
    regression's conditional mean lands. The difference of the two sides is continuous in the price
    and says the same thing — which side reaches further — without the jump.

    *The size was the volatility.* |y| had Spearman 0.56 with the trailing volatility, so a pooled
    loss was spending itself on the volatile pairs and regimes, and a model could lower it by
    reading how agitated the market is, which is known and says nothing of direction.

    What the first form had and this one must keep: its sign was the sign of the 24-bar forward
    return on 84.5% of rows (Spearman 0.85), which is what `swing_leg_target` lacked — imitated at
    Rank IC 0.41, it read -0.04 against the price.

    The same rows, this form: the sign agrees with the forward return on 84.4% (Spearman 0.855),
    so nothing was lost; |y| against the trailing volatility is -0.095 instead of 0.56; one hump
    at zero instead of two. The sign still changes on 11.4% of bars — near zero it has to — but
    by a median 0.43 against 0.195 when it does not, in units where a bar moves 0.116: a crossing,
    not a jump. The argmax's crossings were ten one-bar moves wide.

    Entirely in the future, bounded, and exact to purge: a row's label ends `horizon` bars later.
    NaN on the last `horizon` bars, whose window is not complete, and until the volatility fills.
    """
    if horizon < 1:
        raise ValueError(f"the horizon is a number of bars ahead, not {horizon}")
    # Reversed, a trailing window over bars t..t+h-1 of the original; shifted by one it covers
    # t+1..t+h, the bars ahead of t and not t itself.
    back = close.iloc[::-1].rolling(horizon, min_periods=horizon)
    rise = (back.max().iloc[::-1].shift(-1) / close - 1).clip(lower=0)
    fall = (1 - back.min().iloc[::-1].shift(-1) / close).clip(lower=0)
    return ((rise - fall) / (bar_sigma(close, lookback) * np.sqrt(horizon))).rename("move_balance")


if __name__ == "__main__":
    x = pd.Series(np.abs(np.arange(600) % 20 - 10.0) + 100.0, index=pd.RangeIndex(600))
    piv = find_pivots(x, 5)
    weighted = swing_leg_target(x, piv)
    last = x.index.get_indexer(piv.index)[-1]

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

    # The balance ahead. The example the definition was agreed on, with the scale taken out, then
    # a brute force over a random walk: every row's window read literally.
    def numerator(ahead):
        """The label at the close of 100 after a three-bar warm-up, times its scale: rise - fall."""
        c = pd.Series([100.0, 101, 100, 100] + list(ahead))
        return (move_balance(c, len(ahead), 2) * bar_sigma(c, 2) * np.sqrt(len(ahead))).iloc[3]

    assert np.isclose(numerator([102, 101, 97]), -0.01), "+2% - 3%"
    assert np.isclose(numerator([99, 98, 97]), -0.03), "a side never visited counts zero"
    # Continuous where the argmax jumped: two windows whose sides are nearly equal, a tenth of a
    # percent apart. The argmax read +2% on one and -2.05% on the other; the balance moves 0.1%.
    a1, a2 = numerator([102, 101, 98.05, 100]), numerator([102, 101, 97.95, 100])
    assert a1 > 0 > a2 and np.isclose(a1 - a2, 0.001), (a1, a2)
    walk = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    mb = move_balance(walk, horizon=7, lookback=20)
    assert mb.iloc[-7:].isna().all() and mb.iloc[20:-7].notna().all(), "NaN on the warm-up and the last horizon bars"
    c, sig = walk.to_numpy(), bar_sigma(walk, 20).to_numpy()
    for t in range(20, len(c) - 7):
        r = c[t + 1 : t + 8] / c[t] - 1
        want = (max(r.max(), 0) - max(-r.min(), 0)) / (sig[t] * np.sqrt(7))
        assert np.isclose(mb.iloc[t], want), t
    assert move_balance(walk.iloc[:100], 7, 20).iloc[:93].equals(mb.iloc[:93]), "each row reads its own window only"
    down = pd.Series(100 * np.exp(-0.01 * np.arange(60.0) + rng.normal(0, 1e-4, 60)))
    assert (move_balance(down, 5, 10).dropna() < 0).all(), "a price that only falls has no rise"

    flat = pd.Series([1.0] * 600, index=pd.RangeIndex(600))
    assert remaining_excursion(flat, window=5).isna().all()
    print("ok")
