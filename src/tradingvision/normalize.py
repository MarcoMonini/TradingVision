"""Puts every feature column on one scale, so none of them is ignored during training.

Measured on the store, the 28 raw columns span ~3.2 orders of magnitude: the wick and log-return
columns sit at an IQR of 1e-3 while the bounded ones sit around 1e0. A GRU can in principle
rescale its inputs, but with standard initialisation the gradient on a 1e-3 column is a thousand
times smaller, and weight decay penalises exactly the large weights that would compensate — inside
a finite training budget those columns are ignored.

    z = clip((x - median) / IQR, -CLIP, +CLIP) * SCALE

Quantiles rather than mean and standard deviation because the returns have fat tails. The clip is
what the plain quantile scaling is missing: it normalises the body of the distribution and leaves
outliers at up to 380 IQR out of sample, which weigh on the gates as much as everything else.
Measured at CLIP=5 it truncates 0.56% of the values, and the result has an IQR of exactly SCALE on
every column, 0.093-0.111 out of sample, against 3.2 decades of spread before.

Statistics are fitted on the train period alone and applied unchanged to validation, test and
live — refitting them on the full dataset is leakage. `stats` is a plain DataFrame, so persisting
it next to the model is `stats.to_json(path)` / `pd.read_json(path)`.
"""

from __future__ import annotations

import pandas as pd

# Truncation in IQR units, and the scale the body lands on. CLIP trades tail resolution for a
# bounded input: at 3 it cuts 2.0% of the values, at 8 only 0.15% but the extremes reach 0.8.
CLIP = 5.0
SCALE = 0.1
# A timestamp with fewer symbols than this cannot be ranked into anything meaningful, and it is the
# floor `metrics.by_date` already applies when it refuses to correlate a thin cross-section.
MIN_SYMBOLS = 3


def cross_rank(x: pd.DataFrame, min_symbols: int = MIN_SYMBOLS) -> pd.DataFrame:
    """Every column replaced by its percentile rank inside its own timestamp, centred on zero.

    `x` is long: a (timestamp, symbol) MultiIndex, timestamp first. Output is in [-0.5, +0.5], with
    NaN on the whole row wherever the cross-section is thinner than `min_symbols`.

    **Why.** The cross-sectional label is a ranking inside a timestamp, and `metrics.by_date`
    scores a ranking inside a timestamp. A feature read as a level does not match either: a
    `realized_volatility` of 0.004 means "calm" in one week and "the calmest of the twenty" in
    another, and a model splitting on absolute thresholds has to relearn the mapping in every
    regime it meets. Ranking states the only thing the target rewards and drops everything else.
    Measured, the same LightGBM on the same columns goes from Rank IC 0.054 to 0.098 on this
    transform alone — the largest single gain in the pipeline, and it costs one groupby.

    **Why this is not leakage, which it looks like.** `fit`/`apply` are careful to read the train
    period and never the frame they transform. This one reads the frame it transforms — but only
    across the symbols of a single timestamp, which is exactly the information standing in front of
    a trader at that instant. Nothing is read from another bar, so a slice of dates ranks to what
    it ranks to inside the whole. It composes with `apply` rather than replacing it: rank first,
    scale after, and the scaling then has nothing left to do but move a uniform onto `SCALE`.

    **What it costs.** The prediction stops being computable for one symbol on its own. A rank
    needs peers, so live inference needs the panel of whatever is trading at that instant, and a
    single-pair chart cannot draw this any more than it could draw the label.
    """
    g = x.groupby(level=0)
    ranked = g.rank(pct=True)
    # Centred on the mean of its own timestamp and not by subtracting a flat 0.5: `rank(pct=True)`
    # runs 1/n..1, whose mean is 0.5 + 1/(2n). With a cross-section that thins and fills as symbols
    # gap in and out, a flat shift would leave a small offset that moves with the *count* of
    # symbols — a time-varying signal made of nothing, fed to the model as if it were a feature.
    ranked = ranked - ranked.groupby(level=0).transform("mean")
    return ranked.where(g.transform("size") >= min_symbols)


def fit(x: pd.DataFrame) -> pd.DataFrame:
    """Centre and scale per column — the train period only.

    Raises on a column with no dispersion: dividing by it would silently produce infinities, and
    the feature selection is supposed to have dropped it as degenerate first.
    """
    q = x.quantile([0.25, 0.5, 0.75])
    stats = pd.DataFrame({"center": q.loc[0.5], "scale": q.loc[0.75] - q.loc[0.25]})
    flat = stats.index[stats.scale <= 0].tolist()
    if flat:
        raise ValueError(f"no dispersion between the quartiles, drop these columns first: {flat}")
    return stats


def apply(x: pd.DataFrame, stats: pd.DataFrame, clip: float = CLIP, scale: float = SCALE) -> pd.DataFrame:
    """Normalise `x` with statistics fitted elsewhere. Output is bounded by `clip * scale`."""
    missing = stats.index.difference(x.columns)
    if len(missing):
        raise ValueError(f"columns missing from the frame: {list(missing)}")
    return ((x[stats.index] - stats.center) / stats.scale).clip(-clip, clip) * scale


if __name__ == "__main__":
    import numpy as np

    from tradingvision.features import COLUMNS, features

    rng = np.random.default_rng(0)
    n = 4000
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=pd.date_range("2024", periods=n, freq="15min")
    )
    # Asymmetric wicks: with high and low equidistant from the close, close_position_in_bar is
    # 0.5 everywhere and `fit` rightly refuses it.
    o = close.shift().bfill()
    up, down = rng.lognormal(-6, 0.6, n), rng.lognormal(-6, 0.6, n)
    df = pd.DataFrame(
        {
            "open": o,
            "high": pd.concat([o, close], axis=1).max(axis=1) * (1 + up),
            "low": pd.concat([o, close], axis=1).min(axis=1) * (1 - down),
            "close": close,
            "volume": rng.lognormal(0, 1.5, n),
        }
    )
    x = features(df).iloc[96:]
    train, test = x.iloc[: len(x) // 2], x.iloc[len(x) // 2 :]

    stats = fit(train)
    a, b = apply(train, stats), apply(test, stats)
    assert list(a.columns) == COLUMNS, "column order preserved"
    iqr = a.quantile(0.75) - a.quantile(0.25)
    assert np.allclose(iqr, SCALE, atol=1e-9), f"train IQR must be exactly {SCALE}: {iqr.min()}-{iqr.max()}"
    assert a.abs().max().max() <= CLIP * SCALE + 1e-12 and b.abs().max().max() <= CLIP * SCALE + 1e-12
    assert b.notna().all().all() and np.isfinite(b.to_numpy()).all()
    # No leakage: the transform reads the stats, never the frame it is applied to, so a slice
    # normalises to exactly what it does inside the whole.
    assert apply(test.iloc[:50], stats).equals(b.iloc[:50])
    # Monotone per column (non-strictly, the clip flattens the tails), so the Spearman
    # correlations the feature selection measures are intact.
    for c in COLUMNS:
        order = np.argsort(train[c].to_numpy(), kind="stable")
        assert (np.diff(a[c].to_numpy()[order]) >= -1e-12).all(), f"{c} is not monotone"
    try:
        fit(train.assign(dead=1.0))
    except ValueError as e:
        assert "dead" in str(e)
    else:
        raise AssertionError("a column with no dispersion must be refused")

    # Cross-sectional ranking. Four symbols on one grid, one of them always the largest.
    when = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
    syms = list("abcd")
    idx = pd.MultiIndex.from_product([when, syms], names=["ts", "sym"])
    raw = pd.DataFrame({"v": np.tile([4.0, 1.0, 3.0, 2.0], len(when))}, index=idx)
    r = cross_rank(raw)
    assert r.v.between(-0.5, 0.5).all() and r.groupby(level=0).v.nunique().eq(4).all()
    # Four symbols: pct ranks 0.25..1.0 about a mean of 0.625, so the largest sits at +0.375.
    assert r.xs("a", level=1).v.eq(0.375).all() and r.xs("b", level=1).v.eq(-0.375).all()
    # Exactly centred, whatever the count — the reason the mean is subtracted rather than 0.5.
    assert np.allclose(r.groupby(level=0).v.mean(), 0, atol=1e-12)
    three = cross_rank(raw.drop(index=[(t, "d") for t in when]))
    assert np.allclose(three.groupby(level=0).v.mean(), 0, atol=1e-12), "and with three of them too"
    # The property the transform exists for: any change common to a whole timestamp is invisible.
    # Multiplying a date by ten or shifting it by a hundred cannot move a single rank, which is
    # what makes the feature mean the same thing in a calm week and a violent one.
    per_date = pd.Series(rng.lognormal(0, 2, len(when)), index=when)
    warped = raw.mul(per_date.reindex(idx, level=0), axis=0).add(per_date.reindex(idx, level=0) * 100, axis=0)
    assert cross_rank(warped).equals(r), "a per-timestamp monotone map must leave the ranks alone"
    # Reads inside a timestamp and never across time, so a slice of dates ranks to what it ranks
    # to inside the whole — the leakage question `fit` answers with train-only statistics.
    assert cross_rank(raw.loc[when[:10]]).equals(r.loc[when[:10]])
    # A cross-section too thin to rank is NaN and not 0: "no peers here" is not "average".
    thin = raw.drop(index=[(when[0], s) for s in "bcd"])
    assert cross_rank(thin).loc[when[0]].v.isna().all() and cross_rank(thin).loc[when[1]].notna().all().all()

    clipped = ((train - stats.center).abs() / stats.scale > CLIP).mean().mean()
    print(f"ok — IQR {SCALE} on {len(COLUMNS)} columns, bounded at {CLIP * SCALE}, {clipped * 100:.2f}% clipped")
