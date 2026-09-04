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
    clipped = ((train - stats.center).abs() / stats.scale > CLIP).mean().mean()
    print(f"ok — IQR {SCALE} on {len(COLUMNS)} columns, bounded at {CLIP * SCALE}, {clipped * 100:.2f}% clipped")
