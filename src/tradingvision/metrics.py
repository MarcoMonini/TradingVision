"""The four qlib signal metrics, computed cross-sectionally.

Per timestamp, correlate the predictions of the symbols traded at that instant against their
targets; then average over time (IC) and divide by the dispersion (ICIR). Rank IC is the same on
the ranks and is the primary metric of the spec: Rank IC > 0.02-0.03 and Rank ICIR > 0.3 mark a
signal worth promoting.

With 20 symbols a single cross-section has a standard error of ~0.24, so the individual values are
noise and only the average over thousands of dates carries information. That is why the ratio
matters more than the level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def by_date(pred: pd.Series, target: pd.Series, rank: bool = False) -> pd.Series:
    """Correlation inside each timestamp, dropping the dates that hold fewer than three symbols.

    Spearman is Pearson on the ranks, computed that way here so scipy stays out of the dependency
    list — `Series.corr(method="spearman")` imports it.
    """
    df = pd.DataFrame({"p": pred, "y": target}).dropna()
    g = df.groupby(level=0)
    if rank:
        df = g.rank()
        g = df.groupby(level=0)
    # Pearson by hand: the mean of the standardised product over each date. Cheaper than a
    # group-apply of Series.corr, and it keeps the whole thing vectorised.
    size = g.p.transform("size")
    z = (df - g.transform("mean")) / g.transform("std")
    n = size.groupby(level=0).first()
    return ((z.p * z.y).groupby(level=0).sum() / (n - 1)).where(n > 2)


def signal(pred: pd.Series, target: pd.Series) -> dict[str, float]:
    """IC, ICIR, Rank IC, Rank ICIR for one set of predictions. Both series share a
    (timestamp, symbol) index, timestamp first."""
    out = {}
    for name, rank in (("ic", False), ("rank_ic", True)):
        per_date = by_date(pred, target, rank).dropna()
        out[name] = per_date.mean()
        out[f"{name}ir"] = per_date.mean() / per_date.std()
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=200, freq="h"), list("abcdef")])
    y = pd.Series(rng.normal(size=len(idx)), index=idx)

    assert np.isclose(signal(y, y)["ic"], 1.0) and np.isclose(signal(y, y)["rank_ic"], 1.0)
    assert np.isclose(signal(-y, y)["ic"], -1.0)
    # Monotone but not linear: Rank IC stays perfect, IC does not.
    m = signal(y**3, y)
    assert np.isclose(m["rank_ic"], 1.0) and m["ic"] < 0.95
    # No signal: both near zero, and the ratio stays finite.
    noise = signal(pd.Series(rng.normal(size=len(idx)), index=idx), y)
    assert abs(noise["rank_ic"]) < 0.05 and abs(noise["rank_icir"]) < 1
    # Same as pandas, which needs scipy for the ranks.
    one = y.loc["2024-01-01 00:00"]
    assert np.isclose(by_date(y**3, y, rank=False).iloc[0], (one**3).corr(one))
    print("ok — no signal:", {k: round(v, 3) for k, v in noise.items()})
