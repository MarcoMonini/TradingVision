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


def blocked(per_date: pd.Series, horizon: pd.Timedelta) -> dict[str, float]:
    """`per_date` collapsed onto non-overlapping blocks of `horizon` — the honest denominator.

    A per-date IC series is not a sample of independent numbers when the label reaches forward.
    At a 72h horizon and an hourly grid, two adjacent dates share 71 of the 72 hours their labels
    are made of, so the series is smooth by construction: the dispersion measured across dates is
    far below the dispersion across independent draws, and `mean / std` — which is the ICIR, and
    the number this project promotes a step on — is inflated by roughly the square root of the
    overlap. The same applies to the standard error of the mean, which is why a raw `sem` over
    27,812 dates would claim a precision that 150 blocks of data cannot support.

    Averaging inside each block and then reading mean, dispersion and error across blocks removes
    the overlap instead of modelling it. Blocks are cut on the clock, so two runs on the same
    period compare block for block whatever rows each of them happens to hold.
    """
    block = per_date.groupby(per_date.index.floor(horizon)).mean()
    n = len(block)
    sd = block.std()
    se = sd / np.sqrt(n) if n > 1 else np.nan
    return {
        "mean": float(block.mean()),
        "se": float(se),
        "t": float(block.mean() / se) if n > 1 and sd > 0 else np.nan,
        "ir": float(block.mean() / sd) if n > 1 and sd > 0 else np.nan,
        "blocks": n,
    }


def signal(pred: pd.Series, target: pd.Series, horizon: pd.Timedelta | str | None = None) -> dict[str, float]:
    """IC, ICIR, Rank IC, Rank ICIR for one set of predictions. Both series share a
    (timestamp, symbol) index, timestamp first.

    `horizon` is how far the label reaches. Given, it adds the three numbers that are readable when
    the labels of adjacent dates overlap, all of them from non-overlapping blocks: `rank_ic_se`,
    the standard error of the Rank IC; `rank_ic_t`, the mean over that error; and `blocks`, how
    many independent observations the whole run actually holds.

    `rank_ic_t` is the one to read next to a difference between two steps. `rank_icir` stays in
    the output unchanged — every number already measured in this project was taken on it — but at
    a 72h label on an hourly grid it is not a statement about significance: adjacent dates share
    71 of the 72 hours their labels are made of, so the dispersion in its denominator is that of a
    72-point moving average and the ratio reads several times too high. The count of dates it is
    implicitly divided by is wrong by the same construction, which is what `blocks` states.
    """
    out = {}
    for name, rank in (("ic", False), ("rank_ic", True)):
        per_date = by_date(pred, target, rank).dropna()
        out[name] = per_date.mean()
        out[f"{name}ir"] = per_date.mean() / per_date.std()
        if rank and horizon is not None:
            b = blocked(per_date, pd.Timedelta(horizon))
            out["rank_ic_se"], out["rank_ic_t"], out["blocks"] = b["se"], b["t"], b["blocks"]
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

    # The overlap correction, on the shape the real IC series has: an underlying signal that is
    # independent block by block, observed through a moving average as wide as the label horizon.
    base = pd.Series(rng.normal(0.05, 0.1, 2000), index=pd.date_range("2024", periods=2000, freq="h"))
    overlapped = base.rolling(72, min_periods=1).mean()
    b = blocked(overlapped, pd.Timedelta("72h"))
    assert b["blocks"] == 29, b  # 2000 hourly dates fall into 29 blocks of 72h
    # The mean survives the smoothing; the error bar does not. Divided by the count of dates, the
    # same series claims a t four times larger than 28 independent blocks can support.
    naive_t = overlapped.mean() / (overlapped.std() / np.sqrt(len(overlapped)))
    assert b["t"] < naive_t / 3, (b["t"], naive_t)
    assert np.isclose(b["mean"], overlapped.mean(), rtol=0.05)
    full = signal(y, y, horizon="12h")
    assert full["blocks"] > 1 and full["rank_ic_se"] >= 0 and "rank_icir" in full
    assert signal(y, y).keys() == {"ic", "icir", "rank_ic", "rank_icir"}, "unasked, the output is unchanged"
    print("ok — no signal:", {k: round(v, 3) for k, v in noise.items()})
