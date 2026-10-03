"""Does any single column carry signal *near* the pivot? The cheap check the spec puts before step 3.

Step 2 closed with a warning: the GBM lifted the aggregate Rank IC to 0.156 but left the profile by
distance to the next pivot alone, and under 24 bars that profile is still negative — the momentum
features say "continue" exactly where the leg is about to turn. Two readings fit that. Either the
information about an ending leg is in the 28 columns and no flat model composes it, which is the
question step 3 exists to answer; or it is not in the columns at all, in which case a GRU will
reproduce the same profile at a much higher price and the next month belongs to exhaustion features
(divergence, volume climax, wick asymmetry) instead.

This separates the two for the cost of a few minutes. No model: the Rank IC of each raw column
against the target, restricted to the bars whose next pivot is within `band`. If nothing clears the
noise floor there, the information is not in the inputs and no architecture recovers it.

Train period only, purged, like every other measurement that *decides* something. The floor is
measured and not assumed: pure-noise columns go through the identical pipeline, and a real column
counts only if it beats their spread — with a narrow band the surviving cross-sections are thin, so
the sampling error is much larger than the aggregate numbers elsewhere in the project suggest.

    uv run python -m tradingvision.nearpivot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import dataset, metrics, split
from tradingvision.data.binance import STORE
from tradingvision.dataset import cached
from tradingvision.features import COLUMNS

CACHE = STORE / "step2.parquet"
BAND = 24  # in 5m bars: the band where step 1 and step 2 both go negative
NULLS = 5  # noise columns, enough for a spread rather than a point


def value_columns(columns: list[str], branches: tuple[str, ...] = dataset.BRANCHES) -> list[str]:
    """The value-at-t columns, four branches of the 28 candidates. The lags and window statistics
    are left out: this asks whether an indicator sees the turn, not whether its own past does."""
    return [f"{name}_{tf}" for tf in branches for name in COLUMNS if f"{name}_{tf}" in set(columns)]


def near(df: pd.DataFrame, band: int = BAND) -> pd.DataFrame:
    """The rows whose next pivot is less than `band` 5m bars ahead."""
    horizon = (df.next_pivot - df.index.get_level_values(0)) / pd.Timedelta("5min")
    return df[horizon < band]


def rank_ic(x: pd.Series, target: pd.Series) -> dict[str, float]:
    """Rank IC of one column, with the dispersion that says whether to believe it."""
    per_date = metrics.by_date(x, target, rank=True).dropna()
    mean, n = per_date.mean(), len(per_date)
    return {"rank_ic": mean, "dates": n, "t": mean / per_date.std() * np.sqrt(n) if n > 1 else np.nan}


def run(df: pd.DataFrame, band: int = BAND, nulls: int = NULLS, seed: int = 0) -> pd.DataFrame:
    """Every value column plus `nulls` noise columns, ranked by |Rank IC| on the band."""
    band_rows = near(df, band)
    if band_rows.empty:
        raise ValueError(f"no bar has its next pivot within {band} bars")
    rng = np.random.default_rng(seed)
    noise = {f"__null{i}": pd.Series(rng.normal(size=len(band_rows)), index=band_rows.index) for i in range(nulls)}

    rows = {c: rank_ic(band_rows[c], band_rows.target) for c in value_columns(list(df.columns))}
    rows |= {c: rank_ic(v, band_rows.target) for c, v in noise.items()}
    out = pd.DataFrame(rows).T
    out["abs"] = out.rank_ic.abs()
    return out.sort_values("abs", ascending=False)


def floor(out: pd.DataFrame) -> float:
    """The noise floor: the largest |Rank IC| any pure-noise column reached on the same rows."""
    return out.loc[out.index.str.startswith("__null"), "abs"].max()


def _selfcheck() -> None:
    """A planted column is found on the band and the noise columns are not, so a flat reading on
    the real data is the data talking."""
    idx = pd.MultiIndex.from_product([pd.date_range("2024", periods=400, freq="h", tz="UTC"), list("abcde")])
    rng = np.random.default_rng(0)
    when = idx.get_level_values(0)
    # Half the rows sit inside the band, half far outside it.
    ahead = pd.Series(np.where(rng.random(len(idx)) < 0.5, 60, 6000), index=idx)  # in minutes
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    df = pd.DataFrame(
        {
            # Named like a real column so `value_columns` picks it up, and correlated with the
            # target only where the pivot is close — which is the whole point of the band.
            "ema_slope_5m": np.where(ahead < 120, y * 3, rng.normal(size=len(idx))),
            "log_return_5m": rng.normal(size=len(idx)),
            "target": y,
            "next_pivot": when + pd.to_timedelta(ahead, unit="m"),
        },
        index=idx,
    )
    out = run(df, band=24, nulls=3)
    assert out.index[0] == "ema_slope_5m" and out.iloc[0]["rank_ic"] > 0.8, out
    assert out.loc["log_return_5m", "abs"] <= floor(out) * 2, "an unrelated column must sit at the floor"
    assert len(near(df, 24)) == (ahead < 120).sum()
    # Unrestricted, the same column is half signal and half noise: the band is the measurement.
    assert abs(run(df, band=10**6).loc["ema_slope_5m", "rank_ic"]) < 0.8
    try:
        run(df, band=1)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty band has to raise instead of reporting NaN")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023")
    ap.add_argument("--test-start", default="2025-06", help="everything before it is train; nothing after is read")
    ap.add_argument("--band", type=int, default=BAND, help="5m bars to the next pivot")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    _selfcheck()
    df = cached(args.cache, start=args.start, stride=args.stride, lags=True)
    train, _ = split.temporal(df, args.test_start)
    out = run(train, args.band)
    rows = near(train, args.band)
    print(f"{len(rows):,} of {len(train):,} train bars within {args.band} bars of the next pivot\n")
    print(out.head(args.top).round(4).to_string())
    f = floor(out)
    beat = out[(out["abs"] > f) & ~out.index.str.startswith("__null")]
    print(f"\nnoise floor {f:.4f} ({NULLS} null columns) — {len(beat)} of {len(out) - NULLS} columns beat it")


if __name__ == "__main__":
    main()
