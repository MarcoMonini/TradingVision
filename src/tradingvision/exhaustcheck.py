"""Do the exhaustion features lead the price, where everything else only summarised it?

This is the one road the spec left open, and the only one that could change the *sign* of the
project's result rather than its precision. Everything measured so far predicts the swing label
and not the price: `legcheck` on the trained model gives Rank IC 0.4114 against the label and
**-0.0405 against the forward return**, with the buy decile significantly negative at 24 and 72
bars. The stated reason is that the label is ~70% a clock — `elapsed / (elapsed + remaining)` —
so a model fitted to it learns where the bar sits in a leg, which is arithmetic on the past.

Open point 4 names the alternative in one line: *feature di esaurimento — divergenza, climax di
volume, asimmetria dei wick — invece che di momentum*. `legs.exhaustion` computes six of them and
they are all causal, but nothing has ever isolated them, so nobody knows whether they carry a
lead. This module asks that directly, and asks it of the price rather than of any label.

**What it measures.** For each of the six columns, the cross-sectional Rank IC against the
market-neutral forward return at 6, 24 and 72 bars, per fold, with the dispersion across folds —
because the project's own rule is that a comparison without a dispersion is not a comparison, and
a single cross-section of twenty symbols has a standard error near 0.24.

**Why no model.** Fitting one first would confound two questions. If a GRU on these columns came
out flat, the flatness could be the columns or the fit; a univariate Rank IC per column cannot be
blamed on an architecture, a threshold or a fee. That is the same reason `legcheck` has no rule
inside it. A model is worth building only for a column that already leads on its own.

**The bar to clear.** Not zero — a column has to beat what the trained model already gives against
price, which is -0.0405 (i.e. |IC| 0.04 in the *useful* direction rather than the inverted one),
and it has to do it with a sign that is stable across the four folds. A |Rank IC| of 0.02 that
flips sign between folds is noise on twenty symbols.

Two honest limits. These are univariate reads, so a column that only works in combination will
look flat here; and the forward return is the raw next-`h`-bar return, not a return conditioned on
being near a turn, which is the regime the features are designed for. Both are reasons a negative
result here is weaker evidence than a positive one would have been — stated up front so the
conclusion is not read as stronger than it is.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision import legcheck, legs
from tradingvision.data.binance import SYMBOLS, load

HORIZONS = (6, 24, 72)
FOLDS = 4


def panel(symbols: list[str], timeframe: str = "15m", first: str | None = None) -> pd.DataFrame:
    """The six exhaustion columns plus the close, for every symbol, on one (time, symbol) index.

    Built per symbol and concatenated rather than pivoted: `legs.exhaustion` reads rolling windows
    along a single series, so it has to see each symbol's own bars in order and unmixed.
    """
    frames = []
    for symbol in symbols:
        bars = load(symbol, timeframe)
        if first:
            # Compute on the whole history and cut afterwards: the rolling windows reach back
            # `4 * EXTREMA_WINDOW` bars, so cutting first would leave the opening rows NaN.
            out = legs.exhaustion(bars).assign(close=bars.close)
            out = out[out.index >= pd.Timestamp(first, tz="UTC")]
        else:
            out = legs.exhaustion(bars).assign(close=bars.close)
        frames.append(out.assign(symbol=symbol).set_index("symbol", append=True))
    return pd.concat(frames).sort_index()


def ic_by_fold(x: pd.Series, fwd: pd.Series, folds: int = FOLDS) -> dict[str, float]:
    """Rank IC of one column against one forward return, pooled and split into `folds` by time.

    The folds are contiguous slices of the dates, not random ones: the question is whether the
    sign survives a change of period, and a random split would hand each fold a piece of every
    regime and hide exactly the instability worth finding.
    """
    rows = x.dropna().index.intersection(fwd.dropna().index)
    if len(rows) < folds * 100:
        return {"rank_ic": np.nan, "fold_sd": np.nan, "folds_positive": np.nan, "rows": float(len(rows))}
    when = rows.get_level_values(0)
    edges = pd.qcut(when.astype("int64"), folds, labels=False, duplicates="drop")
    # `legcheck._ic` reads its `index` argument only to choose between a cross-sectional Rank IC
    # and a through-time Spearman -- it does *not* restrict the computation to it. The slice has
    # to be taken here, or every fold silently returns the whole-sample number and the dispersion
    # comes back as exactly zero.
    per = [legcheck._ic(x.loc[sub], fwd.loc[sub], sub) for f in range(folds) if len(sub := rows[edges == f])]
    per = [v for v in per if not np.isnan(v)]
    return {
        "rank_ic": legcheck._ic(x, fwd, rows),
        "fold_sd": float(np.std(per)) if per else np.nan,
        "folds_positive": float(sum(v > 0 for v in per)),
        "rows": float(len(rows)),
    }


# The four columns whose sign held on all four folds at every horizon measured, with the sign that
# makes each one predict the forward return *positively*. Equal weights and no fit, which is the
# project's own law: seven extra factors fitted by ridge on train all lost to a two-column sum, and
# with ~150 independent blocks in three years two parameters is already the ceiling.
SIGNED = {"stretch": -1, "divergence": -1, "streak": -1, "rejection": +1}


def composite(frame: pd.DataFrame, columns: dict[str, int] = SIGNED) -> pd.Series:
    """Equal-weight sum of the signed columns, each ranked inside its own cross-section first.

    Ranked per date because the columns are in incomparable units -- `stretch` is in standard
    deviations, `rejection` is a fraction of a bar's range -- and summing raw values would weight
    them by their variance instead of equally.
    """
    parts = [sign * frame[c].groupby(level=0).rank(pct=True) for c, sign in columns.items()]
    return sum(parts) / len(parts)


def table(frame: pd.DataFrame, horizon: int, neutral: bool = True, lag: int = 0) -> pd.DataFrame:
    """One row per exhaustion column plus the composite, for a single forward horizon.

    `lag` delays every feature by that many bars of its own symbol before measuring. It is the
    check that separates a lead from a bounce: a signal that is really microstructure -- the close
    printing on the other side of the spread, which the next bar undoes -- loses most of its Rank
    IC at one bar of delay, while a signal about where price is going survives it.
    """
    fwd = legcheck.forward(frame.close, horizon, neutral)
    cols = {c: frame[c] for c in legs.EXHAUSTION}
    cols["composite"] = composite(frame)
    if lag:
        cols = {k: v.groupby(level=1).shift(lag) for k, v in cols.items()}
    return pd.DataFrame({c: ic_by_fold(v, fwd) for c, v in cols.items()}).T.rename_axis("column")


def _selfcheck() -> None:
    """A column built to lead the price has to show it here, and a column of noise must not."""
    n, sym = 600, ["a", "b"]
    when = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    frames = []
    for s in sym:
        step = rng.normal(0, 0.01, n)
        close = pd.Series(np.exp(np.cumsum(step)), index=when)
        idx = pd.MultiIndex.from_arrays([when, [s] * n])
        frames.append(
            pd.DataFrame(
                {
                    # Tomorrow's return, known today: the strongest possible lead, and it must
                    # come back with a large positive Rank IC or the plumbing is wrong.
                    "oracle": np.r_[step[1:], 0.0],
                    "noise": rng.normal(size=n),
                    "close": close.to_numpy(),
                },
                index=idx,
            )
        )
    frame = pd.concat(frames).sort_index()

    fwd = legcheck.forward(frame.close, 1, neutral=False)
    lead = ic_by_fold(frame.oracle, fwd)
    assert lead["rank_ic"] > 0.8, lead
    assert lead["folds_positive"] == FOLDS, lead
    flat = ic_by_fold(frame.noise, fwd)
    assert abs(flat["rank_ic"]) < 0.15, flat
    # The dispersion has to be a real number and not the zero that comes back when every fold is
    # silently handed the whole sample -- the failure this module hit on its first real run.
    assert lead["fold_sd"] > 0 and flat["fold_sd"] > 0, (lead, flat)

    # The fold split is contiguous in time and covers every row: four slices, none empty, and the
    # dates of one strictly before the dates of the next.
    rows = frame.oracle.dropna().index
    edges = pd.qcut(rows.get_level_values(0).astype("int64"), FOLDS, labels=False, duplicates="drop")
    assert set(np.unique(edges)) == set(range(FOLDS))
    last = [rows.get_level_values(0)[edges == f].max() for f in range(FOLDS)]
    assert last == sorted(last), last

    # A column that is constant has no ranking and must come back NaN rather than a number: a
    # degenerate correlation is not a zero signal, it is an absent one.
    frame["flat"] = 1.0
    assert np.isnan(ic_by_fold(frame.flat, fwd)["rank_ic"])

    # `panel` keeps each symbol's rows in order and unmixed -- the reason it builds per symbol.
    assert frame.index.is_monotonic_increasing

    # The composite is equal-weighted on per-date ranks, so a column in wild units cannot dominate
    # it: blowing one column up by a thousand must leave the composite's ordering untouched.
    for c in legs.EXHAUSTION:
        frame[c] = rng.normal(size=len(frame))
    before = composite(frame)
    frame["stretch"] = frame["stretch"] * 1000
    assert np.allclose(before.to_numpy(), composite(frame).to_numpy(), equal_nan=True)
    # And the signs enter as written: flipping the sign of a +1 column flips the composite exactly.
    flipped = composite(frame, {**SIGNED, "rejection": -1})
    assert not np.allclose(before.to_numpy(), flipped.to_numpy(), equal_nan=True)

    # `lag` shifts inside each symbol and never across the boundary between two of them: the first
    # row of every symbol has to go NaN, which is what reading a neighbour's last bar would hide.
    lagged = table(frame, 1, neutral=False, lag=1)
    assert lagged.loc["stretch", "rows"] < table(frame, 1, neutral=False).loc["stretch", "rows"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS), help="default is all twenty")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--since", default="2025-06", help="the test slice the model was scored on")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--raw", action="store_true", help="skip the market-neutral step")
    ap.add_argument("--lag", type=int, default=0, help="delay the features by N bars: the bounce-vs-lead check")
    args = ap.parse_args()

    _selfcheck()
    frame = panel(args.symbols, args.timeframe, args.since)
    when = frame.index.get_level_values(0)
    breadth = frame.groupby(level=0).size().mean()
    print(f"{len(frame):,} rows, {frame.index.get_level_values(1).nunique()} symbol(s), breadth {breadth:.2f}")
    print(f"{when.min():%Y-%m-%d} to {when.max():%Y-%m-%d}, {args.timeframe} bars")
    print("returns are raw\n" if args.raw else "returns are market-neutral\n")
    print("the bar to clear: the trained model gives -0.0405 against the forward return at 6 bars")
    print(f"features delayed by {args.lag} bar(s)\n" if args.lag else "features read at the bar they close\n")

    for h in args.horizons:
        print(f"--- {h} bars ahead ---\n")
        print(table(frame, h, not args.raw, args.lag).round(4).to_string())
        print()


if __name__ == "__main__":
    main()
