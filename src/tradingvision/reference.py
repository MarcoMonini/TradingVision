"""Choose `riferimento[k]`: the scale that turns a leg amplitude into a number that means the same
thing on every symbol and in every volatility regime.

`amplitude[k]` (log return of the leg closed at pivot k) is already dimensionless, but it is not
comparable: a 2% leg is an ordinary swing on DOGE and a large one on BTC, and a 2% leg in 2018 is
not a 2% leg in 2025. Anything built on a fixed amplitude threshold therefore hits each symbol
differently — measured on 15m bars, a 1% cut removes 13% of BTC bars and 2% of SOL bars.

    python -m tradingvision.reference --html reference.html

The criterion is invariance, not fit: pick the denominator for which the *same* threshold on
`amplitude / reference` selects the *same* share of legs on every symbol and in every year, while
still separating weak legs from strong ones inside a symbol. A denominator that flattens
everything scores perfectly on invariance and is useless, so both halves are reported.

Every candidate is causal at the pivot: it reads bars up to k and never past it.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision.data.binance import SYMBOLS, load
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import bar_sigma

# Lookback of the volatility estimators, in bars of the reference timeframe. 96 is four times
# `EXTREMA_WINDOW`: a day of 15m bars, long enough to average out a single leg and short enough to
# track a regime. The 24-bar variant is kept in the comparison to show the sensitivity — at the
# same length as the search window the denominator partly measures the very leg it normalises.
LOOKBACK = 96
# Legs used by the self-referential candidate. 20 pivots is roughly a fortnight of 15m swings.
LEG_MEMORY = 20


def atr_pct(df: pd.DataFrame, n: int = LOOKBACK) -> pd.Series:
    """Average true range over `n` bars, as a fraction of price. Wilder's TR, causal."""
    prev = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / df.close


def leg_ratios(
    df: pd.DataFrame,
    pivots: pd.DataFrame | None = None,
    window: int = EXTREMA_WINDOW,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """One row per pivot: the leg it closes, and `amplitude / reference` for every candidate.

    Candidates:
      `atr`      amplitude in units of the average bar range — the spec's example.
      `atr_24`   the same at the search window's own length, to expose the sensitivity to it.
      `sigma_t`  amplitude against sigma * sqrt(duration), the move a random walk of this
                 volatility would make in the time the leg actually took. The only candidate that
                 does not confuse a big leg with a slow one.
      `legs`     amplitude against the median of the previous legs of this symbol — "large for
                 this market lately", with no volatility model at all.
      `raw`      no denominator: the control, i.e. what a fixed percentage threshold does today.
    """
    piv = find_pivots(df.close, window) if pivots is None else pivots
    at = df.index.get_indexer(piv.index)
    if (at < 0).any():
        raise ValueError("pivots do not belong to this frame")

    # Bars of the leg closed at k, matching the backward convention of `amplitude`.
    duration = pd.Series(at, index=piv.index).diff()
    out = pd.DataFrame({"amplitude": piv.amplitude, "duration": duration, "kind": piv.kind})
    out["overshoot"] = _overshoot(df.close.to_numpy(), at, piv.index)
    out["raw"] = piv.amplitude

    for name, n in (("atr", lookback), ("atr_24", window)):
        out[name] = piv.amplitude / atr_pct(df, n).to_numpy()[at]
    sigma = bar_sigma(df.close, lookback).to_numpy()[at]
    out["sigma_t"] = piv.amplitude / (sigma * np.sqrt(duration))
    # Median of the *previous* legs only: including the current one pulls every ratio towards 1.
    out["legs"] = piv.amplitude / piv.amplitude.rolling(LEG_MEMORY).median().shift()
    return out


def _overshoot(c: np.ndarray, at: np.ndarray, index: pd.Index) -> pd.Series:
    """How far outside [0, 1] the price advance runs inside each leg, charged to its closing pivot.

    Independent evidence that a label is broken, and it owes nothing to any candidate denominator:
    the pivots are local extrema of a 24-bar window, not extremes of the leg, so a bar in between
    can trade past either end. When it does so by a lot, the leg is a flat stretch of noise whose
    two pivots happen to sit at nearly the same price, and the label ramps from +1 to -1 across it
    for no reason. 0 on a well-behaved leg.
    """
    i = np.arange(at[0], at[-1] + 1)
    k = np.clip(np.searchsorted(at, i, side="right") - 1, 0, len(at) - 2)
    start, end = at[k], at[k + 1]
    excursion = c[end] - c[start]
    advance = np.divide(c[i] - c[start], excursion, out=np.full(len(i), 0.5), where=excursion != 0)
    g = pd.DataFrame({"k": k, "a": advance}).groupby("k").a.agg(["min", "max"])
    worst = np.maximum(-g["min"], g["max"] - 1).clip(lower=0)
    # Charged to the pivot that closes the leg, like `amplitude`.
    return pd.Series(worst.to_numpy(), index=index[g.index.to_numpy() + 1]).reindex(index)


CANDIDATES = ["raw", "atr", "atr_24", "sigma_t", "legs"]
# A leg whose price advance leaves [0, 1] by more than this is treated as broken: at 0.5 the price
# runs half a leg-length past a pivot without that bar being a pivot itself.
BROKEN = 0.5


def collect(symbols: list[str], timeframe: str = "15m", window: int = EXTREMA_WINDOW, start: str | None = None):
    """Every pivot of every symbol, with its candidate ratios and its year."""
    frames = []
    for i, symbol in enumerate(symbols, 1):
        df = load(symbol, timeframe)
        if start:
            df = df.loc[start:]
        r = leg_ratios(df, window=window)
        frames.append(r.assign(symbol=symbol, year=r.index.year))
        print(f"[{i}/{len(symbols)}] {symbol} — {len(df):,} bars, {len(r):,} pivots", flush=True)
    return pd.concat(frames).dropna(subset=CANDIDATES)


def invariance(data: pd.DataFrame, by: str, share: float = 0.05) -> pd.DataFrame:
    """How unevenly a pooled threshold lands across groups.

    For each candidate the threshold is set on the pooled distribution so that it selects `share`
    of all legs; a perfect denominator then selects that same share inside every symbol and every
    year. What is reported is the spread of the per-group share around it — plus the dispersion
    that survives *inside* a group, which is the signal the threshold is supposed to act on.
    """
    rows = []
    for c in CANDIDATES:
        cut = data[c].quantile(share)
        hit = data.groupby(by)[c].apply(lambda s, cut=cut: (s < cut).mean() * 100)
        spread = np.log(data.groupby(by)[c].median())
        within = data.groupby(by)[c].apply(lambda s: np.log(s).quantile(0.75) - np.log(s).quantile(0.25))
        rows.append(
            {
                "candidate": c,
                "selected % — min": hit.min(),
                "median": hit.median(),
                "max": hit.max(),
                "spread of medians (log)": spread.std(),
                "within-group IQR (log)": within.mean(),
                "separability": within.mean() / spread.std(),
            }
        )
    return pd.DataFrame(rows).set_index("candidate").round(2)


def discrimination(data: pd.DataFrame, share: float = 0.05, broken: float = BROKEN) -> pd.DataFrame:
    """Does the ratio actually rank the broken legs at the bottom?

    Invariance alone is satisfied by any denominator that spreads the ratios evenly; it says
    nothing about whether the low end is where the unusable labels are. `recall` is the share of
    broken legs caught by the pooled threshold, `lift` how much denser they are below it than in
    the panel at large, and `percentile` where the median broken leg sits in the ratio ranking —
    the lower the better, 50 means the ratio ignores them.
    """
    bad = data.overshoot > broken
    rows = []
    for c in CANDIDATES:
        cut = data[c].quantile(share)
        caught = bad & (data[c] < cut)
        rows.append(
            {
                "candidate": c,
                "recall %": caught.sum() / bad.sum() * 100,
                "lift": (caught.sum() / (data[c] < cut).sum()) / bad.mean(),
                "percentile of median broken leg": data[c].rank(pct=True)[bad].median() * 100,
            }
        )
    return pd.DataFrame(rows).set_index("candidate").round(2)


def figure(data: pd.DataFrame):
    """Distribution of each candidate across symbols and across years, on a log scale.

    The good denominator is the one whose boxes line up: same centre, same width, everywhere. Read
    the two columns together — a candidate can align the symbols by crushing the spread, and the
    year column tends to expose that. Boxes are precomputed quantiles rather than raw points: at
    10^5 legs a plain box plot ships every observation into the page and the file reaches tens of
    megabytes.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=len(CANDIDATES),
        cols=2,
        shared_yaxes=True,
        subplot_titles=[f"{c} — {axis}" for c in CANDIDATES for axis in ("by symbol", "by year")],
        vertical_spacing=0.04,
    )
    for row, c in enumerate(CANDIDATES, 1):
        y = np.log10(data[c])
        for col, key in ((1, "symbol"), (2, "year")):
            q = y.groupby(data[key]).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
            fig.add_trace(
                go.Box(
                    x=q.index.astype(str),
                    lowerfence=q[0.05],
                    q1=q[0.25],
                    median=q[0.5],
                    q3=q[0.75],
                    upperfence=q[0.95],
                    marker_color="#3498db",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(title_text="log10 ratio", row=row, col=1)
    # One shared band on every panel: a candidate is invariant when its boxes stay inside it.
    fig.update_yaxes(range=[float(np.log10(data[CANDIDATES].quantile(0.02)).min()), 1.5])
    fig.update_layout(height=300 * len(CANDIDATES), title="riferimento[k] — invariance of amplitude / reference")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--window", type=int, default=EXTREMA_WINDOW)
    ap.add_argument("--start", help="ISO date, to restrict the panel to an aligned period")
    ap.add_argument("--share", type=float, default=0.05, help="fraction of legs the threshold should select")
    ap.add_argument("--html", help="write the comparison figure here")
    ap.add_argument("--csv", help="write the per-pivot ratios here")
    args = ap.parse_args()

    data = collect(args.symbols, args.timeframe, args.window, args.start)
    print(f"\n{len(data):,} legs over {data.symbol.nunique()} symbols, {data.year.min()}-{data.year.max()}")
    for by in ("symbol", "year"):
        print(f"\nthreshold selecting {args.share:.0%} of legs overall — where it actually lands, by {by}:")
        print(invariance(data, by, args.share).to_string())
    bad = data.overshoot > BROKEN
    print(f"\nbroken legs (price advance beyond [0,1] by more than {BROKEN}): {bad.sum():,} ({bad.mean() * 100:.1f}%)")
    print(discrimination(data, args.share).to_string())
    if args.csv:
        data.to_csv(args.csv)
    if args.html:
        figure(data).write_html(args.html, include_plotlyjs="cdn")
        print(f"\nfigure — {args.html}")


if __name__ == "__main__":
    main()
