"""What the strategy pays as a function of the skill the model reaches, and of the fee it pays.

`threshold` prices a prediction that exists. This prices one that does not yet: it synthesises a
signal of *known* skill and runs it through the same rule on the same prices, which answers the
question a measurement on a trained model cannot — how good does the model have to get before any
of this is worth running.

    pred = rho * y + sqrt(1 - rho^2) * noise

The mix is the whole trick. At rho = 0 the prediction is noise and the rule pays fees for nothing;
at rho = 1 it is the label itself and the run reports the ceiling. In between, the realised Rank IC
is measured rather than assumed, so the table is indexed by the number the spec actually reports.

Two decisions in here are not cosmetic.

The noise is persistent, not white. A model fitted on a 12-hour label reads features that move on
that scale and its output moves with them; white hourly noise would invent a turnover no model
produces and charge the strategy for it. The EMA span is a quarter of the label horizon in bars,
which is the horizon in hours on the 15m grid.

The P&L is reported hedged and naked, and the hedged one is the honest half. The label predicts a
return *in excess of the basket*, so that is what a position on it earns; a naked position also
collects the market, which over the twelve months from 2025-09 was worth -0.90 a year and would
have flattered any signal that leaned short. `threshold` already splits by side for this reason.

    uv run python -m tradingvision.simulation --fees 25 10 5
    uv run python -m tradingvision.simulation --pred data/pred-cross-...parquet --smooth 12

The threshold is in units of the cross-section of the day. `book` turns every prediction into its
centred percentile inside its own timestamp before anything is thresholded, so 0.5 means "the top
and bottom quarter of whatever is trading now" on every pair and in every regime. The obvious
alternative — standardise each symbol over the period — reads the period it is trading, which is
the leakage the folds are purged to avoid.

What it is not. The skill is constant through time, where a real model's Rank IC swings and goes
negative in stretches; there is no slippage and no impact; and the hedged leg assumes the basket
is actually shorted, which on perpetuals costs funding. The levels are optimistic by construction
-- halve them, or third them. What the run measures well is the *shape*: the gross scales with the
skill and the cost does not, so where the net crosses zero is a ratio of two things it gets right.
That crossing is the output worth reading.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import metrics, threshold
from tradingvision.data.binance import SYMBOLS, load
from tradingvision.data.target import CROSS_HORIZON, cross_sectional_return
from tradingvision.oracle import FEE

STEP = 4  # 15m bars per decision: hourly, the stride the dataset samples at
RHOS = (0.02, 0.05, 0.10, 0.20, 0.40)
# In units of the cross-section of the day, not of a symbol's own history: `run` turns every
# prediction into its centred percentile inside its timestamp before thresholding, so 0.5 means
# "the top and bottom quarter of whatever is trading now" on every pair and in every regime.
THETAS = (0.2, 0.5, 0.8)
FEES = (0.0025, 0.0010, 0.0005)  # Alpaca taker, and two venues an order of magnitude cheaper
YEAR = pd.Timedelta("365D")


def panel(symbols: list[str], timeframe: str = "15m", since: str = "2024") -> pd.DataFrame:
    """Close prices, one column per symbol, on a shared index."""
    return pd.DataFrame({s: load(s, timeframe).loc[since:].close for s in symbols}).sort_index()


def skilled(y: pd.DataFrame, rho: float, rng: np.random.Generator, span: int) -> pd.Series:
    """A prediction correlated `rho` with the label, standardised per symbol.

    Standardised per symbol and not pooled, because the threshold the rule reads is one number for
    every symbol: without it a pair with a wider prediction would simply be traded more often, and
    the sweep would be measuring that instead of the skill.
    """
    white = pd.DataFrame(rng.normal(size=y.shape), index=y.index, columns=y.columns)
    noise = white.ewm(span=span).mean()
    noise = (noise - noise.mean()) / noise.std()
    pred = (rho * (y - y.mean()) / y.std() + np.sqrt(1 - rho**2) * noise).stack()
    return pred.groupby(level=1).transform(lambda v: (v - v.mean()) / v.std()).sort_index()


def book(pred: pd.Series, theta: float) -> pd.Series:
    """A prediction turned into the signal the rule reads: its centred percentile inside its own
    timestamp, in [-1, +1].

    A threshold has to mean the same thing on every symbol and in every regime, and a raw
    prediction does not: its scale drifts with the model, with volatility, and between pairs. The
    obvious fix — standardise each symbol over the period — reads the period it is trading, which
    is the leakage this project purges folds to avoid. The percentile inside a timestamp reads only
    the cross-section standing there at that instant, which is exactly what a rank-based label
    predicts and what a dollar-neutral book acts on.
    """
    pct = pred.groupby(level=0).rank(pct=True)
    # Centred on the mean of its own timestamp for the reason `normalize.cross_rank` gives: pct
    # ranks run 1/n..1, so a flat shift leaves an offset that moves with how many symbols happen
    # to be trading, and a symmetric threshold would then be asymmetric whenever one gaps out.
    return 2 * (pct - pct.groupby(level=0).transform("mean"))


def run(pred: pd.Series, forward: pd.DataFrame, theta: float, fee: float, span: float) -> dict:
    """One (threshold, fee) pair, priced on both accountings.

    `forward` is the return of each symbol over one decision step. Hedged subtracts the mean of the
    row, which is the basket leg of the trade the label describes.
    """
    pos = threshold.positions(book(pred, theta), theta, sign=1)
    pos = pos.unstack(level=1).reindex(columns=forward.columns).fillna(0.0)
    # Per symbol and per year, in units of position: a flip from long to short is 2. `fillna(pos)`
    # charges the opening trade, as `threshold.pnl` does — the first row has no diff, not no cost.
    turnover = pos.diff().fillna(pos).abs().to_numpy().sum() / pos.shape[1] / span
    out = {
        "theta": theta,
        "fee_bp": round(fee * 1e4),
        "in_market": float((pos != 0).to_numpy().mean()),
        "trades_per_year": turnover / 2,
        "fees_per_year": turnover * fee,
    }
    for name, ret in (("hedged", forward.sub(forward.mean(axis=1), axis=0)), ("naked", forward)):
        # Equal weight across the symbols, so the book is one unit of notional and the fee, charged
        # on the same 1/N weights, stays comparable to the return.
        per_step = (pos * ret.reindex_like(pos)).fillna(0.0).mean(axis=1)
        gross = per_step.sum() / span
        net = gross - turnover * fee
        out[f"gross_{name}"] = gross
        out[f"net_{name}"] = net
        # Annualised on the decision step, with the fee spread evenly over it. An upper bound and
        # nothing more: constant skill and frictionless fills both flatter it. A threshold no
        # prediction ever reaches holds nothing and has no ratio, rather than a zero divided by one.
        risk = per_step.std() * np.sqrt(len(per_step) / span)
        out[f"sharpe_{name}"] = net / span / risk if risk > 0 else np.nan
    return out


def sweep(
    close: pd.DataFrame,
    horizon: int = CROSS_HORIZON,
    step: int = STEP,
    rhos=RHOS,
    thetas=THETAS,
    fees=FEES,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per (skill, threshold, fee), with the realised Rank IC of each synthetic signal."""
    y = cross_sectional_return(close, horizon)
    forward = np.log(close.shift(-step) / close)
    at = np.arange(len(close)) % step == 0  # decide once per step, not once per bar
    y, forward, close = y[at], forward[at], close[at]
    span = (close.index[-1] - close.index[0]) / YEAR
    rng = np.random.default_rng(seed)
    rows = []
    for rho in rhos:
        pred = skilled(y, rho, rng, span=max(2, horizon // step))
        ic = metrics.signal(pred, y.stack().reindex(pred.index))["rank_ic"]
        for theta in thetas:
            for fee in fees:
                rows.append(dict(rank_ic=ic, **run(pred, forward, theta, fee, span)))
    return pd.DataFrame(rows)


def price(pred: pd.Series, thetas=THETAS, fees=FEES, smooth: int = 1) -> pd.DataFrame:
    """A prediction a model actually made, priced with the accounting `sweep` reports.

    `sweep` asks what a model of a given skill would earn. This asks what *this* model earned, so
    the answer lands on the same table and the break-even it has to clear is the one already
    measured rather than a second standard invented for the occasion.

    Long form throughout, and never the wide frame `run` uses. `sweep` builds its own panel by
    resampling one grid, so every symbol has a row at every timestamp; a real prediction does not.
    Measured on the first `--label cross --rank` run: 218,497 rows over 27,812 distinct timestamps
    and twenty symbols, so a timestamp carries eight of them on average. Unstacking that is 60% NaN,
    and filling those with zero reads as "flat" — every symbol dropped and re-entered on every
    timestamp its neighbours have and it does not, which invented 8,400 round trips a year per
    symbol out of nothing. `threshold.pnl` works in long form for this exact reason and says so.

    `smooth` low-passes the prediction over `k` of each symbol's own rows before anything is
    traded. A signal fitted on a 72-hour label should move on a 72-hour scale, and a raw GRU output
    moves every row; the Rank IC barely notices the filter (0.1064 raw against 0.1074 at k=12) so
    what it removes is estimation noise.

    Read `net_hedged`, not `net_naked`. The label is a return in excess of the basket, so the
    hedged column is what the signal predicts; the naked one also collects the market, which over a
    falling test period pays a signal for leaning short and says nothing about what it knows.
    `threshold.sweep` prices the naked single-symbol rule and is the wrong instrument here.
    """
    p = threshold.smoothed(pred.sort_index(), smooth)
    close = threshold.prices(p.index)
    r = threshold.forward_return(close).fillna(0.0)
    # The basket leg, taken over whatever is actually trading at that timestamp rather than over a
    # fixed roster — same rule the label was built under.
    hedged = r - r.groupby(level=0).transform("mean")
    span = (p.index.get_level_values(0).max() - p.index.get_level_values(0).min()) / YEAR
    symbol = p.index.get_level_values(1)

    rows = []
    for theta in thetas:
        pos = threshold.positions(book(p, theta), theta, sign=1)
        turnover = pos.groupby(symbol).diff().fillna(pos).abs().groupby(symbol).sum().mean() / span
        out = {
            "theta": theta,
            "in_market": float((pos != 0).mean()),
            "long_share": float((pos > 0).mean()),
            "trades_per_year": turnover / 2,
        }
        for name, ret in (("hedged", hedged), ("naked", r)):
            # Per timestamp and then summed, so the book is one unit of notional spread over
            # whatever it holds, and a thin cross-section is not silently a bigger bet.
            per_step = (pos * ret).groupby(level=0).mean()
            out[f"gross_{name}"] = per_step.sum() / span
            out[f"risk_{name}"] = per_step.std() * np.sqrt(len(per_step) / span)
        for fee in fees:
            cost = turnover * fee
            rows.append(
                dict(
                    out,
                    fee_bp=round(fee * 1e4),
                    fees_per_year=cost,
                    **{f"net_{n}": out[f"gross_{n}"] - cost for n in ("hedged", "naked")},
                    **{
                        f"sharpe_{n}": (out[f"gross_{n}"] - cost) / out[f"risk_{n}"] if out[f"risk_{n}"] else np.nan
                        for n in ("hedged", "naked")
                    },
                )
            )
    return pd.DataFrame(rows)


def breakeven(table: pd.DataFrame, column: str = "net_hedged") -> pd.DataFrame:
    """The Rank IC at which each (threshold, fee) turns a profit, interpolated between the rows run.

    The one number the whole module exists to produce: the skill the model has to reach before the
    rule pays for itself. NaN when every skill in the sweep is already on the same side of zero.
    """
    out = {}
    for (theta, fee), g in table.groupby(["theta", "fee_bp"]):
        g = g.sort_values("rank_ic")
        below = g[g[column] <= 0].tail(1)
        above = g[g[column] > 0].head(1)
        if below.empty or above.empty or below.rank_ic.iloc[0] > above.rank_ic.iloc[0]:
            out[(theta, fee)] = 0.0 if below.empty else np.nan
            continue
        x0, y0 = below.rank_ic.iloc[0], below[column].iloc[0]
        x1, y1 = above.rank_ic.iloc[0], above[column].iloc[0]
        out[(theta, fee)] = x0 + (x1 - x0) * (-y0) / (y1 - y0)
    return pd.Series(out).unstack().rename_axis(index="theta", columns="fee_bp")


def _selfcheck() -> None:
    """Three symbols whose relative moves are a known signal, so the sweep has a real ceiling.

    A shared random walk carries the market, and a slow sine moves each symbol against the others.
    The label is that sine, so a perfect prediction has to make money and noise has to lose it.
    """
    n, rng = 4000, np.random.default_rng(0)
    when = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    market = np.cumsum(rng.normal(0, 0.004, n))
    phase = np.arange(n) * 2 * np.pi / 400  # a full rotation every 100 hours
    close = pd.DataFrame(
        {f"S{k}": np.exp(market + 0.02 * np.sin(phase + k * 2 * np.pi / 3)) for k in range(3)},
        index=when,
    )
    table = sweep(close, horizon=16, rhos=(0.0, 1.0), thetas=(0.5,), fees=(0.0, FEE))
    perfect = table[(table.rank_ic > 0.5) & (table.fee_bp == 0)].iloc[0]
    blind = table[(table.rank_ic < 0.2) & (table.fee_bp == 0)].iloc[0]
    assert perfect.rank_ic > 0.9, "rho = 1 has to reproduce the label"
    assert perfect.gross_hedged > 0.5, perfect.to_dict()
    assert perfect.gross_hedged > 10 * abs(blind.gross_hedged), "skill has to beat noise"

    # The fee only ever subtracts, and it subtracts what the turnover says it does.
    free, paid = (table[table.fee_bp == f].iloc[-1] for f in (0, round(FEE * 1e4)))
    assert paid.net_hedged < free.net_hedged
    assert np.isclose(free.net_hedged - paid.net_hedged, paid.trades_per_year * 2 * FEE)

    # The signal the rule reads is a percentile inside its timestamp: bounded, centred, and
    # unchanged by any monotone map applied to a whole cross-section — so the scale the model
    # happens to predict on cannot move a single position.
    idx = pd.MultiIndex.from_product([pd.date_range("2025", periods=4, freq="h", tz="UTC"), list("abc")])
    raw = pd.Series([1.0, 5.0, 9.0] * 4, index=idx)
    b = book(raw, 0.5)
    assert b.between(-1, 1).all() and np.allclose(b.groupby(level=0).mean(), 0, atol=1e-12)
    assert np.allclose(book(np.exp(raw * 3) * 100, 0.5), b), "a monotone map per date changes nothing"

    # A wider band trades less. This is the only lever the rule itself has against the fee.
    wide = sweep(close, horizon=16, rhos=(1.0,), thetas=(0.2, 0.9), fees=(FEE,))
    assert wide.trades_per_year.is_monotonic_decreasing, wide.to_string()

    # Market neutrality, carried through to the P&L: a trend common to every symbol leaves the
    # hedged return untouched and moves only the naked one. A common *drift* and not a common
    # random walk, because the label is exactly neutral in its numerator and only nearly so in its
    # denominator — `sigma_i` is the symbol's whole volatility, market component included, so a
    # shock that adds variance rescales the label by a fraction of a percent without touching its
    # ordering. The exact statement is the one worth asserting.
    shocked = close.mul(np.exp(0.0005 * np.arange(n)), axis=0)
    a = sweep(close, horizon=16, rhos=(1.0,), thetas=(0.5,), fees=(0.0,)).iloc[0]
    b = sweep(shocked, horizon=16, rhos=(1.0,), thetas=(0.5,), fees=(0.0,)).iloc[0]
    assert np.isclose(a.gross_hedged, b.gross_hedged, rtol=1e-6), (a.gross_hedged, b.gross_hedged)
    assert not np.isclose(a.gross_naked, b.gross_naked, rtol=1e-3), "the naked leg does move"

    # And the output the module is for: a break-even sits between the skills that bracket it.
    grid = sweep(close, horizon=16, rhos=(0.0, 0.05, 0.2, 1.0), thetas=(0.5,), fees=(FEE,))
    edge = breakeven(grid).iloc[0, 0]
    assert 0 < edge < 1, grid.to_string()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, help="price a real prediction a gru run wrote, instead of sweeping skill")
    ap.add_argument("--smooth", type=int, default=1, help="low-pass the prediction over k of each symbol's own rows")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS[:10]))
    ap.add_argument("--since", default="2024")
    ap.add_argument("--horizon", type=int, default=CROSS_HORIZON, help="label horizon, in 15m bars")
    ap.add_argument("--fees", type=float, nargs="+", default=[f * 1e4 for f in FEES], help="per side, in bp")
    ap.add_argument("--thetas", type=float, nargs="+", default=list(THETAS))
    args = ap.parse_args()

    _selfcheck()
    if args.pred:
        pred = pd.read_parquet(args.pred).iloc[:, 0]
        out = price(pred, tuple(args.thetas), tuple(f / 1e4 for f in args.fees), args.smooth)
        when = pred.index.get_level_values(0)
        print(
            f"{len(pred):,} predictions, {pred.index.get_level_values(1).nunique()} symbols, "
            f"{when.min():%Y-%m-%d} to {when.max():%Y-%m-%d}, smoothing k={args.smooth}\n"
        )
        cols = [
            "theta",
            "fee_bp",
            "in_market",
            "trades_per_year",
            "gross_hedged",
            "fees_per_year",
            "net_hedged",
            "sharpe_hedged",
            "net_naked",
        ]
        print(out[cols].round(4).to_string(index=False))
        return
    close = panel(args.symbols, since=args.since)
    span = (close.index[-1] - close.index[0]) / YEAR
    print(
        f"{len(close):,} bars x {len(close.columns)} symbols, {close.index[0]:%Y-%m} to "
        f"{close.index[-1]:%Y-%m}, span {span:.2f}y, label horizon {args.horizon} bars "
        f"({args.horizon * 15 / 60:.0f}h), decisions every {STEP * 15 / 60:.0f}h"
    )
    table = sweep(close, args.horizon, thetas=tuple(args.thetas), fees=tuple(f / 1e4 for f in args.fees))
    for fee in sorted({int(f) for f in args.fees}, reverse=True):
        cut = table[table.fee_bp == fee]
        print(f"\nnet hedged log return per year, {fee} bp per side — by Rank IC and threshold\n")
        print((cut.pivot_table(index="rank_ic", columns="theta", values="net_hedged") * 100).round(1).to_string())
    print("\nRank IC at which the rule breaks even\n")
    print(breakeven(table).round(3).to_string())
    print("\nanatomy at the middle threshold\n")
    mid = table[table.theta == sorted(args.thetas)[len(args.thetas) // 2]]
    cols = ["rank_ic", "fee_bp", "trades_per_year", "gross_hedged", "fees_per_year", "net_hedged", "sharpe_hedged"]
    print(mid[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
