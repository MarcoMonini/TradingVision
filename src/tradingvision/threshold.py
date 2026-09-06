"""What the signal is worth as a rule, in money rather than in correlation.

Rank IC answers "does the ordering carry information". It does not answer "does trading on it
pay", and the two come apart in exactly the way that matters here: a signal can rank the twenty
symbols correctly every hour and still lose to fees, and a signal with a mediocre IC can pay if
its errors land where nothing is held. This module asks the second question and nothing else.

The rule is the simplest one the label describes, and deliberately not a tuned strategy. The
label says where the bar sits on its leg — `-1` at a low, `+1` at a high — so:

    pred <= -t     go long, the leg should run up from here
    pred >= +t     go short
    in between     hold whatever is already held

Hysteresis and not a flat band: the swing the label describes *is* the hold from one extreme to
the other, and a rule that flattens whenever the signal is unremarkable would pay the round trip
several times inside a single leg. Costs are `oracle.FEE` per side, charged on every unit of
position changed, so a flip from long to short pays twice.

Two honesties about the measurement. The rows are step 2's, one per hour per symbol, so this
trades hourly and never inside the hour. And each symbol is one unit, equally weighted, with no
sizing and no risk limit — the question is whether the edge clears the fee, not what a portfolio
would do with it.

    uv run python -m tradingvision.threshold --pred data/pred-swing-all.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision.data.binance import load
from tradingvision.oracle import FEE

# Quantiles of |prediction| the default grid is taken at: a threshold is only readable next to how
# much of the time it leaves a position open, and the scale of the label changes between targets —
# `swing` lives in [-1, 1], `excursion` in sigma of a 24-bar walk.
QUANTILES = (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)
YEAR = pd.Timedelta("365D")


def prices(index: pd.MultiIndex) -> pd.Series:
    """The 5m close at each (timestamp, symbol) row."""
    out = pd.Series(np.nan, index=index, name="close")
    for symbol, rows in pd.Series(np.arange(len(index)), index=index).groupby(level=1):
        close = load(symbol, "5m").close
        out.iloc[rows.to_numpy()] = close.reindex(rows.index.get_level_values(0)).to_numpy()
    if out.isna().any():
        raise ValueError(f"{out.isna().sum()} rows have no 5m close")
    return out


def positions(pred: pd.Series, threshold: float, sign: int = -1) -> pd.Series:
    """The held position at every row: +1 long, -1 short, 0 before the first signal.

    `sign` is the direction the label points. It is -1 for `swing_leg_target`, where a low
    prediction means the bar sits near a low and the leg runs up from there, and +1 for
    `remaining_excursion`, which is already signed the way the trade is.
    """
    signal = pd.Series(np.nan, index=pred.index)
    signal[pred <= -threshold] = -sign
    signal[pred >= threshold] = sign
    # Forward filled inside each symbol, which is what makes it a hold and not a flicker. The rows
    # arrive sorted by timestamp, so a symbol's rows are already in its own chronological order.
    return signal.groupby(level=1).ffill().fillna(0.0)


def forward_return(close: pd.Series) -> pd.Series:
    """Log return from each row to that symbol's next row — the return a position earns by being
    held there. NaN on the last row of each symbol, which no position can be paid for."""
    return np.log(close.groupby(level=1).shift(-1) / close)


def pnl(pred: pd.Series, close: pd.Series, threshold: float, fee: float = FEE, sign: int = -1) -> dict:
    """One threshold, priced. Log returns throughout, with the fee charged as a log cost too —
    at 0.25% the difference to the exact multiplicative form is in the fifth decimal."""
    span = (pred.index.get_level_values(0).max() - pred.index.get_level_values(0).min()) / YEAR
    pos = positions(pred, threshold, sign)
    r = forward_return(close).fillna(0.0)
    symbol = pos.index.get_level_values(1)
    # Turnover, in units of position: entering costs one side, flipping costs two.
    turnover = pos.groupby(symbol).diff().fillna(pos).abs()
    gross, cost = (pos * r).groupby(symbol).sum(), (turnover * fee).groupby(symbol).sum()

    # Per trade, so the hit rate is a hit rate and not a share of profitable hours: a leg holds a
    # constant position, and the trade is that whole hold.
    f = pd.DataFrame({"pos": pos, "g": pos * r})
    f["leg"] = (f.pos != f.pos.groupby(symbol).shift()).groupby(symbol).cumsum()
    held = f[f.pos != 0]
    per_trade = held.groupby([held.index.get_level_values(1), held.leg]).g.sum() - 2 * fee

    # Split by side, which is the check that separates an edge from a market. Over a period the
    # market spends falling, a rule that is short half the time earns without predicting anything;
    # a signal that reads turning points has to make money on both sides, or say why not.
    gross_by_side = (pos * r).groupby([symbol, np.sign(pos)]).sum().groupby(level=1).mean()
    return {
        "threshold": threshold,
        "in_market": float((pos != 0).mean()),
        "long_share": float((pos > 0).mean()),
        "gross_long": float(gross_by_side.get(1.0, 0.0) / span),
        "gross_short": float(gross_by_side.get(-1.0, 0.0) / span),
        "trades_per_year": len(per_trade) / len(gross) / span,
        "gross_per_year": float(gross.mean() / span),
        "fees_per_year": float(cost.mean() / span),
        "net_per_year": float((gross.mean() - cost.mean()) / span),
        "win_rate": float((per_trade > 0).mean()) if len(per_trade) else np.nan,
        "median_trade": float(per_trade.median()) if len(per_trade) else np.nan,
    }


def sweep(pred: pd.Series, close: pd.Series, quantiles=QUANTILES, fee: float = FEE, sign: int = -1) -> pd.DataFrame:
    """One row per threshold, taken at quantiles of |prediction| so the grid means the same thing
    whatever scale the label lives on. The 0.0 quantile is the always-in rule, which is the
    control: if no threshold beats it, the signal is adding nothing but fees."""
    rows = [pnl(pred, close, float(pred.abs().quantile(q)), fee, sign) for q in quantiles]
    return pd.DataFrame(rows, index=pd.Index(quantiles, name="quantile"))


def buy_and_hold(close: pd.Series) -> dict:
    """The same accounting for holding every symbol throughout — the other control, and the one a
    long-only crypto rule has to beat to be worth running at all."""
    r = forward_return(close).fillna(0.0)
    span = (close.index.get_level_values(0).max() - close.index.get_level_values(0).min()) / YEAR
    per_symbol = r.groupby(close.index.get_level_values(1)).sum()
    return {"net_per_year": float(per_symbol.mean() / span), "trades_per_year": 1 / span}


def _selfcheck() -> None:
    """A price that is a clean saw and a prediction that reads it perfectly: the rule has to make
    the saw's amplitude minus its fees, and lose once the fee exceeds the leg."""
    n = 400
    when = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    idx = pd.MultiIndex.from_arrays([when, ["a"] * n])
    # A triangle wave of 40 bars: 20 up, 20 down, 2% peak to peak.
    phase = np.arange(n) % 40
    ramp = np.where(phase < 20, phase, 40 - phase) / 20.0
    close = pd.Series(np.exp(0.02 * ramp), index=idx)
    # The label read exactly: -1 at the lows, +1 at the peaks.
    pred = pd.Series(2 * ramp - 1, index=idx)

    out = pnl(pred, close, 0.9)
    # Every completed trade wins; the last one is still open when the series ends, so it is
    # carrying its entry fee and nothing else. 21 trades, 20 of them closed.
    assert out["win_rate"] >= 0.95, out
    assert out["median_trade"] > 0.01, out
    # Ten full up legs of 2% each over a 400-hour span, so the yearly rate is that scaled up.
    span = (when[-1] - when[0]) / YEAR
    assert np.isclose(out["gross_per_year"] * span, 10 * 0.02 * 2, atol=0.02), out
    assert out["fees_per_year"] > 0 and out["net_per_year"] < out["gross_per_year"]

    # A fee larger than the leg turns the same perfect signal into a loss. Nothing about the
    # ordering changed, which is the whole point of measuring this and not the Rank IC.
    assert pnl(pred, close, 0.9, fee=0.05)["net_per_year"] < 0

    # Hysteresis: the position is held through the middle of the leg and not flattened there.
    pos = positions(pred, 0.9)
    assert set(pos.unique()) <= {-1.0, 1.0, 0.0} and (pos.iloc[1:] != 0).all()
    assert (pos.groupby(level=1).diff().abs() > 0).sum() < 25, "one flip per leg, not one per bar"
    # And the sign: at the low of the saw the rule is long.
    assert pos.iloc[np.argmin(ramp[1:]) + 1] == 1.0

    # Two symbols out of phase are held independently, and a symbol's last row pays nothing.
    both = pd.concat([close, close.rename(index={"a": "b"}, level=1).iloc[::-1]]).sort_index()
    assert forward_return(both).groupby(level=1).tail(1).isna().all()

    # A wider band flips less often, and the zero-threshold control — which turns at the middle
    # of every leg rather than at its ends — has to lose to a real threshold. That comparison is
    # the reason the control is in the grid: it holds the same signal and the same fee, so what
    # separates them is only where the rule chooses to act.
    table = sweep(pred, close)
    assert table.trades_per_year.is_monotonic_decreasing, table
    assert table.net_per_year.iloc[-1] > table.net_per_year.iloc[0], table
    # A symmetric flip rule is always in the market once it has fired at all: `in_market` says
    # whether the prediction ever reached the band, not how selective the rule is.
    assert table.in_market.iloc[0] == 1.0
    # On the saw both sides earn, because the signal really does read the turns. It is the
    # asymmetry on real data that means something, so the symmetric case has to hold here.
    both = pnl(pred, close, 0.9)
    assert both["gross_long"] > 0 and both["gross_short"] > 0, both
    assert 0.3 < both["long_share"] < 0.7, both


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="the predictions a gru run wrote")
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(QUANTILES),
        help="quantiles of |prediction| to take the thresholds at",
    )
    ap.add_argument("--fee", type=float, default=FEE, help="per side; the default is Alpaca taker tier 1")
    ap.add_argument(
        "--sign",
        type=int,
        default=-1,
        choices=[-1, 1],
        help="-1 for the swing label (low means long), +1 for remaining excursion",
    )
    args = ap.parse_args()

    _selfcheck()
    pred = pd.read_parquet(args.pred).iloc[:, 0]
    close = prices(pred.index)
    print(f"{len(pred):,} rows, {pred.index.get_level_values(1).nunique()} symbols, fee {args.fee * 100:.2f}% per side")
    print(f"{pred.index.get_level_values(0).min():%Y-%m-%d} to {pred.index.get_level_values(0).max():%Y-%m-%d}\n")
    print(sweep(pred, close, args.quantiles, args.fee, args.sign).round(4).to_string())
    print(f"\nbuy and hold, same rows: {buy_and_hold(close)['net_per_year'] * 100:.1f}% log per year")


if __name__ == "__main__":
    main()
