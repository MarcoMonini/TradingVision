"""Step 0 of the validation sequence: calibrate `extrema_window`.

The oracle is a trader with perfect hindsight: it buys the close of every pivot low and sells the
close of the next pivot high. A small window yields many tiny legs eaten by fees, a large one
yields few clean but rare legs — the window range that maximises net P&L is the range of
economically meaningful legs, and everything else in the project (indicator windows, purging
width, steps per branch) is derived from it.

    python -m tradingvision.oracle --symbols BTC/USD ETH/USD --days 365 --windows 6 12 24 48

Long-only and fully invested per leg: it measures how much a leg is worth, not a strategy.
"""

from __future__ import annotations

import argparse

import pandas as pd

from tradingvision.data.candles import SYMBOLS, get_candles
from tradingvision.data.pivots import find_pivots

# Alpaca crypto taker fee, tier 1 (30d volume under $100k). Maker is 0.15%, but the oracle enters
# and exits at the close of a pivot bar, which is a taker fill. Round trip costs 2x.
FEE = 0.0025


def run(close: pd.Series, window: int, fee: float = FEE, pivots: pd.DataFrame | None = None) -> dict:
    """Net P&L of the perfect-hindsight trader over one close series.

    `pivots` skips the detection when the caller already has it for this window.
    """
    piv = find_pivots(close, window) if pivots is None else pivots
    lows = piv[piv.kind == -1]
    # Each low is closed by the next pivot, which after the merge is always a high.
    exits = piv.close.shift(-1).reindex(lows.index)
    legs = (exits / lows.close - 1 - 2 * fee).dropna()
    bars_per_leg = pd.Series(piv.index).diff().dropna()
    return {
        "window": window,
        "trades": len(legs),
        "net_return": float((1 + legs).prod() - 1),
        "gross_leg_pct": float((exits / lows.close - 1).mean() * 100) if len(legs) else float("nan"),
        "win_rate": float((legs > 0).mean()) if len(legs) else float("nan"),
        "median_leg_duration": bars_per_leg.median() if len(bars_per_leg) else pd.NaT,
    }


def sweep(symbols: list[str], timeframe: str, days: int, windows: list[int], fee: float = FEE) -> pd.DataFrame:
    """One row per (symbol, window). Symbols with no data are skipped."""
    rows = []
    for symbol in symbols:
        df = get_candles(symbol, timeframe, days)
        if df.empty:
            print(f"skipped {symbol}: no data")
            continue
        rows += [{"symbol": symbol, **run(df.close, w, fee)} for w in windows]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default="15m", help="reference timeframe for the target")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--windows", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--fee", type=float, default=FEE, help="fee per side, e.g. 0.001 = 10 bps")
    ap.add_argument("--csv", help="write the per-symbol results here")
    args = ap.parse_args()

    res = sweep(args.symbols, args.timeframe, args.days, args.windows, args.fee)
    if res.empty:
        print("no data")
        return
    if args.csv:
        res.to_csv(args.csv, index=False)

    print(res.to_string(index=False))
    print("\nmean over symbols:")
    agg = res.groupby("window")[["trades", "net_return", "gross_leg_pct", "win_rate"]].mean()
    print(agg.to_string())
    print(f"\nbest window by mean net return: {agg.net_return.idxmax()}")


if __name__ == "__main__":
    main()
