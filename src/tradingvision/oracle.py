"""Step 0 of the validation sequence: calibrate `extrema_window`, the parameter every other window
in the project derives from.

The oracle buys the close of every pivot low and sells the close of the next pivot high, net of
fees. Scored on hindsight alone the answer is degenerate — see `run` — so windows are compared
across detection lags, and the window that survives a lag of 2-3 bars is the one a causal model
could plausibly trade. That measurement fixed `EXTREMA_WINDOW` at 24; re-run it when the fee
assumption changes or the volatility regime moves, because the answer tracks both.

    python -m tradingvision.oracle --start 2023-01-01 --csv oracle.csv

Reads the local Binance store, so run `python -m tradingvision.data.binance` first. Fees are
Alpaca's, the venue where the model would actually execute.

Long-only and fully invested per leg: it measures how much a leg is worth, not a strategy.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tradingvision.data.binance import SYMBOLS, load
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots

# Alpaca crypto taker fee, tier 1 (30d volume under $100k). Maker is 0.15%, but the oracle enters
# and exits at the close of a pivot bar, which is a taker fill. Round trip costs 2x.
FEE = 0.0025


def run(
    close: pd.Series,
    window: int = EXTREMA_WINDOW,
    fee: float = FEE,
    pivots: pd.DataFrame | None = None,
    lag: int = 0,
) -> dict:
    """Net P&L of the hindsight trader over one close series.

    `pivots` skips the detection when the caller already has it for this window.

    `lag` fills the trade that many bars after the pivot instead of on it, which is what makes the
    measure usable for choosing a window: at lag 0 the P&L rises monotonically as the window
    shrinks, until the average leg stops clearing costs. That optimum is a cost/volatility ratio,
    not market structure — it lands on the same ~1% median leg on every symbol, at windows from 2
    to 8 bars. Charging a detection lag is what separates legs a causal model could actually hold
    from legs that only exist to a perfect predictor, and a pivot is anyway only confirmable
    `window` bars after the fact, so lag 0 and lag 1 are both unattainable by construction.
    """
    piv = find_pivots(close, window) if pivots is None else pivots
    # Positional, because a lagged fill has no pivot to index by. get_indexer answers -1 for a
    # label it cannot find, which would silently read the wrong bar, so mismatched pivots have to
    # fail loudly: passing a frame built on another slice is an easy mistake and a plausible
    # wrong number is worse than a crash.
    at = close.index.get_indexer(piv.index)
    if (at < 0).any():
        raise ValueError("pivots do not belong to this close series")

    pos = at + lag
    inside = pos < len(close)
    kind, pos, at = piv.kind.to_numpy()[inside], pos[inside], at[inside]
    px = close.to_numpy()[pos]
    # Each low is closed by the next pivot, always a high after the merge. Both fills shift by the
    # same lag, so the exit always follows the entry — but when two pivots sit closer together than
    # `lag` (0.24% of pairs at window 24) the entry lands past the exit pivot, and the trade covers
    # a stretch that no longer overlaps the leg at all. A model that detected the low that late
    # would not take the trade, so neither does the oracle.
    live = pos[:-1] < at[1:]
    opened = (kind[:-1] == -1) & live
    entries, exits = pd.Series(px[:-1][opened]), pd.Series(px[1:][opened])
    skipped = int(((kind[:-1] == -1) & ~live).sum())
    # Fees are charged on the credited side of each fill, so they compound with the price ratio
    # rather than subtracting from the return.
    legs = ((exits / entries) * (1 - fee) ** 2 - 1).dropna()
    bars_per_leg = pd.Series(piv.index).diff().dropna()
    years = (close.index[-1] - close.index[0]) / pd.Timedelta(365.25, "D")
    # Compounded over several years the perfect-hindsight return overflows, so the comparable
    # figure across symbols and history lengths is the log P&L per year.
    net_log = float(np.log1p(legs).sum())
    return {
        "window": window,
        "lag": lag,
        "trades": len(legs),
        "skipped": skipped,  # legs dropped because the lagged entry did not precede the exit
        "log_per_year": net_log / years if years else float("nan"),
        "net_return": float(np.expm1(net_log)) if net_log < 700 else float("inf"),
        "gross_leg_pct": float((exits / entries - 1).mean() * 100) if len(legs) else float("nan"),
        "win_rate": float((legs > 0).mean()) if len(legs) else float("nan"),
        "median_leg_duration": bars_per_leg.median() if len(bars_per_leg) else pd.NaT,
    }


def sweep(symbols, timeframe: str, windows: list[int], lags=(0,), fee: float = FEE, start=None) -> pd.DataFrame:
    """One row per (symbol, window, lag), over the local Binance store."""
    rows = []
    for i, symbol in enumerate(symbols, 1):
        close = load(symbol, timeframe).close
        if start:
            close = close.loc[start:]
        for w in windows:
            piv = find_pivots(close, w)
            rows += [{"symbol": symbol, **run(close, w, fee, piv, lag)} for lag in lags]
        print(f"[{i}/{len(symbols)}] {symbol} — {len(close):,} bars", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default="15m", help="reference timeframe for the target")
    ap.add_argument("--start", help="ISO date, to restrict the panel to an aligned period")
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 18, 20, 22, 24, 26, 28])
    ap.add_argument("--fee", type=float, default=FEE, help="fee per side, e.g. 0.0025 = 25 bps")
    ap.add_argument("--lag", type=int, nargs="+", default=[0, 1, 2, 3], help="detection lag in bars")
    ap.add_argument("--csv", help="write the per-symbol results here")
    args = ap.parse_args()

    res = sweep(args.symbols, args.timeframe, args.windows, args.lag, args.fee, args.start)
    if args.csv:
        res.to_csv(args.csv, index=False)

    print(f"\n{args.timeframe} bars, {args.fee * 200:.2f}% round trip — mean over {len(args.symbols)} symbols")
    for name, values in (("log P&L per year", "log_per_year"), ("share of profitable legs", "win_rate")):
        print(f"\n{name}, by window (rows) and detection lag (columns):")
        print(res.pivot_table(index="window", columns="lag", values=values).round(3).to_string())
    # A single argmax is not meaningful here: at a realistic lag the top of the range is flat to
    # well under a percent, so report the plateau and let the tie break on the share of legs that
    # still pay, which rises monotonically with the window.
    real = res[res.lag >= 2].pivot_table(index="window", values=["log_per_year", "win_rate"])
    flat = real[real.log_per_year >= 0.99 * real.log_per_year.max()]
    print(f"\nat a realistic detection lag (>=2 bars), windows within 1% of the best: {list(flat.index)}")
    print(f"  of those, the best share of profitable legs: {flat.win_rate.idxmax()}  (fixed: {EXTREMA_WINDOW})")


if __name__ == "__main__":
    main()
