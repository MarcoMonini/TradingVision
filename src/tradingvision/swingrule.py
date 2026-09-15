"""The long-only rule the swing label describes, with a threshold that does not read the scale.

`threshold` prices the symmetric flip rule: at every signal it is either long or short, never
flat. This prices the other rule the same label supports — buy near the low of a leg, hold, sell
near its high, then *wait* — and it exists because the symmetric measurement said something
specific about which half of it earns. On `swing_leg_target`, twenty symbols, 25 bp a side, the
gross decomposes by side at every threshold of the grid:

    q       gross long    gross short
    0.70      -0.198        +0.289
    0.95      -0.224        +0.269
    0.99      -0.065        +0.410
    0.998     -0.125        +0.398

The long leg of that rule is entry and exit at the same two points this module trades. So the
prior on a long-only reading of this label is not neutral and not unknown: it is the left column,
which is negative at every threshold measured, over a period when the basket fell 49% a year. The
symmetric rule's +25% net at q=0.99 was the right column carrying it. What this module changes is
the cost — a flat leg is not a position, so a cycle pays two sides instead of the two a flip pays
and then skips the next two — and what it cannot change is the sign of the left column. Run it to
find the number; do not expect the number to be new.

    uv run python -m tradingvision.swingrule --pred data/pred-swing-all.parquet --symbols BTC

Scale is the second thing this module is about, and the reason it does not take a raw number as a
threshold by default. A model fitted under a squared loss predicts a conditional mean, so its
output is the label shrunk by roughly the correlation it achieves — a prediction that lives in
[-0.5, +0.5] against a label that reaches [-1, +1] is that shrinkage and not a bug, and `spread`
measures it. Rescaling to undo it is a *monotone* transform: it moves no bar across any threshold
that is rescaled with it, so "double the prediction and buy at -0.8" and "buy at -0.4" are the
same rule with two spellings. Only the threshold's position inside the prediction's own
distribution decides anything, so `band` takes a quantile of |prediction| over that symbol's own
past rows and the rescaling drops out of the arithmetic. `_selfcheck` asserts that it does.

`--fixed` is there for the other spelling, and for the honest reason: a constant threshold is what
a live system would have to commit to, and seeing how far it drifts from the quantile it was meant
to be is the measurement that says whether committing to one is safe.

What this does not do. One symbol at a time is one path, and the project's own standard is a
dispersion across folds — a win rate over thirty trades has a standard error near 0.09, so a
single number here is a starting point for a comparison and not a result. There is no slippage and
no impact, the fill is the close of the bar that signalled, and `buy_and_hold` on the same rows is
the control a long-only rule has to beat before anything else about it is interesting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import threshold
from tradingvision.oracle import FEE

# Quantiles of |prediction| the default grid takes its band at. The entry and the exit share one
# number: the label is symmetric around zero by construction, so a rule that treats the two ends
# differently is asserting an asymmetry the label does not carry — `--buy`/`--sell` can still
# split them, and the split is then a claim to be measured rather than a default.
QUANTILES = (0.80, 0.90, 0.95, 0.99)
# Rows of a symbol's own history before a quantile of it means anything. 500 hourly rows is three
# weeks, which is short against the year a test slice runs and long enough that the estimate is not
# the last twenty bars wearing a quantile's name.
MIN_PERIODS = 500
YEAR = pd.Timedelta("365D")


def spread(pred: pd.Series, target: pd.Series | None = None) -> dict[str, float]:
    """How far the prediction's scale sits from the label's, and why.

    Under a squared loss the fitted output approaches `E[y|x]`, whose standard deviation is
    `corr(pred, y) * sd(y)` and never more: the part of the label the features do not explain is
    not predicted, it is averaged away. So a prediction reaching half the label's range is a model
    with a through-time correlation near a half, and the two columns this returns should agree.
    They come apart only when something else is also compressing the output — dropout and weight
    decay both do, and early stopping on Rank IC does not push back, since every metric downstream
    reads the ordering and is blind to the scale.

    `implied_rescale` is the factor that would put the prediction back on the label's spread. It
    is reported because it is asked for, and it changes no decision any threshold in this module
    takes: `band` is a quantile of the prediction's own distribution and moves with it.
    """
    out = {
        "pred_sd": float(pred.std()),
        "pred_q99_abs": float(pred.abs().quantile(0.99)),
        "pred_max_abs": float(pred.abs().max()),
    }
    if target is not None:
        y = target.reindex(pred.index)
        out |= {
            "target_sd": float(y.std()),
            "corr": float(pred.corr(y)),
            "sd_ratio": float(pred.std() / y.std()),
            "implied_rescale": float(y.std() / pred.std()),
        }
    return out


def band(pred: pd.Series, q: float, window: int = 0, min_periods: int = MIN_PERIODS) -> pd.Series:
    """The threshold at every row: the `q` quantile of |prediction| over that symbol's own rows.

    Causal and self-inclusive. The row at `t` enters its own quantile because `pred[t]` is known at
    `t` — excluding it would be a stricter rule, not a safer one — and nothing after `t` does. An
    expanding window by default rather than a rolling one: a rolling quantile tracks the model's
    recent output, which on a stretch where the model goes quiet lowers the bar and buys the
    quiet, and the expanding form has the one property that matters for a threshold, which is that
    it stops moving.

    Before `min_periods` rows the quantile is NaN and `positions` reads that as no signal, so the
    first weeks of a test slice are spent measuring the scale rather than trading a guess at it.
    """
    a = pred.abs().sort_index(level=[1, 0])
    g = a.groupby(level=1)
    roll = g.rolling(window, min_periods=min_periods) if window else g.expanding(min_periods=min_periods)
    return roll.quantile(q).droplevel(0).reindex(pred.index)


def positions(pred: pd.Series, buy: pd.Series | float, sell: pd.Series | float, sign: int = -1) -> pd.Series:
    """The held position: 1 long, 0 flat. Never short, which is the whole point of the module.

    `sign` is the direction the label points, as in `threshold.positions`: -1 for
    `swing_leg_target`, where a *low* prediction means the bar sits near a low and the leg runs up
    from there. The entry compares against `-buy` and the exit against `+sell` in the label's own
    orientation, so the caller passes two positive numbers and never has to reason about the sign.

    The state machine is a forward fill and not a loop, for the same reason `threshold.positions`
    is: the two signals are the only two states, so marking the entries 1, the exits 0 and filling
    forward *is* "hold until the other one fires". What it adds over the symmetric rule is that
    the exit leads to 0 rather than to -1, which is what makes "then wait" the third behaviour and
    not a short.
    """
    lo, hi = (-buy, sell) if sign < 0 else (sell, -buy)
    enter, exit_ = (pred <= lo, pred >= hi) if sign < 0 else (pred >= lo, pred <= hi)
    # A NaN threshold — the warm-up of `band` — compares False on both sides and fires neither.
    signal = pd.Series(np.nan, index=pred.index)
    signal[exit_] = 0.0
    signal[enter] = 1.0
    return signal.groupby(level=1).ffill().fillna(0.0)


def trades(pos: pd.Series, r: pd.Series, fee: float) -> pd.Series:
    """The net log P&L of every completed and open hold, one row per hold.

    A trade is the whole hold from entry to exit, not the hour: the label describes a leg and the
    rule is built to sit through one, so a win rate over hours would be counting the same trade
    many times and calling its noise a sample.
    """
    symbol = pos.index.get_level_values(1)
    f = pd.DataFrame({"pos": pos, "g": pos * r})
    f["leg"] = (f.pos != f.pos.groupby(symbol).shift()).groupby(symbol).cumsum()
    held = f[f.pos != 0]
    if not len(held):
        return pd.Series(dtype="float64")
    return held.groupby([held.index.get_level_values(1), held.leg]).g.sum() - 2 * fee


def pnl(pred: pd.Series, close: pd.Series, buy, sell, fee: float = FEE, sign: int = -1) -> dict:
    """One (buy, sell) band, priced. Log returns, the fee charged per side on the units traded."""
    when = pred.index.get_level_values(0)
    span = (when.max() - when.min()) / YEAR
    pos = positions(pred, buy, sell, sign)
    r = threshold.forward_return(close).fillna(0.0)
    symbol = pos.index.get_level_values(1)
    turnover = pos.groupby(symbol).diff().fillna(pos).abs()
    gross, cost = (pos * r).groupby(symbol).sum(), (turnover * fee).groupby(symbol).sum()
    per_trade = trades(pos, r, fee)
    n = len(per_trade) / gross.size
    return {
        "in_market": float((pos != 0).mean()),
        "trades_per_year": n / span,
        "gross_per_year": float(gross.mean() / span),
        "fees_per_year": float(cost.mean() / span),
        "net_per_year": float((gross.mean() - cost.mean()) / span),
        "net_per_trade": float(per_trade.mean()) if len(per_trade) else np.nan,
        "median_trade": float(per_trade.median()) if len(per_trade) else np.nan,
        "win_rate": float((per_trade > 0).mean()) if len(per_trade) else np.nan,
        "trades": len(per_trade),
        # The standard error of a win rate over this many trades, which is the number that says
        # whether the win rate above is a measurement or a sentence about thirty coin flips.
        "win_rate_se": float(np.sqrt(0.25 / len(per_trade))) if len(per_trade) else np.nan,
    }


def sweep(
    pred: pd.Series,
    close: pd.Series,
    quantiles=QUANTILES,
    fee: float = FEE,
    sign: int = -1,
    window: int = 0,
    fixed: bool = False,
) -> pd.DataFrame:
    """One row per quantile, with the band taken dynamically or frozen at its full-period value.

    `fixed` is the comparison and not an option to prefer: it takes the same quantile over the
    whole slice and holds it constant, which reads the period it is trading. The gap between the
    two rows at one quantile is what committing to a constant threshold costs — or, when the gap
    is large, the reason not to.
    """
    rows = {}
    for q in quantiles:
        t = float(pred.abs().quantile(q)) if fixed else band(pred, q, window)
        rows[q] = pnl(pred, close, t, t, fee, sign)
    return pd.DataFrame(rows).T.rename_axis("quantile")


def buy_and_hold(close: pd.Series) -> dict:
    """Holding every symbol throughout, on the same rows and the same accounting — the control a
    long-only rule has to beat before its win rate means anything at all."""
    r = threshold.forward_return(close).fillna(0.0)
    when = close.index.get_level_values(0)
    span = (when.max() - when.min()) / YEAR
    per_symbol = r.groupby(close.index.get_level_values(1)).sum()
    return {"net_per_year": float(per_symbol.mean() / span), "trades_per_year": 1 / span}


def _selfcheck() -> None:
    """A saw the prediction reads perfectly: the rule has to take the up legs and skip the downs."""
    n = 800
    when = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    idx = pd.MultiIndex.from_arrays([when, ["a"] * n])
    phase = np.arange(n) % 40
    ramp = np.where(phase < 20, phase, 40 - phase) / 20.0  # 20 up, 20 down, 2% peak to peak
    close = pd.Series(np.exp(0.02 * ramp), index=idx)
    pred = pd.Series(2 * ramp - 1, index=idx)  # the label read exactly: -1 at the lows, +1 at the peaks

    pos = positions(pred, 0.9, 0.9)
    assert set(pos.unique()) <= {0.0, 1.0}, "long only, never short"
    # It is flat for about the half of the cycle the price spends falling, which the symmetric
    # rule spends short. That difference is the module.
    assert 0.4 < (pos != 0).mean() < 0.6, pos.mean()
    assert pos.iloc[np.argmin(ramp[1:]) + 1] == 1.0, "long at the low of the saw"
    # After an exit it waits: the bars right after a peak are flat, not re-entered.
    after = pos[np.argmax(ramp[:60]) + 1 : np.argmax(ramp[:60]) + 10]
    assert (after == 0).all(), after.tolist()

    out = pnl(pred, close, 0.9, 0.9)
    assert out["win_rate"] >= 0.95 and out["net_per_trade"] > 0.01, out
    # Twenty up legs of 2% each over the span, and only the up legs — the symmetric rule on the
    # same saw collects both and makes twice this.
    span = (when[-1] - when[0]) / YEAR
    assert np.isclose(out["gross_per_year"] * span, 20 * 0.02, atol=0.02), out
    flip = threshold.pnl(pred, close, 0.9)
    assert flip["gross_per_year"] > 1.8 * out["gross_per_year"], (flip, out)
    # And pays less to do it: a cycle here is one round trip, a cycle there is two flips.
    assert out["fees_per_year"] < flip["fees_per_year"], (out, flip)
    # A fee past the size of the leg turns the same perfect signal into a loss. Nothing about the
    # ordering changed, which is why this is measured and not read off the Rank IC.
    assert pnl(pred, close, 0.9, 0.9, fee=0.05)["net_per_year"] < 0

    # The scale argument, in code. A quantile band is invariant to any positive rescaling of the
    # prediction, so doubling the output moves no bar across it — that is the whole content of
    # "raddoppiare la predizione", and it is a no-op unless the threshold is left behind.
    doubled = 2 * pred
    q = 0.9
    assert positions(pred, band(pred, q, min_periods=50), band(pred, q, min_periods=50)).equals(
        positions(doubled, band(doubled, q, min_periods=50), band(doubled, q, min_periods=50))
    )
    # A *fixed* threshold is where the rescaling stops being a no-op, and only because the number
    # was not rescaled with it: -0.8 on the doubled output is -0.4 on the raw one, same rule.
    assert positions(doubled, 0.8, 0.8).equals(positions(pred, 0.4, 0.4))
    # Left behind, it is a different rule and trades differently — the failure mode the fixed
    # spelling invites.
    assert not positions(doubled, 0.8, 0.8).equals(positions(pred, 0.8, 0.8))

    # `band` reads a symbol's own past and nothing later: truncating the series cannot change the
    # thresholds that survive the cut. Same check `dataset` makes on the branches.
    cut = 600
    whole = band(pred, 0.9, min_periods=50)
    part = band(pred.iloc[:cut], 0.9, min_periods=50)
    assert np.allclose(whole.iloc[:cut].to_numpy(), part.to_numpy(), equal_nan=True)
    # The warm-up holds nothing rather than trading a quantile of twenty bars.
    warm = band(pred, 0.9, min_periods=200)
    assert warm.iloc[:199].isna().all() and warm.iloc[199:].notna().all()
    assert (positions(pred, warm, warm).iloc[:199] == 0).all()

    # Two symbols out of phase are held independently, each along its own rows.
    other = close.rename(index={"a": "b"}, level=1).iloc[::-1]
    both = pd.concat([close, other]).sort_index()
    assert threshold.forward_return(both).groupby(level=1).tail(1).isna().all()
    p2 = pd.concat([pred, pred.rename(index={"a": "b"}, level=1).iloc[::-1]]).sort_index()
    assert positions(p2, 0.9, 0.9).groupby(level=1).size().nunique() == 1

    # A band no prediction ever reaches holds nothing and reports no trade, rather than a zero.
    empty = pnl(pred, close, 5.0, 5.0)
    assert empty["trades"] == 0 and empty["in_market"] == 0.0 and np.isnan(empty["win_rate"])

    # `spread` recovers the shrinkage it is there to explain: a prediction correlated `rho` with
    # the label has `rho` times its spread, which is the whole of the [-0.5, +0.5] question.
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=20_000))
    for rho in (0.3, 0.5, 0.8):
        e = rho * y + np.sqrt(1 - rho**2) * pd.Series(rng.normal(size=len(y)))
        s = spread(rho * e / e.std() * y.std(), y)  # E[y|x] under a squared loss, exactly
        assert abs(s["sd_ratio"] - rho) < 0.02 and abs(s["implied_rescale"] - 1 / rho) < 0.1, (rho, s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="the predictions a gru run wrote")
    ap.add_argument("--symbols", nargs="*", help="restrict to these symbols; default is every one in the file")
    ap.add_argument("--since", help="first timestamp to price, e.g. 2025-09 for the last year")
    ap.add_argument("--quantiles", type=float, nargs="+", default=list(QUANTILES), help="band quantiles of |pred|")
    ap.add_argument("--window", type=int, default=0, help="rolling rows for the band; 0 is expanding")
    ap.add_argument("--fixed", nargs="*", type=float, help="price these raw thresholds too, e.g. --fixed 0.4 0.8")
    ap.add_argument("--fee", type=float, default=FEE, help="per side; the default is Alpaca taker tier 1")
    ap.add_argument("--smooth", type=int, default=1, help="output low-pass over each symbol's own rows")
    ap.add_argument("--sign", type=int, default=-1, choices=[-1, 1], help="-1 for the swing label, +1 for excursion")
    args = ap.parse_args()

    _selfcheck()
    pred = pd.read_parquet(args.pred).iloc[:, 0]
    if args.symbols:
        pred = pred[pred.index.get_level_values(1).isin(args.symbols)]
    if args.since:
        pred = pred[pred.index.get_level_values(0) >= pd.Timestamp(args.since, tz="UTC")]
    if not len(pred):
        raise SystemExit("no rows left after --symbols/--since")
    pred = threshold.smoothed(pred.sort_index(), args.smooth)
    close = threshold.prices(pred.index)
    when = pred.index.get_level_values(0)
    print(f"{len(pred):,} rows, {pred.index.get_level_values(1).nunique()} symbol(s), fee {args.fee * 100:.2f}%/side")
    print(f"{when.min():%Y-%m-%d} to {when.max():%Y-%m-%d}\n")
    print("prediction scale: " + ", ".join(f"{k} {v:.3f}" for k, v in spread(pred).items()) + "\n")

    print("dynamic band, expanding quantile of |pred| over each symbol's own past rows\n")
    print(sweep(pred, close, args.quantiles, args.fee, args.sign, args.window).round(4).to_string())
    print("\nthe same quantiles frozen over the whole slice, which reads the period it trades\n")
    print(sweep(pred, close, args.quantiles, args.fee, args.sign, fixed=True).round(4).to_string())
    if args.fixed:
        print("\nfixed thresholds on the raw prediction\n")
        rows = {t: pnl(pred, close, t, t, args.fee, args.sign) for t in args.fixed}
        print(pd.DataFrame(rows).T.rename_axis("threshold").round(4).to_string())
    bh = buy_and_hold(close)
    print(f"\nbuy and hold, same rows: {bh['net_per_year'] * 100:.1f}% log per year")


if __name__ == "__main__":
    main()
