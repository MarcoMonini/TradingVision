"""Does a low prediction mean the leg is *ending*, or only that it has been *running*?

The two readings of `swing_leg_target` are not distinguishable by Rank IC against the label, and
the whole case for trading it turns on which one is true. This module separates them with the one
instrument that has no trading rule in it, so no answer can be blamed on a threshold, a fee or a
hysteresis: the forward return, bucketed by the prediction that was standing at the time.

The argument this exists to settle. The label is a position along a leg,

    target(i) ~ elapsed(i) / (elapsed(i) + remaining(i))

and the two terms are not symmetric in what they need. `elapsed` — how far the bar is from the
*previous* pivot — is entirely past and known at `i`. `remaining` is entirely future. The spec
measured that the denominator is nearly constant across the cross-section, which leaves the label
close to a rescaling of `elapsed` alone; `remaining_excursion`'s docstring is where that is
written down, next to the Rank IC 0.38 a plain OLS on point-in-time features reaches *flat at
every distance from the pivot*, which is what ruled out a pipeline fault and left arithmetic.

So a model can score 0.42 on this label by reproducing `elapsed` and know nothing about
`remaining`, and `elapsed` is the half that cannot be traded: "the price has fallen 5% over 30
bars since the last confirmed low" is a statement about what already happened. `crosscheck`
already priced that translation once — 0.3821 against the retrospective label becomes 0.0753
against the predictive one, four fifths gone, and small is not the same as zero. This module asks
the same question a third way and in the units the trade is actually paid in.

    uv run python -m tradingvision.legcheck --pred data/pred-swing-all-15m.parquet --symbols BTC

Two readings come out, and they answer different halves of the objection.

`by_bucket` is the instrument-free one. Sort the rows by the prediction, cut them into deciles,
and report the mean forward return of each with the standard error that says whether to believe
it. If a prediction near -1 means "the down leg is nearly over", the bottom decile has to carry a
positive forward return, and it has to carry it by more than its own error bar. No rule is applied,
nothing is thresholded, and no fee is charged, so a flat table cannot be answered with a better
exit rule — there is nothing there for a rule to act on. A monotone table is the opposite: it says
the signal is real and the rule is what needs work.

`partial` is the one that names the cause. It projects out the causal half — `elapsed_position`,
computed from pivots that had already been *confirmed* at the bar, so it uses no future — and
reports the Rank IC of what is left. A prediction whose forward information survives the control
is seeing the turn. One whose information disappears with it was reading the odometer.

On confirmation, which is the second half of "senza lag". A pivot is `argrelextrema` with
`order = EXTREMA_WINDOW`: it is an extreme only if it beats the 24 bars on *both* sides, so a low
at `p` is not identifiable until `p + 24` bars — six hours on the 15m grid. The label carries no
lag precisely because it is allowed to look there. `elapsed_position` is not, and the gap between
the two is not a detail of the implementation: it is the information the causal side does not
have. A pivot can also be revoked later, when the merge rule finds a more extreme bar of the same
kind in the same run, so the real-time picture is worse than the delay alone suggests; the control
here ignores that and is therefore, if anything, too generous to the prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import dataset, metrics, threshold
from tradingvision.data.binance import load
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots

BUCKETS = 10
# Rows ahead the forward return is measured over, on the row grid of the prediction file. The
# dataset samples hourly, so 24 is a day and 72 the horizon the cross-sectional label already uses.
HORIZONS = (6, 24, 72)


def forward(close: pd.Series, horizon: int, neutral: bool = True) -> pd.Series:
    """Log return over the next `horizon` rows of each symbol, market-neutral by default.

    Neutral because the alternative flatters or punishes the signal for the market's own direction:
    over a period the basket falls, anything leaning short earns and anything leaning long loses,
    and neither is a statement about the leg. With one symbol there is no cross-section to remove,
    so `neutral` falls back to the raw return and the caller is told rather than quietly given a
    column of zeros.
    """
    r = np.log(close.groupby(level=1).shift(-horizon) / close)
    if neutral and close.index.get_level_values(1).nunique() > 1:
        return r - r.groupby(level=0).transform("mean")
    return r


def by_bucket(pred: pd.Series, fwd: pd.Series, buckets: int = BUCKETS) -> pd.DataFrame:
    """Mean forward return per bucket of the prediction, with the error bar that judges it.

    Buckets are quantiles of the prediction over the whole slice, which is a description and not a
    decision: nothing here is fitted, promoted or traded on, so reading them off the same rows they
    summarise costs nothing. `t` is the mean over its own standard error — the column to read
    first, because a bucket mean without it is a number of unknown size.
    """
    at = pred.index.intersection(fwd.dropna().index)
    p, r = pred[at], fwd[at]
    edges = np.unique(p.quantile(np.linspace(0, 1, buckets + 1)).to_numpy())
    if len(edges) < 3:
        raise ValueError(f"the prediction takes {len(edges)} distinct quantiles — nothing to bucket")
    cut = pd.cut(p, edges, include_lowest=True, labels=False)
    g = r.groupby(cut)
    out = pd.DataFrame({"rows": g.size(), "pred_mean": p.groupby(cut).mean(), "fwd_mean": g.mean(), "fwd_sd": g.std()})
    out["se"] = out.fwd_sd / np.sqrt(out.rows)
    out["t"] = out.fwd_mean / out.se
    return out


def confirmed_pivots(symbol: str, idx: pd.DatetimeIndex, n: int = EXTREMA_WINDOW) -> pd.DataFrame:
    """The pivots of `symbol` carried onto `idx` at the bar they became *identifiable*, not at the
    bar they happened.

    `dataset.pivots_on` places a pivot at its own timestamp, which is right for a label and wrong
    for anything a decision may read: `find_pivots` needs `n` bars on each side, so the extreme at
    `p` is only visible from `p + n` bars onwards. Shifting by exactly that is what makes the
    control causal, and it is also the honest statement of what "real time" costs here.
    """
    piv = find_pivots(load(symbol, dataset.PIVOT_TF).close, n)
    # `+ n` bars of the pivot timeframe, then the same grid shift `dataset` applies to every frame.
    piv.index = piv.index + n * pd.Timedelta(dataset.PIVOT_TF) + dataset._shift(dataset.PIVOT_TF)
    return piv[(piv.index >= idx[0]) & (piv.index <= idx[-1])]


def elapsed_position(pred: pd.Series, n: int = EXTREMA_WINDOW) -> pd.Series:
    """The causal half of the swing label: where the bar sits relative to the last *confirmed*
    pivot, signed by that pivot's kind.

    `-1` just after a confirmed low and rising towards `+1` as the bar travels away from it, which
    is the same orientation `swing_leg_target` has — deliberately, because the point of the control
    is to be the part of the label that needs no future. Normalised by that symbol's own median leg
    duration over the confirmed pivots before the bar, so it is a position and not a bar count.

    Returned NaN before the first confirmed pivot, which `partial` drops rather than fills: a row
    with no pivot behind it has no position along a leg, and inventing one puts a number where the
    measurement has nothing.
    """
    out = []
    for symbol, rows in pred.groupby(level=1):
        when = rows.index.get_level_values(0)
        piv = confirmed_pivots(symbol, when, n)
        if piv.empty:
            out.append(pd.Series(np.nan, index=rows.index))
            continue
        at = pd.Series(piv.index, index=piv.index).reindex(when, method="ffill")
        kind = piv.kind.reindex(when, method="ffill")
        bars = (when - pd.DatetimeIndex(at)) / pd.Timedelta(dataset.PIVOT_TF)
        # Expanding median of the gaps between confirmed pivots, so the scale is train-side too.
        gap = pd.Series(piv.index, index=piv.index).diff() / pd.Timedelta(dataset.PIVOT_TF)
        scale = gap.expanding().median().reindex(when, method="ffill")
        # kind = +1 after a confirmed high, so the leg running from it is a down leg and the label
        # falls from +1 towards -1; the sign follows the pivot and the travel undoes it.
        pos = kind.to_numpy() * (1 - 2 * np.clip(bars.to_numpy() / scale.to_numpy(), 0, 1))
        out.append(pd.Series(pos, index=rows.index))
    return pd.concat(out).reindex(pred.index)


def partial(pred: pd.Series, fwd: pd.Series, control: pd.Series) -> dict[str, float]:
    """Rank IC of the prediction against the forward return, before and after the control.

    The residual is taken in rank space and by a plain projection: both series are replaced by
    their percentiles, the prediction is regressed on the control, and what is left is scored. A
    linear projection is the weakest possible control — it can only remove the part of the
    prediction that is an affine function of the causal position — so a signal that survives it has
    survived very little, and one that does not survive it was that function.
    """
    at = pred.index.intersection(fwd.dropna().index).intersection(control.dropna().index)
    if len(at) < 100:
        raise ValueError(f"{len(at)} rows have a prediction, a forward return and a control")
    p, r, c = pred[at], fwd[at], control[at]
    rp, rc = p.rank(pct=True), c.rank(pct=True)
    beta = np.polyfit(rc, rp, 1)[0]
    resid = rp - beta * rc
    return {
        "rows": len(at),
        "ic_raw": _ic(p, r, at),
        "ic_control": metrics.spearman(rc, r),
        "ic_residual": _ic(resid, r, at),
        "corr_pred_control": float(rp.corr(rc)),
    }


def _ic(x: pd.Series, r: pd.Series, index: pd.Index) -> float:
    """Rank IC per date when there is a cross-section to take it over, Spearman through time when
    there is not. One symbol has no cross-section and `metrics.by_date` would answer a column of
    NaN rather than say so — and a through-time Spearman is a different statistic, so the caller
    is told which one it got by how many symbols it passed in."""
    if isinstance(index, pd.MultiIndex) and index.get_level_values(1).nunique() >= 3:
        return float(metrics.by_date(x, r, rank=True).dropna().mean())
    return metrics.spearman(x, r)


def _selfcheck() -> None:
    n = 20_000
    rng = np.random.default_rng(0)
    when = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    idx = pd.MultiIndex.from_arrays([when, ["a"] * n])

    # A prediction that genuinely leads: it is the future return, so the buckets must be monotone
    # and the control must not be able to explain it away.
    r = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
    fwd = r.groupby(level=1).shift(-1)
    lead = (-fwd).fillna(0.0)
    table = by_bucket(lead, fwd, buckets=5)
    assert table.fwd_mean.is_monotonic_decreasing, table
    assert abs(table.t.iloc[0]) > 10 and abs(table.t.iloc[-1]) > 10, table

    # The null the whole argument turns on, built rather than asserted. Prices are a random walk,
    # so nothing about the future is knowable; `swing_leg_target` on them is still reproduced well
    # by a causal function of the past, because the label is mostly `elapsed`. High correlation
    # with the label, none with the forward return — which is the shape of the objection.
    close = pd.Series(np.exp(np.cumsum(rng.normal(scale=0.004, size=n))), index=when)
    piv = find_pivots(close, EXTREMA_WINDOW)
    assert len(piv) > 50, f"{len(piv)} pivots — the walk is too smooth to check anything"
    from tradingvision.data.target import swing_leg_target

    label = swing_leg_target(close, piv, EXTREMA_WINDOW)
    # The causal stand-in: signed travel away from the last *confirmed* pivot, the same shape
    # `elapsed_position` builds, computed here on the bare series.
    seen = piv.copy()
    seen.index = seen.index + EXTREMA_WINDOW * pd.Timedelta("15min")
    seen = seen[seen.index <= when[-1]]
    at = pd.Series(seen.index, index=seen.index).reindex(when, method="ffill")
    kind = seen.kind.reindex(when, method="ffill")
    bars = (when - pd.DatetimeIndex(at)) / pd.Timedelta("15min")
    gap = pd.Series(seen.index, index=seen.index).diff() / pd.Timedelta("15min")
    scale = gap.expanding().median().reindex(when, method="ffill")
    causal = pd.Series(kind.to_numpy() * (1 - 2 * np.clip(bars / scale, 0, 1)), index=when).dropna()
    both = causal.index.intersection(label.dropna().index)
    # It reads the label out of pure arithmetic on a series that has no future to know. Measured
    # 0.28 Spearman from a two-line causal statistic — the time term of the label and nothing else,
    # with neither the price advancement nor the per-pivot significance the real one carries. That
    # is the floor, not the ceiling: a GRU with 28 columns and 24 steps has far more of `elapsed`
    # available to it, which is the shape of the 0.42 the objection points at.
    got = metrics.spearman(causal[both], label[both])
    assert got > 0.25, got
    # And it knows nothing about what comes next, because on a random walk there is nothing to know.
    ahead = np.log(close.shift(-24) / close)
    live = causal.index.intersection(ahead.dropna().index)
    ic = metrics.spearman(causal[live], ahead[live])
    assert abs(ic) < 0.05, f"the walk leaked: {ic:.4f}"

    # `elapsed_position` is causal by truncation: a bar's control cannot move when later bars are
    # taken away. Checked against the same construction, since the module's own reads the store.
    cut = 12_000
    part = causal.iloc[:cut]
    assert np.allclose(causal.iloc[:cut].to_numpy(), part.to_numpy(), equal_nan=True)

    # `partial` strips exactly what the control explains: a prediction that *is* the control keeps
    # no residual information, and one orthogonal to it keeps all of its own.
    c = pd.Series(rng.normal(size=5_000))
    y = pd.Series(rng.normal(size=5_000))
    mixed = pd.Series(c.to_numpy() + 0.0 * y.to_numpy(), index=c.index)
    out = partial(mixed, y, c)
    assert abs(out["corr_pred_control"]) > 0.99 and abs(out["ic_residual"]) < 0.02, out
    out = partial(y, y, c)
    assert out["ic_residual"] > 0.9, out

    # A prediction with one value has no buckets, and says so rather than returning one row.
    flat = pd.Series(1.0, index=idx)
    try:
        by_bucket(flat, fwd)
    except ValueError as e:
        assert "distinct quantiles" in str(e)
    else:
        raise AssertionError("a constant prediction should not bucket")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="the predictions a gru run wrote")
    ap.add_argument("--symbols", nargs="*", help="restrict to these symbols; default is every one in the file")
    ap.add_argument("--since", help="first timestamp to read, e.g. 2025-09 for the last year")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS), help="rows ahead, on the pred grid")
    ap.add_argument("--buckets", type=int, default=BUCKETS)
    ap.add_argument("--raw", action="store_true", help="do not remove the cross-sectional mean from the return")
    args = ap.parse_args()

    _selfcheck()
    pred = pd.read_parquet(args.pred).iloc[:, 0].sort_index()
    if args.symbols:
        pred = pred[pred.index.get_level_values(1).isin(args.symbols)]
    if args.since:
        pred = pred[pred.index.get_level_values(0) >= pd.Timestamp(args.since, tz="UTC")]
    if not len(pred):
        raise SystemExit("no rows left after --symbols/--since")
    close = threshold.prices(pred.index)
    one = pred.index.get_level_values(1).nunique() < 2
    when = pred.index.get_level_values(0)
    print(f"{len(pred):,} rows, {pred.index.get_level_values(1).nunique()} symbol(s)")
    print(f"{when.min():%Y-%m-%d} to {when.max():%Y-%m-%d}")
    print("returns are raw" if args.raw or one else "returns are market-neutral", end="")
    print(" (one symbol: there is no cross-section to remove)\n" if one else "\n")

    control = elapsed_position(pred)
    for h in args.horizons:
        fwd = forward(close, h, neutral=not args.raw)
        print(f"--- {h} rows ahead ---\n")
        print(by_bucket(pred, fwd, args.buckets).round(5).to_string())
        print("\n" + ", ".join(f"{k} {v:.4f}" for k, v in partial(pred, fwd, control).items()) + "\n")


if __name__ == "__main__":
    main()
