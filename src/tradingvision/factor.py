"""The cross-sectional factor model, and the measurement that says how wide its lens should be.

This is the step the spec left open and named: *no model has been measured against the composite*
(open point 2). The composite is two ranked columns added together with no parameters, and on the
new label it beat every fitted thing in the project. What had never been asked is the one question
that comes before "which columns" — **over how long a window are they measured**.

`N = EXTREMA_WINDOW = 24` bars was measured, carefully, on the *leg* problem: the window that
detects tradable swings once a detection lag is charged. Every feature in `features` inherits it.
But the label is no longer a leg — it is a 72h cross-sectional return, and nothing had ever
checked that six hours of history is the right lens for it. It is not:

    composite  -rank(volatility) + rank(dollar volume), equal weight, four folds from 2025-06

    window      fold 1    fold 2    fold 3    fold 4      mean       sd
    24 (6h)     0.0875    0.0840    0.1234    0.1000    0.0987    0.018
    train-picked (2880 in every fold)
    2880 (30d)  0.1032    0.0996    0.1266    0.1107    0.1100    0.012

Four folds out of four, which is the standard step 3 was promoted on. The window is chosen inside
each fold on the train Rank IC alone and lands on 2880 every time; 24 is the worst of the grid on
train as well, so no test slice was read to get there.

**The money is the larger half.** A six-hour volatility estimate is mostly estimation noise, so
its ranking churns every hour and the book pays for the churn; a thirty-day estimate ranks the
same way for weeks. At the widest threshold — 40% of the cross-section on each side, the most
diversified book the rule can hold — the two windows are not the same strategy at all:

    theta 0.2, hedged log return per year, mean over the four folds

    window      trades/yr    gross    net @25bp    net @10bp    Sharpe @25bp
    24 (6h)         173.4    0.164       -0.703       -0.182          -4.81
    2880 (30d)        2.3    0.187       +0.175       +0.182          +1.18

Open point 1 reads "close the gap between Rank IC 0.106 and a break-even of 0.184 at theta 0.5",
with two levers, more skill or a cheaper venue. There was a third: the gap was never only about
skill. A rule that rebalances a noise estimate 173 times a year cannot pay 25bp a side at any IC
the table reaches, and the same signal measured over a horizon that matches the label pays at the
fee the project already has.

**What was measured and lost.** Seven more candidates, each ranked inside the date and then
residualised against the composite, on train alone: `idiosyncratic_vol` (-0.029), `range_vol`
(-0.026), `corr_to_basket` (+0.025), `beta_to_basket` (-0.004), `amihud` (+0.010), `drawdown`
(-0.013), and trailing returns at 24h/72h/7d/30d, whose residual sign flips between train and
test. Added to the pair and fitted by ridge on the train side they all *lose* out of sample —
0.095 with `corr`, 0.087 with four more, 0.083 with all nine, against 0.110 for the pair. Fitting
the two weights instead of adding them equally scores 0.1146 and wins three folds of four, which
is inside the fold spread: the pair is kept equal-weighted because two parameters are not worth
the third fold.

    uv run python -m tradingvision.factor
    uv run python -m tradingvision.factor --price
    uv run python -m tradingvision.factor --save data/pred-factor.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import metrics
from tradingvision.data.binance import STORE, SYMBOLS, load
from tradingvision.data.target import CROSS_HORIZON, cross_sectional_return

TF = "15m"  # the grid the legs and the label live on — the store's own name for the file
# The same two as durations. `pd.Timedelta("15m")` still parses as minutes but is deprecated, and
# `pd.date_range(freq="15m")` already means *month*: the ambiguity is worth spelling out once here
# rather than meeting it on a grid that is silently a thousand times too wide.
BAR = pd.Timedelta("15min")
BASE_BAR = pd.Timedelta("5min")
SINCE = "2023"  # the period every symbol covers, so the cross-section is full
TEST_START = "2025-06"
FOLDS = 4
# Bars of `TF`: 24h, 72h, 7d, 30d. 24 — the project's `EXTREMA_WINDOW` — is measured next to them
# by `--baseline` and is not in the grid the fold picks from, because the point of the grid is the
# choice and the point of the baseline is what that choice replaced.
WINDOWS = (96, 288, 672, 2880)
BASELINE_WINDOW = 24
DECISION = pd.Timedelta("1h")  # one decision per clock hour, the stride `dataset` samples at


def panel(symbols: list[str] | None = None, since: str = SINCE) -> dict[str, pd.DataFrame]:
    """`close` and `dollar_volume` as wide (time x symbol) frames on one shared `TF` index.

    Straight off the store and not out of `dataset`: a factor is a statement about the panel at an
    instant, so it is built on the panel. It also means the whole study runs on raw bars in a
    couple of minutes instead of on a 1.5 GB flattened cache.
    """
    bars = {s: load(s, TF).loc[since:] for s in (symbols or list(SYMBOLS))}
    idx = None
    for b in bars.values():
        idx = b.index if idx is None else idx.union(b.index)
    close = pd.DataFrame({s: b.close for s, b in bars.items()}).reindex(idx)
    volume = pd.DataFrame({s: b.volume for s, b in bars.items()}).reindex(idx)
    return {"close": close, "dollar_volume": volume * close}


def cross_rank(f: pd.DataFrame) -> pd.DataFrame:
    """Centred percentile inside each row — `normalize.cross_rank` on a wide frame.

    Same statistic and the same reason it is centred on the row mean rather than on a flat 0.5: a
    cross-section that thins would otherwise carry an offset made of nothing but its own count.
    """
    r = f.rank(axis=1, pct=True)
    return r.sub(r.mean(axis=1), axis=0)


def composite(p: dict[str, pd.DataFrame], n: int) -> pd.DataFrame:
    """`-rank(realized volatility) + rank(log dollar volume)` over `n` bars. No parameters.

    The two columns are the ones the spec's selection kept on this label, and the signs are the
    low-volatility anomaly and the size effect. Equal weight because the fitted alternative is
    inside the fold spread — see the module docstring.
    """
    ret = np.log(p["close"]).diff()
    volatility = ret.rolling(n).std()
    size = np.log(p["dollar_volume"].rolling(n).mean().replace(0, np.nan))
    return cross_rank(size) - cross_rank(volatility)


def hourly(index: pd.DatetimeIndex) -> np.ndarray:
    """The `TF` bars that close on a clock hour — one decision an hour, the same one for everyone.

    On the clock and never by position: a positional stride drifts the moment a symbol loses a
    bar, and `dataset.build` records what that cost. `TF` labels are open times, so the bar that
    closes at the hour is the one labelled one `TF` before it.
    """
    closes = index + BAR
    return np.asarray(closes.floor(DECISION) == closes)


def label(p: dict[str, pd.DataFrame], horizon: int = CROSS_HORIZON) -> pd.DataFrame:
    return cross_sectional_return(p["close"], horizon)


def long(f: pd.DataFrame, when: np.ndarray) -> pd.Series:
    """A wide frame on the (timestamp, symbol) index the metrics read, on the rows of `when`."""
    return f[when].stack(future_stack=True).dropna()


def folds(index: pd.DatetimeIndex, start: str = TEST_START, n: int = FOLDS) -> list[tuple]:
    """`(train, test)` boolean masks per expanding fold, the train side purged of the horizon.

    The label reaches a fixed `CROSS_HORIZON` bars ahead and nowhere else, so purging is the
    inequality itself: keep a train bar when its label closes before the cut. Same rule as
    `dataset.relabel_cross` writes into `next_pivot`, stated here directly because this module
    never builds that column.
    """
    at, last = pd.Timestamp(start, tz="UTC"), index.max()
    edges = [at + (last - at) * i / n for i in range(n)] + [None]
    reach = index + CROSS_HORIZON * BAR
    clock = hourly(index)
    out = []
    for cut, until in zip(edges, edges[1:]):
        test = clock & np.asarray(index >= cut)
        if until is not None:
            test &= np.asarray(index < until)
        out.append((clock & np.asarray(reach < cut), test))
    return out


def rank_ic(sig: pd.DataFrame, y: pd.DataFrame, when: np.ndarray) -> pd.Series:
    """The per-date Rank IC of a wide signal against a wide label, vectorised over the panel."""
    a, b = sig[when], y[when]
    ok = a.notna() & b.notna()
    ra, rb = a.rank(axis=1).where(ok), b.rank(axis=1).where(ok)
    count = ok.sum(axis=1)
    za = ra.sub(ra.mean(axis=1), axis=0).div(ra.std(axis=1), axis=0)
    zb = rb.sub(rb.mean(axis=1), axis=0).div(rb.std(axis=1), axis=0)
    return ((za * zb).sum(axis=1) / (count - 1))[count > 2].dropna()


def pick(p: dict[str, pd.DataFrame], y: pd.DataFrame, train: np.ndarray, windows=WINDOWS) -> int:
    """The window with the best *train* Rank IC. The only thing this model fits, and it reads the
    train side of its own fold and nothing else."""
    return max(windows, key=lambda n: rank_ic(composite(p, n), y, train).mean())


def horizon_delta() -> pd.Timedelta:
    """How far the label reaches, as the block width `metrics.blocked` needs."""
    return CROSS_HORIZON * BAR


def walk_forward(p: dict[str, pd.DataFrame], windows=WINDOWS, start: str = TEST_START, n: int = FOLDS):
    """Per-fold test Rank IC of the composite, with the window picked on each fold's train side."""
    y = label(p)
    rows, chosen, preds = {}, [], []
    for i, (train, test) in enumerate(folds(p["close"].index, start, n), 1):
        window = pick(p, y, train, windows)
        sig = composite(p, window)
        ic = rank_ic(sig, y, test)
        b = metrics.blocked(ic, horizon_delta())
        rows[f"fold {i}"] = {"window": window, "dates": len(ic), "rank_ic": b["mean"], "se": b["se"], "t": b["t"]}
        chosen.append(window)
        preds.append(long(sig, test))
    out = pd.DataFrame(rows).T
    out.loc["mean"] = out.mean()
    out.loc["std"] = out.iloc[:-1].std()
    out.loc[["mean", "std"], "window"] = np.nan
    return out, pd.concat(preds).sort_index(), chosen


# Prices are read at the 5m bar that closes with the `TF` candle the signal was built from, which
# is `TF - 5m` past its label — the placement rule `dataset` states and the reason a prediction
# written here can be handed straight to `simulation --pred` without a bar of anticipation.
PRED_SHIFT = BAR - BASE_BAR


def predictions(sig: pd.Series) -> pd.Series:
    """`sig` moved onto the 5m labels the rest of the pipeline prices on."""
    ts, symbol = sig.index.get_level_values(0) + PRED_SHIFT, sig.index.get_level_values(1)
    return pd.Series(sig.to_numpy(), index=pd.MultiIndex.from_arrays([ts, symbol]), name="pred").sort_index()


YEAR = pd.Timedelta("365D")
FEES = (0.0025, 0.0010, 0.0005)
THETAS = (0.2, 0.5, 0.8)


def price(sig: pd.DataFrame, y_close: pd.DataFrame, test: np.ndarray, theta: float, step: int = 4) -> dict:
    """The hedged book of one threshold, priced on the panel's own closes.

    `step` is the decision interval in `TF` bars — 4 is the hour the rows are sampled at. The
    position taken at the close of a bar earns the return to the close one decision later, so no
    second clock enters and there is nowhere for a bar of anticipation to hide.

    Hedged and never naked: the label is a return in excess of the basket, so the basket leg is
    part of the trade the signal describes. `simulation` says the same thing at more length.
    """
    forward = np.log(y_close.shift(-step) / y_close)
    hedged = forward.sub(forward.mean(axis=1), axis=0)
    pct = sig[test].rank(axis=1, pct=True)
    band = 2 * pct.sub(pct.mean(axis=1), axis=0)
    # Hysteresis, as `threshold.positions` does it: hold through the middle rather than flatten,
    # or the book pays a round trip every time the signal is unremarkable.
    pos = pd.DataFrame(np.nan, index=band.index, columns=band.columns)
    pos[band <= -theta], pos[band >= theta] = -1.0, 1.0
    pos = pos.ffill().fillna(0.0)

    span = (pos.index[-1] - pos.index[0]) / YEAR
    turnover = pos.diff().fillna(pos).abs().to_numpy().sum() / pos.shape[1] / span
    per_step = (pos * hedged[test].reindex_like(pos)).fillna(0.0).mean(axis=1)
    gross = per_step.sum() / span
    risk = per_step.std() * np.sqrt(len(per_step) / span)
    out = {
        "theta": theta,
        "in_market": float((pos != 0).to_numpy().mean()),
        # Units of position changed per symbol over the whole test, before any annualising: a book
        # that never changes its mind still pays for opening, and this is the number that says so.
        "turnover": turnover * span,
        "trades_per_year": turnover / 2,
    }
    out["gross_hedged"] = gross
    for fee in FEES:
        out[f"net_{round(fee * 1e4)}bp"] = gross - turnover * fee
    out["sharpe_25bp"] = (gross - turnover * FEES[0]) / risk if risk else np.nan
    return out


def by_quarter(p: dict[str, pd.DataFrame], window: int, theta: float = 0.2, step: int = 4) -> pd.DataFrame:
    """Rank IC and hedged P&L per calendar quarter, over the whole panel and not only the folds.

    The direct measurement for open point 3, which the project could only state as a worry: every
    economic number it holds comes from fifteen months in which the basket fell 51%, and both
    legs of this composite — low volatility and size — are known to be regime dependent. Fifteen
    quarters is not a distribution either, but it is the difference between "it worked once" and
    "it worked in the quarters the basket rose too".

    The long and short legs are reported apart, and against the *naked* return. Split against the
    hedged one they are the same number by construction — a symmetric book on returns that sum to
    zero inside each date earns exactly as much on each side — so the informative split is the one
    that still contains the market. A signal that earns on one side only is the trap this project
    already fell into once, with `remaining_excursion`, where 0.91 of the P&L came from the short
    leg over twelve months the basket spent falling.

    `test` marks the quarters no choice here was made on. The window comes off the train side of
    the folds, so everything up to 2025-06 is in sample for that one parameter and everything
    after it is not.
    """
    y, sig, close = label(p), composite(p, window), p["close"]
    forward = np.log(close.shift(-step) / close)
    hedged = forward.sub(forward.mean(axis=1), axis=0)
    clock = hourly(close.index)
    pct = sig[clock].rank(axis=1, pct=True)
    band = 2 * pct.sub(pct.mean(axis=1), axis=0)
    pos = pd.DataFrame(np.nan, index=band.index, columns=band.columns)
    pos[band <= -theta], pos[band >= theta] = -1.0, 1.0
    pos = pos.ffill().fillna(0.0)
    earned = pos * hedged[clock].reindex_like(pos)
    naked = pos * forward[clock].reindex_like(pos)
    basket = forward[clock].mean(axis=1)
    cut = pd.Timestamp(TEST_START, tz="UTC")

    rows = {}
    for q, when in pos.groupby(pos.index.to_period("Q")).groups.items():
        if len(when) < 200:
            continue
        span = (when.max() - when.min()) / YEAR
        ic = rank_ic(sig, y, np.isin(close.index, when))
        rows[str(q)] = {
            "test": when.min() >= cut,
            "basket": float(basket.loc[when].sum()) / span,
            "rank_ic": float(ic.mean()) if len(ic) else np.nan,
            "long": float(naked.loc[when].where(pos.loc[when] > 0).sum().sum() / pos.shape[1]) / span,
            "short": float(naked.loc[when].where(pos.loc[when] < 0).sum().sum() / pos.shape[1]) / span,
            "gross_hedged": float(earned.loc[when].mean(axis=1).sum()) / span,
        }
    return pd.DataFrame(rows).T


def _selfcheck() -> None:
    """A panel whose ranking is known, so the composite has a sign it cannot get wrong."""
    n = 4000
    when = pd.date_range("2025-01-01", periods=n, freq=BAR, tz="UTC")
    rng = np.random.default_rng(0)
    market = np.cumsum(rng.normal(0, 0.002, n))
    # Three symbols with different volatilities and different sizes; the quiet, large one is the
    # one the composite has to rank first, whatever the market does underneath.
    close = pd.DataFrame(
        {
            "quiet": np.exp(market + np.cumsum(rng.normal(0, 0.001, n))),
            "middle": np.exp(market + np.cumsum(rng.normal(0, 0.004, n))),
            "wild": np.exp(market + np.cumsum(rng.normal(0, 0.010, n))),
        },
        index=when,
    )
    p = {"close": close, "dollar_volume": pd.DataFrame({"quiet": 1e9, "middle": 1e7, "wild": 1e5}, index=when)}
    c = composite(p, 96).dropna()
    assert (c["quiet"] > c["middle"]).mean() > 0.95 and (c["middle"] > c["wild"]).mean() > 0.95, "the sign is fixed"
    assert np.allclose(c.mean(axis=1), 0, atol=1e-12), "a cross-sectional rank is centred on its date"
    # Monotone inside a date: scaling a whole timestamp cannot move a rank, which is what makes
    # the factor mean the same thing in a calm week and a violent one.
    assert np.allclose(cross_rank(close * 1e6), cross_rank(close))

    # The clock. `hourly` has to pick the bars that *close* on the hour, not the ones labelled on
    # it, or every price downstream is read fifteen minutes early.
    at = hourly(when)
    assert (when[at] + BAR).minute.unique().tolist() == [0]
    assert when[at][0].minute == 45 and at.sum() == n // 4

    # Purging. A train bar is kept only when its label has closed before the cut, so the last
    # `CROSS_HORIZON` bars before every boundary are dropped and no train row reads its own test.
    idx = pd.date_range("2025-01-01", periods=CROSS_HORIZON * 8, freq=BAR, tz="UTC")
    cut = idx[CROSS_HORIZON * 4]
    train, test = folds(idx, str(cut), 1)[0]
    assert not (idx[train] + CROSS_HORIZON * BAR >= cut).any(), "an unpurged train row survived"
    assert idx[test].min() >= cut and not (train & test).any()

    # The prediction lands on the 5m label that closes with the candle it was built from — the one
    # place a factor written here could hand `simulation` fifteen minutes of future.
    sig = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_product([[when[3]], ["a", "b"]]))
    assert predictions(sig).index.get_level_values(0)[0] == when[3] + PRED_SHIFT
    assert predictions(sig).index.get_level_values(0)[0] + BASE_BAR == when[3] + BAR

    # And the book: a perfect signal on a panel whose ranking never changes holds one position and
    # pays for it once, which is the property the whole window result rests on.
    y = pd.DataFrame(np.tile([1.0, 0.0, -1.0], (n, 1)), index=when, columns=close.columns)
    everywhere = np.ones(n, dtype=bool)
    out = price(y, close, everywhere, 0.5)
    # A ranking that never changes pays exactly one side per held symbol and nothing after that,
    # which is the whole reason a thirty-day window costs what a six-hour one cannot.
    assert np.isclose(out["turnover"], out["in_market"]), out
    churn = price(pd.DataFrame(rng.normal(size=(n, 3)), index=when, columns=close.columns), close, everywhere, 0.5)
    assert churn["turnover"] > 50 * out["turnover"], (churn, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), help=f"bars of {TF} to choose between")
    ap.add_argument("--price", action="store_true", help="also price the hedged book at each threshold")
    ap.add_argument("--baseline", action="store_true", help=f"also run at the project's N={BASELINE_WINDOW}")
    ap.add_argument("--by-quarter", action="store_true", help="break the signal down by calendar quarter")
    ap.add_argument("--save", type=Path, nargs="?", const=STORE / "pred-factor.parquet", help="write the predictions")
    args = ap.parse_args()

    _selfcheck()
    p = panel()
    print(
        f"{len(p['close']):,} bars x {p['close'].shape[1]} symbols on {TF}, "
        f"{p['close'].index[0]:%Y-%m-%d} to {p['close'].index[-1]:%Y-%m-%d}, "
        f"{args.folds} folds from {args.test_start}, windows {args.windows}\n"
    )
    out, pred, chosen = walk_forward(p, tuple(args.windows), args.test_start, args.folds)
    print("composite, window picked on each fold's train side — test Rank IC with a blocked error\n")
    print(out.round(4).to_string())
    if args.baseline:
        base, _, _ = walk_forward(p, (BASELINE_WINDOW,), args.test_start, args.folds)
        print(f"\nthe project's window, N={BASELINE_WINDOW}, on the same folds\n")
        print(base.round(4).to_string())
    if args.price:
        y = p["close"]
        print("\nhedged book, mean over the folds — log return per year per symbol\n")
        rows = []
        grids = (("N picked on train", tuple(args.windows)), (f"N={BASELINE_WINDOW}", (BASELINE_WINDOW,)))
        for theta in THETAS:
            for name, windows in grids:
                acc = []
                for train, test in folds(p["close"].index, args.test_start, args.folds):
                    acc.append(price(composite(p, pick(p, label(p), train, windows)), y, test, theta))
                rows.append({"signal": name, **pd.DataFrame(acc).mean().to_dict()})
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    if args.by_quarter:
        window = max(set(chosen), key=chosen.count)
        print(f"\nby calendar quarter, N={window}, theta=0.2 — log return per year, per leg\n")
        q = by_quarter(p, window)
        print(q.round(4).to_string())
        print(
            f"\nquarters with a positive hedged gross: {int((q.gross_hedged > 0).sum())} of {len(q)}; "
            f"with a rising basket: {int((q.basket > 0).sum())}, of which "
            f"{int(((q.basket > 0) & (q.gross_hedged > 0)).sum())} also positive"
        )
    if args.save:
        predictions(pred).to_frame().to_parquet(args.save)
        print(f"\n{len(pred):,} predictions written to {args.save}")


if __name__ == "__main__":
    main()
