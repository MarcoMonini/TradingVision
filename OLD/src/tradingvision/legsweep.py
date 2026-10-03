"""Step 8: the swing label has two free numbers — which grid of them the model reads best.

`swing_leg_target` is fixed except for two knobs, and neither was ever measured against the model:

    `window`     the extrema window of the pivots the leg runs between. 24 comes from `oracle`,
                 which scored it on a *lag-penalised hindsight P&L* — a question about market
                 structure, not about what a recurrent net can learn to rank.
    `smoothing`  the spec's `peso_tempo`, the weight of the time term in the blend. 0.7 is
                 documented in `data.target` as "a starting value, tunable", and nothing tuned it.

This module sweeps the 9 x 13 grid the user asked for and writes one checkpoint per cell, which
`app.chart` then picks up so the page draws the model that matches the label on its sliders.

**What changes between two cells, and what does not.** The features stay on `EXTREMA_WINDOW = 24`
for every cell, and the tensor `step3-15m-all.npy` is the same memory-mapped file throughout. Only
the target moves — this is `dataset.relabel`'s contract, spelled out there: recomputing the label
on the rows step 2 already chose keeps the sample identical, so the only difference between two
runs is the question being asked. Changing the feature window too would need step 2 and the tensor
rebuilt per window (16 + 4 minutes x 13) and would confound "a better label" with "better inputs".
`next_pivot` *is* recomputed per window, which `relabel` does not do: the legs are no longer the
same legs, so the purging horizon is no longer the same horizon, and an unpurged split at a longer
window leaks across the cut it exists to protect.

**Why the cells are not strictly comparable, and the column that says so.** A wider window makes a
lower-frequency label — fewer, longer legs, a target that moves less between adjacent hours — and
that is easier to rank whatever the model has learned. Rank IC therefore drifts up with `window`
for a reason that has nothing to do with signal. `base_rank_ic` is the control: `rsi_centered_15m`,
one raw indicator, scored on exactly the rows and the label of that cell. It costs one metric call
and it is the closest cheap answer to "how much of this cell's Rank IC is the label getting
easier". Read `edge = rank_ic - base_rank_ic` next to `rank_ic`, never `rank_ic` alone.

**What came out, on the whole 9 x 13 grid.** Both IC and Rank IC are maximised at the *same* cell
and the surface has no interior optimum at all — it rises monotonically towards short windows and
low time weights:

    cell                 IC       Rank IC   rsi_centered   edge
    0.7 / 24 (current)   0.4188   0.4083    0.3785         0.0298
    0.2 / 12 (best)      0.5033   0.4753    0.4424         0.0330
    0.9 / 60 (worst)     0.3439   0.3254    0.2232         0.1022

Over seeds 1-3 at window 12 the best cell holds: 0.4741 +/- 0.0006 Rank IC at smoothing 0.2,
0.4731 +/- 0.0009 at 0.1, 0.4708 +/- 0.0010 at 0.3. A seed is worth 0.001, so 0.2 is a real
winner over 0.3 and a coin flip against 0.1, and the choice is really "window 12, time weight at
or below 0.3".

**And the reading that matters more than the argmax.** `edge` goes the other way — monotonically
up towards long windows and high time weights, to 0.1022 at 0.9/60. The two surfaces are opposite
corners of the same grid, and neither has an interior maximum, which says the grid is not finding
a better *question* anywhere: it is trading how much of the label a single oscillator already
explains against how much of it is left for anybody. At the best cell one raw RSI reads 0.4424 of
the model's 0.4753 — **93% of the winner is an indicator the model is not needed for**, against
92.7% at the current cell. The +0.067 of Rank IC is worth +0.003 of edge, which is three seeds.
Moving the project to 0.2/12 on this evidence buys a bigger number and no more signal, and step 6
already measured what that distinction is worth: the prediction that reaches 0.4114 against this
label reaches **-0.0405 against the forward return**. Read `--table edge` before moving anything.

**Fast, and openly so.** One temporal split instead of the four walk-forward folds, one seed, a
4-hour sampling grid instead of the hourly one (a quarter of the rows, cross-sections kept whole —
the metric is cross-sectional, so thinning by timestamp and never by row is the only thinning that
leaves it intact), and a short epoch budget. That is ~40x cheaper than the protocol every number
in the spec was taken on, and the numbers here are *not* comparable with those. It is a ranking
device: the winner of this grid is what the full `gru --seeds 5` run then measures properly.
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tradingvision import dataset, gru, metrics, split
from tradingvision.data import binance
from tradingvision.data.pivots import find_pivots
from tradingvision.data.target import swing_leg_target

SMOOTHINGS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WINDOWS = (12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60)
# The cell the project is on today, and the reference every other cell is read against.
CURRENT = (0.7, 24)

STORE = gru.STORE
TENSOR = gru.TENSOR.with_stem(f"{gru.TENSOR.stem}-15m-all")
LABELS = STORE / "legsweep-labels"
RESULTS = STORE / "legsweep.csv"

TEST_START = "2025-06"
# Timestamps kept, of the hourly grid step 2 sampled. Thinning by *timestamp* keeps every
# cross-section whole; thinning by row would delete symbols from the panel the metric averages
# over, which is the mistake `dataset.build`'s docstring is written around.
GRID = "4h"
EPOCHS = 15
PATIENCE = 3
# One raw indicator, as the "how easy is this label" control. 15m because that is the branch the
# tensor carries, and `rsi_centered` because `swing --baseline` already prices it as the rule the
# two-stage model has to beat.
BASELINE = "rsi_centered_15m"


def checkpoint(smoothing: float, window: int, store: Path = STORE) -> Path:
    """Where the model for one cell lives. The name carries both knobs, because a checkpoint whose
    label parameters are not in its filename is a model nobody can match to a chart again."""
    return store / f"gru-swing-s{smoothing:.2f}-w{window:d}.pt"


@functools.lru_cache(maxsize=None)
def _close(symbol: str, tf: str) -> pd.Series:
    """`binance.load(...).close`, kept in memory. The sweep asks for the same twenty series once
    per window, and `load` resamples the 5m store on every call."""
    return binance.load(symbol, tf).close


def _pivots(symbol: str, idx: pd.DatetimeIndex, window: int) -> pd.DataFrame:
    """`dataset.pivots_on`, on the cached close and at an arbitrary window.

    Identical arithmetic, including the check that every pivot lands on a 5m bar — a gap there
    would silently move the next pivot of the preceding bars one leg further and mislabel them.
    """
    piv = find_pivots(_close(symbol, dataset.PIVOT_TF), window)
    piv.index = piv.index + dataset._shift(dataset.PIVOT_TF)
    piv = piv[(piv.index >= idx[0]) & (piv.index <= idx[-1])]
    missing = piv.index.difference(idx)
    if len(missing):
        raise ValueError(f"{len(missing)} pivots have no 5m bar, first at {missing[0]}")
    return piv


def labels(index: pd.MultiIndex, window: int, smoothings=SMOOTHINGS) -> pd.DataFrame:
    """One column per smoothing plus `next_pivot`, for one window, on the rows `index` names.

    The pivots are found once per symbol and shared by every smoothing: the blend weight moves
    where a bar sits *along* a leg and never which legs exist, so detecting them nine times would
    be nine times the work for the same frame.
    """
    since = index.get_level_values(0).min() - dataset.WARMUP
    pieces = []
    for symbol, rows in pd.Series(np.arange(len(index)), index=index).groupby(level=1):
        close = _close(symbol, dataset.BASE_TF).loc[since:]
        when = rows.index.get_level_values(0)
        piv = _pivots(symbol, close.index, window)
        nxt = pd.Series(piv.index, index=piv.index).reindex(close.index, method="bfill")
        cols = {"next_pivot": nxt.reindex(when).set_axis(rows.index)}
        for s in smoothings:
            y = swing_leg_target(close, piv, smoothing=s)
            cols[f"{s:.2f}"] = y.reindex(when).set_axis(rows.index)
        pieces.append(pd.DataFrame(cols))
    return pd.concat(pieces).reindex(index)


def cached_labels(window: int, smoothings=SMOOTHINGS, at: Path = LABELS, index=None) -> pd.DataFrame:
    """`labels` on disk. Half of a cell's wall clock is the label, and the nine smoothings of a
    window share the file, so a resumed or re-read sweep pays for it once.

    The index is checked and not only the columns, which is this file's version of the stamp
    `dataset.cached` and `gru.cached_sequences` both carry. `rows_of` masks `df` with
    `lab.iloc[:, 1]` *positionally*, so a label frame built against an older `step2.parquet` would
    not raise — it would line every row up against the wrong bar and relabel the whole grid. The
    index is the provenance here (`labels` ends on `reindex(index)`), so comparing it is the whole
    check and no JSON is needed beside the file.
    """
    at.mkdir(parents=True, exist_ok=True)
    path = at / f"w{window:d}.parquet"
    if path.exists():
        out = pd.read_parquet(path)
        if list(out.columns) == ["next_pivot"] + [f"{s:.2f}" for s in smoothings] and out.index.equals(index):
            return out
    out = labels(index, window, smoothings)
    out.to_parquet(path)
    return out


def rows_of(df: pd.DataFrame, lab: pd.DataFrame, test_start: str, grid: str):
    """`(inner, valid, test)` for one window — the same three row sets for all nine smoothings.

    Which rows survive does not depend on the blend: `swing_leg_target` is NaN before the first
    pivot, after the last one, and wherever the leg significance is, and none of the three is a
    function of `smoothing`. So the split is taken once per window, on the first column, and the
    target is written onto it afterwards. That is also what lets `Branches` — whose quantiles are
    fitted on the train rows — be built once per window instead of once per cell.
    """
    when = df.index.get_level_values(0)
    on_grid = (when.floor(grid) == when) & lab.iloc[:, 1].notna().to_numpy()
    g = df[on_grid].assign(next_pivot=lab.next_pivot[on_grid])
    train, test = split.temporal(g, test_start)
    inner, valid = split.temporal_fraction(train, gru.VALID_FRACTION)
    return inner, valid, test


def cell(
    z: gru.Branches,
    lab: pd.DataFrame,
    parts: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    smoothing: float,
    window: int,
    base: pd.Series,
    epochs: int = EPOCHS,
    seed: int = 0,
) -> dict:
    """Fit one cell, score it on the test slice and write its checkpoint."""
    y = lab[f"{smoothing:.2f}"]
    inner, valid, test = (p.assign(target=y.loc[p.index].to_numpy()) for p in parts)
    # Huber's delta lives in the units of the label, so it is measured per cell and never carried
    # over — the same criterion `gru.main` applies whenever `--label` moves it. Train only.
    delta = float(inner.target.abs().median())
    model = gru.fit(z, inner, valid, seed=seed, epochs=epochs, patience=PATIENCE, delta=delta)
    pred = gru.predict(model, z, test.row.to_numpy(), test.index)
    out = metrics.signal(pred, test.target)
    out |= {"base_rank_ic": metrics.signal(base.loc[test.index], test.target)["rank_ic"]}
    gru.save(checkpoint(smoothing, window), model, z, ["15m"], gru.FEATURES["all"], "swing", False)
    return {"window": window, "smoothing": smoothing, "delta": delta, "rows": len(inner), **out}


def sweep(
    windows=WINDOWS,
    smoothings=SMOOTHINGS,
    test_start: str = TEST_START,
    grid: str = GRID,
    epochs: int = EPOCHS,
    out: Path = RESULTS,
) -> pd.DataFrame:
    """The whole grid, one row per cell, appended to `out` as it goes so a stop loses one cell."""
    df = gru.meta()
    base = pd.read_parquet(gru.CACHE, columns=[BASELINE])[BASELINE]
    x = [gru.cached_sequences(TENSOR, df.index, keep=gru.FEATURES["all"], tf="15m", steps=gru.STEPS)]
    done = pd.read_csv(out) if out.exists() else None
    have = set() if done is None else set(zip(done.window.astype(int), done.smoothing.astype(float).round(2)))

    for w in windows:
        todo = [s for s in smoothings if (w, round(s, 2)) not in have or not checkpoint(s, w).exists()]
        if not todo:
            continue
        t0 = time.time()
        lab = cached_labels(w, smoothings, index=df.index)
        parts = rows_of(df, lab, test_start, grid)
        z = gru.Branches(x, parts[0].row.to_numpy())
        print(
            f"window {w}: {len(parts[0]):,} train / {len(parts[1]):,} valid / {len(parts[2]):,} test rows, "
            f"labels + scaler in {time.time() - t0:.0f}s"
        )
        for s in todo:
            t1 = time.time()
            row = cell(z, lab, parts, s, w, base, epochs=epochs)
            row["seconds"] = round(time.time() - t1, 1)
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(
                f"  s={s:.1f} w={w:<2d}  ic {row['ic']:+.4f}  rank_ic {row['rank_ic']:+.4f}  "
                f"base {row['base_rank_ic']:+.4f}  edge {row['rank_ic'] - row['base_rank_ic']:+.4f}  "
                f"[{row['seconds']:.0f}s]"
            )
    return pd.read_csv(out)


def table(out: Path = RESULTS, value: str = "rank_ic") -> pd.DataFrame:
    """The grid as a matrix, smoothing down the rows and window across.

    `value="edge"` is the synthetic column, `rank_ic - base_rank_ic`: the part of a cell that is
    not the label becoming easier to rank. It is the one to read for a choice — see the module
    docstring — and `rank_ic` is what the comparison with the project's existing numbers is on.
    """
    df = pd.read_csv(out).drop_duplicates(["window", "smoothing"], keep="last")
    df = df.assign(edge=df.rank_ic - df.base_rank_ic)
    return df.pivot(index="smoothing", columns="window", values=value)


def _selfcheck() -> None:
    """The two things a wrong sweep would get wrong silently: the label's dependence on the knobs,
    and the promise that the row sets do not depend on the smoothing."""
    n = 600
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024", periods=n, freq="15min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=idx)

    # A wider window finds strictly fewer pivots: it is a stronger condition on the same bars.
    counts = [len(find_pivots(close, w)) for w in (8, 16, 32)]
    assert counts == sorted(counts, reverse=True), counts

    piv = find_pivots(close, 8)
    a = swing_leg_target(close, piv, smoothing=0.1)
    b = swing_leg_target(close, piv, smoothing=0.9)
    # Same legs, same defined rows — only where the bar sits along them moves. This is the
    # assumption `rows_of` rests on: one split per window, reused by all nine smoothings.
    assert a.notna().equals(b.notna()), "the smoothing moved which rows carry a label"
    assert not np.allclose(a.dropna(), b.dropna()), "the smoothing moved nothing at all"
    # And the pivots themselves are untouched by the blend: they are the ends of the ramp.
    assert np.allclose(a.loc[piv.index], b.loc[piv.index], equal_nan=True)

    # The filename is the parameters, both of them, and round-trips through a float format.
    assert checkpoint(0.7, 24, Path("d")).name == "gru-swing-s0.70-w24.pt"
    assert checkpoint(0.1, 60, Path("d")) != checkpoint(0.1, 6, Path("d"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, nargs="*", default=list(WINDOWS))
    ap.add_argument("--smoothings", type=float, nargs="*", default=list(SMOOTHINGS))
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--grid", default=GRID, help="sampling of the hourly rows; 4h keeps a quarter")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--table", action="store_true", help="print the grid already measured and stop")
    args = ap.parse_args()

    _selfcheck()
    if not args.table:
        sweep(args.windows, args.smoothings, args.test_start, args.grid, args.epochs, args.out)
    for value, what in (
        ("rank_ic", "Rank IC"),
        ("ic", "IC"),
        ("edge", "Rank IC over `rsi_centered` on the same label"),
    ):
        print(f"\n{what}, smoothing down / window across\n")
        print(table(args.out, value).round(4).to_string())
