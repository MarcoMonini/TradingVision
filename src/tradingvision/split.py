"""One temporal cut, the same date for every symbol, with exact purging.

The leakage is in the labels, not in the bars: `swing_leg_target(i)` interpolates up to the next
pivot, so a bar just before the cut whose next pivot falls after it carries prices from the test
period. A fixed embargo cannot fix that — the distance to the next pivot is not bounded by the
extrema window (p95 124 bars, maximum measured 754), so 3x24 bars would cover about the 87th
percentile and still leak on the rest, while throwing away clean bars on the short legs.

So the rule is exact: drop from train every bar whose `next_pivot` reaches beyond the cut. It
removes the contaminated bars and no others. The same applies to every boundary of the protocol —
train/valid as much as train/test — because an unpurged valid contaminates early stopping.

From step 2 on, one cut is not enough. A single split gives one IC with no error bar, and the
question every later step asks — does the GRU beat the GBM, or is the gap noise — cannot be
answered by two numbers without a dispersion. So `walk_forward` repeats the cut: train on
everything before it, test on the slice that follows, then move the cut forward and retrain on
the enlarged history. N folds, N estimates, mean and standard deviation.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def temporal(df: pd.DataFrame, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, test)` around `cut`, with train purged of the bars that see past it.

    `df` is indexed by (timestamp, symbol) and carries `next_pivot`, as built by `dataset.build`.
    """
    at = pd.Timestamp(cut, tz="UTC")
    when = df.index.get_level_values(0)
    train = df[(when < at) & (df.next_pivot < at)]
    return train, df[when >= at]


def temporal_fraction(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the last `fraction` of the time span — the train/valid cut inside a fold.

    Cut by calendar time and not by row count: the symbols are not equally sampled across the
    period (a late listing has no early bars), so a quantile of the rows would put a different
    date in front of each of them.
    """
    when = df.index.get_level_values(0)
    first, last = when.min(), when.max()
    return temporal(df, str(first + (last - first) * (1 - fraction)))


def cuts(df: pd.DataFrame, start: str, n: int) -> list[pd.Timestamp]:
    """The `n` fold boundaries: `start`, then evenly spaced up to the end of the data."""
    at, last = pd.Timestamp(start, tz="UTC"), df.index.get_level_values(0).max()
    if at >= last:
        raise ValueError(f"the test period starts at {at}, past the last bar {last}")
    return [at + (last - at) * i / n for i in range(n)]


def walk_forward(df: pd.DataFrame, start: str, n: int = 4) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """`(train, test)` for each of `n` expanding folds, the test slices covering `start`..end.

    Expanding and not rolling: with a short crypto history, throwing away the early years to keep
    the train window a fixed length costs more than the regime drift it would avoid. Each train
    side is purged against its own cut by `temporal`, so no fold sees past its boundary.
    """
    edges = cuts(df, start, n) + [None]
    for at, until in zip(edges, edges[1:]):
        train, test = temporal(df, str(at))
        if until is not None:
            test = test[test.index.get_level_values(0) < until]
        yield train, test


if __name__ == "__main__":
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"), ["A"]])
    # Every bar's leg closes two hours later, so the two bars before the cut are contaminated.
    df = pd.DataFrame({"next_pivot": idx.get_level_values(0) + pd.Timedelta(hours=2)}, index=idx)
    train, test = temporal(df, "2024-01-01 06:00")
    assert len(test) == 4
    assert len(train) == 4, "the 05:00 and 04:00 bars see past the cut"
    assert train.index.get_level_values(0).max() == pd.Timestamp("2024-01-01 03:00", tz="UTC")
    assert (train.next_pivot < pd.Timestamp("2024-01-01 06:00", tz="UTC")).all()
    print("ok — purged", 6 - len(train), "of 6 train bars")

    # Walk-forward: the folds tile the test period and never overlap, and every train side stops
    # before its own fold.
    idx = pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"), ["A"]])
    df = pd.DataFrame({"next_pivot": idx.get_level_values(0)}, index=idx)
    folds = list(walk_forward(df, "2024-01-03", n=3))
    assert len(folds) == 3
    assert sum(len(t) for _, t in folds) == 52, "the test slices tile 2024-01-03..end exactly"
    assert [len(tr) for tr, _ in folds] == [48, 65, 82], "train grows with each fold"
    for tr, te in folds:
        assert tr.index.get_level_values(0).max() < te.index.get_level_values(0).min()

    early, late = temporal_fraction(df, 0.2)
    assert len(late) == 20 and len(early) == 80
    try:
        cuts(df, "2030", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("a test period past the data has to raise")
    print("ok — 3 folds, train sizes", [len(tr) for tr, _ in folds])
