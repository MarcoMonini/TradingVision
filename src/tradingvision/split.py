"""One temporal cut, the same date for every symbol, with exact purging.

The leakage is in the labels, not in the bars: `swing_leg_target(i)` interpolates up to the next
pivot, so a bar just before the cut whose next pivot falls after it carries prices from the test
period. A fixed embargo cannot fix that — the distance to the next pivot is not bounded by the
extrema window (p95 124 bars, maximum measured 754), so 3x24 bars would cover about the 87th
percentile and still leak on the rest, while throwing away clean bars on the short legs.

So the rule is exact: drop from train every bar whose `next_pivot` reaches beyond the cut. It
removes the contaminated bars and no others. The same applies to every boundary of the protocol —
train/valid as much as train/test — because an unpurged valid contaminates early stopping.
"""

from __future__ import annotations

import pandas as pd


def temporal(df: pd.DataFrame, cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, test)` around `cut`, with train purged of the bars that see past it.

    `df` is indexed by (timestamp, symbol) and carries `next_pivot`, as built by `dataset.build`.
    """
    at = pd.Timestamp(cut, tz="UTC")
    when = df.index.get_level_values(0)
    train = df[(when < at) & (df.next_pivot < at)]
    return train, df[when >= at]


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
