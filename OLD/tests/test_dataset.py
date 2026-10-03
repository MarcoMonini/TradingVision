"""The one property the branch alignment has to hold: no bar of a higher timeframe is read before
it has closed. Checked by truncation — recomputing a branch on data that stops at `t` must give
the same row at `t` as computing it on the whole series. If the 1h branch anticipated by one bar,
the truncated run would have nothing to read there and the values would differ."""

import numpy as np
import pandas as pd
import pytest

from tradingvision.data.binance import OHLC, STORE
from tradingvision.dataset import BRANCHES, branch, relabel, relabel_cross, symbol_frame


def synthetic(bars: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=bars, freq="5min", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, bars))), index=idx)
    spread = close * rng.uniform(0, 0.004, bars)
    return pd.DataFrame(
        {
            "open": close.shift().fillna(close.iloc[0]),
            "high": np.maximum(close, close.shift().fillna(close.iloc[0])) + spread,
            "low": np.minimum(close, close.shift().fillna(close.iloc[0])) - spread,
            "close": close,
            "volume": rng.uniform(1, 10, bars),
            "quote_volume": rng.uniform(1, 10, bars),
            "trades": rng.integers(1, 100, bars),
        }
    )


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = tf.replace("m", "min") if tf.endswith("m") else tf
    return df.resample(rule).agg(OHLC).dropna(subset=["open"])


def test_branches_never_read_an_unclosed_bar():
    df5 = synthetic()
    full = {tf: branch(resample(df5, tf), tf, df5.index) for tf in BRANCHES}
    # Cut points chosen inside an hour, where a 1h branch that anticipated would show it.
    for t in df5.index[[2000, 2003, 2007, 2011, 2222]]:
        upto = df5.loc[:t]
        for tf in BRANCHES:
            live = branch(resample(upto, tf), tf, upto.index).loc[t]
            assert np.allclose(live, full[tf].loc[t], equal_nan=True), f"{tf} at {t} uses the future"


def test_a_branch_column_only_changes_when_its_bar_closes():
    df5 = synthetic()
    b = branch(resample(df5, "1h"), "1h", df5.index)
    col = b.close_position_in_window_1h.dropna()
    changed = col.diff().dropna().ne(0)
    minutes = changed[changed].index.minute.unique()
    assert list(minutes) == [55], f"the 1h branch should only step at :55 (closing :00), got {minutes}"


# The only test here that reads the downloaded store rather than the synthetic series above. The
# parquet files are gigabytes and gitignored, so CI has nothing to point it at; running it there
# would mean downloading the store on every push to check a property the synthetic tests cannot
# check anyway — that the real data produces a sane label.
@pytest.mark.skipif(not (STORE / "BTCUSDT-5m.parquet").exists(), reason="needs the downloaded store")
def test_symbol_frame_carries_the_purging_horizon():
    df = symbol_frame("BTC", start="2025-06")
    assert df.next_pivot.min() >= df.index.min(), "a bar's next pivot cannot precede it"
    # Volatility units, not price units: the typical leg is a few sigma of a 24-bar walk. The
    # tail is long (a leg can run for hundreds of bars) so this is a sanity bound, not a claim.
    assert df.target.abs().median() < 10, "the label should be O(1) sigma"
    assert not df.isna().any().any(), "warm-up and the undefined tail must be dropped"


@pytest.mark.skipif(not (STORE / "BTCUSDT-5m.parquet").exists(), reason="needs the downloaded store")
def test_relabelling_rows_reproduces_the_label_they_were_built_with():
    """`relabel` writes a different target on rows `dataset` already produced, which is only
    honest if it agrees with `dataset` on the label they share. One symbol is enough: the loop is
    per symbol, so a wrong slice or the wrong pivots would show here."""
    from tradingvision.data.target import remaining_excursion

    df = symbol_frame("BTC", start="2025-06").iloc[::240]
    rows = pd.MultiIndex.from_arrays([df.index, ["BTC"] * len(df)])
    assert np.allclose(relabel(rows, remaining_excursion), df.target)


@pytest.mark.skipif(not (STORE / "BTCUSDT-5m.parquet").exists(), reason="needs the downloaded store")
def test_cross_relabelling_carries_its_own_horizon_and_drops_what_it_cannot_reach():
    """The cross-sectional label is the one that reaches a fixed distance ahead, so it owns the
    purging column too — leaving the pivots' shorter reach in place would under-purge every fold.
    Taken on the tail of the store, which is where the rows it cannot label actually are."""
    from tradingvision.data.binance import load
    from tradingvision.data.target import CROSS_HORIZON, cross_sectional_return
    from tradingvision.dataset import BASE_TF, WARMUP

    syms = ["BTC", "ETH", "SOL", "XRP"]
    when = load("BTC", BASE_TF).index[-8000::240]
    rows = pd.MultiIndex.from_product([when, syms])
    df = pd.DataFrame({"row": np.arange(len(rows)), "next_pivot": rows.get_level_values(0)}, index=rows)
    out = relabel_cross(df)

    bars = CROSS_HORIZON * 3  # 15m bars of the constant, on the 5m grid the panel is built from
    reach = out.next_pivot - out.index.get_level_values(0)
    assert (reach == bars * pd.Timedelta("5min")).all() and reach.iloc[0] == pd.Timedelta("72h")
    # Positions into the tensor survive the drop: they were assigned before it and must not move.
    assert out.row.is_monotonic_increasing and set(out.row) <= set(df.row)
    # The last 72h of the store has no forward return, and those rows leave rather than raise.
    assert len(out) < len(df)
    last = out.index.get_level_values(0).max()
    assert when.max() - last >= pd.Timedelta("72h") - pd.Timedelta(BASE_TF)

    # And the values are the panel's own, not something recomputed one symbol at a time.
    since = df.index.get_level_values(0).min() - WARMUP
    panel = pd.DataFrame({s: load(s, BASE_TF).loc[since:].close for s in syms}).sort_index()
    y = cross_sectional_return(panel, bars)
    at = out.xs("ETH", level=1)
    assert np.allclose(at.target, y.ETH.reindex(at.index))
