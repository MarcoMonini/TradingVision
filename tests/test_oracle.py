"""Offline checks for the P&L oracle: python tests/test_oracle.py

The oracle is what fixed EXTREMA_WINDOW, so its arithmetic is worth pinning down on a series
whose legs can be counted by hand.
"""

import numpy as np
import pandas as pd

from tradingvision.data.pivots import find_pivots
from tradingvision.oracle import run


def triangle(period: int = 20, cycles: int = 10, low: float = 100.0, high: float = 110.0):
    """Sawtooth between `low` and `high`: a trough every `period` bars, a peak halfway between."""
    ramp = np.abs(np.arange(period * cycles) % period - period / 2) / (period / 2)
    idx = pd.date_range("2024-01-01", periods=period * cycles, freq="15min")
    return pd.Series(high - (high - low) * ramp, index=idx)


def test_legs_are_the_low_to_high_moves():
    """Every leg of the triangle runs 100 -> 110, so each trade must be exactly +10% gross."""
    s = triangle()
    r = run(s, window=5, fee=0.0)
    assert r["trades"] == 9, r["trades"]  # 10 cycles, the last peak has no following trough
    assert abs(r["gross_leg_pct"] - 10.0) < 1e-9, r["gross_leg_pct"]
    assert r["win_rate"] == 1.0


def test_fee_compounds_on_both_fills():
    """A 10% leg at 1% per side nets 1.10 * 0.99^2 - 1, not 1.10 - 1 - 0.02."""
    s = triangle()
    r = run(s, window=5, fee=0.01)
    expected = 1.10 * 0.99**2 - 1
    assert abs(np.expm1(r["log_per_year"] * _years(s) / r["trades"]) - expected) < 1e-9


def test_lag_shifts_both_fills_and_shrinks_the_leg():
    """Entering late and exiting late keeps only the middle of each leg."""
    s = triangle()
    # The ramp climbs 1.0 per bar, so k bars of lag buy at 100 + k and sell at 110 - k.
    for k in (0, 1, 2, 3):
        gross = run(s, window=5, fee=0.0, lag=k)["gross_leg_pct"]
        assert abs(gross - ((110 - k) / (100 + k) - 1) * 100) < 1e-9, (k, gross)


def test_lag_drops_trades_that_no_longer_cover_the_leg():
    """When the lagged entry lands past the exit pivot the trade misses the leg entirely."""
    # Spike series: a deep low immediately followed by a high one bar later.
    v = np.full(300, 100.0)
    v[100], v[101] = 90.0, 105.0
    v[200], v[201] = 90.0, 105.0
    s = pd.Series(v, index=pd.date_range("2024-01-01", periods=300, freq="15min"))
    piv = find_pivots(s, 5)
    assert (np.diff(s.index.get_indexer(piv.index)) == 1).any(), "fixture must hold adjacent pivots"

    assert run(s, window=5, fee=0.0, lag=0)["skipped"] == 0
    r = run(s, window=5, fee=0.0, lag=3)
    assert r["skipped"] == 2, r["skipped"]  # both spikes are one bar wide
    assert r["trades"] == 0, r["trades"]


def test_pivots_from_another_series_are_refused():
    """get_indexer answers -1 for a missing label, which would silently read the wrong bar."""
    s = triangle()
    foreign = find_pivots(triangle(period=20, cycles=10, low=1.0, high=2.0).shift(freq="1D").ffill(), 5)
    try:
        run(s, window=5, pivots=foreign)
    except ValueError as e:
        assert "do not belong" in str(e)
    else:
        raise AssertionError("mismatched pivots were accepted")


def _years(s):
    return (s.index[-1] - s.index[0]) / pd.Timedelta(365.25, "D")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
