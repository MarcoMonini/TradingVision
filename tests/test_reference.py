"""Offline checks for the candidate denominators: python tests/test_reference.py

What these pin down is the one property the whole comparison rests on — a denominator has to
cancel the volatility of the symbol it is measured on. If it does not, the choice of
`riferimento[k]` is decided by which coins happen to be in the panel.
"""

import numpy as np
import pandas as pd

from tradingvision.reference import CANDIDATES, atr_pct, bar_sigma, leg_ratios


def walk(scale: float, n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """OHLC of a random walk whose per-bar volatility is `scale`. Same path shape at every scale,
    so ratios can be compared across two runs that differ only in volatility."""
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, scale, n))), index=pd.date_range("2024-01-01", periods=n, freq="15min")
    )
    span = close * scale
    return pd.DataFrame({"open": close, "high": close + span, "low": close - span, "close": close})


def test_volatility_cancels_out():
    """Triple the volatility and the raw amplitude triples; a good denominator does not move."""
    quiet, loud = (leg_ratios(walk(s)) for s in (0.002, 0.006))
    assert 2.5 < loud.raw.median() / quiet.raw.median() < 3.5, "the control must track volatility"
    for c in ("atr", "atr_24", "sigma_t", "legs"):
        shift = loud[c].median() / quiet[c].median()
        assert 0.9 < shift < 1.1, f"{c} moved by {shift:.2f} on a pure volatility change"


def test_price_level_is_irrelevant():
    """Every candidate is a ratio of two relative quantities, so a 10x price tag changes nothing."""
    base = walk(0.003)
    scaled = leg_ratios(base * 10)
    for c in CANDIDATES:
        assert np.allclose(leg_ratios(base)[c].dropna(), scaled[c].dropna()), c


def test_estimators_are_causal():
    """Truncating the future must not change a value already computed in the past."""
    df = walk(0.003)
    for f in (atr_pct, bar_sigma):
        full, cut = f(df), f(df.iloc[:5000])
        assert np.allclose(full.iloc[:5000].dropna(), cut.dropna()), f.__name__


def test_overshoot_is_zero_on_a_clean_leg():
    """A monotone leg never trades past its own pivots, so nothing overshoots."""
    n = 2000
    ramp = np.abs(np.arange(n) % 200 - 100) / 100
    close = pd.Series(100 + 10 * ramp, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    assert leg_ratios(df).overshoot.fillna(0).max() < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok — {name}")
