"""Runs the modules' `__main__` self-checks under pytest.

The project's convention is an assert-based self-check at the bottom of each module rather than a
mirrored test file. Nothing was running them: CI calls `pytest -q`, and pytest never executes a
`__main__` block — so the asserts on the label, the metrics and the purging were dead weight until
someone ran the module by hand.
"""

import runpy
import subprocess
import sys

import pytest

from tradingvision import detect, events, flow, sequential, stopmap, stops, strategy, swingrule, threshold, timing

SELF_CHECKED = [
    "tradingvision.data.candles",
    "tradingvision.legs",
    "tradingvision.data.pivots",
    "tradingvision.data.target",
    "tradingvision.features",
    "tradingvision.normalize",
    "tradingvision.metrics",
]


@pytest.mark.parametrize("module", SELF_CHECKED)
def test_module_selfcheck(module):
    runpy.run_module(module, run_name="__main__")


def test_swingrule_selfcheck():
    """`swingrule` keeps its checks in a function: its `__main__` prices a real prediction file."""
    swingrule._selfcheck()


def test_threshold_selfcheck():
    """`threshold` keeps its checks in a function for the same reason `swingrule` does, and was
    never registered here — the asserts on the always-in rule ran only when someone ran the
    module by hand. They are the ones the traded rule rests on, so they belong in CI."""
    threshold._selfcheck()


def test_stops_selfcheck():
    """`stops` keeps its checks in a function: its `__main__` reads a real prediction file and the
    5m store behind it. The checks themselves run on a saw and need neither."""
    stops._selfcheck()


def test_strategy_selfcheck():
    """`strategy` keeps its checks in a function: its `__main__` reads the v2 prediction file."""
    strategy._selfcheck()


def test_swing_selfcheck():
    """In its own process: it imports torch, which the rest of this file does not need.

    It trains two small models on a saw, which makes it the second slow test in this file.
    """
    code = "from tradingvision import swing; swing._selfcheck()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_detect_selfcheck():
    """`detect` keeps its checks in a function: its `__main__` runs the detectors on v2's predictions."""
    detect._selfcheck()


def test_futures_selfcheck():
    """`data.futures` keeps its checks in a function: its `__main__` downloads the dumps."""
    from tradingvision.data import futures

    futures._selfcheck()


def test_stopmap_selfcheck():
    """`stopmap` keeps its checks in a function: its `__main__` reads the 5m store of fifteen pairs."""
    stopmap._selfcheck()


def test_timing_selfcheck():
    """`timing` keeps its checks in a function: its `__main__` reads the 4h and 15m store."""
    timing._selfcheck()


def test_events_selfcheck():
    """`events` keeps its checks in a function: its `__main__` reads the store, v2 and the futures."""
    events._selfcheck()


def test_sequential_selfcheck():
    """`sequential` keeps its checks in a function: its `__main__` simulates for minutes or reads v2's trades."""
    sequential._selfcheck()


def test_flow_selfcheck():
    """`flow` keeps its checks in a function: its `__main__` reads the taker columns of the store."""
    flow._selfcheck()


def test_binance_selfcheck():
    """`data.binance` keeps its checks in a function: its `__main__` downloads the dumps."""
    from tradingvision.data import binance

    binance._selfcheck()
