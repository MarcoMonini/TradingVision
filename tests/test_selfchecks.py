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

from tradingvision import (
    crosscheck,
    dataset,
    exhaustcheck,
    factor,
    gbm,
    legcheck,
    linear,
    nearpivot,
    selection,
    simulation,
    swingrule,
)

SELF_CHECKED = [
    "tradingvision.data.candles",
    "tradingvision.legs",
    "tradingvision.data.pivots",
    "tradingvision.data.target",
    "tradingvision.features",
    "tradingvision.normalize",
    "tradingvision.metrics",
    "tradingvision.split",
]


@pytest.mark.parametrize("module", SELF_CHECKED)
def test_module_selfcheck(module):
    runpy.run_module(module, run_name="__main__")


def test_dataset_selfcheck():
    """Same reason as `linear`: the module's `__main__` builds from the real store."""
    dataset._selfcheck()


def test_linear_selfcheck():
    """`linear` keeps its checks in a function because its `__main__` is the real run, which
    needs the dataset."""
    linear._selfcheck()


def test_gbm_selfcheck():
    gbm._selfcheck()


def test_gru_selfcheck():
    """In its own process, because torch and lightgbm each bring their own OpenMP runtime.

    lightgbm links `@rpath/libomp.dylib` and finds Homebrew's; torch loads the copy bundled in
    `torch/lib`. Two of them in one process is `OMP: Error #15` and an immediate abort, in either
    import order. No code path imports both — `gru` deliberately stays clear of `gbm` — so the
    only place they meet is this file, and a subprocess is cheaper than making the two share a
    runtime. It trains a few small models, which makes it the slow test here, about ten seconds.
    """
    code = "from tradingvision import gru; gru._selfcheck()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_nearpivot_selfcheck():
    nearpivot._selfcheck()


def test_selection_selfcheck():
    selection._selfcheck()


def test_crosscheck_selfcheck():
    """`crosscheck` keeps its checks in a function too: its `__main__` rebuilds both datasets."""
    crosscheck._selfcheck()


def test_exhaustcheck_selfcheck():
    """`exhaustcheck` keeps its checks in a function: its `__main__` builds the whole panel."""
    exhaustcheck._selfcheck()


def test_legcheck_selfcheck():
    """`legcheck` keeps its checks in a function: its `__main__` reads a real prediction file."""
    legcheck._selfcheck()


def test_swingrule_selfcheck():
    """`swingrule` keeps its checks in a function: its `__main__` prices a real prediction file."""
    swingrule._selfcheck()


def test_simulation_selfcheck():
    """`simulation` keeps its checks in a function: its `__main__` sweeps the real store."""
    simulation._selfcheck()


def test_factor_selfcheck():
    """`factor` keeps its checks in a function too: its `__main__` reads the whole panel."""
    factor._selfcheck()


def test_swing_selfcheck():
    """In its own process, for the reason `gru` is: it imports torch, and lightgbm is in here too.

    It trains two small models on a saw, which makes it the second slow test in this file.
    """
    code = "from tradingvision import swing; swing._selfcheck()"
    subprocess.run([sys.executable, "-c", code], check=True)
