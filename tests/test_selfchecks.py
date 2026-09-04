"""Runs the modules' `__main__` self-checks under pytest.

The project's convention is an assert-based self-check at the bottom of each module rather than a
mirrored test file. Nothing was running them: CI calls `pytest -q`, and pytest never executes a
`__main__` block — so the asserts on the label, the metrics and the purging were dead weight until
someone ran the module by hand.
"""

import runpy

import pytest

from tradingvision import dataset, gbm, linear

SELF_CHECKED = [
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
