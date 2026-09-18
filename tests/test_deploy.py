"""What the deployed image has, and what the chart page is allowed to need.

The page is the only artefact this project ships, and its dependency list is `pyproject`'s runtime
group — not what a developer's `.venv` happens to hold. The two differ by the dev group, and the
gap is not academic: **scipy is in this checkout only because lightgbm pulls it in**, lightgbm is
dev-only on purpose (`CLAUDE.md`: torch and lightgbm never meet in one process, and the image has
torch), and nothing declares scipy anywhere.

That gap ate the page once. `Series.corr(method="spearman")` imports scipy *lazily*, from inside
`pandas.core.nanops`, so the call is invisible to every import-time check and to every test run in
a venv that has it. The page imported fine, drew fine, and raised `ModuleNotFoundError: No module
named 'scipy'` on the caption under the chart, in production, on a line that had been there for
months.

Both tests here are about that class of bug and not about any one call, and each covers the other's
exemption. The first runs the page's own module graph with scipy made unimportable, which is the
deploy host in miniature; it is what watches `metrics`, the one module allowed to call pandas'
Spearman. The second bans that call everywhere else, because a lazy import can sit on a branch no
test walks and no runtime check can reach a branch nobody takes. `metrics.spearman` is the
replacement and is Pearson on the ranks, which is the same number — asserted equal, not close.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "tradingvision"
PAGE = SRC / "app" / "chart.py"
# `selection` is the one module allowed to import scipy: it clusters the feature correlation matrix
# with `scipy.cluster.hierarchy`, it is a by-hand pipeline step, and nothing the page imports
# touches it. It is named here rather than exempted by a rule, so adding a second one is a choice
# somebody makes on purpose.
MAY_IMPORT_SCIPY = {"selection.py"}
# And `metrics` is the one module allowed to *call* pandas' Spearman: its self-check compares the
# replacement against it and asserts they are equal, which cannot be done without calling it. That
# comparison is the reason every other module may stop calling it.
MAY_CALL_PANDAS_SPEARMAN = {"metrics.py"}


def page_imports() -> list[str]:
    """The `tradingvision` modules `chart.py` imports, read off its own source.

    Parsed rather than listed, so a new import on the page is covered the day it is added instead
    of the day somebody remembers this file.
    """
    tree = ast.parse(PAGE.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tradingvision"):
            if node.module == "tradingvision":
                out.update(f"tradingvision.{a.name}" for a in node.names)
            else:
                out.add(node.module)
    return sorted(out)


def test_the_page_imports_something():
    """A guard on the guard: a parse that silently found nothing would make the test below pass by
    doing nothing at all."""
    found = page_imports()
    assert len(found) > 8, found
    assert "tradingvision.metrics" in found and "tradingvision.stops" in found


def test_the_page_computes_with_no_scipy():
    """The deploy host, in a subprocess: scipy poisoned, then the page's numeric paths walked.

    `sys.modules["scipy"] = None` is what makes `import scipy` and `from scipy.stats import ...`
    raise, which is the state the image is really in. The paths exercised are the ones that carry a
    correlation or a rank — the captions and the metrics, where a lazy scipy import can hide.
    """
    modules = "\n".join(f"import {m}" for m in page_imports())
    code = f"""
import sys
sys.modules["scipy"] = None
try:
    import scipy  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("scipy is still importable; the poison did not take")

{modules}

import numpy as np, pandas as pd
from tradingvision import factor, metrics, stops, threshold

n = 400
when = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
rng = np.random.default_rng(0)
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=when)
df = pd.DataFrame({{
    "open": close.shift().fillna(close.iloc[0]),
    "high": close * 1.002, "low": close * 0.998, "close": close, "volume": 1.0,
}}, index=when)
pred = pd.Series(np.tanh(np.cumsum(rng.normal(0, 0.1, n))), index=when)
target = pred.shift(3)

# The caption that broke, and the NaN-tail shape it really has.
assert np.isfinite(metrics.spearman(pred, target))
assert np.isnan(metrics.spearman(pred.iloc[:1], target.iloc[:1]))

# The cross-sectional metrics, which are ranks all the way down.
idx = pd.MultiIndex.from_product([when[:100], list("abcde")])
a = pd.Series(rng.normal(size=len(idx)), index=idx)
b = a + rng.normal(0, 0.5, len(idx))
assert np.isfinite(metrics.signal(b, a)["rank_ic"])
assert np.isfinite(metrics.blocked(metrics.by_date(b, a, rank=True).dropna(), pd.Timedelta("1h"))["t"])
wide = b.unstack()
assert factor.cross_rank(wide).notna().any().any()

# The rule the page draws, with every exit on.
lifted, ohlc = threshold.on_one(pred), threshold.on_one(df[list(stops.OHLC)])
held, trades = stops.run(lifted, ohlc, 0.5, take=("fee", 3.0), stop=("atr", 2.0), trail=True)
assert np.isfinite(stops.price(held, trades)["net_per_year"])
assert np.isfinite(threshold.pnl(lifted, threshold.on_one(close), 0.5)["net_per_year"])

# The features, which the page draws under everything.
from tradingvision.features import COLUMNS, features
assert set(features(df).columns) >= set(COLUMNS)
print("ok")
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.strip().endswith("ok")


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_module_hides_scipy_behind_pandas(path: Path):
    """`method="spearman"` is banned as a *call*, and a bare scipy import outside `selection`.

    The ban is on the spelling and not on the behaviour, deliberately. A lazy import only fails on
    the branch that reaches it, so a test that exercises today's branches proves nothing about
    tomorrow's; the call is what can be checked everywhere at once. `metrics.spearman` is the
    replacement and returns pandas' own number — its self-check asserts equality, not closeness.

    Read off the syntax tree rather than out of the text, so the prose explaining the ban does not
    trip it. Half the point of a rule like this is the comment next to it saying why.
    """
    tree = ast.parse(path.read_text())
    if path.name not in MAY_CALL_PANDAS_SPEARMAN:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            spelled = any(
                k.arg == "method" and isinstance(k.value, ast.Constant) and k.value.value == "spearman"
                for k in node.keywords
            )
            assert not spelled, (
                f"{path.name}:{node.lineno} calls pandas' Spearman, which imports scipy lazily and "
                f"is not in the deployed image. Use metrics.spearman — Pearson on the ranks, same number."
            )
    if path.name not in MAY_IMPORT_SCIPY:
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            assert not any(n.split(".")[0] == "scipy" for n in names), f"{path.name} imports scipy"
