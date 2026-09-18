"""The chart page is the only deployed artefact, and what it can draw is decided at import.

`gru` and `swing` import torch at module scope, so an install without it used to kill the whole app
on Render — candles, pivots and label included — instead of the two model panels alone. torch is a
runtime dependency now, and this checks the page still degrades rather than dies if it is missing,
that a *different* missing module is still an error, and that a checkpoint is found in `models/`,
which is the only directory the image carries one in.

Every test here runs in a subprocess, and not only the ones that block an import. Importing the
page pulls in torch, `tests/test_selfchecks.py` trains a lightgbm model, and the two ship their own
OpenMP runtime: meeting in the pytest process aborts the whole run with `OMP: Error #15` — the rule
`CLAUDE.md` states and the reason the GRU self-check is a subprocess too. `saved` reads nothing off
torch, so those cases block it as well and let the page degrade to `gru is None`, which is the
lightest way to exercise the lookup without loading a tensor library to do it.
"""

import subprocess
import sys

BLOCK = """
import sys

BLOCKED = "%s"

class Block:
    def __init__(self, name):
        self.name = name

    def find_spec(self, name, path=None, target=None):
        if name == self.name or name.startswith(self.name + "."):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return None

sys.meta_path.insert(0, Block(BLOCKED))
"""


def run(blocked: str, body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", (BLOCK % blocked) + body],
        capture_output=True,
        text=True,
    )


def test_page_imports_without_torch():
    done = run(
        "torch",
        "\n".join(
            [
                "from tradingvision.app import chart",
                "assert chart.gru is None and chart.swing is None",
                "assert chart.factor is not None and chart.metrics is not None",
                "assert callable(chart.main)",
            ]
        ),
    )
    assert done.returncode == 0, done.stderr


def test_a_missing_module_that_is_not_torch_still_raises():
    # The fallback is a deployment decision about torch, not a blanket "import what you can".
    done = run("pandas", "from tradingvision.app import chart")
    assert done.returncode != 0
    assert "pandas" in done.stderr


LOOKUP = """
import tempfile
from pathlib import Path
from tradingvision.app import chart


class Module:
    "Stands in for `gru` or `swing`: `saved` reads nothing off them but `CHECKPOINT`."

    def __init__(self, path):
        self.CHECKPOINT = path


with tempfile.TemporaryDirectory() as d:
    store, models = Path(d) / "data", Path(d) / "models"
    store.mkdir(), models.mkdir()
    chart.MODELS = models
    module = Module(store / "gru.pt")
%s
"""


def lookup(body: str):
    done = run("torch", LOOKUP % body)
    assert done.returncode == 0, done.stderr


def test_a_checkpoint_is_read_from_the_store_first_and_from_models_after():
    lookup("""
    assert chart.saved(None) is None
    assert chart.saved(module) is None, "neither directory has it"

    (models / "gru.pt").write_bytes(b"committed")
    assert chart.saved(module) == models / "gru.pt", "the image only ever has this one"

    (store / "gru.pt").write_bytes(b"just trained")
    assert chart.saved(module) == store / "gru.pt", "a fresh --save must not be shadowed"
""")


def test_a_cell_of_the_sweep_is_found_by_the_same_rule_as_the_two_named_checkpoints():
    """`legsweep` writes 117 checkpoints into the same store, and the page picks one by filename.

    The name override must move the file and nothing else — same two directories, same order — or
    a deployed cell would be looked for somewhere the committed one never lands.
    """
    lookup("""
    cell = "gru-swing-s0.20-w12.pt"
    assert chart.saved(module, cell) is None

    (models / cell).write_bytes(b"committed cell")
    assert chart.saved(module, cell) == models / cell

    (store / cell).write_bytes(b"just swept")
    assert chart.saved(module, cell) == store / cell

    # The named checkpoint is untouched by a cell lookup, and a cell by the named one.
    (store / "gru.pt").write_bytes(b"the current cell")
    assert chart.saved(module) == store / "gru.pt"
""")
