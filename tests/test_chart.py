"""The chart page is the only deployed artefact, and what it can draw is decided at import.

`swing` imports torch at module scope, so an install without it used to kill the whole app on
Render — candles, pivots and label included — instead of the model panels alone. torch is a runtime
dependency now, and this checks the page still degrades rather than dies if it is missing, that a
*different* missing module is still an error, and that a checkpoint is found in `models/`, which is
the only directory the image carries one in.

Every test here runs in a subprocess: a blocked import is installed on `sys.meta_path`, and done in
the pytest process it would block that module for every test that runs after. `saved` reads nothing
off torch, so the lookup cases block it as well and let the page degrade to `swing is None`, which
is the lightest way to exercise the lookup without loading a tensor library to do it.
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
                "assert chart.swing is None",
                "assert chart.metrics is not None and chart.stops is not None",
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
    "Stands in for `swing`: `saved` reads nothing off it but `CHECKPOINT`."

    def __init__(self, path):
        self.CHECKPOINT = path


with tempfile.TemporaryDirectory() as d:
    store, models = Path(d) / "data", Path(d) / "models"
    store.mkdir(), models.mkdir()
    chart.MODELS = models
    module = Module(store / "swing.pt")
%s
"""


def lookup(body: str):
    done = run("torch", LOOKUP % body)
    assert done.returncode == 0, done.stderr


def test_a_checkpoint_is_read_from_the_store_first_and_from_models_after():
    lookup("""
    assert chart.saved(None) is None
    assert chart.saved(module) is None, "neither directory has it"

    (models / "swing.pt").write_bytes(b"committed")
    assert chart.saved(module) == models / "swing.pt", "the image only ever has this one"

    (store / "swing.pt").write_bytes(b"just trained")
    assert chart.saved(module) == store / "swing.pt", "a fresh --save must not be shadowed"
""")


def test_v2_is_found_by_the_same_rule_as_the_default_checkpoint():
    """`swing-v2.pt` sits beside `swing.pt`, and the page picks it by filename.

    The name override must move the file and nothing else — same two directories, same order — or
    a deployed v2 would be looked for somewhere the committed one never lands.
    """
    lookup("""
    v2 = chart.V2_CHECKPOINT
    assert chart.saved(module, v2) is None

    (models / v2).write_bytes(b"committed v2")
    assert chart.saved(module, v2) == models / v2

    (store / v2).write_bytes(b"just trained")
    assert chart.saved(module, v2) == store / v2

    # The default checkpoint is untouched by a named lookup, and a named one by the default.
    (store / "swing.pt").write_bytes(b"the default checkpoint")
    assert chart.saved(module) == store / "swing.pt"
""")
