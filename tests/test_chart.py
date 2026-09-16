"""The chart page is the only deployed artefact, and what it can draw is decided at import.

`gru` and `swing` import torch at module scope, so an install without it used to kill the whole app
on Render — candles, pivots and label included — instead of the two model panels alone. torch is a
runtime dependency now, and this checks the page still degrades rather than dies if it is missing,
that a *different* missing module is still an error, and that a checkpoint is found in `models/`,
which is the only directory the image carries one in.
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


class Module:
    """Stands in for `gru` or `swing`: `saved` reads nothing off them but `CHECKPOINT`."""

    def __init__(self, path):
        self.CHECKPOINT = path


def test_a_checkpoint_is_read_from_the_store_first_and_from_models_after(tmp_path, monkeypatch):
    from tradingvision.app import chart

    store, models = tmp_path / "data", tmp_path / "models"
    store.mkdir(), models.mkdir()
    monkeypatch.setattr(chart, "MODELS", models)
    module = Module(store / "gru.pt")

    assert chart.saved(None) is None
    assert chart.saved(module) is None, "neither directory has it"

    (models / "gru.pt").write_bytes(b"committed")
    assert chart.saved(module) == models / "gru.pt", "the image only ever has this one"

    (store / "gru.pt").write_bytes(b"just trained")
    assert chart.saved(module) == store / "gru.pt", "a fresh --save must not be shadowed"
