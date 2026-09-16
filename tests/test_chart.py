"""The chart page is the only deployed artefact, and the image it deploys to has no torch.

`gru` and `swing` import torch at module scope; the runtime dependencies deliberately leave it out
(see `pyproject.toml`). A plain `from tradingvision import gru` at the top of the page therefore
killed the whole app on Render — candles, pivots and label included — for two modules whose
checkpoints live under a `data/` store the image does not ship. This checks the page survives that
install, and that a *different* missing module is still an error rather than a silent None.
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
