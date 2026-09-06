# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline, not a trading system. It measures whether a recurrent net on multi-timeframe
crypto candles can predict which of 20 USDT pairs beats the basket. The deployed artefact is only
the Streamlit chart page; everything else runs by hand as a module.

`swing_dataset_schema.html` (Italian) is the spec and the lab notebook: closed decisions, measured
numbers, open points, and the table of what was tried and failed. **Read it before changing
anything about the label, the windows, or the protocol** — most "obvious" ideas are in it with the
number that killed them. Keep it current when a step lands; the git history reads as a sequence of
measurements, and commit subjects are written that way ("Four branches lose to one, on every fold").

## Commands

```bash
uv sync                              # installs dev group too (torch, lightgbm)
uv run pytest -q
uv run pytest tests/test_dataset.py::test_branches_never_read_an_unclosed_bar -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
```

The Streamlit page: `preview_start` with the `chart` config in `.claude/launch.json`, or
`uv run streamlit run src/tradingvision/app/chart.py`.

Pipeline modules, each a `python -m` entry point, in the order they depend on each other:

```bash
uv run python -m tradingvision.data.binance          # fill data/ first; everything reads it
uv run python -m tradingvision.oracle                # step 0: fixes EXTREMA_WINDOW
uv run python -m tradingvision.linear                # step 1: leakage alarm, expects ~0 Rank IC
uv run python -m tradingvision.gbm --horizon         # step 2: reference IC + builds data/step2.parquet
uv run python -m tradingvision.selection             # step 2: the 28 -> ~12 column cut
uv run python -m tradingvision.gru --seeds 5         # step 3/4: the model
uv run python -m tradingvision.simulation --pred data/pred-*.parquet   # what it is worth in money
```

`gru --save` writes `data/gru.pt`, which is what the chart page draws predictions from.

## Architecture

**The store** — `data/` holds one Parquet per (symbol, interval) from the Binance public dumps
(`data.binance`, no API key, 2017+). Alpaca (`data.candles`) is the live feed the chart page uses
and the venue whose fees every cost figure assumes; its history is too short for training. `data/`
is gitignored; runs are reproduced by re-fetching.

**One label, swappable in one place.** `dataset.build` writes the target into a column called
`target` and `dataset.relabel` / `relabel_cross` rewrite it afterwards, so `gru --label` switches
between `remaining_excursion`, the retrospective `swing_leg_target`, and the cross-sectional
`cross_sectional_return` without touching the pipeline. The spec's section 1 explains why the label
changed twice; numbers taken on different labels are not comparable.

**The alignment rule, which is the one thing that must never break.** Every frame is indexed by the
*open* time of its bar, so a bar labelled `b` on timeframe `tf` closes at `b + tf`. Branch columns
are placed on the 5m grid at `label + tf - 5m` and forward filled. One bar of anticipation on the
1h branch hands the model twelve 5m bars of future and inflates every metric downstream.
`tests/test_dataset.py` checks this by truncation.

**Purging, not embargo.** `split.temporal` drops from train every bar whose `next_pivot` reaches
past the cut — the distance is unbounded (max measured 754 bars), so a fixed embargo both leaks and
throws away clean bars. Applies to train/valid as much as train/test: an unpurged valid contaminates
early stopping. `split.walk_forward` repeats the cut for the four folds every comparison is made on.

**Anything that decides something is measured on train only** — feature selection, normalisation
statistics (`normalize` fits quantiles on train and applies them unchanged), thresholds. Measuring a
choice on the test slice is how a worthless column set gets promoted.

**Metrics.** `metrics.signal` gives the four qlib metrics cross-sectionally; Rank ICIR is what
decides a promotion. A single cross-section of 20 symbols has a standard error of ~0.24, so only the
average over thousands of dates means anything, and a comparison without a dispersion across folds
is not a comparison.

**Cached datasets carry their parameters.** `dataset.cached` writes a JSON stamp next to the Parquet
and refuses to load a file built with different arguments. Don't defeat it — delete the file or pass
another `--cache`.

**torch and lightgbm never meet in one process.** Each ships its own OpenMP runtime and importing
both aborts with `OMP: Error #15`. `gru` deliberately does not import `gbm`; the only place they
coexist is `tests/test_selfchecks.py`, which runs the GRU check in a subprocess.

## Conventions

Self-checks live at the bottom of each module as asserts under `if __name__ == "__main__"` (or a
`_selfcheck()` function when the `__main__` is the real run), not in a mirrored test file.
`tests/test_selfchecks.py` is what makes CI run them — add new modules to its list.

Module docstrings carry the reasoning: what the module measures, which numbers came out, and why an
alternative was rejected. That is where the project's memory lives, so keep them accurate rather
than short. Comments explain the decision, not the syntax.

Constants are measured, not assumed, and say so where they are defined — `EXTREMA_WINDOW = 24`
(`data.pivots`), `CLIP`/`SCALE` (`normalize`), `DELTA` (`gru`, in the units of the label, so it must
be re-measured when the label changes).
