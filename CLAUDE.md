# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline, not a trading system. It measures whether a recurrent net on multi-timeframe
crypto candles can predict which of 20 USDT pairs beats the basket. The deployed artefact is only
the Streamlit chart page; everything else runs by hand as a module.

`HANDOFF.md` is the state of the current branch: what was added, what has *not* been measured yet,
and the order the measurements go in. Read it before starting work here.

`swing_dataset_schema.html` (Italian) is the spec and the lab notebook: closed decisions, measured
numbers, open points, and the table of what was tried and failed. **Read it before changing
anything about the label, the windows, or the protocol** — most "obvious" ideas are in it with the
number that killed them. Keep it current when a step lands; the git history reads as a sequence of
measurements, and commit subjects are written that way ("Four branches lose to one, on every fold").

## Commands

```bash
uv sync                              # installs dev group too (lightgbm); torch is a runtime dep
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
uv run python -m tradingvision.factor --price --baseline --by-quarter # step 6: the cross-sectional factor
uv run python -m tradingvision.swing --timeframe 4h --baseline        # step 7: the tradable swing rule
uv run python -m tradingvision.swingrule --pred data/pred-swing-*.parquet  # the long-only rule on the swing label
uv run python -m tradingvision.threshold --pred data/pred-swing-*.parquet --at 0.5  # the always-in flip rule
uv run python -m tradingvision.stops --pred data/pred-swing-*.parquet --at 0.5 --grid  # the same rule with exits
uv run python -m tradingvision.legcheck  --pred data/pred-swing-*.parquet  # does the prediction lead, or only summarise?
uv run python -m tradingvision.legsweep --table                        # step 8: the 9x13 smoothing/leg-window grid
```

`gru --save` writes `data/gru.pt` and `swing --save` writes `data/swing.pt`; the page reads the store
first and `models/` after, which is the only directory a checkpoint reaches the Render image in —
`data/` is gitignored. On the retrospective label the page instead loads the cell of `legsweep`'s grid
its two sliders name (`gru-swing-s<smoothing>-w<window>.pt`), by the same two-directory rule, and
refuses to draw a model fitted on another cell. `legsweep.CURRENT` — 0.7 / 24 — is the exception:
`gru.pt` *is* that cell, at the full four folds, so it is the one the page draws there.

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

**Sampled on the clock, never by position.** `build` keeps the rows whose timestamp is on the
sampling grid (`index.floor(step) == index`). A positional `iloc[::stride]` over rows a per-symbol
`dropna` has thinned shifts that symbol's phase permanently at its first dropped row, and the
previous build shows what that costs: 31,111 of 64,393 timestamps carrying one symbol, no timestamp
holding all twenty, AVAX absent from every cross-section. Related: `features` returns finite or NaN
and never an infinity, because `dropna` does not see one.

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

**The oracle has two readings and only one of them is a target.** `oracle.run(..., lag=0)` buys
every pivot low with hindsight and is enormous — 4.1 log a year on 4h bars, 12.6 on 15m. The same
oracle at `lag=EXTREMA_WINDOW` is the earliest any reader can *know* a pivot of a centred window,
and it is 0.40 a year: **under 10% of the first**. Everything that makes the hindsight number huge
is the window of future it reads. Quote a causal strategy against the second (`swing` calls it
`reachable`) and mention the first only to say what it is. Trading `swing_leg_target` itself, known
perfectly, earns 54-61% of the hindsight oracle — so "half the oracle" is not an ambitious target,
it is the definition of knowing the label exactly.

**Correlation with the label is not the objective.** A ridge on the features plus the causal leg
state reaches 0.62 time-series correlation with `swing_leg_target` and still loses money, because
its residual concentrates at the turns, which is where every trade is opened and closed. `swing`
therefore trains twice: a Huber on the label to put leg structure in the encoder, then direct
policy optimisation on the net P&L with the fee inside the reward (`swing.fit_policy`). The reward
is paid on *detrended* returns — otherwise the best policy is buy and hold and the stage finds it.

**Pivots have a causal twin.** `data.pivots.find_pivots` is centred and is the label's side of the
problem; `legs.confirmed` is the timeline a live reader would have held, with each turn dated from
its own confirmation and the merge of same-kind runs applied in arrival order. Never approximate
the second by shifting the first — the centred pass has already deleted the lower high a live
reader was acting on for the hours in between.

**Metrics.** `metrics.signal` gives the four qlib metrics cross-sectionally. A single cross-section
of 20 symbols has a standard error of ~0.24, so only the average over thousands of dates means
anything, and a comparison without a dispersion across folds is not a comparison. On the 72h label
the raw Rank ICIR is *not* a significance — adjacent dates share 71 of the 72 hours their labels are
made of, so a naive t over 10,944 dates reads 31.9 where 153 non-overlapping blocks read 5.1. Pass
`horizon=` to `signal` and read `rank_ic_t`.

**Cached datasets carry their parameters.** `dataset.cached` writes a JSON stamp next to the Parquet
and refuses to load a file built with different arguments. Don't defeat it — delete the file or pass
another `--cache`.

**The page may not reach scipy, and pandas hides a path to it.** `Series.corr(method="spearman")`
imports scipy *lazily*, from inside `pandas.core.nanops`, so the call survives every import-time
check and every test run in a venv that has it. scipy reaches this project only as a transitive
dependency of lightgbm, which is dev-only, so the call works everywhere except the one place that
matters — it took the chart page down in production on a caption. Use `metrics.spearman`, which is
Pearson on the ranks and asserted equal to pandas' own. `tests/test_deploy.py` enforces both halves:
the page's module graph is exercised with scipy made unimportable, and the call is banned by an AST
scan everywhere but `metrics`, which has to make it to prove the replacement equals it.

**torch and lightgbm never meet in one process.** Each ships its own OpenMP runtime and importing
both aborts with `OMP: Error #15`. `gru` deliberately does not import `gbm`; the only place they
coexist is `tests/test_selfchecks.py`, which runs the GRU check in a subprocess.

## Conventions

**A strategy that cannot beat one indicator is not a strategy.** `swing --baseline` prices four
one-column rules on exactly the rows the walk-forward tested. `rsi_centered` above 0.3 nets +0.116
log a year on 4h bars against +0.063 for the two-stage model at three times its turnover; keep that
row in front of any claim about the model.

Self-checks live at the bottom of each module as asserts under `if __name__ == "__main__"` (or a
`_selfcheck()` function when the `__main__` is the real run), not in a mirrored test file.
`tests/test_selfchecks.py` is what makes CI run them — add new modules to its list.

Module docstrings carry the reasoning: what the module measures, which numbers came out, and why an
alternative was rejected. That is where the project's memory lives, so keep them accurate rather
than short. Comments explain the decision, not the syntax.

Constants are measured, not assumed, and say so where they are defined — `EXTREMA_WINDOW = 24`
(`data.pivots`), `CLIP`/`SCALE` (`normalize`), `DELTA` (`gru`, in the units of the label, so it must
be re-measured when the label changes).
