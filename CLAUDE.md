# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline, not a trading system. It works on one label, the retrospective
`swing_leg_target` (where each bar sits along the leg between two pivots), on models that predict
it causally, and on the trading rules that read that prediction. The deployed artefact is only the
Streamlit chart page; everything else runs by hand as a module.

**`OLD/` is an archive, not code.** Every predictive label the project tried, the first pipeline
(`dataset` → `gru` → `legsweep`) and the cross-sectional factor live there, frozen at the git tag
`archive-predictive`. `OLD/README.md` says why each one left and with which number. Never import
from `OLD/` and never edit it — it is outside ruff, black and pytest on purpose. To run something
there, check out the tag in a worktree.

`HANDOFF.md` is the state of the current branch: what was added, what has *not* been measured yet,
and the order the measurements go in. Read it before starting work here.

`swing_dataset_schema.html` (Italian) is the spec and the lab notebook: closed decisions, measured
numbers, open points, and the table of what was tried and failed. **Read it before changing
anything about the label, the windows, or the protocol** — most "obvious" ideas are in it with the
number that killed them. Keep it current when a step lands; the git history reads as a sequence of
measurements, and commit subjects are written that way ("Four branches lose to one, on every fold").

`strategy_study.html` (Italian) is the report of the trading-rule study on v2's predictions
(`strategy`, `detect`, HANDOFF §17), step by step with its tables and charts, ending on what is
still open. A copy of a page that was published while the study ran; edit it here now.

`false_alarms.html` (Italian) is in two parts. Part I is why cutting the detectors' false alarms
cannot pay while a filter reads only the price's past (P·W = (1−P)·L under optional stopping), and
what a second model has to be worth. Part II is the plan: the eight routes left, numbered in the
order they run — the residual against the market, the futures columns' IC at the pivots against
everywhere, the liquidity premium at v2's extremes, the flow shock, a volatility-managed slow base,
the weak fast signals as the timing of that base's trades, the decomposed taker flow, pivots as a map
of stop orders — each with why, what to look for, the criterion and what follows, then the
confirmation (2021-2025, paper trading with a test by betting). The Monte Carlo behind each number
and its code are in an appendix. HANDOFF §19 is the same plan with what came out, measured 2026-10-07/08:
no route passes its criterion; route 6's timing is the one lead. The modules are `detect`
(phase 0, routes 1-2), `events` (3-4), `timing` (5-6), `flow` (7), `stopmap` (8), `sequential`
(the betting test), each with `--period dev|holdout|2021`.

`swing_leg_pipeline.html` (Italian) explains the training pipeline of the `swing_leg_target` models.
Part I is a lesson for a reader new to the subject, one concept per chapter with charts on real
candles; Part II walks the pipeline stage by stage with the schema of every frame, tensor and
checkpoint. It was written when `gru --label swing` and `legsweep` trained the label too — those
stages are archived now and the document says so where it describes them; `swing` is the live one.
Its data and JS are generated, not hand-edited numbers, and it describes the code: when the code
changes, update it with it.

## Commands

```bash
uv sync                              # installs the dev group too (scipy, pytest, ruff, black); torch is a runtime dep
uv run pytest -q
uv run pytest tests/test_selfchecks.py::test_swing_selfcheck -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
```

The Streamlit page: `preview_start` with the `chart` config in `.claude/launch.json`, or
`uv run streamlit run src/tradingvision/app/chart.py`. A second page, local only and not deployed:
`uv run streamlit run src/tradingvision/app/decomposer.py`, descriptive statistics of one pair's candles
read from the store, which it never downloads. It grows by request: add a chart or a number to it only
when asked.

Pipeline modules, each a `python -m` entry point, in the order they depend on each other:

```bash
uv run python -m tradingvision.data.binance          # fill data/ first; everything reads it
uv run python -m tradingvision.data.futures          # funding, open interest, taker flow, book depth (BTC/ETH/SOL)
uv run python -m tradingvision.oracle                # step 0: fixes EXTREMA_WINDOW
uv run python -m tradingvision.swing --timeframe 4h --baseline        # step 7: the swing label's band, against one-column rules
uv run python -m tradingvision.swing --timeframe 15m --window 12 --smoothing 0.5 --inputs reduced --steps 48 --test-start 2025-06 --save data/swing-v2.pt  # Swing Leg Position v2
# The three rules below read `pred-swing-*.parquet` in the format the archived `gru` wrote, which
# nothing writes any more; reconnecting them to `swing`'s `pos-swing-*.parquet` is HANDOFF §16, item 1.
uv run python -m tradingvision.swingrule --pred data/pred-swing-*.parquet  # the long-only rule on the swing label
uv run python -m tradingvision.threshold --pred data/pred-swing-*.parquet --at 0.5  # the always-in flip rule
uv run python -m tradingvision.stops --pred data/pred-swing-*.parquet --at 0.5 --grid  # the same rule with exits
uv run python -m tradingvision.strategy --candidates --at 0.40  # the rule study on v2's predictions, ETH/BTC/SOL
uv run python -m tradingvision.detect --shiryaev 0.5 0.9 0.99 --split  # recognising v2's turns causally: what it earns
uv run python -m tradingvision.detect --null 0.5    # does telling true alarms from false read the market? (also --residual BTC)
uv run python -m tradingvision.zones                # every indicator's bands x N: chosen on 2017-2020, read on 2021-2025 and since
```

`swing --save` writes `data/swing.pt`, and v2 is `--save data/swing-v2.pt`, the only checkpoint the
page draws; the page reads the store first and `models/` after, which is the only directory a
checkpoint reaches the Render image in — `data/` is gitignored. v2 is drawn only on the timeframe
it was fitted on, and its checkbox pins the label's smoothing and window to the ones in its checkpoint.

## Architecture

**The store** — `data/` holds one Parquet per (symbol, interval) from the Binance public dumps
(`data.binance`, no API key, 2017+). Alpaca (`data.candles`) is the live feed the chart page uses;
its history is too short for training. Every cost figure is OKX's spot taker fee, `oracle.FEE` =
0.10% a side, the venue the rules would trade on; every number measured before 2026-10-04 was taken
at Alpaca's 0.25%, so reproduce one with `--fee 0.0025`. `data/` is gitignored; runs are reproduced
by re-fetching.

**Three universes, in `data.binance`.** `SYMBOLS` is the training universe (15 pairs, chosen on
liquidity and data quality, no meme coin) and every module's default. `TRADABLE` is its subset
Alpaca lists (13, without BNB and NEAR): a metric that means money is read on these alone, and they
are the chart page's list. `STUDY` is the 20 pairs every number measured before 2026-09-27 was
taken on — pass it as `--symbols` to reproduce one; the cache stamps refuse a file built on the
other list.

**One label.** `swing_leg_target`, in `data.target`. The spec's section 1 explains why the label
changed twice before landing here; numbers taken on different labels are not comparable, and the
predictive ones are archived with the labels they were taken on.

**Every frame is indexed by the *open* time of its bar**, so a bar labelled `b` on timeframe `tf`
closes at `b + tf`. `swing` reads one timeframe per model and never joins two; the multi-branch
alignment rule (`label + tf - 5m` on the 5m grid) and its truncation test went to `OLD/` with
`dataset`. Anything that joins two timeframes again has to bring both back. `features` returns
finite or NaN and never an infinity, because `dropna` does not see one.

**Purging, not embargo.** `swing.purge` drops from train every bar whose label reaches past the cut
— the distance is unbounded, so a fixed embargo both leaks and throws away clean bars. Applies to
train/valid as much as train/test: an unpurged valid contaminates early stopping. `swing.folds`
repeats the cut for the four folds every comparison is made on. The label reads further than its
next pivot: the pivot is only final once its same-kind run closes, `window` bars after the first
opposite extreme, so the reach is `legs.label_reach` (p50 46 bars ahead at window 12, against 13 to
the pivot). The archived first pipeline purged on the pivot, which is one reason it is archived.

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
used to train twice for that reason — a Huber on the label, then direct policy optimisation on the
net P&L with the fee inside the reward — and the second stage is archived (`OLD/README.md`, tag
`archive-policy`): on 4h it lost to one column, on v2 it converged on no trade, and no rule of
`strategy` or `detect` read it. `swing` now fits the label alone and picks a band on its output;
the money is the rules' job, priced on the prediction they read.

**Pivots have a causal twin.** `data.pivots.find_pivots` is centred and is the label's side of the
problem; `legs.confirmed` is the timeline a live reader would have held, with each turn dated from
its own confirmation and the merge of same-kind runs applied in arrival order. Never approximate
the second by shifting the first — the centred pass has already deleted the lower high a live
reader was acting on for the hours in between.

**Metrics.** A comparison without a dispersion across folds is not a comparison. Overlapping
labels inflate any naive t: adjacent rows share most of the future their labels read, so the error
has to be taken over non-overlapping blocks (`metrics.blocked`).

**Cached tensors carry their parameters.** `swing.cached` writes a JSON stamp next to the rows and
refuses to load a file built with different arguments. Don't defeat it — delete the file or pass
another `--cache`. The stamp records what is *not* an argument too (`BUILD`, the store's last bar
per symbol), because those change which rows exist without changing anything a caller passes.

**The page may not reach scipy, and pandas hides a path to it.** `Series.corr(method="spearman")`
imports scipy *lazily*, from inside `pandas.core.nanops`, so the call survives every import-time
check and every test run in a venv that has it. scipy is a dev dependency only, so the call works
everywhere except the one place that
matters — it took the chart page down in production on a caption. Use `metrics.spearman`, which is
Pearson on the ranks and asserted equal to pandas' own. `tests/test_deploy.py` enforces both halves:
the page's module graph is exercised with scipy made unimportable, and the call is banned by an AST
scan everywhere but `metrics`, which has to make it to prove the replacement equals it.

**PSAR is ported, not imported, and has to stay bit-equal.** `ta.trend.PSARIndicator` runs Wilder's
recursion with pandas scalar `.iloc` on both sides, which made it 9.4s of `features`' 10.8s — the
whole cost of the most-called function in the pipeline. `features._psar` is the same algorithm over
numpy arrays, 61x faster, and the module's self-check asserts it **equal** to `ta` rather than
close: the recursion is path dependent, so one bar of drift propagates to the end of the series and
every number measured on that column becomes a different number. `ta` stays the definition.

**If lightgbm ever comes back, it never meets torch in one process.** Each ships its own OpenMP
runtime and importing both aborts with `OMP: Error #15`. It left with `gbm` in the cleanup.

## Conventions

**A strategy that cannot beat one indicator is not a strategy.** `swing --baseline` prices four
one-column rules on exactly the rows the walk-forward tested. `rsi_centered` above 0.3 nets +0.116
log a year on 4h bars against +0.063 for the archived two-stage model at three times its
turnover, at Alpaca's 0.25%; keep that row in front of any claim about the model.

Self-checks live at the bottom of each module as asserts under `if __name__ == "__main__"` (or a
`_selfcheck()` function when the `__main__` is the real run), not in a mirrored test file.
`tests/test_selfchecks.py` is what makes CI run them — add new modules to its list.

Module docstrings carry the reasoning: what the module measures, which numbers came out, and why an
alternative was rejected. That is where the project's memory lives, so keep them accurate rather
than short. Comments explain the decision, not the syntax.

Constants are measured, not assumed, and say so where they are defined — `EXTREMA_WINDOW = 24`
(`data.pivots`), `CLIP`/`SCALE` (`normalize`), `SHARPNESS` (`swing`). The Huber delta is measured on
each fold's train side, in the units of the label.
