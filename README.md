<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000C3D,100:0A4FD6&height=180&section=header&text=TradingVision&fontSize=48&fontColor=ffffff&desc=End-to-end%20ML%20research%20pipeline%20%7C%20PyTorch%20GRU%20%2B%20LightGBM%20on%20crypto%20candles&descSize=15&descAlignY=72" />

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=7CFFA0&center=true&vCenter=true&width=680&lines=35+modules%3A+ingestion+to+cost-aware+backtest;Leakage-resistant+protocol%3A+exact+purging%2C+4-fold+walk-forward;GRU+in+PyTorch+vs+LightGBM+baselines;29+candidate+features+reduced+by+measurement;Every+number+here+was+measured%2C+not+assumed" /></a>

[![CI](https://github.com/MarcoMonini/TradingVision/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoMonini/TradingVision/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=uv&logoColor=white" />
  <img src="https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black" />
  <img src="https://img.shields.io/badge/Black-000000?style=flat-square" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
</p>

</div>

## 🎯 What this is

**A research pipeline, not a trading system.** It measures whether a recurrent network on
multi-timeframe crypto candles can predict *which of 20 USDT pairs will beat the basket over the
next 72 hours*. Every step has to beat the previous one on out-of-sample signal metrics, on the
same four purged walk-forward folds — and several of them did not.

The lab notebook is [`swing_dataset_schema.html`](swing_dataset_schema.html) (Italian): closed
decisions, measured numbers, open points, and the table of what was tried and failed. The git
history reads as a sequence of measurements, and commit subjects are written that way — *"Four
branches lose to one, on every fold"*.

<div align="center">

| 20 USDT pairs | 14.9M 5m candles | 35 modules | 4 walk-forward folds | 5 seeds per fold |
|:---:|:---:|:---:|:---:|:---:|
| **2017 → 2026** | **Binance public dumps** | **ruff · black · pytest in CI** | **exact purging** | **mean ± std** |

</div>

## 🧱 The pipeline — 35 modules

28 pipeline modules and 7 test modules, each pipeline stage a `python -m` entry point, in the
order they depend on each other. Ingestion → labelling → feature engineering → training →
evaluation → cost-aware backtesting → the tradable rule and its exits.

```bash
uv run python -m tradingvision.data.binance        # ingestion: one Parquet per (symbol, interval)
uv run python -m tradingvision.oracle              # step 0: calibrates EXTREMA_WINDOW = 24
uv run python -m tradingvision.linear              # step 1: leakage alarm + lower bound
uv run python -m tradingvision.gbm --horizon       # step 2: LightGBM reference IC
uv run python -m tradingvision.selection           # step 2b: the 28 -> 12 column cut
uv run python -m tradingvision.gru --seeds 5       # steps 3/4: the recurrent model
uv run python -m tradingvision.simulation --pred data/pred-*.parquet   # what it is worth in money
uv run python -m tradingvision.factor --price --baseline --by-quarter  # step 6: the cross-sectional factor
uv run python -m tradingvision.swing --timeframe 4h --baseline         # step 7: the tradable swing rule
uv run python -m tradingvision.threshold --pred data/pred-swing-*.parquet --at 0.5   # the always-in rule
uv run python -m tradingvision.stops --pred data/pred-swing-*.parquet --at 0.5 --grid  # its exits
uv run python -m tradingvision.legcheck --pred data/pred-swing-*.parquet   # does it lead, or summarise?
uv run python -m tradingvision.legsweep --table                        # step 8: the 9x13 label grid
```

<details>
<summary><b>Module map</b></summary>

| Layer | Modules | What it does |
|---|---|---|
| **Ingestion** | `data.binance` · `data.candles` | Bulk OHLCV from `data.binance.vision` (static S3 ZIPs, no API key, 2017+). Alpaca is the live feed and the venue whose fees every cost figure assumes. |
| **Labelling** | `data.pivots` · `data.target` · `legs` · `reference` · `crosscheck` | Local extrema on Close, three interchangeable targets, the *causal* twin of a pivot (`legs.confirmed`, dated from its own confirmation), the scale that makes a leg comparable across symbols and regimes, and the experiment that retired a label. |
| **Features** | `features` · `normalize` | 29 causal candidates on every branch — the original 28, plus `log_dollar_volume`, the only one whose *level* is the information; robust scaling with clip, fitted on train only. |
| **Assembly** | `dataset` | One row per 5m bar, four timeframe branches aligned side by side, plus the purging horizon. |
| **Protocol** | `split` · `metrics` | Exact purging, expanding walk-forward, the four qlib signal metrics cross-sectionally. |
| **Models** | `linear` · `gbm` · `gru` · `swing` · `factor` | OLS floor, LightGBM baseline, GRU encoders + linear head, the two-stage swing model (Huber on the label, then direct policy optimisation on the net P&L), and the two-column composite that beats all of them. |
| **Selection** | `selection` · `nearpivot` · `legsweep` | Five-pass reduction, the per-column signal check near the pivot, and the 9×13 sweep of the label's own two knobs. |
| **Economics** | `oracle` · `threshold` · `stops` · `swingrule` · `simulation` | Hindsight ceiling and its causal twin, the always-in rule priced at a raw number, its exits (two barriers, three re-entry policies, a trailing option), the long-only rule with a rotation null, and the break-even skill table. |
| **Judges** | `legcheck` · `exhaustcheck` | Does the prediction lead the price or only summarise it, and the six exhaustion columns priced against forward return rather than against a label. |
| **App** | `app.chart` | Streamlit page: candles, pivots, label, features, the model, the rule with its fills and exits, and the cross-sectional heatmaps. Containerised, deployed from `models/`. |
| **Tests** | `tests/*` (7) | Invariants, `test_selfchecks.py` which runs every module's own asserts, and `test_deploy.py` which walks the page's import graph with scipy made unimportable. |

</details>

## 🛡️ Leakage-resistant validation

The leakage is not in the bars — no bar is shared between periods and every feature is causal.
It is in the **labels**, and each of these three rules was written against a measurement.

**Exact purging, not a fixed embargo.** A bar near the cut carries a label that interpolates to a
pivot beyond it. The distance to the next pivot is unbounded — p50 28 bars, p95 124, p99 202,
**maximum measured 754** — so a fixed embargo of 72 bars covers about the 87th percentile and still
leaks, while throwing away clean bars on the short legs. The rule drops exactly the contaminated
bars and no others, at **every** boundary: train→valid as much as train→test, because an unpurged
valid contaminates early stopping.

```python
train = train[train.next_pivot < cut_index]
```

**Four-fold expanding walk-forward.** One split gives one number with no error bar, and the
question every step asks — *does this beat the previous one, or is it noise?* — cannot be answered
by two numbers without a dispersion. Four expanding folds from 2025-06; the report is mean and
standard deviation across folds. Per-symbol splits are excluded: 20 correlated assets with BTC in
test while ETH is in train is cross-sectional leakage.

**Multi-seed runs.** 5 initialisations per fold in exploration, reported as mean ± std. The
dispersion between seeds is 0.0014 against 0.0145 between folds — which is itself the evidence
that capacity is not the constraint.

**Anything that decides something is measured on train only.** Feature selection, quantile
normalisation statistics, the Huber δ, thresholds. The alignment rule is enforced by test: every
frame is indexed by the *open* time of its bar, branch columns land on the 5m grid at
`label + tf − 5m`, and `tests/test_dataset.py` checks it by truncation. One bar of anticipation on
the 1h branch hands the model twelve 5m bars of future.

## 🧪 Feature selection — 28 candidates, measured down

Five passes, all on the training period only, on the flattened step-2 dataset (423,093 rows). The
cut ran on the 28 candidates of the time — `log_dollar_volume`, the 29th, was added later by the
cross-sectional label:

| # | Pass | Outcome |
|---|---|---|
| 1 | Sanity | 560 / 560 columns usable. A no-op — which is the answer, not a reason to skip it |
| 2 | Spearman per branch | 28 pairs at \|ρ\| ≥ 0.8; the four branches agree within ~0.01, so the redundancy belongs to the indicators, not the timeframe |
| 3 | Hierarchical clustering at 0.2 | 19 clusters, complete linkage so every in-cluster pair clears the threshold |
| 4 | Group permutation importance | 12 of 19 groups beat the measured noise floor; one group alone is worth 0.130 of a 0.149 baseline |
| 5 | Ablation | 0.1560 ± 0.0150 full set vs 0.1531 ± 0.0166 reduced — a tie, so the reduced set wins |

**And then the result was overturned, which is the point.** The 12 columns cost the GRU 0.005 of
Rank IC and 0.022 of ICIR: a selection measured on flattened trees does not transfer to a
recurrent model, and the two strongest predictors inside 48 bars of the pivot — `tsi_momentum`,
`rsi_centered` — were among the discarded. Rule that follows: **redo the selection per
architecture, or do not do it at all.** The feature set is a parameter (`--features`), not a
deletion.

On the cross-sectional label the ranking changed again — three families separate and then there is
a cliff — and `close_position_in_window`, step 2's most important feature by a factor of thirty,
scores **0.006**. `SELECTED` is now **two columns**, `realized_volatility` and
`log_dollar_volume`, and they are a factor pair rather than a feature set: reversal has real
univariate signal and is measured and then *dropped*, because adding it lowers the combined Rank
IC from 0.121 to 0.103. A selection is only ever valid for the target it was run against, so the
twelve above were redone rather than trimmed.

## 🤖 Models

| | Architecture | Input | Notes |
|---|---|---|---|
| **Baseline** | OLS on the last candle | 112 point-in-time columns | Not a model — a floor and a leakage alarm |
| **Reference** | LightGBM | 560 flattened columns (value at t, t−1, t−4, window mean and std) | Beats nets on tabular finance more often than not, trains on CPU, and reports the per-column importance the GRU cannot |
| **Model** | GRU per branch → concat → linear head | 4 × (B, 24, F) | H = 32, dropout 0.2, AdamW, Huber with δ measured on train, early stopping on **validation Rank IC, never on the loss** |
| **Tradable** | One GRU, two heads: label and policy | (B, 24, 65) — features + leg state at three scales + exhaustion | Trained twice — Huber on the label to put leg structure in the encoder, then **direct policy optimisation on the net P&L with the fee inside the reward**. The gradient is exact because the price is exogenous. The reward is paid on *detrended* returns, or the best policy is buy and hold and the stage finds it |
| **What actually wins** | `−rank(volatility) + rank(dollar volume)` | 2 columns, 30-day window | **Zero parameters.** Beats every fitted thing in the project on the cross-sectional label |

No output activation: `tanh` reached ±1 only asymptotically, so the gradient vanished exactly at
the pivots. GRU over LSTM: at 24 steps the LSTM's advantage does not exist, and the GRU has ~25%
fewer parameters at equal hidden state — which counts on a low signal-to-noise problem.

torch and lightgbm never meet in one process — each ships its own OpenMP runtime and importing
both aborts. The only place they coexist is a test that runs one in a subprocess.

## 📊 Evaluation — Rank IC, Rank ICIR, and the price of a trade

Signal metrics are computed **cross-sectionally**: per timestamp, correlate the predictions of the
symbols trading at that instant against their targets, then average over time (IC) and divide by
the dispersion (ICIR). With 20 symbols a single cross-section has a standard error of ≈0.24, so
individual values are noise and **Rank ICIR — stability, not strength — is what decides a
promotion.**

| Step | Model | Input | Rank IC | Rank ICIR | Outcome |
|---|---|---|---|---|---|
| 0 | Oracle | known extrema | — | — | fixes `EXTREMA_WINDOW = 24` and δ |
| 1 | Linear regression | last candle × 4 branches | 0.124 | 0.28 | lower bound |
| 2 | LightGBM | 560 flattened columns | 0.156 ± 0.015 | 0.578 ± 0.053 | reference |
| 3 | GRU, single branch, 12 features | (B, 24, 12) — 15m | 0.154 ± 0.015 | 0.567 ± 0.052 | ties, does not beat |
| 3 | **GRU, single branch, 28 features** | (B, 24, 28) — 15m | **0.1582 ± 0.0150** | **0.586 ± 0.054** | beats the GBM on all 4 folds |
| 4 | GRU, multi-branch, separate encoders | 4 × (B, 24, 28) | 0.1530 ± 0.0137 | 0.565 ± 0.044 | **loses, 4 folds out of 4** |
| 4 | GRU, multi-branch, shared encoder | + timeframe embedding | 0.1537 ± 0.0147 | 0.566 ± 0.050 | indistinguishable from the above |
| 5 | GRU, cross-sectional rank label | (B, 24, 2) — 15m | 0.106 | 0.305 | first number facing money |
| 6 | GRU, all 29 ranked columns | (B, 24, 29) — 15m | 0.1057 ± 0.0187 | — | **loses to an addition** |
| 6 | **Composite: −rank(vol) + rank(dollar volume)** | 2 columns, 30-day window, **zero parameters** | **0.1118 ± 0.0193** | — | the model of record |

> Step 5 changed the label. Numbers above and below that line are not comparable — which is
> stated here rather than quietly averaged.

**Step 6 is the project's own verdict on itself.** The recurrent net had thirty times the
information and got nothing out of it: half a standard error behind two ranked columns added
together, at twice the fold-to-fold dispersion (0.0241 vs 0.0124). Blending the two gives
0.1136 — +0.0018 on a standard error of 0.019, a tenth of a sigma. On its own two columns the
GRU reproduces the equal-weighted composite at **0.925 correlation**: seven thousand parameters,
twenty-four steps of sequence, AdamW and early stopping, to rediscover an addition.

The window was the larger half of it, and nobody had ever chosen it: `EXTREMA_WINDOW = 24` was
calibrated on the *leg* problem, and the label is a 72h cross-sectional return. Picked on the
train side of each fold it lands on **2880 bars (30 days)** every time — 0.1100 ± 0.012 against
0.0987 ± 0.018 at 24, four folds out of four. A six-hour volatility estimate is mostly
estimation noise, so its ranking churns 174 times a year and the book pays for the churn; the
thirty-day one trades **2.3** times a year and keeps the gross.

### Steps 7 and 8 — the tradable rule, on the other label

Numbers below are on `swing_leg_target` and do not convert to the ones above.

| Rule | Gross | Net @25bp | Trades/yr | Beats hold |
|---|---|---|---|---|
| Two-stage swing model (Huber, then policy gradient on net P&L) | 0.192 | **+0.063** | 25.7 | 12/20 |
| Supervised head, band and sign on validation | 0.095 | +0.060 | 6.8 | 11/20 |
| **`rsi_centered > 0.3` — one column, no model** | 0.172 | **+0.116** | 11.1 | 11/20 |
| Buy and hold | — | +0.057 | — | — |

A one-column momentum rule beats every model here at a third of the turnover, and buy and hold
sits inside the same range. `swing --baseline` prices those rows on exactly the bars the
walk-forward tested, so the comparison lives in the code rather than in somebody's memory of it.

**Step 8 swept the label's own two knobs** — the pivot window and the time weight, never tuned
against the model — across all 9 × 13 = 117 cells, one checkpoint each. The surface has **no
interior optimum**: Rank IC rises monotonically towards short windows and low time weights, to
0.4753 at 0.2/12 against 0.4083 at the current 0.7/24. Every cell also scores one raw
`rsi_centered` on the same label and the same rows, and that control runs the *other* way. At
the winning cell the RSI alone reads 0.4424 of the model's 0.4753 — **93% of the winner is an
indicator the model is not needed for**, against 92.7% at the current cell. The +0.067 of Rank
IC buys +0.003 of edge, which is three seeds. Read `--table edge`, never `rank_ic` alone.

### Cost-aware simulation

`simulation` prices a prediction that does not exist yet: it synthesises a signal of *known* skill
(`pred = ρ·y + √(1−ρ²)·noise`, with persistent rather than white noise) and runs it through the
same rule on the same prices. The gross scales with skill and the cost does not, so where the net
crosses zero is a ratio the run gets right.

**Rank IC at which the rule breaks even**, hedged accounting, 72h label:

| Threshold θ | 5 bp/side | 10 bp/side | 25 bp/side |
|---|---|---|---|
| 0.2 — extremes 40% | 0.091 | 0.155 | — |
| 0.5 — extremes 25% | — | 0.098 | 0.184 |
| 0.8 — extremes 10% | 0.000 | 0.000 | 0.062 |

The trained model, priced on 218,497 out-of-sample predictions at 25 bp/side: **+0.23 net hedged
per year, Sharpe 1.25** with a 12-bar low-pass on the output — against −0.05 unfiltered, because a
signal trained on a 72-hour label should move on a 72-hour scale and a raw GRU output moves every
row. Rank IC does not notice the filter (0.1064 → 0.1074), so what it removes is estimation noise
and the commission that noise was costing.

**And the composite, as a book rather than a correlation.** Weighting by the centred percentile
of the cross-section instead of thresholding its ends, with a no-trade tolerance so a graded
weight does not rebalance every hour for a few basis points — every parameter picked on the train
side of its own fold:

| Book | Window | Turnover | Gross | Net @25bp | Sharpe |
|---|---|---|---|---|---|
| Band | 24 (6h) | 1.24 | 0.086 | +0.076 | 0.69 |
| Band | train-picked | 1.18 | 0.169 | +0.160 | 1.14 |
| Weighted | 24 (6h) | 174.20 | 0.244 | **−1.142** | −6.53 |
| **Weighted** | **train-picked** | 2.03 | 0.254 | **+0.238** | **1.33** |

The third row is the whole study in one line: the same book on a six-hour window earns the same
gross and loses 1.14 a year instead of making 0.24. Nothing about the signal changed — only how
often a noise estimate made it trade.

## ❌ What was measured and did not work

Kept here because re-reading it costs less than re-measuring it.

- **Four branches lose to one, on every fold.** Step 4 was the premise of the whole project. The gate had already said why: `close_position_in_window` lives on the 15m branch alone, and permuting the other three costs 0.0027 out of 0.1525.
- **Shared weights are irrelevant, not a winner.** 0.1537 vs 0.1530 — a 0.0007 gap against 0.0140 between folds. The parametrisation was never the constraint.
- **The original label was beta, not knowledge.** `remaining_excursion` scored Rank IC 0.1582 on itself and **−0.033** against forward market-neutral return, with 0.91 of its P&L on the short side in twelve months when the basket fell 51%.
- **The retrospective label's Rank IC 0.4215 is not a trade.** A linear model on point-in-time features already reproduced it at 0.38 — if a linear model reproduces your label, the label is descriptive, not predictive.
- **Every model gets the direction wrong within 24 bars of the pivot.** A specialist trained only on that band, with all 28 columns, reaches −0.019 and still does not cross zero. It is not instability: per-column Rank IC on train and test correlate at 0.924. The missing thing is a variable telling the model which regime it is in.
- **Less machinery wins, monotonically**, on the cross-sectional label: LightGBM on 35 ranked features 0.104 → ridge on 4 ranked 0.113 → equal-weighted 2-column composite 0.121 → a single factor, zero model, **0.126**.
- **The always-in rule loses at every threshold, and its long leg loses at all of them.** Twenty pairs, fifteen months, 25 bp/side: at ±0.4 the net is −0.894 log a year, at ±0.5 it is −0.458 on a gross of +0.004 against 0.461 of commissions. The best point of the whole grid asks **3.52 bp per side**. All the gross is the short leg (+0.286) on a basket that fell 49% — and the Spearman between a pair's short-leg gross and its own buy-and-hold is **−0.734**. Exposure, not selection.
- **A win rate three sigma over the coin is not an edge.** 0.545 ± 0.013 on 1,577 trades, median trade positive, mean trade −0.0096: many small wins, few large losses. `swingrule.rotation_null` — the same exposure, the positions rotated between symbols — is what says the gross does not beat its own null.
- **Exhaustion features lead the price, and cost more to execute than they pay.** Four of the six hold their sign on all four folds and survive a full bar of delay, so the lead is real and is not the bid-ask bounce. The equal-weighted composite makes +0.0325 ± 0.0100 at 24 bars. Priced as a book, the best cadence asks **1.76 bp per side** against 25 taker and ~10 maker — **6× to 170× under**. Slowing the rebalance, the lever that saved the factor book, saves nothing here: the gross decays as fast as the fees.
- **Fitting the label harder is not the way to the money.** A ridge on the 29 features plus the causal leg state reaches 0.623 time-series correlation with `swing_leg_target` and still loses 0.29 a year, because the residual concentrates at the turns — which is where every trade is opened and closed. Hence the two-stage fit, with the fee inside the reward.
- **Seven extra factors, each with real univariate signal, all lose** when ridge-fitted on train alone (0.083–0.095 against 0.110). Even fitting the composite's two weights instead of adding them loses a fold out of four. With ~150 independent 72h blocks in three years, two parameters are already the ceiling.
- Target z-scored per timestamp (0.1475), demeaned per timestamp (0.1464), low-pass on the prediction (0.1284 at k=2, monotone in the wrong direction), training restricted to h < 48 (0.1334) — all measured, all worse.

## ⚙️ Engineering

- **CI on every push and PR** — `ruff check`, `black --check`, `pytest -q`, plus a Docker image build. Module self-checks live at the bottom of each file as asserts and are executed by `tests/test_selfchecks.py`, so they cannot silently rot.
- **Reproducible builds with uv** — `uv.lock` pins everything; torch resolves to the CPU index on Linux so CI does not pull 73 nvidia wheels for a job that never sees a GPU.
- **Containerised deployment** — multi-stage-style Docker layering (dependencies before sources), non-training runtime deps only, port injected at runtime for Render.
- **Cached datasets carry their parameters** — a JSON stamp next to the Parquet, and loading refuses a file built with different arguments. The stamp also records what is *not* an argument (the lag set, the sampling rule), because those change which rows exist without changing anything a caller passes.
- **The deployed image is guarded by test, not by memory** — the page died in production on a caption, because `Series.corr(method="spearman")` imports scipy *lazily* and scipy reaches this project only through lightgbm, which is dev-only. `test_deploy.py` walks the page's import graph (read off `chart.py` with the AST, so it updates itself) in a subprocess with scipy made unimportable, and bans the call everywhere but `metrics`, which has to make it to prove the replacement equals it.
- **Checkpoints ship in `models/`** — `data/` is gitignored and never copied into the image, so a model reaches Render only by being committed there. The page reads the store first and `models/` after, and says in the sidebar which of its conditions is unmet rather than drawing nothing.
- **Checkpoints are saved device-free** — a file written on `mps` names that device inside the pickle and cannot be unpickled where there is no Metal, which is the deploy host. Saved on CPU at the source, and `map_location` on the way in for the files written the other way.
- **Module docstrings carry the reasoning** — what the module measures, which numbers came out, why the alternative was rejected. That is where the project's memory lives.

```bash
uv sync                                          # torch is a runtime dep; the dev group adds lightgbm, scipy
uv run pytest -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
uv run streamlit run src/tradingvision/app/chart.py
docker build -t tradingvision .
```

## 🔭 Open points

Five were open. Four have since been measured, and the answers are in the spec's section 10 —
kept here with their resolutions, because a closed point is a result and deleting it would
throw the result away.

1. ~~**The gap between IC and commission.**~~ **Closed, and not by either lever it named.** The break-even is not a property of the signal, it is a property of the *(signal, rule)* pair, and half the gap was in the rule: thirty-day window plus a rank-weighted book with a no-trade tolerance nets **+0.238 a year at 25 bp, Sharpe 1.33** — at θ=0.2, the *widest* band, not the most selective. Two things that does not say: the gross is modest in absolute terms (0.254 per symbol per year), and the book holds a position on nearly all twenty pairs 97% of the time. The honest name for what was measured is a **tilt factor portfolio**, and it should be read with that class's limits.
2. ~~**No recurrent model measured against the composite.**~~ **Closed, three ways, and the composite wins all three.** See step 6 above.
3. ~~**One test period, one regime.**~~ **Much better.** Read quarter by quarter over all of 2023+, the Rank IC is positive in **all fifteen quarters** (0.028 to 0.158) and the hedged gross in **thirteen of fifteen**, including six of the seven quarters the basket *rose*. The honest residual: 90.9% of the 30-day composite's variance is a permanent per-symbol level, and that level is essentially the order of market capitalisation. The frozen allocation earns +0.060 of gross against +0.185 for the dynamic one — so the rotation, which is 9% of the variance, makes three quarters of the money.
4. ~~**The band near the pivot.**~~ **The road was travelled.** Exhaustion features do lead the price, and they are one to two orders of magnitude under the cost of executing them — see *what did not work* above. Not a problem of architecture, threshold or venue.
5. ~~**Sampling phase and zero-volume bars.**~~ **Fixed, and it was worse than the point described.** 31,111 of 64,393 timestamps carried a single symbol, no timestamp anywhere held all twenty, and AVAX was one bar out of phase for the entire period — the twentieth symbol was simply absent from every cross-section. Two one-line corrections: `build` samples on the *clock* (`index.floor(step) == index`), and `features` returns finite or NaN and never an infinity. The aggregate Rank IC barely moves (0.1108 ± 0.0203 against 0.1080 ± 0.0193); the breadth goes from **7.86 to 19.99**. The aggregate survived because the thin dates were being discarded; the book did not.

**What is actually open now:**

6. **`stops` is built and has never been run on the store.** Section 10 of the handoff showed that what loses money on the big moves is the *exit*, not the signal, and `stops` is the instrument that tests it — two barriers, three re-entry policies, a trailing option, an explicit intrabar tie-break. No barrier set reproduces `threshold` bit for bit, which is the control every other row has to beat. **No number has been taken.** Start from the fact that at ±0.5 the gross is +0.004 against 0.461 of commissions: an exit has to gain two orders of magnitude, not a few points.
7. **The label grid has no interior optimum, so it is asking the wrong judge.** Step 8's argmax is the cell where the label most resembles an RSI. If the grid is redone, it has to be redone against the *price* (`legcheck`, `swingrule`), not against correlation with the label.
8. **`STEPS = 24` has never been moved**, and it is the one untuned lever that changes *what* the model sees rather than how well it fits: a six-hour window does not reach the start of legs that run up to 754 bars.

---

<div align="center">

**Marco Monini** · [LinkedIn](https://www.linkedin.com/in/marco-monini/) · [marco.monini98@gmail.com](mailto:marco.monini98@gmail.com)

*Research code. Nothing here is investment advice. The project's own conclusion: the recurrent
models do not clear a 25 bp fee, and the one thing that does is two ranked columns added
together, rebalanced twice a year — a slow low-volatility/size tilt, not a trading strategy.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A4FD6,100:000C3D&height=100&section=footer" width="100%" />

</div>
