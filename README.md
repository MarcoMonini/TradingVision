<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000C3D,100:0A4FD6&height=180&section=header&text=TradingVision&fontSize=48&fontColor=ffffff&desc=End-to-end%20ML%20research%20pipeline%20%7C%20PyTorch%20GRU%20%2B%20LightGBM%20on%20crypto%20candles&descSize=15&descAlignY=72" />

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=7CFFA0&center=true&vCenter=true&width=680&lines=27+modules%3A+ingestion+to+cost-aware+backtest;Leakage-resistant+protocol%3A+exact+purging%2C+4-fold+walk-forward;GRU+in+PyTorch+vs+LightGBM+baselines;28+candidate+features+reduced+by+measurement;Every+number+here+was+measured%2C+not+assumed" /></a>

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

| 20 USDT pairs | 14.9M 5m candles | 27 modules | 4 walk-forward folds | 5 seeds per fold |
|:---:|:---:|:---:|:---:|:---:|
| **2017 → 2026** | **Binance public dumps** | **ruff · black · pytest in CI** | **exact purging** | **mean ± std** |

</div>

## 🧱 The pipeline — 27 modules

22 pipeline modules and 5 test modules, each pipeline stage a `python -m` entry point, in the
order they depend on each other. Ingestion → labelling → feature engineering → training →
evaluation → cost-aware backtesting.

```bash
uv run python -m tradingvision.data.binance        # ingestion: one Parquet per (symbol, interval)
uv run python -m tradingvision.oracle              # step 0: calibrates EXTREMA_WINDOW = 24
uv run python -m tradingvision.linear              # step 1: leakage alarm + lower bound
uv run python -m tradingvision.gbm --horizon       # step 2: LightGBM reference IC
uv run python -m tradingvision.selection           # step 2b: the 28 -> 12 column cut
uv run python -m tradingvision.gru --seeds 5       # steps 3/4: the recurrent model
uv run python -m tradingvision.simulation --pred data/pred-*.parquet   # what it is worth in money
```

<details>
<summary><b>Module map</b></summary>

| Layer | Modules | What it does |
|---|---|---|
| **Ingestion** | `data.binance` · `data.candles` | Bulk OHLCV from `data.binance.vision` (static S3 ZIPs, no API key, 2017+). Alpaca is the live feed and the venue whose fees every cost figure assumes. |
| **Labelling** | `data.pivots` · `data.target` · `reference` · `crosscheck` | Local extrema on Close, three interchangeable targets, the scale that makes a leg comparable across symbols and regimes, and the experiment that retired a label. |
| **Features** | `features` · `normalize` | 29 causal candidates on every branch (the spec's 28, plus `log_dollar_volume`); robust scaling with clip, fitted on train only. |
| **Assembly** | `dataset` | One row per 5m bar, four timeframe branches aligned side by side, plus the purging horizon. |
| **Protocol** | `split` · `metrics` | Exact purging, expanding walk-forward, the four qlib signal metrics cross-sectionally. |
| **Models** | `linear` · `gbm` · `gru` | OLS floor, LightGBM baseline, GRU encoders + linear head. |
| **Selection** | `selection` · `nearpivot` | Five-pass reduction and the per-column signal check near the pivot. |
| **Economics** | `oracle` · `threshold` · `simulation` | Hindsight ceiling, the threshold rule priced in money, and the break-even skill table. |
| **App** | `app.chart` | Streamlit page: candles, pivots, label and features, containerised and deployable. |
| **Tests** | `tests/*` (5) | Invariants, plus `test_selfchecks.py`, which runs every module's own asserts. |

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
scores **0.006**.

## 🤖 Models

| | Architecture | Input | Notes |
|---|---|---|---|
| **Baseline** | OLS on the last candle | 112 point-in-time columns | Not a model — a floor and a leakage alarm |
| **Reference** | LightGBM | 560 flattened columns (value at t, t−1, t−4, window mean and std) | Beats nets on tabular finance more often than not, trains on CPU, and reports the per-column importance the GRU cannot |
| **Model** | GRU per branch → concat → linear head | 4 × (B, 24, F) | H = 32, dropout 0.2, AdamW, Huber with δ measured on train, early stopping on **validation Rank IC, never on the loss** |

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

> Step 5 changed the label. Numbers above and below that line are not comparable — which is
> stated here rather than quietly averaged.

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

## ❌ What was measured and did not work

Kept here because re-reading it costs less than re-measuring it.

- **Four branches lose to one, on every fold.** Step 4 was the premise of the whole project. The gate had already said why: `close_position_in_window` lives on the 15m branch alone, and permuting the other three costs 0.0027 out of 0.1525.
- **Shared weights are irrelevant, not a winner.** 0.1537 vs 0.1530 — a 0.0007 gap against 0.0140 between folds. The parametrisation was never the constraint.
- **The original label was beta, not knowledge.** `remaining_excursion` scored Rank IC 0.1582 on itself and **−0.033** against forward market-neutral return, with 0.91 of its P&L on the short side in twelve months when the basket fell 51%.
- **The retrospective label's Rank IC 0.4215 is not a trade.** A linear model on point-in-time features already reproduced it at 0.38 — if a linear model reproduces your label, the label is descriptive, not predictive.
- **Every model gets the direction wrong within 24 bars of the pivot.** A specialist trained only on that band, with all 28 columns, reaches −0.019 and still does not cross zero. It is not instability: per-column Rank IC on train and test correlate at 0.924. The missing thing is a variable telling the model which regime it is in.
- **Less machinery wins, monotonically**, on the cross-sectional label: LightGBM on 35 ranked features 0.104 → ridge on 4 ranked 0.113 → equal-weighted 2-column composite 0.121 → a single factor, zero model, **0.126**.
- Target z-scored per timestamp (0.1475), demeaned per timestamp (0.1464), low-pass on the prediction (0.1284 at k=2, monotone in the wrong direction), training restricted to h < 48 (0.1334) — all measured, all worse.

## ⚙️ Engineering

- **CI on every push and PR** — `ruff check`, `black --check`, `pytest -q`, plus a Docker image build. Module self-checks live at the bottom of each file as asserts and are executed by `tests/test_selfchecks.py`, so they cannot silently rot.
- **Reproducible builds with uv** — `uv.lock` pins everything; torch resolves to the CPU index on Linux so CI does not pull 73 nvidia wheels for a job that never sees a GPU.
- **Containerised deployment** — multi-stage-style Docker layering (dependencies before sources), non-training runtime deps only, port injected at runtime for Render.
- **Cached datasets carry their parameters** — a JSON stamp next to the Parquet, and loading refuses a file built with different arguments.
- **Module docstrings carry the reasoning** — what the module measures, which numbers came out, why the alternative was rejected. That is where the project's memory lives.

```bash
uv sync                                          # dev group included: torch, lightgbm
uv run pytest -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
uv run streamlit run src/tradingvision/app/chart.py
docker build -t tradingvision .
```

## 🔭 Open points

1. **The gap between IC and commission** — the main problem. Measured Rank IC 0.106 against a break-even of 0.062 at θ=0.8 and 0.184 at θ=0.5. You only clear it by staying very selective, and at that selectivity the book makes fewer than one trade per symbol per year.
2. **No recurrent model has been measured against the new composite.** Every piece of machinery removed improves the result — the signature of a low signal-to-noise problem with ~150 independent 72h blocks in three years.
3. **One test period, one regime.** All economic numbers come from fifteen months in which the basket fell 51%.
4. **The band near the pivot** — exhaustion features (divergence, volume climax, wick asymmetry) instead of momentum is the road not yet taken.
5. **Sampling phase and zero-volume bars** — positional stride sampling permanently desynchronises symbols, so real cross-sections are eight symbols thick, not twenty.

---

<div align="center">

**Marco Monini** · [LinkedIn](https://www.linkedin.com/in/marco-monini/) · [marco.monini98@gmail.com](mailto:marco.monini98@gmail.com)

*Research code. Nothing here is investment advice, and the project's own conclusion is that the measured edge does not clear a 25 bp fee.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A4FD6,100:000C3D&height=100&section=footer" width="100%" />

</div>
