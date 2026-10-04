"""Streamlit page: download the candles of a crypto pair and draw them with their pivots, the
swing leg position and the feature columns computed on them.

One label, `swing_leg_target`: where each bar sits along the leg between two pivots, in [-1, +1].
It is retrospective — the pivots are found by a centred window — and since 2026-10-03 it is the
only one the project works on. Three predictive labels used to share this page with it, and the
cross-sectional factor with its heatmaps; each was measured against the price and archived, and
`OLD/README.md` says why one by one.

Over the label the page draws the swing models' prediction of it — the 4h model of step 7, or v2
on 15m — and the always-in rule of `threshold` on that *prediction* and never on the label,
because the label is built from a centred window and a rule trading it would be reading
`EXTREMA_WINDOW` bars of future. Two shapes on the candles keep that distinction visible: hollow
squares are the oracle's pivots, which do read the future, and filled triangles are the rule's
own fills.

The rule is any of `strategy`'s, through `strategy.play`, so the page draws exactly what the study
priced: the band (long under -t, short over +t), re-entry (long when the prediction comes back above
-t, short when it comes back below +t), momentum (the band's sides swapped), and the prediction's own
turns at confirmation. A fifth, the turns found in hindsight, reads the future and is drawn under a
warning, as the diagnostic it is. Two more are `detect`'s turn detectors: the zigzag, which flips
when the prediction comes back h from its extreme, and Shiryaev's, which flips when the probability
that the leg has turned reaches p, on the parameters fitted on v2's development folds, and either
can keep only the signals past a level (`detect.gate`), drawing the dropped ones grey. Each reads
the prediction or its mean over the last k bars, can take every trade on the other side, and can
trade only while BTC is under its 200-day mean, read on Alpaca's daily closes fetched with the
candles.

Every rule reads the model's raw output (`swing.predict_frame`'s `raw`), the units the study
measured its thresholds and detectors in; the line drawn against the label is the same output
calibrated onto the label's +-1, where 0.40 raw is 0.575. Under the label the page draws what the
rule reads, with a mark wherever the rule asked for a side, and for a detector the quantity it
decides on: the posterior and the prior hazard, or the retracement. With the prediction's own turns
switched on, each mark is judged against them, filled on the right side of the leg and hollow when
the leg it bet against was still running.

The rule's exits are `stops`, and they are controls rather than settings: a take profit and a stop
loss, each named in ATR or in round trips or in a flat percentage, each with its own answer to
what the rule holds afterwards — reverse into the other side, stand flat until the opposite
signal, or stand flat until any fresh crossing. The page opens on the configuration the study's
development folds converged on — v2, re-entry at 0.40, the BTC filter, a 6 ATR stop and then flat
until the opposite signal — which is the best the study found and still loses on its hold-out
(`HANDOFF.md` §17). With both barriers off and no filter, the band is exactly `threshold`'s rule,
the one the spec priced. A stop fill is marked with an X on
the candle and at the price it filled at, a take profit with a star; a gap through a level fills
at the open, so the mark can sit well past the level it was aimed at, and that is the point of
drawing it at the fill.

streamlit run src/tradingvision/app/chart.py
"""

import importlib
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingvision import detect, metrics, stops, strategy, threshold
from tradingvision.data.candles import SYMBOLS, TIMEFRAMES, get_candles
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import SMOOTHING, leg_significance, swing_leg_target
from tradingvision.features import COLUMNS, FAMILIES, LABELS, features
from tradingvision.normalize import CLIP, SCALE, apply, fit
from tradingvision.oracle import FEE, run


def optional(name: str) -> ModuleType | None:
    """Import `tradingvision.<name>`, or return None when torch is not installed.

    `swing` imports torch at module scope. torch is a runtime dependency now — the page serves
    predictions from the checkpoints in `models/` and `restore` is a torch call — so on a correct
    install this returns the module. The fallback stays because the failure it replaces was total:
    an image built without torch died at import with `ModuleNotFoundError: No module named 'torch'`
    and Render served nothing, losing the candles, the pivots and the label, none of which need
    torch, to a model that had nothing to draw. Only torch is tolerated: any other missing module
    is a broken install and has to be raised.
    """
    try:
        return importlib.import_module(f"tradingvision.{name}")
    except ModuleNotFoundError as missing:
        if missing.name != "torch":
            raise
        return None


swing = optional("swing")

# Where a checkpoint is looked for besides `data/`. `data/` is the store a training run writes to
# and is gitignored, so nothing under it reaches the image; `models/` is tracked, which is how a
# deployed page gets a model at all — commit `swing.pt` and `swing-v2.pt` there and Render serves the
# predictions. `data/` is read first on purpose: on a machine that has just run `swing --save` the
# fresh checkpoint is the one to draw, and a committed file silently shadowing it is the failure
# mode worth avoiding. `TRADINGVISION_MODELS` overrides the directory for a mounted disk.
MODELS = Path(os.environ.get("TRADINGVISION_MODELS") or Path(__file__).resolve().parents[3] / "models")


def saved(module: ModuleType | None, name: str | None = None) -> Path | None:
    """The checkpoint of `swing`, from the store or from `models/`, or None if neither.

    `name` replaces the filename and leaves the two directories and their order alone: v2 sits in
    the same store as `swing.pt` and has to be found by the same rule rather than by a second one
    that could drift away from it.
    """
    if module is None:
        return None
    at = module.CHECKPOINT.with_name(name) if name else module.CHECKPOINT
    return next((p for p in (at, MODELS / at.name) if p.exists()), None)


MAX_DAYS = 365
# The rule the page opens on: the combination `strategy`'s development folds (1-2) converged on, on
# v2's predictions for ETH, BTC and SOL. Re-entry peaked at 0.35-0.40 for every mean tried; the 6 ATR
# stop with a flat wait for the opposite signal was the best exit; the BTC filter is switched on
# below. Together +37.2 bp a trade on folds 1-2 and -22.8 on the hold-out, folds 3-4 — the default
# is the best the study found, not a rule that earns (`HANDOFF.md` §17).
RULE = "reentry"
# In the head's raw units, `swing.predict_frame`'s `raw`, which is what the study read. Until
# 2026-10-04 the page laid every threshold on the calibrated line instead, where 0.40 is 0.28 raw:
# the calibration stretches the head's output onto the label's +-1 (0.40 raw is 0.575 calibrated).
THRESHOLD = 0.40
STOP = ("atr", 6.0)
# How a barrier width is spelled in the sidebar. The identifiers are `stops.KINDS` and the labels
# are the unit each one is a multiple of: ATR is what this market does anyway, the round trip is
# what the trade costs, and a flat percentage is neither. "off" is first, so turning a barrier off
# is one pick and the bare rule is always one step away.
WIDTHS = {"off": "off", "atr": "× ATR at entry", "fee": f"× round trip ({FEE * 200:.2f}%)", "pct": "% of price"}
# What each policy does after a barrier fires, spelled as the sentence rather than the identifier.
POLICIES = {
    "opposite": "wait for the opposite signal",
    "reverse": "reverse into the other side",
    "rearm": "wait for any fresh signal",
}
# The rules of `strategy`, spelled as what each does. The identifiers are `strategy.SIGNALS`.
RULES = {
    "band": "Band: long under −t, short over +t",
    "reentry": "Re-entry: long back above −t, short back below +t",
    "momentum": "Momentum: long over +t, short under −t",
    "confirmed": "Turns of the prediction, at confirmation",
    "turns": "Turns of the prediction, in hindsight (reads the future)",
    "zigzag": "Zigzag: flip when the prediction comes back h from its extreme",
    "shiryaev": "Bayesian detector: flip when P(the leg has turned) ≥ p",
}
BANDED = ("band", "reentry", "momentum")
# The two turn detectors of `detect`. Shiryaev's runs on `detect.V2_FIT`, fitted on v2's own
# predictions, so it is offered only on v2; the zigzag has nothing fitted and runs on either model.
DETECTORS = ("zigzag", "shiryaev")
POSTERIORS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)  # the thresholds `detect --shiryaev` measured
TURN_WINDOWS = (6, 12, 24, 48)  # the grid `strategy --hindsight` measured
MEANS = (1, 2, 4, 8)  # the moving averages `strategy --smooth` measured
# Days of BTC daily closes the filter needs before the first bar on screen: the 200-day mean, and
# a margin for days the venue has no bar for.
BTC_DAYS = 230

# Starting multiples for a barrier switched to a unit, one per unit. Not tuned — only the stop's
# 6 ATR (`STOP`) was measured — but each is the number the unit makes obvious: three ATR is the textbook stop,
# and three round trips is the smallest barrier that clears its own cost by a margin worth the
# name (one round trip nets exactly zero).
SIZE = {"atr": 3.0, "fee": 3.0, "pct": 2.0}

# Downloads only happen on an explicit click, and only for a (symbol, timeframe, days) triplet
# that is not already cached.
load_candles = st.cache_data(ttl=300, show_spinner="Downloading candles…")(get_candles)


@st.cache_data(show_spinner=False)
def load_pivots(close, window: int):
    """Cached on (series, window): the pivots survive every rerun that does not refetch."""
    return find_pivots(close, window)


# The second swing-leg-position model: 15m, pivots *and* features at 12, time weight 0.5, the 15
# inputs of `REDUCED` over 48 bars. Its label parameters are read from the checkpoint and never
# restated here, so the checkbox cannot draw the label at one setting against a model fitted at
# another.
V2 = "Swing Leg Position v2"
V2_CHECKPOINT = "swing-v2.pt"


def store_name(pair: str) -> str:
    """`BTC/USD` -> `BTC`, the name the training store and a checkpoint's scalers use."""
    return pair.split("/")[0]


@st.cache_data(show_spinner=False)
def load_target(close, window: int, smoothing: float, significance: bool):
    """Recomputed when a target control moves; the pivots underneath come from their own cache."""
    return swing_leg_target(close, load_pivots(close, window), smoothing=smoothing, significance=significance)


@st.cache_data(show_spinner=False)
def load_significance(close, window: int):
    """The raw ratio behind the weighting, shown next to each pivot."""
    return leg_significance(close, load_pivots(close, window))


@st.cache_resource(show_spinner=False)
def load_swing_model(path: str, _mtime: float):
    """The saved swing model, kept across reruns. `cache_resource` and not `cache_data`: a torch
    module is not something to pickle and copy on every widget move. `_mtime` is in the key and
    unused in the body: retraining has to invalidate a checkpoint the path alone cannot date."""
    return swing.restore(Path(path))


@st.cache_data(show_spinner="Running the swing model…")
def load_swing(df, path: str, _mtime: float, symbol: str | None = None):
    """`label`, `logit` and `position` at every bar on screen.

    The model needs no store: its inputs are computed from the candles on screen. `symbol` picks
    the training scaler the checkpoint ships for that pair, when it ships one; a pair the training
    set never held is scaled on its own history instead.
    """
    return swing.predict_frame(*load_swing_model(path, _mtime), df, symbol)


@st.cache_data(show_spinner="Computing features…")
def load_features(df, window: int, columns: tuple[str, ...]):
    """Cached on (frame, window, columns): picking different columns to draw does not recompute
    them, and renaming or adding one re-keys the cache — Streamlit keys on this wrapper's source,
    which does not change when `features` does, so a running app would serve the old schema."""
    return features(df, window)[list(columns)]


def book(
    per: pd.DataFrame,
    weight: pd.Series | None,
    symbol: str,
    what: str = "position",
    benchmark: str = "buy and hold",
) -> go.Figure:
    """What the swing model's position earned over the fetched period, and the position itself.

    The label row says whether the prediction follows the leg; this says what holding it was
    worth, which is a different question with a different answer — a correlation is not a book,
    and a book pays fees. Three curves, and the gaps between them are the whole reading: gross to
    net is the fee the sidebar quotes, net to holding the pair is what timing the legs was worth.

    Cumulative log returns shown as percentages, so the curve compounds the way an account does.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[2, 1],
        vertical_spacing=0.05,
        subplot_titles=[
            "book — cumulative return over the fetched period",
            f"{what} on {symbol.split('/')[0]}",
        ],
    )
    for name, y, color, dash, width, on in (
        ("net of fees", per.gross - per.traded * FEE, "#2ecc71", "solid", 2.0, True),
        ("gross", per.gross, "#95a5a6", "dot", 1.5, True),
        # Off until asked for, alone among the three. Drawn by default the pair's own move sets the
        # axis and the two curves this figure exists for flatten under it. The comparison is not
        # hidden — it is the metric above, in the same period and the same units — and one click
        # on the legend puts it back on the axis for anyone who wants the shape rather than the
        # number.
        (benchmark, per.basket, "#34495e", "dash", 1.0, "legendonly"),
    ):
        fig.add_trace(
            go.Scatter(
                x=per.index,
                y=np.expm1(y.cumsum()) * 100,
                mode="lines",
                name=name,
                visible=on,
                line=dict(color=color, dash=dash, width=width),
                hovertemplate="%{x}<br>" + name + " %{y:+.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if weight is not None:
        # A step and not a line: the weight is held between decisions and interpolating it would
        # draw a position that was never carried.
        fig.add_trace(
            go.Scatter(
                x=weight.index,
                y=weight,
                mode="lines",
                name=what,
                line=dict(color="#e67e22", width=1.5, shape="hv"),
                showlegend=False,
                hovertemplate="%{x}<br>" + what + " %{y:+.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        moved = weight.diff().fillna(weight)
        moved = moved[moved != 0]
        fig.add_trace(
            go.Scatter(
                x=moved.index,
                y=weight.reindex(moved.index),
                mode="markers",
                name="trade",
                marker=dict(size=7, symbol="diamond", color=np.where(moved > 0, "#2ecc71", "#e74c3c")),
                showlegend=False,
                hovertemplate="%{x}<br>to %{y:+.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    fig.update_yaxes(title_text="%", zeroline=True, zerolinecolor="#bbb", row=1, col=1)
    fig.update_yaxes(title_text="units", zeroline=True, zerolinecolor="#bbb", row=2, col=1)
    if weight is not None and len(weight.dropna()):
        # Symmetric around zero and with room for the markers, which otherwise sit half off the
        # floor, so a long and a short of the same size look the same size.
        limit = float(np.abs(weight).max()) * 1.3 or 1.0
        fig.update_yaxes(range=[-limit, limit], row=2, col=1)
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    return fig


def barrier(name: str, key: str, start: tuple[str, float] | None = None) -> tuple[str, float] | None:
    """One sidebar barrier — its unit and its multiple — as the spec `stops` takes, or `None`.

    Two widgets and not one, because the unit is the question and the number is only the answer to
    it. "3" means nothing on its own: three ATR is a barrier that scales with the pair, three round
    trips is a barrier that scales with the fee, and three percent is a barrier that scales with
    neither. Naming the unit first is what keeps the multiple readable when the pair changes under
    it.

    The number is a percentage when the unit is one, and `stops` wants a log distance, so it is
    the one place a division belongs. At these sizes the two agree to the fourth decimal.

    `start` is the barrier the page opens on, in the same spelling; `None` opens it off.
    """
    kinds = list(WIDTHS)
    kind = st.sidebar.selectbox(
        name, kinds, index=kinds.index(start[0]) if start else 0, format_func=WIDTHS.get, key=f"{key}-kind"
    )
    if kind == "off":
        return None
    # The unit is in the widget's key, so each unit keeps its own multiple. Sharing one key would
    # carry the number across a change of unit, and 3 is a textbook stop in ATR and a very
    # different barrier in percent — the kind of silent reinterpretation a reader cannot see.
    size = st.sidebar.number_input(
        f"{name}, {WIDTHS[kind]}",
        min_value=0.1,
        max_value=20.0,
        value=start[1] if start and start[0] == kind else SIZE[kind],
        step=0.1,
        key=f"{key}-{kind}-size",
    )
    return kind, (size / 100 if kind == "pct" else size)


def describe(spec: tuple[str, float] | None) -> str:
    """A barrier spec as the caption says it: `("atr", 3.0)` to `3 × ATR at entry`.

    The percentage goes back the way `barrier` brought it in, so a reader of the caption and a
    reader of the sidebar see the same number.
    """
    if spec is None:
        return "no"
    kind, size = spec
    return f"{size * 100:g}{WIDTHS[kind]}" if kind == "pct" else f"{size:g} {WIDTHS[kind]}"


def chart(
    df,
    pivots,
    target,
    significance,
    feats,
    symbol: str,
    uirevision: str,
    normalized=False,
    pred=None,
    trades=None,
    exits=None,
    panels=None,
) -> go.Figure:
    # Price, then the label under it on the same x: the target is only readable against the leg it
    # describes. Volume next, it is context rather than subject, and the features under everything.
    # One row per family, never one for all of them: adx_trend_strength sits in [0, 1] and
    # log_return around 1e-3, so sharing an axis would flatten the second into the zero line.
    groups = [(fam, [c for c in cols if c in feats.columns]) for fam, cols in FAMILIES.items()]
    groups = [g for g in groups if g[1]]
    # `panels` are the rule's own rows, under the label: what the rule reads and, for a detector,
    # the quantities it decides on. Each is a dict of `title`, `traces`, `hlines` and `range`.
    panels = panels or []
    n = len(panels)
    fig = make_subplots(
        rows=3 + n + len(groups),
        cols=1,
        shared_xaxes=True,
        # Weights, normalised by Plotly: with all seven families open a fixed share for the candles
        # would squeeze every feature row into a line.
        row_heights=[3, 1.5] + [1.4] * n + [1] + [1.5] * len(groups),
        vertical_spacing=0.02,
        subplot_titles=["", ""] + [panel["title"] for panel in panels] + [""] + [f.title() for f, _ in groups],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name=symbol, showlegend=False
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=target.index,
            y=target,
            mode="lines",
            line=dict(width=1.5, color="#3498db"),
            name="target",
            showlegend=False,
            connectgaps=False,  # the unlabelled head and tail must read as gaps, not as a line
            hovertemplate="%{x}<br>target %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if pred is not None:
        # On the target's own axis, and in its units: the swing models calibrate their head back
        # onto the label's range with a monotone map fitted on train, so the gap between the two
        # lines is the error and not a change of scale.
        fig.add_trace(
            go.Scatter(
                x=pred.index,
                y=pred,
                mode="lines",
                line=dict(width=1.5, color="#e67e22"),
                name="prediction",
                showlegend=False,
                connectgaps=False,  # the warm-up has no window behind it and must read as a gap
                hovertemplate="%{x}<br>prediction %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    for kind, color, position in ((1, "#e74c3c", "top center"), (-1, "#2ecc71", "bottom center")):
        p = pivots[pivots.kind == kind]
        # The same pivots on the label axis, where they sit at exactly +/-1 by construction.
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=target.reindex(p.index),
                marker=dict(size=7, color=color, symbol="circle"),
                mode="markers+text",
                # The leg's significance next to the pivot it scored: 1 is a leg worth what the
                # volatility alone would have produced, and the value shown above is tanh of it.
                text=[f"{v:.1f}" for v in significance.reindex(p.index)],
                textposition=position,
                textfont=dict(size=9, color=color),
                name="pivot",
                showlegend=False,
                hovertemplate="%{x}<br>target %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        # Hollow squares, and no leg amplitude written next to them. The oracle and the rule now
        # share this row, so the two have to be told apart at a glance before anything else is
        # readable: the oracle is what hindsight would have taken, the filled triangles below are
        # what the rule actually took, and a shape is a faster distinction than a colour. The
        # percentages went with the circles — they annotated the oracle's legs, which is a
        # different quantity from the rule's trades and the one that made the row unreadable when
        # both are drawn. The amplitude is still in the caption and still in `p.amplitude`.
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p.close,
                mode="markers",
                # Small and thin on purpose. The oracle marks every pivot in the window and the
                # rule marks a handful of fills, so at equal weight the squares are what the eye
                # reads first and the trades disappear into them. The squares are the backdrop.
                marker=dict(size=5, color=color, symbol="square-open", line=dict(width=1, color=color)),
                name="pivot",
                showlegend=False,
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if trades is not None:
        # Where and when, on the price itself. Markers sit at the close of the bar the decision was
        # taken on, which is the price that decision actually paid.
        moved = trades.diff().fillna(trades)
        moved = moved[(moved != 0) & (moved.index >= df.index[0]) & (moved.index <= df.index[-1])]
        # Keyed on the position each change moved *to*, and that is a bug fix and not a style
        # choice. The old version keyed on the sign of the change and called every rise "buy"
        # and every fall "sell", which was right only while the rule had no flat state: an
        # always-in rule goes +1 → −1 and back, so the words alternated by construction. With
        # an exit there is a flat state in between, and closing a short (−1 → 0) and opening a
        # long (0 → +1) are both a rise — two "buy" markers in a row with no "sell" between
        # them, which reads on the chart as a position that was opened twice and never closed.
        # The trade was fine; the caption on it was not. What a marker can always say without
        # ambiguity is the state the rule is in from that bar on, so that is what it says.
        for state, color, shape, word, position in (
            (1.0, "#2ecc71", "triangle-up", "long", "bottom center"),
            (-1.0, "#e74c3c", "triangle-down", "short", "top center"),
            (0.0, "#95a5a6", "line-ew", "flat", "top center"),
        ):
            at = moved.index[trades.reindex(moved.index) == state]
            fig.add_trace(
                go.Scatter(
                    x=at,
                    # Forward filled: a pair that did not trade in a bar has no candle there,
                    # and the decision still happened at the last price the panel carried.
                    y=df.close.reindex(at, method="ffill"),
                    # The words only while they can be read: past a few dozen marks they pile into
                    # a smear over the candles, and the colour and the shape already say it.
                    mode="markers+text" if len(moved) <= 40 else "markers",
                    # Big, filled, and outlined in the page's own background: a bright triangle
                    # sitting on a green candle needs the halo to read as a separate mark.
                    marker=dict(size=16, color=color, symbol=shape, line=dict(width=1.5, color="#0e1117")),
                    text=[word] * len(at),
                    textposition=position,
                    textfont=dict(size=12, color=color),
                    name=word,
                    showlegend=False,
                    hovertemplate="%{x}<br>" + word + " from here, at %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    if exits is not None and len(exits):
        # The barrier fills, at the price they filled at rather than at the bar's close. That
        # distinction is the whole reason to draw them: a level is where the rule aimed, the fill
        # is where it got out, and a bar that gapped through the level puts the two far apart. A
        # third shape, because the triangles above already mean "the signal changed its mind" and
        # a stop is the opposite of that — the price changed it.
        for why, color, shape, size in (("stop", "#e74c3c", "x-thin", 11), ("take", "#2ecc71", "star", 11)):
            fills = exits[(exits.why == why) & exits.exit_time.between(df.index[0], df.index[-1])]
            fig.add_trace(
                go.Scatter(
                    x=fills.exit_time,
                    y=fills.exit,
                    mode="markers",
                    marker=dict(size=size, color=color, symbol=shape, line=dict(width=2, color=color)),
                    name=why,
                    showlegend=False,
                    hovertemplate="%{x}<br>" + why + " at %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    for row, panel in enumerate(panels, start=3):
        for trace in panel["traces"]:
            fig.add_trace(trace, row=row, col=1)
        for y, color in panel.get("hlines", []):
            fig.add_hline(y=y, line=dict(width=1, dash="dash", color=color), row=row, col=1)
        fig.update_yaxes(range=panel.get("range"), zeroline=True, zerolinecolor="#555", row=row, col=1)
    fig.add_trace(
        go.Bar(x=df.index, y=df.volume, name="volume", marker_color="#888", showlegend=False), row=3 + n, col=1
    )
    for row, (_, cols) in enumerate(groups, start=4 + n):
        for col in cols:
            fig.add_trace(
                go.Scatter(x=feats.index, y=feats[col], mode="lines", name=LABELS[col], line=dict(width=1)),
                row=row,
                col=1,
            )
        # Same range on every feature row: the point of normalising is that the rows compare, and
        # an axis fitted to each row would hide it.
        if normalized:
            fig.update_yaxes(range=[-CLIP * SCALE * 1.1, CLIP * SCALE * 1.1], row=row, col=1)
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    # The label lives in [-1, +1] and is read against that ceiling.
    fig.update_yaxes(
        title_text="target vs prediction" if pred is not None else "target",
        range=[-1.3, 1.3],
        zeroline=True,
        zerolinecolor="#bbb",
        row=2,
        col=1,
    )
    # The ceiling the pivots would reach unweighted: the gap to it is the significance discount.
    for y in (-1, 1):
        fig.add_hline(y=y, line=dict(width=1, dash="dot", color="#bbb"), row=2, col=1)
    fig.update_layout(
        height=800 + 150 * len(groups) + 170 * n,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        showlegend=bool(groups),
        xaxis_rangeslider_visible=False,
        margin=dict(t=30, b=10),
        # Keeps zoom and pan across reruns: Plotly patches the existing figure instead of
        # remounting it. The value changes only with the fetched series, so the view resets when
        # the instrument or the period does and survives everything else.
        uirevision=uirevision,
    )
    return fig


def rule_panels(
    raw: pd.Series,
    signal: pd.Series,
    turns: pd.Series | None,
    trace: pd.DataFrame | None,
    rule: str,
    band: float,
    posterior: float,
    h: float,
    dropped: pd.Series | None = None,
    level: float = 0.0,
) -> list[dict]:
    """The rule's rows: what it reads, where it asked for a side, and what a detector decided on.

    `signal` is what the rule handed `stops.walk`, +1 long and -1 short. A marker goes where it asks
    for the other side from the last one it asked for, which is a decision even when a stop or the
    filter left the position flat — the triangles on the candles are what was executed, these are
    what was asked. `turns` are the input's own centred turns, which read the future; with them
    each marker is judged, filled on the right side of the leg it was taken in and hollow when the
    leg it bet against was still running, as `detect.kinds` judges an alarm. `dropped` are the
    detector's alarms the level gate (`detect.gate`) refused, drawn grey, and `level` its two lines.
    """
    asked = signal.replace(0.0, np.nan)
    asked = asked[asked.notna() & (asked != asked.ffill().shift())]
    known = turns[turns != 0] if turns is not None else None
    traces = [
        go.Scatter(
            x=raw.index,
            y=raw,
            mode="lines",
            line=dict(width=1.5, color="#e67e22"),
            name="rule input",
            showlegend=False,
            hovertemplate="%{x}<br>raw prediction %{y:.3f}<extra></extra>",
        )
    ]
    if known is not None:
        traces.append(
            go.Scatter(
                x=known.index,
                y=raw.reindex(known.index),
                mode="markers",
                # Mid grey: it has to read on the light theme and on the dark one alike.
                marker=dict(size=11, symbol="circle-open", color="#7f8c8d", line=dict(width=2)),
                name="turn in hindsight",
                showlegend=False,
                hovertemplate="%{x}<br>a turn of the prediction, known only bars later<extra></extra>",
            )
        )
    for side, color, shape, word in (
        (1.0, "#2ecc71", "triangle-up", "long"),
        (-1.0, "#e74c3c", "triangle-down", "short"),
    ):
        at = asked.index[asked == side]
        right = np.ones(len(at), dtype=bool)
        if known is not None:
            j = known.index.searchsorted(at, side="right") - 1
            right = (j >= 0) & (known.to_numpy()[np.maximum(j, 0)] == side)
        for ok in (True, False):
            pick = at[right == ok]
            traces.append(
                go.Scatter(
                    x=pick,
                    y=raw.reindex(pick),
                    mode="markers",
                    marker=dict(size=11, symbol=shape if ok else f"{shape}-open", color=color, line=dict(width=1.5)),
                    name=word,
                    showlegend=False,
                    hovertemplate="%{x}<br>asks for "
                    + word
                    + ("" if ok else ", against a leg still running")
                    + "<extra></extra>",
                )
            )
    if dropped is not None and dropped.any():
        at = dropped.index[dropped != 0]
        traces.append(
            go.Scatter(
                x=at,
                y=raw.reindex(at),
                mode="markers",
                marker=dict(size=9, symbol="x-thin", color="#7f8c8d", line=dict(width=2, color="#7f8c8d")),
                name="dropped",
                showlegend=False,
                hovertemplate="%{x}<br>a signal the level filter dropped<extra></extra>",
            )
        )
    lines = [band, -band] if rule in BANDED else [level, -level] if level else []
    panels = [
        {
            "title": "What the rule reads — the raw prediction — and where it asked for a side",
            "traces": traces,
            "hlines": [(y, "#888") for y in lines],
        }
    ]
    if rule == "shiryaev":
        panels.append(
            {
                "title": "P(the leg has turned), and the prior hazard of a turn on each bar",
                "traces": [
                    go.Scatter(
                        x=trace.index,
                        y=trace.posterior,
                        mode="lines",
                        line=dict(width=1.5, color="#9b59b6"),
                        name="P(turned)",
                        showlegend=False,
                        hovertemplate="%{x}<br>P(turned) %{y:.2f}<extra></extra>",
                    ),
                    go.Scatter(
                        x=trace.index,
                        y=trace.hazard,
                        mode="lines",
                        line=dict(width=1, color="#95a5a6"),
                        name="hazard",
                        showlegend=False,
                        hovertemplate="%{x}<br>prior hazard %{y:.3f}<extra></extra>",
                    ),
                ],
                "hlines": [(posterior, "#9b59b6")],
                "range": [0, 1.02],
            }
        )
    if rule == "zigzag" or (rule == "shiryaev" and h > 0):
        panels.append(
            {
                "title": "How far the prediction has come back from the extreme of the leg",
                "traces": [
                    go.Scatter(
                        x=trace.index,
                        y=trace.retrace,
                        mode="lines",
                        line=dict(width=1.5, color="#3498db"),
                        name="retracement",
                        showlegend=False,
                        hovertemplate="%{x}<br>back %{y:.3f} from the extreme<extra></extra>",
                    )
                ],
                "hlines": [(h, "#3498db")],
                "range": [0, max(float(trace.retrace.max()), h) * 1.1],
            }
        )
    return panels


def main() -> None:
    st.set_page_config(page_title="Trading Vision", layout="wide")
    st.title("Trading Vision")

    # The tradable pairs of the training universe, and a typed one as well. Alpaca's coverage
    # moves, so a pair it stops serving draws the warning below and nothing else breaks.
    symbol = st.sidebar.selectbox(
        "Pair",
        SYMBOLS,
        accept_new_options=True,
        help="the pairs of the training universe that Alpaca lists, quoted in USD. Type any other Alpaca pair "
        "(`BASE/USD`) to draw it — the page says so if the venue serves nothing for it.",
    )
    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES), index=1)
    days = st.sidebar.slider("History (days)", 1, MAX_DAYS, 30)
    # The v2 swing model, and the checkbox that selects it. Ticked, it pins the label to the two
    # numbers the checkpoint was trained on — the sliders below show them and cannot move them —
    # and it moves the features' window with the pivots', because v2 moved both together. Offered
    # only where there is a checkpoint to read them from.
    v2_at = saved(swing, V2_CHECKPOINT)
    v2_card = load_swing_model(str(v2_at), v2_at.stat().st_mtime)[1] if v2_at else None
    v2 = False
    if v2_card:
        v2 = st.sidebar.checkbox(
            V2,
            value=True,
            help=f"{v2_at.name}: {v2_card['timeframe']} candles, pivots and features at {v2_card['window']} bars, "
            f"time weight {v2_card['smoothing']}, {len(v2_card['inputs'])} inputs x {v2_card['steps']} bars, "
            f"trained to {v2_card['test_start']}",
        )
    elif swing is not None:
        st.sidebar.caption(
            f"No {V2} model. `python -m tradingvision.swing --timeframe 15m --window 12 --smoothing 0.5 "
            f"--inputs reduced --steps 48 --stage label --test-start 2025-06 --save data/{V2_CHECKPOINT}`, "
            f"then copy it into `{MODELS.name}/`."
        )
    feature_window = v2_card["window"] if v2 else EXTREMA_WINDOW
    # The label's two knobs. 0.7 is a starting value for the time weight: 1.0 is a pure time ramp
    # between pivots, 0.0 follows price alone.
    smoothing = st.sidebar.slider(
        "Target smoothing (time weight)", 0.0, 1.0, v2_card["smoothing"] if v2 else SMOOTHING, 0.1, disabled=v2
    )
    # The pivots the label ramps between. `EXTREMA_WINDOW = 24` is calibrated in `oracle` on the
    # hindsight P&L of the legs; v2 moved it to 12. The features stay on 24 whatever this says
    # unless v2 is ticked: moving the label's pivots does not rebuild the columns a model reads.
    leg_window = st.sidebar.slider(
        "Leg window (label pivots)",
        4,
        96,
        v2_card["window"] if v2 else EXTREMA_WINDOW,
        4,
        disabled=v2,
        help=(
            f"fixed by {V2}: the pivots and the features both at its window"
            if v2
            else "bars the pivot detector needs clear on both sides"
        ),
    )
    # Off shows the flat +/-1 labelling, which the weighting is meant to be read against. v2 was
    # trained on the weighted one, so under it the switch is not offered.
    significance = True if v2 else st.sidebar.toggle("Weight pivots by leg significance", value=True)
    # A toggle and not a 29-item checklist. Off: the candle features the swing models read — the
    # columns of `swing.REDUCED` that `features` computes; its leg-state and exhaustion columns come
    # from `legs` and are not drawn here.
    full = st.sidebar.toggle(
        f"All {len(COLUMNS)} candidates",
        value=swing is None,
        help="off: the candle features of `REDUCED`, the cut the swing models read",
    )
    picked = COLUMNS if full or swing is None else [c for c in COLUMNS if c in swing.REDUCED]
    normalized = st.sidebar.toggle("Normalized", value=True, help="clip((x - median) / IQR, ±5) × 0.1")

    # The swing model of step 7, `swing.pt`. Drawn only on the timeframe it was fitted on: its
    # inputs are windows of that bar and nothing rescales them between one bar and another.
    swing_at = saved(swing)
    swing_card = load_swing_model(str(swing_at), swing_at.stat().st_mtime)[1] if swing_at else None
    swinging = False
    if swing is None:
        st.sidebar.caption("This install has no torch, so no swing model.")
    elif v2:
        pass  # the step-7 model reads 24-bar pivots on another timeframe; under v2 it has no line
    elif not swing_at:
        st.sidebar.caption(
            f"No swing model. `python -m tradingvision.swing --save`, then copy it into `{MODELS.name}/`."
        )
    elif timeframe != swing_card["timeframe"]:
        st.sidebar.caption(f"The swing model reads **{swing_card['timeframe']}** candles.")
    else:
        swinging = st.sidebar.toggle(
            "Swing trades",
            value=True,
            help=f"{swing_at.name}, stage {swing_card['stage']}, trained to {swing_card['test_start']}",
        )

    # v2 draws on the timeframe it was fitted on and nowhere else, like every model here.
    v2_drawn = v2 and timeframe == v2_card["timeframe"]
    if v2 and not v2_drawn:
        st.sidebar.caption(f"{V2} reads **{v2_card['timeframe']}** candles.")

    # The trading rule, drawn on whatever swing leg position the row below shows. It reads the
    # *prediction* and never the target: the label is built from a centred window, so a rule trading
    # it would be reading `EXTREMA_WINDOW` bars of future and would draw an oracle wearing a
    # strategy's markers. That is why the toggle appears only once a model is on, and says so when
    # it is not. The one exception is the hindsight-turns rule, which reads the future on purpose
    # and is labelled as the diagnostic it is.
    ruling, rule, band, turn_window, mean, inverse, filtered = False, "band", THRESHOLD, 12, 1, False, False
    h, posterior, prior, hindsight = 0.2, POSTERIORS[0], True, False
    level, level_at, level_close = 0.0, "alarm", True
    take, stop, after_stop, after_take, tie_stop, trail = None, None, stops.AFTER[0], stops.AFTER[0], True, False
    if swinging or v2_drawn:
        ruling = st.sidebar.toggle(
            "Trading rule",
            value=True,
            help="the rules of `strategy`, on the prediction: one code path for the page and for the study",
        )
        if ruling:
            offered = [r for r in RULES if r != "shiryaev" or v2_drawn]
            rule = st.sidebar.selectbox("Rule", offered, index=offered.index(RULE), format_func=RULES.get)
            if rule in BANDED:
                # A constant and not a quantile: a quantile would move with the model, and a
                # constant is what a live system has to commit to.
                band = st.sidebar.slider(
                    "Threshold ±t", 0.05, 1.0, THRESHOLD, 0.05, help="on the raw prediction, the study's units"
                )
            elif rule == "zigzag":
                h = st.sidebar.slider(
                    "Retracement h",
                    0.05,
                    0.8,
                    0.2,
                    0.05,
                    help="a short when the raw prediction has fallen h from its highest since the last signal, a long "
                    "when it has risen h from its lowest. The study measured 0.05 to 0.8; at 0.2 it finds 97% of the "
                    "turns about 5 bars late, with 0.31 false signals a turn",
                )
            elif rule == "shiryaev":
                posterior = st.sidebar.select_slider(
                    "Signal when P(the leg has turned) reaches",
                    POSTERIORS,
                    value=POSTERIORS[0],
                    help="the posterior is updated on every bar from the prior hazard and the bar's move. At 0.5 the "
                    "study found 79% of the turns, 3 bars late at the median, with 0.35 false signals a turn",
                )
                prior = st.sidebar.toggle(
                    "Prior from the level and the leg's age",
                    value=True,
                    help="the chance of a turn on the next bar rises with how far the prediction has gone and how old "
                    "the leg is; off, every bar gets the same chance and the moves alone decide",
                )
                h = st.sidebar.slider(
                    "…and only after a retracement of",
                    0.0,
                    0.5,
                    0.0,
                    0.05,
                    help="0 is off. Above 0 the signal also waits for the zigzag's condition: fewer false signals, "
                    "each one dearer, and every right one later",
                )
            else:
                turn_window = st.sidebar.select_slider(
                    "Turn window (bars each side)",
                    TURN_WINDOWS,
                    value=12,
                    help="a turn is the highest or lowest prediction of this many bars before and after it. "
                    "At confirmation the rule acts that many bars after the turn; in hindsight it acts on the "
                    "turn itself, which nobody can know at the time",
                )
            if rule in DETECTORS:
                # Fewer signals by where they happen: a long only from deep enough, a short only from
                # high enough. `detect --gate` measured 0.1 to 0.5 both ways and found no gate that pays.
                level = st.sidebar.slider(
                    "Keep a signal only past ±L",
                    0.0,
                    0.6,
                    0.0,
                    0.05,
                    help="0 is off. A long only where the raw prediction is at or under −L, a short only at or over "
                    "+L. In the study it cut the trades by up to 50 times and left the gross per trade near zero",
                )
                if level:
                    level_at = st.sidebar.selectbox(
                        "…the level read at",
                        ["alarm", "extreme"],
                        format_func={"alarm": "the signal's bar", "extreme": "the extreme of the leg it closes"}.get,
                    )
                    level_close = st.sidebar.selectbox(
                        "A dropped signal",
                        [True, False],
                        format_func={True: "closes the position, then flat", False: "is ignored"}.get,
                        help="closing stands the rule flat until the next kept signal; ignoring holds the side it had",
                    )
            if rule not in DETECTORS:
                # The detectors were fitted on the prediction bar by bar; a mean would change the very
                # increments their evidence is measured on.
                mean = st.sidebar.select_slider(
                    "Mean of the prediction (bars)",
                    MEANS,
                    value=1,
                    help="the rule reads the mean of the last k predictions; 1 is the raw prediction",
                )
            hindsight = st.sidebar.toggle(
                "Judge the signals against the prediction's turns",
                value=rule in DETECTORS,
                help="marks the prediction's own turns at 12 bars, which are known only 12 bars after them, and "
                "draws every signal filled when it was on the right side of the leg and hollow when it was not",
            )
            inverse = st.sidebar.toggle(
                "Invert every trade", value=False, help="every long becomes a short and every short a long"
            )
            filtered = st.sidebar.toggle(
                "Only while BTC is under its 200-day mean",
                value=True,
                help="read on yesterday's daily close, so it is causal. Flat while it is off; when it switches on, "
                "the rule waits for a fresh signal",
            )
            # The exits. The stop opens on the study's 6 ATR and the take profit off: no take
            # profit tried in the study hurt at every size that fired.
            stop = barrier("Stop loss", "sl", STOP)
            if stop:
                # The fourth lever, and the only one that changes where the barrier *is* rather
                # than how far away it starts. Off is the plain stop; on, the same width hangs off
                # the best price the hold has seen, so the exit turns from "how much am I willing
                # to lose" into "how much of what I am up am I willing to give back".
                trail = st.sidebar.toggle(
                    "Trail the stop",
                    value=False,
                    help="same width, measured from the hold's best price instead of its entry — "
                    "it ratchets one way and only off bars that have already closed",
                )
                after_stop = st.sidebar.selectbox(
                    "After a stop",
                    list(POLICIES),
                    format_func=POLICIES.get,
                    key="after-stop",
                    help="on the study's development folds, waiting for the opposite signal was the best of the three",
                )
            take = barrier("Take profit", "tp")
            if take:
                after_take = st.sidebar.selectbox(
                    "After a take profit", list(POLICIES), format_func=POLICIES.get, key="after-take"
                )
            if take and stop:
                # The one control that is an assumption rather than a rule: which of two exits
                # already inside the same candle happened first, which a candle does not record.
                # Both readings are here because the distance between them is the size of the
                # assumption, and a result that only survives the optimistic one is a result about
                # the intrabar path rather than about the rule.
                tie_stop = st.sidebar.selectbox(
                    "If one candle reaches the stop and the take profit",
                    ["assume the stop came first (prudent)", "assume the take profit came first (optimistic)"],
                    help="a candle gives a high and a low but not the order they arrived in, so a bar "
                    "that reaches both levels has two readings and this picks one. Run it both ways: "
                    "the gap between them is how much of the result is an assumption about the path.",
                ).startswith("assume the stop")
    else:
        st.sidebar.caption("The trading rule needs a **prediction** of the swing leg position, not the label.")

    request = (symbol, timeframe, days)
    if st.sidebar.button("Fetch candles", type="primary", use_container_width=True):
        st.session_state.fetched = (request, load_candles(*request))
        # The filter's input, fetched with the candles so that toggling it never downloads anything.
        st.session_state.btc_daily = load_candles("BTC/USD", "1d", days + BTC_DAYS)

    # The fee is not an input: it is the venue's, and a page that let it be typed would price a
    # rule nobody can trade. The feature window is not one either — every feature derives its own
    # windows from it, so moving it here would draw columns the tensor was not built from. The
    # *label's* window is the one the sidebar moves, and it is a separate number unless v2 ties
    # the two together.
    st.sidebar.divider()
    st.sidebar.caption(
        f"**Feature window** &nbsp; {feature_window} bars — "
        + (f"{V2}'s, with its pivots" if v2 else "calibrated, see the spec")
        + "  \n"
        f"**Fee** &nbsp; {FEE * 100:.2f}% per side, {FEE * 200:.2f}% round trip — Alpaca taker tier 1"
    )

    if "fetched" not in st.session_state:
        st.info("Pick a pair, a timeframe and a period, then press **Fetch candles**.")
        return
    fetched, df = st.session_state.fetched
    if fetched != request:
        st.warning(f"Showing {fetched[0]} {fetched[1]}, {fetched[2]}d — press **Fetch candles** to load the new one.")
    if df.empty:
        st.warning(f"No data for {fetched[0]} on {fetched[1]}.")
        return
    pivots = load_pivots(df.close, leg_window)
    target = load_target(df.close, leg_window, smoothing, significance)
    pair = store_name(fetched[0])
    swung = load_swing(df, str(swing_at), swing_at.stat().st_mtime, pair) if swinging else None
    if swung is not None and not swung.position.notna().any():
        # Not an error and not an empty chart: the model reads 24 bars of history through features
        # that need another hundred behind them, so a short window leaves nothing to score. Said
        # here rather than drawn as a flat line, which would read as "the model does nothing".
        st.info(
            f"The swing model needs about {swing.MIN_BARS} scorable {fetched[1]} bars behind the "
            f"window — this one has {len(df)} candles in total. Widen **History (days)**."
        )
        swung = None
    swing_pos = swung.position.fillna(0.0) if swung is not None else None
    # v2's line, on the label's own range: the calibration fitted on train maps the head's shrunk
    # output back onto +-1, monotonically, so the always-in band reads it in the label's units.
    v2_line = load_swing(df, str(v2_at), v2_at.stat().st_mtime, pair) if v2_drawn else None
    if v2_line is not None and not v2_line.label.notna().any():
        st.info(
            f"{V2} reads {v2_card['steps']} bars through features that need about {6 * v2_card['window']} "
            f"behind them — this window has {len(df)} candles. Widen **History (days)**."
        )
        v2_line = None
    if v2_line is not None and pair not in (v2_card.get("scalers") or {}):
        st.caption(
            f"{fetched[0]} is not one of the pairs {V2} was trained on, so its inputs are scaled on the "
            f"window on screen — including bars after the one each prediction is drawn at."
        )
    strength = load_significance(df.close, leg_window)
    feats = load_features(df, feature_window, tuple(COLUMNS))[picked]
    # One model at a time: v2 when its box is ticked, the step-7 model otherwise. Both are
    # calibrated back onto the label's own range on the way out, so the two lines in the second
    # row share a unit as well as an axis. The map is fitted on the train period and stored in the
    # checkpoint; it is monotone, so it moved no decision on the way here.
    pred, raw = None, None
    if v2_line is not None:
        pred, raw = v2_line.label.rename("prediction"), v2_line.raw
    elif swung is not None:
        pred, raw = swung.label.rename("prediction"), swung.raw
    # The rule, on one pair lifted into the one-symbol panel every `strategy` function reads. Same
    # signals, same state machine (`stops.walk`), same filter and same fill arithmetic as the study's
    # tables — there is no second implementation here to drift away from the one that was priced.
    rule_pos, rule_pnl, fills, on, panels, judged = None, None, None, None, None, None
    if ruling and filtered:
        daily = st.session_state.get("btc_daily")
        if daily is None or daily.empty:
            st.info("The BTC filter needs BTC's daily closes: press **Fetch candles** again.")
            filtered = False
    # The rule reads the head's raw output and not the calibrated line drawn against the label: the
    # raw output is what the study measured every threshold and every detector on.
    if ruling and raw is not None and raw.notna().any():
        live = raw.dropna()
        bars = threshold.on_one(df[list(stops.OHLC)].reindex(live.index))
        if filtered:
            on = strategy.below(bars.index, daily=st.session_state.btc_daily.close)
        trace = None
        if rule == "zigzag":
            trace = detect.zigzag(live.to_numpy(), h, trace=True).set_axis(live.index)
        elif rule == "shiryaev":
            trace = detect.shiryaev(live.to_numpy(), detect.V2_FIT, posterior, not prior, h, trace=True)
            trace = trace.set_axis(live.index)
        # What `walked` may trade on: the BTC filter, and the level gate's flat stretches. Kept apart
        # from `on`, which the caption reads as the BTC filter alone.
        dropped, allowed = None, on
        if trace is not None:
            alarm = threshold.on_one(trace.alarm)
            if level:
                kept, gated = detect.gate(alarm, threshold.on_one(live), level, level_at, level_close)
                dropped, alarm = alarm.where(kept == 0, 0.0).droplevel(1), kept
                allowed = gated if allowed is None else allowed & gated
            sig = -alarm if inverse else alarm
        else:
            sig = strategy.signal(threshold.on_one(live), rule, band, turn_window, mean, inverse)
        got = strategy.walked(
            sig,
            bars,
            take=take,
            stop=stop,
            after=after_stop,
            after_take=after_take,
            trail=trail,
            tie_stop=tie_stop,
            on=allowed,
            atr_window=feature_window,
        )
        rule_pos, rule_pnl, fills = got[0].droplevel(1), got[1].droplevel(1), got[2].reset_index(drop=True)
        turns = strategy.turn_events(threshold.on_one(live), strategy.WINDOW) if hindsight else None
        if turns is not None and (turns != 0).any():
            judged = detect.match(sig, turns, live.index[-1] + pd.Timedelta("1D"))
        if turns is not None:
            turns = turns.droplevel(1)
        panels = rule_panels(live, sig.droplevel(1), turns, trace, rule, band, posterior, h, dropped, level)

    if normalized and len(feats.columns):
        # Fitted on the window on screen, which is what a chart can do and not what the dataset
        # does: there the statistics come from the train period alone.
        alive = feats.columns[(feats.quantile(0.75) - feats.quantile(0.25)) > 0]
        feats = apply(feats[alive], fit(feats[alive]))

    # Oracle: what a perfect-hindsight trader would have made on this window over this period.
    stats = run(df.close, leg_window, FEE, pivots=pivots)
    columns = st.columns(2)
    columns[0].metric("Oracle net return", f"{stats['net_return'] * 100:,.1f}%", f"{stats['trades']} legs")
    columns[1].metric("Avg gross leg", f"{stats['gross_trade_pct']:.2f}%", f"{stats['win_rate'] * 100:.0f}% above fees")

    if swung is not None:
        # What the model made on the window on screen, against the two oracles. The first is the
        # hindsight one the page has always shown; the second is the same oracle filling `W` bars
        # later, which is the earliest a pivot of a centred window can be known to anybody and
        # therefore the only one of the two a causal rule could reach. Measured over twenty pairs
        # the second is 7% of the first, and quoting the model against the first alone would
        # describe a 7x gap that no model can close.
        span = float((df.index[-1] - df.index[0]) / swing.YEAR)
        got = swing.price(swing_pos.to_numpy(), df.close.to_numpy(), span, FEE)
        reachable = run(df.close, leg_window, FEE, pivots=pivots, lag=leg_window)
        mine = got["log_per_year"] * span
        best = stats["log_per_year"] * span
        near = reachable["log_per_year"] * span
        row = st.columns(4)
        row[0].metric(
            "Swing net return",
            f"{np.expm1(mine) * 100:+.1f}%",
            f"{got['trades']} trades"
            + (f", {got['win_rate'] * 100:.0f}% win" if got["trades"] else "")
            + f", {got['in_market'] * 100:.0f}% in market",
        )
        row[1].metric("Buy and hold", f"{(df.close.iloc[-1] / df.close.iloc[0] - 1) * 100:+.1f}%", "same window")
        row[2].metric(
            "Of the reachable oracle",
            f"{mine / near * 100:+.0f}%" if near else "—",
            f"{np.expm1(near) * 100:+.0f}% filling {leg_window} bars after each pivot",
        )
        row[3].metric(
            "Of the hindsight oracle",
            f"{mine / best * 100:+.1f}%" if best else "—",
            f"{np.expm1(best) * 100:,.0f}% — it reads the future",
        )

    if fills is not None:
        # Fees per side on every entry and every exit that happened; a hold still open at the right
        # edge has paid its entry only.
        sides = len(fills) + int((fills.why != "open").sum())
        gross, fees = float(rule_pnl.sum()), sides * FEE
        each = fills.gross * 1e4
        # Two rows of three: six numbers across a page that keeps a sidebar cut every one of them short.
        # The captions under the numbers describe them and say nothing good or bad, so they are not
        # coloured.
        top, bottom = st.columns(3), st.columns(3)
        top[0].metric(
            "Rule net return",
            f"{np.expm1(gross - fees) * 100:+.1f}%",
            f"gross {np.expm1(gross) * 100:+.1f}%, fees {np.expm1(fees) * 100:.1f}%",
            delta_color="off",
        )
        # The study's own yardstick: what a trade makes before the fee, against the round trip it
        # has to pay. Half of it is the fee per side at which the rule breaks even.
        top[1].metric(
            "Gross per trade",
            f"{each.mean():+.1f} bp" if len(fills) else "—",
            (
                (f"breaks even at {each.mean() / 2:.1f} bp a side" if each.mean() > 0 else "loses before any fee")
                if len(fills)
                else "no trade"
            ),
            delta_color="off",
        )
        bottom[0].metric(
            "Long trades, gross",
            f"{np.expm1(fills.gross[fills.side > 0].sum()) * 100:+.1f}%",
            f"short trades {np.expm1(fills.gross[fills.side < 0].sum()) * 100:+.1f}%",
            delta_color="off",
        )
        top[2].metric(
            "Trades",
            f"{len(fills)}",
            (
                f"{(fills.gross > 2 * FEE).mean() * 100:.0f}% win after fees, median {fills.bars.median():.0f} bars"
                if len(fills)
                else "no trade"
            ),
            delta_color="off",
        )
        if take or stop:
            # What actually closed the trades, which is the only reading that says whether a
            # barrier is doing anything at all.
            bottom[1].metric(
                "Closed by a barrier",
                f"{fills.why.isin(['stop', 'take']).mean() * 100:.0f}%" if len(fills) else "—",
                (
                    f"{(fills.why == 'stop').mean() * 100:.0f}% stopped, "
                    f"{(fills.why == 'take').mean() * 100:.0f}% took profit"
                    if len(fills)
                    else ""
                ),
                delta_color="off",
            )
        bottom[2].metric(
            "Buy and hold",
            f"{(df.close.iloc[-1] / df.close.iloc[0] - 1) * 100:+.1f}%",
            "same window, same bars",
            delta_color="off",
        )
        if rule == "turns":
            st.warning(
                f"**This rule reads the future.** A turn is the highest or lowest prediction of the {turn_window} "
                f"bars on each side, so it is known only {turn_window} bars after it happened. Trading it at the "
                "turn itself is the study's diagnostic — how much of the price's turns the prediction's turns "
                "carry — and not a strategy. **At confirmation** is the version a live reader could run."
            )
        entry = {
            "band": f"long at or below **−{band:.2f}**, short at or above **+{band:.2f}**",
            "reentry": f"long when the prediction comes back above **−{band:.2f}**, short when it comes back "
            f"below **+{band:.2f}**",
            "momentum": f"long at or above **+{band:.2f}**, short at or below **−{band:.2f}**",
            "confirmed": f"long at each low of the prediction and short at each high, acted on **{turn_window} bars** "
            "after the turn, when it is confirmed",
            "turns": f"long at each low of the prediction and short at each high, found with **{turn_window} bars** "
            "on each side and acted on at the turn itself",
            "zigzag": f"short when the prediction has fallen **{h:.2f}** from its highest since the last signal, long "
            f"when it has risen **{h:.2f}** from its lowest",
            "shiryaev": f"long or short when the probability that the leg has turned reaches **{posterior:.2f}**, "
            + (
                "with a prior from the prediction's level and the leg's age"
                if prior
                else "with the same prior on every bar"
            )
            + (f", and only once the prediction has come back **{h:.2f}** from its extreme" if h else ""),
        }[rule] + (
            f"; a long kept only where the prediction is at or under **−{level:.2f}** and a short only at or over "
            f"**+{level:.2f}**, read at {'the signal' if level_at == 'alarm' else 'the extreme of the leg it closes'}, "
            + ("and a dropped signal closes the position" if level_close else "and a dropped signal is ignored")
            if level and rule in DETECTORS
            else ""
        )
        st.caption(
            f"Rule: {entry}"
            + (f", read on the mean of the last **{mean}** predictions" if mean > 1 and rule not in DETECTORS else "")
            + (", **every trade inverted**" if inverse else "")
            + (
                f", only while BTC's last daily close is under its 200-day mean — it was on "
                f"**{on.mean() * 100:.0f}%** of these bars"
                if on is not None
                else ""
            )
            + ". Each signal holds until the next one. "
            + (
                f"**{describe(stop)} {'trailing ' if trail else ''}stop**"
                + (" off the best price the hold has seen" if trail else " from the entry price")
                + f", and after it fires the rule {POLICIES[after_stop]}. "
                if stop
                else "No stop. "
            )
            + (f"**{describe(take)} take profit**, and after it the rule {POLICIES[after_take]}. " if take else "")
            + (
                f"A candle that reaches both levels is read as {'a stop' if tie_stop else 'a take profit'}. "
                if take and stop
                else ""
            )
            + (
                "A barrier fills at its level, or at the open when the bar gapped through it, so an X or a "
                "star can sit past the level it was aimed at. "
                if take or stop
                else ""
            )
            + "Every rule reads the model's raw output, drawn in the row under the label with the signals it "
            "produced; the line against the label is the same output calibrated onto the label's ±1. "
            + f"Fee {FEE * 100:.2f}% per side on every entry and exit. Filled triangles on the candles are the "
            f"rule's fills; hollow squares are the oracle's pivots, which read {leg_window} bars of future. One "
            f"pair over one window is one path: `strategy` prices the same rules over ETH, BTC and SOL and "
            f"sixteen months, where no rule earned in both halves."
        )
        if judged is not None:
            st.caption(
                f"**The signals against the prediction's own turns**, centred at {strategy.WINDOW} bars, so each is "
                f"known {strategy.WINDOW} bars after it and the last {strategy.WINDOW} bars have none yet: "
                + (
                    f"{judged['found'] * 100:.0f}% of the turns on screen got a signal on their side, a median "
                    f"{judged['delay_med']:.0f} bars late, "
                    if judged["found"]
                    else "no turn on screen got a signal on its side, "
                )
                + f"and there were {judged['false']:.2f} signals a turn against a leg still running — the hollow "
                "triangles. "
                + (
                    "Shiryaev's detector runs on the parameters fitted on v2's development folds for ETH, BTC and "
                    "SOL; over sixteen months at 0.5 the study found 79% of the turns and earned nothing: right "
                    "signals +31 bp a trade, false ones −80."
                    if rule == "shiryaev"
                    else ""
                )
                + (
                    "Over sixteen months the study's zigzag at 0.2 found 97% of the turns and earned −0.3 bp a trade."
                    if rule == "zigzag"
                    else ""
                )
            )

    st.plotly_chart(
        chart(
            df,
            pivots,
            target,
            strength,
            feats,
            fetched[0],
            "-".join(map(str, fetched)),
            normalized,
            pred,
            rule_pos if rule_pos is not None else swing_pos,
            fills if fills is not None else None,
            panels,
        ),
        use_container_width=True,
        key="chart",
    )
    if fills is not None:
        # The rule's own book, from the same ledger as the numbers above: every entry and every exit
        # that happened is one side of turnover, wherever inside a bar a barrier filled it.
        traded = np.zeros(len(rule_pnl))
        np.add.at(traded, rule_pnl.index.get_indexer(fills.entry), 1.0)
        np.add.at(traded, rule_pnl.index.get_indexer(fills.exit_time[fills.why != "open"]), 1.0)
        rule_per = pd.DataFrame(
            {"gross": rule_pnl, "traded": traded, "basket": np.log(df.close.shift(-1) / df.close).fillna(0.0)},
            index=rule_pnl.index,
        )
        st.plotly_chart(
            book(rule_per, rule_pos, fetched[0], "rule position", "buy and hold"),
            use_container_width=True,
            key="rule-book",
        )
    if swung is not None:
        # The book on the only benchmark a single pair has: holding it. Three curves and the two
        # gaps between them — gross to net is the fee, net to hold is the whole of what timing the
        # legs was worth.
        ret = np.log(df.close.shift(-1) / df.close).fillna(0.0)
        swing_per = pd.DataFrame(
            {"gross": swing_pos * ret, "traded": swing_pos.diff().fillna(swing_pos).abs(), "basket": ret}
        )
        st.plotly_chart(
            book(swing_per, swing_pos, fetched[0], "position", "buy and hold"),
            use_container_width=True,
            key="swing-book",
        )
        held = swing.trades(swing_pos.to_numpy(), df.close.to_numpy(), FEE)
        st.caption(
            f"Long or flat, one unit, entered and exited at the close of the bar the model decided "
            f"on — the oracle's own shape, so the two are priced by the same arithmetic. "
            f"{len(held)} round trips over this window"
            + (
                f", median {held.leg.median() * 100:+.2f}% and {held.bars.median():.0f} bars, "
                f"{(held.leg > 0).mean() * 100:.0f}% of them positive"
                if len(held)
                else ""
            )
            + f". {FEE * 100:.2f}% per side is charged on both fills. Stage "
            f"**{swing_card['stage']}**: the encoder is fitted on the leg label and, past that, "
            f"the position itself is trained on the money — rewarded by what it earned, charged "
            f"for every change of mind at the fee it would really pay."
        )

    st.caption(
        f"{len(df)} candles — {df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC · "
        f"{len(pivots)} pivots, median leg {pivots.amplitude.median() * 100:.2f}% · "
        f"{target.notna().sum()} labelled bars ({target.isna().sum()} unlabelled: head and tail) · "
        f"smoothing {smoothing:.2f} time / {1 - smoothing:.2f} price · "
        f"median leg significance {strength.median():.2f}, "
        f"{(strength < 1).mean() * 100:.0f}% of legs below chance"
        + (
            f" · prediction on {pred.notna().sum()} bars, "
            # Spearman on one symbol through time. It says the model is wired up, not how good it
            # is: the walk-forward in `swing` is where that is measured, over every pair at once.
            # `metrics.spearman` and not `Series.corr(method="spearman")`: the pandas call imports
            # scipy lazily, and scipy is a dev dependency only, which the deployed image
            # deliberately does not carry. This caption is what took the page down
            # with a ModuleNotFoundError.
            f"Spearman {metrics.spearman(pred, target):.2f} through time on this pair alone"
            if pred is not None
            else ""
        )
        if len(pivots)
        else f"{len(df)} candles — no pivot at window {leg_window}"
    )


# Guarded so the page can be imported without drawing it: `streamlit run` executes the script as
# `__main__`, and `tests/test_chart.py` imports it to check it survives a torch-less install.
if __name__ == "__main__":
    main()
