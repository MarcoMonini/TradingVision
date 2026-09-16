"""Streamlit page: download the candles of a crypto pair and draw them with their pivots and
the label and the feature columns computed on them.

Two labels can be drawn, and the choice is the open question of the project rather than a display
option: `remaining_excursion` is what the model is asked to predict, `swing_leg_target` is the
retrospective description a linear model on point-in-time features already reproduces (Rank IC
0.38). Reading them on the same legs is the fastest way to see the difference — the old one ramps
across each leg, the new one collapses to zero at every pivot and says how much is left.

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

from tradingvision import factor, metrics
from tradingvision.data import candles
from tradingvision.data.candles import SYMBOLS, TIMEFRAMES, get_candles
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import (
    CROSS_HORIZON,
    SMOOTHING,
    cross_sectional_return,
    leg_significance,
    remaining_excursion,
    swing_leg_target,
)
from tradingvision.features import COLUMNS, FAMILIES, LABELS, SELECTED, features
from tradingvision.normalize import CLIP, SCALE, apply, fit
from tradingvision.oracle import FEE, run


def optional(name: str) -> ModuleType | None:
    """Import `tradingvision.<name>`, or return None when torch is not installed.

    `gru` and `swing` import torch at module scope. torch is a runtime dependency now — the page
    serves predictions from the checkpoints in `models/` and `restore` is a torch call — so on a
    correct install this returns the module. The fallback stays because the failure it replaces
    was total: an image built without torch died at import with `ModuleNotFoundError: No module
    named 'torch'` and Render served nothing, losing the candles, the pivots, the label and the
    step-6 factor, none of which need torch, to two modules that had nothing to draw. Only torch
    is tolerated: any other missing module is a broken install and has to be raised.
    """
    try:
        return importlib.import_module(f"tradingvision.{name}")
    except ModuleNotFoundError as missing:
        if missing.name != "torch":
            raise
        return None


gru = optional("gru")
swing = optional("swing")

# Where a checkpoint is looked for besides `data/`. `data/` is the store a training run writes to
# and is gitignored, so nothing under it reaches the image; `models/` is tracked, which is how a
# deployed page gets a model at all — commit `gru.pt` and `swing.pt` there and Render serves the
# predictions. `data/` is read first on purpose: on a machine that has just run `gru --save` the
# fresh checkpoint is the one to draw, and a committed file silently shadowing it is the failure
# mode worth avoiding. `TRADINGVISION_MODELS` overrides the directory for a mounted disk.
MODELS = Path(os.environ.get("TRADINGVISION_MODELS") or Path(__file__).resolve().parents[3] / "models")


def saved(module: ModuleType | None) -> Path | None:
    """The checkpoint of `gru` or `swing`, from the store or from `models/`, or None if neither."""
    if module is None:
        return None
    return next((p for p in (module.CHECKPOINT, MODELS / module.CHECKPOINT.name) if p.exists()), None)


MAX_DAYS = 365
# The composite's window, as the duration it was measured as rather than as a count of bars: 2880
# bars of 15m is thirty days, and `factor` picked it on the train side of every fold. Stated in
# time so the same lens holds on whichever timeframe is on screen — a rolling deviation over
# thirty days of 1h bars and over thirty days of 15m bars are the same statistic sampled twice.
FACTOR_WINDOW = pd.Timedelta("30D")
# `TIMEFRAMES` keys as durations. Spelled out rather than parsed: `pd.Timedelta("15m")` is minutes
# but deprecated, and `pd.date_range(freq="15m")` is *months* — not an ambiguity to leave implicit.
BAR = {
    "5m": pd.Timedelta("5min"),
    "15m": pd.Timedelta("15min"),
    "1h": pd.Timedelta("1h"),
    "4h": pd.Timedelta("4h"),
    "1d": pd.Timedelta("1D"),
}
# What the saved model was fitted up to, shown so nobody reads a prediction over the train period
# as if it were out of sample. It is the walk-forward's first cut and `gru`'s own default.
TEST_START = "2025-06"

# Downloads only happen on an explicit click, and only for a (symbol, timeframe, days) triplet
# that is not already cached.
load_candles = st.cache_data(ttl=300, show_spinner="Downloading candles…")(get_candles)


@st.cache_data(show_spinner=False)
def load_pivots(close, window: int):
    """Cached on (series, window): the pivots survive every rerun that does not refetch."""
    return find_pivots(close, window)


# The two labels, keyed by the name shown in the sidebar. `PREDICTIVE` is the default and the one
# the dataset carries.
PREDICTIVE = "remaining excursion (predictive)"
RETROSPECTIVE = "swing leg position (retrospective)"
CROSS = "cross-sectional return (predictive)"
# `gru`'s name for each of them, as written into a checkpoint. No entry for `CROSS`: no model is
# trained on it yet, which is what keeps the prediction from ever being drawn against it.
TRAINED_ON = {"excursion": PREDICTIVE, "swing": RETROSPECTIVE}


@st.cache_data(show_spinner=False)
def load_target(close, window: int, label: str, smoothing: float, significance: bool):
    """Recomputed when a target control moves; the pivots underneath come from their own cache."""
    pivots = load_pivots(close, window)
    if label == PREDICTIVE:
        return remaining_excursion(close, pivots, window)
    return swing_leg_target(close, pivots, smoothing=smoothing, significance=significance)


@st.cache_data(show_spinner="Fetching the panel…")
def load_panel(timeframe: str, days: int, warmup_days: int):
    """`candles.panel` over every peer, with the model's window fetched in front of the display.

    Neither of the two quantities this page draws for the cross-sectional label exists for a
    single series. The label is a forward return *relative to the basket trading at that instant*
    and the model is a rank *inside that instant*, so both need every pair, and one fetch serves
    both. Five pairs is a thin cross-section next to the twenty the dataset carries — the level on
    screen is not the dataset's level — but the shape is the shape the model was measured on,
    which is what a chart is for.

    `warmup_days` is fetched in front of the displayed period and never trimmed here: the
    composite reads thirty days back, so without it the model would be NaN over exactly the range
    the user asked to see. The consumers crop to the display range once the windows are filled.
    """
    return candles.panel(SYMBOLS, timeframe, days + warmup_days, BAR[timeframe])


def load_cross_target(panel: dict, horizon: int):
    """`cross_sectional_return` over the whole panel, one column per pair."""
    return cross_sectional_return(panel["close"], horizon)


def load_factor(panel: dict, window: int):
    """The step-6 composite over the panel on screen — `-rank(volatility) + rank(dollar volume)`.

    This is the model, and unlike the GRU it is drawable. A cross-sectional rank is a statement
    about a symbol *against the others trading at that instant*, so a ranked GRU checkpoint has
    nothing to compute from one series and `gru.predict_frame` refuses it. The composite has the
    same requirement and the chart already meets it: the panel is fetched to build the label, and
    the same panel builds the prediction. No checkpoint, no torch, no training — two columns and
    an addition.

    Read with the cross-section's thinness in mind. `factor` measures on the twenty Binance pairs
    of the store; here there are as many pairs as Alpaca serves, which is five. The mechanism is
    the same and the level is not the project's level.
    """
    return factor.composite(panel, window)


@st.cache_data(show_spinner=False)
def load_significance(close, window: int):
    """The raw ratio behind the weighting, shown next to each pivot."""
    return leg_significance(close, load_pivots(close, window))


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """The saved GRU, kept across reruns. `cache_resource` and not `cache_data`: a torch module is
    not something to pickle and copy on every widget move."""
    return gru.restore(Path(path))


@st.cache_data(show_spinner="Predicting…")
def load_prediction(df, path: str, _mtime: float):
    """The model's output at every bar on screen. `_mtime` is in the key and unused in the body:
    retraining the checkpoint has to invalidate this, and the path alone would not say so."""
    return gru.predict_frame(*load_model(path), df)


@st.cache_resource(show_spinner=False)
def load_swing_model(path: str, _mtime: float):
    """The saved swing model. `cache_resource` for the same reason the GRU uses it, and `_mtime`
    in the key because retraining has to invalidate a checkpoint the path alone cannot date."""
    return swing.restore(Path(path))


@st.cache_data(show_spinner="Running the swing model…")
def load_swing(df, path: str, _mtime: float):
    """`label`, `logit` and `position` at every bar on screen — the model of step 7.

    Unlike the GRU this one needs no store: its inputs are computed from the candles on screen and
    its scaler is fitted on their own history, so it runs on a pair the training set never held.
    """
    return swing.predict_frame(*load_swing_model(path, _mtime), df)


@st.cache_data(show_spinner="Computing features…")
def load_features(df, window: int, columns: tuple[str, ...]):
    """Cached on (frame, window, columns): picking different columns to draw does not recompute
    them, and renaming or adding one re-keys the cache — Streamlit keys on this wrapper's source,
    which does not change when `features` does, so a running app would serve the old schema."""
    return features(df, window)[list(columns)]


def heatmap(z, title: str, unit: str = "sd", limit: float = 2.0):
    """The cross-section itself: one row per pair, time across, colour the label.

    The single-pair line above is a slice of this and cannot show what the label means, because
    the quantity is defined by the pairs that are not on screen. Here it reads directly: at any
    vertical slice, blue is what the basket is about to beat and red is what is about to beat the
    basket. A column that is all one colour is a market move, and the label has already removed
    it — so those columns are pale by construction, which is the property being drawn.

    Clipped at +/- `limit`. The tails are a few percent of the rows and they set the colour scale
    for everything else if left in; the sign and the ordering are what this view is for.

    Drawn twice when the model is on — once for what it predicted and once for what happened —
    and the two together are the only honest picture of a cross-sectional model. Reading them is
    reading whether the reds and blues line up column by column, which is what Rank IC scores.
    """
    fig = go.Figure(
        go.Heatmap(
            z=z.T.to_numpy(),
            x=z.index,
            y=[c.split("/")[0] for c in z.columns],
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            zmin=-limit,
            zmax=limit,
            colorbar=dict(title=unit, thickness=12),
            hovertemplate="%{y} %{x}<br>%{z:.2f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        height=60 + 34 * len(z.columns),
        margin=dict(l=0, r=0, t=30, b=0),
        title_text=title,
        xaxis_rangeslider_visible=False,
    )
    return fig


def book(
    per: pd.DataFrame,
    weight: pd.Series | None,
    symbol: str,
    what: str = "weight",
    benchmark: str = "basket, equal weight, long only",
) -> go.Figure:
    """What the factor's book earned over the fetched period, and this pair's weight inside it.

    The heatmaps say whether the ordering was right; this says what holding it was worth, which is
    a different question with a different answer — a Rank IC is a correlation and a book pays fees.
    Three curves, and the gaps between them are the whole reading: gross to net is the fee the
    sidebar quotes, net to basket is the reason the label is an *excess* return at all. A
    market-neutral curve that trails a rising basket has not failed, it was never long the market.

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
        # Off until asked for, alone among the three. A hedged book earns single digits where a
        # month of this market moves tens, so drawn by default the basket sets the axis and the
        # two curves this figure exists for collapse onto the zero line. The comparison is not
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
        # floor. Symmetric because the book is long and short of the same size by construction,
        # and an axis fitted to the data would draw a tilt the position does not have.
        limit = float(np.abs(weight).max()) * 1.3 or 1.0
        fig.update_yaxes(range=[-limit, limit], row=2, col=1)
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=-0.12, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    return fig


def pinned(pred: pd.Series, peers: int, symbol: str) -> str:
    """What to say when the prediction line does not move, which is not the same as broken.

    A rank across `peers` pairs takes `peers` values and no more, and the composite's ordering is
    mostly a permanent property of each symbol — measured on the twenty pairs of the store, 90.9%
    of its variance is a fixed per-symbol level. A pair at either end of that ordering therefore
    sits at the same rank on every bar, and the flat line is the model's answer rather than a
    fault in it: BTC is the largest and calmest of these five on every bar of a month, and saying
    so *is* the prediction. It is also the spec's open point 3 drawn — the part of this signal
    that is a standing tilt rather than a rotation.

    Said here because a flat line next to a jittery target reads as a bug, and the reader should
    not have to ask. Two values and not one: a cross-section that thins by a pair shifts every
    rank in the row, so a pinned pair still shows a second level wherever a peer is missing.
    """
    values = pred.dropna().nunique()
    if values > 2 or not len(pred.dropna()):
        return ""
    place = "first" if pred.dropna().iloc[-1] > 0 else "last"
    return (
        f" · the rank of {symbol.split('/')[0]} does not move over this window — {place} of "
        f"{peers} on every bar, out of the {peers} values a rank across {peers} pairs can take. "
        f"Not a flat prediction but a confident one; the pairs in the middle of the ordering rotate"
    )


def chart(
    df,
    pivots,
    target,
    significance,
    feats,
    symbol: str,
    uirevision: str,
    normalized=False,
    bounded=True,
    pred=None,
    unit: str = "",
    trades=None,
) -> go.Figure:
    # Price, then the label under it on the same x: the target is only readable against the leg it
    # describes. Volume next, it is context rather than subject, and the features under everything.
    # One row per family, never one for all of them: adx_trend_strength sits in [0, 1] and
    # log_return around 1e-3, so sharing an axis would flatten the second into the zero line.
    groups = [(fam, [c for c in cols if c in feats.columns]) for fam, cols in FAMILIES.items()]
    groups = [g for g in groups if g[1]]
    fig = make_subplots(
        rows=3 + len(groups),
        cols=1,
        shared_xaxes=True,
        # Weights, normalised by Plotly: with all seven families open a fixed share for the candles
        # would squeeze every feature row into a line.
        row_heights=[3, 1.5, 1] + [1.5] * len(groups),
        vertical_spacing=0.02,
        subplot_titles=["", "", ""] + [f.title() for f, _ in groups],
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
        # On the target's own axis, and `unit` is what says the two lines share one. For the GRU
        # over `remaining_excursion` they do by construction — the model is trained in sigma and
        # nothing rescales its output, so the gap between the lines is the error. For the
        # cross-sectional composite they do not: the label is in the dispersion of its date and
        # the model is a centred percentile, so `main` ranks *both* before passing them here and
        # the axis says so. Ranking both is not a cosmetic choice — it is the transform Rank IC
        # applies, so what the eye compares is exactly what the metric scores.
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
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p.close,
                mode="markers+text",
                marker=dict(size=8, color=color, symbol="circle"),
                text=[f"{a * 100:.1f}%" for a in p.amplitude],
                textposition=position,
                textfont=dict(size=9, color=color),
                name="pivot",
                showlegend=False,
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if trades is not None:
        # Where and when, on the price itself. The book is graded rather than on/off, so there is
        # no "entry" and "exit" to mark — only more and less, and the arrow says which way while
        # the text says the weight it moved to. Markers sit at the close of the bar the decision
        # was taken on, which is the price that decision actually paid.
        moved = trades.diff().fillna(trades)
        moved = moved[(moved != 0) & (moved.index >= df.index[0]) & (moved.index <= df.index[-1])]
        # A long/flat book has only two states, so its markers say what was done rather than what
        # the weight became; a graded one has no "in" and "out" and says the level instead.
        binary = set(trades.dropna().unique()) <= {0.0, 1.0}
        for up, color, shape, position in (
            (True, "#2ecc71", "triangle-up", "bottom center"),
            (False, "#e74c3c", "triangle-down", "top center"),
        ):
            side = moved[moved > 0] if up else moved[moved < 0]
            fig.add_trace(
                go.Scatter(
                    x=side.index,
                    # Forward filled: a pair that did not trade in a bar has no candle there, and
                    # the decision still happened at the last price the panel carried.
                    y=df.close.reindex(side.index, method="ffill"),
                    mode="markers+text",
                    marker=dict(size=9, color=color, symbol=shape),
                    text=[("buy" if up else "sell") if binary else f"{v:+.2f}" for v in trades.reindex(side.index)],
                    textposition=position,
                    textfont=dict(size=9, color=color),
                    name="buy" if up else "sell",
                    marker_size=11 if binary else 9,
                    showlegend=False,
                    hovertemplate="%{x}<br>%{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    fig.add_trace(go.Bar(x=df.index, y=df.volume, name="volume", marker_color="#888", showlegend=False), row=3, col=1)
    for row, (_, cols) in enumerate(groups, start=4):
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
    # The retrospective label lives in [-1, +1] and is read against that ceiling; the predictive
    # one is in sigma of a 24-bar walk, unbounded and fat-tailed, so it gets a free axis.
    fig.update_yaxes(
        title_text=("target vs prediction" if pred is not None else "target")
        + (unit or ("" if bounded else " (sigma)")),
        range=[-1.3, 1.3] if bounded else None,
        zeroline=True,
        zerolinecolor="#bbb",
        row=2,
        col=1,
    )
    if bounded:
        # The ceiling the pivots would reach unweighted: the gap to it is the significance discount.
        for y in (-1, 1):
            fig.add_hline(y=y, line=dict(width=1, dash="dot", color="#bbb"), row=2, col=1)
    fig.update_layout(
        height=800 + 150 * len(groups),
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


def main() -> None:
    st.set_page_config(page_title="Trading Vision", layout="wide")
    st.title("TradingVision")

    symbol = st.sidebar.selectbox("Pair", SYMBOLS)
    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES), index=1)
    days = st.sidebar.slider("History (days)", 1, MAX_DAYS, 30)
    label = st.sidebar.radio("Label", [PREDICTIVE, RETROSPECTIVE, CROSS], help="what the model is asked to output")
    # Both controls shape the retrospective label only: the predictive one has no blend to weight
    # (a degenerate leg goes nowhere, so it scores near zero by itself).
    retrospective = label == RETROSPECTIVE
    smoothing = SMOOTHING
    significance = True
    # Counted in bars of the timeframe on screen, like every other window here. 48 bars of 15m is
    # the 12h the horizon sweep pointed at; on another timeframe the same number is another period,
    # which is why it is a control and not a constant.
    horizon = CROSS_HORIZON
    if label == CROSS:
        horizon = st.sidebar.slider("Forward horizon (bars)", 4, 288, CROSS_HORIZON, 4)
    if retrospective:
        # Unlike the window and the fee, this one is explicitly a tunable: 0.7 is a starting value.
        # 1.0 is a pure time ramp between pivots, 0.0 follows price alone.
        smoothing = st.sidebar.slider("Target smoothing (time weight)", 0.0, 1.0, SMOOTHING, 0.05)
        # Off shows the flat +/-1 labelling, which the weighting is meant to be read against.
        significance = st.sidebar.toggle("Weight pivots by leg significance", value=True)
    # The feature windows all derive from the extrema window, so they follow it rather than being
    # tuned here; the per-branch values are still to be measured.
    # A toggle and not a 28-item checklist: after the selection there are only two sets anyone
    # wants to look at, and the reduced one is what the model actually trains on.
    full = st.sidebar.toggle(
        f"All {len(COLUMNS)} candidates",
        value=False,
        help=f"off: the {len(SELECTED)} that survived the feature selection — see the schema doc",
    )
    picked = COLUMNS if full else [c for c in COLUMNS if c in SELECTED]
    normalized = st.sidebar.toggle("Normalized", value=True, help="clip((x - median) / IQR, ±5) × 0.1")

    # The GRU, drawn over the label it was trained on. Three conditions, and each of them is a
    # reason and not a guard: there has to be a checkpoint (`gru --features all --save` writes
    # one), the chart has to be on the branch the model reads, and the label on screen has to be
    # the one the model predicts — over the retrospective label the two lines share an axis
    # without sharing a unit.
    model_at = saved(gru)
    checkpoint = load_model(str(model_at))[1] if model_at else None
    branch = checkpoint["branches"][0] if checkpoint else None
    # Which of the two labels this checkpoint was fitted on. Older files predate the choice and
    # were all fitted on the predictive one.
    # `.get` and not `[...]`: a checkpoint written by `gru --label cross` names a label this page
    # has no line for, and the old subscript turned that into a KeyError on import of the sidebar.
    trained_on = TRAINED_ON.get(checkpoint.get("label", "excursion")) if checkpoint else None
    predicting = False
    if gru is None:
        st.sidebar.caption("This install has no torch, so no GRU prediction. The factor below needs none.")
    elif not model_at:
        st.sidebar.caption(
            f"No model saved. `python -m tradingvision.gru --features all --save`, then copy it into "
            f"`{MODELS.name}/` to deploy it."
        )
    elif trained_on is None:
        st.sidebar.caption("The saved GRU reads cross-sectional ranks, which one pair cannot supply.")
    elif timeframe == branch and label == trained_on:
        predicting = st.sidebar.toggle("GRU prediction", value=True, help=f"{model_at.name}, trained to {TEST_START}")
    else:
        # Never drawn against a label it was not trained on: the two live on different scales, and
        # two lines sharing an axis without sharing a unit is the one reading that misleads.
        st.sidebar.caption(f"GRU prediction needs the **{branch}** timeframe and the **{trained_on}** label.")

    # The swing model of step 7 — the one that trades. It is drawn against the retrospective
    # label because that is the label it predicts, and only on the timeframe it was fitted on:
    # its inputs are windows of that bar and nothing rescales them between one bar and another.
    swing_at = saved(swing)
    swing_card = load_swing_model(str(swing_at), swing_at.stat().st_mtime)[1] if swing_at else None
    swinging = False
    if swing is None:
        st.sidebar.caption("This install has no torch, so no swing trades.")
    elif not swing_at:
        st.sidebar.caption(
            f"No swing model. `python -m tradingvision.swing --save`, then copy it into `{MODELS.name}/`."
        )
    elif label != RETROSPECTIVE:
        st.sidebar.caption(f"The swing model predicts the **{RETROSPECTIVE}** label.")
    elif timeframe != swing_card["timeframe"]:
        st.sidebar.caption(f"The swing model reads **{swing_card['timeframe']}** candles.")
    else:
        swinging = st.sidebar.toggle(
            "Swing trades",
            value=True,
            help=f"{swing_at.name}, stage {swing_card['stage']}, trained to {swing_card['test_start']}",
        )

    # The composite of step 6, which *is* drawable where the GRU is not: it reads the panel this
    # page already fetches for the label, and there is no checkpoint to be missing.
    factoring = False
    if label == CROSS:
        factoring = st.sidebar.toggle(
            "Factor prediction",
            value=True,
            help=f"-rank(volatility) + rank(dollar volume) over {FACTOR_WINDOW.days} days — the step 6 model",
        )
    else:
        st.sidebar.caption(f"The factor predicts the **{CROSS}** label.")

    request = (symbol, timeframe, days)
    if st.sidebar.button("Fetch candles", type="primary", use_container_width=True):
        st.session_state.fetched = (request, load_candles(*request))

    # Not inputs: both were calibrated in oracle.py and changing them here would show pivots the
    # dataset does not contain.
    st.sidebar.divider()
    st.sidebar.caption(
        f"**Extrema window** &nbsp; {EXTREMA_WINDOW} bars — calibrated, see the spec  \n"
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
    pivots = load_pivots(df.close, EXTREMA_WINDOW)
    peers, realised, predicted, factor_pred, skill, unit = 0, None, None, None, None, ""
    per, weight, decision = None, None, ""
    if label == CROSS:
        panel = load_panel(fetched[1], fetched[2], FACTOR_WINDOW.days)
        realised = load_cross_target(panel, horizon)
        peers = len(realised.columns)
        target = (realised[fetched[0]] if fetched[0] in realised else pd.Series(np.nan, index=realised.index)).rename(
            "target"
        )
        target = target.reindex(df.index)
        if factoring and peers:
            # The window as a count of this timeframe's bars, from the duration it was measured
            # as. Two at the floor so a rolling deviation is defined at all.
            bars = max(2, round(FACTOR_WINDOW / BAR[fetched[1]]))
            predicted = load_factor(panel, bars)
            # Both sides ranked inside their own instant before anything is drawn or scored. That
            # is the transform Rank IC applies, and it is the only way the two lines share a unit:
            # the label is in the dispersion of its date, the composite is a sum of two centred
            # percentiles, and neither converts to the other.
            realised, predicted = factor.cross_rank(realised), factor.cross_rank(predicted)
            target = realised[fetched[0]].reindex(df.index).rename("target")
            factor_pred = predicted[fetched[0]].reindex(df.index).rename("prediction")
            unit = " (cross-sectional rank)"
            # Scored over the panel and over the range on screen — not `pred.corr(target)` down in
            # the caption, which is a correlation through time on one pair and blind to the only
            # thing this label says. The error bar is over non-overlapping blocks of the horizon,
            # because adjacent dates share all but one bar of their label.
            on_screen = realised.index.isin(df.index)
            per_date = factor.rank_ic(predicted, realised, on_screen)
            skill = metrics.blocked(per_date, horizon * BAR[fetched[1]]) if len(per_date) else None
            # The book, priced on the panel's own closes. One decision an hour is the clock the
            # study measured on and the stride `dataset` samples at; on a timeframe coarser than
            # an hour the bar is the clock, because there is nothing finer to decide on.
            step = max(1, round(factor.DECISION / BAR[fetched[1]]))
            decision = "hour" if step > 1 else fetched[1]
            at = factor.hourly(predicted.index, BAR[fetched[1]], step * BAR[fetched[1]])
            # `predicted` is already ranked and `weighted` ranks again — a rank of a rank is the
            # same ordering, so this is the book the study prices off the raw composite.
            pos = factor.weighted(predicted, at, factor.TOL)
            # Priced from the first decision the composite exists on and not from the start of the
            # warm-up: the rows in front of it are flat by construction and would only dilute the
            # period the numbers are quoted over. The opening trade is charged at that first row.
            live = pos.index[(pos != 0).any(axis=1)]
            if len(live):
                pos = pos.loc[live[0] :]
                per = factor.pnl(pos, panel["close"], step)
                weight = pos[fetched[0]] if fetched[0] in pos else None
    else:
        target = load_target(df.close, EXTREMA_WINDOW, label, smoothing, significance)
    swung = load_swing(df, str(swing_at), swing_at.stat().st_mtime) if swinging else None
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
    strength = load_significance(df.close, EXTREMA_WINDOW)
    feats = load_features(df, EXTREMA_WINDOW, tuple(COLUMNS))[picked]
    # The two models never draw together: each predicts a different label, and the sidebar only
    # offers whichever one the label on screen belongs to.
    pred = factor_pred if factor_pred is not None else None
    if pred is None and swung is not None:
        # Calibrated back onto the label's own range on the way out of the model, so the two lines
        # in the second row share a unit as well as an axis. The map is fitted on the train period
        # and stored in the checkpoint; it is monotone, so it moved no decision on the way here.
        pred = swung.label.rename("prediction")
    if pred is None and predicting:
        pred = load_prediction(df, str(model_at), model_at.stat().st_mtime)
    if normalized and len(feats.columns):
        # Fitted on the window on screen, which is what a chart can do and not what the dataset
        # does: there the statistics come from the train period alone.
        alive = feats.columns[(feats.quantile(0.75) - feats.quantile(0.25)) > 0]
        feats = apply(feats[alive], fit(feats[alive]))

    # Oracle: what a perfect-hindsight trader would have made on this window over this period.
    stats = run(df.close, EXTREMA_WINDOW, FEE, pivots=pivots)
    columns = st.columns(4 if skill else 2)
    columns[0].metric("Oracle net return", f"{stats['net_return'] * 100:,.1f}%", f"{stats['trades']} legs")
    columns[1].metric("Avg gross leg", f"{stats['gross_trade_pct']:.2f}%", f"{stats['win_rate'] * 100:.0f}% above fees")
    if skill:
        # The error bar is the headline and not a footnote. Over this window and five pairs the
        # standard error is wide enough to contain zero, and a Rank IC quoted without it would
        # read as a result — the project measures 0.110 on twenty pairs over four folds, and this
        # panel cannot say anything that precise.
        columns[2].metric("Rank IC on screen", f"{skill['mean']:+.3f}", f"± {skill['se']:.3f} (se)")
        columns[3].metric("t on blocks", f"{skill['t']:+.2f}", f"{skill['blocks']} independent blocks")
    if per is not None:
        # What the ordering above was worth to hold, which the Rank IC does not say. Net of the
        # fee on the left, the basket next to it: a hedged book is not competing with the market,
        # and quoting its return without what the market did over the same hours invites the
        # comparison anyway — better to draw it than to leave it implied.
        net = per.gross - per.traded * FEE
        moves = int((weight.diff().fillna(weight) != 0).sum()) if weight is not None else 0
        row = st.columns(3)
        row[0].metric(
            "Book net return", f"{np.expm1(net.sum()) * 100:+.2f}%", f"gross {np.expm1(per.gross.sum()) * 100:+.2f}%"
        )
        row[1].metric(
            "Basket, same period",
            f"{np.expm1(per.basket.sum()) * 100:+.2f}%",
            f"{peers} pairs, equal weight, long only",
        )
        row[2].metric(
            f"Trades on {fetched[0].split('/')[0]}", f"{moves}", f"{per.traded.sum():.2f} units traded per pair"
        )

    if swung is not None:
        # What the model made on the window on screen, against the two oracles. The first is the
        # hindsight one the page has always shown; the second is the same oracle filling `W` bars
        # later, which is the earliest a pivot of a centred window can be known to anybody and
        # therefore the only one of the two a causal rule could reach. Measured over twenty pairs
        # the second is 7% of the first, and quoting the model against the first alone would
        # describe a 7x gap that no model can close.
        span = float((df.index[-1] - df.index[0]) / swing.YEAR)
        got = swing.price(swing_pos.to_numpy(), df.close.to_numpy(), span, FEE)
        reachable = run(df.close, EXTREMA_WINDOW, FEE, pivots=pivots, lag=EXTREMA_WINDOW)
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
            f"{np.expm1(near) * 100:+.0f}% filling {EXTREMA_WINDOW} bars after each pivot",
        )
        row[3].metric(
            "Of the hindsight oracle",
            f"{mine / best * 100:+.1f}%" if best else "—",
            f"{np.expm1(best) * 100:,.0f}% — it reads the future",
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
            retrospective or bool(unit),
            pred,
            unit,
            weight if weight is not None else swing_pos,
        ),
        use_container_width=True,
        key="chart",
    )
    if swung is not None:
        # The same figure the factor book gets, on the only benchmark a single pair has: holding
        # it. Three curves and the two gaps between them — gross to net is the fee, net to hold is
        # the whole of what timing the legs was worth.
        ret = np.log(df.close.shift(-1) / df.close).fillna(0.0)
        per = pd.DataFrame(
            {"gross": swing_pos * ret, "traded": swing_pos.diff().fillna(swing_pos).abs(), "basket": ret}
        )
        st.plotly_chart(
            book(per, swing_pos, fetched[0], "position", "buy and hold"),
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

    if realised is not None and peers:
        # Prediction above, outcome below, on one x. The line two rows up is a slice of these and
        # cannot show what either quantity means, because both are defined by the pairs that are
        # not on it. Side by side they read directly: where the two panels agree column by column
        # the model was right about the ordering, and that agreement *is* the Rank IC above.
        limit, u = (0.5, "rank") if unit else (2.0, "sd")
        # Both cropped to the range on screen. The panel reaches `FACTOR_WINDOW` further back to
        # fill the model's window, and drawing that warm-up would put the two panels on different
        # x — which is the one thing that makes them unreadable next to each other.
        window = realised.index.isin(df.index)
        if predicted is not None:
            st.plotly_chart(
                heatmap(predicted[window], "prediction — the factor's ordering, per pair", u, limit),
                use_container_width=True,
                key="predicted",
            )
        st.plotly_chart(
            heatmap(
                realised[window],
                f"outcome — excess return over the next {horizon} bars, per pair",
                u,
                limit,
            ),
            use_container_width=True,
            key="heatmap",
        )
    if per is not None:
        st.plotly_chart(book(per, weight, fetched[0]), use_container_width=True, key="book")
        # What one swap in the ordering moves a weight by, which is the unit the tolerance is
        # quoted in and the number that does not survive a thin cross-section.
        swap = factor.swap(peers)
        span = (pos.index[-1] - pos.index[0]) / factor.YEAR
        st.caption(
            f"Rank-weighted book, one unit of gross notional per pair, dollar neutral inside each "
            f"decision and rebalanced only when a target weight moves more than {factor.TOL} — the "
            f"rule and the tolerance `factor` picked on the train side of all four folds, where it "
            f"made +0.238 a year against +0.179 for the band. One decision per {decision} over "
            f"{len(per)} of them, {FEE * 100:.2f}% per side charged on every unit traded, the "
            f"opening trade included."
            + (
                f" Read the turnover with the width of the cross-section in mind: across "
                f"{peers} pairs one swap in the ordering already moves a weight by {swap:.2f}, "
                f"more than the {factor.TOL} tolerance, so this book rebalances on every swap — "
                f"{per.traded.sum() / span:.0f} units a year, against 6.5 for the same rule on "
                f"the twenty pairs it was measured on, where {factor.TOL} is four swaps wide. "
                f"The rule is the one that was chosen; the panel under it is a quarter of the "
                f"width, and thinness costs fees before it costs anything else."
                if swap > factor.TOL
                else ""
            )
        )
    st.caption(
        f"{len(df)} candles — {df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC · "
        f"{len(pivots)} pivots, median leg {pivots.amplitude.median() * 100:.2f}% · "
        f"{target.notna().sum()} labelled bars ({target.isna().sum()} unlabelled: head and tail) · "
        + (
            f"smoothing {smoothing:.2f} time / {1 - smoothing:.2f} price · "
            f"median leg significance {strength.median():.2f}, "
            f"{(strength < 1).mean() * 100:.0f}% of legs below chance"
            if retrospective
            else (
                f"{horizon}-bar forward return in excess of {peers} pairs"
                + (
                    ", as its rank inside each instant — the transform Rank IC scores, "
                    "and the only unit the two lines share"
                    if unit
                    else (
                        f", in cross-sectional sd · median |target| {target.abs().median():.2f}, "
                        f"99th percentile {target.abs().quantile(0.99):.1f} · "
                        f"|target| ~0.35 is roughly the {FEE * 200:.2f}% round trip"
                    )
                )
                if label == CROSS
                else f"median |target| {target.abs().median():.2f} sigma, "
                f"99th percentile {target.abs().quantile(0.99):.1f} sigma"
            )
        )
        + (
            f" · factor over {round(FACTOR_WINDOW / BAR[fetched[1]])} bars "
            f"({FACTOR_WINDOW.days}d) on {pred.notna().sum()} bars — the Rank IC above is the "
            f"metric, taken per timestamp across the {peers} pairs" + pinned(pred, peers, fetched[0])
            if predicted is not None
            else (
                f" · prediction on {pred.notna().sum()} bars, "
                # Spearman on one symbol through time, which is not the Rank IC of the spec — that
                # one is taken per timestamp across the twenty pairs, and is blind to the common
                # level this one reads. It says the model is wired up, not how good it is.
                f"Spearman {pred.corr(target, method='spearman'):.2f} through time on this pair alone"
                if pred is not None
                else ""
            )
        )
        if len(pivots)
        else f"{len(df)} candles — no pivot at window {EXTREMA_WINDOW}"
    )


# Guarded so the page can be imported without drawing it: `streamlit run` executes the script as
# `__main__`, and `tests/test_chart.py` imports it to check it survives a torch-less install.
if __name__ == "__main__":
    main()
