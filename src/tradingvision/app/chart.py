"""Streamlit page: download the candles of a crypto pair and draw them with their pivots and
the label and the feature columns computed on them.

Two labels can be drawn, and the choice is the open question of the project rather than a display
option: `remaining_excursion` is what the model is asked to predict, `swing_leg_target` is the
retrospective description a linear model on point-in-time features already reproduces (Rank IC
0.38). Reading them on the same legs is the fastest way to see the difference — the old one ramps
across each leg, the new one collapses to zero at every pivot and says how much is left.

streamlit run src/tradingvision/app/chart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingvision import gru
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

MAX_DAYS = 365
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


@st.cache_data(show_spinner=False)
def load_cross_target(symbol: str, timeframe: str, days: int, horizon: int):
    """`cross_sectional_return` for one pair, and the peers it had to be measured against.

    This label does not exist for a single series: it is the forward return of a symbol *relative
    to the basket trading at the same instant*, so drawing it means fetching all of `SYMBOLS` and
    keeping one column. Five pairs is a thin cross-section next to the twenty the dataset carries
    — the level is not the dataset's level and should not be read as one — but the shape on screen
    is the shape the model would be trained on, which is what the chart is for.

    Peers that Alpaca has no data for simply do not join the panel; the label falls back to NaN
    wherever fewer than three of them do.
    """
    closes = {}
    for peer in SYMBOLS:
        bars = get_candles(peer, timeframe, days)
        if not bars.empty:
            closes[peer] = bars.close
    panel = pd.DataFrame(closes).sort_index()
    z = cross_sectional_return(panel, horizon)
    return (z[symbol] if symbol in z else pd.Series(np.nan, index=panel.index)).rename("target"), len(panel.columns)


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


@st.cache_data(show_spinner="Computing features…")
def load_features(df, window: int, columns: tuple[str, ...]):
    """Cached on (frame, window, columns): picking different columns to draw does not recompute
    them, and renaming or adding one re-keys the cache — Streamlit keys on this wrapper's source,
    which does not change when `features` does, so a running app would serve the old schema."""
    return features(df, window)[list(columns)]


def chart(
    df, pivots, target, significance, feats, symbol: str, uirevision: str, normalized=False, bounded=True, pred=None
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
        # On the target's own axis, because it is the target's own unit: the model is trained on
        # `remaining_excursion` in sigma and nothing rescales its output. The two lines are
        # directly comparable, and the gap between them is the error.
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
        title_text=("target vs prediction" if pred is not None else "target") + ("" if bounded else " (sigma)"),
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
    model_at = gru.CHECKPOINT if gru.CHECKPOINT.exists() else None
    checkpoint = load_model(str(model_at))[1] if model_at else None
    branch = checkpoint["branches"][0] if checkpoint else None
    # Which of the two labels this checkpoint was fitted on. Older files predate the choice and
    # were all fitted on the predictive one.
    trained_on = TRAINED_ON[checkpoint.get("label", "excursion")] if checkpoint else None
    predicting = False
    if not model_at:
        st.sidebar.caption("No model saved. `python -m tradingvision.gru --features all --save`")
    elif timeframe == branch and label == trained_on:
        predicting = st.sidebar.toggle("GRU prediction", value=True, help=f"{model_at.name}, trained to {TEST_START}")
    else:
        # Never drawn against a label it was not trained on: the two live on different scales, and
        # two lines sharing an axis without sharing a unit is the one reading that misleads.
        st.sidebar.caption(f"GRU prediction needs the **{branch}** timeframe and the **{trained_on}** label.")

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
    peers = 0
    if label == CROSS:
        target, peers = load_cross_target(fetched[0], fetched[1], fetched[2], horizon)
        target = target.reindex(df.index)
    else:
        target = load_target(df.close, EXTREMA_WINDOW, label, smoothing, significance)
    strength = load_significance(df.close, EXTREMA_WINDOW)
    feats = load_features(df, EXTREMA_WINDOW, tuple(COLUMNS))[picked]
    pred = load_prediction(df, str(model_at), model_at.stat().st_mtime) if predicting else None
    if normalized and len(feats.columns):
        # Fitted on the window on screen, which is what a chart can do and not what the dataset
        # does: there the statistics come from the train period alone.
        alive = feats.columns[(feats.quantile(0.75) - feats.quantile(0.25)) > 0]
        feats = apply(feats[alive], fit(feats[alive]))

    # Oracle: what a perfect-hindsight trader would have made on this window over this period.
    stats = run(df.close, EXTREMA_WINDOW, FEE, pivots=pivots)
    a, b = st.columns(2)
    a.metric("Oracle net return", f"{stats['net_return'] * 100:,.1f}%", f"{stats['trades']} legs")
    b.metric("Avg gross leg", f"{stats['gross_trade_pct']:.2f}%", f"{stats['win_rate'] * 100:.0f}% above fees")

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
            retrospective,
            pred,
        ),
        use_container_width=True,
        key="chart",
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
            else f"{horizon}-bar forward return in excess of {peers} pairs · "
            f"median |target| {target.abs().median():.2f} sigma, "
            f"99th percentile {target.abs().quantile(0.99):.1f} sigma · "
            f"|target| 0.5 is roughly the {FEE * 200:.2f}% round trip"
            if label == CROSS
            else f"median |target| {target.abs().median():.2f} sigma, "
            f"99th percentile {target.abs().quantile(0.99):.1f} sigma"
        )
        + (
            f" · prediction on {pred.notna().sum()} bars, "
            # Spearman on one symbol through time, which is not the Rank IC of the spec — that one
            # is taken per timestamp across the twenty pairs, and is blind to the common level
            # this one reads. It says the model is wired up, not how good it is.
            f"Spearman {pred.corr(target, method='spearman'):.2f} through time on this pair alone"
            if pred is not None
            else ""
        )
        if len(pivots)
        else f"{len(df)} candles — no pivot at window {EXTREMA_WINDOW}"
    )


main()
