"""Streamlit page: download the candles of a crypto pair and draw them with their pivots and
the swing leg target computed on them.

streamlit run src/tradingvision/app/chart.py
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingvision.data.candles import SYMBOLS, TIMEFRAMES, get_candles
from tradingvision.data.pivots import EXTREMA_WINDOW, find_pivots
from tradingvision.data.target import SMOOTHING, leg_significance, swing_leg_target
from tradingvision.oracle import FEE, run

MAX_DAYS = 365

# Downloads only happen on an explicit click, and only for a (symbol, timeframe, days) triplet
# that is not already cached.
load_candles = st.cache_data(ttl=300, show_spinner="Downloading candles…")(get_candles)


@st.cache_data(show_spinner=False)
def load_pivots(close, window: int):
    """Cached on (series, window): the pivots survive every rerun that does not refetch."""
    return find_pivots(close, window)


@st.cache_data(show_spinner=False)
def load_target(close, window: int, smoothing: float, significance: bool):
    """Recomputed when a target control moves; the pivots underneath come from their own cache."""
    return swing_leg_target(close, load_pivots(close, window), smoothing=smoothing, significance=significance)


@st.cache_data(show_spinner=False)
def load_significance(close, window: int):
    """The raw ratio behind the weighting, shown next to each pivot."""
    return leg_significance(close, load_pivots(close, window))


def chart(df, pivots, target, significance, symbol: str, uirevision: str) -> go.Figure:
    # Price, then the label under it on the same x: the target is only readable against the leg it
    # describes. Volume last, it is context rather than subject.
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.27, 0.18], vertical_spacing=0.02)
    fig.add_trace(
        go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name=symbol), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=target.index,
            y=target,
            mode="lines",
            line=dict(width=1.5, color="#3498db"),
            name="swing_leg_target",
            connectgaps=False,  # the unlabelled head and tail must read as gaps, not as a line
            hovertemplate="%{x}<br>target %{y:.2f}<extra></extra>",
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
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Bar(x=df.index, y=df.volume, name="volume", marker_color="#888"), row=3, col=1)
    fig.update_yaxes(title_text="target", range=[-1.3, 1.3], zeroline=True, zerolinecolor="#bbb", row=2, col=1)
    # The ceiling the pivots would reach unweighted: the gap to it is the significance discount.
    for y in (-1, 1):
        fig.add_hline(y=y, line=dict(width=1, dash="dot", color="#bbb"), row=2, col=1)
    fig.update_layout(
        height=800,
        showlegend=False,
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
    # Unlike the window and the fee, this one is explicitly a tunable: 0.7 is a starting value.
    # 1.0 is a pure time ramp between pivots, 0.0 follows price alone.
    smoothing = st.sidebar.slider("Target smoothing (time weight)", 0.0, 1.0, SMOOTHING, 0.05)
    # Off shows the flat +/-1 labelling, which is what the weighting is meant to be read against.
    significance = st.sidebar.toggle("Weight pivots by leg significance", value=True)

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
    target = load_target(df.close, EXTREMA_WINDOW, smoothing, significance)
    strength = load_significance(df.close, EXTREMA_WINDOW)

    # Oracle: what a perfect-hindsight trader would have made on this window over this period.
    stats = run(df.close, EXTREMA_WINDOW, FEE, pivots=pivots)
    a, b = st.columns(2)
    a.metric("Oracle net return", f"{stats['net_return'] * 100:,.1f}%", f"{stats['trades']} legs")
    b.metric("Avg gross leg", f"{stats['gross_trade_pct']:.2f}%", f"{stats['win_rate'] * 100:.0f}% above fees")

    st.plotly_chart(
        chart(df, pivots, target, strength, fetched[0], "-".join(map(str, fetched))),
        use_container_width=True,
        key="chart",
    )
    st.caption(
        f"{len(df)} candles — {df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC · "
        f"{len(pivots)} pivots, median leg {pivots.amplitude.median() * 100:.2f}% · "
        f"{target.notna().sum()} labelled bars ({target.isna().sum()} unlabelled: head and tail) · "
        f"smoothing {smoothing:.2f} time / {1 - smoothing:.2f} price · "
        f"median leg significance {strength.median():.2f}, "
        f"{(strength < 1).mean() * 100:.0f}% of legs below chance"
        if len(pivots)
        else f"{len(df)} candles — no pivot at window {EXTREMA_WINDOW}"
    )


main()
