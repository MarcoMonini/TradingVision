"""Streamlit page: what the candles of one pair look like, read from the store and nothing else.

Descriptive statistics of the market itself, before any label or model. It runs locally on the
Parquet store under `data/` (`data.binance`: 5m candles, aggregated to the timeframe picked with
`binance.load`) and downloads nothing: a pair is offered only if its file is there. It is not
deployed — the image carries no store — and it has no sidebar: the controls sit above what they
move.

A candle's change is open to close, `close / open - 1`, in percent. The dumps trade around the
clock, so a candle opens where the one before closed on almost every bar, and close to close would
give the same numbers.

**The mean is the statistic near zero, not the standard deviation.** Over all candles the mean
change is the drift, a few thousandths of a percent; the standard deviation is the size of a typical
candle — a 50% annual volatility is 0.27% a 15m candle — and is never near zero. The up and down
candles each get their own pair: the mean of the up candles is how far an up candle goes on average,
their standard deviation how much one up candle differs from another. Split at zero, a symmetric
distribution gives the two sides mirror values, and for a normal one the standard deviation of a
side is sqrt(1 - 2/pi) = 0.60 of the whole (asserted in `_selfcheck`). Candles with open equal to
close count in "all" and in neither side.

**The histogram counts every candle loaded.** Its axis is symmetric around zero and spans the larger
of the 0.1th and 99.9th percentiles in absolute value, so a handful of crashes cannot flatten the
bell into a spike; the candles beyond it are counted in its caption, never dropped silently. One bin
is centred on zero. Expect it centred on zero but not normal: crypto returns have a sharper peak and
heavier tails than a normal of the same deviation. The page draws no normal over it.

Under the candles, in a chart of its own over the same period, the same candles each started from
zero: open, high, low and close as changes from the candle's own open, in percent. The price level
and the path between candles go, and what is left is each candle's shape — its body is the change
the histogram counts, its wicks how far it went either way before closing. A control right above
that chart, and moving nothing else, subtracts a statistic from each of its candles (`offset`),
toward zero: the standard deviation of all candles from every candle, each side's standard deviation
from its candles, or each side's mean from its candles. An up candle moves down by it, a down candle
up, a flat one stays, and the whole candle moves, so its shape is kept and its close reads what the
candle did beyond the statistic. The statistics are the ones the page shows, over the whole history
and not over the period. The two charts are separate so that the control sits on the one it moves;
they share a left margin so their time axes line up, but a zoom on one does not move the other. The
period is read by these two alone, and its default, the last 30 days, is there to keep the drawing
light: the numbers and the histogram read the whole history.

    uv run streamlit run src/tradingvision/app/decomposer.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradingvision.data import binance

TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
TIMEFRAME = "15m"  # v2's, the timeframe the project studies
PERIOD = pd.Timedelta(days=30)  # the candle charts' default window
BINS = 201  # odd, so one bin is centred on zero
TAIL = 0.001  # the histogram's axis ends at this quantile on either side, whichever is further out
SIDES = {"all": "All candles", "up": "Up candles", "down": "Down candles"}
# What the lower candle chart can take off each candle, named as the statistics at the top of the page
# are, so a choice points at the numbers it subtracts without repeating them.
OFFSETS = {
    "none": "Nothing",
    "std": "Std dev of all candles",
    "side std": "Std dev of up / down candles",
    "side mean": "Mean of up / down candles",
}


def store() -> Path:
    """The store, looked up at call time so a test can point `binance.STORE` elsewhere."""
    return binance.STORE


def pairs(where: Path) -> list[str]:
    """The symbols with a 5m file in the store: the training universe's order first, then the rest
    alphabetically. The futures files (`*USDT-futures-15m.parquet`) are not candles and stay out."""
    held = {p.name.removesuffix("USDT-5m.parquet") for p in where.glob("*USDT-5m.parquet")}
    return [s for s in binance.SYMBOLS if s in held] + sorted(held - set(binance.SYMBOLS))


@st.cache_data(show_spinner="Reading the store…")
def candles(symbol: str, timeframe: str, mtime: float) -> pd.DataFrame:
    """One pair's OHLC at `timeframe`. `mtime` is the file's and is unused in the body: it is in the
    cache key so that updating the store re-reads it. Not `_mtime` — Streamlit leaves a parameter
    whose name starts with an underscore out of the key."""
    return binance.load(symbol, timeframe, store=store())[["open", "high", "low", "close"]]


def change(df: pd.DataFrame) -> pd.Series:
    """Each candle's change from open to close, in percent."""
    return (df.close / df.open - 1) * 100


def from_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Each candle on its own open: open, high, low and close as percent changes from the open, so every
    candle starts at zero rather than where the one before closed. Its close is `change`."""
    return df[["open", "high", "low", "close"]].div(df.open, axis=0).sub(1).mul(100)


def sides(pct: pd.Series) -> pd.DataFrame:
    """Mean and standard deviation of the change over all candles, the up ones and the down ones."""
    groups = {"all": pct, "up": pct[pct > 0], "down": pct[pct < 0]}
    return pd.DataFrame({k: {"mean": v.mean(), "std": v.std()} for k, v in groups.items()}).T


def offset(pct: pd.Series, stats: pd.DataFrame, how: str) -> pd.Series:
    """What `how` takes off each candle, signed so that subtracting it moves the candle toward zero:
    positive on an up candle, negative on a down one, zero on a flat one. `stats` is `sides`' table;
    the mean of the down candles is negative already, their standard deviation is not."""
    side = np.sign(pct)
    up, down = {
        "none": (0.0, 0.0),
        "std": (stats.loc["all", "std"], -stats.loc["all", "std"]),
        "side std": (stats.loc["up", "std"], -stats.loc["down", "std"]),
        "side mean": (stats.loc["up", "mean"], stats.loc["down", "mean"]),
    }[how]
    return pd.Series(np.select([side > 0, side < 0], [up, down], 0.0), index=pct.index)


def histogram(pct: pd.Series, bins: int = BINS, tail: float = TAIL) -> tuple[np.ndarray, np.ndarray, int]:
    """The counts of `pct` in `bins` equal bins over ±R, their edges, and how many candles lie beyond
    ±R. R is the larger of the `tail` and `1 - tail` quantiles in absolute value."""
    reach = float(np.abs(pct.quantile([tail, 1 - tail])).max()) or float(np.abs(pct).max()) or 1.0
    edges = np.linspace(-reach, reach, bins + 1)
    counts, _ = np.histogram(pct, edges)
    return counts, edges, int(len(pct) - counts.sum())


# The candle charts' left margin, fixed and wide enough for either axis's labels: the two are separate
# figures, and with Plotly sizing each margin to its own labels ("110k" against "-0.5%") their time
# axes would start at different places and a candle would not sit above its own shape.
MARGIN = 64


def candle_figure(df: pd.DataFrame, uirevision: str, height: int = 520, **candle) -> go.Figure:
    """A candlestick chart of `df`; `candle` goes to the trace."""
    fig = go.Figure(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, **candle))
    fig.update_yaxes(automargin=False)
    fig.update_layout(
        height=height,
        margin=dict(l=MARGIN, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        # Keeps zoom and pan across reruns until the pair, the timeframe or the period changes.
        uirevision=uirevision,
    )
    return fig


def zero_figure(df: pd.DataFrame, uirevision: str, minus: pd.Series | None = None) -> go.Figure:
    """The candles of `df` each from zero (`from_zero`), less `minus` (an `offset`) when given."""
    zero = from_zero(df)
    if minus is not None:
        zero = zero.sub(minus.reindex(zero.index), axis=0)
    fig = candle_figure(zero, uirevision, height=340, yhoverformat="+.3f")
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#888")
    return fig


def histogram_figure(counts: np.ndarray, edges: np.ndarray) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2,
            y=counts,
            width=np.diff(edges),
            marker=dict(color="#3498db", line=dict(width=0)),
            customdata=np.c_[edges[:-1], edges[1:]],
            hovertemplate="%{customdata[0]:+.3f}% to %{customdata[1]:+.3f}%<br>%{y:,} candles<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        bargap=0,  # contiguous bins: a gap would read as a range no candle fell in
        xaxis_title="change from open to close, %",
        yaxis_title="candles",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Decomposer", layout="wide", initial_sidebar_state="collapsed")
    st.title("Decomposer")

    where = store()
    held = pairs(where)
    if not held:
        st.info(f"No 5m candles under `{where}`. Fill the store first: `uv run python -m tradingvision.data.binance`.")
        return
    left, right = st.columns(2)
    symbol = left.selectbox("Asset", held, help="the pairs with a file in the store; nothing is downloaded")
    timeframe = right.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=TIMEFRAMES.index(TIMEFRAME),
        help="the store holds 5m candles; the longer ones are aggregated from them",
    )
    df = candles(symbol, timeframe, (where / f"{symbol}USDT-5m.parquet").stat().st_mtime)
    pct = change(df)
    st.caption(
        f"{len(df):,} {timeframe} candles of {symbol}USDT on Binance, {df.index[0]:%Y-%m-%d} to "
        f"{df.index[-1]:%Y-%m-%d}. The numbers and the histogram read all of them; the candle charts read "
        "the period picked above them."
    )

    stats = sides(pct)
    for column, (side, title) in zip(st.columns(len(SIDES)), SIDES.items()):
        column.metric(f"{title}: mean", f"{stats.loc[side, 'mean']:+.4f}%")
        column.metric(f"{title}: standard deviation", f"{stats.loc[side, 'std']:.4f}%")
    flat = int((pct == 0).sum())
    st.caption(
        "Change of a candle: close / open − 1. The mean of all candles is the one near zero; their standard "
        f"deviation is the size of a typical candle. {flat:,} candles ({flat / len(pct):.1%}) closed where they "
        "opened: they count in all candles and in neither side."
    )

    st.subheader("Candles")
    first, last = df.index[0].date(), df.index[-1].date()
    period = st.date_input(
        "Period",
        value=(max(first, last - PERIOD.to_pytimedelta()), last),
        min_value=first,
        max_value=last,
        help="only the two candle charts read it, to keep the drawing light: the numbers and the histogram read every "
        "candle",
    )
    # Mid-pick the widget holds the start alone; the charts run to the last candle until the end is picked.
    start, end = period if len(period) == 2 else (period[0], last)
    window = df.loc[str(start) : str(end)]
    view = f"{symbol}-{timeframe}-{start}-{end}"
    st.plotly_chart(candle_figure(window, view), use_container_width=True, key="candles")

    st.markdown("**The same candles, each from its own open**")
    st.caption(
        "Open, high, low and close as the % change from the candle's own open: every candle starts at zero, its "
        "body is its change, its wicks how far it went either way."
    )
    how = st.radio(
        "Subtract from each candle",
        list(OFFSETS),
        format_func=OFFSETS.get,
        horizontal=True,
        help="toward zero: an up candle moves down by it, a down candle up, a flat one stays. Up / down: an up candle "
        "loses the up candles' statistic, a down candle the down candles'. The whole candle moves, so its shape is "
        "kept and its close reads what it did beyond the statistic. The statistics are the ones at the top of the "
        "page, over every candle loaded",
    )
    minus = offset(pct, stats, how).loc[window.index] if how != "none" else None
    st.plotly_chart(zero_figure(window, view, minus), use_container_width=True, key="from-zero")

    st.subheader("Distribution of the candles' change")
    counts, edges, beyond = histogram(pct)
    st.plotly_chart(histogram_figure(counts, edges), use_container_width=True, key="histogram")
    st.caption(
        f"Every candle loaded, in {BINS} bins of {edges[1] - edges[0]:.4f}% over ±{edges[-1]:.3f}%, the 0.1th or "
        f"99.9th percentile, whichever is further from zero. {beyond:,} candles ({beyond / len(pct):.2%}) lie "
        "beyond it and are not drawn."
    )


def _selfcheck() -> None:
    """The arithmetic on hand-made candles, and the property the docstring quotes."""
    df = pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0, 100.0], "close": [101.0, 99.0, 102.0, 98.0, 100.0]})
    pct = change(df)
    assert np.allclose(pct, [1.0, -1.0, 2.0, -2.0, 0.0])
    s = sides(pct)
    assert np.isclose(s.loc["all", "mean"], 0.0) and np.isclose(s.loc["all", "std"], np.sqrt(2.5))
    # A flat candle is in neither side: two up, two down.
    assert np.isclose(s.loc["up", "mean"], 1.5) and np.isclose(s.loc["down", "mean"], -1.5)
    assert np.isclose(s.loc["up", "std"], np.sqrt(0.5)) and np.isclose(s.loc["down", "std"], np.sqrt(0.5))
    # From zero: every candle opens at 0, closes at its change, and keeps its wicks in percent.
    ohlc = df.assign(high=[103.0, 100.5, 102.0, 100.0, 101.0], low=[99.5, 97.0, 100.0, 96.0, 99.0])
    zero = from_zero(ohlc)
    assert (zero.open == 0).all() and np.allclose(zero.close, pct)
    assert np.allclose(zero.high, [3.0, 0.5, 2.0, 0.0, 1.0]) and np.allclose(zero.low, [-0.5, -3.0, 0.0, -4.0, -1.0])
    # The offsets: toward zero on either side, nothing on a flat candle.
    s = sides(pct)
    assert np.allclose(offset(pct, s, "none"), 0.0)
    assert np.allclose(offset(pct, s, "std"), np.array([1, -1, 1, -1, 0]) * np.sqrt(2.5))
    assert np.allclose(offset(pct, s, "side std"), np.array([1, -1, 1, -1, 0]) * np.sqrt(0.5))
    assert np.allclose(pct - offset(pct, s, "side mean"), [-0.5, 0.5, 0.5, -0.5, 0.0]), "beyond its side's mean"
    assert (zero.low <= zero[["open", "close"]].min(axis=1)).all() and (
        zero.high >= zero[["open", "close"]].max(axis=1)
    ).all()

    # Normal changes: the mean of all is near zero and the standard deviation is not, and each side's
    # standard deviation is sqrt(1 - 2/pi) of the whole.
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(0.0, 0.3, 400_000))
    s = sides(normal)
    assert abs(s.loc["all", "mean"]) < 0.002 and abs(s.loc["all", "std"] - 0.3) < 0.002
    for side in ("up", "down"):
        assert abs(s.loc[side, "std"] / s.loc["all", "std"] - np.sqrt(1 - 2 / np.pi)) < 0.005
    assert np.isclose(s.loc["up", "mean"], -s.loc["down", "mean"], rtol=0.01)

    counts, edges, beyond = histogram(normal)
    assert len(counts) == BINS and np.isclose(edges[0], -edges[-1]), "symmetric around zero"
    assert np.isclose(edges[BINS // 2] + edges[BINS // 2 + 1], 0.0), "one bin centred on zero"
    assert counts.sum() + beyond == len(normal) and beyond <= 2 * TAIL * len(normal) + 1
    assert counts.argmax() in range(BINS // 2 - 5, BINS // 2 + 6), "the bell peaks at zero"
    # All candles flat: a degenerate axis is widened rather than handed to numpy with equal edges.
    counts, edges, beyond = histogram(pd.Series(np.zeros(10)))
    assert counts.sum() == 10 and beyond == 0


if __name__ == "__main__":
    main()
