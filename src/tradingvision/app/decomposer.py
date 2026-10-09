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
heavier tails than a normal of the same deviation. The page draws no normal over it. The same buttons as
the candles' take the same statistic off every candle, over the same bins.

Under the candles, in a chart of its own over the same period, the same candles each started from
zero: open and close as changes from the candle's own open, in percent. The price level, the path
between candles and the wicks go, and what is left is each candle's body, the change the histogram
counts. A control right above that chart, and moving nothing else, subtracts a statistic from each of
its candles (`offset`), toward zero: the standard deviation of all candles from every candle, each
side's standard deviation from its candles, or each side's mean from its candles (`less`). The open
stays at zero and the body shrinks by it: a 1% candle less a 0.2% deviation closes at 0.8%. A candle
smaller than the statistic crosses zero, and keeps the colour of the side it came from: an up candle
is green whichever way it now points. The statistics are the ones the page shows, over the whole
history and not over the period. The two charts are one figure on one time axis, so a zoom on either
moves both. The period is read by these two alone, and its default, the last 30 days, is there to keep
the drawing light: the numbers and the histogram read the whole history.

Beside the histogram, in the same row, one section over every candle loaded, numbers measured on BTC
15m to 2026-10-06 (319,870 candles): **Memory** (`acf`): the autocorrelation of the change and of its
size, lags 1 to 10,000 candles (`ACF_LAGS`), against ±2/√n. The change sits at −0.006 at lag 1 (5m
−0.029, the bid-ask bounce) and near zero after; its size starts at 0.39, is 0.22 a day later and
still 0.12 at lag 2,000 (21 days), with a bump every day of lags from the volatility's daily cycle. So
the global standard deviation the buttons subtract is a long-run average over periods whose typical
candle is several times apart. The ±2/√n band holds for independent candles only; with clustered sizes
the true band is wider, the reason the project takes its errors on non-overlapping blocks
(`metrics.blocked`).

**Time of day and week** (`clock`): the mean and the standard deviation of the change by weekday and
UTC hour of the open, two heatmaps side by side, 1,905 candles a cell on BTC 15m. The standard
deviation is the cycle behind the autocorrelation's daily bumps: its median over the week is 0.48% at
14:00 UTC, the US open, and Saturday and Sunday sit at 0.32-0.33% against 0.39-0.41% on weekdays
(ETH 15m the same shape). A cell's standard deviation is one heavy-tailed sample, though: the
largest, Friday 02:00 at 0.73%, and ETH's Tuesday 04:00 at 2.59%, against a typical 0.5%, are a
handful of crashes rather than an hour. The mean is almost all noise: 12 of 168 cells lie beyond two
standard errors of zero, against 8 by chance, and the standard error is understated because sizes
cluster. The one that stands out is Saturday 00:00 UTC, positive at 3.6 standard errors on BTC 15m,
3.8 on BTC 1h and 4.4 on ETH 15m: past a Bonferroni cut of about 3.4 for 168 cells, measured on the
whole history with nothing held out, so a lead and not a finding.

    uv run streamlit run src/tradingvision/app/decomposer.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingvision.data import binance

TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
TIMEFRAME = "15m"  # v2's, the timeframe the project studies
PERIOD = pd.Timedelta(days=30)  # the candle charts' default window
BINS = 601  # odd, so one bin is centred on zero
TAIL = 0.001  # the histogram's axis ends at this quantile on either side, whichever is further out
ACF_LAGS = (
    10_000  # the autocorrelation's last lag, in candles: 35 days of 5m, 104 of 15m; a quarter of the data at most
)
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # pandas' dayofweek, Monday 0
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


def less(zero: pd.DataFrame, minus: pd.Series) -> pd.DataFrame:
    """The candles of `from_zero` with `minus` (an `offset`) taken off their body, and their wicks dropped:
    the open stays at zero, the close moves by it, high and low are the body's ends."""
    close = zero.close - minus.reindex(zero.index)
    return zero.assign(close=close, high=close.clip(lower=0), low=close.clip(upper=0))


def histogram(pct: pd.Series, bins: int = BINS, tail: float = TAIL) -> tuple[np.ndarray, np.ndarray, int]:
    """The counts of `pct` in `bins` equal bins over ±R, their edges, and how many candles lie beyond
    ±R. R is the larger of the `tail` and `1 - tail` quantiles in absolute value."""
    reach = float(np.abs(pct.quantile([tail, 1 - tail])).max()) or float(np.abs(pct).max()) or 1.0
    edges = np.linspace(-reach, reach, bins + 1)
    counts, _ = np.histogram(pct, edges)
    return counts, edges, int(len(pct) - counts.sum())


def candles_figure(df: pd.DataFrame, uirevision: str, minus: dict[str, pd.Series]) -> go.Figure:
    """The candles of `df` over the same candles each from zero (`from_zero`), less what each choice of
    `OFFSETS` takes off (`minus`, its `offset`). One figure on one time axis, so a zoom or a pan on either chart
    moves both: two Streamlit charts cannot share a zoom, the page never hears of one. The choice is a row of
    buttons inside the figure, right above the lower chart it moves, because a Streamlit widget cannot sit
    between the rows of one figure; every choice is drawn and the buttons show one, in the browser."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close), row=1, col=1)
    zero, up = from_zero(df), df.close > df.open
    for how in OFFSETS:
        cut = less(zero, minus[how])
        # One trace per side of the candle as it was, painted that side's colour whichever way it points now: a
        # candle smaller than what `less` took off crosses zero and would otherwise change colour.
        for mask, colour in ((up, "#3D9970"), (~up, "#FF4136")):  # Plotly's own increasing and decreasing colours
            frame = cut[mask]
            side = dict(line_color=colour, fillcolor=colour)
            fig.add_trace(
                go.Candlestick(
                    x=frame.index,
                    open=frame.open,
                    high=frame.high,
                    low=frame.low,
                    close=frame.close,
                    yhoverformat="+.3f",
                    increasing=side,
                    decreasing=side,
                    visible=how == "none",  # the chart opens on the candles as they are, whatever the order above
                ),
                row=2,
                col=1,
            )
    choices(fig, [[True] + [h == how for h in OFFSETS for _ in "ud"] for how in OFFSETS], fig.layout.yaxis2.domain[1])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#888", row=2, col=1)
    fig.update_layout(
        height=880,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        # Keeps zoom and pan across reruns until the pair, the timeframe or the period changes.
        uirevision=uirevision,
    )
    return fig


def choices(fig: go.Figure, shown: list[list[bool]], y: float) -> None:
    """A row of buttons, one per choice of `OFFSETS`, at height `y` of `fig`'s paper: each shows the traces
    its row of `shown` marks. It opens on Nothing, the figure as drawn."""
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=list(OFFSETS).index("none"),
                buttons=[
                    dict(label=OFFSETS[how], method="restyle", args=[{"visible": v}]) for how, v in zip(OFFSETS, shown)
                ],
                x=0,
                xanchor="left",
                y=y,
                yanchor="bottom",
                pad=dict(b=4),
            )
        ]
    )


def histogram_figure(pct: pd.Series, stats: pd.DataFrame, edges: np.ndarray) -> go.Figure:
    """The counts of `pct` in the bins `edges`, less what each choice of `OFFSETS` takes off each candle
    (`offset`), one at a time on buttons above the chart. The bins are the ones of the candles as they are:
    subtracting toward zero never takes a candle further out unless the statistic is beyond the axis."""
    fig = go.Figure()
    for how in OFFSETS:
        counts, _ = np.histogram(pct - offset(pct, stats, how), edges)
        fig.add_trace(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=counts,
                width=np.diff(edges),
                marker=dict(color="#3498db", line=dict(width=0)),
                customdata=np.c_[edges[:-1], edges[1:]],
                hovertemplate="%{customdata[0]:+.3f}% to %{customdata[1]:+.3f}%<br>%{y:,} candles<extra></extra>",
                visible=how == "none",
            )
        )
    choices(fig, [[h == how for h in OFFSETS] for how in OFFSETS], 1.0)
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0,  # contiguous bins: a gap would read as a range no candle fell in
        xaxis_title="change from open to close, %",
        yaxis_title="candles",
    )
    return fig


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """The autocorrelation of `x` at lags 1 to `lags`: sum of x_t x_(t+k) over sum of x_t², both on
    the demeaned series, through the FFT so a million candles take a fraction of a second."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfft(x, 1 << (2 * len(x) - 1).bit_length())  # padded past 2n: no lag wraps onto another
    cov = np.fft.irfft(f * np.conj(f))[: lags + 1]
    return cov[1:] / cov[0]


def clock(pct: pd.Series) -> pd.DataFrame:
    """Mean, standard deviation and count of the change by weekday and UTC hour of the candle's open."""
    return pct.groupby([pct.index.dayofweek, pct.index.hour]).agg(["mean", "std", "count"])


def clock_figure(table: pd.DataFrame, stat: str) -> go.Figure:
    """`clock`'s `stat` as a heatmap, weekdays down and UTC hours across. The mean is drawn on a scale
    centred on zero, and its hover carries the standard error, std / √n, it has to be read against."""
    grid, n = table[stat].unstack(), table["count"].unstack()
    se = (table["std"] / np.sqrt(table["count"])).unstack()
    mean = stat == "mean"
    fig = go.Figure(
        go.Heatmap(
            x=[f"{h:02d}" for h in grid.columns],
            y=[DAYS[d] for d in grid.index],
            z=grid.to_numpy(),
            customdata=np.dstack([n.to_numpy(), se.to_numpy()]),
            colorscale="RdBu" if mean else "Viridis",
            zmid=0 if mean else None,
            colorbar=dict(ticksuffix="%"),
            hovertemplate="%{y} %{x}:00 UTC<br>"
            + ("mean %{z:+.4f}% ± %{customdata[1]:.4f}" if mean else "std %{z:.4f}%")
            + "<br>%{customdata[0]:,} candles<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")  # Monday on top, as a calendar reads
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="hour of the open, UTC")
    return fig


def acf_figure(pct: pd.Series, step: pd.Timedelta) -> go.Figure:
    """The autocorrelation of the change and of its size, lag by lag, over the band ±2/√n that
    independent candles would stay inside; `step` is one candle, for the hover."""
    lags = min(ACF_LAGS, len(pct) // 4)
    k = np.arange(1, lags + 1)
    days = k * (step / pd.Timedelta(days=1))
    fig = go.Figure()
    for name, colour, x in (("change", "#3498db", pct), ("size of the change", "#e67e22", pct.abs())):
        fig.add_trace(
            go.Scatter(
                x=k,
                y=acf(x.to_numpy(), lags),
                name=name,
                line=dict(color=colour, width=1),
                customdata=days,
                hovertemplate=f"{name}<br>lag %{{x:,}} candles, %{{customdata:.2f}} days<br>"
                "autocorrelation %{y:+.4f}<extra></extra>",
            )
        )
    band = 2 / math.sqrt(len(pct))
    for y in (band, -band):
        fig.add_hline(y=y, line=dict(color="#888", dash="dot"))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.update_xaxes(type="log", title="lag, candles")
    fig.update_yaxes(title="autocorrelation")
    # The histogram's height and top margin, which holds its buttons: side by side, the two plot areas line up.
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0.99, xanchor="right", y=0.99))
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
    st.markdown("**Under the candles, the same candles each from its own open**")
    st.caption(
        "The body alone, as the % change from the candle's own open: every candle starts at zero and its body is "
        "its change. The buttons above it subtract a statistic from "
        "each candle's body: the open stays at zero, a 1% candle less 0.2% closes at 0.8%. Up / down: an up candle "
        "loses the up candles' statistic, a down candle the down candles'. A candle smaller than it crosses zero and "
        "keeps its colour. The statistics are the ones at the top, over every candle loaded."
    )
    minus = {how: offset(pct, stats, how).loc[window.index] for how in OFFSETS}
    st.plotly_chart(candles_figure(window, view, minus), use_container_width=True, key="candles")

    left, right = st.columns(2)
    left.subheader("Distribution of the candles' change")
    counts, edges, beyond = histogram(pct)
    left.plotly_chart(histogram_figure(pct, stats, edges), use_container_width=True, key="histogram")
    left.caption(
        f"Every candle loaded, in {BINS} bins of {edges[1] - edges[0]:.4f}% over ±{edges[-1]:.3f}%, the 0.1th or "
        f"99.9th percentile, whichever is further from zero. {beyond:,} candles ({beyond / len(pct):.2%}) lie "
        "beyond it and are not drawn. The buttons subtract from each candle what they subtract in the chart above."
    )

    right.subheader("Memory")
    step = pd.Timedelta(timeframe.replace("m", "min"))
    right.plotly_chart(acf_figure(pct, step), use_container_width=True, key="acf")
    right.caption(
        "The correlation of each candle with the one k candles later, for the change (its sign) and for its size "
        "(its absolute value), every candle loaded. Dotted, ±2/√n, where independent candles would stay 95% of the "
        "time; candles whose size clusters are not independent and the real band is wider, so a change "
        "autocorrelation just outside it is no evidence. Near zero for the change, positive and slow to decay for "
        "the size: the sign of the next candle is not in this one, its size is. A bump every day of lags is the "
        "volatility's daily cycle."
    )

    st.subheader("Time of day and week")
    table = clock(pct)
    left, right = st.columns(2)
    left.markdown("**Mean of the change**")
    left.plotly_chart(clock_figure(table, "mean"), use_container_width=True, key="clock-mean")
    right.markdown("**Standard deviation of the change**")
    right.plotly_chart(clock_figure(table, "std"), use_container_width=True, key="clock-std")
    se = float((table["std"] / np.sqrt(table["count"])).median())
    st.caption(
        f"Every candle loaded, by the weekday and the UTC hour it opened in: {int(table['count'].median()):,} "
        f"candles a cell. A cell's mean has a standard error of about {se:.4f}%, std / √n, and more than that "
        "because sizes cluster: a mean within two of it of zero is no evidence, and among 168 cells a few lie "
        "beyond by chance. The standard deviation is the cycle the autocorrelation's bumps come from."
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
    # Less: the body shrinks by the offset, the open stays, and the wicks go: high and low are the body's ends.
    cut = less(zero, pd.Series([0.2, -0.2, 2.5, -0.2, 0.0], index=zero.index))
    assert (cut.open == 0).all() and np.allclose(cut.close, [0.8, -0.8, -0.5, -1.8, 0.0])
    assert np.allclose(cut.high, [0.8, 0.0, 0.0, 0.0, 0.0]) and np.allclose(cut.low, [0.0, -0.8, -0.5, -1.8, 0.0])

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

    # Clock: one cell per weekday and hour; 2024-01-01 is a Monday, so Monday 14:00 holds 14 twice.
    hours = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
    table = clock(pd.Series(hours.hour.astype(float), index=hours))
    assert len(table) == 7 * 24 and (table["count"] == 2).all() and (table["std"] == 0).all()
    assert table.loc[(0, 14), "mean"] == 14 and clock_figure(table, "mean").data[0].z.shape == (7, 24)
    # Autocorrelation: equal to the sum it stands for, near zero on independent draws and φ^k on an AR(1).
    x = rng.normal(size=5_000)
    d = x - x.mean()
    assert np.allclose(acf(x, 3), [(d[:-k] * d[k:]).sum() / (d * d).sum() for k in (1, 2, 3)])
    assert np.abs(acf(normal.to_numpy(), 50)).max() < 4 / np.sqrt(len(normal))
    ar = np.zeros(200_000)
    for t, e in enumerate(rng.normal(size=len(ar) - 1), start=1):
        ar[t] = 0.5 * ar[t - 1] + e
    assert np.allclose(acf(ar, 3), [0.5, 0.25, 0.125], atol=0.01)


if __name__ == "__main__":
    main()
