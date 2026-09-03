"""Pagina Streamlit: scarica le candele di una coppia crypto e le disegna.

    streamlit run src/tradingvision/app/chart.py
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tradingvision.data.candles import SYMBOLS, TIMEFRAMES, get_candles

carica_candele = st.cache_data(ttl=300)(get_candles)


def grafico(df, symbol: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.02)
    fig.add_trace(
        go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name=symbol), row=1, col=1
    )
    fig.add_trace(go.Bar(x=df.index, y=df.volume, name="volume", marker_color="#888"), row=2, col=1)
    fig.update_layout(height=700, showlegend=False, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
    return fig


def main() -> None:
    st.set_page_config(page_title="TradingVision", layout="wide")
    st.title("TradingVision")

    col_s, col_t, col_g = st.columns(3)
    symbol = col_s.selectbox("Coppia", SYMBOLS)
    timeframe = col_t.selectbox("Timeframe", list(TIMEFRAMES), index=2)
    days = col_g.slider("Giorni di storico", 1, 365, 30)

    df = carica_candele(symbol, timeframe, days)
    if df.empty:
        st.warning(f"Nessun dato per {symbol} su {timeframe}.")
        return

    st.plotly_chart(grafico(df, symbol), use_container_width=True)
    st.caption(f"{len(df)} candele — da {df.index[0]:%Y-%m-%d %H:%M} a {df.index[-1]:%Y-%m-%d %H:%M} UTC")
    with st.expander("Dati"):
        st.dataframe(df.tail(200), use_container_width=True)


main()
