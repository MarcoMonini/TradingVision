# Streamlit chart app, deployed on Render. Build: docker build -t tradingvision .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true

# Dependencies before sources: the heavy layer survives every code edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The two trained checkpoints, ahead of the sources because they change far less often than the
# code does. `binance.STORE` resolves to /app/data off the editable install, which is where this
# lands. 106 KB, so the layer costs nothing.
COPY data/gru.pt data/swing.pt ./data/

COPY src ./src
RUN uv sync --frozen --no-dev

# Stateless apart from those two files: the page draws Alpaca downloads live and computes the
# label, the pivots and the features on them. It never opens the Parquet store — that is read by
# the Binance fetcher and the pipeline, neither of which runs here — so no disk is mounted and
# the 15 GB under data/ stays out of both the repo and the image.
EXPOSE 8501
# Shell form on purpose: Render injects the port at runtime (PORT, default 10000) and exec form
# would not expand it. The fallback keeps `docker run -p 8501:8501` working locally.
CMD streamlit run src/tradingvision/app/chart.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
