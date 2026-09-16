# Streamlit chart app, deployed on Render. Build: docker build -t tradingvision .
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true

# Dependencies before sources: the heavy layer survives every code edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# The checkpoints the page draws predictions from. `data/` is gitignored and never enters the
# image, so `models/` is the only way a trained model reaches Render — see models/README.md.
COPY models ./models

# Stateless: the page draws Alpaca downloads and the committed checkpoints only. The Parquet store
# under data/ is read by the oracle sweep and the Binance fetcher, neither of which runs here, so
# no disk is mounted. torch is a runtime dependency (inference, not training) and is the CPU build:
# the `pytorch-cpu` index in pyproject.toml keeps the image around a gigabyte instead of the five
# the CUDA wheels would cost for hardware Render does not have.
EXPOSE 8501
# Shell form on purpose: Render injects the port at runtime (PORT, default 10000) and exec form
# would not expand it. The fallback keeps `docker run -p 8501:8501` working locally.
CMD streamlit run src/tradingvision/app/chart.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
