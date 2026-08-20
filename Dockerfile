FROM python:3.11-slim AS retrieval-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/models/fastembed

WORKDIR /app

RUN useradd --create-home appuser \
    && mkdir --parents /models/fastembed \
    && chown --recursive appuser:appuser /models

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir ".[retrieval]"


FROM retrieval-runtime AS api

RUN pip install --no-cache-dir ".[llm]"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.11-slim AS ui

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home appuser

COPY pyproject.toml README.md ./
COPY app ./app
COPY ui ./ui

RUN pip install --no-cache-dir ".[ui]"

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", \
    "--server.port=8501", "--browser.gatherUsageStats=false"]


FROM retrieval-runtime AS ingestion

# The on-demand Phase 7 E2E CLI uses the same structured generator as the API.
# Keep this runtime-only: no model weights are initialized or downloaded at build time.
RUN pip install --no-cache-dir ".[retrieval,ingestion,llm]"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0t64 libxcb1 \
    && rm --recursive --force /var/lib/apt/lists/*

COPY scripts ./scripts

USER appuser

CMD ["python", "--version"]
