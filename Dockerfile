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

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM retrieval-runtime AS ingestion

RUN pip install --no-cache-dir ".[retrieval,ingestion]"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0t64 libxcb1 \
    && rm --recursive --force /var/lib/apt/lists/*

COPY scripts ./scripts

USER appuser

CMD ["python", "--version"]
