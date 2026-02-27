FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY Makefile /app/Makefile
COPY app /app/app
COPY tests /app/tests
COPY scripts /app/scripts
COPY .streamlit /app/.streamlit

RUN pip install --upgrade pip \
    && pip install -e .

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501 8701

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["python3", "-m", "streamlit", "run", "app/ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
