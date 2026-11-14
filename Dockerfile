# CrowdGPT orchestrator — CPU control plane and aggregation.
# Build:
#   docker build -t crowdgpt-orchestrator:local .
# Run:
#   docker compose up -d
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CROWDGPT_ORCHESTRATOR_DB=/data/orchestrator_state.sqlite3 \
    CROWDGPT_OBJECT_DIR=/data/objects

WORKDIR /app
COPY requirements.txt ./
# Prefer the CPU wheel; the coordinator does not need CUDA to aggregate.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY . .
RUN useradd --system --uid 10001 --create-home --home-dir /data crowdgpt \
    && mkdir -p /data \
    && chown -R crowdgpt:crowdgpt /app /data

USER crowdgpt
EXPOSE 5006
# server.py has no dedicated health route; /api/stats is unauthenticated and cheap.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5006/api/stats', timeout=4)"
# No shell, no PID 1 signal surprises.
CMD ["python", "server.py"]
