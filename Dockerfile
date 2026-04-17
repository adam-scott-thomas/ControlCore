# ghostrouter — GCP Cloud Run container
#
# Build:
#   gcloud run deploy ghostrouter \
#     --source . \
#     --region us-central1 \
#     --port 8265 \
#     --memory 512Mi \
#     --cpu 1 \
#     --min-instances 0 \
#     --max-instances 10 \
#     --allow-unauthenticated
#
# Free tier: 2M requests/month, 180k vCPU-seconds, 360k GB-seconds.
# At 512Mi / 1 vCPU, that's ~50 hours of active compute/month free.
# Auto-scales to zero when idle.

FROM python:3.12-slim AS base

# System deps for httpx + maelspine
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml README.md /app/
COPY ghostrouter/ /app/ghostrouter/
COPY config/ /app/config/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Cloud Run expects PORT env var; ghostrouter daemon defaults to 8265.
# Wrap in a shim that respects PORT.
ENV PORT=8265
EXPOSE 8265

# Health check — Cloud Run pings / or custom; ghostrouter daemon exposes /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\",\"8265\")}/health', timeout=3).raise_for_status()" || exit 1

# Entrypoint: run the daemon via the create_app factory so PORT works
CMD ["sh", "-c", "uvicorn --factory ghostrouter.daemon:create_app --host 0.0.0.0 --port ${PORT:-8265}"]
