# ──────────────────────────────────────────────────────────────────────────────
# Shiksha Intelligence AI Server — Production Dockerfile
# Designed for Render (https://render.com) but works on any container host.
#
# Build:  docker build -t shiksha-ai-server .
# Run:    docker run -p 8001:8001 --env-file .env shiksha-ai-server
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps needed to compile some packages (chromadb, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first so Docker can cache this layer
COPY requirements.txt .

# Install into a prefix so we can copy just the installed packages to the final image
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: final runtime image ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install runtime-only system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 \
        libffi8 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appgroup . .

# Render injects PORT at runtime; default to 8001 for local docker runs
ENV PORT=8001 \
    APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose the port (informational — Render uses $PORT env var)
EXPOSE ${PORT}

# Health-check so Render / Docker knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start uvicorn — reads $PORT so Render's port assignment works automatically
CMD uvicorn main:app \
        --host 0.0.0.0 \
        --port ${PORT} \
        --workers 2 \
        --log-level info \
        --no-access-log
