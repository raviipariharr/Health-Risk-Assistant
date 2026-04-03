# ============================================================
# Dockerfile — AI Health Risk Assistant (Backend)
# ============================================================
#
# What is Docker?
#   Docker packages your app + Python + all dependencies into
#   one portable "image". Anyone can run it identically on
#   any machine — no "it works on my laptop" problems.
#
# How a Dockerfile works (read top to bottom):
#   Each line is a LAYER. Docker caches layers — so if you
#   only change your Python code, it re-uses the pip install
#   layer and rebuilds only the changed layers.
#   This makes rebuilds very fast.
#
# Build:  docker build -t health-risk-api .
# Run:    docker run -p 8000:8000 health-risk-api
# ============================================================

# ---- Base image ----
# python:3.11-slim is an official Python image based on Debian.
# "slim" = stripped down (no build tools, docs, etc.)
# Result: ~150MB instead of ~900MB for the full image.
FROM python:3.11-slim

# ---- Metadata ----
LABEL maintainer="your-email@example.com"
LABEL description="AI Health Risk Assistant API"
LABEL version="1.0.0"

# ---- Environment variables ----
# PYTHONDONTWRITEBYTECODE — don't write .pyc files inside container
# PYTHONUNBUFFERED        — print logs immediately (no buffering)
# These are standard best-practices for Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Where pip installs packages
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Working directory ----
# All subsequent commands run from /app inside the container
WORKDIR /app

# ---- System dependencies ----
# Some Python packages need C libraries to compile.
# We install then clean up apt cache to keep image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
# COPY requirements.txt BEFORE copying source code.
# Why? Docker caches this layer. If you only change app code
# (not requirements), Docker skips re-running pip install.
# This is the most important Dockerfile optimisation.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- spaCy model ----
# Download the English spaCy model during build
# so it's baked into the image (no runtime download needed)
RUN python -m spacy download en_core_web_sm || \
    echo "spaCy model download failed — will use blank pipeline"

# ---- Copy source code ----
# Copy everything else AFTER pip install (cache optimisation)
COPY . .

# ---- Train ML model ----
# Run training during build so the model is ready at startup.
# In production you'd load a pre-trained model from cloud storage
# instead — but building it here is great for learning.
RUN python -m app.ml.training.train

# ---- Non-root user ----
# Running as root inside a container is a security risk.
# Create a dedicated user with no home directory.
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---- Port ----
# Tell Docker this container listens on port 8000.
# This is documentation — the actual binding happens at `docker run`.
EXPOSE 8000

# ---- Health check ----
# Docker periodically runs this command to check if the container
# is healthy. If it fails 3 times, Docker marks it "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ---- Startup command ----
# CMD is what runs when the container starts.
# Using array form (exec form) — avoids shell signal handling issues.
# --host 0.0.0.0 — listen on ALL interfaces (not just localhost)
#                  Required in Docker so the host can reach the container
# --workers 2     — 2 worker processes for concurrency
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]