# VectorDB — from-scratch HNSW/KD-tree/brute-force vector search + RAG demo
#
# Build (lean, no sentence-transformers — use this if you'll rely on
# Ollama or Groq only):
#   docker build -t vectordb .
#
# Build with the local-embedding fallback baked in (adds ~a few hundred
# MB for CPU-only torch + sentence-transformers):
#   docker build --build-arg INSTALL_ST=true -t vectordb .
#
# Run:
#   docker run -p 8080:8080 -e GROQ_API_KEY=... -v $(pwd)/data:/app/data vectordb

FROM python:3.12-slim

WORKDIR /app

ARG INSTALL_ST=false

COPY requirements.txt requirements-optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_ST" = "true" ]; then \
        pip install --no-cache-dir -r requirements-optional.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu ; \
    fi

COPY core.py ai_providers.py persistence.py demo_data.py main.py index.html ./

# Persisted vector/doc store lives here — mount a volume in production
# so data survives container restarts/redeploys.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

# IMPORTANT: --workers must stay at 1. VectorDB/DocumentDB state lives
# in each process's memory (not a shared store like Redis/Postgres), so
# multiple worker *processes* would each hold their own drifting copy —
# an insert handled by worker A would be invisible to a read handled by
# worker B until the next restart. Verified this causes real
# read-after-write inconsistency with >1 worker. Threads are safe here
# since they share one process's memory and VectorDB/DocumentDB already
# guard mutations with a lock. Scaling beyond one process would require
# moving state to a real shared store first — noted as a next step in
# the README, out of scope for this project.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "main:app"]
