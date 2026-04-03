FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY . /app/env
WORKDIR /app/env

# Install git for any git-based deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

ENV PYTHONPATH="/app/env:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
