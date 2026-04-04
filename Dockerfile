FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app/env:${PYTHONPATH}"

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen(\
    urllib.request.Request('http://localhost:8000/reset', \
    data=b'{}', headers={'Content-Type':'application/json'}))" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
