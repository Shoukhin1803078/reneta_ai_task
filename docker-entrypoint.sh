#!/bin/sh
set -e

# python:3.12-slim has no curl, so probe Ollama with Python (always present).
echo "Waiting for Ollama to be ready..."
until python -c "import urllib.request; urllib.request.urlopen('http://ollama:11434/api/tags', timeout=3)" >/dev/null 2>&1; do
  echo "Ollama not ready yet, retrying..."
  sleep 2
done
echo "Ollama is ready."

echo "Pulling model llama3.2:3b..."
python -c "
import json, urllib.request
req = urllib.request.Request(
    'http://ollama:11434/api/pull',
    data=json.dumps({'model': 'llama3.2:3b'}).encode(),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=1800) as resp:
    for line in resp:
        pass
print('pull done')
"
echo "Model pulled."

echo "Building the vector database..."
python -m app.ingest
echo "Vector database ready."

echo "Starting the FastAPI service..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
