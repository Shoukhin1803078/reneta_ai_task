#!/bin/sh
set -e

echo "Waiting for Ollama to be ready..."
until curl -fsS http://ollama:11434/api/tags >/dev/null 2>&1; do
  echo "Ollama not ready yet, retrying..."
  sleep 2
done
echo "Ollama is ready."

echo "Pulling model llama3.2:3b..."
ollama pull llama3.2:3b
echo "Model pulled."

echo "Building the vector database..."
python -m app.ingest
echo "Vector database ready."

echo "Starting the FastAPI service..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
