# --- Build stage: download models so the runtime image starts fast ---
FROM python:3.12-slim AS build

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir \
    "sentence-transformers==5.7.0" \
    "transformers==5.15.0" \
    "torch==2.13.0" \
    "safetensors==0.8.0" \
    "huggingface_hub==1.27.0" \
    "tokenizers==0.22.2"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# --- Runtime stage ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://ollama:11434

WORKDIR /app

# Copy the pre-downloaded Hugging Face models from the build stage
COPY --from=build /root/.cache/huggingface /root/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY docker-entrypoint.sh .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
