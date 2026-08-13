# Renata Medicine Leaflet Assistant

A small RAG service that answers staff questions about Renata PLC product
inserts / medicine leaflets, grounded strictly in the provided PDFs, with
citations, and honest when the answer isn't in the documents.

## What's inside

All service code lives in the `app/` package:

- `app/ingest.py` — parse PDFs -> chunk by section -> embed -> store in ChromaDB
- `app/retriever.py` — hybrid retrieval (semantic + BM25) and candidate merging
- `app/reranker.py` — lazy cross-encoder reranker model + `rerank_documents`
- `app/utils.py` — small shared helpers, e.g. `create_context`
- `app/service.py` — shared helpers: vectorstore loading, BM25 corpus, LLM, grounded answer generation
- `app/rag_pipeline.py` — the query flow as a LangGraph state graph (retrieve, rerank, context, generate, citations)
- `app/schemas/` — Pydantic request/response models (`AskRequest`, `Citation`, `AskResponse`)
- `app/main.py` — FastAPI service exposing `POST /ask` and the chat UI
- `static/index.html` — minimal single-page chat interface
- `docs/` — the medicine leaflets (not committed here; place them in this folder)

## Models used

| Component | Model | Notes |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, runs on CPU |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder, runs on CPU |
| LLM | `llama3.2:3b` via Ollama | Local, ~2 GB, no API key |

## Install and run (under 5 minutes)

Prerequisites: Python 3.10+, and [Ollama](https://ollama.com) with the model pulled.

```bash
# 1. Environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Ollama model
ollama pull llama3.2:3b

# 3. Put the medicine leaflets (the 5 PDFs) in ./docs/

# 4. Build the vector database (once)
python -m app.ingest

# 5. Run the service
uvicorn app.main:app --reload
# or: python -m app.main

# 6. Open the chat UI
open http://localhost:8000          # Windows: start http://localhost:8000
```

You can also call the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "I am 5 years old. Can I take Doxicap?"}'
```

Response:

```json
{
  "answer": "No, you should not take Doxicap if you are a child under 8 years of age.",
  "citations": [
    { "source": "doxicap_100mg_doxycycline_leaflet.pdf", "section": "Before you take", "score": 0.82 }
  ]
}
```

## Run with Docker (optional)

Requires Docker (and Docker Compose). The compose file runs an
[Ollama](https://ollama.com) container for the LLM plus the app container.

```bash
# 1. Put the medicine leaflets (the 5 PDFs) in ./docs/

# 2. Build and start both services
docker compose up --build

# 3. Open the chat UI
open http://localhost:8000
```

The first `docker compose up` pulls the Ollama image and downloads the
`llama3.2:3b` model, so it takes a few minutes. The app entrypoint waits for
Ollama, pulls the model, builds the vector database from `./docs/`, then starts
the service. Stop with `Ctrl+C` (or `docker compose down`).

Notes:

- `./docs/` is mounted read-only into the container; add/remove leaflets there
  and restart to rebuild the index.
- `./chroma_db/` is mounted so the vector store can persist across restarts
  (the entrypoint rebuilds it on each start regardless).
- Ollama's models live in the `ollama_data` Docker volume.

## Assumptions

- `docs/` contains the five patient information leaflets provided with the assignment.
- The vector database is built once by `python -m app.ingest` and read at query time by the service.
- Questions that cannot be answered from the context return the exact answer
  `I don't have that information in the provided documents` with an empty
  `citations` list.
