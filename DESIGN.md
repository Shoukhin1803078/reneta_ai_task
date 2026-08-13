# Design

## Plan / architecture

```
docs/*.pdf
   │  ingest.py
   ▼
chunk by section ──► embed (all-MiniLM-L6-v2) ──► ChromaDB (./chroma_db)
                                                       │
                                                       ▼
POST /ask ──► semantic search (top-3) ──┐
            │                           ├─► hybrid candidates ──► rerank (cross-encoder) ──► top-3 ──► context ──► LLM ──► {answer, citations}
            └► BM25 keyword search (top-3)┘
```

- **Ingest** (`app/ingest.py`): parse -> section-based chunking -> metadata -> embeddings -> ChromaDB.
- **Retrieve** (`app/retriever.py`): semantic + BM25 candidates merged and deduplicated.
- **Rerank** (`app/reranker.py`): candidates scored by a cross-encoder, top-k kept.
- **Generate** (`app/service.py` + `app/rag_pipeline.py`): top chunks become the context; a LangGraph state graph runs retrieve -> rerank -> build context -> generate -> format citations; the LLM answers using only that context.
- **UI**: single static HTML page posting to `/ask`, rendering answer + citations.

## Chunking strategy and why

Leaflets follow a fixed structure (What X is and what it is used for, Before
you take, How to take, Possible side effects, Use in pregnancy and
breast-feeding, How to store). The text is first split on those section
headings, so each chunk stays within one topical section, then long sections
are split with `RecursiveCharacterTextSplitter` (size 1000, overlap 100).

This gives chunks that are small enough for good embedding recall, and keeps
the `SECTION` metadata intact — which is what powers the citations returned
by `/ask`.

## Grounding and honesty

- The prompt instructs the LLM to answer **only** from the provided context.
- Citations come from the actual retrieved chunks (`source` + `section`), so
  every claim can be traced back to a document.
- If the question is not covered by the context, the model is expected to say
  it doesn't have that information rather than guessing.

## Model choices and trade-offs

| Model | Why | Trade-off |
| --- | --- | --- |
| `all-MiniLM-L6-v2` embeddings | Small, fast on CPU | Less nuanced than larger embedding models |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker | Strong relevance signal; hybrid retrieval benefits from reranking | Extra compute per request; loaded lazily |
| `llama3.2:3b` via Ollama | Free, local, ~2 GB, fits modest hardware | Smaller model, weaker than 7b/8b but adequate for grounded Q&A |

## What I would improve with more time

- Persist the reranked score / a similarity threshold to flag low-confidence
  answers more rigorously.
- An evaluation set (5-8 question -> expected-answer pairs) to measure
  retrieval quality.
- A Dockerfile to containerize the service.
- Cache the loaded models/vectorstore across requests instead of reloading
  per request.
