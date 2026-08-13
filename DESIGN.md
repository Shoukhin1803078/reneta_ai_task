# Design

## The plan at a glance

The idea is simple: turn the five medicine leaflets into a searchable index,
then answer questions by pulling the most relevant chunks and letting the LLM
read only those.

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

The query flow is wired up as a LangGraph state graph in
`app/rag_pipeline.py` — each stage (retrieve, rerank, build context, generate,
format citations) is its own node, which keeps the pipeline easy to follow and
tweak.

- **Ingest** (`app/ingest.py`): parse -> section-based chunking -> metadata -> embeddings -> ChromaDB.
- **Retrieve** (`app/retriever.py`): semantic + BM25 candidates merged and deduplicated.
- **Rerank** (`app/reranker.py`): candidates scored by a cross-encoder, top-k kept.
- **Generate** (`app/service.py` + `app/rag_pipeline.py`): top chunks become the context; the LLM answers using only that context.
- **UI**: single static HTML page posting to `/ask`, rendering answer + citations.

## Chunking strategy and why

The leaflets all follow the same fixed structure — "What X is and what it is
used for", "Before you take", "How to take", "Possible side effects", "Use in
pregnancy and breast-feeding", "How to store". So instead of blindly chopping
text into equal-size pieces, I split on those section headings first. Each
chunk stays within one topical section. Only the long sections then get
further split with `RecursiveCharacterTextSplitter` (size 1000, overlap 100).

Why this matters: it gives chunks small enough for good embedding recall,
and — more importantly for the citations — it keeps the section name intact
as metadata. That section name is exactly what I surface in the `/ask`
response, so every citation points back to a real heading in a real leaflet.

## Grounding and honesty

The core design goal: the model should only ever answer from the leaflet
content. Nothing else gets into the prompt. Three things enforce that:

**1. Retrieval-limited context.** The LLM prompt contains only the top-k
reranked chunks from the vector database. There's no general-knowledge
injection, and the prompt tells the model to use only the provided context.

**2. A confidence-based honesty gate.** Before generating an answer, the
service checks the reranker confidence of the top retrieved chunk. The
reranker logit is mapped to a 0–1 confidence via sigmoid, and if that
confidence is below a threshold (`MIN_CONFIDENCE_THRESHOLD = 0.5`), the
service refuses with the exact phrase:

> I don't have that information in the provided documents

I tried an LLM-judged gate first — asking the model a yes/no "does the
context answer this?" question — but the small local model (3b) over-thought:
it refused even when the correct chunk was retrieved with 0.98 confidence,
reasoning about edge cases the question didn't ask about. The retrieval
scores are more reliable and deterministic than a small model's judgment, so
the gate now keys off the actual retrieval confidence. This also saves an
LLM call per request.

**3. Citations from the actual retrieval.** Every citation is built from the
real retrieved chunks — `source` (filename) and `section` (leaflet heading)
come straight from chunk metadata, and `score` is the reranker logit mapped
to 0–1 confidence via sigmoid. So an answer can always be traced back to a
specific document and section.

**What the API returns on refusal.** When the gate says the context doesn't
cover the question, the response is:

```json
{
  "answer": "I don't have that information in the provided documents",
  "citations": []
}
```

`citations` is deliberately empty — the service should never claim a source
for a non-answer.

**Known trade-off.** A single threshold is a blunt instrument: set it too
high and real answers get refused; too low and some wrong-context answers
slip through. I calibrated it to the observed reranker scores, but it's worth
tuning against a larger evaluation set.

## Model choices and trade-offs

| Model | Why I picked it | Trade-off |
| --- | --- | --- |
| `all-MiniLM-L6-v2` embeddings | Small, fast on CPU, plenty for this corpus | Less nuanced than larger embedding models |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker | Strong relevance signal; hybrid retrieval really benefits from reranking | Extra compute per request; loaded lazily |
| `llama3.2:3b` via Ollama | Free, local, ~2 GB, fits modest hardware | Smaller model, weaker than 7b/8b but fine for grounded Q&A |

## What I would improve with more time

- Tune the confidence threshold against a larger evaluation set; consider a
  per-section or per-question-type threshold.
- Expand the evaluation set (currently 6 question -> expected-answer pairs in
  `evaluation_set.json`, scored by `python -m app.evaluate`).
- Cache the loaded models/vectorstore across requests instead of reloading
  per request.
