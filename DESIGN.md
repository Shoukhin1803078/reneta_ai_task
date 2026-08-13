# Design

## My Design Overview 

My idea is to turn the five medicine leaflets into a searchable index. When a user asks a question, I retrieve the most relevant chunks using both semantic search and BM25, reranks them and then give only the best chunks to the LLM .
### Pipelines
<img width="848" height="885" alt="Screenshot 2026-08-14 at 1 41 55 AM" src="https://github.com/user-attachments/assets/e37ccd98-5e2d-4953-bd61-fe651ea7dd52" />


I implemented the query flow as a LangGraph state graph in app/rag_pipeline.py. I keep each stage separate—retrieve, rerank, build context, generate and format citations—so I can easily understand and modify each part.

### Main Components

- **Ingest (`app/ingest.py`)**: Parse PDFs → section-based chunking → metadata → embeddings → ChromaDB.
- **Retrieve (`app/retriever.py`)**: Perform semantic and BM25 search, then merge and deduplicate the candidates.
- **Rerank (`app/reranker.py`)**: Score the retrieved candidates using a cross-encoder and select the top-k chunks.
- **Generate (`app/service.py` + `app/rag_pipeline.py`)**: Build the context from the top chunks and generate an answer using only that context.
- **UI**: I build a simple static HTML page sends questions to `/ask` and displays the answer and citations.

```
Query → Semantic + BM25 → Hybrid Search → Rerank → Context → LLM → Answer
```

## Chunking strategy and why

### My Chunking Strategy

The medicine leaflets follow a common structure, such as:

- What X is and what it is used for
- Before you take
- How to take
- Possible side effects
- Use in pregnancy and breast-feeding
- How to store

I decided not to split everything into fixed-size chunks with overlap from the beginning. Instead, I first split the content based on these section headings. If a section is too long, I split it further using `RecursiveCharacterTextSplitter` with a chunk size of 1000 and an overlap of 100.
 
##### Reason for choosing this chunking strategy:
I chose this `section based chunking` approach because I want each chunk to stay focused on one topic. It also allows me to keep the section name in the metadata, which I later use when generating citations.
Why this matters: it gives chunks small enough for good embedding recall,
and — more importantly for the citations — it keeps the section name intact
as metadata. That section name is exactly what I surface in the `/ask`
response, so every citation points back to a real heading in a real leaflet.


## Grounding and honesty

The core design goal: the model should make sure the LLM answers only from the medicine leaflets.

### 1. Confidence-Based Refusal

##### Current way:
Before generating an answer, the system checks the confidence of the top reranked result. The cross-encoder score is converted to a 0–1 confidence using a sigmoid function. If the confidence is below : 
```
MIN_CONFIDENCE_THRESHOLD = 0.5
```
then the system refuses to answer and returns this answer with emty citation.
> I don't have that information in the provided documents

Initially, I tried using the LLM itself as a confidence gate by asking the model a yes/no , whether the context contained enough information to answer the question. But the local 3B model sometimes refused questions even when the correct chunk had a strong retrieval score. 
So I decided to use the retrieval score instead.

### 2.  Retrieval-limited context
The LLM prompt contains only the top-k
reranked chunks from the vector database. There's no general-knowledge
injection, and the prompt tells the model to use only the provided context.

The basic idea is:

```
User Question
     ↓
Retrieve relevant chunks
     ↓
Rerank
     ↓
Top-K chunks
     ↓
LLM
```

### 3. Citations from the actual retrieval.
Citations are generated directly from the retrieved document metadata.
Each citation contains:

- `source` — the original leaflet filename
- `section` — the leaflet section
- `score` — the reranker confidence

This keeps every citation traceable to an actual document and section.

### Refusal Response

When the confidence threshold is not met:

```
{
  "answer":"I don't have that information in the provided documents",
  "citations": []
}
```

The citation list is intentionally empty because the system should not provide sources when it cannot confidently answer the question.

### Known Trade-off

A single confidence threshold is not perfect. A threshold that is too high may reject valid answers, while a threshold that is too low may allow irrelevant context through.
The current threshold was selected based on observed reranker scores but should be further tuned using a larger evaluation dataset.


## Tech Stack
- Python — Core development
- LangChain — RAG and retrieval components
- LangGraph — RAG workflow
- ChromaDB — Vector database
- HuggingFace all-MiniLM-L6-v2 — Embeddings
- BM25 — Keyword search
- Cross-Encoder ms-marco-MiniLM-L-6-v2 — Reranking
- Ollama + Llama 3.2 3B — Local LLM
- FastAPI — API
- HTML/CSS/JavaScript — Simple UI

## Current Configuration

The current codebase uses **top_k = 3** for the final retrieval/reranking stage. So the system passes the top 3 most relevant chunks to the LLM for answer generation.

## Sample Output

- In `README.md`, I have included screenshots of the sample outputs.
- In `sample_question_set.md`, I have included 6 sample questions along with the answers generated by the system.


## Model Choices and Trade-offs

- **`all-MiniLM-L6-v2` embeddings**
    - **Why I picked it:** Small, fast on CPU, and sufficient for this corpus.
    - **Trade-off:** Less nuanced than larger embedding models.
- **`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker**
    - **Why I picked it:** Provides a strong relevance signal and improves the hybrid retrieval results.
    - **Trade-off:** Adds extra computation per request.
- **`llama3.2:3b` via Ollama**
    - **Why I picked it:** Free, runs locally, lightweight, and suitable for grounded Q&A.
    - **Trade-off:** Smaller and less capable than 7B/8B models.


## What I would improve with more time

- Cache the loaded models/vectorstore across requests instead of reloading
  per request.
- Tune the confidence threshold against a larger evaluation set; consider a
  per-section or per-question-type threshold.

