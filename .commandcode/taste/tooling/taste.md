# Taste

## Orchestration & workflow tooling

- Prefers orchestrating multi-step RAG query flows with LangGraph (StateGraph): each stage gets its own node (retrieve → rerank → build context → generate → format citations) in a dedicated module (e.g., `rag_pipeline.py`), rather than an inline procedural sequence in the API handler. Explicitly requested: "add langgraph into this codebase ... you can give a rag_pipeline.py where you put these functions like node." Confidence: 0.8

## Deployment & containerization

- Prefers running the codebase containerized via Docker Compose (a `Dockerfile` plus `docker-compose.yml`) rather than only on the host, with external dependencies (e.g., the Ollama LLM) run as separate compose services and host paths/volumes for data. Explicitly requested: "give me Dockerfile and docker compose file for this for running this codebase into docker." Confidence: 0.6
- Prefers simple, single-stage Dockerfiles over complex multi-stage builds: described the two-stage version (with a model pre-download build stage) as "too much complex" and asked to "make it simple", accepting that models download at runtime on first start instead of being baked into the image. Explicitly requested: "i see your Dockerfile may be too much complex by giving two stage. make it simple." Confidence: 0.8
- Questions the necessity of each step in container startup scripts (e.g., `docker-entrypoint.sh`) and wants only steps with a real purpose kept — prefers the entrypoint to be as minimal as possible, with no pre-download steps at all: first dropped the redundant embedding pre-download, then, even after the rationale that the lazy-loaded reranker pre-download prevents first-request stalls, still asked to remove that line too ("i think we can remove this line also"), accepting models download lazily on first use. Asked directly of the pre-download lines: "is these necessary?" Confidence: 0.75
