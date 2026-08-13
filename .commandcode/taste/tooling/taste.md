# Taste

## Orchestration & workflow tooling

- Prefers orchestrating multi-step RAG query flows with LangGraph (StateGraph): each stage gets its own node (retrieve → rerank → build context → generate → format citations) in a dedicated module (e.g., `rag_pipeline.py`), rather than an inline procedural sequence in the API handler. Explicitly requested: "add langgraph into this codebase ... you can give a rag_pipeline.py where you put these functions like node." Confidence: 0.8

## Deployment & containerization

- Prefers running the codebase containerized via Docker Compose (a `Dockerfile` plus `docker-compose.yml`) rather than only on the host, with external dependencies (e.g., the Ollama LLM) run as separate compose services and host paths/volumes for data. Explicitly requested: "give me Dockerfile and docker compose file for this for running this codebase into docker." Confidence: 0.6
