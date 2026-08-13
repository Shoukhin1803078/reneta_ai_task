from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .rag_pipeline import run_pipeline
from .schemas import AskRequest, AskResponse


app = FastAPI(title="Renata Medicine Leaflet Assistant")

REPO_ROOT = Path(__file__).resolve().parent.parent


@app.get("/")
def read_root():
    return FileResponse(REPO_ROOT / "static" / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = run_pipeline(request.question, top_k=3)

    return AskResponse(answer=result["answer"], citations=result["citations"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
