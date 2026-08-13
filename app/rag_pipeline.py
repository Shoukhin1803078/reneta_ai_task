import math

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .service import (
    NOT_IN_DOCS_PHRASE,
    build_llm,
    generate_answer,
    get_source_documents,
    load_vectorstore,
)
from .reranker import rerank_documents
from .retriever import (
    hybrid_search,
    keyword_search,
    semantic_search,
)
from .utils import create_context


class RAGState(TypedDict):
    question: str
    vectorstore: object
    llm: object
    top_k: int
    candidates: list
    final_reranked_results: list
    context: str
    answer: str
    citations: list


def retrieve(state: RAGState):
    query = state["question"]
    Top_K = state["top_k"]
    vectorstore = state["vectorstore"]

    semantic_results = semantic_search(vectorstore, query, Top_K)

    bm25_results = keyword_search(query, get_source_documents(vectorstore))

    # Get candidates chunks from Hybrid retriever
    candidates = hybrid_search(semantic_results, bm25_results, Top_K)

    return {"candidates": candidates}


def rerank(state: RAGState):
    query = state["question"]
    Top_K = state["top_k"]

    final_reranked_results = rerank_documents(query, state["candidates"], Top_K)

    return {"final_reranked_results": final_reranked_results}


def build_context(state: RAGState):
    context = create_context(
        [doc for doc, _ in state["final_reranked_results"]]
    )

    return {"context": context}


def generate(state: RAGState):
    answer = generate_answer(state["llm"], state["question"], state["context"])

    return {"answer": answer}


def format_citations(state: RAGState):
    answer = state["answer"]

    is_no_info_answer = (
        "don't have that information" in answer.lower()
        or "do not have that information" in answer.lower()
    )

    if is_no_info_answer:
        citations = []
    else:
        citations = []

        for document, score in state["final_reranked_results"]:
            citations.append(
                {
                    "source": document.metadata["source"],
                    "section": document.metadata["section"],
                    "score": round(1 / (1 + math.exp(-score)), 3),
                }
            )

    return {"citations": citations}


def build_pipeline():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("build_context", build_context)
    graph.add_node("generate", generate)
    graph.add_node("format_citations", format_citations)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "format_citations")
    graph.add_edge("format_citations", END)

    return graph.compile()


pipeline = build_pipeline()


# Save graph image
# try:
#     image_bytes = pipeline.get_graph().draw_mermaid_png()
#     with open("rag_graph.png", "wb") as f:
#         f.write(image_bytes)
#     print("Successfully saved graph image to root directory as 'rag_graph.png'")
# except Exception as e:
#     print(f"Failed to save graph image: {e}")


def run_pipeline(question, top_k=3):
    vectorstore = load_vectorstore()
    llm = build_llm()

    result = pipeline.invoke(
        {
            "question": question,
            "vectorstore": vectorstore,
            "llm": llm,
            "top_k": top_k,
        }
    )

    print(f"Ollama final answer=====> {result['answer']}")

    return result
