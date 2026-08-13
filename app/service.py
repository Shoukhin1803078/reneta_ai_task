import math
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from .ingest import load_embeddings


NOT_IN_DOCS_PHRASE = "I don't have that information in the provided documents"

# Minimum reranker confidence score
MIN_CONFIDENCE_THRESHOLD = 0.5


def load_vectorstore(persist_directory="./chroma_db", collection_name="medicine_leaflets"):
    embeddings = load_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectorstore


def get_source_documents(vectorstore):
    """All chunks from the vectorstore, used as the BM25 corpus."""
    from langchain_core.documents import Document

    documents = vectorstore._collection.get(include=["documents", "metadatas"])

    source_documents = []

    for text, metadata in zip(documents["documents"], documents["metadatas"]):
        source_documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return source_documents


def build_llm():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    return llm


def generate_answer(llm, query, context, reranked_results=None):
    # Honesty gate: refuse when there are no retrieved chunks, or when the
    # top chunk is not confidently relevant. This is more reliable for a small
    # local model than asking the LLM to judge "does the context answer it" —
    # the 3b model tends to over-think and refuse even when the right chunk
    # was retrieved.
    if not reranked_results:
        return NOT_IN_DOCS_PHRASE

    top_document, top_score = reranked_results[0]
    top_confidence = 1 / (1 + math.exp(-top_score))

    if top_confidence < MIN_CONFIDENCE_THRESHOLD:
        return NOT_IN_DOCS_PHRASE

    prompt = f"""
    Answer the question using only the provided context.
    Do not use outside knowledge or guess.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content
