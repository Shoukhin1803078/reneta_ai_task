import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from .ingest import load_embeddings


NOT_IN_DOCS_PHRASE = "I don't have that information in the provided documents"


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


def generate_answer(llm, query, context):
    # First: does the context actually answer the question?
    verdict_prompt = f"""
    Does the context below answer the question? Answer with YES or NO only.

    Context:
    {context}

    Question:
    {query}
    """

    verdict = llm.invoke(verdict_prompt).content.strip().upper()

    if "NO" in verdict:
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
