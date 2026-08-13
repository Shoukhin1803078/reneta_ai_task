from langchain_community.retrievers import BM25Retriever


def semantic_search(vectorstore, query, k=5):
    semantic_results = vectorstore.similarity_search_with_score(query, k=k)
    return semantic_results


def keyword_search(query, documents, k=5):
    # BM25 search
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    bm25_results = bm25_retriever.invoke(query)

    return bm25_results


def hybrid_search(semantic_results, bm25_results, k=5):
    results = []
    seen = set()

    # Add semantic results
    for document, score in semantic_results:
        doc_id = (
            document.metadata["source"]
            + "::"
            + document.metadata["section"]
            + "::"
            + document.page_content
        )

        if doc_id not in seen:
            results.append(document)
            seen.add(doc_id)

    # Add BM25 results
    for document in bm25_results:
        doc_id = (
            document.metadata["source"]
            + "::"
            + document.metadata["section"]
            + "::"
            + document.page_content
        )

        if doc_id not in seen:
            results.append(document)
            seen.add(doc_id)

    return results
