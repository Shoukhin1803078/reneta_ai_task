from langchain_community.cross_encoders import HuggingFaceCrossEncoder


_reranker_model = None


def get_reranker_model():
    """Lazily load the cross-encoder reranker so importing this module
    stays fast and the server can run even when the model is not yet cached."""
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _reranker_model


def rerank_documents(query, documents, k=5):
    pairs = []
    for document in documents:
        pair = [query, document.page_content]
        pairs.append(pair)

    reranker_model = get_reranker_model()
    scores = reranker_model.score(pairs)
    scored_documents = []

    for i, (document, score) in enumerate(zip(documents, scores), start=1):
        scored_documents.append([document, score])

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    results = []

    scored_top_k_document = scored_documents[:k]

    for document, score in scored_top_k_document:
        results.append([document, score])

    return results
