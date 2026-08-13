def create_context(documents):
    context = ""

    for i, document in enumerate(documents):
        # context += f"\nDocument {i + 1}:\n"
        context += document.page_content
        context += "\n\n"

    return context
