from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


SECTION_HEADINGS = [
    "What .* is and what it is used for",
    "Before you take",
    "How to take",
    "Possible side effects",
    "Use in pregnancy and breast-feeding",
    "How to store",
]

HEADING_PATTERN = re.compile(
    r"^(?:\d+\.\s*)?(" + "|".join(SECTION_HEADINGS) + r").*$",
    re.IGNORECASE
)


# def split_by_section(text):
#     sections = []
#     current_section = "Overview"
#     current_text = []

#     for line in text.splitlines():
#         line = line.strip()

#         if HEADING_PATTERN.match(line):
#             if current_text:
#                 sections.append((current_section, "\n".join(current_text).strip()))

#             current_section = line
#             current_text = []
#         else:
#             current_text.append(line)

#     if current_text:
#         sections.append((current_section, "\n".join(current_text).strip()))

#     return sections


def split_by_section(text):
    sections = []
    current_section = "Overview"
    current_text = []

    for line in text.splitlines():
        line = line.strip()

        if HEADING_PATTERN.match(line):
            if current_text:
                sections.append(
                    (current_section, "\n".join(current_text).strip())
                )

            # Remove section number: "1. ", "2. ", etc.
            current_section = re.sub(r"^\d+\.\s*", "", line)
            current_text = []

        else:
            current_text.append(line)

    if current_text:
        sections.append(
            (current_section, "\n".join(current_text).strip())
        )

    return sections


def get_medicine_name(text):
    first_lines = text.splitlines()

    for line in first_lines:
        if line.strip():
            return line.strip()

    return "Unknown"


def chunk_documents(docs_path="docs", chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for pdf_path in Path(docs_path).glob("*.pdf"):
        print(f"Processing: {pdf_path.name}")

        pages = PyPDFLoader(str(pdf_path)).load()
        text = "\n".join(page.page_content for page in pages)

        medicine_name = get_medicine_name(text)

        sections = split_by_section(text)

        for section, content in sections:
            chunks = splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                chunk_text = f"MEDICINE: {medicine_name}\nSECTION: {section}\n\n{chunk}"

                all_chunks.append({
                    "id": f"{pdf_path.name}::{section}::{i}",
                    "source": pdf_path.name,
                    "medicine": medicine_name,
                    "section": section,
                    "text": chunk_text
                })

    return all_chunks


def building_langchain_document_and_add_metadata(chunks):
    documents = []

    for chunk in chunks:
        document = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "medicine": chunk["medicine"],
                "section": chunk["section"],
            }
        )

        documents.append(document)

    print(f"Created {len(documents)} LangChain documents.")

    return documents


def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    return embeddings


def main():
    docs_path = "docs"
    chunk_size = 1000
    chunk_overlap = 100
    persist_directory = "./chroma_db"
    collection_name = "medicine_leaflets"

    chunks = chunk_documents(docs_path, chunk_size, chunk_overlap)
    documents = building_langchain_document_and_add_metadata(chunks)

    embeddings = load_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"Stored {len(documents)} documents in ChromaDB.")


if __name__ == "__main__":
    main()
