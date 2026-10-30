"""PDF extraction, chunking, local embedding, and hybrid evidence retrieval."""
from __future__ import annotations

from io import BytesIO
from typing import Iterable

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create embeddings locally without using a paid API."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def pdf_uploads_to_documents(files: Iterable) -> list[Document]:
    """Extract selectable text from uploaded PDFs and split it into chunks."""
    pages: list[Document] = []

    for uploaded in files:
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        name = getattr(uploaded, "name", "uploaded.pdf")

        with fitz.open(stream=BytesIO(data), filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append(
                        Document(
                            page_content=text,
                            metadata={"source": name, "page": page_number},
                        )
                    )

    if not pages:
        raise ValueError(
            "No selectable text was found in the uploaded PDFs. "
            "Scanned PDFs require OCR first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=140,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vector_store(files: Iterable, api_key: str = "") -> tuple[Chroma, int, list[str]]:
    """Build a Chroma index using local embeddings."""
    del api_key  # Retained for compatibility with the existing UI.
    documents = pdf_uploads_to_documents(files)
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="research_agent_local",
    )

    sources = sorted({
        doc.metadata.get("source", "Unknown source")
        for doc in documents
    })

    return vector_store, len(documents), sources


def _lexical_score(question: str, content: str) -> int:
    """Count query terms found in a passage, including common exact phrases."""
    import re

    stopwords = {
        "the", "and", "for", "are", "was", "were", "with", "that", "this",
        "from", "have", "has", "had", "you", "your", "into", "about",
        "what", "which", "who", "why", "how", "include", "explain",
        "based", "using", "each", "page", "cite", "source", "evidence",
    }
    terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z0-9]{3,}", question)
        if term.lower() not in stopwords
    }
    text = content.lower()
    return sum(1 for term in terms if term in text)


def retrieve_evidence(
    vector_store: Chroma, question: str, k: int = 6
) -> list[Document]:
    """Combine semantic and lexical ranking to improve multi-part PDF retrieval."""
    candidate_count = max(k * 4, 20)

    try:
        semantic = vector_store.similarity_search_with_relevance_scores(
            question, k=candidate_count
        )
        candidates = [doc for doc, _score in semantic]
    except (AttributeError, TypeError, ValueError):
        candidates = vector_store.similarity_search(question, k=candidate_count)

    # Deduplicate repeated chunks while preserving their source/page metadata.
    unique: dict[tuple[str, int | str, str], Document] = {}
    for doc in candidates:
        key = (
            str(doc.metadata.get("source", "")),
            doc.metadata.get("page", "?"),
            doc.page_content.strip(),
        )
        unique.setdefault(key, doc)

    ranked = sorted(
        unique.values(),
        key=lambda doc: (
            _lexical_score(question, doc.page_content),
            len(doc.page_content),
        ),
        reverse=True,
    )

    # Reserve some slots for different pages so multi-part questions are less
    # likely to be answered from several near-duplicate chunks from one page.
    selected: list[Document] = []
    seen_pages: set[tuple[str, str]] = set()

    for doc in ranked:
        page_key = (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("page", "?")),
        )
        if page_key not in seen_pages:
            selected.append(doc)
            seen_pages.add(page_key)
        if len(selected) >= k:
            return selected

    for doc in ranked:
        if doc not in selected:
            selected.append(doc)
        if len(selected) >= k:
            break

    return selected
