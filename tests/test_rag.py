from langchain_core.documents import Document

from ragchat.rag import format_sources


def test_format_sources_deduplicates_citations_in_retrieval_order() -> None:
    documents = [
        Document("first", {"filename": "alpha.pdf", "page": 1}),
        Document("duplicate", {"filename": "alpha.pdf", "page": 1}),
        Document("second", {"filename": "beta.pdf", "page": 4}),
    ]

    assert format_sources(documents) == ["alpha.pdf, p. 1", "beta.pdf, p. 4"]
