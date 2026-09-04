from langchain_core.documents import Document

from ragchat.documents import chunk_pages, clean_text, extract_pdf_pages


def test_clean_text_removes_nulls_joins_hyphenated_lines_and_keeps_paragraphs() -> None:
    text = "A hyphen-\nated\x00 word.\nNext line.\n\n\nSecond paragraph."

    assert clean_text(text) == "A hyphenated word. Next line.\n\nSecond paragraph."


def test_extract_pdf_pages_preserves_filename_and_page(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        pages = [FakePage("First page"), FakePage(""), FakePage("Third page")]

    monkeypatch.setattr("ragchat.documents.PdfReader", lambda _: FakeReader())

    pages = extract_pdf_pages(b"not a real PDF", "guide.pdf")

    assert [page.page_content for page in pages] == ["First page", "Third page"]
    assert [page.metadata for page in pages] == [
        {"filename": "guide.pdf", "page": 1},
        {"filename": "guide.pdf", "page": 3},
    ]


def test_chunk_pages_keeps_page_metadata() -> None:
    page = Document(
        page_content="One two three four five six", metadata={"filename": "guide.pdf", "page": 2}
    )

    chunks = chunk_pages([page], chunk_size=12, chunk_overlap=2)

    assert len(chunks) > 1
    assert all(chunk.metadata["filename"] == "guide.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert [chunk.metadata["chunk"] for chunk in chunks] == list(range(1, len(chunks) + 1))
