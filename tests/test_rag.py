from langchain_core.documents import Document

from ragchat.rag import format_sources


    assert format_sources(documents) == ["alpha.pdf, p. 1", "beta.pdf, p. 4"]
