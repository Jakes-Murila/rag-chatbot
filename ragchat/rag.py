"""Vector indexing, grounded retrieval, and answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragchat.config import Settings

SYSTEM_PROMPT = """You answer questions using only the supplied PDF excerpts.
If the excerpts do not contain the answer, say that you could not find it in the uploaded documents.
Do not invent facts, sources, filenames, or page numbers. Be concise and clearly distinguish uncertainty.
At the end of the answer, include a 'Sources:' line containing the relevant [filename, p. page] citations."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]


def build_vector_store(documents: Sequence[Document], settings: Settings) -> FAISS:
    """Embed a non-empty document collection into an in-memory FAISS store."""
    if not documents:
        raise ValueError("Cannot build an index without document chunks.")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return FAISS.from_documents(list(documents), embeddings)


def retrieve(vector_store: FAISS, question: str, k: int) -> list[Document]:
    """Retrieve documents for a non-empty user question."""
    if not question.strip():
        raise ValueError("Ask a question before searching the documents.")
    return vector_store.similarity_search(question, k=k)


def format_sources(documents: Iterable[Document]) -> list[str]:
    """Return unique, presentation-ready citations in retrieval order."""
    citations: list[str] = []
    for document in documents:
        filename = document.metadata.get("filename", "Unknown file")
        page = document.metadata.get("page", "?")
        citation = f"{filename}, p. {page}"
        if citation not in citations:
            citations.append(citation)
    return citations


def answer_question(
    vector_store: FAISS,
    question: str,
    history: Sequence[BaseMessage],
    settings: Settings,
) -> Answer:
    """Generate a grounded answer and return explicit source metadata for the UI."""
    documents = retrieve(vector_store, question, settings.retrieval_k)
    context = "\n\n---\n\n".join(
        f"[{doc.metadata['filename']}, p. {doc.metadata['page']}]\n{doc.page_content}"
        for doc in documents
    )
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    messages.extend(history[-8:])
    messages.append(HumanMessage(content=f"PDF excerpts:\n{context}\n\nQuestion: {question}"))
    response = ChatOpenAI(model=settings.model, api_key=settings.openai_api_key, temperature=0).invoke(
        messages
    )
    if not isinstance(response, AIMessage):
        raise RuntimeError("The language model returned an unexpected response.")
    return Answer(text=str(response.content), sources=format_sources(documents))
