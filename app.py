"""Streamlit interface for grounded questions over uploaded PDFs."""

from __future__ import annotations

import hashlib
from typing import Protocol

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ragchat.config import Settings, load_settings
from ragchat.documents import PdfExtractionError, documents_from_uploads
from ragchat.rag import answer_question, build_vector_store

st.set_page_config(page_title="PDF RAG Chat", page_icon="📚", layout="wide")
st.title("📚 Ask your PDFs")
st.caption("Answers are generated from the uploaded documents and include page-level sources.")


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def get_secret_key() -> str | None:
    """Read the optional Streamlit secret without exposing it to the UI."""
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        return None


def upload_fingerprint(files: list[UploadedPdf]) -> str:
    """Identify the selected document set, including changed file contents."""
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()


def clear_chat() -> None:
    st.session_state.messages = []


def clear_documents() -> None:
    for key in ("vector_store", "document_fingerprint", "document_count", "messages"):
        st.session_state.pop(key, None)
    st.session_state.uploader_nonce += 1


def render_message(message: BaseMessage) -> None:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(str(message.content))


def indexing_error_message(error: Exception) -> str:
    """Turn expected provider failures into instructions that do not expose secrets."""
    error_type = type(error).__name__
    if error_type == "AuthenticationError":
        return "OpenAI rejected the API key. Check that `.env` contains the complete active key."
    if error_type in {"PermissionDeniedError", "NotFoundError"}:
        return "This API key cannot use the configured embedding model. Check its OpenAI project permissions."
    if error_type == "RateLimitError":
        return "OpenAI could not process the embedding request. Check project billing and usage limits, then retry."
    if error_type in {"APIConnectionError", "APITimeoutError"}:
        return "The app could not reach OpenAI. Check your internet connection and retry."
    return "The documents could not be indexed. Ensure the PDF contains selectable text, then try again."


try:
    settings: Settings = load_settings(get_secret_key())
except ValueError:
    st.info("Add `OPENAI_API_KEY` to a local `.env` file or `.streamlit/secrets.toml` to begin.")
    st.code("OPENAI_API_KEY=your_key_here", language="bash")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_nonce", 0)

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploads_{st.session_state.uploader_nonce}",
    )
    st.caption("Scanned/image-only PDFs need OCR before they can be searched.")
    if st.button("Clear chat", width="stretch"):
        clear_chat()
        st.rerun()
    if st.button("Remove documents", width="stretch"):
        clear_documents()
        st.rerun()
    if count := st.session_state.get("document_count"):
        st.success(f"Ready: {count} searchable chunks")

if uploaded_files:
    fingerprint = upload_fingerprint(uploaded_files)
    if fingerprint != st.session_state.get("document_fingerprint"):
        uploads = [(file.name, file.getvalue()) for file in uploaded_files]
        try:
            with st.spinner("Reading PDFs and building the searchable index…"):
                documents = documents_from_uploads(
                    uploads,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                st.session_state.vector_store = build_vector_store(documents, settings)
            st.session_state.document_fingerprint = fingerprint
            st.session_state.document_count = len(documents)
            clear_chat()
            st.error(indexing_error_message(error))

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input(
    "Ask a question about your PDFs",
    disabled="vector_store" not in st.session_state,
)
if question:
    user_message = HumanMessage(content=question)
    st.session_state.messages.append(user_message)
    render_message(user_message)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                answer = answer_question(
                    st.session_state.vector_store,
                    question,
                    st.session_state.messages[:-1],
                    settings,
                )
                st.markdown(answer.text)
                if answer.sources:
                    st.caption("Retrieved sources: " + " · ".join(answer.sources))
                st.session_state.messages.append(AIMessage(content=answer.text))
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("I couldn't answer that right now. Check your API key and try again.")
