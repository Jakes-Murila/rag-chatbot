"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings for indexing and answering questions."""

    openai_api_key: str
    model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 4


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def load_settings(api_key: str | None = None) -> Settings:
    """Load settings from the environment, allowing Streamlit to provide a key."""
    load_dotenv()
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "An OpenAI API key is required. Set OPENAI_API_KEY in .env or Streamlit secrets."
        )

    chunk_size = _positive_int("RAG_CHUNK_SIZE", 1200)
    chunk_overlap = _positive_int("RAG_CHUNK_OVERLAP", 200)
    if chunk_overlap >= chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

    return Settings(
        openai_api_key=key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=_positive_int("RAG_RETRIEVAL_K", 4),
    )
