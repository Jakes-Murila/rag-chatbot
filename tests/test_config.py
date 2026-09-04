import pytest

from ragchat.config import load_settings


def test_load_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "600")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("RAG_RETRIEVAL_K", "2")

    settings = load_settings("test-key")

    assert settings.openai_api_key == "test-key"
    assert settings.model == "test-model"
    assert (settings.chunk_size, settings.chunk_overlap, settings.retrieval_k) == (600, 100, 2)


def test_load_settings_rejects_invalid_chunk_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="smaller"):
        load_settings("test-key")


def test_load_settings_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        load_settings()
