import pytest

from ragchat.config import load_settings


def test_load_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypa
def test_load_settings_rejects_invalid_chunk_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="smaller"):
        load_settings("test-key")


def test_load_settings_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        load_settings()
