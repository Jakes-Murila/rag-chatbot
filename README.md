# PDF RAG Chatbot

A Streamlit application for asking grounded questions across one or more uploaded PDFs. It extracts text page by page, creates provenance-preserving chunks, retrieves the most relevant excerpts with FAISS, and asks an OpenAI model to answer only from that evidence.

> This application is intended for text-based PDFs. Image-only/scanned PDFs need OCR before upload.

## How it works

```text
PDF uploads → text extraction + cleanup → page-aware chunks → OpenAI embeddings + FAISS
                                                                    ↓
Question + recent chat history → relevant excerpts → grounded OpenAI answer + source citations
```

Every chunk retains its original filename and page number. The answer prompt instructs the model to say when the evidence does not answer the question, and the UI shows the retrieved source pages separately.

## Setup

Prerequisites: Python 3.10 or newer and an OpenAI API key with access to the selected model.

```powershell
git clone https://github.com/Jakes-Murila/rag-chatbot.git
cd rag-chatbot
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the app:

```powershell
streamlit run app.py
```

On macOS/Linux, activate the virtual environment with `source .venv/bin/activate`.

### Streamlit secrets (optional)

Instead of `.env`, create `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "your_key_here"
```

Both `.env` and Streamlit secrets are ignored by Git. Never commit a real API key.

## Configuration

All optional configuration uses environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model used for answers. |
| `RAG_CHUNK_SIZE` | `1200` | Characters in each retrieval chunk. |
| `RAG_CHUNK_OVERLAP` | `200` | Shared characters between chunks. Must be smaller than the chunk size. |
| `RAG_RETRIEVAL_K` | `4` | Number of relevant chunks passed to the model. |

## Using the app

1. Add one or more PDF files in the sidebar. A new or changed set is indexed once for the current browser session.
2. Ask a focused question in the chat box.
3. Review the retrieved filename/page citations below each answer.
4. Use **Clear chat** to retain the document index, or **Remove documents** to clear the session completely.

The vector index stays in Streamlit session state only; uploaded documents are not persisted by this project.

## Project structure

```text
app.py                    # Streamlit UI and session-state lifecycle
ragchat/
  config.py               # Environment validation and RAG settings
  documents.py            # PDF extraction, cleanup, and chunking
  rag.py                  # Embeddings, FAISS retrieval, and grounded generation
tests/                    # Fast unit tests with no real API calls
.env.example              # Safe environment-variable template
```

## Development

Run the unit tests after activating the virtual environment:

```powershell
pytest
```

Ruff settings are included in `pyproject.toml`; if you use Ruff locally, run `ruff check .`. Tests cover configuration validation, PDF cleanup and page metadata, chunk provenance, and source-citation formatting. Calls to OpenAI are intentionally not made in tests.

## Troubleshooting

- **“Add OPENAI_API_KEY”**: Copy `.env.example` to `.env`, add a valid key, and restart Streamlit.
- **Indexing fails**: Verify the file is a readable, text-based PDF and that your API key can create embeddings.
- **No answer found**: Rephrase the question or upload a PDF that contains the needed information; the assistant should not infer facts absent from the retrieved excerpts.
- **Scanned PDF**: Run OCR first. `pypdf` extracts embedded text but does not perform OCR.

## License

This project is available under the [MIT License](LICENSE).
