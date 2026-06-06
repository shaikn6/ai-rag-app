# ai-rag-app

Retrieval-Augmented Generation API on **Claude** (Anthropic SDK) + FastAPI.
Demonstrates AI Engineering: embeddings, vector retrieval, grounded LLM answers, eval.

## Stack
- **LLM**: Anthropic Claude (`claude-sonnet-4-6`) via official SDK
- **Retrieval**: in-memory cosine vector store (swap for pgvector/Pinecone in prod)
- **API**: FastAPI — `/ingest`, `/query`, `/health`
- **Quality**: pytest with a mocked LLM (no API key needed for CI), ruff, Docker, CI

## Quickstart
```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...      # only needed for live /query
uvicorn rag.api:app --reload             # http://localhost:8000/docs
pytest                                   # runs offline (LLM mocked)
```

## How it works
```
/ingest  docs ─▶ chunk ─▶ embed ─▶ VectorStore
/query   q ─▶ embed ─▶ top-k retrieve ─▶ build grounded prompt ─▶ Claude ─▶ answer + sources
```

## Design notes
- LLM + embedder behind interfaces (`llm.py`, `embeddings.py`) → swappable, testable
- Prompt grounds Claude on retrieved context and instructs it to cite or say "I don't know"
- Tests inject a fake LLM → deterministic, key-free CI
