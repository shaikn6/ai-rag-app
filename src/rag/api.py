"""FastAPI surface for the RAG app."""
from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel

from .llm import ClaudeLLM, FakeLLM
from .pipeline import answer
from .store import VectorStore

app = FastAPI(title="RAG on Claude", version="0.1.0")
store = VectorStore()


def _llm():
    # Use real Claude when a key is present, else a safe stub.
    return ClaudeLLM() if os.environ.get("ANTHROPIC_API_KEY") else FakeLLM()


class Doc(BaseModel):
    text: str
    source: str = "doc"


class Query(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(store.chunks),
            "live_llm": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/ingest")
def ingest(doc: Doc):
    n = store.add(doc.text, doc.source)
    return {"ingested_chunks": n, "total_chunks": len(store.chunks)}


@app.post("/query")
def query(q: Query):
    return answer(store, _llm(), q.question, k=q.k)
