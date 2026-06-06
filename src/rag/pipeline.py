"""RAG orchestration: retrieve → ground → generate."""
from __future__ import annotations
from .llm import LLM
from .store import VectorStore

SYSTEM = (
    "You are a precise assistant. Answer ONLY from the provided context. "
    "Cite sources as [source]. If the answer is not in the context, say "
    "'I don't know based on the provided documents.'"
)


def answer(store: VectorStore, llm: LLM, question: str, k: int = 3) -> dict:
    hits = store.search(question, k=k)
    if not hits:
        return {"answer": "No documents ingested yet.", "sources": []}
    context = "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return {
        "answer": llm.complete(SYSTEM, user),
        "sources": sorted({c.source for c, _ in hits}),
    }
