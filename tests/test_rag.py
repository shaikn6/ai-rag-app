from rag.store import VectorStore
from rag.llm import FakeLLM
from rag.pipeline import answer
from rag.embeddings import embed
import numpy as np


def test_embed_normalized():
    v = embed("hello world")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_store_retrieves_relevant_chunk():
    s = VectorStore()
    s.add("Cloud cost optimization uses autoscaling and spot instances.", "finops")
    s.add("Bananas are a good source of potassium.", "fruit")
    hits = s.search("how to reduce cloud spend", k=1)
    assert hits and hits[0][0].source == "finops"


def test_answer_grounds_and_cites_sources():
    s = VectorStore()
    s.add("Terraform manages infrastructure as code.", "iac")
    out = answer(s, FakeLLM(), "what is terraform?")
    assert "iac" in out["sources"]
    assert out["answer"]


def test_answer_handles_empty_store():
    out = answer(VectorStore(), FakeLLM(), "anything")
    assert out["sources"] == []
