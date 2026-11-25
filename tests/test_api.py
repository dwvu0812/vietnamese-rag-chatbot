"""Tests for the RAG chatbot API.

Run:  pytest -q
No API key and no Chroma database are needed: the LLM provider and
the retriever are replaced with fakes via FastAPI dependency overrides.
"""

from fastapi.testclient import TestClient

import main
from main import RetrievedChunk, app


class FakeProvider:
    name = "fake:test"

    def generate(self, system_prompt: str, question: str) -> str:
        return f"Tra loi gia cho: {question}"


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def available(self) -> bool:
        return bool(self._chunks)

    def retrieve(self, question: str):
        return self._chunks


def make_client(chunks=None):
    app.dependency_overrides[main.get_llm_provider] = lambda: FakeProvider()
    app.dependency_overrides[main.get_retriever] = lambda: FakeRetriever(chunks)
    return TestClient(app)


def test_health_ok():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "X-Request-ID" in resp.headers


def test_ready_reports_retrieval():
    client = make_client(chunks=[RetrievedChunk("docs/a.md", 0, "hello")])
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["retrieval_available"] is True


def test_chat_empty_question_rejected():
    client = make_client()
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_without_retrieval_returns_empty_citations():
    client = make_client()
    resp = client.post("/chat", json={"question": "RAG là gì?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert body["retrieved"] is False
    assert "RAG là gì?" in body["answer"]


def test_chat_with_retrieval_returns_real_citations():
    chunks = [RetrievedChunk("docs/gioi-thieu.md", 0, "RAG la Retrieval")]
    client = make_client(chunks=chunks)
    resp = client.post("/chat", json={"question": "RAG là gì?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source"] == "docs/gioi-thieu.md"
    assert body["citations"][0]["chunk"] == 0


def test_chat_invalid_body_without_key_returns_500():
    # FastAPI solves dependencies before validating the body, so with no
    # LLM key configured an invalid body still gets the 500 "no key" error
    # instead of 422. With a key set (the real case) it is 422 — see the
    # live-server check. This test pins the no-key behaviour.
    app.dependency_overrides.pop(main.get_llm_provider, None)
    app.dependency_overrides[main.get_retriever] = lambda: FakeRetriever()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code == 500


def test_chat_without_api_key_returns_500():
    # No provider override: the real dependency raises 500 when no key
    # is configured. (In CI there is no .env, so this is the real path.)
    app.dependency_overrides.pop(main.get_llm_provider, None)
    app.dependency_overrides[main.get_retriever] = lambda: FakeRetriever()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/chat", json={"question": "hello"})
    assert resp.status_code == 500


def test_chat_retrieval_end_to_end_with_chroma(tmp_path, monkeypatch):
    """Real Chroma retrieval path, with embeddings faked (no API key)."""
    import chromadb

    from main import ChromaRetriever, Settings

    monkeypatch.setattr(main, "embed_text", lambda text: [1.0, 0.0, 0.0])
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("docs")
    collection.add(
        ids=["docs/gioi-thieu.md-0"],
        documents=["RAG la Retrieval-Augmented Generation"],
        metadatas=[{"source": "docs/gioi-thieu.md", "chunk_index": 0}],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    retriever = ChromaRetriever(Settings(chroma_path=str(tmp_path)))
    app.dependency_overrides[main.get_llm_provider] = lambda: FakeProvider()
    app.dependency_overrides[main.get_retriever] = lambda: retriever
    resp = TestClient(app).post("/chat", json={"question": "RAG là gì?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieved"] is True
    assert body["citations"][0]["source"] == "docs/gioi-thieu.md"


def test_ingest_chunking_needs_no_api_key():
    from ingest import chunk_documents

    chunks = chunk_documents([("docs/a.md", "x" * 1200)])
    assert len(chunks) > 1
    assert all(len(text) <= 500 for _, _, text in chunks)
    assert chunks[0][0] == "docs/a.md"
