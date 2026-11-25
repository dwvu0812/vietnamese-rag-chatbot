"""Vietnamese RAG chatbot — production-grade FastAPI service.

What this is
------------
A small, honest RAG API: it answers in Vietnamese from your own documents
when they exist, and says so when they do not. It is built to the shape
Upwork clients actually hire for (FastAPI + LLM + RAG + vector search),
not as a demo that only calls an LLM.

Design decisions (the parts clients check)
-------------------------------------------
- One LLM provider, chosen from the environment. Only one key is needed:
  ``OPENAI_API_KEY`` -> gpt-4o-mini (takes priority if both keys are set)
  ``GEMINI_API_KEY`` -> gemini-3.5-flash-lite via the Gemini REST API,
  called with ``urllib`` (no extra SDK). Free tier, no card needed.
- Retrieval is optional and degrading, never fatal: if ``chroma_db/`` is
  missing or empty, ``/chat`` still answers, with ``citations: []`` and a
  log line saying retrieval was skipped. A broken vector store must not
  take the whole API down.
- No answer is invented from thin air when documents exist: the system
  prompt is grounded in the retrieved chunks, and the model is told to
  say it does not know when the documents do not cover the question.
- Everything that can fail at runtime is explicit: missing key -> 500 with
  a clear message, bad input -> 400/422, LLM provider error -> 502,
  rate limit -> 429. Stack traces never leak to the client.

Run it
------
::

    cp .env.example .env   # add ONE key
    pip install -r requirements.txt
    python ingest.py       # optional: index docs/ into Chroma first
    uvicorn main:app --reload

Then ``GET /health`` and ``POST /chat {"question": "RAG là gì?"}``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ---------------------------------------------------------------------------
# Logging: one JSON line per event, with the request id attached.
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("rag_chatbot")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.propagate = False


# ---------------------------------------------------------------------------
# Settings: every knob comes from the environment, with a safe default.
# A missing LLM key is not a startup error — the app still boots so
# /health works, and /chat explains the problem (500) instead.
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
    )
    chroma_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db")
    )
    chroma_collection: str = "docs"
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "3"))
    )
    max_question_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    )
    llm_max_retries: int = 3
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
            ).split(",")
            if o.strip()
        ]
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or self.gemini_api_key)

    @property
    def provider_name(self) -> str:
        if self.openai_api_key:
            return f"openai:{self.openai_model}"
        if self.gemini_api_key:
            return f"gemini:{self.gemini_model}"
        return "none"


settings = Settings()
logger.setLevel(settings.log_level.upper())
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{settings.gemini_model}:generateContent"
)

# ---------------------------------------------------------------------------
# API models: the contract. /chat always returns this shape, whether or
# not retrieval found anything — clients never have to guess.
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    source: str
    chunk: int
    text: str = ""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        # Runs before any dependency (e.g. the LLM provider), so a blank
        # question is rejected with 422 even when no API key is set.
        if not value.strip():
            raise ValueError("question trong rong")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    model: str
    retrieved: bool = False


class HealthResponse(BaseModel):
    status: str
    provider: str


class ReadyResponse(BaseModel):
    status: str  # "ok" or "degraded"
    provider: str
    retrieval_available: bool


# ---------------------------------------------------------------------------
# LLM providers: one interface, two implementations. Retries only cover
# transient provider errors (429 / 5xx); a 400 from the provider is
# returned as-is, retrying it would never help.
# ---------------------------------------------------------------------------


class LLMProvider(Protocol):
    name: str

    def generate(self, system_prompt: str, question: str) -> str: ...


def _retryable(status: int | None) -> bool:
    return status in (429, 500, 502, 503, 504)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))


class OpenAIProvider:
    name = f"openai:{settings.openai_model}"

    def __init__(self) -> None:
        self._client = OpenAI(timeout=settings.llm_timeout_seconds)

    def generate(self, system_prompt: str, question: str) -> str:
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - mapped below
                last_error = exc
                status = getattr(exc, "status_code", None)
                logger.warning("openai error attempt=%d status=%s", attempt, status)
                if not _retryable(status):
                    break
                _backoff_sleep(attempt)
        raise HTTPException(
            status_code=502, detail=f"LLM provider error: {last_error}"
        )


class GeminiProvider:
    name = f"gemini:{settings.gemini_model}"

    def generate(self, system_prompt: str, question: str) -> str:
        body = json.dumps(
            {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\n{question}"}
                        ]
                    }
                ]
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries):
            req = urllib.request.Request(GEMINI_URL, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("x-goog-api-key", settings.gemini_api_key or "")
            try:
                with urllib.request.urlopen(req, timeout=settings.llm_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = exc.read().decode("utf-8", errors="replace")
                logger.warning(
                    "gemini error attempt=%d status=%s", attempt, exc.code
                )
                if not _retryable(exc.code):
                    raise HTTPException(
                        status_code=502, detail=f"Gemini loi {exc.code}: {detail}"
                    ) from exc
                _backoff_sleep(attempt)
                continue
            except Exception as exc:  # noqa: BLE001 - network error, retry
                last_error = exc
                logger.warning("gemini network error attempt=%d: %s", attempt, exc)
                _backoff_sleep(attempt)
                continue
            candidates = data.get("candidates", [])
            if not candidates:
                raise HTTPException(
                    status_code=502, detail=f"Gemini tra ve rong: {data}"
                )
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        raise HTTPException(status_code=502, detail=f"LLM provider error: {last_error}")


def get_llm_provider() -> LLMProvider:
    """FastAPI dependency — tests override this with a fake provider."""
    if settings.openai_api_key:
        return OpenAIProvider()
    if settings.gemini_api_key:
        return GeminiProvider()
    raise HTTPException(
        status_code=500,
        detail="Chua co OPENAI_API_KEY hoac GEMINI_API_KEY trong .env",
    )


# ---------------------------------------------------------------------------
# Retrieval: Chroma, lazy and fault-tolerant. Imported lazily so the API
# boots and serves /health even on a machine without chromadb installed.
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: int
    text: str


class ChromaRetriever:
    def __init__(self, settings_: Settings) -> None:
        self._settings = settings_
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        client = chromadb.PersistentClient(path=self._settings.chroma_path)
        self._collection = client.get_or_create_collection(
            self._settings.chroma_collection
        )
        return self._collection

    def available(self) -> bool:
        try:
            return self._get_collection().count() > 0
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            return False

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return []
            query_embedding = embed_text(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self._settings.retrieval_top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # noqa: BLE001 - never fail /chat for retrieval
            logger.exception("retrieval failed, continuing without context")
            return []
        chunks: list[RetrievedChunk] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    source=str(meta.get("source", "docs")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                )
            )
        return chunks


def embed_text(text: str) -> list[float]:
    """Embed one text with the same provider priority as /chat.

    Gemini: gemini-embedding-001 via REST. OpenAI: text-embedding-3-small.
    """
    if settings.gemini_api_key:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_embedding_model}:embedContent"
        )
        body = json.dumps(
            {
                "model": f"models/{settings.gemini_embedding_model}",
                "content": {"parts": [{"text": text}]},
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("x-goog-api-key", settings.gemini_api_key)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini embedding loi {exc.code}",
            ) from exc
        return [float(v) for v in data["embedding"]["values"]]
    if settings.openai_api_key:
        client = OpenAI(timeout=settings.llm_timeout_seconds)
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return [float(v) for v in resp.data[0].embedding]
    raise HTTPException(
        status_code=500,
        detail="Chua co OPENAI_API_KEY hoac GEMINI_API_KEY trong .env",
    )


_retriever: ChromaRetriever | None = None


def get_retriever() -> ChromaRetriever:
    """FastAPI dependency — tests override this with a fake retriever."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever(settings)
    return _retriever


# ---------------------------------------------------------------------------
# App: request id, rate limit, CORS, error handling.
# ---------------------------------------------------------------------------

app = FastAPI(title="rag-chatbot-starter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


_rate_lock = threading.Lock()
_rate_hits: dict[str, list[float]] = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path != "/chat":
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        hits = [t for t in _rate_hits.get(client_ip, []) if now - t < 60]
        if len(hits) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests, try again in a minute."},
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        _rate_hits[client_ip] = hits
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled error path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request_id_var.get()},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness only: the process is up. No dependency is checked here."""
    return HealthResponse(status="ok", provider=settings.provider_name)


@app.get("/ready", response_model=ReadyResponse)
def ready(retriever: ChromaRetriever = Depends(get_retriever)):
    """Readiness: can this instance actually answer a question?"""
    retrieval_available = retriever.available()
    status = "ok" if settings.llm_configured else "degraded"
    return ReadyResponse(
        status=status,
        provider=settings.provider_name,
        retrieval_available=retrieval_available,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    provider: LLMProvider = Depends(get_llm_provider),
    retriever: ChromaRetriever = Depends(get_retriever),
):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question trong rong")
    if len(question) > settings.max_question_length:
        raise HTTPException(
            status_code=400,
            detail=f"question dai qua {settings.max_question_length} ky tu",
        )

    chunks = retriever.retrieve(question)
    retrieved = bool(chunks)
    if retrieved:
        context = "\n\n".join(
            f"[{c.source} chunk {c.chunk_index}]\n{c.text}" for c in chunks
        )
        system_prompt = (
            "Tra loi ngan gon, tieng Viet, chi dua tren tai lieu duoi day. "
            "Neu tai lieu khong co thong tin thi noi thang la khong biet, "
            "dung doan mo.\n\n"
            f"{context}"
        )
    else:
        logger.info("no retrieval context, answering from LLM only")
        system_prompt = "Tra loi ngan gon, tieng Viet, dung nhu cau hoi."

    answer = provider.generate(system_prompt, question)
    return ChatResponse(
        answer=answer,
        citations=[
            Citation(source=c.source, chunk=c.chunk_index, text=c.text)
            for c in chunks
        ],
        model=provider.name,
        retrieved=retrieved,
    )
