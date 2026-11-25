# Vietnamese RAG Chatbot — FastAPI + Gemini / OpenAI

A Vietnamese chatbot API that answers from your own documents, built as a
portfolio project for an AI Engineer pivot — and built the way clients
actually hire for: FastAPI, a real LLM, RAG with citations, tests, Docker
and CI, not a notebook demo.

## What it does

- `GET /health` — liveness, always cheap, no dependency checked.
- `GET /ready` — readiness: is an LLM configured, is retrieval available?
- `POST /chat {"question": "..."}` — answers in Vietnamese and returns:

```json
{
  "answer": "...",
  "citations": [{"source": "docs/gioi-thieu.md", "chunk": 0, "text": "..."}],
  "model": "gemini:gemini-3.5-flash-lite",
  "retrieved": true
}
```

The RAG flow: `docs/*.md` → chunk (500 chars, 50 overlap) → embed
(Gemini `gemini-embedding-001`, or OpenAI `text-embedding-3-small`) →
Chroma → top-3 chunks retrieved per question → grounded answer with
real citations. If the documents do not cover the question, the model
is told to say it does not know instead of inventing an answer. If
Chroma is missing or empty, `/chat` still answers from the LLM alone
with `citations: []` — a broken vector store never takes the API down.

## Production features

- **Validation:** Pydantic request/response models, question length
  limits, whitespace-only questions rejected.
- **Provider abstraction:** one `LLMProvider` interface, OpenAI
  (`gpt-4o-mini`) and Gemini (`gemini-3.5-flash-lite`, REST via `urllib`,
  no extra SDK) behind it. Only one key is needed; OpenAI wins if both
  are set.
- **Retries + timeouts:** transient provider errors (429 / 5xx) are
  retried with exponential backoff, up to 3 attempts, 60s timeout.
  A 400 from the provider is not retried.
- **Observability:** structured JSON logs, `X-Request-ID` on every
  response, `/health` + `/ready`.
- **Safety:** rate limit 60 req/min per IP on `/chat` (429 with
  `Retry-After`), CORS allow-list via env, generic 500 handler that
  never leaks a stack trace, `.env` and `chroma_db/` git-ignored.
- **Tests + CI:** `pytest` suite with fake LLM provider and fake
  retriever (no key, no Chroma needed); GitHub Actions runs
  compile check + tests on every push.
- **Deploy:** `Dockerfile` (Python 3.12-slim, non-root user,
  healthcheck) and `docker-compose.yml` with a persistent Chroma volume.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # put ONE key in .env, never commit it
python ingest.py            # index docs/ into Chroma (needs a key)
uvicorn main:app --reload
```

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG là gì?"}'
```

Docker:

```bash
docker compose up --build
```

Tests (no key needed):

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (priority if both set) |
| `GEMINI_API_KEY` | — | Gemini key, free tier at Google AI Studio |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini chat model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `CHROMA_PATH` | `./chroma_db` | Chroma persistent directory |
| `RETRIEVAL_TOP_K` | `3` | Chunks retrieved per question |
| `MAX_QUESTION_LENGTH` | `2000` | Max question length (chars) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-IP limit on `/chat` |
| `LLM_TIMEOUT_SECONDS` | `60` | LLM request timeout |
| `LOG_LEVEL` | `INFO` | Log level |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed frontend origins |

Notes: `gemini-2.5-flash-lite` is no longer available to new users (404);
`gemini-3.8-flash` was temporarily overloaded (503) when tested on
24 Nov 2026, so the default is `gemini-3.5-flash-lite`.

## Project structure

```
main.py            # FastAPI app: settings, providers, retrieval, routes
ingest.py          # chunk + embed docs/ into Chroma
docs/gioi-thieu.md # sample knowledge base (Vietnamese, about RAG)
tests/test_api.py  # API + chunking tests, no key needed
Dockerfile / docker-compose.yml
.github/workflows/ci.yml
BUOI-1.md / BUOI-2.md / BUOI-3.md   # learning notes in Vietnamese
```

`BUOI-*.md` are the step-by-step learning notes, kept in the repo on
purpose: they show how the project was built, one session at a time.

## What the Upwork JDs actually ask for

The jobs this project is built to match, checked 25 Nov 2026:

| Upwork JD | Asks for | Covered here |
|---|---|---|
| AI Developer for Website Chatbot MVP — Python, FastAPI, OpenAI & Supabase, $1,200 fixed, 2–3 weeks | Chatbot, basic RAG from website content/FAQs, FastAPI, Supabase, deploy + docs | RAG + FastAPI + deploy; Supabase next |
| Senior AI/ML Engineer — RAG, LLMs, AI Agents & Python, $15–30/h, 30+ h/week, 6+ months | Production RAG pipelines, chunking, hybrid retrieval, reranking, LangChain/LangGraph agents | RAG pipeline done; reranking + agents next |
| Full-Stack Next.js Developer for AI RAG Chatbot, $25–47/h, 3–6 months | React/Next.js + TypeScript, FastAPI, RAG (pgvector/Pinecone), LangChain/LangGraph, Supabase, Stripe, Docker, GitHub Actions | API + Docker + CI done; Next.js UI next |

## Roadmap

1. Next.js chat UI with streaming, on top of this API.
2. LangGraph agent (Project 2): a 2-node agent — generate, then check
   and shorten the answer.
3. Reranking + hybrid search, evals (answer groundedness, citation
   precision), Supabase for chat history.
