#!/bin/sh
# Start the API. On first run with a key set, build the search index first
# (ingest.py) so /chat can retrieve real citations; without a key the app
# still starts (RAG stays degraded until a key is set).
set -e
if [ -n "$GEMINI_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
  if [ ! -d "chroma_db" ]; then
    echo "Building search index (first run)..."
    python ingest.py || echo "Index build failed; starting without retrieval."
  fi
fi
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
