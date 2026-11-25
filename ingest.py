"""Index documents into Chroma so /chat can retrieve real citations.

Usage:
    python ingest.py              # index every .md file under docs/
    python ingest.py --reset      # wipe the collection first, then index
    python ingest.py --docs-dir docs --chroma-path ./chroma_db

How it works
------------
1. Read each ``.md`` file under ``--docs-dir``.
2. Split it into chunks (500 chars, 50 overlap) with
   ``RecursiveCharacterTextSplitter`` — small enough to fit in context,
   overlapping enough not to cut a sentence in half.
3. Embed each chunk (Gemini ``gemini-embedding-001`` if a Gemini key is
   set, otherwise OpenAI ``text-embedding-3-small``) and store it in
   Chroma with metadata ``source`` + ``chunk_index``.

``/chat`` retrieves the top-3 chunks for a question and cites them, so
an answer can always be traced back to the document it came from.
No key is ever written to disk: keys come from ``.env`` only.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ingest")

# Imported here, not at module top, so `python ingest.py --help` works
# even before the embedding provider is configured.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import embed_text, settings  # noqa: E402


def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return [(source_name, text)] for every .md file under docs_dir."""
    documents: list[tuple[str, str]] = []
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("skip empty file: %s", path)
            continue
        documents.append((str(path.relative_to(docs_dir.parent)), text))
    return documents


def chunk_documents(
    documents: list[tuple[str, str]],
) -> list[tuple[str, int, str]]:
    """Split documents into (source, chunk_index, chunk_text)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[tuple[str, int, str]] = []
    for source, text in documents:
        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append((source, i, chunk))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Index docs/ into Chroma.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--chroma-path", default=settings.chroma_path)
    parser.add_argument("--reset", action="store_true",
                        help="Delete the collection before indexing.")
    args = parser.parse_args()

    if not settings.llm_configured:
        logger.error(
            "Chua co OPENAI_API_KEY hoac GEMINI_API_KEY trong .env — "
            "khong the tao embedding."
        )
        return 1

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        logger.error("Khong tim thay thu muc tai lieu: %s", docs_dir)
        return 1

    documents = load_documents(docs_dir)
    if not documents:
        logger.error("Khong co file .md nao trong %s", docs_dir)
        return 1

    chunks = chunk_documents(documents)
    logger.info("Tim thay %d file, chia thanh %d chunk.", len(documents), len(chunks))

    import chromadb

    client = chromadb.PersistentClient(path=args.chroma_path)
    if args.reset:
        try:
            client.delete_collection(settings.chroma_collection)
            logger.info("Da xoa collection cu.")
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
    collection = client.get_or_create_collection(settings.chroma_collection)

    embeddings = [embed_text(chunk) for _, _, chunk in chunks]
    collection.add(
        ids=[f"{source}-{i}" for source, i, _ in chunks],
        documents=[chunk for _, _, chunk in chunks],
        metadatas=[
            {"source": source, "chunk_index": i} for source, i, _ in chunks
        ],
        embeddings=embeddings,
    )
    logger.info(
        "Da nap %d chunk vao Chroma (%s, collection %r).",
        len(chunks), args.chroma_path, settings.chroma_collection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
