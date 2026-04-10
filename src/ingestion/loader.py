"""Pinecone vector loading for embedded chunks.

Upserts EmbeddedChunk vectors to a Pinecone index with full metadata.
Supports two modes:

- "rebuild": Delete the existing index and re-ingest everything from scratch.
- "update":  Only upsert chunks whose ID is not already in the index.

Environment variables (loaded from .env):
    PINECONE_API_KEY     — required
    PINECONE_INDEX_NAME  — required
    PINECONE_CLOUD       — optional, default "aws"
    PINECONE_REGION      — optional, default "us-east-1"
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from src.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

_METRIC = "cosine"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client_and_config() -> tuple[Pinecone, str, str, str]:
    """Load credentials and return (client, index_name, cloud, region)."""
    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set.")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")
    return Pinecone(api_key=api_key), index_name, cloud, region


def _make_chunk_id(chunk: EmbeddedChunk) -> str:
    """Return a stable, human-readable ID for a chunk.

    Format: ``{source_file_stem}-{chunk_index}``
    Example: ``codigo_penal_cdmx_31225-42``
    """
    stem = Path(chunk.source_file).stem
    return f"{stem}-{chunk.chunk_index}"


def _to_metadata(chunk: EmbeddedChunk) -> dict:
    """Serialise chunk fields to a Pinecone-compatible metadata dict.

    Pinecone does not support None values. Optional string fields are
    coerced to "" and optional int fields to -1 so filters behave
    predictably (e.g. ``article_number != ""``).
    """
    return {
        "text":             chunk.text,
        "source_file":      chunk.source_file,
        "page_number":      chunk.page_number,
        "chunk_index":      chunk.chunk_index,
        "start_char":       chunk.start_char,
        "end_char":         chunk.end_char,
        "token_count":      chunk.token_count,
        "word_count":       chunk.word_count,
        "chunk_strategy":   chunk.chunk_strategy,
        "article_number":   chunk.article_number or "",
        "title":            chunk.title or "",
        "chapter":          chunk.chapter or "",
        "section":          chunk.section or "",
        "is_continuation":  chunk.is_continuation,
        "prev_chunk_index": chunk.prev_chunk_index if chunk.prev_chunk_index is not None else -1,
        "next_chunk_index": chunk.next_chunk_index if chunk.next_chunk_index is not None else -1,
    }


def _ensure_index(
    pc: Pinecone,
    index_name: str,
    dims: int,
    cloud: str,
    region: str,
) -> None:
    """Create the index if it does not exist yet."""
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        logger.info(
            "Creating index '%s'  dims=%d  metric=%s  cloud=%s  region=%s",
            index_name, dims, _METRIC, cloud, region,
        )
        pc.create_index(
            name=index_name,
            dimension=dims,
            metric=_METRIC,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for index to become ready...")
            time.sleep(2)
        logger.info("Index '%s' is ready.", index_name)
    else:
        logger.info("Index '%s' already exists.", index_name)


def _fetch_existing_ids(index) -> set[str]:
    """Page through the index and return the full set of stored vector IDs."""
    existing: set[str] = set()
    for id_batch in index.list():
        existing.update(id_batch)
    logger.info("Found %d existing vectors in index.", len(existing))
    return existing


def _upsert_batches(
    index,
    chunks: list[EmbeddedChunk],
    batch_size: int,
) -> int:
    """Upsert ``chunks`` in batches. Returns the number of vectors upserted."""
    total_upserted = 0
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

    with tqdm(total=len(chunks), desc="Upserting vectors", unit="vec") as pbar:
        for batch in batches:
            vectors = [
                {
                    "id":       _make_chunk_id(c),
                    "values":   c.embedding,
                    "metadata": _to_metadata(c),
                }
                for c in batch
            ]
            index.upsert(vectors=vectors)
            total_upserted += len(batch)
            pbar.update(len(batch))

    return total_upserted


def _log_stats(index, upserted_count: int, elapsed: float) -> None:
    """Log upsert results and current index statistics."""
    stats = index.describe_index_stats()
    logger.info("=== Pinecone Loader Statistics ===")
    logger.info("Vectors upserted this run : %d", upserted_count)
    logger.info("Total vectors in index    : %d", stats.total_vector_count)
    logger.info(
        "Index fullness            : %.4f%%",
        (stats.index_fullness or 0) * 100,
    )
    logger.info("Elapsed time              : %.1fs", elapsed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: list[EmbeddedChunk],
    mode: str = "update",
    batch_size: int = 100,
) -> int:
    """Upsert embedded chunks into the Pinecone index.

    Args:
        chunks:     Output of ``embed_chunks()``.
        mode:       ``"rebuild"`` deletes the index and re-ingests everything.
                    ``"update"`` only upserts chunks not already in the index.
        batch_size: Vectors per upsert call (Pinecone recommends 100).

    Returns:
        Number of vectors actually upserted in this run.

    Raises:
        ValueError: If required environment variables are missing.
        ValueError: If ``mode`` is not ``"rebuild"`` or ``"update"``.
    """
    if mode not in ("rebuild", "update"):
        raise ValueError(f"mode must be 'rebuild' or 'update', got {mode!r}")
    if not chunks:
        logger.warning("No chunks provided — nothing to upsert.")
        return 0

    pc, index_name, cloud, region = _get_client_and_config()
    dims = len(chunks[0].embedding)
    start_time = time.time()

    # ── rebuild ──────────────────────────────────────────────────────────────
    if mode == "rebuild":
        existing = {idx.name for idx in pc.list_indexes()}
        if index_name in existing:
            logger.info("Rebuild mode: deleting index '%s'.", index_name)
            pc.delete_index(index_name)
            # Wait for deletion to propagate
            while index_name in {idx.name for idx in pc.list_indexes()}:
                logger.info("Waiting for index deletion...")
                time.sleep(2)

        _ensure_index(pc, index_name, dims, cloud, region)
        index = pc.Index(index_name)
        to_upsert = chunks

    # ── update ────────────────────────────────────────────────────────────────
    else:
        _ensure_index(pc, index_name, dims, cloud, region)
        index = pc.Index(index_name)
        existing_ids = _fetch_existing_ids(index)

        to_upsert = [
            c for c in chunks
            if _make_chunk_id(c) not in existing_ids
        ]
        skipped = len(chunks) - len(to_upsert)
        logger.info(
            "Update mode: %d new chunks to upsert, %d already present.",
            len(to_upsert), skipped,
        )

    if not to_upsert:
        logger.info("Nothing new to upsert.")
        return 0

    upserted = _upsert_batches(index, to_upsert, batch_size)
    elapsed = time.time() - start_time
    _log_stats(index, upserted, elapsed)

    return upserted
