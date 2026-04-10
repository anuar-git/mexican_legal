"""Embedding generation for text chunks.

Calls the Pinecone inference API with llama-text-embed-v2 to produce
dense vector embeddings for each TextChunk.

Features:
- Batch processing (configurable batch size, default 64)
- tqdm progress bar
- Checkpointing every N batches to data/processed/checkpoint.json
- Statistics logging on completion
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

_MODEL = "llama-text-embed-v2"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class EmbeddedChunk(TextChunk):
    """A TextChunk with a dense vector embedding attached.

    Inherits all TextChunk fields and adds:
        embedding: Dense vector produced by ``_MODEL``.
    """

    embedding: list[float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pinecone_client() -> Pinecone:
    """Load credentials and return an authenticated Pinecone client."""
    load_dotenv()
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file or export it before running."
        )
    return Pinecone(api_key=api_key)


def _save_checkpoint(
    path: Path,
    embedded: list[EmbeddedChunk],
    total_chunks: int,
) -> None:
    """Persist current progress to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": _MODEL,
                "total_chunks": total_chunks,
                "processed_count": len(embedded),
                "embedded_chunks": [c.model_dump() for c in embedded],
            },
            f,
        )
    logger.info("Checkpoint saved: %d/%d chunks", len(embedded), total_chunks)


def _load_checkpoint(path: Path) -> list[EmbeddedChunk] | None:
    """Return previously embedded chunks from ``path``, or None if absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    embedded = [EmbeddedChunk.model_validate(c) for c in data["embedded_chunks"]]
    logger.info(
        "Resuming from checkpoint: %d chunks already embedded.", len(embedded)
    )
    return embedded


def _log_stats(embedded: list[EmbeddedChunk], elapsed: float) -> None:
    """Log summary statistics after embedding is complete."""
    total = len(embedded)
    avg_tokens = sum(c.token_count for c in embedded) / total
    avg_words = sum(c.word_count for c in embedded) / total
    dims = len(embedded[0].embedding) if embedded else 0

    logger.info("=== Embedding Statistics ===")
    logger.info("Total chunks processed : %d", total)
    logger.info("Average token count    : %.1f", avg_tokens)
    logger.info("Average word count     : %.1f", avg_words)
    logger.info("Embedding dimensions   : %d", dims)
    logger.info(
        "Total time             : %.1fs  (%.3fs per chunk)",
        elapsed,
        elapsed / total if total else 0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[TextChunk],
    batch_size: int = 64,
    checkpoint_every: int = 10,
    checkpoint_path: str = "data/processed/checkpoint.json",
) -> list[EmbeddedChunk]:
    """Embed a list of TextChunks using llama-text-embed-v2 via Pinecone inference.

    Processes chunks in batches, saves a checkpoint after every
    ``checkpoint_every`` batches, and resumes automatically from the most
    recent checkpoint if one exists.

    Args:
        chunks: Output of ``recursive_chunk()`` or any chunking strategy.
        batch_size: Number of chunks per API call (32–64 recommended).
        checkpoint_every: Save checkpoint after this many batches.
        checkpoint_path: Path to the JSON checkpoint file.

    Returns:
        List of ``EmbeddedChunk`` objects in the same order as ``chunks``.
    """
    ckpt_file = Path(checkpoint_path)
    pc = _get_pinecone_client()

    # Resume from checkpoint if available
    embedded: list[EmbeddedChunk] = _load_checkpoint(ckpt_file) or []
    start_index = len(embedded)

    if start_index >= len(chunks):
        logger.info("All %d chunks already embedded — nothing to do.", len(chunks))
        return embedded

    remaining = chunks[start_index:]
    batches = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

    start_time = time.time()

    with tqdm(total=len(remaining), desc="Embedding chunks", unit="chunk") as pbar:
        for batch_num, batch in enumerate(batches):
            response = pc.inference.embed(
                model=_MODEL,
                inputs=[c.text for c in batch],
                parameters={"input_type": "passage", "truncate": "END"},
            )

            for chunk, emb in zip(batch, response, strict=True):
                embedded.append(
                    EmbeddedChunk(**chunk.model_dump(), embedding=emb.values)
                )

            pbar.update(len(batch))

            if (batch_num + 1) % checkpoint_every == 0:
                _save_checkpoint(ckpt_file, embedded, len(chunks))

    # Always save final state
    _save_checkpoint(ckpt_file, embedded, len(chunks))

    elapsed = time.time() - start_time
    _log_stats(embedded, elapsed)

    return embedded
