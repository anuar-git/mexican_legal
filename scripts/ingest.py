#!/usr/bin/env python3
"""Ingestion pipeline CLI — extract, chunk, embed, upsert.

Run from the project root:

    # Full rebuild
    python scripts/ingest.py --source data/raw/ --strategy recursive --rebuild

    # Incremental update (only new chunks)
    python scripts/ingest.py --source data/raw/ --strategy recursive --update

    # Dry run (extract + chunk + embed, skip upsert)
    python scripts/ingest.py --source data/raw/ --strategy recursive --dry-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# When executed as `python scripts/ingest.py`, the project root is not
# automatically on sys.path. Insert it so `src.*` imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import recursive_chunk
from src.ingestion.embedder import embed_chunks
from src.ingestion.extractor import extract_pages
from src.ingestion.loader import upsert_chunks


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Set up stdout logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="Ingest legal PDFs into Pinecone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy to use (default: recursive).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--rebuild",
        action="store_const",
        const="rebuild",
        dest="mode",
        help="Delete the Pinecone index and re-ingest from scratch.",
    )
    mode.add_argument(
        "--update",
        action="store_const",
        const="update",
        dest="mode",
        help="Only upsert chunks not already present in the index.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_const",
        const="dry-run",
        dest="mode",
        help="Extract, chunk and embed — but skip the Pinecone upsert.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_extract(source_dir: Path, log: logging.Logger):
    """Discover PDFs and extract pages from all of them."""
    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        log.error("No PDF files found in %s.", source_dir)
        sys.exit(1)

    log.info(
        "Found %d PDF file(s): %s",
        len(pdf_files),
        [f.name for f in pdf_files],
    )

    all_pages = []
    for pdf in pdf_files:
        t0 = time.time()
        pages = extract_pages(str(pdf))
        log.info(
            "  %-45s  %3d pages  (%.1fs)",
            pdf.name,
            len(pages),
            time.time() - t0,
        )
        all_pages.extend(pages)

    log.info("Total pages extracted: %d", len(all_pages))
    return all_pages


def _step_chunk(pages, strategy: str, log: logging.Logger):
    """Split pages into chunks using the chosen strategy."""
    if strategy == "recursive":
        chunks = recursive_chunk(pages)
    else:
        log.error("Strategy '%s' is not yet implemented.", strategy)
        sys.exit(1)

    log.info(
        "Total chunks produced: %d  (avg %.0f tokens)",
        len(chunks),
        sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
    )
    return chunks


def _step_save_chunks(chunks, log: logging.Logger) -> Path:
    """Serialise chunks to data/processed/chunks.json for offline inspection."""
    out_path = Path("data/processed/chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    log.info("Chunks saved to %s  (%.1f KB,  %d chunks)", out_path, size_kb, len(chunks))
    return out_path


def _step_embed(chunks, log: logging.Logger):
    """Generate embeddings for all chunks."""
    embedded = embed_chunks(chunks)
    dims = len(embedded[0].embedding) if embedded else 0
    log.info(
        "Total chunks embedded: %d  (embedding dims: %d)",
        len(embedded),
        dims,
    )
    return embedded


def _step_upsert(embedded, mode: str, log: logging.Logger) -> int:
    """Upsert vectors to Pinecone, or report what would have been upserted."""
    if mode == "dry-run":
        log.info(
            "Dry run — skipping upsert. Would have sent %d vectors.", len(embedded)
        )
        return 0

    upserted = upsert_chunks(embedded, mode=mode)
    log.info("Vectors upserted this run: %d", upserted)
    return upserted


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    _configure_logging()
    args = _parse_args()
    log = logging.getLogger("ingest")

    log.info("=" * 60)
    log.info("Ingestion pipeline starting")
    log.info("  source   : %s", args.source)
    log.info("  strategy : %s", args.strategy)
    log.info("  mode     : %s", args.mode)
    log.info("=" * 60)

    if not args.source.is_dir():
        log.error("--source '%s' is not a directory.", args.source)
        sys.exit(1)

    pipeline_start = time.time()

    log.info("Step 1/5 — Extracting pages")
    pages = _step_extract(args.source, log)

    log.info("Step 2/5 — Chunking  (strategy=%s)", args.strategy)
    chunks = _step_chunk(pages, args.strategy, log)

    log.info("Step 3/5 — Saving chunks to data/processed/chunks.json")
    _step_save_chunks(chunks, log)

    log.info("Step 4/5 — Embedding  (model=llama-text-embed-v2)")
    embedded = _step_embed(chunks, log)

    log.info("Step 5/5 — Upserting  (mode=%s)", args.mode)
    _step_upsert(embedded, args.mode, log)

    log.info("=" * 60)
    log.info("Pipeline complete in %.1fs.", time.time() - pipeline_start)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
