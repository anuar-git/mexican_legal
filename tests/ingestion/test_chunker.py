"""Unit tests for src/ingestion/chunker.py — recursive chunking strategy.

Verifies:
1. No chunk exceeds the 512-token ceiling.
2. Overlap: tokens from the end of chunk N appear at the start of chunk N+1.
3. Core metadata (source_file, page_number) is faithfully propagated.
4. word_count equals len(text.split()).
5. chunk_strategy is always "recursive".
6. article_number tracks ALL-CAPS ARTÍCULO headers; body references are ignored.
7. title / chapter / section propagate from headings into subsequent chunks.
8. is_continuation correctly distinguishes chunk-opening articles from continuations.
9. prev_chunk_index / next_chunk_index form a valid doubly-linked list.
"""

from src.ingestion.chunker import TextChunk, count_tokens, recursive_chunk
from src.ingestion.extractor import ExtractedPage

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DUMMY_SOURCE = "data/raw/codigo_penal_cdmx_31225.pdf"


def make_page(text: str, page_number: int = 1, source: str = DUMMY_SOURCE) -> ExtractedPage:
    """Construct an ExtractedPage without touching the filesystem."""
    return ExtractedPage(page_number=page_number, raw_text=text, source_file=source)


def long_article(n: int) -> str:
    """Return a single article: ONE ALL-CAPS header followed by a body long enough
    to force the splitter to produce at least 3 chunks (≈1 400 tokens total)."""
    header = f"ARTÍCULO {n}. "
    sentence = (
        "El que dolosamente cause daño en propiedad ajena "
        "será sancionado con prisión de seis meses a cinco años y multa. "
        "Se considerará que hay dolo cuando el agente conoce las circunstancias "
        "del hecho típico y quiere o acepta el resultado. "
    )
    return header + sentence * 30


def page_with_hierarchy() -> str:
    """Return text that contains a TÍTULO, CAPÍTULO, SECCIÓN and two short articles."""
    return (
        "TÍTULO TERCERO\n"
        "CAPÍTULO II\n"
        "SECCIÓN PRIMERA\n"
        "ARTÍCULO 50. Primera disposición del capítulo.\n"
        "ARTÍCULO 51. Segunda disposición del capítulo.\n"
    )


# ---------------------------------------------------------------------------
# 1. Token ceiling — no chunk may exceed 512 tokens
# ---------------------------------------------------------------------------

class TestTokenCeiling:
    def test_single_short_page(self):
        """A page already under 512 tokens must come back as one chunk."""
        text = "ARTÍCULO 1. Nadie puede ser castigado por una acción u omisión."
        pages = [make_page(text)]
        chunks = recursive_chunk(pages)

        assert len(chunks) == 1
        assert chunks[0].token_count <= 512

    def test_long_page_all_chunks_within_limit(self):
        """Every chunk from a page that exceeds 512 tokens must be ≤ 512."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        assert len(chunks) > 1, "Long page should have been split into multiple chunks"
        violations = [c for c in chunks if c.token_count > 512]
        assert violations == [], (
            f"{len(violations)} chunk(s) exceeded 512 tokens: "
            + str([c.token_count for c in violations])
        )

    def test_multi_page_document_all_within_limit(self):
        """Token ceiling holds across every chunk of a multi-page document."""
        pages = [make_page(long_article(i), page_number=i) for i in range(1, 6)]
        chunks = recursive_chunk(pages)

        violations = [c for c in chunks if c.token_count > 512]
        assert violations == [], (
            f"{len(violations)} chunk(s) exceeded 512 tokens across 5 pages"
        )


# ---------------------------------------------------------------------------
# 2. Overlap — content from the end of chunk N appears in chunk N+1
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_consecutive_chunks_share_tokens(self):
        """The last tokens of chunk N must appear somewhere in chunk N+1."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        assert len(chunks) >= 2, "Need ≥2 chunks to verify overlap"

        for i in range(len(chunks) - 1):
            current_text = chunks[i].text
            next_text = chunks[i + 1].text

            # Take the last 20 words as the overlap probe — well within the
            # 64-token overlap window and robust to minor whitespace changes.
            current_tokens = current_text.split()
            probe = " ".join(current_tokens[-20:]) if len(current_tokens) >= 20 else current_text

            assert probe in next_text, (
                f"Chunk {i} tail not found in chunk {i + 1}. "
                f"Probe: {probe!r:.80}"
            )

    def test_overlap_not_full_duplicate(self):
        """Chunks must not be identical — they overlap partially, not fully."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        for i in range(len(chunks) - 1):
            assert chunks[i].text != chunks[i + 1].text, (
                f"Chunk {i} and chunk {i + 1} are identical — overlap misconfigured"
            )


# ---------------------------------------------------------------------------
# 3. Core metadata — source_file, page_number, chunk_index, char offsets
# ---------------------------------------------------------------------------

class TestMetadataPropagation:
    def test_source_file_preserved(self):
        """Every chunk must carry the source_file from its originating page."""
        custom_source = "data/raw/test_document.pdf"
        pages = [make_page(long_article(1), source=custom_source)]
        chunks = recursive_chunk(pages)

        for chunk in chunks:
            assert chunk.source_file == custom_source

    def test_page_number_preserved(self):
        """Chunks must retain the page_number of the page they came from."""
        pages = [
            make_page(long_article(1), page_number=7),
            make_page(long_article(2), page_number=42),
        ]
        chunks = recursive_chunk(pages)

        assert any(c.page_number == 7 for c in chunks), "No chunks for page 7"
        assert any(c.page_number == 42 for c in chunks), "No chunks for page 42"
        unexpected = [c for c in chunks if c.page_number not in (7, 42)]
        assert unexpected == []

    def test_chunk_index_is_globally_sequential(self):
        """chunk_index must be a gapless 0-based sequence across all pages."""
        pages = [make_page(long_article(i), page_number=i) for i in range(1, 4)]
        chunks = recursive_chunk(pages)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_are_skipped(self):
        """Pages with only whitespace must produce no chunks."""
        pages = [
            make_page("   \n\n  "),
            make_page(long_article(1), page_number=2),
        ]
        chunks = recursive_chunk(pages)

        assert all(c.page_number == 2 for c in chunks)

    def test_char_offsets_within_page_bounds(self):
        """start_char / end_char must describe a valid non-empty substring."""
        pages = [make_page(long_article(1))]
        raw_len = len(pages[0].raw_text)
        chunks = recursive_chunk(pages)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= raw_len
            assert chunk.start_char < chunk.end_char


# ---------------------------------------------------------------------------
# 4. word_count
# ---------------------------------------------------------------------------

class TestWordCount:
    def test_word_count_matches_split(self):
        """word_count must equal len(text.split()) for every chunk."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        for chunk in chunks:
            assert chunk.word_count == len(chunk.text.split()), (
                f"chunk {chunk.chunk_index}: word_count={chunk.word_count}, "
                f"len(text.split())={len(chunk.text.split())}"
            )

    def test_word_count_positive(self):
        """word_count must be > 0 — no empty chunks should be produced."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        assert all(c.word_count > 0 for c in chunks)


# ---------------------------------------------------------------------------
# 5. chunk_strategy
# ---------------------------------------------------------------------------

class TestChunkStrategy:
    def test_strategy_is_recursive(self):
        """recursive_chunk must tag every chunk with strategy='recursive'."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        assert all(c.chunk_strategy == "recursive" for c in chunks)


# ---------------------------------------------------------------------------
# 6. article_number
# ---------------------------------------------------------------------------

class TestArticleNumber:
    def test_article_number_detected(self):
        """A chunk that opens with ARTÍCULO N must have article_number='N'."""
        text = "ARTÍCULO 42. El sujeto activo responderá por el delito cometido."
        chunks = recursive_chunk([make_page(text)])

        assert len(chunks) == 1
        assert chunks[0].article_number == "42"

    def test_article_number_propagates_to_continuation_chunks(self):
        """Continuation chunks (no opening header) must inherit the last article."""
        pages = [make_page(long_article(99))]
        chunks = recursive_chunk(pages)

        # Every chunk should carry article_number "99"
        assert all(c.article_number == "99" for c in chunks), (
            "Some continuation chunks lost the article_number"
        )

    def test_article_number_updates_on_new_article(self):
        """When a new article header appears on a new page, article_number must update."""
        pages = [
            make_page("ARTÍCULO 10. Primera disposición.", page_number=1),
            make_page("ARTÍCULO 11. Segunda disposición.", page_number=2),
        ]
        chunks = recursive_chunk(pages)

        article_numbers = [c.article_number for c in chunks]
        assert "10" in article_numbers, "article_number '10' never appeared"
        assert "11" in article_numbers, "article_number '11' never appeared"

    def test_lowercase_body_reference_ignored(self):
        """Lowercase 'artículo N' in body text must not update article_number."""
        text = (
            "ARTÍCULO 5. Texto principal.\n"
            "Véase también el artículo 99 del mismo ordenamiento.\n"
        )
        chunks = recursive_chunk([make_page(text)])

        # article 99 is only a body reference; article_number must stay at 5
        assert all(c.article_number == "5" for c in chunks), (
            "Body reference 'artículo 99' incorrectly updated article_number"
        )

    def test_article_number_none_before_first_article(self):
        """Chunks that precede any article header must have article_number=None."""
        text = "TÍTULO PRIMERO\nDisposiciones generales.\n"
        chunks = recursive_chunk([make_page(text)])

        assert len(chunks) == 1
        assert chunks[0].article_number is None


# ---------------------------------------------------------------------------
# 7. Hierarchy — title, chapter, section
# ---------------------------------------------------------------------------

class TestHierarchyPropagation:
    def test_title_detected(self):
        """A chunk containing a TÍTULO heading must expose it."""
        chunks = recursive_chunk([make_page(page_with_hierarchy())])
        assert any(c.title == "TERCERO" for c in chunks)

    def test_chapter_detected(self):
        """A chunk containing a CAPÍTULO heading must expose it."""
        chunks = recursive_chunk([make_page(page_with_hierarchy())])
        assert any(c.chapter == "II" for c in chunks)

    def test_section_detected(self):
        """A chunk containing a SECCIÓN heading must expose it."""
        chunks = recursive_chunk([make_page(page_with_hierarchy())])
        assert any(c.section == "PRIMERA" for c in chunks)

    def test_hierarchy_propagates_across_pages(self):
        """Hierarchy values set on page N must carry forward onto page N+1."""
        pages = [
            make_page("TÍTULO CUARTO\nCAPÍTULO III\n", page_number=1),
            make_page(long_article(1), page_number=2),
        ]
        chunks = recursive_chunk(pages)

        page2_chunks = [c for c in chunks if c.page_number == 2]
        assert page2_chunks, "No chunks produced for page 2"
        assert all(c.title == "CUARTO" for c in page2_chunks), (
            "Title did not propagate from page 1 to page 2"
        )
        assert all(c.chapter == "III" for c in page2_chunks), (
            "Chapter did not propagate from page 1 to page 2"
        )

    def test_hierarchy_none_before_any_heading(self):
        """title/chapter/section must be None on chunks before any heading appears."""
        text = "ARTÍCULO 1. Texto sin encabezados de título o capítulo."
        chunks = recursive_chunk([make_page(text)])

        assert chunks[0].title is None
        assert chunks[0].chapter is None
        assert chunks[0].section is None


# ---------------------------------------------------------------------------
# 8. is_continuation
# ---------------------------------------------------------------------------

class TestIsContinuation:
    def test_first_chunk_of_article_is_not_continuation(self):
        """A chunk that opens with ARTÍCULO must have is_continuation=False."""
        text = "ARTÍCULO 7. Texto breve del artículo."
        chunks = recursive_chunk([make_page(text)])

        assert len(chunks) == 1
        assert chunks[0].is_continuation is False

    def test_split_article_second_chunk_is_continuation(self):
        """When an article is split, every chunk after the first must be a continuation."""
        pages = [make_page(long_article(1))]
        chunks = recursive_chunk(pages)

        assert len(chunks) >= 2, "Need ≥2 chunks to test continuation"
        assert chunks[0].is_continuation is False, "First chunk should not be a continuation"
        assert all(c.is_continuation for c in chunks[1:]), (
            "All chunks after the first should be continuations"
        )

    def test_chunk_before_any_article_is_not_continuation(self):
        """A chunk with no article context at all must have is_continuation=False."""
        text = "TÍTULO PRIMERO\nDisposiciones generales."
        chunks = recursive_chunk([make_page(text)])

        assert chunks[0].is_continuation is False


# ---------------------------------------------------------------------------
# 9. Neighbor pointers — prev_chunk_index / next_chunk_index
# ---------------------------------------------------------------------------

class TestNeighborPointers:
    def test_first_chunk_has_no_prev(self):
        """The first chunk must have prev_chunk_index=None."""
        chunks = recursive_chunk([make_page(long_article(1))])
        assert chunks[0].prev_chunk_index is None

    def test_last_chunk_has_no_next(self):
        """The last chunk must have next_chunk_index=None."""
        chunks = recursive_chunk([make_page(long_article(1))])
        assert chunks[-1].next_chunk_index is None

    def test_interior_chunks_linked_correctly(self):
        """For every interior chunk, prev and next must point to its neighbors."""
        chunks = recursive_chunk([make_page(long_article(1))])

        assert len(chunks) >= 3, "Need ≥3 chunks to test interior linking"
        for i in range(1, len(chunks) - 1):
            assert chunks[i].prev_chunk_index == chunks[i - 1].chunk_index, (
                f"chunk {i}: prev_chunk_index={chunks[i].prev_chunk_index}, "
                f"expected {chunks[i - 1].chunk_index}"
            )
            assert chunks[i].next_chunk_index == chunks[i + 1].chunk_index, (
                f"chunk {i}: next_chunk_index={chunks[i].next_chunk_index}, "
                f"expected {chunks[i + 1].chunk_index}"
            )

    def test_neighbor_pointers_span_page_boundaries(self):
        """Neighbor pointers must link across page boundaries, not reset per page."""
        pages = [
            make_page(long_article(1), page_number=1),
            make_page(long_article(2), page_number=2),
        ]
        chunks = recursive_chunk(pages)

        # Find the last chunk of page 1 and first chunk of page 2
        page1_last = [c for c in chunks if c.page_number == 1][-1]
        page2_first = [c for c in chunks if c.page_number == 2][0]

        assert page1_last.next_chunk_index == page2_first.chunk_index, (
            "Last chunk of page 1 does not point to first chunk of page 2"
        )
        assert page2_first.prev_chunk_index == page1_last.chunk_index, (
            "First chunk of page 2 does not point back to last chunk of page 1"
        )
