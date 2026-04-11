"""Chunking strategies for extracted legal document pages.

Provides two selectable strategies:
- "recursive": RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
- "semantic":  Semantic chunking (planned, Strategy 2)
"""

import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from src.ingestion.extractor import ExtractedPage

# Encoding used for token counting. cl100k_base (GPT-4) is the closest
# publicly available approximation to Claude's tokenizer.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text`` using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


class TextChunk(BaseModel):
    """A single chunk of text produced by a chunking strategy.

    Attributes:
        text: The chunk content.
        source_file: Path to the originating PDF.
        page_number: 1-based page number within the PDF.
        chunk_index: 0-based position of this chunk across all pages.
        start_char: Start character offset within the page's raw text.
        end_char: End character offset within the page's raw text.
        token_count: Number of tokens in ``text``.
        word_count: Number of whitespace-delimited words in ``text``.
        chunk_strategy: Chunking strategy that produced this chunk.
        article_number: Article number active at this chunk (e.g. "13 BIS").
        title: Document title heading active at this chunk (e.g. "SEGUNDO").
        chapter: Chapter heading active at this chunk (e.g. "IV").
        section: Section heading active at this chunk (e.g. "PRIMERA").
        is_continuation: True when this chunk continues an article that started
            in a previous chunk (i.e. the article was too long to fit in one chunk).
        prev_chunk_index: chunk_index of the immediately preceding chunk, or None.
        next_chunk_index: chunk_index of the immediately following chunk, or None.
    """

    text: str
    source_file: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    word_count: int
    chunk_strategy: str
    article_number: str | None = None
    title: str | None = None
    chapter: str | None = None
    section: str | None = None
    is_continuation: bool = False
    prev_chunk_index: int | None = None
    next_chunk_index: int | None = None


# ---------------------------------------------------------------------------
# Hierarchy extraction helpers
# ---------------------------------------------------------------------------

# Matches ALL-CAPS article headers anywhere in a chunk (e.g. "ARTÍCULO 13",
# "ARTÍCULO20 BIS"). Uppercase-only so that lowercase body references such as
# "conforme al artículo 82" are intentionally excluded.
_RE_ARTICLE_HEADER = re.compile(r'ARTÍCULO\s*(\d+\s*(?:BIS|TER|QUÁTER)?)')

# Matches the same header only at the very start of a chunk (after optional
# whitespace). Used to distinguish a chunk that OPENS a new article from one
# that merely contains an article reference mid-way through.
_RE_ARTICLE_START = re.compile(
    r'^\s*ART[ÍI]CULO\s*(\d+\s*(?:BIS|TER|QUÁTER)?)',
    re.IGNORECASE,
)

# Matches heading lines anywhere in the chunk to update running hierarchy state.
_RE_TITLE   = re.compile(r'(?:^|\n)\s*T[ÍI]TULO\s+([^\n]+)',   re.IGNORECASE)
_RE_CHAPTER = re.compile(r'(?:^|\n)\s*CAP[ÍI]TULO\s+([^\n]+)', re.IGNORECASE)
_RE_SECTION = re.compile(r'(?:^|\n)\s*SECCI[ÓO]N\s+([^\n]+)',  re.IGNORECASE)


def _first_match(pattern: re.Pattern, text: str) -> str | None:
    """Return the first captured group of ``pattern`` in ``text``, or None."""
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _opening_article(text: str) -> str | None:
    """Return the article number if ``text`` opens with an article header, else None.

    Only matches at the very start of the chunk so that lowercase body references
    like "conforme al artículo 82" do not trigger a false positive.
    """
    m = _RE_ARTICLE_START.match(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Separator list
# ---------------------------------------------------------------------------

# Ordered from most to least structurally significant. The splitter tries each
# in turn and only falls back when a segment still exceeds chunk_size.
# Both ALL-CAPS and mixed-case variants are listed for ARTÍCULO because the
# CDMX Penal Code uses ALL-CAPS in headers but mixed-case in body references.
_LEGAL_SEPARATORS: list[str] = [
    "\nARTÍCULO ",  # Article boundary — ALL-CAPS as used in the PDF headers
    "\nArtículo ",  # Mixed-case fallback (other legal documents)
    "\nCAPÍTULO ",  # Chapter boundary
    "\nTÍTULO ",    # Title boundary
    "\nSECCIÓN ",   # Section boundary
    "\n\n",          # Paragraph break
    "\n",            # Line break
    " ",             # Word boundary
    "",              # Character-level last resort
]


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

def recursive_chunk(pages: list[ExtractedPage]) -> list[TextChunk]:
    """Split extracted pages into token-bounded chunks using recursive character splitting.

    Applies LangChain's RecursiveCharacterTextSplitter with legal-document-aware
    separators. Chunk size is measured in tokens (via ``count_tokens``) rather
    than characters, so the 512-token ceiling is consistent with the model's
    context window.

    Each chunk is annotated with the active document hierarchy (title, chapter,
    section, article) at the point it was produced, plus continuation and
    neighbor-pointer metadata.

    Args:
        pages: Output of ``extract_pages()``, one item per PDF page.

    Returns:
        A flat list of ``TextChunk`` objects in document order. ``chunk_index``
        is a global counter across all pages so downstream components can sort
        or reference chunks without needing page context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=count_tokens,
        separators=_LEGAL_SEPARATORS,
        is_separator_regex=False,
    )

    chunks: list[TextChunk] = []
    chunk_index = 0

    # Running document hierarchy — updated as headings are encountered.
    current_title:   str | None = None
    current_chapter: str | None = None
    current_section: str | None = None
    current_article: str | None = None

    for page in pages:
        raw_text = page.raw_text
        if not raw_text.strip():
            continue

        split_texts = splitter.split_text(raw_text)

        # Cursor tracks position in raw_text so character offsets can be computed
        # even when the same phrase appears multiple times.
        search_start = 0

        for split in split_texts:
            start_char = raw_text.find(split, search_start)
            if start_char == -1:
                # Splitter occasionally strips leading/trailing whitespace;
                # fall back to the current cursor position.
                start_char = search_start
            end_char = start_char + len(split)
            search_start = start_char + 1

            # Update hierarchy state from headings found anywhere in this chunk.
            # Title / chapter / section headings are rarer than articles and may
            # appear mid-chunk when the splitter falls through to paragraph breaks.
            if t := _first_match(_RE_TITLE, split):
                current_title = t
            if c := _first_match(_RE_CHAPTER, split):
                current_chapter = c
            if s := _first_match(_RE_SECTION, split):
                current_section = s

            # is_continuation is determined solely by whether the chunk OPENS
            # with a new article header — not by mid-chunk article appearances.
            is_continuation = not bool(_opening_article(split)) and current_article is not None

            # Advance current_article to the LAST uppercase article header found
            # anywhere in the chunk. This handles the case where a whole page fits
            # in one chunk (no split occurs) yet the page introduces new articles.
            headers_in_chunk = _RE_ARTICLE_HEADER.findall(split)
            if headers_in_chunk:
                current_article = headers_in_chunk[-1].strip()

            chunks.append(
                TextChunk(
                    text=split,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=count_tokens(split),
                    word_count=len(split.split()),
                    chunk_strategy="recursive",
                    article_number=current_article,
                    title=current_title,
                    chapter=current_chapter,
                    section=current_section,
                    is_continuation=is_continuation,
                    prev_chunk_index=None,   # filled in second pass below
                    next_chunk_index=None,   # filled in second pass below
                )
            )
            chunk_index += 1

    # Second pass: neighbor pointers can only be set after all chunks exist.
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_index = chunks[i - 1].chunk_index if i > 0 else None
        chunk.next_chunk_index = chunks[i + 1].chunk_index if i < len(chunks) - 1 else None

    return chunks
