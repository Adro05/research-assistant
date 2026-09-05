"""
Text chunking.

Splits page-level extracted text (produced by
backend/services/extraction_service.py) into overlapping, size-bounded
chunks suitable for a future embedding/retrieval stage.

This module is intentionally independent of the filesystem, PDFs, and
FastAPI: it only consumes plain text and the extraction service's result
dataclasses (`PDFExtractionResult` / `ExtractedPage`), and returns plain
dataclasses. It performs no I/O of its own.

Strategy: recursive character-based splitting with paragraph/sentence/word
aware breakpoints, falling back to a hard character cut only when no
natural separator can produce a small-enough piece.

Design rules (fixed for this milestone):
- Chunking never crosses a page boundary — every chunk belongs to exactly
  one page, preserving document/page provenance for future citation
  mapping.
- Output is fully deterministic: the same input always produces the same
  chunks, in the same order.
- An empty or whitespace-only page produces zero chunks.
- A page shorter than the target chunk size produces exactly one chunk
  containing the full page text, unpadded.
- Every chunk is guaranteed to be no longer than `chunk_size`. Overlap is
  carried forward as a tail of up to `overlap` characters from the end of
  the previous chunk; that tail is shrunk whenever necessary to keep the
  new chunk within `chunk_size`.
"""

from dataclasses import dataclass

from backend.services.extraction_service import PDFExtractionResult


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150


# Tried in order, from "most natural" to "least natural" breakpoint.
# Each separator is kept attached to the end of the preceding piece so
# that splitting and rejoining the pieces reconstructs the original text
# exactly.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]


@dataclass
class TextChunk:
    """A single chunk of text, traceable back to its exact source location."""

    document_id: str
    page_number: int
    chunk_index: int
    chunk_index_in_page: int
    text: str
    char_count: int


@dataclass
class ChunkingResult:
    """Structured chunking result for one document."""

    document_id: str
    total_pages: int
    total_chunks: int
    chunks: list[TextChunk]


def _split_text_units(
    text: str,
    max_size: int,
    separators: list[str],
) -> list[str]:
    """
    Recursively break `text` into pieces no longer than `max_size`,
    preferring the earliest-listed separator that appears in the text.

    Concatenating the returned pieces in order always reconstructs `text`
    exactly. This function never drops or reorders characters; it only
    decides where to cut.
    """
    if len(text) <= max_size:
        return [text] if text else []

    if not separators:
        # No natural separator could produce small-enough pieces.
        # Fall back to a hard character cut.
        return [
            text[i : i + max_size]
            for i in range(0, len(text), max_size)
        ]

    separator = separators[0]

    if separator not in text:
        return _split_text_units(
            text,
            max_size,
            separators[1:],
        )

    parts = text.split(separator)
    pieces: list[str] = []

    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            # Reattach the separator to the preceding piece so no
            # characters are lost.
            pieces.append(part + separator)
        elif part:
            # The final part is included only if non-empty.
            # If it is empty, the separator was trailing and has already
            # been preserved in the previous piece.
            pieces.append(part)

    result: list[str] = []

    for piece in pieces:
        if not piece:
            continue

        if len(piece) <= max_size:
            result.append(piece)
        else:
            result.extend(
                _split_text_units(
                    piece,
                    max_size,
                    separators[1:],
                )
            )

    return result


def _pack_units(
    units: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Greedily pack small text units into chunks, carrying an overlap tail
    from the end of one chunk into the start of the next.

    Every returned chunk is guaranteed to be no longer than `chunk_size`.
    """
    if not units:
        return []

    chunks: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit

        elif len(current) + len(unit) <= chunk_size:
            current += unit

        else:
            chunks.append(current)

            # Carry forward as much overlap as possible while still
            # keeping the next chunk within chunk_size.
            max_tail = max(
                0,
                min(overlap, chunk_size - len(unit)),
            )

            tail = (
                current[-max_tail:]
                if max_tail > 0
                else ""
            )

            current = tail + unit

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split raw text into overlapping, size-bounded pieces.

    Returns an empty list for empty text. Text shorter than `chunk_size`
    is returned as a single-element list containing the full text.
    """
    if not text:
        return []

    units = _split_text_units(
        text,
        chunk_size,
        _SEPARATORS,
    )

    return _pack_units(
        units,
        chunk_size,
        overlap,
    )


def chunk_page(
    document_id: str,
    page_number: int,
    text: str,
    start_chunk_index: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """
    Chunk a single page's text into `TextChunk` objects.

    Returns an empty list if the page's text is empty or whitespace-only.
    `start_chunk_index` lets the caller assign globally sequential
    `chunk_index` values across a multi-page document.
    """
    if not text.strip():
        return []

    pieces = chunk_text(
        text,
        chunk_size,
        overlap,
    )

    return [
        TextChunk(
            document_id=document_id,
            page_number=page_number,
            chunk_index=start_chunk_index + position,
            chunk_index_in_page=position,
            text=piece,
            char_count=len(piece),
        )
        for position, piece in enumerate(pieces)
    ]


def chunk_document(
    extraction: PDFExtractionResult,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkingResult:
    """
    Chunk every page of an extraction result, preserving page boundaries
    as hard boundaries and assigning globally sequential chunk indices.
    """
    chunks: list[TextChunk] = []

    for page in extraction.pages:
        page_chunks = chunk_page(
            document_id=extraction.document_id,
            page_number=page.page_number,
            text=page.text,
            start_chunk_index=len(chunks),
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(page_chunks)

    return ChunkingResult(
        document_id=extraction.document_id,
        total_pages=extraction.total_pages,
        total_chunks=len(chunks),
        chunks=chunks,
    )