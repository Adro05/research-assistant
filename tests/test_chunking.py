from __future__ import annotations

from dataclasses import asdict

import pytest

from backend.services.chunking_service import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    ChunkingResult,
    TextChunk,
    _pack_units,
    _split_text_units,
    chunk_document,
    chunk_page,
    chunk_text,
)
from backend.services.extraction_service import (
    ExtractedPage,
    PDFExtractionResult,
)


DOCUMENT_ID = "b2a1c9d4-1234-4abc-8def-0123456789ab"


def _make_extraction(*page_texts: str) -> PDFExtractionResult:
    """Build a simple extraction result for chunking tests."""
    pages = [
        ExtractedPage(
            page_number=index,
            text=text,
        )
        for index, text in enumerate(page_texts, start=1)
    ]

    return PDFExtractionResult(
        document_id=DOCUMENT_ID,
        stored_filename="test.pdf",
        total_pages=len(pages),
        pages=pages,
    )


def _reconstruct_with_overlap(chunks: list[str], overlap: int) -> str:
    """
    Reconstruct the original text from overlapping chunks.

    Each chunk after the first is expected to repeat up to `overlap`
    characters from the previous chunk.
    """
    if not chunks:
        return ""

    result = chunks[0]

    for previous, current in zip(chunks, chunks[1:]):
        max_possible_overlap = min(overlap, len(previous), len(current))

        actual_overlap = 0

        for candidate in range(max_possible_overlap, -1, -1):
            if previous[-candidate:] == current[:candidate]:
                actual_overlap = candidate
                break

        result += current[actual_overlap:]

    return result


def test_default_chunk_configuration() -> None:
    """The milestone defaults are 1000 characters with 150 overlap."""
    assert DEFAULT_CHUNK_SIZE == 1000
    assert DEFAULT_OVERLAP == 150


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_short_text_returns_one_chunk() -> None:
    text = "This is a short document."

    chunks = chunk_text(text)

    assert chunks == [text]


def test_whitespace_only_page_returns_no_chunks() -> None:
    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=1,
        text="   \n\t  ",
    )

    assert chunks == []


def test_page_shorter_than_chunk_size_is_preserved_exactly() -> None:
    text = "A short page with some text.\n\nNothing should be removed."

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=150,
    )

    assert chunks == [text]


def test_split_text_units_preserves_all_characters() -> None:
    text = (
        "Paragraph one has several words.\n\n"
        "Paragraph two has several more words.\n\n"
        "Paragraph three finishes the text."
    )

    units = _split_text_units(
        text,
        max_size=40,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert "".join(units) == text
    assert all(len(unit) <= 40 for unit in units)


def test_split_text_units_falls_back_to_hard_cut() -> None:
    text = "x" * 25

    units = _split_text_units(
        text,
        max_size=10,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert units == [
        "x" * 10,
        "x" * 10,
        "x" * 5,
    ]


def test_split_text_units_prefers_earlier_separator() -> None:
    text = "First paragraph.\n\nSecond paragraph."

    units = _split_text_units(
        text,
        max_size=100,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert units == [text]


def test_chunk_size_is_never_exceeded() -> None:
    text = "word " * 1000

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_overlap_is_present_between_normal_chunks() -> None:
    text = "word " * 100

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert current.startswith(previous[-20:])


def test_character_level_reconstruction_with_overlap() -> None:
    text = (
        "This is a long piece of text designed to verify that the chunking "
        "algorithm preserves every character exactly while introducing "
        "overlap between adjacent chunks. "
        "The original text should be reconstructable from the chunks."
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    reconstructed = _reconstruct_with_overlap(
        chunks,
        overlap=20,
    )

    assert reconstructed == text


def test_character_level_reconstruction_with_small_chunks() -> None:
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )

    chunks = chunk_text(
        text,
        chunk_size=30,
        overlap=8,
    )

    reconstructed = _reconstruct_with_overlap(
        chunks,
        overlap=8,
    )

    assert reconstructed == text


def test_large_unbroken_unit_does_not_exceed_chunk_size() -> None:
    # Use varied characters rather than one repeated character so that
    # the reconstruction helper can distinguish real overlap from
    # coincidental matching characters.
    text = "".join(
        chr(ord("a") + (index % 26))
        for index in range(350)
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)

    reconstructed = _reconstruct_with_overlap(
        chunks,
        overlap=20,
    )

    assert reconstructed == text

def test_overlap_is_shrunk_when_required_by_chunk_size() -> None:
    units = [
        "a" * 80,
        "b" * 80,
    ]

    chunks = _pack_units(
        units,
        chunk_size=100,
        overlap=50,
    )

    assert chunks[0] == "a" * 80
    assert len(chunks[1]) <= 100
    assert chunks[1].endswith("b" * 80)


def test_chunk_page_preserves_document_id() -> None:
    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=3,
        text="Some page content.",
    )

    assert chunks
    assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)


def test_chunk_page_preserves_page_number() -> None:
    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=7,
        text="Some page content.",
    )

    assert chunks
    assert all(chunk.page_number == 7 for chunk in chunks)


def test_chunk_page_assigns_local_indices() -> None:
    text = "word " * 100

    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=1,
        text=text,
        chunk_size=50,
        overlap=10,
    )

    assert [chunk.chunk_index_in_page for chunk in chunks] == list(
        range(len(chunks))
    )


def test_chunk_page_honors_start_chunk_index() -> None:
    text = "word " * 100

    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=1,
        text=text,
        start_chunk_index=10,
        chunk_size=50,
        overlap=10,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(
        range(10, 10 + len(chunks))
    )


def test_chunk_char_count_matches_text_length() -> None:
    text = "word " * 100

    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=1,
        text=text,
        chunk_size=100,
        overlap=20,
    )

    for chunk in chunks:
        assert chunk.char_count == len(chunk.text)


def test_chunk_page_returns_text_chunks() -> None:
    chunks = chunk_page(
        document_id=DOCUMENT_ID,
        page_number=1,
        text="Some page content.",
    )

    assert all(isinstance(chunk, TextChunk) for chunk in chunks)


def test_chunk_document_returns_chunking_result() -> None:
    extraction = _make_extraction(
        "First page content.",
        "Second page content.",
    )

    result = chunk_document(extraction)

    assert isinstance(result, ChunkingResult)


def test_chunk_document_preserves_document_id() -> None:
    extraction = _make_extraction(
        "First page content.",
        "Second page content.",
    )

    result = chunk_document(extraction)

    assert result.document_id == DOCUMENT_ID


def test_chunk_document_preserves_total_pages() -> None:
    extraction = _make_extraction(
        "First page content.",
        "Second page content.",
        "Third page content.",
    )

    result = chunk_document(extraction)

    assert result.total_pages == 3


def test_chunk_document_reports_total_chunks() -> None:
    extraction = _make_extraction(
        "First page content.",
        "Second page content.",
    )

    result = chunk_document(extraction)

    assert result.total_chunks == len(result.chunks)


def test_chunk_document_never_crosses_page_boundaries() -> None:
    extraction = _make_extraction(
        "PAGE_ONE " * 100,
        "PAGE_TWO " * 100,
    )

    result = chunk_document(
        extraction,
        chunk_size=100,
        overlap=20,
    )

    page_one_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.page_number == 1
    ]

    page_two_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.page_number == 2
    ]

    assert page_one_chunks
    assert page_two_chunks

    assert all("PAGE_TWO" not in chunk.text for chunk in page_one_chunks)
    assert all("PAGE_ONE" not in chunk.text for chunk in page_two_chunks)


def test_chunk_document_global_indices_are_sequential() -> None:
    extraction = _make_extraction(
        "First page " * 100,
        "Second page " * 100,
        "Third page " * 100,
    )

    result = chunk_document(
        extraction,
        chunk_size=100,
        overlap=20,
    )

    assert [chunk.chunk_index for chunk in result.chunks] == list(
        range(len(result.chunks))
    )


def test_chunk_document_page_indices_restart_at_zero() -> None:
    extraction = _make_extraction(
        "First page " * 100,
        "Second page " * 100,
    )

    result = chunk_document(
        extraction,
        chunk_size=100,
        overlap=20,
    )

    for page_number in {1, 2}:
        page_chunks = [
            chunk
            for chunk in result.chunks
            if chunk.page_number == page_number
        ]

        assert [
            chunk.chunk_index_in_page
            for chunk in page_chunks
        ] == list(range(len(page_chunks)))


def test_empty_page_produces_no_chunks() -> None:
    extraction = _make_extraction(
        "First page content.",
        "",
        "Third page content.",
    )

    result = chunk_document(extraction)

    assert all(chunk.page_number != 2 for chunk in result.chunks)


def test_whitespace_page_produces_no_chunks() -> None:
    extraction = _make_extraction(
        "First page content.",
        "   \n\t\n   ",
        "Third page content.",
    )

    result = chunk_document(extraction)

    assert all(chunk.page_number != 2 for chunk in result.chunks)


def test_chunking_is_deterministic() -> None:
    text = (
        "This is deterministic test content. "
        "It contains enough text to produce multiple chunks. " * 20
    )

    first = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    second = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert first == second


def test_chunk_document_is_deterministic() -> None:
    extraction = _make_extraction(
        "First page " * 100,
        "Second page " * 100,
    )

    first = chunk_document(
        extraction,
        chunk_size=100,
        overlap=20,
    )

    second = chunk_document(
        extraction,
        chunk_size=100,
        overlap=20,
    )

    assert [asdict(chunk) for chunk in first.chunks] == [
        asdict(chunk) for chunk in second.chunks
    ]


def test_trailing_double_newline_is_preserved() -> None:
    text = "This text ends with two newlines.\n\n"

    units = _split_text_units(
        text,
        max_size=10,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert "".join(units) == text


def test_trailing_single_newline_is_preserved() -> None:
    text = "This text ends with one newline.\n"

    units = _split_text_units(
        text,
        max_size=10,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert "".join(units) == text


def test_recursive_split_piece_ending_with_separator_is_preserved() -> None:
    text = (
        "This is a long paragraph that needs recursive splitting "
        "because it is longer than the maximum size.\n\n"
    )

    units = _split_text_units(
        text,
        max_size=20,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert "".join(units) == text


def test_consecutive_trailing_separators_are_preserved() -> None:
    text = "Text with several separators.\n\n\n\n"

    units = _split_text_units(
        text,
        max_size=10,
        separators=["\n\n", "\n", ". ", " "],
    )

    assert "".join(units) == text


def test_awkward_text_preserves_every_character() -> None:
    texts = [
        "\n\n",
        "\n\n\n",
        "hello\n\n",
        "hello\n",
        "hello\n\nworld",
        "hello  world",
        "hello... world",
        "a\n\nb\n\nc\n\n",
        "  leading whitespace",
        "trailing whitespace  ",
    ]

    for text in texts:
        units = _split_text_units(
            text,
            max_size=5,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        )

        assert "".join(units) == text


def test_public_chunk_text_preserves_trailing_double_newline() -> None:
    text = (
        "This is a longer piece of text that should be split into "
        "multiple chunks while preserving the trailing newline characters."
        * 10
        + "\n\n"
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    reconstructed = _reconstruct_with_overlap(
        chunks,
        overlap=20,
    )

    assert reconstructed == text


@pytest.mark.parametrize(
    "text",
    [
        "alpha beta gamma delta epsilon",
        "one two three four five six seven",
        "paragraph one\n\nparagraph two\n\nparagraph three",
        "line one\nline two\nline three\n",
        "sentence one. sentence two! sentence three?",
    ],
)
def test_reconstruction_for_varied_text(text: str) -> None:
    chunks = chunk_text(
        text,
        chunk_size=20,
        overlap=5,
    )

    reconstructed = _reconstruct_with_overlap(
        chunks,
        overlap=5,
    )

    assert reconstructed == text


def test_chunk_document_handles_empty_document() -> None:
    extraction = _make_extraction()

    result = chunk_document(extraction)

    assert result.total_pages == 0
    assert result.total_chunks == 0
    assert result.chunks == []