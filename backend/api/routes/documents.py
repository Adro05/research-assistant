"""
Document upload, extraction, chunking, and embedding endpoints.

Accepts a PDF upload, validates it, and stores it under data/uploads/.
Also exposes text extraction, chunking, and embedding for an
already-uploaded document. Route handlers stay thin — validation/storage
logic lives in upload_service, extraction logic lives in
extraction_service, chunking logic lives in chunking_service, and
embedding logic lives in embedding_service.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.core.config import get_settings
from backend.schemas.chunking import ChunkingResponse, TextChunkSchema
from backend.schemas.document import PDFUploadResponse
from backend.schemas.embedding import ChunkEmbeddingSchema, EmbeddingResponse
from backend.schemas.extraction import ExtractedPageSchema, PDFExtractionResponse
from backend.services.chunking_service import chunk_document
from backend.services.embedding_service import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    embed_chunking_result,
    get_embedding_model,
)
from backend.services.extraction_service import (
    DocumentNotFoundError,
    InvalidDocumentIdError,
    PDFExtractionError,
    extract_pdf_text,
)
from backend.services.upload_service import (
    UploadValidationError,
    sanitize_filename,
    save_pdf,
    validate_extension,
    validate_not_empty,
    validate_pdf_signature,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_uploads_dir() -> Path:
    """
    FastAPI dependency resolving the upload storage directory.

    Exposed as a dependency (rather than read directly inside the route)
    so tests can override it to point at temporary storage instead of the
    real data/uploads/ directory.
    """
    return Path(get_settings().uploads_dir)


@router.post(
    "/upload",
    response_model=PDFUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: UploadFile = File(...),
    uploads_dir: Path = Depends(get_uploads_dir),
) -> PDFUploadResponse:
    """Validate and store an uploaded PDF file."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file was provided.")

    try:
        safe_filename = sanitize_filename(file.filename)
        validate_extension(safe_filename)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    content = await file.read()

    try:
        validate_not_empty(content)
        validate_pdf_signature(content)
    except UploadValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        stored = save_pdf(content, safe_filename, uploads_dir)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store the uploaded file.",
        ) from exc

    return PDFUploadResponse(
        document_id=stored.document_id,
        original_filename=stored.original_filename,
        stored_filename=stored.stored_filename,
        size_bytes=stored.size_bytes,
        upload_status="success",
    )


@router.post(
    "/{document_id}/extract",
    response_model=PDFExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
) -> PDFExtractionResponse:
    """Extract page-by-page text from an already-uploaded, stored PDF."""
    try:
        result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    return PDFExtractionResponse(
        document_id=result.document_id,
        stored_filename=result.stored_filename,
        total_pages=result.total_pages,
        pages=[ExtractedPageSchema(page_number=p.page_number, text=p.text) for p in result.pages],
    )


@router.post(
    "/{document_id}/chunks",
    response_model=ChunkingResponse,
    status_code=status.HTTP_200_OK,
)
async def chunk_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
) -> ChunkingResponse:
    """Extract and chunk text from an already-uploaded, stored PDF."""
    try:
        extraction_result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    chunking_result = chunk_document(extraction_result)

    return ChunkingResponse(
        document_id=chunking_result.document_id,
        total_pages=chunking_result.total_pages,
        total_chunks=chunking_result.total_chunks,
        chunks=[
            TextChunkSchema(
                document_id=c.document_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                chunk_index_in_page=c.chunk_index_in_page,
                text=c.text,
                char_count=c.char_count,
            )
            for c in chunking_result.chunks
        ],
    )


def get_embedding_model_instance() -> Any:
    """
    FastAPI dependency resolving the embedding model instance.

    Exposed as a dependency (rather than resolved directly inside the
    route) so tests can override it with a lightweight, deterministic
    fake model instead of loading a real Hugging Face model over the
    network. The real model is loaded lazily and cached by
    get_embedding_model() itself, keyed by the configured model name.

    FastAPI resolves dependencies before entering the route body, so any
    exception raised here would otherwise bypass the route's own
    try/except and surface as an unhandled 500. EmbeddingModelUnavailableError
    is therefore caught and converted to HTTPException(503) right here,
    at the point of failure, rather than relying on the route to catch it.
    """
    settings = get_settings()
    try:
        return get_embedding_model(settings.embedding_model_name)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding model is currently unavailable.",
        ) from exc


@router.post(
    "/{document_id}/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
)
async def embed_document_text(
    document_id: str,
    uploads_dir: Path = Depends(get_uploads_dir),
    embedding_model: Any = Depends(get_embedding_model_instance),
) -> EmbeddingResponse:
    """Extract, chunk, and embed text from an already-uploaded, stored PDF."""
    try:
        extraction_result = extract_pdf_text(document_id, uploads_dir)
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PDFExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the stored document.",
        ) from exc

    chunking_result = chunk_document(extraction_result)

    try:
        embedding_result = embed_chunking_result(chunking_result, model=embedding_model)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding model is currently unavailable.",
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embeddings for the document.",
        ) from exc

    return EmbeddingResponse(
        document_id=embedding_result.document_id,
        model_name=embedding_result.model_name,
        embedding_dimension=embedding_result.embedding_dimension,
        total_chunks=embedding_result.total_chunks,
        embeddings=[
            ChunkEmbeddingSchema(
                document_id=e.document_id,
                page_number=e.page_number,
                chunk_index=e.chunk_index,
                chunk_index_in_page=e.chunk_index_in_page,
                text=e.text,
                embedding=e.embedding,
                model_name=e.model_name,
            )
            for e in embedding_result.embeddings
        ],
    )