from pydantic import BaseModel


class PDFUploadResponse(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    size_bytes: int
    upload_status: str