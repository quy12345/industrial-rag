"""Compatibility facade for document ingestion and normalized chunk output."""

from app.domain.documents import SUPPORTED_EXTENSIONS as SUPPORTED_EXTENSIONS
from app.domain.documents import DocumentChunk as DocumentChunk
from app.domain.documents import IngestionError as IngestionError
from app.domain.documents import build_chunk_id as build_chunk_id
from app.domain.documents import build_document_id as build_document_id
from app.domain.documents import build_page_batches as build_page_batches
from app.domain.documents import canonical_chunk_key as canonical_chunk_key
from app.infrastructure.ingestion.docling import get_pdf_page_count as get_pdf_page_count
from app.infrastructure.ingestion.jsonl import write_chunks_jsonl as write_chunks_jsonl
from app.infrastructure.ingestion.pipeline import ingest_document as ingest_document
from app.infrastructure.ingestion.pipeline import validate_input_path as validate_input_path

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "DocumentChunk",
    "IngestionError",
    "build_chunk_id",
    "build_document_id",
    "build_page_batches",
    "canonical_chunk_key",
    "get_pdf_page_count",
    "ingest_document",
    "validate_input_path",
    "write_chunks_jsonl",
]
