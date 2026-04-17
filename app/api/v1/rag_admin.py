"""
POST /v1/rag/ingest   — Upload a PDF or text into the knowledge base
GET  /v1/rag/list     — List all ingested documents
DELETE /v1/rag/clear  — Wipe the entire vector store (super-admin only)
GET  /v1/rag/health   — Check if RAG backend is ready
"""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.rag_service import RagService
from app.core.logging import logger

router = APIRouter()


class IngestResponse(BaseModel):
    success: bool
    filename: str
    chunks_ingested: int
    backend: str


class HealthResponse(BaseModel):
    ready: bool
    backend: str


@router.get("/rag/health", response_model=HealthResponse, tags=["RAG Admin"])
async def rag_health():
    """Check if the RAG vector store is initialised and ready."""
    from app.core.config import settings
    ready = await RagService.ping()
    return HealthResponse(
        ready=ready,
        backend="pinecone" if settings.USE_PINECONE else "chromadb",
    )


@router.post("/rag/ingest", response_model=IngestResponse, tags=["RAG Admin"])
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest into the knowledge base"),
    doc_type: str = Form(default="general", description="e.g. syllabus | policy | notes | exam"),
    subject: str = Form(default="", description="Subject name (optional)"),
    class_name: str = Form(default="", description="Class name (optional)"),
):
    """
    Upload a PDF and chunk it into the vector store.

    - Supports: PDF files only (for now)
    - Chunks are tagged with doc_type, subject, and class metadata
    - Admin/Super-Admin only (add JWT check in production)
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported for ingestion.",
        )

    if not await RagService.ping():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready. Check your API keys in .env.",
        )

    # Save upload to a temp file (PyPDFLoader requires a file path)
    try:
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        metadata = {
            "source": file.filename,
            "doc_type": doc_type,
            "subject": subject,
            "class_name": class_name,
        }

        chunks_count = await RagService.ingest_pdf(tmp_path, metadata=metadata)

    except Exception as exc:
        logger.error("RAG ingest failed", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    from app.core.config import settings
    return IngestResponse(
        success=True,
        filename=file.filename,
        chunks_ingested=chunks_count,
        backend="pinecone" if settings.USE_PINECONE else "chromadb",
    )


@router.post("/rag/ingest-text", tags=["RAG Admin"])
async def ingest_text(
    content: str = Form(..., description="Raw text content to ingest"),
    source_name: str = Form(..., description="Descriptive name for this document"),
    doc_type: str = Form(default="notes"),
    subject: str = Form(default=""),
):
    """
    Ingest raw text (teacher notes, pasted content) into the knowledge base.
    Useful for adding handwritten notes or class summaries without a PDF.
    """
    if not await RagService.ping():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready. Check your API keys in .env.",
        )

    try:
        metadata = {"source": source_name, "doc_type": doc_type, "subject": subject}
        chunks_count = await RagService.ingest_text(content, metadata=metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {exc}",
        )

    from app.core.config import settings
    return {
        "success": True,
        "source": source_name,
        "chunks_ingested": chunks_count,
        "backend": "pinecone" if settings.USE_PINECONE else "chromadb",
    }


@router.get("/rag/list", tags=["RAG Admin"])
async def list_documents():
    """
    List all documents currently stored in the knowledge base.
    Returns unique source names and their metadata.
    """
    docs = await RagService.list_documents()
    return {"documents": docs, "count": len(docs)}


@router.delete("/rag/clear", tags=["RAG Admin"])
async def clear_knowledge_base():
    """
    ⚠️ DESTRUCTIVE: Wipe all vectors from the knowledge base.
    This cannot be undone. Use only to reset the school's document store.
    Super-Admin only — add JWT role check in production.
    """
    try:
        await RagService.clear()
        return {"success": True, "message": "Knowledge base cleared successfully."}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {exc}",
        )
