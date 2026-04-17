"""
RAG Service — Manages the vector store connection for knowledge retrieval.

Supports two backends controlled by settings.USE_PINECONE:
  - Pinecone (production)  — cloud-hosted, free tier covers 1 school
  - ChromaDB (development) — local persistent store, zero cost

Usage:
    from app.services.rag_service import RagService
    chunks = await RagService.query("What is the syllabus for Term 1 Chemistry?")
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger


class RagService:
    """
    Singleton wrapper around the active vector store backend.
    Instantiated at startup via RagService.init() in lifespan.
    """
    _vectorstore: Any = None
    _embeddings: Any = None
    _ready: bool = False

    # ── Initialisation ────────────────────────────────────────────────────

    @classmethod
    async def init(cls) -> None:
        """
        Called once at server startup.
        Initialises the embeddings + vector store. Safe to call even if
        GOOGLE_API_KEY or PINECONE_API_KEY is missing — it will log a warning
        and set _ready=False. RAG queries will then return empty results.
        """
        try:
            cls._embeddings = cls._build_embeddings()

            if settings.USE_PINECONE:
                cls._vectorstore = await asyncio.to_thread(cls._build_pinecone)
                logger.info("RAG backend: Pinecone", index=settings.PINECONE_INDEX_NAME)
            else:
                cls._vectorstore = await asyncio.to_thread(cls._build_chroma)
                logger.info("RAG backend: ChromaDB (local)", path=settings.CHROMA_PERSIST_PATH)

            cls._ready = True

        except Exception as exc:
            logger.warning(
                "RAG service failed to initialise — RAG queries will return empty",
                error=str(exc),
            )
            cls._ready = False

    # ── Query ─────────────────────────────────────────────────────────────

    @classmethod
    async def query(cls, text: str, top_k: int | None = None) -> list[str]:
        """
        Retrieve the top-k most relevant text chunks for a given query.
        Returns a list of strings (page content only, no metadata).
        """
        if not cls._ready or cls._vectorstore is None:
            logger.warning("RAG service not ready — returning empty context")
            return []

        k = top_k or settings.RAG_TOP_K
        try:
            docs = await asyncio.to_thread(
                cls._vectorstore.similarity_search, text, k=k
            )
            chunks = [doc.page_content for doc in docs]
            logger.debug("RAG retrieved", query=text[:60], chunks=len(chunks))
            return chunks
        except Exception as exc:
            logger.error("RAG query failed", error=str(exc))
            return []

    # ── Ingestion ─────────────────────────────────────────────────────────

    @classmethod
    async def ingest_pdf(cls, file_path: str, metadata: dict | None = None) -> int:
        """
        Chunk a PDF file and upsert its embeddings into the vector store.
        Returns the number of chunks ingested.

        Args:
            file_path: Absolute path to the PDF file.
            metadata: Optional dict merged into each chunk's metadata
                      (e.g., {"doc_type": "syllabus", "subject": "Chemistry"})
        """
        if not cls._ready or cls._vectorstore is None:
            raise RuntimeError("RAG service not initialised. Check API keys.")

        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(file_path)
        raw_docs = await asyncio.to_thread(loader.load)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(raw_docs)

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        await asyncio.to_thread(cls._vectorstore.add_documents, chunks)
        logger.info(
            "RAG: PDF ingested",
            file=Path(file_path).name,
            chunks=len(chunks),
        )
        return len(chunks)

    @classmethod
    async def ingest_text(cls, text: str, metadata: dict | None = None) -> int:
        """
        Chunk raw text and upsert into vector store.
        Useful for ingesting teacher notes or pasted content.
        """
        if not cls._ready or cls._vectorstore is None:
            raise RuntimeError("RAG service not initialised. Check API keys.")

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.create_documents([text], metadatas=[metadata or {}])

        await asyncio.to_thread(cls._vectorstore.add_documents, chunks)
        logger.info("RAG: text ingested", chunks=len(chunks))
        return len(chunks)

    # ── Collection Management ─────────────────────────────────────────────

    @classmethod
    async def list_documents(cls) -> list[dict]:
        """
        Returns a list of unique source documents stored in the vector store.
        Works for ChromaDB; Pinecone requires a separate stats call.
        """
        if not cls._ready:
            return []
        try:
            if not settings.USE_PINECONE:
                # ChromaDB: peek at metadata
                collection = cls._vectorstore._collection
                result = await asyncio.to_thread(collection.get, include=["metadatas"])
                seen = {}
                for meta in result.get("metadatas", []):
                    src = meta.get("source", "unknown")
                    if src not in seen:
                        seen[src] = meta
                return list(seen.values())
            else:
                # Pinecone: return index stats
                index_stats = await asyncio.to_thread(
                    cls._vectorstore._index.describe_index_stats
                )
                return [{"backend": "pinecone", "stats": index_stats}]
        except Exception as exc:
            logger.error("RAG list_documents failed", error=str(exc))
            return []

    @classmethod
    async def clear(cls) -> None:
        """Wipe all vectors from the store. Admin-only operation."""
        if not cls._ready:
            raise RuntimeError("RAG service not initialised.")
        try:
            if settings.USE_PINECONE:
                await asyncio.to_thread(cls._vectorstore._index.delete, delete_all=True)
            else:
                cls._vectorstore._collection.delete(
                    where={"source": {"$ne": "__placeholder__"}}
                )
            logger.warning("RAG: knowledge base cleared")
        except Exception as exc:
            logger.error("RAG clear failed", error=str(exc))
            raise

    @classmethod
    async def ping(cls) -> bool:
        """Health check — returns True if RAG is ready."""
        return cls._ready

    # ── Private builders ──────────────────────────────────────────────────

    @staticmethod
    def _build_embeddings():
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GOOGLE_API_KEY,
        )

    @classmethod
    def _build_pinecone(cls):
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        # Create index if it doesn't exist (free plan: 1536 dims for text-embedding-004)
        existing = [idx.name for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=768,          # text-embedding-004 output dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("RAG: Pinecone index created", name=settings.PINECONE_INDEX_NAME)

        index = pc.Index(settings.PINECONE_INDEX_NAME)
        return PineconeVectorStore(index=index, embedding=cls._embeddings)

    @classmethod
    def _build_chroma(cls):
        import chromadb
        from langchain_community.vectorstores import Chroma

        persistent_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)
        return Chroma(
            client=persistent_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=cls._embeddings,
        )
