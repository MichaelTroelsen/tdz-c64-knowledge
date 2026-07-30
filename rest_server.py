#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - REST API Server

FastAPI-based REST API for the C64 Knowledge Base.
Provides HTTP endpoints for all knowledge base operations.

Run with: uvicorn rest_server:app --reload --port 8000
"""

import logging
import os
import time
import io
import tempfile
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, status, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from server import KnowledgeBase
from version import __version__
import rest_models as models


# ============================================================================
# Configuration
# ============================================================================

# Get configuration from environment variables
DATA_DIR = os.getenv('TDZ_DATA_DIR', str(Path.home() / '.tdz-c64-knowledge'))
API_KEYS = os.getenv('TDZ_API_KEYS', '').split(',')
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]  # Remove empty strings
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Defaults to loopback-only: this API can add/remove/scrape (scrape_url can
# be pointed at an arbitrary internal URL - SSRF from the host) with no
# authentication unless TDZ_API_KEYS is set. Binding it to every interface
# by default would expose full read/write KB control to the whole LAN.
REST_HOST = os.getenv('API_HOST', '127.0.0.1')
REST_PORT = int(os.getenv('API_PORT', '8000'))
ALLOW_INSECURE = os.getenv('TDZ_REST_ALLOW_INSECURE', '0') == '1'


# ============================================================================
# Global Knowledge Base Instance
# ============================================================================

kb = None  # Will be initialized in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - initialize KB on startup, cleanup on shutdown."""
    global kb
    # Only initialize if not already initialized (e.g., for testing)
    if kb is None:
        print(f"Initializing Knowledge Base at {DATA_DIR}")
        kb = KnowledgeBase(DATA_DIR)
        print(f"Knowledge Base initialized with {len(kb.documents)} documents")
    else:
        print("Knowledge Base already initialized (test mode)")
    yield
    print("Shutting down REST API server")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="TDZ C64 Knowledge Base API",
    description="REST API for managing and searching Commodore 64 documentation",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ============================================================================
# CORS Middleware
# ============================================================================

# allow_origins=['*'] combined with allow_credentials=True is an invalid
# CORS configuration - browsers reject it inconsistently, and it's also
# strictly more permissive than intended (credentialed requests from any
# origin). Only allow credentials when a specific origin allowlist is set.
_cors_is_wildcard = CORS_ORIGINS == ['*'] or not CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'] if _cors_is_wildcard else CORS_ORIGINS,
    allow_credentials=not _cors_is_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from X-API-Key header."""
    # If no API keys configured, allow all requests (development mode)
    if not API_KEYS:
        return True

    # Check if provided key is valid
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standardized error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "error_type": "http_error"
        }
    )


_logger = logging.getLogger("rest_server")


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other (unhandled/unexpected) exceptions.

    Unlike the HTTPException handler above - which returns deliberate,
    caller-facing operation errors raised by endpoint code - this path
    catches genuinely unexpected failures, which can include internal file
    paths or other implementation detail. Log the detail server-side and
    return a generic message rather than echoing str(exc) to the client.
    """
    _logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_type": "internal_error"
        }
    )


# ============================================================================
# Health & Status Endpoints
# ============================================================================

# Every route handler below is a plain `def`, not `async def`, on purpose.
# KnowledgeBase is entirely synchronous and some calls block for minutes to
# hours (OCR on upload, a whole-site crawl in /scrape, entity extraction).
# Declaring the handlers `async def` while calling those methods directly meant
# they blocked uvicorn's single event loop, so one upload froze every other
# request including /health. FastAPI runs synchronous handlers on its own
# threadpool instead, which is safe now that KnowledgeBase gives each thread its
# own SQLite connection (see KnowledgeBase.db_conn in server.py). `lifespan` and
# the exception handlers stay async - they are not route handlers.

@app.get("/api/v1/health", response_model=models.HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint (no authentication required).

    Returns service health status and component availability.
    """
    try:
        database_ok = kb is not None and kb.db_file.exists()
        semantic_ok = kb is not None and hasattr(kb, 'use_semantic') and kb.use_semantic

        status_str = "healthy" if database_ok else "unhealthy"

        return models.HealthResponse(
            status=status_str,
            version=__version__,
            database_ok=database_ok,
            semantic_search_ok=semantic_ok
        )
    except Exception:
        return models.HealthResponse(
            status="unhealthy",
            version=__version__,
            database_ok=False,
            semantic_search_ok=False
        )


@app.get("/api/v1/stats", response_model=models.StatsResponse, tags=["Analytics"])
def get_stats(authenticated: bool = Depends(verify_api_key)):
    """
    Get knowledge base statistics.

    Returns:
        - Total documents, chunks, entities, relationships
        - Database size
        - Feature availability
    """
    try:
        # Get document count
        total_docs = len(kb.documents)

        # Get chunk count
        cursor = kb.db_conn.cursor()
        total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Get entity count
        try:
            total_entities = cursor.execute(
                "SELECT COUNT(DISTINCT entity_text) FROM document_entities"
            ).fetchone()[0]
        except Exception:
            total_entities = 0

        # Get relationship count
        try:
            total_relationships = cursor.execute(
                "SELECT COUNT(*) FROM entity_relationships"
            ).fetchone()[0]
        except Exception:
            total_relationships = 0

        # Get database size
        db_size_bytes = kb.db_file.stat().st_size
        db_size_mb = db_size_bytes / (1024 * 1024)

        return models.StatsResponse(
            success=True,
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_entities=total_entities,
            total_relationships=total_relationships,
            database_size_mb=round(db_size_mb, 2),
            semantic_search_enabled=kb.use_semantic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/api/v1/search", response_model=models.SearchResponse, tags=["Search"])
def search(
    request: models.SearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Basic FTS5 full-text search.

    Searches across all document chunks using SQLite FTS5.
    """
    try:
        start_time = time.time()

        results = kb.search(
            query=request.query,
            max_results=request.max_results,
            tags=request.tags
        )

        search_time_ms = (time.time() - start_time) * 1000

        search_results = [
            models.SearchResult(
                doc_id=r['doc_id'],
                title=r['title'],
                score=r['score'],
                snippet=r['snippet'],
                tags=r.get('tags', []),
                filename=r.get('filename')
            )
            for r in results
        ]

        return models.SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/v1/search/semantic", response_model=models.SearchResponse, tags=["Search"])
def semantic_search(
    request: models.SemanticSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Semantic search using embeddings.

    Finds documents by meaning/context rather than exact keywords.
    Requires semantic search to be enabled.
    """
    if not kb.use_semantic:
        raise HTTPException(
            status_code=503,
            detail="Semantic search not available. Install sentence-transformers."
        )

    try:
        start_time = time.time()

        results = kb.semantic_search(
            query=request.query,
            max_results=request.top_k,
            tags=request.tags
        )

        search_time_ms = (time.time() - start_time) * 1000

        search_results = [
            models.SearchResult(
                doc_id=r['doc_id'],
                title=r['title'],
                score=r['score'],
                snippet=r['snippet'],
                tags=r.get('tags', []),
                filename=r.get('filename')
            )
            for r in results[:request.max_results]
        ]

        return models.SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.post("/api/v1/search/hybrid", response_model=models.SearchResponse, tags=["Search"])
def hybrid_search(
    request: models.HybridSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Hybrid search combining FTS5 and semantic search.

    Blends keyword and meaning-based search for best results.
    """
    if not kb.use_semantic:
        raise HTTPException(
            status_code=503,
            detail="Hybrid search requires semantic search. Install sentence-transformers."
        )

    try:
        start_time = time.time()

        results = kb.hybrid_search(
            query=request.query,
            max_results=request.max_results,
            semantic_weight=request.semantic_weight,
            tags=request.tags
        )

        search_time_ms = (time.time() - start_time) * 1000

        search_results = [
            models.SearchResult(
                doc_id=r['doc_id'],
                title=r['title'],
                score=r['score'],
                snippet=r['snippet'],
                tags=r.get('tags', []),
                filename=r.get('filename')
            )
            for r in results
        ]

        return models.SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/api/v1/search/faceted", response_model=models.SearchResponse, tags=["Search"])
def faceted_search(
    request: models.FacetedSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Faceted search with filters.

    Search with filters for hardware, instructions, registers, etc.
    """
    try:
        start_time = time.time()

        results = kb.faceted_search(
            query=request.query,
            facet_filters=request.facet_filters or {},
            max_results=request.max_results,
            tags=request.tags
        )

        search_time_ms = (time.time() - start_time) * 1000

        search_results = [
            models.SearchResult(
                doc_id=r['doc_id'],
                title=r['title'],
                score=r['score'],
                snippet=r['snippet'],
                tags=r.get('tags', []),
                filename=r.get('filename')
            )
            for r in results
        ]

        return models.SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Faceted search failed: {str(e)}")


@app.get("/api/v1/documents/{doc_id}/similar", response_model=models.SearchResponse, tags=["Search"])
def find_similar_documents(
    doc_id: str,
    max_results: int = 10,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Find documents similar to the given document.

    Uses TF-IDF cosine similarity to find related documents.
    """
    try:
        # Check if document exists
        if doc_id not in kb.documents:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        start_time = time.time()

        results = kb.find_similar_documents(
            doc_id=doc_id,
            max_results=max_results
        )

        search_time_ms = (time.time() - start_time) * 1000

        search_results = [
            models.SearchResult(
                doc_id=r['doc_id'],
                title=r['title'],
                score=r['similarity'],
                snippet=r.get('snippet', ''),
                tags=r.get('tags', []),
                filename=r.get('filename')
            )
            for r in results
        ]

        doc_title = kb.documents[doc_id].title

        return models.SearchResponse(
            success=True,
            query=f"Similar to: {doc_title}",
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time_ms, 2)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


# ============================================================================
# Document Management Endpoints
# ============================================================================

@app.get("/api/v1/documents", response_model=models.DocumentListResponse, tags=["Documents"])
def list_documents(
    tags: Optional[str] = None,
    authenticated: bool = Depends(verify_api_key)
):
    """
    List all documents in the knowledge base.

    Optional tag filter (comma-separated).
    """
    try:
        # Filter by tags if provided
        tag_list = [t.strip() for t in tags.split(',')] if tags else None

        docs = []
        for doc_id, doc_meta in kb.documents.items():
            # Filter by tags if specified
            if tag_list and not any(tag in doc_meta.tags for tag in tag_list):
                continue

            docs.append(models.DocumentMetadata(
                doc_id=doc_id,
                title=doc_meta.title,
                filename=doc_meta.filename,
                tags=doc_meta.tags,
                created_at=doc_meta.indexed_at,
                num_chunks=doc_meta.total_chunks,
                num_tables=0,  # Not stored in DocumentMeta
                num_code_blocks=0,  # Not stored in DocumentMeta
                source_url=doc_meta.source_url
            ))

        return models.DocumentListResponse(
            success=True,
            documents=docs,
            total=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/api/v1/documents/{doc_id}", tags=["Documents"])
def get_document(
    doc_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get a single document by ID.

    Returns full document metadata.
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        doc_meta = kb.documents[doc_id]

        return {
            "success": True,
            "document": models.DocumentMetadata(
                doc_id=doc_id,
                title=doc_meta.title,
                filename=doc_meta.filename,
                tags=doc_meta.tags,
                created_at=doc_meta.indexed_at,
                num_chunks=doc_meta.total_chunks,
                num_tables=0,  # Not stored in DocumentMeta
                num_code_blocks=0,  # Not stored in DocumentMeta
                source_url=doc_meta.source_url
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
def delete_document(
    doc_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Delete a document from the knowledge base.

    Removes document and all associated chunks, tables, code blocks.
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        doc_title = kb.documents[doc_id].title
        kb.remove_document(doc_id)

        return models.SuccessResponse(
            success=True,
            message=f"Document '{doc_title}' deleted successfully",
            data={"doc_id": doc_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# ============================================================================
# URL Scraping Endpoints
# ============================================================================

@app.post("/api/v1/scrape", response_model=models.ScrapeResponse, tags=["URL Scraping"])
def scrape_url(
    request: models.ScrapeRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Scrape a URL and add to knowledge base.

    Supports:
    - Single page or recursive scraping
    - Automatic frame detection
    - Domain filtering
    - Depth control
    """
    try:
        result = kb.scrape_url(
            url=request.url,
            title=request.title,
            tags=request.tags,
            follow_links=request.follow_links,
            same_domain_only=request.same_domain_only,
            max_pages=request.max_pages,
            depth=request.depth
        )

        if result['status'] != 'success':
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Scraping failed')
            )

        return models.ScrapeResponse(
            success=True,
            url=result['url'],
            files_scraped=result['files_scraped'],
            docs_added=result['docs_added'],
            docs_updated=result.get('docs_updated', 0),
            doc_ids=result.get('doc_ids', []),
            frames_detected=result.get('frames_detected')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


# ============================================================================
# File Upload Endpoint
# ============================================================================

@app.post("/api/v1/documents", tags=["Documents"])
def upload_document(
    file: UploadFile,
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload a document file to the knowledge base.

    Supported formats: PDF, TXT, MD, HTML, XLSX
    """
    try:
        # Parse tags
        tag_list = [t.strip() for t in tags.split(',')] if tags else None

        # Save uploaded file to data_dir/uploads, which is on the
        # KnowledgeBase path whitelist by default (see default_allowed_dirs
        # in server.py) so add_document's security check accepts it.
        uploads_dir = Path(kb.data_dir) / "uploads"
        uploads_dir.mkdir(exist_ok=True)

        # Generate unique filename
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        tmp_filename = f"{Path(file.filename).stem}_{unique_suffix}{Path(file.filename).suffix}"
        tmp_path = str(uploads_dir / tmp_filename)

        # Write uploaded file
        with open(tmp_path, 'wb') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)

        try:
            # Add document to knowledge base
            doc = kb.add_document(
                filepath=tmp_path,
                title=title,
                tags=tag_list
            )

            return {
                "success": True,
                "message": f"Document '{doc.title}' uploaded successfully",
                "data": {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "filename": doc.filename,
                    "num_chunks": doc.total_chunks
                }
            }
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.put("/api/v1/documents/{doc_id}", tags=["Documents"])
def update_document(
    doc_id: str,
    request: models.DocumentUpdateRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Update document metadata (title, tags).
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        doc_meta = kb.documents[doc_id]

        # Update title
        if request.title:
            doc_meta.title = request.title

        # Update tags
        if request.tags is not None:
            doc_meta.tags = request.tags
        elif request.add_tags:
            doc_meta.tags = list(set(doc_meta.tags + request.add_tags))
        elif request.remove_tags:
            doc_meta.tags = [t for t in doc_meta.tags if t not in request.remove_tags]

        # Save to database
        cursor = kb.db_conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET title = ?, tags = ?
            WHERE doc_id = ?
        """, (doc_meta.title, ','.join(doc_meta.tags), doc_id))
        kb.db_conn.commit()

        return models.SuccessResponse(
            success=True,
            message=f"Document '{doc_meta.title}' updated successfully",
            data={"doc_id": doc_id}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


# ============================================================================
# AI Endpoints
# ============================================================================

@app.post("/api/v1/documents/{doc_id}/summarize", tags=["AI Features"])
def summarize_document(
    doc_id: str,
    summary_type: str = "brief",
    authenticated: bool = Depends(verify_api_key)
):
    """
    Generate AI summary of a document.

    Summary types: brief, detailed, bullet
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        result = kb.summarize_document(
            doc_id=doc_id,
            summary_type=summary_type
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "summary_type": summary_type,
            "summary": result['summary'],
            "cached": result.get('cached', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/api/v1/documents/{doc_id}/entities/extract", tags=["AI Features"])
def extract_entities(
    doc_id: str,
    confidence_threshold: float = 0.7,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Extract named entities from a document using AI.

    Entity types: hardware, memory_address, instruction, person, company, product, concept
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        result = kb.extract_entities(
            doc_id=doc_id,
            confidence_threshold=confidence_threshold
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "entities_extracted": result['entities_count'],
            "entities": result['entities']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@app.get("/api/v1/documents/{doc_id}/entities", tags=["AI Features"])
def get_document_entities(
    doc_id: str,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get all entities extracted from a document.
    """
    if doc_id not in kb.documents:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    try:
        entities = kb.get_entities(doc_id=doc_id)

        return {
            "success": True,
            "doc_id": doc_id,
            "total_entities": len(entities),
            "entities": [
                {
                    "text": e['entity_text'],
                    "type": e['entity_type'],
                    "confidence": e['confidence'],
                    "occurrences": e.get('occurrences', 1)
                }
                for e in entities
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entities: {str(e)}")


# ============================================================================
# Export Endpoints
# ============================================================================

@app.get("/api/v1/export/entities", tags=["Export"])
def export_entities(
    format: str = "csv",
    min_confidence: float = 0.0,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Export all entities as CSV or JSON.

    Formats: csv, json
    """
    try:
        result = kb.export_entities(
            format=format,
            min_confidence=min_confidence
        )

        if format == "csv":
            return StreamingResponse(
                io.StringIO(result),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=entities.csv"}
            )
        else:  # json
            return StreamingResponse(
                io.StringIO(result),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=entities.json"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/v1/export/relationships", tags=["Export"])
def export_relationships(
    format: str = "csv",
    min_strength: float = 0.0,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Export all entity relationships as CSV or JSON.

    Formats: csv, json
    """
    try:
        result = kb.export_relationships(
            format=format,
            min_strength=min_strength
        )

        if format == "csv":
            return StreamingResponse(
                io.StringIO(result),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=relationships.csv"}
            )
        else:  # json
            return StreamingResponse(
                io.StringIO(result),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=relationships.json"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    import uvicorn

    # This API can add/remove/update documents and trigger scrape_url
    # (arbitrary-URL fetch from the host - SSRF risk) with zero
    # authentication whenever TDZ_API_KEYS is unset. Binding that to a
    # non-loopback address by default would expose full read/write KB
    # control to the whole LAN/internet. Refuse to start rather than fail
    # open; TDZ_REST_ALLOW_INSECURE=1 opts back in for a deliberately open
    # deployment (e.g. behind its own reverse-proxy auth).
    if REST_HOST not in ('127.0.0.1', 'localhost', '::1') and not API_KEYS and not ALLOW_INSECURE:
        print(
            f"Refusing to bind to {REST_HOST} with no API keys configured "
            "(TDZ_API_KEYS is unset). This would expose unauthenticated "
            "read/write access to the knowledge base. Either set "
            "TDZ_API_KEYS, or set TDZ_REST_ALLOW_INSECURE=1 to override.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Starting TDZ C64 Knowledge Base REST API v{__version__}")
    print(f"Data directory: {DATA_DIR}")
    print(f"API Keys configured: {len(API_KEYS) if API_KEYS else 'None (development mode)'}")
    print(f"CORS origins: {CORS_ORIGINS}")
    print("")
    print(f"Starting server on http://{REST_HOST}:{REST_PORT}")
    print(f"API documentation: http://{REST_HOST}:{REST_PORT}/api/docs")
    print("")

    uvicorn.run(app, host=REST_HOST, port=REST_PORT)
