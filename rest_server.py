#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - REST API Server

FastAPI-based REST API for the C64 Knowledge Base.
Provides HTTP endpoints for all knowledge base operations.

Run with: uvicorn rest_server:app --reload --port 8000
"""

import os
import time
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


# ============================================================================
# Global Knowledge Base Instance
# ============================================================================

kb = None  # Will be initialized in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - initialize KB on startup, cleanup on shutdown."""
    global kb
    print(f"Initializing Knowledge Base at {DATA_DIR}")
    kb = KnowledgeBase(DATA_DIR)
    print(f"Knowledge Base initialized with {len(kb.documents)} documents")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ['*'] else ['*'],
    allow_credentials=True,
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


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": "internal_error"
        }
    )


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=models.HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns service health status and component availability.
    """
    try:
        database_ok = kb is not None and kb.db_path.exists()
        semantic_ok = kb is not None and hasattr(kb, 'use_semantic_search') and kb.use_semantic_search

        status_str = "healthy" if database_ok else "unhealthy"

        return models.HealthResponse(
            status=status_str,
            version=__version__,
            database_ok=database_ok,
            semantic_search_ok=semantic_ok
        )
    except Exception as e:
        return models.HealthResponse(
            status="unhealthy",
            version=__version__,
            database_ok=False,
            semantic_search_ok=False
        )


@app.get("/api/v1/stats", response_model=models.StatsResponse, tags=["Analytics"])
async def get_stats(authenticated: bool = Depends(verify_api_key)):
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
        cursor = kb.conn.cursor()
        total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Get entity count
        try:
            total_entities = cursor.execute(
                "SELECT COUNT(DISTINCT entity_text) FROM document_entities"
            ).fetchone()[0]
        except:
            total_entities = 0

        # Get relationship count
        try:
            total_relationships = cursor.execute(
                "SELECT COUNT(*) FROM entity_relationships"
            ).fetchone()[0]
        except:
            total_relationships = 0

        # Get database size
        db_size_bytes = kb.db_path.stat().st_size
        db_size_mb = db_size_bytes / (1024 * 1024)

        return models.StatsResponse(
            success=True,
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_entities=total_entities,
            total_relationships=total_relationships,
            database_size_mb=round(db_size_mb, 2),
            semantic_search_enabled=kb.use_semantic_search
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/api/v1/search", response_model=models.SearchResponse, tags=["Search"])
async def search(
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
async def semantic_search(
    request: models.SemanticSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Semantic search using embeddings.

    Finds documents by meaning/context rather than exact keywords.
    Requires semantic search to be enabled.
    """
    if not kb.use_semantic_search:
        raise HTTPException(
            status_code=503,
            detail="Semantic search not available. Install sentence-transformers."
        )

    try:
        start_time = time.time()

        results = kb.semantic_search(
            query=request.query,
            top_k=request.top_k,
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
async def hybrid_search(
    request: models.HybridSearchRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Hybrid search combining FTS5 and semantic search.

    Blends keyword and meaning-based search for best results.
    """
    if not kb.use_semantic_search:
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
            top_k=request.top_k,
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
async def faceted_search(
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
async def find_similar_documents(
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
async def list_documents(
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
                created_at=doc_meta.created_at,
                num_chunks=doc_meta.num_chunks,
                num_tables=doc_meta.num_tables,
                num_code_blocks=doc_meta.num_code_blocks,
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
async def get_document(
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
                created_at=doc_meta.created_at,
                num_chunks=doc_meta.num_chunks,
                num_tables=doc_meta.num_tables,
                num_code_blocks=doc_meta.num_code_blocks,
                source_url=doc_meta.source_url
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
async def delete_document(
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
async def scrape_url(
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
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"Starting TDZ C64 Knowledge Base REST API v{__version__}")
    print(f"Data directory: {DATA_DIR}")
    print(f"API Keys configured: {len(API_KEYS) if API_KEYS else 'None (development mode)'}")
    print(f"CORS origins: {CORS_ORIGINS}")
    print("")
    print("Starting server on http://localhost:8000")
    print("API documentation: http://localhost:8000/api/docs")
    print("")

    uvicorn.run(app, host="0.0.0.0", port=8000)
