#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - REST API Server

FastAPI-based REST API server for the C64 Knowledge Base.
Provides HTTP endpoints for all knowledge base operations.

Environment Variables:
    TDZ_DATA_DIR: Data directory for knowledge base (default: ~/.tdz-c64-knowledge)
    TDZ_API_KEYS: Comma-separated list of API keys for authentication
    CORS_ORIGINS: Comma-separated list of allowed CORS origins
    ENABLE_RATE_LIMITING: Enable rate limiting (default: 1)
    API_HOST: Host to bind to (default: 0.0.0.0)
    API_PORT: Port to bind to (default: 8000)

Usage:
    uvicorn rest_server:app --reload --port 8000

    Or use the batch file:
    run_rest_api.bat
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Import version info
from version import __version__, __project_name__

# Import KnowledgeBase
from server import KnowledgeBase

# Import Pydantic models
from rest_models import (
    SearchRequest, SemanticSearchRequest, HybridSearchRequest, FacetedSearchRequest,
    SearchResult, SearchResponse,
    DocumentMetadata, DocumentResponse, DocumentListResponse, DocumentUpdateRequest,
    BulkDeleteRequest, BulkOperationResponse,
    ScrapeRequest, RescrapeRequest, ScrapeResponse, UpdateCheckResponse,
    SummarizeRequest, SummarizeResponse, EntityExtractionRequest, EntitySearchRequest,
    EntityResponse, RelationshipResponse,
    SearchAnalyticsResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global KnowledgeBase instance
kb: Optional[KnowledgeBase] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    global kb

    data_dir = os.environ.get('TDZ_DATA_DIR', str(Path.home() / '.tdz-c64-knowledge'))
    logger.info(f"Initializing KnowledgeBase at {data_dir}")

    try:
        kb = KnowledgeBase(data_dir)
        logger.info(f"KnowledgeBase initialized with {len(kb.documents)} documents")
    except Exception as e:
        logger.error(f"Failed to initialize KnowledgeBase: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down REST API server")


# Create FastAPI app
app = FastAPI(
    title=__project_name__,
    description="REST API for TDZ C64 Knowledge Base - Search and manage Commodore 64 documentation",
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ========== CORS Middleware ==========

# Get allowed origins from environment
cors_origins_env = os.environ.get('CORS_ORIGINS', '')
if cors_origins_env:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(',')]
else:
    # Default: allow localhost
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API Key Authentication ==========

# Get API keys from environment
api_keys_env = os.environ.get('TDZ_API_KEYS', '')
if api_keys_env:
    VALID_API_KEYS = set(key.strip() for key in api_keys_env.split(','))
else:
    # Default: allow unauthenticated access for development
    VALID_API_KEYS = set()
    logger.warning("No API keys configured - authentication disabled!")


async def verify_api_key(request: Request) -> bool:
    """
    Verify API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        True if authenticated or auth disabled

    Raises:
        HTTPException: If authentication fails
    """
    # Skip auth if no keys configured
    if not VALID_API_KEYS:
        return True

    # Get API key from header
    api_key = request.headers.get('X-API-Key')

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header."
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return True


# ========== Request Logging Middleware ==========

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (time.time() - start_time) * 1000

    # Log response
    logger.info(f"Response: {response.status_code} ({duration:.2f}ms)")

    return response


# ========== Error Handlers ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "status_code": 422
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Uncaught exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if os.environ.get('DEBUG') else None,
            "status_code": 500
        }
    )


# ========== Health & Info Endpoints ==========

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": __project_name__,
        "version": __version__,
        "description": "REST API for TDZ C64 Knowledge Base",
        "docs": "/api/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns service status and basic statistics.
    """
    if kb is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "KnowledgeBase not initialized"
            }
        )

    try:
        doc_count = len(kb.documents)

        return {
            "status": "healthy",
            "version": __version__,
            "documents": doc_count,
            "database": "connected",
            "semantic_search": kb.use_semantic,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/api/v1/stats", tags=["Statistics"], dependencies=[Depends(verify_api_key)])
async def get_stats():
    """
    Get knowledge base statistics.

    Requires authentication.

    Returns:
        - Total documents
        - Total chunks
        - Search capabilities
        - Feature flags
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Get document stats
        docs = kb.list_documents()
        total_chunks = sum(doc.total_chunks for doc in docs)
        total_pages = sum(doc.total_pages or 0 for doc in docs)

        # Get all tags
        all_tags = set()
        for doc in docs:
            all_tags.update(doc.tags)

        return {
            "success": True,
            "data": {
                "total_documents": len(docs),
                "total_chunks": total_chunks,
                "total_pages": total_pages,
                "total_tags": len(all_tags),
                "tags": sorted(all_tags),
                "fts5_search": kb._fts5_available(),
                "semantic_search": kb.use_semantic,
                "fuzzy_search": kb.use_fuzzy,
                "ocr": hasattr(kb, 'ocr_enabled') and kb.ocr_enabled or False
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ========== Helper Functions ==========

def format_search_results(results: list, query_time_ms: float, search_mode: str = "keyword") -> SearchResponse:
    """
    Format search results into SearchResponse model.

    Args:
        results: List of search results from KnowledgeBase
        query_time_ms: Query execution time in milliseconds
        search_mode: Search mode used (keyword, semantic, hybrid, faceted)

    Returns:
        SearchResponse with formatted results and metadata
    """
    search_results = []
    for result in results:
        search_results.append(SearchResult(
            doc_id=result.get('doc_id', ''),
            chunk_id=result.get('chunk_id', 0),
            title=result.get('title', ''),
            filename=result.get('filename', ''),
            snippet=result.get('snippet', ''),
            score=result.get('score', 0.0),
            page=result.get('page'),
            tags=result.get('tags', [])
        ))

    return SearchResponse(
        success=True,
        data={"results": search_results},
        metadata={
            "total_results": len(search_results),
            "query_time_ms": round(query_time_ms, 2),
            "search_mode": search_mode
        }
    )


# ========== Search Endpoints ==========

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"], dependencies=[Depends(verify_api_key)])
async def search(request: SearchRequest):
    """
    Basic keyword search using FTS5 or BM25.

    Searches through document chunks using full-text search.
    Returns ranked results based on relevance scores.

    **Example:**
    ```json
    {
        "query": "sprite graphics VIC-II",
        "max_results": 10,
        "tags": ["graphics", "hardware"]
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    # Validate query is not empty
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    try:
        start_time = time.time()

        results = kb.search(
            query=request.query,
            max_results=request.max_results,
            tags=request.tags
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(results, query_time_ms, "keyword")

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/api/v1/search/semantic", response_model=SearchResponse, tags=["Search"], dependencies=[Depends(verify_api_key)])
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search using embeddings and vector similarity.

    Searches based on meaning rather than exact keywords.
    Requires semantic search to be enabled.

    **Example:**
    ```json
    {
        "query": "How do sprites work on the VIC-II chip?",
        "max_results": 5,
        "top_k": 50
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if not kb.use_semantic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Semantic search not enabled. Set USE_SEMANTIC_SEARCH=1"
        )

    try:
        start_time = time.time()

        # Note: top_k parameter is not used by semantic_search method
        # It only accepts query, max_results, and tags
        results = kb.semantic_search(
            query=request.query,
            max_results=request.max_results,
            tags=request.tags
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(results, query_time_ms, "semantic")

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@app.post("/api/v1/search/hybrid", response_model=SearchResponse, tags=["Search"], dependencies=[Depends(verify_api_key)])
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining keyword and semantic search.

    Blends FTS5/BM25 results with semantic results using configurable weighting.
    Provides best of both worlds: exact matches + contextual understanding.

    **Example:**
    ```json
    {
        "query": "SID chip sound programming",
        "max_results": 10,
        "semantic_weight": 0.7,
        "top_k": 100
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if not kb.use_semantic:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1"
        )

    try:
        start_time = time.time()

        # Note: hybrid_search doesn't accept top_k parameter
        # It only accepts query, max_results, tags, semantic_weight
        results = kb.hybrid_search(
            query=request.query,
            max_results=request.max_results,
            tags=request.tags,
            semantic_weight=request.semantic_weight
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(results, query_time_ms, "hybrid")

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(e)}"
        )


@app.post("/api/v1/search/faceted", response_model=SearchResponse, tags=["Search"], dependencies=[Depends(verify_api_key)])
async def faceted_search(request: FacetedSearchRequest):
    """
    Faceted search with entity-based filtering.

    Searches with structured filters on extracted entities.
    Useful for narrowing results to specific hardware, concepts, etc.

    **Example:**
    ```json
    {
        "query": "programming",
        "max_results": 10,
        "facet_filters": {
            "hardware": ["VIC-II", "SID"],
            "concept": ["sprite", "sound"]
        }
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        start_time = time.time()

        results = kb.faceted_search(
            query=request.query,
            facet_filters=request.facet_filters,
            max_results=request.max_results,
            tags=request.tags
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(results, query_time_ms, "faceted")

    except Exception as e:
        logger.error(f"Faceted search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Faceted search failed: {str(e)}"
        )


@app.get("/api/v1/similar/{doc_id}", response_model=SearchResponse, tags=["Search"], dependencies=[Depends(verify_api_key)])
async def find_similar(
    doc_id: str,
    chunk_id: Optional[int] = None,
    max_results: int = 5,
    tags: Optional[str] = None
):
    """
    Find documents similar to a given document.

    Uses TF-IDF or semantic similarity to find related documents.
    Useful for "more like this" functionality.

    **Parameters:**
    - **doc_id**: Document ID to find similar documents for
    - **chunk_id**: Optional specific chunk ID (if None, uses all chunks)
    - **max_results**: Maximum number of results (default: 5)
    - **tags**: Optional comma-separated tags filter (e.g., "graphics,hardware")

    **Example:**
    ```
    GET /api/v1/similar/89d0943d6009?max_results=5&tags=graphics,hardware
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    # Parse tags from query string
    tags_list = None
    if tags:
        tags_list = [tag.strip() for tag in tags.split(',')]

    try:
        start_time = time.time()

        results = kb.find_similar_documents(
            doc_id=doc_id,
            chunk_id=chunk_id,
            max_results=max_results,
            tags=tags_list
        )

        query_time_ms = (time.time() - start_time) * 1000

        return format_search_results(results, query_time_ms, "similar")

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Similar documents error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar search failed: {str(e)}"
        )


# ========== Document CRUD Endpoints ==========

@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def list_documents(
    tags: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
):
    """
    List all documents in the knowledge base.

    **Parameters:**
    - **tags**: Optional comma-separated tags filter
    - **file_type**: Optional file type filter (pdf, txt, md, html, xlsx)
    - **limit**: Maximum number of documents to return
    - **offset**: Number of documents to skip (for pagination)

    **Example:**
    ```
    GET /api/v1/documents?tags=reference,c64&file_type=pdf&limit=20&offset=0
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Parse tags from query string
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',')]

        # Get all documents
        all_docs = kb.list_documents()

        # Apply filters
        filtered_docs = []
        for doc in all_docs:
            # Filter by tags
            if tags_list and not any(tag in doc.tags for tag in tags_list):
                continue
            # Filter by file type
            if file_type and doc.file_type != file_type:
                continue
            filtered_docs.append(doc)

        # Apply pagination
        total_count = len(filtered_docs)
        if offset:
            filtered_docs = filtered_docs[offset:]
        if limit:
            filtered_docs = filtered_docs[:limit]

        # Convert to response format
        doc_metadata_list = []
        for doc in filtered_docs:
            doc_metadata_list.append(DocumentMetadata(
                doc_id=doc.doc_id,
                title=doc.title,
                filename=doc.filename,
                file_type=doc.file_type,
                total_chunks=doc.total_chunks,
                total_pages=doc.total_pages,
                indexed_at=doc.indexed_at,
                tags=doc.tags,
                source_url=doc.source_url
            ))

        return DocumentListResponse(
            success=True,
            data={"documents": doc_metadata_list},
            metadata={
                "total_documents": total_count,
                "returned_count": len(doc_metadata_list),
                "offset": offset,
                "filters": {
                    "tags": tags_list,
                    "file_type": file_type
                }
            }
        )

    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/api/v1/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def get_document(doc_id: str):
    """
    Get metadata for a specific document.

    **Parameters:**
    - **doc_id**: Document ID

    **Example:**
    ```
    GET /api/v1/documents/89d0943d6009
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # kb.get_document() returns a dict, not DocumentMeta
        # Need to get metadata from kb.documents instead
        doc_meta = kb.documents.get(doc_id)
        if not doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document metadata not found: {doc_id}"
            )

        return DocumentResponse(
            success=True,
            data=DocumentMetadata(
                doc_id=doc_meta.doc_id,
                title=doc_meta.title,
                filename=doc_meta.filename,
                file_type=doc_meta.file_type,
                total_chunks=doc_meta.total_chunks,
                total_pages=doc_meta.total_pages,
                indexed_at=doc_meta.indexed_at,
                tags=doc_meta.tags,
                source_url=doc_meta.source_url
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@app.post("/api/v1/documents", response_model=DocumentResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Upload a new document to the knowledge base.

    **Parameters:**
    - **file**: File to upload (PDF, TXT, MD, HTML, XLSX)
    - **tags**: Optional comma-separated tags
    - **title**: Optional custom title (defaults to filename)

    **Example:**
    ```
    POST /api/v1/documents
    Content-Type: multipart/form-data

    file: <binary data>
    tags: "reference,c64"
    title: "C64 Programmer's Reference Guide"
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Save uploaded file to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Parse tags
        tags_list = []
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',')]

        # Add document to knowledge base
        doc = kb.add_document(
            filepath=tmp_path,
            tags=tags_list,
            title=title
        )

        # Clean up temp file
        import os
        os.unlink(tmp_path)

        return DocumentResponse(
            success=True,
            data=DocumentMetadata(
                doc_id=doc.doc_id,
                title=doc.title,
                filename=doc.filename,
                file_type=doc.file_type,
                total_chunks=doc.total_chunks,
                total_pages=doc.total_pages,
                indexed_at=doc.indexed_at,
                tags=doc.tags,
                source_url=doc.source_url
            )
        )

    except Exception as e:
        logger.error(f"Upload document error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@app.put("/api/v1/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def update_document(doc_id: str, request: DocumentUpdateRequest):
    """
    Update document metadata (title, tags).

    **Parameters:**
    - **doc_id**: Document ID
    - **request**: Update request with title and/or tag changes

    **Example:**
    ```json
    {
        "title": "Updated C64 Reference Guide",
        "add_tags": ["updated", "v2"],
        "remove_tags": ["draft"]
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Check if document exists
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # Get document metadata for tags
        doc_meta = kb.documents.get(doc_id)
        if not doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document metadata not found: {doc_id}"
            )

        # Update title if provided
        if request.title:
            kb.update_document_title(doc_id, request.title)

        # Handle tag updates
        if request.tags is not None:
            # Replace all tags
            kb.update_document_tags(doc_id, request.tags)
        else:
            # Add/remove specific tags
            current_tags = set(doc_meta.tags)
            if request.add_tags:
                current_tags.update(request.add_tags)
            if request.remove_tags:
                current_tags.difference_update(request.remove_tags)
            kb.update_document_tags(doc_id, list(current_tags))

        # Get updated document metadata
        updated_doc_meta = kb.documents.get(doc_id)
        if not updated_doc_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document metadata not found: {doc_id}"
            )

        return DocumentResponse(
            success=True,
            data=DocumentMetadata(
                doc_id=updated_doc_meta.doc_id,
                title=updated_doc_meta.title,
                filename=updated_doc_meta.filename,
                file_type=updated_doc_meta.file_type,
                total_chunks=updated_doc_meta.total_chunks,
                total_pages=updated_doc_meta.total_pages,
                indexed_at=updated_doc_meta.indexed_at,
                tags=updated_doc_meta.tags,
                source_url=updated_doc_meta.source_url
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update document error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@app.delete("/api/v1/documents/bulk", response_model=BulkOperationResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def bulk_delete(request: BulkDeleteRequest):
    """
    Delete multiple documents at once.

    **Parameters:**
    - **request**: Bulk delete request with doc_ids and confirmation

    **Example:**
    ```json
    {
        "doc_ids": ["89d0943d6009", "a1b2c3d4e5f6"],
        "confirm": true
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bulk delete requires confirmation (set confirm: true)"
        )

    successful = []
    failed = []
    errors = {}

    for doc_id in request.doc_ids:
        try:
            # Check if exists
            doc = kb.get_document(doc_id)
            if not doc:
                failed.append(doc_id)
                errors[doc_id] = "Document not found"
                continue

            # Delete
            kb.remove_document(doc_id)
            successful.append(doc_id)

        except Exception as e:
            failed.append(doc_id)
            errors[doc_id] = str(e)
            logger.error(f"Failed to delete {doc_id}: {e}")

    return BulkOperationResponse(
        success=len(failed) == 0,
        data={
            "successful": successful,
            "failed": failed,
            "errors": errors
        },
        metadata={
            "total_requested": len(request.doc_ids),
            "successful_count": len(successful),
            "failed_count": len(failed)
        }
    )


@app.delete("/api/v1/documents/{doc_id}", response_model=BulkOperationResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str):
    """
    Delete a document from the knowledge base.

    **Parameters:**
    - **doc_id**: Document ID to delete

    **Example:**
    ```
    DELETE /api/v1/documents/89d0943d6009
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Check if document exists
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # Delete document
        kb.remove_document(doc_id)

        return BulkOperationResponse(
            success=True,
            data={
                "successful": [doc_id],
                "failed": [],
                "errors": {}
            },
            metadata={
                "total_requested": 1,
                "successful_count": 1,
                "failed_count": 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.post("/api/v1/documents/bulk", response_model=BulkOperationResponse, tags=["Documents"], dependencies=[Depends(verify_api_key)])
async def bulk_upload(files: List[UploadFile] = File(...), tags: Optional[str] = None):
    """
    Upload multiple documents at once.

    **Parameters:**
    - **files**: List of files to upload
    - **tags**: Optional comma-separated tags to apply to all files

    **Example:**
    ```
    POST /api/v1/documents/bulk
    Content-Type: multipart/form-data

    files: [<file1>, <file2>, ...]
    tags: "reference,bulk-import"
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    successful = []
    failed = []
    errors = {}

    # Parse tags
    tags_list = []
    if tags:
        tags_list = [tag.strip() for tag in tags.split(',')]

    for file in files:
        try:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            # Add document
            doc = kb.add_document(
                filepath=tmp_path,
                tags=tags_list
            )

            successful.append(doc.doc_id)

            # Clean up
            import os
            os.unlink(tmp_path)

        except Exception as e:
            failed.append(file.filename)
            errors[file.filename] = str(e)
            logger.error(f"Failed to upload {file.filename}: {e}")

    return BulkOperationResponse(
        success=len(failed) == 0,
        data={
            "added": successful,
            "failed": failed,
            "errors": errors
        },
        metadata={
            "total_requested": len(files),
            "successful_count": len(successful),
            "failed_count": len(failed)
        }
    )

# ========== URL Scraping Endpoints ==========

@app.post("/api/v1/scrape", response_model=ScrapeResponse, tags=["URL Scraping"], dependencies=[Depends(verify_api_key)])
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a URL and add content to knowledge base.

    **Parameters:**
    - **request**: Scrape request with URL and options

    **Example:**
    ```json
    {
        "url": "https://www.c64-wiki.com/wiki/VIC",
        "max_depth": 2,
        "max_pages": 20,
        "tags": ["c64-wiki", "reference"],
        "allowed_domains": ["c64-wiki.com"],
        "max_workers": 5
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    # Validate URL format
    from urllib.parse import urlparse
    try:
        parsed = urlparse(request.url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL format. URL must include scheme (http/https) and domain."
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL format"
        )

    try:
        start_time = time.time()

        # Scrape URL
        result = kb.scrape_url(
            url=request.url,
            depth=request.max_depth,
            tags=request.tags or [],
            limit=request.allowed_domains[0] if request.allowed_domains else None,
            threads=request.max_workers
        )

        duration = time.time() - start_time

        return ScrapeResponse(
            success=True,
            data={
                "documents_added": result.get('documents_added', 0),
                "pages_scraped": result.get('pages_scraped', 0),
                "doc_ids": result.get('doc_ids', [])
            },
            metadata={
                "start_url": request.url,
                "duration_seconds": round(duration, 2),
                "errors": result.get('errors', [])
            }
        )

    except Exception as e:
        logger.error(f"Scrape URL error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scrape URL: {str(e)}"
        )


@app.post("/api/v1/scrape/rescrape/{doc_id}", response_model=ScrapeResponse, tags=["URL Scraping"], dependencies=[Depends(verify_api_key)])
async def rescrape_document(doc_id: str, request: RescrapeRequest):
    """
    Re-scrape a URL-sourced document to check for updates.

    **Parameters:**
    - **doc_id**: Document ID to re-scrape
    - **request**: Re-scrape options

    **Example:**
    ```json
    {
        "force": false
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        start_time = time.time()

        # Re-scrape document
        result = kb.rescrape_document(doc_id, force=request.force)

        duration = time.time() - start_time

        if result.get('status') == 'no_changes':
            return ScrapeResponse(
                success=True,
                data={
                    "documents_added": 0,
                    "pages_scraped": 1,
                    "doc_ids": [doc_id],
                    "status": "no_changes"
                },
                metadata={
                    "doc_id": doc_id,
                    "duration_seconds": round(duration, 2),
                    "message": "No changes detected"
                }
            )
        else:
            return ScrapeResponse(
                success=True,
                data={
                    "documents_added": 1,
                    "pages_scraped": 1,
                    "doc_ids": [doc_id],
                    "status": "updated"
                },
                metadata={
                    "doc_id": doc_id,
                    "duration_seconds": round(duration, 2),
                    "message": "Document updated"
                }
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Re-scrape error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to re-scrape document: {str(e)}"
        )


@app.post("/api/v1/scrape/check-updates", response_model=UpdateCheckResponse, tags=["URL Scraping"], dependencies=[Depends(verify_api_key)])
async def check_url_updates():
    """
    Check all URL-sourced documents for updates.

    **Example:**
    ```
    POST /api/v1/scrape/check-updates
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        result = kb.check_url_updates()

        return UpdateCheckResponse(
            success=True,
            data={
                "total_checked": result.get('total_checked', 0),
                "updates_available": result.get('updates_available', 0),
                "updated_docs": result.get('updated_docs', [])
            },
            metadata={
                "check_time": datetime.now().isoformat() + "Z"
            }
        )

    except Exception as e:
        logger.error(f"Check updates error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check for updates: {str(e)}"
        )


# ========== AI Feature Endpoints ==========

@app.post("/api/v1/summarize/{doc_id}", response_model=SummarizeResponse, tags=["AI Features"], dependencies=[Depends(verify_api_key)])
async def summarize_document(doc_id: str, request: SummarizeRequest):
    """
    Generate an AI summary of a document.

    **Parameters:**
    - **doc_id**: Document ID to summarize
    - **request**: Summarization options

    **Example:**
    ```json
    {
        "max_length": 300,
        "style": "technical"
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        start_time = time.time()

        # Get document
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # Generate summary
        summary = kb.summarize_document(
            doc_id=doc_id,
            max_length=request.max_length,
            style=request.style
        )

        generation_time_ms = (time.time() - start_time) * 1000
        word_count = len(summary.split())

        return SummarizeResponse(
            success=True,
            data={
                "summary": summary,
                "doc_id": doc_id,
                "title": doc.title
            },
            metadata={
                "word_count": word_count,
                "generation_time_ms": round(generation_time_ms, 2)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize document: {str(e)}"
        )


@app.post("/api/v1/entities/extract/{doc_id}", response_model=EntityResponse, tags=["AI Features"], dependencies=[Depends(verify_api_key)])
async def extract_entities(doc_id: str, request: EntityExtractionRequest):
    """
    Extract entities from a document using AI.

    **Parameters:**
    - **doc_id**: Document ID
    - **request**: Extraction options

    **Example:**
    ```json
    {
        "confidence_threshold": 0.8,
        "entity_types": ["hardware", "instruction"]
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Check if document exists
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # Extract entities
        kb.extract_entities(
            doc_id=doc_id,
            confidence_threshold=request.confidence_threshold,
            entity_types=request.entity_types
        )

        # Get extracted entities
        entities = kb.get_entities(
            doc_id=doc_id,
            min_confidence=request.confidence_threshold,
            entity_types=request.entity_types
        )

        return EntityResponse(
            success=True,
            data=entities,
            metadata={
                "total_entities": len(entities),
                "doc_id": doc_id
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extract entities error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract entities: {str(e)}"
        )


@app.post("/api/v1/entities/search", response_model=EntityResponse, tags=["AI Features"], dependencies=[Depends(verify_api_key)])
async def search_entities(request: EntitySearchRequest):
    """
    Search for entities across all documents.

    **Parameters:**
    - **request**: Entity search parameters

    **Example:**
    ```json
    {
        "entity_text": "VIC-II",
        "entity_type": "hardware",
        "min_confidence": 0.7
    }
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Search for entities (using search_entities method)
        result = kb.search_entities(
            query=request.entity_text,
            entity_types=[request.entity_type] if request.entity_type else None,
            min_confidence=request.min_confidence,
            max_results=50
        )

        # Flatten results from documents structure to simple list
        entities = []
        if 'documents' in result:
            for doc in result['documents']:
                for match in doc.get('matches', []):
                    entities.append(match)

        return EntityResponse(
            success=True,
            data=entities,
            metadata={
                "total_entities": len(entities),
                "search_term": request.entity_text
            }
        )

    except Exception as e:
        logger.error(f"Search entities error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search entities: {str(e)}"
        )


@app.get("/api/v1/entities/{doc_id}", response_model=EntityResponse, tags=["AI Features"], dependencies=[Depends(verify_api_key)])
async def get_document_entities(
    doc_id: str,
    entity_type: Optional[str] = None,
    min_confidence: float = 0.0
):
    """
    Get all entities for a specific document.

    **Parameters:**
    - **doc_id**: Document ID
    - **entity_type**: Optional entity type filter
    - **min_confidence**: Minimum confidence threshold

    **Example:**
    ```
    GET /api/v1/entities/89d0943d6009?entity_type=hardware&min_confidence=0.7
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Check if document exists
        doc = kb.get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {doc_id}"
            )

        # Get entities
        entity_types = [entity_type] if entity_type else None
        result = kb.get_entities(
            doc_id=doc_id,
            min_confidence=min_confidence,
            entity_types=entity_types
        )

        # kb.get_entities returns a dict with 'entities' list
        # Extract the list for the response
        return EntityResponse(
            success=True,
            data=result['entities'],  # Extract list from dict
            metadata={
                "total_entities": len(result['entities']),
                "doc_id": doc_id,
                "entity_types": result.get('types', {})
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get entities error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get entities: {str(e)}"
        )


@app.get("/api/v1/relationships/{entity_text}", response_model=RelationshipResponse, tags=["AI Features"], dependencies=[Depends(verify_api_key)])
async def get_entity_relationships(
    entity_text: str,
    min_strength: float = 0.0,
    max_results: int = 50
):
    """
    Get relationships for a specific entity.

    **Parameters:**
    - **entity_text**: Entity text to find relationships for
    - **min_strength**: Minimum relationship strength
    - **max_results**: Maximum number of results

    **Example:**
    ```
    GET /api/v1/relationships/VIC-II?min_strength=0.5&max_results=20
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Get relationships using get_entity_relationships method
        relationships = kb.get_entity_relationships(
            entity_text=entity_text,
            relationship_type=None,  # Get all relationship types
            min_strength=min_strength,
            max_results=max_results
        )

        return RelationshipResponse(
            success=True,
            data=relationships,
            metadata={
                "total_relationships": len(relationships),
                "entity": entity_text
            }
        )

    except Exception as e:
        logger.error(f"Get relationships error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relationships: {str(e)}"
        )


# ========== Analytics Endpoints ==========

@app.get("/api/v1/analytics/search", response_model=SearchAnalyticsResponse, tags=["Analytics"], dependencies=[Depends(verify_api_key)])
async def get_search_analytics(time_range_days: int = 30):
    """
    Get search analytics and statistics.

    **Parameters:**
    - **time_range_days**: Number of days to analyze (default: 30)

    **Example:**
    ```
    GET /api/v1/analytics/search?time_range_days=30
    ```

    Note: This is a placeholder implementation. In a production system,
    you would track search queries in a separate analytics table.
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    try:
        # Placeholder analytics data
        # In production, this would query an analytics/search_log table
        analytics_data = {
            "total_searches": 0,
            "unique_queries": 0,
            "avg_results_per_search": 0.0,
            "top_queries": [],
            "search_mode_distribution": {
                "keyword": 0,
                "semantic": 0,
                "hybrid": 0,
                "faceted": 0
            },
            "avg_query_time_ms": 0.0
        }

        return SearchAnalyticsResponse(
            success=True,
            data=analytics_data,
            metadata={
                "time_range_days": time_range_days,
                "generated_at": datetime.now().isoformat() + "Z",
                "note": "Placeholder implementation - analytics tracking not yet implemented"
            }
        )

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


# ========== Export Endpoints ==========

@app.get("/api/v1/export/search", tags=["Export"], dependencies=[Depends(verify_api_key)])
async def export_search_results(
    query: str,
    format: str = "csv",
    max_results: int = 100,
    tags: Optional[str] = None
):
    """
    Export search results to CSV or JSON.

    **Parameters:**
    - **query**: Search query
    - **format**: Export format (csv or json)
    - **max_results**: Maximum results to export
    - **tags**: Optional comma-separated tags filter

    **Example:**
    ```
    GET /api/v1/export/search?query=sprite&format=csv&max_results=50
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if format not in ['csv', 'json']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'csv' or 'json'"
        )

    try:
        # Parse tags
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',')]

        # Perform search
        results = kb.search(query=query, max_results=max_results, tags=tags_list)

        # Export to CSV
        if format == 'csv':
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['doc_id', 'chunk_id', 'title', 'filename', 'snippet', 'score', 'page', 'tags'])

            for result in results:
                writer.writerow([
                    result.get('doc_id', ''),
                    result.get('chunk_id', ''),
                    result.get('title', ''),
                    result.get('filename', ''),
                    result.get('snippet', '').replace('\n', ' '),
                    result.get('score', 0.0),
                    result.get('page', ''),
                    ','.join(result.get('tags', []))
                ])

            from fastapi.responses import Response
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="search_results_{query[:20]}.csv"'
                }
            )

        # Export to JSON
        else:
            import json
            from fastapi.responses import Response
            return Response(
                content=json.dumps(results, indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename="search_results_{query[:20]}.json"'
                }
            )

    except Exception as e:
        logger.error(f"Export search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export search results: {str(e)}"
        )


@app.get("/api/v1/export/documents", tags=["Export"], dependencies=[Depends(verify_api_key)])
async def export_documents(
    format: str = "csv",
    tags: Optional[str] = None,
    file_type: Optional[str] = None
):
    """
    Export document list to CSV or JSON.

    **Parameters:**
    - **format**: Export format (csv or json)
    - **tags**: Optional comma-separated tags filter
    - **file_type**: Optional file type filter

    **Example:**
    ```
    GET /api/v1/export/documents?format=csv&tags=reference
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if format not in ['csv', 'json']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'csv' or 'json'"
        )

    try:
        # Parse tags
        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',')]

        # Get all documents
        all_docs = kb.list_documents()

        # Apply filters
        filtered_docs = []
        for doc in all_docs:
            if tags_list and not any(tag in doc.tags for tag in tags_list):
                continue
            if file_type and doc.file_type != file_type:
                continue
            filtered_docs.append(doc)

        # Export to CSV
        if format == 'csv':
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['doc_id', 'title', 'filename', 'file_type', 'total_chunks',
                           'total_pages', 'indexed_at', 'tags', 'source_url'])

            for doc in filtered_docs:
                writer.writerow([
                    doc.doc_id,
                    doc.title,
                    doc.filename,
                    doc.file_type,
                    doc.total_chunks,
                    doc.total_pages or '',
                    doc.indexed_at,
                    ','.join(doc.tags),
                    doc.source_url or ''
                ])

            from fastapi.responses import Response
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": 'attachment; filename="documents.csv"'
                }
            )

        # Export to JSON
        else:
            import json
            docs_data = []
            for doc in filtered_docs:
                docs_data.append({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'total_chunks': doc.total_chunks,
                    'total_pages': doc.total_pages,
                    'indexed_at': doc.indexed_at,
                    'tags': doc.tags,
                    'source_url': doc.source_url
                })

            from fastapi.responses import Response
            return Response(
                content=json.dumps(docs_data, indent=2),
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="documents.json"'
                }
            )

    except Exception as e:
        logger.error(f"Export documents error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export documents: {str(e)}"
        )


@app.get("/api/v1/export/entities", tags=["Export"], dependencies=[Depends(verify_api_key)])
async def export_entities(
    format: str = "csv",
    entity_type: Optional[str] = None,
    min_confidence: float = 0.0
):
    """
    Export entities to CSV or JSON.

    **Parameters:**
    - **format**: Export format (csv or json)
    - **entity_type**: Optional entity type filter
    - **min_confidence**: Minimum confidence threshold

    **Example:**
    ```
    GET /api/v1/export/entities?format=csv&entity_type=hardware&min_confidence=0.7
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if format not in ['csv', 'json']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'csv' or 'json'"
        )

    try:
        # Use existing export method from KnowledgeBase
        entity_types = [entity_type] if entity_type else None
        export_data = kb.export_entities(
            format=format,
            entity_types=entity_types,
            min_confidence=min_confidence
        )

        from fastapi.responses import Response

        if format == 'csv':
            return Response(
                content=export_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": 'attachment; filename="entities.csv"'
                }
            )
        else:
            return Response(
                content=export_data,
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="entities.json"'
                }
            )

    except Exception as e:
        logger.error(f"Export entities error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export entities: {str(e)}"
        )


@app.get("/api/v1/export/relationships", tags=["Export"], dependencies=[Depends(verify_api_key)])
async def export_relationships(
    format: str = "csv",
    min_strength: float = 0.0,
    entity_type: Optional[str] = None
):
    """
    Export entity relationships to CSV or JSON.

    **Parameters:**
    - **format**: Export format (csv or json)
    - **min_strength**: Minimum relationship strength
    - **entity_type**: Optional entity type filter

    **Example:**
    ```
    GET /api/v1/export/relationships?format=csv&min_strength=0.5
    ```
    """
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KnowledgeBase not initialized"
        )

    if format not in ['csv', 'json']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'csv' or 'json'"
        )

    try:
        # Use existing export method from KnowledgeBase
        entity_types = [entity_type] if entity_type else None
        export_data = kb.export_relationships(
            format=format,
            min_strength=min_strength,
            entity_types=entity_types
        )

        from fastapi.responses import Response

        if format == 'csv':
            return Response(
                content=export_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": 'attachment; filename="relationships.csv"'
                }
            )
        else:
            return Response(
                content=export_data,
                media_type="application/json",
                headers={
                    "Content-Disposition": 'attachment; filename="relationships.json"'
                }
            )

    except Exception as e:
        logger.error(f"Export relationships error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export relationships: {str(e)}"
        )


# ========== End of Sprint 8 Endpoints ==========


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get('API_HOST', '0.0.0.0')
    port = int(os.environ.get('API_PORT', '8000'))

    logger.info(f"Starting REST API server on {host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/api/docs")

    uvicorn.run(
        "rest_server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
