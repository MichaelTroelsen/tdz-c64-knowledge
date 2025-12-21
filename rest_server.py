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
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Import version info
from version import __version__, __project_name__

# Import KnowledgeBase
from server import KnowledgeBase

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
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
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
                "documents": {
                    "total": len(docs),
                    "total_chunks": total_chunks,
                    "total_pages": total_pages
                },
                "tags": {
                    "total": len(all_tags),
                    "tags": sorted(all_tags)
                },
                "capabilities": {
                    "fts5_search": kb.use_fts5,
                    "semantic_search": kb.use_semantic,
                    "fuzzy_search": kb.use_fuzzy,
                    "ocr": kb.ocr_enabled
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ========== Placeholder for future endpoints ==========
# Search endpoints will be added in Sprint 6
# Document CRUD endpoints will be added in Sprint 7
# AI feature endpoints will be added in Sprint 7
# Export endpoints will be added in Sprint 8


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
