#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - REST API Models

Pydantic models for request/response validation and OpenAPI documentation.
Uses Pydantic v2 syntax.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ========== Common Response Models ==========

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "Document not found",
                "details": None,
                "status_code": 404
            }
        }
    )


class SuccessResponse(BaseModel):
    """Generic success response wrapper."""
    success: bool = True
    data: Any = Field(..., description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {"message": "Operation successful"},
                "metadata": {"timestamp": 1234567890}
            }
        }
    )


# ========== Search Models ==========

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=100)
    tags: Optional[List[str]] = Field(None, description="Filter by tags")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "sprite graphics VIC-II",
                "max_results": 10,
                "tags": ["graphics", "hardware"]
            }
        }
    )


class SemanticSearchRequest(SearchRequest):
    """Semantic search request model."""
    top_k: int = Field(100, description="Number of candidates for semantic search", ge=1, le=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How do sprites work on the VIC-II?",
                "max_results": 5,
                "top_k": 50
            }
        }
    )


class HybridSearchRequest(SearchRequest):
    """Hybrid search request model."""
    semantic_weight: float = Field(0.5, description="Weight for semantic results (0.0-1.0)", ge=0.0, le=1.0)
    top_k: int = Field(100, description="Number of candidates for semantic search", ge=1, le=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "SID chip sound programming",
                "max_results": 10,
                "semantic_weight": 0.7,
                "top_k": 100
            }
        }
    )


class FacetedSearchRequest(SearchRequest):
    """Faceted search request model."""
    facet_filters: Dict[str, List[str]] = Field(
        ...,
        description="Facet filters (e.g., {'hardware': ['VIC-II', 'SID']})"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "programming",
                "max_results": 10,
                "facet_filters": {
                    "hardware": ["VIC-II", "SID"],
                    "concept": ["sprite", "sound"]
                }
            }
        }
    )


class SearchResult(BaseModel):
    """Individual search result."""
    doc_id: str = Field(..., description="Document ID")
    chunk_id: int = Field(..., description="Chunk ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Document filename")
    snippet: str = Field(..., description="Matching text snippet")
    score: float = Field(..., description="Relevance score")
    page: Optional[int] = Field(None, description="Page number (if available)")
    tags: List[str] = Field(default_factory=list, description="Document tags")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "89d0943d6009",
                "chunk_id": 42,
                "title": "Commodore 64 Programmer's Reference Guide",
                "filename": "c64-programmers-reference.pdf",
                "snippet": "The VIC-II chip controls sprite graphics...",
                "score": 0.92,
                "page": 127,
                "tags": ["reference", "graphics", "hardware"]
            }
        }
    )


class SearchResponse(BaseModel):
    """Search response model."""
    success: bool = True
    data: List[SearchResult] = Field(..., description="Search results")
    metadata: Dict[str, Any] = Field(
        ...,
        description="Search metadata (total results, query time, etc.)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [
                    {
                        "doc_id": "89d0943d6009",
                        "chunk_id": 42,
                        "title": "C64 Programmer's Reference",
                        "filename": "c64-ref.pdf",
                        "snippet": "VIC-II sprite graphics...",
                        "score": 0.92,
                        "page": 127,
                        "tags": ["reference"]
                    }
                ],
                "metadata": {
                    "total_results": 1,
                    "query_time_ms": 45.2,
                    "search_mode": "hybrid"
                }
            }
        }
    )


# ========== Document Models ==========

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    doc_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (pdf, txt, md, etc.)")
    total_chunks: int = Field(..., description="Number of chunks")
    total_pages: Optional[int] = Field(None, description="Number of pages (if applicable)")
    indexed_at: str = Field(..., description="Index timestamp (ISO format)")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    source_url: Optional[str] = Field(None, description="Source URL (if scraped)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "89d0943d6009",
                "title": "Commodore 64 Programmer's Reference Guide",
                "filename": "c64-programmers-reference.pdf",
                "file_type": "pdf",
                "total_chunks": 93,
                "total_pages": 504,
                "indexed_at": "2025-12-21T10:30:00Z",
                "tags": ["reference", "c64", "programming"],
                "source_url": None
            }
        }
    )


class DocumentResponse(BaseModel):
    """Document response model."""
    success: bool = True
    data: DocumentMetadata = Field(..., description="Document metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "doc_id": "89d0943d6009",
                    "title": "C64 Programmer's Reference",
                    "filename": "c64-ref.pdf",
                    "file_type": "pdf",
                    "total_chunks": 93,
                    "total_pages": 504,
                    "indexed_at": "2025-12-21T10:30:00Z",
                    "tags": ["reference"]
                }
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Document list response model."""
    success: bool = True
    data: List[DocumentMetadata] = Field(..., description="List of documents")
    metadata: Dict[str, Any] = Field(
        ...,
        description="List metadata (total count, filters applied, etc.)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [
                    {
                        "doc_id": "89d0943d6009",
                        "title": "C64 Programmer's Reference",
                        "filename": "c64-ref.pdf",
                        "file_type": "pdf",
                        "total_chunks": 93,
                        "total_pages": 504,
                        "indexed_at": "2025-12-21T10:30:00Z",
                        "tags": ["reference"]
                    }
                ],
                "metadata": {
                    "total_documents": 1,
                    "filters": {}
                }
            }
        }
    )


# ========== Placeholder Models for Future Sprints ==========

# Sprint 7: Document CRUD models
# Sprint 7: URL scraping models
# Sprint 7: AI feature models
# Sprint 8: Export models
# Sprint 8: Analytics models
