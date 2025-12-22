#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - REST API Models

Pydantic v2 models for request/response validation in the REST API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# ============================================================================
# Common Response Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_type: Optional[str] = Field(default=None, description="Error type/category")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": False,
            "error": "Document not found",
            "error_type": "not_found"
        }
    })


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = Field(default=True, description="Always true for success")
    message: str = Field(description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Optional response data")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "message": "Operation completed successfully",
            "data": {"doc_id": "abc123"}
        }
    })


# ============================================================================
# Search Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for basic search."""
    query: str = Field(description="Search query string", min_length=1)
    max_results: int = Field(default=10, description="Maximum results", ge=1, le=100)
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "VIC-II sprites",
            "max_results": 10,
            "tags": ["c64", "hardware"]
        }
    })


class SemanticSearchRequest(SearchRequest):
    """Request model for semantic search."""
    top_k: int = Field(default=5, description="Number of semantic matches", ge=1, le=50)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How do sprites work?",
            "max_results": 10,
            "top_k": 5
        }
    })


class HybridSearchRequest(SearchRequest):
    """Request model for hybrid search."""
    semantic_weight: float = Field(default=0.5, description="Weight for semantic search (0.0-1.0)", ge=0.0, le=1.0)
    top_k: int = Field(default=5, description="Number of semantic matches", ge=1, le=50)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "sprite collision detection",
            "max_results": 10,
            "semantic_weight": 0.5,
            "top_k": 5
        }
    })


class FacetedSearchRequest(SearchRequest):
    """Request model for faceted search."""
    facet_filters: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Facet filters (e.g., {'hardware': ['VIC-II', 'SID']})"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "sound programming",
            "max_results": 10,
            "facet_filters": {
                "hardware": ["SID"],
                "instructions": ["LDA", "STA"]
            }
        }
    })


class SearchResult(BaseModel):
    """Single search result."""
    doc_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    score: float = Field(description="Relevance score")
    snippet: str = Field(description="Text snippet/preview")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    filename: Optional[str] = Field(default=None, description="Source filename")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_id": "abc123",
            "title": "VIC-II Programming Guide",
            "score": 0.85,
            "snippet": "The VIC-II chip controls sprites...",
            "tags": ["c64", "hardware", "vic-ii"],
            "filename": "vic-ii-guide.pdf"
        }
    })


class SearchResponse(BaseModel):
    """Response model for search operations."""
    success: bool = Field(default=True, description="Operation success")
    query: str = Field(description="Original search query")
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    search_time_ms: float = Field(description="Search time in milliseconds")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "query": "VIC-II sprites",
            "results": [
                {
                    "doc_id": "abc123",
                    "title": "VIC-II Programming",
                    "score": 0.92,
                    "snippet": "Sprite programming...",
                    "tags": ["c64", "vic-ii"],
                    "filename": "vic-guide.pdf"
                }
            ],
            "total_results": 1,
            "search_time_ms": 15.3
        }
    })


# ============================================================================
# Document Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata."""
    doc_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    filename: Optional[str] = Field(default=None, description="Source filename")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    created_at: str = Field(description="Creation timestamp")
    num_chunks: int = Field(description="Number of chunks")
    num_tables: int = Field(description="Number of tables extracted")
    num_code_blocks: int = Field(description="Number of code blocks")
    source_url: Optional[str] = Field(default=None, description="Source URL if scraped")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_id": "abc123",
            "title": "C64 Programmer's Reference",
            "filename": "c64-ref.pdf",
            "tags": ["c64", "reference"],
            "created_at": "2025-12-22T08:00:00",
            "num_chunks": 150,
            "num_tables": 5,
            "num_code_blocks": 23,
            "source_url": None
        }
    })


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    success: bool = Field(default=True)
    documents: List[DocumentMetadata] = Field(description="List of documents")
    total: int = Field(description="Total number of documents")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "documents": [
                {
                    "doc_id": "abc123",
                    "title": "C64 Reference",
                    "filename": "c64-ref.pdf",
                    "tags": ["c64"],
                    "created_at": "2025-12-22T08:00:00",
                    "num_chunks": 100,
                    "num_tables": 3,
                    "num_code_blocks": 10,
                    "source_url": None
                }
            ],
            "total": 1
        }
    })


class DocumentCreateRequest(BaseModel):
    """Request model for creating a document."""
    title: Optional[str] = Field(default=None, description="Document title (auto-generated if not provided)")
    tags: Optional[List[str]] = Field(default=None, description="Document tags")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "My C64 Document",
            "tags": ["c64", "custom"]
        }
    })


class DocumentUpdateRequest(BaseModel):
    """Request model for updating document metadata."""
    title: Optional[str] = Field(default=None, description="New title")
    tags: Optional[List[str]] = Field(default=None, description="New tags (replaces existing)")
    add_tags: Optional[List[str]] = Field(default=None, description="Tags to add")
    remove_tags: Optional[List[str]] = Field(default=None, description="Tags to remove")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Updated Title",
            "add_tags": ["important"],
            "remove_tags": ["draft"]
        }
    })


# ============================================================================
# URL Scraping Models
# ============================================================================

class ScrapeRequest(BaseModel):
    """Request model for URL scraping."""
    url: str = Field(description="URL to scrape", pattern=r'^https?://')
    title: Optional[str] = Field(default=None, description="Document title")
    tags: Optional[List[str]] = Field(default=None, description="Document tags")
    follow_links: bool = Field(default=True, description="Follow links to sub-pages")
    same_domain_only: bool = Field(default=True, description="Only follow links on same domain")
    max_pages: int = Field(default=50, description="Maximum pages to scrape", ge=1, le=500)
    depth: int = Field(default=3, description="Maximum link depth", ge=1, le=10)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "url": "http://example.com/c64-docs",
            "title": "C64 Documentation",
            "tags": ["c64", "web"],
            "follow_links": True,
            "same_domain_only": True,
            "max_pages": 50,
            "depth": 3
        }
    })


class ScrapeResponse(BaseModel):
    """Response model for scraping operations."""
    success: bool = Field(default=True)
    url: str = Field(description="Scraped URL")
    files_scraped: int = Field(description="Number of files scraped")
    docs_added: int = Field(description="Documents added")
    docs_updated: int = Field(description="Documents updated")
    doc_ids: List[str] = Field(description="Document IDs")
    frames_detected: Optional[int] = Field(default=None, description="Number of frames detected (if frameset)")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "url": "http://example.com/docs",
            "files_scraped": 18,
            "docs_added": 18,
            "docs_updated": 0,
            "doc_ids": ["abc123", "def456"],
            "frames_detected": 2
        }
    })


# ============================================================================
# Analytics Models
# ============================================================================

class StatsResponse(BaseModel):
    """Response model for knowledge base statistics."""
    success: bool = Field(default=True)
    total_documents: int = Field(description="Total number of documents")
    total_chunks: int = Field(description="Total number of chunks")
    total_entities: int = Field(description="Total unique entities")
    total_relationships: int = Field(description="Total entity relationships")
    database_size_mb: float = Field(description="Database size in MB")
    semantic_search_enabled: bool = Field(description="Semantic search available")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "total_documents": 159,
            "total_chunks": 2612,
            "total_entities": 989,
            "total_relationships": 128,
            "database_size_mb": 45.2,
            "semantic_search_enabled": True
        }
    })


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status", pattern="^(healthy|degraded|unhealthy)$")
    version: str = Field(description="API version")
    database_ok: bool = Field(description="Database accessible")
    semantic_search_ok: bool = Field(description="Semantic search available")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "version": "2.17.1",
            "database_ok": True,
            "semantic_search_ok": True
        }
    })
