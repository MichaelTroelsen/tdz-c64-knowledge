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


# ========== Document CRUD Models (Sprint 7) ==========

class DocumentUpdateRequest(BaseModel):
    """Document update request model."""
    title: Optional[str] = Field(None, description="New title")
    tags: Optional[List[str]] = Field(None, description="New tags (replaces existing)")
    add_tags: Optional[List[str]] = Field(None, description="Tags to add")
    remove_tags: Optional[List[str]] = Field(None, description="Tags to remove")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Updated C64 Reference Guide",
                "add_tags": ["updated", "v2"],
                "remove_tags": ["draft"]
            }
        }
    )


class BulkDeleteRequest(BaseModel):
    """Bulk delete request model."""
    doc_ids: List[str] = Field(..., description="List of document IDs to delete", min_length=1)
    confirm: bool = Field(False, description="Confirmation flag (must be true)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_ids": ["89d0943d6009", "a1b2c3d4e5f6"],
                "confirm": True
            }
        }
    )


class BulkOperationResponse(BaseModel):
    """Bulk operation response model."""
    success: bool = True
    data: Dict[str, Any] = Field(..., description="Operation results")
    metadata: Dict[str, Any] = Field(..., description="Operation metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "successful": ["89d0943d6009"],
                    "failed": [],
                    "errors": {}
                },
                "metadata": {
                    "total_requested": 1,
                    "successful_count": 1,
                    "failed_count": 0
                }
            }
        }
    )


# ========== URL Scraping Models (Sprint 7) ==========

class ScrapeRequest(BaseModel):
    """URL scraping request model."""
    url: str = Field(..., description="URL to scrape", min_length=1)
    max_depth: int = Field(1, description="Maximum crawl depth", ge=1, le=5)
    max_pages: int = Field(10, description="Maximum pages to scrape", ge=1, le=100)
    tags: Optional[List[str]] = Field(None, description="Tags to apply to scraped documents")
    allowed_domains: Optional[List[str]] = Field(None, description="Restrict scraping to these domains")
    max_workers: int = Field(3, description="Number of concurrent workers", ge=1, le=10)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://www.c64-wiki.com/wiki/VIC",
                "max_depth": 2,
                "max_pages": 20,
                "tags": ["c64-wiki", "reference"],
                "allowed_domains": ["c64-wiki.com"],
                "max_workers": 5
            }
        }
    )


class RescrapeRequest(BaseModel):
    """Re-scrape request model."""
    force: bool = Field(False, description="Force re-scrape even if no changes detected")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force": False
            }
        }
    )


class ScrapeResponse(BaseModel):
    """Scrape response model."""
    success: bool = True
    data: Dict[str, Any] = Field(..., description="Scrape results")
    metadata: Dict[str, Any] = Field(..., description="Scrape metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "documents_added": 5,
                    "pages_scraped": 5,
                    "doc_ids": ["abc123", "def456"]
                },
                "metadata": {
                    "start_url": "https://example.com",
                    "duration_seconds": 12.5,
                    "errors": []
                }
            }
        }
    )


class UpdateCheckResponse(BaseModel):
    """URL update check response model."""
    success: bool = True
    data: Dict[str, Any] = Field(..., description="Update check results")
    metadata: Dict[str, Any] = Field(..., description="Check metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "total_checked": 10,
                    "updates_available": 2,
                    "updated_docs": ["abc123", "def456"]
                },
                "metadata": {
                    "check_time": "2025-12-21T10:30:00Z"
                }
            }
        }
    )


# ========== AI Feature Models (Sprint 7) ==========

class SummarizeRequest(BaseModel):
    """Document summarization request model."""
    max_length: int = Field(500, description="Maximum summary length in words", ge=50, le=2000)
    style: str = Field("technical", description="Summary style", pattern="^(technical|simple|detailed)$")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_length": 300,
                "style": "technical"
            }
        }
    )


class SummarizeResponse(BaseModel):
    """Summarization response model."""
    success: bool = True
    data: Dict[str, Any] = Field(..., description="Summarization results")
    metadata: Dict[str, Any] = Field(..., description="Summarization metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "summary": "This document covers VIC-II sprite graphics...",
                    "doc_id": "89d0943d6009",
                    "title": "C64 Programmer's Reference"
                },
                "metadata": {
                    "word_count": 287,
                    "generation_time_ms": 1250.5
                }
            }
        }
    )


class EntityExtractionRequest(BaseModel):
    """Entity extraction request model."""
    confidence_threshold: float = Field(0.7, description="Minimum confidence for entities", ge=0.0, le=1.0)
    entity_types: Optional[List[str]] = Field(None, description="Limit to specific entity types")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confidence_threshold": 0.8,
                "entity_types": ["hardware", "instruction"]
            }
        }
    )


class EntitySearchRequest(BaseModel):
    """Entity search request model."""
    entity_text: str = Field(..., description="Entity text to search for", min_length=1)
    entity_type: Optional[str] = Field(None, description="Filter by entity type")
    min_confidence: float = Field(0.0, description="Minimum confidence", ge=0.0, le=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_text": "VIC-II",
                "entity_type": "hardware",
                "min_confidence": 0.7
            }
        }
    )


class EntityResponse(BaseModel):
    """Entity response model."""
    success: bool = True
    data: List[Dict[str, Any]] = Field(..., description="Entity data")
    metadata: Dict[str, Any] = Field(..., description="Entity metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [
                    {
                        "entity_text": "VIC-II",
                        "entity_type": "hardware",
                        "confidence": 0.95,
                        "occurrences": 42
                    }
                ],
                "metadata": {
                    "total_entities": 1,
                    "doc_id": "89d0943d6009"
                }
            }
        }
    )


class RelationshipResponse(BaseModel):
    """Relationship response model."""
    success: bool = True
    data: List[Dict[str, Any]] = Field(..., description="Relationship data")
    metadata: Dict[str, Any] = Field(..., description="Relationship metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": [
                    {
                        "entity1": "VIC-II",
                        "entity1_type": "hardware",
                        "entity2": "sprite",
                        "entity2_type": "concept",
                        "relationship_strength": 0.89,
                        "co_occurrences": 15
                    }
                ],
                "metadata": {
                    "total_relationships": 1,
                    "entity": "VIC-II"
                }
            }
        }
    )


# ========== Placeholder Models for Future Sprints ==========

# Sprint 8: Export models
# Sprint 8: Analytics models
