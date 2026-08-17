#!/usr/bin/env python3
"""
TDZ C64 Knowledge - shared data types

Dataclasses, the progress-callback type alias, and the exception hierarchy
that KnowledgeBase's methods build and raise. Split out of server.py in the
step after util.py: KnowledgeBase reaches for these from dozens of methods,
so as long as they lived inside server.py itself, splitting that 253-method
class into mixin modules would be circular by construction - the mixins
would need to import server.py for these types, and server.py defines
KnowledgeBase.

Pure data only - no behaviour beyond ProgressUpdate's own percentage
calculation - so this module has no import-cost implications for MCP startup
the way util.py/features.py do.
"""

from dataclasses import dataclass
from typing import Callable, Optional


# Custom Exceptions
class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass


class DocumentNotFoundError(KnowledgeBaseError):
    """Raised when a document is not found."""
    pass


class UnsupportedFileTypeError(KnowledgeBaseError):
    """Raised when an unsupported file type is provided."""
    pass


class SecurityError(KnowledgeBaseError):
    """Raised when a security violation is detected."""
    pass


@dataclass
class DocumentChunk:
    """A searchable chunk of a document."""
    doc_id: str
    filename: str
    title: str
    chunk_id: int
    page: Optional[int]
    content: str
    word_count: int


@dataclass
class DocumentMeta:
    """Metadata about an indexed document."""
    doc_id: str
    filename: str
    title: str
    filepath: str
    file_type: str
    total_pages: Optional[int]
    total_chunks: int
    indexed_at: str
    tags: list[str]
    # PDF metadata (optional)
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    creation_date: Optional[str] = None
    # Update detection fields
    file_mtime: Optional[float] = None  # File modification time
    file_hash: Optional[str] = None  # MD5 hash of file content
    # URL scraping fields (optional)
    source_url: Optional[str] = None  # Original URL if document was scraped
    scrape_date: Optional[str] = None  # ISO timestamp of last scrape
    scrape_config: Optional[str] = None  # JSON string with scraping config
    scrape_status: Optional[str] = None  # 'success', 'partial', 'failed'
    scrape_error: Optional[str] = None  # Error message if scrape failed
    url_last_checked: Optional[str] = None  # ISO timestamp of last update check
    url_content_hash: Optional[str] = None  # Hash of scraped content for change detection
    # Knowledge-card identity (optional)
    card_id: Optional[str] = None  # Logical card id parsed from a ```json id``` block; None for non-card docs
    superseded_by: Optional[str] = None  # doc_id of the card that replaced this one; None if this is the live version


@dataclass
class ProgressUpdate:
    """Progress update for long-running operations."""
    operation: str  # Operation name (e.g., "add_document", "add_documents_bulk")
    current: int  # Current progress (items processed)
    total: int  # Total items to process
    message: str  # Status message
    item: Optional[str] = None  # Current item being processed (e.g., filename)
    percentage: float = 0.0  # Percentage complete (0-100)

    def __post_init__(self):
        """Calculate percentage after initialization."""
        if self.total > 0:
            self.percentage = (self.current / self.total) * 100.0


# Type alias for progress callback function
ProgressCallback = Optional[Callable[[ProgressUpdate], None]]
