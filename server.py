#!/usr/bin/env python3
"""
TDZ C64 Knowledge - MCP Server
A Model Context Protocol server for searching C64 documentation.
"""

import os
import sys
import json
import re
import hashlib
import logging
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env file from current directory

# Import version information
from version import __version__, __project_name__, __build_date__, get_full_version_string

# Caching support
try:
    from cachetools import TTLCache
    CACHE_SUPPORT = True
except ImportError:
    CACHE_SUPPORT = False
    print("Warning: cachetools not installed. Search caching disabled.", file=sys.stderr)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

# Optional imports for PDF support
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed. PDF support disabled.", file=sys.stderr)

# PDF table extraction support
try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("Warning: pdfplumber not installed. Table extraction disabled.", file=sys.stderr)

# BM25 search support
try:
    from rank_bm25 import BM25Okapi
    BM25_SUPPORT = True
except ImportError:
    BM25_SUPPORT = False
    print("Warning: rank-bm25 not installed. Using simple search.", file=sys.stderr)

# NLTK for query preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    NLTK_SUPPORT = True

    # Ensure NLTK data is available
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    NLTK_SUPPORT = False
    print("Warning: nltk not installed. Query preprocessing disabled.", file=sys.stderr)

# Semantic search support
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    SEMANTIC_SUPPORT = True
except ImportError:
    SEMANTIC_SUPPORT = False
    print("Warning: sentence-transformers or faiss-cpu not installed. Semantic search disabled.", file=sys.stderr)

# Fuzzy search support
try:
    from rapidfuzz import fuzz
    FUZZY_SUPPORT = True
except ImportError:
    FUZZY_SUPPORT = False
    print("Warning: rapidfuzz not installed. Fuzzy search disabled.", file=sys.stderr)

# OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("Warning: pytesseract/pdf2image/Pillow not installed. OCR disabled.", file=sys.stderr)


# Custom Exceptions
class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass


class DocumentNotFoundError(KnowledgeBaseError):
    """Raised when a document is not found."""
    pass


class ChunkNotFoundError(KnowledgeBaseError):
    """Raised when a chunk is not found."""
    pass


class UnsupportedFileTypeError(KnowledgeBaseError):
    """Raised when an unsupported file type is provided."""
    pass


class IndexCorruptedError(KnowledgeBaseError):
    """Raised when the index is corrupted."""
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


class KnowledgeBase:
    """Manages the document index and search."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.data_dir / "server.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info(f"{get_full_version_string()}")
        self.logger.info(f"Build Date: {__build_date__}")
        self.logger.info("=" * 60)
        self.logger.info(f"Initializing KnowledgeBase at {data_dir}")

        # Thread safety for parallel document processing
        self._lock = threading.Lock()

        # Database setup
        self.db_file = self.data_dir / "knowledge_base.db"
        self.db_conn = None
        self._init_database()

        # Legacy file paths (for migration)
        self.index_file = self.data_dir / "index.json"
        self.chunks_dir = self.data_dir / "chunks"

        self.documents: dict[str, DocumentMeta] = {}
        self.chunks: list[DocumentChunk] = []  # Only loaded on demand for BM25
        self.bm25 = None  # BM25 index, built on demand

        # Initialize caching layer
        if CACHE_SUPPORT:
            cache_size = int(os.getenv('SEARCH_CACHE_SIZE', '100'))
            cache_ttl = int(os.getenv('SEARCH_CACHE_TTL', '300'))  # 5 minutes
            self._search_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self._similar_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self.logger.info(f"Search caching enabled (size={cache_size}, ttl={cache_ttl}s)")
        else:
            self._search_cache = None
            self._similar_cache = None

        # Initialize query preprocessing
        self.use_preprocessing = NLTK_SUPPORT and os.getenv('USE_QUERY_PREPROCESSING', '1') == '1'
        if self.use_preprocessing:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self.logger.info("Query preprocessing enabled (stemming + stopwords)")
        else:
            self.stemmer = None
            self.stop_words = set()
            if NLTK_SUPPORT:
                self.logger.info("Query preprocessing disabled via USE_QUERY_PREPROCESSING=0")

        # Security: Allowed directories for document ingestion (optional)
        # Set via ALLOWED_DOCS_DIRS environment variable (comma-separated paths)
        allowed_dirs_env = os.getenv('ALLOWED_DOCS_DIRS', '')
        if allowed_dirs_env:
            self.allowed_dirs = [Path(d.strip()).resolve() for d in allowed_dirs_env.split(',') if d.strip()]
            self.logger.info(f"Path traversal protection enabled for: {self.allowed_dirs}")
        else:
            self.allowed_dirs = None  # No restrictions

        # Semantic search initialization
        self.use_semantic = SEMANTIC_SUPPORT and os.getenv('USE_SEMANTIC_SEARCH', '0') == '1'
        self.embeddings_model = None
        self.embeddings_index = None
        self.embeddings_doc_map = []  # Maps FAISS index positions to (doc_id, chunk_id)

        if self.use_semantic:
            try:
                model_name = os.getenv('SEMANTIC_MODEL', 'all-MiniLM-L6-v2')
                self.logger.info(f"Loading embeddings model: {model_name}")
                self.embeddings_model = SentenceTransformer(model_name)
                self.embeddings_file = self.data_dir / "embeddings.faiss"
                self.embeddings_map_file = self.data_dir / "embeddings_map.json"
                self._load_embeddings()
                self.logger.info("Semantic search enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize semantic search: {e}")
                self.use_semantic = False
        else:
            if SEMANTIC_SUPPORT:
                self.logger.info("Semantic search disabled via USE_SEMANTIC_SEARCH=0")

        # Fuzzy search initialization
        self.use_fuzzy = FUZZY_SUPPORT and os.getenv('USE_FUZZY_SEARCH', '1') == '1'
        self.fuzzy_threshold = int(os.getenv('FUZZY_THRESHOLD', '80'))  # 0-100, default 80%

        if self.use_fuzzy:
            self.logger.info(f"Fuzzy search enabled (threshold={self.fuzzy_threshold}%)")
        else:
            if FUZZY_SUPPORT:
                self.logger.info("Fuzzy search disabled via USE_FUZZY_SEARCH=0")

        # OCR initialization
        self.use_ocr = OCR_SUPPORT and os.getenv('USE_OCR', '1') == '1'
        self.poppler_path = os.getenv('POPPLER_PATH', None)  # Optional Poppler path for pdf2image
        self.poppler_available = False  # Track if poppler is actually available

        if self.use_ocr:
            # Check if Tesseract is installed
            try:
                pytesseract.get_tesseract_version()
                self.logger.info("OCR enabled (Tesseract found)")

                # Check if poppler is available
                self.poppler_available = self._check_poppler_available()

                if self.poppler_available:
                    if self.poppler_path:
                        self.logger.info(f"Poppler found at: {self.poppler_path}")
                    else:
                        self.logger.info("Poppler found in system PATH")
                else:
                    self.logger.warning("[WARNING] Poppler not found! OCR will not work for scanned PDFs.")
                    self.logger.warning("Install poppler-utils:")
                    self.logger.warning("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
                    self.logger.warning("  Set POPPLER_PATH environment variable to the bin directory")
                    self.logger.warning("  Example: POPPLER_PATH=C:\\path\\to\\poppler-24.08.0\\Library\\bin")

            except Exception as e:
                self.logger.warning(f"OCR libraries installed but Tesseract not found: {e}")
                self.logger.warning("Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                self.use_ocr = False
        else:
            if OCR_SUPPORT:
                self.logger.info("OCR disabled via USE_OCR=0")

        # Load documents (with automatic migration if needed)
        self._load_documents()
        self.logger.info(f"Loaded {len(self.documents)} documents")

    def _init_database(self):
        """Initialize SQLite database and create schema if needed."""
        db_exists = self.db_file.exists()

        # Connect to database with enable foreign keys
        self.db_conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
        self.db_conn.execute("PRAGMA foreign_keys = ON")

        if not db_exists:
            self.logger.info("Creating new database schema")
            cursor = self.db_conn.cursor()

            # Create documents table
            cursor.execute("""
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filepath TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    total_pages INTEGER,
                    total_chunks INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    author TEXT,
                    subject TEXT,
                    creator TEXT,
                    creation_date TEXT,
                    file_mtime REAL,
                    file_hash TEXT,
                    source_url TEXT,
                    scrape_date TEXT,
                    scrape_config TEXT,
                    scrape_status TEXT,
                    scrape_error TEXT,
                    url_last_checked TEXT,
                    url_content_hash TEXT
                )
            """)

            # Create indexes on documents table
            cursor.execute("CREATE INDEX idx_documents_filepath ON documents(filepath)")
            cursor.execute("CREATE INDEX idx_documents_file_type ON documents(file_type)")
            cursor.execute("CREATE INDEX idx_documents_source_url ON documents(source_url)")
            cursor.execute("CREATE INDEX idx_documents_scrape_status ON documents(scrape_status)")

            # Create chunks table
            cursor.execute("""
                CREATE TABLE chunks (
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    page INTEGER,
                    content TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    PRIMARY KEY (doc_id, chunk_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create index on chunks table
            cursor.execute("CREATE INDEX idx_chunks_doc_id ON chunks(doc_id)")

            # Create FTS5 virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts5 USING fts5(
                    doc_id UNINDEXED,
                    chunk_id UNINDEXED,
                    content,
                    tokenize='porter unicode61'
                )
            """)

            # Trigger: Keep FTS5 in sync on INSERT
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts5_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                    VALUES (new.rowid, new.doc_id, new.chunk_id, new.content);
                END
            """)

            # Trigger: Keep FTS5 in sync on DELETE
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts5_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts5 WHERE rowid = old.rowid;
                END
            """)

            # Trigger: Keep FTS5 in sync on UPDATE
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts5_update AFTER UPDATE ON chunks BEGIN
                    DELETE FROM chunks_fts5 WHERE rowid = old.rowid;
                    INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                    VALUES (new.rowid, new.doc_id, new.chunk_id, new.content);
                END
            """)

            # Create document_tables table
            cursor.execute("""
                CREATE TABLE document_tables (
                    doc_id TEXT NOT NULL,
                    table_id INTEGER NOT NULL,
                    page INTEGER,
                    markdown TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    row_count INTEGER,
                    col_count INTEGER,
                    PRIMARY KEY (doc_id, table_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create FTS5 index for table search
            cursor.execute("""
                CREATE VIRTUAL TABLE tables_fts USING fts5(
                    doc_id UNINDEXED,
                    table_id UNINDEXED,
                    searchable_text,
                    tokenize='porter unicode61'
                )
            """)

            # Triggers for tables_fts
            cursor.execute("""
                CREATE TRIGGER tables_fts_insert AFTER INSERT ON document_tables BEGIN
                    INSERT INTO tables_fts(rowid, doc_id, table_id, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM tables_fts),
                            new.doc_id, new.table_id, new.searchable_text);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER tables_fts_delete AFTER DELETE ON document_tables BEGIN
                    DELETE FROM tables_fts WHERE doc_id = old.doc_id AND table_id = old.table_id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER tables_fts_update AFTER UPDATE ON document_tables BEGIN
                    DELETE FROM tables_fts WHERE doc_id = old.doc_id AND table_id = old.table_id;
                    INSERT INTO tables_fts(rowid, doc_id, table_id, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM tables_fts),
                            new.doc_id, new.table_id, new.searchable_text);
                END
            """)

            # Create document_code_blocks table
            cursor.execute("""
                CREATE TABLE document_code_blocks (
                    doc_id TEXT NOT NULL,
                    block_id INTEGER NOT NULL,
                    page INTEGER,
                    block_type TEXT NOT NULL,
                    code TEXT NOT NULL,
                    searchable_text TEXT NOT NULL,
                    line_count INTEGER,
                    PRIMARY KEY (doc_id, block_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create FTS5 index for code search
            cursor.execute("""
                CREATE VIRTUAL TABLE code_fts USING fts5(
                    doc_id UNINDEXED,
                    block_id UNINDEXED,
                    block_type UNINDEXED,
                    searchable_text,
                    tokenize='porter unicode61'
                )
            """)

            # Triggers for code_fts
            cursor.execute("""
                CREATE TRIGGER code_fts_insert AFTER INSERT ON document_code_blocks BEGIN
                    INSERT INTO code_fts(rowid, doc_id, block_id, block_type, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM code_fts),
                            new.doc_id, new.block_id, new.block_type, new.searchable_text);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER code_fts_delete AFTER DELETE ON document_code_blocks BEGIN
                    DELETE FROM code_fts WHERE doc_id = old.doc_id AND block_id = old.block_id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER code_fts_update AFTER UPDATE ON document_code_blocks BEGIN
                    DELETE FROM code_fts WHERE doc_id = old.doc_id AND block_id = old.block_id;
                    INSERT INTO code_fts(rowid, doc_id, block_id, block_type, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM code_fts),
                            new.doc_id, new.block_id, new.block_type, new.searchable_text);
                END
            """)

            # Create document_facets table for faceted search
            cursor.execute("""
                CREATE TABLE document_facets (
                    doc_id TEXT NOT NULL,
                    facet_type TEXT NOT NULL,
                    facet_value TEXT NOT NULL,
                    PRIMARY KEY (doc_id, facet_type, facet_value),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faceted search
            cursor.execute("CREATE INDEX idx_facets_type_value ON document_facets(facet_type, facet_value)")
            cursor.execute("CREATE INDEX idx_facets_doc_id ON document_facets(doc_id)")

            # Create search_log table for analytics
            cursor.execute("""
                CREATE TABLE search_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    search_mode TEXT NOT NULL,
                    results_count INTEGER NOT NULL,
                    clicked_doc_id TEXT,
                    execution_time_ms REAL,
                    tags TEXT
                )
            """)

            # Create indexes for search analytics
            cursor.execute("CREATE INDEX idx_search_log_query ON search_log(query)")
            cursor.execute("CREATE INDEX idx_search_log_timestamp ON search_log(timestamp)")
            cursor.execute("CREATE INDEX idx_search_log_mode ON search_log(search_mode)")

            # Create cross_references table for content linking
            cursor.execute("""
                CREATE TABLE cross_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    ref_type TEXT NOT NULL,
                    ref_value TEXT NOT NULL,
                    context TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for cross-reference lookup
            cursor.execute("CREATE INDEX idx_xref_type_value ON cross_references(ref_type, ref_value)")
            cursor.execute("CREATE INDEX idx_xref_doc_id ON cross_references(doc_id)")

            # Create query_suggestions table for autocomplete
            cursor.execute("""
                CREATE VIRTUAL TABLE query_suggestions USING fts5(
                    term,
                    frequency UNINDEXED,
                    category UNINDEXED
                )
            """)

            # Create document_summaries table for AI-generated summaries
            cursor.execute("""
                CREATE TABLE document_summaries (
                    doc_id TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    model TEXT,
                    token_count INTEGER,
                    PRIMARY KEY (doc_id, summary_type),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for summary queries
            cursor.execute("CREATE INDEX idx_summaries_doc_id ON document_summaries(doc_id)")
            cursor.execute("CREATE INDEX idx_summaries_type ON document_summaries(summary_type)")

            # Create document_entities table for named entity extraction
            cursor.execute("""
                CREATE TABLE document_entities (
                    doc_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context TEXT,
                    first_chunk_id INTEGER,
                    occurrence_count INTEGER DEFAULT 1,
                    generated_at TEXT NOT NULL,
                    model TEXT,
                    PRIMARY KEY (doc_id, entity_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create FTS5 index for entity search
            cursor.execute("""
                CREATE VIRTUAL TABLE entities_fts USING fts5(
                    doc_id UNINDEXED,
                    entity_id UNINDEXED,
                    entity_text,
                    entity_type UNINDEXED,
                    context,
                    tokenize='porter unicode61'
                )
            """)

            # Triggers to keep entities_fts in sync with document_entities
            cursor.execute("""
                CREATE TRIGGER entities_fts_insert AFTER INSERT ON document_entities BEGIN
                    INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                    VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER entities_fts_delete AFTER DELETE ON document_entities BEGIN
                    DELETE FROM entities_fts WHERE rowid = old.rowid;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER entities_fts_update AFTER UPDATE ON document_entities BEGIN
                    DELETE FROM entities_fts WHERE rowid = old.rowid;
                    INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                    VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                END
            """)

            # Create indexes for entity queries
            cursor.execute("CREATE INDEX idx_entities_doc_id ON document_entities(doc_id)")
            cursor.execute("CREATE INDEX idx_entities_type ON document_entities(entity_type)")
            cursor.execute("CREATE INDEX idx_entities_text ON document_entities(entity_text)")

            # Create entity relationships table
            cursor.execute("""
                CREATE TABLE entity_relationships (
                    entity1_text TEXT NOT NULL,
                    entity1_type TEXT NOT NULL,
                    entity2_text TEXT NOT NULL,
                    entity2_type TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    doc_count INTEGER NOT NULL DEFAULT 1,
                    first_seen_doc TEXT,
                    context_sample TEXT,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY (entity1_text, entity2_text, relationship_type)
                )
            """)

            # Create indexes for relationship queries
            cursor.execute("CREATE INDEX idx_relationships_entity1 ON entity_relationships(entity1_text)")
            cursor.execute("CREATE INDEX idx_relationships_entity2 ON entity_relationships(entity2_text)")
            cursor.execute("CREATE INDEX idx_relationships_type ON entity_relationships(relationship_type)")
            cursor.execute("CREATE INDEX idx_relationships_strength ON entity_relationships(strength)")

            self.db_conn.commit()
            self.logger.info("Database schema created successfully (with FTS5, tables, code blocks, facets, analytics, suggestions, summaries, entities, and relationships)")
        else:
            self.logger.info("Using existing database")

            # Migrate database schema: Add file_mtime and file_hash columns if missing
            cursor = self.db_conn.cursor()
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'file_mtime' not in columns:
                self.logger.info("Migrating database: adding file_mtime column")
                cursor.execute("ALTER TABLE documents ADD COLUMN file_mtime REAL")
                self.db_conn.commit()

            if 'file_hash' not in columns:
                self.logger.info("Migrating database: adding file_hash column")
                cursor.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
                self.db_conn.commit()

            # Migrate: Add URL scraping columns if missing
            if 'source_url' not in columns:
                self.logger.info("Migrating database: adding URL scraping columns")
                cursor.execute("ALTER TABLE documents ADD COLUMN source_url TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN scrape_date TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN scrape_config TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN scrape_status TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN scrape_error TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN url_last_checked TEXT")
                cursor.execute("ALTER TABLE documents ADD COLUMN url_content_hash TEXT")

                # Create indexes for URL columns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents(source_url)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_scrape_status ON documents(scrape_status)")

                self.db_conn.commit()
                self.logger.info("URL scraping columns and indexes added successfully")

            # Check if FTS5 table exists and populate if needed
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks_fts5")
                fts5_count = cursor.fetchone()[0]

                # If FTS5 table is empty but chunks exist, populate it
                if fts5_count == 0:
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    chunks_count = cursor.fetchone()[0]

                    if chunks_count > 0:
                        self.logger.info(f"Populating FTS5 index with {chunks_count} existing chunks")
                        cursor.execute("""
                            INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                            SELECT rowid, doc_id, chunk_id, content FROM chunks
                        """)
                        self.db_conn.commit()
                        self.logger.info("FTS5 index populated successfully")
            except Exception as e:
                # FTS5 table doesn't exist, create it
                self.logger.info(f"Creating FTS5 table for existing database: {e}")
                cursor = self.db_conn.cursor()

                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts5 USING fts5(
                        doc_id UNINDEXED,
                        chunk_id UNINDEXED,
                        content,
                        tokenize='porter unicode61'
                    )
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts5_insert AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                        VALUES (new.rowid, new.doc_id, new.chunk_id, new.content);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts5_delete AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts5 WHERE rowid = old.rowid;
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts5_update AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts5 WHERE rowid = old.rowid;
                        INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                        VALUES (new.rowid, new.doc_id, new.chunk_id, new.content);
                    END
                """)

                # Populate FTS5 from existing chunks
                cursor.execute("""
                    INSERT INTO chunks_fts5(rowid, doc_id, chunk_id, content)
                    SELECT rowid, doc_id, chunk_id, content FROM chunks
                """)

                self.db_conn.commit()
                self.logger.info("FTS5 table created and populated for existing database")

            # Migrate: Add document_tables table if not exists
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_tables'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating document_tables table")
                cursor.execute("""
                    CREATE TABLE document_tables (
                        doc_id TEXT NOT NULL,
                        table_id INTEGER NOT NULL,
                        page INTEGER,
                        markdown TEXT NOT NULL,
                        searchable_text TEXT NOT NULL,
                        row_count INTEGER,
                        col_count INTEGER,
                        PRIMARY KEY (doc_id, table_id),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                # Create FTS5 index for table search
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS tables_fts USING fts5(
                        doc_id UNINDEXED,
                        table_id UNINDEXED,
                        searchable_text,
                        tokenize='porter unicode61'
                    )
                """)

                # Triggers for tables_fts
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS tables_fts_insert AFTER INSERT ON document_tables BEGIN
                        INSERT INTO tables_fts(rowid, doc_id, table_id, searchable_text)
                        VALUES (new.rowid, new.doc_id, new.table_id, new.searchable_text);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS tables_fts_delete AFTER DELETE ON document_tables BEGIN
                        DELETE FROM tables_fts WHERE rowid = old.rowid;
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS tables_fts_update AFTER UPDATE ON document_tables BEGIN
                        DELETE FROM tables_fts WHERE rowid = old.rowid;
                        INSERT INTO tables_fts(rowid, doc_id, table_id, searchable_text)
                        VALUES (new.rowid, new.doc_id, new.table_id, new.searchable_text);
                    END
                """)

                self.db_conn.commit()
                self.logger.info("document_tables and tables_fts created")

            # Migrate: Add document_code_blocks table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_code_blocks'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating document_code_blocks table")
                cursor.execute("""
                    CREATE TABLE document_code_blocks (
                        doc_id TEXT NOT NULL,
                        block_id INTEGER NOT NULL,
                        page INTEGER,
                        block_type TEXT NOT NULL,
                        code TEXT NOT NULL,
                        searchable_text TEXT NOT NULL,
                        line_count INTEGER,
                        PRIMARY KEY (doc_id, block_id),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                # Create FTS5 index for code search
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(
                        doc_id UNINDEXED,
                        block_id UNINDEXED,
                        block_type UNINDEXED,
                        searchable_text,
                        tokenize='porter unicode61'
                    )
                """)

                # Triggers for code_fts
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS code_fts_insert AFTER INSERT ON document_code_blocks BEGIN
                        INSERT INTO code_fts(rowid, doc_id, block_id, block_type, searchable_text)
                        VALUES (new.rowid, new.doc_id, new.block_id, new.block_type, new.searchable_text);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS code_fts_delete AFTER DELETE ON document_code_blocks BEGIN
                        DELETE FROM code_fts WHERE rowid = old.rowid;
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS code_fts_update AFTER UPDATE ON document_code_blocks BEGIN
                        DELETE FROM code_fts WHERE rowid = old.rowid;
                        INSERT INTO code_fts(rowid, doc_id, block_id, block_type, searchable_text)
                        VALUES (new.rowid, new.doc_id, new.block_id, new.block_type, new.searchable_text);
                    END
                """)

                self.db_conn.commit()
                self.logger.info("document_code_blocks and code_fts created")

            # Migrate: Add document_facets table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_facets'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating document_facets table for faceted search")
                cursor.execute("""
                    CREATE TABLE document_facets (
                        doc_id TEXT NOT NULL,
                        facet_type TEXT NOT NULL,
                        facet_value TEXT NOT NULL,
                        PRIMARY KEY (doc_id, facet_type, facet_value),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("CREATE INDEX idx_facets_type_value ON document_facets(facet_type, facet_value)")
                cursor.execute("CREATE INDEX idx_facets_doc_id ON document_facets(doc_id)")

                self.db_conn.commit()
                self.logger.info("document_facets table created")

            # Migrate: Add search_log table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='search_log'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating search_log table for analytics")
                cursor.execute("""
                    CREATE TABLE search_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        query TEXT NOT NULL,
                        search_mode TEXT NOT NULL,
                        results_count INTEGER NOT NULL,
                        clicked_doc_id TEXT,
                        execution_time_ms REAL,
                        tags TEXT
                    )
                """)

                cursor.execute("CREATE INDEX idx_search_log_query ON search_log(query)")
                cursor.execute("CREATE INDEX idx_search_log_timestamp ON search_log(timestamp)")
                cursor.execute("CREATE INDEX idx_search_log_mode ON search_log(search_mode)")

                self.db_conn.commit()
                self.logger.info("search_log table created")

            # Migrate: Add cross_references table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='cross_references'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating cross_references table for content linking")
                cursor.execute("""
                    CREATE TABLE cross_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT NOT NULL,
                        chunk_id INTEGER NOT NULL,
                        ref_type TEXT NOT NULL,
                        ref_value TEXT NOT NULL,
                        context TEXT,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("CREATE INDEX idx_xref_type_value ON cross_references(ref_type, ref_value)")
                cursor.execute("CREATE INDEX idx_xref_doc_id ON cross_references(doc_id)")

                self.db_conn.commit()
                self.logger.info("cross_references table created")

            # Migrate: Add query_suggestions table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='query_suggestions'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating query_suggestions table for autocomplete")
                cursor.execute("""
                    CREATE VIRTUAL TABLE query_suggestions USING fts5(
                        term,
                        frequency UNINDEXED,
                        category UNINDEXED
                    )
                """)

                self.db_conn.commit()
                self.logger.info("query_suggestions table created")

            # Migrate: Add document_summaries table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_summaries'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating document_summaries table for AI-generated summaries")
                cursor.execute("""
                    CREATE TABLE document_summaries (
                        doc_id TEXT NOT NULL,
                        summary_type TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        generated_at TEXT NOT NULL,
                        model TEXT,
                        token_count INTEGER,
                        PRIMARY KEY (doc_id, summary_type),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                # Create indexes for summary queries
                cursor.execute("CREATE INDEX idx_summaries_doc_id ON document_summaries(doc_id)")
                cursor.execute("CREATE INDEX idx_summaries_type ON document_summaries(summary_type)")

                self.db_conn.commit()
                self.logger.info("document_summaries table created")

            # Migrate: Add document_entities table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_entities'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating document_entities table for named entity extraction")
                cursor.execute("""
                    CREATE TABLE document_entities (
                        doc_id TEXT NOT NULL,
                        entity_id INTEGER NOT NULL,
                        entity_text TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        context TEXT,
                        first_chunk_id INTEGER,
                        occurrence_count INTEGER DEFAULT 1,
                        generated_at TEXT NOT NULL,
                        model TEXT,
                        PRIMARY KEY (doc_id, entity_id),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                # Create FTS5 index for entity search
                cursor.execute("""
                    CREATE VIRTUAL TABLE entities_fts USING fts5(
                        doc_id UNINDEXED,
                        entity_id UNINDEXED,
                        entity_text,
                        entity_type UNINDEXED,
                        context,
                        tokenize='porter unicode61'
                    )
                """)

                # Triggers to keep entities_fts in sync
                cursor.execute("""
                    CREATE TRIGGER entities_fts_insert AFTER INSERT ON document_entities BEGIN
                        INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                        VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER entities_fts_delete AFTER DELETE ON document_entities BEGIN
                        DELETE FROM entities_fts WHERE rowid = old.rowid;
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER entities_fts_update AFTER UPDATE ON document_entities BEGIN
                        DELETE FROM entities_fts WHERE rowid = old.rowid;
                        INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                        VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                    END
                """)

                # Create indexes for entity queries
                cursor.execute("CREATE INDEX idx_entities_doc_id ON document_entities(doc_id)")
                cursor.execute("CREATE INDEX idx_entities_type ON document_entities(entity_type)")
                cursor.execute("CREATE INDEX idx_entities_text ON document_entities(entity_text)")

                self.db_conn.commit()
                self.logger.info("document_entities table created")

            # Migrate: Add entity_relationships table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='entity_relationships'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating entity_relationships table for entity co-occurrence tracking")
                cursor.execute("""
                    CREATE TABLE entity_relationships (
                        entity1_text TEXT NOT NULL,
                        entity1_type TEXT NOT NULL,
                        entity2_text TEXT NOT NULL,
                        entity2_type TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        strength REAL NOT NULL,
                        doc_count INTEGER NOT NULL DEFAULT 1,
                        first_seen_doc TEXT,
                        context_sample TEXT,
                        last_updated TEXT NOT NULL,
                        PRIMARY KEY (entity1_text, entity2_text, relationship_type)
                    )
                """)

                # Create indexes for relationship queries
                cursor.execute("CREATE INDEX idx_relationships_entity1 ON entity_relationships(entity1_text)")
                cursor.execute("CREATE INDEX idx_relationships_entity2 ON entity_relationships(entity2_text)")
                cursor.execute("CREATE INDEX idx_relationships_type ON entity_relationships(relationship_type)")
                cursor.execute("CREATE INDEX idx_relationships_strength ON entity_relationships(strength)")

                self.db_conn.commit()
                self.logger.info("entity_relationships table created")

    def _fts5_available(self) -> bool:
        """Check if FTS5 is available and table exists."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks_fts5 LIMIT 1")
            return True
        except Exception:
            return False

    def _load_documents(self):
        """Load documents from database, with automatic migration from JSON if needed."""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        # Check if database is empty but JSON files exist (migration needed)
        if doc_count == 0 and self.index_file.exists():
            self.logger.info("Found legacy JSON index, performing automatic migration to SQLite")
            self._migrate_from_json()
        else:
            # Load documents from database
            self.logger.info(f"Loading {doc_count} documents from database")
            cursor.execute("SELECT * FROM documents")
            rows = cursor.fetchall()

            for row in rows:
                doc = DocumentMeta(
                    doc_id=row[0],
                    filename=row[1],
                    title=row[2],
                    filepath=row[3],
                    file_type=row[4],
                    total_pages=row[5],
                    total_chunks=row[6],
                    indexed_at=row[7],
                    tags=json.loads(row[8]),
                    author=row[9],
                    subject=row[10],
                    creator=row[11],
                    creation_date=row[12],
                    file_mtime=row[13] if len(row) > 13 else None,
                    file_hash=row[14] if len(row) > 14 else None,
                    source_url=row[15] if len(row) > 15 else None,
                    scrape_date=row[16] if len(row) > 16 else None,
                    scrape_config=row[17] if len(row) > 17 else None,
                    scrape_status=row[18] if len(row) > 18 else None,
                    scrape_error=row[19] if len(row) > 19 else None,
                    url_last_checked=row[20] if len(row) > 20 else None,
                    url_content_hash=row[21] if len(row) > 21 else None
                )
                self.documents[doc.doc_id] = doc

    def _migrate_from_json(self):
        """Migrate existing JSON index and chunks to SQLite database."""
        self.logger.info("Starting migration from JSON to SQLite")

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            migrated_count = 0
            for doc_data in data.get('documents', []):
                doc_id = doc_data['doc_id']

                # Load chunks from chunks/{doc_id}.json
                chunk_file = self.chunks_dir / f"{doc_id}.json"
                if not chunk_file.exists():
                    self.logger.warning(f"Missing chunk file for {doc_id}, skipping")
                    continue

                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_list = json.load(f)

                # Insert into database
                doc_meta = DocumentMeta(**doc_data)
                chunks = [DocumentChunk(**c) for c in chunk_list]
                self._add_document_db(doc_meta, chunks)

                self.documents[doc_id] = doc_meta
                migrated_count += 1

            self.logger.info(f"Successfully migrated {migrated_count} documents to SQLite")
            self.logger.info("JSON files preserved as backup (can be manually deleted)")

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise KnowledgeBaseError(f"Failed to migrate from JSON: {e}")

    def _add_document_db(self, doc_meta: DocumentMeta, chunks: list[DocumentChunk],
                         tables: Optional[list[dict]] = None, code_blocks: Optional[list[dict]] = None,
                         facets: Optional[dict[str, set[str]]] = None, cross_refs: Optional[list[dict]] = None):
        """Add a document, chunks, tables, code blocks, facets, and cross-references to the database using a transaction."""
        cursor = self.db_conn.cursor()

        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            # Insert document
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages, total_chunks,
                 indexed_at, tags, author, subject, creator, creation_date, file_mtime, file_hash,
                 source_url, scrape_date, scrape_config, scrape_status, scrape_error,
                 url_last_checked, url_content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_meta.doc_id,
                doc_meta.filename,
                doc_meta.title,
                doc_meta.filepath,
                doc_meta.file_type,
                doc_meta.total_pages,
                doc_meta.total_chunks,
                doc_meta.indexed_at,
                json.dumps(doc_meta.tags),
                doc_meta.author,
                doc_meta.subject,
                doc_meta.creator,
                doc_meta.creation_date,
                doc_meta.file_mtime,
                doc_meta.file_hash,
                doc_meta.source_url,
                doc_meta.scrape_date,
                doc_meta.scrape_config,
                doc_meta.scrape_status,
                doc_meta.scrape_error,
                doc_meta.url_last_checked,
                doc_meta.url_content_hash
            ))

            # Delete old chunks if re-indexing
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_meta.doc_id,))

            # Insert chunks
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks
                    (doc_id, chunk_id, page, content, word_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.page,
                    chunk.content,
                    chunk.word_count
                ))

            # Delete old tables if re-indexing
            cursor.execute("DELETE FROM document_tables WHERE doc_id = ?", (doc_meta.doc_id,))

            # Insert tables
            if tables:
                for table in tables:
                    cursor.execute("""
                        INSERT INTO document_tables
                        (doc_id, table_id, page, markdown, searchable_text, row_count, col_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc_meta.doc_id,
                        table['table_id'],
                        table['page'],
                        table['markdown'],
                        table['searchable_text'],
                        table['row_count'],
                        table['col_count']
                    ))

            # Delete old code blocks if re-indexing
            cursor.execute("DELETE FROM document_code_blocks WHERE doc_id = ?", (doc_meta.doc_id,))

            # Insert code blocks
            if code_blocks:
                for block in code_blocks:
                    cursor.execute("""
                        INSERT INTO document_code_blocks
                        (doc_id, block_id, page, block_type, code, searchable_text, line_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc_meta.doc_id,
                        block['block_id'],
                        block['page'],
                        block['block_type'],
                        block['code'],
                        block['searchable_text'],
                        block['line_count']
                    ))

            # Delete old facets if re-indexing
            cursor.execute("DELETE FROM document_facets WHERE doc_id = ?", (doc_meta.doc_id,))

            # Insert facets
            if facets:
                for facet_type, facet_values in facets.items():
                    for facet_value in facet_values:
                        cursor.execute("""
                            INSERT INTO document_facets (doc_id, facet_type, facet_value)
                            VALUES (?, ?, ?)
                        """, (doc_meta.doc_id, facet_type, facet_value))

            # Delete old cross-references if re-indexing
            cursor.execute("DELETE FROM cross_references WHERE doc_id = ?", (doc_meta.doc_id,))

            # Insert cross-references
            if cross_refs:
                for ref in cross_refs:
                    cursor.execute("""
                        INSERT INTO cross_references (doc_id, chunk_id, ref_type, ref_value, context)
                        VALUES (?, ?, ?, ?, ?)
                    """, (ref['doc_id'], ref['chunk_id'], ref['ref_type'], ref['ref_value'], ref['context']))

            # Commit transaction
            self.db_conn.commit()

        except Exception as e:
            # Rollback on error
            self.db_conn.rollback()
            self.logger.error(f"Error adding document to database: {e}")
            raise KnowledgeBaseError(f"Failed to add document to database: {e}")

    def _remove_document_db(self, doc_id: str) -> bool:
        """Remove a document from the database (chunks cascade automatically)."""
        cursor = self.db_conn.cursor()

        try:
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.db_conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Error removing document from database: {e}")
            raise KnowledgeBaseError(f"Failed to remove document from database: {e}")

    def _get_chunks_db(self, doc_id: Optional[str] = None) -> list[DocumentChunk]:
        """Load chunks from database. If doc_id is None, load all chunks."""
        cursor = self.db_conn.cursor()

        if doc_id:
            cursor.execute("""
                SELECT c.doc_id, d.filename, d.title, c.chunk_id, c.page, c.content, c.word_count
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.doc_id = ?
                ORDER BY c.chunk_id
            """, (doc_id,))
        else:
            cursor.execute("""
                SELECT c.doc_id, d.filename, d.title, c.chunk_id, c.page, c.content, c.word_count
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                ORDER BY c.doc_id, c.chunk_id
            """)

        chunks = []
        for row in cursor.fetchall():
            chunk = DocumentChunk(
                doc_id=row[0],
                filename=row[1],
                title=row[2],
                chunk_id=row[3],
                page=row[4],
                content=row[5],
                word_count=row[6]
            )
            chunks.append(chunk)

        return chunks

    def _load_index(self):
        """Load existing index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for doc_data in data.get('documents', []):
                    doc = DocumentMeta(**doc_data)
                    self.documents[doc.doc_id] = doc
                    
            # Load chunks
            for chunk_file in self.chunks_dir.glob("*.json"):
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    for c in chunk_data:
                        self.chunks.append(DocumentChunk(**c))

    def _build_bm25_index(self):
        """Build BM25 index from chunks for fast searching (lazy loading from database)."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()

        if not BM25_SUPPORT:
            self.logger.info("BM25 index not built (no support)")
            return

        # Lazy load all chunks from database if not already in memory
        if not self.chunks:
            self.logger.info("Loading all chunks from database for BM25 index")
            load_start = time.time()
            self.chunks = self._get_chunks_db()
            load_time = time.time() - load_start
            self.logger.info(f"Loaded {len(self.chunks)} chunks in {load_time:.2f}s")

        if not self.chunks:
            self.logger.info("BM25 index not built (no chunks)")
            return

        # Tokenize all chunk content with preprocessing if enabled
        # Use parallel processing for faster tokenization
        self.logger.info(f"Tokenizing {len(self.chunks)} chunks...")
        tokenize_start = time.time()

        if self.use_preprocessing and len(self.chunks) > 100:
            # Parallel tokenization for large datasets
            tokenized_corpus = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tokenization tasks
                future_to_idx = {
                    executor.submit(self._preprocess_text, chunk.content): idx
                    for idx, chunk in enumerate(self.chunks)
                }

                # Collect results in order
                results = [None] * len(self.chunks)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()

                tokenized_corpus = results
        else:
            # Sequential for small datasets or no preprocessing
            tokenized_corpus = [self._preprocess_text(chunk.content) for chunk in self.chunks]

        tokenize_time = time.time() - tokenize_start
        self.logger.info(f"Tokenization completed in {tokenize_time:.2f}s")

        # Build BM25 index
        index_start = time.time()
        self.bm25 = BM25Okapi(tokenized_corpus)
        index_time = time.time() - index_start

        total_time = time.time() - start_time
        preprocessing_status = "with preprocessing" if self.use_preprocessing else "without preprocessing"
        self.logger.info(f"Built BM25 index with {len(self.chunks)} chunks ({preprocessing_status}) - Total: {total_time:.2f}s (load: {load_time:.2f}s, tokenize: {tokenize_time:.2f}s, index: {index_time:.2f}s)")

    def _fuzzy_match_terms(self, query_terms: list[str], content: str) -> tuple[bool, float]:
        """
        Check if query terms fuzzy match content using rapidfuzz.

        Args:
            query_terms: List of query terms to match
            content: Content to search in

        Returns:
            Tuple of (match_found, average_similarity_score)
        """
        if not self.use_fuzzy:
            # Fuzzy search disabled, do exact matching
            content_lower = content.lower()
            matches = sum(1 for term in query_terms if term.lower() in content_lower)
            return (matches > 0, matches / len(query_terms) if query_terms else 0.0)

        # Split content into words for fuzzy matching
        content_words = content.lower().split()

        match_scores = []
        for query_term in query_terms:
            query_term_lower = query_term.lower()

            # Check for exact match first (fastest)
            if query_term_lower in content.lower():
                match_scores.append(100.0)
                continue

            # Try fuzzy matching against content words
            best_score = 0.0
            for content_word in content_words:
                score = fuzz.ratio(query_term_lower, content_word)
                if score > best_score:
                    best_score = score
                if score >= self.fuzzy_threshold:
                    break  # Found a good enough match

            match_scores.append(best_score)

        # Consider it a match if at least one term meets the threshold
        match_found = any(score >= self.fuzzy_threshold for score in match_scores)
        avg_score = sum(match_scores) / len(match_scores) if match_scores else 0.0

        return (match_found, avg_score)

    def _preprocess_text(self, text: str) -> list[str]:
        """Preprocess text for searching: tokenize, lowercase, remove stopwords, stem.

        Args:
            text: The text to preprocess

        Returns:
            List of processed tokens
        """
        if not self.use_preprocessing:
            # No preprocessing - just lowercase and split
            return text.lower().split()

        # Tokenize and lowercase
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            # Fallback if tokenization fails
            tokens = text.lower().split()

        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            # Keep alphanumeric tokens and hyphenated words (like VIC-II, 6502)
            # Remove pure punctuation tokens
            if token.isalnum() or ('-' in token and any(c.isalnum() for c in token)):
                # Remove stopwords (but keep technical terms with hyphens)
                if token not in self.stop_words:
                    # Apply stemming only to pure alphanumeric tokens
                    # Don't stem technical terms with hyphens/numbers
                    if self.stemmer and token.isalpha():
                        stemmed = self.stemmer.stem(token)
                        processed_tokens.append(stemmed)
                    else:
                        processed_tokens.append(token)

        return processed_tokens

    def _save_index(self):
        """Save index to disk."""
        data = {
            'documents': [asdict(doc) for doc in self.documents.values()],
            'last_updated': datetime.now().isoformat()
        }
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _save_chunks(self, doc_id: str, chunks: list[DocumentChunk]):
        """Save chunks for a document."""
        chunk_file = self.chunks_dir / f"{doc_id}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(c) for c in chunks], f)
    
    def _generate_doc_id(self, filepath: str, text_content: str = None) -> str:
        """
        Generate a unique document ID based on content hash.

        If text_content is provided, generates ID from content hash (deduplication).
        If not provided, falls back to filepath hash (legacy behavior).

        Args:
            filepath: Path to the document
            text_content: Extracted text content (optional)

        Returns:
            12-character hex string document ID
        """
        if text_content:
            # Content-based ID for deduplication
            # Normalize text: lowercase, strip whitespace
            normalized = text_content.lower().strip()
            # Hash first 10k words to handle large documents efficiently
            words = normalized.split()[:10000]
            content_sample = ' '.join(words)
            return hashlib.md5(content_sample.encode('utf-8')).hexdigest()[:12]
        else:
            # Filepath-based ID (legacy)
            return hashlib.md5(filepath.encode()).hexdigest()[:12]
    
    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            start = end - overlap
            
        return chunks
    
    def _extract_pdf_with_ocr(self, filepath: str) -> tuple[str, int]:
        """Extract text from scanned PDF using OCR, returns (text, page_count)."""
        if not self.use_ocr:
            raise RuntimeError("OCR not enabled. Set USE_OCR=1 and install Tesseract.")

        if not self.poppler_available:
            raise RuntimeError(
                "Poppler not found! OCR requires poppler-utils.\n"
                "Install instructions:\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "  Set POPPLER_PATH environment variable to the bin directory\n"
                "  Example: POPPLER_PATH=C:\\path\\to\\poppler-24.08.0\\Library\\bin"
            )

        try:
            self.logger.info(f"Using OCR to extract text from scanned PDF: {filepath}")
            # Convert PDF pages to images
            if self.poppler_path:
                images = convert_from_path(filepath, poppler_path=self.poppler_path)
            else:
                images = convert_from_path(filepath)

            pages = []
            for i, image in enumerate(images):
                try:
                    # Extract text using Tesseract OCR
                    text = pytesseract.image_to_string(image)
                    pages.append(text)
                    self.logger.debug(f"OCR processed page {i + 1}/{len(images)}")
                except Exception as e:
                    self.logger.error(f"OCR failed for page {i + 1}: {e}")
                    pages.append("")  # Empty text for failed page

            full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
            self.logger.info(f"OCR extraction complete: {len(full_text)} characters from {len(images)} pages")
            return full_text, len(images)

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")

    def _check_poppler_available(self) -> bool:
        """Check if poppler is available for pdf2image.

        Returns True if poppler can be found, False otherwise.
        """
        if not OCR_SUPPORT:
            return False

        try:
            # Try to import pdf2image
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoNotInstalledError

            # Create a minimal test - just try to access pdfinfo
            import subprocess

            # Determine the command to check based on poppler_path
            if self.poppler_path:
                # Check if pdfinfo exists in the specified path
                pdfinfo_cmd = os.path.join(self.poppler_path, 'pdfinfo')
                if os.name == 'nt':  # Windows
                    pdfinfo_cmd += '.exe'

                if not os.path.exists(pdfinfo_cmd):
                    return False

                # Try to run pdfinfo -v (poppler uses -v not --version)
                result = subprocess.run(
                    [pdfinfo_cmd, '-v'],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            else:
                # Check if pdfinfo is in PATH
                if os.name == 'nt':  # Windows
                    result = subprocess.run(
                        ['where', 'pdfinfo'],
                        capture_output=True,
                        timeout=5
                    )
                else:  # Unix-like
                    result = subprocess.run(
                        ['which', 'pdfinfo'],
                        capture_output=True,
                        timeout=5
                    )
                return result.returncode == 0

        except Exception as e:
            self.logger.debug(f"Poppler check failed: {e}")
            return False

    def _extract_pdf_text(self, filepath: str) -> tuple[str, int, dict]:
        """Extract text from PDF with automatic OCR fallback for scanned PDFs.

        Returns (text, page_count, metadata).
        """
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install pypdf: pip install pypdf")

        reader = PdfReader(filepath)
        pages = []
        total_text_length = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            total_text_length += len(text.strip())

        # Check if PDF appears to be scanned (very little or no text extracted)
        # Threshold: if we get less than 100 characters total, likely scanned
        is_scanned = total_text_length < 100 and len(reader.pages) > 0

        if is_scanned and self.use_ocr:
            self.logger.info(f"PDF appears to be scanned ({total_text_length} chars extracted), falling back to OCR")
            try:
                ocr_text, page_count = self._extract_pdf_with_ocr(filepath)
                # Use OCR text instead
                full_text = ocr_text
                # Still extract metadata from PDF
            except Exception as e:
                self.logger.warning(f"OCR fallback failed: {e}, using extracted text anyway")
                full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)
        else:
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages)

        # Extract metadata
        metadata = {}
        if reader.metadata:
            # Convert metadata values to strings to handle IndirectObject references
            author = reader.metadata.get('/Author')
            metadata['author'] = str(author) if author else None

            subject = reader.metadata.get('/Subject')
            metadata['subject'] = str(subject) if subject else None

            creator = reader.metadata.get('/Creator')
            metadata['creator'] = str(creator) if creator else None

            creation_date = reader.metadata.get('/CreationDate')
            if creation_date:
                # Try to parse PDF date format (D:YYYYMMDDHHmmSS)
                try:
                    creation_date_str = str(creation_date)
                    if creation_date_str.startswith('D:'):
                        date_str = creation_date_str[2:16]  # Extract YYYYMMDDHHmmSS
                        metadata['creation_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        metadata['creation_date'] = creation_date_str
                except:
                    metadata['creation_date'] = str(creation_date)

        return "\n\n--- PAGE BREAK ---\n\n".join(pages), len(reader.pages), metadata
    
    def _extract_text_file(self, filepath: str) -> str:
        """Extract text from a text file."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode {filepath}")

    def _extract_excel_file(self, filepath: str) -> tuple[str, int]:
        """
        Extract text from Excel file (.xlsx, .xls).

        Returns:
            Tuple of (text_content, sheet_count)
        """
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise RuntimeError("openpyxl not installed. Install with: pip install openpyxl")

        try:
            workbook = load_workbook(filepath, data_only=True)
            sheets_text = []

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]

                # Add sheet header
                sheets_text.append(f"\n{'='*60}\nSheet: {sheet_name}\n{'='*60}\n")

                # Extract all cell values
                rows_text = []
                for row in sheet.iter_rows(values_only=True):
                    # Filter out None values and convert to strings
                    row_values = [str(cell) if cell is not None else '' for cell in row]
                    # Skip completely empty rows
                    if any(val.strip() for val in row_values):
                        rows_text.append('\t'.join(row_values))

                sheets_text.append('\n'.join(rows_text))

            text_content = '\n\n'.join(sheets_text)
            sheet_count = len(workbook.sheetnames)

            self.logger.info(f"Extracted {sheet_count} sheets from Excel file")
            return text_content, sheet_count

        except Exception as e:
            raise RuntimeError(f"Error reading Excel file: {str(e)}")

    def _extract_html_file(self, filepath: str) -> str:
        """
        Extract text from HTML file (.html, .htm).

        Returns:
            Extracted text content
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise RuntimeError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

        try:
            # Read the HTML file with encoding detection
            with open(filepath, 'rb') as f:
                raw_data = f.read()

            # Detect encoding
            import chardet
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')

            # Decode with detected encoding
            html_content = raw_data.decode(encoding, errors='replace')

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()

            # Extract text content
            text_parts = []

            # Get title if available
            if soup.title and soup.title.string:
                text_parts.append(f"Title: {soup.title.string.strip()}\n")

            # Process body or entire document
            body = soup.body if soup.body else soup

            # Handle code blocks specially to preserve formatting
            for pre in body.find_all(['pre', 'code']):
                # Mark code blocks so they're preserved
                pre_text = pre.get_text()
                if pre_text.strip():
                    text_parts.append(f"\n--- CODE BLOCK ---\n{pre_text}\n--- END CODE BLOCK ---\n")
                pre.decompose()  # Remove so we don't process again

            # Get remaining text
            main_text = body.get_text(separator='\n', strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in main_text.split('\n')]
            lines = [line for line in lines if line]  # Remove empty lines
            text_parts.append('\n'.join(lines))

            text_content = '\n\n'.join(text_parts)

            self.logger.info(f"Extracted HTML file ({len(text_content)} characters)")
            return text_content

        except Exception as e:
            raise RuntimeError(f"Error reading HTML file: {str(e)}")

    def _extract_tables(self, filepath: str) -> list[dict]:
        """Extract tables from PDF using pdfplumber.

        Returns a list of table dictionaries with structure:
        {
            'table_id': int,
            'page': int,
            'markdown': str,
            'searchable_text': str,
            'row_count': int,
            'col_count': int
        }
        """
        if not PDFPLUMBER_SUPPORT:
            self.logger.debug("pdfplumber not available, skipping table extraction")
            return []

        tables = []
        table_id = 0

        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract tables from this page
                    page_tables = page.extract_tables()

                    if page_tables:
                        for table_data in page_tables:
                            if not table_data or len(table_data) == 0:
                                continue

                            # Convert table to markdown
                            markdown = self._table_to_markdown(table_data)

                            # Create searchable text (all cells joined with spaces)
                            searchable_text = " ".join(
                                str(cell).strip()
                                for row in table_data
                                for cell in row
                                if cell and str(cell).strip()
                            )

                            tables.append({
                                'table_id': table_id,
                                'page': page_num,
                                'markdown': markdown,
                                'searchable_text': searchable_text,
                                'row_count': len(table_data),
                                'col_count': len(table_data[0]) if table_data else 0
                            })
                            table_id += 1

        except Exception as e:
            self.logger.warning(f"Error extracting tables from {filepath}: {e}")
            return []

        self.logger.info(f"Extracted {len(tables)} tables from PDF")
        return tables

    def _table_to_markdown(self, table_data: list[list]) -> str:
        """Convert a table (list of lists) to markdown format."""
        if not table_data or len(table_data) == 0:
            return ""

        lines = []

        # Header row
        header = table_data[0]
        lines.append("| " + " | ".join(str(cell or "").strip() for cell in header) + " |")

        # Separator row
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        # Data rows
        for row in table_data[1:]:
            lines.append("| " + " | ".join(str(cell or "").strip() for cell in row) + " |")

        return "\n".join(lines)

    def _detect_code_blocks(self, text: str) -> list[dict]:
        """Detect code blocks in text (BASIC, Assembly, Hex dumps).

        Returns a list of code block dictionaries with structure:
        {
            'block_id': int,
            'page': None,  # Page detection happens in add_document
            'block_type': str,  # 'basic', 'assembly', or 'hex'
            'code': str,
            'searchable_text': str,
            'line_count': int
        }
        """
        code_blocks = []
        block_id = 0

        # Pattern 1: BASIC code (lines starting with line numbers)
        # Example: "10 PRINT "HELLO"", "20 GOTO 10"
        basic_pattern = r'(?:^|\n)((?:\d+\s+[A-Z]+[^\n]*\n?){3,})'

        for match in re.finditer(basic_pattern, text, re.MULTILINE):
            code = match.group(1).strip()
            lines = code.split('\n')

            code_blocks.append({
                'block_id': block_id,
                'page': None,
                'block_type': 'basic',
                'code': code,
                'searchable_text': code,
                'line_count': len(lines)
            })
            block_id += 1

        # Pattern 2: Assembly code (mnemonics: LDA, STA, JMP, etc.)
        # Example: "    LDA #$00", "    STA $D020"
        assembly_pattern = r'(?:^|\n)((?:\s*(?:LDA|STA|LDX|STX|LDY|STY|JMP|JSR|RTS|BEQ|BNE|BCC|BCS|ADC|SBC|AND|ORA|EOR|INC|DEC|CMP|CPX|CPY|ASL|LSR|ROL|ROR|BIT|NOP|CLC|SEC|CLI|SEI|CLD|SED|CLV|PHA|PLA|PHP|PLP|TAX|TAY|TXA|TYA|TSX|TXS|INX|INY|DEX|DEY|BMI|BPL|BVC|BVS)[^\n]*\n?){3,})'

        for match in re.finditer(assembly_pattern, text, re.MULTILINE | re.IGNORECASE):
            code = match.group(1).strip()
            lines = code.split('\n')

            # Avoid duplicates (check if this code is already captured)
            if not any(block['code'] == code for block in code_blocks):
                code_blocks.append({
                    'block_id': block_id,
                    'page': None,
                    'block_type': 'assembly',
                    'code': code,
                    'searchable_text': code,
                    'line_count': len(lines)
                })
                block_id += 1

        # Pattern 3: Hex dumps (lines with hex values)
        # Example: "D000: 00 01 02 03 04 05 06 07"
        hex_pattern = r'(?:^|\n)((?:[0-9A-F]{4}:\s*(?:[0-9A-F]{2}\s*){8,}\n?){3,})'

        for match in re.finditer(hex_pattern, text, re.MULTILINE | re.IGNORECASE):
            code = match.group(1).strip()
            lines = code.split('\n')

            code_blocks.append({
                'block_id': block_id,
                'page': None,
                'block_type': 'hex',
                'code': code,
                'searchable_text': code,
                'line_count': len(lines)
            })
            block_id += 1

        basic_count = sum(1 for b in code_blocks if b['block_type'] == 'basic')
        assembly_count = sum(1 for b in code_blocks if b['block_type'] == 'assembly')
        hex_count = sum(1 for b in code_blocks if b['block_type'] == 'hex')
        self.logger.info(f"Detected {len(code_blocks)} code blocks ({basic_count} BASIC, {assembly_count} Assembly, {hex_count} Hex)")
        return code_blocks

    def _extract_facets(self, text: str) -> dict[str, set[str]]:
        """Extract categorizable terms for faceted search.

        Returns a dictionary of facet types to sets of values:
        {
            'hardware': {'SID', 'VIC-II', 'CIA'},
            'instruction': {'LDA', 'STA', 'JMP'},
            'register': {'$D000', '$D400'}
        }
        """
        facets = {
            'hardware': set(),
            'instruction': set(),
            'register': set()
        }

        # Extract hardware components
        facets['hardware'] = self._extract_hardware_refs(text)

        # Extract 6502 instructions
        facets['instruction'] = self._extract_instructions(text)

        # Extract register addresses
        facets['register'] = self._extract_registers(text)

        return facets

    def _extract_hardware_refs(self, text: str) -> set[str]:
        """Extract hardware component mentions from text."""
        hardware = set()

        # Hardware patterns (case-insensitive)
        patterns = {
            'SID': r'\b(?:SID|6581|8580|Sound\s+Interface\s+Device)\b',
            'VIC-II': r'\b(?:VIC-?II|VIC\s*2|6569|6567|Video\s+Interface\s+Chip)\b',
            'CIA': r'\b(?:CIA|6526|Complex\s+Interface\s+Adapter)\b',
            '6502': r'\b6502\b',
            'PLA': r'\b(?:PLA|82S100|Programmable\s+Logic\s+Array)\b',
            'Datasette': r'\b(?:Datasette|1530|C2N)\b',
            'Disk Drive': r'\b(?:1541|1571|1581|Disk\s+Drive)\b',
        }

        for component, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                hardware.add(component)

        return hardware

    def _extract_instructions(self, text: str) -> set[str]:
        """Extract 6502 assembly instructions from text."""
        instructions = set()

        # Common 6502 mnemonics
        mnemonics = [
            'LDA', 'STA', 'LDX', 'STX', 'LDY', 'STY',
            'JMP', 'JSR', 'RTS', 'RTI',
            'BEQ', 'BNE', 'BCC', 'BCS', 'BMI', 'BPL', 'BVC', 'BVS',
            'ADC', 'SBC', 'AND', 'ORA', 'EOR',
            'INC', 'DEC', 'INX', 'INY', 'DEX', 'DEY',
            'CMP', 'CPX', 'CPY',
            'ASL', 'LSR', 'ROL', 'ROR',
            'BIT', 'NOP',
            'CLC', 'SEC', 'CLI', 'SEI', 'CLD', 'SED', 'CLV',
            'PHA', 'PLA', 'PHP', 'PLP',
            'TAX', 'TAY', 'TXA', 'TYA', 'TSX', 'TXS'
        ]

        for mnemonic in mnemonics:
            # Look for mnemonic as whole word (not part of another word)
            pattern = r'\b' + mnemonic + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                instructions.add(mnemonic)

        return instructions

    def _extract_registers(self, text: str) -> set[str]:
        """Extract memory register addresses from text."""
        registers = set()

        # Find all 4-digit hex addresses with $ prefix
        # Common C64 ranges: $D000-$DFFF (I/O), $A000-$BFFF (BASIC ROM), $E000-$FFFF (KERNAL ROM)
        register_pattern = r'\$[0-9A-Fa-f]{4}'
        matches = re.findall(register_pattern, text)

        # Normalize to uppercase and add to set
        for match in matches:
            registers.add(match.upper())

        return registers

    def _extract_cross_references(self, chunks: list[DocumentChunk], doc_id: str) -> list[dict]:
        """
        Extract cross-references from document chunks.

        Returns list of cross-reference dictionaries with keys:
        - doc_id, chunk_id, ref_type, ref_value, context
        """
        cross_refs = []

        for chunk in chunks:
            text = chunk.content
            chunk_id = chunk.chunk_id

            # Extract memory addresses ($D000-$FFFF)
            addresses = self._extract_memory_addresses(text)
            for addr in addresses:
                # Get context (sentence containing the address)
                context = self._get_reference_context(text, addr)
                cross_refs.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'ref_type': 'memory_address',
                    'ref_value': addr,
                    'context': context
                })

            # Extract register offsets (VIC+0, SID+4, etc.)
            offsets = self._extract_register_offsets(text)
            for offset in offsets:
                context = self._get_reference_context(text, offset)
                cross_refs.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'ref_type': 'register_offset',
                    'ref_value': offset,
                    'context': context
                })

            # Extract page references ("see page 156")
            page_refs = self._extract_page_references(text)
            for page_ref in page_refs:
                context = self._get_reference_context(text, str(page_ref))
                cross_refs.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'ref_type': 'page_reference',
                    'ref_value': str(page_ref),
                    'context': context
                })

        return cross_refs

    def _extract_memory_addresses(self, text: str) -> set[str]:
        """Extract memory addresses like $D000, $D020, etc."""
        addresses = set()
        # Match $xxxx format (4-digit hex)
        pattern = r'\$[0-9A-Fa-f]{4}\b'
        matches = re.findall(pattern, text)
        for match in matches:
            addresses.add(match.upper())
        return addresses

    def _extract_register_offsets(self, text: str) -> set[str]:
        """Extract register offset references like VIC+0, SID+4, CIA1+0."""
        offsets = set()
        # Match patterns like: VIC+0, SID+4, CIA1+12, etc.
        pattern = r'\b(VIC|SID|CIA[12]?|PLA)\s*\+\s*(\d+)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for chip, offset in matches:
            offsets.add(f"{chip.upper()}+{offset}")
        return offsets

    def _extract_page_references(self, text: str) -> set[int]:
        """Extract page number references like 'see page 156', 'page 42'."""
        page_nums = set()
        # Match patterns like: "page 123", "see page 456", "on page 789"
        pattern = r'\b(?:see\s+)?page\s+(\d+)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        for page_num in matches:
            page_nums.add(int(page_num))
        return page_nums

    def _get_reference_context(self, text: str, reference: str, context_chars: int = 100) -> str:
        """Get surrounding context for a reference."""
        # Find the reference in text
        pos = text.find(reference)
        if pos == -1:
            # Try case-insensitive
            pos = text.lower().find(reference.lower())

        if pos == -1:
            return ""

        # Get surrounding context
        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(reference) + context_chars)

        context = text[start:end].strip()

        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def build_suggestion_dictionary(self, rebuild: bool = False):
        """
        Build autocomplete suggestion dictionary from all documents.

        Args:
            rebuild: If True, clear existing suggestions and rebuild from scratch
        """
        self.logger.info("Building query suggestion dictionary...")
        start_time = time.time()

        cursor = self.db_conn.cursor()

        # Clear existing if rebuilding
        if rebuild:
            cursor.execute("DELETE FROM query_suggestions")
            self.db_conn.commit()

        # Extract terms from all chunks
        from collections import defaultdict
        terms = defaultdict(int)

        chunks = self._get_chunks_db()
        for chunk in chunks:
            text = chunk.content

            # Extract technical terms (ALL CAPS, 2+ chars)
            tech_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)?\b', text)  # VIC-II, SID, CIA
            for term in tech_terms:
                terms[(term, 'hardware')] += 1

            # Extract memory addresses
            addresses = re.findall(r'\$[0-9A-Fa-f]{4}', text)
            for addr in addresses:
                terms[(addr.upper(), 'register')] += 1

            # Extract 6502 instructions
            instructions = re.findall(
                r'\b(?:LDA|STA|LDX|STX|LDY|STY|TAX|TAY|TXA|TYA|TSX|TXS|'
                r'ADC|SBC|AND|ORA|EOR|INC|DEC|INX|INY|DEX|DEY|'
                r'CMP|CPX|CPY|ASL|LSR|ROL|ROR|BIT|NOP|'
                r'JMP|JSR|RTS|RTI|BEQ|BNE|BCC|BCS|BMI|BPL|BVC|BVS|'
                r'CLC|SEC|CLI|SEI|CLD|SED|CLV|PHA|PLA|PHP|PLP)\b',
                text, re.IGNORECASE
            )
            for instr in instructions:
                terms[(instr.upper(), 'instruction')] += 1

            # Extract common technical phrases (2-3 words)
            # Look for capitalized phrases like "Sprite Multiplexing", "Sound Interface Device"
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            for phrase in phrases:
                if len(phrase) > 5:  # Avoid very short phrases
                    terms[(phrase, 'concept')] += 1

        # Store top N terms (limit to avoid bloat)
        top_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:2000]

        for (term, category), freq in top_terms:
            cursor.execute("""
                INSERT INTO query_suggestions (term, frequency, category)
                VALUES (?, ?, ?)
            """, (term, freq, category))

        self.db_conn.commit()

        elapsed = time.time() - start_time
        self.logger.info(f"Built suggestion dictionary with {len(top_terms)} terms in {elapsed:.2f}s")

    def get_query_suggestions(self, partial: str, max_suggestions: int = 5,
                             category: Optional[str] = None) -> list[dict]:
        """
        Get autocomplete suggestions for partial query.

        Args:
            partial: Partial query string (e.g., "VIC")
            max_suggestions: Maximum number of suggestions to return
            category: Optional category filter ('hardware', 'register', 'instruction', 'concept')

        Returns:
            List of suggestion dicts with 'term', 'frequency', and 'category'
        """
        if not partial or len(partial) < 2:
            return []

        cursor = self.db_conn.cursor()

        # Escape special FTS5 characters by quoting the query
        # FTS5 special chars: $ * " and others need to be quoted
        escaped_partial = f'"{partial}"*'

        # Use FTS5 prefix matching
        if category:
            cursor.execute("""
                SELECT term, frequency, category
                FROM query_suggestions
                WHERE term MATCH ? AND category = ?
                ORDER BY rank, frequency DESC
                LIMIT ?
            """, (escaped_partial, category, max_suggestions))
        else:
            cursor.execute("""
                SELECT term, frequency, category
                FROM query_suggestions
                WHERE term MATCH ?
                ORDER BY rank, frequency DESC
                LIMIT ?
            """, (escaped_partial, max_suggestions))

        results = []
        for row in cursor.fetchall():
            results.append({
                'term': row[0],
                'frequency': row[1],
                'category': row[2]
            })

        return results

    def _update_suggestions_for_chunks(self, chunks: list[DocumentChunk]):
        """
        Incrementally update query suggestions with terms from new chunks.

        Args:
            chunks: List of newly added chunks
        """
        from collections import defaultdict
        terms = defaultdict(int)

        # Extract terms from new chunks
        for chunk in chunks:
            text = chunk.content

            # Extract technical terms (ALL CAPS, 2+ chars)
            tech_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)?\b', text)
            for term in tech_terms:
                terms[(term, 'hardware')] += 1

            # Extract memory addresses
            addresses = re.findall(r'\$[0-9A-Fa-f]{4}', text)
            for addr in addresses:
                terms[(addr.upper(), 'register')] += 1

            # Extract 6502 instructions
            instructions = re.findall(
                r'\b(?:LDA|STA|LDX|STX|LDY|STY|TAX|TAY|TXA|TYA|TSX|TXS|'
                r'ADC|SBC|AND|ORA|EOR|INC|DEC|INX|INY|DEX|DEY|'
                r'CMP|CPX|CPY|ASL|LSR|ROL|ROR|BIT|NOP|'
                r'JMP|JSR|RTS|RTI|BEQ|BNE|BCC|BCS|BMI|BPL|BVC|BVS|'
                r'CLC|SEC|CLI|SEI|CLD|SED|CLV|PHA|PLA|PHP|PLP)\b',
                text, re.IGNORECASE
            )
            for instr in instructions:
                terms[(instr.upper(), 'instruction')] += 1

            # Extract common technical phrases (2-3 words)
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            for phrase in phrases:
                if len(phrase) > 5:
                    terms[(phrase, 'concept')] += 1

        if not terms:
            return

        cursor = self.db_conn.cursor()

        # Update or insert terms (upsert logic)
        for (term, category), freq in terms.items():
            # Check if term already exists
            cursor.execute("""
                SELECT frequency FROM query_suggestions
                WHERE term = ? AND category = ?
            """, (term, category))

            existing = cursor.fetchone()
            if existing:
                # Update frequency
                new_freq = existing[0] + freq
                cursor.execute("""
                    DELETE FROM query_suggestions WHERE term = ? AND category = ?
                """, (term, category))
                cursor.execute("""
                    INSERT INTO query_suggestions (term, frequency, category)
                    VALUES (?, ?, ?)
                """, (term, new_freq, category))
            else:
                # Insert new term
                cursor.execute("""
                    INSERT INTO query_suggestions (term, frequency, category)
                    VALUES (?, ?, ?)
                """, (term, freq, category))

        self.db_conn.commit()
        self.logger.debug(f"Updated suggestion dictionary with {len(terms)} terms")

    def export_search_results(self, results: list[dict], format: str = 'markdown',
                             query: Optional[str] = None) -> str:
        """
        Export search results to various formats.

        Args:
            results: List of search result dicts
            format: Output format ('markdown', 'json', 'html')
            query: Optional query string to include in export

        Returns:
            Formatted string in requested format
        """
        if format == 'markdown':
            return self._export_markdown(results, query)
        elif format == 'json':
            return self._export_json(results, query)
        elif format == 'html':
            return self._export_html(results, query)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'markdown', 'json', or 'html'.")

    def _export_markdown(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as Markdown."""
        output = "# Search Results\n\n"

        if query:
            output += f"**Query:** {query}\n"
        output += f"**Results:** {len(results)}\n"
        output += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        output += "---\n\n"

        for i, result in enumerate(results, 1):
            output += f"## {i}. {result.get('title', 'Untitled')}\n\n"

            # Add score if present
            if 'score' in result:
                output += f"**Score:** {result['score']:.3f}\n"
            elif 'similarity' in result:
                output += f"**Similarity:** {result['similarity']:.3f}\n"

            # Add metadata
            if 'filename' in result:
                output += f"**File:** {result['filename']}\n"
            if 'page' in result and result['page']:
                output += f"**Page:** {result['page']}\n"
            if 'doc_id' in result:
                output += f"**Doc ID:** {result['doc_id']}\n"
            if 'chunk_id' in result:
                output += f"**Chunk:** {result['chunk_id']}\n"

            output += "\n"

            # Add snippet/content
            if 'snippet' in result:
                output += f"### Excerpt\n\n{result['snippet']}\n\n"
            elif 'context' in result:
                output += f"### Context\n\n{result['context']}\n\n"

            # Add tags if present
            if 'tags' in result and result['tags']:
                tags = result['tags'] if isinstance(result['tags'], list) else []
                if tags:
                    output += f"**Tags:** {', '.join(tags)}\n\n"

            output += "---\n\n"

        return output

    def _export_json(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as JSON."""
        export_data = {
            'query': query,
            'result_count': len(results),
            'generated_at': datetime.now().isoformat(),
            'results': results
        }
        return json.dumps(export_data, indent=2)

    def _export_html(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as HTML."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .meta { color: #7f8c8d; margin-bottom: 20px; }
        .result {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .result h2 { color: #2c3e50; margin-top: 0; }
        .result-meta { color: #7f8c8d; font-size: 0.9em; margin: 10px 0; }
        .snippet {
            background: white;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .tags { margin-top: 10px; }
        .tag {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <h1>🔍 Search Results</h1>
"""

        if query:
            html += f"    <div class='meta'><strong>Query:</strong> {query}</div>\n"
        html += f"    <div class='meta'><strong>Results:</strong> {len(results)}</div>\n"
        html += f"    <div class='meta'><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>\n\n"

        for i, result in enumerate(results, 1):
            html += f"    <div class='result'>\n"
            html += f"        <h2>{i}. {result.get('title', 'Untitled')}</h2>\n"

            html += "        <div class='result-meta'>\n"
            if 'score' in result:
                html += f"            <strong>Score:</strong> {result['score']:.3f}<br>\n"
            elif 'similarity' in result:
                html += f"            <strong>Similarity:</strong> {result['similarity']:.3f}<br>\n"
            if 'filename' in result:
                html += f"            <strong>File:</strong> {result['filename']}<br>\n"
            if 'page' in result and result['page']:
                html += f"            <strong>Page:</strong> {result['page']}<br>\n"
            html += "        </div>\n"

            if 'snippet' in result:
                html += f"        <div class='snippet'>{result['snippet']}</div>\n"
            elif 'context' in result:
                html += f"        <div class='snippet'>{result['context']}</div>\n"

            if 'tags' in result and result['tags']:
                tags = result['tags'] if isinstance(result['tags'], list) else []
                if tags:
                    html += "        <div class='tags'>\n"
                    for tag in tags:
                        html += f"            <span class='tag'>{tag}</span>\n"
                    html += "        </div>\n"

            html += "    </div>\n\n"

        html += """</body>
</html>"""

        return html

    def _cache_key(self, method: str, **kwargs) -> str:
        """Generate a cache key from method name and arguments."""
        # Sort kwargs for consistent hashing
        sorted_items = sorted(kwargs.items())
        key_str = f"{method}:{sorted_items}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _invalidate_caches(self):
        """Clear all caches when data changes."""
        if self._search_cache is not None:
            self._search_cache.clear()
        if self._similar_cache is not None:
            self._similar_cache.clear()
        self.logger.info("Search caches invalidated")

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file content."""
        md5_hash = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read file in chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _find_mdscrape_executable(self) -> Optional[str]:
        """Find mdscrape executable in common locations.

        Returns:
            Path to mdscrape executable, or None if not found
        """
        import shutil

        # Check if mdscrape is in PATH
        mdscrape = shutil.which('mdscrape')
        if mdscrape:
            self.logger.info(f"Found mdscrape in PATH: {mdscrape}")
            return mdscrape

        # Check common Windows/Linux paths
        common_paths = [
            Path(r'C:\Users\mit\claude\mdscrape\mdscrape.exe'),  # User-specified location
            Path(r'C:\Users\mit\claude\mdscrape\mdscrape'),
            Path.home() / 'claude' / 'mdscrape' / 'mdscrape.exe',
            Path.home() / 'claude' / 'mdscrape' / 'mdscrape',
            Path(__file__).parent.parent / 'mdscrape' / 'mdscrape.exe',
            Path(__file__).parent.parent / 'mdscrape' / 'mdscrape',
        ]

        for path in common_paths:
            if path.exists():
                self.logger.info(f"Found mdscrape at: {path}")
                return str(path)

        # Check MDSCRAPE_PATH environment variable
        env_path = os.environ.get('MDSCRAPE_PATH')
        if env_path:
            path = Path(env_path)
            if path.exists():
                self.logger.info(f"Found mdscrape via MDSCRAPE_PATH: {path}")
                return str(path)

        self.logger.warning("mdscrape executable not found. Install from: https://github.com/MichaelTroelsen/mdscrape")
        return None

    def _extract_source_url_from_md(self, md_file: Path) -> Optional[str]:
        """Extract source URL from YAML frontmatter in markdown file.

        Args:
            md_file: Path to markdown file

        Returns:
            Source URL if found, None otherwise
        """
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse YAML frontmatter (between --- delimiters)
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    # Simple YAML parsing for 'source:' or 'url:' field
                    for line in frontmatter.split('\n'):
                        line = line.strip()
                        if line.startswith('source:') or line.startswith('url:'):
                            # Extract URL after colon
                            url = line.split(':', 1)[1].strip().strip('"\'')
                            if url:
                                return url
        except Exception as e:
            self.logger.warning(f"Failed to extract URL from {md_file}: {e}")

        return None

    def _add_scraped_document(self, filepath: str, source_url: str, title: Optional[str],
                              tags: Optional[list[str]], scrape_config: str,
                              scrape_date: str) -> DocumentMeta:
        """Add a scraped markdown document with URL metadata.

        Args:
            filepath: Path to scraped markdown file
            source_url: Original URL that was scraped
            title: Optional title for document
            tags: Optional list of tags
            scrape_config: JSON string with scraping configuration
            scrape_date: ISO timestamp of scrape

        Returns:
            DocumentMeta object for added document
        """
        # First, add document using normal flow
        doc = self.add_document(filepath, title, tags)

        # Compute content hash for change detection
        url_content_hash = self._compute_file_hash(filepath)

        # Update database with URL metadata
        with self._lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?,
                    scrape_date = ?,
                    scrape_config = ?,
                    scrape_status = 'success',
                    url_content_hash = ?
                WHERE doc_id = ?
            """, (source_url, scrape_date, scrape_config, url_content_hash, doc.doc_id))

            self.db_conn.commit()

        # Update in-memory object
        doc.source_url = source_url
        doc.scrape_date = scrape_date
        doc.scrape_config = scrape_config
        doc.scrape_status = 'success'
        doc.url_content_hash = url_content_hash

        # Update in documents dict
        self.documents[doc.doc_id] = doc

        self.logger.info(f"Added scraped document: {doc.title} (from {source_url})")
        return doc

    def _is_path_allowed(self, filepath: str) -> bool:
        """
        Check if a file path is within allowed directories.

        Args:
            filepath: Path to check

        Returns:
            True if path is allowed (or no restrictions configured), False otherwise
        """
        # No restrictions if allowed_dirs not configured
        if not self.allowed_dirs:
            return True

        # Resolve to absolute path to prevent path traversal
        try:
            resolved_path = Path(filepath).resolve()
        except (OSError, ValueError):
            # Invalid path
            return False

        # Check if path is within any allowed directory
        return any(
            resolved_path.is_relative_to(allowed_dir)
            for allowed_dir in self.allowed_dirs
        )

    def add_document(self, filepath: str, title: Optional[str] = None, tags: Optional[list[str]] = None,
                     progress_callback: ProgressCallback = None) -> DocumentMeta:
        """Add a document to the knowledge base.

        Args:
            filepath: Path to the document file
            title: Optional title for the document
            tags: Optional list of tags
            progress_callback: Optional callback for progress updates
        """
        # Report progress: Start
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=0,
                total=4,
                message="Starting document ingestion",
                item=filepath
            ))

        # Resolve to absolute path to prevent path traversal
        resolved_path = Path(filepath).resolve()

        # Security: Validate path is within allowed directories
        if not self._is_path_allowed(filepath):
            self.logger.error(f"Security violation: Path outside allowed directories: {resolved_path}")
            raise SecurityError(
                f"Path outside allowed directories. File must be within: {self.allowed_dirs}"
            )

        filepath = str(resolved_path)
        self.logger.info(f"Adding document: {filepath}")

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise DocumentNotFoundError(f"File not found: {filepath}")

        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()

        # Extract text based on file type
        total_pages = None
        pdf_metadata = {}
        try:
            if file_ext == '.pdf':
                text, total_pages, pdf_metadata = self._extract_pdf_text(filepath)
                file_type = 'pdf'
                self.logger.info(f"Extracted {total_pages} pages from PDF")
            elif file_ext in ['.xlsx', '.xls']:
                text, sheet_count = self._extract_excel_file(filepath)
                file_type = 'excel'
                total_pages = sheet_count  # Treat sheets as "pages"
                self.logger.info(f"Extracted Excel file with {sheet_count} sheets ({len(text)} characters)")
            elif file_ext in ['.html', '.htm']:
                text = self._extract_html_file(filepath)
                file_type = 'html'
                self.logger.info(f"Extracted HTML file ({len(text)} characters)")
            elif file_ext in ['.txt', '.md', '.asm', '.bas', '.inc', '.s']:
                text = self._extract_text_file(filepath)
                file_type = 'text'
                self.logger.info(f"Extracted text file ({len(text)} characters)")
            else:
                raise UnsupportedFileTypeError(f"Unsupported file type: {file_ext}")
        except (UnsupportedFileTypeError, DocumentNotFoundError):
            raise
        except Exception as e:
            self.logger.error(f"Error extracting {filepath}: {e}")
            raise KnowledgeBaseError(f"Error extracting document: {e}")

        # Report progress: Text extraction complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=1,
                total=4,
                message=f"Text extraction complete ({len(text)} characters)",
                item=filename
            ))

        # Extract tables from PDFs
        tables = []
        if file_type == 'pdf':
            tables = self._extract_tables(filepath)
            if tables:
                self.logger.info(f"Extracted {len(tables)} tables from PDF")

        # Detect code blocks in text
        code_blocks = self._detect_code_blocks(text)
        if code_blocks:
            self.logger.info(f"Detected {len(code_blocks)} code blocks")

        # Extract facets for faceted search
        facets = self._extract_facets(text)
        facet_count = sum(len(values) for values in facets.values())
        if facet_count > 0:
            self.logger.info(f"Extracted {facet_count} facets ({len(facets['hardware'])} hardware, {len(facets['instruction'])} instructions, {len(facets['register'])} registers)")

        # Generate content-based doc_id for deduplication
        doc_id = self._generate_doc_id(filepath, text)

        # Thread-safe duplicate check
        with self._lock:
            # Check for duplicate content
            if doc_id in self.documents:
                existing_doc = self.documents[doc_id]
                self.logger.warning(f"Duplicate content detected: {filepath}")
                self.logger.warning(f"  Matches existing document: {existing_doc.filepath}")
                self.logger.info(f"Skipping duplicate - returning existing document {doc_id}")
                return existing_doc
        
        # Create chunks
        text_chunks = self._chunk_text(text)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Estimate page number for PDFs based on PAGE BREAK markers
            page_num = None
            if file_type == 'pdf' and '--- PAGE BREAK ---' in text:
                # Count PAGE BREAK markers before this chunk
                chunk_start_pos = text.find(chunk_text[:100])  # Find chunk in full text
                if chunk_start_pos >= 0:
                    page_breaks_before = text[:chunk_start_pos].count('--- PAGE BREAK ---')
                    page_num = page_breaks_before + 1  # Pages are 1-indexed

            chunk = DocumentChunk(
                doc_id=doc_id,
                filename=filename,
                title=title or filename,
                chunk_id=i,
                page=page_num,
                content=chunk_text,
                word_count=len(chunk_text.split())
            )
            chunks.append(chunk)

        # Report progress: Chunking complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=2,
                total=4,
                message=f"Created {len(chunks)} chunks",
                item=filename
            ))

        # Compute file modification time and content hash for update detection
        file_mtime = os.path.getmtime(resolved_path)
        file_hash = self._compute_file_hash(resolved_path)

        # Extract cross-references for content linking
        cross_refs = self._extract_cross_references(chunks, doc_id)
        if cross_refs:
            self.logger.info(f"Extracted {len(cross_refs)} cross-references")

        # Create metadata
        doc_meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            title=title or filename,
            filepath=filepath,
            file_type=file_type,
            total_pages=total_pages,
            total_chunks=len(chunks),
            indexed_at=datetime.now().isoformat(),
            tags=tags or [],
            author=pdf_metadata.get('author'),
            subject=pdf_metadata.get('subject'),
            creator=pdf_metadata.get('creator'),
            creation_date=pdf_metadata.get('creation_date'),
            file_mtime=file_mtime,
            file_hash=file_hash
        )

        # Thread-safe database insertion and cache invalidation
        with self._lock:
            # Add to database (with tables, code blocks, facets, and cross-references)
            self._add_document_db(doc_meta, chunks, tables=tables, code_blocks=code_blocks, facets=facets, cross_refs=cross_refs)
            self.documents[doc_id] = doc_meta

            # Report progress: Database insertion complete
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation="add_document",
                    current=3,
                    total=4,
                    message="Stored in database",
                    item=filename
                ))

            # Invalidate BM25 index (will be rebuilt on next search)
            self.bm25 = None

            # Incrementally add chunks to embeddings (faster than full rebuild)
            if self.use_semantic:
                self._add_chunks_to_embeddings(chunks)

            # Update query suggestions with new terms
            self._update_suggestions_for_chunks(chunks)

            # Invalidate search caches
            self._invalidate_caches()

        self.logger.info(f"Successfully indexed document {doc_id}: {filename} ({len(chunks)} chunks)")

        # Report progress: Complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=4,
                total=4,
                message="Document indexed successfully",
                item=filename
            ))

        return doc_meta

    def scrape_url(self, url: str, title: Optional[str] = None, tags: Optional[list[str]] = None,
                   depth: int = 50, limit: Optional[str] = None, threads: int = 10,
                   delay: int = 100, selector: Optional[str] = None,
                   progress_callback: ProgressCallback = None) -> dict:
        """Scrape a URL using mdscrape and add resulting documents to knowledge base.

        Args:
            url: Starting URL to scrape
            title: Optional base title for scraped documents
            tags: Optional list of tags (domain name auto-added)
            depth: Maximum crawl depth (default: 50)
            limit: Limit scraping to URLs with this prefix (default: url)
            threads: Number of concurrent threads (default: 10)
            delay: Delay between requests in ms (default: 100)
            selector: CSS selector for main content (optional)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with scraping results:
            {
                'status': 'success' | 'partial' | 'failed',
                'url': original_url,
                'output_dir': path_to_scraped_files,
                'files_scraped': count,
                'docs_added': count,
                'docs_updated': count,
                'docs_failed': count,
                'error': error_message (if failed),
                'doc_ids': [list of added doc_ids]
            }
        """
        import subprocess
        from urllib.parse import urlparse
        from datetime import datetime

        # 1. Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(f"Only HTTP/HTTPS URLs supported: {url}")
        except Exception as e:
            return {
                'status': 'failed',
                'url': url,
                'error': f"Invalid URL: {str(e)}"
            }

        # 2. Extract domain for auto-tagging
        domain = parsed.netloc.replace('www.', '')
        if tags is None:
            tags = []
        tags = list(tags) + [domain, 'scraped']

        # 3. Setup output directory in scraped_docs
        scraped_base = self.data_dir / "scraped_docs"
        scraped_base.mkdir(exist_ok=True)

        # Use domain + timestamp for unique output dir
        safe_domain = domain.replace('.', '_').replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = scraped_base / f"{safe_domain}_{timestamp}"

        # 4. Build mdscrape command
        mdscrape_path = self._find_mdscrape_executable()
        if not mdscrape_path:
            return {
                'status': 'failed',
                'url': url,
                'error': 'mdscrape executable not found. Set MDSCRAPE_PATH or install from: https://github.com/MichaelTroelsen/mdscrape'
            }

        cmd = [
            mdscrape_path,
            url,
            '--output', str(output_dir),
            '--depth', str(depth),
            '--threads', str(threads),
            '--delay', str(delay)
        ]

        if limit:
            cmd.extend(['--limit', limit])
        if selector:
            cmd.extend(['--selector', selector])

        # 5. Store scrape config
        scrape_config = {
            'url': url,
            'depth': depth,
            'limit': limit,
            'threads': threads,
            'delay': delay,
            'selector': selector,
            'timestamp': datetime.now().isoformat()
        }
        scrape_config_json = json.dumps(scrape_config)

        # 6. Execute mdscrape
        self.logger.info(f"Scraping URL: {url}")
        self.logger.info(f"Command: {' '.join(cmd)}")

        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="scrape_url",
                current=0,
                total=100,
                message="Starting web scraping",
                item=url
            ))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                self.logger.error(f"Scraping failed: {error_msg}")
                return {
                    'status': 'failed',
                    'url': url,
                    'error': f"mdscrape failed: {error_msg}"
                }

            self.logger.info(f"Scraping completed successfully")

        except subprocess.TimeoutExpired:
            self.logger.error("Scraping timeout (>1 hour)")
            return {
                'status': 'failed',
                'url': url,
                'error': 'Scraping timeout (>1 hour)'
            }
        except Exception as e:
            self.logger.error(f"Scraping error: {e}")
            return {
                'status': 'failed',
                'url': url,
                'error': f"Scraping error: {str(e)}"
            }

        # 7. Find all generated markdown files
        if not output_dir.exists():
            return {
                'status': 'failed',
                'url': url,
                'error': f"Output directory not created: {output_dir}"
            }

        md_files = list(output_dir.rglob('*.md'))

        if not md_files:
            return {
                'status': 'failed',
                'url': url,
                'error': f"No markdown files generated in {output_dir}"
            }

        self.logger.info(f"Found {len(md_files)} markdown files to process")

        # 8. Add each file to knowledge base
        added_docs = []
        failed_docs = []
        scrape_date = datetime.now().isoformat()

        for i, md_file in enumerate(md_files):
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation="scrape_url",
                    current=i,
                    total=len(md_files),
                    message="Adding scraped document",
                    item=md_file.name
                ))

            try:
                # Extract source URL from frontmatter
                source_url_for_file = self._extract_source_url_from_md(md_file)
                if not source_url_for_file:
                    source_url_for_file = url  # Fallback to base URL

                # Generate title from domain + page path
                if title:
                    # Use provided base title
                    doc_title = title
                else:
                    # Extract page name from URL path
                    parsed_source = urlparse(source_url_for_file)
                    page_path = parsed_source.path.strip('/')

                    # If it's the index/root, use domain name
                    if not page_path or page_path.lower() in ['index', 'index.html', 'index.htm']:
                        page_name = "Home"
                    else:
                        # Use the last part of the path as page name
                        page_name = page_path.split('/')[-1]
                        # Remove file extensions
                        page_name = page_name.replace('.html', '').replace('.htm', '').replace('.php', '')
                        # Clean up formatting
                        page_name = page_name.replace('_', ' ').replace('-', ' ').title()

                    # Combine domain + page name
                    doc_title = f"{domain} - {page_name}"

                # Add document with URL metadata
                doc = self._add_scraped_document(
                    filepath=str(md_file),
                    source_url=source_url_for_file,
                    title=doc_title,
                    tags=tags,
                    scrape_config=scrape_config_json,
                    scrape_date=scrape_date
                )
                added_docs.append(doc.doc_id)
                self.logger.info(f"Added: {doc.title} ({doc.doc_id})")

            except Exception as e:
                self.logger.error(f"Failed to add {md_file}: {e}")
                failed_docs.append(str(md_file))

        # 9. Return results
        status = 'success' if not failed_docs else ('partial' if added_docs else 'failed')

        result_dict = {
            'status': status,
            'url': url,
            'output_dir': str(output_dir),
            'files_scraped': len(md_files),
            'docs_added': len(added_docs),
            'docs_updated': 0,
            'docs_failed': len(failed_docs),
            'doc_ids': added_docs
        }

        if failed_docs:
            result_dict['error'] = f"{len(failed_docs)} files failed to add"

        self.logger.info(f"Scraping complete: {status} - Added {len(added_docs)}/{len(md_files)} documents")

        return result_dict

    def rescrape_document(self, doc_id: str, progress_callback: ProgressCallback = None) -> dict:
        """Re-scrape an existing URL-sourced document.

        Args:
            doc_id: Document ID to re-scrape
            progress_callback: Optional progress callback

        Returns:
            Dictionary with re-scrape results (same format as scrape_url)
        """
        # Get document metadata
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Check if document has source URL
        if not doc.source_url:
            raise ValueError(f"Document is not URL-sourced: {doc_id}")

        self.logger.info(f"Re-scraping document: {doc.title} (from {doc.source_url})")

        # Parse original scrape config
        scrape_config = {}
        if doc.scrape_config:
            try:
                scrape_config = json.loads(doc.scrape_config)
            except Exception as e:
                self.logger.warning(f"Failed to parse scrape config: {e}")

        # Remove old document
        self.logger.info(f"Removing old document version: {doc_id}")
        self.remove_document(doc_id)

        # Re-scrape with original config
        result = self.scrape_url(
            url=doc.source_url,
            title=doc.title,
            tags=doc.tags,
            depth=scrape_config.get('depth', 50),
            limit=scrape_config.get('limit'),
            threads=scrape_config.get('threads', 10),
            delay=scrape_config.get('delay', 100),
            selector=scrape_config.get('selector'),
            progress_callback=progress_callback
        )

        # Add rescrape metadata to result
        result['rescrape'] = True
        result['old_doc_id'] = doc_id

        self.logger.info(f"Re-scrape complete: {result['status']}")
        return result

    def check_url_updates(self, auto_rescrape: bool = False) -> dict:
        """Check all URL-sourced documents for updates.

        Args:
            auto_rescrape: If True, automatically re-scrape changed URLs

        Returns:
            Dictionary with lists of unchanged, changed, and failed URLs:
            {
                'unchanged': [list of docs with no changes],
                'changed': [list of docs with updates available],
                'failed': [list of docs where check failed],
                'rescraped': [list of doc_ids that were re-scraped]
            }
        """
        from datetime import datetime

        results = {
            'unchanged': [],
            'changed': [],
            'failed': [],
            'rescraped': []
        }

        # Find all URL-sourced documents
        url_docs = [doc for doc in self.documents.values() if doc.source_url]

        if not url_docs:
            self.logger.info("No URL-sourced documents to check")
            return results

        self.logger.info(f"Checking {len(url_docs)} URL-sourced documents for updates")

        for doc in url_docs:
            try:
                import requests

                # Try HEAD request first (faster)
                response = requests.head(doc.source_url, timeout=10, allow_redirects=True)

                # Update last_checked timestamp
                with self._lock:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        UPDATE documents
                        SET url_last_checked = ?
                        WHERE doc_id = ?
                    """, (datetime.now().isoformat(), doc.doc_id))
                    self.db_conn.commit()

                # Check Last-Modified header if available
                if 'Last-Modified' in response.headers:
                    from email.utils import parsedate_to_datetime
                    last_modified = parsedate_to_datetime(response.headers['Last-Modified'])

                    if doc.scrape_date:
                        scrape_dt = datetime.fromisoformat(doc.scrape_date)
                        if last_modified > scrape_dt:
                            self.logger.info(f"Update available: {doc.title} ({doc.source_url})")
                            results['changed'].append({
                                'doc_id': doc.doc_id,
                                'title': doc.title,
                                'url': doc.source_url,
                                'last_modified': last_modified.isoformat(),
                                'scraped_date': doc.scrape_date
                            })

                            # Auto-rescrape if requested
                            if auto_rescrape:
                                self.logger.info(f"Auto-rescaping: {doc.title}")
                                try:
                                    rescrape_result = self.rescrape_document(doc.doc_id)
                                    if rescrape_result['status'] == 'success':
                                        results['rescraped'].append(doc.doc_id)
                                except Exception as e:
                                    self.logger.error(f"Auto-rescrape failed: {e}")

                            continue

                # No change detected
                results['unchanged'].append({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'url': doc.source_url
                })

            except Exception as e:
                self.logger.error(f"Failed to check {doc.source_url}: {e}")
                results['failed'].append({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'url': doc.source_url,
                    'error': str(e)
                })

        self.logger.info(f"Update check complete: {len(results['unchanged'])} unchanged, "
                        f"{len(results['changed'])} changed, {len(results['failed'])} failed")

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        self.logger.info(f"Removing document: {doc_id}")

        if doc_id not in self.documents:
            self.logger.warning(f"Document not found for removal: {doc_id}")
            return False

        filename = self.documents[doc_id].filename

        # Remove from database (chunks cascade automatically)
        success = self._remove_document_db(doc_id)

        if success:
            # Remove from in-memory index
            del self.documents[doc_id]

            # Invalidate BM25 index (will be rebuilt on next search)
            self.bm25 = None

            # Invalidate embeddings (will be rebuilt on next semantic search)
            if self.use_semantic:
                self.embeddings_index = None
                self.embeddings_doc_map = []

            # Invalidate search caches
            self._invalidate_caches()

            self.logger.info(f"Successfully removed document {doc_id}: {filename}")

        return success

    def needs_reindex(self, filepath: str, doc_id: str) -> bool:
        """
        Check if a document needs re-indexing based on file modification time and content hash.

        Args:
            filepath: Path to the document file
            doc_id: Document ID to check

        Returns:
            True if the document needs re-indexing, False otherwise
        """
        doc = self.documents.get(doc_id)
        if not doc:
            return True  # Document doesn't exist, needs indexing

        # If no mtime/hash stored, can't check - assume needs reindex
        if doc.file_mtime is None or doc.file_hash is None:
            self.logger.info(f"Document {doc_id} has no update detection data, assuming needs reindex")
            return True

        # Quick check: modification time
        try:
            current_mtime = os.path.getmtime(filepath)
            if current_mtime <= doc.file_mtime:
                # File hasn't been modified since last index
                return False
        except OSError:
            # File doesn't exist or can't be accessed
            self.logger.warning(f"Cannot access file: {filepath}")
            return False

        # File was modified - do deep check with content hash
        try:
            current_hash = self._compute_file_hash(filepath)
            if current_hash == doc.file_hash:
                # Content is same despite mtime change (e.g., touched)
                self.logger.info(f"File mtime changed but content unchanged: {filepath}")
                return False
            else:
                # Content has actually changed
                self.logger.info(f"File content changed: {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Error computing hash for {filepath}: {e}")
            return False

    def update_document(self, filepath: str, title: Optional[str] = None, tags: Optional[list[str]] = None) -> DocumentMeta:
        """
        Update an existing document if it has changed, or add it if it doesn't exist.

        Args:
            filepath: Path to the document file
            title: Optional title (if not provided, uses filename)
            tags: Optional list of tags

        Returns:
            DocumentMeta for the document (existing or newly indexed)
        """
        # Find existing doc by filepath
        existing_doc = None
        for doc in self.documents.values():
            if doc.filepath == filepath:
                existing_doc = doc
                break

        if not existing_doc:
            # Document doesn't exist, add it
            self.logger.info(f"Document not found, adding: {filepath}")
            return self.add_document(filepath, title, tags)

        if not self.needs_reindex(filepath, existing_doc.doc_id):
            # Document unchanged
            self.logger.info(f"Document unchanged, skipping reindex: {filepath}")
            return existing_doc

        # Document has changed, re-index it
        self.logger.info(f"Document changed, re-indexing: {filepath}")
        self.remove_document(existing_doc.doc_id)
        return self.add_document(filepath, title, tags)

    def check_all_updates(self, auto_update: bool = False) -> dict:
        """
        Check all indexed documents for updates.

        Args:
            auto_update: If True, automatically re-index changed documents

        Returns:
            Dictionary with lists of unchanged, changed, and missing documents
        """
        results = {
            'unchanged': [],
            'changed': [],
            'missing': [],
            'updated': []  # Only populated if auto_update=True
        }

        for doc_id, doc in list(self.documents.items()):
            filepath = doc.filepath

            # Check if file still exists
            if not os.path.exists(filepath):
                results['missing'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })
                continue

            # Check if needs reindex
            if self.needs_reindex(filepath, doc_id):
                results['changed'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })

                if auto_update:
                    try:
                        updated_doc = self.update_document(filepath, doc.title, doc.tags)
                        results['updated'].append({
                            'doc_id': updated_doc.doc_id,
                            'filepath': filepath,
                            'title': updated_doc.title
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to update {filepath}: {e}")
            else:
                results['unchanged'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })

        return results

    def add_documents_bulk(self, directory: str, pattern: str = "**/*.{pdf,txt,md,html,htm,xlsx,xls}",
                           tags: Optional[list[str]] = None, recursive: bool = True,
                           skip_duplicates: bool = True, progress_callback: ProgressCallback = None) -> dict:
        """
        Add multiple documents from a directory matching a glob pattern.

        Args:
            directory: Directory to search for documents
            pattern: Glob pattern (default: **/*.{pdf,txt})
            tags: Tags to apply to all documents
            recursive: Search subdirectories (default: True)
            skip_duplicates: Skip files with duplicate content (default: True)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with lists of added, skipped, and failed documents
        """
        from pathlib import Path

        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        # Find matching files
        if recursive:
            files = list(dir_path.glob(pattern))
        else:
            # Non-recursive: remove ** from pattern
            non_recursive_pattern = pattern.replace('**/', '')
            files = list(dir_path.glob(non_recursive_pattern))

        results = {
            'added': [],
            'skipped': [],
            'failed': []
        }

        self.logger.info(f"Bulk add: found {len(files)} files matching pattern '{pattern}' in {directory}")

        # Report progress: Start
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_documents_bulk",
                current=0,
                total=len(files),
                message=f"Starting bulk add of {len(files)} files",
                item=directory
            ))

        # Get worker count (configurable via environment variable, default to CPU count)
        max_workers = int(os.getenv('PARALLEL_WORKERS', str(os.cpu_count() or 4)))
        self.logger.info(f"Using {max_workers} workers for parallel processing")

        # Process files in parallel using ThreadPoolExecutor
        def process_file(file_path):
            """Process a single file and return result."""
            if not file_path.is_file():
                return None

            try:
                # Generate title from filename
                title = file_path.stem
                doc = self.add_document(str(file_path), title=title, tags=tags)

                return {
                    'status': 'added',
                    'doc_id': doc.doc_id,
                    'filepath': str(file_path),
                    'title': title,
                    'chunks': doc.total_chunks
                }

            except Exception as e:
                return {
                    'status': 'failed',
                    'filepath': str(file_path),
                    'error': str(e)
                }

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(process_file, fp): fp for fp in files}

            # Process completed tasks as they finish
            completed = 0
            seen_doc_ids = set()

            for future in as_completed(future_to_file):
                completed += 1
                file_path = future_to_file[future]

                # Report progress: Processing file
                if progress_callback:
                    progress_callback(ProgressUpdate(
                        operation="add_documents_bulk",
                        current=completed,
                        total=len(files),
                        message=f"Processing file {completed}/{len(files)}",
                        item=str(file_path.name)
                    ))

                try:
                    result = future.result()
                    if result is None:
                        continue

                    if result['status'] == 'added':
                        # Check for duplicates
                        if skip_duplicates and result['doc_id'] in seen_doc_ids:
                            results['skipped'].append({
                                'filepath': result['filepath'],
                                'reason': 'duplicate content',
                                'doc_id': result['doc_id']
                            })
                        else:
                            seen_doc_ids.add(result['doc_id'])
                            results['added'].append({
                                'doc_id': result['doc_id'],
                                'filepath': result['filepath'],
                                'title': result['title'],
                                'chunks': result['chunks']
                            })
                    elif result['status'] == 'failed':
                        results['failed'].append({
                            'filepath': result['filepath'],
                            'error': result['error']
                        })
                        self.logger.error(f"Failed to add {result['filepath']}: {result['error']}")

                except Exception as e:
                    results['failed'].append({
                        'filepath': str(file_path),
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to process {file_path}: {e}")

        self.logger.info(f"Bulk add complete: {len(results['added'])} added, "
                        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")

        # Report progress: Complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_documents_bulk",
                current=len(files),
                total=len(files),
                message=f"Bulk add complete: {len(results['added'])} added, "
                        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
            ))

        return results

    def remove_documents_bulk(self, doc_ids: Optional[list[str]] = None,
                              tags: Optional[list[str]] = None) -> dict:
        """
        Remove multiple documents by doc IDs or tags.

        Args:
            doc_ids: List of document IDs to remove
            tags: Remove all documents with any of these tags

        Returns:
            Dictionary with lists of removed and failed document IDs
        """
        if not doc_ids and not tags:
            raise ValueError("Must provide either doc_ids or tags")

        results = {
            'removed': [],
            'failed': []
        }

        # Collect doc_ids to remove
        ids_to_remove = set()

        if doc_ids:
            ids_to_remove.update(doc_ids)

        if tags:
            # Find all documents with any of the specified tags
            for doc_id, doc in self.documents.items():
                if any(tag in doc.tags for tag in tags):
                    ids_to_remove.add(doc_id)

        self.logger.info(f"Bulk remove: removing {len(ids_to_remove)} documents")

        for doc_id in ids_to_remove:
            try:
                if self.remove_document(doc_id):
                    results['removed'].append(doc_id)
                else:
                    results['failed'].append({
                        'doc_id': doc_id,
                        'error': 'Document not found'
                    })
            except Exception as e:
                results['failed'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })
                self.logger.error(f"Failed to remove {doc_id}: {e}")

        self.logger.info(f"Bulk remove complete: {len(results['removed'])} removed, "
                        f"{len(results['failed'])} failed")

        return results

    def update_tags_bulk(self, doc_ids: Optional[list[str]] = None,
                         existing_tags: Optional[list[str]] = None,
                         add_tags: Optional[list[str]] = None,
                         remove_tags: Optional[list[str]] = None,
                         replace_tags: Optional[list[str]] = None) -> dict:
        """
        Update tags for multiple documents in bulk.

        Args:
            doc_ids: List of document IDs to update (if None, uses existing_tags to find docs)
            existing_tags: Find documents with any of these tags (alternative to doc_ids)
            add_tags: Tags to add to the documents
            remove_tags: Tags to remove from the documents
            replace_tags: Replace all tags with these tags

        Returns:
            Dictionary with lists of updated and failed document IDs

        Examples:
            # Add 'assembly' tag to specific documents
            kb.update_tags_bulk(doc_ids=['doc1', 'doc2'], add_tags=['assembly'])

            # Remove 'draft' tag from all documents that have it
            kb.update_tags_bulk(existing_tags=['draft'], remove_tags=['draft'])

            # Replace all tags with 'archive' for specific documents
            kb.update_tags_bulk(doc_ids=['doc1', 'doc2'], replace_tags=['archive'])

            # Add 'reviewed' and remove 'draft' for documents with 'pending' tag
            kb.update_tags_bulk(existing_tags=['pending'], add_tags=['reviewed'], remove_tags=['draft'])
        """
        if not doc_ids and not existing_tags:
            raise ValueError("Must provide either doc_ids or existing_tags")

        if not add_tags and not remove_tags and not replace_tags:
            raise ValueError("Must provide at least one of: add_tags, remove_tags, replace_tags")

        results = {
            'updated': [],
            'failed': []
        }

        # Collect doc_ids to update
        ids_to_update = set()

        if doc_ids:
            ids_to_update.update(doc_ids)

        if existing_tags:
            # Find all documents with any of the specified tags
            for doc_id, doc in self.documents.items():
                if any(tag in doc.tags for tag in existing_tags):
                    ids_to_update.add(doc_id)

        self.logger.info(f"Bulk tag update: updating {len(ids_to_update)} documents")

        for doc_id in ids_to_update:
            try:
                if doc_id not in self.documents:
                    results['failed'].append({
                        'doc_id': doc_id,
                        'error': 'Document not found'
                    })
                    continue

                doc = self.documents[doc_id]
                old_tags = doc.tags.copy()

                # Apply tag operations
                if replace_tags is not None:
                    doc.tags = replace_tags.copy()
                else:
                    if add_tags:
                        # Add tags (avoiding duplicates)
                        for tag in add_tags:
                            if tag not in doc.tags:
                                doc.tags.append(tag)

                    if remove_tags:
                        # Remove tags
                        doc.tags = [tag for tag in doc.tags if tag not in remove_tags]

                # Update in database
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET tags = ?
                    WHERE doc_id = ?
                """, (json.dumps(doc.tags), doc_id))
                self.db_conn.commit()

                results['updated'].append({
                    'doc_id': doc_id,
                    'old_tags': old_tags,
                    'new_tags': doc.tags
                })

                self.logger.debug(f"Updated tags for {doc_id}: {old_tags} -> {doc.tags}")

            except Exception as e:
                results['failed'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })
                self.logger.error(f"Failed to update tags for {doc_id}: {e}")

        self.logger.info(f"Bulk tag update complete: {len(results['updated'])} updated, "
                        f"{len(results['failed'])} failed")

        return results

    def auto_tag_document(self, doc_id: str, confidence_threshold: float = 0.7,
                         max_tags: int = 10, append: bool = True) -> dict:
        """
        Generate tags automatically using LLM analysis.

        Args:
            doc_id: Document to tag
            confidence_threshold: Minimum confidence to accept tag (0.0-1.0)
            max_tags: Maximum number of tags to suggest
            append: If True, append to existing tags; if False, replace

        Returns:
            {
                'doc_id': str,
                'suggested_tags': [{'tag': str, 'confidence': float}, ...],
                'applied_tags': [str],
                'skipped_tags': [str],  # Below confidence threshold
                'existing_tags': [str],
                'new_tags': [str]  # Final tag list
            }

        Example:
            result = kb.auto_tag_document('doc123', confidence_threshold=0.7)
            # {
            #     'suggested_tags': [
            #         {'tag': 'sid-programming', 'confidence': 0.95},
            #         {'tag': 'assembly', 'confidence': 0.88},
            #         {'tag': 'beginner', 'confidence': 0.65}  # Below threshold
            #     ],
            #     'applied_tags': ['sid-programming', 'assembly'],
            #     'skipped_tags': ['beginner'],
            #     ...
            # }
        """
        # Import LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Auto-tagging requires LLM integration.")

        # Get LLM client
        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key.")

        # Get document
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Get sample text (first 3 chunks for analysis)
        chunks = self._get_chunks_db(doc_id)
        sample_chunks = chunks[:3] if len(chunks) > 3 else chunks
        sample_text = "\n\n".join([c.content for c in sample_chunks])

        # Limit text size (first 3000 chars)
        if len(sample_text) > 3000:
            sample_text = sample_text[:3000] + "..."

        # Build prompt
        prompt = f"""Analyze this Commodore 64 technical documentation and suggest relevant tags.

Consider these categories:
1. Hardware components (sid, vic-ii, cia, 6502, memory, cartridge, disk-drive, etc.)
2. Programming topics (assembly, basic, machine-code, graphics, sound, sprites, etc.)
3. Document type (tutorial, reference, manual, guide, example, etc.)
4. Difficulty level (beginner, intermediate, advanced, expert)
5. Content area (programming, hardware, history, repair, modification, etc.)

Document title: {doc.title}
Document filename: {doc.filename}

Sample text:
{sample_text}

Return a JSON object with this structure:
{{
    "tags": [
        {{"tag": "sid-programming", "confidence": 0.95, "reason": "Document extensively discusses SID chip programming"}},
        {{"tag": "assembly", "confidence": 0.88, "reason": "Contains assembly code examples"}}
    ]
}}

Important:
- Use lowercase with hyphens (e.g., "sid-programming" not "SID Programming")
- Provide {max_tags} or fewer tags
- Include confidence score (0.0-1.0) for each tag
- Brief reason for each tag suggestion
- Return ONLY the JSON, no other text"""

        # Call LLM
        self.logger.info(f"Auto-tagging document {doc_id} ({doc.title})")

        try:
            response = llm_client.call_json(prompt, max_tokens=1024, temperature=0.3)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise ValueError(f"Failed to generate tags: {e}")

        # Parse response
        suggested_tags = response.get('tags', [])

        # Filter by confidence
        high_confidence_tags = [
            t for t in suggested_tags
            if t['confidence'] >= confidence_threshold
        ]

        low_confidence_tags = [
            t for t in suggested_tags
            if t['confidence'] < confidence_threshold
        ]

        # Extract tag names
        applied_tag_names = [t['tag'] for t in high_confidence_tags]
        skipped_tag_names = [t['tag'] for t in low_confidence_tags]

        # Get existing tags
        existing_tags = doc.tags.copy()

        # Apply tags
        if append:
            # Add new tags to existing (avoid duplicates)
            new_tags = existing_tags.copy()
            for tag in applied_tag_names:
                if tag not in new_tags:
                    new_tags.append(tag)
        else:
            # Replace all tags
            new_tags = applied_tag_names

        # Update document
        doc.tags = new_tags

        # Update in database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET tags = ?
            WHERE doc_id = ?
        """, (json.dumps(new_tags), doc_id))
        self.db_conn.commit()

        result = {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'suggested_tags': suggested_tags,
            'applied_tags': applied_tag_names,
            'skipped_tags': skipped_tag_names,
            'existing_tags': existing_tags,
            'new_tags': new_tags,
            'confidence_threshold': confidence_threshold
        }

        self.logger.info(f"Auto-tagged {doc_id}: applied {len(applied_tag_names)} tags, "
                        f"skipped {len(skipped_tag_names)} low-confidence tags")

        return result

    def auto_tag_all_documents(self, confidence_threshold: float = 0.7,
                               max_tags: int = 10, append: bool = True,
                               skip_tagged: bool = True, max_docs: Optional[int] = None) -> dict:
        """
        Bulk auto-tag all documents using LLM.

        Args:
            confidence_threshold: Minimum confidence to accept tag (0.0-1.0)
            max_tags: Maximum tags per document
            append: If True, append to existing tags; if False, replace
            skip_tagged: If True, skip documents that already have tags
            max_docs: Maximum number of documents to process (None = all)

        Returns:
            {
                'processed': int,
                'skipped': int,
                'failed': int,
                'total_tags_added': int,
                'results': [list of individual results]
            }

        Example:
            results = kb.auto_tag_all_documents(
                confidence_threshold=0.7,
                skip_tagged=True,
                max_docs=10
            )
        """
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_tags_added': 0,
            'results': []
        }

        # Get documents to process
        docs_to_process = []

        for doc_id, doc in self.documents.items():
            # Skip if already has tags (optional)
            if skip_tagged and doc.tags:
                results['skipped'] += 1
                continue

            docs_to_process.append(doc_id)

            # Limit number of documents
            if max_docs and len(docs_to_process) >= max_docs:
                break

        self.logger.info(f"Auto-tagging {len(docs_to_process)} documents "
                        f"(skipped {results['skipped']} already tagged)")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            try:
                self.logger.info(f"Auto-tagging {i}/{len(docs_to_process)}: {doc_id}")

                result = self.auto_tag_document(
                    doc_id,
                    confidence_threshold=confidence_threshold,
                    max_tags=max_tags,
                    append=append
                )

                # Count new tags added
                tags_added = len(set(result['new_tags']) - set(result['existing_tags']))
                results['total_tags_added'] += tags_added

                results['processed'] += 1
                results['results'].append(result)

            except Exception as e:
                results['failed'] += 1
                self.logger.error(f"Failed to auto-tag {doc_id}: {e}")
                results['results'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })

        self.logger.info(f"Auto-tagging complete: processed={results['processed']}, "
                        f"failed={results['failed']}, tags_added={results['total_tags_added']}")

        return results

    def generate_summary(self, doc_id: str, summary_type: str = 'brief',
                        force_regenerate: bool = False) -> str:
        """
        Generate an AI-powered summary of a document.

        Args:
            doc_id: Document ID to summarize
            summary_type: Type of summary ('brief', 'detailed', 'bullet')
                - 'brief': 1-2 paragraph overview (200-300 words)
                - 'detailed': Comprehensive summary with key points (500-800 words)
                - 'bullet': Bullet-point summary of main topics
            force_regenerate: If True, regenerate even if cached summary exists

        Returns:
            Summary text as a string

        Raises:
            ValueError: If document not found or LLM not configured
            DocumentNotFoundError: If document doesn't exist

        Examples:
            # Generate brief summary
            summary = kb.generate_summary('doc123', 'brief')

            # Get detailed summary
            summary = kb.generate_summary('doc456', 'detailed')

            # Force regeneration (bypass cache)
            summary = kb.generate_summary('doc789', 'brief', force_regenerate=True)
        """
        # Validate document exists
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Check for cached summary
        if not force_regenerate:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT summary_text FROM document_summaries
                WHERE doc_id = ? AND summary_type = ?
            """, (doc_id, summary_type))
            result = cursor.fetchone()
            if result:
                self.logger.debug(f"Using cached summary for {doc_id} ({summary_type})")
                return result[0]

        # Import LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Summarization requires LLM integration.")

        # Get LLM client
        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key.")

        # Get document content
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            raise ValueError(f"No content found for document: {doc_id}")

        # For brief summaries, use first 5 chunks; for detailed, use more
        if summary_type == 'brief':
            sample_chunks = chunks[:5]
            word_limit = 300
            length_guidance = "1-2 paragraphs, approximately 200-300 words"
        elif summary_type == 'detailed':
            sample_chunks = chunks[:15] if len(chunks) > 15 else chunks
            word_limit = 800
            length_guidance = "3-5 paragraphs with detailed explanations, approximately 500-800 words"
        elif summary_type == 'bullet':
            sample_chunks = chunks[:10]
            word_limit = 400
            length_guidance = "8-12 bullet points covering main topics"
        else:
            raise ValueError(f"Invalid summary type: {summary_type}. Must be 'brief', 'detailed', or 'bullet'.")

        # Join content
        content = "\n\n".join([c.content for c in sample_chunks])

        # Limit content size to first 10k chars to control API costs
        if len(content) > 10000:
            content = content[:10000] + "..."

        # Build prompt based on summary type
        if summary_type == 'bullet':
            prompt = f"""Create a bullet-point summary of this Commodore 64 technical documentation.

Document Title: {doc.title}
Document Type: {doc.file_type}

Content:
{content}

Create a concise bullet-point summary with 8-12 main topics. Each bullet should be clear and informative.
Return ONLY the bullet points, one per line, starting with "- ". No introduction or explanation needed."""

        else:
            prompt = f"""Create a {summary_type} summary of this Commodore 64 technical documentation.

Document Title: {doc.title}
Document Type: {doc.file_type}

Content:
{content}

Write a {summary_type} summary that is {length_guidance}.
Focus on:
- Key concepts and main topics
- Technical details relevant to programmers
- Important procedures or examples
- Practical applications

Return ONLY the summary text, no preamble."""

        # Call LLM
        self.logger.info(f"Generating {summary_type} summary for {doc_id} ({doc.title})")

        try:
            summary_text = llm_client.call(prompt, max_tokens=word_limit + 200, temperature=0.4)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise ValueError(f"Failed to generate summary: {e}")

        # Clean up summary text
        if not summary_text or not summary_text.strip():
            raise ValueError("LLM returned empty summary")

        summary_text = summary_text.strip()

        # Store summary in database
        cursor = self.db_conn.cursor()
        try:
            # Get model name from LLM client
            model = os.getenv('LLM_MODEL', 'unknown')

            cursor.execute("""
                INSERT OR REPLACE INTO document_summaries
                (doc_id, summary_type, summary_text, generated_at, model, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, summary_type, summary_text, datetime.now().isoformat(),
                  model, len(summary_text.split())))

            self.db_conn.commit()
            self.logger.info(f"Saved {summary_type} summary for {doc_id}")

        except Exception as e:
            self.logger.error(f"Failed to save summary to database: {e}")
            # Return summary even if save failed
            pass

        return summary_text

    def generate_summary_all(self, summary_types: Optional[list[str]] = None,
                            force_regenerate: bool = False,
                            max_docs: Optional[int] = None) -> dict:
        """
        Bulk generate summaries for all documents.

        Args:
            summary_types: List of summary types to generate (['brief'], ['brief', 'detailed'], etc.)
                          Default: ['brief']
            force_regenerate: If True, regenerate all summaries
            max_docs: Maximum number of documents to process (None = all)

        Returns:
            {
                'processed': int,
                'failed': int,
                'total_summaries': int,
                'by_type': {'brief': int, 'detailed': int, 'bullet': int},
                'results': [list of individual results]
            }

        Example:
            results = kb.generate_summary_all(
                summary_types=['brief', 'detailed'],
                max_docs=50
            )
        """
        if summary_types is None:
            summary_types = ['brief']

        results = {
            'processed': 0,
            'failed': 0,
            'total_summaries': 0,
            'by_type': {st: 0 for st in summary_types},
            'results': []
        }

        # Get documents to process
        docs_to_process = list(self.documents.keys())

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Generating summaries for {len(docs_to_process)} documents "
                        f"(types: {', '.join(summary_types)})")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'summaries': {}
            }

            for summary_type in summary_types:
                try:
                    self.logger.info(f"[{i}/{len(docs_to_process)}] {doc_id} ({summary_type})")

                    summary = self.generate_summary(
                        doc_id,
                        summary_type=summary_type,
                        force_regenerate=force_regenerate
                    )

                    doc_results['summaries'][summary_type] = {
                        'success': True,
                        'length': len(summary),
                        'word_count': len(summary.split())
                    }

                    results['total_summaries'] += 1
                    results['by_type'][summary_type] += 1

                except Exception as e:
                    results['failed'] += 1
                    self.logger.error(f"Failed to summarize {doc_id} ({summary_type}): {e}")
                    doc_results['summaries'][summary_type] = {
                        'success': False,
                        'error': str(e)
                    }

            results['processed'] += 1
            results['results'].append(doc_results)

        self.logger.info(f"Summary generation complete: processed={results['processed']}, "
                        f"failed={results['failed']}, total_summaries={results['total_summaries']}")

        return results

    def get_summary(self, doc_id: str, summary_type: str = 'brief') -> Optional[str]:
        """
        Retrieve a cached summary without regenerating.

        Args:
            doc_id: Document ID
            summary_type: Type of summary ('brief', 'detailed', 'bullet')

        Returns:
            Summary text if it exists, None otherwise
        """
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT summary_text FROM document_summaries
            WHERE doc_id = ? AND summary_type = ?
        """, (doc_id, summary_type))
        result = cursor.fetchone()
        return result[0] if result else None

    def extract_entities(self, doc_id: str,
                        confidence_threshold: float = 0.6,
                        force_regenerate: bool = False) -> dict:
        """
        Extract named entities from a document using LLM analysis.

        Args:
            doc_id: Document ID to extract entities from
            confidence_threshold: Minimum confidence to include entity (0.0-1.0, default: 0.6)
            force_regenerate: If True, extract even if entities already exist

        Returns:
            {
                'doc_id': str,
                'doc_title': str,
                'entities': [
                    {
                        'entity_text': 'VIC-II',
                        'entity_type': 'hardware',
                        'confidence': 0.95,
                        'context': '...snippet...',
                        'occurrence_count': 5
                    },
                    ...
                ],
                'entity_count': 42,
                'types': {'hardware': 10, 'memory_address': 8, ...}
            }

        Example:
            result = kb.extract_entities('my-doc-id', confidence_threshold=0.7)
            for entity in result['entities']:
                print(f"{entity['entity_type']}: {entity['entity_text']} ({entity['confidence']})")
        """
        # Validate document exists
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        self.logger.info(f"Extracting entities from document {doc_id} ({doc.title})")

        # Check if entities already exist
        if not force_regenerate:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                self.logger.info(f"Document {doc_id} already has {existing_count} entities (use force_regenerate=True to re-extract)")
                # Return existing entities
                return self.get_entities(doc_id)

        # Get LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Install required dependencies.")

        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)")

        # Get document chunks for sampling
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            raise ValueError(f"No chunks found for document {doc_id}")

        # Sample first 5 chunks (balance between coverage and cost)
        sample_chunks = chunks[:5] if len(chunks) > 5 else chunks
        sample_text = "\n\n".join([c.content for c in sample_chunks])

        # Limit to 5000 characters for cost control
        if len(sample_text) > 5000:
            sample_text = sample_text[:5000] + "..."

        # Build prompt for entity extraction
        prompt = f"""Extract named entities from this Commodore 64 technical documentation.

Document Title: {doc.title}

Content:
{sample_text}

Extract entities in these categories:
1. hardware - Chip names and components (SID, VIC-II, CIA, 6502, 6526, 6581, etc.)
2. memory_address - Memory addresses in any format ($D000, $D020, 53280, 0xD020, etc.)
3. instruction - Assembly instructions (LDA, STA, JMP, JSR, RTS, BRK, etc.)
4. person - People mentioned (Bob Yannes, Jack Tramiel, etc.)
5. company - Organizations (Commodore, MOS Technology, etc.)
6. product - Hardware/software products (VIC-20, C128, 1541, etc.)
7. concept - Technical concepts (sprite, raster interrupt, IRQ, DMA, etc.)

For each entity found, provide:
- entity_text: The entity name as it appears in the document
- entity_type: One of the categories above (lowercase with underscores)
- confidence: How confident you are this is a valid entity (0.0-1.0)
- context: Brief surrounding text showing how the entity is used (max 100 chars)

Return ONLY valid JSON in this exact format:
{{
    "entities": [
        {{
            "entity_text": "VIC-II",
            "entity_type": "hardware",
            "confidence": 0.95,
            "context": "The VIC-II chip controls all graphics"
        }},
        {{
            "entity_text": "$D020",
            "entity_type": "memory_address",
            "confidence": 0.98,
            "context": "Border color is controlled by $D020"
        }}
    ]
}}

Important:
- Extract 20-50 entities maximum
- Include confidence scores (0.0-1.0)
- Provide brief context snippets
- Preserve original capitalization/format
- Return ONLY JSON, no other text
"""

        # Call LLM
        self.logger.info(f"Calling LLM for entity extraction ({len(sample_text)} chars)")
        try:
            response = llm_client.call_json(prompt, max_tokens=2048, temperature=0.3)
        except Exception as e:
            self.logger.error(f"LLM call failed for entity extraction: {e}")
            raise ValueError(f"Failed to extract entities: {e}")

        # Parse response
        all_entities = response.get('entities', [])
        self.logger.info(f"LLM returned {len(all_entities)} entities")

        # Filter by confidence threshold
        filtered_entities = [
            e for e in all_entities
            if e.get('confidence', 0) >= confidence_threshold
        ]

        # Deduplicate entities (case-insensitive, count occurrences)
        entity_map = {}
        for entity in filtered_entities:
            key = (entity['entity_text'].lower(), entity['entity_type'])
            if key in entity_map:
                entity_map[key]['occurrence_count'] += 1
                # Keep highest confidence
                if entity['confidence'] > entity_map[key]['confidence']:
                    entity_map[key]['confidence'] = entity['confidence']
                    entity_map[key]['context'] = entity.get('context', '')
            else:
                entity_map[key] = {
                    'entity_text': entity['entity_text'],  # Preserve original casing
                    'entity_type': entity['entity_type'],
                    'confidence': entity['confidence'],
                    'context': entity.get('context', ''),
                    'occurrence_count': 1
                }

        unique_entities = list(entity_map.values())

        # Store in database
        cursor = self.db_conn.cursor()
        try:
            # Delete existing entities for this document
            cursor.execute("DELETE FROM document_entities WHERE doc_id = ?", (doc_id,))

            # Insert new entities
            from datetime import datetime
            generated_at = datetime.now().isoformat()
            model = llm_client.model if hasattr(llm_client, 'model') else 'unknown'

            for i, entity in enumerate(unique_entities, 1):
                cursor.execute("""
                    INSERT INTO document_entities
                    (doc_id, entity_id, entity_text, entity_type, confidence, context,
                     occurrence_count, generated_at, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    i,
                    entity['entity_text'],
                    entity['entity_type'],
                    entity['confidence'],
                    entity.get('context', ''),
                    entity.get('occurrence_count', 1),
                    generated_at,
                    model
                ))

            self.db_conn.commit()
            self.logger.info(f"Stored {len(unique_entities)} entities for document {doc_id}")

        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Failed to store entities in database: {e}")
            # Return entities even if storage failed
            pass

        # Build result
        types = {}
        for entity in unique_entities:
            entity_type = entity['entity_type']
            types[entity_type] = types.get(entity_type, 0) + 1

        return {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'entities': unique_entities,
            'entity_count': len(unique_entities),
            'types': types
        }

    def get_entities(self, doc_id: str,
                    entity_types: Optional[list[str]] = None,
                    min_confidence: float = 0.0) -> dict:
        """
        Get all entities for a specific document.

        Args:
            doc_id: Document ID
            entity_types: Optional list of entity types to filter by
                         (e.g., ['hardware', 'memory_address'])
            min_confidence: Minimum confidence threshold (default: 0.0)

        Returns:
            {
                'doc_id': str,
                'doc_title': str,
                'entities': [list of entity dicts],
                'entity_count': int,
                'types': {'hardware': count, ...}
            }

        Example:
            # Get all entities
            result = kb.get_entities('my-doc-id')

            # Get only hardware entities with high confidence
            result = kb.get_entities('my-doc-id',
                                    entity_types=['hardware'],
                                    min_confidence=0.8)
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        cursor = self.db_conn.cursor()

        # Build query with optional filters
        query = """
            SELECT entity_text, entity_type, confidence, context, occurrence_count
            FROM document_entities
            WHERE doc_id = ? AND confidence >= ?
        """
        params = [doc_id, min_confidence]

        if entity_types:
            placeholders = ','.join(['?'] * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += " ORDER BY entity_type, confidence DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        entities = []
        types = {}
        for row in rows:
            entity = {
                'entity_text': row[0],
                'entity_type': row[1],
                'confidence': row[2],
                'context': row[3],
                'occurrence_count': row[4]
            }
            entities.append(entity)

            entity_type = row[1]
            types[entity_type] = types.get(entity_type, 0) + 1

        return {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'entities': entities,
            'entity_count': len(entities),
            'types': types
        }

    def search_entities(self, query: str,
                       entity_types: Optional[list[str]] = None,
                       min_confidence: float = 0.0,
                       max_results: int = 20) -> dict:
        """
        Search for entities across all documents using full-text search.

        Args:
            query: Search query (e.g., "VIC-II", "sprite", "$D000")
            entity_types: Filter by entity types (e.g., ['hardware', 'memory_address'])
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results grouped by document:
            {
                'query': str,
                'total_matches': int,
                'documents': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'matches': [
                            {
                                'entity_text': str,
                                'entity_type': str,
                                'confidence': float,
                                'context': str,
                                'occurrence_count': int
                            },
                            ...
                        ],
                        'match_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Search for VIC-II chip mentions
            results = kb.search_entities('VIC-II')

            # Search for memory addresses only
            results = kb.search_entities('$D0', entity_types=['memory_address'])

            # Search with confidence threshold
            results = kb.search_entities('sprite', min_confidence=0.7)
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Build FTS5 query
        # Escape special FTS5 characters and wrap in quotes for literal search
        fts_query = f'"{query.strip().replace('"', '""')}"'

        # Build WHERE clause for filtering
        where_clauses = []
        params = []

        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            where_clauses.append(f"e.entity_type IN ({placeholders})")
            params.extend(entity_types)

        if min_confidence > 0.0:
            where_clauses.append("e.confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Search entities_fts and join with document_entities for full data
        query_sql = f"""
            SELECT e.doc_id, e.entity_text, e.entity_type, e.confidence,
                   e.context, e.occurrence_count
            FROM entities_fts fts
            JOIN document_entities e ON fts.rowid = e.rowid
            WHERE entities_fts MATCH ?{where_clause}
            ORDER BY rank
            LIMIT ?
        """

        # Execute search
        cursor = self.db_conn.cursor()
        cursor.execute(query_sql, [fts_query] + params + [max_results * 10])  # Get extra for grouping
        rows = cursor.fetchall()

        # Group results by document
        doc_matches = {}
        for row in rows:
            doc_id, entity_text, entity_type, confidence, context, occurrence_count = row

            if doc_id not in doc_matches:
                doc_matches[doc_id] = []

            doc_matches[doc_id].append({
                'entity_text': entity_text,
                'entity_type': entity_type,
                'confidence': confidence,
                'context': context or '',
                'occurrence_count': occurrence_count
            })

        # Build result list with document titles
        documents = []
        for doc_id, matches in list(doc_matches.items())[:max_results]:
            doc = self.documents.get(doc_id)
            if doc:
                documents.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'matches': matches,
                    'match_count': len(matches)
                })

        return {
            'query': query,
            'total_matches': sum(len(matches) for matches in doc_matches.values()),
            'documents': documents
        }

    def find_docs_by_entity(self, entity_text: str,
                           entity_type: Optional[str] = None,
                           min_confidence: float = 0.0,
                           max_results: int = 20) -> dict:
        """
        Find all documents that contain a specific entity.

        Args:
            entity_text: Exact entity text to search for (e.g., "VIC-II", "$D000")
            entity_type: Optional entity type filter (e.g., 'hardware', 'memory_address')
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of documents to return

        Returns:
            Dictionary with documents containing the entity:
            {
                'entity_text': str,
                'entity_type': str or None,
                'total_documents': int,
                'documents': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'entity_type': str,
                        'confidence': float,
                        'context': str,
                        'occurrence_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Find all documents mentioning VIC-II
            results = kb.find_docs_by_entity('VIC-II')

            # Find documents with $D000 memory address
            results = kb.find_docs_by_entity('$D000', entity_type='memory_address')

            # Find with confidence threshold
            results = kb.find_docs_by_entity('sprite', min_confidence=0.7)
        """
        if not entity_text or not entity_text.strip():
            raise ValueError("Entity text cannot be empty")

        # Build WHERE clause
        where_clauses = ["e.entity_text = ?"]
        params = [entity_text.strip()]

        if entity_type:
            where_clauses.append("e.entity_type = ?")
            params.append(entity_type)

        if min_confidence > 0.0:
            where_clauses.append("e.confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND ".join(where_clauses)

        # Query database
        query_sql = f"""
            SELECT e.doc_id, e.entity_type, e.confidence, e.context,
                   e.occurrence_count
            FROM document_entities e
            WHERE {where_clause}
            ORDER BY e.confidence DESC, e.occurrence_count DESC
            LIMIT ?
        """

        cursor = self.db_conn.cursor()
        cursor.execute(query_sql, params + [max_results])
        rows = cursor.fetchall()

        # Build result list with document titles
        documents = []
        for row in rows:
            doc_id, ent_type, confidence, context, occurrence_count = row
            doc = self.documents.get(doc_id)
            if doc:
                documents.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'entity_type': ent_type,
                    'confidence': confidence,
                    'context': context or '',
                    'occurrence_count': occurrence_count
                })

        return {
            'entity_text': entity_text.strip(),
            'entity_type': entity_type,
            'total_documents': len(documents),
            'documents': documents
        }

    def get_entity_stats(self, entity_type: Optional[str] = None) -> dict:
        """
        Get statistics about extracted entities in the knowledge base.

        Args:
            entity_type: Optional filter by entity type (e.g., 'hardware', 'memory_address')

        Returns:
            Dictionary with entity statistics:
            {
                'total_entities': int,
                'total_documents_with_entities': int,
                'by_type': {
                    'hardware': int,
                    'memory_address': int,
                    ...
                },
                'top_entities': [
                    {
                        'entity_text': str,
                        'entity_type': str,
                        'document_count': int,
                        'total_occurrences': int,
                        'avg_confidence': float
                    },
                    ...
                ],
                'documents_with_most_entities': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'entity_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Get overall statistics
            stats = kb.get_entity_stats()

            # Get statistics for hardware entities only
            stats = kb.get_entity_stats(entity_type='hardware')
        """
        cursor = self.db_conn.cursor()

        # Total entities
        if entity_type:
            cursor.execute(
                "SELECT COUNT(*) FROM document_entities WHERE entity_type = ?",
                (entity_type,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM document_entities")
        total_entities = cursor.fetchone()[0]

        # Total documents with entities
        if entity_type:
            cursor.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM document_entities WHERE entity_type = ?",
                (entity_type,)
            )
        else:
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM document_entities")
        total_docs = cursor.fetchone()[0]

        # Breakdown by type
        by_type = {}
        if entity_type:
            by_type[entity_type] = total_entities
        else:
            cursor.execute("""
                SELECT entity_type, COUNT(*)
                FROM document_entities
                GROUP BY entity_type
                ORDER BY COUNT(*) DESC
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Top entities by document count
        type_filter = "WHERE entity_type = ?" if entity_type else ""
        params = [entity_type] if entity_type else []

        cursor.execute(f"""
            SELECT entity_text, entity_type,
                   COUNT(DISTINCT doc_id) as doc_count,
                   SUM(occurrence_count) as total_occurrences,
                   AVG(confidence) as avg_confidence
            FROM document_entities
            {type_filter}
            GROUP BY entity_text, entity_type
            ORDER BY doc_count DESC, total_occurrences DESC
            LIMIT 20
        """, params)

        top_entities = [
            {
                'entity_text': row[0],
                'entity_type': row[1],
                'document_count': row[2],
                'total_occurrences': row[3],
                'avg_confidence': round(row[4], 3)
            }
            for row in cursor.fetchall()
        ]

        # Documents with most entities
        cursor.execute(f"""
            SELECT doc_id, COUNT(*) as entity_count
            FROM document_entities
            {type_filter}
            GROUP BY doc_id
            ORDER BY entity_count DESC
            LIMIT 10
        """, params)

        docs_with_most = []
        for row in cursor.fetchall():
            doc_id = row[0]
            entity_count = row[1]
            doc = self.documents.get(doc_id)
            if doc:
                docs_with_most.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'entity_count': entity_count
                })

        return {
            'total_entities': total_entities,
            'total_documents_with_entities': total_docs,
            'by_type': by_type,
            'top_entities': top_entities,
            'documents_with_most_entities': docs_with_most
        }

    def extract_entities_bulk(self, confidence_threshold: float = 0.6,
                             force_regenerate: bool = False,
                             max_docs: Optional[int] = None,
                             skip_existing: bool = True) -> dict:
        """
        Bulk extract entities for multiple documents.

        Args:
            confidence_threshold: Minimum confidence threshold (0.0-1.0, default: 0.6)
            force_regenerate: If True, re-extract even if entities already exist
            max_docs: Maximum number of documents to process (None = all)
            skip_existing: If True, skip documents that already have entities (unless force_regenerate)

        Returns:
            {
                'processed': int,
                'failed': int,
                'skipped': int,
                'total_entities': int,
                'by_type': {'hardware': int, 'memory_address': int, ...},
                'results': [list of individual results]
            }

        Examples:
            # Extract entities for all documents
            results = kb.extract_entities_bulk()

            # Extract for first 10 documents only
            results = kb.extract_entities_bulk(max_docs=10)

            # Force re-extraction for all documents
            results = kb.extract_entities_bulk(force_regenerate=True)
        """
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_entities': 0,
            'by_type': {},
            'results': []
        }

        # Get documents to process
        docs_to_process = list(self.documents.keys())

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Extracting entities for {len(docs_to_process)} documents "
                        f"(confidence threshold: {confidence_threshold})")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'status': 'unknown',
                'entity_count': 0,
                'error': None
            }

            try:
                # Check if entities already exist (unless force_regenerate)
                if skip_existing and not force_regenerate:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM document_entities WHERE doc_id = ?",
                        (doc_id,)
                    )
                    existing_count = cursor.fetchone()[0]

                    if existing_count > 0:
                        self.logger.info(f"[{i}/{len(docs_to_process)}] Skipping {doc_id} (already has {existing_count} entities)")
                        doc_results['status'] = 'skipped'
                        doc_results['entity_count'] = existing_count
                        results['skipped'] += 1
                        results['results'].append(doc_results)
                        continue

                # Extract entities
                self.logger.info(f"[{i}/{len(docs_to_process)}] Extracting entities from {doc_id}")

                result = self.extract_entities(
                    doc_id,
                    confidence_threshold=confidence_threshold,
                    force_regenerate=force_regenerate
                )

                doc_results['status'] = 'success'
                doc_results['entity_count'] = result['entity_count']

                # Update counts
                results['processed'] += 1
                results['total_entities'] += result['entity_count']

                # Update by_type counts
                for entity_type, count in result['types'].items():
                    results['by_type'][entity_type] = results['by_type'].get(entity_type, 0) + count

                results['results'].append(doc_results)

            except Exception as e:
                self.logger.error(f"[{i}/{len(docs_to_process)}] Failed to extract entities from {doc_id}: {e}")
                doc_results['status'] = 'failed'
                doc_results['error'] = str(e)
                results['failed'] += 1
                results['results'].append(doc_results)

        self.logger.info(f"Bulk entity extraction complete: processed={results['processed']}, "
                        f"failed={results['failed']}, skipped={results['skipped']}, "
                        f"total_entities={results['total_entities']}")

        return results

    # ========== ENTITY RELATIONSHIP METHODS ==========

    def extract_entity_relationships(self, doc_id: str, min_confidence: float = 0.6,
                                   force_regenerate: bool = False) -> dict:
        """
        Extract entity co-occurrence relationships from a document.

        This method analyzes how entities appear together in document chunks
        to identify related concepts, hardware, instructions, etc.

        Args:
            doc_id: Document ID to extract relationships from
            min_confidence: Minimum confidence threshold for entities (default: 0.6)
            force_regenerate: If True, regenerate relationships even if they exist

        Returns:
            {
                'doc_id': str,
                'relationship_count': int,
                'relationships': [
                    {
                        'entity1': str,
                        'entity1_type': str,
                        'entity2': str,
                        'entity2_type': str,
                        'strength': float,
                        'context': str
                    },
                    ...
                ]
            }

        Examples:
            # Extract relationships from a document
            result = kb.extract_entity_relationships('doc-id-123')

            # Force regeneration with higher confidence
            result = kb.extract_entity_relationships('doc-id-123',
                min_confidence=0.7, force_regenerate=True)
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        self.logger.info(f"Extracting entity relationships from document {doc_id}")

        # Get all entities for this document
        entities = self.get_entities(doc_id, min_confidence=min_confidence)

        if not entities:
            self.logger.info(f"No entities found for document {doc_id}")
            return {
                'doc_id': doc_id,
                'relationship_count': 0,
                'relationships': []
            }

        # Get all chunks for this document
        chunks = self._get_chunks_db(doc_id)

        if not chunks:
            self.logger.info(f"No chunks found for document {doc_id}")
            return {
                'doc_id': doc_id,
                'relationship_count': 0,
                'relationships': []
            }

        # Build entity -> type mapping
        entity_map = {e['entity_text']: e['entity_type'] for e in entities}

        # Find co-occurrences
        relationships = {}  # (entity1, entity2) -> strength
        relationship_contexts = {}  # (entity1, entity2) -> context

        for chunk in chunks:
            content = chunk.content
            # Find which entities appear in this chunk
            entities_in_chunk = [e for e in entity_map.keys() if e in content]

            # Create pairs of co-occurring entities
            for i, e1 in enumerate(entities_in_chunk):
                for e2 in entities_in_chunk[i+1:]:
                    # Sort entities alphabetically to avoid duplicates (A,B) vs (B,A)
                    pair = tuple(sorted([e1, e2]))

                    # Increment strength
                    relationships[pair] = relationships.get(pair, 0) + 1

                    # Store context (first occurrence)
                    if pair not in relationship_contexts:
                        # Extract context around both entities
                        start_idx = max(0, content.find(e1) - 50)
                        end_idx = min(len(content), content.find(e2) + len(e2) + 50)
                        context = content[start_idx:end_idx].strip()
                        relationship_contexts[pair] = context

        # Normalize strength scores (0.0-1.0)
        max_strength = max(relationships.values()) if relationships else 1
        normalized_relationships = {
            pair: strength / max_strength
            for pair, strength in relationships.items()
        }

        # Store relationships in database
        cursor = self.db_conn.cursor()
        from datetime import datetime
        now = datetime.now().isoformat()

        relationship_list = []
        for (e1, e2), strength in normalized_relationships.items():
            e1_type = entity_map[e1]
            e2_type = entity_map[e2]
            context = relationship_contexts[(e1, e2)]

            # Check if relationship already exists
            cursor.execute("""
                SELECT strength, doc_count FROM entity_relationships
                WHERE entity1_text = ? AND entity2_text = ? AND relationship_type = 'co-occurrence'
            """, (e1, e2))

            existing = cursor.fetchone()

            if existing:
                # Update existing relationship
                old_strength, doc_count = existing
                new_strength = (old_strength * doc_count + strength) / (doc_count + 1)
                new_doc_count = doc_count + 1

                cursor.execute("""
                    UPDATE entity_relationships
                    SET strength = ?, doc_count = ?, last_updated = ?, context_sample = ?
                    WHERE entity1_text = ? AND entity2_text = ? AND relationship_type = 'co-occurrence'
                """, (new_strength, new_doc_count, now, context, e1, e2))
            else:
                # Insert new relationship
                cursor.execute("""
                    INSERT INTO entity_relationships
                    (entity1_text, entity1_type, entity2_text, entity2_type,
                     relationship_type, strength, doc_count, first_seen_doc,
                     context_sample, last_updated)
                    VALUES (?, ?, ?, ?, 'co-occurrence', ?, 1, ?, ?, ?)
                """, (e1, e1_type, e2, e2_type, strength, doc_id, context, now))

            relationship_list.append({
                'entity1': e1,
                'entity1_type': e1_type,
                'entity2': e2,
                'entity2_type': e2_type,
                'strength': strength,
                'context': context
            })

        self.db_conn.commit()

        self.logger.info(f"Extracted {len(relationship_list)} relationships from {doc_id}")

        return {
            'doc_id': doc_id,
            'relationship_count': len(relationship_list),
            'relationships': sorted(relationship_list, key=lambda x: x['strength'], reverse=True)
        }

    def get_entity_relationships(self, entity_text: str,
                                relationship_type: Optional[str] = None,
                                min_strength: float = 0.0,
                                max_results: int = 20) -> list:
        """
        Get all relationships for a given entity.

        Args:
            entity_text: The entity to find relationships for
            relationship_type: Filter by relationship type (default: all types)
            min_strength: Minimum relationship strength (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            List of relationships:
            [
                {
                    'related_entity': str,
                    'related_type': str,
                    'relationship_type': str,
                    'strength': float,
                    'doc_count': int,
                    'context_sample': str
                },
                ...
            ]

        Examples:
            # Find entities related to VIC-II
            relationships = kb.get_entity_relationships('VIC-II')

            # Find strong co-occurrences only
            relationships = kb.get_entity_relationships('VIC-II',
                relationship_type='co-occurrence', min_strength=0.5)
        """
        cursor = self.db_conn.cursor()

        # Build query
        query = """
            SELECT entity1_text, entity1_type, entity2_text, entity2_type,
                   relationship_type, strength, doc_count, context_sample
            FROM entity_relationships
            WHERE (entity1_text = ? OR entity2_text = ?)
              AND strength >= ?
        """
        params = [entity_text, entity_text, min_strength]

        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)

        query += " ORDER BY strength DESC, doc_count DESC LIMIT ?"
        params.append(max_results)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            e1_text, e1_type, e2_text, e2_type, rel_type, strength, doc_count, context = row

            # Determine which is the "other" entity
            if e1_text == entity_text:
                related_entity = e2_text
                related_type = e2_type
            else:
                related_entity = e1_text
                related_type = e1_type

            results.append({
                'related_entity': related_entity,
                'related_type': related_type,
                'relationship_type': rel_type,
                'strength': strength,
                'doc_count': doc_count,
                'context_sample': context
            })

        return results

    def find_related_entities(self, entity_text: str, max_results: int = 10) -> list:
        """
        Discover entities related to a given entity (convenience method).

        This is a simplified version of get_entity_relationships() optimized
        for entity discovery and exploration.

        Args:
            entity_text: The entity to find related entities for
            max_results: Maximum number of related entities to return

        Returns:
            List of related entities with strength scores

        Examples:
            # Discover entities related to SID chip
            related = kb.find_related_entities('SID')
        """
        return self.get_entity_relationships(
            entity_text=entity_text,
            relationship_type='co-occurrence',
            min_strength=0.3,
            max_results=max_results
        )

    def search_by_entity_pair(self, entity1: str, entity2: str,
                             max_results: int = 10) -> list:
        """
        Find documents that contain both entities.

        Args:
            entity1: First entity to search for
            entity2: Second entity to search for
            max_results: Maximum number of documents to return

        Returns:
            List of documents containing both entities:
            [
                {
                    'doc_id': str,
                    'title': str,
                    'entity1_count': int,
                    'entity2_count': int,
                    'contexts': [str, ...]  # Snippets showing both entities
                },
                ...
            ]

        Examples:
            # Find docs about VIC-II and raster interrupts
            docs = kb.search_by_entity_pair('VIC-II', 'raster interrupt')
        """
        cursor = self.db_conn.cursor()

        # Find documents containing both entities
        query = """
            SELECT e1.doc_id,
                   COUNT(DISTINCT e1.entity_id) as entity1_count,
                   COUNT(DISTINCT e2.entity_id) as entity2_count
            FROM document_entities e1
            JOIN document_entities e2 ON e1.doc_id = e2.doc_id
            WHERE e1.entity_text = ? AND e2.entity_text = ?
            GROUP BY e1.doc_id
            ORDER BY entity1_count + entity2_count DESC
            LIMIT ?
        """

        cursor.execute(query, (entity1, entity2, max_results))
        rows = cursor.fetchall()

        results = []
        for doc_id, e1_count, e2_count in rows:
            doc = self.documents.get(doc_id)
            if not doc:
                continue

            # Get context snippets showing both entities
            chunks = self._get_chunks_db(doc_id)
            contexts = []

            for chunk in chunks:
                if entity1 in chunk.content and entity2 in chunk.content:
                    # Extract snippet around both entities
                    idx1 = chunk.content.find(entity1)
                    idx2 = chunk.content.find(entity2)
                    start = max(0, min(idx1, idx2) - 50)
                    end = min(len(chunk.content), max(idx1 + len(entity1), idx2 + len(entity2)) + 50)
                    context = chunk.content[start:end].strip()
                    contexts.append(context)

            results.append({
                'doc_id': doc_id,
                'title': doc.title,
                'entity1_count': e1_count,
                'entity2_count': e2_count,
                'contexts': contexts[:3]  # Return top 3 contexts
            })

        return results

    def extract_relationships_bulk(self, min_confidence: float = 0.6,
                                   max_docs: Optional[int] = None,
                                   skip_existing: bool = False) -> dict:
        """
        Bulk extract entity relationships for multiple documents.

        Args:
            min_confidence: Minimum confidence threshold for entities
            max_docs: Maximum number of documents to process (None = all)
            skip_existing: If True, skip documents that already have relationships

        Returns:
            {
                'processed': int,
                'failed': int,
                'skipped': int,
                'total_relationships': int,
                'results': [list of individual results]
            }

        Examples:
            # Extract relationships for all documents
            results = kb.extract_relationships_bulk()

            # Extract for documents with entities only
            results = kb.extract_relationships_bulk(skip_existing=True)
        """
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_relationships': 0,
            'results': []
        }

        # Get documents with entities
        docs_to_process = []
        cursor = self.db_conn.cursor()

        for doc_id in self.documents.keys():
            cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
            entity_count = cursor.fetchone()[0]

            if entity_count > 0:
                docs_to_process.append(doc_id)

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Extracting entity relationships for {len(docs_to_process)} documents")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'status': 'unknown',
                'relationship_count': 0,
                'error': None
            }

            try:
                self.logger.info(f"[{i}/{len(docs_to_process)}] Extracting relationships from {doc_id}")

                result = self.extract_entity_relationships(doc_id, min_confidence=min_confidence)

                doc_results['status'] = 'success'
                doc_results['relationship_count'] = result['relationship_count']

                results['processed'] += 1
                results['total_relationships'] += result['relationship_count']
                results['results'].append(doc_results)

            except Exception as e:
                self.logger.error(f"[{i}/{len(docs_to_process)}] Failed to extract relationships from {doc_id}: {e}")
                doc_results['status'] = 'failed'
                doc_results['error'] = str(e)
                results['failed'] += 1
                results['results'].append(doc_results)

        self.logger.info(f"Bulk relationship extraction complete: processed={results['processed']}, "
                        f"failed={results['failed']}, total_relationships={results['total_relationships']}")

        return results

    def export_documents_bulk(self, doc_ids: Optional[list[str]] = None,
                              tags: Optional[list[str]] = None,
                              format: str = 'json') -> str:
        """
        Export metadata for multiple documents.

        Args:
            doc_ids: List of document IDs to export (if None, uses tags or exports all)
            tags: Export documents with any of these tags
            format: Export format ('json', 'csv', or 'markdown')

        Returns:
            Exported data as a string

        Examples:
            # Export all documents as JSON
            data = kb.export_documents_bulk(format='json')

            # Export documents with 'reference' tag as CSV
            data = kb.export_documents_bulk(tags=['reference'], format='csv')

            # Export specific documents as Markdown
            data = kb.export_documents_bulk(doc_ids=['doc1', 'doc2'], format='markdown')
        """
        # Collect docs to export
        docs_to_export = []

        if doc_ids:
            # Export specific documents
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    docs_to_export.append(self.documents[doc_id])
        elif tags:
            # Export documents with specified tags
            for doc in self.documents.values():
                if any(tag in doc.tags for tag in tags):
                    docs_to_export.append(doc)
        else:
            # Export all documents
            docs_to_export = list(self.documents.values())

        self.logger.info(f"Bulk export: exporting {len(docs_to_export)} documents as {format}")

        # Format the output
        if format == 'json':
            export_data = []
            for doc in docs_to_export:
                export_data.append({
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'title': doc.title,
                    'filepath': doc.filepath,
                    'file_type': doc.file_type,
                    'total_pages': doc.total_pages,
                    'total_chunks': doc.total_chunks,
                    'indexed_at': doc.indexed_at,
                    'tags': doc.tags,
                    'author': doc.author,
                    'subject': doc.subject,
                    'creator': doc.creator,
                    'creation_date': doc.creation_date
                })
            return json.dumps(export_data, indent=2)

        elif format == 'csv':
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(['doc_id', 'filename', 'title', 'filepath', 'file_type',
                           'total_pages', 'total_chunks', 'indexed_at', 'tags',
                           'author', 'subject', 'creator', 'creation_date'])

            # Write data
            for doc in docs_to_export:
                writer.writerow([
                    doc.doc_id,
                    doc.filename,
                    doc.title,
                    doc.filepath,
                    doc.file_type,
                    doc.total_pages,
                    doc.total_chunks,
                    doc.indexed_at,
                    ', '.join(doc.tags),
                    doc.author or '',
                    doc.subject or '',
                    doc.creator or '',
                    doc.creation_date or ''
                ])

            return output.getvalue()

        elif format == 'markdown':
            lines = []
            lines.append(f"# Document Export")
            lines.append(f"\n**Total Documents:** {len(docs_to_export)}")
            lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append("---\n")

            for i, doc in enumerate(docs_to_export, 1):
                lines.append(f"## {i}. {doc.title}")
                lines.append(f"- **ID:** `{doc.doc_id}`")
                lines.append(f"- **Filename:** {doc.filename}")
                lines.append(f"- **Type:** {doc.file_type}")
                lines.append(f"- **Pages:** {doc.total_pages}")
                lines.append(f"- **Chunks:** {doc.total_chunks}")
                lines.append(f"- **Tags:** {', '.join(doc.tags) if doc.tags else 'None'}")
                if doc.author:
                    lines.append(f"- **Author:** {doc.author}")
                if doc.subject:
                    lines.append(f"- **Subject:** {doc.subject}")
                lines.append(f"- **Indexed:** {doc.indexed_at}")
                lines.append(f"- **Path:** `{doc.filepath}`")
                lines.append("")

            return '\n'.join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'markdown'")

    def add_relationship(self, from_doc_id: str, to_doc_id: str,
                        relationship_type: str = "related", note: str = "") -> dict:
        """
        Add a relationship between two documents.

        Args:
            from_doc_id: Source document ID
            to_doc_id: Target document ID
            relationship_type: Type of relationship (e.g., 'related', 'references', 'prerequisite', 'sequel')
            note: Optional note about the relationship

        Returns:
            Dictionary with relationship details

        Examples:
            # Mark document as related
            kb.add_relationship("doc1", "doc2", "related", "Both cover VIC-II graphics")

            # Mark as prerequisite
            kb.add_relationship("basic_guide", "advanced_guide", "prerequisite", "Read basic first")
        """
        # Validate documents exist
        if from_doc_id not in self.documents:
            raise ValueError(f"Source document not found: {from_doc_id}")
        if to_doc_id not in self.documents:
            raise ValueError(f"Target document not found: {to_doc_id}")

        # Create relationships table if it doesn't exist
        cursor = self.db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_doc_id TEXT NOT NULL,
                to_doc_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY (to_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                UNIQUE(from_doc_id, to_doc_id, relationship_type)
            )
        """)

        # Check if relationship already exists
        cursor.execute("""
            SELECT id FROM document_relationships
            WHERE from_doc_id = ? AND to_doc_id = ? AND relationship_type = ?
        """, (from_doc_id, to_doc_id, relationship_type))

        if cursor.fetchone():
            raise ValueError(f"Relationship already exists: {from_doc_id} -> {to_doc_id} ({relationship_type})")

        # Insert relationship
        created_at = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO document_relationships (from_doc_id, to_doc_id, relationship_type, note, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (from_doc_id, to_doc_id, relationship_type, note, created_at))

        self.db_conn.commit()

        self.logger.info(f"Added relationship: {from_doc_id} -> {to_doc_id} ({relationship_type})")

        return {
            'from_doc_id': from_doc_id,
            'to_doc_id': to_doc_id,
            'relationship_type': relationship_type,
            'note': note,
            'created_at': created_at
        }

    def remove_relationship(self, from_doc_id: str, to_doc_id: str,
                           relationship_type: Optional[str] = None) -> bool:
        """
        Remove a relationship between two documents.

        Args:
            from_doc_id: Source document ID
            to_doc_id: Target document ID
            relationship_type: Optional specific relationship type to remove

        Returns:
            True if relationship was removed, False if not found
        """
        cursor = self.db_conn.cursor()

        if relationship_type:
            cursor.execute("""
                DELETE FROM document_relationships
                WHERE from_doc_id = ? AND to_doc_id = ? AND relationship_type = ?
            """, (from_doc_id, to_doc_id, relationship_type))
        else:
            cursor.execute("""
                DELETE FROM document_relationships
                WHERE from_doc_id = ? AND to_doc_id = ?
            """, (from_doc_id, to_doc_id))

        self.db_conn.commit()
        removed = cursor.rowcount > 0

        if removed:
            self.logger.info(f"Removed relationship: {from_doc_id} -> {to_doc_id}")

        return removed

    def get_relationships(self, doc_id: str, direction: str = "both") -> list[dict]:
        """
        Get all relationships for a document.

        Args:
            doc_id: Document ID
            direction: 'outgoing', 'incoming', or 'both' (default)

        Returns:
            List of relationship dictionaries
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        cursor = self.db_conn.cursor()
        relationships = []

        # Get outgoing relationships (this doc -> others)
        if direction in ["outgoing", "both"]:
            cursor.execute("""
                SELECT from_doc_id, to_doc_id, relationship_type, note, created_at
                FROM document_relationships
                WHERE from_doc_id = ?
            """, (doc_id,))

            for row in cursor.fetchall():
                relationships.append({
                    'direction': 'outgoing',
                    'from_doc_id': row[0],
                    'to_doc_id': row[1],
                    'relationship_type': row[2],
                    'note': row[3],
                    'created_at': row[4],
                    'related_doc_id': row[1]  # For convenience
                })

        # Get incoming relationships (others -> this doc)
        if direction in ["incoming", "both"]:
            cursor.execute("""
                SELECT from_doc_id, to_doc_id, relationship_type, note, created_at
                FROM document_relationships
                WHERE to_doc_id = ?
            """, (doc_id,))

            for row in cursor.fetchall():
                relationships.append({
                    'direction': 'incoming',
                    'from_doc_id': row[0],
                    'to_doc_id': row[1],
                    'relationship_type': row[2],
                    'note': row[3],
                    'created_at': row[4],
                    'related_doc_id': row[0]  # For convenience
                })

        return relationships

    def get_related_documents(self, doc_id: str, relationship_type: Optional[str] = None) -> list[dict]:
        """
        Get all documents related to a given document with full metadata.

        Args:
            doc_id: Document ID
            relationship_type: Optional filter by relationship type

        Returns:
            List of related documents with relationship info
        """
        relationships = self.get_relationships(doc_id)

        if relationship_type:
            relationships = [r for r in relationships if r['relationship_type'] == relationship_type]

        related_docs = []
        for rel in relationships:
            related_doc_id = rel['related_doc_id']
            if related_doc_id in self.documents:
                doc_meta = self.documents[related_doc_id]
                related_docs.append({
                    'doc_id': doc_meta.doc_id,
                    'title': doc_meta.title,
                    'filename': doc_meta.filename,
                    'tags': doc_meta.tags,
                    'relationship_type': rel['relationship_type'],
                    'relationship_direction': rel['direction'],
                    'note': rel['note'],
                    'created_at': rel['created_at']
                })

        return related_docs

    def get_relationship_graph(self, tags: Optional[list[str]] = None,
                              relationship_types: Optional[list[str]] = None) -> dict:
        """
        Get relationship graph data for visualization.

        Args:
            tags: Optional list of tags to filter documents
            relationship_types: Optional list of relationship types to include

        Returns:
            Dictionary with 'nodes' and 'edges' for graph visualization
        """
        cursor = self.db_conn.cursor()

        # Build WHERE clauses for filtering
        where_clauses = []
        params = []

        if relationship_types:
            placeholders = ','.join('?' * len(relationship_types))
            where_clauses.append(f"relationship_type IN ({placeholders})")
            params.extend(relationship_types)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get all relationships
        query = f"""
            SELECT from_doc_id, to_doc_id, relationship_type, note
            FROM document_relationships
            {where_sql}
        """

        cursor.execute(query, params)
        relationships = cursor.fetchall()

        # Build nodes and edges
        nodes = {}
        edges = []

        for from_id, to_id, rel_type, note in relationships:
            # Check if documents match tag filter
            if tags:
                from_doc = self.documents.get(from_id)
                to_doc = self.documents.get(to_id)
                if not from_doc or not to_doc:
                    continue
                if not any(tag in from_doc.tags for tag in tags) and \
                   not any(tag in to_doc.tags for tag in tags):
                    continue

            # Add nodes if not already present
            if from_id not in nodes:
                doc = self.documents.get(from_id)
                if doc:
                    nodes[from_id] = {
                        'id': from_id,
                        'label': doc.title,
                        'title': f"{doc.title}\n{doc.filename}\nTags: {', '.join(doc.tags)}",
                        'tags': doc.tags,
                        'chunks': doc.total_chunks
                    }

            if to_id not in nodes:
                doc = self.documents.get(to_id)
                if doc:
                    nodes[to_id] = {
                        'id': to_id,
                        'label': doc.title,
                        'title': f"{doc.title}\n{doc.filename}\nTags: {', '.join(doc.tags)}",
                        'tags': doc.tags,
                        'chunks': doc.total_chunks
                    }

            # Add edge
            if from_id in nodes and to_id in nodes:
                edges.append({
                    'from': from_id,
                    'to': to_id,
                    'type': rel_type,
                    'label': rel_type,
                    'title': note if note else rel_type
                })

        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'relationship_types': list(set(e['type'] for e in edges))
            }
        }

    def search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """Search the knowledge base using BM25 ranking or simple term frequency."""
        start_time = time.time()

        # Check cache first
        if self._search_cache is not None:
            cache_key = self._cache_key('search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            if cache_key in self._search_cache:
                results = self._search_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Cache hit for query: '{query}' ({len(results)} results, {elapsed_ms:.2f}ms)")
                return results

        self.logger.info(f"Search query: '{query}' (max_results={max_results}, tags={tags})")

        # Extract phrase queries (text in quotes)
        phrase_pattern = r'"([^"]*)"'
        phrases = re.findall(phrase_pattern, query)
        # Remove phrases from query to get regular terms
        query_without_phrases = re.sub(phrase_pattern, '', query)

        # Preprocess query terms (tokenize, remove stopwords, stem)
        query_terms_list = self._preprocess_text(query_without_phrases)
        query_terms = set(query_terms_list)
        query_terms = {term for term in query_terms if term}  # Remove empty strings

        # Check search backend preference (in priority order)
        # FTS5 can be enabled with environment variable: USE_FTS5=1
        # BM25 can be disabled with environment variable: USE_BM25=0
        use_fts5 = os.environ.get('USE_FTS5', '0') == '1'
        use_bm25 = os.environ.get('USE_BM25', '1') == '1'

        if use_fts5 and self._fts5_available():
            # Use SQLite FTS5 full-text search
            results = self._search_fts5(query, query_terms, phrases, tags, max_results)
            # Fall back to BM25/simple if FTS5 returns no results
            if not results:
                if use_bm25 and BM25_SUPPORT:
                    if self.bm25 is None:
                        self._build_bm25_index()
                    if self.bm25 is not None:
                        results = self._search_bm25(query, query_terms, phrases, tags, max_results)
                    else:
                        results = self._search_simple(query_terms, phrases, tags, max_results)
                else:
                    results = self._search_simple(query_terms, phrases, tags, max_results)
        elif use_bm25 and BM25_SUPPORT:
            # Build BM25 index if not already built
            if self.bm25 is None:
                self._build_bm25_index()

            if self.bm25 is not None:
                results = self._search_bm25(query, query_terms, phrases, tags, max_results)
            else:
                results = self._search_simple(query_terms, phrases, tags, max_results)
        else:
            results = self._search_simple(query_terms, phrases, tags, max_results)

        # Store in cache
        if self._search_cache is not None:
            cache_key = self._cache_key('search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            self._search_cache[cache_key] = results

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Search completed: {len(results)} results in {elapsed_ms:.2f}ms")

        # Log search for analytics
        search_mode = 'fts5' if (use_fts5 and self._fts5_available() and results) else ('bm25' if use_bm25 else 'simple')
        self._log_search(query, search_mode, len(results), elapsed_ms, tags)

        return results

    def _search_bm25(self, query: str, query_terms: set, phrases: list, tags: Optional[list[str]], max_results: int) -> list[dict]:
        """Search using BM25 algorithm."""
        # Preprocess query for BM25
        tokenized_query = self._preprocess_text(query)

        # Get BM25 scores for all chunks
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Build results with scores
        results = []
        for idx, chunk in enumerate(self.chunks):
            # Filter by tags if specified
            if tags:
                doc = self.documents.get(chunk.doc_id)
                if doc and not any(t in doc.tags for t in tags):
                    continue

            score = bm25_scores[idx]

            # Boost score for phrase matches
            if phrases:
                content_lower = chunk.content.lower()
                for phrase in phrases:
                    if phrase.lower() in content_lower:
                        score *= 2  # 2x boost for phrase match

            # Combine query_terms and phrases for snippet extraction
            all_terms = query_terms | {p.lower() for p in phrases}
            snippet = self._extract_snippet(chunk.content, all_terms)
            results.append({
                'doc_id': chunk.doc_id,
                'filename': chunk.filename,
                'title': chunk.title,
                'chunk_id': chunk.chunk_id,
                'score': float(score),
                'snippet': snippet,
                'word_count': chunk.word_count
            })

        # Sort by score and return top results
        # BM25 scores can be negative for very small documents
        # Accept any non-zero score (positive or moderately negative)
        # Filter only exact zeros (true non-matches)
        results = [r for r in results if abs(r['score']) > 0.0001]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def _search_simple(self, query_terms: set, phrases: list, tags: Optional[list[str]], max_results: int) -> list[dict]:
        """Simple term frequency search with fuzzy matching support (fallback when BM25 not available)."""
        results = []

        for chunk in self.chunks:
            # Filter by tags if specified
            if tags:
                doc = self.documents.get(chunk.doc_id)
                if doc and not any(t in doc.tags for t in tags):
                    continue

            content_lower = chunk.content.lower()

            # Score based on term frequency (exact matches)
            score = 0
            for term in query_terms:
                # Exact word match (higher score)
                score += len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower)) * 2
                # Partial match
                score += content_lower.count(term)

            # If fuzzy search is enabled and no exact matches, try fuzzy matching
            if self.use_fuzzy and score == 0:
                match_found, fuzzy_score = self._fuzzy_match_terms(list(query_terms), chunk.content)
                if match_found:
                    # Use fuzzy score (scaled down since it's less reliable than exact match)
                    score = fuzzy_score / 10.0

            # Boost score for phrase matches
            for phrase in phrases:
                if phrase.lower() in content_lower:
                    score += len(phrase.split()) * 10  # High boost for phrase match

            if score > 0:
                # Combine query_terms and phrases for snippet extraction
                all_terms = query_terms | {p.lower() for p in phrases}
                snippet = self._extract_snippet(chunk.content, all_terms)
                results.append({
                    'doc_id': chunk.doc_id,
                    'filename': chunk.filename,
                    'title': chunk.title,
                    'chunk_id': chunk.chunk_id,
                    'score': score,
                    'snippet': snippet,
                    'word_count': chunk.word_count
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def _search_fts5(self, query: str, query_terms: set, phrases: list,
                     tags: Optional[list[str]], max_results: int) -> list[dict]:
        """Search using SQLite FTS5 full-text search."""
        cursor = self.db_conn.cursor()

        # Build FTS5 query (FTS5 supports boolean operators and phrases natively)
        # Escape special characters that could cause syntax errors
        # Quote terms with hyphens or other special chars to treat them as phrases
        import re
        words = query.split()
        escaped_words = []
        for word in words:
            # If word contains hyphen or other special chars, quote it
            if re.search(r'[-:]', word):
                escaped_words.append(f'"{word}"')
            else:
                escaped_words.append(word)
        fts_query = ' '.join(escaped_words)

        try:
            # Execute FTS5 search with BM25 ranking
            cursor.execute("""
                SELECT
                    c.doc_id,
                    c.chunk_id,
                    c.content,
                    c.word_count,
                    c.page,
                    d.filename,
                    d.title,
                    fts.rank as score
                FROM chunks_fts5 fts
                JOIN chunks c ON c.rowid = fts.rowid
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE chunks_fts5 MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, max_results * 2))  # Get 2x for tag filtering

            results = []
            for row in cursor.fetchall():
                doc_id, chunk_id, content, word_count, page, filename, title, score = row

                # Filter by tags if specified
                if tags:
                    doc = self.documents.get(doc_id)
                    if doc and not any(t in doc.tags for t in tags):
                        continue

                # Extract snippet with highlighting
                snippet = self._extract_snippet(content, query_terms | {p.lower() for p in phrases})

                results.append({
                    'doc_id': doc_id,
                    'filename': filename,
                    'title': title,
                    'chunk_id': chunk_id,
                    'score': abs(score),  # FTS5 returns negative scores (lower is better)
                    'snippet': snippet,
                    'word_count': word_count
                })

                if len(results) >= max_results:
                    break

            return results

        except Exception as e:
            self.logger.error(f"FTS5 search failed: {e}")
            # Fall back to simple search
            return []

    def _extract_snippet(self, content: str, query_terms: set, snippet_size: int = 300) -> str:
        """
        Extract a relevant snippet from content with highlighted search terms.
        Enhanced to extract complete sentences and find regions with high term density.
        """
        content_lower = content.lower()

        # Calculate term density across content in sliding windows
        window_size = snippet_size
        best_score = 0
        best_pos = 0

        # Slide through content and score each window by term matches
        for i in range(0, max(1, len(content) - window_size + 1), 50):  # Step by 50 chars
            window = content_lower[i:i + window_size]
            score = sum(window.count(term) for term in query_terms if term)
            if score > best_score:
                best_score = score
                best_pos = i

        # If no matches found, use beginning
        if best_score == 0:
            best_pos = 0

        # Expand to complete sentences
        # Find sentence boundaries (., !, ?, or newline followed by capital letter or code)
        sentence_pattern = r'[.!?\n][\s\n]+'

        # Find start of sentence
        start = best_pos
        # Look backwards for sentence start
        for match in re.finditer(sentence_pattern, content[:best_pos]):
            start = match.end()

        # If we're too far from best_pos, adjust
        if best_pos - start > snippet_size // 2:
            start = max(0, best_pos - snippet_size // 3)

        # Find end of sentence
        end = min(len(content), start + snippet_size)
        matches = list(re.finditer(sentence_pattern, content[start:]))
        for match in matches:
            potential_end = start + match.end()
            if potential_end - start >= snippet_size * 0.8:  # At least 80% of desired size
                end = potential_end
                break

        # If we couldn't find sentence end, use hard cutoff
        if end - start < snippet_size * 0.5:
            end = min(len(content), start + snippet_size)

        snippet = content[start:end].strip()

        # Preserve code blocks (lines starting with spaces/tabs)
        # Don't break in the middle of code
        lines = snippet.split('\n')
        if lines and (lines[0].startswith('    ') or lines[0].startswith('\t')):
            # Start of snippet is code, find complete code block
            code_end = 0
            for i, line in enumerate(lines):
                if line and not line[0].isspace():
                    code_end = i
                    break
            if code_end > 0:
                snippet = '\n'.join(lines[:code_end])

        # Highlight matching terms (case-insensitive, whole words)
        for term in query_terms:
            if len(term) >= 2:  # Only highlight terms with 2+ characters
                # Use word boundary for whole word matching when possible
                pattern = re.compile(f'\\b({re.escape(term)})\\b', re.IGNORECASE)
                snippet = pattern.sub(r'**\1**', snippet)

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _load_embeddings(self):
        """Load FAISS embeddings index from disk."""
        if not self.use_semantic:
            return

        try:
            if self.embeddings_file.exists() and self.embeddings_map_file.exists():
                self.embeddings_index = faiss.read_index(str(self.embeddings_file))
                with open(self.embeddings_map_file, 'r') as f:
                    self.embeddings_doc_map = json.load(f)
                self.logger.info(f"Loaded embeddings index with {len(self.embeddings_doc_map)} vectors")
            else:
                self.logger.info("No existing embeddings found, will build on first use")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            self.embeddings_index = None
            self.embeddings_doc_map = []

    def _save_embeddings(self):
        """Save FAISS embeddings index to disk."""
        if not self.use_semantic or self.embeddings_index is None:
            return

        try:
            faiss.write_index(self.embeddings_index, str(self.embeddings_file))
            with open(self.embeddings_map_file, 'w') as f:
                json.dump(self.embeddings_doc_map, f)
            self.logger.info(f"Saved embeddings index with {len(self.embeddings_doc_map)} vectors")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")

    def _build_embeddings(self):
        """Build FAISS index from all document chunks."""
        if not self.use_semantic or self.embeddings_model is None:
            return

        self.logger.info("Building embeddings index for all chunks...")
        start_time = time.time()

        # Load all chunks
        chunks = self._get_chunks_db()
        if not chunks:
            self.logger.warning("No chunks to embed")
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.embeddings_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.embeddings_index.add(embeddings)

        # Build doc map
        self.embeddings_doc_map = [(chunk.doc_id, chunk.chunk_id) for chunk in chunks]

        # Save to disk
        self._save_embeddings()

        # Invalidate similarity cache since embeddings changed
        if self._similar_cache is not None:
            self._similar_cache.clear()
            self.logger.info("Similarity cache invalidated after rebuilding embeddings")

        elapsed = time.time() - start_time
        self.logger.info(f"Built embeddings index in {elapsed:.2f}s")

    def _add_chunks_to_embeddings(self, chunks: list[DocumentChunk]):
        """
        Incrementally add new chunks to the FAISS embeddings index.
        This avoids rebuilding the entire index from scratch.

        Args:
            chunks: List of DocumentChunk objects to add to embeddings
        """
        if not self.use_semantic or self.embeddings_model is None:
            return

        if not chunks:
            self.logger.debug("No chunks to add to embeddings")
            return

        self.logger.info(f"Adding {len(chunks)} chunks to embeddings index...")
        start_time = time.time()

        # Generate embeddings for new chunks
        texts = [chunk.content for chunk in chunks]
        new_embeddings = self.embeddings_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(new_embeddings)

        # If no existing index, create one
        if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
            self.logger.info("Creating new embeddings index")
            dimension = new_embeddings.shape[1]
            self.embeddings_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.embeddings_doc_map = []

        # Add to existing index
        self.embeddings_index.add(new_embeddings)

        # Update doc map
        for chunk in chunks:
            self.embeddings_doc_map.append((chunk.doc_id, chunk.chunk_id))

        # Save updated index to disk
        self._save_embeddings()

        # Invalidate similarity cache since embeddings changed
        if self._similar_cache is not None:
            self._similar_cache.clear()
            self.logger.debug("Similarity cache invalidated after adding chunks")

        elapsed = time.time() - start_time
        self.logger.info(f"Added {len(chunks)} chunks to embeddings index in {elapsed:.2f}s")

    def semantic_search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """
        Perform semantic search using embeddings and vector similarity.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of search results with scores
        """
        if not self.use_semantic or self.embeddings_model is None:
            raise RuntimeError("Semantic search not available. Enable with USE_SEMANTIC_SEARCH=1")

        # Build embeddings index if not yet built
        if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
            self._build_embeddings()
            if self.embeddings_index is None:
                return []

        self.logger.info(f"Semantic search query: '{query}' (max_results={max_results}, tags={tags})")
        start_time = time.time()

        # Encode query
        query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index (get more results for filtering)
        k = min(max_results * 5, len(self.embeddings_doc_map))
        scores, indices = self.embeddings_index.search(query_embedding, k)

        # Build results
        results = []
        seen_docs = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.embeddings_doc_map):
                continue

            doc_id, chunk_id = self.embeddings_doc_map[idx]

            # Filter by tags if specified
            if tags:
                doc = self.documents.get(doc_id)
                if doc and not any(t in doc.tags for t in tags):
                    continue

            # Get chunk
            chunk = self.get_chunk(doc_id, chunk_id)
            if not chunk:
                continue

            # Extract snippet (highlight query terms)
            query_terms = set(query.lower().split())
            snippet = self._extract_snippet(chunk.content, query_terms)

            doc = self.documents.get(doc_id)
            results.append({
                'doc_id': doc_id,
                'filename': chunk.filename,
                'title': chunk.title,
                'chunk_id': chunk_id,
                'score': float(score),
                'snippet': snippet,
                'word_count': chunk.word_count,
                'similarity': float(score)  # Cosine similarity score
            })

            if len(results) >= max_results:
                break

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Semantic search completed: {len(results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'semantic', len(results), elapsed, tags)

        return results

    def hybrid_search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None,
                     semantic_weight: float = 0.3) -> list[dict]:
        """
        Perform hybrid search combining FTS5 keyword search and semantic search.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by
            semantic_weight: Weight for semantic score (0.0-1.0). Default 0.3 means 70% FTS5, 30% semantic.

        Returns:
            List of search results with combined scores
        """
        if not self.use_semantic or self.embeddings_model is None:
            # Fall back to regular search if semantic not available
            self.logger.warning("Semantic search not available, falling back to FTS5/BM25")
            return self.search(query, max_results, tags)

        self.logger.info(f"Hybrid search query: '{query}' (max_results={max_results}, tags={tags}, semantic_weight={semantic_weight})")
        start_time = time.time()

        # Get results from both search methods (request 2x max_results for better merging)
        fts_results = self.search(query, max_results * 2, tags)
        semantic_results = self.semantic_search(query, max_results * 2, tags)

        # Normalize scores to 0-1 range
        # FTS5 scores: higher absolute value is better, normalize by max
        if fts_results:
            max_fts_score = max(r['score'] for r in fts_results)
            if max_fts_score > 0:
                for r in fts_results:
                    r['fts_score_normalized'] = r['score'] / max_fts_score
            else:
                for r in fts_results:
                    r['fts_score_normalized'] = 0.0

        # Semantic scores: already 0-1 cosine similarity
        for r in semantic_results:
            r['semantic_score_normalized'] = r.get('similarity', r['score'])

        # Merge results by (doc_id, chunk_id)
        merged = {}

        # Add FTS5 results
        for r in fts_results:
            key = (r['doc_id'], r['chunk_id'])
            merged[key] = {
                'doc_id': r['doc_id'],
                'filename': r['filename'],
                'title': r['title'],
                'chunk_id': r['chunk_id'],
                'snippet': r['snippet'],
                'word_count': r['word_count'],
                'fts_score': r['fts_score_normalized'],
                'semantic_score': 0.0  # Will be updated if found in semantic results
            }

        # Update with semantic results
        for r in semantic_results:
            key = (r['doc_id'], r['chunk_id'])
            if key in merged:
                merged[key]['semantic_score'] = r['semantic_score_normalized']
            else:
                # Not in FTS5 results, add it
                merged[key] = {
                    'doc_id': r['doc_id'],
                    'filename': r['filename'],
                    'title': r['title'],
                    'chunk_id': r['chunk_id'],
                    'snippet': r['snippet'],
                    'word_count': r['word_count'],
                    'fts_score': 0.0,
                    'semantic_score': r['semantic_score_normalized']
                }

        # Calculate hybrid scores
        results = []
        for item in merged.values():
            hybrid_score = (1.0 - semantic_weight) * item['fts_score'] + semantic_weight * item['semantic_score']
            results.append({
                'doc_id': item['doc_id'],
                'filename': item['filename'],
                'title': item['title'],
                'chunk_id': item['chunk_id'],
                'score': hybrid_score,
                'fts_score': item['fts_score'],
                'semantic_score': item['semantic_score'],
                'snippet': item['snippet'],
                'word_count': item['word_count']
            })

        # Sort by hybrid score (descending) and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:max_results]

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Hybrid search completed: {len(results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'hybrid', len(results), elapsed, tags)

        return results

    def faceted_search(self, query: str, facet_filters: Optional[dict[str, list[str]]] = None,
                      max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """
        Perform search with faceted filtering.

        Args:
            query: Search query
            facet_filters: Dict of facet_type -> list of values to filter by.
                          Example: {'hardware': ['SID', 'VIC-II'], 'instruction': ['LDA', 'STA']}
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of search results filtered by facets, with facets included
        """
        self.logger.info(f"Faceted search query: '{query}' (facets={facet_filters}, max_results={max_results}, tags={tags})")
        start_time = time.time()

        # First get regular search results (request more for filtering)
        results = self.search(query, max_results * 3, tags)

        # If no facet filters specified, just return regular results with facets added
        if not facet_filters:
            # Add facets to each result
            for result in results:
                result['facets'] = self._get_document_facets(result['doc_id'])
            elapsed = (time.time() - start_time) * 1000
            self.logger.info(f"Faceted search (no filters) completed: {len(results[:max_results])} results in {elapsed:.2f}ms")
            return results[:max_results]

        # Filter results by facets
        filtered_results = []
        for result in results:
            # Get document facets
            doc_facets = self._get_document_facets(result['doc_id'])

            # Check if document matches all facet filters
            matches = True
            for facet_type, required_values in facet_filters.items():
                doc_values = doc_facets.get(facet_type, set())
                # Document must have at least one of the required values for this facet type
                if not any(val in doc_values for val in required_values):
                    matches = False
                    break

            if matches:
                result['facets'] = doc_facets
                filtered_results.append(result)

                if len(filtered_results) >= max_results:
                    break

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Faceted search completed: {len(filtered_results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'faceted', len(filtered_results), elapsed, tags)

        return filtered_results

    def _get_document_facets(self, doc_id: str) -> dict[str, set[str]]:
        """Get all facets for a document from the database."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT facet_type, facet_value
            FROM document_facets
            WHERE doc_id = ?
        """, (doc_id,))

        facets = {'hardware': set(), 'instruction': set(), 'register': set()}
        for row in cursor.fetchall():
            facet_type, facet_value = row
            if facet_type in facets:
                facets[facet_type].add(facet_value)

        return facets

    def find_by_reference(self, ref_type: str, ref_value: str, max_results: int = 10) -> list[dict]:
        """
        Find documents by cross-reference type and value.

        Args:
            ref_type: Type of reference ('memory_address', 'register_offset', 'page_reference')
            ref_value: The reference value to search for (e.g., '$D020', 'VIC+0', '156')
            max_results: Maximum number of results to return

        Returns:
            List of results with document info, chunk info, and context
        """
        self.logger.info(f"Finding documents by reference: {ref_type}={ref_value}")
        start_time = time.time()

        cursor = self.db_conn.cursor()

        # Query cross_references table
        cursor.execute("""
            SELECT
                xr.doc_id,
                xr.chunk_id,
                xr.ref_type,
                xr.ref_value,
                xr.context,
                d.filename,
                d.title,
                d.tags
            FROM cross_references xr
            JOIN documents d ON xr.doc_id = d.doc_id
            WHERE xr.ref_type = ? AND xr.ref_value = ?
            ORDER BY d.title, xr.chunk_id
            LIMIT ?
        """, (ref_type, ref_value, max_results))

        results = []
        for row in cursor.fetchall():
            doc_id, chunk_id, ref_type, ref_value, context, filename, title, tags_json = row
            results.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'filename': filename,
                'title': title,
                'ref_type': ref_type,
                'ref_value': ref_value,
                'context': context,
                'tags': json.loads(tags_json) if tags_json else []
            })

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Found {len(results)} references in {elapsed:.2f}ms")

        return results

    def find_similar_documents(self, doc_id: str, chunk_id: Optional[int] = None,
                               max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """
        Find documents similar to the given document or chunk.

        Args:
            doc_id: Document ID to find similar documents for
            chunk_id: Optional chunk ID (if None, uses all chunks from document)
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of similar documents with similarity scores
        """
        # Check cache first
        if self._similar_cache is not None:
            cache_key = self._cache_key('find_similar',
                                       doc_id=doc_id,
                                       chunk_id=chunk_id,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            if cache_key in self._similar_cache:
                results = self._similar_cache[cache_key]
                self.logger.debug(f"Cache hit for find_similar: doc_id={doc_id}, chunk_id={chunk_id}")
                return results

        # Prefer semantic search if available
        if self.use_semantic and self.embeddings_index is not None:
            results = self._find_similar_semantic(doc_id, chunk_id, max_results, tags)
        else:
            # Fall back to TF-IDF similarity
            results = self._find_similar_tfidf(doc_id, chunk_id, max_results, tags)

        # Store in cache
        if self._similar_cache is not None:
            cache_key = self._cache_key('find_similar',
                                       doc_id=doc_id,
                                       chunk_id=chunk_id,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            self._similar_cache[cache_key] = results

        return results

    def _find_similar_semantic(self, doc_id: str, chunk_id: Optional[int],
                               max_results: int, tags: Optional[list[str]]) -> list[dict]:
        """Find similar documents using semantic embeddings."""
        if not self.use_semantic or self.embeddings_model is None:
            raise RuntimeError("Semantic search not available")

        # Build embeddings index if not yet built
        if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
            self._build_embeddings()
            if self.embeddings_index is None:
                return []

        # Get target embedding(s)
        if chunk_id is not None:
            # Find specific chunk's embedding
            try:
                target_idx = self.embeddings_doc_map.index((doc_id, chunk_id))
                target_embedding = self.embeddings_index.reconstruct(target_idx)
                target_embedding = target_embedding.reshape(1, -1)
            except ValueError:
                self.logger.error(f"Chunk not found in embeddings: {doc_id}, {chunk_id}")
                return []
        else:
            # Average all chunk embeddings for this document
            doc_indices = [i for i, (d, c) in enumerate(self.embeddings_doc_map) if d == doc_id]
            if not doc_indices:
                self.logger.error(f"Document not found in embeddings: {doc_id}")
                return []

            embeddings = np.array([self.embeddings_index.reconstruct(i) for i in doc_indices])
            target_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(target_embedding)

        # Search for similar chunks (get more for filtering)
        k = min(max_results * 10, len(self.embeddings_doc_map))
        scores, indices = self.embeddings_index.search(target_embedding, k)

        # Build results, aggregating by document
        doc_scores = {}  # doc_id -> max similarity score
        doc_chunks = {}  # doc_id -> list of matching chunks

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.embeddings_doc_map):
                continue

            found_doc_id, found_chunk_id = self.embeddings_doc_map[idx]

            # Skip the source document/chunk
            if found_doc_id == doc_id:
                if chunk_id is None or found_chunk_id == chunk_id:
                    continue

            # Filter by tags if specified
            if tags:
                doc = self.documents.get(found_doc_id)
                if doc and not any(t in doc.tags for t in tags):
                    continue

            # Track best score per document
            if found_doc_id not in doc_scores or score > doc_scores[found_doc_id]:
                doc_scores[found_doc_id] = float(score)
                doc_chunks[found_doc_id] = found_chunk_id

        # Sort documents by similarity score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]

        # Build result list with document info
        results = []
        for found_doc_id, similarity in sorted_docs:
            doc = self.documents.get(found_doc_id)
            if not doc:
                continue

            best_chunk_id = doc_chunks[found_doc_id]
            chunk = self.get_chunk(found_doc_id, best_chunk_id)

            results.append({
                'doc_id': found_doc_id,
                'filename': doc.filename,
                'title': doc.title,
                'chunk_id': best_chunk_id,
                'similarity': similarity,
                'snippet': chunk.content[:300] + "..." if chunk else "",
                'total_chunks': doc.total_chunks,
                'tags': doc.tags
            })

        return results

    def _find_similar_tfidf(self, doc_id: str, chunk_id: Optional[int],
                            max_results: int, tags: Optional[list[str]]) -> list[dict]:
        """Find similar documents using TF-IDF (fallback when semantic search unavailable)."""
        from collections import Counter
        import math

        # Get target document chunks
        target_chunks = self._get_chunks_db(doc_id)
        if not target_chunks:
            return []

        # If specific chunk requested, use only that chunk
        if chunk_id is not None:
            target_chunks = [c for c in target_chunks if c.chunk_id == chunk_id]
            if not target_chunks:
                return []

        # Build target document term vector
        target_terms = []
        for chunk in target_chunks:
            words = chunk.content.lower().split()
            target_terms.extend(words)

        target_tf = Counter(target_terms)
        target_length = math.sqrt(sum(count**2 for count in target_tf.values()))

        # Calculate similarity with all other documents
        doc_similarities = []

        for other_doc_id, other_doc in self.documents.items():
            # Skip source document
            if other_doc_id == doc_id:
                continue

            # Filter by tags
            if tags and not any(t in other_doc.tags for t in tags):
                continue

            # Get chunks for comparison document
            other_chunks = self._get_chunks_db(other_doc_id)
            if not other_chunks:
                continue

            # Build term vector for other document
            other_terms = []
            for chunk in other_chunks:
                words = chunk.content.lower().split()
                other_terms.extend(words)

            other_tf = Counter(other_terms)
            other_length = math.sqrt(sum(count**2 for count in other_tf.values()))

            # Calculate cosine similarity
            dot_product = sum(target_tf[term] * other_tf[term] for term in target_tf if term in other_tf)

            if target_length > 0 and other_length > 0:
                similarity = dot_product / (target_length * other_length)
            else:
                similarity = 0.0

            if similarity > 0:
                # Find best matching chunk
                best_chunk = other_chunks[0] if other_chunks else None

                doc_similarities.append({
                    'doc_id': other_doc_id,
                    'filename': other_doc.filename,
                    'title': other_doc.title,
                    'chunk_id': best_chunk.chunk_id if best_chunk else 0,
                    'similarity': similarity,
                    'snippet': best_chunk.content[:300] + "..." if best_chunk else "",
                    'total_chunks': other_doc.total_chunks,
                    'tags': other_doc.tags
                })

        # Sort by similarity and return top results
        doc_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return doc_similarities[:max_results]

    def get_chunk(self, doc_id: str, chunk_id: int) -> Optional[DocumentChunk]:
        """Get a specific chunk from database."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT c.doc_id, d.filename, d.title, c.chunk_id, c.page, c.content, c.word_count
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.doc_id = ? AND c.chunk_id = ?
        """, (doc_id, chunk_id))

        row = cursor.fetchone()
        if not row:
            return None

        return DocumentChunk(
            doc_id=row[0],
            filename=row[1],
            title=row[2],
            chunk_id=row[3],
            page=row[4],
            content=row[5],
            word_count=row[6]
        )

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document with all its chunks.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with document metadata and chunks, or None if not found
        """
        if doc_id not in self.documents:
            return None

        doc = self.documents[doc_id]
        chunks = self._get_chunks_db(doc_id)

        return {
            'doc_id': doc.doc_id,
            'title': doc.title,
            'filename': doc.filename,
            'chunks': [
                {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'page': chunk.page,
                    'word_count': chunk.word_count
                }
                for chunk in chunks
            ]
        }

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get the full content of a document from database."""
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            return None
        return "\n\n".join(c.content for c in chunks)

    def list_documents(self) -> list[DocumentMeta]:
        """List all indexed documents."""
        return list(self.documents.values())
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics from database."""
        cursor = self.db_conn.cursor()

        # Count total chunks and words
        cursor.execute("SELECT COUNT(*), SUM(word_count) FROM chunks")
        total_chunks, total_words = cursor.fetchone()
        total_chunks = total_chunks or 0
        total_words = total_words or 0

        return {
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_words': total_words,
            'file_types': list(set(d.file_type for d in self.documents.values())),
            'all_tags': list(set(t for d in self.documents.values() for t in d.tags))
        }

    def health_check(self) -> dict:
        """
        Perform health check on the knowledge base system.

        Returns:
            Dictionary with health metrics and status information
        """
        health = {
            'status': 'healthy',
            'issues': [],
            'metrics': {},
            'features': {},
            'database': {},
            'performance': {}
        }

        try:
            # Database health
            cursor = self.db_conn.cursor()

            # Check database file size
            db_file = Path(self.data_dir) / "knowledge_base.db"
            if db_file.exists():
                db_size_mb = db_file.stat().st_size / (1024 * 1024)
                health['database']['size_mb'] = round(db_size_mb, 2)

                # Warn if database is very large
                if db_size_mb > 1000:  # 1GB
                    health['issues'].append(f"Database size is large: {db_size_mb:.2f} MB")

            # Check table integrity
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            health['database']['integrity'] = integrity
            if integrity != 'ok':
                health['status'] = 'warning'
                health['issues'].append(f"Database integrity check failed: {integrity}")

            # Document and chunk counts
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            health['metrics']['documents'] = doc_count

            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            health['metrics']['chunks'] = chunk_count

            cursor.execute("SELECT SUM(word_count) FROM chunks")
            total_words = cursor.fetchone()[0] or 0
            health['metrics']['total_words'] = total_words

            # Check for orphaned chunks (shouldn't happen with foreign keys)
            cursor.execute("""
                SELECT COUNT(*) FROM chunks c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
                WHERE d.doc_id IS NULL
            """)
            orphaned = cursor.fetchone()[0]
            if orphaned > 0:
                health['status'] = 'warning'
                health['issues'].append(f"Found {orphaned} orphaned chunks")
                health['database']['orphaned_chunks'] = orphaned

            # Feature availability
            health['features']['fts5_enabled'] = os.environ.get('USE_FTS5', '0') == '1'
            health['features']['fts5_available'] = self._fts5_available()
            health['features']['semantic_search_enabled'] = self.use_semantic
            health['features']['semantic_search_available'] = self.embeddings_index is not None
            health['features']['bm25_enabled'] = os.environ.get('USE_BM25', '1') == '1'
            health['features']['query_preprocessing'] = os.environ.get('USE_QUERY_PREPROCESSING', '1') == '1'

            # Check FTS5 index if enabled
            if health['features']['fts5_enabled']:
                if not health['features']['fts5_available']:
                    health['issues'].append("FTS5 is enabled but index not found")
                    health['status'] = 'warning'

            # Check semantic search if enabled
            if health['features']['semantic_search_enabled']:
                if not health['features']['semantic_search_available']:
                    health['issues'].append("Semantic search enabled but embeddings not built")
                    health['status'] = 'warning'
                else:
                    # Check embeddings file size
                    if self.embeddings_file.exists():
                        emb_size_mb = self.embeddings_file.stat().st_size / (1024 * 1024)
                        health['features']['embeddings_size_mb'] = round(emb_size_mb, 2)
                        health['features']['embeddings_count'] = len(self.embeddings_doc_map)

            # Performance metrics
            health['performance']['cache_enabled'] = self._search_cache is not None
            if self._search_cache is not None:
                from cachetools import TTLCache
                if isinstance(self._search_cache, TTLCache):
                    health['performance']['cache_size'] = len(self._search_cache)
                    health['performance']['cache_capacity'] = self._search_cache.maxsize

            # BM25 index status
            if health['features']['bm25_enabled']:
                health['features']['bm25_index_built'] = self.bm25 is not None

            # Disk space check
            import shutil
            disk_usage = shutil.disk_usage(self.data_dir)
            free_gb = disk_usage.free / (1024 ** 3)
            health['database']['disk_free_gb'] = round(free_gb, 2)

            if free_gb < 1:  # Less than 1GB free
                health['status'] = 'warning'
                health['issues'].append(f"Low disk space: {free_gb:.2f} GB free")

            # Overall status
            if not health['issues']:
                health['status'] = 'healthy'
                health['message'] = 'All systems operational'
            else:
                health['message'] = f"System functional with {len(health['issues'])} issue(s)"

        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Health check error: {str(e)}")
            health['message'] = 'Health check failed'
            self.logger.error(f"Health check error: {e}", exc_info=True)

        return health

    def _log_search(self, query: str, search_mode: str, results_count: int, execution_time_ms: float,
                    tags: Optional[list[str]] = None, clicked_doc_id: Optional[str] = None):
        """Log a search query to the search_log table."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO search_log (query, search_mode, results_count, execution_time_ms, tags, clicked_doc_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query,
                search_mode,
                results_count,
                execution_time_ms,
                ','.join(tags) if tags else None,
                clicked_doc_id
            ))
            self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging search: {e}")

    def get_search_analytics(self, days: int = 30, limit: int = 100) -> dict:
        """
        Get search analytics and insights.

        Args:
            days: Number of days to analyze (default: 30)
            limit: Maximum number of results for top queries (default: 100)

        Returns:
            Dictionary with analytics data including:
            - total_searches: Total number of searches
            - unique_queries: Number of unique queries
            - avg_results: Average number of results per search
            - avg_execution_time_ms: Average execution time
            - top_queries: Most frequent queries
            - failed_searches: Queries with zero results
            - search_modes: Breakdown by search mode
            - popular_tags: Most frequently used tags
        """
        cursor = self.db_conn.cursor()

        # Calculate cutoff date
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        analytics = {}

        try:
            # Total searches
            cursor.execute("""
                SELECT COUNT(*) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            analytics['total_searches'] = cursor.fetchone()[0]

            # Unique queries
            cursor.execute("""
                SELECT COUNT(DISTINCT query) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            analytics['unique_queries'] = cursor.fetchone()[0]

            # Average results count
            cursor.execute("""
                SELECT AVG(results_count) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            avg_results = cursor.fetchone()[0]
            analytics['avg_results'] = round(avg_results, 2) if avg_results else 0

            # Average execution time
            cursor.execute("""
                SELECT AVG(execution_time_ms) FROM search_log
                WHERE timestamp >= ? AND execution_time_ms IS NOT NULL
            """, (cutoff_date,))
            avg_time = cursor.fetchone()[0]
            analytics['avg_execution_time_ms'] = round(avg_time, 2) if avg_time else 0

            # Top queries (most frequent)
            cursor.execute("""
                SELECT query, COUNT(*) as count, AVG(results_count) as avg_results
                FROM search_log
                WHERE timestamp >= ?
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, limit))
            analytics['top_queries'] = [
                {
                    'query': row[0],
                    'count': row[1],
                    'avg_results': round(row[2], 1) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]

            # Failed searches (zero results)
            cursor.execute("""
                SELECT query, COUNT(*) as count
                FROM search_log
                WHERE timestamp >= ? AND results_count = 0
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, min(limit, 20)))
            analytics['failed_searches'] = [
                {'query': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]

            # Search mode breakdown
            cursor.execute("""
                SELECT search_mode, COUNT(*) as count, AVG(results_count) as avg_results
                FROM search_log
                WHERE timestamp >= ?
                GROUP BY search_mode
                ORDER BY count DESC
            """, (cutoff_date,))
            analytics['search_modes'] = [
                {
                    'mode': row[0],
                    'count': row[1],
                    'avg_results': round(row[2], 1) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]

            # Popular tags
            cursor.execute("""
                SELECT tags, COUNT(*) as count
                FROM search_log
                WHERE timestamp >= ? AND tags IS NOT NULL
                GROUP BY tags
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, 20))
            tag_counts = {}
            for row in cursor.fetchall():
                tags_str = row[0]
                count = row[1]
                # Split tags and count individually
                for tag in tags_str.split(','):
                    tag = tag.strip()
                    tag_counts[tag] = tag_counts.get(tag, 0) + count

            analytics['popular_tags'] = [
                {'tag': tag, 'count': count}
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ]

        except Exception as e:
            self.logger.error(f"Error getting search analytics: {e}")
            analytics['error'] = str(e)

        return analytics

    def search_tables(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """Search for tables in documents using FTS5.

        Returns a list of table dictionaries with structure:
        {
            'doc_id': str,
            'doc_title': str,
            'table_id': int,
            'page': int,
            'markdown': str,
            'row_count': int,
            'col_count': int,
            'score': float
        }
        """
        cursor = self.db_conn.cursor()

        # Check if tables_fts exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='tables_fts'
        """)
        if not cursor.fetchone():
            self.logger.warning("tables_fts index not found, no tables to search")
            return []

        # Build FTS5 query
        # Escape special characters in query
        fts_query = query.replace('"', '""')
        fts_query = f'"{fts_query}"' if ' ' in fts_query else fts_query

        # Search tables_fts
        sql = """
            SELECT t.doc_id, d.title, t.table_id, t.page, t.markdown, t.row_count, t.col_count,
                   tables_fts.rank as score
            FROM tables_fts
            JOIN document_tables t ON tables_fts.doc_id = t.doc_id AND tables_fts.table_id = t.table_id
            JOIN documents d ON t.doc_id = d.doc_id
            WHERE tables_fts MATCH ?
        """

        # Add tag filtering if specified
        if tags:
            tag_conditions = " OR ".join(["d.tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_conditions})"

        sql += " ORDER BY score DESC LIMIT ?"

        # Execute query
        params = [fts_query]
        if tags:
            params.extend([f'%"{tag}"%' for tag in tags])
        params.append(max_results)

        cursor.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'doc_title': row[1],
                'table_id': row[2],
                'page': row[3],
                'markdown': row[4],
                'row_count': row[5],
                'col_count': row[6],
                'score': abs(row[7])  # FTS5 rank is negative, take absolute value
            })

        self.logger.info(f"Table search for '{query}' returned {len(results)} results")
        return results

    def search_code(self, query: str, max_results: int = 5, block_type: Optional[str] = None,
                    tags: Optional[list[str]] = None) -> list[dict]:
        """Search for code blocks in documents using FTS5.

        Args:
            query: Search query
            max_results: Maximum number of results
            block_type: Filter by code type ('basic', 'assembly', 'hex')
            tags: Filter by document tags

        Returns a list of code block dictionaries with structure:
        {
            'doc_id': str,
            'doc_title': str,
            'block_id': int,
            'page': int,
            'block_type': str,
            'code': str,
            'line_count': int,
            'score': float
        }
        """
        cursor = self.db_conn.cursor()

        # Check if code_fts exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='code_fts'
        """)
        if not cursor.fetchone():
            self.logger.warning("code_fts index not found, no code blocks to search")
            return []

        # Build FTS5 query
        # Escape special characters in query
        fts_query = query.replace('"', '""')
        fts_query = f'"{fts_query}"' if ' ' in fts_query else fts_query

        # Search code_fts
        sql = """
            SELECT c.doc_id, d.title, c.block_id, c.page, c.block_type, c.code, c.line_count,
                   code_fts.rank as score
            FROM code_fts
            JOIN document_code_blocks c ON code_fts.doc_id = c.doc_id AND code_fts.block_id = c.block_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE code_fts MATCH ?
        """

        # Add block type filtering if specified
        if block_type:
            sql += " AND c.block_type = ?"

        # Add tag filtering if specified
        if tags:
            tag_conditions = " OR ".join(["d.tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_conditions})"

        sql += " ORDER BY score DESC LIMIT ?"

        # Execute query
        params = [fts_query]
        if block_type:
            params.append(block_type)
        if tags:
            params.extend([f'%"{tag}"%' for tag in tags])
        params.append(max_results)

        cursor.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'doc_title': row[1],
                'block_id': row[2],
                'page': row[3],
                'block_type': row[4],
                'code': row[5],
                'line_count': row[6],
                'score': abs(row[7])  # FTS5 rank is negative, take absolute value
            })

        self.logger.info(f"Code search for '{query}' returned {len(results)} results (type={block_type or 'all'})")
        return results

    def create_backup(self, dest_dir: str, compress: bool = True) -> str:
        """
        Create full backup of knowledge base.

        Args:
            dest_dir: Destination directory for backup
            compress: Whether to compress backup to zip file (default: True)

        Returns:
            Path to backup (directory or zip file)
        """
        import shutil

        self.logger.info(f"Creating backup to {dest_dir}")
        start_time = time.time()

        # Create backup directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"kb_backup_{timestamp}"
        backup_path = Path(dest_dir) / backup_name

        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup database file
            db_source = Path(self.data_dir) / "knowledge_base.db"
            db_dest = backup_path / "knowledge_base.db"
            if db_source.exists():
                shutil.copy2(db_source, db_dest)
                self.logger.info(f"Backed up database: {db_source.stat().st_size} bytes")

            # Backup embeddings if they exist
            embeddings_path = Path(self.data_dir) / "embeddings.faiss"
            embeddings_map_path = Path(self.data_dir) / "embeddings_map.json"

            if embeddings_path.exists():
                shutil.copy2(embeddings_path, backup_path / "embeddings.faiss")
                self.logger.info(f"Backed up embeddings index: {embeddings_path.stat().st_size} bytes")

            if embeddings_map_path.exists():
                shutil.copy2(embeddings_map_path, backup_path / "embeddings_map.json")
                self.logger.info(f"Backed up embeddings map")

            # Create metadata file
            metadata = {
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'document_count': len(self.documents),
                'total_chunks': sum(doc.total_chunks for doc in self.documents.values()),
                'database_size_bytes': db_source.stat().st_size if db_source.exists() else 0,
                'has_embeddings': embeddings_path.exists(),
                'version': '2.5.0'
            }

            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Created backup metadata: {metadata}")

            # Compress if requested
            if compress:
                self.logger.info("Compressing backup...")
                zip_path = shutil.make_archive(str(backup_path), 'zip', backup_path)
                shutil.rmtree(backup_path)  # Remove uncompressed directory

                elapsed = time.time() - start_time
                self.logger.info(f"Backup completed in {elapsed:.2f}s: {zip_path}")
                return zip_path
            else:
                elapsed = time.time() - start_time
                self.logger.info(f"Backup completed in {elapsed:.2f}s: {backup_path}")
                return str(backup_path)

        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise

    def restore_from_backup(self, backup_path: str, verify: bool = True) -> dict:
        """
        Restore knowledge base from backup.

        Args:
            backup_path: Path to backup (directory or zip file)
            verify: Whether to verify backup integrity before restoring (default: True)

        Returns:
            Restoration metadata dict
        """
        import shutil
        import zipfile

        self.logger.info(f"Restoring from backup: {backup_path}")
        start_time = time.time()

        backup_path_obj = Path(backup_path)
        temp_dir = None

        try:
            # Extract if compressed
            if backup_path_obj.suffix == '.zip':
                self.logger.info("Extracting compressed backup...")
                temp_dir = Path(self.data_dir) / f"temp_restore_{int(time.time())}"
                temp_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find the backup directory inside temp_dir
                extracted_dirs = list(temp_dir.iterdir())
                if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
                    restore_from = extracted_dirs[0]
                else:
                    restore_from = temp_dir
            else:
                restore_from = backup_path_obj

            # Verify backup if requested
            if verify:
                self.logger.info("Verifying backup integrity...")

                # Check for required files
                db_file = restore_from / "knowledge_base.db"
                metadata_file = restore_from / "metadata.json"

                if not db_file.exists():
                    raise ValueError("Backup is missing database file")

                if not metadata_file.exists():
                    raise ValueError("Backup is missing metadata file")

                # Load and validate metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.logger.info(f"Backup metadata: {metadata}")

            # Close current database connection
            self.close()

            # Backup current database before overwriting (safety measure)
            current_db = Path(self.data_dir) / "knowledge_base.db"
            if current_db.exists():
                safety_backup = Path(self.data_dir) / f"knowledge_base_pre_restore_{int(time.time())}.db"
                shutil.copy2(current_db, safety_backup)
                self.logger.info(f"Created safety backup: {safety_backup}")

            # Restore database
            db_source = restore_from / "knowledge_base.db"
            db_dest = Path(self.data_dir) / "knowledge_base.db"
            shutil.copy2(db_source, db_dest)
            self.logger.info(f"Restored database: {db_source.stat().st_size} bytes")

            # Restore embeddings if they exist in backup
            embeddings_source = restore_from / "embeddings.faiss"
            embeddings_map_source = restore_from / "embeddings_map.json"

            if embeddings_source.exists():
                embeddings_dest = Path(self.data_dir) / "embeddings.faiss"
                shutil.copy2(embeddings_source, embeddings_dest)
                self.logger.info(f"Restored embeddings index")

            if embeddings_map_source.exists():
                embeddings_map_dest = Path(self.data_dir) / "embeddings_map.json"
                shutil.copy2(embeddings_map_source, embeddings_map_dest)
                self.logger.info(f"Restored embeddings map")

            # Reload knowledge base
            self.logger.info("Reloading knowledge base...")
            self._init_database()
            self._load_documents()

            # Reload embeddings if they exist
            if self.use_semantic and embeddings_source.exists():
                self._load_embeddings()

            elapsed = time.time() - start_time

            result = {
                'success': True,
                'backup_metadata': metadata,
                'restored_documents': len(self.documents),
                'elapsed_seconds': elapsed
            }

            self.logger.info(f"Restore completed in {elapsed:.2f}s: {len(self.documents)} documents")
            return result

        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            raise
        finally:
            # Cleanup temporary extraction directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def close(self):
        """Close the database connection."""
        if self.db_conn:
            self.db_conn.close()
            self.logger.info("Database connection closed")


# Initialize the MCP server
server = Server("tdz-c64-knowledge")

# Get data directory from environment or use default
DATA_DIR = os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge"))
kb = KnowledgeBase(DATA_DIR)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_docs",
            description="Search the C64 knowledge base for information. Use this to find documentation about memory maps, opcodes, BASIC commands, SID, VIC-II, CIA chips, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords or phrases)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_chunk",
            description="Get the full content of a specific document chunk. Use after search_docs to read more context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from search results"
                    },
                    "chunk_id": {
                        "type": "integer",
                        "description": "Chunk ID from search results"
                    }
                },
                "required": ["doc_id", "chunk_id"]
            }
        ),
        Tool(
            name="get_document",
            description="Get the full content of a document by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="list_docs",
            description="List all documents in the C64 knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="add_document",
            description="Add a PDF or text file to the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Full path to the PDF or text file"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title (optional, defaults to filename)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization (e.g., 'memory-map', 'sid', 'basic', 'assembly')"
                    }
                },
                "required": ["filepath"]
            }
        ),
        Tool(
            name="scrape_url",
            description="Scrape a documentation website and add all pages to the knowledge base using mdscrape. Useful for ingesting online documentation, tutorials, and technical resources. Follows links and converts HTML to searchable markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Starting URL to scrape (e.g., https://docs.example.com/api/)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Base title for scraped documents (optional, defaults to page titles)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for scraped documents (domain name auto-added)"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum link depth to follow (default: 50)",
                        "default": 50
                    },
                    "limit": {
                        "type": "string",
                        "description": "Only scrape URLs with this prefix (default: starting URL)"
                    },
                    "threads": {
                        "type": "integer",
                        "description": "Number of concurrent download threads (default: 10)",
                        "default": 10
                    },
                    "delay": {
                        "type": "integer",
                        "description": "Delay between requests in milliseconds (default: 100)",
                        "default": 100
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for main content (optional, auto-detected)"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="rescrape_document",
            description="Re-scrape a URL-sourced document to check for updates. Removes the old version and re-scrapes the original URL with the same configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to re-scrape (must be a URL-sourced document)"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="check_url_updates",
            description="Check all URL-sourced documents for updates by comparing Last-Modified headers. Detects when source URLs have been modified since last scrape.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_rescrape": {
                        "type": "boolean",
                        "description": "Automatically re-scrape changed URLs (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="remove_document",
            description="Remove a document from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to remove"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="semantic_search",
            description="Search the knowledge base using semantic/conceptual similarity (requires USE_SEMANTIC_SEARCH=1). Finds documents based on meaning, not just keywords. Example: searching for 'movable objects' can find 'sprites'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="find_similar",
            description="Find documents similar to a given document. Uses semantic embeddings if available, falls back to TF-IDF. Great for discovering related content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to find similar documents for"
                    },
                    "chunk_id": {
                        "type": "integer",
                        "description": "Optional chunk ID (if omitted, uses entire document)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="hybrid_search",
            description="Perform hybrid search combining FTS5 keyword search and semantic search. Best of both worlds - finds exact keyword matches AND conceptually related content. Returns results ranked by weighted combination of both scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    },
                    "semantic_weight": {
                        "type": "number",
                        "description": "Weight for semantic score, 0.0-1.0 (default: 0.3). Higher values favor conceptual matches, lower values favor exact keyword matches.",
                        "default": 0.3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="health_check",
            description="Perform health check on the knowledge base system. Returns status, metrics, feature availability, and any issues detected.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_updates",
            description="Check all indexed documents for updates. Detects files that have been modified since indexing and optionally re-indexes them automatically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_update": {
                        "type": "boolean",
                        "description": "Automatically re-index changed documents (default: false)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="add_documents_bulk",
            description="Add multiple documents from a directory at once. Supports glob patterns for file matching.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for documents"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (default: **/*.{pdf,txt})",
                        "default": "**/*.{pdf,txt}"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to apply to all documents (optional)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search subdirectories (default: true)",
                        "default": True
                    },
                    "skip_duplicates": {
                        "type": "boolean",
                        "description": "Skip files with duplicate content (default: true)",
                        "default": True
                    }
                },
                "required": ["directory"]
            }
        ),
        Tool(
            name="remove_documents_bulk",
            description="Remove multiple documents by IDs or tags. Useful for cleaning up the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to remove (optional)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Remove all documents with these tags (optional)"
                    }
                }
            }
        ),
        Tool(
            name="update_tags_bulk",
            description="Update tags for multiple documents in bulk. Add, remove, or replace tags for documents selected by ID or existing tags. Useful for reorganizing and categorizing the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to update (optional, use existing_tags to find documents)"
                    },
                    "existing_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Find documents with any of these tags (alternative to doc_ids)"
                    },
                    "add_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add to the documents"
                    },
                    "remove_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to remove from the documents"
                    },
                    "replace_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Replace all tags with these tags"
                    }
                }
            }
        ),
        Tool(
            name="export_documents_bulk",
            description="Export metadata for multiple documents in JSON, CSV, or Markdown format. Useful for creating reports, backups, or sharing document lists.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to export (optional, defaults to all or filtered by tags)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Export documents with any of these tags (optional)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format (default: json)",
                        "enum": ["json", "csv", "markdown"],
                        "default": "json"
                    }
                }
            }
        ),
        Tool(
            name="search_tables",
            description="Search for tables in PDF documents. Tables contain structured data like memory maps, register definitions, and command references. Returns tables in markdown format with page numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for table content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_code",
            description="Search for code blocks in documents (BASIC, Assembly, Hex dumps). Finds programming examples and code snippets. Returns code with type (basic/assembly/hex), line count, and page numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for code content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "block_type": {
                        "type": "string",
                        "description": "Filter by code type: 'basic', 'assembly', or 'hex' (optional)",
                        "enum": ["basic", "assembly", "hex"]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="faceted_search",
            description="Search with faceted filtering. Filter results by hardware components (SID, VIC-II, CIA, etc.), assembly instructions (LDA, STA, etc.), or memory registers ($D000, etc.). Great for narrowing down search results to specific technical domains.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "facet_filters": {
                        "type": "object",
                        "description": "Facet filters as dict of facet_type -> list of values. Example: {'hardware': ['SID', 'VIC-II'], 'instruction': ['LDA', 'STA']}",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by document tags (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_analytics",
            description="Get search analytics and insights. Shows popular queries, failed searches, search mode usage, and performance metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 30)",
                        "default": 30
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results for top queries (default: 100)",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="find_by_reference",
            description="Find documents by cross-reference. Search for documents containing specific memory addresses ($D020), register offsets (VIC+0, SID+4), or page references (page 156). Great for tracking how specific registers or memory locations are documented.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ref_type": {
                        "type": "string",
                        "description": "Type of reference to search for",
                        "enum": ["memory_address", "register_offset", "page_reference"]
                    },
                    "ref_value": {
                        "type": "string",
                        "description": "The reference value (e.g., '$D020', 'VIC+0', '156')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["ref_type", "ref_value"]
            }
        ),
        Tool(
            name="suggest_queries",
            description="Get autocomplete suggestions for partial queries. Suggests technical terms, memory addresses, instructions, and concepts based on indexed content. Great for discovering searchable content and learning proper terminology.",
            inputSchema={
                "type": "object",
                "properties": {
                    "partial": {
                        "type": "string",
                        "description": "Partial query string (e.g., 'VIC', 'SID', '$D0')"
                    },
                    "max_suggestions": {
                        "type": "integer",
                        "description": "Maximum number of suggestions (default: 5)",
                        "default": 5
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter",
                        "enum": ["hardware", "register", "instruction", "concept"]
                    }
                },
                "required": ["partial"]
            }
        ),
        Tool(
            name="export_results",
            description="Export search results to various formats (markdown, json, html). Use this to save search results for offline use, sharing, or creating custom reference guides.",
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "Search results array (from any search method)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Export format",
                        "enum": ["markdown", "json", "html"],
                        "default": "markdown"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query string to include in export"
                    }
                },
                "required": ["results"]
            }
        ),
        Tool(
            name="create_backup",
            description="Create a backup of the knowledge base. Backs up database and embeddings to a zip file. Use this regularly for data safety and before making major changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dest_dir": {
                        "type": "string",
                        "description": "Destination directory for backup"
                    },
                    "compress": {
                        "type": "boolean",
                        "description": "Whether to compress backup to zip file (default: true)",
                        "default": True
                    }
                },
                "required": ["dest_dir"]
            }
        ),
        Tool(
            name="restore_backup",
            description="Restore knowledge base from a backup. WARNING: This will replace the current database. A safety backup is created automatically before restoration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backup_path": {
                        "type": "string",
                        "description": "Path to backup file or directory"
                    },
                    "verify": {
                        "type": "boolean",
                        "description": "Whether to verify backup integrity before restoring (default: true)",
                        "default": True
                    }
                },
                "required": ["backup_path"]
            }
        ),
        Tool(
            name="auto_tag_document",
            description="Automatically generate tags for a document using AI analysis. Analyzes document content and suggests relevant tags across categories: hardware (sid, vic-ii), programming (assembly, basic), document type (tutorial, reference), and difficulty level (beginner, advanced). Requires LLM configuration (set LLM_PROVIDER and API key).",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to tag"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to accept tag (0.0-1.0, default: 0.7)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tags": {
                        "type": "integer",
                        "description": "Maximum number of tags to suggest (default: 10)",
                        "default": 10
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing tags; if false, replace (default: true)",
                        "default": True
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="auto_tag_all",
            description="Bulk auto-tag multiple documents using AI. Analyzes content and suggests relevant tags for all documents (or subset). Useful for initial organization or re-tagging collections. Can skip already-tagged documents and limit processing count. Requires LLM configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to accept tag (0.0-1.0, default: 0.7)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_tags": {
                        "type": "integer",
                        "description": "Maximum tags per document (default: 10)",
                        "default": 10
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing tags; if false, replace (default: true)",
                        "default": True
                    },
                    "skip_tagged": {
                        "type": "boolean",
                        "description": "If true, skip documents that already have tags (default: true)",
                        "default": True
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing or rate limiting)"
                    }
                }
            }
        ),
        Tool(
            name="summarize_document",
            description="Generate an AI-powered summary of a document. Supports brief (200-300 words), detailed (500-800 words), or bullet-point summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to summarize"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet"],
                        "description": "Type of summary: 'brief' (default), 'detailed', or 'bullet'",
                        "default": "brief"
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "If true, regenerate summary even if cached version exists (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_summary",
            description="Retrieve a cached summary of a document without regenerating it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet"],
                        "description": "Type of summary (default: 'brief')",
                        "default": "brief"
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="summarize_all",
            description="Bulk generate summaries for all documents in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["brief", "detailed", "bullet"]},
                        "description": "Types of summaries to generate (default: ['brief'])"
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "If true, regenerate all summaries (default: false)",
                        "default": False
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing)"
                    }
                }
            }
        ),
        Tool(
            name="extract_entities",
            description="Extract named entities from a C64 document using AI. Identifies hardware (SID, VIC-II, CIA, 6502), memory addresses ($D000), assembly instructions (LDA, STA), people, companies, products, and technical concepts. Returns entities with type, confidence score, and context. Requires LLM configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to extract entities from"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to include entity (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "Force re-extraction even if entities already exist (default: false)",
                        "default": False
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="list_entities",
            description="List all entities extracted from a document, grouped by type (hardware, memory_address, instruction, person, company, product, concept). Great for getting an overview of what a document covers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        },
                        "description": "Filter by entity types (optional, returns all types if omitted)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0, default: 0.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="search_entities",
            description="Search for entities across all documents using full-text search. Find all documents mentioning specific hardware, addresses, instructions, people, companies, products, or concepts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'VIC-II', 'sprite', '$D000')"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"]
                        },
                        "description": "Filter by entity types (optional)"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0, default: 0.0)",
                        "default": 0.0,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="entity_stats",
            description="Get statistics about extracted entities in the knowledge base. Shows breakdown by type, top entities, and documents with most entities. Useful for understanding the knowledge base content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": ["hardware", "memory_address", "instruction", "person", "company", "product", "concept"],
                        "description": "Filter statistics by entity type (optional, shows all types if omitted)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="extract_entities_bulk",
            description="Bulk extract entities from multiple documents in the knowledge base. Processes documents in batch, skips documents that already have entities (unless force_regenerate). Returns statistics about processed documents and extracted entities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence to include entity (0.0-1.0, default: 0.6)",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "force_regenerate": {
                        "type": "boolean",
                        "description": "Force re-extraction even if entities already exist (default: false)",
                        "default": False
                    },
                    "max_docs": {
                        "type": "integer",
                        "description": "Maximum number of documents to process (optional, for testing)",
                        "minimum": 1
                    },
                    "skip_existing": {
                        "type": "boolean",
                        "description": "Skip documents that already have entities (default: true)",
                        "default": True
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "search_docs":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")
        
        results = kb.search(query, max_results, tags)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]
        
        output = f"Found {len(results)} results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"--- Result {i} ---\n"
            output += f"Document: {r['title']} ({r['filename']})\n"
            output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
            output += f"Score: {r['score']}\n"
            output += f"Snippet:\n{r['snippet']}\n\n"
        
        return [TextContent(type="text", text=output)]

    elif name == "semantic_search":
        if not kb.use_semantic:
            return [TextContent(
                type="text",
                text="Semantic search is not enabled. Set USE_SEMANTIC_SEARCH=1 and install sentence-transformers and faiss-cpu."
            )]

        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")

        try:
            results = kb.semantic_search(query, max_results, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Semantic search error: {str(e)}")]

        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]

        output = f"Found {len(results)} semantic results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"--- Result {i} ---\n"
            output += f"Document: {r['title']} ({r['filename']})\n"
            output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
            output += f"Similarity Score: {r['similarity']:.4f}\n"
            output += f"Snippet:\n{r['snippet']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "hybrid_search":
        if not kb.use_semantic:
            return [TextContent(
                type="text",
                text="Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1 and install sentence-transformers and faiss-cpu."
            )]

        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")
        semantic_weight = arguments.get("semantic_weight", 0.3)

        try:
            results = kb.hybrid_search(query, max_results, tags, semantic_weight)
        except Exception as e:
            return [TextContent(type="text", text=f"Hybrid search error: {str(e)}")]

        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]

        output = f"Found {len(results)} hybrid results for '{query}' (semantic_weight={semantic_weight}):\n\n"
        for i, r in enumerate(results, 1):
            output += f"--- Result {i} ---\n"
            output += f"Document: {r['title']} ({r['filename']})\n"
            output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
            output += f"Hybrid Score: {r['score']:.4f} (FTS: {r['fts_score']:.4f}, Semantic: {r['semantic_score']:.4f})\n"
            output += f"Snippet:\n{r['snippet']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "faceted_search":
        query = arguments.get("query", "")
        facet_filters = arguments.get("facet_filters")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")

        try:
            results = kb.faceted_search(query, facet_filters, max_results, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Faceted search error: {str(e)}")]

        if not results:
            if facet_filters:
                return [TextContent(type="text", text=f"No results found for '{query}' with facet filters: {facet_filters}")]
            else:
                return [TextContent(type="text", text=f"No results found for: {query}")]

        # Format output
        filter_desc = f" with facets: {facet_filters}" if facet_filters else ""
        output = f"Found {len(results)} results for '{query}'{filter_desc}:\n\n"

        for i, r in enumerate(results, 1):
            output += f"--- Result {i} ---\n"
            output += f"Document: {r['title']} ({r['filename']})\n"
            output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
            output += f"Score: {r['score']:.4f}\n"

            # Show document facets
            if 'facets' in r:
                facets = r['facets']
                facet_parts = []
                if facets.get('hardware'):
                    facet_parts.append(f"Hardware: {', '.join(sorted(facets['hardware']))}")
                if facets.get('instruction'):
                    facet_parts.append(f"Instructions: {', '.join(sorted(facets['instruction']))}")
                if facets.get('register'):
                    # Show only first 5 registers if many
                    regs = sorted(facets['register'])
                    if len(regs) > 5:
                        facet_parts.append(f"Registers: {', '.join(regs[:5])} (+{len(regs)-5} more)")
                    else:
                        facet_parts.append(f"Registers: {', '.join(regs)}")
                if facet_parts:
                    output += f"Facets: {' | '.join(facet_parts)}\n"

            output += f"Snippet:\n{r['snippet']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "get_chunk":
        doc_id = arguments.get("doc_id")
        chunk_id = arguments.get("chunk_id")
        
        chunk = kb.get_chunk(doc_id, chunk_id)
        if not chunk:
            return [TextContent(type="text", text=f"Chunk not found: {doc_id}/{chunk_id}")]
        
        output = f"Document: {chunk.title}\n"
        output += f"Chunk {chunk.chunk_id} ({chunk.word_count} words):\n\n"
        output += chunk.content
        
        return [TextContent(type="text", text=output)]
    
    elif name == "get_document":
        doc_id = arguments.get("doc_id")
        
        content = kb.get_document_content(doc_id)
        if not content:
            return [TextContent(type="text", text=f"Document not found: {doc_id}")]
        
        doc = kb.documents.get(doc_id)
        output = f"Document: {doc.title}\n"
        output += f"File: {doc.filename}\n"
        output += f"{'='*50}\n\n"
        output += content
        
        return [TextContent(type="text", text=output)]
    
    elif name == "list_docs":
        docs = kb.list_documents()
        
        if not docs:
            return [TextContent(type="text", text="No documents in knowledge base. Use add_document to add PDFs or text files.")]
        
        output = f"Documents in knowledge base ({len(docs)}):\n\n"
        for doc in docs:
            output += f"- {doc.title}\n"
            output += f"  ID: {doc.doc_id}\n"
            output += f"  File: {doc.filename} ({doc.file_type})\n"
            if doc.total_pages:
                output += f"  Pages: {doc.total_pages}\n"
            output += f"  Chunks: {doc.total_chunks}\n"
            if doc.tags:
                output += f"  Tags: {', '.join(doc.tags)}\n"
            output += f"  Indexed: {doc.indexed_at}\n\n"
        
        return [TextContent(type="text", text=output)]
    
    elif name == "add_document":
        filepath = arguments.get("filepath")
        title = arguments.get("title")
        tags = arguments.get("tags", [])
        
        try:
            doc = kb.add_document(filepath, title, tags)
            output = f"Successfully added document:\n"
            output += f"  Title: {doc.title}\n"
            output += f"  ID: {doc.doc_id}\n"
            output += f"  Type: {doc.file_type}\n"
            output += f"  Chunks: {doc.total_chunks}\n"
            if doc.total_pages:
                output += f"  Pages: {doc.total_pages}\n"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error adding document: {str(e)}")]

    elif name == "scrape_url":
        url = arguments.get("url")
        title = arguments.get("title")
        tags = arguments.get("tags", [])
        depth = arguments.get("depth", 50)
        limit = arguments.get("limit")
        threads = arguments.get("threads", 10)
        delay = arguments.get("delay", 100)
        selector = arguments.get("selector")

        try:
            result = kb.scrape_url(
                url=url,
                title=title,
                tags=tags,
                depth=depth,
                limit=limit,
                threads=threads,
                delay=delay,
                selector=selector
            )

            output = f"Scraping Result:\n\n"
            output += f"Status: {result['status']}\n"
            output += f"URL: {result['url']}\n"
            output += f"Files scraped: {result['files_scraped']}\n"
            output += f"Documents added: {result['docs_added']}\n"

            if result['docs_failed'] > 0:
                output += f"Documents failed: {result['docs_failed']}\n"

            if result.get('error'):
                output += f"\nError: {result['error']}\n"

            if result.get('doc_ids'):
                output += f"\nAdded document IDs:\n"
                for doc_id in result['doc_ids'][:10]:  # Show first 10
                    doc = kb.documents.get(doc_id)
                    if doc:
                        output += f"  - {doc.title} ({doc_id})\n"
                if len(result['doc_ids']) > 10:
                    output += f"  ... and {len(result['doc_ids']) - 10} more\n"

            if result['status'] == 'success':
                output += f"\nOutput directory: {result['output_dir']}\n"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error scraping URL: {str(e)}")]

    elif name == "rescrape_document":
        doc_id = arguments.get("doc_id")

        try:
            result = kb.rescrape_document(doc_id)

            output = f"Re-scrape Result:\n\n"
            output += f"Status: {result['status']}\n"
            output += f"Original doc ID: {result['old_doc_id']}\n"
            output += f"Documents added: {result['docs_added']}\n"

            if result.get('error'):
                output += f"Error: {result['error']}\n"

            if result.get('doc_ids'):
                output += f"\nNew document IDs:\n"
                for doc_id in result['doc_ids'][:5]:
                    doc = kb.documents.get(doc_id)
                    if doc:
                        output += f"  - {doc.title} ({doc_id})\n"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error re-scraping: {str(e)}")]

    elif name == "check_url_updates":
        auto_rescrape = arguments.get("auto_rescrape", False)

        try:
            results = kb.check_url_updates(auto_rescrape)

            output = "URL Update Check:\n\n"

            if results['unchanged']:
                output += f"✓ {len(results['unchanged'])} documents unchanged\n"

            if results['changed']:
                output += f"⚠ {len(results['changed'])} documents have updates:\n"
                for doc in results['changed'][:10]:  # Show first 10
                    output += f"  - {doc['title']}\n"
                    output += f"    URL: {doc['url']}\n"
                    output += f"    Last modified: {doc['last_modified']}\n"
                if len(results['changed']) > 10:
                    output += f"  ... and {len(results['changed']) - 10} more\n"
                output += "\n"

            if results['failed']:
                output += f"✗ {len(results['failed'])} checks failed:\n"
                for doc in results['failed'][:5]:
                    output += f"  - {doc['title']}: {doc['error']}\n"
                output += "\n"

            if auto_rescrape and results['rescraped']:
                output += f"✓ {len(results['rescraped'])} documents re-scraped\n"
            elif not auto_rescrape and results['changed']:
                output += "\nTip: Use auto_rescrape=true to automatically update changed documents.\n"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error checking updates: {str(e)}")]

    elif name == "remove_document":
        doc_id = arguments.get("doc_id")
        
        if kb.remove_document(doc_id):
            return [TextContent(type="text", text=f"Successfully removed document: {doc_id}")]
        else:
            return [TextContent(type="text", text=f"Document not found: {doc_id}")]
    
    elif name == "find_similar":
        doc_id = arguments.get("doc_id")
        chunk_id = arguments.get("chunk_id")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")

        # Check if document exists
        if doc_id not in kb.documents:
            return [TextContent(type="text", text=f"Document not found: {doc_id}")]

        try:
            results = kb.find_similar_documents(doc_id, chunk_id, max_results, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Error finding similar documents: {str(e)}")]

        if not results:
            return [TextContent(type="text", text=f"No similar documents found for: {doc_id}")]

        source_doc = kb.documents[doc_id]
        if chunk_id is not None:
            output = f"Documents similar to '{source_doc.title}' (chunk {chunk_id}):\n\n"
        else:
            output = f"Documents similar to '{source_doc.title}':\n\n"

        for i, r in enumerate(results, 1):
            output += f"--- {i}. {r['title']} ({r['filename']}) ---\n"
            output += f"Doc ID: {r['doc_id']}, Similarity: {r['similarity']:.4f}\n"
            output += f"Tags: {', '.join(r['tags']) if r['tags'] else 'none'}\n"
            output += f"Snippet:\n{r['snippet']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "kb_stats":
        stats = kb.get_stats()
        output = "Knowledge Base Statistics:\n"
        output += f"  Documents: {stats['total_documents']}\n"
        output += f"  Chunks: {stats['total_chunks']}\n"
        output += f"  Total Words: {stats['total_words']:,}\n"
        output += f"  File Types: {', '.join(stats['file_types']) or 'none'}\n"
        output += f"  Tags: {', '.join(stats['all_tags']) or 'none'}\n"
        return [TextContent(type="text", text=output)]

    elif name == "health_check":
        health = kb.health_check()

        # Format output
        output = f"System Health Check\n{'='*50}\n\n"
        output += f"Status: {health['status'].upper()}\n"
        output += f"Message: {health['message']}\n\n"

        # Metrics
        if health['metrics']:
            output += "Metrics:\n"
            for key, value in health['metrics'].items():
                if isinstance(value, int):
                    output += f"  {key}: {value:,}\n"
                else:
                    output += f"  {key}: {value}\n"
            output += "\n"

        # Database
        if health['database']:
            output += "Database:\n"
            for key, value in health['database'].items():
                output += f"  {key}: {value}\n"
            output += "\n"

        # Features
        if health['features']:
            output += "Features:\n"
            for key, value in health['features'].items():
                status = "✓" if value else "✗"
                output += f"  {status} {key}: {value}\n"
            output += "\n"

        # Performance
        if health['performance']:
            output += "Performance:\n"
            for key, value in health['performance'].items():
                output += f"  {key}: {value}\n"
            output += "\n"

        # Issues
        if health['issues']:
            output += f"Issues ({len(health['issues'])}):\n"
            for i, issue in enumerate(health['issues'], 1):
                output += f"  {i}. {issue}\n"
        else:
            output += "✓ No issues detected\n"

        return [TextContent(type="text", text=output)]

    elif name == "search_analytics":
        days = arguments.get("days", 30)
        limit = arguments.get("limit", 100)

        analytics = kb.get_search_analytics(days, limit)

        if 'error' in analytics:
            return [TextContent(type="text", text=f"Error getting analytics: {analytics['error']}")]

        # Format output
        output = f"Search Analytics (Last {days} days)\n{'='*50}\n\n"

        # Overall stats
        output += "Overview:\n"
        output += f"  Total Searches: {analytics.get('total_searches', 0):,}\n"
        output += f"  Unique Queries: {analytics.get('unique_queries', 0):,}\n"
        output += f"  Avg Results per Search: {analytics.get('avg_results', 0)}\n"
        output += f"  Avg Execution Time: {analytics.get('avg_execution_time_ms', 0):.2f}ms\n\n"

        # Search modes
        if analytics.get('search_modes'):
            output += "Search Mode Usage:\n"
            for mode in analytics['search_modes']:
                output += f"  {mode['mode']}: {mode['count']:,} searches (avg {mode['avg_results']} results)\n"
            output += "\n"

        # Top queries
        if analytics.get('top_queries'):
            output += f"Top {min(10, len(analytics['top_queries']))} Most Popular Queries:\n"
            for i, query in enumerate(analytics['top_queries'][:10], 1):
                output += f"  {i}. \"{query['query']}\" - {query['count']} times (avg {query['avg_results']} results)\n"
            output += "\n"

        # Failed searches
        if analytics.get('failed_searches'):
            output += f"Top {min(10, len(analytics['failed_searches']))} Failed Searches (0 results):\n"
            for i, failed in enumerate(analytics['failed_searches'][:10], 1):
                output += f"  {i}. \"{failed['query']}\" - {failed['count']} times\n"
            output += "\n"

        # Popular tags
        if analytics.get('popular_tags'):
            output += f"Top {min(10, len(analytics['popular_tags']))} Most Used Tags:\n"
            for i, tag in enumerate(analytics['popular_tags'][:10], 1):
                output += f"  {i}. {tag['tag']}: {tag['count']} searches\n"

        return [TextContent(type="text", text=output)]

    elif name == "find_by_reference":
        ref_type = arguments.get("ref_type")
        ref_value = arguments.get("ref_value")
        max_results = arguments.get("max_results", 10)

        if not ref_type or not ref_value:
            return [TextContent(type="text", text="Error: ref_type and ref_value are required")]

        results = kb.find_by_reference(ref_type, ref_value, max_results)

        if not results:
            return [TextContent(type="text", text=f"No references found for {ref_type}={ref_value}")]

        output = f"Found {len(results)} references for {ref_type}={ref_value}:\n\n"
        for i, r in enumerate(results, 1):
            output += f"--- Reference {i} ---\n"
            output += f"Document: {r['title']} ({r['filename']})\n"
            output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
            output += f"Type: {r['ref_type']}, Value: {r['ref_value']}\n"
            output += f"Context:\n{r['context']}\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "check_updates":
        auto_update = arguments.get("auto_update", False)
        results = kb.check_all_updates(auto_update)

        output = "Document Update Check:\n\n"

        if results['unchanged']:
            output += f"✓ {len(results['unchanged'])} documents unchanged\n"

        if results['changed']:
            output += f"⚠ {len(results['changed'])} documents changed:\n"
            for doc in results['changed']:
                output += f"  - {doc['title']} ({doc['filepath']})\n"
            output += "\n"

        if results['missing']:
            output += f"✗ {len(results['missing'])} documents missing (files not found):\n"
            for doc in results['missing']:
                output += f"  - {doc['title']} ({doc['filepath']})\n"
            output += "\n"

        if auto_update and results['updated']:
            output += f"✓ {len(results['updated'])} documents re-indexed:\n"
            for doc in results['updated']:
                output += f"  - {doc['title']} ({doc['filepath']})\n"

        if not auto_update and results['changed']:
            output += "\nRun with auto_update=true to automatically re-index changed documents.\n"

        return [TextContent(type="text", text=output)]

    elif name == "add_documents_bulk":
        directory = arguments.get("directory")
        pattern = arguments.get("pattern", "**/*.{pdf,txt}")
        tags = arguments.get("tags")
        recursive = arguments.get("recursive", True)
        skip_duplicates = arguments.get("skip_duplicates", True)

        try:
            results = kb.add_documents_bulk(directory, pattern, tags, recursive, skip_duplicates)
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk add: {str(e)}")]

        output = "Bulk Document Add Results:\n\n"

        if results['added']:
            output += f"✓ {len(results['added'])} documents added:\n"
            for doc in results['added']:
                output += f"  - {doc['title']} ({doc['filename']})\n"
                output += f"    ID: {doc['doc_id']}, Chunks: {doc['chunks']}\n"
            output += "\n"

        if results['skipped']:
            output += f"⊘ {len(results['skipped'])} documents skipped (duplicates):\n"
            for doc in results['skipped']:
                output += f"  - {doc['filepath']}\n"
            output += "\n"

        if results['failed']:
            output += f"✗ {len(results['failed'])} documents failed:\n"
            for failure in results['failed']:
                output += f"  - {failure['filepath']}: {failure['error']}\n"
            output += "\n"

        output += f"Total: {len(results['added'])} added, {len(results['skipped'])} skipped, {len(results['failed'])} failed"

        return [TextContent(type="text", text=output)]

    elif name == "remove_documents_bulk":
        doc_ids = arguments.get("doc_ids")
        tags = arguments.get("tags")

        try:
            results = kb.remove_documents_bulk(doc_ids, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk remove: {str(e)}")]

        output = "Bulk Document Remove Results:\n\n"

        if results['removed']:
            output += f"✓ {len(results['removed'])} documents removed:\n"
            for doc_id in results['removed']:
                output += f"  - {doc_id}\n"
            output += "\n"

        if results['failed']:
            output += f"✗ {len(results['failed'])} documents failed to remove:\n"
            for failure in results['failed']:
                output += f"  - {failure['doc_id']}: {failure['error']}\n"
            output += "\n"

        output += f"Total: {len(results['removed'])} removed, {len(results['failed'])} failed"

        return [TextContent(type="text", text=output)]

    elif name == "update_tags_bulk":
        doc_ids = arguments.get("doc_ids")
        existing_tags = arguments.get("existing_tags")
        add_tags = arguments.get("add_tags")
        remove_tags = arguments.get("remove_tags")
        replace_tags = arguments.get("replace_tags")

        try:
            results = kb.update_tags_bulk(
                doc_ids=doc_ids,
                existing_tags=existing_tags,
                add_tags=add_tags,
                remove_tags=remove_tags,
                replace_tags=replace_tags
            )
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk tag update: {str(e)}")]

        output = "Bulk Tag Update Results:\n\n"

        if results['updated']:
            output += f"✓ {len(results['updated'])} documents updated:\n"
            for update in results['updated']:
                output += f"  - {update['doc_id']}\n"
                output += f"    Old tags: {', '.join(update['old_tags']) if update['old_tags'] else 'None'}\n"
                output += f"    New tags: {', '.join(update['new_tags']) if update['new_tags'] else 'None'}\n"
            output += "\n"

        if results['failed']:
            output += f"✗ {len(results['failed'])} documents failed to update:\n"
            for failure in results['failed']:
                output += f"  - {failure['doc_id']}: {failure['error']}\n"
            output += "\n"

        output += f"Total: {len(results['updated'])} updated, {len(results['failed'])} failed"

        return [TextContent(type="text", text=output)]

    elif name == "export_documents_bulk":
        doc_ids = arguments.get("doc_ids")
        tags = arguments.get("tags")
        format = arguments.get("format", "json")

        try:
            export_data = kb.export_documents_bulk(doc_ids=doc_ids, tags=tags, format=format)
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk export: {str(e)}")]

        # Determine document count from the export
        if format == "json":
            import json as json_lib
            doc_count = len(json_lib.loads(export_data))
        else:
            doc_count = export_data.count('\n') if format == 'csv' else export_data.count('## ') - 1

        output = f"Document Export ({format.upper()}):\n\n"
        output += f"Exported {doc_count} document(s)\n\n"
        output += "=" * 80 + "\n\n"
        output += export_data

        return [TextContent(type="text", text=output)]

    elif name == "search_tables":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        tags = arguments.get("tags")

        try:
            results = kb.search_tables(query, max_results, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching tables: {str(e)}")]

        if not results:
            return [TextContent(type="text", text=f"No tables found for query: '{query}'")]

        output = f"Found {len(results)} table(s) for '{query}':\n\n"

        for i, result in enumerate(results, 1):
            output += f"Result {i}:\n"
            output += f"  Document: {result['doc_title']}\n"
            output += f"  Page: {result['page']}\n"
            output += f"  Size: {result['row_count']} rows × {result['col_count']} columns\n"
            output += f"  Score: {result['score']:.2f}\n\n"
            output += f"Table content:\n{result['markdown']}\n\n"
            output += "-" * 80 + "\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "search_code":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        block_type = arguments.get("block_type")
        tags = arguments.get("tags")

        try:
            results = kb.search_code(query, max_results, block_type, tags)
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching code: {str(e)}")]

        if not results:
            type_filter = f" (type: {block_type})" if block_type else ""
            return [TextContent(type="text", text=f"No code blocks found for query: '{query}'{type_filter}")]

        output = f"Found {len(results)} code block(s) for '{query}':\n\n"

        for i, result in enumerate(results, 1):
            output += f"Result {i}:\n"
            output += f"  Document: {result['doc_title']}\n"
            output += f"  Page: {result['page'] or 'N/A'}\n"
            output += f"  Type: {result['block_type']}\n"
            output += f"  Lines: {result['line_count']}\n"
            output += f"  Score: {result['score']:.2f}\n\n"
            output += f"Code:\n```{result['block_type']}\n{result['code']}\n```\n\n"
            output += "-" * 80 + "\n\n"

        return [TextContent(type="text", text=output)]

    elif name == "suggest_queries":
        partial = arguments.get("partial", "")
        max_suggestions = arguments.get("max_suggestions", 5)
        category = arguments.get("category")

        if not partial:
            return [TextContent(type="text", text="Error: partial query string is required")]

        try:
            suggestions = kb.get_query_suggestions(partial, max_suggestions, category)
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting suggestions: {str(e)}")]

        if not suggestions:
            return [TextContent(type="text", text=f"No suggestions found for: '{partial}'")]

        output = f"Query suggestions for '{partial}':\n\n"
        for i, sug in enumerate(suggestions, 1):
            output += f"{i}. {sug['term']} ({sug['category']}) - used {sug['frequency']} times\n"

        return [TextContent(type="text", text=output)]

    elif name == "export_results":
        results = arguments.get("results", [])
        format = arguments.get("format", "markdown")
        query = arguments.get("query")

        if not results:
            return [TextContent(type="text", text="Error: results array is required")]

        try:
            exported = kb.export_search_results(results, format, query)

            # Return the exported content
            output = f"Search results exported to {format} format:\n\n"
            output += "=" * 80 + "\n\n"
            output += exported

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error exporting results: {str(e)}")]

    elif name == "create_backup":
        dest_dir = arguments.get("dest_dir")
        compress = arguments.get("compress", True)

        if not dest_dir:
            return [TextContent(type="text", text="Error: dest_dir is required")]

        try:
            backup_path = kb.create_backup(dest_dir, compress)

            output = f"✓ Backup created successfully!\n\n"
            output += f"Location: {backup_path}\n"
            output += f"Format: {'Compressed (ZIP)' if compress else 'Uncompressed directory'}\n\n"
            output += f"The backup includes:\n"
            output += f"- Database ({len(kb.documents)} documents)\n"
            output += f"- Embeddings (if available)\n"
            output += f"- Metadata file with timestamp and version info\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error creating backup: {str(e)}")]

    elif name == "restore_backup":
        backup_path = arguments.get("backup_path")
        verify = arguments.get("verify", True)

        if not backup_path:
            return [TextContent(type="text", text="Error: backup_path is required")]

        try:
            result = kb.restore_from_backup(backup_path, verify)

            output = f"✓ Restore completed successfully!\n\n"
            output += f"Backup source: {backup_path}\n"
            output += f"Documents restored: {result['restored_documents']}\n"
            output += f"Time elapsed: {result['elapsed_seconds']:.2f}s\n\n"

            if 'backup_metadata' in result:
                metadata = result['backup_metadata']
                output += f"Backup info:\n"
                output += f"- Created: {metadata.get('created_at', 'Unknown')}\n"
                output += f"- Version: {metadata.get('version', 'Unknown')}\n"
                output += f"- Original document count: {metadata.get('document_count', 'Unknown')}\n"

            output += f"\nNote: A safety backup was created before restoration."

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error restoring backup: {str(e)}")]

    elif name == "auto_tag_document":
        doc_id = arguments.get("doc_id")
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        max_tags = arguments.get("max_tags", 10)
        append = arguments.get("append", True)

        if not doc_id:
            return [TextContent(type="text", text="Error: doc_id is required")]

        try:
            result = kb.auto_tag_document(
                doc_id,
                confidence_threshold=confidence_threshold,
                max_tags=max_tags,
                append=append
            )

            output = f"✓ Auto-tagged document successfully!\n\n"
            output += f"**Document:** {result['doc_title']}\n"
            output += f"**Document ID:** {doc_id}\n\n"

            output += f"**Applied Tags ({len(result['applied_tags'])}):**\n"
            for tag_info in result['suggested_tags']:
                if tag_info['tag'] in result['applied_tags']:
                    output += f"  - {tag_info['tag']} (confidence: {tag_info['confidence']:.2f})\n"
                    output += f"    Reason: {tag_info.get('reason', 'N/A')}\n"

            if result['skipped_tags']:
                output += f"\n**Skipped Tags (below {confidence_threshold} threshold):**\n"
                for tag_info in result['suggested_tags']:
                    if tag_info['tag'] in result['skipped_tags']:
                        output += f"  - {tag_info['tag']} (confidence: {tag_info['confidence']:.2f})\n"

            output += f"\n**Tag Summary:**\n"
            output += f"  - Existing tags: {', '.join(result['existing_tags']) if result['existing_tags'] else 'None'}\n"
            output += f"  - New tags: {', '.join(result['new_tags'])}\n"
            output += f"  - Total tags added: {len(set(result['new_tags']) - set(result['existing_tags']))}\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error auto-tagging document: {str(e)}\n\nNote: Auto-tagging requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    elif name == "auto_tag_all":
        confidence_threshold = arguments.get("confidence_threshold", 0.7)
        max_tags = arguments.get("max_tags", 10)
        append = arguments.get("append", True)
        skip_tagged = arguments.get("skip_tagged", True)
        max_docs = arguments.get("max_docs")

        try:
            results = kb.auto_tag_all_documents(
                confidence_threshold=confidence_threshold,
                max_tags=max_tags,
                append=append,
                skip_tagged=skip_tagged,
                max_docs=max_docs
            )

            output = f"✓ Bulk auto-tagging complete!\n\n"
            output += f"**Statistics:**\n"
            output += f"  - Documents processed: {results['processed']}\n"
            output += f"  - Documents skipped: {results['skipped']}\n"
            output += f"  - Documents failed: {results['failed']}\n"
            output += f"  - Total tags added: {results['total_tags_added']}\n\n"

            if results['processed'] > 0:
                output += f"**Sample Results (first 5):**\n"
                for i, result in enumerate(results['results'][:5], 1):
                    if 'error' in result:
                        output += f"\n{i}. {result['doc_id']} - ERROR: {result['error']}\n"
                    else:
                        output += f"\n{i}. {result.get('doc_title', 'Unknown')}\n"
                        output += f"   - Applied: {', '.join(result['applied_tags']) if result['applied_tags'] else 'None'}\n"
                        output += f"   - Total tags: {len(result['new_tags'])}\n"

                if results['processed'] > 5:
                    output += f"\n... and {results['processed'] - 5} more documents\n"

            if results['failed'] > 0:
                output += f"\n**Warning:** {results['failed']} documents failed to process. Check logs for details.\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk auto-tagging: {str(e)}\n\nNote: Auto-tagging requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    elif name == "summarize_document":
        doc_id = arguments.get("doc_id")
        summary_type = arguments.get("summary_type", "brief")
        force_regenerate = arguments.get("force_regenerate", False)

        if not doc_id:
            return [TextContent(type="text", text="Error: doc_id is required")]

        try:
            summary = kb.generate_summary(
                doc_id,
                summary_type=summary_type,
                force_regenerate=force_regenerate
            )

            output = f"✓ Summary generated ({summary_type})\n\n"
            output += f"**Document:** {kb.documents[doc_id].title}\n\n"
            output += f"**Summary ({summary_type}):**\n\n"
            output += summary

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating summary: {str(e)}\n\nNote: Summarization requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    elif name == "get_summary":
        doc_id = arguments.get("doc_id")
        summary_type = arguments.get("summary_type", "brief")

        if not doc_id:
            return [TextContent(type="text", text="Error: doc_id is required")]

        try:
            summary = kb.get_summary(doc_id, summary_type)

            if not summary:
                return [TextContent(type="text", text=f"No cached summary found for {doc_id} ({summary_type}). Use 'summarize_document' to generate one.")]

            output = f"✓ Cached summary retrieved ({summary_type})\n\n"
            output += f"**Document:** {kb.documents[doc_id].title}\n\n"
            output += f"**Summary ({summary_type}):**\n\n"
            output += summary

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error retrieving summary: {str(e)}")]

    elif name == "summarize_all":
        summary_types = arguments.get("summary_types", ["brief"])
        force_regenerate = arguments.get("force_regenerate", False)
        max_docs = arguments.get("max_docs")

        try:
            results = kb.generate_summary_all(
                summary_types=summary_types,
                force_regenerate=force_regenerate,
                max_docs=max_docs
            )

            output = f"✓ Bulk summarization complete!\n\n"
            output += f"**Statistics:**\n"
            output += f"  - Documents processed: {results['processed']}\n"
            output += f"  - Documents failed: {results['failed']}\n"
            output += f"  - Total summaries generated: {results['total_summaries']}\n"
            output += f"  - By type:\n"
            for summary_type, count in results['by_type'].items():
                output += f"    - {summary_type}: {count}\n"

            output += f"\n**Sample Results (first 3):**\n"
            for i, result in enumerate(results['results'][:3], 1):
                output += f"\n{i}. {result['title']}\n"
                for summary_type, summary_result in result['summaries'].items():
                    if summary_result['success']:
                        output += f"   - {summary_type}: {summary_result['word_count']} words\n"
                    else:
                        output += f"   - {summary_type}: ERROR - {summary_result['error']}\n"

            if results['processed'] > 3:
                output += f"\n... and {results['processed'] - 3} more documents\n"

            if results['failed'] > 0:
                output += f"\n**Warning:** {results['failed']} documents failed to process. Check logs for details.\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk summarization: {str(e)}\n\nNote: Summarization requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    elif name == "extract_entities":
        doc_id = arguments.get("doc_id")
        confidence_threshold = arguments.get("confidence_threshold", 0.6)
        force_regenerate = arguments.get("force_regenerate", False)

        if not doc_id:
            return [TextContent(type="text", text="Error: doc_id is required")]

        try:
            result = kb.extract_entities(
                doc_id,
                confidence_threshold=confidence_threshold,
                force_regenerate=force_regenerate
            )

            output = f"✓ Entity extraction complete!\n\n"
            output += f"**Document:** {result['doc_title']}\n"
            output += f"**Document ID:** {doc_id}\n"
            output += f"**Entities Found:** {result['entity_count']}\n\n"

            # Group by entity type
            if result['entities']:
                for entity_type in sorted(result['types'].keys()):
                    entities_of_type = [e for e in result['entities'] if e['entity_type'] == entity_type]
                    output += f"**{entity_type.upper().replace('_', ' ')}** ({len(entities_of_type)}):\n"

                    # Show first 5 of each type
                    for entity in entities_of_type[:5]:
                        output += f"  - **{entity['entity_text']}** (confidence: {entity['confidence']:.2f}"
                        if entity.get('occurrence_count', 1) > 1:
                            output += f", occurs {entity['occurrence_count']}x"
                        output += ")\n"
                        if entity.get('context'):
                            context = entity['context'][:80] + "..." if len(entity['context']) > 80 else entity['context']
                            output += f"    *{context}*\n"

                    if len(entities_of_type) > 5:
                        output += f"  ... and {len(entities_of_type) - 5} more\n"
                    output += "\n"

                output += f"Use `list_entities` tool to see all entities with filtering options.\n"
            else:
                output += "No entities found with the current confidence threshold.\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error extracting entities: {str(e)}\n\nNote: Entity extraction requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    elif name == "list_entities":
        doc_id = arguments.get("doc_id")
        entity_types = arguments.get("entity_types")
        min_confidence = arguments.get("min_confidence", 0.0)

        if not doc_id:
            return [TextContent(type="text", text="Error: doc_id is required")]

        try:
            result = kb.get_entities(
                doc_id,
                entity_types=entity_types,
                min_confidence=min_confidence
            )

            output = f"**Entities for Document:** {result['doc_title']}\n"
            output += f"**Document ID:** {doc_id}\n"

            # Show filters if applied
            if entity_types or min_confidence > 0:
                output += f"**Filters:** "
                filters = []
                if entity_types:
                    filters.append(f"types={', '.join(entity_types)}")
                if min_confidence > 0:
                    filters.append(f"min_confidence={min_confidence}")
                output += ', '.join(filters) + "\n"

            output += f"**Total Entities:** {result['entity_count']}\n\n"

            if result['entities']:
                # Group by type
                for entity_type in sorted(result['types'].keys()):
                    entities_of_type = [e for e in result['entities'] if e['entity_type'] == entity_type]
                    output += f"**{entity_type.upper().replace('_', ' ')}** ({len(entities_of_type)}):\n"

                    for entity in entities_of_type:
                        output += f"  - **{entity['entity_text']}** (conf: {entity['confidence']:.2f}"
                        if entity.get('occurrence_count', 1) > 1:
                            output += f", {entity['occurrence_count']}x"
                        output += ")\n"
                        if entity.get('context'):
                            context = entity['context'][:100] + "..." if len(entity['context']) > 100 else entity['context']
                            output += f"    *{context}*\n"

                    output += "\n"
            else:
                output += "No entities found. Use `extract_entities` tool to extract entities first.\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error listing entities: {str(e)}")]

    elif name == "search_entities":
        query = arguments.get("query")
        entity_types = arguments.get("entity_types")
        min_confidence = arguments.get("min_confidence", 0.0)
        max_results = arguments.get("max_results", 20)

        if not query:
            return [TextContent(type="text", text="Error: query is required")]

        try:
            result = kb.search_entities(
                query,
                entity_types=entity_types,
                min_confidence=min_confidence,
                max_results=max_results
            )

            output = f"**Entity Search Results for:** {result['query']}\n"
            output += f"**Total Matches:** {result['total_matches']}\n"

            # Show filters if applied
            if entity_types or min_confidence > 0:
                output += f"**Filters:** "
                filters = []
                if entity_types:
                    filters.append(f"types={', '.join(entity_types)}")
                if min_confidence > 0:
                    filters.append(f"min_confidence={min_confidence}")
                output += ', '.join(filters) + "\n"

            output += f"**Documents Found:** {len(result['documents'])}\n\n"

            if result['documents']:
                for doc in result['documents']:
                    output += f"**{doc['doc_title']}** ({doc['doc_id']})\n"
                    output += f"  Matches: {doc['match_count']}\n"

                    # Show first 3 matches per document
                    for match in doc['matches'][:3]:
                        output += f"  - **{match['entity_text']}** ({match['entity_type']}, conf: {match['confidence']:.2f}"
                        if match.get('occurrence_count', 1) > 1:
                            output += f", {match['occurrence_count']}x"
                        output += ")\n"
                        if match.get('context'):
                            context = match['context'][:80] + "..." if len(match['context']) > 80 else match['context']
                            output += f"    *{context}*\n"

                    if doc['match_count'] > 3:
                        output += f"  ... and {doc['match_count'] - 3} more matches\n"
                    output += "\n"
            else:
                output += "No entities found matching your query.\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error searching entities: {str(e)}")]

    elif name == "entity_stats":
        entity_type = arguments.get("entity_type")

        try:
            result = kb.get_entity_stats(entity_type=entity_type)

            output = f"**Entity Statistics**\n"
            if entity_type:
                output += f"**Type Filter:** {entity_type}\n"
            output += "\n"

            output += f"**Total Entities:** {result['total_entities']}\n"
            output += f"**Documents with Entities:** {result['total_documents_with_entities']}\n\n"

            # Breakdown by type
            if result['by_type']:
                output += f"**Entities by Type:**\n"
                for ent_type, count in sorted(result['by_type'].items(), key=lambda x: x[1], reverse=True):
                    output += f"  - {ent_type.replace('_', ' ')}: {count}\n"
                output += "\n"

            # Top entities
            if result['top_entities']:
                output += f"**Top Entities (by document count):**\n"
                for i, entity in enumerate(result['top_entities'][:10], 1):
                    output += f"{i}. **{entity['entity_text']}** ({entity['entity_type']})\n"
                    output += f"   - Found in {entity['document_count']} document(s)\n"
                    output += f"   - Total occurrences: {entity['total_occurrences']}\n"
                    output += f"   - Avg confidence: {entity['avg_confidence']:.2f}\n"
                output += "\n"

            # Documents with most entities
            if result['documents_with_most_entities']:
                output += f"**Documents with Most Entities:**\n"
                for i, doc in enumerate(result['documents_with_most_entities'], 1):
                    output += f"{i}. **{doc['doc_title']}**: {doc['entity_count']} entities\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting entity stats: {str(e)}")]

    elif name == "extract_entities_bulk":
        confidence_threshold = arguments.get("confidence_threshold", 0.6)
        force_regenerate = arguments.get("force_regenerate", False)
        max_docs = arguments.get("max_docs")
        skip_existing = arguments.get("skip_existing", True)

        try:
            result = kb.extract_entities_bulk(
                confidence_threshold=confidence_threshold,
                force_regenerate=force_regenerate,
                max_docs=max_docs,
                skip_existing=skip_existing
            )

            output = f"**Bulk Entity Extraction Complete**\n\n"
            output += f"**Processed:** {result['processed']} documents\n"
            output += f"**Skipped:** {result['skipped']} documents (already have entities)\n"
            output += f"**Failed:** {result['failed']} documents\n"
            output += f"**Total Entities Extracted:** {result['total_entities']}\n\n"

            if result['by_type']:
                output += f"**Entities by Type:**\n"
                for ent_type, count in sorted(result['by_type'].items(), key=lambda x: x[1], reverse=True):
                    output += f"  - {ent_type.replace('_', ' ')}: {count}\n"
                output += "\n"

            # Show sample results
            if result['results']:
                output += f"**Sample Results (first 10):**\n"
                for i, doc_result in enumerate(result['results'][:10], 1):
                    status_emoji = "✓" if doc_result['status'] == 'success' else "⊗" if doc_result['status'] == 'failed' else "⊘"
                    output += f"{i}. {status_emoji} **{doc_result['title']}**"
                    if doc_result['status'] == 'success':
                        output += f" - {doc_result['entity_count']} entities"
                    elif doc_result['status'] == 'skipped':
                        output += f" - skipped ({doc_result['entity_count']} entities)"
                    elif doc_result['status'] == 'failed':
                        output += f" - ERROR: {doc_result.get('error', 'unknown error')}"
                    output += "\n"

            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error in bulk entity extraction: {str(e)}\n\nNote: Entity extraction requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    resources = []
    for doc in kb.list_documents():
        resources.append(Resource(
            uri=f"c64kb://{doc.doc_id}",
            name=doc.title,
            description=f"{doc.file_type.upper()} document with {doc.total_chunks} chunks",
            mimeType="text/plain"
        ))
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri.startswith("c64kb://"):
        doc_id = uri[8:]
        content = kb.get_document_content(doc_id)
        if content:
            return content
    return f"Resource not found: {uri}"


async def main():
    """Run the MCP server."""
    # Log version information
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"Starting {get_full_version_string()}")
    logger.info(f"Build Date: {__build_date__}")
    logger.info("=" * 60)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
