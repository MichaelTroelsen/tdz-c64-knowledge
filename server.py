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
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

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
        self.logger.info(f"Initializing KnowledgeBase at {data_dir}")

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
        if self.use_ocr:
            # Check if Tesseract is installed
            try:
                pytesseract.get_tesseract_version()
                self.logger.info("OCR enabled (Tesseract found)")
                if self.poppler_path:
                    self.logger.info(f"Using Poppler from: {self.poppler_path}")
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
                    file_hash TEXT
                )
            """)

            # Create indexes on documents table
            cursor.execute("CREATE INDEX idx_documents_filepath ON documents(filepath)")
            cursor.execute("CREATE INDEX idx_documents_file_type ON documents(file_type)")

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

            self.db_conn.commit()
            self.logger.info("Database schema created successfully (with FTS5, tables, and code blocks)")
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
                    file_hash=row[14] if len(row) > 14 else None
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
                         tables: Optional[list[dict]] = None, code_blocks: Optional[list[dict]] = None):
        """Add a document, chunks, tables, and code blocks to the database using a transaction."""
        cursor = self.db_conn.cursor()

        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            # Insert document
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages, total_chunks,
                 indexed_at, tags, author, subject, creator, creation_date, file_mtime, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                doc_meta.file_hash
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
        if not BM25_SUPPORT:
            self.logger.info("BM25 index not built (no support)")
            return

        # Lazy load all chunks from database if not already in memory
        if not self.chunks:
            self.logger.info("Loading all chunks from database for BM25 index")
            self.chunks = self._get_chunks_db()

        if not self.chunks:
            self.logger.info("BM25 index not built (no chunks)")
            return

        # Tokenize all chunk content with preprocessing if enabled
        tokenized_corpus = [self._preprocess_text(chunk.content) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        preprocessing_status = "with preprocessing" if self.use_preprocessing else "without preprocessing"
        self.logger.info(f"Built BM25 index with {len(self.chunks)} chunks ({preprocessing_status})")

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

        self.logger.info(f"Detected {len(code_blocks)} code blocks ({sum(1 for b in code_blocks if b['block_type']=='basic')} BASIC, {sum(1 for b in code_blocks if b['block_type']=='assembly')} Assembly, {sum(1 for b in code_blocks if b['block_type']=='hex')} Hex)")
        return code_blocks

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
        if self.allowed_dirs:
            # Check if path is within any allowed directory
            is_allowed = any(
                resolved_path.is_relative_to(allowed_dir)
                for allowed_dir in self.allowed_dirs
            )
            if not is_allowed:
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

        # Generate content-based doc_id for deduplication
        doc_id = self._generate_doc_id(filepath, text)

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

        # Add to database (with tables and code blocks)
        self._add_document_db(doc_meta, chunks, tables=tables, code_blocks=code_blocks)
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

        # Invalidate embeddings (will be rebuilt on next semantic search)
        if self.use_semantic:
            self.embeddings_index = None
            self.embeddings_doc_map = []

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

    def add_documents_bulk(self, directory: str, pattern: str = "**/*.{pdf,txt}",
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

        for idx, file_path in enumerate(files, 1):
            if not file_path.is_file():
                continue

            # Report progress: Processing file
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation="add_documents_bulk",
                    current=idx,
                    total=len(files),
                    message=f"Processing file {idx}/{len(files)}",
                    item=str(file_path.name)
                ))

            try:
                # Generate title from filename
                title = file_path.stem

                doc = self.add_document(str(file_path), title=title, tags=tags)

                # Check if this was a duplicate
                if skip_duplicates:
                    # Check if any previously added doc has same doc_id
                    is_duplicate = any(d['doc_id'] == doc.doc_id for d in results['added'])
                    if is_duplicate:
                        results['skipped'].append({
                            'filepath': str(file_path),
                            'reason': 'duplicate content',
                            'doc_id': doc.doc_id
                        })
                        continue

                results['added'].append({
                    'doc_id': doc.doc_id,
                    'filepath': str(file_path),
                    'title': title,
                    'chunks': doc.total_chunks
                })

            except Exception as e:
                results['failed'].append({
                    'filepath': str(file_path),
                    'error': str(e)
                })
                self.logger.error(f"Failed to add {file_path}: {e}")

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
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
