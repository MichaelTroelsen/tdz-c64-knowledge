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
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

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
                    creation_date TEXT
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

            self.db_conn.commit()
            self.logger.info("Database schema created successfully (with FTS5)")
        else:
            self.logger.info("Using existing database")

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
                    creation_date=row[12]
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

    def _add_document_db(self, doc_meta: DocumentMeta, chunks: list[DocumentChunk]):
        """Add a document and its chunks to the database using a transaction."""
        cursor = self.db_conn.cursor()

        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            # Insert document
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages, total_chunks,
                 indexed_at, tags, author, subject, creator, creation_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                doc_meta.creation_date
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
    
    def _generate_doc_id(self, filepath: str) -> str:
        """Generate a unique document ID."""
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
    
    def _extract_pdf_text(self, filepath: str) -> tuple[str, int, dict]:
        """Extract text from PDF, returns (text, page_count, metadata)."""
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install pypdf: pip install pypdf")

        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

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
    
    def add_document(self, filepath: str, title: Optional[str] = None, tags: Optional[list[str]] = None) -> DocumentMeta:
        """Add a document to the knowledge base."""
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

        doc_id = self._generate_doc_id(filepath)
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
            creation_date=pdf_metadata.get('creation_date')
        )

        # Add to database
        self._add_document_db(doc_meta, chunks)
        self.documents[doc_id] = doc_meta

        # Invalidate BM25 index (will be rebuilt on next search)
        self.bm25 = None

        # Invalidate embeddings (will be rebuilt on next semantic search)
        if self.use_semantic:
            self.embeddings_index = None
            self.embeddings_doc_map = []

        self.logger.info(f"Successfully indexed document {doc_id}: {filename} ({len(chunks)} chunks)")
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

            self.logger.info(f"Successfully removed document {doc_id}: {filename}")

        return success
    
    def search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """Search the knowledge base using BM25 ranking or simple term frequency."""
        start_time = time.time()
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
        """Simple term frequency search (fallback when BM25 not available)."""
        results = []

        for chunk in self.chunks:
            # Filter by tags if specified
            if tags:
                doc = self.documents.get(chunk.doc_id)
                if doc and not any(t in doc.tags for t in tags):
                    continue

            content_lower = chunk.content.lower()

            # Score based on term frequency
            score = 0
            for term in query_terms:
                # Exact word match (higher score)
                score += len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower)) * 2
                # Partial match
                score += content_lower.count(term)

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
        fts_query = query

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
        """Extract a relevant snippet from content with highlighted search terms."""
        content_lower = content.lower()

        # Find the first occurrence of any query term
        best_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        if best_pos == len(content):
            best_pos = 0

        # Extract snippet around that position
        start = max(0, best_pos - snippet_size // 2)
        end = min(len(content), start + snippet_size)

        snippet = content[start:end]

        # Highlight matching terms (case-insensitive)
        for term in query_terms:
            if len(term) >= 2:  # Only highlight terms with 2+ characters
                # Use regex to match whole words and preserve case
                pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
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
            name="kb_stats",
            description="Get statistics about the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
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
    
    elif name == "kb_stats":
        stats = kb.get_stats()
        output = "Knowledge Base Statistics:\n"
        output += f"  Documents: {stats['total_documents']}\n"
        output += f"  Chunks: {stats['total_chunks']}\n"
        output += f"  Total Words: {stats['total_words']:,}\n"
        output += f"  File Types: {', '.join(stats['file_types']) or 'none'}\n"
        output += f"  Tags: {', '.join(stats['all_tags']) or 'none'}\n"
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
