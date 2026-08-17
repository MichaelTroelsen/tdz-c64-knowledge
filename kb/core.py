"""Database connections, schema creation and migration, the document cache, and object lifecycle.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from dataclasses import asdict
from datetime import datetime
from features import ANOMALY_SUPPORT
from features import AnomalyDetector
from features import CACHE_SUPPORT
from features import FUZZY_SUPPORT
from features import NLTK_SUPPORT
from features import OCR_SUPPORT
from features import SEMANTIC_SUPPORT
from features import TTLCache
from features import pytesseract
from models import DocumentChunk
from models import DocumentMeta
from models import KnowledgeBaseError
from pathlib import Path
from typing import Optional
from util import _cross_process_lock
from version import __build_date__
from version import get_full_version_string
import hashlib
import json
import logging
import os
import queue
import sqlite3
import sys
import threading


class CoreMixin:

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging with UTF-8 encoding to handle Unicode characters
        log_file = self.data_dir / "server.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stderr)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Configure StreamHandler encoding for Windows console compatibility
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                # Set UTF-8 encoding, with error handling for Windows console
                handler.stream.reconfigure(encoding='utf-8', errors='replace') if hasattr(handler.stream, 'reconfigure') else None
        self.logger.info("=" * 60)
        self.logger.info(f"{get_full_version_string()}")
        self.logger.info(f"Build Date: {__build_date__}")
        self.logger.info("=" * 60)
        self.logger.info(f"Initializing KnowledgeBase at {data_dir}")

        # Thread safety for parallel document processing
        self._lock = threading.Lock()

        # Database setup
        self.db_file = self.data_dir / "knowledge_base.db"
        # One sqlite3 connection per thread - see the db_conn property. A single
        # shared connection was read *and committed* from the event-loop thread
        # (_log_mcp_call on every MCP call), the background extraction worker,
        # and asyncio.to_thread/ThreadPoolExecutor workers. sqlite3
        # transactions are per-connection, so any thread's commit() committed
        # whatever transaction another thread had only half-built: a trivial
        # kb_stats call could commit a partial bulk ingest, whose subsequent
        # rollback then rolled back nothing.
        self._thread_local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()  # guards _all_conns only, never queries
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
            self._semantic_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self._hybrid_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
            self._faceted_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)

            # Query embedding cache for semantic search (1 hour TTL for embeddings)
            embedding_cache_ttl = int(os.getenv('EMBEDDING_CACHE_TTL', '3600'))  # 1 hour
            self._embedding_cache = TTLCache(maxsize=cache_size, ttl=embedding_cache_ttl)

            # Entity extraction result cache (24 hour TTL for expensive LLM operations)
            entity_cache_ttl = int(os.getenv('ENTITY_CACHE_TTL', '86400'))  # 24 hours
            self._entity_cache = TTLCache(maxsize=50, ttl=entity_cache_ttl)

            # Health check cache (5 minute TTL for system metrics)
            health_cache_ttl = int(os.getenv('HEALTH_CACHE_TTL', '300'))  # 5 minutes
            self._health_cache = TTLCache(maxsize=1, ttl=health_cache_ttl)

            # Stats cache (1 minute TTL for statistics)
            stats_cache_ttl = int(os.getenv('STATS_CACHE_TTL', '60'))  # 1 minute
            self._stats_cache = TTLCache(maxsize=1, ttl=stats_cache_ttl)

            self.logger.info(f"Search result caching enabled (size={cache_size}, ttl={cache_ttl}s)")
            self.logger.info(f"Embedding caching enabled (size={cache_size}, ttl={embedding_cache_ttl}s)")
            self.logger.info(f"Entity caching enabled (size=50, ttl={entity_cache_ttl}s)")
            self.logger.info(f"Health/stats caching enabled (health={health_cache_ttl}s, stats={stats_cache_ttl}s)")
        else:
            self._search_cache = None
            self._similar_cache = None
            self._semantic_cache = None
            self._hybrid_cache = None
            self._faceted_cache = None
            self._embedding_cache = None
            self._entity_cache = None
            self._health_cache = None
            self._stats_cache = None

        # Initialize query preprocessing
        self.use_preprocessing = NLTK_SUPPORT and os.getenv('USE_QUERY_PREPROCESSING', '1') == '1'
        # The stemmer and stopword set are built on first search, not here -
        # constructing them forces the nltk import and keeps startup slow.
        self.stemmer = None
        self.stop_words = set()
        self._preprocessing_ready = False
        if self.use_preprocessing:
            self.logger.info("Query preprocessing enabled (stemming + stopwords, lazy init)")
        else:
            self._preprocessing_ready = True  # nothing to initialize
            if NLTK_SUPPORT:
                self.logger.info("Query preprocessing disabled via USE_QUERY_PREPROCESSING=0")

        # Security: Allowed directories for document ingestion (optional)
        # Set via ALLOWED_DOCS_DIRS environment variable (comma-separated paths)
        # Always include: scraped_docs, downloads, temp, and uploads (all
        # server-controlled subdirectories of data_dir).
        #
        # The current working directory is deliberately NOT included here.
        # This server is commonly registered at Claude Code user scope, which
        # means it is launched with the cwd of whatever project the caller
        # happens to be in - unconditionally allowing cwd would make that
        # entire project tree ingestible and defeat this whitelist. Opt in
        # explicitly with TDZ_ALLOW_CWD=1 if the CLI/dev workflow needs it.
        allowed_dirs_env = os.getenv('ALLOWED_DOCS_DIRS', '')

        default_allowed_dirs = [
            self.data_dir / "scraped_docs",  # Always allow scraped documents
            self.data_dir / "downloads",     # Always allow downloaded files (Archive Search)
            self.data_dir / "temp",          # Always allow temp files (Quick Add)
            self.data_dir / "uploads",       # Always allow REST API uploads
        ]
        if os.getenv('TDZ_ALLOW_CWD', '0') == '1':
            default_allowed_dirs.append(Path.cwd())

        # Resolve all default dirs the same way user-supplied dirs are
        # resolved, so is_relative_to() can't be defeated by case/symlink
        # differences between the two sets.
        default_allowed_dirs = [d.resolve() for d in default_allowed_dirs]

        if allowed_dirs_env:
            # Add user-specified directories
            user_dirs = [Path(d.strip()).resolve() for d in allowed_dirs_env.split(',') if d.strip()]
            # Combine and remove duplicates while preserving order
            all_dirs = default_allowed_dirs + user_dirs
            seen = set()
            self.allowed_dirs = []
            for d in all_dirs:
                if d not in seen:
                    seen.add(d)
                    self.allowed_dirs.append(d)
        else:
            self.allowed_dirs = default_allowed_dirs

        self.logger.info(f"Path traversal protection enabled for: {self.allowed_dirs}")

        # Semantic search initialization (lazy loading for faster startup)
        self.use_semantic = SEMANTIC_SUPPORT and os.getenv('USE_SEMANTIC_SEARCH', '0') == '1'
        self.embeddings_model = None
        self.embeddings_index = None
        self.embeddings_doc_map = []  # Maps FAISS index positions to (doc_id, chunk_id)
        self._embeddings_loaded = False  # Track if model has been loaded

        # Cross-encoder reranking. Off by default like the other heavy search
        # backends: enabling it downloads a ~90MB model on first use.
        self.use_reranker = SEMANTIC_SUPPORT and os.getenv('USE_RERANKER', '0') == '1'
        self.reranker_model = None
        self._reranker_loaded = False

        # Local NLI entailment check for answer_question's grounding
        # verification (Tier 2 - see GROUNDEDNESS-CHECK-SCOPE.md). Off by
        # default: this replaces the always-on LLM-based check
        # (_verify_claims_llm) with a local cross-encoder, which needs its
        # own ~700MB model download on first use.
        self.use_nli_verification = SEMANTIC_SUPPORT and os.getenv('USE_NLI_VERIFICATION', '0') == '1'
        self.nli_model = None
        self._nli_loaded = False
        # (contradiction_idx, entailment_idx, neutral_idx) in the loaded
        # model's own class order - read from its config, not assumed.
        self._nli_label_indices = None

        if self.use_semantic:
            # Don't load model yet - will load on first use for faster startup
            self.embeddings_file = self.data_dir / "embeddings.faiss"
            self.embeddings_map_file = self.data_dir / "embeddings_map.json"
            self.embeddings_lock_file = self.data_dir / "embeddings.lock"
            self.logger.info("Semantic search enabled (lazy loading - model will load on first use)")
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
        # pytesseract only ever shells out to bare 'tesseract', so a perfectly
        # good install that isn't on PATH looks identical to no install at all.
        # TESSERACT_PATH accepts either the exe or its directory, mirroring
        # POPPLER_PATH.
        self.tesseract_path = os.getenv('TESSERACT_PATH', None)

        if self.use_ocr:
            if self.tesseract_path:
                tess = Path(self.tesseract_path)
                if tess.is_dir():
                    tess = tess / 'tesseract.exe' if os.name == 'nt' else tess / 'tesseract'
                pytesseract.pytesseract.tesseract_cmd = str(tess)

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
        self._documents_data_version = self._current_data_version()
        self.logger.info(f"Loaded {len(self.documents)} documents")

        # Initialize background entity extraction queue
        self._extraction_queue = queue.Queue()
        self._extraction_shutdown = threading.Event()
        self._extraction_worker = threading.Thread(
            target=self._extraction_worker_loop,
            daemon=True,
            name="EntityExtractionWorker"
        )
        self._extraction_worker.start()
        self.logger.info("Background entity extraction worker started")

        # Pick up jobs a previously-killed process left behind.
        self._recover_extraction_jobs()

        # Initialize anomaly detection
        self.anomaly_detector = None
        if ANOMALY_SUPPORT and os.getenv('USE_ANOMALY_DETECTION', '1') == '1':
            try:
                self.anomaly_detector = AnomalyDetector(self)
                self.logger.info("Anomaly detection enabled")
            except Exception as e:
                self.logger.warning(f"Anomaly detection initialization failed: {e}")
                self.anomaly_detector = None
        else:
            if ANOMALY_SUPPORT:
                self.logger.info("Anomaly detection disabled via USE_ANOMALY_DETECTION=0")

    @property
    def db_conn(self) -> sqlite3.Connection:
        """This thread's SQLite connection, opened on first use.

        Exposed as a property (rather than a _conn() helper) so that the ~190
        existing `self.db_conn.execute(...)` / `.cursor()` / `.commit()` call
        sites - plus external consumers like rest_server/admin_gui/wiki_export
        that read `kb.db_conn` - keep working unchanged while each thread
        transparently gets its own transaction scope.
        """
        conn = getattr(self._thread_local, 'conn', None)
        if conn is None:
            conn = self._make_conn()
            self._thread_local.conn = conn
        return conn

    def _make_conn(self) -> sqlite3.Connection:
        """Open one new connection and apply the per-connection PRAGMAs.

        `journal_mode = WAL` is deliberately not set here: it is a property of
        the database *file*, applied once under the cross-process schema lock
        in _init_database_locked. foreign_keys and busy_timeout are
        per-connection settings and so must be re-applied to every thread's
        connection.
        """
        conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        # A lingering orphaned process (e.g. a prior session's server that
        # never exited) can hold the file lock indefinitely; without a
        # timeout, a second process blocks on it forever with no log output
        # instead of failing fast with a clear "database is locked" error.
        conn.execute(f"PRAGMA busy_timeout = {int(os.getenv('TDZ_DB_BUSY_TIMEOUT_MS', '30000'))}")
        with self._conns_lock:
            self._all_conns.append(conn)
        return conn

    def _close_all_conns(self):
        """Close every connection handed out by db_conn, from any thread.

        check_same_thread=False is what makes this legal: the owning thread may
        already be gone (a retired ThreadPoolExecutor worker) and would
        otherwise leak its file handle for the life of the process.
        """
        with self._conns_lock:
            conns, self._all_conns = self._all_conns, []
        # Threads keep their conn in _thread_local; swapping the whole object
        # out makes every thread - including ones already parked in the pool -
        # lazily re-open instead of reusing a connection we just closed.
        self._thread_local = threading.local()
        for conn in conns:
            try:
                conn.close()
            except Exception:
                pass  # already closed, or its thread died mid-transaction

    def _init_database(self):
        """Initialize SQLite database and create schema if needed."""
        # restore_backup() calls this again after replacing the .db file, so any
        # connection still open against the old file must go; the db_conn
        # property then re-opens per thread against the new one on next use.
        self._close_all_conns()

        # Serialise WAL-mode activation and the check-then-create-or-migrate
        # sequence across processes. sqlite3.connect() creates the
        # physical .db file immediately even with zero tables, so a
        # file-existence check is not a reliable signal of "has the schema
        # been created" when two processes start against the same brand-new
        # data directory at once - and switching journal mode is itself a
        # schema-file mutation with the same race potential (observed
        # directly: two connections enabling WAL at almost the same instant
        # on a brand-new file occasionally left one connection with a view of
        # the database that didn't yet include the other's committed schema,
        # surfacing as a transient "no such table: documents"). Every CREATE
        # statement below is now idempotent (IF NOT EXISTS), so once
        # serialised it's harmless for a racing process to run the "create
        # schema" branch again - it just no-ops through statements a peer
        # already committed.
        with _cross_process_lock(self.data_dir / "schema_init.lock", timeout=60.0):
            self._init_database_locked()

    def _init_database_locked(self):
        """Create/migrate the schema. Caller must hold schema_init.lock.

        db_exists is determined here, under the lock, via a live query
        against this connection rather than a pre-lock filesystem check -
        that check would be stale by the time we get the lock if a
        concurrent process created the schema while we were waiting for it.
        """
        # Every Claude Code session spawns its own server process against this
        # same database file. In the default 'delete' journal mode a single
        # writer takes an exclusive lock on the whole file and blocks every
        # reader in every other process, so concurrent sessions serialised
        # behind each other and timed out. WAL lets readers proceed during a
        # write, which is what makes multi-session use viable.
        try:
            mode = self.db_conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode and str(mode[0]).lower() != 'wal':
                self.logger.warning(f"Could not enable WAL journal mode (got '{mode[0]}')")
        except sqlite3.Error as e:
            # Non-fatal: e.g. the DB lives on a network share that lacks WAL support
            self.logger.warning(f"Could not enable WAL journal mode: {e}")

        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        db_exists = cursor.fetchone() is not None

        if not db_exists:
            self.logger.info("Creating new database schema")
            cursor = self.db_conn.cursor()

            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
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
                    url_content_hash TEXT,
                    card_id TEXT,
                    superseded_by TEXT
                )
            """)

            # Create indexes on documents table
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_filepath ON documents(filepath)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents(source_url)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_scrape_status ON documents(scrape_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_card_id ON documents(card_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_superseded_by ON documents(superseded_by)")

            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")

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
                CREATE TABLE IF NOT EXISTS document_tables (
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
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM tables_fts),
                            new.doc_id, new.table_id, new.searchable_text);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS tables_fts_delete AFTER DELETE ON document_tables BEGIN
                    DELETE FROM tables_fts WHERE doc_id = old.doc_id AND table_id = old.table_id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS tables_fts_update AFTER UPDATE ON document_tables BEGIN
                    DELETE FROM tables_fts WHERE doc_id = old.doc_id AND table_id = old.table_id;
                    INSERT INTO tables_fts(rowid, doc_id, table_id, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM tables_fts),
                            new.doc_id, new.table_id, new.searchable_text);
                END
            """)

            # Create document_code_blocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_code_blocks (
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
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM code_fts),
                            new.doc_id, new.block_id, new.block_type, new.searchable_text);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS code_fts_delete AFTER DELETE ON document_code_blocks BEGIN
                    DELETE FROM code_fts WHERE doc_id = old.doc_id AND block_id = old.block_id;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS code_fts_update AFTER UPDATE ON document_code_blocks BEGIN
                    DELETE FROM code_fts WHERE doc_id = old.doc_id AND block_id = old.block_id;
                    INSERT INTO code_fts(rowid, doc_id, block_id, block_type, searchable_text)
                    VALUES ((SELECT COALESCE(MAX(rowid), 0) + 1 FROM code_fts),
                            new.doc_id, new.block_id, new.block_type, new.searchable_text);
                END
            """)

            # Create document_facets table for faceted search
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_facets (
                    doc_id TEXT NOT NULL,
                    facet_type TEXT NOT NULL,
                    facet_value TEXT NOT NULL,
                    PRIMARY KEY (doc_id, facet_type, facet_value),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faceted search
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facets_type_value ON document_facets(facet_type, facet_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facets_doc_id ON document_facets(doc_id)")

            # Create search_log table for analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_log (
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log(query)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_timestamp ON search_log(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_mode ON search_log(search_mode)")

            # Create cross_references table for content linking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cross_references (
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_xref_type_value ON cross_references(ref_type, ref_value)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_xref_doc_id ON cross_references(doc_id)")

            # Create query_suggestions table for autocomplete
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS query_suggestions USING fts5(
                    term,
                    frequency UNINDEXED,
                    category UNINDEXED
                )
            """)

            # Create document_summaries table for AI-generated summaries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_summaries (
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_doc_id ON document_summaries(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_type ON document_summaries(summary_type)")

            # Create document_entities table for named entity extraction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_entities (
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
                CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
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
                CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON document_entities BEGIN
                    INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                    VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON document_entities BEGIN
                    DELETE FROM entities_fts WHERE rowid = old.rowid;
                END
            """)

            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON document_entities BEGIN
                    DELETE FROM entities_fts WHERE rowid = old.rowid;
                    INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                    VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                END
            """)

            # Create indexes for entity queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_doc_id ON document_entities(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON document_entities(entity_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_text ON document_entities(entity_text)")

            # Create entity relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_relationships (
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entity1 ON entity_relationships(entity1_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entity2 ON entity_relationships(entity2_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON entity_relationships(relationship_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_strength ON entity_relationships(strength)")

            # Create extraction_jobs table for background entity extraction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    queued_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    entities_extracted INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for extraction jobs queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_doc_id ON extraction_jobs(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status ON extraction_jobs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_queued_at ON extraction_jobs(queued_at)")

            # Create graph_cache table for knowledge graph caching (v2.24.0)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_cache (
                    cache_id TEXT PRIMARY KEY,
                    graph_version INTEGER NOT NULL,
                    graph_data BLOB NOT NULL,
                    node_count INTEGER NOT NULL,
                    edge_count INTEGER NOT NULL,
                    created_date TEXT NOT NULL,
                    last_accessed TEXT
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_cache_created ON graph_cache(created_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_cache_accessed ON graph_cache(last_accessed)")

            # Create graph_metrics table for graph analysis metrics (v2.24.0)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_metrics (
                    metric_id TEXT PRIMARY KEY,
                    entity_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    pagerank REAL,
                    betweenness_centrality REAL,
                    closeness_centrality REAL,
                    degree_centrality REAL,
                    community_id INTEGER,
                    computed_date TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_entity ON graph_metrics(entity_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_pagerank ON graph_metrics(pagerank DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_community ON graph_metrics(community_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_type ON graph_metrics(entity_type)")

            # Create graph_paths table for path finding cache (v2.24.0)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_paths (
                    path_id TEXT PRIMARY KEY,
                    entity1 TEXT NOT NULL,
                    entity2 TEXT NOT NULL,
                    path_length INTEGER NOT NULL,
                    path_nodes TEXT NOT NULL,
                    path_weight REAL,
                    computed_date TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entity1 ON graph_paths(entity1)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entity2 ON graph_paths(entity2)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entities ON graph_paths(entity1, entity2)")

            self.db_conn.commit()
            self.logger.info("Database schema created successfully (with FTS5, tables, code blocks, facets, analytics, suggestions, summaries, entities, relationships, extraction jobs, and knowledge graph)")
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

            # Migrate: Add knowledge-card identity columns if missing
            if 'card_id' not in columns:
                self.logger.info("Migrating database: adding card_id column")
                cursor.execute("ALTER TABLE documents ADD COLUMN card_id TEXT")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_card_id ON documents(card_id)")
                self.db_conn.commit()

            if 'superseded_by' not in columns:
                self.logger.info("Migrating database: adding superseded_by column")
                cursor.execute("ALTER TABLE documents ADD COLUMN superseded_by TEXT")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_superseded_by ON documents(superseded_by)")
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
                    CREATE TABLE IF NOT EXISTS document_tables (
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
                    CREATE TABLE IF NOT EXISTS document_code_blocks (
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
                    CREATE TABLE IF NOT EXISTS document_facets (
                        doc_id TEXT NOT NULL,
                        facet_type TEXT NOT NULL,
                        facet_value TEXT NOT NULL,
                        PRIMARY KEY (doc_id, facet_type, facet_value),
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_facets_type_value ON document_facets(facet_type, facet_value)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_facets_doc_id ON document_facets(doc_id)")

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
                    CREATE TABLE IF NOT EXISTS search_log (
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

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log(query)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_timestamp ON search_log(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_log_mode ON search_log(search_mode)")

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
                    CREATE TABLE IF NOT EXISTS cross_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT NOT NULL,
                        chunk_id INTEGER NOT NULL,
                        ref_type TEXT NOT NULL,
                        ref_value TEXT NOT NULL,
                        context TEXT,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_xref_type_value ON cross_references(ref_type, ref_value)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_xref_doc_id ON cross_references(doc_id)")

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
                    CREATE VIRTUAL TABLE IF NOT EXISTS query_suggestions USING fts5(
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
                    CREATE TABLE IF NOT EXISTS document_summaries (
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
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_doc_id ON document_summaries(doc_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_type ON document_summaries(summary_type)")

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
                    CREATE TABLE IF NOT EXISTS document_entities (
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
                    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
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
                    CREATE TRIGGER IF NOT EXISTS entities_fts_insert AFTER INSERT ON document_entities BEGIN
                        INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                        VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS entities_fts_delete AFTER DELETE ON document_entities BEGIN
                        DELETE FROM entities_fts WHERE rowid = old.rowid;
                    END
                """)

                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS entities_fts_update AFTER UPDATE ON document_entities BEGIN
                        DELETE FROM entities_fts WHERE rowid = old.rowid;
                        INSERT INTO entities_fts(rowid, doc_id, entity_id, entity_text, entity_type, context)
                        VALUES (new.rowid, new.doc_id, new.entity_id, new.entity_text, new.entity_type, new.context);
                    END
                """)

                # Create indexes for entity queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_doc_id ON document_entities(doc_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON document_entities(entity_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_text ON document_entities(entity_text)")

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
                    CREATE TABLE IF NOT EXISTS entity_relationships (
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
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entity1 ON entity_relationships(entity1_text)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entity2 ON entity_relationships(entity2_text)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON entity_relationships(relationship_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_strength ON entity_relationships(strength)")

                self.db_conn.commit()
                self.logger.info("entity_relationships table created")

            # Migrate: Add extraction_jobs table if not exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='extraction_jobs'
            """)
            if not cursor.fetchone():
                self.logger.info("Creating extraction_jobs table for background entity extraction")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS extraction_jobs (
                        job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        doc_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        confidence_threshold REAL NOT NULL,
                        queued_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        error_message TEXT,
                        entities_extracted INTEGER,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                    )
                """)

                # Create indexes for extraction jobs queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_doc_id ON extraction_jobs(doc_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status ON extraction_jobs(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_extraction_jobs_queued_at ON extraction_jobs(queued_at)")

                self.db_conn.commit()
                self.logger.info("extraction_jobs table created")

            # Check for graph_cache table (v2.24.0 - Knowledge Graph Analysis)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_cache'")
            if not cursor.fetchone():
                self.logger.info("Creating graph_cache table for knowledge graph caching")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_cache (
                        cache_id TEXT PRIMARY KEY,
                        graph_version INTEGER NOT NULL,
                        graph_data BLOB NOT NULL,
                        node_count INTEGER NOT NULL,
                        edge_count INTEGER NOT NULL,
                        created_date TEXT NOT NULL,
                        last_accessed TEXT
                    )
                """)

                # Create index for cache management
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_cache_created ON graph_cache(created_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_cache_accessed ON graph_cache(last_accessed)")

                self.db_conn.commit()
                self.logger.info("graph_cache table created")

            # Check for graph_metrics table (v2.24.0 - Knowledge Graph Analysis)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_metrics'")
            if not cursor.fetchone():
                self.logger.info("Creating graph_metrics table for graph analysis metrics")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_metrics (
                        metric_id TEXT PRIMARY KEY,
                        entity_text TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        pagerank REAL,
                        betweenness_centrality REAL,
                        closeness_centrality REAL,
                        degree_centrality REAL,
                        community_id INTEGER,
                        computed_date TEXT NOT NULL
                    )
                """)

                # Create indexes for metric queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_entity ON graph_metrics(entity_text)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_pagerank ON graph_metrics(pagerank DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_community ON graph_metrics(community_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_type ON graph_metrics(entity_type)")

                self.db_conn.commit()
                self.logger.info("graph_metrics table created")

            # Check for graph_paths table (v2.24.0 - Knowledge Graph Analysis)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_paths'")
            if not cursor.fetchone():
                self.logger.info("Creating graph_paths table for path finding cache")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS graph_paths (
                        path_id TEXT PRIMARY KEY,
                        entity1 TEXT NOT NULL,
                        entity2 TEXT NOT NULL,
                        path_length INTEGER NOT NULL,
                        path_nodes TEXT NOT NULL,
                        path_weight REAL,
                        computed_date TEXT NOT NULL
                    )
                """)

                # Create indexes for path queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entity1 ON graph_paths(entity1)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entity2 ON graph_paths(entity2)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_paths_entities ON graph_paths(entity1, entity2)")

                self.db_conn.commit()
                self.logger.info("graph_paths table created")

            # ============================================================
            # Phase 2 Topic & Cluster Tables (v2.24.0 - Discovery)
            # ============================================================


        # Always run migrations for schema updates (regardless of db_exists)
        self._migrate_phase3_schema()
        self._migrate_mcp_log_schema()
        self._migrate_figures_schema()
        self._migrate_topics_clusters_schema()

    def _migrate_topics_clusters_schema(self):
        """Create the topic-modelling and clustering tables if absent.

        These used to be created only in _init_database_locked's
        existing-database branch, so a brand-new database never got them and
        every topic/cluster tool failed with "no such table" until the next
        schema migration happened to run. Kept as check-then-create so it is
        safe to call on every startup.
        """
        cursor = self.db_conn.cursor()

        # Check for topics table (v2.24.0 - Topic Modeling)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topics'")
        if not cursor.fetchone():
            self.logger.info("Creating topics table for topic modeling")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    topic_number INTEGER NOT NULL,
                    top_words TEXT NOT NULL,
                    word_weights TEXT NOT NULL,
                    num_documents INTEGER DEFAULT 0,
                    coherence_score REAL,
                    created_date TEXT NOT NULL,
                    UNIQUE(model_type, topic_number)
                )
            """)

            # Create indexes for topic queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_model ON topics(model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_number ON topics(model_type, topic_number)")

            self.db_conn.commit()
            self.logger.info("topics table created")

        # Check for document_topics table (v2.24.0 - Topic Modeling)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_topics'")
        if not cursor.fetchone():
            self.logger.info("Creating document_topics table for topic assignments")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_topics (
                    assignment_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    probability REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    assigned_date TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (topic_id) REFERENCES topics(topic_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for topic assignment queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_topics_doc ON document_topics(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_topics_topic ON document_topics(topic_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_topics_model ON document_topics(model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_topics_probability ON document_topics(probability DESC)")

            self.db_conn.commit()
            self.logger.info("document_topics table created")

        # Check for clusters table (v2.24.0 - Document Clustering)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clusters'")
        if not cursor.fetchone():
            self.logger.info("Creating clusters table for clustering analysis")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    cluster_number INTEGER NOT NULL,
                    centroid_vector BLOB,
                    num_documents INTEGER DEFAULT 0,
                    representative_docs TEXT,
                    top_terms TEXT,
                    silhouette_score REAL,
                    created_date TEXT NOT NULL,
                    UNIQUE(algorithm, cluster_number)
                )
            """)

            # Create indexes for cluster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_algorithm ON clusters(algorithm)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_clusters_number ON clusters(algorithm, cluster_number)")

            self.db_conn.commit()
            self.logger.info("clusters table created")

        # Check for document_clusters table (v2.24.0 - Document Clustering)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_clusters'")
        if not cursor.fetchone():
            self.logger.info("Creating document_clusters table for cluster assignments")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_clusters (
                    assignment_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    cluster_id TEXT NOT NULL,
                    distance REAL,
                    algorithm TEXT NOT NULL,
                    assigned_date TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
                )
            """)

            # Phase 3 temporal analysis tables (events, document_events,
            # timeline_entries) are NOT created here - _migrate_phase3_schema(),
            # called unconditionally below regardless of db_exists, already
            # creates them idempotently. Duplicating that here caused a crash
            # the first time an existing (non-fresh) database reached this
            # branch: _migrate_phase3_schema() had already created 'events' on
            # the previous init, so the unconditional CREATE TABLE IF NOT EXISTS here raised
            # "table events already exists".

            # Create indexes for cluster assignment queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_clusters_doc ON document_clusters(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_clusters_cluster ON document_clusters(cluster_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_clusters_algorithm ON document_clusters(algorithm)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_clusters_distance ON document_clusters(distance)")

            self.db_conn.commit()
            self.logger.info("document_clusters table created")

    def _migrate_figures_schema(self):
        """Create the figure-OCR tables and add extraction_jobs.job_type.

        Figures live in their own table rather than being appended to the
        document's chunks: a chunk edit is indistinguishable from original
        document text once merged, so OCR output could never be re-run or
        corrected without reingesting the whole document. Keeping them
        separate also preserves the page/index provenance needed to say
        "figure 2 on page 7".
        """
        cursor = self.db_conn.cursor()

        # extraction_jobs predates having more than one kind of job.
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(extraction_jobs)")}
        if cols and 'job_type' not in cols:
            self.logger.info("Adding job_type column to extraction_jobs")
            cursor.execute(
                "ALTER TABLE extraction_jobs ADD COLUMN job_type TEXT NOT NULL DEFAULT 'entities'"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_jobs_type ON extraction_jobs(job_type)"
            )
            self.db_conn.commit()

        if cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='document_figures'"
        ).fetchone():
            # document_figures predates page rasterization, when every row was
            # necessarily an embedded bitmap.
            fig_cols = {row[1] for row in cursor.execute("PRAGMA table_info(document_figures)")}
            if 'source' not in fig_cols:
                self.logger.info("Adding source column to document_figures")
                cursor.execute(
                    "ALTER TABLE document_figures ADD COLUMN source TEXT NOT NULL DEFAULT 'embedded'"
                )
                self.db_conn.commit()
            return

        self.logger.info("Creating document_figures tables for figure OCR")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_figures (
                figure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER,
                image_index INTEGER NOT NULL,
                ocr_text TEXT,
                char_count INTEGER DEFAULT 0,
                image_path TEXT,
                width INTEGER,
                height INTEGER,
                extracted_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'embedded',
                UNIQUE(doc_id, page_number, image_index),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_figures_doc ON document_figures(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_figures_page ON document_figures(doc_id, page_number)")

        # Mirror the chunks_fts5 arrangement so figure text is searchable by
        # the same FTS5 machinery, kept in sync by triggers rather than by
        # every writer remembering to update two tables.
        if self._fts5_available_raw():
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS figures_fts5 USING fts5(
                    doc_id UNINDEXED,
                    figure_id UNINDEXED,
                    ocr_text,
                    tokenize='porter unicode61'
                )
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS figures_fts5_insert AFTER INSERT ON document_figures BEGIN
                    INSERT INTO figures_fts5(rowid, doc_id, figure_id, ocr_text)
                    VALUES (new.rowid, new.doc_id, new.figure_id, new.ocr_text);
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS figures_fts5_delete AFTER DELETE ON document_figures BEGIN
                    DELETE FROM figures_fts5 WHERE rowid = old.rowid;
                END
            """)
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS figures_fts5_update AFTER UPDATE ON document_figures BEGIN
                    DELETE FROM figures_fts5 WHERE rowid = old.rowid;
                    INSERT INTO figures_fts5(rowid, doc_id, figure_id, ocr_text)
                    VALUES (new.rowid, new.doc_id, new.figure_id, new.ocr_text);
                END
            """)

        self.db_conn.commit()
        self.logger.info("document_figures tables created")

    def _fts5_available_raw(self) -> bool:
        """Whether this SQLite build has the FTS5 extension compiled in.

        Distinct from _fts5_available(), which reports whether the chunk
        index actually exists and is populated.
        """
        try:
            self.db_conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)"
            )
            self.db_conn.execute("DROP TABLE IF EXISTS _fts5_probe")
            return True
        except sqlite3.OperationalError:
            return False

    def _migrate_mcp_log_schema(self):
        """Create the MCP call log table if it doesn't exist yet."""
        cursor = self.db_conn.cursor()

        result = cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='mcp_call_log'
        """).fetchone()

        if result:
            return

        self.logger.info("Creating mcp_call_log table")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_call_log (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                called_at TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                args_summary TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcp_log_tool ON mcp_call_log(tool_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcp_log_time ON mcp_call_log(called_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcp_log_success ON mcp_call_log(success)")

        self.db_conn.commit()
        self.logger.info("mcp_call_log table created")

    def _migrate_phase3_schema(self):
        """Migrate existing databases to include Phase 3 temporal analysis tables."""
        cursor = self.db_conn.cursor()

        # Check if Phase 3 tables already exist
        result = cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='events'
        """).fetchone()

        if result:
            # Phase 3 tables already exist, skip migration
            return

        self.logger.info("Creating Phase 3 temporal analysis tables")

        try:
            # Create events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    date_extracted TEXT,
                    date_normalized TEXT,
                    year INTEGER,
                    month INTEGER,
                    day INTEGER,
                    confidence REAL DEFAULT 0.5,
                    entities TEXT,
                    metadata TEXT,
                    created_date TEXT NOT NULL
                )
            """)

            # Create document-events mapping table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_events (
                    mapping_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    context TEXT,
                    position INTEGER,
                    created_date TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
                )
            """)

            # Create timeline entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_entries (
                    entry_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    display_date TEXT NOT NULL,
                    sort_order INTEGER NOT NULL,
                    category TEXT,
                    importance INTEGER DEFAULT 3,
                    created_date TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for Phase 3 tables
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_year ON events(year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date_normalized)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_events_doc ON document_events(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_events_event ON document_events(event_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_entries_event ON timeline_entries(event_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_entries_sort ON timeline_entries(sort_order)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_entries_category ON timeline_entries(category)")

            self.db_conn.commit()
            self.logger.info("Phase 3 temporal analysis tables created successfully")

        except Exception as e:
            self.logger.exception("Failed to create Phase 3 tables")
            self.db_conn.rollback()
            raise

    def _fts5_available(self) -> bool:
        """Check if FTS5 is available and table exists."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks_fts5 LIMIT 1")
            return True
        except Exception:
            return False

    def _current_data_version(self) -> Optional[int]:
        """SQLite's built-in per-connection change counter, or None on error.

        PRAGMA data_version increments whenever a DIFFERENT connection
        commits a change to this database file, but does NOT change for
        commits made by this connection itself - exactly the signal needed
        to detect "another agent process changed something" without a
        schema change (a new revision column) and without re-querying the
        documents table on every call just to check.
        """
        try:
            return self.db_conn.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error:
            return None

    def _reload_documents(self):
        """Authoritative refresh of self.documents from the database.

        Unlike _load_documents() (which only ever adds/updates entries and
        is meant for the one-time startup/migration path), this REPLACES the
        dict wholesale so a document another agent process removed is
        actually dropped here too, not just never updated.
        """
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT * FROM documents")
        rows = cursor.fetchall()

        documents = {}
        for row in rows:
            doc = DocumentMeta(
                doc_id=row[0], filename=row[1], title=row[2], filepath=row[3],
                file_type=row[4], total_pages=row[5], total_chunks=row[6],
                indexed_at=row[7], tags=json.loads(row[8]), author=row[9],
                subject=row[10], creator=row[11], creation_date=row[12],
                file_mtime=row[13] if len(row) > 13 else None,
                file_hash=row[14] if len(row) > 14 else None,
                source_url=row[15] if len(row) > 15 else None,
                scrape_date=row[16] if len(row) > 16 else None,
                scrape_config=row[17] if len(row) > 17 else None,
                scrape_status=row[18] if len(row) > 18 else None,
                scrape_error=row[19] if len(row) > 19 else None,
                url_last_checked=row[20] if len(row) > 20 else None,
                url_content_hash=row[21] if len(row) > 21 else None,
                card_id=row[22] if len(row) > 22 else None,
                superseded_by=row[23] if len(row) > 23 else None
            )
            documents[doc.doc_id] = doc
        self.documents = documents

    def _sync_documents_if_needed(self):
        """Refresh self.documents if another agent process changed the DB.

        self.documents was historically loaded once at startup and never
        refreshed, so a long-running session kept serving a frozen view:
        a document another Claude Code session added was invisible to
        search_docs/get_document/list_docs until this process restarted, and
        one that peer removed kept being served indefinitely. Called at the
        top of the read paths that matter most (search, semantic_search,
        hybrid_search, get_document, list_documents, get_stats) - a single
        cheap PRAGMA query on every call, and a full reload only on the rare
        call where something actually changed.
        """
        version = self._current_data_version()
        if version is None:
            return
        if version != self._documents_data_version:
            self._documents_data_version = version
            self._reload_documents()

            # self.chunks/self.bm25 have the identical staleness problem once
            # loaded in this process (_build_bm25_index only reloads chunks
            # from the DB when self.chunks is empty - see
            # reconcile_chunk_cache for the same reasoning). Clearing them
            # here piggybacks on the same change-detection signal instead of
            # needing a second one; this mirrors what add_document/
            # remove_document already do for THEIR OWN writes, just extended
            # to writes made by a peer process.
            if self.chunks:
                self.chunks = []
            self.bm25 = None
            self._invalidate_caches()

            self.logger.debug(f"Refreshed documents cache ({len(self.documents)} docs) - detected change from another process")

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
                    url_content_hash=row[21] if len(row) > 21 else None,
                    card_id=row[22] if len(row) > 22 else None,
                    superseded_by=row[23] if len(row) > 23 else None
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
                 url_last_checked, url_content_hash, card_id, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                doc_meta.url_content_hash,
                doc_meta.card_id,
                doc_meta.superseded_by
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
            try:
                self.db_conn.commit()
            except (SystemError, Exception) as commit_error:
                # Python 3.14 SQLite bug workaround - commit may fail with various errors
                # Check if "not an error" message (Python 3.14 bug) or SystemError
                error_msg = str(commit_error).lower()
                if isinstance(commit_error, SystemError) or "not an error" in error_msg:
                    # Verify document was actually added by checking it exists
                    check_cursor = self.db_conn.cursor()
                    result = check_cursor.execute(
                        "SELECT 1 FROM documents WHERE doc_id = ?", (doc_meta.doc_id,)
                    ).fetchone()
                    if result is not None:
                        # Document was added, commit succeeded despite error
                        self.logger.warning(
                            f"Commit error (Python 3.14 bug), but document {doc_meta.doc_id} was added successfully: {commit_error}"
                        )
                        return  # Success despite error
                # Re-raise if not the Python 3.14 bug or if verification failed
                raise

        except Exception as e:
            # Rollback on error (but skip if it's the Python 3.14 "not an error" bug that we already handled)
            error_msg = str(e).lower()
            if not (isinstance(e, SystemError) or "not an error" in error_msg):
                try:
                    self.db_conn.rollback()
                except SystemError:
                    # Rollback may also fail with same bug, ignore
                    pass
                # exception(), not error(): a failed write here means a bug or
                # real DB fault, and the wrapped KnowledgeBaseError below keeps
                # only the message - the traceback would otherwise be lost.
                self.logger.exception("Error adding document to database")
                raise KnowledgeBaseError(f"Failed to add document to database: {e}")
            # If it's the "not an error" bug, we already verified and returned above, so this shouldn't be reached
            # But if it is, don't wrap it again
            raise

    def _remove_document_db(self, doc_id: str) -> bool:
        """Remove a document from the database (chunks cascade automatically)."""
        cursor = self.db_conn.cursor()

        try:
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            rowcount = cursor.rowcount
            try:
                self.db_conn.commit()
            except SystemError as se:
                # Python 3.14 SQLite bug workaround - commit may return NULL
                # Check if deletion actually happened by verifying doc doesn't exist
                check_cursor = self.db_conn.cursor()
                result = check_cursor.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                if result is None:
                    # Document was deleted, commit succeeded despite SystemError
                    self.logger.warning(f"SystemError during commit (Python 3.14 bug), but deletion succeeded: {se}")
                    return rowcount > 0
                else:
                    # Deletion failed, re-raise
                    raise
            return rowcount > 0
        except Exception as e:
            try:
                self.db_conn.rollback()
            except SystemError:
                # Rollback may also fail with same bug, ignore
                pass
            self.logger.exception("Error removing document from database")
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

    def _document_source_missing(self, filepath: Optional[str]) -> bool:
        """True if `filepath` is empty/NULL or points at a file that is no
        longer on disk. Shared by get_figure_ocr_coverage and health_check's
        missing_source_files metric so the two never disagree about what
        "missing" means."""
        return not (filepath and os.path.exists(filepath))

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
        if self._semantic_cache is not None:
            self._semantic_cache.clear()
        if self._hybrid_cache is not None:
            self._hybrid_cache.clear()
        if self._faceted_cache is not None:
            self._faceted_cache.clear()
        self.logger.info("All search result caches invalidated")

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file content."""
        md5_hash = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read file in chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def close(self):
        """Close the database connection and shutdown background workers."""
        # Signal worker threads to shutdown
        if hasattr(self, '_extraction_shutdown'):
            self._extraction_shutdown.set()
            self.logger.info("Signaled entity extraction worker to shutdown")

            # Wait for worker thread to finish (with timeout)
            if hasattr(self, '_extraction_worker') and self._extraction_worker.is_alive():
                self._extraction_worker.join(timeout=10.0)
                if self._extraction_worker.is_alive():
                    self.logger.warning("Entity extraction worker did not shutdown cleanly")
                else:
                    self.logger.info("Entity extraction worker shutdown complete")

        # Close every per-thread database connection (see the db_conn property)
        self._close_all_conns()
        self.logger.info("Database connections closed")
