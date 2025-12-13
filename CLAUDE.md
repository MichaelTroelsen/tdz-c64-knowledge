# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. It provides tools to ingest PDFs and text files, build a searchable knowledge base, and query it through the MCP interface.

## Core Architecture

### Main Components

**server.py** - MCP server implementation
- `KnowledgeBase` class: Core data management (index + chunks storage + tables + code blocks)
- MCP tool handlers: `search_docs`, `semantic_search`, `hybrid_search`, `add_document`, `get_chunk`, `get_document`, `list_docs`, `remove_document`, `kb_stats`, `health_check`, `find_similar`, `check_updates`, `add_documents_bulk`, `remove_documents_bulk`, `search_tables` (NEW), `search_code` (NEW)
- MCP resource handlers: Exposes documents as `c64kb://` URIs
- Async server running on stdio transport

**cli.py** - Command-line interface for batch operations
- Wraps `KnowledgeBase` for CLI usage
- Commands: `add`, `add-folder`, `search`, `list`, `remove`, `stats`
- Useful for bulk importing documents

### Data Storage Model

The knowledge base uses **SQLite database** for efficient storage and querying:
- **knowledge_base.db** - SQLite database with four main tables:
  - `documents` table - Stores DocumentMeta objects (13 fields including doc_id, title, tags, metadata)
  - `chunks` table - Stores DocumentChunk objects (5 fields: doc_id, chunk_id, page, content, word_count)
  - `document_tables` table (NEW) - Stores extracted tables from PDFs (7 fields: doc_id, table_id, page, markdown, searchable_text, row_count, col_count)
  - `document_code_blocks` table (NEW) - Stores detected code blocks (7 fields: doc_id, block_id, page, block_type, code, searchable_text, line_count)
  - Foreign key relationships: all child tables reference documents.doc_id with CASCADE delete
  - FTS5 indexes: `chunks_fts5`, `tables_fts` (NEW), `code_fts` (NEW) for full-text search
  - Indexes on filepath, file_type, and doc_id for fast queries

**Migration from JSON**: Legacy JSON files (index.json, chunks/*.json) are automatically migrated to SQLite on first run. JSON files are preserved as backup.

**Lazy Loading**: Only document metadata is loaded at startup. Chunks are loaded on-demand for search (building BM25 index) and retrieval operations. This enables the system to scale to 100k+ documents without memory issues.

**Chunking Strategy**: Documents are split into overlapping chunks (default 1500 words, 200 word overlap) to enable granular search and retrieval.

### Document Processing Pipeline

1. File ingestion (PDF via pypdf, text files with encoding detection)
2. Text extraction (pages joined with "--- PAGE BREAK ---")
3. **Table extraction** (NEW) - For PDFs, extracts structured tables using pdfplumber, converts to markdown
4. **Code block detection** (NEW) - Detects BASIC, Assembly, and Hex dump code blocks using regex patterns
5. Chunking with overlap (_chunk_text method)
6. **Content-based ID generation** (doc_id from MD5 hash of normalized text content, first 10k words)
7. **Duplicate detection** (checks if content hash already exists, returns existing doc if duplicate)
8. Database persistence (document + chunks + tables + code blocks inserted in ACID transaction via `_add_document_db()`)

**Duplicate Detection Details**:
- `_generate_doc_id()` accepts optional `text_content` parameter
- If provided, generates content-based hash from normalized text (lowercase, first 10k words)
- In `add_document()`, checks if doc_id already exists in `self.documents`
- If duplicate detected, logs warning and returns existing document (non-destructive)
- Prevents duplicate indexing regardless of file path or filename
- Backward compatible: filepath-based IDs still supported for legacy code

### Search Implementation

**Semantic Search with Embeddings** - Conceptual/meaning-based search (optional):
- Uses sentence-transformers for generating embeddings
- FAISS vector similarity search with cosine distance
- Finds documents based on meaning, not just keywords
- Example: "movable objects" finds documents about "sprites"
- Performance: ~7-16ms per query after initial embeddings generation
- Embeddings persisted to disk (embeddings.faiss, embeddings_map.json)
- Can be enabled with `USE_SEMANTIC_SEARCH=1` environment variable
- Configurable model via `SEMANTIC_MODEL` (default: all-MiniLM-L6-v2)

**SQLite FTS5 Full-Text Search** - Native database search (recommended for keywords):
- Uses SQLite's FTS5 virtual table with Porter stemming tokenizer
- 480x faster than BM25 (50ms vs 24,000ms for typical queries)
- Native BM25 ranking built into SQLite
- No need to load all chunks into memory
- Automatic triggers keep FTS5 in sync with chunks table
- Can be enabled with `USE_FTS5=1` environment variable
- Falls back to BM25/simple search if FTS5 returns no results

**BM25 (Okapi BM25)** - Industry-standard probabilistic ranking (fallback):
- Uses rank-bm25 library for accurate relevance scoring
- Handles document length normalization
- Tokenizes documents and queries for matching
- Accepts negative scores for small documents (filters by abs(score) > 0.0001)
- Can be disabled with `USE_BM25=0` environment variable

**Security - Path Traversal Protection**:
- Optional directory whitelisting via `ALLOWED_DOCS_DIRS` environment variable
- Validates all file paths in `add_document()` are within allowed directories
- Blocks path traversal attacks (e.g., `../../../etc/passwd`)
- Raises `SecurityError` on violations
- Backward compatible (no restrictions if not configured)

**Phrase Search**:
- Detects quoted phrases with regex pattern `r'"([^"]*)"'`
- Exact phrase matches get 2x score boost
- Combined with term search for comprehensive results

**Query Preprocessing** (NLTK-powered):
- Tokenization with word_tokenize()
- Stopword removal using English stopwords corpus
- Porter Stemmer for word normalization
- Preserves technical terms (hyphenated words like "VIC-II", numbers like "6502")
- Applied to both queries and corpus during BM25 indexing
- Can be disabled with `USE_QUERY_PREPROCESSING=0` environment variable
- Implemented in `_preprocess_text()` method

**Hybrid Search** (NEW in v2.0.0):
- Combines FTS5 keyword search with semantic search
- Configurable weighting via `semantic_weight` parameter (0.0-1.0, default 0.3)
- Score normalization for fair comparison (both normalized to 0-1 range)
- Intelligent result merging by (doc_id, chunk_id)
- Best of both worlds: exact keyword matching + conceptual understanding
- Example: "SID sound programming" finds exact matches AND related content about audio synthesis
- Performance: ~60-180ms (combines two searches)

**Enhanced Snippet Extraction** (NEW in v2.0.0):
- Term density scoring via sliding window analysis
- Complete sentence extraction (no mid-sentence cuts)
- Code block preservation (detects and preserves indented blocks)
- Whole word boundary highlighting for better accuracy
- 80% size threshold ensures adequate context
- More natural, readable snippets with proper sentence boundaries

**Health Monitoring** (NEW in v2.0.0):
- Comprehensive system diagnostics via `health_check()` method
- Database health: integrity checking, size monitoring, orphaned chunk detection
- Feature status: FTS5, semantic search, BM25, embeddings availability
- Performance metrics: cache utilization, index status
- Disk space warnings (< 1GB free)
- Returns structured health report with status, metrics, and issues

**Table Extraction from PDFs** (NEW in v2.1.0):
- Automatic extraction of structured tables from PDF documents using pdfplumber
- Tables converted to markdown format for display
- FTS5 full-text search on table content via `tables_fts` index
- Searchable via `search_tables()` method and `search_tables` MCP tool
- Results include page number, row/column count, and relevance scores
- Critical for C64 documentation (memory maps, register tables, opcode references)
- Stored in `document_tables` database table with automatic FTS5 synchronization

**Code Block Detection** (NEW in v2.1.0):
- Automatic detection of code blocks in all document types (PDFs and text files)
- Supports three code types:
  - **BASIC**: Line-numbered BASIC programs (e.g., "10 PRINT", "20 GOTO")
  - **Assembly**: 6502 assembly mnemonics (LDA, STA, JMP, etc.)
  - **Hex dumps**: Memory dumps with addresses (e.g., "D000: 00 01 02 03")
- Uses regex pattern matching (requires 3+ consecutive lines for detection)
- FTS5 full-text search on code content via `code_fts` index
- Searchable via `search_code()` method and `search_code` MCP tool
- Results include block type, line count, page number, and relevance scores
- Stored in `document_code_blocks` database table with automatic FTS5 synchronization

**Additional Features**:
- Search term highlighting in snippets (markdown bold)
- PDF page number tracking in results
- Tag-based filtering
- Comprehensive logging to server.log

### Similarity Search (Find Similar Documents)

**find_similar_documents()** - Find documents similar to a given document or chunk:
- Dual-mode implementation: semantic embeddings (preferred) or TF-IDF (fallback)
- Supports document-level similarity (when chunk_id is None) and chunk-level similarity
- Tag filtering support for narrowing results

**Semantic Similarity** (`_find_similar_semantic`):
- Uses FAISS embeddings index for fast nearest-neighbor search
- Computes cosine similarity between embedding vectors
- Aggregates chunk scores by document (mean similarity)
- Requires embeddings to be built (`USE_SEMANTIC_SEARCH=1`)

**TF-IDF Similarity** (`_find_similar_tfidf`):
- Builds TF-IDF vectors using sklearn's TfidfVectorizer
- Computes cosine similarity between document/chunk vectors
- No external dependencies beyond sklearn (included in rank-bm25)
- Works without embeddings generation

**MCP Tool**: `find_similar`
- Input: doc_id (required), chunk_id (optional), max_results (default: 5), tags (optional)
- Returns: List of similar documents with similarity scores and snippets
- Automatically selects best available method (semantic > TF-IDF)

## Development Commands

### Setup
```cmd
# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install mcp pypdf rank-bm25 nltk

# For development (includes pytest)
pip install -e ".[dev]"
```

### Testing the Server
```cmd
# Run server standalone (will wait for stdio input)
python server.py

# Test CLI commands
python cli.py stats
python cli.py search "your query"
```

### CLI Usage
```cmd
# Add single document
python cli.py add "C:/path/to/doc.pdf" --title "Title" --tags tag1 tag2

# Bulk add from folder
python cli.py add-folder "C:/docs" --tags reference --recursive

# Search
python cli.py search "query text" --max 10 --tags sid vic-ii

# List all documents
python cli.py list

# Show stats
python cli.py stats

# Remove document
python cli.py remove <doc_id>
```

## Environment Configuration

**TDZ_DATA_DIR** - Directory for database file storage (default: `~/.tdz-c64-knowledge`)
**USE_FTS5** - Enable/disable SQLite FTS5 full-text search (default: `0` for disabled, set to `1` to enable - recommended for best performance)
**USE_SEMANTIC_SEARCH** - Enable/disable semantic search with embeddings (default: `0` for disabled, set to `1` to enable for conceptual search)
**SEMANTIC_MODEL** - Sentence-transformers model to use (default: `all-MiniLM-L6-v2`)
**USE_BM25** - Enable/disable BM25 search algorithm (default: `1` for enabled, set to `0` to disable)
**USE_QUERY_PREPROCESSING** - Enable/disable NLTK query preprocessing (default: `1` for enabled, set to `0` to disable)
**ALLOWED_DOCS_DIRS** - Comma-separated list of allowed document directories for security (optional, no restrictions if not set)

When adding to Claude Code or Claude Desktop, set these in the MCP config `env` section:
```json
"env": {
  "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data",
  "USE_FTS5": "1",
  "USE_SEMANTIC_SEARCH": "1",
  "SEMANTIC_MODEL": "all-MiniLM-L6-v2",
  "USE_BM25": "1",
  "USE_QUERY_PREPROCESSING": "1",
  "ALLOWED_DOCS_DIRS": "C:\\Users\\YourName\\Documents\\C64Docs"
}
```

## Code Patterns

### Adding New Tools

1. Add tool definition in `list_tools()` with proper inputSchema
2. Implement handler in `call_tool()` function
3. Use KnowledgeBase methods for data operations
4. Return list of `TextContent` objects

### Search Algorithm Architecture

Search is implemented in `KnowledgeBase.search()` starting at server.py line ~350.

**Current Implementation:**
- SQLite FTS5 via `_search_fts5()` method (when USE_FTS5=1, recommended)
- BM25 ranking via `_search_bm25()` method (fallback/default)
- Simple term frequency via `_search_simple()` method (fallback)
- Phrase detection and boosting
- Search term highlighting via `_extract_snippet()`

**Key Methods:**
- `search()` - Main entry point, dispatches to FTS5, BM25, or simple search based on environment variables
- `semantic_search()` - Semantic/conceptual search using embeddings and FAISS
- `hybrid_search()` - Combines FTS5 + semantic with configurable weighting (default: 0.3)
- `health_check()` - Comprehensive system diagnostics (database, features, performance)
- `search_tables()` - **NEW** Search for tables in PDFs using FTS5 with tag filtering
- `search_code()` - **NEW** Search for code blocks (BASIC/Assembly/Hex) with type filtering
- `_extract_tables()` - **NEW** Extract tables from PDFs using pdfplumber, convert to markdown
- `_table_to_markdown()` - **NEW** Convert table data to markdown format
- `_detect_code_blocks()` - **NEW** Detect BASIC, Assembly, and Hex dump code blocks via regex
- `_build_embeddings()` - Generates embeddings for all chunks and builds FAISS index
- `_load_embeddings()` - Loads persisted FAISS index from disk
- `_save_embeddings()` - Saves FAISS index to disk
- `_search_fts5()` - SQLite FTS5 search with native BM25 ranking (480x faster)
- `_fts5_available()` - Checks if FTS5 table exists and is ready
- `_search_bm25()` - BM25 scoring with phrase boosting
- `_search_simple()` - Fallback term frequency scoring
- `_build_bm25_index()` - Builds BM25 index from chunks on init/update
- `_extract_snippet()` - Extracts context with term density scoring, complete sentences, code preservation

**Completed Enhancements:**
- ✅ v2.0.0: Hybrid search combining FTS5 + semantic (configurable weighting)
- ✅ v2.0.0: Enhanced snippet extraction (term density, complete sentences, code blocks)
- ✅ v2.0.0: Health monitoring system (diagnostics, metrics, status reporting)
- ✅ v2.1.0: Table extraction from PDFs (pdfplumber, markdown conversion, FTS5 search)
- ✅ v2.1.0: Code block detection (BASIC/Assembly/Hex, regex patterns, FTS5 search)

**Future Enhancements:**
- Query autocompletion based on indexed content
- Fuzzy search / typo tolerance (Levenshtein distance)
- Multi-language support beyond English
- See FUTURE_IMPROVEMENTS.md for detailed roadmap

### Database Access Patterns

All database operations go through KnowledgeBase methods:

**Adding Documents**:
```python
# server.py add_document() -> _add_document_db()
# Uses transaction for ACID guarantees
cursor.execute("BEGIN TRANSACTION")
# Insert document + chunks
cursor.execute("INSERT INTO documents ...")
cursor.execute("INSERT INTO chunks ...")
self.db_conn.commit()
```

**Retrieving Data**:
```python
# Lazy loading - only load what's needed
chunks = self._get_chunks_db(doc_id)  # Load chunks for one document
chunks = self._get_chunks_db()        # Load all chunks (for BM25)
```

**Search Flow (with FTS5 enabled)**:
1. `search()` called → checks if `USE_FTS5=1` and `_fts5_available()`
2. If FTS5 available → `_search_fts5()` executes native SQLite search (~50ms)
3. FTS5 returns results with native BM25 ranking
4. If FTS5 returns no results → falls back to BM25/simple search
5. Results filtered by tags (if specified) and returned

**Search Flow (BM25 fallback)**:
1. `search()` called → checks if `self.bm25` is None
2. If None → `_build_bm25_index()` → `_get_chunks_db()` loads all chunks (~24s first time)
3. BM25 scores calculated → results filtered and sorted
4. Subsequent searches use cached BM25 index (fast)
5. Add/remove operations invalidate cache (`self.bm25 = None`)

**Key Methods**:
- `_init_database()` - Create schema and tables
- `_add_document_db(doc, chunks)` - Insert with transaction
- `_remove_document_db(doc_id)` - Delete (chunks cascade)
- `_get_chunks_db(doc_id)` - Load chunks with JOIN to get filename/title
- `get_chunk(doc_id, chunk_id)` - Query single chunk
- `close()` - Close database connection (important for tests)

### Extending File Type Support

Currently supported formats:
- **PDF** (.pdf) - via pypdf, method: `_extract_pdf_text()`
- **Text** (.txt) - native Python with encoding detection
- **Markdown** (.md) - treated as text files
- **HTML** (.html, .htm) - via BeautifulSoup4, method: `_extract_html_file()`
- **Excel** (.xlsx, .xls) - via openpyxl, method: `_extract_excel_file()`

File type detection is in `add_document()` at server.py:~2230. To add new formats:
1. Add file extension to condition check
2. Implement extraction method (like `_extract_pdf_text` or `_extract_excel_file`)
3. Update tool description and README
4. Update GUI file uploaders in admin_gui.py
5. Update bulk add pattern default in `add_documents_bulk()`

## Windows-Specific Notes

- Uses Windows-style paths (`C:\Users\...`)
- Batch files (.bat) provided for convenience (setup.bat, run.bat, tdz.bat)
- Virtual environment activation: `.venv\Scripts\activate`
- Python executable path for MCP config: `C:\...\\.venv\Scripts\python.exe`
