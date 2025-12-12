# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. It provides tools to ingest PDFs and text files, build a searchable knowledge base, and query it through the MCP interface.

## Core Architecture

### Main Components

**server.py** - MCP server implementation
- `KnowledgeBase` class: Core data management (index + chunks storage)
- MCP tool handlers: `search_docs`, `add_document`, `get_chunk`, `get_document`, `list_docs`, `remove_document`, `kb_stats`
- MCP resource handlers: Exposes documents as `c64kb://` URIs
- Async server running on stdio transport

**cli.py** - Command-line interface for batch operations
- Wraps `KnowledgeBase` for CLI usage
- Commands: `add`, `add-folder`, `search`, `list`, `remove`, `stats`
- Useful for bulk importing documents

### Data Storage Model

The knowledge base uses **SQLite database** for efficient storage and querying:
- **knowledge_base.db** - SQLite database with two main tables:
  - `documents` table - Stores DocumentMeta objects (13 fields including doc_id, title, tags, metadata)
  - `chunks` table - Stores DocumentChunk objects (5 fields: doc_id, chunk_id, page, content, word_count)
  - Foreign key relationship: chunks.doc_id → documents.doc_id (CASCADE delete)
  - Indexes on filepath, file_type, and doc_id for fast queries

**Migration from JSON**: Legacy JSON files (index.json, chunks/*.json) are automatically migrated to SQLite on first run. JSON files are preserved as backup.

**Lazy Loading**: Only document metadata is loaded at startup. Chunks are loaded on-demand for search (building BM25 index) and retrieval operations. This enables the system to scale to 100k+ documents without memory issues.

**Chunking Strategy**: Documents are split into overlapping chunks (default 1500 words, 200 word overlap) to enable granular search and retrieval.

### Document Processing Pipeline

1. File ingestion (PDF via pypdf, text files with encoding detection)
2. Text extraction (pages joined with "--- PAGE BREAK ---")
3. Chunking with overlap (_chunk_text method)
4. Index generation (doc_id from MD5 hash of filepath)
5. Database persistence (document + chunks inserted in ACID transaction via `_add_document_db()`)

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

**Additional Features**:
- Search term highlighting in snippets (markdown bold)
- PDF page number tracking in results
- Tag-based filtering
- Comprehensive logging to server.log

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
- `_build_embeddings()` - Generates embeddings for all chunks and builds FAISS index
- `_load_embeddings()` - Loads persisted FAISS index from disk
- `_save_embeddings()` - Saves FAISS index to disk
- `_search_fts5()` - SQLite FTS5 search with native BM25 ranking (480x faster)
- `_fts5_available()` - Checks if FTS5 table exists and is ready
- `_search_bm25()` - BM25 scoring with phrase boosting
- `_search_simple()` - Fallback term frequency scoring
- `_build_bm25_index()` - Builds BM25 index from chunks on init/update
- `_extract_snippet()` - Extracts context with term highlighting

**Future Enhancements:**
- Implement semantic search with embeddings (sentence-transformers)
- Add query preprocessing (stemming, stopwords)
- Fuzzy search / typo tolerance (Levenshtein distance)
- See IMPROVEMENTS.md for detailed recommendations

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

File type detection is in `add_document()` at server.py:~560. To add new formats:
1. Add file extension to condition check
2. Implement extraction method (like `_extract_pdf_text`)
3. Update tool description and README

## Windows-Specific Notes

- Uses Windows-style paths (`C:\Users\...`)
- Batch files (.bat) provided for convenience (setup.bat, run.bat, tdz.bat)
- Virtual environment activation: `.venv\Scripts\activate`
- Python executable path for MCP config: `C:\...\\.venv\Scripts\python.exe`
