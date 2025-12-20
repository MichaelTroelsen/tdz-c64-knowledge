# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Structure

- **[README.md](README.md)** - User-facing documentation (installation, configuration, features, tools, usage)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Developer documentation (architecture, database schema, algorithms, code patterns)
- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide
- **[FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md)** - Roadmap and planned enhancements

## Project Overview

This is an MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. It provides tools to ingest PDFs and text files, build a searchable knowledge base, and query it through the MCP interface.

**Key Technologies:**
- MCP server (stdio transport)
- SQLite database with FTS5 full-text search
- Optional semantic search with sentence-transformers + FAISS
- BM25 ranking algorithm for relevance scoring
- URL scraping with mdscrape integration

## Quick Reference

### File Structure

- `server.py` - MCP server implementation, KnowledgeBase class
- `cli.py` - Command-line interface for batch operations
- `admin_gui.py` - Streamlit web interface
- `test_server.py` - Pytest test suite
- `knowledge_base.db` - SQLite database (in TDZ_DATA_DIR)

### Main Components

**KnowledgeBase class** (server.py):
- Core data management (documents, chunks, tables, code blocks)
- Search algorithms (FTS5, BM25, semantic, hybrid)
- Document processing pipeline
- Database operations with ACID transactions

**MCP Tools** - See README.md "Tools" section for user-facing API

### Development Commands

```cmd
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest test_server.py -v

# Run CLI
python cli.py stats
python cli.py search "query" --max 5

# Run GUI
python -m streamlit run admin_gui.py
```

### Environment Configuration

See README.md "Environment Variables" section for full list.

**Key variables:**
- `TDZ_DATA_DIR` - Database storage directory
- `USE_FTS5` - Enable SQLite FTS5 (recommended: `1`)
- `USE_SEMANTIC_SEARCH` - Enable embeddings-based search
- `ALLOWED_DOCS_DIRS` - Security whitelist for document directories
- `MDSCRAPE_PATH` - Path to mdscrape executable for URL scraping

**MCP config example:**
```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\path\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\c64-knowledge-data",
        "USE_FTS5": "1",
        "USE_SEMANTIC_SEARCH": "1",
        "ALLOWED_DOCS_DIRS": "C:\\Downloads\\tdz-c64-knowledge-input"
      }
    }
  }
}
```

## Architecture Quick Reference

For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Data Storage

- **SQLite database** with 4 tables: documents, chunks, document_tables, document_code_blocks
- **FTS5 indexes** for full-text search on chunks, tables, and code
- **Lazy loading** - metadata loaded at startup, chunks on-demand
- **Chunking** - 1500 words with 200 word overlap

### Search Algorithms

1. **FTS5** (recommended) - Native SQLite full-text search, 480x faster than BM25
2. **Semantic** - Embeddings + FAISS for meaning-based search
3. **Hybrid** - Combines FTS5 + semantic with configurable weighting
4. **BM25** - Fallback probabilistic ranking
5. **Simple** - Basic term frequency (fallback)

See ARCHITECTURE.md "Search Implementation" for details.

### Document Processing Pipeline

1. File ingestion (PDF/text/HTML/Excel)
2. Text extraction
3. Table extraction (PDFs only, via pdfplumber)
4. Code block detection (BASIC/Assembly/Hex)
5. Chunking with overlap
6. Content-based ID generation (MD5 hash)
7. Duplicate detection
8. Database persistence (ACID transaction)

### URL Scraping (v2.14.0)

**Methods:**
- `scrape_url()` - Scrape website, add to knowledge base
- `rescrape_document()` - Re-scrape existing URL-sourced doc
- `check_url_updates()` - Check all scraped docs for updates

**Features:**
- Concurrent scraping (configurable threads)
- Smart content extraction via mdscrape
- Depth control, URL filtering, rate limiting
- Update detection (Last-Modified headers, content hashes)

See ARCHITECTURE.md "URL Scraping" for implementation details.

## Code Patterns

### Adding New MCP Tools

1. Add tool definition in `list_tools()` with proper inputSchema
2. Implement handler in `call_tool()` function
3. Use KnowledgeBase methods for data operations
4. Return list of `TextContent` objects

### Database Operations

All database operations use KnowledgeBase methods with ACID transactions:
- `_add_document_db(doc, chunks)` - Insert with transaction
- `_remove_document_db(doc_id)` - Delete (cascades to chunks)
- `_get_chunks_db(doc_id)` - Load chunks (lazy loading)

### Extending File Type Support

Currently: PDF, TXT, MD, HTML, Excel

To add new format:
1. Add extension check in `add_document()` at server.py:~2230
2. Implement extraction method (e.g., `_extract_docx_file()`)
3. Update README.md, admin_gui.py file uploaders

See ARCHITECTURE.md "Extending File Type Support" for details.

## Testing

```cmd
# Run all tests
pytest test_server.py -v

# Run specific test
pytest test_server.py::TestKnowledgeBase::test_search_basic -v

# With coverage
pytest test_server.py -v --cov=server --cov-report=term
```

## Windows-Specific Notes

- Uses Windows-style paths (`C:\Users\...`)
- Batch files (.bat) for convenience
- Virtual environment: `.venv\Scripts\activate`
- Python path: `.venv\Scripts\python.exe`
