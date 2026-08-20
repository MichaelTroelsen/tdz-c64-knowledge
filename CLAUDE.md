# CLAUDE.md

Quick reference for Claude Code when working with this codebase.

## Documentation Index

- **README.md** - Installation, features, tools, usage examples
- **docs/ARCHITECTURE.md** - Technical details, database schema, algorithms
- **CONTEXT.md** - Project status, version history, quick overview
- **docs/QUICKSTART.md** - Fast setup guide
- **docs/ROADMAP.md** - Roadmap

## Project Summary

MCP server for searching Commodore 64 documentation. Ingests PDFs/text/web pages, builds searchable SQLite knowledge base with FTS5, semantic search, and AI-powered features (RAG, entity extraction, anomaly detection).

**Stack:** Python 3.10+, SQLite+FTS5, MCP protocol, FAISS, sentence-transformers, FastAPI (optional REST)

## File Structure

- `server.py` - MCP server entry point (576 lines): transports (stdio + streamable HTTP) and tool dispatch
- `kb/` - `KnowledgeBase` class, split into domain mixins (`core.py`, `ingest/`, `search/`, `entities/`, `graph.py`, `topics.py`, `temporal.py`, `figures.py`, `admin.py`)
- `mcp_tools/` - MCP tool layer: `schemas.py` (the 94 `Tool(...)` literals), `handlers.py` (aggregator), plus 8 domain handler modules (`admin.py`, `documents.py`, `entities.py`, `figures.py`, `knowledge_graph.py`, `search.py`, `temporal.py`, `topics.py`)
- `util.py`, `models.py`, `text_utils.py`, `features.py` - shared module preamble (formerly the top of server.py)
- `rest_server.py` - FastAPI REST API (18 endpoints, optional)
- `cli.py` - Command-line interface
- `admin_gui.py` - Streamlit entry point (193 lines), dispatches to `admin_pages/`
- `admin_pages/` - 16 page bodies, each exposing `render(kb)`
- `admin_common.py` - 4 shared helpers used by `admin_pages/`
- `test_card_updates.py` - Pytest test suite
- `test_mcp_startup.py` - MCP startup/connectivity regression tests (handshake speed, concurrent sessions, lazy imports, WAL, DB thread-safety, extraction-job recovery)
- `test_mcp_tool_dispatch.py` - Smoke-calls every registered MCP tool
- `test_rest_api.py` - REST endpoint/auth tests
- `test_figure_ocr.py` - Background figure-OCR batch pass
- `test_wiki_safety.py` - Wiki export output-safety (URL scheme validation, escaping)
- `test_scrape_politeness.py` - robots.txt, User-Agent, backoff
- `knowledge_base.db` - SQLite database (in TDZ_DATA_DIR)

## Development Commands

```cmd
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Test
pytest test_card_updates.py -v

# CLI
python cli.py stats
python cli.py search "VIC-II" --max 5

# GUI
python -m streamlit run admin_gui.py
```

## Key Environment Variables

- `TDZ_DATA_DIR` - Database directory (default: ~/.tdz-c64-knowledge)
- `USE_FTS5=1` - Enable FTS5 full-text search (recommended)
- `USE_SEMANTIC_SEARCH=1` - Enable embeddings-based search
- `ALLOWED_DOCS_DIRS` - Security whitelist for document paths
- `MDSCRAPE_PATH` - Path to mdscrape executable

See README.md for complete environment variable list.

## MCP Configuration Example

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\path\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\data",
        "USE_FTS5": "1"
      }
    }
  }
}
```

## Architecture Overview

**Database:** SQLite with 22 tables (documents, chunks, entities, relationships, etc.)
**Search:** FTS5 (480x faster), semantic (FAISS), hybrid, BM25 fallback
**Chunking:** 1500 words, 200 word overlap, lazy loading
**Processing:** PDF/text/HTML/Excel → extract → chunk → index → search

See docs/ARCHITECTURE.md for detailed technical documentation.

## Common Code Patterns

### Adding MCP Tools
1. Add a `Tool(...)` schema to `TOOL_SCHEMAS` in `mcp_tools/schemas.py`
2. Implement the handler in the relevant `mcp_tools/<domain>.py` module and register it in that module's `HANDLERS_<DOMAIN>` dict
3. Return list of `TextContent` objects

### Database Operations
Use KnowledgeBase methods (ACID transactions):
- `_add_document_db(doc, chunks)` - Insert with transaction
- `_remove_document_db(doc_id)` - Delete (cascades)
- `_get_chunks_db(doc_id)` - Lazy load chunks

### Extending File Types
1. Add extension check in `add_document()` (`kb/ingest/_documents.py`)
2. Implement `_extract_X_file()` method (`kb/ingest/_extraction.py`)
3. Update README.md and `admin_pages/documents.py`

See docs/ARCHITECTURE.md "Extending File Type Support" for details.

## Testing

```cmd
pytest test_card_updates.py -v                                          # All tests
pytest test_card_updates.py::test_no_orphaned_chunks_after_remove -v    # Specific test
pytest test_card_updates.py --cov=server --cov-report=term              # With coverage
pytest test_mcp_startup.py -v                                           # MCP connectivity
```

### MCP startup performance

`test_mcp_startup.py` guards the constraint that broke multi-session use:
Claude Code allows **30s** for the MCP initialize handshake, and every session
spawns its own `server.py` process against the shared database.

Keep startup fast — **do not import heavy optional dependencies at module
level**. `sentence-transformers`, `torch`, `transformers` and `nltk` cost ~16s
between them and are imported on first use instead (see `_LazyModule`,
`_ensure_nltk`, and the `SentenceTransformer` factory near the top of
`server.py`). Availability is detected with `importlib.util.find_spec`, so
feature flags stay accurate without paying the import cost.

The database runs in **WAL** journal mode so concurrent server processes do not
serialise behind a single exclusive writer lock. `TDZ_DB_BUSY_TIMEOUT_MS`
(default 30000) tunes the SQLite busy timeout.

## Windows Notes

- Paths: `C:\Users\...`
- Activate: `.venv\Scripts\activate`
- Python: `.venv\Scripts\python.exe`
- Batch files: setup.bat, start-all.bat
