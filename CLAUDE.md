# CLAUDE.md

Quick reference for Claude Code when working with this codebase.

## Documentation Index

- **README.md** - Installation, features, tools, usage examples
- **ARCHITECTURE.md** - Technical details, database schema, algorithms
- **CONTEXT.md** - Project status, version history, quick overview
- **QUICKSTART.md** - Fast setup guide
- **FUTURE_IMPROVEMENTS.md** - Roadmap

## Project Summary

MCP server for searching Commodore 64 documentation. Ingests PDFs/text/web pages, builds searchable SQLite knowledge base with FTS5, semantic search, and AI-powered features (RAG, entity extraction, anomaly detection).

**Stack:** Python 3.10+, SQLite+FTS5, MCP protocol, FAISS, sentence-transformers, FastAPI (optional REST)

## File Structure

- `server.py` - MCP server, KnowledgeBase class, 50+ tools
- `rest_server.py` - FastAPI REST API (27 endpoints, optional)
- `cli.py` - Command-line interface
- `admin_gui.py` - Streamlit web UI
- `test_server.py` - Pytest test suite
- `knowledge_base.db` - SQLite database (in TDZ_DATA_DIR)

## Development Commands

```cmd
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Test
pytest test_server.py -v

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

**Database:** SQLite with 12+ tables (documents, chunks, entities, relationships, etc.)
**Search:** FTS5 (480x faster), semantic (FAISS), hybrid, BM25 fallback
**Chunking:** 1500 words, 200 word overlap, lazy loading
**Processing:** PDF/text/HTML/Excel → extract → chunk → index → search

See ARCHITECTURE.md for detailed technical documentation.

## Common Code Patterns

### Adding MCP Tools
1. Add definition in `list_tools()` with inputSchema
2. Implement handler in `call_tool()`
3. Return list of `TextContent` objects

### Database Operations
Use KnowledgeBase methods (ACID transactions):
- `_add_document_db(doc, chunks)` - Insert with transaction
- `_remove_document_db(doc_id)` - Delete (cascades)
- `_get_chunks_db(doc_id)` - Lazy load chunks

### Extending File Types
1. Add extension check in `add_document()` (~line 2230)
2. Implement `_extract_X_file()` method
3. Update README.md and admin_gui.py

See ARCHITECTURE.md "Extending File Type Support" for details.

## Testing

```cmd
pytest test_server.py -v                                    # All tests
pytest test_server.py::TestKnowledgeBase::test_search -v   # Specific test
pytest test_server.py --cov=server --cov-report=term       # With coverage
```

## Windows Notes

- Paths: `C:\Users\...`
- Activate: `.venv\Scripts\activate`
- Python: `.venv\Scripts\python.exe`
- Batch files: setup.bat, start-all.bat
