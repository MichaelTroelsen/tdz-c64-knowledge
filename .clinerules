# TDZ C64 Knowledge MCP Server - Project Rules

## Project Type
This is an MCP (Model Context Protocol) server that provides C64 knowledge base access to Claude

## Available Resources
- README.md - Main project documentation and setup instructions
- ARCHITECTURE.md - Developer documentation (architecture, database schema, algorithms)
- CLAUDE.md - Quick reference for Claude Code
- QUICKSTART.md - Quick setup guide
- FUTURE_IMPROVEMENTS.md - Roadmap and planned enhancements
- API documentation in /docs/API.md (if exists)
- MCP server implementation in Python
- Knowledge base tools and search functionality
- Test files and examples

## Before Answering Questions
1. Check README.md for project architecture and setup
2. Review ARCHITECTURE.md for implementation details
3. Review Python source files for specific code questions
4. Check existing documentation in /docs/ folder
5. Look at test files for usage examples

## Key Technologies
- Python MCP SDK
- SQLite FTS5 (full-text search)
- Semantic search with embeddings (optional)
- BM25 ranking algorithm
- PDF/text document processing
- Web scraping capabilities (mdscrape)
- Streamlit GUI (admin_gui.py)

## Workflow Guidelines
1. This server is USED BY other projects (like SIDM2) - don't confuse server code with client usage
2. When modifying tools, update both implementation AND documentation
3. Test changes with actual C64 documentation files
4. Consider backwards compatibility with existing clients
5. All database operations must use ACID transactions
6. Maintain lazy loading architecture for scalability

## Common Tasks
- Adding new search tools/capabilities
- Processing and indexing documentation
- Improving search relevance
- Web scraping integration
- Embedding generation and semantic search
- Extending file type support (currently: PDF, TXT, MD, HTML, Excel)

## Architecture Notes
- SQLite database with 4 tables: documents, chunks, document_tables, document_code_blocks
- FTS5 indexes for full-text search
- Lazy loading: metadata at startup, chunks on-demand
- Content-based duplicate detection via MD5 hashing
- Search algorithms: FTS5 (primary), Semantic, Hybrid, BM25 (fallback), Simple (fallback)

## Testing
- Run tests: `pytest test_server.py -v`
- Use CLI for quick testing: `python cli.py search "query"`
- Use GUI for visual testing: `python -m streamlit run admin_gui.py`
