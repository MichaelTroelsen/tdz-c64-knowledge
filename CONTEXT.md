# TDZ C64 Knowledge Server - Development Context

## Quick Reference
**Before asking questions, check:**
1. This CONTEXT.md file
2. README.md for architecture and setup
3. ARCHITECTURE.md for deep technical details
4. CLAUDE.md for quick reference
5. Python source files (server.py, cli.py, admin_gui.py)
6. /docs/ folder for API documentation (if exists)
7. Test files for usage examples

## Project Overview
TDZ C64 Knowledge is an MCP (Model Context Protocol) server that provides Claude with access to a searchable knowledge base of Commodore 64 documentation, including:
- Memory maps
- Hardware specifications (SID, VIC-II, CIA)
- BASIC and assembly programming references
- Technical manuals and programmer's guides

## Architecture
- **MCP Server**: Python-based server implementing Model Context Protocol (stdio transport)
- **REST API Server**: FastAPI-based HTTP/REST interface (27 endpoints, optional)
- **Knowledge Base**: SQLite database with FTS5 full-text search
- **Document Processing**: PDF, text, Markdown, HTML, and Excel file indexing
- **Search Capabilities**:
  - Keyword search (FTS5 - 480x faster than BM25)
  - Semantic search (embeddings + FAISS, optional, 43% faster with caching)
  - Hybrid search (combines keyword + semantic, 22% faster with parallelization)
  - Natural language query translation (AI-powered)
  - Table extraction (PDF tables → markdown)
  - Code block detection (BASIC/Assembly/Hex)
  - BM25 ranking (fallback)
  - Similarity search (find related docs)
- **AI Features**:
  - Entity extraction (LLM-powered, background processing)
  - Entity analytics and relationship mapping
  - Document summarization
  - Anomaly detection (ML-based baseline learning)

## Key Components
- **server.py** - MCP server, KnowledgeBase class, tool handlers, AI features
- **rest_server.py** - FastAPI REST API server (27 endpoints, optional)
- **rest_models.py** - Pydantic v2 request/response models
- **cli.py** - Command-line interface for batch operations
- **admin_gui.py** - Streamlit web interface with analytics dashboard
- **test_server.py** - Pytest test suite
- **knowledge_base.db** - SQLite database (12+ tables including entities, relationships, extraction_jobs)

## MCP Tools

### Search Tools
- `search_docs` - Full-text search
- `semantic_search` - Meaning-based search
- `hybrid_search` - Combined search
- `translate_query` - Natural language query translation
- `search_tables` - Search extracted tables
- `search_code` - Search code blocks
- `find_similar` - Find similar documents

### Document Management
- `add_document` - Index a file
- `add_documents_bulk` - Bulk import
- `remove_document` - Remove a document
- `remove_documents_bulk` - Bulk remove
- `list_docs` - List all documents
- `get_chunk` - Retrieve specific chunk
- `get_document` - Retrieve full document
- `check_updates` - Check for file updates

### URL Scraping
- `scrape_url` - Scrape documentation website
- `rescrape_document` - Re-scrape existing URL doc
- `check_url_updates` - Check scraped docs for updates

### AI & Analytics
- `extract_entities` - Extract entities from document (LLM)
- `get_entities` - Search extracted entities
- `get_entity_analytics` - Comprehensive entity statistics
- `queue_entity_extraction` - Background entity extraction
- `get_extraction_status` - Check extraction job status
- `get_extraction_jobs` - Monitor extraction queue
- `compare_documents` - Side-by-side document comparison
- `export_entities` - Export entities to CSV/JSON
- `export_relationships` - Export relationships to CSV/JSON

### System
- `kb_stats` - Knowledge base statistics
- `health_check` - System diagnostics

## Integration Points
- Used by Claude Desktop via MCP configuration
- Used by Claude Code for C64 development projects
- Can be used by any MCP-compatible client

## Common Development Tasks
- Adding new document types (see ARCHITECTURE.md "Extending File Type Support")
- Improving search algorithms (see ARCHITECTURE.md "Search Implementation")
- Adding new MCP tools (see ARCHITECTURE.md "Adding New MCP Tools")
- Optimizing chunking strategies (default: 1500 words, 200 word overlap)
- Enhancing metadata extraction
- Extending URL scraping capabilities

## Testing
```cmd
# Run all tests
pytest test_server.py -v

# Test CLI
python cli.py stats
python cli.py search "VIC-II sprite" --max 5

# Test GUI
python -m streamlit run admin_gui.py
```

- Test with real C64 documentation PDFs
- Verify search quality and relevance
- Check MCP protocol compliance
- Test from Claude Desktop/Code

## Environment Variables

### Core Settings
- `TDZ_DATA_DIR` - Database directory (default: ~/.tdz-c64-knowledge)
- `USE_FTS5` - Enable FTS5 search (recommended: `1`)
- `USE_SEMANTIC_SEARCH` - Enable semantic search
- `USE_BM25` - Enable BM25 fallback (default: `1`)
- `USE_QUERY_PREPROCESSING` - Enable NLTK preprocessing (default: `1`)

### Performance & Caching
- `EMBEDDING_CACHE_TTL` - Embedding cache duration (default: 3600s)
- `ENTITY_CACHE_TTL` - Entity cache duration (default: 86400s)

### Security & Integration
- `ALLOWED_DOCS_DIRS` - Security whitelist for document directories
- `MDSCRAPE_PATH` - Path to mdscrape executable
- `AUTO_EXTRACT_ENTITIES` - Auto-queue entity extraction (default: `1`)
- `REST_API_KEY` - API key for REST server authentication

## Version History Highlights

### v2.21.0 (Current) - Anomaly Detection
- Intelligent anomaly detection with ML-based baseline learning
- 1500x performance improvement (3400+ docs/second)
- Automated content change detection for URL-sourced documents

### v2.20.x - Performance & Reliability
- Enhanced URL update checking with concurrent processing
- Performance optimizations and security fixes
- Improved monitoring dashboards

### v2.19.0 - Speed Optimizations
- Semantic search 43% faster with embedding cache
- Hybrid search 22% faster with parallel execution
- Entity extraction caching (4x faster)

### v2.18.0 - REST API & Analytics
- Complete FastAPI REST server (27 endpoints)
- Background entity extraction (zero-delay processing)
- Interactive entity analytics dashboard with network graphs

### v2.17.0 - AI-Powered Intelligence
- Natural language query translation
- AI-powered query parsing and entity detection
- Automatic search mode recommendation

### v2.15-v2.16 - Entity Intelligence
- Entity relationship mapping
- Document comparison features
- Export capabilities (CSV/JSON)

## Related Projects
- **SIDM2**: C64 project that USES this server
- **mcp-c64**: Another C64-related MCP server (development tools)

## Important Notes
- This is SERVER code - it provides tools TO Claude
- Don't confuse with client-side usage (SIDM2 is a CLIENT)
- Changes here affect all projects using the server
- Restart Claude Desktop/Code after server changes
- Database uses ACID transactions for integrity
- Lazy loading architecture for scalability (100k+ documents)
- Content-based duplicate detection via MD5 hashing

## Version
Current: v2.21.0 (Anomaly Detection, Performance Optimizations, REST API, Entity Analytics)
