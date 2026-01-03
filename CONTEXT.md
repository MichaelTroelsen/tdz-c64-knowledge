# TDZ C64 Knowledge Server - Development Context

## Quick Reference

**Before asking questions, check:**
1. CONTEXT.md (this file) - Current status, quick stats
2. CLAUDE.md - Dev commands, code patterns
3. README.md - Features, installation, tools
4. ARCHITECTURE.md - Technical deep dive
5. Source files (server.py, cli.py, admin_gui.py, rest_server.py)

## Project Overview

MCP server providing Claude with searchable Commodore 64 documentation (memory maps, hardware specs, programming references, technical manuals).

**Architecture:**
- MCP Server (Python, stdio transport) + optional REST API (FastAPI, 27 endpoints)
- SQLite database (12+ tables, FTS5 full-text search)
- Multi-format ingestion: PDF, text, Markdown, HTML, Excel, web scraping
- Search: FTS5 (480x faster), semantic (FAISS), hybrid, fuzzy, RAG
- AI: Entity extraction, relationship mapping, anomaly detection, question answering

## Current Status - v2.23.1

**Development Phase:**
- ✅ Phase 1: AI-Powered Intelligence (v2.13-v2.22.0) - Complete
- ✅ Phase 2: Advanced Search & Discovery (v2.23.0-v2.23.1) - Complete
  - RAG question answering with citations
  - Fuzzy search with typo tolerance
  - Progressive search refinement
  - Smart document tagging
- ✅ Phase 3: Content Intelligence (v2.15-v2.22.0) - Complete
  - Entity extraction, relationship mapping
  - Version tracking, anomaly detection
- 🎯 **Current Focus:** Production stability, maintenance, feature refinement

## Key Stats & Performance

- **Scalability:** Tested to 5,000+ documents with excellent performance
- **Search Performance:** FTS5 85ms avg, Semantic 16ms avg, Hybrid 142ms avg
- **Throughput:** 5,712 concurrent queries/sec (10 workers), 3,400+ docs/sec anomaly detection
- **Entity Extraction:** 5000x faster with C64-specific regex (1ms vs 5s LLM-only)
- **Database:** 12+ tables, ACID transactions, lazy loading, content-based deduplication

## Core Components

- **server.py** - MCP server, KnowledgeBase class, 50+ tools, AI features
- **rest_server.py** - FastAPI REST API (27 endpoints, optional)
- **rest_models.py** - Pydantic v2 models
- **cli.py** - Command-line interface
- **admin_gui.py** - Streamlit dashboard
- **test_server.py** - Pytest test suite
- **knowledge_base.db** - SQLite database (in TDZ_DATA_DIR)

## MCP Tools Summary

**50+ tools organized by category:**
- Search (11): search_docs, semantic_search, hybrid_search, fuzzy_search, search_within_results, answer_question, translate_query, search_tables, search_code, find_similar, faceted_search
- Documents (6): add_document, add_documents_bulk, remove_document, remove_documents_bulk, list_docs, get_document, get_chunk, check_updates
- URL Scraping (3): scrape_url, rescrape_document, check_url_updates
- AI & Analytics (14): extract_entities, get_entities, search_entities, entity_stats, extract_entities_bulk, extract_entity_relationships, get_entity_relationships, find_related_entities, search_entity_pair, extract_relationships_bulk, get_entity_analytics, compare_documents, suggest_tags, add_tags_to_document, get_tags_by_category
- Export (3): export_entities, export_relationships, export_documents_bulk
- System (2): kb_stats, health_check

See README.md for complete tool documentation.

## Integration Points

- **Claude Desktop** - Via MCP configuration (%APPDATA%\Claude\claude_desktop_config.json)
- **Claude Code** - Via `.claude/settings.json` or `claude mcp add`
- **REST API** - FastAPI server on port 8000 (optional)
- **CLI** - Direct command-line usage
- **GUI** - Streamlit web interface

## Environment Variables

**Essential:**
- `TDZ_DATA_DIR` - Database directory (default: ~/.tdz-c64-knowledge)
- `USE_FTS5=1` - Enable FTS5 search (recommended)
- `USE_SEMANTIC_SEARCH=1` - Enable semantic search (optional)

**Security:**
- `ALLOWED_DOCS_DIRS` - Whitelist document directories
- `REST_API_KEY` - API authentication

**Performance:**
- `EMBEDDING_CACHE_TTL=3600` - Cache duration (seconds)
- `ENTITY_CACHE_TTL=86400` - Entity cache duration

See README.md for complete list.

## Recent Version Highlights

**v2.23.0-v2.23.1** - RAG Question Answering & Advanced Search (Phase 2 Complete)
- RAG-based answer_question with citations, confidence scoring
- Fuzzy search (rapidfuzz) with typo tolerance
- Progressive search refinement (search_within_results)
- Smart tagging (suggest_tags, get_tags_by_category, add_tags_to_document)

**v2.22.0** - Phase 1 Complete & Search Optimizations
- Enhanced entity analytics with relationship tracking
- C64-specific regex patterns (5000x faster entity extraction)
- Distance-based relationship strength scoring
- Health check optimization (93% faster)

**v2.21.0** - Anomaly Detection
- ML-based baseline learning (30-day rolling window)
- 1500x performance improvement (3400+ docs/second)
- Multi-dimensional anomaly scoring

## Development Tasks

**Common operations:**
- Adding new file types → ARCHITECTURE.md "Extending File Type Support"
- Improving search algorithms → ARCHITECTURE.md "Search Implementation"
- Adding MCP tools → ARCHITECTURE.md "Adding New MCP Tools"
- Optimizing chunking → Default: 1500 words, 200 word overlap
- Extending URL scraping → Uses mdscrape integration

## Testing

```cmd
pytest test_server.py -v                    # All tests
python cli.py stats                         # Test CLI
python -m streamlit run admin_gui.py       # Test GUI
```

**Test with real C64 PDFs to verify:**
- Search quality and relevance
- MCP protocol compliance
- Claude Desktop/Code integration

## Important Notes

- **This is SERVER code** - Provides tools TO Claude (not client code)
- Changes affect ALL projects using this server
- Restart Claude Desktop/Code after server changes
- Database uses ACID transactions for integrity
- Lazy loading enables 100k+ document scalability
- Content-based duplicate detection via MD5 hashing

## Related Projects

- **SIDM2** - C64 project that USES this server (client)
- **mcp-c64** - Another C64 MCP server (development tools)
