# TDZ C64 Knowledge

[![Version](https://img.shields.io/badge/version-2.23.15-brightgreen.svg)](https://github.com/MichaelTroelsen/tdz-c64-knowledge)
[![CI/CD Pipeline](https://github.com/MichaelTroelsen/tdz-c64-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelTroelsen/tdz-c64-knowledge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

MCP server for managing and searching Commodore 64 documentation. Ingest PDFs, text, Markdown, HTML, Excel, and web pages into a searchable knowledge base accessible via Claude Code or other MCP clients.

## 🚀 Quick Start

```cmd
# 1. Install
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 2. Configure Claude Code
claude mcp add tdz-c64-knowledge -- .venv\Scripts\python.exe server.py

# 3. Add documents
.venv\Scripts\python.exe cli.py add-folder "C:\c64docs" --tags reference --recursive

# 4. Search via Claude Code
# Ask: "Search the C64 docs for VIC-II sprite registers"
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.

## Features

### Search & Retrieval
- **FTS5 full-text search** - 480x faster queries (50ms vs 24s)
- **Semantic search** - Find by meaning, not keywords (e.g., "movable objects" → "sprites")
- **RAG question answering** - Answer questions by synthesizing docs with citations
- **Fuzzy search** - Typo tolerance ("VIC2" → "VIC-II", "asembly" → "assembly")
- **Progressive refinement** - Search within results to narrow down
- **Hybrid search** - Combines keyword + semantic with configurable weighting
- **Similarity search** - Discover related documentation automatically
- **Query preprocessing** - NLTK stemming and stopword removal
- **Smart tagging** - AI-powered tag suggestions by category
- **Table/code search** - Search extracted tables and code blocks

### Document Management
- **Multi-format** - PDF, TXT, MD, HTML, Excel, web scraping
- **Duplicate detection** - Content-based deduplication
- **Chunked retrieval** - Get specific sections without loading entire docs
- **Metadata extraction** - Author, subject, page numbers
- **Persistent index** - Documents stay indexed between sessions

### AI-Powered Features
- **Entity extraction** - Extract hardware, memory addresses, instructions, concepts (5000x faster with C64 regex patterns)
- **Relationship mapping** - Co-occurrence analysis with distance-based strength scoring
- **Document comparison** - Side-by-side analysis with similarity scores
- **Natural language query translation** - Parse queries into structured search parameters
- **Anomaly detection** - ML-based baseline learning for URL-sourced content (3400+ docs/second)
- **Temporal analysis** - Event detection, timeline construction, historical context (5 event types, 8 date formats)
- **Advanced visualizations** - 3D knowledge graphs, hierarchical bundling, Sankey flow diagrams

### Wiki Export (NEW in v2.23.15)
- **Static HTML wiki** - Export entire knowledge base to browsable website
- **Document similarity map** - 2D visualization using UMAP/t-SNE dimensionality reduction
- **Interactive timeline** - Horizontal scrollable timeline with zoom levels and event filters
- **Knowledge graph** - D3.js force-directed graph (178 entities, 20 relationships)
- **Enhanced UI** - Explanation boxes, prominent ASK AI button, file type detection
- **Clickable clusters** - Browse k-means clusters with linked documents
- **No server required** - Pure client-side JavaScript, works offline
- **Full-text search** - Fuse.js powered search across all content
- See [WIKI_EXPORT_GUIDE.md](WIKI_EXPORT_GUIDE.md) for usage

### REST API (Optional)
- **27 endpoints** - Full CRUD, search, analytics, export
- **OpenAPI/Swagger docs** - Interactive API at `/api/docs`
- **API authentication** - Secure via X-API-Key header
- See [docs/REST_API.md](docs/REST_API.md) for details

### Performance
- **Scalability** - Tested to 5,000+ documents
- **Concurrent throughput** - 5,712 queries/sec (10 workers)
- **Lazy loading** - 100k+ document support
- **Search caching** - 50-100x speedup for repeated queries

## Installation (Windows)

### Prerequisites
- **Python 3.10+** - https://python.org (check "Add Python to PATH")
- **uv** (recommended) or pip: `pip install uv`

### Setup

```cmd
cd C:\Users\YourName\mcp-servers\tdz-c64-knowledge

# Using uv (faster)
uv venv
.venv\Scripts\activate
uv pip install mcp pypdf rank-bm25 nltk

# Or using pip
python -m venv .venv
.venv\Scripts\activate
pip install mcp pypdf rank-bm25 nltk

# Test
python server.py  # Press Ctrl+C to stop
```

## Configuration

### Claude Code

```cmd
claude mcp add tdz-c64-knowledge -- C:\path\.venv\Scripts\python.exe C:\path\server.py
```

Or add to `.claude/settings.json`:
```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\path\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\c64-knowledge-data"
      }
    }
  }
}
```

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\path\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\c64-knowledge-data"
      }
    }
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TDZ_DATA_DIR` | Database directory | `~/.tdz-c64-knowledge` |
| `USE_FTS5` | Enable FTS5 search (recommended) | `0` |
| `USE_SEMANTIC_SEARCH` | Enable semantic search | `0` |
| `SEMANTIC_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `USE_BM25` | Enable BM25 fallback | `1` |
| `USE_QUERY_PREPROCESSING` | Enable NLTK preprocessing | `1` |
| `USE_FUZZY_SEARCH` | Enable fuzzy search | `1` |
| `FUZZY_THRESHOLD` | Fuzzy similarity (0-100) | `80` |
| `USE_OCR` | Enable OCR for scanned PDFs | `1` |
| `SEARCH_CACHE_SIZE` | Max cached results | `100` |
| `SEARCH_CACHE_TTL` | Cache TTL (seconds) | `300` |
| `ALLOWED_DOCS_DIRS` | Document directory whitelist | None |

## Search Features

### FTS5 Full-Text Search (Recommended)

Enable with `USE_FTS5=1` for maximum performance:
- **480x faster** than BM25
- Native SQLite BM25 ranking
- Porter stemming tokenizer

### Semantic Search

Enable with `USE_SEMANTIC_SEARCH=1`:
- Meaning-based search (e.g., "movable objects" finds "sprites")
- FAISS vector similarity with sentence-transformers
- ~7-16ms per query after embeddings built
- Pre-build embeddings: `pip install sentence-transformers faiss-cpu`

### Phrase Search

Use double quotes for exact phrases:
```
search_docs(query='"VIC-II chip" registers')
```

### Fuzzy Search

Handles typos automatically with `USE_FUZZY_SEARCH=1`:
- "VIC-I" → "VIC-II" (83% similarity)
- "grafics" → "graphics" (88% similarity)
- Configurable threshold (default: 80%)

### OCR for Scanned PDFs

Automatic with `USE_OCR=1`:
- Detects scanned PDFs (< 100 chars extracted)
- Uses Tesseract OCR
- Install: `pip install pytesseract pdf2image Pillow` + Tesseract binary
- ~1-2 seconds per page

## Temporal Analysis & Visualizations

Extract events, construct timelines, and visualize knowledge graphs.

### Event Detection

Automatically detect significant events in documents:
- **5 Event Types** - Product releases, company milestones, technical innovations, cultural events, version updates
- **8 Date Formats** - Full dates, month-year, year ranges, decades, parenthetical dates
- **Confidence Scoring** - Pattern matching with proximity-based confidence (0.0-1.0)
- **Entity Association** - Automatically link entities to events

```python
# Extract events from a document
result = kb.extract_document_events('doc_id', min_confidence=0.7)
# Returns: event_count, filtered_count, stored_count, events list
```

### Timeline Construction

Build chronological timelines with flexible querying:
- **Automatic Timeline Building** - Chronologically sorted by date (YYYYMMDD integer sort)
- **Category Organization** - Group by decade-type combinations (e.g., "1980s-release")
- **Importance Levels** - 1-5 scale based on confidence
- **Date Range Filtering** - Query events by year range, type, importance

```python
# Build timeline from events
timeline_result = kb.build_timeline(min_confidence=0.5)

# Query timeline
timeline = kb.get_timeline(start_year=1980, end_year=1989, min_importance=3)

# Get historical context
context = kb.get_historical_context(year=1982, context_years=2)
```

### Interactive Visualizations

Generate interactive HTML visualizations with Plotly and NetworkX:

**Timeline Visualizations:**
- **Interactive Timeline** - Horizontal timeline with zoom/pan, color-coded by event type
- **Event Network** - Spring layout showing event relationships
- **Trend Charts** - Multi-subplot dashboard (bar chart, stacked area, cumulative line)

**Advanced Graph Visualizations:**
- **3D Knowledge Graph** - Interactive 3D entity-relationship graph with rotation controls
- **Hierarchical Bundling** - Circular layout with curved edges bundled through center
- **Sankey Diagrams** - Topic flow over time (decade or year grouping)

```python
# Generate visualizations
kb.visualize_timeline(start_year=1980, end_year=1990, output_path="timeline.html")
kb.visualize_knowledge_graph_3d(max_entities=50, output_path="graph_3d.html")
kb.visualize_hierarchical_bundling(max_entities=30, output_path="bundling.html")
kb.visualize_topic_flow_sankey(time_period='decade', output_path="flow.html")
```

### MCP Tools for Timeline

4 timeline-specific MCP tools:
- `extract_document_events` - Extract and store events from documents
- `get_timeline` - Query chronological timeline with filters
- `search_events_by_date` - Search events by date range and type
- `get_historical_context` - Get events around a specific year

See [PHASE3_TEMPORAL_ANALYSIS.md](PHASE3_TEMPORAL_ANALYSIS.md) for complete documentation.

## Tools

62 MCP tools organized by category. Key tools listed below.

### Search Tools

**search_docs** - Full-text search
```
search_docs(query="SID register", max_results=5, tags=["sid"])
```

**semantic_search** - Meaning-based search
```
semantic_search(query="How do sprites work?", max_results=5)
```

**hybrid_search** - Combined keyword + semantic
```
hybrid_search(query="SID chip", semantic_weight=0.7, max_results=10)
```

**answer_question** - RAG-based Q&A with citations
```
answer_question(
  question="How do I program sprites on the VIC-II?",
  max_sources=5,
  search_mode="auto"
)
```

**fuzzy_search** - Typo-tolerant search
```
fuzzy_search(query="VIC2 asembly", similarity_threshold=80)
```

**search_within_results** - Progressive refinement
```
# Broad search, then refine
results = search_docs(query="VIC-II", max_results=50)
refined = search_within_results(results, "sprite collision", max_results=5)
```

**find_similar** - Find related documents
```
find_similar(doc_id="abc123", max_results=5)
```

### Document Management

**add_document** - Add a file
```
add_document(
  filepath="C:/docs/c64_ref.pdf",
  title="C64 Programmer's Reference",
  tags=["reference", "memory-map"]
)
```

**add_documents_bulk** - Bulk import
```
add_documents_bulk(
  directory="C:/c64docs",
  pattern="**/*.{pdf,txt}",
  tags=["reference"],
  recursive=true
)
```

**list_docs** - List all documents

**get_chunk** - Get specific chunk
```
get_chunk(doc_id="abc123", chunk_id=5)
```

**remove_document** - Remove a document

**remove_documents_bulk** - Bulk remove by IDs or tags
```
remove_documents_bulk(tags=["outdated"])
```

**check_updates** - Check for file changes
```
check_updates(auto_update=false)
```

### URL Scraping

**scrape_url** - Scrape documentation website
```
scrape_url(
  url="https://www.c64-wiki.com/wiki/VIC",
  tags=["wiki"],
  depth=2,
  threads=5
)
```

**rescrape_document** - Re-scrape for updates
```
rescrape_document(doc_id="abc123", force=false)
```

**check_url_updates** - Check all scraped docs
```
check_url_updates(auto_rescrape=false, check_structure=true)
```

### AI & Analytics

**extract_entities** - Extract named entities
```
extract_entities(doc_id="abc123", confidence_threshold=0.6)
```

**search_entities** - Search across entities
```
search_entities(query="VIC-II", entity_types=["hardware"])
```

**get_entity_analytics** - Comprehensive entity statistics

**extract_entity_relationships** - Extract co-occurrences
```
extract_entity_relationships(doc_id="abc123", min_strength=0.3)
```

**search_entity_pair** - Find docs with entity pair
```
search_entity_pair(entity1="VIC-II", entity2="sprite")
```

**compare_documents** - Side-by-side comparison
```
compare_documents(doc_id_1="abc", doc_id_2="def", comparison_type="full")
```

**suggest_tags** - AI-powered tag suggestions
```
suggest_tags(doc_id="abc123", confidence_threshold=0.6)
```

**get_tags_by_category** - Browse tags by category

**translate_query** - Parse natural language queries
```
translate_query(query="find sprites on VIC-II chip")
```

### Export Tools

**export_entities** - Export to CSV/JSON
```
export_entities(format="csv", output_path="entities.csv", min_confidence=0.7)
```

**export_relationships** - Export relationships
```
export_relationships(format="json", output_path="rels.json", min_strength=0.5)
```

### System

**kb_stats** - Knowledge base statistics

**health_check** - System diagnostics

## Data Storage

SQLite database with 12+ tables:
- **documents** - Document metadata
- **chunks** - Chunked content (1500 words, 200 overlap)
- **document_tables** - Extracted PDF tables
- **document_code_blocks** - Detected code blocks
- **document_entities** - Extracted entities
- **entity_relationships** - Co-occurrence tracking
- Plus: summaries, extraction_jobs, monitoring_history, etc.

**Benefits:**
- Lazy loading (metadata at startup, chunks on-demand)
- ACID transactions
- Scalable to 100k+ documents
- FTS5 full-text indexes

Default location: `~/.tdz-c64-knowledge` or `TDZ_DATA_DIR`

## Usage Examples

Ask Claude Code:
- "Search the C64 docs for SID voice registers"
- "What does the memory map say about $D400?"
- "Find information about sprite multiplexing"
- "Add C:/docs/mapping_the_c64.pdf with tags memory-map, reference"
- "How do I program raster interrupts on the VIC-II?" (uses RAG)

## Suggested Tags

Organize docs with consistent tags:
- `reference`, `memory-map`, `basic`, `assembly`
- `sid`, `vic-ii`, `cia`, `kernal`
- `hardware`, `disk`, `graphics`, `sound`

## Troubleshooting

**"pypdf not installed"** - Run: `pip install pypdf rank-bm25`

**"mcp module not found"** - Run: `pip install mcp`

**Server not responding** - Use Python from virtual environment, not system Python

**PDF extraction issues** - Use OCR or add plain text version

**BM25 issues** - Check logs in `TDZ_DATA_DIR/server.log`, try `USE_BM25=0`

## Development

### Testing

```cmd
pip install -e ".[dev]"

# Run all tests
pytest test_server.py test_wiki_export.py -v

# With coverage
pytest test_server.py -v --cov=server --cov-report=term

# Wiki export tests only
pytest test_wiki_export.py -v
```

**Test Coverage:**
- `test_server.py` - Core server functionality (search, entities, RAG, etc.)
- `test_wiki_export.py` - Wiki generation features (16 tests):
  - Document coordinate export (UMAP/t-SNE)
  - File type detection (HTML/MD)
  - Cluster document export
  - HTML generation with explanation boxes
  - JavaScript generation for interactive features

### CI/CD

GitHub Actions workflow tests on Python 3.10/3.11/3.12 across Windows/Linux/macOS with Ruff code quality checks.

## Documentation

### Core Documentation

- **[README.md](README.md)** (this file) - Installation, features, tools, usage
- **[QUICKSTART.md](QUICKSTART.md)** - Fast setup guide (5 minutes)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical deep dive, database schema, algorithms
- **[CONTEXT.md](CONTEXT.md)** - Project status, quick stats, version history
- **[CLAUDE.md](CLAUDE.md)** - Quick reference for Claude Code integration
- **[CHANGELOG.md](CHANGELOG.md)** - Complete version history

### Feature Documentation

Browse [docs/](docs/) for detailed guides on specific features:

**API & Integration:**
- [REST API](docs/REST_API.md) - FastAPI REST server (27 endpoints)

**AI-Powered Features:**
- [Entity Extraction](docs/ENTITY_EXTRACTION.md) - Extract hardware, memory addresses, instructions
- [Anomaly Detection](docs/ANOMALY_DETECTION.md) - ML-based URL content monitoring
- [Summarization](docs/SUMMARIZATION.md) - AI-powered document summarization

**Data Sources:**
- [Web Scraping](docs/WEB_SCRAPING.md) - Scrape documentation websites
- [Web Monitoring](docs/WEB_MONITORING.md) - Track URL-sourced content changes

**Setup & Deployment:**
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [Docker Setup](docs/DOCKER.md) - Docker configuration
- [Environment Setup](docs/ENVIRONMENT_SETUP.md) - Environment variables
- [Poppler Setup](docs/POPPLER_SETUP.md) - Poppler installation for PDFs

**User Interfaces:**
- [GUI Guide](docs/GUI.md) - Streamlit web interface

**Development:**
- [Testing Guide](docs/TESTING.md) - Test suite and CI/CD
- [Examples](docs/EXAMPLES.md) - Usage examples and performance analysis
- [Monitoring Setup](docs/MONITORING.md) - Scheduled monitoring configuration
- [Roadmap](docs/ROADMAP.md) - Future improvements and features

## Version History

**v2.23.0** - RAG Question Answering & Advanced Search (Phase 2 Complete)
- RAG-based answer_question with citations
- Fuzzy search with rapidfuzz
- Progressive search refinement
- Smart tagging system

**v2.22.0** - Search Improvements (Phase 1 Complete)
- Enhanced entity analytics
- C64-specific regex patterns (5000x faster)
- Performance optimizations

**v2.21.0** - Anomaly Detection
- ML-based baseline learning
- 1500x performance improvement

**v2.18.0** - REST API & Background Processing
- FastAPI REST server (27 endpoints)
- Background entity extraction

**v2.15.0+** - Entity Intelligence
- Entity extraction, relationships, analytics

See CONTEXT.md for complete version history.

## License

MIT License - Use freely for your retro computing projects!
