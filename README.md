# TDZ C64 Knowledge

[![Version](https://img.shields.io/badge/version-2.20.0-brightgreen.svg)](https://github.com/MichaelTroelsen/tdz-c64-knowledge)
[![CI/CD Pipeline](https://github.com/MichaelTroelsen/tdz-c64-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelTroelsen/tdz-c64-knowledge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. Add PDFs, text files, Markdown, HTML, and Excel files to build a searchable knowledge base that Claude Code or other MCP clients can query.

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

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Features

### Search & Retrieval
- **SQLite FTS5 full-text search** - Native database search with 480x faster queries (50ms vs 24s)
- **Semantic search with embeddings** - Find documents by meaning, not just keywords (e.g., "movable objects" finds "sprites")
- **Find similar documents** - Discover related documentation automatically using semantic or TF-IDF similarity
- **BM25 search algorithm** - Industry-standard ranking for accurate results (fallback)
- **Query preprocessing** - Intelligent stemming and stopword removal with NLTK
- **Phrase search** - Use quotes for exact phrase matching (e.g., `"VIC-II chip"`)
- **Search term highlighting** - Matching terms highlighted in search results
- **Tag-based filtering** - Organize docs by topic (memory-map, sid, vic-ii, basic, assembly, etc.)

### Document Management
- **Multiple file formats** - Ingest PDFs, text files (.txt), Markdown (.md), HTML (.html, .htm), and Excel files (.xlsx, .xls)
- **Duplicate detection** - Content-based deduplication prevents indexing the same document twice
- **Chunked retrieval** - Get specific sections without loading entire documents
- **PDF metadata extraction** - Author, subject, creator, and creation date
- **Page number tracking** - Results show PDF page numbers for easy reference
- **Persistent index** - Documents stay indexed between sessions

### Security & Reliability
- **Path traversal protection** - Whitelist allowed directories for document ingestion
- **Comprehensive logging** - File and console logging for debugging
- **ACID transactions** - SQLite ensures data integrity

### AI-Powered Features
- **Natural Language Query Translation** - Convert natural language queries to structured search parameters
  - **AI-powered parsing** - Understands "find sprite info on VIC-II" and translates to structured query
  - **Entity extraction** - Identifies hardware, concepts, and technical terms in user queries
  - **Search mode recommendation** - Suggests keyword/semantic/hybrid based on query
  - **Facet filter generation** - Automatically creates filters from detected entities
  - **Confidence scoring** - 0.0-1.0 confidence in translation accuracy
  - **Graceful fallback** - Works without LLM using regex patterns
- **Named Entity Extraction** - Extract and catalog technical entities from documentation using LLM
  - **7 entity types**: hardware (SID, VIC-II), memory addresses ($D000), instructions (LDA, STA), people, companies, products, concepts
  - **Confidence scoring** - Each entity has 0.0-1.0 confidence score
  - **Occurrence counting** - Track how many times each entity appears
  - **Context snippets** - See surrounding text for each entity mention
  - **Full-text search** - Search across all entities with FTS5
  - **Statistics dashboard** - View top entities and documents
  - **Bulk processing** - Extract entities from entire knowledge base
- **Entity Relationship Tracking** - Discover connections between technical entities
  - **Co-occurrence analysis** - Tracks which entities appear together in documents
  - **Relationship strength** - 0.0-1.0 scoring based on frequency and context
  - **Interactive network visualization** - Drag-and-drop graph with pyvis
    - **Color-coded nodes** - 7 colors for entity types (hardware: red, memory_address: teal, instruction: blue, person: coral, company: mint, product: yellow, concept: purple)
    - **Edge thickness** - Width scaled by relationship strength
    - **Edge transparency** - Alpha scaled by strength (stronger = more opaque)
    - **Physics simulation** - Barnes-Hut algorithm for natural node arrangement
    - **Hover tooltips** - Entity type, relationship strength, shared document count
    - **Adjustable controls** - Max nodes (10-100), min strength threshold (0.0-1.0)
    - **Responsive design** - 600px height, dark theme optimized
    - **Performance** - Handles 100+ nodes smoothly with drag interaction
  - **Context extraction** - Shows example sentences where entities co-occur
  - **Bulk extraction** - Process relationships across entire knowledge base
  - **Search by entity pair** - Find documents discussing specific entity combinations
- **Document Summarization** - Generate AI summaries (brief, detailed, bullet-point)
- **Smart Auto-Tagging** - AI-powered tag suggestions with confidence scoring

### REST API Server
- **FastAPI-based REST interface** - HTTP/REST API with 27 endpoints
- **OpenAPI/Swagger documentation** - Interactive API docs at `/api/docs`
- **6 endpoint categories**:
  - Health & Stats (2 endpoints) - System monitoring
  - Search (5 endpoints) - All search modes (keyword, semantic, hybrid, faceted)
  - Documents (7 endpoints) - Full CRUD operations with bulk support
  - URL Scraping (3 endpoints) - Web content ingestion
  - AI Features (5 endpoints) - Summarization, entity extraction, relationships
  - Analytics & Export (5 endpoints) - CSV/JSON export capabilities
- **API key authentication** - Secure access via X-API-Key header
- **CORS support** - Cross-origin resource sharing configuration
- **Pydantic v2 validation** - Request/response validation
- **Production ready** - Can run alongside MCP server
- See [README_REST_API.md](README_REST_API.md) for complete API documentation

## Installation (Windows)

### Prerequisites

1. **Python 3.10+** - Download from https://python.org
   - During install, check "Add Python to PATH"

2. **uv** (recommended) or pip
   ```cmd
   pip install uv
   ```

### Setup

1. **Clone or download this folder** to a location like:
   ```
   C:\Users\YourName\mcp-servers\tdz-c64-knowledge
   ```

2. **Create virtual environment and install dependencies:**
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
   ```

3. **Test the server:**
   ```cmd
   python server.py
   ```
   (Press Ctrl+C to stop - it will just wait for input since it's an MCP server)

## Configuration

### For Claude Code

Add to your Claude Code MCP settings (usually in `.claude/settings.json` or via `claude mcp add`):

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data"
      }
    }
  }
}
```

Or add via command line:
```cmd
claude mcp add tdz-c64-knowledge -- C:\Users\YourName\mcp-servers\tdz-c64-knowledge\.venv\Scripts\python.exe C:\Users\YourName\mcp-servers\tdz-c64-knowledge\server.py
```

### For Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data"
      }
    }
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TDZ_DATA_DIR` | Directory to store index and chunks | `~/.tdz-c64-knowledge` |
| `USE_FTS5` | Enable SQLite FTS5 full-text search (0=disabled, 1=enabled) | `0` (disabled) |
| `USE_SEMANTIC_SEARCH` | Enable semantic search with embeddings (0=disabled, 1=enabled) | `0` (disabled) |
| `SEMANTIC_MODEL` | Sentence-transformers model to use | `all-MiniLM-L6-v2` |
| `USE_BM25` | Enable BM25 search algorithm (0=disabled, 1=enabled) | `1` (enabled) |
| `USE_QUERY_PREPROCESSING` | Enable query preprocessing with NLTK (0=disabled, 1=enabled) | `1` (enabled) |
| `USE_FUZZY_SEARCH` | Enable fuzzy search for typo tolerance (0=disabled, 1=enabled) | `1` (enabled) |
| `FUZZY_THRESHOLD` | Fuzzy search similarity threshold (0-100) | `80` (80% similarity) |
| `USE_OCR` | Enable OCR for scanned PDFs (0=disabled, 1=enabled) | `1` (enabled if Tesseract installed) |
| `SEARCH_CACHE_SIZE` | Maximum number of cached search results | `100` |
| `SEARCH_CACHE_TTL` | Cache time-to-live in seconds | `300` (5 minutes) |
| `ALLOWED_DOCS_DIRS` | Comma-separated whitelist of allowed document directories (optional) | None (no restrictions) |

## Search Features

### SQLite FTS5 Full-Text Search (Recommended)

For maximum performance, enable **SQLite FTS5** full-text search by setting `USE_FTS5=1` in your environment:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "env": {
        "USE_FTS5": "1"
      }
    }
  }
}
```

**Performance:**
- **480x faster** than BM25 (50ms vs 24,000ms for typical queries)
- No need to load all chunks into memory
- Native SQLite BM25 ranking
- Porter stemming tokenizer for better matching

**How it works:**
- Uses SQLite's FTS5 virtual table with Porter stemming
- Automatic triggers keep FTS5 index in sync with chunks table
- Falls back to BM25/simple search if FTS5 returns no results

### Semantic Search with Embeddings (Recommended for Natural Language)

For conceptual/meaning-based search, enable **semantic search** by setting `USE_SEMANTIC_SEARCH=1`:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "env": {
        "USE_SEMANTIC_SEARCH": "1",
        "SEMANTIC_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

**How it works:**
- Uses sentence-transformers to generate embeddings for all chunks
- Stores embeddings in FAISS vector index for fast similarity search
- Finds documents based on semantic meaning, not just keywords
- Example: searching "movable objects" will find documents about "sprites"

**Performance:**
- First search: ~7-16ms per query (if embeddings pre-built)
- First search without embeddings: ~1-2 minutes (builds embeddings index for all chunks)
- Subsequent searches: ~7-16ms per query
- Embeddings are persisted to disk (embeddings.faiss, embeddings_map.json)
- **Tip:** Pre-build embeddings with `python enable_semantic_search.py` before first use

**Requirements:**
- Install with: `pip install sentence-transformers faiss-cpu`
- Or: `pip install -e ".[semantic]"`

### BM25 Ranking Algorithm (Fallback)

When FTS5 and semantic search are disabled, the server uses the **BM25 (Okapi BM25)** algorithm for search ranking. BM25 is an industry-standard probabilistic ranking function that provides much better relevance scoring than simple term frequency.

**Benefits:**
- More accurate ranking of search results
- Better handling of document length variations
- Improved scoring for multi-term queries

**How it works:**
- Tokenizes documents and queries into words
- Scores based on term frequency, document frequency, and document length
- Handles small documents (may produce negative scores, which are handled correctly)

To disable BM25 and use simple term frequency search, set `USE_BM25=0` in your environment.

### Phrase Search

Use double quotes to search for exact phrases:

```
search_docs(query='"VIC-II chip" registers')
```

This will:
- Match documents containing the exact phrase "VIC-II chip"
- Also search for the term "registers"
- Boost scores for documents with exact phrase matches

### Search Term Highlighting

Search results automatically highlight matching terms in snippets using markdown bold (`**term**`). This makes it easy to see why a result matched your query.

### Page Number Tracking

For PDF documents, search results include the estimated page number where the content appears. This helps you quickly find the information in the original document.

### Query Preprocessing

The server uses NLTK for intelligent query preprocessing to improve search accuracy:

**Features:**
- **Stemming** - Matches word variations (searching "running" finds "run", "runs", "runner")
- **Stopword removal** - Filters out common words ("the", "a", "is") that don't add meaning
- **Technical term preservation** - Keeps hyphenated terms like "VIC-II" and numbers like "6502"
- **Smart tokenization** - Properly handles punctuation and special characters

**Benefits:**
- Find more relevant results with natural language queries
- Matches conceptually similar terms (plural/singular, verb tenses)
- Reduces noise from common words
- Preserves important technical terminology

**Configuration:**
- Enabled by default when NLTK is installed
- Disable with environment variable: `USE_QUERY_PREPROCESSING=0`
- Works with both BM25 and simple search algorithms

**Example:**
- Query: "How does the SID chip generate sounds?"
- Preprocessed: ["sid", "chip", "generat", "sound"]
- Matches: "generate", "generating", "generated", "sounds", "sound"

### Fuzzy Search / Typo Tolerance

The server uses rapidfuzz for fuzzy string matching to handle typos and similar terms:

**Features:**
- **Levenshtein distance** - Finds similar words based on edit distance
- **Configurable threshold** - Adjust sensitivity (default: 80% similarity)
- **Exact match priority** - Exact matches always score higher than fuzzy matches
- **Smart fallback** - Only uses fuzzy matching when exact matches not found

**Configuration:**
```json
{
  "env": {
    "USE_FUZZY_SEARCH": "1",
    "FUZZY_THRESHOLD": "80"
  }
}
```

**Benefits:**
- Handles typos automatically ("VIC-I" finds "VIC-II")
- Finds similar terms ("grafics" finds "graphics")
- Improves search recall without sacrificing precision
- Configurable tolerance for your use case

**Examples:**
- Query: "registr" → Finds: "register" (88% similarity)
- Query: "VIC-I" → Finds: "VIC-II" (83% similarity)
- Query: "grafics" → Finds: "graphics" (88% similarity)

**Installation:**
```bash
pip install rapidfuzz
```

### OCR Support for Scanned PDFs

The server automatically detects and processes scanned PDFs using Tesseract OCR:

**Features:**
- **Automatic detection** - Detects PDFs with little/no extractable text
- **Seamless fallback** - Automatically uses OCR when needed
- **Per-page processing** - Handles multi-page scanned documents
- **Graceful degradation** - Falls back to extracted text if OCR fails

**How it works:**
1. Attempts normal text extraction from PDF
2. If < 100 characters extracted, assumes scanned document
3. Automatically converts PDF pages to images
4. Runs Tesseract OCR on each page
5. Indexes the OCR-extracted text

**Configuration:**
```json
{
  "env": {
    "USE_OCR": "1"
  }
}
```

**System Requirements:**
- **Python libraries**: `pytesseract`, `pdf2image`, `Pillow`
- **System binary**: Tesseract-OCR must be installed

**Installation:**
```bash
# Python libraries
pip install pytesseract pdf2image Pillow

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**Benefits:**
- Handles scanned manuals and documentation automatically
- No manual intervention required
- Preserves PDF metadata
- Works with any quality scanned documents

**Performance:**
- Text-based PDFs: Instant extraction
- Scanned PDFs: ~1-2 seconds per page

### Search Result Caching

The knowledge base automatically caches search results for improved performance:

**How it works:**
- Search results are cached with a Time-To-Live (TTL) of 5 minutes by default
- Repeated queries return cached results instantly (50-100x speedup)
- Cache automatically invalidates when documents are added/removed
- Separate caches for `search()` and `find_similar()` operations

**Configuration:**
```json
{
  "env": {
    "SEARCH_CACHE_SIZE": "100",
    "SEARCH_CACHE_TTL": "300"
  }
}
```

**Benefits:**
- Dramatically faster repeated queries in interactive sessions
- Minimal memory overhead
- Thread-safe implementation
- Zero configuration required

### Document Update Detection

The knowledge base tracks file modification times and content hashes to detect when indexed documents have changed:

**How it works:**
1. **Quick check**: Compares file modification time with stored mtime
2. **Deep check**: If mtime changed, computes MD5 hash of file content
3. **Smart update**: Only re-indexes if content hash differs

**Check for updates:**
```
check_updates(auto_update=false)
```

Returns status of all documents (unchanged/changed/missing) and optionally re-indexes changed files automatically.

**Benefits:**
- Avoid re-indexing unchanged files
- Detect real content changes (not just mtime changes)
- Batch checking with detailed status reporting
- Optional auto-update for hands-free workflow

## Tools

The server exposes these tools to MCP clients:

### search_docs
Search the knowledge base for information.

```
search_docs(query="SID register", max_results=5, tags=["sid"])
```

### add_document
Add a PDF or text file to the knowledge base.

```
add_document(
  filepath="C:/c64docs/Programmer_Reference.pdf",
  title="C64 Programmer's Reference Guide",
  tags=["reference", "memory-map", "basic", "assembly"]
)
```

### list_docs
List all indexed documents.

### get_chunk
Get the full content of a specific search result chunk.

```
get_chunk(doc_id="abc123", chunk_id=5)
```

### get_document
Get the full content of a document.

### remove_document
Remove a document from the knowledge base.

### kb_stats
Get statistics about the knowledge base.

### find_similar
Find documents similar to a given document or chunk.

```
find_similar(
  doc_id="abc123",
  chunk_id=5,  # Optional - for chunk-level similarity
  max_results=5,
  tags=["graphics"]  # Optional - filter by tags
)
```

Uses semantic embeddings (FAISS) if available, otherwise falls back to TF-IDF cosine similarity. Returns documents sorted by similarity score with snippets.

### semantic_search
Search using semantic/conceptual similarity (requires USE_SEMANTIC_SEARCH=1).

```
semantic_search(
  query="How do sprites work?",
  max_results=5,
  tags=["graphics"]
)
```

Finds documents based on meaning, not just keywords. Example: searching for "movable objects" can find documents about "sprites".

### hybrid_search
Perform hybrid search combining FTS5 keyword search and semantic search.

```
hybrid_search(
  query="SID chip sound programming",
  max_results=10,
  semantic_weight=0.7,  # 0.0-1.0, higher favors semantic
  tags=["audio"]
)
```

Best of both worlds - finds exact keyword matches AND conceptually related content. Results ranked by weighted combination of both scores.

### faceted_search
Search with entity-based facet filtering.

```
faceted_search(
  query="graphics programming",
  max_results=10,
  facet_filters={
    "hardware": ["VIC-II"],
    "concept": ["sprite", "raster"]
  }
)
```

Filters results by detected entities. Requires entity extraction to be run on documents first.

### search_tables
Search specifically within extracted PDF tables.

```
search_tables(
  query="memory map",
  max_results=5,
  tags=["reference"]
)
```

Searches only table content extracted from PDFs, useful for finding structured data like register maps, memory layouts, and opcode tables.

### search_code
Search specifically within detected code blocks.

```
search_code(
  query="LDA STA",
  max_results=5,
  code_type="assembly"  # or "basic", "hex"
)
```

Searches only code blocks (BASIC, Assembly, Hex) extracted from documents. Useful for finding code examples and programming patterns.

### find_by_reference
Find documents by cross-reference or citation.

```
find_by_reference(
  reference="page 142",
  max_results=5
)
```

Searches for document cross-references, page numbers, and citations within the knowledge base.

### suggest_queries
Get query suggestions based on knowledge base content.

```
suggest_queries(
  partial_query="VIC",
  max_suggestions=5
)
```

Returns suggested search queries based on indexed content, helpful for discovery and exploration.

### check_updates
Check all indexed documents for updates and optionally re-index changed files.

```
check_updates(auto_update=false)
```

Returns a report showing:
- ✓ Unchanged documents
- ⚠ Changed documents (content modified)
- ✗ Missing documents (files not found)
- ✓ Updated documents (if auto_update=true)

**Example output:**
```
Document Update Check:

✓ 15 documents unchanged

⚠ 2 documents changed:
  - VIC-II Guide (C:/docs/vic-ii-guide.pdf)
  - SID Manual (C:/docs/sid-manual.pdf)

✗ 1 documents missing (files not found):
  - Old Reference (C:/docs/deleted.pdf)

Run with auto_update=true to automatically re-index changed documents.
```

### add_documents_bulk
Add multiple documents from a directory at once using glob patterns.

```
add_documents_bulk(
  directory="C:/c64docs",
  pattern="**/*.{pdf,txt}",  # Default: **/*.{pdf,txt}
  tags=["reference", "c64"],
  recursive=true,  # Search subdirectories
  skip_duplicates=true  # Skip files with duplicate content
)
```

Returns a report showing:
- ✓ Added documents (with doc IDs and chunk counts)
- ⊘ Skipped documents (duplicates)
- ✗ Failed documents (with error messages)

**Example output:**
```
Bulk Document Add Results:

✓ 15 documents added:
  - mapping_the_c64.txt (mapping_the_c64)
    ID: a1b2c3d4, Chunks: 8
  - programmers_reference.pdf (programmers_reference)
    ID: e5f6g7h8, Chunks: 45
  ...

⊘ 2 documents skipped (duplicates):
  - C:/c64docs/backup/mapping_the_c64_copy.txt

✗ 1 documents failed:
  - C:/c64docs/corrupted.pdf: Error extracting document

Total: 15 added, 2 skipped, 1 failed
```

**Benefits:**
- Efficiently import large document collections
- Automatic duplicate detection during bulk operations
- Graceful error handling (failures don't stop the operation)
- Supports glob patterns for flexible file matching

### remove_documents_bulk
Remove multiple documents by IDs or tags.

```
# Remove by document IDs
remove_documents_bulk(doc_ids=["abc123", "def456", "ghi789"])

# Remove by tags
remove_documents_bulk(tags=["outdated", "test"])

# Remove by both
remove_documents_bulk(doc_ids=["abc123"], tags=["test"])
```

Returns a report showing:
- ✓ Removed documents (by doc ID)
- ✗ Failed removals (with error messages)

**Example output:**
```
Bulk Document Remove Results:

✓ 3 documents removed:
  - abc123
  - def456
  - ghi789

Total: 3 removed, 0 failed
```

**Benefits:**
- Clean up multiple documents at once
- Tag-based removal for flexible document management
- Detailed failure reporting

### URL Scraping Tools

#### scrape_url
Scrape a documentation website and add all pages to the knowledge base.

```
scrape_url(
  url="https://www.c64-wiki.com/wiki/VIC",
  tags=["c64-wiki", "reference"],
  depth=2,          # Maximum link depth
  limit="https://www.c64-wiki.com",  # Only scrape URLs with this prefix
  threads=5,        # Concurrent download threads
  delay=100        # Delay between requests (ms)
)
```

Uses mdscrape to convert HTML to markdown. Automatically follows links and indexes all discovered pages.

#### rescrape_document
Re-scrape a URL-sourced document to check for updates.

```
rescrape_document(
  doc_id="abc123",  # Must be a URL-sourced document
  force=false       # Force rescrape even if not modified
)
```

Removes the old version and re-scrapes the original URL with the same configuration. Checks Last-Modified headers to avoid unnecessary downloads.

#### check_url_updates
Check all URL-sourced documents for updates with comprehensive structure discovery.

```
check_url_updates(
  auto_rescrape=false,     # Automatically re-scrape changed URLs
  check_structure=true     # Discover new/missing sub-pages (slower)
)
```

**Quick Mode** (`check_structure=false`): Fast check using Last-Modified headers only
- Checks all scraped pages for content updates
- ~1 second per site
- Returns: unchanged, changed, failed

**Full Mode** (`check_structure=true`): Comprehensive structure discovery
- Crawls websites to discover all current pages
- Detects new pages not in database
- Identifies missing/removed pages
- ~10-60 seconds per site (max 100 pages, depth 5)
- Returns: unchanged, changed, new_pages, missing_pages, scrape_sessions, failed

**Features:**
- Groups documents by scrape session (base URL)
- Concurrent crawling with BeautifulSoup
- Respects original scrape configuration (depth, limit, same_domain)
- Automatic 404 detection for removed pages
- Per-session statistics and progress logging

**Example Response:**
```json
{
  "unchanged": 34,
  "changed": 0,
  "new_pages": [
    {"url": "https://example.com/new-page.html", "base_url": "https://example.com"}
  ],
  "missing_pages": [
    {"doc_id": "abc123", "url": "https://example.com/deleted.html", "reason": "404"}
  ],
  "scrape_sessions": [
    {"base_url": "https://example.com", "docs_count": 10, "new": 1, "missing": 1}
  ]
}
```

### Document Analysis Tools

#### compare_documents
Compare two documents side-by-side with similarity scoring.

```
compare_documents(
  doc_id_1="abc123",
  doc_id_2="def456",
  comparison_type="full"  # or "metadata", "content", "entities"
)
```

**Returns:**
- Similarity score (0.0-1.0) using cosine similarity
- Metadata differences (title, tags, etc.)
- Content diff (unified diff format)
- Entity comparison (common, unique to each)
- Chunk count comparison

Useful for finding duplicate or related documents, tracking document changes, and analyzing content overlap.

#### health_check
Perform health check on the knowledge base system.

```
health_check()
```

**Returns:**
- System status (healthy/degraded/unhealthy)
- Database connectivity and integrity
- Feature availability (FTS5, semantic search, OCR)
- Performance metrics (query times, cache hit rates)
- Any detected issues or warnings

Use for monitoring and diagnostics.

### System Management Tools

#### update_tags_bulk
Update tags for multiple documents at once.

```
update_tags_bulk(
  doc_ids=["abc123", "def456"],
  add_tags=["updated", "v2"],
  remove_tags=["draft", "old"]
)
```

Efficiently update tags across many documents. Can add tags, remove tags, or both in a single operation.

#### create_backup
Create a backup of the knowledge base.

```
create_backup(
  backup_path="C:/backups/kb-2025-12-21.db",
  include_embeddings=true  # Include FAISS indexes
)
```

Creates a complete backup including database, embeddings, and configuration. Returns backup file path and size.

#### restore_backup
Restore knowledge base from a backup file.

```
restore_backup(
  backup_path="C:/backups/kb-2025-12-21.db",
  confirm=true  # Required safety check
)
```

**Warning:** This will replace the current knowledge base. All existing data will be lost.

#### search_analytics
Get search analytics and usage statistics.

```
search_analytics(
  time_range_days=30,  # Analysis period
  group_by="query"     # or "tag", "date"
)
```

**Returns:**
- Top search queries
- Search volume trends
- Popular tags
- Average results per query
- Query performance metrics

Useful for understanding usage patterns and optimizing content.

### Export Tools

#### export_results
Export search results to CSV or JSON.

```
export_results(
  query="VIC-II",
  format="csv",  # or "json"
  output_path="search_results.csv",
  max_results=100
)
```

Exports search results with all metadata, snippets, and scores for external analysis.

#### export_documents_bulk
Export document metadata for all documents.

```
export_documents_bulk(
  format="csv",  # or "json"
  output_path="documents.csv",
  tags=["reference"],  # Optional filter
  include_stats=true   # Include chunk counts, entity counts
)
```

Exports comprehensive document metadata including titles, tags, file info, statistics, and timestamps.

#### export_entities
Export all extracted entities to CSV or JSON.

```
export_entities(
  format="csv",  # or "json"
  output_path="entities.csv",
  entity_type="hardware",  # Optional filter by type
  min_confidence=0.7       # Optional confidence threshold
)
```

**CSV columns:**
- entity_text, entity_type, confidence, document_count, total_occurrences

**JSON format:**
- Full entity objects with contexts, document IDs, and metadata

Useful for analyzing entity distribution, building taxonomies, and external processing.

#### export_relationships
Export all entity relationships to CSV or JSON.

```
export_relationships(
  format="csv",  # or "json"
  output_path="relationships.csv",
  min_strength=0.5,  # Optional strength threshold
  entity_type=null   # Optional filter by entity type
)
```

**CSV columns:**
- entity1, entity1_type, entity2, entity2_type, strength, co_occurrence_count, document_count

**JSON format:**
- Full relationship objects with contexts, documents, and statistics

Perfect for network analysis, knowledge graphs, and relationship visualization.

### AI-Powered Tools

#### extract_entities
Extract named entities from a document using AI (requires LLM configuration).

```
extract_entities(
  doc_id="abc123",
  confidence_threshold=0.6,  # 0.0-1.0, default: 0.6
  force_regenerate=false     # Re-extract even if cached
)
```

**Entity types extracted:**
- `hardware` - Chip names (SID, VIC-II, CIA, 6502, 6526, 6581)
- `memory_address` - Memory addresses ($D000, $D020, $0400)
- `instruction` - Assembly instructions (LDA, STA, JMP, JSR, RTS)
- `person` - People mentioned (Bob Yannes, Jack Tramiel)
- `company` - Organizations (Commodore, MOS Technology)
- `product` - Hardware products (VIC-20, C128, 1541)
- `concept` - Technical concepts (sprite, raster interrupt, IRQ)

**Returns:** Entities grouped by type with confidence scores, occurrence counts, and context snippets.

**Example output:**
```
✓ Entity extraction complete!

Document: C64 Programmer's Reference
Entities Found: 42

HARDWARE (10):
  - VIC-II (confidence: 0.95, 5x)
    *The VIC-II chip at $D000 controls video output...*
  - SID (confidence: 0.92, 3x)
  - CIA (confidence: 0.88, 2x)
  ...

MEMORY ADDRESS (8):
  - $D000 (confidence: 0.98, 12x)
  - $D020 (confidence: 0.96, 4x)
  ...
```

#### list_entities
List all entities from a document with optional filtering.

```
list_entities(
  doc_id="abc123",
  entity_types=["hardware", "memory_address"],  # Optional filter
  min_confidence=0.7  # Optional threshold
)
```

Returns all extracted entities matching the filters.

#### search_entities
Search for entities across all documents using full-text search.

```
search_entities(
  query="VIC-II",
  entity_types=["hardware"],  # Optional filter
  min_confidence=0.7,
  max_results=20
)
```

**Returns:** Documents containing matching entities with contexts and match counts.

**Example output:**
```
Entity Search Results for: VIC-II
Total Matches: 15
Documents Found: 8

1. C64 Programmer's Reference (programmers_ref)
   Matches: 5
   - VIC-II (hardware, conf: 0.95, 5x)
     *The VIC-II chip at $D000 controls...*
   - VIC-II registers (concept, conf: 0.88, 3x)
   ...

2. VIC-II Technical Guide (vic_ii_guide)
   Matches: 3
   ...
```

#### entity_stats
Show entity extraction statistics for the knowledge base.

```
entity_stats(
  entity_type="hardware"  # Optional filter by type
)
```

**Returns:**
- Total entities and documents with entities
- Breakdown by entity type
- Top 20 entities by document count
- Top 10 documents by entity count

**Example output:**
```
Entity Statistics

Total Entities: 1,247
Documents with Entities: 42

Entities by Type:
  - hardware: 325
  - memory address: 289
  - instruction: 198
  - concept: 167
  - product: 142
  - person: 78
  - company: 48

Top 10 Entities (by document count):
1. VIC-II (hardware)
   - Found in 28 document(s)
   - Total occurrences: 145
   - Avg confidence: 0.94

2. $D000 (memory_address)
   - Found in 24 document(s)
   - Total occurrences: 98
   - Avg confidence: 0.97
...
```

#### extract_entities_bulk
Bulk extract entities from multiple documents.

```
extract_entities_bulk(
  confidence_threshold=0.6,
  force_regenerate=false,
  max_docs=null,  # Process all documents
  skip_existing=true
)
```

**Returns:** Processing statistics including processed/skipped/failed counts and entity breakdowns.

**Example output:**
```
Bulk Entity Extraction Complete

Processed: 42 documents
Skipped: 15 documents (already have entities)
Failed: 0 documents
Total Entities Extracted: 1,247

Entities by Type:
  - hardware: 325
  - memory address: 289
  - instruction: 198
  - concept: 167
  - product: 142
  - person: 78
  - company: 48

Sample Results (first 10):
1. ✓ C64 Programmer's Reference - 58 entities
2. ✓ VIC-II Guide - 34 entities
3. ⊘ Memory Map - skipped (28 entities)
...
```

**Note:** Entity extraction requires LLM configuration. Set `LLM_PROVIDER` and appropriate API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).

#### summarize_document
Generate an AI summary of a document.

```
summarize_document(
  doc_id="abc123",
  summary_type="brief",  # or "detailed", "bullet"
  max_length=300,
  style="technical"      # or "simple", "detailed"
)
```

**Summary types:**
- `brief` - 2-3 sentence overview
- `detailed` - Comprehensive multi-paragraph summary
- `bullet` - Key points as bullet list

**Styles:**
- `technical` - Preserves technical terminology
- `simple` - Accessible language
- `detailed` - In-depth analysis

Summaries are cached in the database for fast retrieval.

#### summarize_all
Bulk summarize all documents in the knowledge base.

```
summarize_all(
  summary_type="brief",
  force_regenerate=false,  # Re-summarize even if cached
  max_docs=null,           # Process all documents
  skip_existing=true
)
```

**Returns:** Processing statistics with summary counts and status for each document.

**Example output:**
```
Bulk Summarization Complete

Processed: 42 documents
Skipped: 15 documents (already have summaries)
Failed: 2 documents (LLM errors)

Sample Summaries:
1. ✓ C64 Programmer's Reference - Brief summary generated
2. ✓ VIC-II Guide - Brief summary generated
3. ⊘ Memory Map - skipped (has summary)
...
```

#### get_summary
Retrieve cached summary for a document.

```
get_summary(
  doc_id="abc123",
  summary_type="brief"
)
```

Returns previously generated summary if available, or generates a new one if needed.

#### auto_tag_document
Generate AI-powered tag suggestions for a document.

```
auto_tag_document(
  doc_id="abc123",
  max_tags=10,
  confidence_threshold=0.7,  # Minimum confidence for suggestions
  apply_tags=false           # Auto-apply high-confidence tags
)
```

**Returns:** Suggested tags with confidence scores (0.0-1.0).

**Example output:**
```
Tag Suggestions for: C64 Programmer's Reference

High Confidence (>0.8):
  - reference (0.95)
  - programming (0.92)
  - memory-map (0.88)

Medium Confidence (0.7-0.8):
  - basic (0.75)
  - assembly (0.72)

Low Confidence (<0.7):
  - graphics (0.65)
```

#### auto_tag_all
Bulk auto-tag all documents in the knowledge base.

```
auto_tag_all(
  confidence_threshold=0.8,  # Only apply tags with high confidence
  apply_tags=true,           # Automatically apply tags
  max_tags_per_doc=5
)
```

**Returns:** Processing statistics with tag counts and application results.

Useful for automatically organizing and categorizing large document collections.

#### translate_query
Translate natural language queries to structured search parameters using AI.

```
translate_query(
  query="find information about sprites on the VIC-II chip",
  confidence_threshold=0.7  # Minimum confidence for entity extraction
)
```

**Returns:** Structured search parameters including:
- Extracted search terms
- Detected entities by type
- Recommended search mode (keyword/semantic/hybrid)
- Auto-generated facet filters
- Confidence score

**Example output:**
```
Natural Language Query Translation

Original Query: "find information about sprites on the VIC-II chip"

Search Parameters:
  Terms: sprite, information
  Mode: hybrid (recommended)
  Confidence: 0.85

Detected Entities:
  - VIC-II (hardware, conf: 0.95)
  - sprite (concept, conf: 0.88)

Suggested Facet Filters:
  hardware: [VIC-II]
  concept: [sprite]

Ready to execute search with these parameters.
```

#### extract_entity_relationships
Extract entity relationships (co-occurrences) from a document.

```
extract_entity_relationships(
  doc_id="abc123",
  min_strength=0.3  # Minimum relationship strength to include
)
```

**Returns:** Entity pairs found in the document with strength scores and contexts.

**Example output:**
```
✓ Relationship extraction complete!

Document: C64 Programmer's Reference
Relationships Found: 28

Top 10 Relationships:
1. VIC-II ↔ sprite (strength: 0.92, 8 co-occurrences)
   *The VIC-II chip controls sprite graphics with hardware registers...*

2. $D000 ↔ VIC-II (strength: 0.88, 6 co-occurrences)
   *Register $D000 is the base address for VIC-II sprite control...*
   ...
```

#### get_entity_relationships
Get all relationships for a specific entity across all documents.

```
get_entity_relationships(
  entity_text="VIC-II",
  min_strength=0.5,
  max_results=20
)
```

**Returns:** All entities related to the specified entity with strength scores.

#### find_related_entities
Find entities most strongly related to a given entity.

```
find_related_entities(
  entity_text="VIC-II",
  entity_type="hardware",  # Optional filter
  min_strength=0.5,
  max_results=10
)
```

**Returns:** Top related entities sorted by relationship strength.

#### search_entity_pair
Find documents where two specific entities co-occur.

```
search_entity_pair(
  entity1="VIC-II",
  entity2="sprite",
  max_results=10
)
```

**Returns:** Documents containing both entities with relationship contexts.

**Example output:**
```
Entity Pair Search: VIC-II ↔ sprite
Documents Found: 12

1. C64 Programmer's Reference (programmers_ref)
   Relationship Strength: 0.92
   Co-occurrences: 8
   Contexts:
     *The VIC-II chip controls sprite graphics...*
     *Eight hardware sprites are managed by VIC-II registers...*
   ...
```

#### extract_relationships_bulk
Bulk extract relationships from multiple documents.

```
extract_relationships_bulk(
  min_strength=0.3,
  max_docs=null,  # Process all documents
  skip_existing=true
)
```

**Returns:** Processing statistics and relationship breakdowns.

**Note:** Relationship extraction requires prior entity extraction. Run `extract_entities` or `extract_entities_bulk` first.

## Duplicate Detection

The knowledge base automatically detects and prevents duplicate content from being indexed multiple times. When you add a document, the system:

1. **Generates content-based ID** - Creates a hash from the document's text content (first 10,000 words)
2. **Checks for duplicates** - Compares against existing documents using the content hash
3. **Returns existing document** - If duplicate detected, returns the existing document instead of creating a new entry

**Benefits:**
- Prevents storage bloat from duplicate content
- Improves search quality by eliminating duplicate results
- Works regardless of file path or filename
- Clear logging when duplicates are detected

**Example:**
```
# First time - document is indexed
doc1 = add_document("C:/docs/vic-ii-guide.pdf", title="VIC-II Guide")

# Second time - same content, different path - duplicate detected
doc2 = add_document("C:/backup/vic-ii-guide-copy.pdf", title="Copy of VIC-II Guide")

# doc2.doc_id == doc1.doc_id (same content hash)
# No duplicate entry created in knowledge base
```

## Suggested Tags

Organize your C64 docs with consistent tags:

- `reference` - General reference guides
- `memory-map` - Memory maps and addresses
- `basic` - BASIC programming
- `assembly` - 6502/6510 assembly
- `sid` - SID chip / sound
- `vic-ii` - VIC-II chip / graphics
- `cia` - CIA chips / I/O
- `kernal` - Kernal ROM routines
- `hardware` - Hardware specifications
- `disk` - Disk drives and DOS

## Usage Examples

Once configured, you can ask Claude Code things like:

- "Search the C64 docs for SID voice registers"
- "What does the memory map say about $D400?"
- "Find information about sprite multiplexing"
- "Search for the exact phrase 'VIC-II chip' in the docs"
- "Find documentation about 'raster interrupts' in graphics-related docs"
- "Add C:/docs/c64/mapping_the_c64.pdf to the knowledge base with tags memory-map, reference"

**Phrase search examples:**
```
search_docs(query='"VIC-II chip"')
search_docs(query='"raster interrupt" timing')
search_docs(query='"SID register" $D400')
```

## Data Storage

The knowledge base uses an **SQLite database** for efficient storage and querying:
- `knowledge_base.db` - SQLite database containing documents and chunks
  - `documents` table - Document metadata with full-text indexes
  - `chunks` table - Chunked document content with foreign key relationships

**Automatic Migration**: If you're upgrading from a previous version with JSON files (`index.json` and `chunks/*.json`), the server will automatically migrate your data to SQLite on first run. The JSON files are preserved as backup and can be manually deleted after verification.

**Benefits of SQLite**:
- Lazy loading - Only loads document metadata at startup, not all chunks
- ACID transactions - Data integrity guaranteed
- Efficient queries - Fast chunk retrieval and statistics
- Scalable - Supports 100,000+ documents without memory issues

Default location: `~/.tdz-c64-knowledge` (or set via `TDZ_DATA_DIR`)

## Troubleshooting

### "pypdf not installed" or "rank_bm25 not found"
Run: `pip install pypdf rank-bm25` (in your virtual environment)

### "mcp module not found"
Run: `pip install mcp` (in your virtual environment)

### Server not responding
Make sure you're using the Python from your virtual environment, not the system Python.

### PDF extraction issues
Some scanned PDFs may not extract text well. Consider using OCR tools to convert them first, or add the plain text version instead.

### BM25 issues
If you experience search problems with BM25:
1. Check the logs in `TDZ_DATA_DIR/server.log`
2. Try disabling BM25 with `USE_BM25=0` environment variable
3. Ensure rank-bm25 is installed: `pip show rank-bm25`

## Development

### Running Tests

Install development dependencies:
```cmd
pip install -e ".[dev]"
```

Run the test suite:
```cmd
pytest test_server.py -v
```

Run tests with coverage:
```cmd
pytest test_server.py -v --cov=server --cov-report=term
```

### CI/CD Pipeline

This project includes a GitHub Actions workflow that:
- Runs tests on Python 3.10, 3.11, and 3.12
- Tests on Windows, Linux, and macOS
- Performs code quality checks with Ruff
- Validates documentation completeness
- Runs integration tests

The pipeline runs automatically on push to main/master/develop branches and on pull requests.

## License

MIT License - Use freely for your retro computing projects!
