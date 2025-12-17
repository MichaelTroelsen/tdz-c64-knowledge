# Complete Feature Documentation

**TDZ C64 Knowledge Base v2.12.0 - All Features & Capabilities**

---

## Table of Contents

1. [Search Features](#search-features)
2. [Document Management](#document-management)
3. [Data Extraction](#data-extraction)
4. [User Interfaces](#user-interfaces)
5. [Advanced Features](#advanced-features)
6. [Performance Features](#performance-features)
7. [Security Features](#security-features)
8. [Integration Features](#integration-features)

---

## Search Features

### 1. Full-Text Search (FTS5)
**Status:** ✅ Enabled | **Performance:** 50ms (480x faster than BM25)

Native SQLite FTS5 implementation with Porter stemming tokenizer.

**Features:**
- Word stemming (running → run, programming → program)
- Phrase matching ("VIC-II chip")
- Boolean operators (AND, OR, NOT)
- Ranking by relevance

**Example:**
```bash
search "memory mapping"
# Returns: Documents about memory organization
```

**Technical Details:**
- Tokenizer: Porter stemming
- Language: English
- Storage: Virtual FTS5 table in SQLite
- Sync: Automatic triggers keep index current

---

### 2. Semantic/Conceptual Search
**Status:** ✅ Installed | **Performance:** 7-16ms per query

Meaning-based search using embeddings and vector similarity.

**Features:**
- Concept matching ("movable objects" → "sprites")
- Intent understanding
- Related documents discovery
- No keyword matching required

**Example:**
```bash
set USE_SEMANTIC_SEARCH=1
search "how do I make sounds?"
# Returns: SID programming documentation
```

**Technical Details:**
- Model: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: FAISS with cosine similarity
- Embeddings: Pre-computed for all chunks
- Search Speed: ~7-16ms

---

### 3. BM25 Ranking Algorithm
**Status:** ✅ Enabled | **Performance:** 24000ms (fallback)

Industry-standard probabilistic ranking for keyword matching.

**Features:**
- Document length normalization
- Term frequency scoring
- Inverse document frequency weighting
- Parameter tuning (k1=1.2, b=0.75)

**When Used:**
- Primary search if FTS5 returns no results
- Fallback when FTS5 unavailable
- Configurable via USE_BM25 environment variable

---

### 4. Hybrid Search
**Status:** ✅ Enabled | **Performance:** 60-180ms

Combines FTS5 keyword search with semantic search.

**Features:**
- Weighted scoring (default: 30% semantic, 70% keyword)
- Intelligent result merging
- Best of both worlds: exact + conceptual
- Configurable semantic weight (0.0-1.0)

**Example:**
```bash
search "SID sound programming" --hybrid
# Returns: Both exact keyword matches AND related concepts
```

**Algorithm:**
1. FTS5 search for exact matches
2. Semantic search for concepts
3. Normalize both scores (0-1)
4. Weighted combination
5. Sort by combined score

---

### 5. Fuzzy/Typo-Tolerant Search
**Status:** ✅ Enabled | **Threshold:** 80%

Handles misspellings using Levenshtein distance.

**Features:**
- Automatic typo detection
- Configurable similarity threshold (0-100%)
- Word-level matching
- Smart fallback to exact matching

**Examples:**
- "registr" → finds "register" (89% similar)
- "VIC-I" → finds "VIC-II" (83% similar)
- "asembly" → finds "assembly" (88% similar)

**Configuration:**
```bash
set USE_FUZZY_SEARCH=1
set FUZZY_THRESHOLD=80  # 0-100, default 80%
```

---

### 6. Phrase Search
**Status:** ✅ Enabled

Exact phrase matching with quote detection.

**Features:**
- Exact phrase matching (whole words)
- Score boost (2x for BM25, 10x for simple)
- Quote detection via regex
- Combined with term search

**Example:**
```bash
search "VIC-II chip register"
# Exact phrase gets higher score than individual words
```

---

### 7. Query Preprocessing
**Status:** ✅ Enabled

NLTK-powered text normalization.

**Features:**
- Tokenization (word segmentation)
- Stopword removal (the, a, an, etc.)
- Porter stemming (running → run)
- Technical term preservation (VIC-II, 6502)

**Applied To:**
- User queries
- Indexed corpus during search
- Both search methods

**Example:**
```bash
Query: "How to program the SID chip?"
Processed: ["program", "sid", "chip"]  # Stopwords removed
```

---

### 8. Search Result Ranking
**Status:** ✅ Enabled | **Performance:** 50-100x speedup for cached results

Results automatically ranked by relevance.

**Ranking Factors:**
1. Exact phrase matches (highest weight)
2. Term frequency in document
3. Inverse document frequency (IDF)
4. Query term coverage
5. Document length normalization

**Search Caching:**
- TTL: 5 minutes (configurable)
- Size: 100 entries (configurable)
- Hit rate: 50-100x speedup for repeated queries
- Auto-invalidation on document changes

---

### 9. Search Term Highlighting
**Status:** ✅ Enabled

Matching terms highlighted in results.

**Features:**
- Markdown bold (`**term**`)
- Case-insensitive matching
- Whole-word boundary detection
- Multiple term highlighting

**Example Result:**
```
...The **SID** chip controls **sound** synthesis on the **C64**...
```

---

### 10. Tag-Based Filtering
**Status:** ✅ Enabled

Filter search results by document tags.

**Features:**
- Multiple tag support
- AND/OR filtering
- Tag autocomplete (in GUI)
- Hierarchical tags (category/subtopic)

**Example:**
```bash
search "programming" --tags assembly --max 10
# Only returns assembly-tagged documents
```

**Available Tags:**
- c64, c128 (platform)
- reference, codebase64 (category)
- assembly, basic (language)
- sid, vic-ii, cia (hardware)
- music, graphics (topic)

---

## Document Management

### 1. Multi-Format Support
**Status:** ✅ Enabled

Support for 5+ document formats.

**Formats:**
- ✅ **PDF** (.pdf) - Via pypdf with OCR fallback
- ✅ **Text** (.txt) - With encoding detection
- ✅ **Markdown** (.md) - Treated as text
- ✅ **HTML** (.html, .htm) - Via BeautifulSoup4
- ✅ **Excel** (.xlsx, .xls) - Via openpyxl

**Detection:** Automatic by file extension

---

### 2. Bulk Document Import
**Status:** ✅ Enabled

Batch import entire directories.

**Features:**
- Recursive folder scanning
- Glob pattern matching (default: `**/*.{pdf,txt,md,html,xlsx}`)
- Duplicate skipping
- Progress reporting
- Error handling (failed files don't stop process)

**Example:**
```bash
add-folder "C:\docs" --tags reference --recursive
# Processes all supported formats recursively
```

---

### 3. Duplicate Detection
**Status:** ✅ Enabled

Content-based deduplication prevents duplicate indexing.

**Algorithm:**
- MD5 hash of normalized content (first 10k words)
- Case-insensitive comparison
- Lowercase normalization
- Returns existing document ID

**Benefits:**
- Saves storage (same file at different paths)
- Improves search quality (no duplicate results)
- Detects truly same content

---

### 4. Document Metadata Extraction
**Status:** ✅ Enabled

Automatic extraction of document metadata from PDFs.

**Extracted Fields:**
- Author
- Subject
- Creator
- Creation date
- Page count
- File type
- File size

**Storage:** In SQLite documents table

---

### 5. PDF Page Number Tracking
**Status:** ✅ Enabled

Tracks PDF page numbers for accurate referencing.

**Features:**
- PAGE BREAK marker insertion during extraction
- Page number calculation based on markers
- Returned in search results
- Accurate to actual page

**Example Result:**
```
Page: 47 (of 504)
Chunk: 142
...content from page 47...
```

---

### 6. OCR for Scanned PDFs
**Status:** ✅ Enabled (Tesseract installed)

Automatic fallback OCR for scanned documents.

**Features:**
- Automatic detection (< 100 chars = scanned)
- Tesseract OCR processing
- Per-page processing with error handling
- Graceful fallback if OCR unavailable
- Optional Poppler requirement (not installed)

**Configuration:**
```bash
set USE_OCR=1
set POPPLER_PATH=C:\poppler\Library\bin  # Optional
```

---

## Data Extraction

### 1. Table Extraction
**Status:** ✅ Enabled | **Performance:** ~1-2 sec per table

Automatic extraction of structured tables from PDFs.

**Features:**
- pdfplumber-based extraction
- Markdown format conversion
- Row/column metadata
- Full-text indexing
- Page tracking

**Extraction Details:**
- Tables detected per page
- Converted to markdown format
- Stored in document_tables table
- FTS5 indexed for searching

**Example:**
```
Current KB has 209+ extracted tables
- Memory mapping tables
- Register reference tables
- Opcode tables
- Hardware specifications
```

**Search:**
```bash
search "memory address" --max 5
# Returns: Tables with memory addresses
```

---

### 2. Code Block Detection
**Status:** ✅ Enabled | **Performance:** Instant detection

Automatic detection of code blocks in documents.

**Supported Code Types:**
- **BASIC** - Line-numbered programs (10 PRINT, 20 GOTO)
- **Assembly** - 6502 mnemonics (LDA, STA, JMP, BRK)
- **Hex** - Memory dumps with addresses ($D000: 00 01 02)

**Detection Method:**
- Regex pattern matching
- Minimum 3 consecutive lines
- Mnemonic dictionary for assembly

**Detection Examples:**
- BASIC: Lines starting with numbers + keywords
- Assembly: 6502 instruction mnemonics
- Hex: Address format ($XXXX:) with bytes

**Storage:**
- document_code_blocks table
- FTS5 indexed
- Block type, content, line count tracked

**Current KB:**
- 1,876+ detected code blocks
- 50+ BASIC blocks
- 208+ Assembly blocks
- 0 Hex blocks

---

### 3. Facet Extraction
**Status:** ✅ Enabled | **Performance:** Instant

Extraction of technical metadata (facets).

**Facet Types:**
- **Hardware Components** (VIC-II, SID, CIA, 6502)
- **Assembly Instructions** (LDA, STA, JMP, BRK, etc.)
- **Register References** ($D000, $D020, $1C04, etc.)

**Current KB:**
- 1,876+ total facets
- 6 hardware components
- 55 instructions
- 1,829 registers

**Use Cases:**
- Hardware-specific documentation discovery
- Instruction reference
- Memory address lookup

---

## User Interfaces

### 1. Web GUI (Streamlit)
**Status:** ✅ Running | **URL:** http://localhost:8501

Beautiful web-based dashboard.

**Pages:**
- **Search** - Full search interface with filters
- **Documents** - Browse and manage documents
- **Tags** - Tag management interface
- **Dashboard** - Statistics and analytics
- **Settings** - Feature configuration
- **Analytics** - Query metrics and trends

**Features:**
- Real-time search
- Filter by tags
- Sort results
- Preview documents
- Responsive design
- Dark/light mode (Streamlit default)

**Launch:**
```bash
launch-gui-full-features.bat
```

---

### 2. Command-Line Interface (CLI)
**Status:** ✅ Enabled

Powerful command-line tool.

**Commands:**
- `search` - Search knowledge base
- `list` - List all documents
- `add` - Add single document
- `add-folder` - Bulk import from directory
- `remove` - Delete document
- `stats` - Show statistics

**Usage:**
```bash
launch-cli-full-features.bat search "query" --max 10 --tags tag1
launch-cli-full-features.bat add-folder "path" --tags reference --recursive
launch-cli-full-features.bat list
launch-cli-full-features.bat stats
```

---

### 3. MCP Server (Model Context Protocol)
**Status:** ✅ Ready | **Transport:** stdio

Integration with Claude Code and Claude Desktop.

**Exposed Tools:**
- search_docs
- semantic_search
- hybrid_search
- find_similar
- search_tables
- search_code
- add_document
- add_documents_bulk
- remove_document
- list_docs
- get_document
- kb_stats
- health_check
- check_updates

**Configuration:**
```json
{
  "command": "path/to/.venv/Scripts/python.exe",
  "args": ["path/to/server.py"],
  "env": {
    "USE_FTS5": "1",
    "USE_SEMANTIC_SEARCH": "1"
  }
}
```

---

## Advanced Features

### 1. Hybrid Search Combination
**Status:** ✅ Enabled

Smart combination of multiple search methods.

**Algorithm:**
1. Execute FTS5 search (fast, exact)
2. Execute semantic search (conceptual)
3. Normalize scores (0-1 range)
4. Weighted combination (default 70% FTS5, 30% semantic)
5. Merge duplicate results
6. Sort by combined score

**Configuration:**
```bash
set USE_SEMANTIC_SEARCH=1  # Enables hybrid
# semantic_weight defaults to 0.3
```

---

### 2. Smart Similarity Search
**Status:** ✅ Enabled

Find similar documents automatically.

**Modes:**
- **Semantic** - Uses FAISS embeddings (if available)
- **TF-IDF** - Fallback using sklearn
- **Both** - Automatic selection

**Features:**
- Document-level similarity
- Chunk-level similarity (optional)
- Tag filtering
- Configurable result count

**Example:**
```bash
find_similar <doc_id> --max 5
# Returns: 5 most similar documents
```

---

### 3. Health Monitoring
**Status:** ✅ Enabled

Comprehensive system diagnostics.

**Monitored Systems:**
- Database integrity (table checksums, orphan detection)
- Feature availability (FTS5, semantic, BM25, embeddings)
- Performance metrics (cache utilization, index status)
- Disk space (warnings if < 1GB free)

**Information Provided:**
- System status (OK/WARNING/ERROR)
- Detailed metrics
- Issue list with suggestions

---

## Performance Features

### 1. Search Result Caching
**Status:** ✅ Enabled

TTL-based caching for query results.

**Configuration:**
```bash
set SEARCH_CACHE_SIZE=100      # Max cached queries
set SEARCH_CACHE_TTL=300       # 5-minute expiry
```

**Benefits:**
- 50-100x speedup for repeated queries
- Automatic invalidation on document changes
- Minimal memory overhead

**Cache Stats:**
- Entry count tracking
- Hit/miss ratio
- Automatic cleanup of expired entries

---

### 2. Index Building Optimization
**Status:** ✅ Enabled

Lazy-loaded, efficient index building.

**Optimizations:**
- On-demand BM25 index building (~50 seconds one-time)
- FTS5 indexes (pre-built, instantly searchable)
- Semantic embeddings (one-time generation ~1 min)
- Caching of built indexes

**Process:**
- First search builds BM25 index
- Subsequent searches use cache
- Document changes invalidate caches

---

### 3. Database Query Optimization
**Status:** ✅ Enabled

SQLite optimizations for fast queries.

**Optimizations:**
- FTS5 virtual tables (native search)
- Automatic trigger sync (keeps indexes current)
- Query result caching
- Prepared statement reuse
- Transaction batching

**Performance:**
- Search queries: 50ms (FTS5) vs 24s (BM25)
- Document addition: ~5-10 seconds per document
- Duplicate detection: < 1 second

---

## Security Features

### 1. Path Traversal Protection
**Status:** ✅ Enabled

Prevents access to files outside allowed directories.

**Features:**
- Directory whitelisting via ALLOWED_DOCS_DIRS
- Path normalization and resolution
- Blocks path traversal attempts (../ sequences)
- Clear error messages

**Configuration:**
```bash
set ALLOWED_DOCS_DIRS=C:\docs\allowed,C:\research
```

---

### 2. Content-Based Deduplication
**Status:** ✅ Enabled

Prevents duplicate content indexing.

**Security Benefits:**
- Prevents accidental re-indexing of malicious files
- Content hash comparison (MD5)
- Normalized comparison (case-insensitive)

---

### 3. Database Integrity
**Status:** ✅ Enabled

SQLite ACID compliance ensures data safety.

**Features:**
- Atomic transactions
- Consistency checks
- Durability (writes to disk)
- Isolation between transactions

---

## Integration Features

### 1. Claude Code Integration
**Status:** ✅ Ready

Full integration with Claude Code MCP system.

**Features:**
- MCP server on stdio
- All tools exposed to Claude
- Full context awareness
- Async support

**Usage:**
> "Search the C64 knowledge base for assembly examples"

---

### 2. Environment Variable Configuration
**Status:** ✅ Enabled

Comprehensive configuration system.

**Configurable Items:**
- Search algorithms (FTS5, semantic, BM25)
- Performance settings (cache size/TTL)
- Security (allowed directories)
- LLM integration (provider, API key, model)

---

### 3. Logging & Diagnostics
**Status:** ✅ Enabled

Comprehensive logging system.

**Log Levels:**
- INFO - Normal operations
- WARNING - Potential issues
- ERROR - Error conditions

**Output:**
- File: server.log
- Console: stderr

---

## Feature Status Summary

| Feature | Status | Performance |
|---------|--------|-------------|
| FTS5 Search | ✅ | 50ms |
| Semantic Search | ✅ | 7-16ms |
| BM25 Ranking | ✅ | 24s (cached) |
| Hybrid Search | ✅ | 60-180ms |
| Fuzzy Search | ✅ | Configurable |
| Query Preprocessing | ✅ | 5-10ms |
| Result Caching | ✅ | 50-100x speedup |
| Table Extraction | ✅ | 1-2s per table |
| Code Detection | ✅ | Instant |
| Facet Extraction | ✅ | Instant |
| OCR Support | ✅ | 1-2s per page |
| Duplicate Detection | ✅ | <1s |
| Streamlit GUI | ✅ | - |
| CLI Tool | ✅ | - |
| MCP Server | ✅ | - |

---

## Next Generation Features (Planned)

Coming in future releases:

1. **RAG Question Answering** - Natural language Q&A with citations
2. **Document Summarization** - Auto-generate summaries
3. **Entity Extraction** - Extract names, dates, technical terms
4. **VICE Emulator Integration** - Link to running C64 emulator
5. **REST API** - HTTP interface for web apps
6. **Plugin System** - Extensible architecture

---

**Last Updated:** 2025-12-17
**Version:** 2.12.0
**Status:** ✅ All Features Operational
