# Changelog

All notable changes to the TDZ C64 Knowledge Base project.

## [2.1.0] - 2025-12-12

### Added - Table Extraction and Code Block Detection

#### 📊 Table Extraction from PDFs
- **New Method:** `_extract_tables(filepath)` extracts structured tables from PDF documents using pdfplumber
- **Markdown Conversion:** Tables automatically converted to markdown format via `_table_to_markdown()`
- **Database Storage:** New `document_tables` table with 7 fields (doc_id, table_id, page, markdown, searchable_text, row_count, col_count)
- **FTS5 Search Index:** New `tables_fts` virtual table for full-text search on table content
- **MCP Tool:** New `search_tables` tool for searching tables with page numbers and metadata
- **Search Method:** New `search_tables(query, max_results, tags)` method with FTS5-powered search
- **Benefits:**
  - Critical for C64 documentation (memory maps, register tables, opcode references)
  - Preserves table structure and searchability
  - Returns results with row/column counts and page numbers
  - Fully integrated with document lifecycle (automatic cleanup on delete)

**Usage:**
```python
# Python API
results = kb.search_tables("VIC-II sprite registers", max_results=5)

# Via MCP in Claude Desktop
# "Search for tables about sprite registers"
```

#### 🔧 Code Block Detection
- **New Method:** `_detect_code_blocks(text)` detects BASIC, Assembly, and Hex dump code blocks using regex
- **Three Code Types Supported:**
  - **BASIC:** Line-numbered programs (e.g., "10 PRINT", "20 GOTO")
  - **Assembly:** 6502 mnemonics (LDA, STA, JMP, etc.)
  - **Hex Dumps:** Memory dumps with addresses (e.g., "D000: 00 01 02 03")
- **Database Storage:** New `document_code_blocks` table with 7 fields (doc_id, block_id, page, block_type, code, searchable_text, line_count)
- **FTS5 Search Index:** New `code_fts` virtual table for full-text search on code content
- **MCP Tool:** New `search_code` tool with optional block_type filtering
- **Search Method:** New `search_code(query, max_results, block_type, tags)` method
- **Pattern Matching:** Requires 3+ consecutive lines for detection to avoid false positives
- **Benefits:**
  - Automatically extracts programming examples from documentation
  - Searchable code snippets with syntax highlighting
  - Filter results by code type (BASIC/Assembly/Hex)
  - Returns line counts and page numbers

**Usage:**
```python
# Python API
results = kb.search_code("LDA STA", max_results=5, block_type="assembly")

# Via MCP in Claude Desktop
# "Search for assembly code that uses LDA and STA"
```

### Testing
- Added 4 new test cases (32 total tests, 31 passed, 1 skipped)
- `test_code_block_detection()` - Validates BASIC, Assembly, and Hex dump detection
- `test_table_extraction()` - Verifies pdfplumber integration
- `test_table_search()` - Tests FTS5 table search functionality
- `test_code_search()` - Tests code block search with type filtering
- All existing tests continue to pass

### Performance
New features maintain sub-second performance:
- Table extraction: Depends on PDF size and table count
- Code block detection: ~10-50ms for typical documents
- Table search: ~50-150ms (FTS5-powered)
- Code search: ~50-150ms (FTS5-powered)

### Documentation
- Updated CLAUDE.md with new architecture details
- Added sections for table extraction and code block detection
- Updated Key Methods section with new methods
- This CHANGELOG documents all improvements

### Developer Notes
**Implementation Details:**
- Uses pdfplumber for table extraction (optional dependency)
- Regex patterns for code detection (BASIC: `\d+\s+[A-Z]+`, Assembly: mnemonics, Hex: `[0-9A-F]{4}:\s*(?:[0-9A-F]{2}\s*)+`)
- Both features use FTS5 virtual tables with automatic triggers for synchronization
- Foreign key cascade deletes ensure data integrity
- Schema migrations handle existing databases automatically

**Database Schema Changes:**
- New tables: `document_tables`, `document_code_blocks`
- New FTS5 indexes: `tables_fts`, `code_fts`
- Automatic triggers for INSERT/UPDATE/DELETE synchronization
- Backward compatible with existing databases (migrations run automatically)

**Breaking Changes:** None - all new features are additive

---

## [2.0.0] - 2025-12-12

### Added - Priority 1 Improvements

#### 🎯 Hybrid Search
- **New Method:** `hybrid_search(query, max_results, tags, semantic_weight)` combines FTS5 keyword search with semantic search
- **Configurable Weighting:** Adjust balance between keyword precision and semantic recall (default: 70% FTS5, 30% semantic)
- **Score Normalization:** Both search types normalized to 0-1 range for fair comparison
- **Result Merging:** Intelligent merging of results from both search backends
- **MCP Tool:** New `hybrid_search` tool available in Claude Desktop integration
- **Benefits:**
  - Best of both worlds: exact keyword matching AND conceptual understanding
  - Better handles technical terms + synonyms simultaneously
  - Example: "6502 assembler" finds exact matches AND related content about machine code, opcodes

**Usage:**
```python
# Python API
results = kb.hybrid_search("SID chip sound", max_results=5, semantic_weight=0.3)

# Via MCP in Claude Desktop
# "Use hybrid search to find information about graphics sprites"
```

#### ✨ Enhanced Snippet Extraction
- **Term Density Scoring:** Sliding window analysis finds regions with highest query term concentration
- **Complete Sentences:** Expands to sentence boundaries instead of hard cutoffs
- **Code Block Preservation:** Detects and preserves complete code blocks (indented lines)
- **Whole Word Highlighting:** Uses word boundaries for better highlighting accuracy
- **Better Context:** More relevant snippets with natural boundaries

**Improvements:**
- No more mid-sentence cuts
- Better context around search terms
- Code snippets preserved intact
- 80% snippet size threshold ensures adequate context

#### 🏥 Health Monitoring
- **New Method:** `health_check()` performs comprehensive system diagnostics
- **Database Health:**
  - Integrity checking (PRAGMA integrity_check)
  - Database file size monitoring
  - Orphaned chunks detection
  - Disk space warnings (< 1GB free)
- **Feature Status:**
  - FTS5 enabled/available
  - Semantic search enabled/available (with embeddings count)
  - BM25 enabled
  - Query preprocessing status
- **Performance Metrics:**
  - Cache status and utilization
  - BM25 index build status
- **MCP Tool:** New `health_check` tool returns formatted system status

**Health Check Output:**
```
Status: HEALTHY
Message: All systems operational

Metrics:
  documents: 145
  chunks: 4,665
  total_words: 6,870,642

Database:
  size_mb: 45.23
  integrity: ok
  disk_free_gb: 125.5

Features:
  ✓ fts5_enabled: True
  ✓ fts5_available: True
  ✓ semantic_search_enabled: True
  ✓ semantic_search_available: True
  ✓ embeddings_count: 2347

Performance:
  cache_enabled: True
  cache_size: 23
  cache_capacity: 100

✓ No issues detected
```

### Testing
- Added 3 new test cases (28 total tests, all passing)
- `test_health_check()` - Validates health check structure and metrics
- `test_enhanced_snippet_extraction()` - Verifies improved snippet quality
- `test_hybrid_search()` - Tests hybrid search score normalization and merging
- Test coverage: 27 passed, 1 skipped (semantic search in test env)

### Performance
All new features maintain sub-second performance:
- Hybrid search: ~60-180ms (combines two searches)
- Enhanced snippets: Same as regular search (minimal overhead from sentence detection)
- Health check: ~50-100ms (database queries + file stats)

### Documentation
- Updated CLAUDE.md with new methods
- This CHANGELOG documents all improvements
- Test suite expanded to cover new functionality

### Developer Notes
**Implementation Details:**
- Hybrid search uses score normalization (max-value scaling for FTS5, native 0-1 for semantic)
- Enhanced snippets use regex sentence boundary detection with 80% size threshold
- Health check uses SQLite PRAGMA commands for integrity + Python shutil for disk stats
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## Previous Versions

### [1.0.0] - 2025-12-11
- Initial production release
- 145 documents, 4,665 chunks, 6.8M words
- FTS5 full-text search (480x speedup vs BM25)
- Semantic search with all-MiniLM-L6-v2 embeddings
- OCR support for scanned PDFs (Tesseract + Poppler)
- Content-based duplicate detection
- SQLite database backend
- MCP integration for Claude Desktop
- 25 comprehensive tests
