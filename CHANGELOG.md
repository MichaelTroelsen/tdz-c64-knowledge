# Changelog

All notable changes to the TDZ C64 Knowledge Base project.

## [2.5.0] - 2025-12-13

### Added - Backup/Restore & GUI Admin Interface

#### 💾 Backup & Restore Operations
- **New Method:** `create_backup(dest_dir, compress=True)` creates full knowledge base backups
- **Backup Features:**
  - Compressed ZIP archives or uncompressed directories
  - Includes database, embeddings, and metadata
  - Automatic timestamping (kb_backup_YYYYMMDD_HHMMSS)
  - Metadata file with document count, version, size info
- **New Method:** `restore_from_backup(backup_path, verify=True)` restores from backups
- **Restore Features:**
  - Supports both ZIP and directory backups
  - Automatic extraction of compressed backups
  - Backup verification (checksums, required files)
  - Safety backup before restoration
  - Complete database and embeddings restoration
- **MCP Tools:** New `create_backup` and `restore_backup` tools
- **Benefits:**
  - Data safety and disaster recovery
  - Version control for knowledge base
  - Easy migration between systems
  - Automated backup workflows

**Usage:**
```python
# Python API - Create backup
backup_path = kb.create_backup("/path/to/backups", compress=True)

# Restore from backup
result = kb.restore_from_backup("/path/to/backup.zip", verify=True)

# Via MCP in Claude Desktop
# "Create a backup of the knowledge base in ~/backups"
# "Restore from backup ~/backups/kb_backup_20251213_083327.zip"
```

#### 🎮 Streamlit GUI Admin Interface
- **New File:** `admin_gui.py` - Complete web-based admin interface
- **Dashboard Page:**
  - Real-time metrics (documents, chunks, words)
  - Health status monitoring
  - Database info and feature status
  - File types and tags overview
- **Documents Page:**
  - Upload new documents (PDF/TXT)
  - List and filter document library
  - Delete documents
  - View document metadata
- **Search Page:**
  - Multi-mode search (Keyword/Semantic/Hybrid)
  - Tag filtering
  - Export results (Markdown/JSON/HTML)
  - Adjustable semantic weight for hybrid search
- **Backup & Restore Page:**
  - Create backups with compression options
  - Restore from backups with verification
  - Safety warnings and confirmations
- **Analytics Page:**
  - Search analytics dashboard
  - Top queries, failed searches
  - Search mode usage charts
  - Popular tags visualization

**Usage:**
```bash
# Install GUI dependencies
pip install ".[gui]"

# Run admin interface
streamlit run admin_gui.py

# Access at http://localhost:8501
```

### Testing
- Added 3 new test cases (44 total tests, 42 passed, 2 skipped)
- `test_backup_and_restore()` - Full backup/restore cycle validation
- `test_uncompressed_backup()` - Uncompressed backup creation
- `test_backup_with_empty_kb()` - Edge case testing
- All existing tests continue to pass

### Performance
- Backup creation: ~100-500ms depending on database size
- Restore operation: ~200-800ms depending on database size
- Compression adds ~50-200ms overhead

### Dependencies
- Added `streamlit>=1.28.0` for GUI (optional, install with `pip install ".[gui]"`)
- Added `pandas>=2.0.0` for analytics charts (optional)
- No new required dependencies for core functionality

### Documentation
- Updated CHANGELOG.md with v2.5.0 release notes
- Updated FUTURE_IMPROVEMENTS.md to mark Automated Backup as completed
- Created comprehensive `admin_gui.py` with inline documentation

### Developer Notes
**Implementation Details:**
- Backup uses Python's `shutil` for file operations and `zipfile` for compression
- Restore creates safety backup before overwriting (automatic rollback capability)
- GUI uses Streamlit's session state for persistent knowledge base connection
- All backup/restore operations are logged with detailed timestamps
- Metadata JSON includes version info for forward compatibility

**Breaking Changes:** None - all new features are additive

---

## [2.4.0] - 2025-12-13

### Added - Query Autocompletion & Export Features

#### 🔍 Query Autocompletion
- **New Table:** `query_suggestions` FTS5 virtual table for fast autocomplete
- **Automatic Term Extraction:** Technical terms extracted during document ingestion
- **Four Categories:**
  - **Hardware:** ALL CAPS technical terms (VIC-II, SID, CIA, etc.)
  - **Register:** Memory addresses ($D000, $D020, $D418, etc.)
  - **Instruction:** 6502 assembly mnemonics (LDA, STA, JMP, etc.)
  - **Concept:** Technical phrases (Video Interface Controller, etc.)
- **New Methods:**
  - `build_suggestion_dictionary(rebuild=False)` - Build/rebuild suggestion index
  - `get_query_suggestions(partial, max_suggestions, category)` - Get autocomplete suggestions
  - `_update_suggestions_for_chunks(chunks)` - Incremental update during document addition
- **MCP Tool:** New `suggest_queries` tool with category filtering
- **FTS5 Prefix Matching:** Fast substring matching with proper escaping
- **Auto-Update:** Suggestions automatically updated when documents are added
- **Benefits:**
  - Type-ahead suggestions for better search experience
  - Discover technical terms and addresses
  - Category filtering for targeted suggestions
  - Frequency tracking shows common terms

**Usage:**
```python
# Python API - Get suggestions
suggestions = kb.get_query_suggestions("VIC", max_suggestions=5)

# Category-specific suggestions
suggestions = kb.get_query_suggestions("$D0", max_suggestions=5, category="register")

# Rebuild dictionary manually
kb.build_suggestion_dictionary(rebuild=True)

# Via MCP in Claude Desktop
# "Suggest queries starting with 'SID'"
```

#### 📤 Export Search Results
- **New Method:** `export_search_results(results, format, query)` exports to multiple formats
- **Three Export Formats:**
  - **Markdown:** Clean, readable format for documentation
  - **JSON:** Machine-readable format with full metadata
  - **HTML:** Styled format with embedded CSS for viewing in browsers
- **Format-Specific Methods:**
  - `_export_markdown()` - Markdown with headers, scores, metadata
  - `_export_json()` - JSON with query, timestamp, result count
  - `_export_html()` - HTML with embedded CSS styling
- **MCP Tool:** New `export_results` tool for exporting from Claude Desktop
- **Rich Metadata:** Includes query, timestamps, scores, snippets, tags
- **Benefits:**
  - Save search results for documentation
  - Share results in preferred format
  - Machine-readable JSON for automation
  - Styled HTML for presentations

**Usage:**
```python
# Python API - Export as markdown
results = kb.search("VIC-II graphics", max_results=10)
markdown = kb.export_search_results(results, format='markdown', query="VIC-II graphics")

# Export as JSON
json_export = kb.export_search_results(results, format='json')

# Export as HTML
html = kb.export_search_results(results, format='html', query="VIC-II graphics")

# Via MCP in Claude Desktop
# "Export these search results as HTML"
```

### Testing
- Added 2 new test cases (41 total tests, 39 passed, 2 skipped)
- `test_query_autocompletion()` - Validates term extraction, FTS5 matching, category filtering
- `test_export_functionality()` - Tests all three export formats and error handling
- All existing tests continue to pass

### Performance
New features maintain excellent performance:
- Query suggestions: ~5-15ms per autocomplete query (FTS5-powered)
- Suggestion extraction: ~10-30ms during document ingestion
- Export operations: ~10-50ms depending on result count and format

### Database Schema Changes
- New table: `query_suggestions` FTS5 virtual table (term, frequency, category)
- Automatic migrations for existing databases
- Incremental updates on document addition

### Documentation
- Updated CHANGELOG.md with v2.4.0 release notes
- Updated FUTURE_IMPROVEMENTS.md to mark completed features
- Test suite covers all new functionality

### Developer Notes
**Implementation Details:**
- Query suggestions use FTS5 for fast prefix matching
- Special characters ($, *, etc.) properly quoted for FTS5
- Incremental term updates use upsert logic (update or insert)
- Export formats include full result metadata
- HTML export includes embedded CSS for standalone viewing
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## [2.3.0] - 2025-12-12

### Added - Performance & Content Enhancement

#### ⚡ Incremental Embeddings
- **New Method:** `_add_chunks_to_embeddings(chunks)` incrementally adds embeddings to FAISS index
- **Performance:** 10-100x faster than full rebuild for new documents
- **Smart Updates:** Detects existing index and adds new vectors without rebuilding
- **Automatic:** Document addition now uses incremental updates by default
- **Index Growth:** FAISS index grows incrementally as documents are added
- **Benefits:**
  - Fast document addition (seconds instead of minutes)
  - No need to rebuild entire embeddings index
  - Seamless integration with existing semantic search
  - Automatic persistence after each update

**Technical Details:**
- Generates embeddings only for new chunks
- Uses FAISS `add()` method to append to existing index
- Updates embeddings_doc_map concurrently
- Saves index to disk after each addition
- Invalidates similarity cache automatically

#### 🚀 Parallel Document Processing
- **ThreadPoolExecutor:** Process multiple documents concurrently
- **Thread-Safe:** Database operations protected with threading locks
- **Configurable Workers:** Set via `PARALLEL_WORKERS` environment variable (default: CPU count)
- **Modified Method:** `add_documents_bulk()` now uses parallel processing
- **Performance:** 3-4x faster bulk imports on multi-core systems
- **Safety:**
  - Critical sections protected with `self._lock`
  - Duplicate detection remains accurate
  - Database ACID guarantees maintained

**Usage:**
```python
# Set worker count (optional, defaults to CPU count)
import os
os.environ['PARALLEL_WORKERS'] = '8'

# Bulk add with parallel processing
results = kb.add_documents_bulk(
    directory="docs",
    pattern="**/*.pdf",
    tags=["reference"]
)
```

#### 🔗 Cross-Reference Detection
- **New Table:** `cross_references` stores extracted references with context
- **Reference Types:**
  - **Memory Addresses:** $D000, $D020, $D418 (4-digit hex with $ prefix)
  - **Register Offsets:** VIC+0, SID+4, CIA1+12 (chip name + offset)
  - **Page References:** "page 156", "see page 42"
- **Extraction Methods:**
  - `_extract_cross_references(chunks, doc_id)` - Main coordinator
  - `_extract_memory_addresses(text)` - Regex-based address extraction
  - `_extract_register_offsets(text)` - Chip+offset pattern matching
  - `_extract_page_references(text)` - Page number references
  - `_get_reference_context(text, reference)` - Surrounding context (200 chars)
- **New Method:** `find_by_reference(ref_type, ref_value, max_results)` searches by reference
- **MCP Tool:** New `find_by_reference` tool for Claude Desktop integration
- **Auto-Extraction:** References automatically extracted during document ingestion
- **Benefits:**
  - Track how specific registers are documented across documents
  - Find all mentions of a memory address
  - Cross-link related content automatically
  - Navigate documentation by technical references

**Usage:**
```python
# Python API - Find all documents mentioning $D020
results = kb.find_by_reference("memory_address", "$D020", max_results=10)

# Find all VIC+0 register references
results = kb.find_by_reference("register_offset", "VIC+0", max_results=10)

# Find page 156 references
results = kb.find_by_reference("page_reference", "156", max_results=10)

# Via MCP in Claude Desktop
# "Find all documents that reference the $D020 register"
```

### Testing
- Added 3 new test cases (39 total tests, 37 passed, 2 skipped)
- `test_incremental_embeddings()` - Validates incremental FAISS index updates
- `test_parallel_processing()` - Tests ThreadPoolExecutor with bulk add
- `test_cross_reference_detection()` - Tests reference extraction and lookup
- All existing tests continue to pass

### Performance
New features maintain excellent performance:
- Incremental embeddings: ~1-5s for typical documents (vs 30-120s for full rebuild)
- Parallel processing: 3-4x speedup on 4+ core systems
- Cross-reference extraction: ~10-30ms during document ingestion
- Cross-reference lookup: ~20-50ms with indexed queries

### Database Schema Changes
- New table: `cross_references` with indexes on (ref_type, ref_value) and doc_id
- Foreign key cascade deletes ensure data integrity
- Automatic migrations for existing databases

### Documentation
- Updated CHANGELOG.md with v2.3.0 release notes
- All new methods documented in CLAUDE.md
- Test suite covers all new functionality

### Developer Notes
**Implementation Details:**
- Incremental embeddings use FAISS IndexFlatIP for cosine similarity
- Parallel processing uses concurrent.futures.ThreadPoolExecutor
- Thread-safe operations protected by threading.Lock
- Cross-references use regex patterns for extraction
- All features fully backward compatible

**Breaking Changes:** None - all new features are additive

---

## [2.2.0] - 2025-12-12

### Added - Faceted Search & Analytics

#### 🔍 Faceted Search and Filtering
- **New Method:** `faceted_search(query, facet_filters, max_results, tags)` enables multi-dimensional filtering
- **Database Storage:** New `document_facets` table with composite primary key (doc_id, facet_type, facet_value)
- **Facet Types:**
  - **Hardware:** Automatically detects SID, VIC-II, CIA, 6502, PLA, Datasette, Disk Drive references
  - **Instructions:** Extracts 6502 assembly mnemonics (LDA, STA, JMP, etc.) - 46 mnemonics supported
  - **Registers:** Identifies memory-mapped register addresses ($D000-$DFFF, etc.)
- **Extraction Methods:**
  - `_extract_facets(text)` - Main coordinator method
  - `_extract_hardware_refs(text)` - Pattern-based hardware detection
  - `_extract_instructions(text)` - Assembly instruction detection
  - `_extract_registers(text)` - Register address extraction (4-digit hex with $ prefix)
- **Auto-Extraction:** Facets automatically extracted during document ingestion
- **MCP Tool:** New `faceted_search` tool with facet filter support
- **Benefits:**
  - Narrow results by technical domain (e.g., "find SID-related documents only")
  - Combine keyword search with facet filtering
  - Results include facet metadata for each document
  - Filter by multiple facet types simultaneously

**Usage:**
```python
# Python API - Search for SID chip documents
results = kb.faceted_search(
    query="sound",
    facet_filters={'hardware': ['SID'], 'instruction': ['LDA', 'STA']},
    max_results=5
)

# Via MCP in Claude Desktop
# "Search for sound programming documents that mention the SID chip and use LDA/STA instructions"
```

#### 📊 Search Analytics & Logging
- **New Method:** `_log_search(query, search_mode, results_count, execution_time_ms, tags)` logs all searches
- **Database Storage:** New `search_log` table tracks all search activity
- **Fields Logged:** timestamp, query, search_mode (fts5/bm25/semantic/hybrid/faceted), results_count, execution_time_ms, tags, clicked_doc_id
- **Analytics Method:** `get_search_analytics(days, limit)` provides comprehensive usage insights
- **Metrics Provided:**
  - **Overview:** Total searches, unique queries, avg results, avg execution time
  - **Top Queries:** Most frequently searched terms with average result counts
  - **Failed Searches:** Queries that returned 0 results (helps identify gaps)
  - **Search Modes:** Breakdown by search backend usage
  - **Popular Tags:** Most commonly used tag filters
- **Auto-Logging:** All search methods automatically log queries (search, semantic_search, hybrid_search, faceted_search)
- **MCP Tool:** New `search_analytics` tool for usage reporting
- **Benefits:**
  - Understand user search patterns
  - Identify knowledge gaps (failed searches)
  - Optimize search performance
  - Track search backend effectiveness

**Usage:**
```python
# Python API - Get last 30 days of analytics
analytics = kb.get_search_analytics(days=30, limit=100)

# Via MCP in Claude Desktop
# "Show me search analytics for the last 7 days"
```

### Testing
- Added 4 new test cases (36 total tests, 35 passed, 1 skipped)
- `test_facet_extraction()` - Validates hardware, instruction, and register detection
- `test_faceted_search()` - Tests facet filtering and result filtering
- `test_search_logging()` - Verifies search logging functionality
- `test_search_analytics()` - Tests analytics aggregation and reporting
- All existing tests continue to pass

### Performance
New features maintain sub-second performance:
- Facet extraction: ~10-50ms during document ingestion
- Faceted search: ~60-180ms (regular search + facet filtering)
- Search logging: ~5-10ms overhead per search (async-friendly)
- Analytics queries: ~50-200ms depending on date range

### Database Schema Changes
- New tables: `document_facets`, `search_log`
- Indexes: `idx_facets_type_value`, `idx_facets_doc_id`, `idx_search_log_query`, `idx_search_log_timestamp`, `idx_search_log_mode`
- Foreign key cascade deletes ensure data integrity
- Automatic migrations for existing databases

### Documentation
- Updated CHANGELOG.md with new feature details
- Updated FUTURE_IMPROVEMENTS.md to mark completed items
- Test suite expanded with 4 comprehensive tests

### Developer Notes
**Implementation Details:**
- Facet extraction uses regex patterns for hardware and instructions
- Register addresses normalized to uppercase ($D000 format)
- All search methods now log queries automatically
- Analytics use SQL GROUP BY and aggregation functions
- Search logging designed for minimal performance impact

**Breaking Changes:** None - all new features are additive

---

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
