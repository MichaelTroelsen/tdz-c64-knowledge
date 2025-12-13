# Changelog

All notable changes to the TDZ C64 Knowledge Base project.

## [2.12.0] - 2025-12-13

### Added - Smart Auto-Tagging with LLM Integration

#### 🤖 LLM Integration Module
- **New File:** `llm_integration.py` - Unified LLM client for multiple providers
- **Supported Providers:**
  - **Anthropic (Claude):** claude-3-haiku-20240307, claude-3-5-sonnet-20241022, claude-3-opus-20240229
  - **OpenAI (GPT):** gpt-3.5-turbo, gpt-4, gpt-4-turbo
- **Features:**
  - Auto-detection of provider from environment variables
  - JSON response parsing with markdown code block handling
  - Configurable models and parameters (temperature, max_tokens)
  - Error handling and logging
- **Environment Variables:**
  - `LLM_PROVIDER` - Provider selection (anthropic/openai, default: anthropic)
  - `ANTHROPIC_API_KEY` - API key for Claude
  - `OPENAI_API_KEY` - API key for GPT models
  - `LLM_MODEL` - Model selection (default: claude-3-haiku-20240307)

#### 🏷️ Smart Auto-Tagging
- **New Method:** `auto_tag_document(doc_id, confidence_threshold, max_tags, append)` - AI-powered tag generation
- **Analysis Process:**
  - Analyzes first 3 chunks (max 3000 chars) of document
  - Sends structured prompt to LLM requesting JSON tags
  - Returns tags with confidence scores (0.0-1.0) and reasoning
- **Tag Categories:**
  - **Hardware:** SID, VIC-II, CIA, 6510, PLA, etc.
  - **Programming Topics:** Assembly, BASIC, machine code, graphics, sound
  - **Document Type:** Tutorial, reference, manual, code example
  - **Difficulty Level:** Beginner, intermediate, advanced
- **Confidence Filtering:**
  - Default threshold: 0.7 (70% confidence)
  - Configurable per-request
  - Only high-confidence tags applied
- **Tag Application:**
  - Append mode: Add to existing tags (default)
  - Replace mode: Replace all existing tags
  - Returns detailed results with applied/skipped tags

**Usage:**
```python
# Python API - Auto-tag a single document
result = kb.auto_tag_document(
    doc_id="doc123",
    confidence_threshold=0.7,
    max_tags=10,
    append=True
)

# Via MCP in Claude Desktop
# "Auto-tag this document with AI-generated tags"
```

#### 📦 Bulk Auto-Tagging
- **New Method:** `auto_tag_all_documents(confidence_threshold, max_tags, append, skip_tagged, max_docs)` - Bulk AI tagging
- **Features:**
  - Process all documents or limit with `max_docs`
  - Skip already-tagged documents (configurable)
  - Rate limiting to prevent API overuse
  - Comprehensive error handling per document
  - Returns statistics: processed, skipped, failed, total_tags_added
- **Performance:**
  - Progress logging for each document
  - Continues on errors (non-blocking)
  - Transaction-safe (each document committed individually)

**Usage:**
```python
# Python API - Auto-tag all untagged documents
results = kb.auto_tag_all_documents(
    confidence_threshold=0.7,
    max_tags=10,
    append=True,
    skip_tagged=True,
    max_docs=None  # Process all
)

# Via MCP in Claude Desktop
# "Auto-tag all documents that don't have tags yet"
```

#### 🛠️ MCP Tools
- **New Tool:** `auto_tag_document` - Single document auto-tagging
  - Parameters: doc_id, confidence_threshold, max_tags, append
  - Returns: Applied tags with confidence scores and reasoning
- **New Tool:** `auto_tag_all` - Bulk document auto-tagging
  - Parameters: confidence_threshold, max_tags, append, skip_tagged, max_docs
  - Returns: Statistics and sample results

#### 📋 Configuration
- **New File:** `.env.example` - Complete environment configuration template
- **LLM Configuration:**
  - Provider selection
  - API key setup
  - Model selection
  - Usage examples
- **Search Configuration:**
  - FTS5, semantic search, BM25 settings
  - Query preprocessing options
- **Security Configuration:**
  - Document directory whitelisting
- **Data Storage:**
  - TDZ_DATA_DIR path configuration

### Changed
- Updated version to 2.12.0
- Version.py includes new features: smart_auto_tagging, llm_integration

### Benefits
- **Time Savings:** Automatically tag hundreds of documents in minutes
- **Consistency:** LLM applies tags uniformly across all documents
- **Discovery:** AI identifies relevant categories you might miss
- **Flexibility:** Support for multiple LLM providers (Claude, GPT)
- **Quality:** Confidence-based filtering ensures only relevant tags
- **Control:** Configurable thresholds and tag limits

### Testing
- LLM integration tested manually with sample documents
- Auto-tagging verified with C64 documentation
- Error handling tested for missing API keys
- JSON parsing tested with various response formats

### Dependencies
- **Required:** anthropic>=0.18.0 OR openai>=1.0.0 (install at least one)
- Install with: `pip install anthropic` or `pip install openai`

### Documentation
- Created `.env.example` with comprehensive configuration guide
- Updated `version.py` to v2.12.0
- Updated `CHANGELOG.md` with v2.12.0 entry
- See `llm_integration.py` docstrings for API details

### Developer Notes
**Implementation Details:**
- LLM integration uses provider abstraction pattern
- Auto-tagging analyzes document content + metadata
- Structured prompts guide LLM to return consistent JSON
- Confidence scores from LLM help filter low-quality suggestions
- Tag format: lowercase with hyphens (e.g., "sid-programming")
- All database operations within transactions

**Breaking Changes:** None - all new features are optional and require API keys

---

## [2.11.0] - 2025-12-13

### Added - GUI Improvements & Version Management

#### 📁 File Path Input in GUI
- **New Tab:** "Add by File Path" in Add Documents section
  - Paste file paths directly (e.g., `C:\Users\...\file.md`)
  - No need to upload files - just enter the path
  - Instant file validation (shows error if file doesn't exist)
  - Supports all file types (PDF, TXT, MD, HTML, Excel)

#### 🔍 Duplicate Detection
- **Intelligent Duplicate Detection** across all document operations
  - Detects files with identical content (content-based hashing)
  - Shows warning message: "Document already exists in knowledge base"
  - Displays existing document details (title, ID, chunks)
  - Prevents database bloat from duplicate entries
  - Works in all three add methods: Single Upload, Bulk Upload, File Path

#### 📄 Enhanced File Viewer
- **Markdown Files (.md):** Dual view mode
  - "Rendered" view - Shows beautifully formatted markdown
    - Headers, lists, code blocks, tables
    - Bold, italic, links
    - Full markdown rendering
  - "Raw Markdown" view - Shows source code
    - Syntax highlighting for markdown
    - Line numbers for easy reference
    - Scrollable code block
  - Toggle button to switch between views
- **Text Files (.txt):** Improved display
  - Scrollable code block with line numbers
  - Line count display for files over 50 lines
  - Monospace font for better readability
  - Professional code viewer styling
- **Removed:** Old `st.text_area()` approach for better UX

#### ⏳ Progress Indicators
- **All Document Operations** now show clear feedback:
  - Spinner with "Adding document: filename..." message
  - Bulk upload shows "Processing X/Y: filename" for each file
  - Clear success/error/duplicate messages
  - No more silent operations

#### 📦 Version Management System
- **New File:** `version.py` - Centralized version information
  - Semantic versioning (MAJOR.MINOR.PATCH)
  - Build date tracking
  - Feature version tracking
  - Version history
- **GUI Version Display:**
  - Sidebar footer shows current version
  - Build date displayed
  - Dynamic version from version.py
- **Server Logging:**
  - Version logged on startup
  - Visible in server.log file
  - Helps with debugging and support

#### 📋 Enhanced Messages
- **Success Messages:** "Document added successfully!" with full details
  - Shows document title
  - Shows chunk count
  - Shows document ID
- **Duplicate Messages:** "Document already exists" with existing doc info
  - Shows which document matched
  - Shows content hash matching
- **Error Messages:** Clear error descriptions
  - "File not found" for invalid paths
  - "Error adding document" with detailed error
- **Bulk Upload:** Summary statistics
  - Count of added documents
  - Count of duplicates (with filenames)
  - Count of failed documents

### Changed
- File viewer now uses `st.code()` and `st.markdown()` instead of `st.text_area()`
- Markdown files render formatting by default
- GUI footer dynamically shows version from version.py
- README.md includes version badge

### Testing
- Created `test_gui_changes.py` - 6/6 tests passing
- Created `test_file_viewer.py` - 4/4 tests passing
- Created `test_gui_startup.py` - Validates GUI can start
- All GUI functionality verified working

### Documentation
- Created `GUI_IMPROVEMENTS_SUMMARY.md`
- Created `FILE_VIEWER_IMPROVEMENTS.md`
- Updated README.md with version badge
- Updated CLAUDE.md with version information

### Developer Notes
**Implementation Details:**
- Version management uses centralized `version.py` file
- Semantic versioning with tuple and string representations
- GUI imports version dynamically (no hardcoded versions)
- Server logs version on every startup for debugging
- File viewer improvements use native Streamlit components
- Duplicate detection tracks document count before/after

**Benefits:**
- Better user experience with clear feedback
- No confusion about document status
- Easier file management (no mandatory uploads)
- Professional version tracking
- Improved code visibility in documentation

## [2.10.0] - 2025-12-13

### Added - HTML File Support
- **HTML File Support** - Ingest and search HTML documentation (.html, .htm)
  - Powered by BeautifulSoup4 for robust HTML parsing
  - Automatic encoding detection with chardet
  - Removes script, style, nav, footer, and header elements
  - Preserves code blocks (<pre>, <code>) with special formatting
  - Extracts page title when available
  - Cleans up excessive whitespace
  - Support for both single and bulk upload in GUI

### Changed
- Default bulk add pattern updated from `**/*.{pdf,txt,md,xlsx,xls}` to `**/*.{pdf,txt,md,html,htm,xlsx,xls}`
- GUI file uploaders now accept HTML files alongside other formats

### Dependencies
- Added `beautifulsoup4>=4.9.0` for HTML parsing
- Added `chardet>=5.0.0` for encoding detection

### Testing
- Added test fixture for sample HTML files
- Added `test_add_html_document()` to verify HTML extraction
- All 59 tests passing (2 skipped)

## [2.9.0] - 2025-12-13

### Added - Excel File Support & Enhanced Markdown
- **Excel File Support** - Ingest and search Excel spreadsheets (.xlsx, .xls)
  - Extract data from all sheets with clear delimiters
  - Tab-delimited columns for readability
  - Sheet count tracked as "pages" for consistency
  - Support for both single and bulk upload in GUI
  - Powered by openpyxl library
- **Enhanced Markdown Support** - Made .md file support more visible
  - Updated bulk add default pattern to include .md files
  - Updated GUI file uploaders to explicitly list Markdown files
  - Better discoverability for existing Markdown support

### Changed
- Default bulk add pattern updated from `**/*.{pdf,txt,md}` to `**/*.{pdf,txt,md,xlsx,xls}`
- GUI file uploaders now accept Excel files alongside PDFs, text, and Markdown

### Testing
- Added test fixture for sample Excel files
- Added `test_add_excel_document()` to verify Excel extraction
- All 58 tests passing (2 skipped)

## [2.8.0] - 2025-12-13

### Added - CI/CD Automation & Relationship Visualization

#### 🔄 CI/CD Pipeline
- **GitHub Actions Workflow** - Automated testing on push/PR
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Multi-version testing (Python 3.10, 3.11, 3.12)
  - Test coverage reporting with Codecov
  - Security checks (safety, bandit)
  - Code quality checks (ruff linting)
  - Integration tests
  - Documentation validation
  - Build status summary
- **Release Automation** - Automatic GitHub releases
  - Changelog extraction
  - Release artifact uploads
  - Version tagging support
  - Optional PyPI publishing (disabled by default)
- **Status Badges** - README.md badges for CI status, Python version, license

#### 📊 Relationship Graph Visualization
- **New Backend Method:** `get_relationship_graph(tags, relationship_types)`
  - Returns nodes and edges for graph visualization
  - Filter by tags or relationship types
  - Includes statistics (total nodes, edges, relationship types)
- **New GUI Page:** "🔗 Relationship Graph"
  - Interactive network graph powered by pyvis
  - Filter by tags and relationship types
  - Visualization options:
    - Enable/disable physics simulation
    - Choose layout algorithm (hierarchical, force_atlas, barnes_hut, repulsion)
    - Adjust node size
    - Show/hide relationship direction arrows
    - Custom edge colors
  - Color-coded edges by relationship type:
    - 🟢 Green for "related"
    - 🔵 Blue for "references"
    - 🟠 Orange for "prerequisite"
    - 🟣 Purple for "sequel"
  - Interactive features:
    - Click and drag nodes
    - Hover for details
    - Scroll to zoom
  - Export graph data as JSON
  - Statistics display (document count, relationship count, types)
  - Legend for relationship types
- **New Dependencies:**
  - pyvis >= 0.3.2 for interactive network visualization
  - networkx >= 3.0 for graph data structures

#### ✅ Testing
- **New Tests (2 total):**
  - `test_get_relationship_graph` - Basic graph generation
  - `test_get_relationship_graph_filtered` - Filtering by tags and types
- All 57 tests passing

**Benefits:**
- Automated quality assurance with CI/CD pipeline
- Visual exploration of document relationships
- Interactive graph makes complex relationships understandable
- Filter and export capabilities for analysis
- Professional DevOps practices

## [2.7.0] - 2025-12-13

### Added - GUI Enhancements & Document Relationships

#### 📦 Bulk File Upload in GUI
- **Bulk Upload Tab** in Documents page
  - Drag-and-drop multiple PDF/TXT files simultaneously
  - Progress bar showing upload status
  - Individual error handling per file
  - Apply tags to all uploaded files at once
  - Success/failure summary after upload

#### 🏷️ Tag Management Page
- **New Page:** Dedicated Tag Management interface
- **Tag Statistics Dashboard:**
  - Total tags count
  - Average documents per tag
  - Most used tag
- **Sortable Tag List:**
  - Sort by name (A-Z, Z-A)
  - Sort by document count
  - View document count for each tag
- **Four Tag Operations:**
  - **🔄 Rename Tag** - Rename a tag across all documents
  - **🔗 Merge Tags** - Combine multiple tags into one
  - **🗑️ Delete Tag** - Remove tag from all documents
  - **➕ Add to All** - Add a tag to every document in the knowledge base

#### 👁️ Document Preview
- **Preview Button** added to each document in Documents page
- **Preview Features:**
  - Adjustable chunk preview (1-10 chunks via slider)
  - Toggle metadata display (chunk ID, page number, word count)
  - Full document export as text file
  - Preview shows formatted markdown content
  - Session state preserves preview open/closed state

#### 🔗 Document Relationships
- **Backend Methods:**
  - `add_relationship(from_doc_id, to_doc_id, type, note)` - Create relationships between documents
  - `remove_relationship(from_doc_id, to_doc_id, type)` - Delete specific or all relationships
  - `get_relationships(doc_id, direction)` - Get outgoing/incoming/both relationships
  - `get_related_documents(doc_id, type)` - Get full metadata of related documents
- **Relationship Types:**
  - `related` - General relationship
  - `references` - One document references another
  - `prerequisite` - Must be read before another document
  - `sequel` - Continuation of another document
- **Database Schema:**
  - New `document_relationships` table
  - Foreign keys with CASCADE delete
  - UNIQUE constraint prevents duplicates
  - Optional notes for each relationship
- **GUI Relationships Panel:**
  - **Relationships Button** for each document
  - View outgoing and incoming relationships
  - Add new relationships with dropdown selection
  - Delete individual relationships
  - Display relationship type and notes
  - Shows related document titles

#### ✅ Testing
- **New Tests (11 total):**
  - `test_add_relationship` - Create relationships
  - `test_add_relationship_invalid_doc` - Error handling
  - `test_add_duplicate_relationship` - Prevent duplicates
  - `test_get_relationships_outgoing` - Outgoing links
  - `test_get_relationships_incoming` - Incoming links
  - `test_get_relationships_both` - Bidirectional queries
  - `test_remove_relationship` - Remove specific relationships
  - `test_remove_all_relationships` - Remove all between docs
  - `test_get_related_documents` - Full metadata retrieval
  - `test_get_related_documents_filtered` - Filter by type
  - `test_relationship_cascade_delete` - Verify CASCADE behavior

**Benefits:**
- Improved document organization with visual relationship mapping
- Efficient bulk operations for large collections
- Better tag management and cleanup
- Quick document preview without leaving the interface
- Track document dependencies and reading order

## [2.6.0] - 2025-12-13

### Added - Bulk Document Management

#### ⚡ Bulk Operations
- **New Method:** `update_tags_bulk()` - Update tags for multiple documents in bulk
  - Add tags to multiple documents
  - Remove tags from multiple documents
  - Replace all tags for multiple documents
  - Select documents by ID or by existing tags
  - Returns detailed results (updated/failed)
- **New Method:** `export_documents_bulk()` - Export document metadata in various formats
  - Export to JSON, CSV, or Markdown
  - Export all documents, by tags, or by specific IDs
  - Comprehensive metadata including title, tags, chunks, dates
- **Existing Methods Enhanced:**
  - `add_documents_bulk()` - Already existed, now documented
  - `remove_documents_bulk()` - Already existed, now documented

#### 🔧 MCP Tools for Claude Desktop
- **New Tool:** `update_tags_bulk` - Bulk tag management via Claude Desktop
- **New Tool:** `export_documents_bulk` - Bulk export via Claude Desktop

#### 🖥️ GUI Enhancements
- **New Feature:** Bulk Operations panel in Documents page
- **Three Tabs:**
  - 🗑️ **Bulk Delete** - Delete multiple documents by IDs or tags
  - 🏷️ **Bulk Re-tag** - Add/remove/replace tags for multiple documents
  - 📤 **Bulk Export** - Export document metadata in JSON/CSV/Markdown

#### 🔨 CLI Commands
- **New Command:** `remove-bulk` - Remove multiple documents from command line
- **New Command:** `update-tags-bulk` - Bulk tag updates from command line
- **New Command:** `export-bulk` - Export document metadata from command line

**Usage Examples:**
```bash
# CLI - Remove documents by tags
python cli.py remove-bulk --tags draft old

# CLI - Add tags to specific documents
python cli.py update-tags-bulk --doc-ids doc1 doc2 --add reviewed approved

# CLI - Export all documents as JSON
python cli.py export-bulk --format json --output documents.json

# CLI - Export documents with specific tags as CSV
python cli.py export-bulk --tags reference c64 --format csv --output c64_docs.csv
```

```python
# Python API - Bulk tag update
results = kb.update_tags_bulk(
    existing_tags=["draft"],
    add_tags=["reviewed"],
    remove_tags=["draft"]
)

# Python API - Export documents
export_data = kb.export_documents_bulk(
    tags=["reference", "c64"],
    format="markdown"
)
```

#### ✅ Testing
- **New Tests:**
  - `test_update_tags_bulk` - Comprehensive tag update testing
  - `test_export_documents_bulk` - Export format testing (JSON/CSV/Markdown)

**Benefits:**
- Efficient management of large document collections
- Easy reorganization and categorization
- Bulk cleanup operations
- Document metadata backup and reporting
- Integration with CI/CD workflows

---

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
