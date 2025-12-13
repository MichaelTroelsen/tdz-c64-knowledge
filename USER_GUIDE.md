# TDZ C64 Knowledge Base - User Guide

Complete guide for using the Commodore 64 documentation knowledge base with Claude Desktop.

## Table of Contents
- [What's New in v2.8.0](#whats-new-in-v280) ⭐⭐⭐⭐⭐
- [What's New in v2.7.0](#whats-new-in-v270) ⭐⭐⭐⭐
- [What's New in v2.5.0](#whats-new-in-v250) ⭐⭐⭐
- [What's New in v2.0.0](#whats-new-in-v200) ⭐
- [Quick Start](#quick-start)
- [Features](#features)
- [Search Modes](#search-modes)
- [GUI Admin Interface](#gui-admin-interface) ⭐ NEW
- [Backup & Restore](#backup--restore) ⭐ NEW
- [Claude Desktop Integration](#claude-desktop-integration)
- [Command Line Usage](#command-line-usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## What's New in v2.8.0

### 🔄 CI/CD Automation
Automated testing and release workflows for professional quality assurance.

**CI/CD Pipeline Features:**
- ✅ **Automated Testing** - Runs on every push and pull request
  - Multi-platform: Ubuntu, Windows, macOS
  - Multi-version: Python 3.10, 3.11, 3.12
  - 57 tests across all features
- ✅ **Test Coverage** - Codecov integration for coverage tracking
- ✅ **Security Checks** - Dependency vulnerability scanning (safety) and code security analysis (bandit)
- ✅ **Code Quality** - Automated linting with ruff
- ✅ **Documentation Checks** - Validates README and other docs
- ✅ **Status Badges** - README shows CI status, Python version, license

**Release Automation:**
- 🏷️ Automatic GitHub releases when tags are pushed
- 📄 Changelog extraction from CHANGELOG.md
- 📦 Artifact uploads (README, guides, docs)
- 🔖 Version tagging support
- 📦 Optional PyPI publishing (when ready)

### 📊 Relationship Graph Visualization
**Visual exploration of document relationships with interactive network graphs.**

**Features:**
- 🎨 **Interactive Network Graph** - Click, drag, zoom, hover
- 🔍 **Smart Filters:**
  - Filter by document tags
  - Filter by relationship types (related, references, prerequisite, sequel)
- ⚙️ **Visualization Options:**
  - Physics simulation on/off
  - Layout algorithms: hierarchical, force_atlas, barnes_hut, repulsion
  - Adjustable node size
  - Show/hide relationship arrows
  - Custom edge colors
- 🎨 **Color-Coded Edges:**
  - 🟢 Green = "related" (general relationship)
  - 🔵 Blue = "references" (cites/quotes)
  - 🟠 Orange = "prerequisite" (must read first)
  - 🟣 Purple = "sequel" (continuation)
- 📊 **Statistics Dashboard:**
  - Total documents in graph
  - Total relationships
  - Relationship types count
- 📥 **Export** - Download graph data as JSON

**How to Use:**
1. Navigate to "🔗 Relationship Graph" page in admin GUI
2. **Optional:** Apply filters (tags or relationship types)
3. Explore the interactive graph:
   - Click and drag nodes to rearrange
   - Hover over nodes/edges for details
   - Scroll to zoom in/out
4. **Optional:** Adjust visualization settings in ⚙️ Options
5. **Optional:** Export graph data for external analysis

**Use Cases:**
```
📚 Learning Paths - Visualize prerequisite chains for educational content
🔍 Research - See how topics reference each other
📖 Documentation - Understand document dependencies
🗺️ Content Mapping - Explore your knowledge base structure
```

**Python API:**
```python
# Get full relationship graph
graph = kb.get_relationship_graph()

# Filter by tags
graph = kb.get_relationship_graph(tags=["basic", "tutorial"])

# Filter by relationship type
graph = kb.get_relationship_graph(relationship_types=["prerequisite"])

# Returns: {'nodes': [...], 'edges': [...], 'stats': {...}}
```

---

## What's New in v2.7.0

### 📦 Bulk File Upload in GUI
Upload multiple documents at once with drag-and-drop support.

**Features:**
- 📁 Drag and drop multiple PDF/TXT files simultaneously
- 📊 Real-time progress bar showing upload status
- ⚠️ Individual error handling - one failure doesn't stop the rest
- 🏷️ Apply tags to all uploaded files at once
- ✅ Success/failure summary after upload

**How to Use:**
1. Navigate to Documents page in admin GUI
2. Expand "➕ Add Documents" section
3. Click "📦 Bulk Upload" tab
4. Drag multiple files or click to browse
5. Optionally add tags (comma-separated)
6. Click "📦 Add All Documents"

### 🏷️ Tag Management Page
Dedicated page for comprehensive tag operations.

**Tag Statistics Dashboard:**
- 📊 Total tags count
- 📈 Average documents per tag
- 🏆 Most used tag

**Tag Operations:**
- **🔄 Rename Tag** - Rename a tag across all documents that use it
- **🔗 Merge Tags** - Combine multiple tags into a single tag
- **🗑️ Delete Tag** - Remove a tag from all documents
- **➕ Add to All** - Add a tag to every document in the knowledge base

**Sortable Tag List:**
- Sort by name (A-Z or Z-A)
- Sort by document count (most used first)
- View which tags are applied to how many documents

**Example Use Cases:**
```
✅ Rename "c-64" to "c64" across all documents
✅ Merge "ref", "reference", "documentation" into "reference"
✅ Delete obsolete tags like "draft" or "old"
✅ Add "reviewed" tag to all documents
```

### 👁️ Document Preview
Quick preview of document content without leaving the Documents page.

**Preview Features:**
- 📄 Adjustable chunk preview (1-10 chunks)
- 📊 Optional metadata display (chunk ID, page number, word count)
- 📥 Export full document as text file
- 📝 Formatted markdown content rendering
- 💾 Session state preserves preview open/closed

**How to Use:**
1. In Documents page, click "👁️ Preview" button for any document
2. Use slider to adjust number of chunks to preview
3. Toggle "Show metadata" for technical details
4. Click "📥 Export Full Document" to download complete text
5. Click "👁️ Preview" again to close

### 🔗 Document Relationships
Track connections between documents (prerequisites, references, related content).

**Relationship Types:**
- `related` - General relationship between documents
- `references` - One document references/cites another
- `prerequisite` - Must be read before another document
- `sequel` - Continuation of another document

**Relationships Panel Features:**
- **Outgoing** - Documents this one links to
- **Incoming** - Documents that link to this one
- **Add New Relationship** - Create links with type and optional note
- **Delete Relationships** - Remove individual relationships
- **Display Notes** - View relationship descriptions

**How to Use:**
1. In Documents page, click "🔗 Relationships" button for any document
2. View existing outgoing and incoming relationships
3. To add new relationship:
   - Select target document from dropdown
   - Choose relationship type
   - Optionally add a note
   - Click "➕ Add Relationship"
4. To delete: Click 🗑️ button next to any relationship

**Example Use Case:**
```
Document: "C64 BASIC Introduction"
└─ references → "C64 Programmer's Reference"
└─ prerequisite → "C64 Advanced Programming"

Document: "C64 BASIC Introduction"
├─ incoming: "Getting Started with C64"
└─ incoming: "First Steps in Programming"
```

**Python API:**
```python
# Add relationship
kb.add_relationship("intro_doc", "advanced_doc", "prerequisite", "Read intro first")

# Get all relationships for a document
relationships = kb.get_relationships("intro_doc", direction="both")

# Get related documents with full metadata
related = kb.get_related_documents("intro_doc", relationship_type="prerequisite")

# Remove relationship
kb.remove_relationship("intro_doc", "advanced_doc", "prerequisite")
```

---

## What's New in v2.5.0

### 🖥️ Web-Based Admin GUI
Complete Streamlit-based admin interface for managing your knowledge base.

**Key Features:**
- 📊 **Dashboard** - Real-time metrics, health monitoring, system overview
- 📚 **Documents** - Upload PDFs/TXT files with drag-and-drop, browse library, manage documents
- 🔍 **Search** - Three search modes (Keyword/Semantic/Hybrid) with export to Markdown/JSON/HTML
- 💾 **Backup & Restore** - Create compressed backups, restore from backups with verification
- 📈 **Analytics** - Search analytics, query frequency, mode usage, popular tags

**How to Use:**
```bash
# Install GUI dependencies
pip install ".[gui]"

# Run the admin interface
streamlit run admin_gui.py
```

Opens automatically at `http://localhost:8501` - see [GUI Admin Interface](#gui-admin-interface) section for full guide.

### 💾 Backup & Restore Operations
Production-ready backup and restore system for data safety.

**Backup Features:**
- Compressed ZIP archives or uncompressed directories
- Automatic timestamping (kb_backup_YYYYMMDD_HHMMSS)
- Includes database, embeddings, and metadata
- Metadata file with document count, version, size info

**Restore Features:**
- Supports ZIP and directory backups
- Automatic backup extraction
- Safety backup before restoration (rollback capability)
- Verification of backup integrity
- Statistics on restoration

**Usage:**
```python
# Via Python API
kb.create_backup("/path/to/backups", compress=True)
kb.restore_from_backup("/path/to/backup.zip", verify=True)

# Via MCP tools in Claude Desktop
"Create a backup of the knowledge base"
"Restore from backup at C:\backups\kb_backup_20251213_103045.zip"
```

See [Backup & Restore](#backup--restore) section for detailed guide.

---

## What's New in v2.0.0

### 🎯 Hybrid Search
Combines FTS5 keyword precision with semantic understanding for best results.

**Key Benefits:**
- Best of both worlds: exact matches + conceptual understanding
- Configurable weighting (default: 70% FTS5, 30% semantic)
- Example: "SID sound" finds exact "SID" mentions AND audio synthesis concepts
- Performance: 60-180ms

**Usage:**
```python
results = kb.hybrid_search("graphics programming", max_results=5, semantic_weight=0.3)
```

### ✨ Enhanced Snippet Extraction
Smarter, more readable search result snippets.

**Improvements:**
- ✅ Complete sentences (no mid-sentence cuts)
- ✅ Code block preservation (doesn't break indented code)
- ✅ Term density scoring (finds best context)
- ✅ Whole word highlighting

**Example:**
```
Before: "...II chip controls all graphics and vi..."
After:  "The VIC-II chip controls all graphics and video output on the Commodore 64."
```

### 🏥 Health Monitoring
System diagnostics at your fingertips.

**Provides:**
- Database integrity and size
- Feature availability (FTS5, semantic, embeddings)
- Performance metrics (cache, indexes)
- Disk space warnings
- Issue detection

**Usage:**
```python
health = kb.health_check()
# Returns: {'status': 'healthy', 'metrics': {...}, 'features': {...}}
```

---

## Quick Start

### 1. Installation

```bash
# Clone and setup
cd C:\Users\mit\claude\c64server\tdz-c64-knowledge
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Import Documents

```bash
# Import PDF documents
.venv\Scripts\python.exe cli.py add-folder "pdf" --tags reference c64 --recursive

# Import text documents
.venv\Scripts\python.exe cli.py add-folder "txt" --tags reference c64 --recursive
```

### 3. Enable Optimizations

```bash
# Enable FTS5 (480x faster keyword search)
.venv\Scripts\python.exe enable_fts5.py

# Enable semantic search (meaning-based search)
.venv\Scripts\python.exe enable_semantic_search.py
```

### 4. Test It

```bash
# Search for SID chip information
.venv\Scripts\python.exe cli.py search "SID chip sound synthesis" --max 5

# View stats
.venv\Scripts\python.exe cli.py stats
```

---

## Features

### Current Knowledge Base
- **145 Documents** - Comprehensive C64 documentation
- **4,665 Chunks** - Segmented for efficient retrieval
- **6.8M Words** - Extensive technical content
- **File Types:** PDF, Text, Markdown, and Excel (.xlsx, .xls)
- **Topics:** Assembly language, BASIC, hardware (SID, VIC-II, CIA), memory maps, programming guides

### Search Capabilities
1. **Hybrid Search** ⭐ NEW - Combines FTS5 + Semantic for best results (60-180ms)
2. **FTS5 Full-Text Search** - Lightning-fast keyword matching (50-140ms)
3. **Semantic Search** - AI-powered meaning-based search (12-25ms)
4. **Enhanced Snippets** ⭐ NEW - Complete sentences, code preservation, term density
5. **BM25 Ranking** - Industry-standard relevance scoring (fallback)
6. **Fuzzy Matching** - Handles typos (80% threshold)
7. **Query Preprocessing** - Stemming and stopword removal
8. **Phrase Search** - Exact phrase matching with quotes
9. **Tag Filtering** - Filter by document categories
10. **Health Monitoring** ⭐ NEW - System diagnostics and status reporting

---

## Search Modes

### FTS5 Full-Text Search (Recommended for Keywords)

**Best for:** Exact terms, technical keywords, register names

```bash
# Search for specific technical terms
.venv\Scripts\python.exe cli.py search "VIC-II sprite registers" --max 5

# Search with tags
.venv\Scripts\python.exe cli.py search "$D000 register" --tags sid vic-ii --max 10
```

**Performance:** 50-140ms per query (800x faster than BM25)

**Features:**
- Native SQLite FTS5 with Porter stemming
- Automatic synchronization with chunks table
- BM25 ranking built-in
- Falls back to BM25 if no results

### Semantic Search (AI-Powered)

**Best for:** Natural language questions, conceptual searches

```python
# Via Python API
from server import KnowledgeBase
kb = KnowledgeBase("~/.tdz-c64-knowledge")
results = kb.semantic_search("How do I create graphics?", max_results=5)
```

**Performance:** 12-25ms per query

**Features:**
- Understands meaning and context
- Finds related concepts (e.g., "movable objects" → sprites)
- Uses sentence-transformers (all-MiniLM-L6-v2 model)
- FAISS vector similarity search
- Persistent embeddings cache

**Example:**
```
Query: "movable objects"
→ Finds: Sprite documentation, VIC-II programming guides
```

### Hybrid Search ⭐ NEW (Best of Both Worlds)

**Best for:** Maximum precision and recall - combines exact keywords with conceptual understanding

```python
# Via Python API
from server import KnowledgeBase
kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Balanced (default: 70% FTS5, 30% semantic)
results = kb.hybrid_search("SID sound programming", max_results=5)

# More semantic (40% FTS5, 60% semantic)
results = kb.hybrid_search("audio synthesis", max_results=5, semantic_weight=0.6)

# More keyword-focused (90% FTS5, 10% semantic)
results = kb.hybrid_search("$D400 register", max_results=5, semantic_weight=0.1)
```

**Performance:** 60-180ms per query (combines two searches)

**Features:**
- Combines FTS5 keyword precision + semantic recall
- Configurable weighting (semantic_weight: 0.0-1.0)
- Score normalization for fair comparison
- Intelligent result merging
- Returns combined scores: `hybrid_score`, `fts_score`, `semantic_score`

**When to Use:**
- General searches where you want best results
- Technical terms + conceptual understanding needed
- Example: "6502 assembler" finds exact matches AND related content about machine code, opcodes

**Example Results:**
```
Query: "SID sound programming"
Result 1: hybrid_score=0.85 (fts=0.95, semantic=0.45)
  → "Programming the SID Chip" - exact keyword match
Result 2: hybrid_score=0.72 (fts=0.40, semantic=0.95)
  → "Audio Synthesis Techniques" - conceptually related
```

### BM25 Search (Fallback)

**Best for:** When FTS5 returns no results

Automatically used as fallback. No manual configuration needed.

---

## GUI Admin Interface

### Overview

The web-based admin interface provides a user-friendly way to manage your C64 Knowledge Base without using the command line or Claude Desktop.

### Installation

```bash
# Install GUI dependencies (Streamlit and pandas)
pip install ".[gui]"

# Or install individually
pip install streamlit>=1.28.0 pandas>=2.0.0
```

### Running the GUI

```bash
# Start the admin interface
streamlit run admin_gui.py

# Or on a custom port
streamlit run admin_gui.py --server.port 8502
```

The interface will automatically open in your default browser at `http://localhost:8501` (or your custom port).

### Pages Overview

#### 📊 Dashboard

**Purpose:** Monitor knowledge base health and performance

**Displays:**
- **Top Metrics**
  - Total documents count
  - Total chunks count
  - Total words indexed
  - System status (healthy/warning/error)

- **Health Monitoring**
  - Database integrity and size
  - Disk space available
  - Feature status (FTS5, semantic search, embeddings)
  - Performance metrics (cache, indexes)
  - Issue detection and warnings

- **Overview Cards**
  - File types distribution
  - Tags in use
  - System configuration
  - Data directory location

**Best Practices:**
- Check dashboard daily for health status
- Monitor disk space warnings
- Verify features are enabled (green checkmarks)

#### 📚 Documents

**Purpose:** Upload and manage documents in the knowledge base

**Features:**

**Upload Documents:**
1. Click "➕ Add New Document" expander
2. Drag and drop PDF or TXT file (or click to browse)
3. Optional: Enter custom title (defaults to filename)
4. Optional: Add tags (comma-separated, e.g., "reference, sid, programming")
5. Click "Add Document"
6. Wait for processing confirmation

**Document Library:**
- **Browse:** View all indexed documents in a table
- **Filter:** Filter by tags using multiselect dropdown
- **Sort:** Sortable columns (Title, Type, Size, Chunks, Date)
- **Metadata:** See file path, chunk count, word count, added date
- **Delete:** Remove documents with confirmation dialog

**Tips:**
- Use consistent tags: reference, c64, assembly, basic, sid, vic-ii, etc.
- Check for duplicates before uploading (library shows all docs)
- Large PDFs may take 10-30 seconds to process

#### 🔍 Search

**Purpose:** Search the knowledge base with multiple modes and export results

**Search Modes:**

**1. Keyword (FTS5)** - Fast exact matching
- Best for: Technical terms, register names, specific commands
- Speed: 50-140ms
- Example queries: "VIC-II sprites", "$D000 register", "6502 opcodes"

**2. Semantic** - Meaning-based search
- Best for: Conceptual queries, "how to" questions
- Speed: 12-25ms
- Example queries: "How do I create graphics?", "sound programming basics"

**3. Hybrid** - Combines keyword + semantic
- Best for: General search, balanced results
- Speed: 60-180ms
- Adjustable semantic weight (0-1 slider)
- Example queries: "SID programming", "sprite collision"

**Search Options:**
- **Max Results:** Number of results to return (default: 10)
- **Tag Filter:** Filter results by document tags (multiselect)
- **Semantic Weight:** For hybrid search, balance between keyword (0) and semantic (1)

**Results Display:**
- Document title and snippet (with highlighted terms)
- Score (relevance)
- Source (filename)
- Page number (if PDF)
- Metadata (tags, word count)

**Export Results:**
1. Perform a search
2. Select export format: Markdown, JSON, or HTML
3. Click "📤 Export Results"
4. Download file with timestamp

**Export Formats:**
- **Markdown:** Human-readable, includes all metadata
- **JSON:** Machine-readable, includes all fields
- **HTML:** Formatted for web viewing

#### 💾 Backup & Restore

**Purpose:** Create and restore backups for data safety

**Create Backup:**
1. Navigate to "💾 Backup & Restore" page
2. Enter backup directory path (e.g., `C:\backups\c64kb`)
   - Directory will be created if it doesn't exist
3. Check "Compress to ZIP" for compressed archive (recommended)
4. Click "🔄 Create Backup"
5. Note the backup path for future restoration

**Backup Contents:**
- Database file (knowledge_base.db)
- Embeddings files (if available)
- Metadata file (timestamp, version, document count)

**Backup Naming:**
- Format: `kb_backup_YYYYMMDD_HHMMSS.zip`
- Example: `kb_backup_20251213_103045.zip`

**Restore Backup:**
1. Navigate to "💾 Backup & Restore" page
2. Read the warning: **Restoring replaces current database**
3. Enter backup path (file or directory)
   - Supports: `.zip` files or uncompressed backup directories
4. Check "Verify backup before restoring" (recommended)
5. Click "⚠️ Restore Backup"
6. Wait for confirmation
7. GUI will reload with restored data

**Safety Features:**
- Automatic safety backup created before restoration
- Backup verification checks for database and metadata files
- Shows restoration statistics (document count, time elapsed)

**Best Practices:**
- Create backups **before** major changes
- Store backups in a different location (not same directory as data)
- Test restoration periodically
- Keep multiple backup versions (weekly/monthly)
- Enable compression to save disk space

#### 📈 Analytics

**Purpose:** View search analytics and usage patterns

**Metrics Displayed:**

**Summary Stats:**
- Total searches performed
- Unique queries count
- Average results per query
- Average execution time

**Top Queries:**
- Most frequent search queries
- Search count for each query
- Ordered by frequency

**Failed Searches:**
- Queries that returned 0 results
- Helps identify gaps in knowledge base
- Consider adding more documentation for these topics

**Search Mode Usage:**
- Pie chart showing distribution:
  - FTS5 keyword searches
  - Semantic searches
  - Hybrid searches
- Helps understand user preferences

**Popular Tags:**
- Most frequently searched tags
- Tag usage frequency
- Identifies most valuable document categories

**Time Range Selection:**
- 7 days / 14 days / 30 days / 60 days / 90 days
- Filter analytics by time period
- Track trends over time

**Use Cases:**
- Identify popular topics
- Find knowledge gaps (failed searches)
- Optimize search modes based on usage
- Track knowledge base effectiveness

### Configuration

The GUI uses the same data directory as the MCP server.

**Default:** `~/.tdz-c64-knowledge`

**Custom Directory:**
```bash
# Set environment variable
export TDZ_DATA_DIR="/path/to/data"
streamlit run admin_gui.py
```

**Windows:**
```cmd
set TDZ_DATA_DIR=C:\Users\YourName\c64-knowledge-data
streamlit run admin_gui.py
```

### Remote Access

**Warning:** Only use on trusted networks!

```bash
# Allow remote connections
streamlit run admin_gui.py --server.address 0.0.0.0

# Custom port + remote access
streamlit run admin_gui.py --server.address 0.0.0.0 --server.port 8502
```

Access from other devices: `http://YOUR_IP:8501`

### Troubleshooting

**GUI Won't Start:**
```bash
# Check if streamlit is installed
streamlit --version

# Reinstall if needed
pip install --force-reinstall streamlit
```

**Can't Upload Documents:**
- Verify file format (only PDF and TXT supported)
- Check data directory has write permissions
- Check available disk space
- Review error message in GUI

**Search Returns No Results:**
- Try different search modes (Keyword/Semantic/Hybrid)
- Remove tag filters
- Check Dashboard to verify documents are indexed
- Try broader search terms

**Backup Fails:**
- Verify destination directory exists and is writable
- Check available disk space
- Ensure knowledge base is not being modified during backup
- Review error message details

---

## Bulk Document Management

### Overview

Bulk operations allow you to efficiently manage large collections of documents. Available via GUI, CLI, Python API, and Claude Desktop (MCP).

### Bulk Operations Available

1. **Bulk Delete** - Remove multiple documents at once
2. **Bulk Re-tag** - Update tags for multiple documents
3. **Bulk Export** - Export document metadata in various formats

### Via GUI (Easiest)

1. Open admin interface: `streamlit run admin_gui.py`
2. Navigate to "📚 Documents" page
3. Click "⚡ Bulk Operations" expander
4. Choose from three tabs: Bulk Delete, Bulk Re-tag, or Bulk Export

#### Bulk Delete Tab

**Delete by Document IDs:**
1. Enter document IDs (one per line) in text area
2. Click "🗑️ Delete Selected Documents"
3. Confirm deletion

**Delete by Tags:**
1. Enter tags (comma-separated)
2. System shows how many documents will be deleted
3. Click "⚠️ Confirm Delete"

#### Bulk Re-tag Tab

**Select Documents:**
- By Document IDs (one per line)
- By Existing Tags (finds all documents with those tags)

**Operations:**
- **Add Tags** - Adds new tags while preserving existing ones
- **Remove Tags** - Removes specified tags
- **Replace All Tags** - Replaces entire tag set

**Example:**
1. Select "Existing Tags" method
2. Enter "draft" to find all draft documents
3. Choose "Add Tags" operation
4. Enter "reviewed, approved"
5. Click "➕ Add Tags"
6. All draft documents now have "reviewed" and "approved" tags added

#### Bulk Export Tab

**Export Options:**
- **All Documents** - Export entire knowledge base
- **Documents with Tags** - Filter by specific tags
- **Specific Documents** - Export selected document IDs

**Formats:**
- **JSON** - Machine-readable, includes all metadata
- **CSV** - Spreadsheet format, easy import to Excel
- **Markdown** - Human-readable, formatted document list

**Example:**
1. Select "Documents with Tags"
2. Enter "reference, c64"
3. Choose "CSV" format
4. Click "📤 Export by Tags"
5. Download CSV file

### Via Python API

#### Bulk Tag Update

```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Add tags to specific documents
results = kb.update_tags_bulk(
    doc_ids=["doc_id_1", "doc_id_2"],
    add_tags=["assembly", "advanced"]
)

# Remove tags from documents with certain tags
results = kb.update_tags_bulk(
    existing_tags=["draft", "pending"],
    remove_tags=["draft"],
    add_tags=["reviewed"]
)

# Replace all tags for specific documents
results = kb.update_tags_bulk(
    doc_ids=["doc_id_1"],
    replace_tags=["archive", "old"]
)

print(f"Updated: {len(results['updated'])}")
print(f"Failed: {len(results['failed'])}")

kb.close()
```

#### Bulk Export

```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Export all documents as JSON
json_data = kb.export_documents_bulk(format="json")

# Export documents with specific tags as CSV
csv_data = kb.export_documents_bulk(
    tags=["reference", "c64"],
    format="csv"
)

# Export specific documents as Markdown
markdown_data = kb.export_documents_bulk(
    doc_ids=["doc1", "doc2", "doc3"],
    format="markdown"
)

# Save to file
with open("documents_export.json", "w") as f:
    f.write(json_data)

kb.close()
```

#### Bulk Delete

```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Remove documents by IDs
results = kb.remove_documents_bulk(
    doc_ids=["doc1", "doc2", "doc3"]
)

# Remove all documents with certain tags
results = kb.remove_documents_bulk(
    tags=["draft", "old", "archive"]
)

print(f"Removed: {len(results['removed'])}")
print(f"Failed: {len(results['failed'])}")

kb.close()
```

### Via CLI

#### Bulk Remove

```bash
# Remove specific documents
.venv\Scripts\python.exe cli.py remove-bulk --doc-ids doc1 doc2 doc3

# Remove all documents with tags
.venv\Scripts\python.exe cli.py remove-bulk --tags draft old
```

#### Bulk Tag Update

```bash
# Add tags to specific documents
.venv\Scripts\python.exe cli.py update-tags-bulk --doc-ids doc1 doc2 --add reviewed approved

# Remove tags from documents with existing tags
.venv\Scripts\python.exe cli.py update-tags-bulk --existing-tags draft --remove draft --add completed

# Replace all tags
.venv\Scripts\python.exe cli.py update-tags-bulk --doc-ids doc1 --replace archive legacy
```

#### Bulk Export

```bash
# Export all documents as JSON (stdout)
.venv\Scripts\python.exe cli.py export-bulk --format json

# Export to file
.venv\Scripts\python.exe cli.py export-bulk --format json --output documents.json

# Export documents with specific tags as CSV
.venv\Scripts\python.exe cli.py export-bulk --tags reference c64 --format csv --output c64_docs.csv

# Export specific documents as Markdown
.venv\Scripts\python.exe cli.py export-bulk --doc-ids doc1 doc2 --format markdown --output docs.md
```

### Via Claude Desktop (MCP)

Ask Claude to perform bulk operations:

**Bulk Tag Updates:**
- "Update all documents tagged 'draft' to remove the draft tag and add 'reviewed'"
- "Add the 'assembly' tag to documents doc1, doc2, and doc3"
- "Replace all tags on document doc1 with 'archive' and 'legacy'"

**Bulk Export:**
- "Export all documents in the knowledge base as JSON"
- "Export all documents tagged 'reference' as CSV"
- "Export documents doc1, doc2, doc3 as Markdown"

**Bulk Delete:**
- "Remove all documents tagged 'old' and 'archive'"
- "Delete documents doc1, doc2, and doc3"

### Use Cases

**Workflow Example 1: Review Process**
```python
# 1. Find all pending documents
results = kb.update_tags_bulk(
    existing_tags=["pending"],
    add_tags=["under-review"]
)

# 2. After review, mark as approved
results = kb.update_tags_bulk(
    existing_tags=["under-review"],
    remove_tags=["under-review", "pending"],
    add_tags=["approved"]
)

# 3. Archive old approved documents
results = kb.update_tags_bulk(
    existing_tags=["approved"],
    add_tags=["archive"]
)
```

**Workflow Example 2: Content Reorganization**
```python
# 1. Export current state for backup
backup_data = kb.export_documents_bulk(format="json")
with open("before_reorganization.json", "w") as f:
    f.write(backup_data)

# 2. Reorganize tags
kb.update_tags_bulk(existing_tags=["basic"], replace_tags=["programming", "beginner"])
kb.update_tags_bulk(existing_tags=["assembly"], replace_tags=["programming", "advanced"])
kb.update_tags_bulk(existing_tags=["hardware"], add_tags=["technical", "reference"])

# 3. Export new state
after_data = kb.export_documents_bulk(format="json")
with open("after_reorganization.json", "w") as f:
    f.write(after_data)
```

**Workflow Example 3: Cleanup**
```python
# 1. Export documents before deletion (safety measure)
to_delete = kb.export_documents_bulk(tags=["old", "draft"], format="json")
with open("deleted_documents_backup.json", "w") as f:
    f.write(to_delete)

# 2. Delete old and draft documents
results = kb.remove_documents_bulk(tags=["old", "draft"])
print(f"Deleted {len(results['removed'])} documents")
```

### Best Practices

**Safety:**
- Always export or backup before bulk deletions
- Use tag-based operations for flexibility
- Test on a few documents first
- Review results (updated/failed counts)

**Efficiency:**
- Use bulk operations instead of loops
- Combine add/remove in single operation
- Export regularly for reporting
- Use tags consistently for easier management

**Organization:**
- Maintain a standard tag taxonomy
- Document tag meanings
- Use hierarchical tags (e.g., "programming:basic", "programming:advanced")
- Regular tag cleanup and consolidation

---

## Backup & Restore

### Overview

The backup and restore system provides production-ready data protection for your knowledge base. All operations can be performed via the GUI, Claude Desktop (MCP), Python API, or command line.

### Creating Backups

#### Via GUI (Easiest)

1. Open admin interface: `streamlit run admin_gui.py`
2. Navigate to "💾 Backup & Restore" page
3. Enter backup directory path
4. Enable "Compress to ZIP" (recommended)
5. Click "🔄 Create Backup"

#### Via Python API

```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Create compressed backup
backup_path = kb.create_backup("/path/to/backups", compress=True)
print(f"Backup created: {backup_path}")

# Create uncompressed backup
backup_path = kb.create_backup("/path/to/backups", compress=False)

kb.close()
```

#### Via Claude Desktop (MCP)

Ask Claude:
```
"Create a backup of the C64 knowledge base to C:\backups\c64kb"
"Make a compressed backup of the knowledge base"
```

#### Via CLI (Not implemented yet, use Python API)

### Backup Contents

Each backup includes:

**1. Database File** (`knowledge_base.db`)
- All documents metadata
- All chunks
- FTS5 index (if enabled)
- Indexes and triggers

**2. Embeddings Files** (if semantic search enabled)
- `embeddings.faiss` - FAISS vector index
- `embeddings_map.json` - Chunk ID mapping

**3. Metadata File** (`metadata.json`)
```json
{
  "timestamp": "20251213_103045",
  "created_at": "2025-12-13T10:30:45",
  "document_count": 145,
  "total_chunks": 4665,
  "database_size_bytes": 125829120,
  "has_embeddings": true,
  "version": "2.5.0"
}
```

### Backup Naming

**Compressed (ZIP):**
- Format: `kb_backup_YYYYMMDD_HHMMSS.zip`
- Example: `kb_backup_20251213_103045.zip`

**Uncompressed (Directory):**
- Format: `kb_backup_YYYYMMDD_HHMMSS/`
- Example: `kb_backup_20251213_103045/`

### Restoring Backups

#### Via GUI (Easiest)

1. Open admin interface: `streamlit run admin_gui.py`
2. Navigate to "💾 Backup & Restore" page
3. Read warning: **Current database will be replaced**
4. Enter backup path (ZIP file or directory)
5. Enable "Verify backup before restoring" (recommended)
6. Click "⚠️ Restore Backup"
7. Wait for confirmation and GUI reload

#### Via Python API

```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Restore from ZIP with verification
result = kb.restore_from_backup("/path/to/kb_backup_20251213_103045.zip", verify=True)

print(f"Restored {result['restored_documents']} documents")
print(f"Elapsed time: {result['elapsed_seconds']:.2f}s")
print(f"Backup metadata: {result['backup_metadata']}")

kb.close()
```

#### Via Claude Desktop (MCP)

Ask Claude:
```
"Restore the knowledge base from C:\backups\kb_backup_20251213_103045.zip"
"Restore from backup with verification"
```

### Safety Features

**1. Automatic Safety Backup**
- Before restoring, current database is backed up to:
  - `knowledge_base_pre_restore_<timestamp>.db`
- Allows rollback if restoration fails
- Located in data directory

**2. Backup Verification**
- Checks for required files (database, metadata)
- Validates metadata.json structure
- Ensures backup is complete before proceeding
- Can be disabled with `verify=False` (not recommended)

**3. Extraction Handling**
- Automatically extracts ZIP archives to temporary directory
- Cleans up temporary files after restoration
- Handles both ZIP and directory backups

### Best Practices

**Backup Schedule:**
- **Daily:** During active development
- **Weekly:** For production use
- **Before major changes:** Adding bulk documents, schema updates
- **Before version upgrades:** Test restoration on dev environment first

**Storage Recommendations:**
- **Location:** Store backups on different drive/device than data directory
- **Retention:** Keep 7 daily + 4 weekly + 3 monthly backups
- **Compression:** Always enable (saves 50-70% disk space)
- **Verification:** Always verify backups (default: enabled)

**Testing Restoration:**
```bash
# Test restore on a different data directory
export TDZ_DATA_DIR=/tmp/test-restore
python -c "
from server import KnowledgeBase
kb = KnowledgeBase()
result = kb.restore_from_backup('/path/to/backup.zip')
print(f'Test restore successful: {result}')
kb.close()
"
```

**Automation:**

Windows Task Scheduler:
```cmd
# Create batch file: backup_c64kb.bat
@echo off
set TDZ_DATA_DIR=C:\Users\YourName\.tdz-c64-knowledge
cd C:\Users\mit\claude\c64server\tdz-c64-knowledge
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb.create_backup('C:\\backups\\c64kb', compress=True); kb.close()"
```

Schedule this script to run daily/weekly via Task Scheduler.

### Troubleshooting

**Backup Creation Fails:**
```
Error: Permission denied
```
**Solution:** Check destination directory permissions, ensure write access

```
Error: No space left on device
```
**Solution:** Free up disk space or use different destination directory

**Restore Fails:**
```
ValueError: Backup is missing database file
```
**Solution:** Backup is corrupted or incomplete, use different backup

```
Error: Backup is missing metadata file
```
**Solution:** Backup is from older version, missing metadata.json (can skip verification)

**After Restore:**
- Verify document count matches backup metadata
- Test search functionality (all search modes)
- Check embeddings loaded (if semantic search enabled)
- Review logs for any warnings

### Backup Size Estimates

Based on current knowledge base (145 docs, 4,665 chunks):

| Component | Uncompressed | Compressed (ZIP) |
|-----------|--------------|------------------|
| Database  | 120 MB       | 35 MB            |
| Embeddings| 15 MB        | 8 MB             |
| Metadata  | 1 KB         | 1 KB             |
| **Total** | **135 MB**   | **43 MB**        |

Compression saves ~68% disk space.

---

## Claude Desktop Integration

### Configuration

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\Users\\mit/.tdz-c64-knowledge",
        "USE_FTS5": "1",
        "USE_SEMANTIC_SEARCH": "1",
        "SEMANTIC_MODEL": "all-MiniLM-L6-v2",
        "USE_OCR": "1",
        "POPPLER_PATH": "C:\\Users\\mit\\claude\\c64server\\tdz-c64-knowledge\\poppler-25.12.0\\Library\\bin"
      }
    }
  }
}
```

### Usage in Claude Desktop

Once configured, you can ask Claude:

**Hybrid Searches** ⭐ NEW (Recommended):
- "Use hybrid search to find information about sound programming on the C64"
- "Search for sprite collision detection using hybrid search with semantic weight 0.4"
- "Find VIC-II graphics programming guides with hybrid search"

**Keyword Searches:**
- "Search the C64 knowledge base for sprite collision detection"
- "Find information about SID register $D400"
- "What's in the VIC-II documentation?"

**Semantic Searches:**
- "How do I make sounds on the C64?"
- "What are the graphics capabilities?"
- "How does memory management work?"

**System Health** ⭐ NEW:
- "Check the health of the knowledge base system"
- "Show me system diagnostics and performance metrics"
- "Is everything working correctly with the knowledge base?"

**Document Management:**
- "List all documents in the knowledge base"
- "Show me documents tagged with 'assembly'"
- "Get statistics about the knowledge base"

---

## Command Line Usage

### Search

```bash
# Basic search
.venv\Scripts\python.exe cli.py search "your query" --max 10

# Filter by tags
.venv\Scripts\python.exe cli.py search "SID" --tags reference --max 5

# Search all tags
.venv\Scripts\python.exe cli.py search "6502" --tags reference c64 codebase64
```

### Add Documents

```bash
# Add single document
.venv\Scripts\python.exe cli.py add "path/to/doc.pdf" --title "Title" --tags tag1 tag2

# Add folder (recursive)
.venv\Scripts\python.exe cli.py add-folder "docs" --tags reference --recursive

# Add with specific tags
.venv\Scripts\python.exe cli.py add-folder "assembly-guides" --tags assembly reference --recursive
```

### List and Stats

```bash
# List all documents
.venv\Scripts\python.exe cli.py list

# Show statistics
.venv\Scripts\python.exe cli.py stats

# Remove document
.venv\Scripts\python.exe cli.py remove <doc_id>
```

---

## Performance Optimization

### FTS5 Setup

**Initial Setup:**
```bash
.venv\Scripts\python.exe enable_fts5.py
```

**Performance Test Results:**
- SID chip search: 68,000ms → 137ms (496x faster)
- VIC-II sprites: 66,000ms → 66ms (1000x faster)
- 6502 assembly: 66,000ms → 58ms (1138x faster)

**How It Works:**
- Creates SQLite FTS5 virtual table on first search
- Automatic triggers keep index synchronized
- Native BM25 ranking
- Porter stemmer for word normalization

### Semantic Search Setup

**Initial Setup:**
```bash
# Install dependencies (if not already installed)
pip install sentence-transformers faiss-cpu

# Build embeddings index
.venv\Scripts\python.exe enable_semantic_search.py
```

**Performance:**
- Query processing: 12-25ms
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Index size: ~2,347 vectors for 4,665 chunks
- Persistent cache: Saved to disk for fast startup

**Customization:**
```bash
# Use different model
export SEMANTIC_MODEL=all-mpnet-base-v2

# Higher quality (slower)
export SEMANTIC_MODEL=multi-qa-mpnet-base-dot-v1
```

### Environment Variables

```bash
# Core settings
TDZ_DATA_DIR=C:\Users\mit/.tdz-c64-knowledge  # Data directory
USE_FTS5=1                                    # Enable FTS5 (recommended)
USE_SEMANTIC_SEARCH=1                         # Enable semantic search
SEMANTIC_MODEL=all-MiniLM-L6-v2              # Embedding model

# BM25 settings
USE_BM25=1                                    # Enable BM25 (fallback)
USE_QUERY_PREPROCESSING=1                     # Enable stemming/stopwords

# OCR settings (for scanned PDFs)
USE_OCR=1                                     # Enable OCR
POPPLER_PATH=C:\...\poppler-25.12.0\Library\bin

# Security
ALLOWED_DOCS_DIRS=C:\Users\...\C64Docs        # Whitelist directories
```

---

## Troubleshooting

### Search is Slow (>1 second)

**Problem:** FTS5 not enabled or index not built

**Solution:**
```bash
# Enable FTS5
.venv\Scripts\python.exe enable_fts5.py

# Verify in logs
# Should see: "Search completed: 5 results in 50-150ms"
```

### Semantic Search Not Working

**Problem:** Dependencies missing or embeddings not built

**Solution:**
```bash
# Install dependencies
pip install sentence-transformers faiss-cpu

# Build embeddings
.venv\Scripts\python.exe enable_semantic_search.py

# Check for embeddings files
dir C:\Users\mit\.tdz-c64-knowledge\embeddings*
```

### OCR Failing on Scanned PDFs

**Problem:** Poppler not configured

**Solution:**
```bash
# Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases
# Extract to project directory
# Set environment variable
export POPPLER_PATH=C:\...\poppler-25.12.0\Library\bin
```

### Duplicate Documents

**Problem:** Same content indexed multiple times

**Solution:**
The knowledge base automatically detects duplicates using content-based hashing. Duplicates are skipped during import. No action needed.

### MCP Server Not Connecting

**Problem:** Claude Desktop can't find the server

**Solution:**
1. Check config path: `%APPDATA%\Claude\claude_desktop_config.json`
2. Verify Python path is absolute (not relative)
3. Check logs in Claude Desktop developer console
4. Test server manually:
   ```bash
   .venv\Scripts\python.exe server.py
   ```

### Search Returns No Results

**Problem:** Query too specific or typos

**Solutions:**
1. Use broader terms: "SID chip" instead of "SID $D400-$D41C"
2. Enable fuzzy search (already enabled by default)
3. Try semantic search for conceptual queries
4. Check available tags: `.venv\Scripts\python.exe cli.py stats`

---

## Tips and Best Practices

### Search Tips
1. **Use FTS5 for technical terms** - Register names, chip names, keywords
2. **Use semantic search for questions** - "How do I...", "What is..."
3. **Combine with tags** - Narrow results to specific topics
4. **Use quotes for phrases** - "sprite collision detection"
5. **Try variations** - "VIC-II" vs "VIC II" vs "VIC 2"

### Document Management
1. **Tag consistently** - Use standard tags (reference, c64, assembly, etc.)
2. **Use descriptive titles** - Helps with search and organization
3. **Organize by topic** - Group related docs with same tags
4. **Check for duplicates** - Use `.venv\Scripts\python.exe cli.py list`

### Performance
1. **Keep FTS5 enabled** - Massive speed improvement
2. **Use semantic search sparingly** - Great for exploration, FTS5 for precision
3. **Cache is your friend** - Repeat searches are instant (5 min TTL)
4. **Monitor stats** - Track knowledge base growth

---

## Advanced Usage

### Python API

```python
from server import KnowledgeBase
import os

# Initialize
os.environ['USE_FTS5'] = '1'
os.environ['USE_SEMANTIC_SEARCH'] = '1'
kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Search
results = kb.search("VIC-II sprites", max_results=10, tags=["reference"])

# Semantic search
semantic_results = kb.semantic_search("How do I create graphics?", max_results=5)

# Find similar documents
similar = kb.find_similar_documents(doc_id="abc123", max_results=5)

# Get specific chunk
chunk = kb.get_chunk(doc_id="abc123", chunk_id=0)

# Stats
stats = {
    "documents": len(kb.documents),
    "chunks": len(kb._get_chunks_db()),
    "file_types": set(doc.file_type for doc in kb.documents.values())
}

# Cleanup
kb.close()
```

### Batch Operations

```bash
# Import multiple folders
for dir in pdf txt codebase64; do
  .venv\Scripts\python.exe cli.py add-folder "$dir" --tags reference c64 --recursive
done

# Bulk tag update (requires custom script)
# Search and re-tag specific documents

# Export document list
.venv\Scripts\python.exe cli.py list > documents.txt
```

---

## Resources

- **CLAUDE.md** - Developer documentation and architecture
- **POPPLER_SETUP.md** - OCR configuration guide
- **QUICK_START.md** - Production deployment guide
- **README.md** - Project overview

## Support

For issues or questions:
1. Check logs: `server.log` in project directory
2. Review CLAUDE.md for technical details
3. Test with `enable_fts5.py` and `enable_semantic_search.py`
4. Check Claude Desktop console for MCP errors
