# TDZ C64 Knowledge Base - Comprehensive Documentation

[![Version](https://img.shields.io/badge/version-2.12.0-brightgreen.svg)](https://github.com/MichaelTroelsen/tdz-c64-knowledge)
[![Status](https://img.shields.io/badge/status-fully%20configured-success.svg)](#status)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced Commodore 64/128 Documentation Search Engine with AI Integration**

A production-ready MCP (Model Context Protocol) server for managing and searching comprehensive Commodore 64/128 technical documentation. Features advanced search capabilities, semantic understanding, automatic table/code extraction, and OCR support.

---

## 🎯 Current Status

✅ **Fully Configured & Operational**

- **Knowledge Base:** 148+ documents | 4,680+ chunks | 6.9M+ words
- **Search Features:** 480x faster FTS5 | Semantic search | Fuzzy matching
- **Document Support:** PDF | TXT | Markdown | HTML | Excel
- **Advanced Features:** Table extraction | Code detection | OCR | Auto-tagging
- **UI:** Web-based Streamlit dashboard at http://localhost:8501

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Use GUI (Easiest)
```bash
# Streamlit web interface
launch-gui-full-features.bat

# Access: http://localhost:8501
```

### Option 2: Use CLI
```bash
# Search via command line
launch-cli-full-features.bat search "VIC-II sprites" --max 5

# List documents
launch-cli-full-features.bat list

# Add new documents
launch-cli-full-features.bat add-folder "C:\path\to\docs" --tags reference --recursive
```

### Option 3: Use with Claude Code
```bash
# Add to Claude Code MCP
claude mcp add tdz-c64-knowledge -- ".venv\Scripts\python.exe" "server.py"

# Then ask Claude: "Search the C64 docs for assembly programming"
```

---

## 📊 Feature Overview

### ⚡ Search Capabilities

| Feature | Performance | Status |
|---------|-------------|--------|
| **FTS5 Full-Text Search** | 50ms (480x faster) | ✅ Enabled |
| **Semantic/Conceptual Search** | 7-16ms per query | ✅ Installed |
| **Fuzzy Search (Typo Tolerance)** | Adjustable threshold | ✅ Enabled (80%) |
| **BM25 Ranking Algorithm** | Fallback method | ✅ Enabled |
| **Query Preprocessing** | Stemming + stopwords | ✅ Enabled |
| **Search Result Caching** | 50-100x speedup | ✅ Enabled |

**Example Queries That Work:**
- `"SID chip sound"` - Keyword search
- `"how to program sprites"` - Conceptual search
- `"registr"` - Fuzzy search (finds "register")
- `"memory map"` - Phrase search
- Search with tags: `vid-ii` tag filter

### 📄 Document Processing

- ✅ **PDF Extraction** - Text, tables, metadata
- ✅ **Table Detection** - Automated extraction from PDFs (209+ tables indexed)
- ✅ **Code Block Detection** - BASIC, Assembly, Hex dumps (1,876+ blocks indexed)
- ✅ **Facet Extraction** - Hardware components, instructions, registers
- ✅ **Duplicate Detection** - Content-based deduplication
- ✅ **OCR Support** - Tesseract for scanned PDFs (Poppler optional)
- ✅ **Multi-Format Support** - PDF, TXT, MD, HTML, XLSX

### 🎨 User Interfaces

1. **Web GUI (Streamlit)**
   - Search interface with filters
   - Document browser
   - Tag management
   - Analytics dashboard
   - Settings configuration
   - URL: http://localhost:8501

2. **Command-Line Interface (CLI)**
   - Batch operations
   - Scripting support
   - Advanced queries
   - Document management

3. **MCP Server**
   - Integration with Claude Code/Desktop
   - Programmatic API
   - Tool-based access

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.10+**
- **~2GB free disk space** (for database + indices)
- **Optional:** Poppler (for scanned PDF OCR)

### Step 1: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# Core dependencies (already installed)
pip install mcp pypdf rank-bm25 nltk cachetools openpyxl beautifulsoup4 chardet

# Optional features (already installed)
pip install pdfplumber sentence-transformers faiss-cpu pytesseract pdf2image rapidfuzz streamlit pandas pyvis networkx
```

### Step 3: Run Your Preferred Interface

**GUI:**
```bash
launch-gui-full-features.bat
```

**CLI:**
```bash
launch-cli-full-features.bat search "query"
```

**MCP Server:**
```bash
launch-server-full-features.bat
```

---

## 📝 AI-Powered Summarization

**Status:** ✅ New in v2.13.0 - Fully Implemented

Generate intelligent AI summaries of C64 documentation with three summary types:

**Features:**
- **Brief:** 200-300 word overviews (1-2 paragraphs)
- **Detailed:** 500-800 word comprehensive summaries (3-5 paragraphs)
- **Bullet Points:** 8-12 key topics in bullet format

**Quick Start:**
```bash
# Generate brief summary for a document
launch-cli-full-features.bat summarize "doc-id"

# Detailed summary
launch-cli-full-features.bat summarize "doc-id" --type detailed

# Generate for all documents
launch-cli-full-features.bat summarize-all --types brief detailed

# MCP Tool (in Claude Desktop)
User: "Summarize document 'assembly-guide' with detailed summary"
Claude: (uses summarize_document tool)
```

**Benefits:**
- Intelligent caching for instant retrieval
- Bulk processing for entire knowledge base
- Supports Claude or GPT models
- Seamless integration with Claude Desktop
- Multi-level detail options for different needs

**Requirements:**
- LLM API key (Anthropic Claude or OpenAI GPT)
- See [SUMMARIZATION.md](SUMMARIZATION.md) for complete guide

---

## ⚙️ Configuration

All features are pre-configured. To customize:

### Environment Variables
Create or edit `.env` file:
```bash
# Search Configuration
USE_FTS5=1                    # Enable FTS5 (fastest)
USE_SEMANTIC_SEARCH=1         # Enable semantic search
USE_BM25=1                    # Enable BM25 fallback
USE_QUERY_PREPROCESSING=1     # Enable stemming
USE_FUZZY_SEARCH=1            # Enable typo tolerance

# Performance
SEARCH_CACHE_SIZE=100         # Cache size
SEARCH_CACHE_TTL=300          # Cache expiry (seconds)

# Data Storage
TDZ_DATA_DIR=~/.tdz-c64-knowledge

# LLM Integration (optional)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku-20240307
ANTHROPIC_API_KEY=sk-ant-...
```

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for complete reference.

---

## 🎯 Use Cases

### 1. Research & Documentation
```bash
# Find all VIC-II related documentation
launch-cli-full-features.bat search "VIC-II" --tags vic-ii --max 20
```

### 2. Programming Reference
```bash
# Find assembly programming examples
launch-cli-full-features.bat search "6502 assembly" --max 10
```

### 3. Hardware Understanding
```bash
# Learn about SID chip programming
launch-cli-full-features.bat search "SID sound synthesis" --max 5
```

### 4. Integration with Claude AI
```
Ask Claude: "Based on the C64 knowledge base,
how do I program sprite collision detection?"
```

---

## 🔍 Search Examples

### Exact Phrase Search
```
"memory map" "I/O control"
```

### Tag-Based Search
```
search "SID programming" --tags sid music
```

### Fuzzy/Typo-Tolerant Search
```
search "sprits" --max 10
# Finds: "sprites" (fuzzy match)
```

### Semantic Conceptual Search
```
search "how to make sounds"
# Finds: SID chip documentation
```

---

## 📈 Knowledge Base Statistics

**Current Status (as of 2025-12-17):**

| Metric | Value |
|--------|-------|
| **Total Documents** | 148+ |
| **Total Chunks** | 4,680+ |
| **Total Words** | 6.9M+ |
| **Code Blocks** | 1,876+ (BASIC/Assembly/Hex) |
| **Extracted Tables** | 209+ |
| **Facets (Hardware/Instructions/Registers)** | 1,876+ |
| **Average Search Time** | 50-100ms |
| **File Types** | PDF, TXT, MD, HTML, XLSX |

**Document Categories:**
- Commodore 64 Programmers References
- Assembly Language Guides
- Machine Language Programming
- Sound & Graphics Programming (SID/VIC-II)
- System Internals
- Diagnostic Manuals
- "Compute! Hacking" Magazine Articles
- C128 Documentation

---

## 🛠️ Architecture

### Core Components

1. **KnowledgeBase Class** (`server.py`)
   - Document ingestion
   - Search algorithms
   - Database management
   - Caching layer

2. **SQLite Database** (`knowledge_base.db`)
   - Documents table (metadata)
   - Chunks table (content)
   - Tables table (extracted tables)
   - Code blocks table (detected code)
   - FTS5 indexes (full-text search)

3. **Search Engine**
   - FTS5: Native SQLite search (fastest)
   - Semantic: Sentence-transformers + FAISS
   - BM25: Fallback ranking algorithm

4. **User Interfaces**
   - Streamlit GUI (web-based)
   - CLI (command-line)
   - MCP Server (programmatic)

### Data Flow
```
Input Files (PDF/TXT/etc)
    ↓
Text Extraction + Table Detection + Code Block Detection
    ↓
Chunking (1500-word chunks with 200-word overlap)
    ↓
Database Storage (SQLite with FTS5 indexes)
    ↓
Search Queries → FTS5/Semantic/BM25
    ↓
Results with Snippets, Page Numbers, Scores
```

---

## 📚 Available Tools (MCP)

### Search Tools
- `search_docs` - Full-text keyword search
- `semantic_search` - Conceptual/meaning-based search
- `hybrid_search` - Combined keyword + semantic
- `find_similar` - Find similar documents
- `search_tables` - Search extracted tables
- `search_code` - Search code blocks

### Management Tools
- `add_document` - Add single file
- `add_documents_bulk` - Batch import
- `remove_document` - Delete document
- `list_docs` - List all documents
- `get_document` - Retrieve full document

### Information Tools
- `kb_stats` - Database statistics
- `health_check` - System diagnostics
- `check_updates` - Find modified files

---

## 🚀 Next Steps

### Recommended Priorities

1. **RAG Question Answering** ⭐⭐⭐⭐⭐
   - Ask natural language questions
   - Get synthesized answers with citations
   - Integration with Claude API

2. **Document Summarization** ⭐⭐⭐⭐
   - Auto-generate document summaries
   - Improve discoverability
   - Quick overviews

3. **Entity Extraction** ⭐⭐⭐⭐
   - Extract names, dates, technical terms
   - Build knowledge graph
   - Historical research

4. **VICE Emulator Integration** ⭐⭐⭐⭐⭐
   - Link documentation to running emulator
   - Memory address lookup
   - Real-time programming assistance

### Optional Enhancements

- Poppler OCR for scanned PDFs
- Semantic search embeddings generation
- REST API server
- Multi-user support
- Plugin system

---

## 📝 Common Tasks

### Add More Documents
```bash
# Single file
launch-cli-full-features.bat add "path/to/file.pdf" --title "My Doc" --tags reference

# Entire folder
launch-cli-full-features.bat add-folder "C:\docs" --tags reference --recursive
```

### List All Documents
```bash
launch-cli-full-features.bat list
```

### Get Statistics
```bash
launch-cli-full-features.bat stats
```

### Remove Document
```bash
launch-cli-full-features.bat remove <doc_id>
```

---

## 🔐 Security

- **Path Traversal Protection** - Whitelist allowed directories
- **Content-Based Deduplication** - MD5 hashing prevents duplicates
- **ACID Transactions** - SQLite ensures consistency
- **Comprehensive Logging** - All operations logged to `server.log`

---

## 📊 Performance Tips

1. **Use FTS5 Search** - 480x faster than BM25
   ```bash
   set USE_FTS5=1
   ```

2. **Enable Caching** - Already enabled (100 entries, 5-min TTL)

3. **Use Exact Phrase Search** - When searching for specific terms
   ```bash
   search "VIC-II chip"  # vs  search VIC-II chip
   ```

4. **Filter by Tags** - Narrow results
   ```bash
   search "programming" --tags assembly --max 10
   ```

---

## 🐛 Troubleshooting

### Issue: Semantic search not working
**Solution:** Enable with `set USE_SEMANTIC_SEARCH=1`

### Issue: Search results empty
**Solution:** Try fuzzy search or reduce specificity

### Issue: Slow initial search
**Solution:** First search builds BM25 index (~50s). Subsequent searches use cache.

### Issue: Scanned PDFs not extracted
**Solution:** Optional - Install Poppler from https://github.com/oschwartz10612/poppler-windows

---

## 📚 Documentation Files

- **README.md** - This file (project overview)
- **QUICKSTART.md** - Getting started guide
- **ENVIRONMENT_SETUP.md** - Environment variables & configuration
- **CLAUDE.md** - Development guidelines
- **CHANGELOG.md** - Version history
- **EXAMPLES.md** - Code examples

---

## 🤝 Contributing

Ideas for enhancements:
1. RAG Question Answering system
2. Document summarization
3. Entity extraction
4. VICE emulator integration
5. REST API server
6. Mobile app frontend

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- Commodore 64 documentation authors and archivists
- Open-source projects: pypdf, NLTK, sentence-transformers, FAISS, SQLite
- Claude AI for MCP and API integration

---

**Ready to use! Start with:**
```bash
launch-gui-full-features.bat
# Open: http://localhost:8501
```

**Last Updated:** 2025-12-17
**Version:** 2.12.0
**Status:** ✅ Production Ready
