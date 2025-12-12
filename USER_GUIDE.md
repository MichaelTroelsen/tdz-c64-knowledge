# TDZ C64 Knowledge Base - User Guide

Complete guide for using the Commodore 64 documentation knowledge base with Claude Desktop.

## Table of Contents
- [What's New in v2.0.0](#whats-new-in-v200) ⭐
- [Quick Start](#quick-start)
- [Features](#features)
- [Search Modes](#search-modes)
- [Claude Desktop Integration](#claude-desktop-integration)
- [Command Line Usage](#command-line-usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

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
- **File Types:** PDF and Text
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
