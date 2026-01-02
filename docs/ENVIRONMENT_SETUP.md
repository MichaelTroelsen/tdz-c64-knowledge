# TDZ C64 Knowledge Base - Environment Setup Guide

**Status:** ✅ All optional features installed and configured

## 📋 Quick Start

### Option 1: Use Launcher Batch Files (Easiest)

```bash
# GUI with all features enabled
launch-gui-full-features.bat

# CLI with all features enabled
launch-cli-full-features.bat search "VIC-II sprites"

# MCP Server with all features enabled
launch-server-full-features.bat
```

### Option 2: Set Environment Variables Manually

```bash
# Command-line (Windows)
set USE_FTS5=1
set USE_SEMANTIC_SEARCH=1
set USE_BM25=1
set USE_QUERY_PREPROCESSING=1
set USE_FUZZY_SEARCH=1

# Then run your command
python cli.py search "query"
```

### Option 3: Use .env File (For IDE Integration)

The `.env` file has been created with all recommended settings. However, Python doesn't automatically load it - you need to use a library like `python-dotenv` or set variables manually.

---

## 🚀 Features Enabled

All of the following features are installed and ready to use:

| Feature | Status | Command to Enable |
|---------|--------|-------------------|
| **FTS5 Full-Text Search** | ✅ | `USE_FTS5=1` |
| **Semantic Search** | ✅ | `USE_SEMANTIC_SEARCH=1` |
| **BM25 Ranking** | ✅ | `USE_BM25=1` |
| **Query Preprocessing** | ✅ | `USE_QUERY_PREPROCESSING=1` |
| **Fuzzy Search (Typo Tolerance)** | ✅ | `USE_FUZZY_SEARCH=1` |
| **Table Extraction** | ✅ | (Automatic when pdfplumber installed) |
| **OCR Support** | ✅ | `USE_OCR=1` |
| **Search Caching** | ✅ | (Default: 100 entries, 5-min TTL) |

---

## 🔧 Environment Variables Reference

### Search Configuration

```bash
# Enable/disable search algorithms
USE_FTS5=1                    # SQLite FTS5 (480x faster than BM25)
USE_BM25=1                    # BM25 algorithm (fallback)
USE_QUERY_PREPROCESSING=1     # NLTK stemming + stopword removal
USE_FUZZY_SEARCH=1            # Typo tolerance (requires rapidfuzz)

# Semantic search (requires sentence-transformers + faiss-cpu)
USE_SEMANTIC_SEARCH=1         # Enable embeddings-based search
SEMANTIC_MODEL=all-MiniLM-L6-v2  # Default model

# Fuzzy search parameters
FUZZY_THRESHOLD=80            # Similarity threshold (0-100)

# Search cache
SEARCH_CACHE_SIZE=100         # Number of cached results
SEARCH_CACHE_TTL=300          # Cache expiry time (seconds)
```

### Data Storage

```bash
TDZ_DATA_DIR=C:\Users\YourName\.tdz-c64-knowledge
```

### LLM Integration

```bash
LLM_PROVIDER=anthropic        # or 'openai'
LLM_MODEL=claude-3-haiku-20240307
ANTHROPIC_API_KEY=sk-ant-...  # Optional (for RAG features)
OPENAI_API_KEY=sk-...         # Optional (for OpenAI models)
```

### OCR & Document Processing

```bash
USE_OCR=1                     # Enable OCR for scanned PDFs
POPPLER_PATH=C:\path\to\poppler\bin  # Optional (for scanned PDFs)
```

### Security

```bash
# Comma-separated list of allowed document directories
ALLOWED_DOCS_DIRS=C:\docs\allowed,C:\research\docs
```

---

## 📊 Performance Impact

| Feature | Impact | Notes |
|---------|--------|-------|
| **FTS5** | 480x faster search | ~50ms vs 24,000ms for BM25 |
| **Semantic Search** | 7-16ms per query | Requires embeddings generation (~1 min one-time) |
| **Fuzzy Matching** | +10-20% latency | Provides typo tolerance |
| **Query Preprocessing** | +5-10% latency | Improves relevance |
| **Search Caching** | 50-100x faster | For repeated queries |

---

## 🎯 Recommended Configuration

**For best performance**, enable:

```bash
USE_FTS5=1                    # Fastest search (native SQLite)
USE_SEMANTIC_SEARCH=1         # Conceptual understanding
USE_BM25=1                    # Fallback if FTS5 returns no results
USE_QUERY_PREPROCESSING=1     # Better relevance
USE_FUZZY_SEARCH=1            # User-friendly typo tolerance
```

This combination provides:
- Fast keyword search (FTS5)
- Conceptual understanding (Semantic)
- Graceful degradation (BM25 fallback)
- Better matching (Preprocessing + Fuzzy)

---

## 🔌 Integration with Claude Code/Claude Desktop

To use with Claude Code's MCP system:

### Step 1: Add MCP Configuration

Create/edit your Claude Desktop config.json:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\Users\\YourName\\claude\\c64server\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\YourName\\claude\\c64server\\tdz-c64-knowledge\\server.py"],
      "env": {
        "USE_FTS5": "1",
        "USE_SEMANTIC_SEARCH": "1",
        "USE_BM25": "1",
        "USE_QUERY_PREPROCESSING": "1",
        "USE_FUZZY_SEARCH": "1",
        "USE_OCR": "1",
        "SEARCH_CACHE_SIZE": "100",
        "SEARCH_CACHE_TTL": "300"
      }
    }
  }
}
```

### Step 2: Restart Claude Desktop

Restart Claude Desktop to load the new configuration.

---

## 📈 Optional: Enable RAG Question Answering

To use the upcoming RAG feature, add your API key:

```bash
ANTHROPIC_API_KEY=sk-ant-...    # For Claude models
# OR
OPENAI_API_KEY=sk-...           # For GPT models
```

---

## ✅ Verification Checklist

- [x] Python 3.10+ installed
- [x] Virtual environment created (.venv)
- [x] Core dependencies installed (mcp, pypdf, rank-bm25, nltk, cachetools)
- [x] Optional features installed:
  - [x] pdfplumber (table extraction)
  - [x] sentence-transformers (semantic search)
  - [x] faiss-cpu (vector search)
  - [x] pytesseract (OCR)
  - [x] pdf2image (PDF to images)
  - [x] rapidfuzz (fuzzy matching)
  - [x] streamlit (GUI)
  - [x] pandas, pyvis, networkx (GUI enhancements)
- [x] .env file created with settings
- [x] Launcher batch files created

---

## 🆘 Troubleshooting

### Semantic Search Not Enabled

**Problem:** Logs show "Semantic search disabled"

**Solution:** Set `USE_SEMANTIC_SEARCH=1` in environment before running:
```bash
set USE_SEMANTIC_SEARCH=1
python cli.py search "query"
```

### FTS5 Search Not Enabled

**Problem:** "FTS5 is disabled"

**Solution:** Ensure SQLite has FTS5 support and set:
```bash
set USE_FTS5=1
```

### OCR Not Working

**Problem:** "Poppler not found" warning

**Solution (optional):** Download Poppler for Windows:
1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
2. Extract to C:\poppler
3. Set: `POPPLER_PATH=C:\poppler\Library\bin`

### Knowledge Base Not Loading

**Problem:** Error initializing knowledge base

**Solution:**
1. Check `TDZ_DATA_DIR` path exists
2. Ensure database file is not corrupted
3. Check file permissions

---

## 📚 Knowledge Base Stats

- **Documents:** 147
- **Chunks:** 4,678
- **Total Words:** ~6.9 million
- **File Types:** PDF, TXT, MD, HTML, XLSX

---

## 🎓 Next Steps

1. ✅ **Environment Setup** - DONE
2. ⏭️ **RAG Question Answering** - Next major feature
3. ⏭️ **Document Summarization** - Auto-generate summaries
4. ⏭️ **Entity Extraction** - Extract names, dates, technical terms
5. ⏭️ **VICE Emulator Integration** - Link to running C64 emulator

---

**Last Updated:** 2025-12-17
**Version:** 2.12.0
