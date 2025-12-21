# TDZ C64 Knowledge - Quick Start Guide

Get up and running with the C64 documentation knowledge base in 5 minutes!

## ⚡ Quick Install (Windows)

```cmd
# 1. Navigate to installation directory
cd C:\Users\YourName\mcp-servers

# 2. Clone or download this repository
git clone https://github.com/MichaelTroelsen/tdz-c64-knowledge.git
cd tdz-c64-knowledge

# 3. Create virtual environment and install
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 4. Test the server
python server.py
# Press Ctrl+C to stop
```

## 🔧 Configure Claude Code

Add to your Claude Code MCP settings:

```cmd
claude mcp add tdz-c64-knowledge -- C:\Users\YourName\mcp-servers\tdz-c64-knowledge\.venv\Scripts\python.exe C:\Users\YourName\mcp-servers\tdz-c64-knowledge\server.py
```

Or manually edit `.claude/settings.json`:

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

## 📚 Add Your First Documents

```cmd
# Using CLI (recommended for bulk imports)
.venv\Scripts\python.exe cli.py add-folder "C:\c64docs" --tags reference c64 --recursive

# Or via Claude Code
# Just ask: "Add C:/c64docs/programmers_reference.pdf to the knowledge base with tags reference, memory-map"
```

## 🔍 Search Examples

Ask Claude Code:
- "Search the C64 docs for VIC-II sprite registers"
- "Find information about SID chip sound synthesis"
- "What does the memory map say about $D000?"
- "Search for raster interrupts in graphics-related docs"

## ⚙️ Recommended Configuration

For best performance, enable these features in your MCP config:

```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data",
    "USE_FTS5": "1",
    "USE_FUZZY_SEARCH": "1",
    "USE_QUERY_PREPROCESSING": "1",
    "SEARCH_CACHE_SIZE": "100"
  }
}
```

## 🚀 Advanced Features (Optional)

### Enable Semantic Search (Meaning-Based Search)
```cmd
# Install dependencies
.venv\Scripts\python.exe -m pip install sentence-transformers faiss-cpu

# Pre-build embeddings index (takes 1-2 minutes)
.venv\Scripts\python.exe enable_semantic_search.py

# Add to your MCP config env section:
"USE_SEMANTIC_SEARCH": "1"
```

**What you get:**
- Search by meaning, not just keywords
- Example: "How do I create graphics?" finds VIC-II documentation
- ~7-16ms search speed after embeddings are built
- Particularly useful for natural language questions

### Enable AI-Powered Features
```cmd
# Set up your LLM provider (choose one)
# For Anthropic Claude:
set ANTHROPIC_API_KEY=sk-ant-...
set LLM_PROVIDER=anthropic

# For OpenAI GPT:
set OPENAI_API_KEY=sk-...
set LLM_PROVIDER=openai
```

**Available AI Features:**
- **Natural Language Query Translation** - "find sprite info on VIC-II" → structured search
- **Entity Extraction** - Extract hardware, memory addresses, instructions, concepts
- **Relationship Tracking** - Discover connections between entities
- **Document Summarization** - Generate AI summaries (brief, detailed, bullet-point)
- **Smart Auto-Tagging** - AI-powered tag suggestions

**Quick Examples:**
```cmd
# Extract entities from a document
.venv\Scripts\python.exe cli.py extract-entities <doc-id>

# Search for entities
.venv\Scripts\python.exe cli.py search-entity "VIC-II"

# Show entity relationships
.venv\Scripts\python.exe cli.py show-relationships "VIC-II" --max 10

# Translate natural language query
.venv\Scripts\python.exe cli.py translate-query "find sprite graphics info"
```

### Run the REST API Server
```cmd
# Install REST dependencies
.venv\Scripts\pip.exe install -e ".[rest]"

# Start the server
.venv\Scripts\python.exe -m uvicorn rest_server:app --reload --port 8000

# Or use the batch file (Windows)
run_rest_api.bat

# Access API documentation
# http://localhost:8000/api/docs
```

**What you get:**
- 27 HTTP endpoints for all knowledge base operations
- API key authentication for secure access
- OpenAPI/Swagger documentation
- Can run alongside MCP server
- See [README_REST_API.md](README_REST_API.md) for complete API docs

### Run the Web GUI (Streamlit)
```cmd
# Start the Streamlit admin interface
.venv\Scripts\python.exe -m streamlit run admin_gui.py

# Opens browser at http://localhost:8501
```

**What you get:**
- Document management (upload, delete, view)
- Interactive search with all modes (keyword, semantic, hybrid, faceted)
- Entity analytics dashboard with network graph visualization
- Document comparison tool
- Bulk operations and statistics
- Natural language query translation interface
- Entity relationship explorer

### Enable OCR for Scanned PDFs
```bash
pip install pytesseract pdf2image Pillow
# Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```
Add to config: `"USE_OCR": "1"`

## 📊 Verify Installation

```cmd
# Check stats
.venv\Scripts\python.exe cli.py stats

# Test search
.venv\Scripts\python.exe cli.py search "VIC-II" --max 3

# List documents
.venv\Scripts\python.exe cli.py list
```

## 🎯 Common Use Cases

### Bulk Import All PDFs
```cmd
.venv\Scripts\python.exe cli.py add-folder "C:\c64docs" --tags reference --recursive
```

### Search with Tags
Ask Claude: "Search for 'raster interrupt' in graphics-related docs"

### Find Similar Documents
Ask Claude: "Find documents similar to the VIC-II guide"

### Check for Updates
Ask Claude: "Check if any indexed documents have been updated"

## 🆘 Troubleshooting

**Server not starting?**
- Make sure you're using the venv Python: `.venv\Scripts\python.exe`
- Check logs in: `%TDZ_DATA_DIR%\server.log`

**No search results?**
- Verify documents are indexed: `cli.py list`
- Check search terms: Try broader queries first
- Enable fuzzy search for typo tolerance

**PDF extraction issues?**
- Scanned PDFs: Install Tesseract for OCR support
- Modern PDFs: Should work out of the box with pypdf

## 📖 Full Documentation

- [README.md](README.md) - Complete feature documentation
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Technical implementation details
- [CLAUDE.md](CLAUDE.md) - Development guide

## 🎉 You're Ready!

Start asking Claude Code about your C64 documentation. The knowledge base will handle:

**Core Search:**
- ✅ SQLite FTS5 full-text search (480x faster than BM25)
- ✅ Semantic search with embeddings
- ✅ Hybrid search (combines keyword + semantic)
- ✅ Fuzzy search for typo tolerance
- ✅ Phrase search with quotes
- ✅ Tag-based filtering
- ✅ Search result caching

**AI-Powered:**
- ✅ Natural Language Query Translation
- ✅ Entity Extraction (7 entity types)
- ✅ Entity Relationship Tracking with network graphs
- ✅ Document Summarization
- ✅ Smart Auto-Tagging

**Document Management:**
- ✅ Automatic duplicate detection
- ✅ Multiple file formats (PDF, TXT, MD, HTML, Excel)
- ✅ OCR for scanned PDFs (if enabled)
- ✅ Content-based update detection
- ✅ Bulk operations

**Access Methods:**
- ✅ MCP Server for Claude Code/Desktop
- ✅ REST API Server (27 HTTP endpoints)
- ✅ Streamlit Web GUI
- ✅ CLI for batch operations

Happy retro computing! 🎮

**Version:** 2.18.0 (2025-12-21)
