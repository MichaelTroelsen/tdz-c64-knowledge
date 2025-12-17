# Quick Start Guide - TDZ C64 Knowledge Base

**Get started in 5 minutes!**

---

## 🎯 Choose Your Path

### Path A: Web GUI (Recommended for New Users)
Best for: Exploring the knowledge base visually

```bash
launch-gui-full-features.bat
```
Then open: **http://localhost:8501**

✨ Features:
- Search with filters
- Browse documents
- Manage tags
- View analytics

---

### Path B: Command Line (Recommended for Power Users)
Best for: Scripting, automation, batch operations

```bash
# Search
launch-cli-full-features.bat search "VIC-II sprites"

# Add documents
launch-cli-full-features.bat add-folder "C:\path\to\docs" --tags reference --recursive

# List all documents
launch-cli-full-features.bat list

# Get statistics
launch-cli-full-features.bat stats
```

---

### Path C: Claude Code/Desktop Integration (Recommended for AI Users)
Best for: Using with Claude AI assistant

**Add to MCP Configuration:**

Windows config typically at: `%APPDATA%\Claude\claude_desktop_config.json`

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
        "USE_FUZZY_SEARCH": "1"
      }
    }
  }
}
```

Then restart Claude Desktop and ask:
> "Search the C64 knowledge base for assembly programming examples"

---

## 📚 Common Search Examples

### Exact Phrase Search
```bash
launch-cli-full-features.bat search "memory map" --max 5
```

### Find Code Examples
```bash
launch-cli-full-features.bat search "6502 assembly" --tags assembly --max 10
```

### Look Up Hardware
```bash
launch-cli-full-features.bat search "SID chip" --tags sid --max 5
```

### Fuzzy/Typo-Tolerant Search
```bash
launch-cli-full-features.bat search "sprits" --max 5
# Finds "sprites" even with typo!
```

---

## 🔍 Understanding Search Results

Each result shows:
```
Document Title (relevance score)
├─ Chunk ID: Content preview
├─ Page: PDF page number
└─ Score: 0-100 (higher = more relevant)
```

**Score Interpretation:**
- **90+** - Highly relevant
- **70-90** - Relevant
- **50-70** - Somewhat relevant
- **<50** - May not be relevant

---

## 📁 Adding Your Own Documents

### Single File
```bash
launch-cli-full-features.bat add "C:\docs\MyDocument.pdf" --title "My Doc" --tags reference assembly
```

### Entire Folder
```bash
launch-cli-full-features.bat add-folder "C:\my-docs" --tags reference --recursive
```

### Supported Formats
- ✅ PDF (.pdf)
- ✅ Text (.txt)
- ✅ Markdown (.md)
- ✅ HTML (.html, .htm)
- ✅ Excel (.xlsx, .xls)

---

## 🎛️ Configuration Options

### Enable All Advanced Features (Already Done!)
```bash
set USE_FTS5=1
set USE_SEMANTIC_SEARCH=1
set USE_BM25=1
set USE_QUERY_PREPROCESSING=1
set USE_FUZZY_SEARCH=1
```

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for all options.

---

## ✨ AI-Powered Summarization (NEW!)

Generate intelligent summaries of documents:

```bash
# Brief summary (200-300 words)
launch-cli-full-features.bat summarize "doc-id"

# Detailed summary (500-800 words)
launch-cli-full-features.bat summarize "doc-id" --type detailed

# Bullet-point summary
launch-cli-full-features.bat summarize "doc-id" --type bullet

# Generate for all documents
launch-cli-full-features.bat summarize-all --types brief detailed
```

**Features:**
- Three summary types (brief, detailed, bullet)
- Intelligent caching for instant retrieval
- Works with Claude or GPT
- Bulk processing support

Requires LLM API key. See [SUMMARIZATION.md](SUMMARIZATION.md) for complete guide.

---

## 🚀 Advanced Features

### Semantic Search (Conceptual)
Find documents by meaning, not just keywords:
```
"How do I make sounds?" → Finds SID programming docs
"Moving graphics" → Finds sprite/animation docs
```

### Fuzzy Matching
Automatically corrects typos:
```
"VIC-I" → Finds "VIC-II"
"registr" → Finds "register"
"asembly" → Finds "assembly"
```

### Table Search
Automatically extracts and searches tables from PDFs:
```bash
launch-cli-full-features.bat search "memory address" --max 5
# Finds: Memory mapping tables, register lists, etc.
```

### Code Block Search
Finds assembly, BASIC, and hex code:
```bash
launch-cli-full-features.bat search "LDA" --max 5
# Finds: Assembly code examples
```

---

## 📊 Current Knowledge Base

**What's Included:**
- 148+ Commodore 64 & 128 technical documents
- 4,680+ searchable chunks
- 6.9+ million words
- 209+ extracted tables
- 1,876+ detected code blocks

**Key Topics:**
- Assembly language programming
- SID music programming
- VIC-II graphics programming
- Memory mapping & I/O
- BASIC programming
- System internals
- Diagnostic guides

---

## 🎓 Learning Paths

### Beginner: Learn BASIC Programming
```bash
launch-cli-full-features.bat search "BASIC programming" --max 10
```

### Intermediate: Assembly Language
```bash
launch-cli-full-features.bat search "6502 assembly" --max 10
```

### Advanced: System Programming
```bash
launch-cli-full-features.bat search "memory address" --max 10
```

### Composer: SID Music
```bash
launch-cli-full-features.bat search "SID music" --tags sid --max 10
```

### Artist: Graphics Programming
```bash
launch-cli-full-features.bat search "VIC-II sprite" --max 10
```

---

## 💾 Database Management

### List All Documents
```bash
launch-cli-full-features.bat list
```

### See Statistics
```bash
launch-cli-full-features.bat stats
```

### Remove a Document
```bash
launch-cli-full-features.bat remove <doc_id>
```

### Add Bulk Documents
```bash
launch-cli-full-features.bat add-folder "C:\Downloads\tdz-c64-knowledge-input" --tags reference --recursive
```

---

## ⚡ Performance Tips

1. **First search will take ~50 seconds** (builds search index)
2. **Subsequent searches are cached** (instant results for same query)
3. **Use FTS5 search** (480x faster than BM25)
4. **Filter by tags** (narrows results)
5. **Use exact phrases** for specific terms

---

## 🔗 Integration Examples

### Use with Claude Desktop

> **User:** "Based on the C64 documentation, explain how sprite collision detection works"
>
> **Claude:** (searches KB) "According to the documentation... [citation with page numbers]"

### Use in Scripts
```batch
@echo off
for /f %%i in ('date /t') do set SEARCHDATE=%%i
echo C64 Knowledge Base Search - %SEARCHDATE% > search_results.txt
launch-cli-full-features.bat search "%~1" >> search_results.txt
```

### Batch Processing
```bash
# Add all C64 docs from downloads
launch-cli-full-features.bat add-folder "C:\Users\%username%\Downloads\C64-Books" --tags reference --recursive

# Then search them all
launch-cli-full-features.bat search "memory" --max 20 > results.txt
```

---

## 📋 Cheat Sheet

| Task | Command |
|------|---------|
| Open GUI | `launch-gui-full-features.bat` |
| Search | `launch-cli-full-features.bat search "query"` |
| Add docs | `launch-cli-full-features.bat add-folder "path" --tags tag1 --recursive` |
| List all | `launch-cli-full-features.bat list` |
| Stats | `launch-cli-full-features.bat stats` |
| Remove | `launch-cli-full-features.bat remove <id>` |

---

## ❓ FAQ

**Q: How do I search for a specific phrase?**
A: Use quotes: `search "VIC-II memory"`

**Q: Can I search across multiple topics?**
A: Yes: `search "sprite" "collision"` or use tag filters

**Q: How do I add documents?**
A: `launch-cli-full-features.bat add-folder "path" --tags reference --recursive`

**Q: What if my PDF is scanned?**
A: System will automatically try OCR (requires Poppler for best results)

**Q: Can I use this with Claude AI?**
A: Yes! Configure MCP in Claude Desktop settings

**Q: How do I find code examples?**
A: Search for: `search "6502" --max 10` or `search "LDA"` for assembly

**Q: Is my data secure?**
A: Yes - all stored locally in SQLite database with no external calls by default

---

## 🔧 Troubleshooting

**Issue:** Batch file not found
- **Solution:** Make sure you're in the project directory:
  ```bash
  cd C:\Users\mit\claude\c64server\tdz-c64-knowledge
  ```

**Issue:** Python not found
- **Solution:** Activate virtual environment first:
  ```bash
  .venv\Scripts\activate
  ```

**Issue:** First search is slow
- **Solution:** Normal (indexes 4,680 chunks). Subsequent searches use cache.

**Issue:** No results found
- **Solution:** Try fuzzy search or simpler query terms

**Issue:** Semantic search not working
- **Solution:** Requires sentence-transformers (already installed)

---

## 📚 More Help

- Full documentation: [README.md](README_UPDATED.md)
- Environment configuration: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- Development info: [CLAUDE.md](CLAUDE.md)
- Examples: [EXAMPLES.md](EXAMPLES.md)

---

## 🎉 You're Ready!

Pick one:

```bash
# Option 1: GUI (Easiest)
launch-gui-full-features.bat

# Option 2: CLI (Powerful)
launch-cli-full-features.bat search "what interests you?"

# Option 3: Claude Code (AI-powered)
# Ask Claude: "Search the C64 knowledge base for..."
```

**Happy exploring!** 🚀

---

**Version:** 2.12.0 | **Last Updated:** 2025-12-17 | **Status:** ✅ Ready to Use
