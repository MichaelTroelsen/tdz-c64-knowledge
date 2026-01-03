# Troubleshooting Guide

**Version:** 2.23.1
**Last Updated:** 2026-01-03

Common issues and solutions for the TDZ C64 Knowledge Base.

---

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [MCP Integration Issues](#mcp-integration-issues)
- [Search Problems](#search-problems)
- [Document Processing](#document-processing)
- [Performance Issues](#performance-issues)
- [Database Issues](#database-issues)
- [API Issues](#api-issues)
- [Advanced Troubleshooting](#advanced-troubleshooting)

---

## 🔧 Installation Issues

### ModuleNotFoundError: No module named 'mcp'

**Symptom:** Server won't start, missing mcp module

**Solution:**
```cmd
.venv\Scripts\activate
pip install mcp
```

---

### ModuleNotFoundError: No module named 'pypdf'

**Symptom:** PDF extraction fails

**Solution:**
```cmd
pip install pypdf rank-bm25 nltk
```

Or install all dependencies:
```cmd
pip install -e .
```

---

### "Python not found" when running server

**Symptom:** System Python used instead of venv

**Solution:**
Always use the full path to venv Python:
```cmd
.venv\Scripts\python.exe server.py
```

For MCP config:
```json
{
  "command": "C:\\full\\path\\.venv\\Scripts\\python.exe"
}
```

---

### Permission denied when installing packages

**Symptom:** pip install fails with permission error

**Solutions:**
1. Run as administrator
2. Use virtual environment (recommended)
3. Use `pip install --user` (not recommended for venv)

---

## 🔌 MCP Integration Issues

### Server not appearing in Claude Code

**Symptom:** MCP server doesn't show in Claude Code tools

**Diagnostic steps:**
1. Check MCP configuration:
   ```cmd
   claude mcp list
   ```

2. Verify paths are absolute (not relative):
   ```json
   {
     "command": "C:\\Users\\YourName\\...\\python.exe",
     "args": ["C:\\Users\\YourName\\...\\server.py"]
   }
   ```

3. Test server manually:
   ```cmd
   .venv\Scripts\python.exe server.py
   # Should start without errors
   # Press Ctrl+C to stop
   ```

4. Check logs:
   - Claude Code: Check developer console
   - Server: Check `%TDZ_DATA_DIR%\server.log`

**Common fixes:**
- Use absolute paths everywhere
- Restart Claude Code after config changes
- Ensure Python path has no spaces (or use quotes)
- Check environment variables are set

---

### "Connection refused" or server crashes

**Symptom:** MCP server crashes on startup or during operation

**Solutions:**
1. Check database integrity:
   ```cmd
   .venv\Scripts\python.exe cli.py stats
   ```

2. Check data directory exists:
   ```cmd
   echo %TDZ_DATA_DIR%
   dir %TDZ_DATA_DIR%
   ```

3. Review server logs:
   ```cmd
   type %TDZ_DATA_DIR%\server.log
   ```

4. Try with fresh database:
   ```cmd
   set TDZ_DATA_DIR=C:\temp\kb-test
   .venv\Scripts\python.exe server.py
   ```

---

### Tools not responding or timing out

**Symptom:** MCP tool calls hang or timeout

**Causes:**
- Large documents being processed
- Semantic search model loading
- Database lock

**Solutions:**
1. Check if operation is in progress:
   ```cmd
   .venv\Scripts\python.exe cli.py stats
   ```

2. Increase timeout in Claude Code settings (if available)

3. For semantic search, first query is slow (5-6 seconds for model loading)

4. Check database isn't locked:
   - Close any other connections
   - Restart server

---

## 🔍 Search Problems

### No search results when documents exist

**Symptom:** Search returns empty results

**Diagnostic:**
```cmd
.venv\Scripts\python.exe cli.py list
.venv\Scripts\python.exe cli.py search "test" --max 10
```

**Common causes:**

1. **FTS5 index not built**
   ```cmd
   # Check if FTS5 enabled
   echo %USE_FTS5%

   # If not, enable it
   set USE_FTS5=1
   ```

2. **Tags filtering too restrictive**
   ```cmd
   # Try without tags
   .venv\Scripts\python.exe cli.py search "test" --max 10
   ```

3. **BM25 index not built**
   - First search builds index (slow)
   - Subsequent searches are fast

**Fix:**
```cmd
# Force rebuild indexes
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); print('Indexes built')"
```

---

### Search results are irrelevant

**Symptom:** Search returns wrong documents

**Solutions:**

1. **Use fuzzy search** for typo tolerance:
   ```json
   {"query": "VIC2", "tool": "fuzzy_search"}
   ```

2. **Use semantic search** for concepts:
   ```json
   {"query": "how do sprites work", "tool": "semantic_search"}
   ```

3. **Use hybrid search** for best results:
   ```json
   {"query": "VIC-II sprite", "tool": "hybrid_search", "semantic_weight": 0.3}
   ```

4. **Add tags** to filter:
   ```json
   {"query": "sprite", "tags": ["graphics", "hardware"]}
   ```

---

### Semantic search not working

**Symptom:** semantic_search tool fails or returns error

**Requirements:**
```cmd
pip install sentence-transformers faiss-cpu
```

**Enable:**
```cmd
set USE_SEMANTIC_SEARCH=1
```

**Build embeddings:**
```cmd
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._build_embeddings()"
```

**Note:** First query loads model (~5 seconds), subsequent queries are fast (16ms avg)

---

## 📄 Document Processing

### PDF extraction fails with "pypdf not installed"

**Solution:**
```cmd
pip install pypdf
```

---

### Scanned PDFs return no text

**Symptom:** PDF added but no searchable content

**Solution:** Enable OCR
```cmd
pip install pytesseract pdf2image Pillow
# Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
set USE_OCR=1
```

---

### "Duplicate document" when adding file

**Symptom:** Document rejected as duplicate

**Explanation:** Content-based duplicate detection using MD5 hash

**Solutions:**
1. If truly different, it's likely the same content with different filename
2. Force add by removing old version first:
   ```cmd
   .venv\Scripts\python.exe cli.py remove <doc_id>
   .venv\Scripts\python.exe cli.py add "path/to/file.pdf"
   ```

---

### Large files cause timeout or crash

**Symptom:** Server hangs when processing large PDFs

**Solutions:**
1. Process in smaller batches
2. Increase system memory
3. Split large PDFs into smaller files
4. Disable OCR if not needed:
   ```cmd
   set USE_OCR=0
   ```

---

### "Path not allowed" security error

**Symptom:** SecurityError: Path outside allowed directories

**Cause:** Path traversal protection enabled

**Solution:** Add directory to whitelist:
```cmd
set ALLOWED_DOCS_DIRS=C:\c64docs;C:\more\docs;D:\backups
```

**Disable** (not recommended for production):
```cmd
set ALLOWED_DOCS_DIRS=
```

---

## ⚡ Performance Issues

### Slow search performance

**Symptom:** Searches take >1 second

**Solutions:**

1. **Enable FTS5** (480x faster):
   ```cmd
   set USE_FTS5=1
   ```

2. **Enable search cache**:
   ```cmd
   set SEARCH_CACHE_SIZE=100
   set SEARCH_CACHE_TTL=300
   ```

3. **Use appropriate search mode**:
   - FTS5: 85ms avg (keywords)
   - Semantic: 16ms avg (concepts)
   - Hybrid: 142ms avg (best quality)

4. **Reduce max_results**:
   ```json
   {"query": "test", "max_results": 5}
   ```

---

### High memory usage

**Symptom:** Python process uses excessive RAM

**Causes:**
- BM25 index loaded (all chunks in memory)
- Embeddings loaded (FAISS index)
- Many large documents

**Solutions:**
1. Use FTS5 instead of BM25 (no memory for chunks)
2. Use lazy loading (default)
3. Reduce `SEARCH_CACHE_SIZE`
4. Process documents in batches

**Expected memory usage:**
- Baseline: ~100MB
- + BM25: ~200MB per 1000 documents
- + Embeddings: ~4MB per 1000 chunks
- + Cache: Varies by cache size

---

### Database growing too large

**Symptom:** knowledge_base.db is multiple GB

**Solutions:**
1. Remove old/unused documents
2. Run VACUUM:
   ```cmd
   .venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('path/to/knowledge_base.db'); conn.execute('VACUUM'); conn.close()"
   ```
3. Check for duplicate documents
4. Clear extraction jobs table

**Expected sizes:**
- 100 documents: ~30MB
- 500 documents: ~150MB
- 1000 documents: ~300MB

---

## 💾 Database Issues

### "Database is locked" error

**Symptom:** Database locked by another process

**Solutions:**
1. Close all connections:
   - Stop MCP server
   - Stop REST API server
   - Stop Streamlit GUI
   - Close CLI

2. Wait 30 seconds for locks to release

3. If persists, find and kill processes:
   ```cmd
   tasklist | findstr python
   taskkill /PID <process_id> /F
   ```

4. Last resort - delete lock files:
   ```cmd
   del %TDZ_DATA_DIR%\knowledge_base.db-wal
   del %TDZ_DATA_DIR%\knowledge_base.db-shm
   ```

---

### "Database integrity check failed"

**Symptom:** Corruption detected

**Solutions:**
1. Run integrity check:
   ```cmd
   .venv\Scripts\python.exe cli.py health-check
   ```

2. Try SQLite repair:
   ```cmd
   sqlite3 knowledge_base.db ".recover" | sqlite3 knowledge_base_recovered.db
   ```

3. Restore from backup:
   ```cmd
   copy backup\knowledge_base.db %TDZ_DATA_DIR%\knowledge_base.db
   ```

4. Rebuild from source documents (if no backup)

---

### Lost all documents after crash

**Symptom:** Database empty after restart

**Check:**
1. Correct data directory:
   ```cmd
   echo %TDZ_DATA_DIR%
   dir %TDZ_DATA_DIR%
   ```

2. Database file exists:
   ```cmd
   dir %TDZ_DATA_DIR%\knowledge_base.db
   ```

3. Check for WAL files (crash recovery):
   ```cmd
   dir %TDZ_DATA_DIR%\*.db-*
   ```

**Recovery:**
- SQLite automatically recovers from WAL files on next open
- If recovery fails, restore from backup

---

## 🌐 API Issues

### REST API won't start

**Symptom:** uvicorn fails to start

**Solutions:**
1. Install dependencies:
   ```cmd
   pip install -e ".[rest]"
   ```

2. Check port not in use:
   ```cmd
   netstat -ano | findstr :8000
   ```

3. Try different port:
   ```cmd
   uvicorn rest_server:app --port 8001
   ```

---

### "Authentication disabled" warning

**Symptom:** Server starts but shows security warning

**Solution:** Set API keys:
```cmd
set TDZ_API_KEYS=your-secret-key-1,your-secret-key-2
```

---

### CORS errors in browser

**Symptom:** Browser blocks requests from web app

**Solution:** Configure CORS:
```cmd
set CORS_ORIGINS=http://localhost:3000,https://yourapp.com
```

---

## 🔬 Advanced Troubleshooting

### Enable debug logging

```cmd
set LOG_LEVEL=DEBUG
.venv\Scripts\python.exe server.py
```

---

### Test individual components

**Test database:**
```cmd
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); print(f'Docs: {len(kb.documents)}')"
```

**Test search:**
```cmd
.venv\Scripts\python.exe cli.py search "test" --max 5
```

**Test entity extraction:**
```cmd
.venv\Scripts\python.exe cli.py extract-entities <doc_id>
```

---

### Rebuild indexes

**FTS5:**
```cmd
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._rebuild_fts5_indexes(); print('FTS5 rebuilt')"
```

**Embeddings:**
```cmd
.venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._build_embeddings(); print('Embeddings rebuilt')"
```

---

### Get system diagnostics

```cmd
.venv\Scripts\python.exe cli.py health-check
```

**Returns:**
- Database health
- Feature status
- Performance metrics
- Issues detected

---

### Common Error Messages

**"No LLM provider configured"**
- Set `LLM_PROVIDER=anthropic` or `openai`
- Set API key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

**"mdscrape not found"**
- Install mdscrape: https://github.com/MichaelTroelsen/mdscrape
- Set `MDSCRAPE_PATH=C:\path\to\mdscrape\mdscrape.exe`

**"FTS5 not available"**
- Check SQLite version: `python -c "import sqlite3; print(sqlite3.sqlite_version)"`
- Requires SQLite 3.9.0+ (usually fine on modern systems)

---

## 💡 Getting Help

If your issue isn't covered here:

1. **Check the logs:** `%TDZ_DATA_DIR%\server.log`
2. **Run health check:** `cli.py health-check`
3. **Review documentation:**
   - [README](../README.md)
   - [ARCHITECTURE](../ARCHITECTURE.md)
   - [MCP Integration](MCP_INTEGRATION.md)
4. **GitHub Issues:** Report bugs with:
   - Error message
   - Steps to reproduce
   - System info (Windows version, Python version)
   - Relevant logs

---

**Version:** 2.23.1
**Platform:** Windows (with cross-platform notes)
**Last Updated:** 2026-01-03
