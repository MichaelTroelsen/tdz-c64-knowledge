# MCP Integration Guide

**Version:** 2.23.1
**Last Updated:** 2026-01-03

Complete guide for integrating the TDZ C64 Knowledge Base with Claude Code, Claude Desktop, and other MCP clients.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Claude Code Integration](#claude-code-integration)
- [Claude Desktop Integration](#claude-desktop-integration)
- [Configuration Options](#configuration-options)
- [Testing & Verification](#testing--verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## 🎯 Overview

### What is MCP?

**Model Context Protocol (MCP)** is a standard protocol that allows AI assistants like Claude to interact with external tools and data sources.

The TDZ C64 Knowledge Base implements an MCP server that provides:
- **50+ tools** for searching and managing C64 documentation
- **Stdio transport** for local integration
- **Full-featured API** accessible from Claude

---

## 🔧 Claude Code Integration

### Quick Setup

**Method 1: Automatic (Recommended)**

```cmd
cd C:\Users\YourName\mcp-servers\tdz-c64-knowledge
claude mcp add tdz-c64-knowledge -- .venv\Scripts\python.exe server.py
```

**Method 2: Manual Configuration**

1. **Open Claude Code MCP settings:**
   - Location: `.claude/settings.json` in your project
   - Or use: `claude mcp list` to see config location

2. **Add server configuration:**
   ```json
   {
     "mcpServers": {
       "tdz-c64-knowledge": {
         "command": "C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
         "args": ["C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\server.py"],
         "env": {
           "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data",
           "USE_FTS5": "1",
           "USE_FUZZY_SEARCH": "1",
           "SEARCH_CACHE_SIZE": "100"
         }
       }
     }
   }
   ```

3. **Restart Claude Code**

---

### Verify Installation

**Test the server manually:**
```cmd
cd C:\Users\YourName\mcp-servers\tdz-c64-knowledge
.venv\Scripts\activate
python server.py
# Should start without errors
# Press Ctrl+C to stop
```

**Check available tools:**
```cmd
claude mcp list
# Should show "tdz-c64-knowledge" with 50+ tools
```

**Test a tool:**
Ask Claude Code: "List all C64 documents in the knowledge base"
- Should call the `list_docs` tool

---

### Configuration Best Practices

**✅ DO:**
- Use absolute paths for `command` and `args`
- Keep venv inside project directory
- Set `TDZ_DATA_DIR` to a dedicated location
- Enable `USE_FTS5=1` for best performance

**❌ DON'T:**
- Use relative paths (they won't work reliably)
- Use system Python (always use venv)
- Store data in temp directories
- Forget to restart Claude Code after config changes

---

## 🖥️ Claude Desktop Integration

### Configuration

1. **Find config file:**
   ```cmd
   %APPDATA%\Claude\claude_desktop_config.json
   ```
   (Usually: `C:\Users\YourName\AppData\Roaming\Claude\claude_desktop_config.json`)

2. **Edit or create the file:**
   ```json
   {
     "mcpServers": {
       "tdz-c64-knowledge": {
         "command": "C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\.venv\\Scripts\\python.exe",
         "args": ["C:\\Users\\YourName\\mcp-servers\\tdz-c64-knowledge\\server.py"],
         "env": {
           "TDZ_DATA_DIR": "C:\\Users\\YourName\\c64-knowledge-data",
           "USE_FTS5": "1"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop**

---

### Multiple Servers

You can run multiple MCP servers simultaneously:

```json
{
  "mcpServers": {
    "tdz-c64-knowledge": {
      "command": "C:\\...\\python.exe",
      "args": ["C:\\...\\server.py"],
      "env": {...}
    },
    "another-mcp-server": {
      "command": "C:\\...\\python.exe",
      "args": ["C:\\...\\other_server.py"],
      "env": {...}
    }
  }
}
```

---

## ⚙️ Configuration Options

### Required Settings

| Setting | Description | Example |
|---------|-------------|---------|
| `command` | Python executable (absolute path) | `C:\...\python.exe` |
| `args` | Server script path (absolute) | `["C:\...\server.py"]` |

### Environment Variables

#### Essential
| Variable | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `TDZ_DATA_DIR` | Database directory | `~/.tdz-c64-knowledge` | Custom path |
| `USE_FTS5` | Enable FTS5 search | `0` | `1` |

#### Search Features
| Variable | Description | Default |
|----------|-------------|---------|
| `USE_SEMANTIC_SEARCH` | Enable embeddings | `0` |
| `USE_FUZZY_SEARCH` | Enable typo tolerance | `1` |
| `USE_QUERY_PREPROCESSING` | NLTK preprocessing | `1` |
| `FUZZY_THRESHOLD` | Similarity threshold (0-100) | `80` |

#### Performance
| Variable | Description | Default |
|----------|-------------|---------|
| `SEARCH_CACHE_SIZE` | Max cached results | `100` |
| `SEARCH_CACHE_TTL` | Cache TTL (seconds) | `300` |
| `EMBEDDING_CACHE_TTL` | Embedding cache (seconds) | `3600` |

#### Security
| Variable | Description | Default |
|----------|-------------|---------|
| `ALLOWED_DOCS_DIRS` | Whitelist directories | None |

#### AI Features
| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `anthropic` or `openai` | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `LLM_MODEL` | Model override | Auto |

#### External Tools
| Variable | Description | Default |
|----------|-------------|---------|
| `MDSCRAPE_PATH` | Path to mdscrape | Auto-detect |

---

### Example Configurations

**Minimal (Fast startup):**
```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\c64-knowledge",
    "USE_FTS5": "1"
  }
}
```

**Recommended (Balanced):**
```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\c64-knowledge",
    "USE_FTS5": "1",
    "USE_FUZZY_SEARCH": "1",
    "SEARCH_CACHE_SIZE": "100"
  }
}
```

**Full-Featured (All capabilities):**
```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\c64-knowledge",
    "USE_FTS5": "1",
    "USE_SEMANTIC_SEARCH": "1",
    "USE_FUZZY_SEARCH": "1",
    "USE_QUERY_PREPROCESSING": "1",
    "SEARCH_CACHE_SIZE": "100",
    "LLM_PROVIDER": "anthropic",
    "ANTHROPIC_API_KEY": "sk-ant-xxxxx",
    "ALLOWED_DOCS_DIRS": "C:\\c64docs;D:\\retro\\docs"
  }
}
```

**Production (Secure):**
```json
{
  "env": {
    "TDZ_DATA_DIR": "/var/lib/tdz-c64-knowledge",
    "USE_FTS5": "1",
    "USE_SEMANTIC_SEARCH": "1",
    "SEARCH_CACHE_SIZE": "200",
    "ALLOWED_DOCS_DIRS": "/data/c64docs",
    "LOG_LEVEL": "INFO"
  }
}
```

---

## ✅ Testing & Verification

### Manual Server Test

```cmd
cd C:\Users\YourName\mcp-servers\tdz-c64-knowledge
.venv\Scripts\activate
python server.py
```

**Expected output:**
```
TDZ C64 Knowledge Base MCP Server v2.23.1
Data directory: C:\Users\YourName\c64-knowledge-data
Loading documents...
Loaded 158 documents
Server initialized successfully
Listening on stdio...
```

Press `Ctrl+C` to stop.

---

### Test Tools via Claude

**Basic search:**
```
Ask Claude: "Search the C64 docs for VIC-II sprites"
```
Expected: Results with document titles and snippets

**List documents:**
```
Ask Claude: "List all documents in the knowledge base"
```
Expected: Document list with metadata

**Get statistics:**
```
Ask Claude: "What are the knowledge base statistics?"
```
Expected: Document counts, chunk stats, etc.

---

### Check Tool Availability

**In Claude Code:**
```cmd
claude mcp list
```

Should show:
- Server name: `tdz-c64-knowledge`
- Status: Active
- Tools: 50+

**In Claude Desktop:**
Look for hammer icon (🔨) in chat interface showing available tools.

---

## 🔍 Troubleshooting

### Server Not Appearing

**Symptoms:**
- No tools available in Claude
- MCP server not in list

**Solutions:**

1. **Check paths are absolute:**
   ```json
   {
     "command": "C:\\full\\path\\python.exe",  // ✅ Absolute
     // NOT: ".venv\\Scripts\\python.exe"     // ❌ Relative
   }
   ```

2. **Test server manually:**
   ```cmd
   .venv\Scripts\python.exe server.py
   # Should start without errors
   ```

3. **Check for errors:**
   - Claude Code: Developer console
   - Claude Desktop: Check logs in `%APPDATA%\Claude\logs`

4. **Restart the application**

---

### Tools Not Working

**Symptoms:**
- Tools appear but calls fail
- Timeout errors

**Solutions:**

1. **Check environment variables:**
   ```cmd
   echo %TDZ_DATA_DIR%
   dir %TDZ_DATA_DIR%
   ```

2. **Verify database exists:**
   ```cmd
   dir %TDZ_DATA_DIR%\knowledge_base.db
   ```

3. **Test CLI:**
   ```cmd
   .venv\Scripts\python.exe cli.py stats
   ```

4. **Check server logs:**
   ```cmd
   type %TDZ_DATA_DIR%\server.log
   ```

---

### "Database locked" Errors

**Cause:** Multiple connections to database

**Solution:**
1. Stop all servers:
   - MCP server (restart Claude)
   - REST API server
   - Streamlit GUI
   - CLI

2. Wait 30 seconds

3. Restart MCP server only

---

### Performance Issues

**Symptoms:**
- Slow tool responses
- Timeouts

**Solutions:**

1. **Enable FTS5:**
   ```json
   {"USE_FTS5": "1"}
   ```

2. **Enable caching:**
   ```json
   {"SEARCH_CACHE_SIZE": "100"}
   ```

3. **Reduce max_results:**
   Ask for fewer results (5-10 instead of 20-50)

4. **Check database size:**
   Large databases (>1GB) may be slow
   - Run VACUUM
   - Remove old documents

---

### Semantic Search Not Available

**Symptoms:**
- `semantic_search` tool fails
- "Embeddings not built" error

**Solution:**

1. **Install dependencies:**
   ```cmd
   pip install sentence-transformers faiss-cpu
   ```

2. **Enable feature:**
   ```json
   {"USE_SEMANTIC_SEARCH": "1"}
   ```

3. **Build embeddings:**
   ```cmd
   .venv\Scripts\python.exe -c "from server import KnowledgeBase; kb = KnowledgeBase(); kb._build_embeddings()"
   ```

---

## 🚀 Advanced Configuration

### Custom Data Directories

**Use case:** Separate knowledge bases for different projects

```json
{
  "mcpServers": {
    "c64-hardware-docs": {
      "command": "C:\\...\\python.exe",
      "args": ["C:\\...\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\kb\\hardware"
      }
    },
    "c64-software-docs": {
      "command": "C:\\...\\python.exe",
      "args": ["C:\\...\\server.py"],
      "env": {
        "TDZ_DATA_DIR": "C:\\kb\\software"
      }
    }
  }
}
```

---

### Shared Knowledge Base

**Use case:** Multiple users accessing same database

```json
{
  "env": {
    "TDZ_DATA_DIR": "\\\\server\\shared\\c64-knowledge",
    "USE_FTS5": "1"
  }
}
```

**Note:** Ensure network path is accessible and has proper permissions

---

### Development vs Production

**Development:**
```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\dev\\kb-test",
    "LOG_LEVEL": "DEBUG",
    "USE_FTS5": "1"
  }
}
```

**Production:**
```json
{
  "env": {
    "TDZ_DATA_DIR": "C:\\production\\c64-kb",
    "LOG_LEVEL": "INFO",
    "USE_FTS5": "1",
    "USE_SEMANTIC_SEARCH": "1",
    "ALLOWED_DOCS_DIRS": "C:\\trusted\\docs",
    "SEARCH_CACHE_SIZE": "200"
  }
}
```

---

### Running Multiple Versions

**Use case:** Testing new version while keeping stable version

```json
{
  "mcpServers": {
    "tdz-c64-stable": {
      "command": "C:\\servers\\v2.22\\python.exe",
      "args": ["C:\\servers\\v2.22\\server.py"],
      "env": {"TDZ_DATA_DIR": "C:\\kb\\stable"}
    },
    "tdz-c64-beta": {
      "command": "C:\\servers\\v2.23\\python.exe",
      "args": ["C:\\servers\\v2.23\\server.py"],
      "env": {"TDZ_DATA_DIR": "C:\\kb\\beta"}
    }
  }
}
```

---

## 📚 See Also

- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues
- [MCP Tools Reference](MCP_TOOLS_REFERENCE.md) - All 50+ tools
- [Quick Start](../QUICKSTART.md) - Fast setup guide
- [Security Guide](SECURITY.md) - Production security

---

**Version:** 2.23.1
**Platform:** Windows (cross-platform compatible)
**Last Updated:** 2026-01-03
