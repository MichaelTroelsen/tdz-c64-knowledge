# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. It provides tools to ingest PDFs and text files, build a searchable knowledge base, and query it through the MCP interface.

## Core Architecture

### Main Components

**server.py** - MCP server implementation
- `KnowledgeBase` class: Core data management (index + chunks storage)
- MCP tool handlers: `search_docs`, `add_document`, `get_chunk`, `get_document`, `list_docs`, `remove_document`, `kb_stats`
- MCP resource handlers: Exposes documents as `c64kb://` URIs
- Async server running on stdio transport

**cli.py** - Command-line interface for batch operations
- Wraps `KnowledgeBase` for CLI usage
- Commands: `add`, `add-folder`, `search`, `list`, `remove`, `stats`
- Useful for bulk importing documents

### Data Storage Model

The knowledge base uses a simple file-based storage:
- **index.json** - Document metadata (DocumentMeta objects serialized)
- **chunks/\<doc_id\>.json** - Content chunks for each document (DocumentChunk objects)

Documents are split into overlapping chunks (default 1500 words, 200 word overlap) to enable granular search and retrieval.

### Document Processing Pipeline

1. File ingestion (PDF via pypdf, text files with encoding detection)
2. Text extraction (pages joined with "--- PAGE BREAK ---")
3. Chunking with overlap (_chunk_text method)
4. Index generation (doc_id from MD5 hash of filepath)
5. Persistence (chunks + metadata saved separately)

### Search Implementation

Simple term-frequency scoring:
- Query split into terms
- Exact word matches score 2x partial matches
- Results sorted by score, limited by max_results
- Snippet extraction centers on first query term occurrence

## Development Commands

### Setup
```cmd
# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install mcp pypdf
```

### Testing the Server
```cmd
# Run server standalone (will wait for stdio input)
python server.py

# Test CLI commands
python cli.py stats
python cli.py search "your query"
```

### CLI Usage
```cmd
# Add single document
python cli.py add "C:/path/to/doc.pdf" --title "Title" --tags tag1 tag2

# Bulk add from folder
python cli.py add-folder "C:/docs" --tags reference --recursive

# Search
python cli.py search "query text" --max 10 --tags sid vic-ii

# List all documents
python cli.py list

# Show stats
python cli.py stats

# Remove document
python cli.py remove <doc_id>
```

## Environment Configuration

**TDZ_DATA_DIR** - Directory for index and chunks storage (default: `~/.tdz-c64-knowledge`)

When adding to Claude Code or Claude Desktop, set this in the MCP config `env` section to specify where the knowledge base files are stored.

## Code Patterns

### Adding New Tools

1. Add tool definition in `list_tools()` with proper inputSchema
2. Implement handler in `call_tool()` function
3. Use KnowledgeBase methods for data operations
4. Return list of `TextContent` objects

### Modifying Search Algorithm

Search is in `KnowledgeBase.search()` at server.py:227. Current implementation is simple term frequency. To enhance:
- Consider implementing BM25 or TF-IDF
- Add phrase matching
- Implement semantic search with embeddings

### Extending File Type Support

File type detection is in `add_document()` at server.py:152. To add new formats:
1. Add file extension to condition check
2. Implement extraction method (like `_extract_pdf_text`)
3. Update tool description and README

## Windows-Specific Notes

- Uses Windows-style paths (`C:\Users\...`)
- Batch files (.bat) provided for convenience (setup.bat, run.bat, tdz.bat)
- Virtual environment activation: `.venv\Scripts\activate`
- Python executable path for MCP config: `C:\...\\.venv\Scripts\python.exe`
