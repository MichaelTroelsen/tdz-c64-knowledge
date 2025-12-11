# TDZ C64 Knowledge

An MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. Add PDFs and text files to build a searchable knowledge base that Claude Code or other MCP clients can query.

## Features

- **PDF and text file ingestion** - Extract and index content from documentation
- **Full-text search** - Find relevant information across all documents
- **Tag-based filtering** - Organize docs by topic (memory-map, sid, vic-ii, basic, assembly, etc.)
- **Chunked retrieval** - Get specific sections without loading entire documents
- **Persistent index** - Documents stay indexed between sessions

## Installation (Windows)

### Prerequisites

1. **Python 3.10+** - Download from https://python.org
   - During install, check "Add Python to PATH"

2. **uv** (recommended) or pip
   ```cmd
   pip install uv
   ```

### Setup

1. **Clone or download this folder** to a location like:
   ```
   C:\Users\YourName\mcp-servers\tdz-c64-knowledge
   ```

2. **Create virtual environment and install dependencies:**
   ```cmd
   cd C:\Users\YourName\mcp-servers\tdz-c64-knowledge
   
   # Using uv (faster)
   uv venv
   .venv\Scripts\activate
   uv pip install mcp pypdf
   
   # Or using pip
   python -m venv .venv
   .venv\Scripts\activate
   pip install mcp pypdf
   ```

3. **Test the server:**
   ```cmd
   python server.py
   ```
   (Press Ctrl+C to stop - it will just wait for input since it's an MCP server)

## Configuration

### For Claude Code

Add to your Claude Code MCP settings (usually in `.claude/settings.json` or via `claude mcp add`):

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

Or add via command line:
```cmd
claude mcp add tdz-c64-knowledge -- C:\Users\YourName\mcp-servers\tdz-c64-knowledge\.venv\Scripts\python.exe C:\Users\YourName\mcp-servers\tdz-c64-knowledge\server.py
```

### For Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TDZ_DATA_DIR` | Directory to store index and chunks | `~/.tdz-c64-knowledge` |

## Tools

The server exposes these tools to MCP clients:

### search_docs
Search the knowledge base for information.

```
search_docs(query="SID register", max_results=5, tags=["sid"])
```

### add_document
Add a PDF or text file to the knowledge base.

```
add_document(
  filepath="C:/c64docs/Programmer_Reference.pdf",
  title="C64 Programmer's Reference Guide",
  tags=["reference", "memory-map", "basic", "assembly"]
)
```

### list_docs
List all indexed documents.

### get_chunk
Get the full content of a specific search result chunk.

```
get_chunk(doc_id="abc123", chunk_id=5)
```

### get_document
Get the full content of a document.

### remove_document
Remove a document from the knowledge base.

### kb_stats
Get statistics about the knowledge base.

## Suggested Tags

Organize your C64 docs with consistent tags:

- `reference` - General reference guides
- `memory-map` - Memory maps and addresses
- `basic` - BASIC programming
- `assembly` - 6502/6510 assembly
- `sid` - SID chip / sound
- `vic-ii` - VIC-II chip / graphics
- `cia` - CIA chips / I/O
- `kernal` - Kernal ROM routines
- `hardware` - Hardware specifications
- `disk` - Disk drives and DOS

## Usage Examples

Once configured, you can ask Claude Code things like:

- "Search the C64 docs for SID voice registers"
- "What does the memory map say about $D400?"
- "Find information about sprite multiplexing"
- "Add C:/docs/c64/mapping_the_c64.pdf to the knowledge base with tags memory-map, reference"

## Data Storage

The knowledge base stores:
- `index.json` - Document metadata
- `chunks/*.json` - Chunked document content

Default location: `~/.tdz-c64-knowledge` (or set via `TDZ_DATA_DIR`)

## Troubleshooting

### "pypdf not installed"
Run: `pip install pypdf` (in your virtual environment)

### "mcp module not found"
Run: `pip install mcp` (in your virtual environment)

### Server not responding
Make sure you're using the Python from your virtual environment, not the system Python.

### PDF extraction issues
Some scanned PDFs may not extract text well. Consider using OCR tools to convert them first, or add the plain text version instead.

## Development

### Running Tests

Install development dependencies:
```cmd
pip install -e ".[dev]"
```

Run the test suite:
```cmd
pytest test_server.py -v
```

Run tests with coverage:
```cmd
pytest test_server.py -v --cov=server --cov-report=term
```

### CI/CD Pipeline

This project includes a GitHub Actions workflow that:
- Runs tests on Python 3.10, 3.11, and 3.12
- Tests on Windows, Linux, and macOS
- Performs code quality checks with Ruff
- Validates documentation completeness
- Runs integration tests

The pipeline runs automatically on push to main/master/develop branches and on pull requests.

## License

MIT License - Use freely for your retro computing projects!
