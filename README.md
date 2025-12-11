# TDZ C64 Knowledge

An MCP (Model Context Protocol) server for managing and searching Commodore 64 documentation. Add PDFs and text files to build a searchable knowledge base that Claude Code or other MCP clients can query.

## Features

- **PDF and text file ingestion** - Extract and index content from documentation
- **BM25 search algorithm** - Industry-standard ranking for accurate results
- **Query preprocessing** - Intelligent stemming and stopword removal with NLTK
- **Phrase search** - Use quotes for exact phrase matching (e.g., `"VIC-II chip"`)
- **Search term highlighting** - Matching terms highlighted in search results
- **Tag-based filtering** - Organize docs by topic (memory-map, sid, vic-ii, basic, assembly, etc.)
- **Chunked retrieval** - Get specific sections without loading entire documents
- **PDF metadata extraction** - Author, subject, creator, and creation date
- **Page number tracking** - Results show PDF page numbers for easy reference
- **Persistent index** - Documents stay indexed between sessions
- **Comprehensive logging** - File and console logging for debugging

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
   uv pip install mcp pypdf rank-bm25 nltk

   # Or using pip
   python -m venv .venv
   .venv\Scripts\activate
   pip install mcp pypdf rank-bm25 nltk
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
| `USE_BM25` | Enable BM25 search algorithm (0=disabled, 1=enabled) | `1` (enabled) |
| `USE_QUERY_PREPROCESSING` | Enable query preprocessing with NLTK (0=disabled, 1=enabled) | `1` (enabled) |

## Search Features

### BM25 Ranking Algorithm

By default, the server uses the **BM25 (Okapi BM25)** algorithm for search ranking. BM25 is an industry-standard probabilistic ranking function that provides much better relevance scoring than simple term frequency.

**Benefits:**
- More accurate ranking of search results
- Better handling of document length variations
- Improved scoring for multi-term queries

**How it works:**
- Tokenizes documents and queries into words
- Scores based on term frequency, document frequency, and document length
- Handles small documents (may produce negative scores, which are handled correctly)

To disable BM25 and use simple term frequency search, set `USE_BM25=0` in your environment.

### Phrase Search

Use double quotes to search for exact phrases:

```
search_docs(query='"VIC-II chip" registers')
```

This will:
- Match documents containing the exact phrase "VIC-II chip"
- Also search for the term "registers"
- Boost scores for documents with exact phrase matches

### Search Term Highlighting

Search results automatically highlight matching terms in snippets using markdown bold (`**term**`). This makes it easy to see why a result matched your query.

### Page Number Tracking

For PDF documents, search results include the estimated page number where the content appears. This helps you quickly find the information in the original document.

### Query Preprocessing

The server uses NLTK for intelligent query preprocessing to improve search accuracy:

**Features:**
- **Stemming** - Matches word variations (searching "running" finds "run", "runs", "runner")
- **Stopword removal** - Filters out common words ("the", "a", "is") that don't add meaning
- **Technical term preservation** - Keeps hyphenated terms like "VIC-II" and numbers like "6502"
- **Smart tokenization** - Properly handles punctuation and special characters

**Benefits:**
- Find more relevant results with natural language queries
- Matches conceptually similar terms (plural/singular, verb tenses)
- Reduces noise from common words
- Preserves important technical terminology

**Configuration:**
- Enabled by default when NLTK is installed
- Disable with environment variable: `USE_QUERY_PREPROCESSING=0`
- Works with both BM25 and simple search algorithms

**Example:**
- Query: "How does the SID chip generate sounds?"
- Preprocessed: ["sid", "chip", "generat", "sound"]
- Matches: "generate", "generating", "generated", "sounds", "sound"

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
- "Search for the exact phrase 'VIC-II chip' in the docs"
- "Find documentation about 'raster interrupts' in graphics-related docs"
- "Add C:/docs/c64/mapping_the_c64.pdf to the knowledge base with tags memory-map, reference"

**Phrase search examples:**
```
search_docs(query='"VIC-II chip"')
search_docs(query='"raster interrupt" timing')
search_docs(query='"SID register" $D400')
```

## Data Storage

The knowledge base uses an **SQLite database** for efficient storage and querying:
- `knowledge_base.db` - SQLite database containing documents and chunks
  - `documents` table - Document metadata with full-text indexes
  - `chunks` table - Chunked document content with foreign key relationships

**Automatic Migration**: If you're upgrading from a previous version with JSON files (`index.json` and `chunks/*.json`), the server will automatically migrate your data to SQLite on first run. The JSON files are preserved as backup and can be manually deleted after verification.

**Benefits of SQLite**:
- Lazy loading - Only loads document metadata at startup, not all chunks
- ACID transactions - Data integrity guaranteed
- Efficient queries - Fast chunk retrieval and statistics
- Scalable - Supports 100,000+ documents without memory issues

Default location: `~/.tdz-c64-knowledge` (or set via `TDZ_DATA_DIR`)

## Troubleshooting

### "pypdf not installed" or "rank_bm25 not found"
Run: `pip install pypdf rank-bm25` (in your virtual environment)

### "mcp module not found"
Run: `pip install mcp` (in your virtual environment)

### Server not responding
Make sure you're using the Python from your virtual environment, not the system Python.

### PDF extraction issues
Some scanned PDFs may not extract text well. Consider using OCR tools to convert them first, or add the plain text version instead.

### BM25 issues
If you experience search problems with BM25:
1. Check the logs in `TDZ_DATA_DIR/server.log`
2. Try disabling BM25 with `USE_BM25=0` environment variable
3. Ensure rank-bm25 is installed: `pip show rank-bm25`

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
