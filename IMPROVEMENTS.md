# MCP Server Improvement Suggestions

## 🎯 Priority Levels
- **P0**: Critical improvements (security, correctness, major performance)
- **P1**: High-value improvements (user experience, scalability)
- **P2**: Nice-to-have enhancements (convenience, polish)

---

## 1. Search Quality Improvements

### P0: Implement Better Search Algorithms
**Current Issue**: Simple term frequency scoring is not very accurate

**Recommendations**:
- **BM25 Algorithm**: Industry-standard for text search, better than TF-IDF
  ```python
  # Pseudo-code
  def bm25_score(term_freq, doc_length, avg_doc_length, k1=1.5, b=0.75):
      return (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
  ```
- **Benefits**: Much more accurate ranking, handles document length bias
- **Effort**: Medium (1-2 days)
- **Library**: Consider `rank-bm25` package

### P1: Add Phrase Search Support
**Current Issue**: Cannot search for exact phrases like "VIC-II register"

**Recommendations**:
```python
def search_phrase(self, query: str):
    """Search for exact phrase match."""
    if '"' in query:
        # Extract phrases in quotes
        phrases = re.findall(r'"([^"]*)"', query)
        # Score higher for exact phrase matches
```

### P1: Implement Query Preprocessing
**Current Issue**: No stemming, lemmatization, or stopword removal

**Recommendations**:
- Add `nltk` or `spacy` for text processing
- Stem words ("running" → "run")
- Remove stopwords ("the", "a", "is")
- Handle synonyms (SID = Sound Interface Device)

### P2: Fuzzy Search / Typo Tolerance
**Current Issue**: "VIC-I" won't find "VIC-II"

**Recommendations**:
- Use Levenshtein distance for fuzzy matching
- Library: `python-Levenshtein` or `rapidfuzz`
```python
from rapidfuzz import fuzz
if fuzz.ratio(term, token) > 85:  # 85% similarity
    score += partial_score
```

---

## 2. Performance & Scalability

### P0: Move to SQLite Database
**Current Issue**: All chunks loaded into memory, doesn't scale beyond ~1000 docs

**Recommendations**:
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT,
    title TEXT,
    filepath TEXT,
    file_type TEXT,
    total_pages INTEGER,
    total_chunks INTEGER,
    indexed_at TIMESTAMP,
    tags TEXT  -- JSON array
);

CREATE TABLE chunks (
    doc_id TEXT,
    chunk_id INTEGER,
    content TEXT,
    word_count INTEGER,
    PRIMARY KEY (doc_id, chunk_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX idx_chunks_content ON chunks(content);
CREATE INDEX idx_documents_tags ON documents(tags);
```

**Benefits**:
- Scales to 100k+ documents
- Efficient filtering and sorting
- Full-text search (FTS5)
- ACID compliance

**Effort**: High (3-5 days)

### P1: Implement SQLite FTS5 for Full-Text Search
```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content_rowid UNINDEXED,
    tokenize='porter unicode61'
);
```

**Benefits**: 10-100x faster search with built-in ranking

### P1: Add Caching Layer
**Current Issue**: Repeatedly searching same queries is inefficient

**Recommendations**:
```python
from functools import lru_cache
import hashlib

class KnowledgeBase:
    def __init__(self, data_dir):
        self._search_cache = {}  # LRU cache with TTL

    def search(self, query: str, max_results: int = 5, tags=None):
        cache_key = hashlib.md5(
            f"{query}:{max_results}:{tags}".encode()
        ).hexdigest()

        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = self._search_uncached(query, max_results, tags)
        self._search_cache[cache_key] = results
        return results
```

### P2: Lazy Loading of Chunks
**Current Issue**: All chunks loaded at startup

**Recommendations**:
- Load chunks on-demand from disk/database
- Keep index in memory, stream chunks as needed

---

## 3. Feature Enhancements

### P0: Add Semantic Search with Embeddings
**Current Issue**: Cannot find conceptually similar content

**Recommendations**:
```python
# Use sentence-transformers for semantic search
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, data_dir):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_document(self, ...):
        # Generate embeddings for each chunk
        chunk_embeddings = self.model.encode([c.content for c in chunks])
        # Store in vector database (e.g., FAISS, ChromaDB)

    def semantic_search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query])
        # Find nearest neighbors in embedding space
```

**Benefits**:
- Finds "sprites" when searching for "movable objects"
- Understands context and meaning
- Better for natural language queries

**Effort**: Medium (2-3 days)
**Libraries**: `sentence-transformers`, `faiss-cpu`, or `chromadb`

### P1: Highlight Search Terms in Results
```python
def _extract_snippet(self, content: str, query_terms: set, snippet_size: int = 300):
    # ... existing code ...

    # Highlight matching terms
    for term in query_terms:
        snippet = re.sub(
            f'({re.escape(term)})',
            r'**\1**',  # Bold in markdown
            snippet,
            flags=re.IGNORECASE
        )
    return snippet
```

### P1: Add "More Like This" Tool
**Recommendation**:
```python
Tool(
    name="find_similar",
    description="Find documents similar to a given document",
    inputSchema={
        "properties": {
            "doc_id": {"type": "string"},
            "max_results": {"type": "integer", "default": 5}
        }
    }
)
```

### P1: Document Update Detection
**Current Issue**: No way to detect if source file has changed

**Recommendations**:
```python
@dataclass
class DocumentMeta:
    # ... existing fields ...
    file_hash: str  # MD5 of file contents
    file_mtime: float  # Modification time

def needs_reindex(self, filepath: str, doc_meta: DocumentMeta) -> bool:
    current_mtime = os.path.getmtime(filepath)
    return current_mtime > doc_meta.file_mtime
```

### P2: Pagination for Large Result Sets
```python
def search(self, query: str, max_results: int = 5, offset: int = 0, tags=None):
    # ... search logic ...
    return {
        'results': results[offset:offset + max_results],
        'total': len(results),
        'offset': offset,
        'has_more': offset + max_results < len(results)
    }
```

### P2: Export/Backup Functionality
```python
Tool(
    name="export_kb",
    description="Export knowledge base to a portable format",
    inputSchema={
        "properties": {
            "output_path": {"type": "string"},
            "format": {"enum": ["json", "sqlite", "zip"]}
        }
    }
)
```

---

## 4. Data Quality & Processing

### P0: Track Page Numbers in PDFs
**Current Issue**: Line 183 has `page=None` comment

**Recommendations**:
```python
def _extract_pdf_text(self, filepath: str) -> tuple[list[tuple[str, int]], int]:
    """Extract text with page numbers."""
    reader = PdfReader(filepath)
    pages_with_numbers = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages_with_numbers.append((text, page_num))
    return pages_with_numbers, len(reader.pages)

def _chunk_text_with_pages(self, pages_with_numbers: list[tuple[str, int]], ...):
    """Chunk text while preserving page information."""
    # Track which page each chunk came from
```

**Benefits**: Users can reference exact page numbers

### P1: Extract Document Metadata
```python
def _extract_pdf_metadata(self, reader: PdfReader) -> dict:
    """Extract PDF metadata."""
    metadata = reader.metadata
    return {
        'author': metadata.get('/Author'),
        'title': metadata.get('/Title'),
        'subject': metadata.get('/Subject'),
        'creation_date': metadata.get('/CreationDate')
    }
```

### P1: Duplicate Detection
**Current Issue**: Same document can be indexed multiple times with different paths

**Recommendations**:
```python
def _generate_doc_id(self, filepath: str) -> str:
    """Generate ID based on content hash, not filepath."""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash[:12]
```

### P2: OCR Support for Scanned PDFs
**Recommendations**:
- Use `pytesseract` for OCR
- Detect if PDF is text-based or image-based
- Fall back to OCR if text extraction fails

---

## 5. Error Handling & Robustness

### P0: Better Error Handling in Async Functions
**Current Issue**: Generic exceptions don't provide context

**Recommendations**:
```python
class KnowledgeBaseError(Exception):
    """Base exception for KB errors."""
    pass

class DocumentNotFoundError(KnowledgeBaseError):
    pass

class IndexCorruptedError(KnowledgeBaseError):
    pass

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        # ... tool logic ...
    except DocumentNotFoundError as e:
        return [TextContent(type="text", text=f"Document not found: {e}")]
    except KnowledgeBaseError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        # Log unexpected errors
        logger.exception("Unexpected error in tool call")
        return [TextContent(type="text", text=f"Unexpected error: {str(e)}")]
```

### P1: Add Logging
```python
import logging

logging.basicConfig(
    filename=DATA_DIR / 'server.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def search(self, query: str, ...):
    logger.info(f"Search query: '{query}', tags={tags}")
    results = # ... search logic ...
    logger.info(f"Found {len(results)} results")
    return results
```

### P1: Index Validation & Repair
```python
def validate_index(self) -> list[str]:
    """Validate index integrity, return list of issues."""
    issues = []

    # Check for orphaned chunks
    for chunk_file in self.chunks_dir.glob("*.json"):
        doc_id = chunk_file.stem
        if doc_id not in self.documents:
            issues.append(f"Orphaned chunks for doc_id: {doc_id}")

    # Check for missing chunk files
    for doc_id in self.documents:
        chunk_file = self.chunks_dir / f"{doc_id}.json"
        if not chunk_file.exists():
            issues.append(f"Missing chunks for doc_id: {doc_id}")

    return issues

def repair_index(self):
    """Attempt to repair index issues."""
    # Remove orphaned chunks, re-index missing documents
```

---

## 6. Observability & Monitoring

### P1: Add Metrics Collection
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SearchMetrics:
    query: str
    results_count: int
    search_time_ms: float
    timestamp: datetime

class KnowledgeBase:
    def __init__(self, data_dir):
        self.metrics = []

    def search(self, query: str, ...):
        start_time = time.time()
        results = # ... search ...
        elapsed_ms = (time.time() - start_time) * 1000

        self.metrics.append(SearchMetrics(
            query=query,
            results_count=len(results),
            search_time_ms=elapsed_ms,
            timestamp=datetime.now()
        ))

        return results
```

### P2: Add Search Analytics Tool
```python
Tool(
    name="search_analytics",
    description="Get analytics about search queries",
    inputSchema={
        "properties": {
            "days": {"type": "integer", "default": 7}
        }
    }
)

# Returns:
# - Most common queries
# - Queries with no results
# - Average search time
# - Most searched tags
```

---

## 7. Developer Experience

### P1: Bulk Operations
```python
Tool(
    name="add_documents_bulk",
    description="Add multiple documents at once",
    inputSchema={
        "properties": {
            "directory": {"type": "string"},
            "pattern": {"type": "string", "default": "**/*.pdf"},
            "tags": {"type": "array"}
        }
    }
)
```

### P1: Progress Reporting for Long Operations
```python
async def add_document(self, filepath: str, progress_callback=None):
    if progress_callback:
        await progress_callback("Extracting text...", 0.2)
    text = self._extract_pdf_text(filepath)

    if progress_callback:
        await progress_callback("Creating chunks...", 0.5)
    chunks = self._chunk_text(text)

    # ... etc
```

### P2: Configuration File Support
```yaml
# config.yml
knowledge_base:
  chunk_size: 1500
  chunk_overlap: 200
  search:
    algorithm: bm25
    default_max_results: 10
  semantic_search:
    enabled: true
    model: all-MiniLM-L6-v2
```

---

## 8. Testing Improvements

### P1: Add Performance Benchmarks
```python
# tests/benchmark.py
def test_search_performance():
    """Search should complete in <100ms for typical queries."""
    kb = setup_kb_with_1000_docs()

    start = time.time()
    results = kb.search("VIC-II register")
    elapsed = time.time() - start

    assert elapsed < 0.1, f"Search took {elapsed}s, expected <0.1s"
```

### P1: Add Integration Tests for MCP Protocol
```python
async def test_mcp_tool_call():
    """Test actual MCP tool invocation."""
    result = await call_tool("search_docs", {"query": "SID"})
    assert len(result) > 0
    assert "SID" in result[0].text
```

---

## 9. Security Considerations

### P0: Path Traversal Protection
**Current Issue**: `add_document` could access files outside intended directory

**Recommendations**:
```python
def add_document(self, filepath: str, ...):
    filepath = Path(filepath).resolve()

    # Validate path doesn't escape allowed directories
    if allowed_dirs:
        if not any(filepath.is_relative_to(d) for d in allowed_dirs):
            raise SecurityError("Path outside allowed directories")
```

### P1: Resource Limits
```python
# Prevent abuse
MAX_CHUNK_SIZE = 10_000  # words
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_SEARCH_RESULTS = 100

def add_document(self, filepath: str, ...):
    file_size = os.path.getsize(filepath)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {file_size} bytes")
```

---

## Recommended Implementation Order

### Phase 1: Foundation (1-2 weeks)
1. Add logging (P1)
2. Better error handling (P0)
3. Track PDF page numbers (P0)
4. SQLite database migration (P0)

### Phase 2: Search Quality (1-2 weeks)
5. Implement BM25 search (P0)
6. Add phrase search (P1)
7. Query preprocessing (P1)
8. Highlight search terms (P1)

### Phase 3: Advanced Features (2-3 weeks)
9. Semantic search with embeddings (P0)
10. SQLite FTS5 integration (P1)
11. Caching layer (P1)
12. Update detection (P1)

### Phase 4: Polish (1 week)
13. Bulk operations (P1)
14. Progress reporting (P1)
15. Search analytics (P2)
16. Configuration file (P2)

---

## Quick Wins (Can Implement in <1 Day Each)

1. **Add logging** - Copy-paste logging setup
2. **Highlight search terms** - Simple regex replacement
3. **Better error messages** - Add custom exception classes
4. **Document metadata extraction** - pypdf already provides this
5. **Configuration from environment variables** - Add more env vars
6. **Duplicate detection** - Change ID generation to content-based hash
7. **Index validation tool** - Add CLI command for health check

---

## Breaking Changes to Consider

If you're willing to accept breaking changes for better long-term architecture:

1. **Change storage from JSON to SQLite** - Requires data migration
2. **Redesign doc_id generation** - Old IDs won't work
3. **Change search result format** - Add more metadata fields
4. **Split into multiple files** - Better organization (server.py, search.py, storage.py, models.py)

---

## Resources & Libraries

**Search & NLP**:
- `rank-bm25` - BM25 algorithm
- `sentence-transformers` - Semantic search
- `spacy` or `nltk` - Text processing
- `rapidfuzz` - Fuzzy matching

**Storage & Performance**:
- `sqlite3` (built-in) - Database
- `faiss-cpu` - Vector similarity search
- `chromadb` - Vector database alternative

**PDF Processing**:
- `pytesseract` - OCR
- `pdfplumber` - Better PDF parsing

**Monitoring**:
- `prometheus-client` - Metrics export
- `structlog` - Structured logging
