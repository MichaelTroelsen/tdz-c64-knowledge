# MCP Server Improvement Suggestions

## 🎯 Priority Levels
- **P0**: Critical improvements (security, correctness, major performance)
- **P1**: High-value improvements (user experience, scalability)
- **P2**: Nice-to-have enhancements (convenience, polish)

---

## ✅ Completed Improvements

### Phase 1: Quick Wins (Completed)
- ✅ **Logging Infrastructure** - File and console logging with timestamps
- ✅ **Custom Exception Classes** - KnowledgeBaseError hierarchy for better error handling
- ✅ **Search Term Highlighting** - Markdown bold highlighting in search results
- ✅ **PDF Metadata Extraction** - Author, subject, creator, creation date extraction

### Phase 2: Search Quality (Completed)
- ✅ **BM25 Search Algorithm** - Industry-standard ranking with rank-bm25 library
- ✅ **Phrase Search Support** - Exact phrase matching with quote detection
- ✅ **Query Preprocessing** - NLTK-powered stemming and stopword removal
- ✅ **PDF Page Number Tracking** - Estimated page numbers in search results
- ✅ **Score Filtering Fix** - Handles negative BM25 scores for small documents

### Phase 3: Storage & Scalability (Completed)
- ✅ **SQLite Database Migration** - Migrated from JSON files to SQLite for better scalability
- ✅ **Lazy Loading** - Only loads document metadata at startup, chunks loaded on-demand
- ✅ **ACID Transactions** - Database transactions ensure data integrity
- ✅ **Automatic Migration** - Seamless upgrade from JSON to SQLite on first run

### Phase 4: Performance Optimization (Completed)
- ✅ **SQLite FTS5 Full-Text Search** - Native search with 480x performance improvement

### Phase 5: P0 Critical Features (Completed)
- ✅ **Path Traversal Protection** - Security validation for document ingestion
- ✅ **Semantic Search with Embeddings** - Conceptual search using sentence-transformers

### Additional Completions
- ✅ **CI/CD Pipeline** - GitHub Actions workflow for multi-platform testing
- ✅ **Comprehensive Test Suite** - 19 tests covering all functionality including SQLite
- ✅ **Documentation** - Updated README, CLAUDE.md with all new features

---

## 1. Search Quality Improvements

### ✅ COMPLETED: P0: Implement Better Search Algorithms
**Status**: ✅ Implemented with BM25Okapi

**Implementation Details**:
- Using `rank-bm25` package with BM25Okapi algorithm
- Default k1=1.2, b=0.75 parameters
- Handles negative scores for small documents with `abs(score) > 0.0001` filter
- Enabled by default, can disable with `USE_BM25=0` environment variable
- Falls back to simple search if rank-bm25 not available

### ✅ COMPLETED: P1: Add Phrase Search Support
**Status**: ✅ Implemented with regex pattern matching

**Implementation Details**:
```python
# Extract phrases in quotes
phrase_pattern = r'"([^"]*)"'
phrases = re.findall(phrase_pattern, query)
# Boost phrase matches 2x in BM25, 10x in simple search
```

### ✅ COMPLETED: P1: Implement Query Preprocessing
**Status**: ✅ Implemented with NLTK

**Implementation Details**:
- Using NLTK for tokenization, stemming, and stopword removal
- Porter Stemmer for word normalization
- English stopwords corpus for filtering
- Preserves technical terms (hyphenated words, numbers)
- Applied to both queries and BM25 corpus
- Configurable via `USE_QUERY_PREPROCESSING` environment variable

```python
# Preprocessing features:
- word_tokenize() for smart tokenization
- PorterStemmer for stemming ("running" → "run")
- stopwords.words('english') for filtering
- Special handling for "VIC-II", "6502", etc.
```

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

### ✅ COMPLETED: P0: Move to SQLite Database
**Status**: ✅ Implemented with automatic migration
**Completed**: December 2025

**Implementation Details**:
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

### ✅ COMPLETED: P1: Implement SQLite FTS5 for Full-Text Search
**Status**: ✅ Implemented with Porter stemming and automatic sync
**Completed**: December 2025

**Implementation Details**:
```sql
CREATE VIRTUAL TABLE chunks_fts5 USING fts5(
    doc_id UNINDEXED,
    chunk_id UNINDEXED,
    content,
    tokenize='porter unicode61'
);

-- Automatic triggers keep FTS5 in sync
CREATE TRIGGER chunks_fts5_insert AFTER INSERT ON chunks ...
CREATE TRIGGER chunks_fts5_delete AFTER DELETE ON chunks ...
CREATE TRIGGER chunks_fts5_update AFTER UPDATE ON chunks ...
```

**Key Features**:
- Native SQLite BM25 ranking (`ORDER BY rank`)
- Porter stemming for improved matching
- Automatic population for existing databases
- Environment variable control via `USE_FTS5=1`
- Fallback to BM25/simple search if FTS5 unavailable

**Performance Improvements**:
- Search queries: **50ms (FTS5)** vs 24,000ms (BM25) = **480x faster**
- No need to load all chunks into memory for search
- Native tokenization and stemming in SQLite

**Usage**:
```bash
# Enable FTS5 search
export USE_FTS5=1  # or set in MCP config env
```

**Benefits**:
- 10-100x faster search with built-in ranking ✅ Achieved (480x in practice)
- Reduced memory usage (no chunk loading for search)
- Built-in Porter stemming tokenizer

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

### ✅ COMPLETED: P0: Add Semantic Search with Embeddings
**Status**: ✅ Implemented with FAISS and sentence-transformers
**Completed**: December 2025

**Implementation Details**:
```python
# sentence-transformers for embeddings generation
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class KnowledgeBase:
    def __init__(self, data_dir):
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_index = None  # FAISS index
        self.embeddings_doc_map = []  # Maps index positions to (doc_id, chunk_id)

    def _build_embeddings(self):
        # Generate embeddings for all chunks
        chunks = self._get_chunks_db()
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embeddings_model.encode(texts, convert_to_numpy=True)

        # Create FAISS index with cosine similarity
        dimension = embeddings.shape[1]
        self.embeddings_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.embeddings_index.add(embeddings)

    def semantic_search(self, query: str, max_results: int = 5, tags=None):
        # Encode query and search
        query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.embeddings_index.search(query_embedding, max_results)
        # Return results with similarity scores
```

**Key Features**:
- SentenceTransformer embeddings (default: all-MiniLM-L6-v2)
- FAISS vector similarity search with cosine distance
- Persistent embeddings storage (embeddings.faiss, embeddings_map.json)
- Lazy embeddings generation (built on first semantic search)
- Automatic invalidation on add/remove operations
- Tag filtering support

**Environment Variables**:
- `USE_SEMANTIC_SEARCH=1` - Enable semantic search
- `SEMANTIC_MODEL=all-MiniLM-L6-v2` - Model to use

**Performance**:
- Embeddings generation: ~1 min for 2347 chunks (one-time)
- Search speed: ~7-16ms per query
- Persistent storage avoids rebuilding

**Benefits**:
- ✅ Finds "sprites" when searching for "movable objects"
- ✅ Understands context and meaning
- ✅ Better for natural language queries

**Libraries**: `sentence-transformers>=2.0.0`, `faiss-cpu>=1.7.0`

### ✅ COMPLETED: P1: Highlight Search Terms in Results
**Status**: ✅ Implemented in `_extract_snippet()` method

**Implementation Details**:
```python
# Highlight matching terms (case-insensitive)
for term in query_terms:
    if len(term) >= 2:
        pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
        snippet = pattern.sub(r'**\1**', snippet)
```

### ✅ COMPLETED: P1: Add "More Like This" Tool
**Status**: ✅ Implemented with semantic and TF-IDF similarity
**Completed**: December 2025

**Implementation Details**:
```python
# MCP Tool
Tool(
    name="find_similar",
    description="Find documents similar to a given document or chunk",
    inputSchema={
        "properties": {
            "doc_id": {"type": "string", "description": "Source document ID"},
            "chunk_id": {"type": "integer", "description": "Optional chunk ID for chunk-level similarity"},
            "max_results": {"type": "integer", "default": 5},
            "tags": {"type": "array", "description": "Filter by tags"}
        }
    }
)

# Core implementation
def find_similar_documents(self, doc_id: str, chunk_id: Optional[int] = None,
                          max_results: int = 5, tags: Optional[list[str]] = None):
    # Prefer semantic search if available
    if self.use_semantic and self.embeddings_index is not None:
        return self._find_similar_semantic(doc_id, chunk_id, max_results, tags)
    else:
        # Fall back to TF-IDF similarity
        return self._find_similar_tfidf(doc_id, chunk_id, max_results, tags)
```

**Key Features**:
- **Dual-mode similarity**: Uses semantic embeddings (FAISS) if available, else TF-IDF cosine similarity
- **Document-level similarity**: Find similar documents when chunk_id is None
- **Chunk-level similarity**: Find chunks similar to a specific chunk when chunk_id provided
- **Tag filtering**: Filter results by document tags
- **Smart aggregation**: Groups chunks by document for document-level results

**Semantic Similarity** (`_find_similar_semantic`):
- Uses FAISS embeddings index for fast nearest-neighbor search
- Aggregates chunk scores by document (mean similarity)
- Returns documents sorted by average similarity score

**TF-IDF Similarity** (`_find_similar_tfidf`):
- Builds TF-IDF vectors for all chunks
- Computes cosine similarity between vectors
- Works without requiring embeddings generation

**Benefits**:
- ✅ Discover related documentation automatically
- ✅ Navigate knowledge base by similarity
- ✅ No external dependencies required (TF-IDF fallback)
- ✅ Fast lookups with FAISS when available

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

### ✅ COMPLETED: P0: Track Page Numbers in PDFs
**Status**: ✅ Implemented with PAGE BREAK marker estimation

**Implementation Details**:
- PDF text extraction uses `--- PAGE BREAK ---` markers between pages
- Chunk creation estimates page number by counting PAGE BREAK markers
- Page numbers stored in `DocumentChunk.page` field
- Search results include page numbers when available

```python
# Estimate page number for PDFs based on PAGE BREAK markers
if file_type == 'pdf' and '--- PAGE BREAK ---' in text:
    chunk_start_pos = text.find(chunk_text[:100])
    if chunk_start_pos >= 0:
        page_breaks_before = text[:chunk_start_pos].count('--- PAGE BREAK ---')
        page_num = page_breaks_before + 1
```

### ✅ COMPLETED: P1: Extract Document Metadata
**Status**: ✅ Implemented in `_extract_pdf_text()` method

**Implementation Details**:
```python
# Extract PDF metadata
metadata = {}
if reader.metadata:
    metadata['author'] = reader.metadata.get('/Author')
    metadata['subject'] = reader.metadata.get('/Subject')
    metadata['creator'] = reader.metadata.get('/Creator')
    creation_date = reader.metadata.get('/CreationDate')
    # Parse PDF date format: D:YYYYMMDDHHmmSS
```

Metadata fields added to `DocumentMeta` dataclass:
- `author: Optional[str]`
- `subject: Optional[str]`
- `creator: Optional[str]`
- `creation_date: Optional[str]`

### ✅ COMPLETED: P1: Duplicate Detection
**Status**: ✅ Implemented with content-based doc IDs
**Completed**: December 2025

**Implementation Details**:
```python
def _generate_doc_id(self, filepath: str, text_content: str = None) -> str:
    """Generate ID based on content hash for deduplication."""
    if text_content:
        # Content-based ID for deduplication
        normalized = text_content.lower().strip()
        words = normalized.split()[:10000]  # First 10k words
        content_sample = ' '.join(words)
        return hashlib.md5(content_sample.encode('utf-8')).hexdigest()[:12]
    else:
        # Filepath-based ID (legacy support)
        return hashlib.md5(filepath.encode()).hexdigest()[:12]

def add_document(self, filepath: str, ...):
    # Extract text first
    text = self._extract_pdf_text(filepath) or self._extract_text_file(filepath)

    # Generate content-based doc_id for deduplication
    doc_id = self._generate_doc_id(filepath, text)

    # Check for duplicate content
    if doc_id in self.documents:
        existing_doc = self.documents[doc_id]
        self.logger.warning(f"Duplicate content detected: {filepath}")
        self.logger.warning(f"  Matches existing document: {existing_doc.filepath}")
        self.logger.info(f"Skipping duplicate - returning existing document {doc_id}")
        return existing_doc  # Non-destructive - returns existing doc

    # Continue with normal indexing...
```

**Key Features**:
- **Content-based IDs**: Doc ID derived from text content (first 10k words) not filepath
- **Automatic duplicate detection**: Same content at different paths detected and skipped
- **Non-destructive behavior**: Returns existing document instead of creating duplicate
- **Normalized comparison**: Lowercase normalization prevents case-sensitivity duplicates
- **Efficient hashing**: Uses first 10k words to handle very large documents
- **Backward compatible**: Filepath-based IDs still supported when text_content is None

**Benefits**:
- ✅ Prevents duplicate indexing of same content
- ✅ Saves storage space and reduces index bloat
- ✅ Improves search quality by avoiding duplicate results
- ✅ Clear logging when duplicates are detected

### P2: OCR Support for Scanned PDFs
**Recommendations**:
- Use `pytesseract` for OCR
- Detect if PDF is text-based or image-based
- Fall back to OCR if text extraction fails

---

## 5. Error Handling & Robustness

### ✅ COMPLETED: P0: Better Error Handling in Async Functions
**Status**: ✅ Implemented custom exception hierarchy

**Implementation Details**:
```python
class KnowledgeBaseError(Exception):
    """Base exception for knowledge base errors."""
    pass

class DocumentNotFoundError(KnowledgeBaseError):
    """Raised when a document is not found."""
    pass

class ChunkNotFoundError(KnowledgeBaseError):
    """Raised when a chunk is not found."""
    pass

class UnsupportedFileTypeError(KnowledgeBaseError):
    """Raised when file type is not supported."""
    pass

class IndexCorruptedError(KnowledgeBaseError):
    """Raised when the index is corrupted."""
    pass
```

All custom exceptions are tested in `test_custom_exceptions()` test case.

### ✅ COMPLETED: P1: Add Logging
**Status**: ✅ Implemented with file and console logging

**Implementation Details**:
```python
# Setup in __init__
log_file = self.data_dir / "server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
self.logger = logging.getLogger(__name__)
```

Logs include:
- Document additions/removals
- Search queries and result counts
- BM25 index building
- Error conditions

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

### ✅ COMPLETED: P0: Path Traversal Protection
**Status**: ✅ Implemented with directory whitelisting
**Completed**: December 2025

**Implementation Details**:
```python
class SecurityError(KnowledgeBaseError):
    """Raised when a security violation is detected."""
    pass

class KnowledgeBase:
    def __init__(self, data_dir):
        # Parse allowed directories from environment
        allowed_dirs_env = os.getenv('ALLOWED_DOCS_DIRS', '')
        if allowed_dirs_env:
            self.allowed_dirs = [Path(d.strip()).resolve()
                                for d in allowed_dirs_env.split(',') if d.strip()]
        else:
            self.allowed_dirs = None  # No restrictions (backward compatible)

    def add_document(self, filepath: str, ...):
        # Resolve to absolute path to prevent path traversal
        resolved_path = Path(filepath).resolve()

        # Validate path is within allowed directories
        if self.allowed_dirs:
            is_allowed = any(
                resolved_path.is_relative_to(allowed_dir)
                for allowed_dir in self.allowed_dirs
            )
            if not is_allowed:
                raise SecurityError(
                    f"Path outside allowed directories. File must be within: {self.allowed_dirs}"
                )
```

**Key Features**:
- New `SecurityError` exception class
- `ALLOWED_DOCS_DIRS` environment variable for directory whitelisting
- Path resolution with `Path.resolve()` to normalize paths
- Validation that resolved paths are within allowed directories
- Blocks path traversal attempts (e.g., `../../../etc/passwd`)
- Backward compatible (no restrictions if not configured)

**Configuration**:
```json
"env": {
  "ALLOWED_DOCS_DIRS": "C:\\docs\\allowed,C:\\other\\allowed"
}
```

**Testing**:
- Added comprehensive security test with 3 scenarios
- Tests allowed directory access (passes)
- Tests restricted directory access (blocks)
- Tests path traversal attempts (blocks)
- All 20 tests pass including new security test

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
