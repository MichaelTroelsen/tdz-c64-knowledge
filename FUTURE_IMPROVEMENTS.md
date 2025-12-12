# Future Improvements for TDZ C64 Knowledge Base

Additional enhancement suggestions building on the current state (145 documents, FTS5 + semantic search enabled).

**Last Updated:** December 2025
**Current State:** Production-ready with FTS5, semantic search, OCR, duplicate detection, and comprehensive test coverage.

---

## 🎯 Priority 1: High-Impact, Quick Wins

### 1.1 Hybrid Search (Combine FTS5 + Semantic)
**Impact:** ⭐⭐⭐⭐⭐ | **Effort:** ⭐⭐ | **Time:** 2-4 hours

**Current State:** FTS5 and semantic work independently
**Proposed:** Combine results for better precision + recall

```python
def hybrid_search(self, query: str, max_results: int = 10,
                 semantic_weight: float = 0.3, tags=None):
    """
    Combine FTS5 (keyword) and semantic (meaning) search.

    Args:
        semantic_weight: 0.0 = pure keyword, 1.0 = pure semantic, 0.3 = balanced
    """
    # Get both result sets (2x max to allow for merging)
    fts_results = self.search(query, max_results=max_results*2, tags=tags)
    sem_results = self.semantic_search(query, max_results=max_results*2, tags=tags)

    # Normalize scores to 0-1 range
    fts_normalized = self._normalize_scores(fts_results)
    sem_normalized = self._normalize_scores(sem_results)

    # Merge with weighted scores
    merged = {}
    for result in fts_normalized:
        doc_id = result['doc_id']
        merged[doc_id] = {
            'fts_score': result['score'],
            'sem_score': 0.0,
            **result
        }

    for result in sem_normalized:
        doc_id = result['doc_id']
        if doc_id in merged:
            merged[doc_id]['sem_score'] = result['score']
        else:
            merged[doc_id] = {
                'fts_score': 0.0,
                'sem_score': result['score'],
                **result
            }

    # Calculate hybrid score
    for doc_id, data in merged.items():
        data['hybrid_score'] = (
            (1 - semantic_weight) * data['fts_score'] +
            semantic_weight * data['sem_score']
        )

    # Sort and return top N
    sorted_results = sorted(merged.values(),
                          key=lambda x: x['hybrid_score'],
                          reverse=True)
    return sorted_results[:max_results]
```

**Benefits:**
- Best of both worlds: exact matching + conceptual understanding
- Configurable balance (tune for your use case)
- Better handles typos, synonyms, and technical terms simultaneously
- Example: "6502 assembler" finds both exact matches AND related content about machine code, opcodes, etc.

**MCP Tool:**
```python
Tool(
    name="hybrid_search",
    description="Search using combined keyword + semantic understanding",
    inputSchema={
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 10},
            "semantic_weight": {"type": "number", "default": 0.3, "minimum": 0, "maximum": 1},
            "tags": {"type": "array"}
        }
    }
)
```

---

### 1.2 Query Autocompletion
**Impact:** ⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 4-6 hours

**Proposed:** Suggest queries based on indexed content

```sql
-- New table for autocomplete dictionary
CREATE VIRTUAL TABLE query_suggestions USING fts5(
    term,
    frequency,  -- How often it appears
    category    -- hardware, instruction, register, etc.
);
```

```python
def build_suggestion_dictionary(self):
    """Extract common terms/phrases for autocomplete."""
    # Extract important terms from all chunks
    terms = defaultdict(int)

    for chunk in self._get_chunks_db():
        # Extract technical terms (ALL CAPS, $-prefixed, hyphenated)
        tech_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)?\b', chunk.content)  # VIC-II, SID
        registers = re.findall(r'\$[0-9A-F]{4}', chunk.content)  # $D000
        instructions = re.findall(r'\b(?:LDA|STA|JMP|JSR|RTS|...)\b', chunk.content)

        for term in tech_terms + registers + instructions:
            terms[term] += 1

    # Store top N terms in suggestions table
    for term, freq in sorted(terms.items(), key=lambda x: x[1], reverse=True)[:1000]:
        # categorize and store

def get_query_suggestions(self, partial: str, max_suggestions: int = 5):
    """Get autocomplete suggestions for partial query."""
    cursor = self.db_conn.cursor()
    cursor.execute("""
        SELECT term, frequency, category
        FROM query_suggestions
        WHERE term MATCH ?
        ORDER BY rank, frequency DESC
        LIMIT ?
    """, (f"{partial}*", max_suggestions))

    return [{'term': row[0], 'category': row[2]} for row in cursor.fetchall()]
```

**Example:**
- User types: "VIC"
- Suggestions: ["VIC-II sprites", "VIC-II registers", "VIC-II programming", "VIC chip", "VIC-II colors"]

**Benefits:**
- Discover searchable content
- Reduce typos
- Learn proper terminology (VIC-II vs VIC II vs VIC2)

**MCP Tool:**
```python
Tool(name="suggest_queries", ...)
```

---

### 1.3 Enhanced Result Snippets
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐ | **Time:** 2-3 hours

**Current:** Basic text extraction with keyword highlighting
**Proposed:** Context-aware snippets

```python
def _extract_smart_snippet(self, chunk_text: str, query: str,
                          context_window: int = 200) -> str:
    """Extract optimal snippet with better context."""
    query_terms = set(query.lower().split())

    # Find best position (highest term density)
    best_score = 0
    best_start = 0

    words = chunk_text.split()
    for i in range(len(words) - context_window):
        window = ' '.join(words[i:i+context_window]).lower()
        score = sum(window.count(term) for term in query_terms)

        if score > best_score:
            best_score = score
            best_start = i

    # Extract full sentences around best position
    snippet = ' '.join(words[max(0, best_start-20):best_start+context_window+20])

    # Special handling for code blocks
    if '    ' in snippet or snippet.count('\n') > 3:  # Likely code
        # Preserve formatting
        lines = snippet.split('\n')
        # Find lines with matches
        matched_lines = [l for l in lines if any(term in l.lower() for term in query_terms)]
        snippet = '\n'.join(matched_lines[:5])  # Up to 5 lines of code

    # Add ellipsis
    if best_start > 20:
        snippet = '...' + snippet
    if best_start + context_window + 20 < len(words):
        snippet = snippet + '...'

    return snippet
```

**Benefits:**
- Better context (complete sentences)
- Code blocks stay formatted
- Multiple snippets if term appears often
- Related terms highlighted (not just exact matches)

---

## Priority 2: Search & Discovery Enhancements

### 2.1 Faceted Search & Filtering ✅ **COMPLETED in v2.2.0**
**Impact:** ⭐⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 6-8 hours

**Proposed:** Multi-dimensional filtering

```python
def faceted_search(self, query: str, facets: dict, max_results: int = 10):
    """
    Search with multiple filters.

    facets = {
        "tags": ["assembly", "reference"],
        "file_type": ["pdf"],
        "page_range": {"min": 100, "max": 500},
        "hardware": ["SID", "VIC-II"],  # extracted from content
        "year": [1983, 1984]  # from metadata
    }
    """
    # Build WHERE clause from facets
    # ...

# Extract facet values during indexing
def _extract_facets(self, text: str) -> dict:
    """Extract categorizable terms for faceting."""
    return {
        'hardware': self._extract_hardware_refs(text),
        'instructions': self._extract_6502_instructions(text),
        'registers': self._extract_registers(text),
    }

def _extract_hardware_refs(self, text: str) -> list[str]:
    """Find hardware component mentions."""
    hardware = []
    patterns = {
        'SID': r'\b(?:SID|6581|Sound\s+Interface\s+Device)\b',
        'VIC-II': r'\b(?:VIC-II|VIC\s*2|6569|6567)\b',
        'CIA': r'\b(?:CIA|6526|Complex\s+Interface\s+Adapter)\b',
        '6502': r'\b6502\b',
    }

    for component, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            hardware.append(component)

    return hardware
```

**New Schema:**
```sql
CREATE TABLE document_facets (
    doc_id TEXT,
    facet_type TEXT,  -- 'hardware', 'instruction', 'register'
    facet_value TEXT,
    PRIMARY KEY (doc_id, facet_type, facet_value)
);
```

**MCP Tool:**
```python
Tool(
    name="faceted_search",
    description="Search with multiple filters (tags, hardware, year, etc.)",
    ...
)
```

**Benefits:**
- Narrow large result sets quickly
- Explore by category ("Show me all SID programming documents")
- Power-user feature

---

### 2.2 Search History & Analytics ✅ **COMPLETED in v2.2.0**
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 4-6 hours

**Proposed:** Learn from usage patterns

```sql
CREATE TABLE search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    query TEXT,
    search_mode TEXT,  -- fts5, semantic, hybrid
    results_count INTEGER,
    clicked_doc_id TEXT,  -- which result was used
    user_agent TEXT  -- MCP client identifier
);

CREATE INDEX idx_search_log_query ON search_log(query);
CREATE INDEX idx_search_log_timestamp ON search_log(timestamp);
```

```python
def log_search(self, query: str, mode: str, results: list, clicked_doc: str = None):
    """Log search for analytics."""
    cursor = self.db_conn.cursor()
    cursor.execute("""
        INSERT INTO search_log (query, search_mode, results_count, clicked_doc_id)
        VALUES (?, ?, ?, ?)
    """, (query, mode, len(results), clicked_doc))
    self.db_conn.commit()

def get_search_analytics(self, days: int = 7) -> dict:
    """Analyze recent search patterns."""
    cursor = self.db_conn.cursor()

    # Most common queries
    cursor.execute("""
        SELECT query, COUNT(*) as count
        FROM search_log
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        GROUP BY query
        ORDER BY count DESC
        LIMIT 10
    """, (days,))
    popular_queries = cursor.fetchall()

    # Queries with no results
    cursor.execute("""
        SELECT DISTINCT query
        FROM search_log
        WHERE results_count = 0
          AND timestamp > datetime('now', '-' || ? || ' days')
        LIMIT 10
    """, (days,))
    no_results = cursor.fetchall()

    # Average results per mode
    cursor.execute("""
        SELECT search_mode, AVG(results_count) as avg_results
        FROM search_log
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        GROUP BY search_mode
    """, (days,))
    mode_performance = cursor.fetchall()

    return {
        'popular_queries': popular_queries,
        'no_results_queries': no_results,
        'search_mode_stats': mode_performance
    }
```

**MCP Tool:**
```python
Tool(name="search_analytics", description="View search usage patterns and trends", ...)
```

**Benefits:**
- Identify knowledge gaps (queries with no results)
- Popular topics
- Improve autocomplete suggestions
- Optimize search modes

**Privacy:** All data stays local, never transmitted.

---

## Priority 3: Performance & Scalability

### 3.1 Incremental Embeddings ✅ COMPLETED in v2.3.0
**Impact:** ⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 4-6 hours

**Current:** Full rebuild when adding documents
**Proposed:** Incremental updates

```python
def add_document(self, filepath: str, ...):
    doc = # ... normal add ...

    # Generate embeddings for new chunks only
    if self.use_semantic and self.embeddings_index is not None:
        new_chunks = self._get_chunks_db(doc.doc_id)
        self._add_chunks_to_embeddings(new_chunks)

    return doc

def _add_chunks_to_embeddings(self, chunks: list[DocumentChunk]):
    """Add new chunks to existing FAISS index."""
    if not chunks:
        return

    # Generate embeddings for new chunks
    texts = [c.content for c in chunks]
    new_embeddings = self.embeddings_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(new_embeddings)

    # Add to existing index
    self.embeddings_index.add(new_embeddings)

    # Update mapping
    for chunk in chunks:
        self.embeddings_doc_map.append((chunk.doc_id, chunk.chunk_id))

    # Save updated index
    self._save_embeddings()
```

**Benefits:**
- Fast document addition (seconds instead of minutes)
- Scales to 10k+ documents
- No interruption to search

---

### 3.2 Parallel Document Processing ✅ COMPLETED in v2.3.0
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 4-6 hours

**Proposed:** Process multiple PDFs simultaneously

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def add_documents_bulk(self, directory: str, max_workers: int = 4, **kwargs):
    """Process multiple documents in parallel."""
    files = list(Path(directory).glob(kwargs.get('pattern', '**/*.{pdf,txt}')))

    results = {'added': [], 'skipped': [], 'failed': []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(self._add_document_safe, str(f), **kwargs): f
            for f in files
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                doc = future.result()
                results['added'].append(doc)
            except Exception as e:
                results['failed'].append({'file': str(file_path), 'error': str(e)})

    return results

def _add_document_safe(self, filepath: str, **kwargs):
    """Thread-safe wrapper for add_document."""
    # Use thread-local DB connection
    # Or use lock for shared connection
    with self._db_lock:
        return self.add_document(filepath, **kwargs)
```

**Configuration:**
```bash
# Control parallelism
export MAX_WORKERS=4  # CPU count
```

**Benefits:**
- 3-4x faster bulk imports
- Better CPU utilization
- Especially helpful for large collections

---

## Priority 4: Content Enhancements

### 4.1 Table Extraction from PDFs
**Impact:** ⭐⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐⭐⭐ | **Time:** 12-16 hours

**Critical for C64 docs:** Memory maps, register tables, opcode tables

```python
import pdfplumber

def _extract_tables(self, pdf_path: str) -> list[dict]:
    """Extract structured tables from PDF."""
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()

            for table in page_tables:
                if not table or len(table) < 2:
                    continue

                # Convert to structured format
                headers = table[0]
                rows = table[1:]

                tables.append({
                    'page': page_num + 1,
                    'headers': headers,
                    'rows': rows,
                    'markdown': self._table_to_markdown(headers, rows),
                    'text': self._table_to_text(headers, rows)
                })

    return tables

def _table_to_markdown(self, headers: list, rows: list) -> str:
    """Convert table to markdown format."""
    md = '| ' + ' | '.join(headers) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'

    for row in rows:
        md += '| ' + ' | '.join(str(cell or '') for cell in row) + ' |\n'

    return md
```

**Storage:**
```sql
CREATE TABLE document_tables (
    doc_id TEXT,
    table_id INTEGER,
    page INTEGER,
    markdown TEXT,
    searchable_text TEXT,
    PRIMARY KEY (doc_id, table_id)
);

CREATE VIRTUAL TABLE tables_fts USING fts5(
    doc_id UNINDEXED,
    table_id UNINDEXED,
    searchable_text
);
```

**Search Enhancement:**
```python
def search_tables(self, query: str, max_results: int = 5):
    """Search specifically in tables."""
    # Use tables_fts for targeted table search
    # Return formatted markdown tables in results
```

**Benefits:**
- ✅ Crucial for C64 memory maps ($D000-$DFFF register tables)
- ✅ Opcode tables
- ✅ Better formatting in results
- ✅ Searchable table content

---

### 4.2 Code Block Detection
**Impact:** ⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 6-8 hours

**Proposed:** Identify and preserve code

```python
def _detect_code_blocks(self, text: str) -> list[dict]:
    """Detect assembly code, BASIC, hex dumps."""
    code_blocks = []

    # Pattern 1: BASIC (line numbers)
    basic_pattern = r'^\d{1,5}\s+[A-Z]+.*$'

    # Pattern 2: Assembly (mnemonics + addressing modes)
    asm_pattern = r'^\s*(?:LDA|STA|JMP|JSR|...)\s+[#$%]?[\w,()]+\s*;?.*$'

    # Pattern 3: Hex dumps
    hex_pattern = r'^[\dA-F]{4}:\s+(?:[\dA-F]{2}\s+)+.*$'

    lines = text.split('\n')
    current_block = []
    block_type = None

    for line in lines:
        if re.match(basic_pattern, line, re.IGNORECASE):
            if block_type != 'basic':
                if current_block:
                    code_blocks.append({'type': block_type, 'code': '\n'.join(current_block)})
                current_block = []
                block_type = 'basic'
            current_block.append(line)

        elif re.match(asm_pattern, line, re.IGNORECASE):
            if block_type != 'assembly':
                if current_block:
                    code_blocks.append({'type': block_type, 'code': '\n'.join(current_block)})
                current_block = []
                block_type = 'assembly'
            current_block.append(line)

        # ... similar for hex dumps ...

    return code_blocks

def _format_code_snippet(self, code: str, lang: str) -> str:
    """Format code for display with syntax highlighting hints."""
    return f"```{lang}\n{code}\n```"
```

**Benefits:**
- Preserve formatting in search results
- "Find assembly examples for LDA instruction"
- Better snippet quality
- Syntax highlighting in supported clients

---

### 4.3 Cross-Reference Detection ✅ COMPLETED in v2.3.0
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 8-10 hours

**Proposed:** Link related content

```python
def _extract_references(self, text: str) -> dict:
    """Extract cross-references: memory addresses, register names, page refs."""
    refs = {
        'memory_addresses': set(),
        'registers': set(),
        'page_numbers': set(),
        'chips': set()
    }

    # Memory addresses: $D000-$D3FF
    refs['memory_addresses'] = set(re.findall(r'\$[0-9A-F]{4}', text))

    # Register references: VIC+0, SID+4
    refs['registers'] = set(re.findall(r'(?:VIC|SID|CIA)\+\d+', text, re.IGNORECASE))

    # Page references: "see page 156"
    refs['page_numbers'] = set(re.findall(r'page\s+(\d+)', text, re.IGNORECASE))

    # Chip mentions
    chips = ['6502', '6581', 'SID', '6569', '6567', 'VIC-II', '6526', 'CIA']
    for chip in chips:
        if chip.lower() in text.lower():
            refs['chips'].add(chip)

    return refs
```

**Schema:**
```sql
CREATE TABLE cross_references (
    doc_id TEXT,
    ref_type TEXT,  -- 'memory', 'register', 'page', 'chip'
    ref_value TEXT,
    chunk_id INTEGER,
    PRIMARY KEY (doc_id, ref_type, ref_value, chunk_id)
);

CREATE INDEX idx_xref_value ON cross_references(ref_type, ref_value);
```

**MCP Tool:**
```python
Tool(
    name="find_by_reference",
    description="Find all documents mentioning a specific address/register/chip",
    inputSchema={
        "properties": {
            "ref_type": {"enum": ["memory", "register", "chip"]},
            "ref_value": {"type": "string"}  # e.g., "$D000" or "SID"
        }
    }
)
```

**Benefits:**
- "Show all docs about $D000"
- Build knowledge graph
- Navigate by hardware component

---

## Priority 5: User Experience

### 5.1 Export Results
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 6-8 hours

**Proposed:** Export search results

```python
def export_search_results(self, results: list, format: str = 'markdown') -> str:
    """Export results to various formats."""

    if format == 'markdown':
        output = f"# Search Results\n\n"
        output += f"**Query:** {results['query']}\n"
        output += f"**Results:** {len(results['items'])}\n\n"

        for i, result in enumerate(results['items'], 1):
            output += f"## {i}. {result['title']}\n\n"
            output += f"**Score:** {result['score']:.2f}\n\n"
            output += f"{result['snippet']}\n\n"
            output += f"**Source:** {result['filename']} (page {result['page']})\n\n"
            output += "---\n\n"

        return output

    elif format == 'json':
        return json.dumps(results, indent=2)

    elif format == 'html':
        # HTML template with styling
        ...
```

**MCP Tool:**
```python
Tool(name="export_results", ...)
```

**Use Cases:**
- Create custom reference guides
- Share findings
- Print for offline use

---

## Priority 6: Operational Excellence

### 6.1 Health Monitoring
**Impact:** ⭐⭐⭐ | **Effort:** ⭐⭐ | **Time:** 3-4 hours

```python
def health_check(self) -> dict:
    """Comprehensive health status."""
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': {'status': 'unknown'},
        'indexes': {},
        'disk': {},
        'performance': {}
    }

    try:
        # Database connectivity
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        health['database'] = {
            'status': 'healthy',
            'documents': doc_count
        }

        # FTS5 index status
        if self.use_fts5:
            cursor.execute("SELECT COUNT(*) FROM chunks_fts")
            fts_count = cursor.fetchone()[0]
            health['indexes']['fts5'] = {
                'status': 'healthy' if fts_count > 0 else 'empty',
                'entries': fts_count
            }

        # Embeddings status
        if self.use_semantic:
            health['indexes']['embeddings'] = {
                'status': 'loaded' if self.embeddings_index else 'not_loaded',
                'vectors': self.embeddings_index.ntotal if self.embeddings_index else 0
            }

        # Disk usage
        data_dir_size = sum(f.stat().st_size for f in Path(self.data_dir).rglob('*'))
        health['disk'] = {
            'data_dir_size_mb': data_dir_size / (1024 * 1024),
            'db_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
        }

    except Exception as e:
        health['status'] = 'unhealthy'
        health['error'] = str(e)

    return health
```

**MCP Tool:**
```python
Tool(name="health_check", description="Get system health status", ...)
```

---

### 6.2 Automated Backup
**Impact:** ⭐⭐⭐⭐ | **Effort:** ⭐⭐⭐ | **Time:** 6-8 hours

```python
import shutil
from datetime import datetime

def create_backup(self, dest_dir: str, compress: bool = True) -> str:
    """Create full backup of knowledge base."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"kb_backup_{timestamp}"
    backup_path = Path(dest_dir) / backup_name

    backup_path.mkdir(parents=True, exist_ok=True)

    # Backup database
    shutil.copy2(self.db_path, backup_path / "knowledge_base.db")

    # Backup embeddings if they exist
    if self.embeddings_index:
        self._save_embeddings_to(backup_path)

    # Export metadata
    metadata = {
        'timestamp': timestamp,
        'document_count': len(self.documents),
        'version': self._get_version()
    }
    with open(backup_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    if compress:
        # Create zip file
        shutil.make_archive(str(backup_path), 'zip', backup_path)
        shutil.rmtree(backup_path)  # Remove uncompressed
        return f"{backup_path}.zip"

    return str(backup_path)

def restore_from_backup(self, backup_path: str):
    """Restore knowledge base from backup."""
    # Extract if compressed
    # Validate backup integrity
    # Restore database
    # Restore embeddings
    # Reload index
```

---

## Summary: Recommended Next Steps

### Immediate (1-2 weeks)
1. **Hybrid Search** - Best ROI, easy to implement
2. **Enhanced Snippets** - Better UX
3. **Health Monitoring** - Operational necessity

### Short Term (1 month)
4. **Query Autocomplete** - Great UX improvement
5. **Faceted Search** - Essential for growing collection
6. **Table Extraction** - Critical for C64 memory maps

### Medium Term (2-3 months)
7. **Search Analytics** - Learn usage patterns
8. **Code Block Detection** - Better for technical docs
9. **Incremental Embeddings** - Scalability
10. **Export Results** - Sharing capability

### Long Term (3-6 months)
11. **Cross-References** - Build knowledge graph
12. **Parallel Processing** - Performance at scale
13. **Automated Backups** - Data safety

---

## Conclusion

The knowledge base is already production-ready with excellent search capabilities. These improvements focus on:

1. **Better Search Quality** - Hybrid, autocomplete, facets
2. **Richer Content** - Tables, code, cross-refs
3. **Scalability** - Incremental updates, parallelization
4. **Operations** - Monitoring, backups, analytics

**Recommended Start:** Hybrid search (2-4 hours, high impact)

Would you like me to implement any of these?
