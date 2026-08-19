# Architecture & Developer Guide

This document provides detailed technical information for developers working on the tdz-c64-knowledge codebase.

For user-facing documentation, see [README.md](README.md).

## Core Architecture

### Main Components

**server.py** - MCP server entry point (576 lines)
- Transports: stdio and streamable HTTP
- `list_tools()` returns `mcp_tools/schemas.py`'s `TOOL_SCHEMAS` (93 `Tool(...)` literals)
- `call_tool()` dispatches via `mcp_tools/handlers.py`'s `HANDLERS` dict (a lookup, not an `elif` chain)
- MCP resource handlers: Exposes documents as `c64kb://` URIs

**kb/** - `KnowledgeBase` class, split into domain mixins
- `kb/core.py` - base class, schema/connection setup, `_add_document_db()`
- `kb/ingest/` - `_documents.py` (`add_document()`, `add_documents_bulk()`), `_extraction.py`, `_tagging.py`
- `kb/search/` - `_retrieval.py` (`search()`), `_query.py`, `_rag.py`
- `kb/entities/` - `_extraction.py`, `_relationships.py`
- `kb/graph.py`, `kb/topics.py`, `kb/temporal.py`, `kb/figures.py`, `kb/admin.py` - remaining domain mixins

**mcp_tools/** - MCP tool layer
- `schemas.py` - the `Tool(...)` literals (`TOOL_SCHEMAS`)
- `handlers.py` - aggregates `HANDLERS` from 8 domain modules: `admin.py`, `documents.py`, `entities.py`, `figures.py`, `knowledge_graph.py`, `search.py`, `temporal.py`, `topics.py`

**cli.py** - Command-line interface for batch operations
- Wraps `KnowledgeBase` for CLI usage
- Commands: `add`, `add-folder`, `search`, `list`, `remove`, `stats`
- Useful for bulk importing documents

**admin_gui.py** - Streamlit entry point (193 lines); dispatches to `admin_pages/`
- `admin_pages/` - 16 page bodies, each exposing `render(kb)`; dispatch is `PAGES[page](kb)`
- `admin_common.py` - 4 shared helpers (`build_bm25_index_background`, `trigger_background_reindex`, `format_bytes`, `format_timestamp`)
- Document management UI, search interface with filters, statistics/health monitoring, URL scraping interface

### Data Storage Model

#### Source-file retention: `uploads/` is permanent storage

**Decision (2026-08-19):** files placed in `<TDZ_DATA_DIR>/uploads/` are
**permanent document storage**, not a staging area. Ingest records the
uploads path in `documents.filepath` and that path is expected to keep
resolving for the life of the document. Nothing in the system should delete,
prune or rotate `uploads/`.

This is recorded because the answer is not derivable from the code, and two
behaviours depend on it:

- **A missing source file is a defect, not housekeeping.** 106 of the 178
  PDFs in the live knowledge base no longer resolve at their recorded
  `documents.filepath`. Under this policy that is a real fault to repair,
  which is why `health_check` reports `missing_source_files` and raises its
  status to `warning` when the count is non-zero - that metric is an alarm,
  not noise.
- **Re-pointing is repair, not relocation.** A tool that re-points a document
  at a moved file is fixing a broken record; the new path must still pass the
  `ALLOWED_DOCS_DIRS` whitelist, and `uploads/` remains a legitimate target
  rather than somewhere files are only passing through.

The indexed text is therefore not the only artifact. Anything that needs the
original bytes - figure OCR, page rasterisation, re-chunking, the PDF viewer
in the exported wiki - depends on the source file still being there.

The knowledge base uses **SQLite database** for efficient storage and querying:

**knowledge_base.db** - SQLite database with 22 tables (the six most central are detailed below; see server.py for the complete schema):

#### Tables Schema

**documents table** - Stores DocumentMeta objects (22 fields):
- Core fields: `doc_id`, `filepath`, `title`, `file_type`, `tags`, `word_count`, `added_date`
- PDF metadata: `author`, `subject`, `creator`, `pdf_creation_date`
- URL scraping fields (v2.14.0): `source_url`, `scrape_date`, `scrape_config`, `scrape_status`, `scrape_error`, `url_last_checked`, `url_content_hash`

**chunks table** - Stores DocumentChunk objects (5 fields):
- `doc_id`, `chunk_id`, `page`, `content`, `word_count`
- Foreign key: references documents.doc_id with CASCADE delete

**document_tables table** - Stores extracted tables from PDFs (7 fields):
- `doc_id`, `table_id`, `page`, `markdown`, `searchable_text`, `row_count`, `col_count`
- Foreign key: references documents.doc_id with CASCADE delete

**document_code_blocks table** - Stores detected code blocks (7 fields):
- `doc_id`, `block_id`, `page`, `block_type`, `code`, `searchable_text`, `line_count`
- Foreign key: references documents.doc_id with CASCADE delete

**document_entities table** - Stores extracted named entities (v2.15.0) (10 fields):
- `doc_id`, `entity_id`, `entity_text`, `entity_type`, `confidence`, `context`, `first_chunk_id`, `occurrence_count`, `generated_at`, `model`
- Foreign key: references documents.doc_id with CASCADE delete
- Primary key: (doc_id, entity_id)

**document_summaries table** - Stores AI-generated summaries (v2.13.0) (7 fields):
- `doc_id`, `summary_type`, `summary_text`, `generated_at`, `model`, `token_count`, `updated_at`
- Foreign key: references documents.doc_id with CASCADE delete
- Primary key: (doc_id, summary_type)

#### Indexes

- FTS5 indexes: `chunks_fts5`, `tables_fts`, `code_fts`, `entities_fts` for full-text search
- Performance indexes: `filepath`, `file_type`, `doc_id`, `source_url`, `scrape_status`, `idx_entities_doc_id`, `idx_entities_type`, `idx_entities_text`
- Summary index: `idx_summaries_doc_id`

#### Migration from JSON

Legacy JSON files (index.json, chunks/*.json) are automatically migrated to SQLite on first run. JSON files are preserved as backup.

#### Lazy Loading

Only document metadata is loaded at startup. Chunks are loaded on-demand for search (building BM25 index) and retrieval operations. This enables the system to scale to 100k+ documents without memory issues.

#### Chunking Strategy

Documents are split into overlapping chunks (default 1500 words, 200 word overlap) to enable granular search and retrieval.

## Document Processing Pipeline

1. **File ingestion** - PDF via pypdf, text files with encoding detection
2. **Text extraction** - Pages joined with "--- PAGE BREAK ---"
3. **Table extraction** - For PDFs, extracts structured tables using pdfplumber, converts to markdown
4. **Code block detection** - Detects BASIC, Assembly, and Hex dump code blocks using regex patterns
5. **Chunking** - Overlapping chunks via `_chunk_text()` method
6. **Content-based ID generation** - doc_id from MD5 hash of normalized text content (first 10k words)
7. **Duplicate detection** - Checks if content hash already exists, returns existing doc if duplicate
8. **Database persistence** - Document + chunks + tables + code blocks inserted in ACID transaction via `_add_document_db()`

### Duplicate Detection Details

- `_generate_doc_id()` accepts optional `text_content` parameter
- If provided, generates content-based hash from normalized text (lowercase, first 10k words)
- In `add_document()`, checks if doc_id already exists in `self.documents`
- If duplicate detected, logs warning and returns existing document (non-destructive)
- Prevents duplicate indexing regardless of file path or filename
- Backward compatible: filepath-based IDs still supported for legacy code

## URL Scraping & Web Content Ingestion (v2.14.0)

**Overview**: Automatically scrape and index documentation websites using the integrated mdscrape tool. Converts HTML documentation to searchable markdown with full metadata tracking.

### Core Methods

- `scrape_url()` - Scrape a website and add all pages to knowledge base
- `rescrape_document()` - Re-scrape an existing URL-sourced document to check for updates
- `check_url_updates()` - Check all URL-sourced documents for changes (HEAD request, Last-Modified header)

### Features

- **Concurrent scraping** with configurable thread pools (default: 10 threads)
- **Smart content extraction** - mdscrape automatically identifies main content and removes navigation/ads
- **Depth control** - Follow links up to configurable depth (default: 50 levels)
- **URL filtering** - Limit scraping to specific URL prefixes with `--limit` parameter
- **Rate limiting** - Configurable delay between requests (default: 100ms)
- **CSS selectors** - Optional custom selectors for content extraction
- **Auto-tagging** - Automatically tags documents with domain name + "scraped"
- **Persistent storage** - Scraped markdown files saved to `scraped_docs/{domain}_{timestamp}/`
- **Update detection** - Tracks Last-Modified headers and content hashes for change detection
- **Re-scraping** - Maintains original scrape configuration for easy updates
- **YAML frontmatter** - Extracts source URLs and titles from scraped markdown

### Security

- Only HTTP/HTTPS URLs supported (blocks file:// and other protocols)
- URL validation with urlparse
- Sanitized domain names for filesystem paths (replaces '.', ':', etc.)
- Subprocess timeout (1 hour maximum)
- Thread-safe database operations

### Performance

- Concurrent scraping with configurable threads (1-20)
- Incremental document addition (processes files as they're scraped)
- Progress callbacks for long operations
- Efficient HEAD requests for update checking

### Dependencies

- **mdscrape** executable required (path: `C:\Users\mit\claude\mdscrape` or set via `MDSCRAPE_PATH` env var)
- Install from: https://github.com/MichaelTroelsen/mdscrape

## Named Entity Extraction (v2.15.0)

**Overview**: AI-powered named entity extraction identifies and catalogs technical entities from C64 documentation using Large Language Models (LLM). Extracts 7 entity types with confidence scoring, occurrence counting, and full-text search capabilities.

### Core Methods

#### extract_entities(doc_id, confidence_threshold=0.6, force_regenerate=False)
Extract named entities from a single document using LLM.

**Implementation** (`kb/entities/_extraction.py`, `extract_entities`):
1. **Validation** - Check document exists
2. **Cache check** - Return existing entities unless force_regenerate
3. **Content sampling** - Sample first 5 chunks (up to 5000 chars for cost control)
4. **LLM prompt construction** - Build detailed prompt with 7 entity categories
5. **LLM call** - Temperature 0.3 for deterministic results, max_tokens 2048
6. **JSON parsing** - Extract entities from LLM response
7. **Confidence filtering** - Filter by threshold (default 0.6)
8. **Deduplication** - Case-insensitive matching, count occurrences
9. **Database storage** - Store in document_entities table with transaction
10. **Return results** - Structured dict with entities grouped by type

**Entity Types Extracted:**
- `hardware` - Chip names (SID, VIC-II, CIA, 6502, 6526, 6581)
- `memory_address` - Memory addresses ($D000, $D020, $0400)
- `instruction` - Assembly instructions (LDA, STA, JMP, JSR, RTS)
- `person` - People mentioned (Bob Yannes, Jack Tramiel)
- `company` - Organizations (Commodore, MOS Technology)
- `product` - Hardware products (VIC-20, C128, 1541)
- `concept` - Technical concepts (sprite, raster interrupt, IRQ)

**Return Structure:**
```python
{
    'doc_id': str,
    'doc_title': str,
    'entities': [
        {
            'entity_text': 'VIC-II',
            'entity_type': 'hardware',
            'confidence': 0.95,
            'context': '...snippet...',
            'occurrence_count': 5
        },
        ...
    ],
    'entity_count': 42,
    'types': {'hardware': 10, 'memory_address': 8, ...}
}
```

#### get_entities(doc_id, entity_types=None, min_confidence=0.0)
Retrieve stored entities from database with optional filtering.

**Implementation** (`kb/entities/_extraction.py`, `get_entities`):
- Query document_entities table
- Filter by entity_types array (optional)
- Filter by min_confidence threshold
- Order by entity_type, then confidence DESC
- Returns same structure as extract_entities()

#### search_entities(query, entity_types=None, min_confidence=0.0, max_results=20)
Search entities across all documents using FTS5 full-text search.

**Implementation** (`kb/entities/_extraction.py`, `search_entities`):
1. **FTS5 query** - Search entities_fts virtual table
2. **Filtering** - Apply entity_types and min_confidence filters
3. **Ranking** - Order by FTS5 rank (relevance)
4. **Grouping** - Group results by document
5. **Enrichment** - Add document titles and match counts
6. **Return** - Documents with matching entities and contexts

**Return Structure:**
```python
{
    'query': str,
    'total_matches': int,
    'documents': [
        {
            'doc_id': str,
            'doc_title': str,
            'matches': [
                {
                    'entity_text': str,
                    'entity_type': str,
                    'confidence': float,
                    'context': str,
                    'occurrence_count': int
                },
                ...
            ],
            'match_count': int
        },
        ...
    ]
}
```

#### find_docs_by_entity(entity_text, entity_type=None, min_confidence=0.0, max_results=20)
Find all documents containing a specific entity (exact match).

**Implementation** (`kb/entities/_extraction.py`, `find_docs_by_entity`):
- Exact text matching on entity_text field
- Optional entity_type and min_confidence filtering
- Order by confidence DESC, occurrence_count DESC
- Returns documents with entity details

#### get_entity_stats(entity_type=None)
Get comprehensive statistics about extracted entities.

**Implementation** (`kb/entities/_extraction.py`, `get_entity_stats`):
- Total entities and documents with entities
- Breakdown by entity type (7 categories)
- Top 20 entities by document count with avg confidence
- Top 10 documents by entity count
- Optional filtering by entity_type

#### extract_entities_bulk(confidence_threshold=0.6, force_regenerate=False, max_docs=None, skip_existing=True)
Bulk extract entities from multiple documents with progress tracking.

**Implementation** (`kb/entities/_extraction.py`, `extract_entities_bulk`):
1. **Document selection** - Get all documents or limit with max_docs
2. **Existing check** - Skip documents that already have entities (unless force_regenerate)
3. **Batch processing** - Process each document with error handling
4. **Statistics** - Track processed/skipped/failed counts
5. **Aggregation** - Aggregate entity counts by type
6. **Return** - Comprehensive results with statistics

### Database Schema

#### document_entities Table
```sql
CREATE TABLE document_entities (
    doc_id TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    context TEXT,
    first_chunk_id INTEGER,
    occurrence_count INTEGER DEFAULT 1,
    generated_at TEXT NOT NULL,
    model TEXT,
    PRIMARY KEY (doc_id, entity_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
)
```

#### entities_fts Virtual Table (FTS5)
```sql
CREATE VIRTUAL TABLE entities_fts USING fts5(
    doc_id UNINDEXED,
    entity_id UNINDEXED,
    entity_text,
    entity_type UNINDEXED,
    context,
    tokenize='porter unicode61'
)
```

#### Triggers (3 total)
- `entities_fts_insert` - Sync INSERT operations to FTS5
- `entities_fts_delete` - Sync DELETE operations to FTS5
- `entities_fts_update` - Sync UPDATE operations to FTS5

#### Indexes (3 total)
- `idx_entities_doc_id` - Fast lookups by document
- `idx_entities_type` - Fast filtering by entity type
- `idx_entities_text` - Fast lookups by entity text

### Features

- **Confidence Scoring** - Each entity has 0.0-1.0 confidence score from LLM
- **Occurrence Counting** - Tracks how many times each entity appears in document
- **Context Snippets** - Stores surrounding text (up to 100 chars) for each entity
- **Database Caching** - Avoids re-extraction unless forced
- **Case-Insensitive Deduplication** - Merges duplicate entities (e.g., "VIC-II" and "vic-ii")
- **Full-Text Search** - FTS5 index enables fast search across entity text and context
- **Type Filtering** - Filter searches and retrievals by entity type
- **Confidence Filtering** - Filter by minimum confidence threshold
- **Bulk Processing** - Process entire knowledge base with skip_existing optimization
- **LLM Provider Support** - Works with Anthropic Claude and OpenAI GPT models

### LLM Integration

**Configuration** (environment variables):
- `LLM_PROVIDER` - "anthropic" or "openai"
- `ANTHROPIC_API_KEY` - Anthropic API key (if using Claude)
- `OPENAI_API_KEY` - OpenAI API key (if using GPT)
- `LLM_MODEL` - Optional model override

**Prompt Engineering:**
- Detailed instructions with 7 entity categories
- Examples for each entity type
- Request for specific JSON format
- Temperature 0.3 for deterministic results
- Max tokens 2048 for response

**Cost Control:**
- Samples only first 5 chunks (not entire document)
- Limits to 5000 characters max
- Database caching prevents re-extraction
- Typical cost: $0.01-0.04 per document

### Security

- Input validation on doc_id, confidence thresholds
- SQL injection prevention via parameterized queries
- Transaction safety with BEGIN/COMMIT
- CASCADE delete ensures referential integrity

### Performance

- **FTS5 Search** - Fast full-text search on entity text and context
- **Lazy Loading** - Entities loaded on-demand, not at startup
- **Indexes** - 3 B-tree indexes for fast filtering and lookups
- **Caching** - Database-backed caching avoids LLM calls
- **Batch Processing** - Efficient bulk extraction with error handling

### MCP Tools (5 tools)

**Tool Definitions** (`mcp_tools/schemas.py`, `TOOL_SCHEMAS` entries for `extract_entities`, `list_entities`, `search_entities`, `entity_stats`, `extract_entities_bulk`):
1. `extract_entities` - Extract from single document
2. `list_entities` - List entities with filtering
3. `search_entities` - Search across all documents
4. `entity_stats` - Statistics dashboard
5. `extract_entities_bulk` - Bulk extraction

**Tool Handlers** (`mcp_tools/entities.py`: `handle_extract_entities`, `handle_list_entities`, `handle_search_entities`, `handle_entity_stats`, `handle_extract_entities_bulk`):
- Formatted markdown output
- Error handling with helpful LLM configuration messages
- Sample results (first N per category)
- Statistics summaries

### CLI Commands (4 commands)

**Command Definitions** (`cli.py`, `main()`: `extract-entities`, `extract-all-entities`, `search-entity`, `entity-stats` subparsers):
1. `extract-entities <doc_id>` - Single document extraction
2. `extract-all-entities` - Bulk extraction
3. `search-entity <query>` - Search for entities
4. `entity-stats` - Show statistics

**Command Handlers** (`cli.py`, `main()`: `extract-entities`, `extract-all-entities`, `search-entity`, `entity-stats` handlers):
- Formatted console output
- Progress indicators
- Error messages with LLM setup instructions
- Statistics displays

## Search Implementation

### Search Algorithm Architecture

Search is implemented in `KnowledgeBase.search()` (`kb/search/_retrieval.py`, `_RetrievalMixin.search`).

**Current Implementation:**
- SQLite FTS5 via `_search_fts5()` method (when USE_FTS5=1, recommended)
- BM25 ranking via `_search_bm25()` method (fallback/default)
- Simple term frequency via `_search_simple()` method (fallback)
- Phrase detection and boosting
- Search term highlighting via `_extract_snippet()`

### Key Methods

**Search Methods:**
- `search()` - Main entry point, dispatches to FTS5, BM25, or simple search based on environment variables
- `semantic_search()` - Semantic/conceptual search using embeddings and FAISS
- `hybrid_search()` - Combines FTS5 + semantic via RRF, configurable weighting (default: 0.7)
- `search_tables()` - Search for tables in PDFs using FTS5 with tag filtering
- `search_code()` - Search for code blocks (BASIC/Assembly/Hex) with type filtering

**Helper Methods:**
- `health_check()` - Comprehensive system diagnostics (database, features, performance)
- `_extract_tables()` - Extract tables from PDFs using pdfplumber, convert to markdown
- `_table_to_markdown()` - Convert table data to markdown format
- `_detect_code_blocks()` - Detect BASIC, Assembly, and Hex dump code blocks via regex
- `_build_embeddings()` - Generates embeddings for all chunks and builds FAISS index
- `_load_embeddings()` - Loads persisted FAISS index from disk
- `_save_embeddings()` - Saves FAISS index to disk
- `_search_fts5()` - SQLite FTS5 search with native BM25 ranking (480x faster)
- `_fts5_available()` - Checks if FTS5 table exists and is ready
- `_search_bm25()` - BM25 scoring with phrase boosting
- `_search_simple()` - Fallback term frequency scoring
- `_build_bm25_index()` - Builds BM25 index from chunks on init/update
- `_extract_snippet()` - Extracts context with term density scoring, complete sentences, code preservation

### Semantic Search with Embeddings

**Implementation:**
- Uses sentence-transformers for generating embeddings
- FAISS vector similarity search with cosine distance
- Finds documents based on meaning, not just keywords
- Performance: ~7-16ms per query after initial embeddings generation
- Embeddings persisted to disk (embeddings.faiss, embeddings_map.json)
- Configurable model via `SEMANTIC_MODEL` (default: all-MiniLM-L6-v2)

### SQLite FTS5 Full-Text Search

**Implementation:**
- Uses SQLite's FTS5 virtual table with Porter stemming tokenizer
- 480x faster than BM25 (50ms vs 24,000ms for typical queries)
- Native BM25 ranking built into SQLite
- No need to load all chunks into memory
- Automatic triggers keep FTS5 in sync with chunks table
- Falls back to BM25/simple search if FTS5 returns no results

### BM25 (Okapi BM25)

**Implementation:**
- Uses rank-bm25 library for accurate relevance scoring
- Handles document length normalization
- Tokenizes documents and queries for matching
- Accepts negative scores for small documents (filters by abs(score) > 0.0001)

### Security - Path Traversal Protection

- Optional directory whitelisting via `ALLOWED_DOCS_DIRS` environment variable
- Validates all file paths in `add_document()` are within allowed directories
- Blocks path traversal attacks (e.g., `../../../etc/passwd`)
- Raises `SecurityError` on violations
- Backward compatible (no restrictions if not configured)

### Query Preprocessing (NLTK-powered)

- Tokenization with word_tokenize()
- Stopword removal using English stopwords corpus
- Porter Stemmer for word normalization
- Preserves technical terms (hyphenated words like "VIC-II", numbers like "6502")
- Applied to both queries and corpus during BM25 indexing
- Implemented in `_preprocess_text()` method

### Hybrid Search (v2.0.0)

- Combines FTS5 keyword search with semantic search
- Fused by default via Reciprocal Rank Fusion (RRF): `sum(weight / (RRF_K + rank))` over the
  rankers that returned each result, weighted by `semantic_weight` (0.0-1.0, default 0.7)
- Set `HYBRID_FUSION=weighted` to restore the legacy max-normalized score blend (each side scaled
  to 0-1 by its own max, then combined by weight) - superseded because a single high-scoring FTS
  hit could rescale, and so distort, the whole keyword side
- `RRF_K` (default 60) damps how much the top rank dominates the fused score
- Intelligent result merging by (doc_id, chunk_id)
- Optional cross-encoder reranking of the fused results: `USE_RERANKER=1` (default off - net
  regresses recall/MRR on the project's retrieval eval, see `eval_retrieval.py`, though it does
  promote some buried-answer cases a bi-encoder ranks low)
- Performance: ~25ms (semantic-dominated queries) to ~260ms (combines two searches); +~1s if
  reranking is enabled

### RAG Answer Grounding

`answer_question()` no longer trusts its own citations. Before this, confidence was a
hardcoded 0.85/0.70 keyed only on whether the literal string "Source N" appeared in the
LLM's output - a self-report of citing, not evidence the cited passage actually supports
the claim next to it.

- `_extract_claims_for_verification` splits the generated answer into sentence-sized
  claims and keeps only the ones with a resolvable "Source N" citation
- `_passages_for_claims` builds one windowed excerpt per **(claim, cited source)** pair -
  not one shared excerpt per source - since a single chunk can hold several distinct facts
  scattered across it; sharing one window across every claim that cites the same source
  means a claim about a fact outside that window gets judged against text that never
  mentions it. Windowed on that claim's own words, with the question's words unioned in as
  a fallback for pronoun-heavy claims ("It also has this feature.") with no distinguishing
  words of their own
- Two backends, dispatched by `_verify_answer_grounding`:
  - **Tier 1 (default):** one extra LLM call (`_verify_claims_llm`) asking, for every
    claim at once, whether its cited source supports/contradicts/doesn't mention it
  - **Tier 2:** `USE_NLI_VERIFICATION=1` swaps in a local NLI entailment cross-encoder
    (`_verify_claims_nli`, default `cross-encoder/nli-deberta-v3-base`) - near-zero
    marginal cost per call once loaded, at the cost of a large one-time model download.
    Its class-index-to-label mapping is read from the loaded model's own `config.id2label`
    rather than hardcoded, so a differently-labelled checkpoint via `NLI_MODEL` can't
    silently invert every verdict
  - Cascades on failure: NLI unavailable/erroring falls back to the LLM check; either
    backend failing falls back to the old citation-presence heuristic. A broken verifier
    never takes down `answer_question` itself
- `confidence` becomes `supported_claims / checked_claims`; unsupported claims are named
  in the response (`unverified_claims`), not just folded into a number
- `USE_ANSWER_VERIFICATION=0` disables the check outright; `VERIFY_PASSAGE_CHARS` (default
  800) bounds how much of each cited source the check sees

Verified live against real corpus content and a real LLM, which found and fixed five bugs
no mock could have caught - see the "Fix five bugs found by live-testing" commit for the
full account. One limitation was found and deliberately left unresolved: the passage
windowing is a substring-density heuristic (see below), and can still miss a specific/rare
term when most of a multi-term match set clusters together nearby but that one term sits
just outside the window.

### Enhanced Snippet Extraction (v2.0.0)

- Term density scoring via sliding window analysis
- Complete sentence extraction, when it doesn't cost the match that won the density search -
  otherwise the window falls back to the raw best-scoring span rather than a "clean" one that
  excludes it (compares density, not raw length, so a legitimately shorter window in
  uniform/repetitive text is left alone)
- Code block preservation (detects and preserves indented blocks)
- Whole word boundary highlighting for better accuracy
- 80% size threshold ensures adequate context
- More natural, readable snippets with proper sentence boundaries
- Every caller's query terms go through `_content_terms()` (raw text) or `_filter_snippet_terms()`
  (an already-tokenized set, e.g. `search()`'s NLTK-preprocessed terms) before reaching this
  function's density scoring - dropping stopwords and sub-3-character tokens. A bare digit like
  "3" was found live to substring-match densely in unrelated regions ("voice 3", "REG 3"),
  stealing the window from the sentence that actually said "three voices". Applies to every
  `_extract_snippet` call site: the reranker, the RAG context builder, answer-verification
  windowing, `search()`'s BM25/simple/FTS5 backends, `semantic_search()`, and `search_figures()`
- `search_figures()` had a separate, more severe defect alongside this: it passed the raw query
  **string** to `_extract_snippet`, which expects a term set - Python iterates a string character
  by character, so density scoring scored windows by letter frequency and highlighting was
  silently disabled outright (single characters never clear the function's own len>=2 highlight
  threshold)

### Health Monitoring (v2.0.0)

- Comprehensive system diagnostics via `health_check()` method
- Database health: integrity checking, size monitoring, orphaned chunk detection
- Feature status: FTS5, semantic search, BM25, embeddings availability
- Performance metrics: cache utilization, index status
- Disk space warnings (< 1GB free)
- Returns structured health report with status, metrics, and issues

### Table Extraction from PDFs (v2.1.0)

- Automatic extraction of structured tables from PDF documents using pdfplumber
- Tables converted to markdown format for display
- FTS5 full-text search on table content via `tables_fts` index
- Searchable via `search_tables()` method and `search_tables` MCP tool
- Results include page number, row/column count, and relevance scores
- Stored in `document_tables` database table with automatic FTS5 synchronization

### Code Block Detection (v2.1.0)

**Supported Code Types:**
- **BASIC**: Line-numbered BASIC programs (e.g., "10 PRINT", "20 GOTO")
- **Assembly**: 6502 assembly mnemonics (LDA, STA, JMP, etc.)
- **Hex dumps**: Memory dumps with addresses (e.g., "D000: 00 01 02 03")

**Implementation:**
- Uses regex pattern matching (requires 3+ consecutive lines for detection)
- FTS5 full-text search on code content via `code_fts` index
- Searchable via `search_code()` method and `search_code` MCP tool
- Results include block type, line count, page number, and relevance scores
- Stored in `document_code_blocks` database table with automatic FTS5 synchronization

## Similarity Search (Find Similar Documents)

**find_similar_documents()** - Find documents similar to a given document or chunk:
- Dual-mode implementation: semantic embeddings (preferred) or TF-IDF (fallback)
- Supports document-level similarity (when chunk_id is None) and chunk-level similarity
- Tag filtering support for narrowing results

**Semantic Similarity** (`_find_similar_semantic`):
- Uses FAISS embeddings index for fast nearest-neighbor search
- Computes cosine similarity between embedding vectors
- Aggregates chunk scores by document (mean similarity)
- Requires embeddings to be built (`USE_SEMANTIC_SEARCH=1`)

**TF-IDF Similarity** (`_find_similar_tfidf`):
- Builds TF-IDF vectors using sklearn's TfidfVectorizer
- Computes cosine similarity between document/chunk vectors
- No external dependencies beyond sklearn (included in rank-bm25)
- Works without embeddings generation

## Database Access Patterns

All database operations go through KnowledgeBase methods:

### Adding Documents

```python
# kb/ingest/_documents.py add_document() -> kb/core.py _add_document_db()
# Uses transaction for ACID guarantees
cursor.execute("BEGIN TRANSACTION")
# Insert document + chunks
cursor.execute("INSERT INTO documents ...")
cursor.execute("INSERT INTO chunks ...")
self.db_conn.commit()
```

### Retrieving Data

```python
# Lazy loading - only load what's needed
chunks = self._get_chunks_db(doc_id)  # Load chunks for one document
chunks = self._get_chunks_db()        # Load all chunks (for BM25)
```

### Search Flow (with FTS5 enabled)

1. `search()` called → checks if `USE_FTS5=1` and `_fts5_available()`
2. If FTS5 available → `_search_fts5()` executes native SQLite search (~50ms)
3. FTS5 returns results with native BM25 ranking
4. If FTS5 returns no results → falls back to BM25/simple search
5. Results filtered by tags (if specified) and returned

### Search Flow (BM25 fallback)

1. `search()` called → checks if `self.bm25` is None
2. If None → `_build_bm25_index()` → `_get_chunks_db()` loads all chunks (~24s first time)
3. BM25 scores calculated → results filtered and sorted
4. Subsequent searches use cached BM25 index (fast)
5. Add/remove operations invalidate cache (`self.bm25 = None`)

### Key Database Methods

- `_init_database()` - Create schema and tables
- `_add_document_db(doc, chunks)` - Insert with transaction
- `_remove_document_db(doc_id)` - Delete (chunks cascade)
- `_get_chunks_db(doc_id)` - Load chunks with JOIN to get filename/title
- `get_chunk(doc_id, chunk_id)` - Query single chunk
- `close()` - Close database connection (important for tests)

## Extending the Codebase

### Adding New MCP Tools

1. Add a `Tool(...)` schema to `TOOL_SCHEMAS` in `mcp_tools/schemas.py`
2. Implement the handler function in the relevant `mcp_tools/<domain>.py` module and register it in that module's `HANDLERS_<DOMAIN>` dict (picked up by `mcp_tools/handlers.py`'s aggregation)
3. Use KnowledgeBase methods for data operations
4. Return list of `TextContent` objects

### Extending File Type Support

Currently supported formats:
- **PDF** (.pdf) - via pypdf, method: `_extract_pdf_text()`
- **Text** (.txt) - native Python with encoding detection
- **Markdown** (.md) - treated as text files
- **HTML** (.html, .htm) - via BeautifulSoup4, method: `_extract_html_file()`
- **Excel** (.xlsx, .xls) - via openpyxl, method: `_extract_excel_file()`

**To add new formats:**
1. Add file extension to condition check in `add_document()` (`kb/ingest/_documents.py`)
2. Implement extraction method (like `_extract_pdf_text` or `_extract_excel_file`)
3. Update tool description and README
4. Update GUI file uploaders in `admin_pages/documents.py`
5. Update bulk add pattern default in `add_documents_bulk()`

## Version History

### Completed Enhancements

- ✅ v2.0.0: Hybrid search combining FTS5 + semantic (configurable weighting)
- ✅ v2.0.0: Enhanced snippet extraction (term density, complete sentences, code blocks)
- ✅ v2.0.0: Health monitoring system (diagnostics, metrics, status reporting)
- ✅ v2.1.0: Table extraction from PDFs (pdfplumber, markdown conversion, FTS5 search)
- ✅ v2.1.0: Code block detection (BASIC/Assembly/Hex, regex patterns, FTS5 search)
- ✅ v2.14.0: URL scraping & web content ingestion (mdscrape integration)

### Future Enhancements

- Multi-language support beyond English
- See docs/ROADMAP.md for detailed roadmap

(Query autocompletion and fuzzy search / typo tolerance were on this list previously but have since shipped - see the version history above and docs/ROADMAP.md.)

## Wiki Export (v2.23.15)

### Overview

The wiki export system generates a static HTML wiki from the knowledge base, providing an offline-browsable interface with advanced visualizations. No server required - all functionality is client-side JavaScript.

### Architecture

**wiki_export.py** - Wiki generation orchestrator
- `WikiExporter` class: Handles full export pipeline
- Parallelized document page generation (ThreadPoolExecutor, 8 workers max)
- JSON data exports for client-side rendering
- CSS/JS asset generation with visualization libraries

### Export Pipeline

1. **Data Extraction** (`wikigen/data.py`)
   - `_export_documents()` - Document metadata with enhanced file type detection
   - `_export_entities()` - Entity groupings by type
   - `_export_graph()` - Graph nodes (178) and edges (20) for knowledge graph
   - `_export_document_coordinates()` - 2D UMAP/t-SNE coordinates for similarity map
   - `_export_topics()` - LDA topic models
   - `_export_clusters()` - k-means clusters with document lists (limit 50)
   - `_export_events()` - Timeline events (8 types)
   - `_export_chunks()` - All text chunks with document references

2. **JSON Data Files** (`wiki_export.py`, `export()`)
   - `documents.json` - Full document metadata
   - `entities.json` - Grouped entities with occurrences
   - `graph.json` - Knowledge graph structure
   - `coordinates.json` - 2D document positions (UMAP/t-SNE)
   - `clusters.json` - Cluster assignments with member documents
   - `topics.json` - Topic models with top words
   - `events.json` - Timeline events with metadata
   - `chunks.json` - Full chunk data
   - `search-index.json` - Fuse.js search index
   - `similarities.json` - Document similarity matrix

3. **HTML Generation** (`wikigen/pages.py`, `wikigen/browsers.py`, `wikigen/visualizations.py`)
   - `_generate_index_html()` - Home page with stats
   - `_generate_documents_browser_html()` - Document list with filters
   - `_generate_chunks_browser_html()` - Chunk browser with pagination
   - `_generate_entities_html()` - Entity explorer by type
   - `_generate_knowledge_graph_html()` - D3.js force-directed graph (887 lines)
   - `_generate_similarity_map_html()` - Canvas 2D similarity map (660 lines)
   - `_generate_topics_html()` - Topic/cluster browser
   - `_generate_timeline_html()` - Interactive horizontal timeline (750 lines)
   - `_generate_pdf_viewer_html()` - PDF.js viewer integration
   - `_generate_doc_html()` - Individual document pages (parallelized)

4. **Static Assets** (`wiki_export.py`, `_create_css`/`_create_javascript`/`_download_libraries`; content lives in `wiki_assets/`)
   - `_create_css()` - Complete stylesheet (~5500 lines)
   - `_create_javascript()` - All JS modules (9 files, ~1650 lines total)
   - `_download_libraries()` - Fuse.js, PDF.js from CDN

### Enhanced File Type Detection (v2.23.15)

In `wikigen/data.py`, `_export_documents()`:
```python
# Detect file type from extension for better display
file_type = doc_meta.file_type
if file_type == 'text':
    filename_lower = doc_meta.filename.lower()
    if filename_lower.endswith('.html') or filename_lower.endswith('.htm'):
        file_type = 'html'
    elif filename_lower.endswith('.md') or filename_lower.endswith('.markdown'):
        file_type = 'markdown'
```

Enables proper filtering on documents page (PDF, HTML, Markdown, Text separate).

### Document Similarity Map

**Implementation** (`wikigen/data.py`, `_export_document_coordinates`):

Uses dimensionality reduction to visualize document relationships in 2D:
1. Loads document embeddings from FAISS index
2. Reduces to 2D using UMAP (preferred) or t-SNE fallback
3. Normalizes coordinates to 0-1000 range for canvas rendering
4. Retrieves k-means cluster assignments from database
5. Exports to `coordinates.json`

**Visualization** (`wikigen/visualizations.py`, `_generate_similarity_map_html`):
- Canvas-based rendering with pan/zoom
- Color-coded by cluster (15 colors)
- Hover tooltips show document details
- Click navigation to document pages
- Search and filter by cluster/file type
- Stats dashboard: total docs, clusters, reduction method

**Key Parameters:**
- UMAP: `n_components=2`, `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`, `random_state=42`
- t-SNE: `n_components=2`, `perplexity=min(30, n_docs-1)`, `random_state=42`

### UI Enhancements (v2.23.15)

**Explanation Boxes:**
- Added to Documents, Chunks, Topics pages
- Gradient background, left border accent
- Explains page purpose and features
- CSS class: `.explanation-box` (`wiki_assets/css/style.css`)

**ASK AI Button:**
- Changed icon: 💬 → 🤖 with "Ask AI" label
- Increased size: 60px → 85px
- Gradient background
- Pulse animation (2s, infinite)
- Enhanced hover effects (scale 1.1, rotate 5deg)
- CSS classes: `.chat-toggle`, `.bot-icon`, `.bot-label` (`wiki_assets/css/style.css`)

**Clickable Clusters:**
- Topics page shows up to 10 documents per cluster
- Clickable links to document pages
- "...and N more" indicator for large clusters
- JavaScript: `displayClusters()` function (`wiki_assets/js/topics.js`)

**Timeline Viewport:**
- Height: `calc(100vh - 400px)` with `min-height: 500px`
- Better utilizes available browser space
- CSS: `.timeline-container` (`wikigen/pages.py`, `_generate_timeline_html`)

### Visualization Libraries

**D3.js v7** - Force-directed knowledge graph
- 178 nodes (entities) sized by document frequency
- 20 edges (relationships) with strength visualization
- Interactive zoom, pan, drag
- Node highlighting on hover/click

**UMAP/t-SNE** - Dimensionality reduction
- UMAP via `umap-learn` package (preferred)
- t-SNE via `sklearn.manifold.TSNE` (fallback)
- Reduces high-dimensional embeddings to 2D

**Canvas API** - Similarity map rendering
- 2D context for drawing document points
- Mouse interaction: drag to pan, wheel to zoom
- Hover detection within 10px radius
- Click navigation

**Fuse.js** - Client-side search
- Fuzzy search across all content
- No server required
- Configurable threshold and keys

**PDF.js** - PDF viewing
- Client-side PDF rendering
- Page navigation, zoom controls
- Download functionality

### Testing (test_wiki_export.py)

**16 unit tests** covering:
1. **Document Coordinate Export** (3 tests)
   - UMAP/t-SNE dimensionality reduction
   - No embeddings fallback
   - Insufficient data handling

2. **File Type Detection** (4 tests)
   - HTML file detection (.html, .htm)
   - Markdown detection (.md, .markdown)
   - PDF preservation
   - Plain text preservation

3. **Cluster Document Export** (2 tests)
   - Document lists in clusters
   - Bytes cluster number handling

4. **HTML Generation** (6 tests)
   - Explanation boxes (Documents, Chunks, Topics)
   - Similarity map page
   - Timeline viewport height
   - ASK AI button styling

5. **JavaScript Generation** (1 test)
   - Clickable cluster documents in topics.js

All tests use mocked `KnowledgeBase` to avoid database dependencies.

### Performance Characteristics

- **Export time**: ~30 seconds for 215 documents (parallel generation)
- **Coordinate generation**: Depends on embeddings (UMAP ~5s for 200 docs)
- **File size**: ~15MB total wiki (includes libraries, data, HTML)
- **Load time**: <2s for index page, instant navigation
- **Search**: Client-side Fuse.js, <100ms for most queries

### Browser Compatibility

- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
- JavaScript ES6+ features required
- Canvas API for similarity map
- PDF.js for PDF viewing
- No IE11 support

## Windows-Specific Notes

- Uses Windows-style paths (`C:\Users\...`)
- Batch files (.bat) provided for convenience (setup.bat, run.bat, tdz.bat)
- Virtual environment activation: `.venv\Scripts\activate`
- Python executable path for MCP config: `C:\...\\.venv\Scripts\python.exe`
