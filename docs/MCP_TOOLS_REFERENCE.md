# MCP Tools Quick Reference

**Version:** 2.23.1
**Total Tools:** 50+
**Last Updated:** 2026-01-03

Complete reference for all MCP tools available in the TDZ C64 Knowledge Base server.

---

## 📑 Table of Contents

- [Search Tools (11)](#search-tools)
- [Document Management (6)](#document-management)
- [URL Scraping (3)](#url-scraping)
- [AI & Analytics (14)](#ai--analytics)
- [Export Tools (3)](#export-tools)
- [System Tools (2)](#system-tools)
- [Advanced Tools (14)](#advanced-tools)

---

## 🔍 Search Tools

### search_docs
**Full-text search using FTS5 or BM25**

```json
{
  "query": "VIC-II sprite registers",
  "max_results": 10,
  "tags": ["hardware", "graphics"]
}
```

**Returns:** Search results with snippets, scores, and metadata
**Performance:** 85ms avg (FTS5), 480x faster than BM25
**Use when:** Keyword-based search, technical terms, exact matches

---

### semantic_search
**Meaning-based search using embeddings**

```json
{
  "query": "How do sprites work on the VIC-II chip?",
  "max_results": 5,
  "top_k": 50
}
```

**Returns:** Semantically similar documents
**Performance:** 16ms avg (after model loading)
**Use when:** Conceptual searches, natural language questions

---

### hybrid_search
**Combined keyword + semantic search**

```json
{
  "query": "SID chip sound programming",
  "max_results": 10,
  "semantic_weight": 0.7,
  "tags": ["audio"]
}
```

**Parameters:**
- `semantic_weight` (0.0-1.0): Balance between keyword (0.0) and semantic (1.0), default: 0.3

**Returns:** Merged and ranked results from both search modes
**Performance:** 142ms avg
**Use when:** Best overall search quality needed

---

### fuzzy_search
**Typo-tolerant search with string similarity**

```json
{
  "query": "VIC2 asembly",
  "similarity_threshold": 80,
  "max_results": 10
}
```

**Features:**
- Handles typos: "VIC2" → "VIC-II", "asembly" → "assembly"
- Configurable threshold (0-100)
- Falls back to exact search if good matches found

**Use when:** User queries may have typos, technical term variations

---

### search_within_results
**Progressive search refinement**

```json
{
  "previous_results": [...],
  "refinement_query": "sprite collision",
  "max_results": 5
}
```

**Workflow:**
1. Broad search: `search_docs("VIC-II", max_results=50)`
2. Refine: `search_within_results(results, "sprite collision")`

**Use when:** Narrowing down large result sets

---

### answer_question
**RAG-based question answering with citations**

```json
{
  "question": "How do I program sprites on the VIC-II?",
  "max_sources": 5,
  "search_mode": "auto",
  "confidence_threshold": 0.7
}
```

**Returns:**
- Natural language answer
- Source citations
- Confidence score
- Context used

**Use when:** Users ask questions rather than searching for keywords

---

### translate_query
**Convert natural language to structured search**

```json
{
  "query": "find sprites on VIC-II chip",
  "explain": true
}
```

**Returns:** Structured search parameters and explanation
**Use when:** Parsing user intent from natural language

---

### search_tables
**Search extracted PDF tables**

```json
{
  "query": "memory map",
  "tags": ["reference"],
  "max_results": 10
}
```

**Returns:** Table results with markdown formatting, page numbers
**Use when:** Looking for tabular data in documents

---

### search_code
**Search code blocks (BASIC/Assembly/Hex)**

```json
{
  "query": "LDA #$00",
  "block_type": "assembly",
  "max_results": 10
}
```

**Block types:** `basic`, `assembly`, `hex`, or `null` (all types)
**Use when:** Finding code examples

---

### faceted_search
**Search with entity-based filtering**

```json
{
  "query": "programming",
  "facet_filters": {
    "hardware": ["VIC-II", "SID"],
    "concept": ["sprite", "sound"]
  },
  "max_results": 10
}
```

**Use when:** Filtering by extracted entities

---

### find_similar
**Find documents similar to a given document**

```json
{
  "doc_id": "89d0943d6009",
  "max_results": 5,
  "tags": ["graphics"]
}
```

**Methods:** Semantic (FAISS) or TF-IDF
**Use when:** Discovering related documentation

---

## 📄 Document Management

### add_document
**Add a single file to the knowledge base**

```json
{
  "filepath": "C:/docs/c64_ref.pdf",
  "title": "C64 Programmer's Reference",
  "tags": ["reference", "memory-map"]
}
```

**Supported formats:** PDF, TXT, MD, HTML, Excel
**Features:** Duplicate detection, automatic chunking

---

### list_docs
**List all documents with filtering**

```json
{
  "tags": ["reference"],
  "file_type": "pdf",
  "limit": 20,
  "offset": 0
}
```

**Returns:** Paginated document list with metadata

---

### get_document
**Get full document metadata**

```json
{
  "doc_id": "89d0943d6009",
  "include_chunks": false
}
```

**Returns:** Complete document metadata, optionally with chunks

---

### get_chunk
**Get a specific chunk**

```json
{
  "doc_id": "89d0943d6009",
  "chunk_id": 5
}
```

**Returns:** Single chunk content with page info

---

### remove_document
**Delete a document**

```json
{
  "doc_id": "89d0943d6009"
}
```

**Cascade:** Removes all chunks, entities, summaries

---

### check_updates
**Check for file changes**

```json
{
  "auto_update": false
}
```

**Returns:** Modified, deleted, and unchanged document lists

---

## 🌐 URL Scraping

### scrape_url
**Scrape documentation website**

```json
{
  "url": "https://www.c64-wiki.com/wiki/VIC",
  "tags": ["wiki"],
  "depth": 2,
  "threads": 5,
  "max_pages": 20
}
```

**Features:**
- Concurrent scraping
- Depth control
- URL filtering
- Auto-tagging

**Dependencies:** Requires mdscrape executable

---

### rescrape_document
**Re-scrape for updates**

```json
{
  "doc_id": "89d0943d6009",
  "force": false
}
```

**Checks:** Last-Modified headers, content hash
**Use when:** Updating scraped content

---

### check_url_updates
**Check all scraped docs for changes**

```json
{
  "auto_rescrape": false,
  "check_structure": true
}
```

**Features:** Anomaly detection, bulk checking

---

## 🤖 AI & Analytics

### extract_entities
**Extract named entities from document**

```json
{
  "doc_id": "89d0943d6009",
  "confidence_threshold": 0.6,
  "force_regenerate": false
}
```

**Entity types:** hardware, memory_address, instruction, person, company, product, concept
**Performance:** 5000x faster with regex patterns
**Requires:** LLM API key (Anthropic or OpenAI)

---

### list_entities / get_entities
**Retrieve stored entities**

```json
{
  "doc_id": "89d0943d6009",
  "entity_types": ["hardware", "instruction"],
  "min_confidence": 0.7
}
```

---

### search_entities
**Search across all entities**

```json
{
  "query": "VIC-II",
  "entity_types": ["hardware"],
  "min_confidence": 0.7,
  "max_results": 20
}
```

**Uses:** FTS5 full-text search on entity table

---

### entity_stats
**Get entity extraction statistics**

```json
{
  "entity_type": "hardware"
}
```

**Returns:** Total counts, top entities, document distribution

---

### get_entity_analytics
**Comprehensive entity analytics**

```json
{
  "entity_type": "all"
}
```

**Returns:**
- Entity distribution by type
- Top entities by occurrence
- Relationship strength scores
- Co-occurrence patterns

---

### extract_entity_relationships
**Extract entity co-occurrences**

```json
{
  "doc_id": "89d0943d6009",
  "min_strength": 0.3
}
```

**Calculates:** Distance-based relationship strength

---

### get_entity_relationships
**Get relationships for entity**

```json
{
  "entity_text": "VIC-II",
  "min_strength": 0.5,
  "max_results": 20
}
```

---

### search_entity_pair
**Find docs with entity pair**

```json
{
  "entity1": "VIC-II",
  "entity2": "sprite",
  "max_results": 10
}
```

---

### compare_documents
**Side-by-side document comparison**

```json
{
  "doc_id_1": "abc123",
  "doc_id_2": "def456",
  "comparison_type": "full"
}
```

**Types:** `metadata`, `content`, `full`

---

### suggest_tags
**AI-powered tag suggestions**

```json
{
  "doc_id": "89d0943d6009",
  "confidence_threshold": 0.6
}
```

**Returns:** Suggested tags by category

---

### get_tags_by_category
**Browse tags organized by category**

```json
{
  "category": "hardware"
}
```

---

### summarize_document
**Generate AI summary**

```json
{
  "doc_id": "89d0943d6009",
  "summary_type": "brief",
  "max_length": 300
}
```

**Types:** `brief`, `detailed`, `bullet`

---

## 📤 Export Tools

### export_entities
**Export entities to CSV/JSON**

```json
{
  "format": "csv",
  "output_path": "entities.csv",
  "min_confidence": 0.7,
  "entity_types": ["hardware"]
}
```

---

### export_relationships
**Export relationships to CSV/JSON**

```json
{
  "format": "json",
  "output_path": "relationships.json",
  "min_strength": 0.5
}
```

---

### export_documents_bulk
**Bulk export documents**

```json
{
  "format": "json",
  "output_path": "documents.json",
  "tags": ["reference"],
  "include_chunks": false
}
```

---

## ⚙️ System Tools

### kb_stats
**Knowledge base statistics**

```json
{}
```

**Returns:**
- Document count and total size
- Chunk statistics
- Entity counts
- Search performance metrics

---

### health_check
**System diagnostics**

```json
{}
```

**Returns:**
- Database health (integrity, size, orphans)
- Feature status (FTS5, semantic, embeddings)
- Performance metrics
- Issues and warnings

---

## 🚀 Advanced Tools

### add_documents_bulk
**Bulk document import**

```json
{
  "directory": "C:/c64docs",
  "pattern": "**/*.{pdf,txt}",
  "tags": ["reference"],
  "recursive": true
}
```

---

### remove_documents_bulk
**Bulk delete by IDs or tags**

```json
{
  "doc_ids": ["abc", "def"],
  "tags": ["outdated"],
  "confirm": true
}
```

---

### update_tags_bulk
**Bulk tag operations**

```json
{
  "add_tags": ["updated"],
  "remove_tags": ["draft"],
  "doc_ids": ["abc", "def"]
}
```

---

### extract_entities_bulk
**Bulk entity extraction**

```json
{
  "confidence_threshold": 0.6,
  "max_docs": 10,
  "skip_existing": true
}
```

---

### extract_relationships_bulk
**Bulk relationship extraction**

```json
{
  "min_strength": 0.3,
  "doc_ids": ["abc", "def"]
}
```

---

### auto_tag_document
**Auto-tag single document**

```json
{
  "doc_id": "89d0943d6009",
  "confidence_threshold": 0.7,
  "apply_tags": false
}
```

---

### auto_tag_all
**Auto-tag all documents**

```json
{
  "confidence_threshold": 0.7,
  "reindex": false
}
```

---

### queue_entity_extraction
**Queue background extraction**

```json
{
  "doc_id": "89d0943d6009",
  "priority": "normal"
}
```

---

### get_extraction_status
**Check extraction job status**

```json
{
  "job_id": "abc123"
}
```

---

### get_extraction_jobs
**List all extraction jobs**

```json
{
  "status": "pending",
  "limit": 20
}
```

---

### find_by_reference
**Find by reference type**

```json
{
  "reference_type": "memory_address",
  "reference_value": "$D000",
  "max_results": 10
}
```

---

### suggest_queries
**Get query suggestions**

```json
{
  "partial_query": "VIC",
  "max_suggestions": 5
}
```

---

### search_analytics
**Search analytics**

```json
{
  "time_range_days": 30
}
```

---

### create_backup / restore_backup
**Database backup/restore**

```json
{
  "backup_path": "C:/backups/kb_backup.db"
}
```

---

## 💡 Usage Tips

### Performance
- **FTS5 search:** Fastest for keywords (85ms avg)
- **Semantic search:** Best for concepts (16ms avg after loading)
- **Hybrid search:** Best quality (142ms avg)

### Cost Optimization
- Use regex entity extraction first (5000x faster than LLM)
- Cache entity/summary results in database
- Set appropriate confidence thresholds

### Best Practices
- Tag documents consistently for better filtering
- Use fuzzy search for user-facing applications
- Enable FTS5 for production (`USE_FTS5=1`)
- Build embeddings for semantic search

---

## 📚 See Also

- [REST API Reference](REST_API.md) - HTTP endpoints for all tools
- [Architecture Guide](../ARCHITECTURE.md) - Technical implementation details
- [Examples](EXAMPLES.md) - Practical usage examples
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

---

**Version:** 2.23.1
**Total Tools:** 50+
**Platform:** Windows (cross-platform compatible)
