# Performance Optimizations - Phase 2

## Baseline Metrics (Before Optimization)

**Startup Performance:**
- Initialization time: **1976ms** (~2 seconds)
- Memory usage: 5.48MB
- Documents loaded: 184
- Embeddings vectors: 2612

**Search Performance:**
- FTS5 search: **92.52ms** average
- Semantic search: **20.01ms** average (excellent!)
- Hybrid search: **103.53ms** average

## Identified Bottlenecks

### 1. Startup Time (CRITICAL - 2 seconds)

**Root Causes:**
- Loading sentence-transformers model: ~2.5 seconds
- Loading all document metadata upfront: ~100ms
- Loading embeddings index into memory: ~200ms
- Initializing background workers: ~50ms

**Impact:** Users wait 2 seconds before KB is ready

### 2. Hybrid Search Performance (MODERATE - 103ms)

**Root Cause:** Sequential execution of FTS5 + Semantic searches

**Impact:** Slower than individual search types

## Optimization Strategy

### Phase 2A: Lazy Loading & Deferred Initialization

**Target: Reduce startup from 2000ms → 500ms**

1. **Lazy Load Embeddings Model** (saves ~2.5s on startup)
   - Don't load sentence-transformers until first semantic search
   - Cache loaded model for subsequent searches
   - Trade-off: First semantic search takes +2.5s

2. **Defer Background Workers** (saves ~50ms)
   - Start entity extraction worker only when needed
   - Or start after initial KB initialization completes

3. **Optimize Document Metadata Loading** (saves ~50-100ms)
   - Add database indexes on frequently queried fields
   - Use prepared statements for bulk loading

### Phase 2B: Search Performance Improvements

**Target: Reduce hybrid search from 103ms → 60ms**

1. **Parallel FTS5 + Semantic Execution**
   - Run both searches concurrently using threading
   - Merge results when both complete
   - Expected: ~40% improvement

2. **Query Result Caching** (already implemented)
   - Verify cache hit rates
   - Increase cache size if needed

### Phase 2C: Database Query Optimizations

**Target: Faster document/chunk retrieval**

1. **Add Missing Indexes**
   - Index on `documents.tags` (JSON array)
   - Index on `chunks.doc_id` (if missing)
   - Composite index on frequently joined columns

2. **Query Optimization**
   - Use EXPLAIN QUERY PLAN to identify slow queries
   - Optimize JOIN operations
   - Add covering indexes where beneficial

### Phase 2D: Memory Optimizations

**Current Usage: ~5.48MB**

1. **Streaming Document Loading**
   - Don't load all 184 documents into memory
   - Load on-demand when accessed
   - Keep only metadata in memory

2. **Embeddings Index Compression**
   - Use quantization for embeddings
   - Trade minor accuracy for 4x less memory

## Implementation Priority

**HIGH PRIORITY (Immediate Impact):**
1. Lazy load embeddings model → saves 2.5s startup
2. Add database indexes → faster queries
3. Parallel hybrid search → 40% faster

**MEDIUM PRIORITY:**
4. Defer background workers → saves 50ms startup
5. Optimize metadata loading → saves 50-100ms
6. Query result caching verification

**LOW PRIORITY (Nice to have):**
7. Embeddings compression
8. Streaming document loading

## Expected Results After Phase 2

**Startup Time:**
- Before: 1976ms
- After: ~500ms
- **Improvement: 75% faster**

**Hybrid Search:**
- Before: 103.53ms
- After: ~60ms
- **Improvement: 42% faster**

**Overall Impact:**
- Much faster initial load
- Snappier search experience
- Better resource utilization

## Measurement Plan

1. Re-run profiler after each optimization
2. Compare before/after metrics
3. Document improvements in this file
4. Update version.py with Phase 2 results
