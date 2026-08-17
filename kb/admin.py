"""Stats, health, backup/restore, reconciliation and the MCP call log.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from models import DocumentChunk
from models import DocumentMeta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import json
import os
import time


class AdminMixin:

    @staticmethod
    def _summarize_call_args(arguments: dict, max_value_len: int = 150) -> str:
        """
        Render tool call arguments as a short, privacy-conscious JSON summary
        for logging - long string values (e.g. full document content pasted
        into a tool call) are truncated rather than stored in full.
        """
        if not arguments:
            return "{}"
        summarized = {}
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > max_value_len:
                summarized[key] = value[:max_value_len] + f"...[{len(value)} chars]"
            else:
                summarized[key] = value
        try:
            return json.dumps(summarized, default=str)[:2000]
        except Exception:
            return str(summarized)[:2000]

    def _log_mcp_call(self, tool_name: str, duration_ms: float, success: bool,
                       error_message: Optional[str] = None, arguments: Optional[dict] = None):
        """Record one MCP tool invocation to mcp_call_log. Never raises - a
        logging failure must not break the tool call it's trying to record."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """INSERT INTO mcp_call_log
                   (tool_name, called_at, duration_ms, success, error_message, args_summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    tool_name,
                    datetime.now(timezone.utc).isoformat(),
                    duration_ms,
                    1 if success else 0,
                    (error_message or None),
                    self._summarize_call_args(arguments),
                )
            )
            self.db_conn.commit()
        except Exception as e:
            self.logger.warning(f"Failed to log MCP call for {tool_name}: {e}")

    def get_mcp_call_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Aggregate MCP usage stats over the trailing window (default 24h)."""
        cursor = self.db_conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        row = cursor.execute(
            """SELECT COUNT(*), SUM(success = 0), AVG(duration_ms), MAX(duration_ms)
               FROM mcp_call_log WHERE called_at >= ?""",
            (cutoff,)
        ).fetchone()
        total, errors, avg_ms, max_ms = row
        total = total or 0
        errors = errors or 0

        top_tools = cursor.execute(
            """SELECT tool_name, COUNT(*) as cnt, AVG(duration_ms) as avg_ms,
                      SUM(success = 0) as errors
               FROM mcp_call_log WHERE called_at >= ?
               GROUP BY tool_name ORDER BY cnt DESC LIMIT 20""",
            (cutoff,)
        ).fetchall()

        return {
            'window_hours': hours,
            'total_calls': total,
            'error_count': errors,
            'error_rate': (errors / total) if total else 0.0,
            'avg_duration_ms': avg_ms or 0.0,
            'max_duration_ms': max_ms or 0.0,
            'top_tools': [
                {'tool_name': t, 'calls': c, 'avg_duration_ms': a or 0.0, 'errors': e}
                for t, c, a, e in top_tools
            ]
        }

    def get_recent_mcp_calls(self, limit: int = 200, tool_name: Optional[str] = None,
                              only_errors: bool = False) -> List[Dict[str, Any]]:
        """Fetch the most recent MCP calls, optionally filtered, newest first."""
        cursor = self.db_conn.cursor()
        query = "SELECT call_id, tool_name, called_at, duration_ms, success, error_message, args_summary FROM mcp_call_log WHERE 1=1"
        params = []
        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)
        if only_errors:
            query += " AND success = 0"
        query += " ORDER BY call_id DESC LIMIT ?"
        params.append(limit)

        rows = cursor.execute(query, params).fetchall()
        return [
            {
                'call_id': r[0], 'tool_name': r[1], 'called_at': r[2],
                'duration_ms': r[3], 'success': bool(r[4]),
                'error_message': r[5], 'args_summary': r[6],
            }
            for r in rows
        ]

    def get_mcp_calls_over_time(self, hours: int = 24, bucket_minutes: int = 60) -> List[Dict[str, Any]]:
        """Bucket call counts over time for a simple usage-over-time chart."""
        cursor = self.db_conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = cursor.execute(
            """SELECT called_at, success FROM mcp_call_log WHERE called_at >= ? ORDER BY called_at""",
            (cutoff,)
        ).fetchall()

        buckets = {}
        for called_at, success in rows:
            try:
                dt = datetime.fromisoformat(called_at)
            except ValueError:
                continue
            bucket_key = dt.replace(
                minute=(dt.minute // bucket_minutes) * bucket_minutes if bucket_minutes < 60 else 0,
                second=0, microsecond=0
            )
            if bucket_minutes >= 60:
                bucket_key = bucket_key.replace(hour=(dt.hour // (bucket_minutes // 60)) * (bucket_minutes // 60))
            entry = buckets.setdefault(bucket_key.isoformat(), {'calls': 0, 'errors': 0})
            entry['calls'] += 1
            if not success:
                entry['errors'] += 1

        return [
            {'bucket': k, 'calls': v['calls'], 'errors': v['errors']}
            for k, v in sorted(buckets.items())
        ]

    def get_chunk(self, doc_id: str, chunk_id: int) -> Optional[DocumentChunk]:
        """Get a specific chunk from database."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT c.doc_id, d.filename, d.title, c.chunk_id, c.page, c.content, c.word_count
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.doc_id = ? AND c.chunk_id = ?
        """, (doc_id, chunk_id))

        row = cursor.fetchone()
        if not row:
            return None

        return DocumentChunk(
            doc_id=row[0],
            filename=row[1],
            title=row[2],
            chunk_id=row[3],
            page=row[4],
            content=row[5],
            word_count=row[6]
        )

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get document with all its chunks.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with document metadata and chunks, or None if not found
        """
        self._sync_documents_if_needed()

        if doc_id not in self.documents:
            return None

        doc = self.documents[doc_id]
        chunks = self._get_chunks_db(doc_id)

        return {
            'doc_id': doc.doc_id,
            'title': doc.title,
            'filename': doc.filename,
            'chunks': [
                {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'page': chunk.page,
                    'word_count': chunk.word_count
                }
                for chunk in chunks
            ]
        }

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get the full content of a document from database."""
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            return None
        return "\n\n".join(c.content for c in chunks)

    def list_documents(self, include_superseded: bool = False) -> list[DocumentMeta]:
        """List indexed documents. Excludes superseded card versions by default."""
        self._sync_documents_if_needed()
        if include_superseded:
            return list(self.documents.values())
        return [d for d in self.documents.values() if not d.superseded_by]

    def get_stats(self, use_cache: bool = True) -> dict:
        """
        Get knowledge base statistics from database.

        Args:
            use_cache: If True, use cached results if available (1 minute TTL)

        Returns:
            Dictionary with statistics
        """
        # Check cache first
        if use_cache and self._stats_cache is not None:
            cached_result = self._stats_cache.get('stats')
            if cached_result is not None:
                return cached_result

        self._sync_documents_if_needed()
        cursor = self.db_conn.cursor()

        # Count total chunks and words
        cursor.execute("SELECT COUNT(*), SUM(word_count) FROM chunks")
        total_chunks, total_words = cursor.fetchone()
        total_chunks = total_chunks or 0
        total_words = total_words or 0

        # OPTIMIZED: Use sets comprehension more efficiently
        # Collect file_types and tags in a single pass
        file_types_set = set()
        all_tags_set = set()
        for doc in self.documents.values():
            file_types_set.add(doc.file_type)
            all_tags_set.update(doc.tags)

        stats = {
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_words': total_words,
            'file_types': sorted(list(file_types_set)),  # Sorted for consistent output
            'all_tags': sorted(list(all_tags_set))  # Sorted for consistent output
        }

        # Cache the result
        if use_cache and self._stats_cache is not None:
            self._stats_cache['stats'] = stats

        return stats

    def reconcile_chunk_cache(self) -> dict:
        """
        Reconcile the in-memory chunk cache (self.chunks) against the database.

        _build_bm25_index() only (re)loads chunks from the DB when self.chunks
        is empty, so a process that already had chunks in memory before a
        remove_document() bugfix landed (or before any future cache/DB
        divergence) will keep serving stale content from search_docs/BM25
        forever, even though get_document()/list_docs() correctly report the
        DB state. This unconditionally reloads self.chunks from the DB - the
        chunks/documents JOIN in _get_chunks_db() naturally drops any chunk
        whose document no longer exists - and invalidates every derived index
        and cache so the next search rebuilds from the reconciled data.

        Returns:
            Dictionary with before/after chunk counts and the doc_ids that
            were pruned as orphans (present in the cache, absent from the DB).
        """
        before_count = len(self.chunks)
        before_doc_ids = {c.doc_id for c in self.chunks}

        self.chunks = self._get_chunks_db()

        after_count = len(self.chunks)
        after_doc_ids = {c.doc_id for c in self.chunks}
        orphaned_doc_ids = sorted(before_doc_ids - after_doc_ids)

        # Invalidate BM25 index (will be rebuilt on next search)
        self.bm25 = None

        # Remove the orphaned docs' vectors from the shared embeddings index
        # in place, the same way remove_document() does - nulling the index
        # here would leave the on-disk files (which still contain the
        # orphans' vectors) untouched, then have the next add_document()
        # overwrite that full-corpus file with an index containing only its
        # own new chunks.
        if self.use_semantic:
            for doc_id in orphaned_doc_ids:
                self._remove_doc_embeddings(doc_id)

        # Invalidate search result caches
        self._invalidate_caches()

        self.logger.info(
            f"Reconciled chunk cache: {before_count} -> {after_count} chunks "
            f"({len(orphaned_doc_ids)} orphaned doc_id(s) pruned)"
        )

        return {
            'chunks_before': before_count,
            'chunks_after': after_count,
            'chunks_pruned': before_count - after_count,
            'orphaned_doc_ids': orphaned_doc_ids,
        }

    def reconcile_embeddings(self, max_docs: Optional[int] = None) -> dict:
        """
        Backfill embeddings for documents that have chunks but no vectors.

        The embeddings rebuild trigger only fires when the index is
        COMPLETELY empty (see _build_embeddings/_ensure_embeddings_loaded),
        so a partially populated index - built while USE_SEMANTIC_SEARCH was
        off, or from before add_document loaded the embeddings model - looks
        healthy forever and silently leaves those documents invisible to
        semantic_search/find_similar. This finds every document with chunks
        but no embedded vectors and embeds them via the same locked
        incremental path add_document uses, so it is safe to run
        concurrently with other agents.

        Args:
            max_docs: Optional cap on how many missing documents to process
                      in this call, for backfilling a large gap incrementally
                      instead of one long-running call.

        Returns:
            Dictionary with before/after coverage and the doc_ids processed.
        """
        if not self.use_semantic:
            return {'error': 'Semantic search is not enabled (set USE_SEMANTIC_SEARCH=1)'}

        self._ensure_embeddings_loaded()
        if self.embeddings_model is None:
            return {'error': 'Embeddings model is unavailable - check server logs'}

        embedded_docs_before = {d for d, _ in self.embeddings_doc_map}
        chunks_before = len(self.embeddings_doc_map)

        missing_doc_ids = [d for d in self.documents.keys() if d not in embedded_docs_before]
        if max_docs is not None:
            missing_doc_ids = missing_doc_ids[:max_docs]

        processed = []
        for doc_id in missing_doc_ids:
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                self._add_chunks_to_embeddings(chunks)
                processed.append(doc_id)

        embedded_docs_after = {d for d, _ in self.embeddings_doc_map}
        chunks_after = len(self.embeddings_doc_map)
        total_docs = len(self.documents)

        self.logger.info(
            f"Reconciled embeddings: {len(embedded_docs_before)} -> {len(embedded_docs_after)} "
            f"docs covered ({len(processed)} backfilled this call)"
        )

        return {
            'docs_covered_before': len(embedded_docs_before),
            'docs_covered_after': len(embedded_docs_after),
            'total_documents': total_docs,
            'docs_still_missing': total_docs - len(embedded_docs_after),
            'docs_backfilled_this_call': len(processed),
            'chunks_before': chunks_before,
            'chunks_after': chunks_after,
        }

    def health_check(self, quick_check: bool = True, use_cache: bool = True) -> dict:
        """
        Perform health check on the knowledge base system.

        Args:
            quick_check: If True, skip expensive operations (integrity check, orphaned chunks)
            use_cache: If True, use cached results if available (5 minute TTL)

        Returns:
            Dictionary with health metrics and status information
        """
        # Check cache first
        cache_key = f"health_quick_{quick_check}"
        if use_cache and self._health_cache is not None:
            cached_result = self._health_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        health = {
            'status': 'healthy',
            'issues': [],
            'metrics': {},
            'features': {},
            'database': {},
            'performance': {}
        }

        try:
            # Database health
            cursor = self.db_conn.cursor()

            # Check database file size
            db_file = Path(self.data_dir) / "knowledge_base.db"
            if db_file.exists():
                db_size_mb = db_file.stat().st_size / (1024 * 1024)
                health['database']['size_mb'] = round(db_size_mb, 2)

                # Warn if database is very large
                if db_size_mb > 1000:  # 1GB
                    health['issues'].append(f"Database size is large: {db_size_mb:.2f} MB")

            # Check table integrity (EXPENSIVE - only on full check)
            if not quick_check:
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                health['database']['integrity'] = integrity
                if integrity != 'ok':
                    health['status'] = 'warning'
                    health['issues'].append(f"Database integrity check failed: {integrity}")

            # Document and chunk counts - OPTIMIZED: Single query instead of 3
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM documents) as doc_count,
                    COUNT(*) as chunk_count,
                    SUM(word_count) as total_words
                FROM chunks
            """)
            doc_count, chunk_count, total_words = cursor.fetchone()
            health['metrics']['documents'] = doc_count or 0
            health['metrics']['chunks'] = chunk_count or 0
            health['metrics']['total_words'] = total_words or 0

            # Check for orphaned chunks (EXPENSIVE - only on full check)
            if not quick_check:
                cursor.execute("""
                    SELECT COUNT(*) FROM chunks c
                    LEFT JOIN documents d ON c.doc_id = d.doc_id
                    WHERE d.doc_id IS NULL
                """)
                orphaned = cursor.fetchone()[0]
                if orphaned > 0:
                    health['status'] = 'warning'
                    health['issues'].append(f"Found {orphaned} orphaned chunks")
                    health['database']['orphaned_chunks'] = orphaned

            # Feature availability
            health['features']['fts5_enabled'] = os.environ.get('USE_FTS5', '0') == '1'
            health['features']['fts5_available'] = self._fts5_available()
            health['features']['semantic_search_enabled'] = self.use_semantic
            # Check if embeddings are loaded OR if embeddings files exist (lazy loading).
            # self.embeddings_file/embeddings_map_file only exist as attributes when
            # use_semantic is True (see __init__) - and USE_SEMANTIC_SEARCH defaults
            # to off, so this raised AttributeError on every default-config health
            # check before the use_semantic guard was added here.
            health['features']['semantic_search_available'] = self.use_semantic and (
                self.embeddings_index is not None or
                (self.embeddings_file.exists() and self.embeddings_map_file.exists())
            )
            health['features']['bm25_enabled'] = os.environ.get('USE_BM25', '1') == '1'
            health['features']['query_preprocessing'] = os.environ.get('USE_QUERY_PREPROCESSING', '1') == '1'

            # Background extraction worker: queued jobs are only ever drained
            # by this thread, so if it has died every queue_entity_extraction
            # call silently accumulates work that will never run.
            worker = getattr(self, '_extraction_worker', None)
            worker_alive = bool(worker and worker.is_alive())
            health['features']['extraction_worker_alive'] = worker_alive
            health['features']['extraction_queue_depth'] = self._extraction_queue.qsize()
            if not worker_alive:
                health['issues'].append(
                    "Entity extraction worker is not running - queued extraction jobs will not be processed"
                )
                health['status'] = 'warning'

            # Documents whose recorded filepath no longer resolves on disk:
            # previously only discoverable by accident (figure OCR silently
            # skips them - see get_figure_ocr_coverage). Any future re-chunk,
            # re-OCR or page-image feature would hit the same wall silently,
            # so surface it here for the whole corpus, not just PDFs.
            missing_source_files = sum(
                1 for doc in self.documents.values()
                if self._document_source_missing(doc.filepath)
            )
            health['metrics']['missing_source_files'] = missing_source_files
            if missing_source_files > 0:
                health['issues'].append(
                    f"{missing_source_files} document(s) have a filepath that no longer exists on disk"
                )
                health['status'] = 'warning'

            # Check FTS5 index if enabled
            if health['features']['fts5_enabled']:
                if not health['features']['fts5_available']:
                    health['issues'].append("FTS5 is enabled but index not found")
                    health['status'] = 'warning'

            # Check semantic search if enabled
            if health['features']['semantic_search_enabled']:
                if not health['features']['semantic_search_available']:
                    health['issues'].append("Semantic search enabled but embeddings not built")
                    health['status'] = 'warning'
                else:
                    # Check embeddings file size (works for both loaded and lazy-loaded)
                    if self.embeddings_file.exists():
                        emb_size_mb = self.embeddings_file.stat().st_size / (1024 * 1024)
                        health['features']['embeddings_size_mb'] = round(emb_size_mb, 2)

                        # Get the doc map from the loaded index or straight
                        # from the map file - needed for both the raw count
                        # and the drift check below.
                        embeddings_map = None
                        if self.embeddings_index is not None:
                            embeddings_map = self.embeddings_doc_map
                        elif self.embeddings_map_file.exists():
                            try:
                                with open(self.embeddings_map_file, 'r') as f:
                                    embeddings_map = json.load(f)
                            except Exception:
                                embeddings_map = None

                        if embeddings_map is not None:
                            health['features']['embeddings_count'] = len(embeddings_map)

                            # Drift check: the rebuild trigger only fires on a
                            # COMPLETELY empty index (see _build_embeddings/
                            # _ensure_embeddings_loaded), so a partially
                            # populated one - e.g. built while
                            # USE_SEMANTIC_SEARCH was off, or before
                            # add_document loaded the model - looks healthy
                            # forever with no signal that most documents are
                            # actually invisible to semantic_search.
                            embedded_docs = {d for d, _ in embeddings_map}
                            total_docs = health['metrics'].get('documents', 0)
                            total_chunks = health['metrics'].get('chunks', 0)
                            if total_docs > 0:
                                doc_coverage_pct = round(100.0 * len(embedded_docs) / total_docs, 1)
                                health['features']['embeddings_doc_coverage_pct'] = doc_coverage_pct
                                # A newly-added, not-yet-embedded document or two is
                                # expected between searches; a large gap is not.
                                missing_docs = total_docs - len(embedded_docs)
                                if missing_docs > max(5, total_docs * 0.1):
                                    health['status'] = 'warning'
                                    health['issues'].append(
                                        f"Semantic search coverage gap: {missing_docs} of {total_docs} "
                                        f"documents ({100 - doc_coverage_pct:.1f}%) have no embeddings - "
                                        "run reconcile_embeddings to backfill"
                                    )
                            if total_chunks > 0:
                                health['features']['embeddings_chunk_coverage_pct'] = round(
                                    100.0 * len(embeddings_map) / total_chunks, 1
                                )

            # Performance metrics
            health['performance']['cache_enabled'] = self._search_cache is not None
            if self._search_cache is not None:
                from cachetools import TTLCache
                if isinstance(self._search_cache, TTLCache):
                    health['performance']['cache_size'] = len(self._search_cache)
                    health['performance']['cache_capacity'] = self._search_cache.maxsize

            # BM25 index status
            if health['features']['bm25_enabled']:
                health['features']['bm25_index_built'] = self.bm25 is not None

            # Disk space check
            import shutil
            disk_usage = shutil.disk_usage(self.data_dir)
            free_gb = disk_usage.free / (1024 ** 3)
            health['database']['disk_free_gb'] = round(free_gb, 2)

            if free_gb < 1:  # Less than 1GB free
                health['status'] = 'warning'
                health['issues'].append(f"Low disk space: {free_gb:.2f} GB free")

            # Overall status
            if not health['issues']:
                health['status'] = 'healthy'
                health['message'] = 'All systems operational'
            else:
                health['message'] = f"System functional with {len(health['issues'])} issue(s)"

            # Cache the result
            if use_cache and self._health_cache is not None:
                self._health_cache[cache_key] = health

        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Health check error: {str(e)}")
            health['message'] = 'Health check failed'
            self.logger.error(f"Health check error: {e}", exc_info=True)

        return health

    def _log_search(self, query: str, search_mode: str, results_count: int, execution_time_ms: float,
                    tags: Optional[list[str]] = None, clicked_doc_id: Optional[str] = None):
        """Log a search query to the search_log table."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO search_log (query, search_mode, results_count, execution_time_ms, tags, clicked_doc_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query,
                search_mode,
                results_count,
                execution_time_ms,
                ','.join(tags) if tags else None,
                clicked_doc_id
            ))
            self.db_conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging search: {e}")

    def get_search_analytics(self, days: int = 30, limit: int = 100) -> dict:
        """
        Get search analytics and insights.

        Args:
            days: Number of days to analyze (default: 30)
            limit: Maximum number of results for top queries (default: 100)

        Returns:
            Dictionary with analytics data including:
            - total_searches: Total number of searches
            - unique_queries: Number of unique queries
            - avg_results: Average number of results per search
            - avg_execution_time_ms: Average execution time
            - top_queries: Most frequent queries
            - failed_searches: Queries with zero results
            - search_modes: Breakdown by search mode
            - popular_tags: Most frequently used tags
        """
        cursor = self.db_conn.cursor()

        # Calculate cutoff date
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        analytics = {}

        try:
            # Total searches
            cursor.execute("""
                SELECT COUNT(*) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            analytics['total_searches'] = cursor.fetchone()[0]

            # Unique queries
            cursor.execute("""
                SELECT COUNT(DISTINCT query) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            analytics['unique_queries'] = cursor.fetchone()[0]

            # Average results count
            cursor.execute("""
                SELECT AVG(results_count) FROM search_log
                WHERE timestamp >= ?
            """, (cutoff_date,))
            avg_results = cursor.fetchone()[0]
            analytics['avg_results'] = round(avg_results, 2) if avg_results else 0

            # Average execution time
            cursor.execute("""
                SELECT AVG(execution_time_ms) FROM search_log
                WHERE timestamp >= ? AND execution_time_ms IS NOT NULL
            """, (cutoff_date,))
            avg_time = cursor.fetchone()[0]
            analytics['avg_execution_time_ms'] = round(avg_time, 2) if avg_time else 0

            # Top queries (most frequent)
            cursor.execute("""
                SELECT query, COUNT(*) as count, AVG(results_count) as avg_results
                FROM search_log
                WHERE timestamp >= ?
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, limit))
            analytics['top_queries'] = [
                {
                    'query': row[0],
                    'count': row[1],
                    'avg_results': round(row[2], 1) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]

            # Failed searches (zero results)
            cursor.execute("""
                SELECT query, COUNT(*) as count
                FROM search_log
                WHERE timestamp >= ? AND results_count = 0
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, min(limit, 20)))
            analytics['failed_searches'] = [
                {'query': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]

            # Search mode breakdown
            cursor.execute("""
                SELECT search_mode, COUNT(*) as count, AVG(results_count) as avg_results
                FROM search_log
                WHERE timestamp >= ?
                GROUP BY search_mode
                ORDER BY count DESC
            """, (cutoff_date,))
            analytics['search_modes'] = [
                {
                    'mode': row[0],
                    'count': row[1],
                    'avg_results': round(row[2], 1) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]

            # Popular tags
            cursor.execute("""
                SELECT tags, COUNT(*) as count
                FROM search_log
                WHERE timestamp >= ? AND tags IS NOT NULL
                GROUP BY tags
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff_date, 20))
            tag_counts = {}
            for row in cursor.fetchall():
                tags_str = row[0]
                count = row[1]
                # Split tags and count individually
                for tag in tags_str.split(','):
                    tag = tag.strip()
                    tag_counts[tag] = tag_counts.get(tag, 0) + count

            analytics['popular_tags'] = [
                {'tag': tag, 'count': count}
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            ]

        except Exception as e:
            self.logger.error(f"Error getting search analytics: {e}")
            analytics['error'] = str(e)

        return analytics

    def create_backup(self, dest_dir: str, compress: bool = True) -> str:
        """
        Create full backup of knowledge base.

        Args:
            dest_dir: Destination directory for backup
            compress: Whether to compress backup to zip file (default: True)

        Returns:
            Path to backup (directory or zip file)
        """
        import shutil

        self.logger.info(f"Creating backup to {dest_dir}")
        start_time = time.time()

        # Create backup directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"kb_backup_{timestamp}"
        backup_path = Path(dest_dir) / backup_name

        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup database file
            db_source = Path(self.data_dir) / "knowledge_base.db"
            db_dest = backup_path / "knowledge_base.db"
            if db_source.exists():
                shutil.copy2(db_source, db_dest)
                self.logger.info(f"Backed up database: {db_source.stat().st_size} bytes")

            # Backup embeddings if they exist
            embeddings_path = Path(self.data_dir) / "embeddings.faiss"
            embeddings_map_path = Path(self.data_dir) / "embeddings_map.json"

            if embeddings_path.exists():
                shutil.copy2(embeddings_path, backup_path / "embeddings.faiss")
                self.logger.info(f"Backed up embeddings index: {embeddings_path.stat().st_size} bytes")

            if embeddings_map_path.exists():
                shutil.copy2(embeddings_map_path, backup_path / "embeddings_map.json")
                self.logger.info("Backed up embeddings map")

            # Create metadata file
            metadata = {
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'document_count': len(self.documents),
                'total_chunks': sum(doc.total_chunks for doc in self.documents.values()),
                'database_size_bytes': db_source.stat().st_size if db_source.exists() else 0,
                'has_embeddings': embeddings_path.exists(),
                'version': '2.5.0'
            }

            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Created backup metadata: {metadata}")

            # Compress if requested
            if compress:
                self.logger.info("Compressing backup...")
                zip_path = shutil.make_archive(str(backup_path), 'zip', backup_path)
                shutil.rmtree(backup_path)  # Remove uncompressed directory

                elapsed = time.time() - start_time
                self.logger.info(f"Backup completed in {elapsed:.2f}s: {zip_path}")
                return zip_path
            else:
                elapsed = time.time() - start_time
                self.logger.info(f"Backup completed in {elapsed:.2f}s: {backup_path}")
                return str(backup_path)

        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise

    def restore_from_backup(self, backup_path: str, verify: bool = True) -> dict:
        """
        Restore knowledge base from backup.

        Args:
            backup_path: Path to backup (directory or zip file)
            verify: Whether to verify backup integrity before restoring (default: True)

        Returns:
            Restoration metadata dict
        """
        import shutil
        import zipfile

        self.logger.info(f"Restoring from backup: {backup_path}")
        start_time = time.time()

        backup_path_obj = Path(backup_path)
        temp_dir = None

        try:
            # Extract if compressed
            if backup_path_obj.suffix == '.zip':
                self.logger.info("Extracting compressed backup...")
                temp_dir = Path(self.data_dir) / f"temp_restore_{int(time.time())}"
                temp_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find the backup directory inside temp_dir
                extracted_dirs = list(temp_dir.iterdir())
                if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
                    restore_from = extracted_dirs[0]
                else:
                    restore_from = temp_dir
            else:
                restore_from = backup_path_obj

            # Verify backup if requested
            if verify:
                self.logger.info("Verifying backup integrity...")

                # Check for required files
                db_file = restore_from / "knowledge_base.db"
                metadata_file = restore_from / "metadata.json"

                if not db_file.exists():
                    raise ValueError("Backup is missing database file")

                if not metadata_file.exists():
                    raise ValueError("Backup is missing metadata file")

                # Load and validate metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.logger.info(f"Backup metadata: {metadata}")

            # Close current database connection
            self.close()

            # Backup current database before overwriting (safety measure)
            current_db = Path(self.data_dir) / "knowledge_base.db"
            if current_db.exists():
                safety_backup = Path(self.data_dir) / f"knowledge_base_pre_restore_{int(time.time())}.db"
                shutil.copy2(current_db, safety_backup)
                self.logger.info(f"Created safety backup: {safety_backup}")

            # Restore database
            db_source = restore_from / "knowledge_base.db"
            db_dest = Path(self.data_dir) / "knowledge_base.db"
            shutil.copy2(db_source, db_dest)
            self.logger.info(f"Restored database: {db_source.stat().st_size} bytes")

            # Restore embeddings if they exist in backup
            embeddings_source = restore_from / "embeddings.faiss"
            embeddings_map_source = restore_from / "embeddings_map.json"

            if embeddings_source.exists():
                embeddings_dest = Path(self.data_dir) / "embeddings.faiss"
                shutil.copy2(embeddings_source, embeddings_dest)
                self.logger.info("Restored embeddings index")

            if embeddings_map_source.exists():
                embeddings_map_dest = Path(self.data_dir) / "embeddings_map.json"
                shutil.copy2(embeddings_map_source, embeddings_map_dest)
                self.logger.info("Restored embeddings map")

            # Reload knowledge base
            self.logger.info("Reloading knowledge base...")
            self._init_database()
            self._load_documents()

            # Reload embeddings if they exist
            if self.use_semantic and embeddings_source.exists():
                self._load_embeddings()

            elapsed = time.time() - start_time

            result = {
                'success': True,
                'backup_metadata': metadata,
                'restored_documents': len(self.documents),
                'elapsed_seconds': elapsed
            }

            self.logger.info(f"Restore completed in {elapsed:.2f}s: {len(self.documents)} documents")
            return result

        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            raise
        finally:
            # Cleanup temporary extraction directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
