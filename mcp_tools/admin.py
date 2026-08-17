"""Admin, stats, analytics, backup, bulk-operation, tagging and
summarization handlers. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""

import os

from mcp.types import TextContent


def handle_kb_stats(kb, name: str, arguments: dict) -> list[TextContent]:
    stats = kb.get_stats()
    output = "Knowledge Base Statistics:\n"
    output += f"  Documents: {stats['total_documents']}\n"
    output += f"  Chunks: {stats['total_chunks']}\n"
    output += f"  Total Words: {stats['total_words']:,}\n"
    output += f"  File Types: {', '.join(stats['file_types']) or 'none'}\n"
    output += f"  Tags: {', '.join(stats['all_tags']) or 'none'}\n"
    return [TextContent(type="text", text=output)]


def handle_health_check(kb, name: str, arguments: dict) -> list[TextContent]:
    health = kb.health_check()

    # Format output
    output = f"System Health Check\n{'='*50}\n\n"
    output += f"Status: {health['status'].upper()}\n"
    output += f"Message: {health['message']}\n\n"

    # Metrics
    if health['metrics']:
        output += "Metrics:\n"
        for key, value in health['metrics'].items():
            if isinstance(value, int):
                output += f"  {key}: {value:,}\n"
            else:
                output += f"  {key}: {value}\n"
        output += "\n"

    # Database
    if health['database']:
        output += "Database:\n"
        for key, value in health['database'].items():
            output += f"  {key}: {value}\n"
        output += "\n"

    # Features
    if health['features']:
        output += "Features:\n"
        # Lazily-built features are expected to read False until their
        # first use (e.g. the BM25 index only builds on first BM25
        # search) - that's not a problem, so don't flag it with a ✗
        # right next to "No issues detected" below.
        lazy_features = {'bm25_index_built'}
        for key, value in health['features'].items():
            if key in lazy_features and not value:
                status = "ⓘ"
                output += f"  {status} {key}: {value} (builds lazily on first use)\n"
            else:
                status = "✓" if value else "✗"
                output += f"  {status} {key}: {value}\n"
        output += "\n"

    # Performance
    if health['performance']:
        output += "Performance:\n"
        for key, value in health['performance'].items():
            output += f"  {key}: {value}\n"
        output += "\n"

    # Issues
    if health['issues']:
        output += f"Issues ({len(health['issues'])}):\n"
        for i, issue in enumerate(health['issues'], 1):
            output += f"  {i}. {issue}\n"
    else:
        output += "✓ No issues detected\n"

    return [TextContent(type="text", text=output)]


def handle_reconcile_chunk_cache(kb, name: str, arguments: dict) -> list[TextContent]:
    result = kb.reconcile_chunk_cache()

    output = f"Chunk Cache Reconciliation\n{'='*50}\n\n"
    output += f"Chunks before: {result['chunks_before']:,}\n"
    output += f"Chunks after:  {result['chunks_after']:,}\n"
    output += f"Chunks pruned: {result['chunks_pruned']:,}\n\n"

    if result['orphaned_doc_ids']:
        output += f"Orphaned doc_ids removed from cache ({len(result['orphaned_doc_ids'])}):\n"
        for doc_id in result['orphaned_doc_ids']:
            output += f"  - {doc_id}\n"
    else:
        output += "No orphaned doc_ids found - cache was already consistent with the database.\n"

    return [TextContent(type="text", text=output)]


def handle_reconcile_embeddings(kb, name: str, arguments: dict) -> list[TextContent]:
    max_docs = arguments.get("max_docs")
    result = kb.reconcile_embeddings(max_docs=max_docs)

    if 'error' in result:
        return [TextContent(type="text", text=f"Error: {result['error']}")]

    output = f"Embeddings Reconciliation\n{'='*50}\n\n"
    output += f"Documents covered before: {result['docs_covered_before']:,} / {result['total_documents']:,}\n"
    output += f"Documents covered after:  {result['docs_covered_after']:,} / {result['total_documents']:,}\n"
    output += f"Backfilled this call:     {result['docs_backfilled_this_call']:,}\n"
    output += f"Still missing:            {result['docs_still_missing']:,}\n\n"
    output += f"Chunks before: {result['chunks_before']:,}\n"
    output += f"Chunks after:  {result['chunks_after']:,}\n"

    if result['docs_still_missing'] > 0:
        output += (
            f"\n{result['docs_still_missing']} document(s) still have no embeddings. "
            "Run this tool again (optionally with max_docs to process in smaller batches).\n"
        )

    return [TextContent(type="text", text=output)]


def handle_detect_anomalies(kb, name: str, arguments: dict) -> list[TextContent]:
    min_severity = arguments.get("min_severity", "moderate")
    days = arguments.get("days", 7)

    results = kb.detect_anomalies(min_severity=min_severity, days=days)

    if 'error' in results:
        return [TextContent(type="text", text=f"Error: {results['error']}")]

    # Format output
    output = f"Anomaly Detection Results\n{'='*60}\n\n"
    output += f"Time Range: Last {results['time_range_days']} days\n"
    output += f"Minimum Severity: {min_severity}\n"
    output += f"Total Anomalies: {results['total_anomalies']}\n"

    if results['by_severity']:
        output += f"\nBreakdown by Severity:\n"
        for severity, count in sorted(results['by_severity'].items()):
            output += f"  {severity.title()}: {count}\n"

    if results['total_anomalies'] > 0:
        output += f"\nAverage Score: {results['avg_score']}/100\n\n"
        output += f"Anomalies Detected:\n"
        output += "-" * 60 + "\n\n"

        for anomaly in results['anomalies']:
            output += f"Document: {anomaly.get('doc_title', 'Unknown')}\n"
            output += f"  Severity: {anomaly.get('severity', 'unknown').upper()} (Score: {anomaly.get('score', 0)}/100)\n"
            output += f"  Check Date: {anomaly.get('check_date', 'N/A')}\n"
            output += f"  Status: {anomaly.get('status', 'N/A')}\n"

            if 'components' in anomaly:
                output += f"  Score Components:\n"
                for component, score in anomaly['components'].items():
                    output += f"    - {component}: {score:.1f}\n"

            if 'change_type' in anomaly and anomaly['change_type']:
                output += f"  Change Type: {anomaly['change_type']}\n"

            if 'http_status' in anomaly and anomaly['http_status']:
                output += f"  HTTP Status: {anomaly['http_status']}\n"

            if 'error_message' in anomaly and anomaly['error_message']:
                output += f"  Error: {anomaly['error_message']}\n"

            output += "\n"
    else:
        output += "\n✓ No anomalies detected in the specified time range.\n"

    return [TextContent(type="text", text=output)]


def handle_search_analytics(kb, name: str, arguments: dict) -> list[TextContent]:
    days = arguments.get("days", 30)
    limit = arguments.get("limit", 100)

    analytics = kb.get_search_analytics(days, limit)

    if 'error' in analytics:
        return [TextContent(type="text", text=f"Error getting analytics: {analytics['error']}")]

    # Format output
    output = f"Search Analytics (Last {days} days)\n{'='*50}\n\n"

    # Overall stats
    output += "Overview:\n"
    output += f"  Total Searches: {analytics.get('total_searches', 0):,}\n"
    output += f"  Unique Queries: {analytics.get('unique_queries', 0):,}\n"
    output += f"  Avg Results per Search: {analytics.get('avg_results', 0)}\n"
    output += f"  Avg Execution Time: {analytics.get('avg_execution_time_ms', 0):.2f}ms\n\n"

    # Search modes
    if analytics.get('search_modes'):
        output += "Search Mode Usage:\n"
        for mode in analytics['search_modes']:
            output += f"  {mode['mode']}: {mode['count']:,} searches (avg {mode['avg_results']} results)\n"
        output += "\n"

    # Top queries
    if analytics.get('top_queries'):
        output += f"Top {min(10, len(analytics['top_queries']))} Most Popular Queries:\n"
        for i, query in enumerate(analytics['top_queries'][:10], 1):
            output += f"  {i}. \"{query['query']}\" - {query['count']} times (avg {query['avg_results']} results)\n"
        output += "\n"

    # Failed searches
    if analytics.get('failed_searches'):
        output += f"Top {min(10, len(analytics['failed_searches']))} Failed Searches (0 results):\n"
        for i, failed in enumerate(analytics['failed_searches'][:10], 1):
            output += f"  {i}. \"{failed['query']}\" - {failed['count']} times\n"
        output += "\n"

    # Popular tags
    if analytics.get('popular_tags'):
        output += f"Top {min(10, len(analytics['popular_tags']))} Most Used Tags:\n"
        for i, tag in enumerate(analytics['popular_tags'][:10], 1):
            output += f"  {i}. {tag['tag']}: {tag['count']} searches\n"

    return [TextContent(type="text", text=output)]


def handle_find_by_reference(kb, name: str, arguments: dict) -> list[TextContent]:
    ref_type = arguments.get("ref_type")
    ref_value = arguments.get("ref_value")
    max_results = arguments.get("max_results", 10)

    if not ref_type or not ref_value:
        return [TextContent(type="text", text="Error: ref_type and ref_value are required")]

    results = kb.find_by_reference(ref_type, ref_value, max_results)

    if not results:
        return [TextContent(type="text", text=f"No references found for {ref_type}={ref_value}")]

    output = f"Found {len(results)} references for {ref_type}={ref_value}:\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- Reference {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Type: {r['ref_type']}, Value: {r['ref_value']}\n"
        output += f"Context:\n{r['context']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_check_updates(kb, name: str, arguments: dict) -> list[TextContent]:
    auto_update = arguments.get("auto_update", False)
    results = kb.check_all_updates(auto_update)

    output = "Document Update Check:\n\n"

    if results['unchanged']:
        output += f"✓ {len(results['unchanged'])} documents unchanged\n"

    if results['changed']:
        output += f"⚠ {len(results['changed'])} documents changed:\n"
        for doc in results['changed']:
            output += f"  - {doc['title']} ({doc['filepath']})\n"
        output += "\n"

    if results['missing']:
        output += f"✗ {len(results['missing'])} documents missing (files not found):\n"
        for doc in results['missing']:
            output += f"  - {doc['title']} ({doc['filepath']})\n"
        output += "\n"

    if auto_update and results['updated']:
        output += f"✓ {len(results['updated'])} documents re-indexed:\n"
        for doc in results['updated']:
            output += f"  - {doc['title']} ({doc['filepath']})\n"

    if not auto_update and results['changed']:
        output += "\nRun with auto_update=true to automatically re-index changed documents.\n"

    return [TextContent(type="text", text=output)]


def handle_add_documents_bulk(kb, name: str, arguments: dict) -> list[TextContent]:
    directory = arguments.get("directory")
    pattern = arguments.get("pattern", "**/*.{pdf,txt}")
    tags = arguments.get("tags")
    recursive = arguments.get("recursive", True)
    skip_duplicates = arguments.get("skip_duplicates", True)

    try:
        # add_documents_bulk's driving loop (`for future in
        # as_completed(...)`) is a plain blocking call. Running it on the
        # asyncio event loop thread made every other request on this MCP
        # session (even a trivial read-only kb_stats call) queue behind the
        # entire batch, hanging the session for as long as the whole bulk
        # operation took (issue #13). The per-tool asyncio.to_thread that
        # first fixed that is gone: this entire dispatch now runs on a
        # worker thread (see call_tool), so a direct call is already
        # off-loop.
        results = kb.add_documents_bulk(directory, pattern, tags, recursive, skip_duplicates)
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk add: {str(e)}")]

    output = "Bulk Document Add Results:\n\n"

    if results['added']:
        output += f"✓ {len(results['added'])} documents added:\n"
        for doc in results['added']:
            # add_documents_bulk's returned dicts carry 'filepath', not
            # 'filename' - unlike search results and other tool outputs
            # in this dispatch. Every real (non-duplicate) call to this
            # tool raised KeyError here before this fix.
            output += f"  - {doc['title']} ({os.path.basename(doc['filepath'])})\n"
            output += f"    ID: {doc['doc_id']}, Chunks: {doc['chunks']}\n"
        output += "\n"

    if results['skipped']:
        output += f"⊘ {len(results['skipped'])} documents skipped (duplicates):\n"
        for doc in results['skipped']:
            output += f"  - {doc['filepath']}\n"
        output += "\n"

    if results['failed']:
        output += f"✗ {len(results['failed'])} documents failed:\n"
        for failure in results['failed']:
            output += f"  - {failure['filepath']}: {failure['error']}\n"
        output += "\n"

    output += f"Total: {len(results['added'])} added, {len(results['skipped'])} skipped, {len(results['failed'])} failed"

    return [TextContent(type="text", text=output)]


def handle_remove_documents_bulk(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_ids = arguments.get("doc_ids")
    tags = arguments.get("tags")

    try:
        results = kb.remove_documents_bulk(doc_ids, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk remove: {str(e)}")]

    output = "Bulk Document Remove Results:\n\n"

    if results['removed']:
        output += f"✓ {len(results['removed'])} documents removed:\n"
        for doc_id in results['removed']:
            output += f"  - {doc_id}\n"
        output += "\n"

    if results['failed']:
        output += f"✗ {len(results['failed'])} documents failed to remove:\n"
        for failure in results['failed']:
            output += f"  - {failure['doc_id']}: {failure['error']}\n"
        output += "\n"

    output += f"Total: {len(results['removed'])} removed, {len(results['failed'])} failed"

    return [TextContent(type="text", text=output)]


def handle_update_tags_bulk(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_ids = arguments.get("doc_ids")
    existing_tags = arguments.get("existing_tags")
    add_tags = arguments.get("add_tags")
    remove_tags = arguments.get("remove_tags")
    replace_tags = arguments.get("replace_tags")

    try:
        results = kb.update_tags_bulk(
            doc_ids=doc_ids,
            existing_tags=existing_tags,
            add_tags=add_tags,
            remove_tags=remove_tags,
            replace_tags=replace_tags
        )
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk tag update: {str(e)}")]

    output = "Bulk Tag Update Results:\n\n"

    if results['updated']:
        output += f"✓ {len(results['updated'])} documents updated:\n"
        for update in results['updated']:
            output += f"  - {update['doc_id']}\n"
            output += f"    Old tags: {', '.join(update['old_tags']) if update['old_tags'] else 'None'}\n"
            output += f"    New tags: {', '.join(update['new_tags']) if update['new_tags'] else 'None'}\n"
        output += "\n"

    if results['failed']:
        output += f"✗ {len(results['failed'])} documents failed to update:\n"
        for failure in results['failed']:
            output += f"  - {failure['doc_id']}: {failure['error']}\n"
        output += "\n"

    output += f"Total: {len(results['updated'])} updated, {len(results['failed'])} failed"

    return [TextContent(type="text", text=output)]


def handle_export_documents_bulk(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_ids = arguments.get("doc_ids")
    tags = arguments.get("tags")
    format = arguments.get("format", "json")

    try:
        export_data = kb.export_documents_bulk(doc_ids=doc_ids, tags=tags, format=format)
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk export: {str(e)}")]

    # Determine document count from the export
    if format == "json":
        import json as json_lib
        doc_count = len(json_lib.loads(export_data))
    else:
        doc_count = export_data.count('\n') if format == 'csv' else export_data.count('## ') - 1

    output = f"Document Export ({format.upper()}):\n\n"
    output += f"Exported {doc_count} document(s)\n\n"
    output += "=" * 80 + "\n\n"
    output += export_data

    return [TextContent(type="text", text=output)]


def handle_search_tables(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")

    try:
        results = kb.search_tables(query, max_results, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching tables: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No tables found for query: '{query}'")]

    output = f"Found {len(results)} table(s) for '{query}':\n\n"

    for i, result in enumerate(results, 1):
        output += f"Result {i}:\n"
        output += f"  Document: {result['doc_title']}\n"
        output += f"  Page: {result['page']}\n"
        output += f"  Size: {result['row_count']} rows × {result['col_count']} columns\n"
        output += f"  Score: {result['score']:.2f}\n\n"
        output += f"Table content:\n{result['markdown']}\n\n"
        output += "-" * 80 + "\n\n"

    return [TextContent(type="text", text=output)]


def handle_search_code(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    block_type = arguments.get("block_type")
    tags = arguments.get("tags")

    try:
        results = kb.search_code(query, max_results, block_type, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching code: {str(e)}")]

    if not results:
        type_filter = f" (type: {block_type})" if block_type else ""
        return [TextContent(type="text", text=f"No code blocks found for query: '{query}'{type_filter}")]

    output = f"Found {len(results)} code block(s) for '{query}':\n\n"

    for i, result in enumerate(results, 1):
        output += f"Result {i}:\n"
        output += f"  Document: {result['doc_title']}\n"
        output += f"  Page: {result['page'] or 'N/A'}\n"
        output += f"  Type: {result['block_type']}\n"
        output += f"  Lines: {result['line_count']}\n"
        output += f"  Score: {result['score']:.2f}\n\n"
        output += f"Code:\n```{result['block_type']}\n{result['code']}\n```\n\n"
        output += "-" * 80 + "\n\n"

    return [TextContent(type="text", text=output)]


def handle_suggest_queries(kb, name: str, arguments: dict) -> list[TextContent]:
    partial = arguments.get("partial", "")
    max_suggestions = arguments.get("max_suggestions", 5)
    category = arguments.get("category")

    if not partial:
        return [TextContent(type="text", text="Error: partial query string is required")]

    try:
        suggestions = kb.get_query_suggestions(partial, max_suggestions, category)
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting suggestions: {str(e)}")]

    if not suggestions:
        return [TextContent(type="text", text=f"No suggestions found for: '{partial}'")]

    output = f"Query suggestions for '{partial}':\n\n"
    for i, sug in enumerate(suggestions, 1):
        output += f"{i}. {sug['term']} ({sug['category']}) - used {sug['frequency']} times\n"

    return [TextContent(type="text", text=output)]


def handle_export_results(kb, name: str, arguments: dict) -> list[TextContent]:
    results = arguments.get("results", [])
    format = arguments.get("format", "markdown")
    query = arguments.get("query")

    if not results:
        return [TextContent(type="text", text="Error: results array is required")]

    try:
        exported = kb.export_search_results(results, format, query)

        # Return the exported content
        output = f"Search results exported to {format} format:\n\n"
        output += "=" * 80 + "\n\n"
        output += exported

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error exporting results: {str(e)}")]


def handle_create_backup(kb, name: str, arguments: dict) -> list[TextContent]:
    dest_dir = arguments.get("dest_dir")
    compress = arguments.get("compress", True)

    if not dest_dir:
        return [TextContent(type="text", text="Error: dest_dir is required")]

    try:
        backup_path = kb.create_backup(dest_dir, compress)

        output = "✓ Backup created successfully!\n\n"
        output += f"Location: {backup_path}\n"
        output += f"Format: {'Compressed (ZIP)' if compress else 'Uncompressed directory'}\n\n"
        output += "The backup includes:\n"
        output += f"- Database ({len(kb.documents)} documents)\n"
        output += "- Embeddings (if available)\n"
        output += "- Metadata file with timestamp and version info\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating backup: {str(e)}")]


def handle_restore_backup(kb, name: str, arguments: dict) -> list[TextContent]:
    backup_path = arguments.get("backup_path")
    verify = arguments.get("verify", True)

    if not backup_path:
        return [TextContent(type="text", text="Error: backup_path is required")]

    try:
        result = kb.restore_from_backup(backup_path, verify)

        output = "✓ Restore completed successfully!\n\n"
        output += f"Backup source: {backup_path}\n"
        output += f"Documents restored: {result['restored_documents']}\n"
        output += f"Time elapsed: {result['elapsed_seconds']:.2f}s\n\n"

        if 'backup_metadata' in result:
            metadata = result['backup_metadata']
            output += "Backup info:\n"
            output += f"- Created: {metadata.get('created_at', 'Unknown')}\n"
            output += f"- Version: {metadata.get('version', 'Unknown')}\n"
            output += f"- Original document count: {metadata.get('document_count', 'Unknown')}\n"

        output += "\nNote: A safety backup was created before restoration."

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error restoring backup: {str(e)}")]


def handle_auto_tag_document(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    confidence_threshold = arguments.get("confidence_threshold", 0.7)
    max_tags = arguments.get("max_tags", 10)
    append = arguments.get("append", True)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        result = kb.auto_tag_document(
            doc_id,
            confidence_threshold=confidence_threshold,
            max_tags=max_tags,
            append=append
        )

        output = "✓ Auto-tagged document successfully!\n\n"
        output += f"**Document:** {result['doc_title']}\n"
        output += f"**Document ID:** {doc_id}\n\n"

        output += f"**Applied Tags ({len(result['applied_tags'])}):**\n"
        for tag_info in result['suggested_tags']:
            if tag_info['tag'] in result['applied_tags']:
                output += f"  - {tag_info['tag']} (confidence: {tag_info['confidence']:.2f})\n"
                output += f"    Reason: {tag_info.get('reason', 'N/A')}\n"

        if result['skipped_tags']:
            output += f"\n**Skipped Tags (below {confidence_threshold} threshold):**\n"
            for tag_info in result['suggested_tags']:
                if tag_info['tag'] in result['skipped_tags']:
                    output += f"  - {tag_info['tag']} (confidence: {tag_info['confidence']:.2f})\n"

        output += "\n**Tag Summary:**\n"
        output += f"  - Existing tags: {', '.join(result['existing_tags']) if result['existing_tags'] else 'None'}\n"
        output += f"  - New tags: {', '.join(result['new_tags'])}\n"
        output += f"  - Total tags added: {len(set(result['new_tags']) - set(result['existing_tags']))}\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error auto-tagging document: {str(e)}\n\nNote: Auto-tagging requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


def handle_auto_tag_all(kb, name: str, arguments: dict) -> list[TextContent]:
    confidence_threshold = arguments.get("confidence_threshold", 0.7)
    max_tags = arguments.get("max_tags", 10)
    append = arguments.get("append", True)
    skip_tagged = arguments.get("skip_tagged", True)
    max_docs = arguments.get("max_docs")

    try:
        results = kb.auto_tag_all_documents(
            confidence_threshold=confidence_threshold,
            max_tags=max_tags,
            append=append,
            skip_tagged=skip_tagged,
            max_docs=max_docs
        )

        output = "✓ Bulk auto-tagging complete!\n\n"
        output += "**Statistics:**\n"
        output += f"  - Documents processed: {results['processed']}\n"
        output += f"  - Documents skipped: {results['skipped']}\n"
        output += f"  - Documents failed: {results['failed']}\n"
        output += f"  - Total tags added: {results['total_tags_added']}\n\n"

        if results['processed'] > 0:
            output += "**Sample Results (first 5):**\n"
            for i, result in enumerate(results['results'][:5], 1):
                if 'error' in result:
                    output += f"\n{i}. {result['doc_id']} - ERROR: {result['error']}\n"
                else:
                    output += f"\n{i}. {result.get('doc_title', 'Unknown')}\n"
                    output += f"   - Applied: {', '.join(result['applied_tags']) if result['applied_tags'] else 'None'}\n"
                    output += f"   - Total tags: {len(result['new_tags'])}\n"

            if results['processed'] > 5:
                output += f"\n... and {results['processed'] - 5} more documents\n"

        if results['failed'] > 0:
            output += f"\n**Warning:** {results['failed']} documents failed to process. Check logs for details.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk auto-tagging: {str(e)}\n\nNote: Auto-tagging requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


def handle_summarize_document(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    summary_type = arguments.get("summary_type", "brief")
    force_regenerate = arguments.get("force_regenerate", False)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        summary = kb.generate_summary(
            doc_id,
            summary_type=summary_type,
            force_regenerate=force_regenerate
        )

        output = f"✓ Summary generated ({summary_type})\n\n"
        output += f"**Document:** {kb.documents[doc_id].title}\n\n"
        output += f"**Summary ({summary_type}):**\n\n"
        output += summary

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating summary: {str(e)}\n\nNote: Summarization requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


def handle_get_summary(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    summary_type = arguments.get("summary_type", "brief")

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        summary = kb.get_summary(doc_id, summary_type)

        if not summary:
            return [TextContent(type="text", text=f"No cached summary found for {doc_id} ({summary_type}). Use 'summarize_document' to generate one.")]

        output = f"✓ Cached summary retrieved ({summary_type})\n\n"
        output += f"**Document:** {kb.documents[doc_id].title}\n\n"
        output += f"**Summary ({summary_type}):**\n\n"
        output += summary

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error retrieving summary: {str(e)}")]


def handle_summarize_all(kb, name: str, arguments: dict) -> list[TextContent]:
    summary_types = arguments.get("summary_types", ["brief"])
    force_regenerate = arguments.get("force_regenerate", False)
    max_docs = arguments.get("max_docs")

    try:
        results = kb.generate_summary_all(
            summary_types=summary_types,
            force_regenerate=force_regenerate,
            max_docs=max_docs
        )

        output = "✓ Bulk summarization complete!\n\n"
        output += "**Statistics:**\n"
        output += f"  - Documents processed: {results['processed']}\n"
        output += f"  - Documents failed: {results['failed']}\n"
        output += f"  - Total summaries generated: {results['total_summaries']}\n"
        output += "  - By type:\n"
        for summary_type, count in results['by_type'].items():
            output += f"    - {summary_type}: {count}\n"

        output += "\n**Sample Results (first 3):**\n"
        for i, result in enumerate(results['results'][:3], 1):
            output += f"\n{i}. {result['title']}\n"
            for summary_type, summary_result in result['summaries'].items():
                if summary_result['success']:
                    output += f"   - {summary_type}: {summary_result['word_count']} words\n"
                else:
                    output += f"   - {summary_type}: ERROR - {summary_result['error']}\n"

        if results['processed'] > 3:
            output += f"\n... and {results['processed'] - 3} more documents\n"

        if results['failed'] > 0:
            output += f"\n**Warning:** {results['failed']} documents failed to process. Check logs for details.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk summarization: {str(e)}\n\nNote: Summarization requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


HANDLERS_ADMIN = {
    "kb_stats": handle_kb_stats,
    "health_check": handle_health_check,
    "reconcile_chunk_cache": handle_reconcile_chunk_cache,
    "reconcile_embeddings": handle_reconcile_embeddings,
    "detect_anomalies": handle_detect_anomalies,
    "search_analytics": handle_search_analytics,
    "find_by_reference": handle_find_by_reference,
    "check_updates": handle_check_updates,
    "add_documents_bulk": handle_add_documents_bulk,
    "remove_documents_bulk": handle_remove_documents_bulk,
    "update_tags_bulk": handle_update_tags_bulk,
    "export_documents_bulk": handle_export_documents_bulk,
    "search_tables": handle_search_tables,
    "search_code": handle_search_code,
    "suggest_queries": handle_suggest_queries,
    "export_results": handle_export_results,
    "create_backup": handle_create_backup,
    "restore_backup": handle_restore_backup,
    "auto_tag_document": handle_auto_tag_document,
    "auto_tag_all": handle_auto_tag_all,
    "summarize_document": handle_summarize_document,
    "get_summary": handle_get_summary,
    "summarize_all": handle_summarize_all,
}
