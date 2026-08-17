"""Entity extraction, search, relationships, comparison, export, and
async extraction-job handlers. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


def handle_extract_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    confidence_threshold = arguments.get("confidence_threshold", 0.6)
    force_regenerate = arguments.get("force_regenerate", False)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        result = kb.extract_entities(
            doc_id,
            confidence_threshold=confidence_threshold,
            force_regenerate=force_regenerate
        )

        output = "✓ Entity extraction complete!\n\n"
        output += f"**Document:** {result['doc_title']}\n"
        output += f"**Document ID:** {doc_id}\n"
        output += f"**Entities Found:** {result['entity_count']}\n\n"

        # Group by entity type
        if result['entities']:
            for entity_type in sorted(result['types'].keys()):
                entities_of_type = [e for e in result['entities'] if e['entity_type'] == entity_type]
                output += f"**{entity_type.upper().replace('_', ' ')}** ({len(entities_of_type)}):\n"

                # Show first 5 of each type
                for entity in entities_of_type[:5]:
                    output += f"  - **{entity['entity_text']}** (confidence: {entity['confidence']:.2f}"
                    if entity.get('occurrence_count', 1) > 1:
                        output += f", occurs {entity['occurrence_count']}x"
                    output += ")\n"
                    if entity.get('context'):
                        context = entity['context'][:80] + "..." if len(entity['context']) > 80 else entity['context']
                        output += f"    *{context}*\n"

                if len(entities_of_type) > 5:
                    output += f"  ... and {len(entities_of_type) - 5} more\n"
                output += "\n"

            output += "Use `list_entities` tool to see all entities with filtering options.\n"
        else:
            output += "No entities found with the current confidence threshold.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting entities: {str(e)}\n\nNote: Entity extraction requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


def handle_list_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    entity_types = arguments.get("entity_types")
    min_confidence = arguments.get("min_confidence", 0.0)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        result = kb.get_entities(
            doc_id,
            entity_types=entity_types,
            min_confidence=min_confidence
        )

        output = f"**Entities for Document:** {result['doc_title']}\n"
        output += f"**Document ID:** {doc_id}\n"

        # Show filters if applied
        if entity_types or min_confidence > 0:
            output += "**Filters:** "
            filters = []
            if entity_types:
                filters.append(f"types={', '.join(entity_types)}")
            if min_confidence > 0:
                filters.append(f"min_confidence={min_confidence}")
            output += ', '.join(filters) + "\n"

        output += f"**Total Entities:** {result['entity_count']}\n\n"

        if result['entities']:
            # Group by type
            for entity_type in sorted(result['types'].keys()):
                entities_of_type = [e for e in result['entities'] if e['entity_type'] == entity_type]
                output += f"**{entity_type.upper().replace('_', ' ')}** ({len(entities_of_type)}):\n"

                for entity in entities_of_type:
                    output += f"  - **{entity['entity_text']}** (conf: {entity['confidence']:.2f}"
                    if entity.get('occurrence_count', 1) > 1:
                        output += f", {entity['occurrence_count']}x"
                    output += ")\n"
                    if entity.get('context'):
                        context = entity['context'][:100] + "..." if len(entity['context']) > 100 else entity['context']
                        output += f"    *{context}*\n"

                output += "\n"
        else:
            output += "No entities found. Use `extract_entities` tool to extract entities first.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing entities: {str(e)}")]


def handle_search_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query")
    entity_types = arguments.get("entity_types")
    min_confidence = arguments.get("min_confidence", 0.0)
    max_results = arguments.get("max_results", 20)

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    try:
        result = kb.search_entities(
            query,
            entity_types=entity_types,
            min_confidence=min_confidence,
            max_results=max_results
        )

        output = f"**Entity Search Results for:** {result['query']}\n"
        output += f"**Total Matches:** {result['total_matches']}\n"

        # Show filters if applied
        if entity_types or min_confidence > 0:
            output += "**Filters:** "
            filters = []
            if entity_types:
                filters.append(f"types={', '.join(entity_types)}")
            if min_confidence > 0:
                filters.append(f"min_confidence={min_confidence}")
            output += ', '.join(filters) + "\n"

        output += f"**Documents Found:** {len(result['documents'])}\n\n"

        if result['documents']:
            for doc in result['documents']:
                output += f"**{doc['doc_title']}** ({doc['doc_id']})\n"
                output += f"  Matches: {doc['match_count']}\n"

                # Show first 3 matches per document
                for match in doc['matches'][:3]:
                    output += f"  - **{match['entity_text']}** ({match['entity_type']}, conf: {match['confidence']:.2f}"
                    if match.get('occurrence_count', 1) > 1:
                        output += f", {match['occurrence_count']}x"
                    output += ")\n"
                    if match.get('context'):
                        context = match['context'][:80] + "..." if len(match['context']) > 80 else match['context']
                        output += f"    *{context}*\n"

                if doc['match_count'] > 3:
                    output += f"  ... and {doc['match_count'] - 3} more matches\n"
                output += "\n"
        else:
            output += "No entities found matching your query.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching entities: {str(e)}")]


def handle_entity_stats(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_type = arguments.get("entity_type")

    try:
        result = kb.get_entity_stats(entity_type=entity_type)

        output = "**Entity Statistics**\n"
        if entity_type:
            output += f"**Type Filter:** {entity_type}\n"
        output += "\n"

        output += f"**Total Entities:** {result['total_entities']}\n"
        output += f"**Documents with Entities:** {result['total_documents_with_entities']}\n\n"

        # Breakdown by type
        if result['by_type']:
            output += "**Entities by Type:**\n"
            for ent_type, count in sorted(result['by_type'].items(), key=lambda x: x[1], reverse=True):
                output += f"  - {ent_type.replace('_', ' ')}: {count}\n"
            output += "\n"

        # Top entities
        if result['top_entities']:
            output += "**Top Entities (by document count):**\n"
            for i, entity in enumerate(result['top_entities'][:10], 1):
                output += f"{i}. **{entity['entity_text']}** ({entity['entity_type']})\n"
                output += f"   - Found in {entity['document_count']} document(s)\n"
                output += f"   - Total occurrences: {entity['total_occurrences']}\n"
                output += f"   - Avg confidence: {entity['avg_confidence']:.2f}\n"
            output += "\n"

        # Documents with most entities
        if result['documents_with_most_entities']:
            output += "**Documents with Most Entities:**\n"
            for i, doc in enumerate(result['documents_with_most_entities'], 1):
                output += f"{i}. **{doc['doc_title']}**: {doc['entity_count']} entities\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting entity stats: {str(e)}")]


def handle_get_entity_analytics(kb, name: str, arguments: dict) -> list[TextContent]:
    time_range_days = arguments.get("time_range_days", 30)

    try:
        result = kb.get_entity_analytics(time_range_days=time_range_days)

        output = "**Entity Analytics Dashboard**\n"
        output += f"**Time Range:** Last {time_range_days} days\n\n"

        # Summary Metrics
        if 'summary_metrics' in result:
            metrics = result['summary_metrics']
            output += "**Summary Metrics:**\n"
            output += f"  - Total Entities: {metrics.get('total_entities', 0)}\n"
            output += f"  - Unique Entity Texts: {metrics.get('unique_entity_texts', 0)}\n"
            output += f"  - Total Relationships: {metrics.get('total_relationships', 0)}\n"
            output += f"  - Documents with Entities: {metrics.get('documents_with_entities', 0)}\n"
            output += f"  - Avg Entities per Document: {metrics.get('avg_entities_per_doc', 0):.2f}\n\n"

        # Entity Distribution by Type
        if 'entity_distribution' in result and result['entity_distribution']:
            output += "**Entity Distribution by Type:**\n"
            for entity_type, count in sorted(result['entity_distribution'].items(), key=lambda x: x[1], reverse=True):
                output += f"  - {entity_type.replace('_', ' ').title()}: {count}\n"
            output += "\n"

        # Top Entities
        if 'top_entities' in result and result['top_entities']:
            output += "**Top 10 Entities (by document count):**\n"
            for i, entity in enumerate(result['top_entities'][:10], 1):
                output += f"{i}. **{entity['entity_text']}** ({entity['entity_type']})\n"
                output += f"   - Documents: {entity['doc_count']}\n"
                output += f"   - Avg Confidence: {entity['avg_confidence']:.2f}\n"
            output += "\n"

        # Relationship Statistics
        if 'relationship_stats' in result and result['relationship_stats']:
            stats = result['relationship_stats']
            output += "**Relationship Statistics:**\n"
            output += f"  - Total Relationships: {stats.get('total', 0)}\n"
            output += f"  - Avg Strength: {stats.get('avg_strength', 0):.3f}\n"
            output += f"  - Max Strength: {stats.get('max_strength', 0):.3f}\n"
            if 'by_type' in stats and stats['by_type']:
                output += "  - By Type:\n"
                for rel_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    output += f"    - {rel_type}: {count}\n"
            output += "\n"

        # Top Relationships
        if 'top_relationships' in result and result['top_relationships']:
            output += "**Top 10 Entity Relationships:**\n"
            for i, rel in enumerate(result['top_relationships'][:10], 1):
                output += f"{i}. **{rel['entity1']}** ({rel['entity1_type']}) <-> **{rel['entity2']}** ({rel['entity2_type']})\n"
                output += f"   - Strength: {rel['strength']:.3f}\n"
                output += f"   - Documents: {rel['doc_count']}\n"
            output += "\n"

        # Extraction Timeline
        if 'extraction_timeline' in result and result['extraction_timeline']:
            output += "**Extraction Timeline (Recent Activity):**\n"
            for entry in result['extraction_timeline'][:7]:  # Last 7 days
                output += f"  - {entry['date']}: {entry['count']} entities extracted\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting entity analytics: {str(e)}")]


def handle_extract_entities_bulk(kb, name: str, arguments: dict) -> list[TextContent]:
    confidence_threshold = arguments.get("confidence_threshold", 0.6)
    force_regenerate = arguments.get("force_regenerate", False)
    max_docs = arguments.get("max_docs")
    skip_existing = arguments.get("skip_existing", True)

    try:
        result = kb.extract_entities_bulk(
            confidence_threshold=confidence_threshold,
            force_regenerate=force_regenerate,
            max_docs=max_docs,
            skip_existing=skip_existing
        )

        output = "**Bulk Entity Extraction Complete**\n\n"
        output += f"**Processed:** {result['processed']} documents\n"
        output += f"**Skipped:** {result['skipped']} documents (already have entities)\n"
        output += f"**Failed:** {result['failed']} documents\n"
        output += f"**Total Entities Extracted:** {result['total_entities']}\n\n"

        if result['by_type']:
            output += "**Entities by Type:**\n"
            for ent_type, count in sorted(result['by_type'].items(), key=lambda x: x[1], reverse=True):
                output += f"  - {ent_type.replace('_', ' ')}: {count}\n"
            output += "\n"

        # Show sample results
        if result['results']:
            output += "**Sample Results (first 10):**\n"
            for i, doc_result in enumerate(result['results'][:10], 1):
                status_emoji = "✓" if doc_result['status'] == 'success' else "⊗" if doc_result['status'] == 'failed' else "⊘"
                output += f"{i}. {status_emoji} **{doc_result['title']}**"
                if doc_result['status'] == 'success':
                    output += f" - {doc_result['entity_count']} entities"
                elif doc_result['status'] == 'skipped':
                    output += f" - skipped ({doc_result['entity_count']} entities)"
                elif doc_result['status'] == 'failed':
                    output += f" - ERROR: {doc_result.get('error', 'unknown error')}"
                output += "\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in bulk entity extraction: {str(e)}\n\nNote: Entity extraction requires LLM configuration. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")]


def handle_extract_entity_relationships(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    min_confidence = arguments.get("min_confidence", 0.6)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        result = kb.extract_entity_relationships(doc_id, min_confidence=min_confidence)

        output = "**Entity Relationship Extraction Complete**\n\n"
        output += f"**Document:** {kb.documents[doc_id].title}\n"
        output += f"**Relationships Found:** {result['relationship_count']}\n\n"

        if result['relationships']:
            output += "**Top Relationships (by strength):**\n\n"
            # Show top 10 relationships
            for i, rel in enumerate(result['relationships'][:10], 1):
                output += f"{i}. **{rel['entity1']}** ({rel['entity1_type']}) ↔ **{rel['entity2']}** ({rel['entity2_type']})\n"
                output += f"   Strength: {rel['strength']:.2f}\n"
                if rel.get('context'):
                    context = rel['context'][:100] + "..." if len(rel['context']) > 100 else rel['context']
                    output += f"   *{context}*\n"
                output += "\n"

            if len(result['relationships']) > 10:
                output += f"... and {len(result['relationships']) - 10} more relationships\n\n"

            output += "Use `get_entity_relationships` to explore specific entities.\n"
        else:
            output += "No relationships found. The document may not have enough entities extracted.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting relationships: {str(e)}")]


def handle_get_entity_relationships(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    min_strength = arguments.get("min_strength", 0.0)
    max_results = arguments.get("max_results", 20)

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        relationships = kb.get_entity_relationships(
            entity_text=entity_text,
            min_strength=min_strength,
            max_results=max_results
        )

        if not relationships:
            return [TextContent(type="text", text=f"No relationships found for entity '{entity_text}'.\n\nThis entity may not have been extracted yet, or it doesn't co-occur with other entities.")]

        output = f"**Entities Related to '{entity_text}'**\n\n"
        output += f"Found {len(relationships)} related entities:\n\n"

        for i, rel in enumerate(relationships, 1):
            output += f"{i}. **{rel['related_entity']}** ({rel['related_type']})\n"
            output += f"   Strength: {rel['strength']:.2f} | Found in {rel['doc_count']} document(s)\n"
            if rel.get('context_sample'):
                context = rel['context_sample'][:100] + "..." if len(rel['context_sample']) > 100 else rel['context_sample']
                output += f"   *{context}*\n"
            output += "\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting relationships: {str(e)}")]


def handle_find_related_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    max_results = arguments.get("max_results", 10)

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        related = kb.find_related_entities(entity_text=entity_text, max_results=max_results)

        if not related:
            return [TextContent(type="text", text=f"No related entities found for '{entity_text}'.")]

        output = f"**Entities Related to '{entity_text}'** (Top {len(related)})\n\n"

        for i, rel in enumerate(related, 1):
            output += f"{i}. **{rel['related_entity']}** ({rel['related_type']}) - strength: {rel['strength']:.2f}\n"

        output += "\nUse `get_entity_relationships` for more details and context.\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error finding related entities: {str(e)}")]


def handle_search_entity_pair(kb, name: str, arguments: dict) -> list[TextContent]:
    entity1 = arguments.get("entity1")
    entity2 = arguments.get("entity2")
    max_results = arguments.get("max_results", 10)

    if not entity1 or not entity2:
        return [TextContent(type="text", text="Error: Both entity1 and entity2 are required")]

    try:
        results = kb.search_by_entity_pair(entity1=entity1, entity2=entity2, max_results=max_results)

        if not results:
            return [TextContent(type="text", text=f"No documents found containing both '{entity1}' and '{entity2}'.")]

        output = f"**Documents Containing Both '{entity1}' AND '{entity2}'**\n\n"
        output += f"Found {len(results)} document(s):\n\n"

        for i, doc in enumerate(results, 1):
            output += f"**{i}. {doc['title']}**\n"
            output += f"   '{entity1}': {doc['entity1_count']} mention(s) | '{entity2}': {doc['entity2_count']} mention(s)\n"
            output += f"   Doc ID: `{doc['doc_id']}`\n"

            if doc.get('contexts'):
                output += "   **Context snippets:**\n"
                for j, context in enumerate(doc['contexts'][:2], 1):
                    ctx_short = context[:150] + "..." if len(context) > 150 else context
                    output += f"   {j}. *{ctx_short}*\n"
            output += "\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error searching entity pair: {str(e)}")]


def handle_compare_documents(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id_1 = arguments.get("doc_id_1")
    doc_id_2 = arguments.get("doc_id_2")
    comparison_type = arguments.get("comparison_type", "full")

    if not doc_id_1 or not doc_id_2:
        return [TextContent(type="text", text="Error: Both doc_id_1 and doc_id_2 are required")]

    try:
        result = kb.compare_documents(doc_id_1, doc_id_2, comparison_type)

        # Build formatted output
        output = "**Document Comparison**\n\n"
        output += f"**Similarity Score:** {result['similarity_score']:.1%}\n"
        output += f"**Summary:** {result['summary']}\n\n"

        # Metadata differences
        output += "**Metadata Comparison:**\n"
        md = result['metadata_diff']
        output += "- **Titles:** \n"
        output += f"  - Doc 1: {md['title'][0]}\n"
        output += f"  - Doc 2: {md['title'][1]}\n"
        output += f"- **Files:** {md['filename'][0]} vs {md['filename'][1]}\n"
        output += f"- **Types:** {md['file_type'][0]} vs {md['file_type'][1]}\n"
        output += f"- **Chunks:** {result['chunk_count'][0]} vs {result['chunk_count'][1]}\n\n"

        # Tags comparison
        tags = md['tags']
        if tags['common']:
            output += f"**Common Tags ({len(tags['common'])}):** {', '.join(tags['common'])}\n"
        if tags['only_in_doc1']:
            output += f"**Only in Doc 1 ({len(tags['only_in_doc1'])}):** {', '.join(tags['only_in_doc1'])}\n"
        if tags['only_in_doc2']:
            output += f"**Only in Doc 2 ({len(tags['only_in_doc2'])}):** {', '.join(tags['only_in_doc2'])}\n"
        output += "\n"

        # Entity comparison
        ec = result['entity_comparison']
        if ec['total_doc1'] > 0 or ec['total_doc2'] > 0:
            output += "**Entity Comparison:**\n"
            output += f"- Total in Doc 1: {ec['total_doc1']}\n"
            output += f"- Total in Doc 2: {ec['total_doc2']}\n"
            output += f"- Common Entities: {len(ec['common_entities'])}\n"

            if ec['common_entities'][:5]:  # Show first 5
                output += "\n**Top Common Entities:**\n"
                for ent in ec['common_entities'][:5]:
                    output += f"  - **{ent['text']}** ({ent['type']})\n"

            if ec['unique_to_doc1'][:3]:
                output += "\n**Sample Unique to Doc 1:**\n"
                for ent in ec['unique_to_doc1'][:3]:
                    output += f"  - **{ent['text']}** ({ent['type']})\n"

            if ec['unique_to_doc2'][:3]:
                output += "\n**Sample Unique to Doc 2:**\n"
                for ent in ec['unique_to_doc2'][:3]:
                    output += f"  - **{ent['text']}** ({ent['type']})\n"

        # Content diff preview
        if result['content_diff']:
            output += f"\n**Content Diff Preview:** (First {min(len(result['content_diff']), 20)} lines)\n"
            output += "```diff\n"
            for line in result['content_diff'][:20]:
                output += line + "\n"
            output += "```\n"
            if len(result['content_diff']) > 20:
                output += f"... {len(result['content_diff']) - 20} more lines\n"

        return [TextContent(type="text", text=output)]

    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error comparing documents: {str(e)}")]


def handle_export_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    format = arguments.get("format", "csv")
    entity_types = arguments.get("entity_types")
    min_confidence = arguments.get("min_confidence", 0.0)
    output_path = arguments.get("output_path")

    try:
        result = kb.export_entities(
            format=format,
            entity_types=entity_types,
            min_confidence=min_confidence,
            output_path=output_path
        )

        # Count entities
        if format.lower() == 'csv':
            entity_count = result.count('\n') - 1  # Subtract header
        else:  # json
            import json
            entity_count = len(json.loads(result))

        output = "**Entity Export Complete**\n\n"
        output += f"**Format:** {format.upper()}\n"
        output += f"**Entities Exported:** {entity_count}\n"
        output += f"**Min Confidence:** {min_confidence:.2f}\n"

        if entity_types:
            output += f"**Filtered Types:** {', '.join(entity_types)}\n"

        if output_path:
            output += f"**Saved to:** `{output_path}`\n\n"
        else:
            output += "\n**Preview (first 500 chars):**\n```\n"
            output += result[:500]
            if len(result) > 500:
                output += "\n... (truncated)"
            output += "\n```\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error exporting entities: {str(e)}")]


def handle_export_relationships(kb, name: str, arguments: dict) -> list[TextContent]:
    format = arguments.get("format", "csv")
    min_strength = arguments.get("min_strength", 0.0)
    entity_types = arguments.get("entity_types")
    output_path = arguments.get("output_path")

    try:
        result = kb.export_relationships(
            format=format,
            min_strength=min_strength,
            entity_types=entity_types,
            output_path=output_path
        )

        # Count relationships
        if format.lower() == 'csv':
            rel_count = result.count('\n') - 1  # Subtract header
        else:  # json
            import json
            rel_count = len(json.loads(result))

        output = "**Relationship Export Complete**\n\n"
        output += f"**Format:** {format.upper()}\n"
        output += f"**Relationships Exported:** {rel_count}\n"
        output += f"**Min Strength:** {min_strength:.2f}\n"

        if entity_types:
            output += f"**Filtered Types:** {', '.join(entity_types)}\n"

        if output_path:
            output += f"**Saved to:** `{output_path}`\n\n"
        else:
            output += "\n**Preview (first 500 chars):**\n```\n"
            output += result[:500]
            if len(result) > 500:
                output += "\n... (truncated)"
            output += "\n```\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error exporting relationships: {str(e)}")]


def handle_queue_entity_extraction(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    confidence_threshold = arguments.get("confidence_threshold", 0.6)
    skip_if_exists = arguments.get("skip_if_exists", True)

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        result = kb.queue_entity_extraction(
            doc_id=doc_id,
            confidence_threshold=confidence_threshold,
            skip_if_exists=skip_if_exists
        )

        if result.get('queued'):
            output = "**Entity Extraction Queued**\n\n"
            output += f"**Job ID:** {result['job_id']}\n"
            output += f"**Document ID:** {doc_id}\n"
            output += f"**Confidence Threshold:** {confidence_threshold:.1%}\n\n"
            output += "✅ Extraction job has been queued and will run in the background.\n"
            output += "Use `get_extraction_status` to check progress.\n"
        else:
            output = "**Entity Extraction Not Queued**\n\n"
            output += f"**Reason:** {result.get('reason', 'Unknown')}\n"
            if 'existing_job_id' in result:
                output += f"**Existing Job ID:** {result['existing_job_id']}\n"
            if 'existing_entities' in result:
                output += f"**Existing Entities:** {result['existing_entities']}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error queuing extraction: {str(e)}")]


def handle_get_extraction_status(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")

    if not doc_id:
        return [TextContent(type="text", text="Error: doc_id is required")]

    try:
        status = kb.get_extraction_status(doc_id)

        output = f"**Entity Extraction Status for Document: {doc_id}**\n\n"
        output += f"**Has Entities:** {'✅ Yes' if status['has_entities'] else '❌ No'}\n"
        output += f"**Entity Count:** {status['entity_count']}\n\n"

        if status['jobs']:
            output += "**Extraction Jobs:**\n\n"
            for job in status['jobs']:
                output += f"- **Job {job['job_id']}**\n"
                output += f"  - Status: {job['status'].upper()}\n"
                output += f"  - Confidence: {job['confidence_threshold']:.1%}\n"
                output += f"  - Queued: {job['queued_at']}\n"
                if job['started_at']:
                    output += f"  - Started: {job['started_at']}\n"
                if job['completed_at']:
                    output += f"  - Completed: {job['completed_at']}\n"
                if job['entities_extracted']:
                    output += f"  - Entities Extracted: {job['entities_extracted']}\n"
                if job['error_message']:
                    output += f"  - Error: {job['error_message']}\n"
                output += "\n"
        else:
            output += "**No extraction jobs found for this document.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting extraction status: {str(e)}")]


def handle_get_extraction_jobs(kb, name: str, arguments: dict) -> list[TextContent]:
    status_filter = arguments.get("status_filter")
    limit = arguments.get("limit", 100)

    try:
        jobs = kb.get_all_extraction_jobs(
            status_filter=status_filter,
            limit=limit
        )

        output = "**Entity Extraction Jobs**\n\n"
        if status_filter:
            output += f"**Filter:** {status_filter.upper()}\n"
        output += f"**Total Jobs:** {len(jobs)}\n\n"

        if jobs:
            # Group by status
            by_status = {}
            for job in jobs:
                status = job['status']
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(job)

            # Show summary
            output += "**Summary:**\n"
            for status, job_list in sorted(by_status.items()):
                output += f"- {status.upper()}: {len(job_list)}\n"
            output += "\n"

            # Show recent jobs (limit to 10 for readability)
            output += f"**Recent Jobs (showing {min(len(jobs), 10)} of {len(jobs)}):**\n\n"
            for job in jobs[:10]:
                output += f"- **Job {job['job_id']}** ({job['status'].upper()})\n"
                output += f"  - Document: {job['doc_title'][:50]}...\n"
                output += f"  - Doc ID: {job['doc_id']}\n"
                output += f"  - Queued: {job['queued_at']}\n"
                if job['entities_extracted']:
                    output += f"  - Entities: {job['entities_extracted']}\n"
                if job['error_message']:
                    output += f"  - Error: {job['error_message'][:100]}...\n"
                output += "\n"
        else:
            output += "**No jobs found.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting extraction jobs: {str(e)}")]


# ============================================================
# Figure OCR Tool Handlers
# ============================================================


HANDLERS_ENTITIES = {
    "extract_entities": handle_extract_entities,
    "list_entities": handle_list_entities,
    "search_entities": handle_search_entities,
    "entity_stats": handle_entity_stats,
    "get_entity_analytics": handle_get_entity_analytics,
    "extract_entities_bulk": handle_extract_entities_bulk,
    "extract_entity_relationships": handle_extract_entity_relationships,
    "get_entity_relationships": handle_get_entity_relationships,
    "find_related_entities": handle_find_related_entities,
    "search_entity_pair": handle_search_entity_pair,
    "compare_documents": handle_compare_documents,
    "export_entities": handle_export_entities,
    "export_relationships": handle_export_relationships,
    "queue_entity_extraction": handle_queue_entity_extraction,
    "get_extraction_status": handle_get_extraction_status,
    "get_extraction_jobs": handle_get_extraction_jobs,
}
