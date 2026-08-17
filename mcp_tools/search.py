"""Search-tool handlers: full-text/semantic/hybrid/fuzzy search, tag
facets, and query refinement. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


def handle_search_docs(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")
    include_superseded = arguments.get("include_superseded", False)

    results = kb.search(query, max_results, tags, include_superseded)

    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    output = f"Found {len(results)} results for '{query}':\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Score: {r['score']}\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_translate_query(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    confidence_threshold = arguments.get("confidence_threshold", 0.7)

    try:
        result = kb.translate_nl_query(query, confidence_threshold)
    except Exception as e:
        return [TextContent(type="text", text=f"Query translation error: {str(e)}")]

    # Format the translation result
    output = "Natural Language Query Translation\n"
    output += f"{'='*60}\n\n"
    output += f"Original Query: \"{result['original_query']}\"\n\n"

    output += f"Search Mode: {result['search_mode']} (confidence: {result['confidence']:.2f})\n"
    output += f"Reasoning: {result['reasoning']}\n\n"

    if result['search_terms']:
        output += "Search Terms:\n"
        for term in result['search_terms']:
            output += f"  - {term}\n"
        output += "\n"

    if result['facet_filters']:
        output += "Facet Filters:\n"
        for facet_type, values in result['facet_filters'].items():
            output += f"  {facet_type}: {', '.join(values)}\n"
        output += "\n"

    if result['entities_found']:
        output += f"Entities Detected ({len(result['entities_found'])}):\n"
        for entity in result['entities_found']:
            output += f"  - {entity['text']} ({entity['type']}, confidence: {entity['confidence']:.2f})\n"
        output += "\n"

    # Add suggested next steps
    output += "Suggested Action:\n"
    if result['search_mode'] == 'keyword':
        output += f"  Use search_docs with terms: {', '.join(result['search_terms'][:3])}\n"
    elif result['search_mode'] == 'semantic':
        output += f"  Use semantic_search with query: \"{result['original_query']}\"\n"
    else:  # hybrid
        output += "  Use hybrid_search for best results\n"

    if result['facet_filters']:
        output += f"  Apply facet filters: {result['facet_filters']}\n"

    return [TextContent(type="text", text=output)]


def handle_semantic_search(kb, name: str, arguments: dict) -> list[TextContent]:
    if not kb.use_semantic:
        return [TextContent(
            type="text",
            text="Semantic search is not enabled. Set USE_SEMANTIC_SEARCH=1 and install sentence-transformers and faiss-cpu."
        )]

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")

    try:
        results = kb.semantic_search(query, max_results, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Semantic search error: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    output = f"Found {len(results)} semantic results for '{query}':\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Similarity Score: {r['similarity']:.4f}\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_hybrid_search(kb, name: str, arguments: dict) -> list[TextContent]:
    if not kb.use_semantic:
        return [TextContent(
            type="text",
            text="Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1 and install sentence-transformers and faiss-cpu."
        )]

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")
    semantic_weight = arguments.get("semantic_weight", 0.7)

    try:
        results = kb.hybrid_search(query, max_results, tags, semantic_weight)
    except Exception as e:
        return [TextContent(type="text", text=f"Hybrid search error: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    output = f"Found {len(results)} hybrid results for '{query}' (semantic_weight={semantic_weight}):\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Hybrid Score: {r['score']:.4f} (FTS: {r['fts_score']:.4f}, Semantic: {r['semantic_score']:.4f})\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_fuzzy_search(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")
    similarity_threshold = arguments.get("similarity_threshold", 80)

    try:
        results = kb.fuzzy_search(query, max_results, tags, similarity_threshold)
    except Exception as e:
        return [TextContent(type="text", text=f"Fuzzy search error: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    output = f"Found {len(results)} fuzzy search results for '{query}':\n\n"

    # Show corrections if any
    if results and 'fuzzy_corrections' in results[0]:
        corrections = results[0]['fuzzy_corrections']
        output += "Corrections applied:\n"
        for corr in corrections:
            output += f"  '{corr['original']}' → '{corr['corrected']}' (similarity: {corr['similarity']}%)\n"
        output += f"Corrected query: '{results[0]['corrected_query']}'\n\n"

    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Score: {r['score']}\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_search_within_results(kb, name: str, arguments: dict) -> list[TextContent]:
    previous_results = arguments.get("previous_results", [])
    refinement_query = arguments.get("refinement_query", "")
    max_results = arguments.get("max_results", 5)

    try:
        results = kb.search_within_results(previous_results, refinement_query, max_results)
    except Exception as e:
        return [TextContent(type="text", text=f"Search within results error: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No matching results in previous set for refinement query: {refinement_query}")]

    output = f"Refined search: found {len(results)} results for '{refinement_query}' within previous results:\n\n"

    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Original Score: {r.get('score', 'N/A')}\n"
        output += f"Refinement Score: {r.get('refinement_score', 'N/A')}\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_suggest_tags(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id", "")
    confidence_threshold = arguments.get("confidence_threshold", 0.6)

    try:
        suggestions = kb.suggest_tags(doc_id, confidence_threshold)
    except Exception as e:
        return [TextContent(type="text", text=f"Tag suggestion error: {str(e)}")]

    if not suggestions:
        return [TextContent(type="text", text=f"No tag suggestions found for document {doc_id}. Content may not match known categories.")]

    # Organize by category
    by_category = {}
    for suggestion in suggestions:
        category = suggestion['category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(suggestion)

    output = f"Tag suggestions for document {doc_id}:\n\n"

    for category in sorted(by_category.keys()):
        tags_in_cat = by_category[category]
        output += f"**{category.replace('-', ' ').title()}:**\n"
        for tag in tags_in_cat:
            output += f"  - {tag['tag']} (confidence: {tag['confidence']:.0%})\n"
        output += "\n"

    output += "\n**To apply these tags:**\n"
    output += "Use the 'add_document' tool to update the document with tags.\n"
    output += "Or use update_document_tags tool to modify tags directly.\n"

    return [TextContent(type="text", text=output)]


def handle_get_tags_by_category(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        tag_categories = kb.get_tags_by_category()
    except Exception as e:
        return [TextContent(type="text", text=f"Tag browsing error: {str(e)}")]

    output = "Knowledge Base Tags by Category\n"
    output += "=" * 60 + "\n\n"

    total_tags = 0

    for category in sorted(tag_categories.keys()):
        tags = tag_categories[category]
        if not tags:
            continue

        output += f"**{category.replace('-', ' ').title()}** ({len(tags)} tags):\n"

        for tag_info in tags:
            tag = tag_info['tag']
            count = tag_info['count']
            output += f"  - {tag}: {count} document{'s' if count != 1 else ''}\n"
            total_tags += 1

        output += "\n"

    output += f"\nTotal: {total_tags} tags across {len(tag_categories)} categories\n"

    return [TextContent(type="text", text=output)]


def handle_faceted_search(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    facet_filters = arguments.get("facet_filters")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")

    try:
        results = kb.faceted_search(query, facet_filters, max_results, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Faceted search error: {str(e)}")]

    if not results:
        if facet_filters:
            return [TextContent(type="text", text=f"No results found for '{query}' with facet filters: {facet_filters}")]
        else:
            return [TextContent(type="text", text=f"No results found for: {query}")]

    # Format output
    filter_desc = f" with facets: {facet_filters}" if facet_filters else ""
    output = f"Found {len(results)} results for '{query}'{filter_desc}:\n\n"

    for i, r in enumerate(results, 1):
        output += f"--- Result {i} ---\n"
        output += f"Document: {r['title']} ({r['filename']})\n"
        output += f"Doc ID: {r['doc_id']}, Chunk: {r['chunk_id']}\n"
        output += f"Score: {r['score']:.4f}\n"

        # Show document facets
        if 'facets' in r:
            facets = r['facets']
            facet_parts = []
            if facets.get('hardware'):
                facet_parts.append(f"Hardware: {', '.join(sorted(facets['hardware']))}")
            if facets.get('instruction'):
                facet_parts.append(f"Instructions: {', '.join(sorted(facets['instruction']))}")
            if facets.get('register'):
                # Show only first 5 registers if many
                regs = sorted(facets['register'])
                if len(regs) > 5:
                    facet_parts.append(f"Registers: {', '.join(regs[:5])} (+{len(regs)-5} more)")
                else:
                    facet_parts.append(f"Registers: {', '.join(regs)}")
            if facet_parts:
                output += f"Facets: {' | '.join(facet_parts)}\n"

        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


HANDLERS_SEARCH = {
    "search_docs": handle_search_docs,
    "translate_query": handle_translate_query,
    "semantic_search": handle_semantic_search,
    "hybrid_search": handle_hybrid_search,
    "fuzzy_search": handle_fuzzy_search,
    "search_within_results": handle_search_within_results,
    "suggest_tags": handle_suggest_tags,
    "get_tags_by_category": handle_get_tags_by_category,
    "faceted_search": handle_faceted_search,
}
