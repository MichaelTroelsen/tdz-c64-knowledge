"""One function per MCP tool, replacing server.py's 3,400-line
`elif name == ...` chain (R12).

Handlers take the KnowledgeBase explicitly rather than reaching for a
module global: server.py rebinds its `kb` singleton in get_kb(), so an
imported copy would go stale the moment that happened.

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""
import os

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


def handle_get_chunk(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    chunk_id = arguments.get("chunk_id")

    chunk = kb.get_chunk(doc_id, chunk_id)
    if not chunk:
        return [TextContent(type="text", text=f"Chunk not found: {doc_id}/{chunk_id}")]

    output = f"Document: {chunk.title}\n"
    output += f"Chunk {chunk.chunk_id} ({chunk.word_count} words):\n\n"
    output += chunk.content

    return [TextContent(type="text", text=output)]


def handle_get_document(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")

    content = kb.get_document_content(doc_id)
    if not content:
        return [TextContent(type="text", text=f"Document not found: {doc_id}")]

    doc = kb.documents.get(doc_id)
    output = f"Document: {doc.title}\n"
    output += f"File: {doc.filename}\n"
    output += f"{'='*50}\n\n"
    output += content

    return [TextContent(type="text", text=output)]


def handle_list_docs(kb, name: str, arguments: dict) -> list[TextContent]:
    include_superseded = arguments.get("include_superseded", False)
    docs = kb.list_documents(include_superseded)

    if not docs:
        return [TextContent(type="text", text="No documents in knowledge base. Use add_document to add PDFs or text files.")]

    output = f"Documents in knowledge base ({len(docs)}):\n\n"
    for doc in docs:
        output += f"- {doc.title}\n"
        output += f"  ID: {doc.doc_id}\n"
        output += f"  File: {doc.filename} ({doc.file_type})\n"
        if doc.card_id:
            output += f"  Card ID: {doc.card_id}\n"
        if doc.superseded_by:
            output += f"  Superseded by: {doc.superseded_by}\n"
        if doc.total_pages:
            output += f"  Pages: {doc.total_pages}\n"
        output += f"  Chunks: {doc.total_chunks}\n"
        if doc.tags:
            output += f"  Tags: {', '.join(doc.tags)}\n"
        output += f"  Indexed: {doc.indexed_at}\n\n"

    return [TextContent(type="text", text=output)]


def handle_get_document_by_card_id(kb, name: str, arguments: dict) -> list[TextContent]:
    card_id = arguments.get("card_id")
    include_superseded = arguments.get("include_superseded", False)

    doc = kb.get_document_by_card_id(card_id, include_superseded)
    if not doc:
        return [TextContent(type="text", text=f"No card found with id: {card_id}")]

    output = f"Card: {card_id}\n"
    output += f"  Title: {doc.title}\n"
    output += f"  Doc ID: {doc.doc_id}\n"
    if doc.superseded_by:
        output += f"  Status: SUPERSEDED by {doc.superseded_by}\n"
    else:
        output += f"  Status: live\n"
    output += f"  Chunks: {doc.total_chunks}\n"
    output += f"  Indexed: {doc.indexed_at}\n"
    return [TextContent(type="text", text=output)]


def handle_add_document(kb, name: str, arguments: dict) -> list[TextContent]:
    filepath = arguments.get("filepath")
    title = arguments.get("title")
    tags = arguments.get("tags", [])
    replace = arguments.get("replace", False)

    try:
        doc = kb.add_document(filepath, title, tags, replace=replace)
        output = "Successfully added document:\n"
        output += f"  Title: {doc.title}\n"
        output += f"  ID: {doc.doc_id}\n"
        output += f"  Type: {doc.file_type}\n"
        if doc.card_id:
            output += f"  Card ID: {doc.card_id}\n"
        output += f"  Chunks: {doc.total_chunks}\n"
        if doc.total_pages:
            output += f"  Pages: {doc.total_pages}\n"
        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error adding document: {str(e)}")]


def handle_update_document(kb, name: str, arguments: dict) -> list[TextContent]:
    card_id_or_doc_id = arguments.get("card_id_or_doc_id")
    filepath = arguments.get("filepath")
    title = arguments.get("title")
    tags = arguments.get("tags")

    try:
        doc = kb.update_document(card_id_or_doc_id, filepath, title, tags)
        output = "Successfully updated document:\n"
        output += f"  Title: {doc.title}\n"
        output += f"  ID: {doc.doc_id}\n"
        if doc.card_id:
            output += f"  Card ID: {doc.card_id}\n"
        output += f"  Chunks: {doc.total_chunks}\n"
        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error updating document: {str(e)}")]


def handle_scrape_url(kb, name: str, arguments: dict) -> list[TextContent]:
    url = arguments.get("url")
    title = arguments.get("title")
    tags = arguments.get("tags", [])
    follow_links = arguments.get("follow_links", True)
    same_domain_only = arguments.get("same_domain_only", True)
    max_pages = arguments.get("max_pages", 50)
    depth = arguments.get("depth", 3)
    limit = arguments.get("limit")
    threads = arguments.get("threads", 3)
    delay = arguments.get("delay", 500)
    selector = arguments.get("selector")

    try:
        result = kb.scrape_url(
            url=url,
            title=title,
            tags=tags,
            follow_links=follow_links,
            same_domain_only=same_domain_only,
            max_pages=max_pages,
            depth=depth,
            limit=limit,
            threads=threads,
            delay=delay,
            selector=selector
        )

        output = "Scraping Result:\n\n"
        output += f"Status: {result['status']}\n"
        output += f"URL: {result['url']}\n"
        output += f"Files scraped: {result['files_scraped']}\n"
        output += f"Documents added: {result['docs_added']}\n"

        if result['docs_failed'] > 0:
            output += f"Documents failed: {result['docs_failed']}\n"

        if result.get('error'):
            output += f"\nError: {result['error']}\n"

        if result.get('doc_ids'):
            output += "\nAdded document IDs:\n"
            for doc_id in result['doc_ids'][:10]:  # Show first 10
                doc = kb.documents.get(doc_id)
                if doc:
                    output += f"  - {doc.title} ({doc_id})\n"
            if len(result['doc_ids']) > 10:
                output += f"  ... and {len(result['doc_ids']) - 10} more\n"

        if result['status'] == 'success':
            output += f"\nOutput directory: {result['output_dir']}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error scraping URL: {str(e)}")]


def handle_rescrape_document(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")

    try:
        result = kb.rescrape_document(doc_id)

        output = "Re-scrape Result:\n\n"
        output += f"Status: {result['status']}\n"
        output += f"Original doc ID: {result['old_doc_id']}\n"
        output += f"Documents added: {result['docs_added']}\n"

        if result.get('error'):
            output += f"Error: {result['error']}\n"

        if result.get('doc_ids'):
            output += "\nNew document IDs:\n"
            for doc_id in result['doc_ids'][:5]:
                doc = kb.documents.get(doc_id)
                if doc:
                    output += f"  - {doc.title} ({doc_id})\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error re-scraping: {str(e)}")]


def handle_check_url_updates(kb, name: str, arguments: dict) -> list[TextContent]:
    auto_rescrape = arguments.get("auto_rescrape", False)

    try:
        results = kb.check_url_updates(auto_rescrape)

        output = "URL Update Check:\n\n"

        if results['unchanged']:
            output += f"✓ {len(results['unchanged'])} documents unchanged\n"

        if results['changed']:
            output += f"⚠ {len(results['changed'])} documents have updates:\n"
            for doc in results['changed'][:10]:  # Show first 10
                output += f"  - {doc['title']}\n"
                output += f"    URL: {doc['url']}\n"
                output += f"    Last modified: {doc['last_modified']}\n"
            if len(results['changed']) > 10:
                output += f"  ... and {len(results['changed']) - 10} more\n"
            output += "\n"

        if results['failed']:
            output += f"✗ {len(results['failed'])} checks failed:\n"
            for doc in results['failed'][:5]:
                output += f"  - {doc['title']}: {doc['error']}\n"
            output += "\n"

        if auto_rescrape and results['rescraped']:
            output += f"✓ {len(results['rescraped'])} documents re-scraped\n"
        elif not auto_rescrape and results['changed']:
            output += "\nTip: Use auto_rescrape=true to automatically update changed documents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error checking updates: {str(e)}")]


def handle_remove_document(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")

    if kb.remove_document(doc_id):
        return [TextContent(type="text", text=f"Successfully removed document: {doc_id}")]
    else:
        return [TextContent(type="text", text=f"Document not found: {doc_id}")]


def handle_find_similar(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    chunk_id = arguments.get("chunk_id")
    max_results = arguments.get("max_results", 5)
    tags = arguments.get("tags")

    # Check if document exists
    if doc_id not in kb.documents:
        return [TextContent(type="text", text=f"Document not found: {doc_id}")]

    try:
        results = kb.find_similar_documents(doc_id, chunk_id, max_results, tags)
    except Exception as e:
        return [TextContent(type="text", text=f"Error finding similar documents: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=f"No similar documents found for: {doc_id}")]

    source_doc = kb.documents[doc_id]
    if chunk_id is not None:
        output = f"Documents similar to '{source_doc.title}' (chunk {chunk_id}):\n\n"
    else:
        output = f"Documents similar to '{source_doc.title}':\n\n"

    for i, r in enumerate(results, 1):
        output += f"--- {i}. {r['title']} ({r['filename']}) ---\n"
        output += f"Doc ID: {r['doc_id']}, Similarity: {r['similarity']:.4f}\n"
        output += f"Tags: {', '.join(r['tags']) if r['tags'] else 'none'}\n"
        output += f"Snippet:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_answer_question(kb, name: str, arguments: dict) -> list[TextContent]:
    question = arguments.get("question", "")
    max_sources = arguments.get("max_sources", 5)
    search_mode = arguments.get("search_mode", "auto")

    # Validate input
    if not question or len(question.strip()) < 3:
        return [TextContent(type="text", text="Error: Question must be at least 3 characters long")]

    # Call RAG implementation
    try:
        result = kb.answer_question(
            question,
            max_context_chunks=max_sources,
            force_search_mode=None if search_mode == "auto" else search_mode
        )
    except Exception as e:
        return [TextContent(type="text", text=f"Error answering question: {str(e)}")]

    # Format response
    output = f"# Answer\n\n{result['answer']}\n\n"

    # Add sources section if available
    if result['sources']:
        output += "## Sources\n\n"
        for i, source in enumerate(result['sources'], 1):
            doc = kb.documents.get(source['doc_id'])
            if doc:
                output += f"**{i}. {doc.title}** ({doc.filename})\n"
                output += f"   - Chunk ID: {source['chunk_id']}\n"
                output += f"   - Relevance: {source['score']:.2%}\n"
                if 'page' in source and source['page']:
                    output += f"   - Page: {source['page']}\n"
                output += "\n"

    # Add metadata section
    output += "## Answer Metadata\n\n"
    output += f"- **Confidence**: {result['confidence']:.1%}\n"
    output += f"- **Search Mode**: {result.get('reasoning', 'Unknown')}\n"
    output += f"- **LLM Model**: {result.get('model', 'Fallback')}\n"
    output += f"- **Sources Used**: {len(result['sources'])} documents\n"

    # Flag claims the grounding check couldn't confirm against their
    # cited source - the whole point of running the check is to surface
    # this, not just fold it into a number nobody can act on.
    unverified = result.get('unverified_claims')
    if unverified:
        output += f"\n## ⚠️ Unverified Claims ({len(unverified)})\n\n"
        output += "These statements cited a source but the citation didn't check out against it:\n\n"
        for claim in unverified:
            output += f"- {claim}\n"

    # Add error note if applicable
    if result.get('error'):
        output += f"\n⚠️ **Note**: {result['error']}\n"

    return [TextContent(type="text", text=output)]


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
def handle_batch_ocr_figures(kb, name: str, arguments: dict) -> list[TextContent]:
    limit = arguments.get("limit")
    reprocess = arguments.get("reprocess", False)

    try:
        result = kb.queue_figure_ocr_all(limit=limit, skip_if_exists=not reprocess)
    except Exception as e:
        return [TextContent(type="text", text=f"Error queueing figure OCR: {str(e)}")]

    output = "**Batch Figure OCR Queued**\n\n"
    output += f"- PDF documents in knowledge base: {result['pdf_documents']}\n"
    output += f"- Queued for OCR: {result['queued']}\n"
    output += f"- Skipped: {result['skipped']}\n\n"

    if result['queued']:
        output += (
            "Work runs in the background - this call does not wait for it. "
            "Use `figure_ocr_status` to watch progress and `search_figures` "
            "to query the results.\n"
        )
    if result['skipped_details']:
        output += f"\n**Skipped (first {len(result['skipped_details'])}):**\n"
        for item in result['skipped_details']:
            output += f"- {item['doc_id']}: {item['reason']}\n"

    return [TextContent(type="text", text=output)]


def handle_ocr_document_figures(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    reprocess = arguments.get("reprocess", False)

    try:
        result = kb.queue_figure_ocr(doc_id, skip_if_exists=not reprocess)
    except Exception as e:
        return [TextContent(type="text", text=f"Error queueing figure OCR: {str(e)}")]

    if result.get('queued'):
        return [TextContent(type="text", text=(
            f"Queued figure OCR for {doc_id} (job {result['job_id']}).\n"
            "Runs in the background - check `figure_ocr_status` for progress."
        ))]
    return [TextContent(type="text", text=(
        f"Not queued for {doc_id}: {result.get('reason', 'unknown reason')}"
    ))]


def handle_figure_ocr_status(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        stats = kb.get_figure_ocr_coverage()
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting figure OCR status: {str(e)}")]

    output = "**Figure OCR Status**\n\n"
    if not stats['available']:
        output += f"[UNAVAILABLE] {stats['unavailable_reason']}\n\n"

    output += f"- PDF documents: {stats['pdf_documents']}\n"
    output += f"- Documents processed: {stats['documents_processed']}\n"
    output += f"- Documents remaining: {stats['documents_remaining']}\n"
    output += f"  - Reachable (file on disk): {stats['documents_remaining_reachable']}\n"
    output += f"  - Unreachable (source file missing): {stats['documents_remaining_unreachable']}\n"
    output += f"- Figures extracted: {stats['figures_extracted']}\n"
    output += f"- Figures containing text: {stats['figures_with_text']}\n"
    output += f"- Jobs pending: {stats['jobs_pending']}\n"

    if stats['pdf_documents']:
        pct = 100.0 * stats['documents_processed'] / stats['pdf_documents']
        output += f"\nCoverage: {pct:.1f}% of PDFs\n"

    return [TextContent(type="text", text=output)]


def handle_search_figures(kb, name: str, arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 10)
    doc_id = arguments.get("doc_id")

    try:
        results = kb.search_figures(query, max_results, doc_id)
    except Exception as e:
        return [TextContent(type="text", text=f"Figure search error: {str(e)}")]

    if not results:
        return [TextContent(type="text", text=(
            f"No figure text found for: {query}\n\n"
            "If no figures have been OCR'd yet, run `batch_ocr_figures` first "
            "(check with `figure_ocr_status`)."
        ))]

    output = f"Found {len(results)} figure match(es) for '{query}':\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- Figure {i} ---\n"
        output += f"Document: {r['doc_title'] or r['doc_id']}\n"
        output += f"Doc ID: {r['doc_id']}, Page: {r['page_number']}, Figure: {r['image_index']}\n"
        output += f"Image: {r['image_path']}\n"
        output += f"Text:\n{r['snippet']}\n\n"

    return [TextContent(type="text", text=output)]


def handle_get_document_figures(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    with_text_only = arguments.get("with_text_only", False)

    try:
        figures = kb.get_document_figures(doc_id, with_text_only)
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting figures: {str(e)}")]

    if not figures:
        return [TextContent(type="text", text=(
            f"No extracted figures for {doc_id}. "
            "Run `ocr_document_figures` to extract them."
        ))]

    with_text = sum(1 for f in figures if f['ocr_text'])
    output = f"**Figures in {doc_id}**\n\n"
    output += f"- Total: {len(figures)}\n- With OCR text: {with_text}\n\n"

    for f in figures:
        output += f"**Page {f['page_number']}, figure {f['image_index']}** "
        output += f"({f['width']}x{f['height']}px, {f.get('source') or 'embedded'})\n"
        output += f"  Image: {f['image_path']}\n"
        if f['ocr_text']:
            preview = f['ocr_text'][:300].replace('\n', ' ')
            output += f"  Text ({f['char_count']} chars): {preview}"
            output += "...\n" if f['char_count'] > 300 else "\n"
        else:
            output += "  Text: (none detected)\n"
        output += "\n"

    return [TextContent(type="text", text=output)]


# ============================================================
# Knowledge Graph Tool Handlers (v2.24.0 - Phase 1, Task 1.4)
# ============================================================
def handle_build_knowledge_graph(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    use_cache = arguments.get("use_cache", True)

    try:
        import networkx as nx

        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            use_cache=use_cache
        )

        # Build comprehensive output
        output = "# Knowledge Graph Built\n\n"
        output += f"**Graph Statistics:**\n"
        output += f"- Nodes (entities): {G.number_of_nodes()}\n"
        output += f"- Edges (relationships): {G.number_of_edges()}\n"

        if G.number_of_nodes() > 0:
            density = nx.density(G)
            output += f"- Density: {density:.4f}\n"
            output += f"- Connected components: {nx.number_connected_components(G)}\n"

            # Show sample nodes
            sample_nodes = list(G.nodes())[:10]
            output += f"\n**Sample Entities (first 10):**\n"
            for node in sample_nodes:
                node_data = G.nodes[node]
                output += f"- {node} (type: {node_data.get('type', 'unknown')}, "
                output += f"occurrences: {node_data.get('occurrences', 0)})\n"

            # Show sample edges
            if G.number_of_edges() > 0:
                sample_edges = list(G.edges(data=True))[:10]
                output += f"\n**Sample Relationships (first 10):**\n"
                for e1, e2, data in sample_edges:
                    output += f"- {e1} ↔ {e2} (strength: {data.get('weight', 0):.3f})\n"
        else:
            output += "\n**No entities found matching the criteria.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error building knowledge graph: {str(e)}")]


def handle_compute_graph_metrics(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    store_results = arguments.get("store_results", True)

    try:
        metrics = kb.compute_graph_metrics(
            G=None,  # Will build new graph
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            store_results=store_results
        )

        # Build output
        output = "# Graph Metrics Computed\n\n"

        # Graph statistics
        output += f"**Graph Statistics:**\n"
        stats = metrics['graph_stats']
        output += f"- Nodes: {stats['nodes']}\n"
        output += f"- Edges: {stats['edges']}\n"
        output += f"- Density: {stats['density']:.4f}\n"
        output += f"- Connected components: {stats['connected_components']}\n"
        output += f"- Communities detected: {metrics['num_communities']}\n"
        output += f"- Computed: {stats['computed_at']}\n"

        # Top entities by PageRank
        if metrics['pagerank']:
            output += f"\n**Top 15 Entities by PageRank:**\n"
            top_pr = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:15]
            for rank, (entity, score) in enumerate(top_pr, 1):
                community = metrics['communities'].get(entity, 'N/A')
                output += f"{rank:2d}. {entity:30s} (PR: {score:.6f}, Community: {community})\n"

        # Top entities by Betweenness
        if metrics['betweenness']:
            output += f"\n**Top 10 Bridge Entities (Betweenness):**\n"
            top_bt = sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (entity, score) in enumerate(top_bt, 1):
                output += f"{rank:2d}. {entity:30s} (Betweenness: {score:.6f})\n"

        if store_results:
            output += f"\n**Metrics stored to database for {len(metrics['pagerank'])} entities.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error computing graph metrics: {str(e)}")]


def handle_get_entity_metrics(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    metric_types = arguments.get("metric_types")

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        result = kb.get_entity_metrics(entity_text, metric_types)

        if not result['found']:
            return [TextContent(type="text", text=f"No metrics found for entity: '{entity_text}'\n\nThis entity may not be in the knowledge graph, or metrics have not been computed yet. Run 'compute_graph_metrics' first.")]

        # Build output
        output = f"# Entity Metrics: {entity_text}\n\n"
        output += f"**Entity Type:** {result['entity_type']}\n"
        output += f"**Computed:** {result['computed_date']}\n\n"

        output += f"**Metrics:**\n"
        metrics = result['metrics']

        if 'pagerank' in metrics and metrics['pagerank'] is not None:
            output += f"- PageRank: {metrics['pagerank']:.6f} (importance score)\n"

        if 'betweenness' in metrics and metrics['betweenness'] is not None:
            output += f"- Betweenness: {metrics['betweenness']:.6f} (bridge score)\n"

        if 'degree' in metrics and metrics['degree'] is not None:
            output += f"- Degree: {metrics['degree']:.6f} (connection score)\n"

        if 'community' in metrics and metrics['community'] is not None:
            output += f"- Community ID: {metrics['community']}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error retrieving entity metrics: {str(e)}")]


def handle_find_entity_path(kb, name: str, arguments: dict) -> list[TextContent]:
    entity1 = arguments.get("entity1")
    entity2 = arguments.get("entity2")
    max_path_length = arguments.get("max_path_length", 6)
    store_result = arguments.get("store_result", True)

    if not entity1 or not entity2:
        return [TextContent(type="text", text="Error: both entity1 and entity2 are required")]

    try:
        result = kb.find_shortest_path(
            entity1=entity1,
            entity2=entity2,
            G=None,  # Will build graph
            max_path_length=max_path_length,
            store_result=store_result
        )

        if result is None:
            return [TextContent(type="text", text=f"Error: Could not compute path. One or both entities may not exist in the graph.")]

        # Build output
        output = f"# Path: {entity1} → {entity2}\n\n"

        if result['exists']:
            output += f"**Path Found:** {result['length']} edges\n\n"
            output += f"**Path:**\n"
            output += " → ".join(result['path']) + "\n\n"

            if result['relationships']:
                output += f"**Relationship Details:**\n"
                for i, rel in enumerate(result['relationships'], 1):
                    output += f"{i}. {rel['from']} → {rel['to']}\n"
                    output += f"   - Strength: {rel['weight']:.3f}\n"
                    output += f"   - Co-occurrences: {rel['doc_count']} documents\n"

            if store_result:
                output += f"\n**Path stored to database.**\n"
        else:
            output += f"**No path found.**\n\n"
            output += f"The entities '{entity1}' and '{entity2}' are in different connected components "
            output += f"of the knowledge graph (no relationship path exists between them).\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error finding path: {str(e)}")]


def handle_get_entity_community(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    max_members = arguments.get("max_members", 50)

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        # First, get the entity's community ID
        entity_metrics = kb.get_entity_metrics(entity_text)

        if not entity_metrics['found']:
            return [TextContent(type="text", text=f"Entity '{entity_text}' not found in graph metrics.\n\nRun 'compute_graph_metrics' first to detect communities.")]

        if 'community' not in entity_metrics['metrics'] or entity_metrics['metrics']['community'] is None:
            return [TextContent(type="text", text=f"No community information for '{entity_text}'.\n\nRun 'compute_graph_metrics' first.")]

        community_id = entity_metrics['metrics']['community']

        # Get all entities in the same community
        cursor = kb.db_conn.cursor()
        rows = cursor.execute("""
            SELECT entity_text, entity_type, pagerank, degree_centrality
            FROM graph_metrics
            WHERE community_id = ?
            ORDER BY pagerank DESC
            LIMIT ?
        """, (community_id, max_members)).fetchall()

        # Build output
        output = f"# Community {community_id}\n\n"
        output += f"**Anchor Entity:** {entity_text} ({entity_metrics['entity_type']})\n"
        output += f"**Total Members:** {len(rows)}\n\n"

        output += f"**Community Members (ranked by PageRank):**\n"
        for i, (ent_text, ent_type, pr, degree) in enumerate(rows, 1):
            marker = " ← anchor" if ent_text == entity_text else ""
            output += f"{i:2d}. {ent_text:30s} (type: {ent_type}, PR: {pr:.6f}){marker}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting community: {str(e)}")]


def handle_get_top_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    metric = arguments.get("metric", "pagerank")
    limit = arguments.get("limit", 10)
    entity_types = arguments.get("entity_types")

    try:
        # Build query based on metric
        metric_column_map = {
            'pagerank': 'pagerank',
            'betweenness': 'betweenness_centrality',
            'degree': 'degree_centrality'
        }

        metric_column = metric_column_map.get(metric, 'pagerank')

        cursor = kb.db_conn.cursor()

        # Build query
        query = f"""
            SELECT entity_text, entity_type, pagerank, betweenness_centrality,
                   degree_centrality, community_id
            FROM graph_metrics
            WHERE {metric_column} IS NOT NULL
        """
        params = []

        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += f" ORDER BY {metric_column} DESC LIMIT ?"
        params.append(limit)

        rows = cursor.execute(query, params).fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No entities found.\n\nRun 'compute_graph_metrics' first to generate metrics.")]

        # Build output
        metric_display = metric.replace('_', ' ').title()
        output = f"# Top {len(rows)} Entities by {metric_display}\n\n"

        if entity_types:
            output += f"**Filtered by types:** {', '.join(entity_types)}\n\n"

        output += f"**Ranking:**\n"
        for rank, (ent_text, ent_type, pr, betw, deg, comm) in enumerate(rows, 1):
            output += f"{rank:2d}. {ent_text:30s}\n"
            output += f"    Type: {ent_type}, Community: {comm}\n"
            output += f"    PageRank: {pr:.6f}, Betweenness: {betw:.6f}, Degree: {deg:.6f}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting top entities: {str(e)}")]


def handle_visualize_graph(kb, name: str, arguments: dict) -> list[TextContent]:
    output_path = arguments.get("output_path", "knowledge_graph.html")
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    color_by = arguments.get("color_by", "entity_type")
    size_by = arguments.get("size_by", "pagerank")
    physics_enabled = arguments.get("physics_enabled", True)
    height = arguments.get("height", "800px")
    width = arguments.get("width", "100%")

    try:
        # Generate visualization
        saved_path = kb.visualize_knowledge_graph(
            G=None,  # Will build graph
            output_path=output_path,
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            color_by=color_by,
            size_by=size_by,
            physics_enabled=physics_enabled,
            height=height,
            width=width
        )

        if not saved_path:
            return [TextContent(type="text", text="Error: Could not generate visualization (empty graph or error occurred).")]

        # Build output message
        output = "# Knowledge Graph Visualization Generated\n\n"
        output += f"**File saved to:** `{saved_path}`\n\n"

        output += f"**Visualization Settings:**\n"
        output += f"- Nodes colored by: {color_by}\n"
        output += f"- Node size based on: {size_by}\n"
        output += f"- Physics simulation: {'Enabled' if physics_enabled else 'Disabled'}\n"

        if entity_types:
            output += f"- Filtered to types: {', '.join(entity_types)}\n"

        output += f"- Min occurrences: {min_occurrences}\n"
        output += f"- Min relationship strength: {min_relationship_strength}\n\n"

        output += f"**Interactive Features:**\n"
        output += f"- Hover over nodes to see entity details and metrics\n"
        output += f"- Hover over edges to see relationship strength\n"
        output += f"- Click and drag nodes to rearrange\n"
        output += f"- Zoom with mouse wheel\n"
        output += f"- Pan by dragging the background\n\n"

        output += f"Open the HTML file in your web browser to explore the interactive visualization!\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating visualization: {str(e)}")]


def handle_train_lda_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    max_iter = arguments.get("max_iter", 100)
    max_features = arguments.get("max_features", 1000)

    try:
        results = kb.train_lda_model(
            num_topics=num_topics,
            max_iter=max_iter,
            max_features=max_features,
            store_results=True
        )

        output = f"# LDA Topic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Vocabulary size: {results['vocabulary_size']}\n"
        output += f"- Perplexity: {results['perplexity']:.2f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training LDA model: {str(e)}")]


def handle_train_nmf_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    max_iter = arguments.get("max_iter", 200)
    max_features = arguments.get("max_features", 1000)

    try:
        results = kb.train_nmf_model(
            num_topics=num_topics,
            max_iter=max_iter,
            max_features=max_features,
            store_results=True
        )

        output = f"# NMF Topic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Vocabulary size: {results['vocabulary_size']}\n"
        output += f"- Reconstruction error: {results['reconstruction_error']:.2f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training NMF model: {str(e)}")]


def handle_train_bertopic(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    min_topic_size = arguments.get("min_topic_size", 5)

    try:
        results = kb.train_bertopic_model(
            num_topics=num_topics,
            min_topic_size=min_topic_size,
            store_results=True
        )

        output = f"# BERTopic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Outliers: {results['outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training BERTopic model: {str(e)}")]


def handle_get_document_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    model_type = arguments.get("model_type")
    doc_id = arguments.get("doc_id")
    min_probability = arguments.get("min_probability", 0.1)

    try:
        cursor = kb.db_conn.cursor()

        if doc_id:
            # Get topics for specific document
            assignments = cursor.execute("""
                SELECT dt.topic_id, t.topic_number, t.top_words, dt.probability
                FROM document_topics dt
                JOIN topics t ON dt.topic_id = t.topic_id
                WHERE dt.doc_id = ? AND dt.model_type = ? AND dt.probability >= ?
                ORDER BY dt.probability DESC
            """, (doc_id, model_type, min_probability)).fetchall()

            if not assignments:
                return [TextContent(type="text", text=f"No topic assignments found for document {doc_id} with model {model_type}")]

            doc = kb.documents.get(doc_id)
            doc_title = doc.title if doc else "Unknown"

            output = f"# Topic Assignments for: {doc_title}\n\n"
            output += f"**Model:** {model_type}\n\n"

            for topic_id, topic_num, top_words_json, probability in assignments:
                import json
                top_words = json.loads(top_words_json)
                output += f"- **Topic {topic_num}** (prob: {probability:.3f}): {', '.join(top_words[:5])}\n"

        else:
            # Get summary of all documents
            stats = cursor.execute("""
                SELECT COUNT(DISTINCT doc_id), COUNT(*)
                FROM document_topics
                WHERE model_type = ?
            """, (model_type,)).fetchone()

            if not stats or stats[0] == 0:
                return [TextContent(type="text", text=f"No topic assignments found for model {model_type}")]

            num_docs, num_assignments = stats

            output = f"# Topic Assignments Summary\n\n"
            output += f"**Model:** {model_type}\n"
            output += f"**Documents:** {num_docs}\n"
            output += f"**Total assignments:** {num_assignments}\n\n"

            # Get topic distribution
            topic_dist = cursor.execute("""
                SELECT t.topic_number, t.top_words, COUNT(DISTINCT dt.doc_id) as doc_count
                FROM topics t
                LEFT JOIN document_topics dt ON t.topic_id = dt.topic_id
                WHERE t.model_type = ?
                GROUP BY t.topic_number
                ORDER BY doc_count DESC
            """, (model_type,)).fetchall()

            output += f"**Topic Distribution:**\n\n"
            for topic_num, top_words_json, doc_count in topic_dist:
                import json
                top_words = json.loads(top_words_json)
                output += f"- **Topic {topic_num}** ({doc_count} docs): {', '.join(top_words[:5])}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting document topics: {str(e)}")]


def handle_cluster_documents_kmeans(kb, name: str, arguments: dict) -> list[TextContent]:
    num_clusters = arguments.get("num_clusters", 10)

    try:
        results = kb.cluster_documents_kmeans(
            num_clusters=num_clusters,
            store_results=True
        )

        output = f"# K-Means Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Silhouette score: {results['silhouette_score']:.3f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Cluster Sizes:**\n\n"
        for i, cluster in enumerate(results['clusters'][:10], 1):
            output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_cluster_documents_dbscan(kb, name: str, arguments: dict) -> list[TextContent]:
    eps = arguments.get("eps", 0.5)
    min_samples = arguments.get("min_samples", 3)

    try:
        results = kb.cluster_documents_dbscan(
            eps=eps,
            min_samples=min_samples,
            store_results=True
        )

        output = f"# DBSCAN Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Outliers: {results['num_outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        if results['clusters']:
            output += f"**Cluster Sizes:**\n\n"
            for i, cluster in enumerate(results['clusters'][:10], 1):
                output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_cluster_documents_hdbscan(kb, name: str, arguments: dict) -> list[TextContent]:
    min_cluster_size = arguments.get("min_cluster_size", 5)
    min_samples = arguments.get("min_samples", 3)

    try:
        results = kb.cluster_documents_hdbscan(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            store_results=True
        )

        output = f"# HDBSCAN Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Outliers: {results['num_outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        if results['clusters']:
            output += f"**Cluster Sizes:**\n\n"
            for i, cluster in enumerate(results['clusters'][:10], 1):
                output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_get_cluster_documents(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    cluster_number = arguments.get("cluster_number")

    try:
        cursor = kb.db_conn.cursor()

        if cluster_number is not None:
            # Get documents in specific cluster
            cluster_id = f"{algorithm}_cluster_{cluster_number}"

            # Get cluster info
            cluster_info = cursor.execute("""
                SELECT num_documents, top_terms, silhouette_score
                FROM clusters
                WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()

            if not cluster_info:
                return [TextContent(type="text", text=f"Cluster {cluster_number} not found for algorithm {algorithm}")]

            num_docs, top_terms_json, silhouette = cluster_info
            import json
            top_terms = json.loads(top_terms_json)

            # Get documents
            docs = cursor.execute("""
                SELECT dc.doc_id, dc.distance
                FROM document_clusters dc
                WHERE dc.cluster_id = ?
                ORDER BY dc.distance ASC
                LIMIT 20
            """, (cluster_id,)).fetchall()

            output = f"# Cluster {cluster_number} ({algorithm})\n\n"
            output += f"**Cluster Info:**\n"
            output += f"- Documents: {num_docs}\n"
            output += f"- Top terms: {', '.join(top_terms[:10])}\n"
            if silhouette:
                output += f"- Silhouette score: {silhouette:.3f}\n"
            output += f"\n**Documents (closest to centroid):**\n\n"

            for doc_id, distance in docs:
                doc = kb.documents.get(doc_id)
                if doc:
                    output += f"- {doc.title[:60]} (distance: {distance:.3f})\n"

        else:
            # Get summary of all clusters
            clusters = cursor.execute("""
                SELECT cluster_number, num_documents, top_terms
                FROM clusters
                WHERE algorithm = ?
                ORDER BY num_documents DESC
            """, (algorithm,)).fetchall()

            if not clusters:
                return [TextContent(type="text", text=f"No clusters found for algorithm {algorithm}")]

            output = f"# Clusters ({algorithm})\n\n"
            output += f"**Total clusters:** {len(clusters)}\n\n"

            for cluster_num, num_docs, top_terms_json in clusters[:20]:
                import json
                top_terms = json.loads(top_terms_json)
                output += f"- **Cluster {cluster_num}** ({num_docs} docs): {', '.join(top_terms[:5])}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting cluster documents: {str(e)}")]


# Phase 3: Temporal Analysis Tools
def handle_extract_document_events(kb, name: str, arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("doc_id")
    min_confidence = arguments.get("min_confidence", 0.5)

    try:
        result = kb.extract_document_events(doc_id, min_confidence=min_confidence)

        output = f"# Event Extraction Complete\n\n"
        output += f"**Document:** {result['title']}\n"
        output += f"**Statistics:**\n"
        output += f"- Total events detected: {result['event_count']}\n"
        output += f"- Events with confidence >= {min_confidence}: {result['filtered_count']}\n"
        output += f"- Events stored to database: {result['stored_count']}\n\n"

        if result['events']:
            output += f"**Extracted Events:**\n\n"
            for i, event in enumerate(result['events'][:10], 1):
                date_str = event['date_info']['text'] if event['date_info'] else 'No date'
                output += f"{i}. **[{event['type']}]** {event['title'][:80]}...\n"
                output += f"   - Date: {date_str}\n"
                output += f"   - Confidence: {event['confidence']:.2f}\n"
                if event['entities']:
                    output += f"   - Entities: {', '.join(event['entities'][:3])}\n"
                output += "\n"

            if len(result['events']) > 10:
                output += f"... and {len(result['events']) - 10} more events\n\n"

        output += f"Events stored to database. Use `get_timeline` or `search_events_by_date` to query.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting events: {str(e)}")]


def handle_get_timeline(kb, name: str, arguments: dict) -> list[TextContent]:
    start_year = arguments.get("start_year")
    end_year = arguments.get("end_year")
    category = arguments.get("category")
    min_importance = arguments.get("min_importance", 1)
    limit = arguments.get("limit")

    try:
        timeline = kb.get_timeline(
            start_year=start_year,
            end_year=end_year,
            category=category,
            min_importance=min_importance,
            limit=limit
        )

        if not timeline:
            # Try to build timeline if empty
            kb.build_timeline()
            timeline = kb.get_timeline(
                start_year=start_year,
                end_year=end_year,
                category=category,
                min_importance=min_importance,
                limit=limit
            )

        if not timeline:
            return [TextContent(type="text", text="No timeline entries found. Extract events from documents first using `extract_document_events`.")]

        output = f"# Timeline\n\n"

        # Add filter info
        filters = []
        if start_year:
            filters.append(f"from {start_year}")
        if end_year:
            filters.append(f"to {end_year}")
        if category:
            filters.append(f"category: {category}")
        if min_importance > 1:
            filters.append(f"importance >= {min_importance}")

        if filters:
            output += f"**Filters:** {', '.join(filters)}\n"

        output += f"**Total entries:** {len(timeline)}\n\n"
        output += f"**Timeline Entries:**\n\n"

        for entry in timeline[:50]:  # Limit to 50 entries
            importance_stars = "⭐" * entry['importance']
            output += f"**[{entry['display_date']}]** {entry['title'][:70]}...\n"
            output += f"  - Type: {entry['event_type']} | Importance: {importance_stars} ({entry['importance']}/5)\n"
            output += f"  - Confidence: {entry['confidence']:.2f}\n\n"

        if len(timeline) > 50:
            output += f"... and {len(timeline) - 50} more entries (use `limit` parameter to see more)\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting timeline: {str(e)}")]


def handle_search_events_by_date(kb, name: str, arguments: dict) -> list[TextContent]:
    start_year = arguments.get("start_year")
    end_year = arguments.get("end_year")
    event_type = arguments.get("event_type")
    min_confidence = arguments.get("min_confidence", 0.5)

    try:
        events = kb.search_events_by_date(
            start_year=start_year,
            end_year=end_year,
            event_type=event_type,
            min_confidence=min_confidence
        )

        if not events:
            return [TextContent(type="text", text="No events found matching the criteria.")]

        output = f"# Event Search Results\n\n"

        # Add search criteria
        criteria = []
        if start_year and end_year:
            criteria.append(f"{start_year}-{end_year}")
        elif start_year:
            criteria.append(f"from {start_year}")
        elif end_year:
            criteria.append(f"to {end_year}")
        if event_type:
            criteria.append(f"type: {event_type}")
        criteria.append(f"confidence >= {min_confidence}")

        output += f"**Search criteria:** {', '.join(criteria)}\n"
        output += f"**Results:** {len(events)} events\n\n"

        # Group events by year
        events_by_year = {}
        for event in events:
            year = event['year']
            if year not in events_by_year:
                events_by_year[year] = []
            events_by_year[year].append(event)

        for year in sorted(events_by_year.keys()):
            year_events = events_by_year[year]
            output += f"## {year} ({len(year_events)} events)\n\n"

            for event in year_events[:20]:  # Limit per year
                output += f"- **[{event['date_normalized']}]** {event['title'][:70]}...\n"
                output += f"  Type: {event['event_type']}, Confidence: {event['confidence']:.2f}\n"

            if len(year_events) > 20:
                output += f"  ... and {len(year_events) - 20} more events from {year}\n"

            output += "\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error searching events: {str(e)}")]


def handle_get_historical_context(kb, name: str, arguments: dict) -> list[TextContent]:
    year = arguments.get("year")
    context_years = arguments.get("context_years", 2)

    try:
        context = kb.get_historical_context(year, context_years=context_years)

        if context['total_events'] == 0:
            return [TextContent(type="text", text=f"No events found in the period {context['year_range'][0]}-{context['year_range'][1]}.")]

        output = f"# Historical Context for {year}\n\n"
        output += f"**Context range:** {context['year_range'][0]} - {context['year_range'][1]}\n"
        output += f"**Total events:** {context['total_events']}\n\n"

        # Show events by year
        for event_year in sorted(context['events_by_year'].keys()):
            year_events = context['events_by_year'][event_year]
            marker = "📍 **TARGET YEAR**" if event_year == year else ""

            output += f"## {event_year} ({len(year_events)} events) {marker}\n\n"

            for event in year_events[:10]:
                output += f"- **[{event['event_type']}]** {event['title'][:70]}...\n"
                output += f"  Date: {event['date_normalized']}, Confidence: {event['confidence']:.2f}\n"

            if len(year_events) > 10:
                output += f"  ... and {len(year_events) - 10} more events\n"

            output += "\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting historical context: {str(e)}")]


def handle_analyze_graph_pagerank(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    top_n = arguments.get("top_n", 20)
    alpha = arguments.get("alpha", 0.85)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate PageRank
        pagerank = kb.analyze_pagerank(G, alpha=alpha)

        # Format results
        output = f"# PageRank Analysis\n\n"
        output += f"**Total entities:** {len(pagerank)}\n"
        output += f"**Damping factor (alpha):** {alpha}\n\n"
        output += f"## Top {top_n} Entities by PageRank\n\n"

        for i, (entity, score) in enumerate(list(pagerank.items())[:top_n], 1):
            node_data = G.nodes[entity]
            output += f"{i}. **{entity}**\n"
            output += f"   - PageRank: {score:.6f}\n"
            output += f"   - Type: {node_data['type']}\n"
            output += f"   - Occurrences: {node_data['occurrences']}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing PageRank: {str(e)}")]


def handle_detect_graph_communities(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    algorithm = arguments.get("algorithm", "louvain")

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Detect communities
        communities = kb.detect_communities(G, algorithm=algorithm)
        num_communities = len(set(communities.values()))

        # Format results
        output = f"# Community Detection\n\n"
        output += f"**Algorithm:** {algorithm}\n"
        output += f"**Total entities:** {len(communities)}\n"
        output += f"**Communities detected:** {num_communities}\n\n"

        # Group entities by community
        community_groups = {}
        for entity, comm_id in communities.items():
            if comm_id not in community_groups:
                community_groups[comm_id] = []
            community_groups[comm_id].append(entity)

        # Show communities sorted by size
        sorted_communities = sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)

        output += "## Communities (by size)\n\n"
        for i, (comm_id, members) in enumerate(sorted_communities[:10], 1):
            output += f"### Community {comm_id} ({len(members)} members)\n\n"
            # Show first 10 members
            for entity in members[:10]:
                node_type = G.nodes[entity].get('type', 'unknown')
                output += f"- {entity} (type: {node_type})\n"
            if len(members) > 10:
                output += f"- ... and {len(members) - 10} more\n"
            output += "\n"

        if len(sorted_communities) > 10:
            output += f"\n... and {len(sorted_communities) - 10} more communities\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error detecting communities: {str(e)}")]


def handle_calculate_graph_centrality(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    top_n = arguments.get("top_n", 10)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate centrality
        centrality = kb.calculate_centrality(G)

        # Format results
        output = f"# Centrality Analysis\n\n"
        output += f"**Total entities:** {G.number_of_nodes()}\n\n"

        # Show top entities for each centrality measure
        for measure_name, measure_values in centrality.items():
            sorted_values = sorted(measure_values.items(), key=lambda x: x[1], reverse=True)

            output += f"## Top {top_n} by {measure_name.capitalize()} Centrality\n\n"
            output += f"*{measure_name.capitalize()} measures "
            if measure_name == 'betweenness':
                output += "entities that bridge different parts of the graph*\n\n"
            elif measure_name == 'closeness':
                output += "entities that are close to all other entities*\n\n"
            elif measure_name == 'degree':
                output += "entities with many direct connections*\n\n"

            for i, (entity, score) in enumerate(sorted_values[:top_n], 1):
                node_data = G.nodes[entity]
                output += f"{i}. **{entity}**\n"
                output += f"   - Score: {score:.6f}\n"
                output += f"   - Type: {node_data['type']}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating centrality: {str(e)}")]


def handle_get_graph_statistics(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate statistics
        density = nx.density(G)
        num_components = nx.number_connected_components(G)

        # Get degree distribution
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        # Format results
        output = f"# Knowledge Graph Statistics\n\n"
        output += f"## Basic Metrics\n\n"
        output += f"- **Nodes (entities):** {G.number_of_nodes()}\n"
        output += f"- **Edges (relationships):** {G.number_of_edges()}\n"
        output += f"- **Density:** {density:.4f} (0 = sparse, 1 = complete)\n"
        output += f"- **Connected components:** {num_components}\n\n"

        # Show largest connected component
        if num_components > 0:
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            output += f"## Connected Components\n\n"
            output += f"- Largest: {len(components[0])} nodes ({len(components[0])/G.number_of_nodes()*100:.1f}%)\n"
            if len(components) > 1:
                output += f"- Second largest: {len(components[1])} nodes\n"
            output += f"- Isolated nodes: {sum(1 for c in components if len(c) == 1)}\n\n"

        # Degree statistics
        output += f"## Degree Distribution\n\n"
        output += f"- Average degree: {avg_degree:.2f}\n"
        output += f"- Maximum degree: {max_degree}\n"
        output += f"- Minimum degree: {min_degree}\n\n"

        # Entity type distribution
        type_counts = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        output += f"## Entity Types\n\n"
        for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            output += f"- {etype}: {count} ({count/G.number_of_nodes()*100:.1f}%)\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting graph statistics: {str(e)}")]


# Phase 2: Visualization Tools
def handle_generate_topic_wordcloud(kb, name: str, arguments: dict) -> list[TextContent]:
    topic_id = arguments.get("topic_id")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 800)
    height = arguments.get("height", 400)
    background_color = arguments.get("background_color", "white")

    try:
        result = kb.generate_topic_wordcloud(
            topic_id=topic_id,
            output_path=output_path,
            width=width,
            height=height,
            background_color=background_color
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Topic Word Cloud Generated\n{'='*60}\n\n"
        output += f"Topic: {result['topic_number']} ({result['model_type'].upper()})\n"
        output += f"Output File: {result['output_path']}\n"
        output += f"Image Size: {width}x{height}px\n"
        output += f"Words Visualized: {result['num_words']}\n\n"
        output += "The word cloud has been saved. Larger words indicate higher importance in the topic.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating word cloud: {str(e)}")]


def handle_visualize_cluster_scatter(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 1200)
    height = arguments.get("height", 800)
    n_neighbors = arguments.get("n_neighbors", 15)
    min_dist = arguments.get("min_dist", 0.1)

    try:
        result = kb.visualize_cluster_scatter(
            algorithm=algorithm,
            output_path=output_path,
            width=width,
            height=height,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Cluster Scatter Plot Generated\n{'='*60}\n\n"
        output += f"Algorithm: {result['algorithm'].upper()}\n"
        output += f"Clusters Visualized: {result['num_clusters']}\n"
        output += f"Documents Plotted: {result['num_documents']}\n"
        output += f"Output File: {result['output_path']}\n"
        output += f"Image Size: {width}x{height}px\n\n"
        output += "The scatter plot shows document clusters in 2D space using UMAP projection.\n"
        output += "Documents in the same cluster appear closer together in the visualization.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating scatter plot: {str(e)}")]


def handle_generate_topic_heatmap(kb, name: str, arguments: dict) -> list[TextContent]:
    model_type = arguments.get("model_type")
    output_path = arguments.get("output_path")
    max_topics = arguments.get("max_topics", 20)
    max_documents = arguments.get("max_documents", 50)

    try:
        result = kb.generate_topic_heatmap(
            model_type=model_type,
            output_path=output_path,
            max_topics=max_topics,
            max_documents=max_documents
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Topic Heatmap Generated\n{'='*60}\n\n"
        output += f"Model Type: {result['model_type'].upper()}\n"
        output += f"Topics: {result['num_topics']}\n"
        output += f"Documents: {result['num_documents']}\n"
        output += f"Output File: {result['output_path']}\n\n"
        output += "The heatmap shows topic-document probability matrix.\n"
        output += "Brighter colors indicate higher topic probability for a document.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating heatmap: {str(e)}")]


def handle_visualize_cluster_distribution(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 1000)
    height = arguments.get("height", 600)

    try:
        result = kb.visualize_cluster_distribution(
            algorithm=algorithm,
            output_path=output_path,
            width=width,
            height=height
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Cluster Distribution Chart Generated\n{'='*60}\n\n"
        output += f"Algorithm: {result['algorithm'].upper()}\n"
        output += f"Clusters: {result['num_clusters']}\n"
        output += f"Output File: {result['output_path']}\n\n"
        output += "Cluster Sizes:\n"

        # Show cluster sizes
        for cluster_name, size in sorted(result['cluster_sizes'].items()):
            if cluster_name == 'outliers':
                output += f"  Outliers: {size} documents\n"
            else:
                cluster_num = cluster_name.replace('cluster_', '')
                output += f"  Cluster {cluster_num}: {size} documents\n"

        output += "\nThe bar chart shows the distribution of documents across clusters.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating distribution chart: {str(e)}")]


HANDLERS = {
    "search_docs": handle_search_docs,
    "translate_query": handle_translate_query,
    "semantic_search": handle_semantic_search,
    "hybrid_search": handle_hybrid_search,
    "fuzzy_search": handle_fuzzy_search,
    "search_within_results": handle_search_within_results,
    "suggest_tags": handle_suggest_tags,
    "get_tags_by_category": handle_get_tags_by_category,
    "faceted_search": handle_faceted_search,
    "get_chunk": handle_get_chunk,
    "get_document": handle_get_document,
    "list_docs": handle_list_docs,
    "get_document_by_card_id": handle_get_document_by_card_id,
    "add_document": handle_add_document,
    "update_document": handle_update_document,
    "scrape_url": handle_scrape_url,
    "rescrape_document": handle_rescrape_document,
    "check_url_updates": handle_check_url_updates,
    "remove_document": handle_remove_document,
    "find_similar": handle_find_similar,
    "answer_question": handle_answer_question,
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
    "batch_ocr_figures": handle_batch_ocr_figures,
    "ocr_document_figures": handle_ocr_document_figures,
    "figure_ocr_status": handle_figure_ocr_status,
    "search_figures": handle_search_figures,
    "get_document_figures": handle_get_document_figures,
    "build_knowledge_graph": handle_build_knowledge_graph,
    "compute_graph_metrics": handle_compute_graph_metrics,
    "get_entity_metrics": handle_get_entity_metrics,
    "find_entity_path": handle_find_entity_path,
    "get_entity_community": handle_get_entity_community,
    "get_top_entities": handle_get_top_entities,
    "visualize_graph": handle_visualize_graph,
    "train_lda_topics": handle_train_lda_topics,
    "train_nmf_topics": handle_train_nmf_topics,
    "train_bertopic": handle_train_bertopic,
    "get_document_topics": handle_get_document_topics,
    "cluster_documents_kmeans": handle_cluster_documents_kmeans,
    "cluster_documents_dbscan": handle_cluster_documents_dbscan,
    "cluster_documents_hdbscan": handle_cluster_documents_hdbscan,
    "get_cluster_documents": handle_get_cluster_documents,
    "extract_document_events": handle_extract_document_events,
    "get_timeline": handle_get_timeline,
    "search_events_by_date": handle_search_events_by_date,
    "get_historical_context": handle_get_historical_context,
    "analyze_graph_pagerank": handle_analyze_graph_pagerank,
    "detect_graph_communities": handle_detect_graph_communities,
    "calculate_graph_centrality": handle_calculate_graph_centrality,
    "get_graph_statistics": handle_get_graph_statistics,
    "generate_topic_wordcloud": handle_generate_topic_wordcloud,
    "visualize_cluster_scatter": handle_visualize_cluster_scatter,
    "generate_topic_heatmap": handle_generate_topic_heatmap,
    "visualize_cluster_distribution": handle_visualize_cluster_distribution,
}
