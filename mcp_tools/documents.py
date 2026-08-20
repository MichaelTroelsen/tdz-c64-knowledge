"""Document/chunk CRUD and ingest handlers: lookup, add/update/remove,
URL scraping, similarity, and RAG answer_question. Split out of
handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


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


def handle_add_deepsid_document(kb, name: str, arguments: dict) -> list[TextContent]:
    """Ingest one DeepSID tune's metadata as a document."""
    fullname = arguments.get("fullname")
    tags = arguments.get("tags")

    if not fullname:
        return [TextContent(type="text", text="Error: fullname is required")]

    try:
        doc = kb.add_deepsid_document(fullname, tags)
    except Exception as e:
        # DeepSidError's messages name their own cause (missing XHR header,
        # wrong path format, no matching tune), so surfacing the text is more
        # useful here than classifying it.
        return [TextContent(type="text", text=f"Error ingesting DeepSID tune: {e}")]

    return [TextContent(type="text", text=(
        f"Added DeepSID document: {doc.title}\n"
        f"  doc_id: {doc.doc_id}\n"
        f"  source: {doc.source_url}\n"
        f"  stored at: {doc.filepath}\n"
        f"  chunks: {doc.total_chunks}"
    ))]


def handle_repoint_document(kb, name: str, arguments: dict) -> list[TextContent]:
    """Repair a document whose source file has moved, without reingesting."""
    doc_id = arguments.get("doc_id")
    new_filepath = arguments.get("new_filepath")
    force = bool(arguments.get("force", False))

    if not doc_id or not new_filepath:
        return [TextContent(type="text", text="Error: doc_id and new_filepath are both required")]

    try:
        result = kb.repoint_document(doc_id, new_filepath, force=force)
    except Exception as e:
        return [TextContent(type="text", text=f"Error re-pointing document: {e}")]

    if result["hash_verified"]:
        provenance = "content hash matches the recorded hash"
    elif result["forced"]:
        provenance = "WARNING: forced past a content-hash mismatch"
    else:
        provenance = "no recorded hash was available to verify against"

    output = (
        f"# Re-pointed {result['doc_id']}\n\n"
        f"- Was: {result['old_filepath']}\n"
        f"- Now: {result['new_filepath']}\n"
        f"- Verification: {provenance}\n\n"
        "The document's text, chunks and embeddings were not touched - only the "
        "recorded source path. Run health_check to confirm missing_source_files dropped."
    )
    return [TextContent(type="text", text=output)]


HANDLERS_DOCUMENTS = {
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
    "repoint_document": handle_repoint_document,
    "add_deepsid_document": handle_add_deepsid_document,
    "find_similar": handle_find_similar,
    "answer_question": handle_answer_question,
}
