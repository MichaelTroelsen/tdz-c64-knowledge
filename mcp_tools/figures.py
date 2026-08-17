"""Figure OCR Tool Handlers. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


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


HANDLERS_FIGURES = {
    "batch_ocr_figures": handle_batch_ocr_figures,
    "ocr_document_figures": handle_ocr_document_figures,
    "figure_ocr_status": handle_figure_ocr_status,
    "search_figures": handle_search_figures,
    "get_document_figures": handle_get_document_figures,
}
