"""IngestMixin assembled from its sub-mixins (R12 step 4).

kb/ingest.py grew to 3,977 lines / 55 methods, over the ~2,500-line module
target. It is split here into three sub-mixins by responsibility -
file-type extraction, document lifecycle (add/update/remove/scrape/bulk),
and tagging/summarisation - each its own module, still under the target.

The public name `IngestMixin` (and `from kb import IngestMixin`) is
unchanged: this package's __init__ composes it from the sub-mixins below,
so server.py needs no edit.

Splitting a mixin into pieces and recombining by inheritance means a
method defined in two pieces would silently shadow one of them, and the
method count would still look right. Guard it explicitly, the same way
mcp_tools/handlers.py guards its handler-dict merge: assert no method name
appears twice across the sub-mixins, and that the union equals the
original 55-name set.
"""
from ._documents import _DocumentsMixin
from ._extraction import _ExtractionMixin
from ._tagging import _TaggingMixin

_SUB_MIXINS = [_ExtractionMixin, _DocumentsMixin, _TaggingMixin]

_EXPECTED_METHODS = frozenset({
    "_chunk_text", "_extract_pdf_with_ocr", "_check_poppler_available",
    "_extract_pdf_text", "_extract_text_file", "_extract_excel_file",
    "_extract_html_file", "_extract_tables", "_table_to_markdown",
    "_detect_code_blocks", "_extract_facets", "_extract_hardware_refs",
    "_extract_instructions", "_extract_registers", "_extract_cross_references",
    "_extract_memory_addresses", "_extract_register_offsets",
    "_extract_page_references", "_get_reference_context",
    "_extract_text_for_file", "_extract_card_id", "_detect_and_extract_frames",
    "_find_mdscrape_executable", "_extract_source_url_from_md",
    "_add_scraped_document", "_is_path_allowed", "get_document_by_card_id",
    "_rebuild_entity_relationships", "_mark_superseded", "add_document",
    "scrape_url", "rescrape_document", "_discover_urls", "check_url_updates",
    "remove_document", "needs_reindex", "_reindex_document_if_changed",
    "update_document", "update_document_title", "update_document_tags",
    "check_all_updates", "add_documents_bulk", "remove_documents_bulk",
    "export_documents_bulk", "suggest_tags", "add_tags_to_document",
    "get_tags_by_category", "_call_llm", "summarize_document",
    "update_tags_bulk", "auto_tag_document", "auto_tag_all_documents",
    "generate_summary", "generate_summary_all", "get_summary",
})

_seen: dict = {}
for _cls in _SUB_MIXINS:
    for _name in vars(_cls):
        if _name.startswith("__"):
            continue
        if _name in _seen:
            raise RuntimeError(
                f"duplicate method {_name!r} defined in both "
                f"{_seen[_name].__name__} and {_cls.__name__} - this would "
                "silently shadow one of them"
            )
        _seen[_name] = _cls

_actual = frozenset(_seen)
if _actual != _EXPECTED_METHODS:
    _missing = _EXPECTED_METHODS - _actual
    _extra = _actual - _EXPECTED_METHODS
    raise RuntimeError(
        f"IngestMixin sub-mixin method set changed unexpectedly: "
        f"missing={sorted(_missing)} extra={sorted(_extra)}"
    )


class IngestMixin(_ExtractionMixin, _DocumentsMixin, _TaggingMixin):
    pass
