"""Dispatch smoke test: every registered MCP tool must be callable.

R17 from CODE-REVIEW.md: the 87-tool dispatch in server.py's
_call_tool_impl (an ~3300-line `elif name == "..."` chain) had zero test
coverage. Most branches wrap their real logic in try/except and return a
handled `Error: ...` TextContent on failure - that's a legitimate outcome
this test accepts. What it catches is a bug in the *dispatch* code itself:
a typo'd arguments.get() key, a wrong variable name, or any other mistake
that raises an unhandled exception before or after that inner try/except,
across the whole tool surface cheaply and in one pass.

This is intentionally NOT a correctness test for each tool's business
logic (see the "Known limitations" tools below, and the TODOs in
CODE-REVIEW.md R17 for follow-up: FTS5/BM25 ranking correctness, entity
extraction accuracy, backup/restore round-tripping).
"""
import asyncio
import os

import pytest

import server as server_module


# Tools whose required args need a specific, meaningfully-typed value to
# reach their dispatch code at all (a generic string/int placeholder would
# either KeyError inside a nested lookup unrelated to dispatch correctness,
# or the tool would trivially no-op). Resolved lazily against the fixture
# in _build_args() since some depend on the fixture's real doc_id/tmp_path.
def _special_args(tool_name: str, fixture: dict) -> dict:
    doc_id = fixture["doc_id"]
    tmp = fixture["tmp_path"]
    specials = {
        "get_chunk": {"doc_id": doc_id, "chunk_id": 0},
        "get_document": {"doc_id": doc_id},
        "update_document": {"card_id_or_doc_id": doc_id, "filepath": fixture["doc_path"]},
        "get_document_by_card_id": {"card_id": doc_id},
        "find_similar": {"doc_id": doc_id},
        "compare_documents": {"doc_id_1": doc_id, "doc_id_2": doc_id},
        "add_document": {"filepath": fixture["doc_path"]},
        "add_documents_bulk": {"directory": str(tmp)},
        "scrape_url": {"url": "http://127.0.0.1:9/does-not-resolve"},  # must fail fast, not hang
        "search_within_results": {
            "previous_results": [{"doc_id": doc_id, "chunk_id": 0, "title": "t", "snippet": "s", "score": 1.0}],
            "refinement_query": "test",
        },
        "export_results": {"results": [{"doc_id": doc_id, "chunk_id": 0, "title": "t", "snippet": "s", "score": 1.0}]},
        "find_by_reference": {"ref_type": "register", "ref_value": "$D020"},
        "create_backup": {"dest_dir": str(tmp / "backup")},
        "restore_backup": {"backup_path": str(tmp / "does-not-exist")},
        "extract_document_events": {"doc_id": doc_id},
        "get_historical_context": {"year": 1982},
        "summarize_document": {"doc_id": doc_id},
        "auto_tag_document": {"doc_id": doc_id},
        "suggest_tags": {"doc_id": doc_id},
        "extract_entities": {"doc_id": doc_id},
        "generate_summary": {"doc_id": doc_id},
        "get_summary": {"doc_id": doc_id},
        "get_document_topics": {"doc_id": doc_id},
        "get_cluster_documents": {"doc_id": doc_id},
        "generate_topic_wordcloud": {"topic_id": 0, "output_path": str(tmp / "wc.png")},
        "visualize_cluster_scatter": {"algorithm": "kmeans", "output_path": str(tmp / "scatter.png")},
        "generate_topic_heatmap": {"model_type": "lda", "output_path": str(tmp / "heatmap.png")},
        "visualize_cluster_distribution": {"algorithm": "kmeans", "output_path": str(tmp / "dist.png")},
        "train_lda_topics": {"model_type": "lda"},
        "cluster_documents_kmeans": {"algorithm": "kmeans"},
        "cluster_documents_dbscan": {"algorithm": "dbscan"},
        "cluster_documents_hdbscan": {"algorithm": "hdbscan"},
    }
    return specials.get(tool_name, {})


def _placeholder_for(prop_name: str, prop_schema: dict, fixture: dict):
    if "enum" in prop_schema and prop_schema["enum"]:
        return prop_schema["enum"][0]
    ptype = prop_schema.get("type")
    if ptype == "string":
        return prop_schema.get("default", "test")
    if ptype == "integer":
        return prop_schema.get("default", 1)
    if ptype == "number":
        return prop_schema.get("default", 1.0)
    if ptype == "boolean":
        return prop_schema.get("default", False)
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return "test"


def _build_args(tool, fixture: dict) -> dict:
    schema = tool.inputSchema or {}
    props = schema.get("properties", {})
    required = schema.get("required", [])

    args = dict(_special_args(tool.name, fixture))
    for key in required:
        if key in args:
            continue
        args[key] = _placeholder_for(key, props.get(key, {}), fixture)
    return args


@pytest.fixture
def dispatch_fixture(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TDZ_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "0")  # keep this smoke test fast

    kb = server_module.KnowledgeBase(str(data_dir))

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "The VIC-II chip handles sprites and raster interrupts. "
        "The SID chip at $D400 handles sound synthesis.",
        encoding="utf-8",
    )
    doc = kb.add_document(str(doc_path))

    monkeypatch.setattr(server_module, "kb", kb)
    try:
        yield {"kb": kb, "doc_id": doc.doc_id, "doc_path": str(doc_path), "tmp_path": tmp_path}
    finally:
        kb.close()


def test_every_registered_tool_is_dispatchable(dispatch_fixture):
    tools = asyncio.run(server_module.list_tools())
    assert len(tools) >= 80, (
        f"expected ~87 registered tools, found {len(tools)} - "
        "list_tools() may not be returning its normal full set"
    )

    failures = []
    for tool in tools:
        args = _build_args(tool, dispatch_fixture)
        try:
            result = asyncio.run(server_module.call_tool(tool.name, args))
        except Exception as e:  # noqa: BLE001 - intentionally broad, this IS the check
            failures.append(f"{tool.name}({args}): raised {type(e).__name__}: {e}")
            continue
        if not result:
            failures.append(f"{tool.name}({args}): returned an empty result list")

    assert not failures, "Dispatch failures:\n" + "\n".join(failures)
