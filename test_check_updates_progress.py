"""check_updates(auto_update=True) must report progress, not run silently.

Issue #17's second half was mis-scoped as "check_updates hangs on a
disallowed path" (see ISSUE-17-RESCOPE.md) - check_all_updates takes no
path at all, so there is nothing for it to reject. What actually happened:
the reporter called it with auto_update=true, which turns a several-second
read-only scan into a full remove-then-add re-index of every changed
document, and watched 1800s of silence before aborting it as a hang. It
was not hanging - there was just no way to tell a correctly-running
multi-hour rebuild apart from a wedged one.

This test builds a small isolated corpus (never the live database - see
CLAUDE.md / the task's own constraint) with one changed document, and
checks that check_all_updates(auto_update=True) drives a progress_callback
at least once before it returns, so a caller has *something* to observe
progressing during the run.
"""

import os
import time

import pytest


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A KnowledgeBase on an isolated data dir - never the live one."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    from server import KnowledgeBase
    instance = KnowledgeBase(str(tmp_path))
    yield instance
    instance.close()


def _make_changed_doc(kb, tmp_path):
    """Index one real document, then edit it on disk and push its mtime
    forward so needs_reindex() sees it as changed - the same condition
    check_all_updates(auto_update=True) re-indexes in production.
    """
    doc_dir = tmp_path / "temp"
    doc_dir.mkdir(exist_ok=True)
    filepath = doc_dir / "note.txt"
    filepath.write_text("original content", encoding="utf-8")

    doc = kb.add_document(str(filepath), title="Note")

    # Edit the file and move its mtime past what was recorded at index time.
    filepath.write_text("edited content, different from what was indexed", encoding="utf-8")
    future = time.time() + 5
    os.utime(filepath, (future, future))

    return doc, filepath


def test_changed_file_on_disk_before_check(kb, tmp_path):
    """Sanity check the mutation actually landed before trusting anything
    check_all_updates reports about it (paste-the-measurement-first, per
    the task's sabotage instructions).
    """
    doc, filepath = _make_changed_doc(kb, tmp_path)
    on_disk = filepath.read_text(encoding="utf-8")
    assert on_disk == "edited content, different from what was indexed"
    assert kb.needs_reindex(str(filepath), doc.doc_id) is True


def test_auto_update_reports_progress(kb, tmp_path):
    """The defect: a caller had no way to tell check_updates(auto_update=true)
    was progressing. check_all_updates must drive progress_callback at least
    once for the changed document it re-indexes.
    """
    _make_changed_doc(kb, tmp_path)

    updates = []
    results = kb.check_all_updates(auto_update=True, progress_callback=updates.append)

    assert len(results["changed"]) == 1
    assert len(results["updated"]) == 1

    # At minimum: a start event and one event for the re-indexed document.
    assert len(updates) >= 2, (
        f"expected check_all_updates to report progress at least twice "
        f"(start + one re-indexed document), got {len(updates)}: {updates}"
    )

    messages = [u.message for u in updates]
    assert any("note.txt" in (u.item or "") for u in updates), (
        f"expected a progress update naming the file being re-indexed, got: {messages}"
    )


def test_no_progress_reported_without_callback(kb, tmp_path):
    """progress_callback is optional - omitting it must not break the call."""
    _make_changed_doc(kb, tmp_path)
    results = kb.check_all_updates(auto_update=True)
    assert len(results["updated"]) == 1


def test_mcp_handler_wires_progress_callback_to_logger(kb, tmp_path, monkeypatch, caplog):
    """The MCP handler (mcp_tools/admin.py:handle_check_updates) is the
    thing an actual tool call goes through. MCP tool calls in this server
    are single blocking request/response, so nothing reaches the MCP
    client mid-call regardless of what the kb-level callback does (see
    ISSUE-17-RESCOPE.md and the docstring added to check_all_updates) -
    the observable channel this wires up is the server's own log, for an
    operator tailing it. This confirms that channel actually fires rather
    than only being present in the standalone kb-level test above.
    """
    import logging
    from mcp_tools.admin import handle_check_updates

    _make_changed_doc(kb, tmp_path)

    caplog.set_level(logging.INFO, logger=kb.logger.name)
    handle_check_updates(kb, "check_updates", {"auto_update": True})

    progress_records = [r for r in caplog.records if "check_updates:" in r.message]
    assert progress_records, "expected handle_check_updates(auto_update=True) to log progress"
