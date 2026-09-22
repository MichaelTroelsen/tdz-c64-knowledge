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


def test_handle_check_updates_emits_progress_per_document(kb, tmp_path):
    """The progress channel server.py built (emit_tool_progress /
    _progress_channel) was unused outside server.py itself until this task.
    handle_check_updates's progress_callback is the seam: it must forward
    kb's own ProgressUpdate.current/total/message to emit_tool_progress
    alongside the existing log line, not instead of it.

    This drives the seam directly via the contextvar, without a live MCP
    session, as an in-process proxy for "a client that sent a
    progressToken" - see test_client_receives_per_document_progress_for_
    check_updates below for the real wire-level version.
    """
    import server as server_module
    from mcp_tools.admin import handle_check_updates

    _make_changed_doc(kb, tmp_path)

    sent = []

    class _FakeChannel:
        def send(self, progress, total=None, message=None):
            sent.append((progress, total, message))
            return True

    token = server_module._progress_channel.set(_FakeChannel())
    try:
        handle_check_updates(kb, "check_updates", {"auto_update": True})
    finally:
        server_module._progress_channel.reset(token)

    assert len(sent) >= 2, (
        f"expected at least a start event and one re-indexed-document event "
        f"to reach emit_tool_progress, got {len(sent)}: {sent}"
    )
    assert any("Re-indexed" in (message or "") for _, _, message in sent), (
        f"expected a per-document re-index notification, got: {sent}"
    )
    # The corpus built by _make_changed_doc has exactly one document, so the
    # scan's total is known before any re-indexing starts - the first event
    # must carry that real total, not omit it or fabricate one.
    first_progress, first_total, first_message = sent[0]
    assert first_total == 1, (
        f"expected the real known total (1 document) on the first progress "
        f"event, got total={first_total} message={first_message!r}"
    )

    # The per-document re-index events' own total (reindexed_count) is not a
    # real total - check_all_updates itself doesn't know how many documents
    # will end up changed until the scan is over, so that total always
    # equals current (1/1, 2/2, ...). Forwarding it verbatim would render as
    # a finished bar on every single document. The handler must send no
    # total on those events, and keep sending the honest total_docs on the
    # up-front one.
    reindex_events = [
        (progress, total, message) for progress, total, message in sent
        if "Re-indexed" in (message or "")
    ]
    assert reindex_events, f"expected at least one re-indexed-document event, got: {sent}"
    for progress, total, message in reindex_events:
        assert total is None, (
            f"expected no total on a per-document re-index event (its own "
            f"total is fabricated - always equal to current), got "
            f"total={total} message={message!r}"
        )


def test_client_receives_per_document_progress_for_check_updates(tmp_path, monkeypatch):
    """Wire-level proof: a client that sends a progressToken and calls
    check_updates with auto_update=true gets a notifications/progress frame
    per document re-indexed, not just the generic ack/heartbeat every tool
    call already gets (server.py's _open_progress_channel /
    _progress_heartbeat).
    """
    from test_mcp_startup import MCPSession, HANDSHAKE_BUDGET_S, _env
    import server as server_module

    for key, value in _env(tmp_path).items():
        monkeypatch.setenv(key, value)

    kb_pre = server_module.KnowledgeBase(str(tmp_path))
    try:
        _make_changed_doc(kb_pre, tmp_path)
    finally:
        kb_pre.close()

    with MCPSession(tmp_path, client_name="check-updates-progress") as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None and "result" in response, response
        frames = s.call_tool_frames(
            "check_updates", {"auto_update": True}, progress_token="cu-1",
            req_id=51, timeout=90,
        )
        stderr = s.stderr_text()

    progress = [
        f for f in frames
        if f.get("method") == "notifications/progress"
        and f.get("params", {}).get("progressToken") == "cu-1"
    ]
    assert len(progress) >= 2, (
        "expected more than the generic ack/heartbeat for "
        f"check_updates(auto_update=true); got {len(progress)} progress "
        f"frames: {progress}. stderr: {stderr[-1500:]!r}"
    )
    messages = [p["params"].get("message") for p in progress]
    assert any("Re-indexed" in (m or "") for m in messages), (
        f"expected a progress notification for the re-indexed document, "
        f"got messages: {messages}"
    )
    assert "result" in frames[-1] or "error" in frames[-1], frames[-1]


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
