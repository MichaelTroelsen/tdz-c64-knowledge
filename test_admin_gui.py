"""Regression coverage for the admin GUI's page modules (R18).

admin_gui.py dispatches to admin_pages.PAGES[label](kb) from a sidebar
radio. This drives each of those 16 render(kb) functions through
Streamlit's AppTest harness and asserts the page renders without an
uncaught exception.

Isolation: admin_gui.py builds its KnowledgeBase from TDZ_DATA_DIR (or
~/.tdz-c64-knowledge if unset) at script-exec time, and KnowledgeBase
writes to that directory (WAL files, caches, background indexing). Every
test below monkeypatches TDZ_DATA_DIR to a pytest tmp_path *before*
AppTest execs admin_gui.py, and the session-scoped guard_live_db fixture
below proves the user's real ~/.tdz-c64-knowledge/knowledge_base.db was
never touched.
"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import admin_common
import admin_pages
from admin_pages import PAGES


def test_format_timestamp_valid_iso8601():
    """Regression: admin_common.py had no datetime import at all, so this
    raised NameError instead of formatting the timestamp."""
    result = admin_common.format_timestamp("2024-03-15T10:30:00")
    assert result == "2024-03-15 10:30:00"


def test_format_timestamp_non_iso_string_passthrough():
    """The except clause must still absorb genuinely malformed input and
    return it unchanged."""
    result = admin_common.format_timestamp("not-a-timestamp")
    assert result == "not-a-timestamp"

LIVE_DATA_DIR = Path(os.path.expanduser("~/.tdz-c64-knowledge"))
LIVE_DB = LIVE_DATA_DIR / "knowledge_base.db"


@pytest.fixture(scope="session", autouse=True)
def guard_live_db():
    """Session-wide proof: the user's live database is byte-for-byte
    unchanged (mtime and size) before vs. after this whole test run."""
    before = LIVE_DB.stat() if LIVE_DB.exists() else None
    yield
    after = LIVE_DB.stat() if LIVE_DB.exists() else None
    if before is None:
        assert after is None, "live knowledge_base.db appeared during the test run"
    else:
        assert after is not None, "live knowledge_base.db disappeared during the test run"
        assert before.st_mtime == after.st_mtime, (
            f"live knowledge_base.db mtime changed: {before.st_mtime} -> {after.st_mtime}"
        )
        assert before.st_size == after.st_size, (
            f"live knowledge_base.db size changed: {before.st_size} -> {after.st_size}"
        )


def _run_page(tmp_path, monkeypatch, label):
    """Boot admin_gui.py against an isolated tmp KB and select one page."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    at = AppTest.from_file("admin_gui.py", default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(label).run()
    return at


@pytest.mark.parametrize("label", list(PAGES.keys()))
def test_page_renders_without_exception(tmp_path, monkeypatch, label):
    at = _run_page(tmp_path, monkeypatch, label)

    assert list(at.exception) == [], f"{label!r} raised: {list(at.exception)}"

    # Proves the KB this run actually used is the isolated tmp copy, not
    # the user's live ~/.tdz-c64-knowledge database.
    kb = at.session_state["kb"]
    assert str(kb.data_dir) == str(tmp_path)

    # "Rendered" has to mean something: the page must have actually
    # produced output, not silently rendered nothing.
    assert len(at.main) > 0, f"{label!r} produced no elements"


def test_url_monitoring_renders_with_url_sourced_document(tmp_path, monkeypatch):
    """Regression: render() imported `json` and `datetime` mid-function (in
    a button branch near the end), making both names local to the WHOLE
    function per Python scoping rules. Every reference above those inline
    imports was unbound unless that button had already been clicked.

    The default PAGES fixture has no url-sourced documents, so the loop
    body touching `json` never executes and the bug stays hidden. This
    test injects a document with BOTH source_url and a non-null
    scrape_config so the json.loads() call on page load is actually
    exercised, then clicks 'Run Check' (patching check_url_updates to
    avoid real network I/O) to exercise the datetime.now() call too.
    """
    from models import DocumentMeta

    label = "🌐 URL Monitoring"
    at = _run_page(tmp_path, monkeypatch, label)
    assert list(at.exception) == [], f"initial boot raised: {list(at.exception)}"

    kb = at.session_state["kb"]
    kb.documents["fake-doc-1"] = DocumentMeta(
        doc_id="fake-doc-1",
        filename="fake.html",
        title="Fake Scraped Page",
        filepath="http://example.invalid/fake.html",
        file_type="html",
        total_pages=None,
        total_chunks=1,
        indexed_at="2024-01-01T00:00:00",
        tags=[],
        source_url="http://example.invalid/fake.html",
        scrape_config='{"base_url": "http://example.invalid"}',
    )

    # Re-render (same AppTest instance, so session_state incl. our injected
    # doc and the kb object persist) with the url-sourced document present:
    # this alone must hit the json.loads() call on page load (previously
    # UnboundLocalError).
    at.run()
    assert list(at.exception) == [], (
        f"rendering with a source_url+scrape_config doc raised: {list(at.exception)}"
    )

    # Now drive the 'Run Update Check' button branch, which reaches the
    # datetime.now() call (previously also UnboundLocalError). Patch
    # check_url_updates to avoid a real network call.
    kb.check_url_updates = lambda **kwargs: {
        "unchanged": [],
        "changed": [],
        "new_pages": [],
        "missing_pages": [],
        "failed": [],
        "scrape_sessions": [
            {
                "base_url": "http://example.invalid",
                "docs_count": 1,
                "unchanged": 1,
                "changed": 0,
            }
        ],
    }

    run_buttons = [b for b in at.button if b.label == "▶️ Run Check"]
    assert run_buttons, "Run Check button not found"
    run_buttons[0].click().run()
    assert list(at.exception) == [], f"Run Check click raised: {list(at.exception)}"
    assert at.session_state["last_url_check"] is not None


def test_non_vacuous_a_broken_page_is_caught(tmp_path, monkeypatch):
    """Demonstrate the harness actually detects a broken page.

    Temporarily replaces one page's render() with a function that raises,
    confirms AppTest surfaces that as at.exception, then restores the
    original so it never leaks into other tests.
    """
    label = "📊 Dashboard"
    original = admin_pages.PAGES[label]

    def _boom(kb):
        raise RuntimeError("intentional failure injected by test_non_vacuous_a_broken_page_is_caught")

    admin_pages.PAGES[label] = _boom
    try:
        at = _run_page(tmp_path, monkeypatch, label)
        exceptions = list(at.exception)
        assert len(exceptions) == 1
        assert "intentional failure injected" in str(exceptions[0].value)
    finally:
        admin_pages.PAGES[label] = original
        assert admin_pages.PAGES[label] is original
