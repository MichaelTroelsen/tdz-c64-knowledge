#!/usr/bin/env python3
"""
Regression tests for scrape_url/rescrape_document reliability fixes.

Background, found during an audit of the MCP server's reliability (the same
investigation that found and fixed issue #12's search hang):

  1. scrape_url's monitor loop had no real wall-clock deadline. It only
     LOGGED a warning after 60s of no progress and never terminated the
     process; the sole timeout guard, process.wait(timeout=60), ran AFTER
     the loop already exited, making it a no-op for a genuinely stalled
     process. A hung mdscrape run blocked the call (and, per issue #12,
     the whole asyncio event loop) indefinitely.
  2. max_pages was accepted as a parameter but never enforced - mdscrape
     itself doesn't support it (an explicit code comment said so), and
     nothing in Python enforced it either, so a crawl "capped" at N pages
     was actually unbounded.
  3. rescrape_document() removed the existing document BEFORE attempting
     the re-scrape, with no rollback. A dead/renamed page, or the failure
     modes above, permanently destroyed the only copy of the content.
     depth also silently fell back to 50 - much deeper than scrape_url's
     own default of 3 - for any document scraped before that field existed.

Run with:  pytest test_scrape_reliability.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from server import KnowledgeBase

REPO = Path(__file__).parent
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def kb(temp_data_dir):
    original = {k: os.environ.get(k) for k in ('ALLOWED_DOCS_DIRS', 'AUTO_EXTRACT_ENTITIES')}
    os.environ['ALLOWED_DOCS_DIRS'] = temp_data_dir
    os.environ['AUTO_EXTRACT_ENTITIES'] = '0'

    kb_instance = KnowledgeBase(temp_data_dir)
    yield kb_instance
    kb_instance.close()

    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _fake_popen_running(fake_code):
    """Build a subprocess.Popen replacement that ignores the real mdscrape
    argv and runs `fake_code` under this venv's python instead - a real
    Popen object, so scrape_url's .stdout/.poll()/.terminate()/.wait()
    calls all work exactly as they would against the real subprocess.
    """
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        return real_popen([PYTHON, "-c", fake_code], **kwargs)

    return fake_popen


def test_scrape_url_enforces_wall_clock_timeout(kb, monkeypatch):
    """A stalled mdscrape process must be killed, not block forever.

    Regression: previously nothing in the monitor loop ever called
    terminate()/kill() - a process that just sat there blocked scrape_url
    (and the whole event loop, since nothing here runs in a thread pool)
    indefinitely. mdscrape is replaced with a process that sleeps for 5
    minutes and never emits a single progress line.
    """
    monkeypatch.setenv('SCRAPE_TIMEOUT_S', '2')
    monkeypatch.setattr(kb, '_find_mdscrape_executable', lambda: 'fake-mdscrape')
    monkeypatch.setattr(subprocess, 'Popen', _fake_popen_running("import time; time.sleep(300)"))

    start = time.time()
    result = kb.scrape_url("http://example.invalid/", max_pages=5, threads=1, delay=100)
    elapsed = time.time() - start

    assert elapsed < 30, (
        f"scrape_url blocked for {elapsed:.1f}s - the 2s SCRAPE_TIMEOUT_S was not enforced"
    )
    assert result['stop_reason'] == 'timeout', result
    assert result['status'] in ('partial', 'failed'), result


def test_scrape_url_enforces_max_pages_cap(kb, monkeypatch):
    """max_pages must actually stop the crawl - mdscrape itself doesn't
    support it (the real cap has to be enforced on the Python side), and
    previously nothing did: a "cap" of 5 pages let the crawl run unbounded.

    mdscrape is replaced with a process that emits progress lines for 1000
    pages in rapid succession and would otherwise run for a long time.
    """
    monkeypatch.setattr(kb, '_find_mdscrape_executable', lambda: 'fake-mdscrape')
    fake_code = (
        "import sys, time\n"
        "for i in range(1000):\n"
        "    print(f'Scraping: http://example.invalid/page{i}', flush=True)\n"
        "    time.sleep(0.02)\n"
    )
    monkeypatch.setattr(subprocess, 'Popen', _fake_popen_running(fake_code))

    start = time.time()
    result = kb.scrape_url("http://example.invalid/", max_pages=5, threads=1, delay=100)
    elapsed = time.time() - start

    assert elapsed < 15, (
        f"scrape_url took {elapsed:.1f}s - max_pages=5 should have stopped the "
        "1000-page fake crawl almost immediately"
    )
    assert result['stop_reason'] == 'max_pages', result


def test_rescrape_keeps_original_when_scrape_fails(kb, monkeypatch, temp_data_dir):
    """rescrape_document must not destroy the original on a failed re-scrape.

    Regression: remove_document() ran BEFORE the scrape attempt, with no
    rollback, so a dead page or a failed scrape permanently lost the only
    copy of the content.
    """
    doc_path = Path(temp_data_dir) / "original.txt"
    doc_path.write_text("Original content about the 6502 addressing modes.")
    doc = kb.add_document(str(doc_path), title="Original Page")
    kb.documents[doc.doc_id].source_url = 'http://example.invalid/original'

    monkeypatch.setattr(kb, 'scrape_url', lambda **kw: {
        'status': 'failed', 'url': kw.get('url'), 'doc_ids': [], 'error': 'site unreachable'
    })

    result = kb.rescrape_document(doc.doc_id)

    assert result['old_doc_kept'] is True
    assert doc.doc_id in kb.documents, "original document was removed despite the re-scrape failing"
    assert kb.get_document(doc.doc_id) is not None, "original document's content was lost"


def test_rescrape_keeps_original_when_dedup_matches(kb, monkeypatch, temp_data_dir):
    """If the re-scrape finds byte-identical content, nothing should be
    removed - add_document's content-hash dedup already returned the SAME
    doc_id, so removing "the old one" would delete the only copy that exists.
    """
    doc_path = Path(temp_data_dir) / "original.txt"
    doc_path.write_text("Unchanged page content about VIC-II sprites.")
    doc = kb.add_document(str(doc_path), title="Unchanged Page")
    kb.documents[doc.doc_id].source_url = 'http://example.invalid/unchanged'

    monkeypatch.setattr(kb, 'scrape_url', lambda **kw: {
        'status': 'success', 'url': kw.get('url'), 'doc_ids': [doc.doc_id], 'docs_added': 0
    })

    result = kb.rescrape_document(doc.doc_id)

    assert result['old_doc_kept'] is True
    assert doc.doc_id in kb.documents


def test_rescrape_removes_old_only_after_successful_new_scrape(kb, monkeypatch, temp_data_dir):
    """Once a re-scrape genuinely succeeds with new/different content, the
    old version should be retired - confirming the fix doesn't just make
    rescrape_document permanently keep stale content either.
    """
    old_path = Path(temp_data_dir) / "old.txt"
    old_path.write_text("Old page content, version one.")
    old_doc = kb.add_document(str(old_path), title="Page")
    kb.documents[old_doc.doc_id].source_url = 'http://example.invalid/page'

    new_path = Path(temp_data_dir) / "new.txt"
    new_path.write_text("New page content, version two - substantially different.")
    new_doc = kb.add_document(str(new_path), title="Page (rescraped)")

    monkeypatch.setattr(kb, 'scrape_url', lambda **kw: {
        'status': 'success', 'url': kw.get('url'), 'doc_ids': [new_doc.doc_id], 'docs_added': 1
    })

    result = kb.rescrape_document(old_doc.doc_id)

    assert result['old_doc_kept'] is False
    assert old_doc.doc_id not in kb.documents, "superseded document was not retired"
    assert new_doc.doc_id in kb.documents
