"""Output-safety tests for the static wiki exporter (R16 in CODE-REVIEW.md).

The exported wiki is built from scraped content, so a document's recorded
source_url is attacker-influenced: a `javascript:` URL contains no HTML
metacharacters at all, so html.escape() passes it through untouched and it
lands in the generated page as a live script-executing link.
"""
import inspect
import re

import pytest

from wiki_export import WikiExporter, safe_external_url, url_query_value


@pytest.mark.parametrize("blocked", [
    "javascript:alert(document.cookie)",
    "JavaScript:alert(1)",              # scheme matching must be case-insensitive
    "  javascript:alert(1)  ",          # ...and must not be defeated by padding
    "data:text/html,<script>alert(1)</script>",
    "vbscript:msgbox(1)",
    "file:///etc/passwd",
])
def test_unsafe_url_schemes_are_dropped(blocked):
    assert safe_external_url(blocked) == "", f"{blocked!r} survived sanitisation"


@pytest.mark.parametrize("allowed", [
    "http://example.com",
    "https://example.com/a/b?c=d#e",
    "mailto:someone@example.com",
])
def test_safe_url_schemes_are_preserved(allowed):
    assert safe_external_url(allowed) == allowed


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_empty_urls_are_dropped(empty):
    assert safe_external_url(empty) == ""


def test_attribute_breakout_is_escaped():
    """Even an http URL must not be able to close the href and add handlers."""
    out = safe_external_url('https://x.com/" onmouseover="alert(1)')
    assert '"' not in out
    assert "&quot;" in out


def test_query_values_are_percent_encoded():
    """viewer.html params are interpolated into a URL, not just HTML text."""
    assert url_query_value('docs/a b&c=d.pdf') == 'docs%2Fa%20b%26c%3Dd.pdf'
    out = url_query_value('x" onload="alert(1)')
    assert '"' not in out and ' ' not in out


def test_doc_page_never_interpolates_source_url_unsanitised():
    """Guard the call site, not just the helper.

    _generate_doc_html previously did html.escape(doc["source_url"]) inline,
    which is exactly the bug: escaping is not scheme validation.
    """
    src = inspect.getsource(WikiExporter._generate_doc_html)
    assert 'html.escape(doc["source_url"])' not in src
    assert "html.escape(doc['source_url'])" not in src
    assert 'safe_source_url' in src, "expected the sanitised value to be used"


def _exporter_source() -> str:
    """Every line of source that makes up the exporter.

    WikiExporter's page methods live in wikigen mixins since the split, so
    inspect.getsource(WikiExporter) on its own returns just the small class
    body. These scans would then pass by finding nothing at all, which is
    worse than failing - hence the MRO walk and the non-vacuity assertions
    in the callers.
    """
    return chr(10).join(
        inspect.getsource(klass) for klass in WikiExporter.__mro__ if klass is not object
    )


def test_no_external_script_srcs_remain_in_generated_html():
    """R16: every third-party script must be vendored, or the page breaks offline."""
    src = _exporter_source()
    external = re.findall(r'<script src="(https?://[^"]+)"', src)
    assert not external, f"generated HTML still loads scripts from a CDN: {external}"


def test_every_vendored_library_has_a_download_entry():
    """A lib/<name> reference with no _JS_LIBRARIES entry is a 404 in the export."""
    src = _exporter_source()
    referenced = set(re.findall(r'<script src="(?:\.\./)?lib/([^"]+)"', src))
    declared = {filename for _, _, filename, _ in WikiExporter._JS_LIBRARIES}
    assert referenced, "scan found no lib/ references at all - it has gone blind"
    missing = referenced - declared
    assert not missing, f"referenced but never downloaded: {missing}"


def test_generated_js_has_no_control_characters():
    """Regression: Python read `\\b` in a non-raw literal as backspace (0x08).

    The emitted JS then carried a literal control character where a regex
    word-boundary assertion was intended, silently breaking BASIC REM and
    assembly number syntax highlighting in every exported wiki.
    """
    # The JS used to be a triple-quoted literal inside _create_javascript, so
    # this test parsed the method's source and eval'd the literal back. It now
    # ships as wiki_assets/js/enhancements.js and is read verbatim, which is
    # what the export actually writes - so assert against that instead of
    # against source text.
    emitted = WikiExporter._read_asset("js", "enhancements.js")

    control = {c for c in emitted if ord(c) < 32 and c not in '\n\r\t'}
    assert not control, (
        f"emitted JS contains control characters {[hex(ord(c)) for c in control]} - "
        "an unescaped backslash sequence was interpreted by Python instead of "
        "being passed through to JavaScript"
    )
    assert r'/\bREM\s+.*/gi' in emitted, "BASIC REM word-boundary regex regressed"
