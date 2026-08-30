"""Cross-format doc_id dedup: a PDF and a .txt of the same content must share an id.

The defect this pins: `_extract_pdf_text` joins pages with PDF_PAGE_BREAK, which
injects four words per page boundary into text that is otherwise identical to the
same document ingested as .txt. `_generate_doc_id` hashed those invented words, so
the two formats produced different ids and `add_document`'s duplicate-content
short-circuit could never fire across formats.

Two properties are tested, and the second matters as much as the first:
  1. marker-bearing and marker-free renderings of the same content hash EQUAL;
  2. text containing NO marker hashes exactly as it did before the fix - because
     976 non-PDF documents in the live corpus are stored under ids computed by the
     old algorithm, and any change there would orphan every one of them.
"""
import hashlib
import logging

import pytest

from kb.core import CoreMixin
from text_utils import PDF_PAGE_BREAK


PAGES = [
    "VIC-II sprite zero X coordinate lives at 53248 decimal.",
    "The SID filter has low-pass, band-pass and high-pass modes.",
    "CIA timer A underflow can chain into timer B for long periods.",
]


class _Hasher(CoreMixin):
    """CoreMixin is a mixin over KnowledgeBase; _generate_doc_id needs no state."""

    logger = logging.getLogger("test_doc_id_dedup")

    def __init__(self):  # deliberately does not call CoreMixin.__init__
        pass


@pytest.fixture
def hasher():
    return _Hasher()


def _as_plain_text(pages):
    """What _extract_text_file yields for a .txt of the same document."""
    return "\n\n".join(pages)


def _as_pdf_text(pages):
    """What _extract_pdf_text yields - pages joined with the injected marker."""
    return f"\n\n{PDF_PAGE_BREAK}\n\n".join(pages)


def _pre_fix_doc_id(text):
    """The exact algorithm that produced every doc_id currently in the live DB."""
    normalized = text.lower().strip()
    sample = " ".join(normalized.split()[:10000])
    return hashlib.md5(sample.encode("utf-8")).hexdigest()[:12]


def test_pdf_and_txt_of_same_content_share_a_doc_id(hasher):
    txt_id = hasher._generate_doc_id("issue.txt", _as_plain_text(PAGES))
    pdf_id = hasher._generate_doc_id("issue.pdf", _as_pdf_text(PAGES))
    assert txt_id == pdf_id, (
        "a PDF and a .txt of identical content must dedupe; the extractor's "
        f"page-break markers are leaking into the hash again ({txt_id} != {pdf_id})"
    )


def test_marker_free_text_hashes_exactly_as_before_the_fix(hasher):
    """The 976 non-PDF documents in the live corpus must keep their stored ids."""
    plain = _as_plain_text(PAGES)
    assert PDF_PAGE_BREAK not in plain
    assert hasher._generate_doc_id("issue.txt", plain) == _pre_fix_doc_id(plain)


def test_the_fix_is_what_changed_the_pdf_id(hasher):
    """Guards the claim, not just the outcome: the PDF id moved, the .txt id did not."""
    pdf_text = _as_pdf_text(PAGES)
    assert hasher._generate_doc_id("issue.pdf", pdf_text) != _pre_fix_doc_id(pdf_text)


def test_many_page_breaks_do_not_shift_the_hash(hasher):
    """Page count must not affect identity - a re-scan may paginate differently."""
    three = hasher._generate_doc_id("a.pdf", _as_pdf_text(PAGES))
    one_page = hasher._generate_doc_id("b.pdf", _as_plain_text(PAGES))
    assert three == one_page


def test_genuinely_different_content_still_differs(hasher):
    """The fix must not collapse distinct documents into one id."""
    other = list(PAGES)
    other[1] = "The 6510 has an on-chip I/O port at addresses zero and one."
    assert hasher._generate_doc_id("a.pdf", _as_pdf_text(PAGES)) != \
        hasher._generate_doc_id("b.pdf", _as_pdf_text(other))


def test_filepath_fallback_is_untouched(hasher):
    """No text_content still means the legacy filepath hash."""
    expected = hashlib.md5(b"some/path.pdf").hexdigest()[:12]
    assert hasher._generate_doc_id("some/path.pdf") == expected


def test_marker_constant_is_what_the_extractor_actually_injects():
    """Couples the hash's assumption to the producer, so a reworded marker fails here."""
    import kb.ingest._extraction as extraction

    src = open(extraction.__file__, encoding="utf-8").read()
    assert "PDF_PAGE_BREAK" in src, (
        "the extractor no longer references the shared constant - if it went back to "
        "a hard-coded literal, _generate_doc_id will silently stop stripping it"
    )
