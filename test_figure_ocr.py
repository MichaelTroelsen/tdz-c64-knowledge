"""Tests for the background figure-OCR batch pass.

Motivation: the ingest-time OCR path only fires for PDFs detected as
*entirely* scanned, so a normal text PDF gets its text layer indexed while its
embedded figures - which in C64 documentation is often where the actual
reference data lives (memory maps, register tables, pinouts) - were never read
at all.

Only the Tesseract call itself is stubbed (the binary is not installed in CI).
Everything else runs for real: a genuine PDF is built on disk, images are
pulled out of it with PyMuPDF, rows are written to SQLite, and search runs
through the real FTS5 index.
"""
import os

import pytest

import server as server_module

fitz = pytest.importorskip("fitz", reason="PyMuPDF required for figure OCR")
PIL_Image = pytest.importorskip("PIL.Image", reason="Pillow required for figure OCR")

from PIL import Image, ImageDraw  # noqa: E402


# Text the fake OCR engine will "read" out of each figure, keyed by size so
# different figures can return different content.
FAKE_OCR_TEXT = {}


def _make_figure_png(path, size, label):
    """A plain image with a label drawn on it, big enough to pass the size gate."""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, size[0] - 5, size[1] - 5], outline='black', width=2)
    draw.text((12, 12), label, fill='black')
    img.save(path)
    return path


def _make_pdf_with_figures(pdf_path, figure_specs):
    """Build a real PDF whose pages carry embedded raster images.

    figure_specs: list of (page_index, size, label)
    """
    doc = fitz.open()
    pages_needed = max(spec[0] for spec in figure_specs) + 1
    # The body text must differ per file: add_document rejects a second
    # document whose content hash matches one already indexed, so PDFs with
    # identical text collapse into a single document and any test that needs
    # two distinct documents would silently be testing one.
    unique = os.path.splitext(os.path.basename(pdf_path))[0]
    for page_index in range(pages_needed):
        page = doc.new_page()
        # A text layer, so this is NOT detected as a scanned PDF - which is
        # exactly the case ingest-time OCR skips.
        page.insert_text(
            (72, 72),
            f"VIC-II register reference for {unique} page {page_index}. See figure below."
        )

    tmpdir = os.path.dirname(pdf_path)
    for i, (page_index, size, label) in enumerate(figure_specs):
        png = _make_figure_png(os.path.join(tmpdir, f"_fig{i}.png"), size, label)
        rect = fitz.Rect(72, 120, 72 + size[0], 120 + size[1])
        doc[page_index].insert_image(rect, filename=png)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def figure_kb(tmp_path, monkeypatch):
    """A KB with figure OCR forced available and Tesseract stubbed out."""
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "0")

    kb = server_module.KnowledgeBase(str(tmp_path / "data"))
    # Tesseract itself is not installed in CI; force the capability on and
    # stub the single call that needs the binary.
    kb.use_ocr = True

    def fake_image_to_string(image, *a, **kw):
        return FAKE_OCR_TEXT.get(image.size, "")

    monkeypatch.setattr(server_module.pytesseract, "image_to_string", fake_image_to_string)

    try:
        yield kb, docs
    finally:
        kb.close()


def test_figures_are_extracted_and_ocrd_from_a_text_pdf(figure_kb, tmp_path):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "SPRITE COLLISION REGISTER $D01E"

    pdf = _make_pdf_with_figures(str(docs / "vic.pdf"), [(0, (400, 300), "fig1")])
    doc = kb.add_document(pdf)

    result = kb.extract_document_figures(doc.doc_id)

    assert result['status'] == 'completed', result
    assert result['figures_found'] == 1, result
    assert result['figures_with_text'] == 1, result

    figures = kb.get_document_figures(doc.doc_id)
    assert len(figures) == 1
    assert figures[0]['page_number'] == 1
    assert "SPRITE COLLISION" in figures[0]['ocr_text']
    assert os.path.exists(figures[0]['image_path']), "extracted image was not saved"


def test_tiny_decorative_images_are_skipped(figure_kb):
    """Rules, bullets and logo fragments are never figures with readable text."""
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "REAL FIGURE TEXT"
    FAKE_OCR_TEXT[(20, 20)] = "noise"

    pdf = _make_pdf_with_figures(str(docs / "mixed.pdf"), [
        (0, (400, 300), "real"),
        (0, (20, 20), "tiny"),
    ])
    doc = kb.add_document(pdf)

    result = kb.extract_document_figures(doc.doc_id)

    assert result['figures_found'] == 1, f"tiny image was not filtered out: {result}"
    assert result['figures_skipped_small'] >= 1, result
    texts = [f['ocr_text'] for f in kb.get_document_figures(doc.doc_id)]
    assert not any(t and 'noise' in t for t in texts)


def test_figures_with_no_readable_text_are_recorded_but_left_empty(figure_kb):
    """A row still proves we looked, but empty text must not pollute search."""
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "   \n  \n "  # whitespace only

    pdf = _make_pdf_with_figures(str(docs / "blank.pdf"), [(0, (400, 300), "blank")])
    doc = kb.add_document(pdf)

    result = kb.extract_document_figures(doc.doc_id)
    assert result['figures_found'] == 1
    assert result['figures_with_text'] == 0

    figures = kb.get_document_figures(doc.doc_id)
    assert len(figures) == 1
    assert not figures[0]['ocr_text']
    assert kb.get_document_figures(doc.doc_id, with_text_only=True) == []


def test_search_finds_text_that_exists_only_inside_a_figure(figure_kb):
    """The whole point: content invisible to normal document search."""
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(500, 400)] = "RASTER IRQ latch at $D012 controls the scanline compare"

    pdf = _make_pdf_with_figures(str(docs / "raster.pdf"), [(0, (500, 400), "raster")])
    doc = kb.add_document(pdf)
    kb.extract_document_figures(doc.doc_id)

    # The phrase exists only in the image, never in the PDF's text layer.
    assert not kb.search("scanline"), "fixture is wrong - term leaked into document text"

    hits = kb.search_figures("scanline")
    assert hits, "figure OCR text is not searchable"
    assert hits[0]['doc_id'] == doc.doc_id
    assert hits[0]['page_number'] == 1
    assert 'scanline' in hits[0]['snippet'].lower()


def test_search_can_be_scoped_to_one_document(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "shared keyword sprite"

    a = kb.add_document(_make_pdf_with_figures(str(docs / "a.pdf"), [(0, (400, 300), "a")]))
    b = kb.add_document(_make_pdf_with_figures(str(docs / "b.pdf"), [(0, (400, 300), "b")]))
    kb.extract_document_figures(a.doc_id)
    kb.extract_document_figures(b.doc_id)

    assert len(kb.search_figures("sprite")) == 2
    scoped = kb.search_figures("sprite", doc_id=a.doc_id)
    assert len(scoped) == 1 and scoped[0]['doc_id'] == a.doc_id


def test_reextraction_is_skipped_unless_forced(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "first pass text"

    pdf = _make_pdf_with_figures(str(docs / "again.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)
    kb.extract_document_figures(doc.doc_id)

    again = kb.extract_document_figures(doc.doc_id)
    assert again['status'] == 'skipped', again

    FAKE_OCR_TEXT[(400, 300)] = "second pass text"
    forced = kb.extract_document_figures(doc.doc_id, force=True)
    assert forced['status'] == 'completed', forced

    figures = kb.get_document_figures(doc.doc_id)
    assert len(figures) == 1, "force must replace rows, not duplicate them"
    assert "second pass" in figures[0]['ocr_text']


def test_non_pdf_documents_are_skipped_cleanly(figure_kb):
    kb, docs = figure_kb
    txt = docs / "notes.txt"
    txt.write_text("Plain text notes about the SID chip.", encoding="utf-8")
    doc = kb.add_document(str(txt))

    result = kb.extract_document_figures(doc.doc_id)
    assert result['status'] == 'skipped'
    assert 'not a PDF' in result['reason']


def test_unknown_document_raises(figure_kb):
    kb, _ = figure_kb
    with pytest.raises(server_module.DocumentNotFoundError):
        kb.extract_document_figures("does-not-exist")


def test_missing_source_file_is_reported_not_crashed(figure_kb):
    kb, docs = figure_kb
    pdf = _make_pdf_with_figures(str(docs / "gone.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)
    os.remove(pdf)

    result = kb.extract_document_figures(doc.doc_id)
    assert result['status'] == 'skipped'
    assert 'no longer on disk' in result['reason']


# --- background queue integration ---


def test_queueing_creates_a_figures_typed_job(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    pdf = _make_pdf_with_figures(str(docs / "queued.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)

    result = kb.queue_figure_ocr(doc.doc_id)
    assert result['queued'] is True

    row = kb.db_conn.execute(
        "SELECT job_type, status FROM extraction_jobs WHERE job_id = ?",
        (result['job_id'],)
    ).fetchone()
    assert row == ('figures', 'queued') or row[0] == 'figures'


def test_a_pending_figure_job_does_not_block_entity_extraction(figure_kb):
    """Both job kinds share one table; their dedup checks must not collide."""
    kb, docs = figure_kb
    pdf = _make_pdf_with_figures(str(docs / "both.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)

    kb.queue_figure_ocr(doc.doc_id)
    entity_result = kb.queue_entity_extraction(doc.doc_id)

    assert entity_result.get('queued') is True, (
        f"a pending figures job was mistaken for pending entity work: {entity_result}"
    )


def test_extraction_status_reports_the_job_type(figure_kb):
    kb, docs = figure_kb
    pdf = _make_pdf_with_figures(str(docs / "typed.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)
    kb.queue_figure_ocr(doc.doc_id)

    status = kb.get_extraction_status(doc.doc_id)
    assert any(j.get('job_type') == 'figures' for j in status['jobs']), status['jobs']


def test_worker_processes_a_queued_figure_job_end_to_end(figure_kb):
    """The real worker thread must actually drain a 'figures' job."""
    import time

    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(450, 320)] = "BORDER COLOUR REGISTER $D020"

    pdf = _make_pdf_with_figures(str(docs / "worker.pdf"), [(0, (450, 320), "x")])
    doc = kb.add_document(pdf)

    queued = kb.queue_figure_ocr(doc.doc_id)
    assert queued['queued'] is True

    deadline = time.time() + 60
    while time.time() < deadline:
        row = kb.db_conn.execute(
            "SELECT status FROM extraction_jobs WHERE job_id = ?", (queued['job_id'],)
        ).fetchone()
        if row and row[0] in ('completed', 'failed'):
            break
        time.sleep(0.1)
    else:
        pytest.fail("figure OCR job was never drained by the worker")

    assert row[0] == 'completed', (
        kb.db_conn.execute(
            "SELECT error_message FROM extraction_jobs WHERE job_id = ?", (queued['job_id'],)
        ).fetchone()
    )
    hits = kb.search_figures("BORDER")
    assert hits and hits[0]['doc_id'] == doc.doc_id


def test_batch_queues_every_pdf_but_not_other_file_types(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()

    for name in ("one.pdf", "two.pdf"):
        kb.add_document(_make_pdf_with_figures(str(docs / name), [(0, (400, 300), "x")]))
    txt = docs / "three.txt"
    txt.write_text("not a pdf", encoding="utf-8")
    kb.add_document(str(txt))

    result = kb.queue_figure_ocr_all()
    assert result['pdf_documents'] == 2, result
    assert result['queued'] == 2, result


def test_batch_respects_a_limit(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        kb.add_document(_make_pdf_with_figures(str(docs / name), [(0, (400, 300), "x")]))

    result = kb.queue_figure_ocr_all(limit=2)
    assert result['queued'] == 2, result


def test_coverage_reporting(figure_kb):
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "some register text"

    a = kb.add_document(_make_pdf_with_figures(str(docs / "cov1.pdf"), [(0, (400, 300), "x")]))
    kb.add_document(_make_pdf_with_figures(str(docs / "cov2.pdf"), [(0, (400, 300), "y")]))

    before = kb.get_figure_ocr_coverage()
    assert before['pdf_documents'] == 2
    assert before['documents_processed'] == 0
    assert before['documents_remaining'] == 2

    kb.extract_document_figures(a.doc_id)

    after = kb.get_figure_ocr_coverage()
    assert after['documents_processed'] == 1
    assert after['documents_remaining'] == 1
    assert after['figures_extracted'] == 1
    assert after['figures_with_text'] == 1


def test_removing_a_document_removes_its_figures(figure_kb):
    """FK cascade: orphaned figure rows would leak into every search."""
    kb, docs = figure_kb
    FAKE_OCR_TEXT.clear()
    FAKE_OCR_TEXT[(400, 300)] = "doomed figure text"

    pdf = _make_pdf_with_figures(str(docs / "doomed.pdf"), [(0, (400, 300), "x")])
    doc = kb.add_document(pdf)
    kb.extract_document_figures(doc.doc_id)
    assert kb.search_figures("doomed")

    kb.remove_document(doc.doc_id)

    left = kb.db_conn.execute(
        "SELECT COUNT(*) FROM document_figures WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()[0]
    assert left == 0, "figure rows outlived their document"
    assert not kb.search_figures("doomed"), "deleted figure text is still searchable"


def test_availability_is_reported_rather_than_crashing(figure_kb, monkeypatch):
    kb, _ = figure_kb

    kb.use_ocr = False
    ok, reason = kb.figure_ocr_available()
    assert ok is False and 'OCR' in reason

    with pytest.raises(server_module.KnowledgeBaseError, match="Figure OCR unavailable"):
        kb.extract_document_figures("whatever")
