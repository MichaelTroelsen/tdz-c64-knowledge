"""Re-pointing a document at a relocated source file.

uploads/ is permanent storage (docs/ARCHITECTURE.md, Data Storage Model), so a
document whose filepath stops resolving is a broken record. These tests cover
the repair path and, as much as the repair itself, the two ways it could do
harm: binding a document to the wrong file, or accepting a path that
add_document would have refused.
"""


import pytest

from models import DocumentNotFoundError, KnowledgeBaseError, SecurityError


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A KnowledgeBase on an isolated data dir - never the live one."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    from server import KnowledgeBase
    instance = KnowledgeBase(str(tmp_path))
    yield instance
    instance.close()


@pytest.fixture
def uploads(tmp_path):
    d = tmp_path / "uploads"
    d.mkdir(exist_ok=True)
    return d


def _add_text_doc(kb, uploads, name="vic2.txt", body=None):
    body = body or ("The VIC-II video chip drives sprites and raster interrupts "
                    "on the Commodore 64.\n" * 8)
    src = uploads / name
    src.write_text(body, encoding="utf-8")
    return kb.add_document(str(src)), src


def test_repoint_updates_filepath_in_memory_and_in_the_database(kb, uploads):
    doc, src = _add_text_doc(kb, uploads)
    moved = uploads / "moved" / "vic2.txt"
    moved.parent.mkdir(exist_ok=True)
    src.rename(moved)

    result = kb.repoint_document(doc.doc_id, str(moved))

    assert result["hash_verified"] is True
    assert kb.documents[doc.doc_id].filepath == str(moved.resolve())
    row = kb.db_conn.cursor().execute(
        "SELECT filepath FROM documents WHERE doc_id = ?", (doc.doc_id,)
    ).fetchone()
    assert row[0] == str(moved.resolve()), "the database still holds the old path"


def test_repoint_is_refused_outside_the_allowed_directories(kb, uploads, tmp_path):
    """The same whitelist that gates add_document must gate this."""
    doc, src = _add_text_doc(kb, uploads)
    outside = tmp_path.parent / "outside-the-whitelist.txt"
    outside.write_text("anything", encoding="utf-8")
    try:
        with pytest.raises(SecurityError):
            kb.repoint_document(doc.doc_id, str(outside))
        # and the record is untouched
        assert kb.documents[doc.doc_id].filepath == str(src.resolve())
    finally:
        outside.unlink()


def test_repoint_refuses_a_different_file(kb, uploads):
    """Binding a document to the wrong file is worse than a missing path.

    This is the failure the hash check exists to prevent, and it is silent
    without it: both files exist, both are inside the whitelist, and the only
    thing distinguishing them is their content.
    """
    doc, src = _add_text_doc(kb, uploads)
    impostor = uploads / "sid.txt"
    impostor.write_text("The SID 6581 is the sound chip.\n" * 8, encoding="utf-8")
    src.unlink()

    with pytest.raises(KnowledgeBaseError) as exc:
        kb.repoint_document(doc.doc_id, str(impostor))
    assert "Content mismatch" in str(exc.value)
    assert kb.documents[doc.doc_id].filepath == str(src.resolve()), "record was mutated anyway"


def test_force_accepts_a_mismatch_and_says_so(kb, uploads):
    doc, src = _add_text_doc(kb, uploads)
    impostor = uploads / "sid.txt"
    impostor.write_text("The SID 6581 is the sound chip.\n" * 8, encoding="utf-8")
    src.unlink()

    result = kb.repoint_document(doc.doc_id, str(impostor), force=True)
    assert result["forced"] is True
    assert result["hash_verified"] is False
    assert kb.documents[doc.doc_id].filepath == str(impostor.resolve())


def test_repoint_rejects_a_path_that_is_not_a_file(kb, uploads):
    doc, _ = _add_text_doc(kb, uploads)
    with pytest.raises(KnowledgeBaseError) as exc:
        kb.repoint_document(doc.doc_id, str(uploads / "does-not-exist.txt"))
    assert "not a file on disk" in str(exc.value)


def test_unknown_document_raises(kb, uploads):
    src = uploads / "orphan.txt"
    src.write_text("x", encoding="utf-8")
    with pytest.raises(DocumentNotFoundError):
        kb.repoint_document("deadbeefcafe", str(src))


def test_repoint_drops_missing_source_files_by_exactly_one(kb, uploads):
    """health_check memoises for five minutes, hence use_cache=False."""
    doc_a, src_a = _add_text_doc(kb, uploads, name="a.txt", body="Sprite multiplexing.\n" * 8)
    doc_b, src_b = _add_text_doc(kb, uploads, name="b.txt", body="Raster interrupt timing.\n" * 8)

    moved_a = uploads / "a-moved.txt"
    src_a.rename(moved_a)
    src_b.unlink()   # a second casualty, deliberately left broken

    before = kb.health_check(use_cache=False)["metrics"]["missing_source_files"]
    assert before == 2, f"expected both documents to be missing, got {before}"

    kb.repoint_document(doc_a.doc_id, str(moved_a))

    after = kb.health_check(use_cache=False)["metrics"]["missing_source_files"]
    assert after == before - 1, f"expected exactly one repair, went {before} -> {after}"


def test_repoint_makes_figure_extraction_reach_the_file(kb, uploads, monkeypatch):
    """extract_document_figures skips a PDF whose source is gone.

    figure_ocr_available() is stubbed True so the filepath guard is actually
    reached. Without the stub this test is vacuous on any machine without
    Tesseract: the method raises at the OCR gate and never looks at the path,
    so the very thing being asserted never executes. The assertion is that the
    'no longer on disk' skip stops firing - not that OCR produces text, which
    is a different subsystem.
    """
    fitz = pytest.importorskip("fitz")

    pdf_path = uploads / "schematic.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "VIC-II pinout reference for the C64 mainboard.")
    pdf.save(str(pdf_path))
    pdf.close()

    meta = kb.add_document(str(pdf_path))
    assert meta.file_type == "pdf"

    monkeypatch.setattr(type(kb), "figure_ocr_available", lambda self: (True, ""))

    moved = uploads / "schematics" / "schematic.pdf"
    moved.parent.mkdir(exist_ok=True)
    pdf_path.rename(moved)

    before = kb.extract_document_figures(meta.doc_id)
    assert before["status"] == "skipped"
    assert "no longer on disk" in before["reason"]

    kb.repoint_document(meta.doc_id, str(moved))

    after = kb.extract_document_figures(meta.doc_id, force=True)
    assert after["status"] != "skipped" or "no longer on disk" not in after.get("reason", ""), after
