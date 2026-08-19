"""Embedded-figure extraction and OCR.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timezone
from features import FIGURE_SUPPORT
from features import OCR_SUPPORT
from features import pytesseract
from models import DocumentNotFoundError
from models import KnowledgeBaseError
from pathlib import Path
from text_utils import _content_terms
from typing import Optional
import os
import sqlite3


class FiguresMixin:

    def figure_ocr_available(self) -> tuple[bool, Optional[str]]:
        """Whether figure OCR can run, and if not, why not."""
        if not FIGURE_SUPPORT:
            return False, "PyMuPDF (fitz) is not installed. Install with: pip install PyMuPDF"
        if not OCR_SUPPORT:
            return False, "pytesseract/Pillow are not installed. Install with: pip install pytesseract Pillow"
        if not self.use_ocr:
            # self.use_ocr collapses two distinct causes (kb/core.py): the env
            # var was 0, or Tesseract's binary could not be found. Re-read the
            # env var here to tell them apart rather than repeating whichever
            # one happens to be listed first.
            if os.getenv('USE_OCR', '1') == '0':
                return False, "OCR is disabled (USE_OCR=0)"
            return False, (
                "USE_OCR is enabled but the Tesseract binary was not found"
                + (f" at TESSERACT_PATH={self.tesseract_path!r}" if self.tesseract_path else " on PATH")
                + ". Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        return True, None

    def extract_document_figures(self, doc_id: str, force: bool = False) -> dict:
        """Extract and OCR every embedded figure in one PDF document.

        Returns a summary dict; raises KnowledgeBaseError only for conditions
        the caller can act on (unknown doc, OCR unavailable). Per-figure
        failures are counted, not raised - one unreadable image must not
        abandon the rest of the document.
        """
        ok, reason = self.figure_ocr_available()
        if not ok:
            raise KnowledgeBaseError(f"Figure OCR unavailable: {reason}")

        doc = self.documents.get(doc_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        if (doc.file_type or '').lower() != 'pdf':
            return {
                'doc_id': doc_id, 'status': 'skipped',
                'reason': f"not a PDF (file_type={doc.file_type!r})",
                'figures_found': 0, 'figures_with_text': 0, 'figures_failed': 0,
            }

        if not doc.filepath or not os.path.exists(doc.filepath):
            return {
                'doc_id': doc_id, 'status': 'skipped',
                'reason': f"source file is no longer on disk: {doc.filepath}",
                'figures_found': 0, 'figures_with_text': 0, 'figures_failed': 0,
            }

        cursor = self.db_conn.cursor()
        if not force:
            already = cursor.execute(
                "SELECT COUNT(*) FROM document_figures WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
            if already:
                return {
                    'doc_id': doc_id, 'status': 'skipped',
                    'reason': f"{already} figure(s) already extracted (pass force=True to redo)",
                    'figures_found': already, 'figures_with_text': 0, 'figures_failed': 0,
                }

        import fitz
        from PIL import Image

        figures_dir = self.data_dir / "figures" / doc_id
        figures_dir.mkdir(parents=True, exist_ok=True)

        found = with_text = failed = skipped_small = vector = 0

        # Two passes. PyMuPDF objects are not thread-safe, so every fitz call
        # stays on this thread and only writes PNGs; the OCR pass below is
        # what actually costs time, and that one parallelises freely.
        pending: list[dict] = []

        pdf = fitz.open(doc.filepath)
        try:
            for page_number in range(pdf.page_count):
                page = pdf[page_number]
                images = page.get_images(full=True)

                for image_index, img in enumerate(images):
                    xref = img[0]
                    try:
                        pix = fitz.Pixmap(pdf, xref)

                        if pix.width < self.FIGURE_MIN_WIDTH or pix.height < self.FIGURE_MIN_HEIGHT:
                            skipped_small += 1
                            continue

                        # CMYK/alpha pixmaps can't be handed to PIL directly.
                        if pix.n - pix.alpha >= 4 or pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        found += 1
                        image_path = figures_dir / f"p{page_number + 1:04d}_i{image_index:02d}.png"
                        pix.save(str(image_path))
                        pending.append({
                            'page_number': page_number + 1,
                            'image_index': image_index,
                            'image_path': image_path,
                            'width': pix.width,
                            'height': pix.height,
                            'source': 'embedded',
                        })
                    except Exception as e:
                        failed += 1
                        self.logger.debug(
                            f"Could not extract image {image_index} on page {page_number + 1} "
                            f"of {doc_id}: {e}"
                        )

                if self.FIGURE_RASTERIZE_PAGES:
                    # Numbered after this page's embedded images so the
                    # UNIQUE(doc_id, page_number, image_index) key still holds.
                    drawn, drawn_skipped, drawn_failed = self._extract_vector_figures(
                        page, page_number, images, figures_dir, len(images)
                    )
                    pending.extend(drawn)
                    found += len(drawn)
                    vector += len(drawn)
                    skipped_small += drawn_skipped
                    failed += drawn_failed
        finally:
            pdf.close()

        def _ocr(item: dict) -> dict:
            text = ''
            try:
                with Image.open(item['image_path']) as im:
                    text = pytesseract.image_to_string(im) or ''
            except Exception as e:
                item['ocr_failed'] = True
                self.logger.debug(f"OCR failed for {item['image_path'].name}: {e}")
            item['text'] = text.strip()
            return item

        if self.FIGURE_OCR_WORKERS > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=self.FIGURE_OCR_WORKERS) as pool:
                pending = list(pool.map(_ocr, pending))
        else:
            pending = [_ocr(item) for item in pending]

        rows = []
        extracted_at = datetime.now(timezone.utc).isoformat()
        for item in sorted(pending, key=lambda i: (i['page_number'], i['image_index'])):
            if item.get('ocr_failed'):
                failed += 1

            text = item['text']
            if len(text) < self.FIGURE_MIN_CHARS:
                # Keep the row (it records that we looked) but with no text,
                # so it never pollutes search results.
                text = ''
            else:
                with_text += 1

            rows.append((
                doc_id, item['page_number'], item['image_index'], text or None, len(text),
                str(item['image_path']), item['width'], item['height'],
                extracted_at, item['source'],
            ))

        # force=True clears the previous rows first, so a failure between the
        # DELETE and the INSERT would otherwise leave the document with no
        # figures at all - worse than the state we started from.
        try:
            if force:
                cursor.execute("DELETE FROM document_figures WHERE doc_id = ?", (doc_id,))
            cursor.executemany("""
                INSERT OR REPLACE INTO document_figures
                    (doc_id, page_number, image_index, ocr_text, char_count,
                     image_path, width, height, extracted_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            self.db_conn.commit()
        except Exception:
            self.db_conn.rollback()
            self.logger.exception(f"Storing figures for {doc_id} failed; rolled back")
            raise

        self.logger.info(
            f"Figure OCR for {doc_id}: {found} figure(s) ({vector} rasterized), "
            f"{with_text} with text, {failed} failed, {skipped_small} skipped on size"
        )
        return {
            'doc_id': doc_id,
            'status': 'completed',
            'figures_found': found,
            'figures_with_text': with_text,
            'figures_failed': failed,
            'figures_skipped_small': skipped_small,
            'figures_rasterized': vector,
        }

    def _extract_vector_figures(self, page, page_number: int, images: list,
                                figures_dir: Path, first_index: int) -> tuple[list[dict], int, int]:
        """Render the vector-drawing clusters on one page as figure images.

        Only the clustered drawing regions are rendered, never the whole page:
        a full-page render would OCR the body text a second time and duplicate
        what the PDF's own text layer already contributed to the chunk index.

        Returns (pending items, skipped, failed). Must stay on the thread that
        owns `page` - PyMuPDF objects are not thread-safe.
        """
        import fitz

        pending: list[dict] = []
        skipped = failed = 0

        page_area = abs(page.rect.get_area())
        if not page_area:
            return pending, skipped, failed

        # Where this page's bitmaps sit, so a cluster that merely frames one
        # is not OCR'd a second time as a "drawing".
        image_rects = []
        for img in images:
            try:
                image_rects.extend(page.get_image_rects(img[0]))
            except Exception:
                # A figure we can't locate just loses this dedup check.
                pass

        zoom = self.FIGURE_RASTER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        index = first_index

        for cluster in page.cluster_drawings():
            rect = fitz.Rect(cluster) & page.rect
            if rect.is_empty:
                continue

            area = abs(rect.get_area())
            if area / page_area > self.FIGURE_RASTER_MAX_AREA:
                skipped += 1
                continue

            if any(abs((rect & ir).get_area()) >= 0.5 * area
                   for ir in image_rects if not (rect & ir).is_empty):
                continue

            try:
                pix = page.get_pixmap(matrix=matrix, clip=rect)

                if pix.width < self.FIGURE_MIN_WIDTH or pix.height < self.FIGURE_MIN_HEIGHT:
                    skipped += 1
                    continue

                if pix.n - pix.alpha >= 4 or pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_path = figures_dir / f"p{page_number + 1:04d}_v{index:02d}.png"
                pix.save(str(image_path))
                pending.append({
                    'page_number': page_number + 1,
                    'image_index': index,
                    'image_path': image_path,
                    'width': pix.width,
                    'height': pix.height,
                    'source': 'vector',
                })
                index += 1
            except Exception as e:
                failed += 1
                self.logger.debug(
                    f"Could not rasterize drawing cluster {index} on page "
                    f"{page_number + 1}: {e}"
                )

        return pending, skipped, failed

    def get_document_figures(self, doc_id: str, with_text_only: bool = False) -> list[dict]:
        """Every extracted figure for a document, in page order."""
        sql = """
            SELECT figure_id, page_number, image_index, ocr_text, char_count,
                   image_path, width, height, extracted_at, source
            FROM document_figures WHERE doc_id = ?
        """
        if with_text_only:
            sql += " AND ocr_text IS NOT NULL AND ocr_text != ''"
        sql += " ORDER BY page_number, image_index"

        return [
            {
                'figure_id': r[0], 'page_number': r[1], 'image_index': r[2],
                'ocr_text': r[3], 'char_count': r[4], 'image_path': r[5],
                'width': r[6], 'height': r[7], 'extracted_at': r[8],
                'source': r[9],
            }
            for r in self.db_conn.execute(sql, (doc_id,)).fetchall()
        ]

    def search_figures(self, query: str, max_results: int = 10,
                       doc_id: Optional[str] = None) -> list[dict]:
        """Search OCR'd figure text. Uses FTS5 when present, else LIKE."""
        results = []
        # _extract_snippet expects a set of terms, not the raw query string -
        # passing the string iterates it character by character, scoring
        # windows by letter frequency instead of word matches, and silently
        # disabling highlighting entirely (single characters never clear its
        # own len>=2 threshold).
        query_terms = _content_terms(query)
        use_fts = False
        try:
            use_fts = self.db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='figures_fts5'"
            ).fetchone() is not None
        except sqlite3.Error:
            pass

        if use_fts:
            try:
                sql = """
                    SELECT f.figure_id, f.doc_id, f.page_number, f.image_index,
                           f.ocr_text, f.image_path, bm25(figures_fts5) AS score
                    FROM figures_fts5
                    JOIN document_figures f ON f.figure_id = figures_fts5.figure_id
                    WHERE figures_fts5 MATCH ?
                """
                params: list = [query]
                if doc_id:
                    sql += " AND f.doc_id = ?"
                    params.append(doc_id)
                sql += " ORDER BY score LIMIT ?"
                params.append(max_results)
                rows = self.db_conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as e:
                # A malformed FTS5 query (unbalanced quote, bare operator) is
                # a user-input problem, not a reason to return nothing.
                self.logger.debug(f"FTS5 figure search failed, falling back to LIKE: {e}")
                use_fts = False
                rows = []
        if not use_fts:
            sql = """
                SELECT figure_id, doc_id, page_number, image_index,
                       ocr_text, image_path, 0.0 AS score
                FROM document_figures
                WHERE ocr_text LIKE ?
            """
            params = [f"%{query}%"]
            if doc_id:
                sql += " AND doc_id = ?"
                params.append(doc_id)
            sql += " ORDER BY char_count DESC LIMIT ?"
            params.append(max_results)
            rows = self.db_conn.execute(sql, params).fetchall()

        for r in rows:
            figure_doc = self.documents.get(r[1])
            results.append({
                'figure_id': r[0],
                'doc_id': r[1],
                'doc_title': figure_doc.title if figure_doc else None,
                'page_number': r[2],
                'image_index': r[3],
                'snippet': self._extract_snippet(r[4] or '', query_terms),
                'image_path': r[5],
                'score': round(r[6], 4) if r[6] else 0.0,
            })
        return results

    def get_figure_ocr_coverage(self) -> dict:
        """How much of the PDF corpus has been through figure OCR."""
        cursor = self.db_conn.cursor()
        pdf_ids = [d.doc_id for d in self.documents.values()
                   if (d.file_type or '').lower() == 'pdf']

        processed_ids = {row[0] for row in cursor.execute(
            "SELECT DISTINCT doc_id FROM document_figures"
        ).fetchall()}
        processed = len(processed_ids)
        figures, with_text = cursor.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN ocr_text IS NOT NULL AND ocr_text != '' "
            "THEN 1 ELSE 0 END), 0) FROM document_figures"
        ).fetchone()

        pending = cursor.execute(
            "SELECT COUNT(*) FROM extraction_jobs WHERE job_type = 'figures' "
            "AND status IN ('queued', 'running')"
        ).fetchone()[0]

        # A queued-but-unprocessed PDF whose documents.filepath no longer
        # exists on disk will never be processed - ocr_document_figures skips
        # it with "source file is no longer on disk" (see above). Splitting
        # documents_remaining keeps that number from implying a re-run can
        # reach 100% coverage when 106/178 PDFs in the live KB are gone.
        remaining_ids = [doc_id for doc_id in pdf_ids if doc_id not in processed_ids]
        unreachable = sum(
            1 for doc_id in remaining_ids
            if self._document_source_missing(self.documents[doc_id].filepath)
        )
        reachable = len(remaining_ids) - unreachable

        return {
            'pdf_documents': len(pdf_ids),
            'documents_processed': processed,
            # reachable + unreachable by construction (both derived from
            # remaining_ids) - keep documents_remaining defined the same way
            # rather than via len(pdf_ids) - processed, so the two never
            # drift apart if a document_figures row references a doc_id
            # that isn't (or is no longer) in pdf_ids.
            'documents_remaining': reachable + unreachable,
            'documents_remaining_reachable': reachable,
            'documents_remaining_unreachable': unreachable,
            'figures_extracted': figures,
            'figures_with_text': with_text,
            'jobs_pending': pending,
            'available': self.figure_ocr_available()[0],
            'unavailable_reason': self.figure_ocr_available()[1],
        }

    def queue_figure_ocr(self, doc_id: str, skip_if_exists: bool = True) -> dict:
        """Queue one document for background figure OCR."""
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        cursor = self.db_conn.cursor()
        if skip_if_exists:
            existing = cursor.execute(
                "SELECT COUNT(*) FROM document_figures WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
            if existing:
                return {'queued': False, 'reason': f"already has {existing} extracted figure(s)"}

            pending = cursor.execute(
                "SELECT job_id FROM extraction_jobs WHERE doc_id = ? AND job_type = 'figures' "
                "AND status IN ('queued', 'running') ORDER BY queued_at DESC LIMIT 1",
                (doc_id,)
            ).fetchone()
            if pending:
                return {'queued': False, 'reason': 'figure OCR already pending',
                        'existing_job_id': pending[0]}

        cursor.execute("""
            INSERT INTO extraction_jobs (doc_id, status, confidence_threshold, queued_at, job_type)
            VALUES (?, 'queued', 0.0, ?, 'figures')
        """, (doc_id, datetime.now(timezone.utc).isoformat()))
        self.db_conn.commit()
        job_id = cursor.lastrowid

        self._extraction_queue.put({
            'job_id': job_id, 'doc_id': doc_id,
            'confidence_threshold': 0.0, 'job_type': 'figures',
        })
        self.logger.info(f"Queued figure OCR job {job_id} for document {doc_id}")
        return {'queued': True, 'job_id': job_id}

    def queue_figure_ocr_all(self, limit: Optional[int] = None,
                             skip_if_exists: bool = True) -> dict:
        """Queue every PDF in the knowledge base for background figure OCR."""
        ok, reason = self.figure_ocr_available()
        if not ok:
            raise KnowledgeBaseError(f"Figure OCR unavailable: {reason}")

        pdf_docs = [d for d in self.documents.values()
                    if (d.file_type or '').lower() == 'pdf']
        pdf_docs.sort(key=lambda d: d.doc_id)

        queued, skipped = [], []
        for doc in pdf_docs:
            if limit is not None and len(queued) >= limit:
                break
            try:
                result = self.queue_figure_ocr(doc.doc_id, skip_if_exists=skip_if_exists)
            except Exception as e:
                skipped.append({'doc_id': doc.doc_id, 'reason': str(e)})
                continue
            if result.get('queued'):
                queued.append({'doc_id': doc.doc_id, 'job_id': result['job_id']})
            else:
                skipped.append({'doc_id': doc.doc_id, 'reason': result.get('reason', '')})

        self.logger.info(f"Figure OCR batch: queued {len(queued)}, skipped {len(skipped)}")
        return {
            'pdf_documents': len(pdf_docs),
            'queued': len(queued),
            'skipped': len(skipped),
            'jobs': queued,
            'skipped_details': skipped[:20],
        }
