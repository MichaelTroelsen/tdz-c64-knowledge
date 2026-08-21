#!/usr/bin/env python3
"""
Unit tests for "In KB" status detection in Archive Search

This test validates the entire workflow:
1. File URL construction from archive.org
2. Quick Add saves source_url correctly
3. Duplicate detection finds existing files
4. "In KB" status would be displayed
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase, DocumentMeta


class TestInKBStatus(unittest.TestCase):
    """Test suite for In KB status detection."""

    def setUp(self):
        """Set up test fixtures with temporary KB."""
        # Create temp directory for test KB
        self.test_dir = Path(tempfile.mkdtemp())

        # Set ALLOWED_DOCS_DIRS to include test directory
        os.environ['ALLOWED_DOCS_DIRS'] = str(self.test_dir)

        # Disable auto entity extraction for faster tests
        os.environ['AUTO_EXTRACT_ENTITIES'] = '0'

        self.kb = KnowledgeBase(str(self.test_dir))

        # Create a test document file in allowed directory (temp subfolder)
        temp_dir = self.test_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        self.test_file = temp_dir / "test_document.txt"
        self.test_file.write_text("This is a test C64 document about VIC-II.", encoding='utf-8')

    def tearDown(self):
        """Clean up test directory."""
        # Close database connection before cleanup
        if hasattr(self, 'kb') and hasattr(self.kb, 'db_conn'):
            try:
                self.kb.db_conn.close()
            except:
                pass

        # Clean up test directory
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                # On Windows, database file might still be locked
                import time
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.test_dir)
                except:
                    pass  # Best effort cleanup

    def test_file_url_construction(self):
        """Test that file URLs are constructed correctly."""
        # Simulate archive.org data
        identifier = "back-in-time-1"
        filename = "Back in Time 1/Extras/BIT 1 - DD Booklet.pdf"

        # URL construction (same as admin_gui.py line 3873)
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        # Verify URL format
        self.assertTrue(file_url.startswith("https://archive.org/download/"))
        self.assertIn(identifier, file_url)
        self.assertNotIn(" ", file_url)  # No spaces
        self.assertIn("%20", file_url)  # Spaces encoded

        expected = "https://archive.org/download/back-in-time-1/Back%20in%20Time%201%2FExtras%2FBIT%201%20-%20DD%20Booklet.pdf"
        self.assertEqual(file_url, expected)

        print(f"✅ File URL constructed correctly: {file_url[:80]}...")

    def test_quick_add_saves_source_url(self):
        """Test that Quick Add saves file URL as source_url."""
        # Simulate archive.org file URL
        identifier = "test-item"
        filename = "Test Document.pdf"
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        # Add document (simulating Quick Add)
        doc = self.kb.add_document(
            str(self.test_file),
            title="Test Document",
            tags=["test", "archive"]
        )

        # Update source_url (same as admin_gui.py lines 4022-4030)
        with self.kb.db_conn:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?,
                    scrape_date = ?,
                    scrape_status = 'success'
                WHERE doc_id = ?
            """, (file_url, datetime.now(timezone.utc).isoformat(), doc.doc_id))

        # Update in-memory object
        doc.source_url = file_url
        self.kb.documents[doc.doc_id] = doc

        # Verify source_url is saved
        cursor = self.kb.db_conn.cursor()
        cursor.execute("SELECT source_url FROM documents WHERE doc_id = ?", (doc.doc_id,))
        saved_url = cursor.fetchone()[0]

        self.assertEqual(saved_url, file_url)
        print(f"✅ source_url saved to database: {saved_url[:80]}...")

    def test_source_url_loaded_on_kb_init(self):
        """Test that source_url is loaded when KB is reinitialized."""
        # Add document with source_url
        identifier = "test-item"
        filename = "Test Document.pdf"
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        doc = self.kb.add_document(
            str(self.test_file),
            title="Test Document",
            tags=["test"]
        )

        # Update source_url in database
        with self.kb.db_conn:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?
                WHERE doc_id = ?
            """, (file_url, doc.doc_id))

        # Create NEW KB instance (simulates Streamlit rerun)
        new_kb = KnowledgeBase(str(self.test_dir))

        # Verify source_url is loaded
        loaded_doc = new_kb.documents[doc.doc_id]
        self.assertIsNotNone(loaded_doc.source_url)
        self.assertEqual(loaded_doc.source_url, file_url)

        print(f"✅ source_url loaded on KB init: {loaded_doc.source_url[:80]}...")

    def test_duplicate_detection_logic(self):
        """Test the duplicate detection logic used in admin_gui.py."""
        # Add document with source_url
        identifier = "test-item"
        filename = "Test Document.pdf"
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        doc = self.kb.add_document(
            str(self.test_file),
            title="Test Document",
            tags=["test"]
        )

        # Update source_url
        with self.kb.db_conn:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?
                WHERE doc_id = ?
            """, (file_url, doc.doc_id))

        # Reload KB (simulates Streamlit rerun)
        kb = KnowledgeBase(str(self.test_dir))

        # Simulate the duplicate detection check (admin_gui.py lines 3957-3961)
        file = {
            'name': filename,
            'url': file_url
        }

        existing_doc = None
        for doc in kb.documents.values():
            if hasattr(doc, 'source_url') and doc.source_url == file['url']:
                existing_doc = doc
                break

        # Verify duplicate was found
        self.assertIsNotNone(existing_doc, "Duplicate detection should find the existing document")
        self.assertEqual(existing_doc.source_url, file_url)

        print(f"✅ Duplicate detection found existing document: {existing_doc.doc_id[:12]}...")

    def test_in_kb_status_would_be_shown(self):
        """Test that 'In KB' status would be shown (not Download/Quick Add buttons)."""
        # Add document with source_url
        identifier = "back-in-time-1"
        filename = "Back in Time 1 Booklet.pdf"
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        doc = self.kb.add_document(
            str(self.test_file),
            title="Back in Time 1 Booklet",
            tags=["disk-magazine"]
        )

        # Update source_url
        with self.kb.db_conn:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?
                WHERE doc_id = ?
            """, (file_url, doc.doc_id))

        # Reload KB
        kb = KnowledgeBase(str(self.test_dir))

        # Simulate archive search file
        file = {
            'name': filename,
            'size': 1024000,
            'format': 'PDF',
            'url': file_url
        }

        # Duplicate detection
        existing_doc = None
        for doc in kb.documents.values():
            if hasattr(doc, 'source_url') and doc.source_url == file['url']:
                existing_doc = doc
                break

        if existing_doc:
            # This is where "✅ In KB" would be shown
            status = "✅ In KB"
            show_buttons = False
        else:
            # This is where Download/Quick Add buttons would be shown
            status = "Show buttons"
            show_buttons = True

        # Verify correct status
        self.assertEqual(status, "✅ In KB")
        self.assertFalse(show_buttons, "Download/Quick Add buttons should NOT be shown")

        print(f"✅ 'In KB' status would be displayed (doc: {existing_doc.doc_id[:12]}...)")

    def test_multiple_files_duplicate_detection(self):
        """Test duplicate detection with multiple files."""
        # Add multiple documents (use .txt extension to avoid PDF parsing issues)
        files_to_add = [
            ("item-1", "Document_A.txt"),
            ("item-2", "Document_B.txt"),
            ("item-3", "Document_C.txt"),
        ]

        added_docs = []
        temp_dir = self.test_dir / "temp"
        for identifier, filename in files_to_add:
            # Create unique test file in allowed temp directory
            test_file = temp_dir / filename
            test_file.write_text(f"Test content for {filename}", encoding='utf-8')

            # Add document
            doc = self.kb.add_document(
                str(test_file),
                title=filename,
                tags=["test"]
            )

            # Update source_url
            encoded_filename = quote(filename, safe='')
            file_url = f"https://archive.org/download/{identifier}/{encoded_filename}"

            with self.kb.db_conn:
                cursor = self.kb.db_conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET source_url = ?
                    WHERE doc_id = ?
                """, (file_url, doc.doc_id))

            added_docs.append((doc.doc_id, file_url))

            # Clean up test file
            test_file.unlink()

        # Reload KB
        kb = KnowledgeBase(str(self.test_dir))

        # Check each file for duplicate
        for doc_id, file_url in added_docs:
            file = {'url': file_url}

            existing_doc = None
            for doc in kb.documents.values():
                if hasattr(doc, 'source_url') and doc.source_url == file['url']:
                    existing_doc = doc
                    break

            self.assertIsNotNone(existing_doc, f"Should find doc {doc_id[:12]}...")
            self.assertEqual(existing_doc.doc_id, doc_id)

        print(f"✅ All {len(added_docs)} documents detected as duplicates correctly")

    def test_url_encoding_consistency(self):
        """Test that URL encoding is consistent between save and check."""
        # Test various problematic filenames
        test_cases = [
            ("simple.pdf", "simple.pdf"),
            ("With Spaces.pdf", "With%20Spaces.pdf"),
            ("Special (chars).pdf", "Special%20%28chars%29.pdf"),
            ("Path/To/File.pdf", "Path%2FTo%2FFile.pdf"),
        ]

        for original, expected_encoded in test_cases:
            with self.subTest(filename=original):
                # Encode filename
                encoded = quote(original, safe='')
                self.assertEqual(encoded, expected_encoded)

                # Construct URL
                file_url = f"https://archive.org/download/test/{encoded}"

                # Verify no spaces in URL
                self.assertNotIn(" ", file_url)

        print(f"✅ URL encoding consistent across {len(test_cases)} test cases")

    def test_no_source_url_shows_buttons(self):
        """Test that documents without source_url show Download/Quick Add buttons."""
        # Add document WITHOUT source_url
        doc = self.kb.add_document(
            str(self.test_file),
            title="Document Without Source URL",
            tags=["test"]
        )

        # Reload KB
        kb = KnowledgeBase(str(self.test_dir))

        # Simulate checking for a file that doesn't match
        file = {
            'url': "https://archive.org/download/different-item/different-file.pdf"
        }

        # Duplicate detection
        existing_doc = None
        for doc in kb.documents.values():
            if hasattr(doc, 'source_url') and doc.source_url == file['url']:
                existing_doc = doc
                break

        # Verify no match found
        self.assertIsNone(existing_doc, "Should not find duplicate")

        # Buttons should be shown
        show_buttons = (existing_doc is None)
        self.assertTrue(show_buttons, "Download/Quick Add buttons SHOULD be shown")

        print("✅ Documents without matching source_url correctly show buttons")

    def test_old_format_item_url_detection(self):
        """Test that old format (item URL) documents are detected as duplicates."""
        # Create a file with realistic Quick Add filename (spaces, not underscores)
        # This matches the actual format from Quick Add: quick_add_TIMESTAMP_filename.pdf
        temp_dir = self.test_dir / "temp"
        test_file = temp_dir / "quick_add_20260102_000000_Back in Time 3 Booklet.txt"
        test_file.write_text("Test content for Back in Time 3", encoding='utf-8')

        # Add document with OLD format source_url (item URL, not file URL)
        doc = self.kb.add_document(
            str(test_file),
            title="Back in Time 3 by C64Audio.com - Back in Time 3 Booklet",
            tags=["disk-magazine"]
        )

        # Clean up test file
        test_file.unlink()

        # Update with OLD format source_url (as was saved before fix)
        item_id = "back-in-time-3"
        old_format_url = f"https://archive.org/details/{item_id}"

        with self.kb.db_conn:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?
                WHERE doc_id = ?
            """, (old_format_url, doc.doc_id))

        # Reload KB
        kb = KnowledgeBase(str(self.test_dir))

        # Simulate archive search file (new format with file URL)
        # Archive.org file path, but we'll match on filename similarity
        filename = "Back in Time 3/Extras/Back in Time 3 Booklet.PDF"
        encoded_filename = quote(filename, safe='')
        file_url = f"https://archive.org/download/{item_id}/{encoded_filename}"

        file = {
            'name': filename,
            'url': file_url
        }

        result = {
            'identifier': item_id
        }

        # Enhanced duplicate detection (same logic as admin_gui.py)
        existing_doc = None

        for doc in kb.documents.values():
            if not hasattr(doc, 'source_url') or not doc.source_url:
                continue

            # Try exact file URL match (new format)
            if doc.source_url == file['url']:
                existing_doc = doc
                break

            # Try item identifier match (old format)
            if item_id in doc.source_url:
                # Also check if filename matches (ignore extension for flexibility)
                safe_filename = Path(file['name']).stem  # Get filename without extension
                doc_stem = Path(doc.filename).stem  # Get doc filename without extension
                if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                    existing_doc = doc
                    break

        # Verify old format document was found
        self.assertIsNotNone(existing_doc, "Should find document with old format URL")
        self.assertEqual(existing_doc.source_url, old_format_url)

        print(f"✅ Old format (item URL) detected as duplicate: {existing_doc.doc_id[:12]}...")


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 70)
    print("IN KB STATUS DETECTION TESTS")
    print("=" * 70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInKBStatus)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Fix Windows console encoding
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    success = run_tests()
    sys.exit(0 if success else 1)
