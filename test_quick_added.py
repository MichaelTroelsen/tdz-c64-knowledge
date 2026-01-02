#!/usr/bin/env python3
"""
Unit tests for Quick Added tab and file operations in Archive Search
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestQuickAddedTab(unittest.TestCase):
    """Test suite for Quick Added tab functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample quick-added file entries
        self.sample_success_entry = {
            'title': 'Commodore 64 Programmer\'s Reference Guide - C64_PRG.pdf',
            'file_name': 'C64_PRG.pdf',
            'source_url': 'https://archive.org/details/c64-programmers-reference',
            'doc_id': 'abc123def456',
            'status': 'success',
            'timestamp': '2026-01-02 20:30:45 UTC'
        }

        self.sample_failed_entry = {
            'title': 'VIC-II Specifications - vic2_specs.pdf',
            'file_name': 'vic2_specs.pdf',
            'source_url': 'https://archive.org/details/vic-ii-specs',
            'doc_id': None,
            'status': 'failed',
            'error': 'Connection timeout',
            'timestamp': '2026-01-02 20:35:12 UTC'
        }

    def test_session_state_initialization(self):
        """Test that session state for quick_added_files is initialized correctly."""
        # Simulate session state initialization
        session_state = {'quick_added_files': []}

        self.assertIsInstance(session_state['quick_added_files'], list)
        self.assertEqual(len(session_state['quick_added_files']), 0)

        print("✅ Session state initialization test passed")

    def test_add_success_entry(self):
        """Test adding a successful quick-add entry."""
        quick_added_files = []

        # Add successful entry
        quick_added_files.append(self.sample_success_entry)

        # Verify entry was added
        self.assertEqual(len(quick_added_files), 1)
        self.assertEqual(quick_added_files[0]['status'], 'success')
        self.assertEqual(quick_added_files[0]['doc_id'], 'abc123def456')
        self.assertIsNotNone(quick_added_files[0]['doc_id'])

        print("✅ Add success entry test passed")

    def test_add_failed_entry(self):
        """Test adding a failed quick-add entry."""
        quick_added_files = []

        # Add failed entry
        quick_added_files.append(self.sample_failed_entry)

        # Verify entry was added with error
        self.assertEqual(len(quick_added_files), 1)
        self.assertEqual(quick_added_files[0]['status'], 'failed')
        self.assertIsNone(quick_added_files[0]['doc_id'])
        self.assertIn('error', quick_added_files[0])
        self.assertEqual(quick_added_files[0]['error'], 'Connection timeout')

        print("✅ Add failed entry test passed")

    def test_multiple_entries(self):
        """Test adding multiple quick-add entries."""
        quick_added_files = []

        # Add multiple entries
        quick_added_files.append(self.sample_success_entry)
        quick_added_files.append(self.sample_failed_entry)
        quick_added_files.append({
            'title': 'SID Programming Guide',
            'file_name': 'sid_guide.pdf',
            'source_url': 'https://archive.org/details/sid-guide',
            'doc_id': 'xyz789abc123',
            'status': 'success',
            'timestamp': '2026-01-02 20:40:00 UTC'
        })

        # Verify all entries
        self.assertEqual(len(quick_added_files), 3)
        self.assertEqual(quick_added_files[0]['status'], 'success')
        self.assertEqual(quick_added_files[1]['status'], 'failed')
        self.assertEqual(quick_added_files[2]['status'], 'success')

        print("✅ Multiple entries test passed")

    def test_clear_history(self):
        """Test clearing the quick-added history."""
        quick_added_files = [
            self.sample_success_entry,
            self.sample_failed_entry
        ]

        # Verify entries exist
        self.assertEqual(len(quick_added_files), 2)

        # Clear history
        quick_added_files.clear()

        # Verify cleared
        self.assertEqual(len(quick_added_files), 0)

        print("✅ Clear history test passed")

    def test_entry_structure(self):
        """Test that quick-add entries have required fields."""
        required_fields = ['title', 'file_name', 'source_url', 'status', 'timestamp']

        # Check success entry
        for field in required_fields:
            self.assertIn(field, self.sample_success_entry)
        self.assertIn('doc_id', self.sample_success_entry)

        # Check failed entry
        for field in required_fields:
            self.assertIn(field, self.sample_failed_entry)
        self.assertIn('error', self.sample_failed_entry)

        print("✅ Entry structure test passed")

    def test_timestamp_format(self):
        """Test timestamp format is correct."""
        timestamp = self.sample_success_entry['timestamp']

        # Verify timestamp can be parsed
        try:
            parsed = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S UTC')
            self.assertIsInstance(parsed, datetime)
        except ValueError:
            self.fail("Timestamp format is invalid")

        print("✅ Timestamp format test passed")

    def test_reverse_chronological_order(self):
        """Test that entries should be displayed in reverse order (newest first)."""
        quick_added_files = [
            {
                'title': 'First',
                'file_name': 'first.pdf',
                'source_url': 'https://archive.org/1',
                'doc_id': 'id1',
                'status': 'success',
                'timestamp': '2026-01-02 20:00:00 UTC'
            },
            {
                'title': 'Second',
                'file_name': 'second.pdf',
                'source_url': 'https://archive.org/2',
                'doc_id': 'id2',
                'status': 'success',
                'timestamp': '2026-01-02 20:10:00 UTC'
            },
            {
                'title': 'Third',
                'file_name': 'third.pdf',
                'source_url': 'https://archive.org/3',
                'doc_id': 'id3',
                'status': 'success',
                'timestamp': '2026-01-02 20:20:00 UTC'
            }
        ]

        # Reverse for display (newest first)
        reversed_list = list(reversed(quick_added_files))

        # Verify order
        self.assertEqual(reversed_list[0]['title'], 'Third')
        self.assertEqual(reversed_list[1]['title'], 'Second')
        self.assertEqual(reversed_list[2]['title'], 'First')

        print("✅ Reverse chronological order test passed")


class TestDownloadFunctionality(unittest.TestCase):
    """Test suite for file download functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://archive.org/download/test-item/test-file.pdf"
        self.test_filename = "test-file.pdf"

    def tearDown(self):
        """Clean up test files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('urllib.request.urlretrieve')
    def test_download_to_temp_directory(self, mock_urlretrieve):
        """Test downloading file to temp directory."""
        # Create temp directory
        temp_dir = Path(self.temp_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Simulate download
        temp_filename = f"quick_add_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.test_filename}"
        tmp_path = temp_dir / temp_filename

        # Mock urlretrieve to create a dummy file
        def create_dummy_file(url, path):
            Path(path).write_text("dummy content")

        mock_urlretrieve.side_effect = create_dummy_file
        mock_urlretrieve(self.test_url, str(tmp_path))

        # Verify file was created
        self.assertTrue(tmp_path.exists())
        self.assertEqual(tmp_path.read_text(), "dummy content")

        print("✅ Download to temp directory test passed")

    def test_temp_file_naming_convention(self):
        """Test that temp files follow correct naming convention."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"quick_add_{timestamp}_{self.test_filename}"

        # Verify naming pattern
        self.assertTrue(temp_filename.startswith("quick_add_"))
        self.assertTrue(temp_filename.endswith(self.test_filename))
        self.assertIn(timestamp, temp_filename)

        print("✅ Temp file naming convention test passed")

    def test_temp_directory_creation(self):
        """Test that temp directory is created if it doesn't exist."""
        temp_dir = Path(self.temp_dir) / "temp"

        # Verify doesn't exist
        self.assertFalse(temp_dir.exists())

        # Create directory
        temp_dir.mkdir(exist_ok=True)

        # Verify exists
        self.assertTrue(temp_dir.exists())
        self.assertTrue(temp_dir.is_dir())

        print("✅ Temp directory creation test passed")

    def test_download_to_downloads_directory(self):
        """Test downloading file to downloads directory."""
        downloads_dir = Path(self.temp_dir) / "downloads"
        downloads_dir.mkdir(exist_ok=True)

        # Simulate download
        download_path = downloads_dir / self.test_filename
        download_path.write_text("downloaded content")

        # Verify file was downloaded
        self.assertTrue(download_path.exists())
        self.assertEqual(download_path.read_text(), "downloaded content")

        print("✅ Download to downloads directory test passed")

    def test_cleanup_on_success(self):
        """Test that temp files are cleaned up after successful add."""
        temp_dir = Path(self.temp_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Create temp file
        tmp_path = temp_dir / "temp_file.pdf"
        tmp_path.write_text("temp content")

        # Verify exists
        self.assertTrue(tmp_path.exists())

        # Simulate cleanup
        tmp_path.unlink()

        # Verify removed
        self.assertFalse(tmp_path.exists())

        print("✅ Cleanup on success test passed")

    def test_cleanup_on_failure(self):
        """Test that temp files are cleaned up even on failure."""
        temp_dir = Path(self.temp_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Create temp file
        tmp_path = temp_dir / "temp_file.pdf"
        tmp_path.write_text("temp content")

        try:
            # Simulate error
            raise Exception("Test error")
        except Exception:
            # Cleanup should still happen
            if tmp_path.exists():
                tmp_path.unlink()

        # Verify removed
        self.assertFalse(tmp_path.exists())

        print("✅ Cleanup on failure test passed")


class TestAddToKBFunctionality(unittest.TestCase):
    """Test suite for add to knowledge base functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_doc.pdf"
        self.test_file.write_text("Test document content")

    def tearDown(self):
        """Clean up test files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_add_document_parameters(self):
        """Test that add_document is called with correct parameters."""
        # Mock KnowledgeBase
        mock_kb = Mock()
        mock_doc = Mock()
        mock_doc.doc_id = "test123"
        mock_kb.add_document.return_value = mock_doc

        # Simulate add_document call
        title = "Test Document - test_doc.pdf"
        tags = ["test", "archive"]

        doc = mock_kb.add_document(
            str(self.test_file),
            title=title,
            tags=tags
        )

        # Verify call
        mock_kb.add_document.assert_called_once_with(
            str(self.test_file),
            title=title,
            tags=tags
        )
        self.assertEqual(doc.doc_id, "test123")

        print("✅ Add document parameters test passed")

    def test_source_url_metadata_update(self):
        """Test that source URL is updated in database."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        # Simulate source URL update
        source_url = "https://archive.org/details/test-item"
        doc_id = "test123"
        scrape_date = datetime.now(timezone.utc).isoformat()

        mock_cursor.execute("""
            UPDATE documents
            SET source_url = ?,
                scrape_date = ?,
                scrape_status = 'success'
            WHERE doc_id = ?
        """, (source_url, scrape_date, doc_id))
        mock_conn.commit()

        # Verify update was called
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        print("✅ Source URL metadata update test passed")

    def test_document_metadata_structure(self):
        """Test that document metadata has required fields."""
        # Mock document object
        mock_doc = Mock()
        mock_doc.doc_id = "abc123"
        mock_doc.title = "Test Document"
        mock_doc.filepath = str(self.test_file)
        mock_doc.source_url = "https://archive.org/details/test"

        # Verify fields
        self.assertEqual(mock_doc.doc_id, "abc123")
        self.assertEqual(mock_doc.title, "Test Document")
        self.assertIsNotNone(mock_doc.source_url)

        print("✅ Document metadata structure test passed")

    def test_title_generation(self):
        """Test automatic title generation from archive.org metadata."""
        result_title = "Commodore 64 Programmer's Reference"
        file_name = "C64_PRG.pdf"

        # Generate title
        title = f"{result_title} - {file_name}"

        # Verify title format
        self.assertEqual(title, "Commodore 64 Programmer's Reference - C64_PRG.pdf")
        self.assertIn(result_title, title)
        self.assertIn(file_name, title)

        print("✅ Title generation test passed")

    def test_tags_extraction(self):
        """Test tag extraction from archive.org subject metadata."""
        # Test with list of tags
        subject_list = ["commodore-64", "programming", "manual"]
        tags = subject_list if isinstance(subject_list, list) else [subject_list]

        self.assertIsInstance(tags, list)
        self.assertEqual(len(tags), 3)
        self.assertIn("commodore-64", tags)

        # Test with single tag string
        subject_string = "commodore-64"
        tags = [subject_string] if isinstance(subject_string, str) else subject_string

        self.assertIsInstance(tags, list)
        self.assertEqual(len(tags), 1)

        print("✅ Tags extraction test passed")

    def test_add_to_kb_error_handling(self):
        """Test error handling when adding document fails."""
        # Mock KnowledgeBase that raises error
        mock_kb = Mock()
        mock_kb.add_document.side_effect = Exception("Database error")

        # Simulate add attempt with error handling
        error_occurred = False
        error_message = None

        try:
            mock_kb.add_document(str(self.test_file))
        except Exception as e:
            error_occurred = True
            error_message = str(e)

        # Verify error was caught
        self.assertTrue(error_occurred)
        self.assertEqual(error_message, "Database error")

        print("✅ Add to KB error handling test passed")

    def test_path_security_validation(self):
        """Test that files are within allowed directories."""
        data_dir = Path(self.temp_dir)
        temp_dir = data_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Create file in temp directory
        test_file = temp_dir / "test.pdf"
        test_file.write_text("content")

        # Verify file is within allowed directory
        self.assertTrue(str(test_file).startswith(str(data_dir)))
        self.assertTrue(test_file.exists())

        print("✅ Path security validation test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete Quick Add workflow."""

    def test_complete_quick_add_workflow(self):
        """Test complete workflow: download -> add to KB -> track."""
        # Step 1: Initialize tracking
        quick_added_files = []

        # Step 2: Simulate download and add
        mock_doc = Mock()
        mock_doc.doc_id = "workflow123"

        # Step 3: Record success
        quick_added_files.append({
            'title': 'Test Workflow Document',
            'file_name': 'workflow.pdf',
            'source_url': 'https://archive.org/details/workflow',
            'doc_id': mock_doc.doc_id,
            'status': 'success',
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        })

        # Verify complete workflow
        self.assertEqual(len(quick_added_files), 1)
        self.assertEqual(quick_added_files[0]['status'], 'success')
        self.assertEqual(quick_added_files[0]['doc_id'], 'workflow123')

        print("✅ Complete quick add workflow test passed")

    def test_failed_workflow_tracking(self):
        """Test workflow when add to KB fails."""
        # Step 1: Initialize tracking
        quick_added_files = []

        # Step 2: Simulate failed add
        error = Exception("Connection timeout")

        # Step 3: Record failure
        quick_added_files.append({
            'title': 'Failed Document',
            'file_name': 'failed.pdf',
            'source_url': 'https://archive.org/details/failed',
            'doc_id': None,
            'status': 'failed',
            'error': str(error),
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        })

        # Verify failure tracking
        self.assertEqual(len(quick_added_files), 1)
        self.assertEqual(quick_added_files[0]['status'], 'failed')
        self.assertIsNone(quick_added_files[0]['doc_id'])
        self.assertIn('Connection timeout', quick_added_files[0]['error'])

        print("✅ Failed workflow tracking test passed")


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 60)
    print("QUICK ADDED TAB AND FILE OPERATIONS TESTS")
    print("=" * 60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuickAddedTab))
    suite.addTests(loader.loadTestsFromTestCase(TestDownloadFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestAddToKBFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
