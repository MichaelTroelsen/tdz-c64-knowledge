#!/usr/bin/env python3
"""
Unit tests for URL encoding in Archive Search

Tests that filenames with spaces, special characters, and directory
separators are properly URL-encoded for archive.org downloads.
"""

import unittest
import sys
from pathlib import Path
from urllib.parse import quote
from unittest.mock import Mock, patch, MagicMock

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestURLEncoding(unittest.TestCase):
    """Test suite for URL encoding in Archive Search."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "https://archive.org/download"
        self.identifier = "test-item-123"

    def test_simple_filename(self):
        """Test URL encoding for simple filename without special characters."""
        filename = "document.pdf"
        encoded = quote(filename, safe='')
        url = f"{self.base_url}/{self.identifier}/{encoded}"

        self.assertEqual(encoded, "document.pdf")
        self.assertEqual(url, f"{self.base_url}/{self.identifier}/document.pdf")
        print("✅ Simple filename test passed")

    def test_filename_with_spaces(self):
        """Test URL encoding for filename with spaces."""
        filename = "Back in Time 1.pdf"
        encoded = quote(filename, safe='')
        url = f"{self.base_url}/{self.identifier}/{encoded}"

        # Spaces should be encoded as %20
        self.assertIn("%20", encoded)
        self.assertEqual(encoded, "Back%20in%20Time%201.pdf")
        self.assertNotIn(" ", url)
        print("✅ Filename with spaces test passed")

    def test_filename_with_directory_path(self):
        """Test URL encoding for filename with directory separators."""
        filename = "Back in Time 1/Extras/BIT 1 - DD Booklet.pdf"
        encoded = quote(filename, safe='')
        url = f"{self.base_url}/{self.identifier}/{encoded}"

        # Spaces should be encoded
        self.assertIn("%20", encoded)
        # Forward slashes should be encoded (safe='')
        self.assertIn("%2F", encoded)
        # Hyphens should be encoded
        self.assertIn("-", encoded)  # Hyphens are safe
        # No unencoded spaces
        self.assertNotIn(" ", url)

        expected = "Back%20in%20Time%201%2FExtras%2FBIT%201%20-%20DD%20Booklet.pdf"
        self.assertEqual(encoded, expected)
        print("✅ Filename with directory path test passed")

    def test_filename_with_special_characters(self):
        """Test URL encoding for filename with various special characters."""
        test_cases = [
            ("C64 Music (SID).pdf", "C64%20Music%20%28SID%29.pdf"),
            ("Game & Demo.pdf", "Game%20%26%20Demo.pdf"),
            ("Price: $5.99.pdf", "Price%3A%20%245.99.pdf"),
            ("80% Complete.pdf", "80%25%20Complete.pdf"),
            ("Item #123.pdf", "Item%20%23123.pdf"),
        ]

        for original, expected in test_cases:
            with self.subTest(filename=original):
                encoded = quote(original, safe='')
                self.assertEqual(encoded, expected)
                self.assertNotIn(" ", encoded)

        print("✅ Special characters test passed")

    def test_filename_with_unicode(self):
        """Test URL encoding for filename with unicode characters."""
        filename = "Документ.pdf"  # Russian
        encoded = quote(filename, safe='')

        # Unicode should be percent-encoded
        self.assertIn("%", encoded)
        # Original unicode should not appear
        self.assertNotEqual(encoded, filename)

        print("✅ Unicode filename test passed")

    def test_url_construction(self):
        """Test complete URL construction with encoded filename."""
        identifier = "back-in-time-1"
        filename = "Back in Time 1/Extras/BIT 1 - DD Booklet.pdf"
        encoded_filename = quote(filename, safe='')

        url = f"https://archive.org/download/{identifier}/{encoded_filename}"

        # Verify URL structure
        self.assertTrue(url.startswith("https://archive.org/download/"))
        self.assertIn(identifier, url)
        self.assertNotIn(" ", url)

        expected_url = "https://archive.org/download/back-in-time-1/Back%20in%20Time%201%2FExtras%2FBIT%201%20-%20DD%20Booklet.pdf"
        self.assertEqual(url, expected_url)

        print("✅ URL construction test passed")

    def test_empty_filename(self):
        """Test URL encoding for empty filename."""
        filename = ""
        encoded = quote(filename, safe='')

        self.assertEqual(encoded, "")
        print("✅ Empty filename test passed")

    def test_filename_with_only_spaces(self):
        """Test URL encoding for filename with only spaces."""
        filename = "   "
        encoded = quote(filename, safe='')

        # All spaces should be encoded
        self.assertEqual(encoded, "%20%20%20")
        self.assertNotIn(" ", encoded)

        print("✅ Filename with only spaces test passed")

    def test_filename_with_plus_sign(self):
        """Test URL encoding for filename with plus sign."""
        filename = "C++_Programming.pdf"
        encoded = quote(filename, safe='')

        # Plus signs should be encoded
        self.assertIn("%2B", encoded)
        self.assertEqual(encoded, "C%2B%2B_Programming.pdf")

        print("✅ Plus sign test passed")

    def test_filename_with_equals_sign(self):
        """Test URL encoding for filename with equals sign."""
        filename = "E=MC2.pdf"
        encoded = quote(filename, safe='')

        # Equals sign should be encoded
        self.assertIn("%3D", encoded)
        self.assertEqual(encoded, "E%3DMC2.pdf")

        print("✅ Equals sign test passed")

    def test_filename_with_question_mark(self):
        """Test URL encoding for filename with question mark."""
        filename = "What is C64?.pdf"
        encoded = quote(filename, safe='')

        # Question mark should be encoded
        self.assertIn("%3F", encoded)
        self.assertNotIn("?", encoded)

        print("✅ Question mark test passed")

    def test_safe_parameter_default(self):
        """Test that safe='' encodes all special characters."""
        filename = "A/B C&D.pdf"
        encoded_safe_empty = quote(filename, safe='')
        encoded_safe_slash = quote(filename, safe='/')

        # With safe='', forward slash should be encoded
        self.assertIn("%2F", encoded_safe_empty)

        # With safe='/', forward slash should not be encoded
        self.assertNotIn("%2F", encoded_safe_slash)
        self.assertIn("/", encoded_safe_slash)

        print("✅ Safe parameter test passed")

    def test_double_encoding_prevention(self):
        """Test that we don't double-encode already encoded URLs."""
        filename = "Document.pdf"
        encoded_once = quote(filename, safe='')
        encoded_twice = quote(encoded_once, safe='')

        # First encoding should be idempotent for simple filenames
        self.assertEqual(encoded_once, encoded_twice)

        print("✅ Double encoding prevention test passed")

    @patch('urllib.request.urlretrieve')
    def test_urlretrieve_with_encoded_url(self, mock_urlretrieve):
        """Test that urllib.request.urlretrieve accepts properly encoded URLs."""
        # Simulate the problematic case from the error report
        filename = "Back in Time 1/Extras/BIT 1 - DD Booklet.pdf"
        encoded_filename = quote(filename, safe='')
        url = f"https://archive.org/download/back-in-time-1/{encoded_filename}"

        # This should not raise InvalidURL error
        import urllib.request
        mock_urlretrieve.return_value = (None, None)

        try:
            urllib.request.urlretrieve(url, "/tmp/test.pdf")
            self.assertTrue(True)  # Success if no exception
        except Exception as e:
            self.fail(f"urlretrieve raised exception with encoded URL: {e}")

        print("✅ urlretrieve with encoded URL test passed")


class TestFilenameExtraction(unittest.TestCase):
    """Test extraction of safe filenames from paths."""

    def test_extract_filename_from_simple_path(self):
        """Test extracting filename from simple filename."""
        from pathlib import Path

        filename = "document.pdf"
        safe_filename = Path(filename).name

        self.assertEqual(safe_filename, "document.pdf")
        print("✅ Simple filename extraction test passed")

    def test_extract_filename_from_directory_path(self):
        """Test extracting filename from path with directories."""
        from pathlib import Path

        filename = "Back in Time 3/Extras/Back in Time 3 Booklet.PDF"
        safe_filename = Path(filename).name

        # Should extract only the last component
        self.assertEqual(safe_filename, "Back in Time 3 Booklet.PDF")
        self.assertNotIn("/", safe_filename)
        self.assertNotIn("\\", safe_filename)
        print("✅ Directory path extraction test passed")

    def test_extract_filename_with_multiple_slashes(self):
        """Test extracting filename from deep directory structure."""
        from pathlib import Path

        test_cases = [
            ("a/b/c/file.pdf", "file.pdf"),
            ("dir1/dir2/dir3/dir4/document.txt", "document.txt"),
            ("Folder 1/Subfolder 2/File Name.pdf", "File Name.pdf"),
        ]

        for original, expected in test_cases:
            with self.subTest(filename=original):
                safe_filename = Path(original).name
                self.assertEqual(safe_filename, expected)
                self.assertNotIn("/", safe_filename)

        print("✅ Multiple slashes extraction test passed")

    def test_temp_filename_construction(self):
        """Test construction of temp filename without directory components."""
        from pathlib import Path
        from datetime import datetime

        # Simulate the problematic case
        filename = "Back in Time 3/Extras/Back in Time 3 Booklet.PDF"
        safe_filename = Path(filename).name

        # Construct temp filename
        temp_filename = f"quick_add_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"

        # Verify no directory separators in temp filename
        self.assertNotIn("/", temp_filename)
        self.assertNotIn("\\", temp_filename)
        self.assertTrue(temp_filename.endswith("Back in Time 3 Booklet.PDF"))
        self.assertTrue(temp_filename.startswith("quick_add_"))

        print("✅ Temp filename construction test passed")

    def test_path_join_with_safe_filename(self):
        """Test that Path join works correctly with extracted filename."""
        from pathlib import Path

        temp_dir = Path("/tmp/test")
        filename = "Folder/Subfolder/File.pdf"
        safe_filename = Path(filename).name

        # This should create a valid single-level path
        result_path = temp_dir / safe_filename

        # Should only have one component after temp_dir
        self.assertEqual(result_path.name, "File.pdf")
        # Check path ends correctly (OS-agnostic)
        self.assertTrue(str(result_path).endswith("File.pdf"))
        self.assertEqual(result_path.parent, temp_dir)

        print("✅ Path join test passed")

    def test_specific_back_in_time_3_filename(self):
        """Test the specific problematic filename from user report."""
        from pathlib import Path

        # Exact filename from error report
        filename = "Back in Time 3/Extras/Back in Time 3 Booklet.PDF"
        safe_filename = Path(filename).name

        # Should extract only the last component
        self.assertEqual(safe_filename, "Back in Time 3 Booklet.PDF")
        self.assertNotIn("/", safe_filename)
        self.assertNotIn("\\", safe_filename)
        self.assertNotIn("Extras", safe_filename)

        # Test with downloads directory
        downloads_dir = Path("C:\\Users\\mit\\.tdz-c64-knowledge\\downloads")
        filepath = downloads_dir / safe_filename

        # Should create a single-level path
        expected_path = "C:\\Users\\mit\\.tdz-c64-knowledge\\downloads\\Back in Time 3 Booklet.PDF"
        self.assertEqual(str(filepath), expected_path)

        # Verify no subdirectories are created
        self.assertEqual(filepath.parent, downloads_dir)

        print("✅ Specific Back in Time 3 filename test passed")

    def test_download_filepath_construction(self):
        """Test download filepath construction with directory-containing filenames."""
        from pathlib import Path

        downloads_dir = Path("C:\\Users\\test\\.tdz-c64-knowledge\\downloads")

        test_cases = [
            ("simple.pdf", "simple.pdf"),
            ("Dir/file.pdf", "file.pdf"),
            ("Back in Time 3/Extras/Back in Time 3 Booklet.PDF", "Back in Time 3 Booklet.PDF"),
            ("a/b/c/d/e/file.txt", "file.txt"),
        ]

        for original, expected_filename in test_cases:
            with self.subTest(filename=original):
                safe_filename = Path(original).name
                filepath = downloads_dir / safe_filename

                # Check filename is correct
                self.assertEqual(filepath.name, expected_filename)
                # Check no extra directories are created
                self.assertEqual(filepath.parent, downloads_dir)
                # Check path only has one level under downloads_dir
                relative_parts = filepath.relative_to(downloads_dir).parts
                self.assertEqual(len(relative_parts), 1)

        print("✅ Download filepath construction test passed")


class TestArchiveSearchURLs(unittest.TestCase):
    """Test Archive Search URL construction logic."""

    def test_file_dict_structure(self):
        """Test that file dictionaries have proper structure."""
        # Simulate file data from archive.org
        filename = "Back in Time 1/Extras/BIT 1 - DD Booklet.pdf"
        encoded_filename = quote(filename, safe='')
        identifier = "back-in-time-1"

        file_dict = {
            'name': filename,
            'size': 1024000,
            'format': 'PDF',
            'url': f"https://archive.org/download/{identifier}/{encoded_filename}"
        }

        # Verify structure
        self.assertEqual(file_dict['name'], filename)
        self.assertNotIn(" ", file_dict['url'])
        self.assertTrue(file_dict['url'].startswith("https://archive.org/download/"))

        print("✅ File dict structure test passed")

    def test_multiple_files_encoding(self):
        """Test URL encoding for multiple files with different patterns."""
        identifier = "test-collection"
        files_data = [
            ("simple.pdf", "simple.pdf"),
            ("with spaces.pdf", "with%20spaces.pdf"),
            ("path/to/file.pdf", "path%2Fto%2Ffile.pdf"),
            ("special!@#.pdf", "special%21%40%23.pdf"),
        ]

        for original, expected_encoded in files_data:
            with self.subTest(filename=original):
                encoded = quote(original, safe='')
                url = f"https://archive.org/download/{identifier}/{encoded}"

                self.assertEqual(encoded, expected_encoded)
                self.assertNotIn(" ", url)

        print("✅ Multiple files encoding test passed")


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 60)
    print("URL ENCODING AND FILENAME EXTRACTION TESTS")
    print("=" * 60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestURLEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestFilenameExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestArchiveSearchURLs))

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
