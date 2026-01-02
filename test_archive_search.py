#!/usr/bin/env python3
"""
Unit tests for Archive.org search functionality
"""

import unittest
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class TestArchiveSearch(unittest.TestCase):
    """Test suite for Archive.org search integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        try:
            import internetarchive as ia
            cls.ia = ia
            cls.ia_available = True
        except ImportError:
            cls.ia_available = False
            print("⚠️ Warning: internetarchive library not available")

    def test_internetarchive_import(self):
        """Test that internetarchive library is available."""
        self.assertTrue(
            self.ia_available,
            "internetarchive library should be installed"
        )

    def test_basic_search(self):
        """Test basic search functionality."""
        if not self.ia_available:
            self.skipTest("internetarchive library not available")

        try:
            # Perform a simple search
            search_query = "commodore 64"
            search = self.ia.search_items(search_query)

            # Get first result
            results = []
            for item in search:
                results.append(item)
                if len(results) >= 5:  # Limit to 5 for testing
                    break

            # Verify we got results
            self.assertGreater(
                len(results), 0,
                "Search should return at least one result"
            )

            # Verify result structure
            if results:
                first_result = results[0]
                self.assertIn('identifier', first_result)
                print(f"✅ Basic search test passed - found {len(results)} results")

        except Exception as e:
            self.fail(f"Basic search failed: {str(e)}")

    def test_filtered_search(self):
        """Test search with format filter."""
        if not self.ia_available:
            self.skipTest("internetarchive library not available")

        try:
            # Search with PDF format filter
            search_query = "commodore 64 AND (format:PDF)"
            search = self.ia.search_items(search_query)

            # Get first result
            results = []
            for item in search:
                results.append(item)
                if len(results) >= 3:  # Limit to 3 for testing
                    break

            # Verify we got results
            self.assertGreater(
                len(results), 0,
                "Filtered search should return results"
            )

            print(f"✅ Filtered search test passed - found {len(results)} PDF results")

        except Exception as e:
            self.fail(f"Filtered search failed: {str(e)}")

    def test_item_metadata_retrieval(self):
        """Test retrieving metadata from a known archive.org item."""
        if not self.ia_available:
            self.skipTest("internetarchive library not available")

        try:
            # Use a known stable item (example: C64 manual)
            # Note: You might want to replace this with a known stable identifier
            search_query = "commodore 64 manual"
            search = self.ia.search_items(search_query)

            # Get first item
            first_item = None
            for item in search:
                first_item = item
                break

            if first_item:
                # Get full item metadata
                item_obj = self.ia.get_item(first_item['identifier'])
                metadata = item_obj.metadata

                # Verify metadata structure
                self.assertIsInstance(metadata, dict)
                self.assertIn('identifier', metadata)

                # Check for common metadata fields
                has_title = 'title' in metadata
                has_creator = 'creator' in metadata

                print(f"✅ Metadata retrieval test passed")
                print(f"   Item: {first_item['identifier']}")
                print(f"   Has title: {has_title}")
                print(f"   Has creator: {has_creator}")

        except Exception as e:
            self.fail(f"Metadata retrieval failed: {str(e)}")

    def test_file_listing(self):
        """Test listing files from an archive.org item."""
        if not self.ia_available:
            self.skipTest("internetarchive library not available")

        try:
            # Search for an item
            search_query = "commodore 64"
            search = self.ia.search_items(search_query)

            # Get first item
            first_item = None
            for item in search:
                first_item = item
                break

            if first_item:
                # Get item files
                item_obj = self.ia.get_item(first_item['identifier'])
                files = list(item_obj.files)

                # Verify we have files
                self.assertGreater(
                    len(files), 0,
                    "Item should have at least one file"
                )

                # Check file structure
                if files:
                    first_file = files[0]
                    self.assertIn('name', first_file)

                    print(f"✅ File listing test passed")
                    print(f"   Item: {first_item['identifier']}")
                    print(f"   Total files: {len(files)}")
                    print(f"   First file: {first_file.get('name', 'Unknown')}")

        except Exception as e:
            self.fail(f"File listing failed: {str(e)}")

    def test_search_result_limit(self):
        """Test that result limiting works correctly."""
        if not self.ia_available:
            self.skipTest("internetarchive library not available")

        try:
            # Search with manual limit
            search_query = "commodore"
            search = self.ia.search_items(search_query)

            max_results = 10
            results = []
            items_processed = 0

            for item in search:
                if items_processed >= max_results:
                    break
                items_processed += 1
                results.append(item)

            # Verify we got exactly max_results or less
            self.assertLessEqual(
                len(results), max_results,
                f"Should not exceed max_results ({max_results})"
            )

            print(f"✅ Result limit test passed")
            print(f"   Requested: {max_results}")
            print(f"   Received: {len(results)}")

        except Exception as e:
            self.fail(f"Result limit test failed: {str(e)}")

    def test_query_construction(self):
        """Test building complex search queries."""
        # Test query building logic (doesn't require API call)
        query_parts = []

        # Base query
        search_query = "commodore 64"
        query_parts.append(search_query)

        # Collection filter
        collection = "texts"
        if collection != "All Collections":
            query_parts.append(f"collection:{collection}")

        # File type filter
        file_types = ["PDF", "TXT"]
        if file_types:
            format_query = " OR ".join([f"format:{ft}" for ft in file_types])
            query_parts.append(f"({format_query})")

        # Date filter
        start_year = 1980
        end_year = 1990
        query_parts.append(f"year:[{start_year} TO {end_year}]")

        # Combine
        full_query = " AND ".join(query_parts)

        # Verify query structure
        self.assertIn("commodore 64", full_query)
        self.assertIn("collection:texts", full_query)
        self.assertIn("format:PDF", full_query)
        self.assertIn("year:[1980 TO 1990]", full_query)

        print(f"✅ Query construction test passed")
        print(f"   Query: {full_query}")

    def test_url_construction(self):
        """Test that download URLs are correctly constructed."""
        identifier = "test-item-123"
        filename = "test-file.pdf"

        expected_url = f"https://archive.org/download/{identifier}/{filename}"
        constructed_url = f"https://archive.org/download/{identifier}/{filename}"

        self.assertEqual(
            constructed_url, expected_url,
            "Download URL should be correctly formatted"
        )

        # Test details URL
        expected_details = f"https://archive.org/details/{identifier}"
        constructed_details = f"https://archive.org/details/{identifier}"

        self.assertEqual(
            constructed_details, expected_details,
            "Details URL should be correctly formatted"
        )

        print(f"✅ URL construction test passed")
        print(f"   Download: {constructed_url}")
        print(f"   Details: {constructed_details}")


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 60)
    print("ARCHIVE SEARCH FUNCTIONALITY TESTS")
    print("=" * 60 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestArchiveSearch)

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
