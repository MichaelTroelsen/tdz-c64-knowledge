#!/usr/bin/env python3
"""
Unit tests for path security validation in KnowledgeBase.

Tests the ALLOWED_DOCS_DIRS security feature to ensure:
1. Paths within allowed directories are accepted
2. Paths outside allowed directories are rejected
3. Path traversal attacks are blocked
"""

import unittest
import os
import tempfile
from pathlib import Path
from server import KnowledgeBase


class TestPathSecurity(unittest.TestCase):
    """Test path security validation"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.kb = KnowledgeBase(self.test_dir)

    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_allowed_single_directory(self):
        """Test that files within a single allowed directory are accepted"""
        # Set allowed directory
        os.environ['ALLOWED_DOCS_DIRS'] = r'C:\Users\mit\Downloads\tdz-c64-knowledge-input'

        # Reinitialize to pick up environment variable
        self.kb = KnowledgeBase(self.test_dir)

        # Test valid path
        valid_path = r'C:\Users\mit\Downloads\tdz-c64-knowledge-input\test.pdf'
        self.assertTrue(
            self.kb._is_path_allowed(valid_path),
            f"Path should be allowed: {valid_path}"
        )

        # Test invalid path
        invalid_path = r'C:\Users\mit\Documents\test.pdf'
        self.assertFalse(
            self.kb._is_path_allowed(invalid_path),
            f"Path should be blocked: {invalid_path}"
        )

    def test_allowed_multiple_directories(self):
        """Test that files within multiple allowed directories are accepted"""
        # Set multiple allowed directories (comma-separated)
        os.environ['ALLOWED_DOCS_DIRS'] = (
            r'C:\Users\mit\Downloads\tdz-c64-knowledge-input,'
            r'C:\Users\mit\.tdz-c64-knowledge\scraped_docs'
        )

        # Reinitialize to pick up environment variable
        self.kb = KnowledgeBase(self.test_dir)

        # Test first allowed directory
        path1 = r'C:\Users\mit\Downloads\tdz-c64-knowledge-input\doc.pdf'
        self.assertTrue(
            self.kb._is_path_allowed(path1),
            f"Path in first allowed dir should be allowed: {path1}"
        )

        # Test second allowed directory
        path2 = r'C:\Users\mit\.tdz-c64-knowledge\scraped_docs\site\index.md'
        self.assertTrue(
            self.kb._is_path_allowed(path2),
            f"Path in second allowed dir should be allowed: {path2}"
        )

        # Test subdirectory of allowed directory
        subdir_path = r'C:\Users\mit\.tdz-c64-knowledge\scraped_docs\blog_chordian_net_20251218_193643\index.md'
        self.assertTrue(
            self.kb._is_path_allowed(subdir_path),
            f"Path in subdirectory of allowed dir should be allowed: {subdir_path}"
        )

        # Test path outside allowed directories
        invalid_path = r'C:\Users\mit\Documents\secret.pdf'
        self.assertFalse(
            self.kb._is_path_allowed(invalid_path),
            f"Path outside allowed dirs should be blocked: {invalid_path}"
        )

    def test_path_traversal_attack(self):
        """Test that path traversal attacks are blocked"""
        os.environ['ALLOWED_DOCS_DIRS'] = r'C:\Users\mit\Downloads\tdz-c64-knowledge-input'

        # Reinitialize to pick up environment variable
        self.kb = KnowledgeBase(self.test_dir)

        # Test path traversal attempts
        traversal_paths = [
            r'C:\Users\mit\Downloads\tdz-c64-knowledge-input\..\..\..\Windows\System32\config\SAM',
            r'C:\Users\mit\Downloads\tdz-c64-knowledge-input\..\..\Documents\secrets.txt',
        ]

        for path in traversal_paths:
            self.assertFalse(
                self.kb._is_path_allowed(path),
                f"Path traversal should be blocked: {path}"
            )

    def test_no_restrictions_when_not_configured(self):
        """Test that all paths are allowed when ALLOWED_DOCS_DIRS is not set"""
        # Clear environment variable
        if 'ALLOWED_DOCS_DIRS' in os.environ:
            del os.environ['ALLOWED_DOCS_DIRS']

        # Reinitialize to pick up environment variable
        self.kb = KnowledgeBase(self.test_dir)

        # Any path should be allowed
        test_paths = [
            r'C:\Users\mit\Documents\test.pdf',
            r'C:\Windows\System32\test.pdf',
            r'D:\SomeOtherDrive\test.pdf',
        ]

        for path in test_paths:
            self.assertTrue(
                self.kb._is_path_allowed(path),
                f"All paths should be allowed when not configured: {path}"
            )

    def test_case_insensitive_windows_paths(self):
        """Test that Windows path comparison is case-insensitive"""
        os.environ['ALLOWED_DOCS_DIRS'] = r'C:\Users\mit\Downloads\tdz-c64-knowledge-input'

        # Reinitialize to pick up environment variable
        self.kb = KnowledgeBase(self.test_dir)

        # Test different case variations
        test_paths = [
            r'c:\users\mit\downloads\tdz-c64-knowledge-input\test.pdf',  # lowercase
            r'C:\USERS\MIT\DOWNLOADS\TDZ-C64-KNOWLEDGE-INPUT\test.pdf',  # uppercase
            r'C:\Users\Mit\Downloads\TDZ-C64-Knowledge-Input\test.pdf',  # mixed case
        ]

        for path in test_paths:
            self.assertTrue(
                self.kb._is_path_allowed(path),
                f"Case-insensitive path should be allowed: {path}"
            )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
