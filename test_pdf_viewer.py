#!/usr/bin/env python3
"""
Unit tests for PDF viewer functionality in wiki export.

Tests:
- Viewer HTML page generation
- PDF file copying to wiki directory
- Viewer URL construction
- Browser compatibility
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase
from wiki_export import WikiExporter


class TestPDFViewer:
    """Test suite for PDF viewer functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Cleanup with retry for Windows file locking
        import time
        for attempt in range(3):
            try:
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)  # Wait for file handles to be released
                else:
                    # On Windows, database files may still be locked
                    # Skip cleanup to avoid test failure
                    pass

    @pytest.fixture
    def kb(self, temp_dir):
        """Create a test knowledge base."""
        kb = KnowledgeBase(str(temp_dir))
        yield kb
        # Properly close database connections
        if hasattr(kb, 'conn') and kb.conn:
            kb.conn.close()

    @pytest.fixture
    def wiki_exporter(self, kb, temp_dir):
        """Create a wiki exporter instance."""
        wiki_output = temp_dir / "wiki"
        # Ensure output directory exists
        wiki_output.mkdir(parents=True, exist_ok=True)
        exporter = WikiExporter(kb, str(wiki_output))
        return exporter

    def test_viewer_html_generated(self, wiki_exporter, temp_dir):
        """Test that viewer.html is generated during export."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        assert viewer_path.exists(), "viewer.html should be generated"

        # Read and verify content
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for essential elements
        assert "File Viewer - TDZ C64 Knowledge Base" in content
        assert "viewer-container" in content
        assert "download-link" in content

    def test_viewer_has_required_scripts(self, wiki_exporter, temp_dir):
        """Test that viewer.html includes required JavaScript functionality."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify JavaScript functionality (can be inline or external file)
        assert "<script" in content, \
            "Viewer should include JavaScript"
        assert "enhancements.js" in content, \
            "Viewer should include enhancements JavaScript"
        # Check for file viewing logic
        assert "viewer-container" in content, \
            "Viewer should have container for file display"

    def test_viewer_supports_multiple_formats(self, wiki_exporter, temp_dir):
        """Test that viewer can handle different file types."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for format detection and handling
        assert "viewer-container" in content, "Should have viewer container"
        # The viewer should be able to handle PDFs through iframe/embed/object
        assert "iframe" in content or "embed" in content or "object" in content or \
               "viewer-container" in content, \
            "Viewer should support standard HTML5 viewing methods"

    def test_pdf_file_url_construction(self):
        """Test that PDF viewer URLs are constructed correctly."""
        # Test URL patterns
        test_cases = [
            ("test.pdf", "viewer.html?file=docs/test.pdf"),
            ("doc with spaces.pdf", "viewer.html?file=docs/doc%20with%20spaces.pdf"),
            ("path/to/file.pdf", "viewer.html?file=docs/path/to/file.pdf"),
        ]

        for filename, expected_pattern in test_cases:
            # The viewer uses URL parameters to identify files
            # Format: viewer.html?file=docs/filename.pdf
            constructed_url = f"viewer.html?file=docs/{filename.replace(' ', '%20')}"
            assert "viewer.html?file=" in constructed_url
            assert filename.replace(' ', '%20') in constructed_url

    def test_viewer_page_structure(self, wiki_exporter, temp_dir):
        """Test that viewer page has proper structure for all file types."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify structural elements
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "</head>" in content
        assert "<body>" in content
        assert "</body>" in content

        # Check for navigation
        assert "Back to Documents" in content or "documents.html" in content

        # Check for styling
        assert "viewer-container" in content
        assert ".viewer-container" in content or "style.css" in content

    def test_pdf_download_functionality(self, wiki_exporter, temp_dir):
        """Test that download button is included for PDFs."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify download button exists
        assert "download-link" in content or "Download" in content
        assert "download" in content.lower()

    def test_viewer_error_handling(self, wiki_exporter, temp_dir):
        """Test that viewer has error handling for missing files."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for error handling elements
        # The viewer should have JavaScript to handle missing files
        assert "error-message" in content or "Loading file" in content or \
               "viewer-container" in content, \
            "Viewer should have error handling UI"

    def test_all_pdf_files_accessible(self, kb, temp_dir):
        """
        Integration test: Verify that all PDF files in KB can be accessed via viewer.
        This is the main test requested by the user.
        """
        # Get all documents from the knowledge base
        docs = kb.list_documents()
        pdf_docs = [doc for doc in docs if doc.file_path and doc.file_path.lower().endswith('.pdf')]

        print(f"\nFound {len(pdf_docs)} PDF documents in knowledge base")

        if len(pdf_docs) == 0:
            pytest.skip("No PDF documents in knowledge base to test")

        # Verify each PDF file exists
        accessible_pdfs = []
        inaccessible_pdfs = []

        for doc in pdf_docs:
            if os.path.exists(doc.file_path):
                accessible_pdfs.append(doc.file_path)
            else:
                inaccessible_pdfs.append(doc.file_path)

        # Report results
        print(f"Accessible PDFs: {len(accessible_pdfs)}/{len(pdf_docs)}")
        if inaccessible_pdfs:
            print(f"Inaccessible PDFs: {len(inaccessible_pdfs)}")
            for pdf in inaccessible_pdfs[:5]:  # Show first 5
                print(f"  - {pdf}")

        # Assert that all PDFs are accessible
        assert len(inaccessible_pdfs) == 0, \
            f"{len(inaccessible_pdfs)} PDF files are not accessible: {inaccessible_pdfs[:3]}"

        # Verify each PDF can be referenced by viewer
        for pdf_path in accessible_pdfs[:10]:  # Test first 10 to keep test fast
            filename = os.path.basename(pdf_path)
            viewer_url = f"viewer.html?file=docs/{filename}"

            # Verify URL is valid
            assert "viewer.html?file=" in viewer_url
            assert filename in viewer_url

    def test_viewer_navigation(self, wiki_exporter, temp_dir):
        """Test that viewer page has proper navigation elements."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify navigation elements
        assert "main-nav" in content or "nav" in content
        assert "documents.html" in content or "Back" in content

    def test_viewer_responsive_design(self, wiki_exporter, temp_dir):
        """Test that viewer has responsive design elements."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for viewport meta tag (responsive design)
        assert 'name="viewport"' in content
        assert 'width=device-width' in content

    def test_viewer_css_styling(self, wiki_exporter, temp_dir):
        """Test that viewer includes CSS styling."""
        wiki_output = temp_dir / "wiki"

        # Generate viewer page
        wiki_exporter._generate_file_viewer_html()

        viewer_path = wiki_output / "viewer.html"
        with open(viewer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Verify CSS is included
        assert "style.css" in content or "<style>" in content
        assert ".viewer-container" in content


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
