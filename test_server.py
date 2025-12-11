#!/usr/bin/env python3
"""
Tests for TDZ C64 Knowledge MCP Server
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
import pytest

# Import the server module
from server import KnowledgeBase, DocumentMeta, DocumentChunk


class TestKnowledgeBase:
    """Test suite for KnowledgeBase class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def kb(self, temp_data_dir):
        """Create a KnowledgeBase instance with temp directory."""
        return KnowledgeBase(temp_data_dir)

    @pytest.fixture
    def sample_text_file(self, temp_data_dir):
        """Create a sample text file for testing."""
        test_file = Path(temp_data_dir) / "test_doc.txt"
        content = """VIC-II Graphics Chip

The VIC-II chip controls all graphics and video output on the Commodore 64.
It has 47 registers mapped to memory locations $D000-$D02E.

Key features:
- 320x200 high resolution graphics
- 8 hardware sprites
- Smooth scrolling
- Raster interrupts

The SID chip handles sound generation with 3 voices.
"""
        test_file.write_text(content)
        return str(test_file)

    def test_initialization(self, temp_data_dir):
        """Test KnowledgeBase initialization."""
        kb = KnowledgeBase(temp_data_dir)
        assert kb.data_dir == Path(temp_data_dir)
        assert (kb.data_dir / "chunks").exists()
        assert len(kb.documents) == 0

    def test_add_text_document(self, kb, sample_text_file):
        """Test adding a text file."""
        doc = kb.add_document(sample_text_file, "Test VIC-II Doc", ["vic-ii", "graphics"])

        assert doc.filename == "test_doc.txt"
        assert doc.title == "Test VIC-II Doc"
        assert "vic-ii" in doc.tags
        assert "graphics" in doc.tags
        assert doc.total_chunks > 0
        assert doc.file_type == "text"

    def test_search_basic(self, kb, sample_text_file):
        """Test basic search functionality."""
        kb.add_document(sample_text_file, "Test Doc", ["test"])

        # Search for "VIC-II"
        results = kb.search("VIC-II", max_results=5)
        assert len(results) > 0
        assert "VIC-II" in results[0]["snippet"]

    def test_search_multiple_terms(self, kb, sample_text_file):
        """Test search with multiple terms."""
        kb.add_document(sample_text_file, "Test Doc", ["test"])

        results = kb.search("sprite scrolling", max_results=5)
        assert len(results) > 0

    def test_search_no_results(self, kb, sample_text_file):
        """Test search with no matching results."""
        kb.add_document(sample_text_file, "Test Doc", ["test"])

        results = kb.search("nonexistent keyword xyz", max_results=5)
        assert len(results) == 0

    def test_search_with_tag_filter(self, kb, sample_text_file):
        """Test search with tag filtering."""
        kb.add_document(sample_text_file, "Test Doc", ["vic-ii", "graphics"])

        # Search with matching tag
        results = kb.search("VIC-II", max_results=5, tags=["vic-ii"])
        assert len(results) > 0

        # Search with non-matching tag
        results = kb.search("VIC-II", max_results=5, tags=["sid"])
        assert len(results) == 0

    def test_list_documents(self, kb, sample_text_file):
        """Test listing documents."""
        kb.add_document(sample_text_file, "Test Doc", ["test"])

        docs = kb.list_documents()
        assert len(docs) == 1
        assert docs[0].filename == "test_doc.txt"

    def test_get_chunk(self, kb, sample_text_file):
        """Test retrieving a specific chunk."""
        doc = kb.add_document(sample_text_file, "Test Doc", ["test"])

        chunk = kb.get_chunk(doc.doc_id, 0)
        assert chunk is not None
        assert "VIC-II" in chunk.content

    def test_get_document(self, kb, sample_text_file):
        """Test retrieving full document content."""
        doc = kb.add_document(sample_text_file, "Test Doc", ["test"])

        content = kb.get_document_content(doc.doc_id)
        assert content is not None
        assert isinstance(content, str)
        assert "VIC-II" in content

    def test_remove_document(self, kb, sample_text_file):
        """Test removing a document."""
        doc = kb.add_document(sample_text_file, "Test Doc", ["test"])

        kb.remove_document(doc.doc_id)

        docs = kb.list_documents()
        assert len(docs) == 0

    def test_persistence(self, temp_data_dir, sample_text_file):
        """Test that data persists between instances."""
        # Add document with first instance
        kb1 = KnowledgeBase(temp_data_dir)
        doc = kb1.add_document(sample_text_file, "Test Doc", ["test"])
        doc_id = doc.doc_id

        # Create new instance and verify data is still there
        kb2 = KnowledgeBase(temp_data_dir)
        docs = kb2.list_documents()
        assert len(docs) == 1
        assert docs[0].doc_id == doc_id

    def test_chunking(self, kb, temp_data_dir):
        """Test document chunking with larger text."""
        # Create a large document
        large_file = Path(temp_data_dir) / "large_doc.txt"
        content = "This is a test sentence. " * 500  # ~2500 words
        large_file.write_text(content)

        doc = kb.add_document(str(large_file), "Large Doc", ["test"])

        # Should be split into multiple chunks
        assert doc.total_chunks >= 2

    def test_stats(self, kb, sample_text_file):
        """Test statistics calculation."""
        kb.add_document(sample_text_file, "Test Doc", ["vic-ii", "test"])

        stats = kb.get_stats()

        assert stats["total_documents"] == 1
        assert stats["total_chunks"] > 0
        assert stats["total_words"] > 0
        assert "text" in stats["file_types"]
        assert "vic-ii" in stats["all_tags"]


    def test_search_highlighting(self, kb, sample_text_file):
        """Test that search terms are highlighted in snippets."""
        kb.add_document(sample_text_file, "Test Doc", ["test"])

        results = kb.search("VIC-II", max_results=5)
        assert len(results) > 0

        # Check that terms are highlighted with **term**
        snippet = results[0]["snippet"]
        assert "**VIC" in snippet or "**vic" in snippet  # Case-insensitive highlighting

    def test_pdf_metadata(self, kb):
        """Test PDF metadata extraction."""
        # This test would require an actual PDF with metadata
        # For now, we'll just verify that the metadata fields exist in DocumentMeta
        import server
        from dataclasses import fields

        field_names = {f.name for f in fields(server.DocumentMeta)}
        assert "author" in field_names
        assert "subject" in field_names
        assert "creator" in field_names
        assert "creation_date" in field_names

    def test_custom_exceptions(self):
        """Test that custom exceptions are defined."""
        from server import (
            KnowledgeBaseError,
            DocumentNotFoundError,
            ChunkNotFoundError,
            UnsupportedFileTypeError,
            IndexCorruptedError
        )

        # Verify exception hierarchy
        assert issubclass(DocumentNotFoundError, KnowledgeBaseError)
        assert issubclass(ChunkNotFoundError, KnowledgeBaseError)
        assert issubclass(UnsupportedFileTypeError, KnowledgeBaseError)
        assert issubclass(IndexCorruptedError, KnowledgeBaseError)

    def test_unsupported_file_type(self, kb, temp_data_dir):
        """Test that unsupported file types raise appropriate exception."""
        from server import UnsupportedFileTypeError

        unsupported_file = Path(temp_data_dir) / "test.xyz"
        unsupported_file.write_text("some content")

        with pytest.raises(UnsupportedFileTypeError):
            kb.add_document(str(unsupported_file))


def test_imports():
    """Test that required modules can be imported."""
    import server
    assert hasattr(server, 'KnowledgeBase')
    assert hasattr(server, 'DocumentMeta')
    assert hasattr(server, 'DocumentChunk')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
