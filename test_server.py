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
        kb = KnowledgeBase(temp_data_dir)
        yield kb
        # Clean up: close database connection
        kb.close()

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
        assert (kb.data_dir / "knowledge_base.db").exists()
        assert kb.db_conn is not None
        assert len(kb.documents) == 0
        kb.close()

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
        kb1.close()

        # Create new instance and verify data is still there
        kb2 = KnowledgeBase(temp_data_dir)
        docs = kb2.list_documents()
        assert len(docs) == 1
        assert docs[0].doc_id == doc_id
        kb2.close()

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
            IndexCorruptedError,
            SecurityError
        )

        # Verify exception hierarchy
        assert issubclass(DocumentNotFoundError, KnowledgeBaseError)
        assert issubclass(ChunkNotFoundError, KnowledgeBaseError)
        assert issubclass(UnsupportedFileTypeError, KnowledgeBaseError)
        assert issubclass(IndexCorruptedError, KnowledgeBaseError)
        assert issubclass(SecurityError, KnowledgeBaseError)

    def test_unsupported_file_type(self, kb, temp_data_dir):
        """Test that unsupported file types raise appropriate exception."""
        from server import UnsupportedFileTypeError

        unsupported_file = Path(temp_data_dir) / "test.xyz"
        unsupported_file.write_text("some content")

        with pytest.raises(UnsupportedFileTypeError):
            kb.add_document(str(unsupported_file))

    def test_path_traversal_protection(self, temp_data_dir):
        """Test path traversal protection with allowed directories."""
        import os
        from server import KnowledgeBase, SecurityError

        # Create allowed and restricted directories
        allowed_dir = Path(temp_data_dir) / "allowed"
        allowed_dir.mkdir(parents=True, exist_ok=True)

        restricted_dir = Path(temp_data_dir) / "restricted"
        restricted_dir.mkdir(parents=True, exist_ok=True)

        # Create test files
        allowed_file = allowed_dir / "test.txt"
        allowed_file.write_text("Allowed content")

        restricted_file = restricted_dir / "test.txt"
        restricted_file.write_text("Restricted content")

        # Initialize KB with allowed directory restriction
        os.environ['ALLOWED_DOCS_DIRS'] = str(allowed_dir)
        kb_restricted = KnowledgeBase(str(Path(temp_data_dir) / "kb_restricted"))

        # Test 1: Adding file from allowed directory should succeed
        doc = kb_restricted.add_document(str(allowed_file), title="Allowed Doc")
        assert doc.filename == "test.txt"
        kb_restricted.remove_document(doc.doc_id)

        # Test 2: Adding file from restricted directory should fail
        with pytest.raises(SecurityError) as exc_info:
            kb_restricted.add_document(str(restricted_file))
        assert "outside allowed directories" in str(exc_info.value).lower()

        # Test 3: Path traversal attempt should fail
        traversal_path = str(allowed_dir / ".." / "restricted" / "test.txt")
        with pytest.raises(SecurityError):
            kb_restricted.add_document(traversal_path)

        # Clean up
        del os.environ['ALLOWED_DOCS_DIRS']
        kb_restricted.close()

    def test_duplicate_detection(self, kb, temp_data_dir):
        """Test content-based duplicate detection."""
        # Create a document
        doc1_path = Path(temp_data_dir) / "original.txt"
        doc1_path.write_text("This is unique content about the VIC-II chip and sprites.")

        # Add first document
        doc1 = kb.add_document(str(doc1_path), title="Original", tags=["test"])
        original_doc_id = doc1.doc_id

        # Create an exact duplicate at a different path
        doc2_path = Path(temp_data_dir) / "duplicate.txt"
        doc2_path.write_text("This is unique content about the VIC-II chip and sprites.")

        # Add duplicate - should return existing document
        doc2 = kb.add_document(str(doc2_path), title="Duplicate", tags=["test"])

        # Should have same doc_id (content-based)
        assert doc2.doc_id == original_doc_id
        assert doc2.filepath == doc1.filepath  # Returns first document

        # Should only have one document in the knowledge base
        docs = kb.list_documents()
        matching_docs = [d for d in docs if d.doc_id == original_doc_id]
        assert len(matching_docs) == 1

        # Create a document with different content
        doc3_path = Path(temp_data_dir) / "different.txt"
        doc3_path.write_text("This is completely different content about the SID chip.")

        # Add different document - should get new doc_id
        doc3 = kb.add_document(str(doc3_path), title="Different", tags=["test"])
        assert doc3.doc_id != original_doc_id

        # Clean up
        kb.remove_document(doc1.doc_id)
        kb.remove_document(doc3.doc_id)

    def test_add_documents_bulk(self, kb, temp_data_dir):
        """Test bulk document addition."""
        # Create multiple test files in a subdirectory
        test_dir = Path(temp_data_dir) / "test_docs"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create PDF and TXT files
        for i in range(3):
            txt_file = test_dir / f"doc_{i}.txt"
            txt_file.write_text(f"This is test document number {i} about VIC-II and sprites.")

        # Bulk add all txt files
        results = kb.add_documents_bulk(str(test_dir), pattern="*.txt", tags=["bulk-test"], recursive=False)

        # Verify results
        assert len(results['added']) == 3
        assert len(results['skipped']) == 0
        assert len(results['failed']) == 0

        # Verify documents are in the knowledge base
        docs = kb.list_documents()
        bulk_docs = [d for d in docs if "bulk-test" in d.tags]
        assert len(bulk_docs) == 3

        # Test with recursive pattern
        subdir = test_dir / "subdir"
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "nested.txt").write_text("Nested document about SID chip.")

        results = kb.add_documents_bulk(str(test_dir), pattern="**/*.txt", tags=["nested-test"], recursive=True)
        # Should find 1 new document (the nested one, others are duplicates)
        assert len(results['added']) >= 1

        # Clean up
        for doc in kb.list_documents():
            if "bulk-test" in doc.tags or "nested-test" in doc.tags:
                kb.remove_document(doc.doc_id)

    def test_remove_documents_bulk(self, kb, temp_data_dir):
        """Test bulk document removal."""
        # Create and add multiple test documents
        docs_to_add = []
        for i in range(5):
            test_file = Path(temp_data_dir) / f"bulk_remove_{i}.txt"
            test_file.write_text(f"Document {i} for bulk removal testing.")
            if i < 3:
                doc = kb.add_document(str(test_file), f"Doc {i}", ["remove-test", "group-a"])
            else:
                doc = kb.add_document(str(test_file), f"Doc {i}", ["remove-test", "group-b"])
            docs_to_add.append(doc.doc_id)

        # Test removal by doc_ids
        results = kb.remove_documents_bulk(doc_ids=docs_to_add[:2])
        assert len(results['removed']) == 2
        assert len(results['failed']) == 0

        # Test removal by tags
        results = kb.remove_documents_bulk(tags=["group-b"])
        assert len(results['removed']) == 2  # 2 docs with group-b tag

        # Verify only 1 document remains (index 2, which has group-a)
        remaining_docs = [d for d in kb.list_documents() if "remove-test" in d.tags]
        assert len(remaining_docs) == 1

        # Clean up remaining document
        kb.remove_document(remaining_docs[0].doc_id)

    def test_progress_reporting(self, kb, temp_data_dir):
        """Test progress reporting for add_document and add_documents_bulk."""
        from server import ProgressUpdate

        # Test progress reporting for add_document
        test_file = Path(temp_data_dir) / "progress_test.txt"
        test_file.write_text("Testing progress reporting for document ingestion.")

        progress_updates = []

        def progress_callback(update: ProgressUpdate):
            """Capture progress updates."""
            progress_updates.append(update)

        # Add document with progress callback
        doc = kb.add_document(str(test_file), title="Progress Test", tags=["test"],
                             progress_callback=progress_callback)

        # Verify progress updates were called
        assert len(progress_updates) > 0
        assert progress_updates[0].operation == "add_document"
        assert progress_updates[0].current == 0
        assert progress_updates[0].total == 4
        assert progress_updates[-1].current == 4  # Final update
        assert progress_updates[-1].percentage == 100.0

        # Clean up
        kb.remove_document(doc.doc_id)

        # Test progress reporting for add_documents_bulk
        test_dir = Path(temp_data_dir) / "bulk_progress"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 test files
        for i in range(3):
            bulk_file = test_dir / f"bulk_{i}.txt"
            bulk_file.write_text(f"Bulk test file {i}")

        progress_updates.clear()

        # Bulk add with progress callback
        results = kb.add_documents_bulk(str(test_dir), pattern="*.txt",
                                       tags=["bulk-progress-test"],
                                       progress_callback=progress_callback)

        # Verify bulk progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0].operation == "add_documents_bulk"
        assert progress_updates[0].total == 3
        assert progress_updates[-1].percentage == 100.0

        # Clean up
        for doc in kb.list_documents():
            if "bulk-progress-test" in doc.tags:
                kb.remove_document(doc.doc_id)


def test_query_preprocessing(tmpdir):
    """Test query preprocessing (stemming, stopword removal)."""
    from server import KnowledgeBase

    kb = KnowledgeBase(str(tmpdir))

    # Test with preprocessing enabled (default)
    if kb.use_preprocessing:
        # Test stopword removal
        tokens = kb._preprocess_text("the quick brown fox")
        assert "the" not in tokens  # Stopword removed
        assert "quick" in tokens or "quickli" in tokens  # May be stemmed

        # Test stemming
        tokens = kb._preprocess_text("running runs runner")
        # All should stem to "run"
        assert all(t.startswith("run") for t in tokens)

        # Test hyphenated technical terms preserved
        tokens = kb._preprocess_text("VIC-II chip")
        assert "vic-ii" in tokens
        assert "chip" in tokens

        # Test that numbers are preserved
        tokens = kb._preprocess_text("6502 processor")
        assert "6502" in tokens
        assert "processor" in tokens or "process" in tokens  # May be stemmed

    kb.close()


def test_health_check(tmpdir):
    """Test health check functionality."""
    # Create knowledge base
    kb = KnowledgeBase(str(tmpdir))

    # Create sample document
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("Test content for health check")

    # Add document
    doc = kb.add_document(str(test_file), "Test Doc", ["test"])
    assert doc is not None

    # Run health check
    health = kb.health_check()

    # Verify structure
    assert 'status' in health
    assert 'issues' in health
    assert 'metrics' in health
    assert 'features' in health
    assert 'database' in health
    assert 'performance' in health
    assert 'message' in health

    # Check status
    assert health['status'] in ['healthy', 'warning', 'error']

    # Check metrics
    assert health['metrics']['documents'] >= 1
    assert health['metrics']['chunks'] >= 1
    assert health['metrics']['total_words'] > 0

    # Check database info
    assert 'integrity' in health['database']
    assert health['database']['integrity'] == 'ok'

    # Check features
    assert 'fts5_enabled' in health['features']
    assert 'semantic_search_enabled' in health['features']
    assert 'bm25_enabled' in health['features']

    # Check performance
    assert 'cache_enabled' in health['performance']

    print(f"\nHealth check status: {health['status']}")
    print(f"Message: {health['message']}")

    kb.close()


def test_enhanced_snippet_extraction(tmpdir):
    """Test enhanced snippet extraction with term density scoring."""
    # Create knowledge base
    kb = KnowledgeBase(str(tmpdir))

    # Create sample document
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("VIC-II graphics chip. The VIC-II has sprites for moving graphics on screen.")

    # Add document
    doc = kb.add_document(str(test_file), "Test Doc", ["test"])
    assert doc is not None

    # Enable FTS5 for testing
    os.environ['USE_FTS5'] = '1'

    # Search with terms that appear multiple times
    results = kb.search("VIC-II graphics sprites", max_results=1)

    if results:
        snippet = results[0]['snippet']

        # Check that snippet contains highlighted terms
        assert '**' in snippet, "Snippet should contain highlighted terms"

        # Check that snippet doesn't cut mid-word (should have ellipsis or complete sentences)
        # Enhanced snippet should favor complete sentences
        if snippet.startswith('...'):
            # If it starts with ellipsis, the next character should be uppercase or space
            assert len(snippet) > 3
        if snippet.endswith('...'):
            # If it ends with ellipsis, it was properly truncated
            assert len(snippet) > 3

        print(f"\nEnhanced snippet: {snippet[:100]}...")

    kb.close()


def test_hybrid_search(tmpdir):
    """Test hybrid search combining FTS5 and semantic search."""
    # Create knowledge base
    kb = KnowledgeBase(str(tmpdir))

    # Skip if semantic search not available
    if not kb.use_semantic:
        kb.close()
        pytest.skip("Semantic search not enabled (USE_SEMANTIC_SEARCH=1 required)")

    # Create sample document
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("VIC-II graphics chip with hardware sprites for moving graphics.")

    # Add document
    doc = kb.add_document(str(test_file), "VIC-II Test Doc", ["test", "vic-ii"])
    assert doc is not None

    # Enable FTS5
    os.environ['USE_FTS5'] = '1'

    try:
        # Test hybrid search with different weights
        results = kb.hybrid_search("graphics sprites", max_results=5, semantic_weight=0.3)

        # Should return results
        assert isinstance(results, list)

        if results:
            # Check result structure
            assert 'doc_id' in results[0]
            assert 'score' in results[0]
            assert 'fts_score' in results[0]
            assert 'semantic_score' in results[0]
            assert 'snippet' in results[0]

            # Scores should be normalized (0-1 range)
            assert 0 <= results[0]['fts_score'] <= 1
            assert 0 <= results[0]['semantic_score'] <= 1
            assert 0 <= results[0]['score'] <= 1

            # Results should be sorted by score descending
            for i in range(len(results) - 1):
                assert results[i]['score'] >= results[i + 1]['score']

            print(f"\nHybrid search results: {len(results)}")
            print(f"Top result score: {results[0]['score']:.4f} (FTS: {results[0]['fts_score']:.4f}, Semantic: {results[0]['semantic_score']:.4f})")

        # Test with different semantic weight
        results_high_semantic = kb.hybrid_search("graphics", max_results=5, semantic_weight=0.8)
        assert isinstance(results_high_semantic, list)

    except RuntimeError as e:
        if "Semantic search not available" in str(e):
            kb.close()
            pytest.skip("Semantic search dependencies not installed")
        raise
    finally:
        kb.close()


def test_imports():
    """Test that required modules can be imported."""
    import server
    assert hasattr(server, 'KnowledgeBase')
    assert hasattr(server, 'DocumentMeta')
    assert hasattr(server, 'DocumentChunk')


def test_ocr_configuration():
    """Test OCR configuration and Tesseract availability."""
    import server

    # Check if OCR libraries are available
    assert server.OCR_SUPPORT is not None, "OCR_SUPPORT should be defined"

    if server.OCR_SUPPORT:
        # Libraries are installed, check if Tesseract is available
        import pytesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"\nTesseract found: version {version}")
            assert version is not None, "Tesseract version should be retrievable"
        except Exception as e:
            pytest.skip(f"Tesseract not found: {e}")
    else:
        pytest.skip("OCR libraries (pytesseract/pdf2image/Pillow) not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
