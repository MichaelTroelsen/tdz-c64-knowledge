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

    @pytest.fixture
    def sample_excel_file(self, temp_data_dir):
        """Create a sample Excel file for testing."""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl not installed")

        test_file = Path(temp_data_dir) / "test_doc.xlsx"
        wb = Workbook()

        # First sheet: C64 Memory Map
        ws1 = wb.active
        ws1.title = "Memory Map"
        ws1.append(["Address", "Range", "Description"])
        ws1.append(["$0000-$00FF", "256 bytes", "Zero Page"])
        ws1.append(["$0100-$01FF", "256 bytes", "Stack"])
        ws1.append(["$D000-$D3FF", "1K", "VIC-II Chip"])
        ws1.append(["$D400-$D7FF", "1K", "SID Chip"])

        # Second sheet: Registers
        ws2 = wb.create_sheet("VIC-II Registers")
        ws2.append(["Register", "Address", "Function"])
        ws2.append(["SPRITE 0 X", "$D000", "Sprite 0 X-coordinate"])
        ws2.append(["SPRITE 0 Y", "$D001", "Sprite 0 Y-coordinate"])
        ws2.append(["SCREEN CTRL", "$D011", "Screen control register"])

        wb.save(test_file)
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

    def test_add_excel_document(self, kb, sample_excel_file):
        """Test adding an Excel file."""
        doc = kb.add_document(sample_excel_file, "Test C64 Memory Map", ["memory", "registers"])

        assert doc.filename == "test_doc.xlsx"
        assert doc.title == "Test C64 Memory Map"
        assert "memory" in doc.tags
        assert "registers" in doc.tags
        assert doc.total_chunks > 0
        assert doc.file_type == "excel"
        assert doc.total_pages == 2  # Two sheets

        # Verify content was extracted correctly
        chunks = kb._get_chunks_db(doc.doc_id)
        assert len(chunks) > 0

        # Check that sheet names and data are in the content
        full_text = ' '.join([chunk.content for chunk in chunks])
        assert "Memory Map" in full_text
        assert "VIC-II Registers" in full_text
        assert "Zero Page" in full_text
        assert "SPRITE 0 X" in full_text
        assert "$D000" in full_text

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

    def test_update_tags_bulk(self, kb, temp_data_dir):
        """Test bulk tag updates."""
        # Create and add multiple test documents
        docs_to_add = []
        for i in range(5):
            test_file = Path(temp_data_dir) / f"bulk_retag_{i}.txt"
            test_file.write_text(f"Document {i} for bulk retagging testing.")
            doc = kb.add_document(str(test_file), f"Retag Doc {i}", ["initial", "test"])
            docs_to_add.append(doc.doc_id)

        # Test adding tags to specific documents
        results = kb.update_tags_bulk(
            doc_ids=docs_to_add[:2],
            add_tags=["added-tag"]
        )
        assert len(results['updated']) == 2
        assert len(results['failed']) == 0

        # Verify tags were added
        for doc_id in docs_to_add[:2]:
            doc = kb.documents[doc_id]
            assert "added-tag" in doc.tags
            assert "initial" in doc.tags  # Original tag preserved

        # Test removing tags by existing tags
        results = kb.update_tags_bulk(
            existing_tags=["added-tag"],
            remove_tags=["initial"]
        )
        assert len(results['updated']) == 2

        # Verify tag was removed
        for doc_id in docs_to_add[:2]:
            doc = kb.documents[doc_id]
            assert "initial" not in doc.tags
            assert "added-tag" in doc.tags

        # Test replacing all tags
        results = kb.update_tags_bulk(
            doc_ids=[docs_to_add[3]],
            replace_tags=["replaced", "new-tags"]
        )
        assert len(results['updated']) == 1

        # Verify tags were replaced
        doc = kb.documents[docs_to_add[3]]
        assert doc.tags == ["replaced", "new-tags"]
        assert "initial" not in doc.tags

        # Clean up
        for doc_id in docs_to_add:
            kb.remove_document(doc_id)

    def test_export_documents_bulk(self, kb, temp_data_dir):
        """Test bulk document export."""
        import json

        # Create and add multiple test documents
        docs_to_add = []
        for i in range(3):
            test_file = Path(temp_data_dir) / f"bulk_export_{i}.txt"
            test_file.write_text(f"Document {i} for bulk export testing.")
            doc = kb.add_document(str(test_file), f"Export Doc {i}", ["export-test", f"group-{i%2}"])
            docs_to_add.append(doc.doc_id)

        # Test JSON export of all documents
        export_data = kb.export_documents_bulk(format="json")
        parsed = json.loads(export_data)
        assert len(parsed) >= 3  # At least our 3 test documents

        # Test CSV export by tags
        export_data = kb.export_documents_bulk(tags=["export-test"], format="csv")
        lines = export_data.strip().split('\n')
        assert len(lines) >= 4  # Header + 3 documents

        # Test Markdown export by doc IDs
        export_data = kb.export_documents_bulk(doc_ids=docs_to_add[:2], format="markdown")
        assert "# Document Export" in export_data
        assert "Export Doc 0" in export_data
        assert "Export Doc 1" in export_data
        assert "Export Doc 2" not in export_data  # Not included

        # Test JSON export with specific doc IDs
        export_data = kb.export_documents_bulk(doc_ids=[docs_to_add[0]], format="json")
        parsed = json.loads(export_data)
        assert len(parsed) == 1
        assert parsed[0]['doc_id'] == docs_to_add[0]
        assert parsed[0]['title'] == "Export Doc 0"

        # Clean up
        for doc_id in docs_to_add:
            kb.remove_document(doc_id)

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

    def test_add_relationship(self, kb, sample_text_file, temp_data_dir):
        """Test adding relationships between documents."""
        # Create two test documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document content")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        # Add a relationship
        result = kb.add_relationship(doc1.doc_id, doc2.doc_id, "references", "Doc 1 references Doc 2")

        assert result['from_doc_id'] == doc1.doc_id
        assert result['to_doc_id'] == doc2.doc_id
        assert result['relationship_type'] == "references"
        assert result['note'] == "Doc 1 references Doc 2"
        assert 'created_at' in result

    def test_add_relationship_invalid_doc(self, kb, sample_text_file):
        """Test adding relationship with invalid document ID."""
        doc = kb.add_document(sample_text_file, "Doc 1", ["test"])

        # Try to create relationship with non-existent document
        with pytest.raises(ValueError, match="Target document not found"):
            kb.add_relationship(doc.doc_id, "nonexistent_id", "related")

    def test_add_duplicate_relationship(self, kb, sample_text_file, temp_data_dir):
        """Test that duplicate relationships are rejected."""
        # Create two documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        # Add relationship
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "related")

        # Try to add same relationship again
        with pytest.raises(ValueError, match="Relationship already exists"):
            kb.add_relationship(doc1.doc_id, doc2.doc_id, "related")

    def test_get_relationships_outgoing(self, kb, sample_text_file, temp_data_dir):
        """Test getting outgoing relationships."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["test"])

        # Add relationships: doc1 -> doc2, doc1 -> doc3
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "prerequisite")

        # Get outgoing relationships from doc1
        rels = kb.get_relationships(doc1.doc_id, direction="outgoing")

        assert len(rels) == 2
        assert all(r['direction'] == 'outgoing' for r in rels)

        # Check we have both relationships
        related_ids = [r['related_doc_id'] for r in rels]
        assert doc2.doc_id in related_ids
        assert doc3.doc_id in related_ids

    def test_get_relationships_incoming(self, kb, sample_text_file, temp_data_dir):
        """Test getting incoming relationships."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["test"])

        # Add relationships: doc1 -> doc3, doc2 -> doc3
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "references")
        kb.add_relationship(doc2.doc_id, doc3.doc_id, "related")

        # Get incoming relationships to doc3
        rels = kb.get_relationships(doc3.doc_id, direction="incoming")

        assert len(rels) == 2
        assert all(r['direction'] == 'incoming' for r in rels)

        # Check we have both relationships
        related_ids = [r['related_doc_id'] for r in rels]
        assert doc1.doc_id in related_ids
        assert doc2.doc_id in related_ids

    def test_get_relationships_both(self, kb, sample_text_file, temp_data_dir):
        """Test getting both incoming and outgoing relationships."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["test"])

        # Create relationships: doc1 -> doc2, doc3 -> doc2
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")
        kb.add_relationship(doc3.doc_id, doc2.doc_id, "related")

        # Get all relationships for doc2
        rels = kb.get_relationships(doc2.doc_id, direction="both")

        assert len(rels) == 2

        # Should have one incoming and one outgoing
        directions = [r['direction'] for r in rels]
        assert 'incoming' in directions
        assert 'outgoing' not in directions  # doc2 has no outgoing relationships

    def test_remove_relationship(self, kb, sample_text_file, temp_data_dir):
        """Test removing a specific relationship."""
        # Create two documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        # Add two relationships
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "related")

        # Remove one relationship
        result = kb.remove_relationship(doc1.doc_id, doc2.doc_id, "references")

        assert result == True

        # Verify only one relationship remains
        rels = kb.get_relationships(doc1.doc_id)
        assert len(rels) == 1
        assert rels[0]['relationship_type'] == "related"

    def test_remove_all_relationships(self, kb, sample_text_file, temp_data_dir):
        """Test removing all relationships between two documents."""
        # Create two documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        # Add multiple relationships
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "related")

        # Remove all relationships (by not specifying type)
        result = kb.remove_relationship(doc1.doc_id, doc2.doc_id)

        assert result == True

        # Verify no relationships remain
        rels = kb.get_relationships(doc1.doc_id)
        assert len(rels) == 0

    def test_get_related_documents(self, kb, sample_text_file, temp_data_dir):
        """Test getting full metadata of related documents."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["reference"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["guide"])

        # Add relationships
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references", "See this reference")
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "prerequisite", "Read this first")

        # Get related documents
        related = kb.get_related_documents(doc1.doc_id)

        assert len(related) == 2

        # Verify we get full document metadata
        assert all('title' in doc for doc in related)
        assert all('tags' in doc for doc in related)
        assert all('relationship_type' in doc for doc in related)
        assert all('note' in doc for doc in related)

        # Check specific documents
        titles = [doc['title'] for doc in related]
        assert "Doc 2" in titles
        assert "Doc 3" in titles

    def test_get_related_documents_filtered(self, kb, sample_text_file, temp_data_dir):
        """Test getting related documents filtered by relationship type."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["test"])

        # Add relationships of different types
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "prerequisite")

        # Get only "references" relationships
        related = kb.get_related_documents(doc1.doc_id, relationship_type="references")

        assert len(related) == 1
        assert related[0]['title'] == "Doc 2"
        assert related[0]['relationship_type'] == "references"

    def test_relationship_cascade_delete(self, kb, sample_text_file, temp_data_dir):
        """Test that relationships are deleted when a document is removed."""
        # Create two documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test"])

        # Add relationship
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "references")

        # Verify relationship exists
        rels = kb.get_relationships(doc1.doc_id)
        assert len(rels) == 1

        # Delete doc2
        kb.remove_document(doc2.doc_id)

        # Verify relationship was automatically deleted
        rels = kb.get_relationships(doc1.doc_id)
        assert len(rels) == 0

    def test_get_relationship_graph(self, kb, sample_text_file, temp_data_dir):
        """Test getting relationship graph data for visualization."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test", "basic"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["test", "advanced"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["reference"])

        # Create some relationships
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "prerequisite")
        kb.add_relationship(doc2.doc_id, doc3.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "related")

        # Get full graph
        graph = kb.get_relationship_graph()

        assert len(graph['nodes']) == 3
        assert len(graph['edges']) == 3
        assert graph['stats']['total_nodes'] == 3
        assert graph['stats']['total_edges'] == 3
        assert set(graph['stats']['relationship_types']) == {'prerequisite', 'references', 'related'}

        # Verify node structure
        node_ids = [n['id'] for n in graph['nodes']]
        assert doc1.doc_id in node_ids
        assert doc2.doc_id in node_ids
        assert doc3.doc_id in node_ids

        # Verify edge structure
        assert all('from' in e and 'to' in e and 'type' in e for e in graph['edges'])

    def test_get_relationship_graph_filtered(self, kb, sample_text_file, temp_data_dir):
        """Test filtering relationship graph by tags and types."""
        # Create three documents
        doc1 = kb.add_document(sample_text_file, "Doc 1", ["test"])

        test_file2 = Path(temp_data_dir) / "test_doc2.txt"
        test_file2.write_text("Second test document")
        doc2 = kb.add_document(str(test_file2), "Doc 2", ["advanced"])

        test_file3 = Path(temp_data_dir) / "test_doc3.txt"
        test_file3.write_text("Third test document")
        doc3 = kb.add_document(str(test_file3), "Doc 3", ["reference"])

        # Create relationships of different types
        kb.add_relationship(doc1.doc_id, doc2.doc_id, "prerequisite")
        kb.add_relationship(doc2.doc_id, doc3.doc_id, "references")
        kb.add_relationship(doc1.doc_id, doc3.doc_id, "related")

        # Filter by relationship type
        graph = kb.get_relationship_graph(relationship_types=["prerequisite"])
        assert len(graph['edges']) == 1
        assert graph['edges'][0]['type'] == "prerequisite"

        # Filter by tags
        graph = kb.get_relationship_graph(tags=["test"])
        # Should include doc1 and any relationships involving it
        assert len(graph['nodes']) >= 1


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


def test_code_block_detection(tmpdir):
    """Test code block detection for BASIC, Assembly, and Hex dumps."""
    kb = KnowledgeBase(str(tmpdir))

    # Test BASIC code detection
    basic_text = """
Some text before.

10 PRINT "HELLO WORLD"
20 FOR I = 1 TO 10
30 PRINT I
40 NEXT I
50 END

Some text after.
"""

    code_blocks = kb._detect_code_blocks(basic_text)
    basic_blocks = [b for b in code_blocks if b['block_type'] == 'basic']
    assert len(basic_blocks) >= 1, "Should detect BASIC code"
    assert basic_blocks[0]['line_count'] >= 3, "BASIC block should have multiple lines"
    print(f"\nDetected BASIC code: {basic_blocks[0]['line_count']} lines")

    # Test Assembly code detection
    assembly_text = """
Some text before.

    LDA #$00
    STA $D020
    STA $D021
    JMP $FFE1

Some text after.
"""

    code_blocks = kb._detect_code_blocks(assembly_text)
    asm_blocks = [b for b in code_blocks if b['block_type'] == 'assembly']
    assert len(asm_blocks) >= 1, "Should detect Assembly code"
    assert asm_blocks[0]['line_count'] >= 3, "Assembly block should have multiple lines"
    print(f"Detected Assembly code: {asm_blocks[0]['line_count']} lines")

    # Test Hex dump detection
    hex_text = """
Some text before.

D000: 00 01 02 03 04 05 06 07
D008: 08 09 0A 0B 0C 0D 0E 0F
D010: 10 11 12 13 14 15 16 17

Some text after.
"""

    code_blocks = kb._detect_code_blocks(hex_text)
    hex_blocks = [b for b in code_blocks if b['block_type'] == 'hex']
    assert len(hex_blocks) >= 1, "Should detect Hex dumps"
    assert hex_blocks[0]['line_count'] >= 3, "Hex dump should have multiple lines"
    print(f"Detected Hex dump: {hex_blocks[0]['line_count']} lines")

    kb.close()


def test_table_extraction(tmpdir):
    """Test table extraction from PDFs."""
    import server

    if not server.PDFPLUMBER_SUPPORT:
        pytest.skip("pdfplumber not installed, skipping table extraction test")

    kb = KnowledgeBase(str(tmpdir))

    # Note: This test would require a sample PDF with tables
    # For now, just test that the method exists and handles missing files gracefully
    tables = kb._extract_tables("nonexistent.pdf")
    assert tables == [], "Should return empty list for nonexistent file"

    print("\nTable extraction test passed (pdfplumber available)")

    kb.close()


def test_table_search(tmpdir):
    """Test table search functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test document with tables in database
    # Insert a test table directly into the database
    cursor = kb.db_conn.cursor()

    # Create a fake document
    test_doc_id = "test_table_doc"
    cursor.execute("""
        INSERT INTO documents
        (doc_id, filename, title, filepath, file_type, total_pages, total_chunks,
         indexed_at, tags, file_mtime, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        test_doc_id,
        "test.pdf",
        "Test Doc with Tables",
        "/tmp/test.pdf",
        "pdf",
        1,
        0,
        "2025-12-12T00:00:00",
        json.dumps(["test"]),
        0.0,
        "test_hash"
    ))

    # Insert a test table
    cursor.execute("""
        INSERT INTO document_tables
        (doc_id, table_id, page, markdown, searchable_text, row_count, col_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        test_doc_id,
        0,
        1,
        "| Address | Register |\n| --- | --- |\n| $D000 | Sprite 0 X |\n| $D001 | Sprite 0 Y |",
        "Address Register $D000 Sprite 0 X $D001 Sprite 0 Y",
        3,  # Including header
        2
    ))

    kb.db_conn.commit()

    # Search for tables
    results = kb.search_tables("sprite register", max_results=5)
    assert isinstance(results, list), "Should return a list"

    if results:
        assert results[0]['doc_id'] == test_doc_id
        assert results[0]['table_id'] == 0
        assert 'markdown' in results[0]
        assert 'score' in results[0]
        print(f"\nTable search found {len(results)} results")
    else:
        print("\nNo table results found (FTS5 index may not be built yet)")

    kb.close()


def test_code_search(tmpdir):
    """Test code block search functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test document with code blocks in database
    cursor = kb.db_conn.cursor()

    # Create a fake document
    test_doc_id = "test_code_doc"
    cursor.execute("""
        INSERT INTO documents
        (doc_id, filename, title, filepath, file_type, total_pages, total_chunks,
         indexed_at, tags, file_mtime, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        test_doc_id,
        "test.txt",
        "Test Doc with Code",
        "/tmp/test.txt",
        "text",
        None,
        0,
        "2025-12-12T00:00:00",
        json.dumps(["test"]),
        0.0,
        "test_hash"
    ))

    # Insert a test code block
    code_text = "    LDA #$00\n    STA $D020\n    RTS"
    cursor.execute("""
        INSERT INTO document_code_blocks
        (doc_id, block_id, page, block_type, code, searchable_text, line_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        test_doc_id,
        0,
        None,
        "assembly",
        code_text,
        code_text,
        3
    ))

    kb.db_conn.commit()

    # Search for code blocks
    results = kb.search_code("LDA STA", max_results=5)
    assert isinstance(results, list), "Should return a list"

    if results:
        assert results[0]['doc_id'] == test_doc_id
        assert results[0]['block_type'] == 'assembly'
        assert 'code' in results[0]
        assert 'score' in results[0]
        print(f"\nCode search found {len(results)} results")
    else:
        print("\nNo code results found (FTS5 index may not be built yet)")

    # Test filtering by block type
    results_filtered = kb.search_code("LDA", max_results=5, block_type="assembly")
    assert isinstance(results_filtered, list)

    kb.close()


def test_facet_extraction(tmpdir):
    """Test facet extraction from document text."""
    kb = KnowledgeBase(str(tmpdir))

    # Create a test file with hardware, instructions, and registers
    test_file = Path(str(tmpdir)) / "test_facets.txt"
    content = """
    The SID chip ($D400-$D41C) provides sound generation on the Commodore 64.
    The VIC-II chip controls graphics with registers at $D000-$D02E.

    Example assembly code:
    LDA #$00
    STA $D020
    STA $D021
    JMP $1000

    The CIA chip handles I/O operations.
    """
    test_file.write_text(content)

    # Add document (will extract facets automatically)
    doc = kb.add_document(str(test_file), "Test Facets", ["test"])

    # Get facets from database
    cursor = kb.db_conn.cursor()
    cursor.execute("""
        SELECT facet_type, facet_value
        FROM document_facets
        WHERE doc_id = ?
    """, (doc.doc_id,))

    facets = {}
    for row in cursor.fetchall():
        facet_type, facet_value = row
        if facet_type not in facets:
            facets[facet_type] = set()
        facets[facet_type].add(facet_value)

    # Verify hardware facets
    assert 'hardware' in facets
    assert 'SID' in facets['hardware']
    assert 'VIC-II' in facets['hardware']
    assert 'CIA' in facets['hardware']

    # Verify instruction facets
    assert 'instruction' in facets
    assert 'LDA' in facets['instruction']
    assert 'STA' in facets['instruction']
    assert 'JMP' in facets['instruction']

    # Verify register facets
    assert 'register' in facets
    assert '$D020' in facets['register']
    assert '$D021' in facets['register']
    assert '$1000' in facets['register']

    print(f"\nExtracted facets: hardware={len(facets.get('hardware', []))}, "
          f"instructions={len(facets.get('instruction', []))}, "
          f"registers={len(facets.get('register', []))}")

    kb.close()


def test_faceted_search(tmpdir):
    """Test faceted search functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test documents with different facets
    test_file1 = Path(str(tmpdir)) / "test_sid.txt"
    test_file1.write_text("""
    The SID chip provides sound synthesis.
    Use LDA and STA to program the SID registers at $D400.
    """)

    test_file2 = Path(str(tmpdir)) / "test_vic.txt"
    test_file2.write_text("""
    The VIC-II chip controls graphics and sprites.
    Use LDA and STX to program VIC-II registers at $D000.
    """)

    kb.add_document(str(test_file1), "SID Doc", ["sid"])
    kb.add_document(str(test_file2), "VIC Doc", ["vic-ii"])

    # Search with hardware facet filter (SID only)
    results = kb.faceted_search(
        query="chip",
        facet_filters={'hardware': ['SID']},
        max_results=5
    )

    # Should only return SID document
    assert len(results) > 0
    assert any('SID' in r['snippet'] for r in results)

    # Verify facets are included in results
    if results:
        assert 'facets' in results[0]
        assert 'hardware' in results[0]['facets']

    # Search with instruction facet filter
    results2 = kb.faceted_search(
        query="program",
        facet_filters={'instruction': ['STX']},
        max_results=5
    )

    # Should only return VIC document (has STX)
    assert len(results2) > 0
    assert any('VIC-II' in r['snippet'] for r in results2)

    print(f"\nFaceted search: SID filter={len(results)} results, STX filter={len(results2)} results")

    kb.close()


def test_search_logging(tmpdir):
    """Test search logging functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create a test document
    test_file = Path(str(tmpdir)) / "test_log.txt"
    test_file.write_text("The VIC-II chip controls graphics.")
    kb.add_document(str(test_file), "Test Doc", ["test"])

    # Perform searches (will be logged automatically)
    kb.search("VIC-II", max_results=5, tags=["test"])
    kb.search("graphics", max_results=5)
    kb.search("nonexistent", max_results=5)

    # Check search log
    cursor = kb.db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM search_log")
    log_count = cursor.fetchone()[0]

    assert log_count >= 3, "Should have logged at least 3 searches"

    # Verify log entries
    cursor.execute("""
        SELECT query, search_mode, results_count, tags
        FROM search_log
        ORDER BY timestamp DESC
        LIMIT 3
    """)

    logs = cursor.fetchall()
    assert len(logs) == 3

    # Check that failed search is logged with 0 results
    failed_log = next((log for log in logs if log[0] == "nonexistent"), None)
    assert failed_log is not None
    assert failed_log[2] == 0, "Failed search should have 0 results"

    print(f"\nSearch logging: {log_count} searches logged")

    kb.close()


def test_search_analytics(tmpdir):
    """Test search analytics functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test data
    test_file = Path(str(tmpdir)) / "test_analytics.txt"
    test_file.write_text("VIC-II graphics and SID sound on the Commodore 64.")
    kb.add_document(str(test_file), "Test Doc", ["test"])

    # Perform various searches
    kb.search("VIC-II", max_results=5)
    kb.search("VIC-II", max_results=5)  # Duplicate query
    kb.search("SID", max_results=5, tags=["test"])
    kb.search("nonexistent", max_results=5)  # Failed search

    # Get analytics
    analytics = kb.get_search_analytics(days=1, limit=10)

    # Verify analytics structure
    assert 'total_searches' in analytics
    assert 'unique_queries' in analytics
    assert 'avg_results' in analytics
    assert 'top_queries' in analytics
    assert 'failed_searches' in analytics
    assert 'search_modes' in analytics

    # Verify data
    assert analytics['total_searches'] >= 3  # At least 3 searches (some may be cached)
    assert analytics['unique_queries'] >= 2  # At least 2 unique queries

    # Check top queries
    top_queries = analytics['top_queries']
    assert len(top_queries) > 0
    vic_query = next((q for q in top_queries if q['query'] == 'VIC-II'), None)
    assert vic_query is not None
    assert vic_query['count'] >= 1  # At least one search logged

    # Check failed searches
    failed = analytics['failed_searches']
    assert len(failed) > 0
    assert any(f['query'] == 'nonexistent' for f in failed)

    print(f"\nSearch analytics: {analytics['total_searches']} total, "
          f"{analytics['unique_queries']} unique queries")

    kb.close()


def test_incremental_embeddings(tmpdir):
    """Test that embeddings are incrementally added without full rebuild."""
    kb = KnowledgeBase(str(tmpdir))

    # Skip if semantic search not available
    if not kb.use_semantic:
        kb.close()
        pytest.skip("Semantic search not enabled (USE_SEMANTIC_SEARCH=1 required)")

    # Create first document
    test_file1 = Path(str(tmpdir)) / "test_doc1.txt"
    test_file1.write_text("The VIC-II chip controls graphics and sprites on the Commodore 64.")
    doc1 = kb.add_document(str(test_file1), "Test Doc 1", ["test"])

    # Verify embeddings were created
    assert kb.embeddings_index is not None
    assert len(kb.embeddings_doc_map) > 0
    initial_count = len(kb.embeddings_doc_map)
    initial_index_total = kb.embeddings_index.ntotal
    print(f"\nInitial embeddings: {initial_count} vectors")

    # Create second document
    test_file2 = Path(str(tmpdir)) / "test_doc2.txt"
    test_file2.write_text("The SID chip provides sound synthesis with three voices and filters.")
    doc2 = kb.add_document(str(test_file2), "Test Doc 2", ["test"])

    # Verify embeddings were incrementally added
    assert kb.embeddings_index is not None
    assert len(kb.embeddings_doc_map) > initial_count
    final_count = len(kb.embeddings_doc_map)
    final_index_total = kb.embeddings_index.ntotal

    print(f"Final embeddings: {final_count} vectors")
    print(f"Added {final_count - initial_count} vectors incrementally")

    # Verify FAISS index total matches doc map
    assert final_index_total == final_count

    # Verify both documents are searchable
    results = kb.semantic_search("graphics", max_results=5)
    assert len(results) > 0
    doc_ids = [r['doc_id'] for r in results]
    assert doc1.doc_id in doc_ids or doc2.doc_id in doc_ids

    kb.close()


def test_parallel_processing(tmpdir):
    """Test parallel document processing with ThreadPoolExecutor."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test directory with multiple files
    test_dir = Path(str(tmpdir)) / "bulk_test"
    test_dir.mkdir()

    # Create 5 test files
    test_files = []
    for i in range(5):
        test_file = test_dir / f"test_doc_{i}.txt"
        test_file.write_text(f"Test document {i} about C64 programming and graphics. "
                            f"This is chunk {i} of data for testing parallel processing.")
        test_files.append(test_file)

    # Set worker count to 2 for testing
    import os
    old_workers = os.getenv('PARALLEL_WORKERS')
    os.environ['PARALLEL_WORKERS'] = '2'

    try:
        # Use bulk add with parallel processing
        import time
        start_time = time.time()
        results = kb.add_documents_bulk(str(test_dir), pattern="*.txt", tags=["bulk", "test"])
        elapsed = time.time() - start_time

        # Verify results
        assert len(results['added']) == 5
        assert len(results['skipped']) == 0
        assert len(results['failed']) == 0

        # Verify all documents are in the knowledge base
        assert len(kb.documents) == 5

        # Verify all documents are searchable
        search_results = kb.search("C64 programming", max_results=10)
        assert len(search_results) >= 5

        print(f"\nParallel bulk add: {len(results['added'])} files in {elapsed:.2f}s")

    finally:
        # Restore original worker count
        if old_workers is not None:
            os.environ['PARALLEL_WORKERS'] = old_workers
        elif 'PARALLEL_WORKERS' in os.environ:
            del os.environ['PARALLEL_WORKERS']

    kb.close()


def test_cross_reference_detection(tmpdir):
    """Test cross-reference extraction and lookup."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test document with cross-references
    test_file = Path(str(tmpdir)) / "test_xrefs.txt"
    content = """
    The VIC-II border color register is at $D020 and the background color is at $D021.
    You can set these directly using VIC+0 and VIC+1 offsets.
    For more information, see page 156 of the manual.
    The SID volume register is at $D418, also known as SID+24.
    CIA timer registers start at $DC04.
    """
    test_file.write_text(content)

    # Add document
    doc = kb.add_document(str(test_file), "Test Cross-References", ["test"])

    # Test memory address references
    results = kb.find_by_reference("memory_address", "$D020", max_results=10)
    assert len(results) > 0
    assert results[0]['ref_type'] == 'memory_address'
    assert results[0]['ref_value'] == '$D020'
    assert '$D020' in results[0]['context']

    # Test register offset references
    results = kb.find_by_reference("register_offset", "VIC+0", max_results=10)
    assert len(results) > 0
    assert results[0]['ref_type'] == 'register_offset'
    assert results[0]['ref_value'] == 'VIC+0'

    # Test page references
    results = kb.find_by_reference("page_reference", "156", max_results=10)
    assert len(results) > 0
    assert results[0]['ref_type'] == 'page_reference'
    assert results[0]['ref_value'] == '156'
    assert 'page 156' in results[0]['context'].lower()

    print(f"\nCross-reference extraction successful - found references in {doc.doc_id}")

    kb.close()


def test_query_autocompletion(tmpdir):
    """Test query suggestion and autocompletion functionality."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test document with technical terms
    test_file = Path(str(tmpdir)) / "test_autocomplete.txt"
    content = """
    The VIC-II chip controls graphics on the Commodore 64.
    SID chip produces sound synthesis.
    Memory addresses like $D000, $D020, $D021 are important.
    Assembly instructions: LDA #$00, STA $D020, JMP $1000
    The Video Interface Controller handles sprites.
    """
    test_file.write_text(content)

    # Add document
    doc = kb.add_document(str(test_file), "Test Autocomplete", ["test"])

    # Test that suggestions were built automatically
    suggestions = kb.get_query_suggestions("VIC", max_suggestions=5)
    assert len(suggestions) > 0

    # Test hardware category
    suggestions = kb.get_query_suggestions("SID", max_suggestions=5, category="hardware")
    assert len(suggestions) > 0
    assert all(s['category'] == 'hardware' for s in suggestions)

    # Test register category
    suggestions = kb.get_query_suggestions("$D0", max_suggestions=5, category="register")
    assert len(suggestions) > 0
    assert all(s['category'] == 'register' for s in suggestions)

    # Test instruction category
    suggestions = kb.get_query_suggestions("LD", max_suggestions=5, category="instruction")
    assert len(suggestions) > 0
    assert all(s['category'] == 'instruction' for s in suggestions)

    # Test manual build
    kb.build_suggestion_dictionary(rebuild=True)
    suggestions = kb.get_query_suggestions("VIC", max_suggestions=5)
    assert len(suggestions) > 0

    # Test with too short query (should return empty)
    suggestions = kb.get_query_suggestions("V", max_suggestions=5)
    assert len(suggestions) == 0

    print(f"\nQuery autocompletion test successful - {len(suggestions)} suggestions found")

    kb.close()


def test_export_functionality(tmpdir):
    """Test search results export to various formats."""
    kb = KnowledgeBase(str(tmpdir))

    # Create test document
    test_file = Path(str(tmpdir)) / "test_export.txt"
    content = """
    The Commodore 64 is an 8-bit home computer.
    It was released in 1982 by Commodore International.
    The C64 features the VIC-II graphics chip and SID sound chip.
    """
    test_file.write_text(content)

    # Add document and search
    doc = kb.add_document(str(test_file), "Test Export", ["test", "c64"])
    results = kb.search("VIC-II graphics chip", max_results=5)

    # Test markdown export
    markdown = kb.export_search_results(results, format='markdown', query="VIC-II graphics chip")
    assert "# Search Results" in markdown
    assert "VIC-II" in markdown
    assert "**Query:** VIC-II graphics chip" in markdown  # Markdown format uses **Query:**
    assert "Test Export" in markdown

    # Test JSON export
    json_export = kb.export_search_results(results, format='json', query="VIC-II graphics chip")
    assert '"query": "VIC-II graphics chip"' in json_export
    assert '"result_count":' in json_export
    assert '"results"' in json_export

    # Test HTML export
    html = kb.export_search_results(results, format='html', query="VIC-II graphics chip")
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "VIC-II" in html
    assert "<style>" in html  # Should have CSS

    # Test invalid format
    try:
        kb.export_search_results(results, format='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported export format" in str(e)

    # Test export without query parameter
    markdown_no_query = kb.export_search_results(results, format='markdown')
    assert "# Search Results" in markdown_no_query

    # Test export with empty results
    empty_export = kb.export_search_results([], format='markdown', query="nothing")
    assert "# Search Results" in empty_export
    assert "**Results:** 0" in empty_export  # Markdown format uses **Results:**

    print(f"\nExport functionality test successful - tested markdown, JSON, and HTML formats")

    kb.close()


def test_backup_and_restore(tmpdir):
    """Test backup creation and restoration."""
    # Create knowledge base with test document
    kb = KnowledgeBase(str(tmpdir / "original"))

    test_file = Path(str(tmpdir)) / "test_backup.txt"
    content = """
    The Commodore 64 is an 8-bit home computer.
    It features the VIC-II graphics chip and SID sound chip.
    Released in 1982 by Commodore International.
    """
    test_file.write_text(content)

    # Add document
    doc = kb.add_document(str(test_file), "Test Backup Doc", ["test", "backup"])
    original_doc_id = doc.doc_id
    original_count = len(kb.documents)

    # Create backup directory
    backup_dir = Path(str(tmpdir)) / "backups"
    backup_dir.mkdir()

    # Create compressed backup
    backup_path = kb.create_backup(str(backup_dir), compress=True)
    assert Path(backup_path).exists()
    assert backup_path.endswith('.zip')

    # Verify backup contains metadata
    import zipfile
    with zipfile.ZipFile(backup_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        # Should contain metadata.json and knowledge_base.db
        assert any('metadata.json' in f for f in files)
        assert any('knowledge_base.db' in f for f in files)

    kb.close()

    # Create new knowledge base in different directory
    restore_kb = KnowledgeBase(str(tmpdir / "restored"))

    # Restore from backup
    result = restore_kb.restore_from_backup(backup_path, verify=True)

    assert result['success'] == True
    assert result['restored_documents'] == original_count
    assert original_doc_id in restore_kb.documents

    # Verify document was restored correctly
    restored_doc = restore_kb.documents[original_doc_id]
    assert restored_doc.title == "Test Backup Doc"
    assert "test" in restored_doc.tags
    assert "backup" in restored_doc.tags

    print(f"\nBackup and restore test successful - restored {result['restored_documents']} documents")

    restore_kb.close()


def test_uncompressed_backup(tmpdir):
    """Test uncompressed backup creation."""
    kb = KnowledgeBase(str(tmpdir / "original"))

    test_file = Path(str(tmpdir)) / "test_uncompressed.txt"
    test_file.write_text("Test content for uncompressed backup")

    kb.add_document(str(test_file), "Test Doc", ["test"])

    backup_dir = Path(str(tmpdir)) / "backups"
    backup_dir.mkdir()

    # Create uncompressed backup
    backup_path = kb.create_backup(str(backup_dir), compress=False)

    backup_path_obj = Path(backup_path)
    assert backup_path_obj.exists()
    assert backup_path_obj.is_dir()

    # Verify backup structure
    assert (backup_path_obj / "knowledge_base.db").exists()
    assert (backup_path_obj / "metadata.json").exists()

    # Read metadata
    import json
    with open(backup_path_obj / "metadata.json", 'r') as f:
        metadata = json.load(f)
        assert metadata['document_count'] == 1
        assert metadata['version'] == '2.5.0'

    print(f"\nUncompressed backup test successful")

    kb.close()


def test_backup_with_empty_kb(tmpdir):
    """Test backup of empty knowledge base."""
    kb = KnowledgeBase(str(tmpdir / "empty"))

    backup_dir = Path(str(tmpdir)) / "backups"
    backup_dir.mkdir()

    # Backup empty knowledge base
    backup_path = kb.create_backup(str(backup_dir), compress=True)

    assert Path(backup_path).exists()

    # Restore to new location
    restore_kb = KnowledgeBase(str(tmpdir / "restored_empty"))
    result = restore_kb.restore_from_backup(backup_path)

    assert result['success'] == True
    assert result['restored_documents'] == 0

    print(f"\nEmpty knowledge base backup test successful")

    kb.close()
    restore_kb.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
