"""
Unit tests for wiki export functionality.

Tests the new features added in v2.23.15:
- Document coordinate export (UMAP/t-SNE)
- File type detection (HTML/MD)
- Cluster document export
- HTML generation with explanation boxes
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wiki_export import WikiExporter
from server import KnowledgeBase


class TestDocumentCoordinateExport:
    """Test document coordinate export for similarity map."""

    def test_export_coordinates_with_umap(self, tmp_path):
        """Test coordinate export using UMAP."""
        # Create mock knowledge base with embeddings
        kb = Mock(spec=KnowledgeBase)
        kb.embeddings = {
            'doc1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'doc2': [0.2, 0.3, 0.4, 0.5, 0.6],
            'doc3': [0.3, 0.4, 0.5, 0.6, 0.7],
            'doc4': [0.4, 0.5, 0.6, 0.7, 0.8],
            'doc5': [0.5, 0.6, 0.7, 0.8, 0.9],
        }

        # Mock database connection and cursor for cluster data
        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = [
            ('doc1', 0, 'kmeans'),
            ('doc2', 0, 'kmeans'),
            ('doc3', 1, 'kmeans'),
            ('doc4', 1, 'kmeans'),
            ('doc5', 2, 'kmeans'),
        ]
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)

        documents_data = [
            {'id': 'doc1', 'title': 'Doc 1', 'filename': 'doc1.txt', 'file_type': 'text', 'tags': ['tag1'], 'total_chunks': 5},
            {'id': 'doc2', 'title': 'Doc 2', 'filename': 'doc2.txt', 'file_type': 'text', 'tags': ['tag2'], 'total_chunks': 3},
            {'id': 'doc3', 'title': 'Doc 3', 'filename': 'doc3.txt', 'file_type': 'text', 'tags': ['tag3'], 'total_chunks': 4},
            {'id': 'doc4', 'title': 'Doc 4', 'filename': 'doc4.txt', 'file_type': 'text', 'tags': ['tag4'], 'total_chunks': 6},
            {'id': 'doc5', 'title': 'Doc 5', 'filename': 'doc5.txt', 'file_type': 'text', 'tags': ['tag5'], 'total_chunks': 2},
        ]

        result = exporter._export_document_coordinates(documents_data)

        # Verify result structure
        assert 'documents' in result
        assert 'method' in result
        assert 'count' in result
        assert result['method'] in ['umap', 'tsne']
        assert result['count'] == 5
        assert len(result['documents']) == 5

        # Verify each document has required fields
        for doc in result['documents']:
            assert 'id' in doc
            assert 'title' in doc
            assert 'filename' in doc
            assert 'file_type' in doc
            assert 'tags' in doc
            assert 'total_chunks' in doc
            assert 'cluster' in doc
            assert 'x' in doc
            assert 'y' in doc

            # Verify coordinates are normalized (0-1000 range)
            assert 0 <= doc['x'] <= 1000
            assert 0 <= doc['y'] <= 1000

    def test_export_coordinates_no_embeddings(self, tmp_path):
        """Test coordinate export when no embeddings available."""
        kb = Mock(spec=KnowledgeBase)
        kb.embeddings = None

        exporter = WikiExporter(kb, tmp_path)
        documents_data = [
            {'id': 'doc1', 'title': 'Doc 1', 'filename': 'doc1.txt', 'file_type': 'text', 'tags': [], 'total_chunks': 5}
        ]

        result = exporter._export_document_coordinates(documents_data)

        assert result['method'] == 'none'
        assert result['count'] == 0
        assert len(result['documents']) == 0

    def test_export_coordinates_insufficient_data(self, tmp_path):
        """Test coordinate export with insufficient documents."""
        kb = Mock(spec=KnowledgeBase)
        kb.embeddings = {'doc1': [0.1, 0.2, 0.3]}

        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = [('doc1', 0, 'kmeans')]
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        documents_data = [
            {'id': 'doc1', 'title': 'Doc 1', 'filename': 'doc1.txt', 'file_type': 'text', 'tags': [], 'total_chunks': 5}
        ]

        result = exporter._export_document_coordinates(documents_data)

        assert result['method'] == 'none'
        assert len(result['documents']) == 0


class TestFileTypeDetection:
    """Test enhanced file type detection for HTML/MD files."""

    def test_detect_html_file_type(self, tmp_path):
        """Test HTML file type detection."""
        kb = Mock(spec=KnowledgeBase)
        kb.documents = {
            'doc1': Mock(
                doc_id='doc1',
                title='Test HTML',
                filename='test.html',
                file_type='text',
                total_pages=0,
                total_chunks=5,
                indexed_at='2024-01-01',
                tags=['html']
            )
        }

        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = [
            ('chunk1', 'Content 1', None),
            ('chunk2', 'Content 2', None),
        ]
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        documents = exporter._export_documents()

        assert len(documents) == 1
        assert documents[0]['file_type'] == 'html'
        assert documents[0]['filename'] == 'test.html'

    def test_detect_markdown_file_type(self, tmp_path):
        """Test Markdown file type detection."""
        kb = Mock(spec=KnowledgeBase)
        kb.documents = {
            'doc1': Mock(
                doc_id='doc1',
                title='Test Markdown',
                filename='test.md',
                file_type='text',
                total_pages=0,
                total_chunks=3,
                indexed_at='2024-01-01',
                tags=['markdown']
            ),
            'doc2': Mock(
                doc_id='doc2',
                title='Test Markdown 2',
                filename='test.markdown',
                file_type='text',
                total_pages=0,
                total_chunks=2,
                indexed_at='2024-01-01',
                tags=['markdown']
            )
        }

        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = [
            ('chunk1', 'Content 1', None),
        ]
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        documents = exporter._export_documents()

        assert len(documents) == 2
        assert documents[0]['file_type'] == 'markdown'
        assert documents[1]['file_type'] == 'markdown'

    def test_preserve_pdf_file_type(self, tmp_path):
        """Test that PDF file type is preserved."""
        kb = Mock(spec=KnowledgeBase)
        kb.documents = {
            'doc1': Mock(
                doc_id='doc1',
                title='Test PDF',
                filename='test.pdf',
                file_type='pdf',
                total_pages=10,
                total_chunks=15,
                indexed_at='2024-01-01',
                tags=['pdf']
            )
        }

        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = []
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        documents = exporter._export_documents()

        assert documents[0]['file_type'] == 'pdf'

    def test_preserve_text_file_type(self, tmp_path):
        """Test that plain text files remain as text."""
        kb = Mock(spec=KnowledgeBase)
        kb.documents = {
            'doc1': Mock(
                doc_id='doc1',
                title='Test Text',
                filename='test.txt',
                file_type='text',
                total_pages=0,
                total_chunks=5,
                indexed_at='2024-01-01',
                tags=['text']
            )
        }

        mock_cursor = Mock()
        mock_cursor.execute.return_value.fetchall.return_value = []
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        documents = exporter._export_documents()

        assert documents[0]['file_type'] == 'text'


class TestClusterDocumentExport:
    """Test cluster export with document lists."""

    def test_export_clusters_with_documents(self, tmp_path):
        """Test cluster export includes document lists."""
        kb = Mock(spec=KnowledgeBase)

        mock_cursor = Mock()

        # Mock clusters query
        def execute_side_effect(query, params=None):
            result = Mock()
            if 'SELECT DISTINCT algorithm' in query:
                result.fetchall.return_value = [('kmeans',), ('dbscan',)]
            elif 'COUNT(dc.doc_id)' in query:
                if params and params[0] == 'kmeans':
                    result.fetchall.return_value = [
                        ('cluster1', 0, 5),
                        ('cluster2', 1, 3),
                    ]
                else:  # dbscan
                    result.fetchall.return_value = [
                        ('cluster3', 0, 4),
                    ]
            elif 'SELECT d.doc_id, d.title, d.filename' in query:
                # Mock documents in cluster
                result.fetchall.return_value = [
                    ('doc1', 'Document 1', 'doc1.txt'),
                    ('doc2', 'Document 2', 'doc2.txt'),
                    ('doc3', 'Document 3', 'doc3.txt'),
                ]
            return result

        mock_cursor.execute.side_effect = execute_side_effect
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        clusters = exporter._export_clusters()

        # Verify structure
        assert 'kmeans' in clusters
        assert 'dbscan' in clusters

        # Verify kmeans clusters
        assert len(clusters['kmeans']) == 2
        assert clusters['kmeans'][0]['number'] == 0
        assert clusters['kmeans'][0]['doc_count'] == 5
        assert 'documents' in clusters['kmeans'][0]
        assert len(clusters['kmeans'][0]['documents']) == 3

        # Verify document structure
        doc = clusters['kmeans'][0]['documents'][0]
        assert 'id' in doc
        assert 'title' in doc
        assert 'filename' in doc

    def test_export_clusters_handles_bytes(self, tmp_path):
        """Test cluster export handles bytes cluster numbers."""
        kb = Mock(spec=KnowledgeBase)

        mock_cursor = Mock()

        def execute_side_effect(query, params=None):
            result = Mock()
            if 'SELECT DISTINCT algorithm' in query:
                result.fetchall.return_value = [('kmeans',)]
            elif 'COUNT(dc.doc_id)' in query:
                # Return cluster_num as bytes
                result.fetchall.return_value = [
                    ('cluster1', b'\x00\x00\x00\x00', 5),
                ]
            elif 'SELECT d.doc_id, d.title, d.filename' in query:
                result.fetchall.return_value = [
                    ('doc1', 'Document 1', 'doc1.txt'),
                ]
            return result

        mock_cursor.execute.side_effect = execute_side_effect
        mock_db_conn = Mock()
        mock_db_conn.cursor.return_value = mock_cursor
        kb.db_conn = mock_db_conn

        exporter = WikiExporter(kb, tmp_path)
        clusters = exporter._export_clusters()

        # Verify bytes were converted to int
        assert clusters['kmeans'][0]['number'] == 0
        assert isinstance(clusters['kmeans'][0]['number'], int)


class TestHTMLGeneration:
    """Test HTML generation with new features."""

    def test_documents_page_has_explanation_box(self, tmp_path):
        """Test documents page includes explanation box."""
        kb = Mock(spec=KnowledgeBase)
        exporter = WikiExporter(kb, tmp_path)

        exporter._generate_documents_browser_html()

        # Read generated file
        html_file = tmp_path / "documents.html"
        assert html_file.exists()

        content = html_file.read_text(encoding='utf-8')

        # Verify explanation box exists
        assert 'class="explanation-box"' in content
        assert '📚 About Documents' in content
        assert 'Filter by file type' in content

    def test_chunks_page_has_explanation_box(self, tmp_path):
        """Test chunks page includes explanation box."""
        kb = Mock(spec=KnowledgeBase)
        exporter = WikiExporter(kb, tmp_path)

        exporter._generate_chunks_browser_html()

        html_file = tmp_path / "chunks.html"
        assert html_file.exists()

        content = html_file.read_text(encoding='utf-8')

        assert 'class="explanation-box"' in content
        assert '🧩 About Chunks' in content
        assert '1500 words' in content
        assert '200-word overlap' in content

    def test_topics_page_has_explanation_box(self, tmp_path):
        """Test topics page includes explanation box."""
        kb = Mock(spec=KnowledgeBase)
        kb.db_conn = Mock()

        exporter = WikiExporter(kb, tmp_path)
        exporter._generate_topics_html()

        html_file = tmp_path / "topics.html"
        assert html_file.exists()

        content = html_file.read_text(encoding='utf-8')

        assert 'class="explanation-box"' in content
        assert '🔍 About Topics & Clusters' in content
        assert 'Topic Models' in content
        assert 'Document Clusters' in content
        assert 'Click on any document' in content

    def test_similarity_map_page_generation(self, tmp_path):
        """Test similarity map page is generated."""
        kb = Mock(spec=KnowledgeBase)
        kb.db_conn = Mock()

        exporter = WikiExporter(kb, tmp_path)
        exporter._generate_similarity_map_html()

        html_file = tmp_path / "similarity-map.html"
        assert html_file.exists()

        content = html_file.read_text(encoding='utf-8')

        # Verify key elements
        assert 'Similarity Map' in content or 'similarity' in content.lower()
        assert 'id="similarity-canvas"' in content
        assert 'coordinates.json' in content
        assert 'renderCanvas' in content
        assert 'zoom' in content.lower() or 'pan' in content.lower()

    def test_timeline_viewport_height(self, tmp_path):
        """Test timeline uses viewport height."""
        kb = Mock(spec=KnowledgeBase)
        kb.db_conn = Mock()

        exporter = WikiExporter(kb, tmp_path)
        exporter._generate_timeline_html()

        html_file = tmp_path / "timeline.html"
        content = html_file.read_text(encoding='utf-8')

        # Verify viewport height CSS
        assert 'calc(100vh - 400px)' in content
        assert 'min-height: 500px' in content

    def test_ask_ai_button_enhanced(self, tmp_path):
        """Test ASK AI button has enhanced styling."""
        kb = Mock(spec=KnowledgeBase)
        exporter = WikiExporter(kb, tmp_path)

        # Create necessary directories
        (tmp_path / "assets" / "css").mkdir(parents=True, exist_ok=True)

        # Generate CSS
        exporter._create_css()

        css_file = tmp_path / "assets" / "css" / "style.css"
        assert css_file.exists()

        content = css_file.read_text(encoding='utf-8')

        # Verify enhanced button styling
        assert '.bot-icon' in content
        assert '.bot-label' in content
        assert '@keyframes pulse' in content
        assert 'linear-gradient' in content
        assert 'width: 85px' in content
        assert 'height: 85px' in content


class TestJavaScriptGeneration:
    """Test JavaScript generation for new features."""

    def test_topics_js_has_clickable_clusters(self, tmp_path):
        """Test topics.js generates clickable cluster documents."""
        kb = Mock(spec=KnowledgeBase)
        exporter = WikiExporter(kb, tmp_path)

        # Create necessary directories
        (tmp_path / "assets" / "js").mkdir(parents=True, exist_ok=True)

        exporter._create_javascript()

        js_file = tmp_path / "assets" / "js" / "topics.js"
        assert js_file.exists()

        content = js_file.read_text(encoding='utf-8')

        # Verify clickable document list generation
        assert 'cluster.documents' in content
        assert 'slice(0, 10)' in content
        assert 'safeFilename' in content
        assert 'href="docs/' in content
        assert '...and' in content  # "...and N more"


@pytest.fixture
def tmp_path():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
