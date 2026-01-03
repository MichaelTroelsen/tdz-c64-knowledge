#!/usr/bin/env python3
"""
Wiki Export Script

Exports the TDZ C64 Knowledge Base to a static HTML/JavaScript wiki.

Features:
- Exports all documents, chunks, entities, topics, clusters, events
- Generates search index for client-side search
- Creates navigation structure
- Builds interactive visualizations
- Produces fully static site (no server needed)

Usage:
    python wiki_export.py --output wiki/
"""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import html
import re


class WikiExporter:
    """Exports knowledge base to static HTML wiki."""

    def __init__(self, kb: KnowledgeBase, output_dir: str):
        self.kb = kb
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "docs"
        self.assets_dir = self.output_dir / "assets"
        self.data_dir = self.assets_dir / "data"

        # Statistics
        self.stats = {
            'documents': 0,
            'chunks': 0,
            'entities': 0,
            'topics': 0,
            'clusters': 0,
            'events': 0,
            'export_date': datetime.now().isoformat()
        }

    def export(self):
        """Main export function."""
        print("=" * 60)
        print("TDZ C64 Knowledge Base - Wiki Export")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")

        # Create directory structure
        print("\n[1/7] Creating directory structure...")
        self._create_directories()

        # Export data
        print("[2/7] Exporting documents...")
        documents_data = self._export_documents()

        print("[3/7] Exporting entities...")
        entities_data = self._export_entities()

        print("[4/7] Exporting topics and clusters...")
        topics_data = self._export_topics()
        clusters_data = self._export_clusters()

        print("[5/7] Exporting events...")
        events_data = self._export_events()

        print("[6/9] Building search index...")
        search_index = self._build_search_index(documents_data)

        print("[7/9] Generating navigation...")
        navigation = self._build_navigation(documents_data)

        # Export chunks data
        print("[8/9] Exporting chunks...")
        chunks_data = self._export_chunks()

        # Save data files
        print("\nSaving data files...")
        self._save_json('documents.json', documents_data)
        self._save_json('entities.json', entities_data)
        self._save_json('topics.json', topics_data)
        self._save_json('clusters.json', clusters_data)
        self._save_json('events.json', events_data)
        self._save_json('search-index.json', search_index)
        self._save_json('navigation.json', navigation)
        self._save_json('chunks.json', chunks_data)
        self._save_json('stats.json', self.stats)

        # Generate HTML pages
        print("\nGenerating HTML pages...")
        self._generate_html_pages(documents_data)

        # Copy PDFs
        print("[9/10] Copying PDF files...")
        self._copy_pdfs(documents_data)

        # Generate articles
        articles_data = self._generate_articles(entities_data)

        # Copy static assets
        print("\nCopying static assets...")
        self._copy_static_assets()

        # Print summary
        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  Documents: {self.stats['documents']}")
        print(f"  Chunks: {self.stats['chunks']}")
        print(f"  Entities: {self.stats['entities']}")
        print(f"  Topics: {self.stats['topics']}")
        print(f"  Clusters: {self.stats['clusters']}")
        print(f"  Events: {self.stats['events']}")
        print(f"  Articles: {self.stats.get('articles', 0)}")
        print(f"\nWiki location: {self.output_dir.absolute()}")
        print(f"Open: {(self.output_dir / 'index.html').absolute()}")
        print("\n" + "=" * 60)

    def _create_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        (self.assets_dir / "css").mkdir(exist_ok=True)
        (self.assets_dir / "js").mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        (self.output_dir / "lib").mkdir(exist_ok=True)

    def _export_documents(self) -> List[Dict]:
        """Export all documents with metadata."""
        documents = []
        cursor = self.kb.db_conn.cursor()

        for doc_id, doc_meta in self.kb.documents.items():
            # Get chunks
            chunks = cursor.execute(
                "SELECT chunk_id, content, page FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
                (doc_id,)
            ).fetchall()

            # Get tags
            tags = doc_meta.tags if doc_meta.tags else []

            doc_data = {
                'id': doc_id,
                'title': doc_meta.title,
                'filename': doc_meta.filename,
                'file_type': doc_meta.file_type,
                'total_pages': doc_meta.total_pages,
                'total_chunks': len(chunks),
                'indexed_at': doc_meta.indexed_at,
                'tags': tags,
                'source_url': getattr(doc_meta, 'source_url', None),
                'chunks': [
                    {
                        'id': chunk_id,
                        'content': content,
                        'page': page
                    }
                    for chunk_id, content, page in chunks
                ]
            }

            documents.append(doc_data)
            self.stats['documents'] += 1
            self.stats['chunks'] += len(chunks)

        # Sort by title
        documents.sort(key=lambda d: d['title'].lower())

        return documents

    def _export_entities(self) -> Dict:
        """Export entities grouped by type with document mappings."""
        cursor = self.kb.db_conn.cursor()

        # Get all entities with counts and document references
        entities_by_type = {}

        entity_types = cursor.execute(
            "SELECT DISTINCT entity_type FROM document_entities ORDER BY entity_type"
        ).fetchall()

        for (entity_type,) in entity_types:
            entities = cursor.execute("""
                SELECT entity_text, COUNT(DISTINCT doc_id) as doc_count,
                       AVG(confidence) as avg_confidence
                FROM document_entities
                WHERE entity_type = ?
                GROUP BY entity_text
                ORDER BY doc_count DESC, entity_text
            """, (entity_type,)).fetchall()

            entity_list = []
            for text, count, conf in entities:
                # Get all documents containing this entity
                doc_refs = cursor.execute("""
                    SELECT DISTINCT de.doc_id, d.title
                    FROM document_entities de
                    JOIN documents d ON de.doc_id = d.doc_id
                    WHERE de.entity_type = ? AND de.entity_text = ?
                    ORDER BY d.title
                """, (entity_type, text)).fetchall()

                entity_list.append({
                    'text': text,
                    'doc_count': count,
                    'confidence': round(conf, 2),
                    'documents': [
                        {
                            'id': doc_id,
                            'title': title,
                            'filename': re.sub(r'[^\w\-]', '_', doc_id) + '.html'
                        }
                        for doc_id, title in doc_refs
                    ]
                })

            entities_by_type[entity_type] = entity_list
            self.stats['entities'] += len(entities)

        return entities_by_type

    def _export_topics(self) -> Dict:
        """Export topic models."""
        cursor = self.kb.db_conn.cursor()

        topics_by_model = {}

        # Get topics for each model type
        model_types = cursor.execute(
            "SELECT DISTINCT model_type FROM topics ORDER BY model_type"
        ).fetchall()

        for (model_type,) in model_types:
            topics = cursor.execute("""
                SELECT topic_id, topic_number, top_words, coherence_score
                FROM topics
                WHERE model_type = ?
                ORDER BY topic_number
            """, (model_type,)).fetchall()

            topics_by_model[model_type] = [
                {
                    'id': topic_id,
                    'number': topic_num,
                    'words': top_words,
                    'coherence': round(coherence, 3) if coherence else None
                }
                for topic_id, topic_num, top_words, coherence in topics
            ]

            self.stats['topics'] += len(topics)

        return topics_by_model

    def _export_clusters(self) -> Dict:
        """Export document clusters."""
        cursor = self.kb.db_conn.cursor()

        clusters_by_algo = {}

        # Get clusters for each algorithm
        algorithms = cursor.execute(
            "SELECT DISTINCT algorithm FROM clusters ORDER BY algorithm"
        ).fetchall()

        for (algorithm,) in algorithms:
            clusters = cursor.execute("""
                SELECT c.cluster_id, c.cluster_number, COUNT(dc.doc_id) as doc_count
                FROM clusters c
                LEFT JOIN document_clusters dc ON c.cluster_id = dc.cluster_id
                WHERE c.algorithm = ?
                GROUP BY c.cluster_id, c.cluster_number
                ORDER BY c.cluster_number
            """, (algorithm,)).fetchall()

            # Convert cluster data, handling bytes/memoryview objects
            processed_clusters = []
            for cluster_id, cluster_num, doc_count in clusters:
                # Handle cluster_num which might be bytes/memoryview/int
                if isinstance(cluster_num, (bytes, memoryview)):
                    # Convert bytes to int
                    cluster_num = int.from_bytes(bytes(cluster_num), byteorder='little')

                processed_clusters.append({
                    'id': cluster_id,
                    'number': cluster_num,
                    'doc_count': doc_count
                })

            clusters_by_algo[algorithm] = processed_clusters

            self.stats['clusters'] += len(clusters)

        return clusters_by_algo

    def _export_events(self) -> List[Dict]:
        """Export timeline events."""
        cursor = self.kb.db_conn.cursor()

        events = cursor.execute("""
            SELECT event_id, event_type, title, description,
                   date_normalized, year, confidence
            FROM events
            ORDER BY year, date_normalized
        """).fetchall()

        events_data = [
            {
                'id': event_id,
                'type': event_type,
                'title': title,
                'description': description,
                'date': date_norm,
                'year': year,
                'confidence': round(conf, 2)
            }
            for event_id, event_type, title, description, date_norm, year, conf in events
        ]

        self.stats['events'] = len(events_data)

        return events_data

    def _export_chunks(self) -> List[Dict]:
        """Export all chunks with document references."""
        cursor = self.kb.db_conn.cursor()

        chunks = cursor.execute("""
            SELECT c.chunk_id, c.doc_id, c.content, c.page, d.title, d.file_type
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            ORDER BY d.title, c.chunk_id
        """).fetchall()

        chunks_data = []
        for chunk_id, doc_id, content, page, doc_title, file_type in chunks:
            chunks_data.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'doc_title': doc_title,
                'doc_filename': re.sub(r'[^\w\-]', '_', doc_id) + '.html',
                'file_type': file_type,
                'content': content[:500] + '...' if len(content) > 500 else content,  # Preview
                'full_content': content,
                'page': page,
                'content_length': len(content)
            })

        return chunks_data

    def _build_search_index(self, documents: List[Dict]) -> List[Dict]:
        """Build search index for client-side search."""
        search_index = []

        for doc in documents:
            # Combine all chunk content
            content = ' '.join(chunk['content'] for chunk in doc['chunks'])

            # Truncate for preview
            preview = content[:500] + '...' if len(content) > 500 else content

            search_index.append({
                'id': doc['id'],
                'title': doc['title'],
                'content': content,
                'preview': preview,
                'tags': doc['tags'],
                'file_type': doc['file_type'],
                'chunks': len(doc['chunks'])
            })

        return search_index

    def _build_navigation(self, documents: List[Dict]) -> Dict:
        """Build navigation structure."""
        # Group by tags
        by_tags = {}
        for doc in documents:
            for tag in doc['tags']:
                if tag not in by_tags:
                    by_tags[tag] = []
                by_tags[tag].append({
                    'id': doc['id'],
                    'title': doc['title']
                })

        # Group by file type
        by_type = {}
        for doc in documents:
            file_type = doc['file_type']
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append({
                'id': doc['id'],
                'title': doc['title']
            })

        return {
            'by_tags': by_tags,
            'by_type': by_type,
            'all_tags': sorted(by_tags.keys()),
            'all_types': sorted(by_type.keys())
        }

    def _save_json(self, filename: str, data: Any):
        """Save data as JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filename}")

    def _generate_html_pages(self, documents: List[Dict]):
        """Generate HTML pages for all documents."""
        # Generate index page
        self._generate_index_html()

        # Generate browser pages
        self._generate_documents_browser_html()
        self._generate_chunks_browser_html()

        # Generate document pages
        for doc in documents:
            self._generate_doc_html(doc)

        # Generate entity pages
        self._generate_entities_html()

        # Generate topics page
        self._generate_topics_html()

        # Generate timeline page
        self._generate_timeline_html()

        # Generate PDF viewer page
        self._generate_pdf_viewer_html()

    def _generate_index_html(self):
        """Generate main index page."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDZ C64 Knowledge Base - Wiki</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Comprehensive Commodore 64 Documentation</p>
        </header>

        <nav class="main-nav">
            <a href="index.html" class="active">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <div class="search-section">
            <input type="text" id="search-input" placeholder="Search the knowledge base..." autocomplete="off">
            <div id="search-results" class="search-results"></div>
        </div>

        <main>
            <section class="stats-grid">
                <div class="stat-card">
                    <h3>{self.stats['documents']}</h3>
                    <p>Documents</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['chunks']}</h3>
                    <p>Text Chunks</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['entities']}</h3>
                    <p>Entities</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['topics']}</h3>
                    <p>Topics</p>
                </div>
            </section>

            <section class="recent-docs">
                <h2>Browse Documents</h2>
                <div class="doc-grid" id="doc-list">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>

            <section class="browse-section">
                <h2>Browse by Category</h2>
                <div class="category-grid" id="category-list">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>
        </main>

        <footer>
            <p>Exported: {self.stats['export_date']}</p>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/main.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "index.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Generated: index.html")

    def _generate_doc_html(self, doc: Dict):
        """Generate HTML page for a single document."""
        doc_id = doc['id']
        safe_filename = re.sub(r'[^\w\-]', '_', doc_id) + '.html'

        # Escape HTML in content
        title_escaped = html.escape(doc['title'])

        # Build chunks HTML
        chunks_html = []
        for chunk in doc['chunks']:
            content_escaped = html.escape(chunk['content'])
            page_info = f" (Page {chunk['page']})" if chunk['page'] else ""
            chunks_html.append(f"""
                <div class="chunk">
                    <div class="chunk-header">Chunk {chunk['id']}{page_info}</div>
                    <div class="chunk-content">{content_escaped}</div>
                </div>
            """)

        # Build tags HTML
        tags_html = ' '.join(f'<span class="tag">{html.escape(tag)}</span>' for tag in doc['tags'])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_escaped} - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="../assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle"><a href="../index.html">← Back to Home</a></p>
        </header>

        <nav class="main-nav">
            <a href="../index.html">Home</a>
            <a href="../articles.html">Articles</a>
            <a href="../documents.html">Documents</a>
            <a href="../chunks.html">Chunks</a>
            <a href="../entities.html">Entities</a>
            <a href="../topics.html">Topics</a>
            <a href="../timeline.html">Timeline</a>
        </nav>

        <main>
            <article class="document">
                <div class="doc-header">
                    <h1>{title_escaped}</h1>
                    <div class="doc-meta">
                        <span class="meta-item">📄 {html.escape(doc['file_type'])}</span>
                        <span class="meta-item">📊 {doc['total_chunks']} chunks</span>
                        {f'<span class="meta-item">📑 {doc["total_pages"]} pages</span>' if doc['total_pages'] else ''}
                    </div>
                    {f'<div class="doc-tags">{tags_html}</div>' if tags_html else ''}
                    {f'<div class="doc-url"><a href="{html.escape(doc["source_url"])}" target="_blank">🔗 Source URL</a></div>' if doc.get('source_url') else ''}
                </div>

                <div class="chunks-container">
                    {''.join(chunks_html)}
                </div>
            </article>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="../assets/js/main.js"></script>
    <script src="../assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.docs_dir / safe_filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_entities_html(self):
        """Generate entities browser page."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entities - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Entity Browser - Click any entity to see related documents</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html" class="active">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <!-- Filter and Navigation Controls -->
            <div class="entity-controls">
                <div class="filter-section">
                    <input type="text" id="entity-filter" placeholder="🔍 Search entities..." autocomplete="off">
                </div>

                <div class="entity-type-nav" id="entity-type-nav">
                    <!-- Will be populated by JavaScript -->
                </div>

                <div class="entity-stats" id="entity-stats">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>

            <!-- Entities Container -->
            <div id="entities-container">
                <!-- Will be populated by JavaScript -->
            </div>

            <!-- Back to Top Button -->
            <button id="back-to-top" class="back-to-top" title="Back to top">
                ↑ Top
            </button>
        </main>

        <!-- Entity Details Modal -->
        <div id="entity-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 id="modal-title"></h2>
                    <span class="modal-close">&times;</span>
                </div>
                <div class="modal-body" id="modal-body">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>
        </div>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/entities.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "entities.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: entities.html")

    def _generate_topics_html(self):
        """Generate topics browser page."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topics & Clusters - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Topics & Clusters</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html" class="active">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <section>
                <h2>Topic Models</h2>
                <div id="topics-container">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>

            <section>
                <h2>Document Clusters</h2>
                <div id="clusters-container">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/topics.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "topics.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: topics.html")

    def _generate_timeline_html(self):
        """Generate timeline page."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Timeline - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Historical Timeline</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html" class="active">Timeline</a>
        </nav>

        <main>
            <div id="timeline-container">
                <!-- Will be populated by JavaScript -->
            </div>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/timeline.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "timeline.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: timeline.html")

    def _generate_documents_browser_html(self):
        """Generate documents browser page."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documents - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Browse All Documents</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="documents.html" class="active">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <div class="browser-controls">
                <input type="text" id="doc-search" placeholder="🔍 Search documents..." autocomplete="off">

                <div class="filter-buttons">
                    <button class="filter-btn active" data-type="all">All Types</button>
                    <button class="filter-btn" data-type="pdf">PDF</button>
                    <button class="filter-btn" data-type="text">Text</button>
                    <button class="filter-btn" data-type="html">HTML</button>
                    <button class="filter-btn" data-type="markdown">Markdown</button>
                </div>

                <div class="sort-controls">
                    <label>Sort by:</label>
                    <select id="sort-select">
                        <option value="title">Title (A-Z)</option>
                        <option value="title-desc">Title (Z-A)</option>
                        <option value="chunks">Chunks (Most)</option>
                        <option value="chunks-asc">Chunks (Least)</option>
                        <option value="date">Date Added</option>
                    </select>
                </div>
            </div>

            <div id="documents-grid" class="documents-grid">
                <!-- Populated by JavaScript -->
            </div>

            <button id="back-to-top" class="back-to-top">↑ Top</button>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/documents.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "documents.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: documents.html")

    def _generate_chunks_browser_html(self):
        """Generate chunks browser page."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chunks - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Browse Text Chunks</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html" class="active">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <div class="browser-controls">
                <input type="text" id="chunk-search" placeholder="🔍 Search chunks..." autocomplete="off">

                <div class="chunk-stats" id="chunk-stats">
                    <!-- Populated by JavaScript -->
                </div>
            </div>

            <div id="chunks-list" class="chunks-list">
                <!-- Populated by JavaScript -->
            </div>

            <div class="pagination" id="pagination">
                <!-- Populated by JavaScript -->
            </div>

            <button id="back-to-top" class="back-to-top">↑ Top</button>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/chunks.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "chunks.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: chunks.html")

    def _generate_pdf_viewer_html(self):
        """Generate PDF viewer page."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Viewer - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        #pdf-container {
            width: 100%;
            height: calc(100vh - 200px);
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: auto;
        }
        #pdf-canvas {
            display: block;
            margin: 20px auto;
        }
        .pdf-controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            padding: 20px;
            background: var(--bg-color);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .pdf-controls button {
            padding: 10px 20px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .pdf-controls button:hover {
            background: var(--secondary-color);
        }
        .pdf-controls button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle" id="pdf-title">PDF Viewer</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <div class="pdf-controls">
                <button id="prev-page">← Previous</button>
                <span id="page-info">Page 1 of 1</span>
                <button id="next-page">Next →</button>
                <button id="zoom-in">Zoom In</button>
                <button id="zoom-out">Zoom Out</button>
                <a id="download-link" href="#" download style="padding: 10px 20px; background: var(--primary-color); color: white; border-radius: 5px; text-decoration: none;">Download PDF</a>
            </div>

            <div id="pdf-container">
                <canvas id="pdf-canvas"></canvas>
            </div>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="lib/pdf.min.js"></script>
    <script src="assets/js/pdf-viewer.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "pdf-viewer.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: pdf-viewer.html")

    def _copy_pdfs(self, documents: List[Dict]):
        """Copy PDF files to wiki directory."""
        pdfs_dir = self.output_dir / "pdfs"
        pdfs_dir.mkdir(exist_ok=True)

        pdf_count = 0
        for doc in documents:
            if doc['file_type'].lower() == 'pdf':
                # Try to find the original PDF file
                doc_meta = self.kb.documents.get(doc['id'])
                if doc_meta and hasattr(doc_meta, 'filename'):
                    source_path = Path(doc_meta.filename)
                    if source_path.exists() and source_path.suffix.lower() == '.pdf':
                        dest_filename = re.sub(r'[^\w\-]', '_', doc['id']) + '.pdf'
                        dest_path = pdfs_dir / dest_filename
                        try:
                            shutil.copy2(source_path, dest_path)
                            pdf_count += 1
                        except Exception as e:
                            print(f"  Warning: Could not copy {source_path.name}: {e}")

        print(f"  Copied {pdf_count} PDF files")

    def _copy_static_assets(self):
        """Copy CSS, JS, and library files."""
        self._create_css()
        self._create_javascript()
        self._download_libraries()

    def _create_css(self):
        """Create CSS stylesheet."""
        css = """/* TDZ C64 Knowledge Base - Wiki Stylesheet */

:root {
    --primary-color: #4a5568;
    --secondary-color: #2d3748;
    --accent-color: #4299e1;
    --bg-color: #f7fafc;
    --text-color: #2d3748;
    --border-color: #e2e8f0;
    --card-bg: #ffffff;
    --code-bg: #2d3748;
    --code-color: #68d391;
}

/* Dark Theme */
[data-theme="dark"] {
    --primary-color: #cbd5e0;
    --secondary-color: #e2e8f0;
    --accent-color: #63b3ed;
    --bg-color: #1a202c;
    --text-color: #f7fafc;
    --border-color: #4a5568;
    --card-bg: #2d3748;
    --code-bg: #1a202c;
    --code-color: #68d391;
}

/* C64 Classic Theme */
[data-theme="c64"] {
    --primary-color: #c5cae9;
    --secondary-color: #e8eaf6;
    --accent-color: #ffc107;
    --bg-color: #3f51b5;
    --text-color: #e8eaf6;
    --border-color: #7986cb;
    --card-bg: #5c6bc0;
    --code-bg: #283593;
    --code-color: #00e676;
}

/* Theme transition */
* {
    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--bg-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
header {
    text-align: center;
    padding: 40px 0 20px;
    border-bottom: 2px solid var(--border-color);
}

header h1 {
    font-size: 2.5em;
    color: var(--secondary-color);
    margin-bottom: 10px;
}

.subtitle {
    font-size: 1.2em;
    color: var(--primary-color);
}

.subtitle a {
    color: var(--accent-color);
    text-decoration: none;
}

.subtitle a:hover {
    text-decoration: underline;
}

/* Navigation */
.main-nav {
    display: flex;
    justify-content: center;
    gap: 20px;
    padding: 20px 0;
    border-bottom: 1px solid var(--border-color);
}

.main-nav a {
    padding: 10px 20px;
    text-decoration: none;
    color: var(--text-color);
    border-radius: 5px;
    transition: all 0.3s;
}

.main-nav a:hover,
.main-nav a.active {
    background-color: var(--accent-color);
    color: white;
}

/* Search Section */
.search-section {
    margin: 30px 0;
    position: relative;
}

#search-input,
#entity-filter {
    width: 100%;
    padding: 15px 20px;
    font-size: 1.1em;
    border: 2px solid var(--border-color);
    border-radius: 10px;
    outline: none;
    transition: border-color 0.3s;
}

#search-input:focus,
#entity-filter:focus {
    border-color: var(--accent-color);
}

.search-results {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    max-height: 500px;
    overflow-y: auto;
    background: white;
    border: 2px solid var(--border-color);
    border-top: none;
    border-radius: 0 0 10px 10px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    z-index: 1000;
    display: none;
}

.search-results.active {
    display: block;
}

.search-result {
    padding: 15px 20px;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background-color 0.2s;
}

.search-result:hover {
    background-color: var(--bg-color);
}

.search-result:last-child {
    border-bottom: none;
}

.search-result-title {
    font-weight: 600;
    color: var(--secondary-color);
    margin-bottom: 5px;
}

.search-result-preview {
    font-size: 0.9em;
    color: var(--primary-color);
    line-height: 1.4;
}

.search-result-meta {
    font-size: 0.8em;
    color: #718096;
    margin-top: 5px;
}

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.stat-card {
    background: var(--card-bg);
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    text-align: center;
    transition: transform 0.3s, box-shadow 0.3s;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.stat-card h3 {
    font-size: 2.5em;
    color: var(--accent-color);
    margin-bottom: 10px;
}

.stat-card p {
    color: var(--primary-color);
    font-size: 1.1em;
}

/* Document Grid */
.doc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.doc-card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.3s, box-shadow 0.3s;
}

.doc-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.doc-card h3 {
    font-size: 1.2em;
    margin-bottom: 10px;
    color: var(--secondary-color);
}

.doc-card h3 a {
    color: inherit;
    text-decoration: none;
}

.doc-card h3 a:hover {
    color: var(--accent-color);
}

.doc-card-meta {
    font-size: 0.9em;
    color: var(--primary-color);
    margin-bottom: 10px;
}

.doc-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 10px;
}

.tag {
    display: inline-block;
    padding: 3px 10px;
    background: var(--bg-color);
    border-radius: 5px;
    font-size: 0.85em;
    color: var(--primary-color);
}

/* Document Page */
.document {
    background: var(--card-bg);
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin: 30px 0;
}

.doc-header {
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 20px;
    margin-bottom: 30px;
}

.doc-header h1 {
    color: var(--secondary-color);
    margin-bottom: 15px;
}

.doc-meta {
    display: flex;
    gap: 20px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.meta-item {
    color: var(--primary-color);
    font-size: 0.95em;
}

.doc-url {
    margin-top: 10px;
}

.doc-url a {
    color: var(--accent-color);
    text-decoration: none;
}

.doc-url a:hover {
    text-decoration: underline;
}

.chunks-container {
    margin-top: 30px;
}

.chunk {
    margin-bottom: 30px;
    padding: 20px;
    background: var(--bg-color);
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
}

.chunk-header {
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: 10px;
    font-size: 0.9em;
}

.chunk-content {
    color: var(--text-color);
    white-space: pre-wrap;
    line-height: 1.8;
}

/* Sections */
section {
    margin: 40px 0;
}

section h2 {
    color: var(--secondary-color);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--border-color);
}

/* Category Grid */
.category-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 15px;
}

.category-card {
    background: var(--card-bg);
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    cursor: pointer;
    transition: all 0.3s;
}

.category-card:hover {
    background-color: var(--accent-color);
    color: white;
    transform: translateY(-3px);
}

.category-card h3 {
    font-size: 1.1em;
    margin-bottom: 5px;
}

.category-card p {
    font-size: 0.9em;
    opacity: 0.8;
}

/* Entities */
.entity-type {
    margin-bottom: 30px;
}

.entity-type h3 {
    color: var(--secondary-color);
    margin-bottom: 15px;
    padding: 10px;
    background: var(--bg-color);
    border-radius: 5px;
}

.entity-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 10px;
}

.entity-item {
    background: var(--card-bg);
    padding: 10px 15px;
    border-radius: 5px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.entity-name {
    font-weight: 500;
    color: var(--text-color);
}

.entity-count {
    font-size: 0.85em;
    color: var(--primary-color);
    background: var(--bg-color);
    padding: 2px 8px;
    border-radius: 10px;
}

/* Timeline */
.timeline {
    position: relative;
    padding: 20px 0;
}

.timeline-event {
    background: var(--card-bg);
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border-left: 4px solid var(--accent-color);
}

.timeline-year {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--accent-color);
    margin-bottom: 10px;
}

.timeline-title {
    font-size: 1.2em;
    color: var(--secondary-color);
    margin-bottom: 10px;
}

.timeline-desc {
    color: var(--text-color);
    line-height: 1.6;
}

.timeline-meta {
    margin-top: 10px;
    font-size: 0.9em;
    color: var(--primary-color);
}

/* Footer */
footer {
    text-align: center;
    padding: 40px 0 20px;
    border-top: 2px solid var(--border-color);
    margin-top: 60px;
    color: var(--primary-color);
}

/* Entity Controls */
.entity-controls {
    margin: 30px 0;
}

.entity-type-nav {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 20px 0;
}

.entity-type-btn {
    padding: 8px 16px;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.95em;
}

.entity-type-btn:hover,
.entity-type-btn.active {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

.entity-stats {
    padding: 15px;
    background: var(--bg-color);
    border-radius: 5px;
    margin: 20px 0;
    font-size: 0.95em;
    color: var(--primary-color);
}

.entity-item {
    cursor: pointer;
    transition: all 0.2s;
}

.entity-item:hover {
    background-color: var(--accent-color);
    color: white;
    transform: translateX(5px);
}

.entity-item:hover .entity-count {
    background-color: white;
    color: var(--accent-color);
}

/* Back to Top Button */
.back-to-top {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: var(--accent-color);
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s;
    z-index: 999;
}

.back-to-top.visible {
    opacity: 1;
    visibility: visible;
}

.back-to-top:hover {
    background: var(--secondary-color);
    transform: translateY(-5px);
}

/* Modal */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.5);
    animation: fadeIn 0.3s;
}

.modal.active {
    display: block;
}

.modal-content {
    background-color: var(--card-bg);
    margin: 5% auto;
    padding: 0;
    border-radius: 10px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    width: 90%;
    max-width: 800px;
    animation: slideDown 0.3s;
}

.modal-header {
    padding: 20px 30px;
    border-bottom: 2px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.modal-header h2 {
    margin: 0;
    color: var(--secondary-color);
}

.modal-close {
    color: var(--primary-color);
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
    transition: color 0.3s;
}

.modal-close:hover {
    color: var(--accent-color);
}

.modal-body {
    padding: 30px;
    max-height: 60vh;
    overflow-y: auto;
}

.modal-section {
    margin-bottom: 20px;
}

.modal-section h3 {
    color: var(--primary-color);
    margin-bottom: 15px;
    font-size: 1.1em;
}

.document-link {
    display: block;
    padding: 10px 15px;
    margin: 5px 0;
    background: var(--bg-color);
    border-radius: 5px;
    color: var(--text-color);
    text-decoration: none;
    transition: all 0.2s;
}

.document-link:hover {
    background: var(--accent-color);
    color: white;
    transform: translateX(5px);
}

.entity-type-section {
    margin-bottom: 40px;
}

.entity-type-section.collapsed .entity-list {
    display: none;
}

.entity-type-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    padding: 15px;
    background: var(--bg-color);
    border-radius: 5px;
    transition: background 0.3s;
}

.entity-type-header:hover {
    background: var(--border-color);
}

.entity-type-header h3 {
    margin: 0;
}

.collapse-icon {
    font-size: 1.2em;
    transition: transform 0.3s;
}

.entity-type-section.collapsed .collapse-icon {
    transform: rotate(-90deg);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideDown {
    from {
        transform: translateY(-50px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Responsive */
@media (max-width: 768px) {
    header h1 {
        font-size: 2em;
    }

    .stats-grid,
    .doc-grid,
    .category-grid {
        grid-template-columns: 1fr;
    }

    .main-nav {
        flex-wrap: wrap;
    }

    .modal-content {
        width: 95%;
        margin: 10% auto;
    }

    .back-to-top {
        bottom: 20px;
        right: 20px;
        padding: 10px 16px;
    }

    .entity-type-nav {
        gap: 5px;
    }

    .entity-type-btn {
        padding: 6px 12px;
        font-size: 0.9em;
    }
}

/* Browser Controls */
.browser-controls {
    margin: 30px 0;
    padding: 20px;
    background: var(--bg-color);
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

.search-box {
    width: 100%;
    padding: 12px 20px;
    font-size: 1em;
    border: 2px solid var(--border-color);
    border-radius: 5px;
    margin-bottom: 15px;
    transition: border-color 0.3s;
}

.search-box:focus {
    outline: none;
    border-color: var(--accent-color);
}

.filter-sort-row {
    display: flex;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
}

.filter-buttons {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.filter-btn {
    padding: 8px 16px;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9em;
}

.filter-btn:hover {
    background: var(--border-color);
}

.filter-btn.active {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

.sort-controls {
    display: flex;
    align-items: center;
    gap: 10px;
}

.sort-controls label {
    font-weight: 600;
    color: var(--primary-color);
}

.sort-controls select {
    padding: 8px 12px;
    border: 2px solid var(--border-color);
    border-radius: 5px;
    background: var(--card-bg);
    cursor: pointer;
    font-size: 0.9em;
    transition: border-color 0.3s;
}

.sort-controls select:focus {
    outline: none;
    border-color: var(--accent-color);
}

/* Documents Grid */
.documents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
    margin: 30px 0;
}

.doc-card {
    background: var(--card-bg);
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: all 0.3s;
}

.doc-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.doc-card h3 {
    margin: 0 0 10px 0;
    font-size: 1.1em;
}

.doc-card h3 a {
    color: var(--secondary-color);
    text-decoration: none;
    transition: color 0.3s;
}

.doc-card h3 a:hover {
    color: var(--accent-color);
}

.doc-card-meta {
    font-size: 0.9em;
    color: var(--primary-color);
    margin: 10px 0;
}

.doc-tags {
    margin: 10px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}

.view-pdf-btn {
    display: inline-block;
    margin-top: 10px;
    padding: 8px 16px;
    background: var(--accent-color);
    color: white;
    border-radius: 5px;
    text-decoration: none;
    font-size: 0.9em;
    transition: all 0.3s;
}

.view-pdf-btn:hover {
    background: var(--secondary-color);
    transform: translateX(5px);
}

/* Chunks List */
.chunks-list {
    margin: 30px 0;
}

.chunk-item {
    background: var(--card-bg);
    padding: 20px;
    margin-bottom: 15px;
    border-radius: 8px;
    border-left: 4px solid var(--primary-color);
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: all 0.2s;
}

.chunk-item:hover {
    transform: translateX(5px);
    box-shadow: 0 3px 15px rgba(0,0,0,0.1);
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
    gap: 10px;
}

.chunk-doc-link {
    color: var(--secondary-color);
    text-decoration: none;
    font-weight: 600;
    transition: color 0.3s;
}

.chunk-doc-link:hover {
    color: var(--accent-color);
}

.chunk-meta {
    font-size: 0.85em;
    color: var(--primary-color);
}

.chunk-content {
    color: var(--text-color);
    line-height: 1.6;
    margin-top: 10px;
    padding: 10px;
    background: var(--bg-color);
    border-radius: 5px;
}

.chunks-stats {
    padding: 15px;
    background: var(--bg-color);
    border-radius: 5px;
    margin: 20px 0;
    font-size: 0.95em;
    color: var(--primary-color);
}

/* Pagination */
.pagination {
    margin: 40px 0;
    text-align: center;
}

.pagination-controls {
    display: inline-flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
}

.pagination-controls button {
    padding: 10px 16px;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.95em;
}

.pagination-controls button:hover {
    background: var(--border-color);
}

.pagination-controls button.active {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

.pagination-controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* PDF Viewer */
.pdf-controls {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin: 20px 0;
    padding: 15px;
    background: var(--bg-color);
    border-radius: 8px;
    flex-wrap: wrap;
}

.pdf-controls button {
    padding: 10px 20px;
    background: var(--accent-color);
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.95em;
}

.pdf-controls button:hover:not(:disabled) {
    background: var(--secondary-color);
}

.pdf-controls button:disabled {
    background: var(--border-color);
    cursor: not-allowed;
    opacity: 0.6;
}

.pdf-controls span {
    color: var(--primary-color);
    font-weight: 600;
}

.pdf-canvas-container {
    text-align: center;
    margin: 20px 0;
    padding: 20px;
    background: var(--bg-color);
    border-radius: 8px;
    overflow: auto;
}

#pdf-canvas {
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    max-width: 100%;
    height: auto;
}

.pdf-error {
    padding: 20px;
    background: #fee;
    border: 2px solid #fcc;
    border-radius: 8px;
    color: #c33;
    text-align: center;
    margin: 20px 0;
}

/* Loading indicator */
.loading {
    text-align: center;
    padding: 40px;
    color: var(--primary-color);
}

.loading::after {
    content: '...';
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}

/* Responsive adjustments for new components */
@media (max-width: 768px) {
    .documents-grid {
        grid-template-columns: 1fr;
    }

    .filter-sort-row {
        flex-direction: column;
        align-items: stretch;
    }

    .filter-buttons,
    .sort-controls {
        width: 100%;
    }

    .sort-controls {
        justify-content: space-between;
    }

    .chunk-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .pdf-controls {
        flex-direction: column;
        gap: 10px;
    }

    .pdf-controls button {
        width: 100%;
    }
}

/* Theme Toggle Button */
.theme-toggle {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 1000;
    background: var(--accent-color);
    color: white;
    border: none;
    padding: 12px 18px;
    border-radius: 25px;
    cursor: pointer;
    font-size: 1.3em;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s;
}

.theme-toggle:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}

/* Code Block Enhancements */
.code-block-wrapper {
    position: relative;
    margin: 20px 0;
}

.copy-button {
    position: absolute;
    top: 10px;
    right: 10px;
    background: var(--accent-color);
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.85em;
    opacity: 0.8;
    transition: all 0.3s;
    z-index: 10;
}

.copy-button:hover {
    opacity: 1;
    transform: translateY(-2px);
}

.copy-button.copied {
    background: #48bb78;
}

/* Reading Progress Bar */
.reading-progress {
    position: fixed;
    top: 0;
    left: 0;
    width: 0%;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-color), var(--secondary-color));
    z-index: 9999;
    transition: width 0.1s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Search Highlighting */
.highlight {
    background-color: #ffc107;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: bold;
    color: #000;
}

[data-theme="dark"] .highlight {
    background-color: #ff9800;
}

[data-theme="c64"] .highlight {
    background-color: #ffeb3b;
}

/* Article Meta (Reading Time) */
.article-meta {
    display: flex;
    gap: 15px;
    padding: 15px;
    background: var(--bg-color);
    border-radius: 8px;
    margin: 20px 0;
    font-size: 0.9em;
    color: var(--primary-color);
    border: 1px solid var(--border-color);
}

.article-meta span {
    display: flex;
    align-items: center;
    gap: 5px;
}

/* Enhanced Back to Top */
.back-to-top {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: var(--accent-color);
    color: white;
    border: none;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 1.5em;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s;
    z-index: 999;
}

.back-to-top.visible {
    opacity: 1;
    visibility: visible;
}

.back-to-top:hover {
    background: var(--secondary-color);
    transform: translateY(-5px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.3);
}

/* Smooth Scroll */
html {
    scroll-behavior: smooth;
}
"""
        css_path = self.assets_dir / "css" / "style.css"
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css)
        print(f"  Created: assets/css/style.css")

    def _create_javascript(self):
        """Create JavaScript files."""
        # Main JS
        main_js = """// Main JavaScript for TDZ C64 Knowledge Base Wiki

// Load navigation data and populate categories
async function loadNavigation() {
    try {
        const response = await fetch('assets/data/navigation.json');
        const nav = await response.json();

        // Populate category list
        const categoryList = document.getElementById('category-list');
        if (categoryList) {
            categoryList.innerHTML = '';

            // Tags
            for (const tag of nav.all_tags) {
                const count = nav.by_tags[tag].length;
                const card = document.createElement('div');
                card.className = 'category-card';
                card.innerHTML = `<h3>${escapeHtml(tag)}</h3><p>${count} documents</p>`;
                card.onclick = () => window.location.href = `#tag-${tag}`;
                categoryList.appendChild(card);
            }
        }
    } catch (error) {
        console.error('Error loading navigation:', error);
    }
}

// Load and display documents
async function loadDocuments() {
    try {
        const response = await fetch('assets/data/documents.json');
        const documents = await response.json();

        const docList = document.getElementById('doc-list');
        if (docList) {
            docList.innerHTML = '';

            // Show first 20 documents
            for (const doc of documents.slice(0, 20)) {
                const card = createDocCard(doc);
                docList.appendChild(card);
            }
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// Create document card element
function createDocCard(doc) {
    const card = document.createElement('div');
    card.className = 'doc-card';

    const safeFilename = doc.id.replace(/[^\\w\\-]/g, '_') + '.html';
    const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

    card.innerHTML = `
        <h3><a href="docs/${safeFilename}">${escapeHtml(doc.title)}</a></h3>
        <div class="doc-card-meta">
            ${doc.file_type} • ${doc.total_chunks} chunks
        </div>
        <div class="doc-tags">${tags}</div>
    `;

    return card;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        loadNavigation();
        loadDocuments();
    });
} else {
    loadNavigation();
    loadDocuments();
}
"""

        # Search JS
        search_js = """// Search functionality using Fuse.js

let searchIndex = [];
let fuse = null;

// Load search index
async function loadSearchIndex() {
    try {
        const response = await fetch('assets/data/search-index.json');
        searchIndex = await response.json();

        // Initialize Fuse.js
        fuse = new Fuse(searchIndex, {
            keys: ['title', 'content', 'tags'],
            threshold: 0.3,
            includeScore: true,
            minMatchCharLength: 2,
            ignoreLocation: true
        });

        console.log('Search index loaded:', searchIndex.length, 'documents');
    } catch (error) {
        console.error('Error loading search index:', error);
    }
}

// Handle search input
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');

if (searchInput && searchResults) {
    searchInput.addEventListener('input', handleSearch);

    // Close results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.remove('active');
        }
    });
}

function handleSearch(event) {
    const query = event.target.value.trim();

    if (query.length < 2) {
        searchResults.classList.remove('active');
        searchResults.innerHTML = '';
        return;
    }

    if (!fuse) {
        searchResults.innerHTML = '<div class="search-result">Loading search index...</div>';
        searchResults.classList.add('active');
        return;
    }

    // Perform search
    const results = fuse.search(query, { limit: 10 });

    if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result">No results found</div>';
        searchResults.classList.add('active');
        return;
    }

    // Display results
    searchResults.innerHTML = '';
    for (const result of results) {
        const doc = result.item;
        const resultDiv = createSearchResult(doc);
        searchResults.appendChild(resultDiv);
    }
    searchResults.classList.add('active');
}

function createSearchResult(doc) {
    const div = document.createElement('div');
    div.className = 'search-result';

    const safeFilename = doc.id.replace(/[^\\w\\-]/g, '_') + '.html';

    div.innerHTML = `
        <div class="search-result-title">${escapeHtml(doc.title)}</div>
        <div class="search-result-preview">${escapeHtml(doc.preview)}</div>
        <div class="search-result-meta">${doc.file_type} • ${doc.chunks} chunks</div>
    `;

    div.onclick = () => {
        window.location.href = `docs/${safeFilename}`;
    };

    return div;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Load search index on page load
loadSearchIndex();
"""

        # Entities JS
        entities_js = """// Enhanced entities browser with clickable entities and navigation

let entitiesData = {};
let currentFilter = '';
let currentTypeFilter = 'all';

async function loadEntities() {
    try {
        const response = await fetch('assets/data/entities.json');
        entitiesData = await response.json();
        initializeUI();
        displayEntities(entitiesData);
    } catch (error) {
        console.error('Error loading entities:', error);
    }
}

function initializeUI() {
    // Create type navigation buttons
    const typeNav = document.getElementById('entity-type-nav');
    typeNav.innerHTML = '';

    const allBtn = createTypeButton('all', 'All Types', true);
    typeNav.appendChild(allBtn);

    for (const entityType of Object.keys(entitiesData)) {
        const count = entitiesData[entityType].length;
        const btn = createTypeButton(entityType, `${entityType} (${count})`);
        typeNav.appendChild(btn);
    }

    // Update stats
    updateStats();

    // Setup modal close handlers
    setupModal();

    // Setup back to top button
    setupBackToTop();
}

function createTypeButton(type, label, active = false) {
    const btn = document.createElement('button');
    btn.className = 'entity-type-btn' + (active ? ' active' : '');
    btn.textContent = label;
    btn.onclick = () => filterByType(type);
    return btn;
}

function filterByType(type) {
    currentTypeFilter = type;

    // Update button states
    document.querySelectorAll('.entity-type-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Filter and display
    const filtered = type === 'all' ? entitiesData : { [type]: entitiesData[type] };
    displayEntities(filtered, currentFilter);
}

function updateStats() {
    const stats = document.getElementById('entity-stats');
    const totalEntities = Object.values(entitiesData).reduce((sum, arr) => sum + arr.length, 0);
    const totalTypes = Object.keys(entitiesData).length;
    const totalDocs = new Set(
        Object.values(entitiesData).flatMap(entities =>
            entities.flatMap(e => e.documents.map(d => d.id))
        )
    ).size;

    stats.innerHTML = `
        <strong>${totalEntities}</strong> entities across
        <strong>${totalTypes}</strong> types, referenced in
        <strong>${totalDocs}</strong> documents
    `;
}

function displayEntities(data, searchQuery = '') {
    const container = document.getElementById('entities-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [entityType, entities] of Object.entries(data)) {
        // Filter by search query
        let filteredEntities = entities;
        if (searchQuery) {
            filteredEntities = entities.filter(e =>
                e.text.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        if (filteredEntities.length === 0) continue;

        const section = document.createElement('div');
        section.className = 'entity-type-section';
        section.id = `type-${entityType}`;

        // Collapsible header
        const header = document.createElement('div');
        header.className = 'entity-type-header';
        header.innerHTML = `
            <h3>${entityType} (${filteredEntities.length})</h3>
            <span class="collapse-icon">▼</span>
        `;
        header.onclick = () => toggleSection(section);
        section.appendChild(header);

        // Entity list
        const list = document.createElement('div');
        list.className = 'entity-list';

        for (const entity of filteredEntities) {
            const item = document.createElement('div');
            item.className = 'entity-item';
            item.innerHTML = `
                <span class="entity-name">${escapeHtml(entity.text)}</span>
                <span class="entity-count">${entity.doc_count} docs</span>
            `;
            item.onclick = () => showEntityDetails(entity, entityType);
            list.appendChild(item);
        }

        section.appendChild(list);
        container.appendChild(section);
    }

    if (container.children.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #718096; padding: 40px;">No entities found matching your search.</p>';
    }
}

function toggleSection(section) {
    section.classList.toggle('collapsed');
}

function showEntityDetails(entity, entityType) {
    const modal = document.getElementById('entity-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    modalTitle.textContent = `${entity.text} (${entityType})`;

    modalBody.innerHTML = `
        <div class="modal-section">
            <h3>Overview</h3>
            <p>
                <strong>Type:</strong> ${entityType}<br>
                <strong>Documents:</strong> ${entity.doc_count}<br>
                <strong>Confidence:</strong> ${(entity.confidence * 100).toFixed(0)}%
            </p>
        </div>

        <div class="modal-section">
            <h3>Related Documents (${entity.documents.length})</h3>
            ${entity.documents.map(doc => `
                <a href="docs/${doc.filename}" class="document-link">
                    📄 ${escapeHtml(doc.title)}
                </a>
            `).join('')}
        </div>
    `;

    modal.classList.add('active');
}

function setupModal() {
    const modal = document.getElementById('entity-modal');
    const closeBtn = document.querySelector('.modal-close');

    closeBtn.onclick = () => {
        modal.classList.remove('active');
    };

    window.onclick = (event) => {
        if (event.target === modal) {
            modal.classList.remove('active');
        }
    };

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modal.classList.remove('active');
        }
    });
}

function setupBackToTop() {
    const backToTop = document.getElementById('back-to-top');

    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });

    backToTop.onclick = () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    };
}

// Filter entities by search
const filterInput = document.getElementById('entity-filter');
if (filterInput) {
    filterInput.addEventListener('input', (e) => {
        currentFilter = e.target.value.trim();

        const filtered = currentTypeFilter === 'all'
            ? entitiesData
            : { [currentTypeFilter]: entitiesData[currentTypeFilter] };

        displayEntities(filtered, currentFilter);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
loadEntities();
"""

        # Topics JS
        topics_js = """// Topics and clusters browser

async function loadTopicsAndClusters() {
    try {
        const [topicsResp, clustersResp] = await Promise.all([
            fetch('assets/data/topics.json'),
            fetch('assets/data/clusters.json')
        ]);

        const topics = await topicsResp.json();
        const clusters = await clustersResp.json();

        displayTopics(topics);
        displayClusters(clusters);
    } catch (error) {
        console.error('Error loading topics/clusters:', error);
    }
}

function displayTopics(topicsData) {
    const container = document.getElementById('topics-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [modelType, topics] of Object.entries(topicsData)) {
        const section = document.createElement('div');
        section.innerHTML = `<h3>${modelType.toUpperCase()} (${topics.length} topics)</h3>`;

        const grid = document.createElement('div');
        grid.className = 'doc-grid';

        for (const topic of topics) {
            const card = document.createElement('div');
            card.className = 'doc-card';
            card.innerHTML = `
                <h3>Topic ${topic.number}</h3>
                <div class="doc-card-meta">${escapeHtml(topic.words)}</div>
                ${topic.coherence ? `<div class="doc-tags"><span class="tag">Coherence: ${topic.coherence}</span></div>` : ''}
            `;
            grid.appendChild(card);
        }

        section.appendChild(grid);
        container.appendChild(section);
    }
}

function displayClusters(clustersData) {
    const container = document.getElementById('clusters-container');
    if (!container) return;

    container.innerHTML = '';

    for (const [algorithm, clusters] of Object.entries(clustersData)) {
        const section = document.createElement('div');
        section.innerHTML = `<h3>${algorithm.toUpperCase()} (${clusters.length} clusters)</h3>`;

        const grid = document.createElement('div');
        grid.className = 'doc-grid';

        for (const cluster of clusters) {
            const card = document.createElement('div');
            card.className = 'doc-card';
            card.innerHTML = `
                <h3>Cluster ${cluster.number}</h3>
                <div class="doc-card-meta">${cluster.doc_count} documents</div>
            `;
            grid.appendChild(card);
        }

        section.appendChild(grid);
        container.appendChild(section);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadTopicsAndClusters();
"""

        # Timeline JS
        timeline_js = """// Timeline browser

async function loadTimeline() {
    try {
        const response = await fetch('assets/data/events.json');
        const events = await response.json();
        displayTimeline(events);
    } catch (error) {
        console.error('Error loading timeline:', error);
    }
}

function displayTimeline(events) {
    const container = document.getElementById('timeline-container');
    if (!container) return;

    container.innerHTML = '';

    if (events.length === 0) {
        container.innerHTML = '<p>No timeline events found.</p>';
        return;
    }

    const timeline = document.createElement('div');
    timeline.className = 'timeline';

    for (const event of events) {
        const eventDiv = document.createElement('div');
        eventDiv.className = 'timeline-event';

        eventDiv.innerHTML = `
            <div class="timeline-year">${event.year || 'Unknown'}</div>
            <div class="timeline-title">${escapeHtml(event.title)}</div>
            <div class="timeline-desc">${escapeHtml(event.description || '')}</div>
            <div class="timeline-meta">
                ${event.type} • Confidence: ${(event.confidence * 100).toFixed(0)}%
            </div>
        `;

        timeline.appendChild(eventDiv);
    }

    container.appendChild(timeline);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadTimeline();
"""

        # Documents JS
        documents_js = """// Documents browser with filtering and sorting

let allDocuments = [];
let filteredDocuments = [];
let currentTypeFilter = 'all';
let currentSort = 'title';

async function loadDocuments() {
    try {
        const response = await fetch('assets/data/documents.json');
        allDocuments = await response.json();
        filteredDocuments = allDocuments;
        displayDocuments();
        setupEventListeners();
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function setupEventListeners() {
    // Search
    document.getElementById('doc-search').addEventListener('input', (e) => {
        filterAndDisplay();
    });

    // Type filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentTypeFilter = e.target.dataset.type;
            filterAndDisplay();
        });
    });

    // Sort
    document.getElementById('sort-select').addEventListener('change', (e) => {
        currentSort = e.target.value;
        displayDocuments();
    });

    // Back to top
    const backToTop = document.getElementById('back-to-top');
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    backToTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
}

function filterAndDisplay() {
    const searchQuery = document.getElementById('doc-search').value.toLowerCase();

    filteredDocuments = allDocuments.filter(doc => {
        // Type filter
        if (currentTypeFilter !== 'all' && doc.file_type.toLowerCase() !== currentTypeFilter) {
            return false;
        }

        // Search filter
        if (searchQuery) {
            return doc.title.toLowerCase().includes(searchQuery) ||
                   doc.tags.some(tag => tag.toLowerCase().includes(searchQuery));
        }

        return true;
    });

    displayDocuments();
}

function displayDocuments() {
    // Sort
    const sorted = [...filteredDocuments].sort((a, b) => {
        switch (currentSort) {
            case 'title':
                return a.title.localeCompare(b.title);
            case 'title-desc':
                return b.title.localeCompare(a.title);
            case 'chunks':
                return b.total_chunks - a.total_chunks;
            case 'chunks-asc':
                return a.total_chunks - b.total_chunks;
            case 'date':
                return new Date(b.indexed_at) - new Date(a.indexed_at);
            default:
                return 0;
        }
    });

    const grid = document.getElementById('documents-grid');
    grid.innerHTML = '';

    if (sorted.length === 0) {
        grid.innerHTML = '<p style="text-align: center; padding: 40px; color: #718096;">No documents found.</p>';
        return;
    }

    for (const doc of sorted) {
        const card = createDocCard(doc);
        grid.appendChild(card);
    }
}

function createDocCard(doc) {
    const card = document.createElement('div');
    card.className = 'doc-card';

    const safeFilename = doc.id.replace(/[^\\w\\-]/g, '_') + '.html';
    const isPDF = doc.file_type.toLowerCase() === 'pdf';
    const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

    card.innerHTML = `
        <h3><a href="docs/${safeFilename}">${escapeHtml(doc.title)}</a></h3>
        <div class="doc-card-meta">
            ${doc.file_type} • ${doc.total_chunks} chunks
            ${doc.total_pages ? ` • ${doc.total_pages} pages` : ''}
        </div>
        <div class="doc-tags">${tags}</div>
        ${isPDF ? `<a href="pdf-viewer.html?file=${doc.id.replace(/[^\\w\\-]/g, '_')}.pdf" class="view-pdf-btn">📄 View PDF</a>` : ''}
    `;

    return card;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadDocuments();
"""

        # Chunks JS
        chunks_js = """// Chunks browser with search and pagination

let allChunks = [];
let filteredChunks = [];
let currentPage = 1;
const chunksPerPage = 50;

async function loadChunks() {
    try {
        const response = await fetch('assets/data/chunks.json');
        allChunks = await response.json();
        filteredChunks = allChunks;
        updateStats();
        displayChunks();
        setupEventListeners();
    } catch (error) {
        console.error('Error loading chunks:', error);
    }
}

function setupEventListeners() {
    // Search
    document.getElementById('chunk-search').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        if (query) {
            filteredChunks = allChunks.filter(chunk =>
                chunk.full_content.toLowerCase().includes(query) ||
                chunk.doc_title.toLowerCase().includes(query)
            );
        } else {
            filteredChunks = allChunks;
        }
        currentPage = 1;
        updateStats();
        displayChunks();
    });

    // Back to top
    const backToTop = document.getElementById('back-to-top');
    window.addEventListener('scroll', () => {
        if (window.pageYOffset > 300) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    });
    backToTop.onclick = () => window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateStats() {
    const stats = document.getElementById('chunk-stats');
    const totalChunks = filteredChunks.length;
    const uniqueDocs = new Set(filteredChunks.map(c => c.doc_id)).size;

    stats.innerHTML = `
        Showing <strong>${totalChunks.toLocaleString()}</strong> chunks from
        <strong>${uniqueDocs}</strong> documents
    `;
}

function displayChunks() {
    const container = document.getElementById('chunks-list');
    container.innerHTML = '';

    const startIdx = (currentPage - 1) * chunksPerPage;
    const endIdx = Math.min(startIdx + chunksPerPage, filteredChunks.length);
    const pageChunks = filteredChunks.slice(startIdx, endIdx);

    if (pageChunks.length === 0) {
        container.innerHTML = '<p style="text-align: center; padding: 40px; color: #718096;">No chunks found.</p>';
        return;
    }

    for (const chunk of pageChunks) {
        const item = createChunkItem(chunk);
        container.appendChild(item);
    }

    displayPagination();
}

function createChunkItem(chunk) {
    const item = document.createElement('div');
    item.className = 'chunk-item';

    const pageInfo = chunk.page ? ` • Page ${chunk.page}` : '';

    item.innerHTML = `
        <div class="chunk-header">
            <a href="docs/${chunk.doc_filename}" class="chunk-doc-link">
                📄 ${escapeHtml(chunk.doc_title)}
            </a>
            <span class="chunk-meta">${chunk.file_type}${pageInfo} • ${chunk.content_length} chars</span>
        </div>
        <div class="chunk-content">${escapeHtml(chunk.content)}</div>
    `;

    return item;
}

function displayPagination() {
    const pagination = document.getElementById('pagination');
    const totalPages = Math.ceil(filteredChunks.length / chunksPerPage);

    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = '<div class="pagination-controls">';

    // Previous
    if (currentPage > 1) {
        html += `<button onclick="goToPage(${currentPage - 1})">← Previous</button>`;
    }

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    if (startPage > 1) {
        html += `<button onclick="goToPage(1)">1</button>`;
        if (startPage > 2) html += '<span>...</span>';
    }

    for (let i = startPage; i <= endPage; i++) {
        const active = i === currentPage ? ' class="active"' : '';
        html += `<button${active} onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) html += '<span>...</span>';
        html += `<button onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }

    // Next
    if (currentPage < totalPages) {
        html += `<button onclick="goToPage(${currentPage + 1})">Next →</button>`;
    }

    html += '</div>';
    pagination.innerHTML = html;
}

function goToPage(page) {
    currentPage = page;
    displayChunks();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

loadChunks();
"""

        # PDF Viewer JS
        pdf_viewer_js = """// PDF.js viewer integration

let pdfDoc = null;
let pageNum = 1;
let pageRendering = false;
let pageNumPending = null;
let scale = 1.5;
const canvas = document.getElementById('pdf-canvas');
const ctx = canvas.getContext('2d');

// Get PDF filename from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const pdfFile = urlParams.get('file');

if (!pdfFile) {
    document.getElementById('pdf-title').textContent = 'No PDF specified';
    document.getElementById('pdf-container').innerHTML = '<p style="text-align: center; padding: 40px;">No PDF file specified. Please select a document to view.</p>';
} else {
    loadPDF(`pdfs/${pdfFile}`);
}

function loadPDF(url) {
    // Configure PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'lib/pdf.worker.min.js';

    // Load the PDF
    pdfjsLib.getDocument(url).promise.then(function(pdf) {
        pdfDoc = pdf;
        document.getElementById('page-info').textContent = `Page ${pageNum} of ${pdfDoc.numPages}`;

        // Set download link
        document.getElementById('download-link').href = url;
        document.getElementById('download-link').download = pdfFile;

        // Initial render
        renderPage(pageNum);
    }).catch(function(error) {
        console.error('Error loading PDF:', error);
        document.getElementById('pdf-container').innerHTML = `
            <p style="text-align: center; padding: 40px; color: #e53e3e;">
                Error loading PDF: ${error.message}<br>
                File: ${url}
            </p>
        `;
    });
}

function renderPage(num) {
    pageRendering = true;

    pdfDoc.getPage(num).then(function(page) {
        const viewport = page.getViewport({ scale: scale });
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
            canvasContext: ctx,
            viewport: viewport
        };

        const renderTask = page.render(renderContext);

        renderTask.promise.then(function() {
            pageRendering = false;
            if (pageNumPending !== null) {
                renderPage(pageNumPending);
                pageNumPending = null;
            }
        });
    });

    document.getElementById('page-info').textContent = `Page ${num} of ${pdfDoc.numPages}`;
    updateButtons();
}

function queueRenderPage(num) {
    if (pageRendering) {
        pageNumPending = num;
    } else {
        renderPage(num);
    }
}

function onPrevPage() {
    if (pageNum <= 1) return;
    pageNum--;
    queueRenderPage(pageNum);
}

function onNextPage() {
    if (pageNum >= pdfDoc.numPages) return;
    pageNum++;
    queueRenderPage(pageNum);
}

function onZoomIn() {
    scale += 0.25;
    queueRenderPage(pageNum);
}

function onZoomOut() {
    if (scale <= 0.5) return;
    scale -= 0.25;
    queueRenderPage(pageNum);
}

function updateButtons() {
    document.getElementById('prev-page').disabled = (pageNum <= 1);
    document.getElementById('next-page').disabled = (pageNum >= pdfDoc.numPages);
}

// Event listeners
document.getElementById('prev-page').addEventListener('click', onPrevPage);
document.getElementById('next-page').addEventListener('click', onNextPage);
document.getElementById('zoom-in').addEventListener('click', onZoomIn);
document.getElementById('zoom-out').addEventListener('click', onZoomOut);

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') onPrevPage();
    if (e.key === 'ArrowRight') onNextPage();
    if (e.key === '+' || e.key === '=') onZoomIn();
    if (e.key === '-' || e.key === '_') onZoomOut();
});
"""

        # Enhancements JS - Theme switcher, copy buttons, progress bar, etc.
        enhancements_js = """// Wiki Enhancements - UX Improvements
// Theme switcher, copy buttons, reading progress, back-to-top

// ===== THEME SWITCHER =====
const themes = ['light', 'dark', 'c64'];
let currentTheme = localStorage.getItem('theme') || 'light';

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    currentTheme = theme;
    updateThemeIcon();
}

function cycleTheme() {
    const currentIndex = themes.indexOf(currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    setTheme(themes[nextIndex]);
}

function updateThemeIcon() {
    const button = document.getElementById('theme-toggle');
    if (button) {
        const icons = { 'light': '☀️', 'dark': '🌙', 'c64': '💙' };
        const labels = { 'light': 'Light Theme', 'dark': 'Dark Theme', 'c64': 'C64 Classic' };
        button.textContent = icons[currentTheme];
        button.title = labels[currentTheme];
    }
}

// ===== COPY BUTTONS =====
function addCopyButtons() {
    document.querySelectorAll('pre code').forEach(block => {
        // Skip if already wrapped
        if (block.parentNode.classList.contains('code-block-wrapper')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';

        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = '📋 Copy';
        button.title = 'Copy code to clipboard';

        button.onclick = async () => {
            try {
                await navigator.clipboard.writeText(block.textContent);
                button.textContent = '✅ Copied!';
                button.classList.add('copied');
                setTimeout(() => {
                    button.textContent = '📋 Copy';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
                button.textContent = '❌ Failed';
                setTimeout(() => {
                    button.textContent = '📋 Copy';
                }, 2000);
            }
        };

        const pre = block.parentNode;
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        wrapper.appendChild(button);
    });
}

// ===== READING PROGRESS =====
function addReadingProgress() {
    const progress = document.createElement('div');
    progress.className = 'reading-progress';
    document.body.appendChild(progress);

    function updateProgress() {
        const scrolled = window.pageYOffset;
        const height = document.documentElement.scrollHeight - window.innerHeight;
        const percent = height > 0 ? (scrolled / height) * 100 : 0;
        progress.style.width = percent + '%';
    }

    window.addEventListener('scroll', updateProgress);
    updateProgress();
}

// ===== BACK TO TOP =====
function addBackToTop() {
    const button = document.createElement('button');
    button.className = 'back-to-top';
    button.textContent = '↑';
    button.title = 'Back to top';
    button.onclick = () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    document.body.appendChild(button);

    function updateBackToTop() {
        if (window.pageYOffset > 300) {
            button.classList.add('visible');
        } else {
            button.classList.remove('visible');
        }
    }

    window.addEventListener('scroll', updateBackToTop);
    updateBackToTop();
}

// ===== THEME TOGGLE BUTTON =====
function addThemeToggle() {
    const button = document.createElement('button');
    button.id = 'theme-toggle';
    button.className = 'theme-toggle';
    button.onclick = cycleTheme;
    document.body.appendChild(button);
    updateThemeIcon();
}

// ===== SEARCH HIGHLIGHTING =====
function highlightSearchTerms() {
    const urlParams = new URLSearchParams(window.location.search);
    const query = urlParams.get('q');
    if (!query) return;

    const terms = query.toLowerCase().split(/\\s+/).filter(t => t.length >= 3);
    if (terms.length === 0) return;

    // Highlight in paragraphs, list items, and table cells
    document.querySelectorAll('main p, main li, main td, .chunk-content').forEach(element => {
        let html = element.innerHTML;
        let modified = false;

        terms.forEach(term => {
            const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
            if (regex.test(html)) {
                html = html.replace(regex, '<mark class="highlight">$1</mark>');
                modified = true;
            }
        });

        if (modified) {
            element.innerHTML = html;
        }
    });
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
}

// ===== KEYBOARD SHORTCUTS =====
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Alt+T = Toggle theme
        if (e.altKey && e.key === 't') {
            e.preventDefault();
            cycleTheme();
        }
        // Alt+H = Go to home
        if (e.altKey && e.key === 'h') {
            e.preventDefault();
            window.location.href = '/';
        }
    });
}

// ===== INITIALIZE ALL =====
document.addEventListener('DOMContentLoaded', () => {
    // Initialize theme first (before anything renders)
    setTheme(currentTheme);

    // Add all enhancements
    addThemeToggle();
    addCopyButtons();
    addReadingProgress();
    addBackToTop();
    highlightSearchTerms();
    setupKeyboardShortcuts();

    console.log('✅ Wiki enhancements loaded');
    console.log('💡 Keyboard shortcuts: Alt+T (theme), Alt+H (home)');
});

// Export for use in other scripts
window.wikiEnhancements = {
    setTheme,
    cycleTheme,
    addCopyButtons
};
"""

        # Save all JavaScript files
        js_dir = self.assets_dir / "js"

        files = {
            'main.js': main_js,
            'search.js': search_js,
            'entities.js': entities_js,
            'topics.js': topics_js,
            'timeline.js': timeline_js,
            'documents.js': documents_js,
            'chunks.js': chunks_js,
            'pdf-viewer.js': pdf_viewer_js,
            'enhancements.js': enhancements_js
        }

        for filename, content in files.items():
            filepath = js_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created: assets/js/{filename}")

    def _download_libraries(self):
        """Download required JavaScript libraries."""
        import urllib.request

        lib_dir = self.output_dir / "lib"

        # Fuse.js - lightweight fuzzy search library
        fuse_url = "https://cdn.jsdelivr.net/npm/fuse.js@7.0.0/dist/fuse.min.js"
        fuse_path = lib_dir / "fuse.min.js"

        try:
            print(f"  Downloading Fuse.js...")
            urllib.request.urlretrieve(fuse_url, fuse_path)
            print(f"  Downloaded: lib/fuse.min.js")
        except Exception as e:
            print(f"  Warning: Could not download Fuse.js: {e}")
            print(f"  Creating fallback notice...")

            # Create a fallback file with instructions
            fallback = """// Fuse.js could not be downloaded automatically.
// Please download from: https://cdn.jsdelivr.net/npm/fuse.js@7.0.0/dist/fuse.min.js
// Or use CDN in HTML: <script src="https://cdn.jsdelivr.net/npm/fuse.js@7.0.0"></script>

console.warn('Fuse.js not loaded - search functionality will be limited');
"""
            with open(fuse_path, 'w', encoding='utf-8') as f:
                f.write(fallback)

        # PDF.js - PDF viewer library
        pdf_url = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js"
        pdf_worker_url = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js"
        pdf_path = lib_dir / "pdf.min.js"
        pdf_worker_path = lib_dir / "pdf.worker.min.js"

        try:
            print(f"  Downloading PDF.js...")
            urllib.request.urlretrieve(pdf_url, pdf_path)
            print(f"  Downloaded: lib/pdf.min.js")

            print(f"  Downloading PDF.js worker...")
            urllib.request.urlretrieve(pdf_worker_url, pdf_worker_path)
            print(f"  Downloaded: lib/pdf.worker.min.js")
        except Exception as e:
            print(f"  Warning: Could not download PDF.js: {e}")
            print(f"  PDF viewing will not be available")

            # Create a fallback file with instructions
            fallback = """// PDF.js could not be downloaded automatically.
// Please download from: https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js
// and https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js

console.warn('PDF.js not loaded - PDF viewing will not work');
"""
            with open(pdf_path, 'w', encoding='utf-8') as f:
                f.write(fallback)

    def _generate_articles(self, entities_data: Dict):
        """Generate articles for major entities and topics."""
        print("\n[10/10] Generating articles...")

        # Create articles directory
        articles_dir = self.output_dir / "articles"
        articles_dir.mkdir(exist_ok=True)

        # Define major topics to generate articles for
        article_topics = {
            'HARDWARE': ['SID', 'VIC-II', 'VIC', 'CIA', '6510', '6502', '1541'],
            'MUSIC': ['Music', 'Sound', 'Composer', 'Editor', 'Tracker'],
            'GRAPHICS': ['Sprite', 'Bitmap', 'Graphics', 'Color', 'Screen'],
            'PROGRAMMING': ['Assembly', 'BASIC', 'Kernal', 'ROM', 'Memory'],
            'TOOLS': ['Assembler', 'Editor', 'Debugger', 'Monitor', 'Emulator']
        }

        articles_generated = []

        # Generate articles for each category
        for category, keywords in article_topics.items():
            for keyword in keywords:
                # Find matching entities
                matching_entities = self._find_entities_for_article(entities_data, keyword)

                if matching_entities:
                    article = self._create_article(keyword, category, matching_entities, entities_data)
                    if article:
                        articles_generated.append(article)
                        print(f"  Generated: {article['title']}")

        # Generate articles browser page
        self._generate_articles_browser_html(articles_generated)

        # Save articles index
        articles_index_path = self.data_dir / "articles.json"
        with open(articles_index_path, 'w', encoding='utf-8') as f:
            json.dump(articles_generated, f, indent=2)
        print(f"  Saved: articles.json ({len(articles_generated)} articles)")

        self.stats['articles'] = len(articles_generated)
        return articles_generated

    def _find_entities_for_article(self, entities_data: Dict, keyword: str) -> List[Dict]:
        """Find entities matching a keyword for article generation."""
        matching = []

        for entity_type, entities in entities_data.items():
            for entity in entities:
                # Case-insensitive partial match
                if keyword.lower() in entity['text'].lower():
                    matching.append({
                        'type': entity_type,
                        'text': entity['text'],
                        'doc_count': entity['doc_count'],
                        'confidence': entity['confidence'],
                        'documents': entity['documents']
                    })

        # Sort by document count (most referenced first)
        matching.sort(key=lambda x: x['doc_count'], reverse=True)
        return matching

    def _create_article(self, keyword: str, category: str, entities: List[Dict], all_entities: Dict) -> Dict:
        """Create an article from entity data."""
        if not entities:
            return None

        # Get main entity (highest doc count)
        main_entity = entities[0]

        # Generate article filename
        safe_keyword = re.sub(r'[^\w\-]', '_', keyword.lower())
        filename = f"{safe_keyword}.html"
        filepath = self.output_dir / "articles" / filename

        # Gather related content
        related_entities = self._find_related_entities(main_entity, all_entities)
        code_examples = self._extract_code_examples(main_entity)

        # Generate article HTML
        html_content = self._generate_article_html(
            keyword, category, main_entity, entities, related_entities, code_examples
        )

        # Write article file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return {
            'title': keyword,
            'category': category,
            'filename': filename,
            'entity_count': len(entities),
            'doc_count': main_entity['doc_count'],
            'related_count': len(related_entities)
        }

    def _find_related_entities(self, entity: Dict, all_entities: Dict, max_related: int = 10) -> List[Dict]:
        """Find entities related to the main entity (appear in same documents)."""
        # Get document IDs for main entity
        main_doc_ids = set(doc['id'] for doc in entity['documents'])

        related = []
        for entity_type, entities in all_entities.items():
            for ent in entities:
                if ent['text'] == entity['text']:
                    continue  # Skip self

                # Check for document overlap
                ent_doc_ids = set(doc['id'] for doc in ent['documents'])
                overlap = main_doc_ids & ent_doc_ids

                if overlap:
                    related.append({
                        'type': entity_type,
                        'text': ent['text'],
                        'doc_count': ent['doc_count'],
                        'overlap_count': len(overlap),
                        'overlap_ratio': len(overlap) / len(main_doc_ids)
                    })

        # Sort by overlap ratio
        related.sort(key=lambda x: (x['overlap_ratio'], x['doc_count']), reverse=True)
        return related[:max_related]

    def _extract_code_examples(self, entity: Dict, max_examples: int = 5) -> List[Dict]:
        """Extract code examples from documents mentioning this entity."""
        examples = []

        # Get chunks from documents
        for doc in entity['documents'][:max_examples]:
            # Fetch chunks for this document
            cursor = self.kb.db_conn.cursor()
            chunks = cursor.execute("""
                SELECT content, page
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_id
                LIMIT 3
            """, (doc['id'],)).fetchall()

            for content, page in chunks:
                # Look for code-like patterns
                if any(indicator in content.lower() for indicator in ['$', 'lda', 'sta', 'jsr', 'rts', 'register']):
                    examples.append({
                        'doc_title': doc['title'],
                        'doc_id': doc['id'],
                        'doc_filename': doc['filename'],
                        'content': content[:500] + '...' if len(content) > 500 else content,
                        'page': page
                    })
                    break  # One example per document

        return examples

    def _calculate_reading_time(self, content: str) -> int:
        """Calculate estimated reading time in minutes."""
        words = len(content.split())
        # Average reading speed: 200 words per minute
        minutes = max(1, round(words / 200))
        return minutes

    def _generate_article_html(self, title: str, category: str, main_entity: Dict,
                               all_matches: List[Dict], related: List[Dict],
                               code_examples: List[Dict]) -> str:
        """Generate HTML for an article page."""
        title_escaped = html.escape(title)

        # Calculate reading time (count words from all sections)
        total_words = 0
        for entity in all_matches[:10]:
            total_words += len(entity['text'].split())
        for rel in related:
            total_words += len(rel['text'].split())
        for example in code_examples:
            total_words += len(example['content'].split())
        reading_time = self._calculate_reading_time(' ' * total_words)
        word_count = total_words

        # Build overview section
        overview_html = f"""
        <div class="article-overview">
            <h2>Overview</h2>
            <p><strong>Category:</strong> {html.escape(category)}</p>
            <p><strong>Referenced in:</strong> {main_entity['doc_count']} documents</p>
            <p><strong>Entity Type:</strong> {html.escape(main_entity['type'])}</p>
            <p><strong>Confidence:</strong> {(main_entity['confidence'] * 100):.0f}%</p>
        </div>
        """

        # Build entities section
        entities_html = '<div class="article-section"><h2>Related Entities</h2><ul class="entity-list-article">'
        for entity in all_matches[:10]:
            entities_html += f'<li><strong>{html.escape(entity["text"])}</strong> ({entity["type"]}) - {entity["doc_count"]} docs</li>'
        entities_html += '</ul></div>'

        # Build related topics section
        related_html = ''
        if related:
            related_html = '<div class="article-section"><h2>Related Topics</h2><ul class="related-list">'
            for rel in related:
                related_html += f'<li><strong>{html.escape(rel["text"])}</strong> ({rel["type"]}) - appears in {rel["overlap_count"]} common documents</li>'
            related_html += '</ul></div>'

        # Build code examples section
        code_html = ''
        if code_examples:
            code_html = '<div class="article-section"><h2>Code Examples & Technical Details</h2>'
            for i, example in enumerate(code_examples, 1):
                code_html += f"""
                <div class="code-example">
                    <h3>Example {i} - from <a href="../docs/{example['doc_filename']}">{html.escape(example['doc_title'])}</a></h3>
                    <pre><code>{html.escape(example['content'])}</code></pre>
                </div>
                """
            code_html += '</div>'

        # Build documents section
        docs_html = '<div class="article-section"><h2>Source Documents</h2><ul class="doc-list-article">'
        for doc in main_entity['documents'][:20]:
            docs_html += f'<li><a href="../docs/{doc["filename"]}">{html.escape(doc["title"])}</a></li>'
        if len(main_entity['documents']) > 20:
            docs_html += f'<li><em>...and {len(main_entity["documents"]) - 20} more documents</em></li>'
        docs_html += '</ul></div>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_escaped} - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <style>
        .article-overview {{
            background: var(--bg-color);
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid var(--accent-color);
        }}
        .article-section {{
            margin: 30px 0;
        }}
        .article-section h2 {{
            color: var(--secondary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .entity-list-article, .related-list, .doc-list-article {{
            list-style: none;
            padding: 0;
        }}
        .entity-list-article li, .related-list li, .doc-list-article li {{
            padding: 10px;
            margin: 5px 0;
            background: var(--card-bg);
            border-radius: 5px;
        }}
        .code-example {{
            margin: 20px 0;
            padding: 15px;
            background: var(--bg-color);
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
        }}
        .code-example h3 {{
            margin-top: 0;
            color: var(--primary-color);
        }}
        .code-example pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }}
        .article-category {{
            display: inline-block;
            padding: 5px 15px;
            background: var(--accent-color);
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle"><a href="../index.html">← Back to Home</a> | <a href="../articles.html">← Back to Articles</a></p>
        </header>

        <nav class="main-nav">
            <a href="../index.html">Home</a>
            <a href="../articles.html">Articles</a>
            <a href="../documents.html">Documents</a>
            <a href="../chunks.html">Chunks</a>
            <a href="../entities.html">Entities</a>
            <a href="../topics.html">Topics</a>
            <a href="../timeline.html">Timeline</a>
        </nav>

        <main>
            <article class="article-content">
                <span class="article-category">{html.escape(category)}</span>
                <h1>{title_escaped}</h1>

                <div class="article-meta">
                    <span class="reading-time">⏱️ {reading_time} min read</span>
                    <span class="word-count">📄 ~{word_count} words</span>
                    <span class="doc-count">📚 {main_entity['doc_count']} documents</span>
                </div>

                {overview_html}
                {entities_html}
                {related_html}
                {code_html}
                {docs_html}
            </article>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="../assets/js/enhancements.js"></script>
</body>
</html>
"""
        return html_content

    def _generate_articles_browser_html(self, articles: List[Dict]):
        """Generate articles browser page."""
        # Group articles by category
        by_category = {}
        for article in articles:
            cat = article['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(article)

        # Sort each category by doc count
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x['doc_count'], reverse=True)

        # Build category sections
        categories_html = ''
        for category in sorted(by_category.keys()):
            articles_in_cat = by_category[category]
            categories_html += f"""
            <div class="article-category-section">
                <h2>{html.escape(category)}</h2>
                <div class="articles-grid">
            """

            for article in articles_in_cat:
                categories_html += f"""
                <div class="article-card">
                    <h3><a href="articles/{article['filename']}">{html.escape(article['title'])}</a></h3>
                    <div class="article-meta">
                        <span>📚 {article['doc_count']} documents</span>
                        <span>🔗 {article['related_count']} related topics</span>
                    </div>
                </div>
                """

            categories_html += '</div></div>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Articles - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .article-category-section {{
            margin: 40px 0;
        }}
        .article-category-section h2 {{
            color: var(--secondary-color);
            border-bottom: 3px solid var(--accent-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .articles-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .article-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s;
        }}
        .article-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .article-card h3 {{
            margin: 0 0 10px 0;
        }}
        .article-card h3 a {{
            color: var(--secondary-color);
            text-decoration: none;
        }}
        .article-card h3 a:hover {{
            color: var(--accent-color);
        }}
        .article-meta {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: var(--primary-color);
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Articles & Guides</p>
        </header>

        <nav class="main-nav">
            <a href="index.html">Home</a>
            <a href="articles.html" class="active">Articles</a>
            <a href="documents.html">Documents</a>
            <a href="chunks.html">Chunks</a>
            <a href="entities.html">Entities</a>
            <a href="topics.html">Topics</a>
            <a href="timeline.html">Timeline</a>
        </nav>

        <main>
            <section class="intro">
                <h2>Knowledge Base Articles</h2>
                <p>Automatically generated articles based on entity extraction and document analysis.
                   Each article aggregates information from multiple sources to provide comprehensive coverage
                   of key Commodore 64 topics.</p>
                <p><strong>Total Articles:</strong> {len(articles)}</p>
            </section>

            {categories_html}
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.15</p>
        </footer>
    </div>

    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "articles.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: articles.html")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Export KB to static HTML wiki')
    parser.add_argument('--output', default='wiki', help='Output directory (default: wiki/)')
    parser.add_argument('--data-dir', help='Knowledge base data directory')

    args = parser.parse_args()

    # Initialize KB
    data_dir = args.data_dir or os.path.expanduser('~/.tdz-c64-knowledge')
    print(f"Loading knowledge base from: {data_dir}")
    kb = KnowledgeBase(data_dir)

    # Export
    exporter = WikiExporter(kb, args.output)
    exporter.export()


if __name__ == '__main__':
    main()
