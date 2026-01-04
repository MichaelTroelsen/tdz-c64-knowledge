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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing


class WikiExporter:
    """Exports knowledge base to static HTML wiki."""

    def __init__(self, kb: KnowledgeBase, output_dir: str):
        self.kb = kb
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "docs"
        self.assets_dir = self.output_dir / "assets"
        self.data_dir = self.assets_dir / "data"
        self.files_dir = self.output_dir / "files"  # Directory for actual source files

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

    def _get_unified_about_box(self) -> str:
        """Generate unified about box HTML for all pages."""
        return """
            <div class="explanation-box">
                <h3>📚 About This Knowledge Base</h3>
                <p>
                    The <strong>TDZ C64 Knowledge Base</strong> is a comprehensive collection of Commodore 64 documentation,
                    tutorials, technical references, and historical information. All content has been processed and indexed
                    for easy searching and exploration.
                </p>
                <p>
                    <strong>Features:</strong> Full-text search across all documents • Entity extraction (chips, hardware, software) •
                    Interactive knowledge graph • Topic modeling and clustering • Timeline of C64 history • AI-powered Q&A •
                    Automatic article generation from extracted knowledge
                </p>
                <p>
                    <strong>Navigation:</strong> Use the menu above to explore different views of the knowledge base. Click the 🤖 Ask AI
                    button (bottom right) to ask questions about any C64 topic.
                </p>
            </div>
"""

    def _get_main_nav(self, active_page: str = '') -> str:
        """Generate consistent main navigation HTML with logo and theme switcher."""
        pages = [
            ('articles', 'Articles'),
            ('documents', 'Documents'),
            ('chunks', 'Chunks'),
            ('entities', 'Entities'),
            ('knowledge-graph', 'Knowledge Graph'),
            ('similarity-map', 'Similarity Map'),
            ('topics', 'Topics'),
            ('timeline', 'Timeline')
        ]

        nav_items = []
        for page_key, display_name in pages:
            active_class = ' class="active"' if page_key == active_page else ''
            page_file = page_key + '.html'
            nav_items.append(f'            <a href="{page_file}"{active_class}>{display_name}</a>')

        return f"""    <nav class="main-nav">
        <div class="nav-left">
            <a href="index.html" class="nav-logo">📚 TDZ C64 KB</a>
        </div>
        <div class="nav-center">
{chr(10).join(nav_items)}
        </div>
        <div class="nav-right">
            <button class="theme-switcher" id="theme-toggle" aria-label="Toggle theme">🌙</button>
        </div>
    </nav>"""

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
        self._copy_source_files(documents_data)

        print("[3/7] Exporting entities...")
        entities_data = self._export_entities()
        graph_data = self._export_graph()

        print("  Exporting document coordinates...")
        coordinates_data = self._export_document_coordinates(documents_data)

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
        # Save data files
        print("\nSaving data files...")
        self._save_json('documents.json', documents_data)
        self._save_json('entities.json', entities_data)
        self._save_json('graph.json', graph_data)
        self._save_json('coordinates.json', coordinates_data)
        self._save_json('topics.json', topics_data)
        self._save_json('clusters.json', clusters_data)
        self._save_json('events.json', events_data)
        self._save_json('search-index.json', search_index)
        self._save_json('navigation.json', navigation)
        self._save_json('chunks.json', chunks_data)
        self._save_json('stats.json', self.stats)

        # Calculate document similarities
        print("\nCalculating document similarities...")
        similarities = self._calculate_document_similarities(documents_data, entities_data)
        self._save_json('similarities.json', similarities)

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
        self.files_dir.mkdir(exist_ok=True)  # For actual source files

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

            # Detect file type from extension for better display
            file_type = doc_meta.file_type
            if file_type == 'text':
                filename_lower = doc_meta.filename.lower()
                if filename_lower.endswith('.html') or filename_lower.endswith('.htm'):
                    file_type = 'html'
                elif filename_lower.endswith('.md') or filename_lower.endswith('.markdown'):
                    file_type = 'markdown'

            # Get file path if available
            filepath = getattr(doc_meta, 'filepath', None)

            doc_data = {
                'id': doc_id,
                'title': doc_meta.title,
                'filename': doc_meta.filename,
                'filepath': filepath,
                'file_type': file_type,
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

    def _copy_source_files(self, documents_data: List[Dict]):
        """Copy source files to the files directory for direct viewing."""
        print("  Copying source files to wiki...")
        copied_count = 0

        for doc in documents_data:
            filepath = doc.get('filepath')
            if not filepath or not os.path.exists(filepath):
                continue

            # Create a safe filename
            doc_id = doc['id']
            file_ext = Path(filepath).suffix
            safe_filename = re.sub(r'[^\w\-]', '_', doc_id) + file_ext
            dest_path = self.files_dir / safe_filename

            try:
                shutil.copy2(filepath, dest_path)
                # Store the relative path for linking
                doc['file_path_in_wiki'] = f"files/{safe_filename}"
                copied_count += 1
            except Exception as e:
                print(f"    Warning: Could not copy {filepath}: {e}")

        print(f"  Copied {copied_count} source files")

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

    def _export_graph(self) -> Dict:
        """Export entity graph data for visualization."""
        cursor = self.kb.db_conn.cursor()

        # Build nodes from all entities with their counts
        nodes_dict = {}

        # Get all entities with document counts
        entities_query = """
            SELECT entity_text, entity_type, COUNT(DISTINCT doc_id) as doc_count
            FROM document_entities
            GROUP BY entity_text, entity_type
            HAVING doc_count >= 2
            ORDER BY doc_count DESC
        """

        entities = cursor.execute(entities_query).fetchall()

        for entity_text, entity_type, doc_count in entities:
            nodes_dict[entity_text] = {
                'id': entity_text,
                'label': entity_text,
                'type': entity_type or 'UNKNOWN',
                'count': doc_count,
                'value': doc_count  # For node sizing
            }

        # Get relationships (edges)
        relationships_query = """
            SELECT entity1_text, entity2_text, relationship_type,
                   strength, doc_count
            FROM entity_relationships
            WHERE strength >= 0.3
            ORDER BY strength DESC
            LIMIT 5000
        """

        relationships = cursor.execute(relationships_query).fetchall()

        edges = []
        for e1, e2, rel_type, strength, doc_count in relationships:
            # Only include edges between nodes we have
            if e1 in nodes_dict and e2 in nodes_dict:
                edges.append({
                    'source': e1,
                    'target': e2,
                    'type': rel_type,
                    'weight': round(strength, 2),
                    'doc_count': doc_count,
                    'value': doc_count  # For edge thickness
                })

        # Convert nodes dict to list
        nodes = list(nodes_dict.values())

        print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")

        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': len(set(n['type'] for n in nodes))
            }
        }

    def _export_document_coordinates(self, documents_data: List[Dict]) -> Dict:
        """Export 2D coordinates for document similarity visualization."""
        try:
            import numpy as np
            try:
                import umap
                use_umap = True
            except ImportError:
                from sklearn.manifold import TSNE
                use_umap = False

            # Check if embeddings are available
            if not hasattr(self.kb, 'embeddings') or self.kb.embeddings is None:
                print("  No embeddings available, loading...")
                self.kb._load_embeddings()

            # Check again after loading
            if not hasattr(self.kb, 'embeddings') or self.kb.embeddings is None or len(self.kb.embeddings) == 0:
                print("  No embeddings found, skipping coordinates export")
                return {'documents': [], 'method': 'none', 'count': 0}

            # Get embeddings for documents
            doc_ids = [doc['id'] for doc in documents_data]
            embeddings_list = []
            valid_docs = []

            for doc_id in doc_ids:
                if doc_id in self.kb.embeddings:
                    embeddings_list.append(self.kb.embeddings[doc_id])
                    valid_docs.append(next(d for d in documents_data if d['id'] == doc_id))

            if len(embeddings_list) < 2:
                print("  Insufficient embeddings for visualization")
                return {'documents': [], 'method': 'none'}

            embeddings_array = np.array(embeddings_list)

            # Reduce to 2D
            print(f"  Reducing {len(embeddings_list)} embeddings to 2D using {'UMAP' if use_umap else 't-SNE'}...")

            if use_umap:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
            else:
                reducer = TSNE(
                    n_components=2,
                    perplexity=min(30, len(embeddings_list) - 1),
                    random_state=42
                )

            coordinates_2d = reducer.fit_transform(embeddings_array)

            # Normalize coordinates to 0-1000 range for easier visualization
            x_coords = coordinates_2d[:, 0]
            y_coords = coordinates_2d[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            x_norm = ((x_coords - x_min) / (x_max - x_min)) * 1000
            y_norm = ((y_coords - y_min) / (y_max - y_min)) * 1000

            # Get cluster information
            cursor = self.kb.db_conn.cursor()
            doc_clusters = {}
            clusters = cursor.execute("""
                SELECT dc.doc_id, c.cluster_number, c.algorithm
                FROM document_clusters dc
                JOIN clusters c ON dc.cluster_id = c.cluster_id
                WHERE c.algorithm = 'kmeans'
            """).fetchall()

            for doc_id, cluster_num, algorithm in clusters:
                if isinstance(cluster_num, (bytes, memoryview)):
                    cluster_num = int.from_bytes(bytes(cluster_num), byteorder='little')
                doc_clusters[doc_id] = cluster_num

            # Build coordinate data
            coord_data = []
            for i, doc in enumerate(valid_docs):
                coord_data.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'filename': doc['filename'],
                    'file_type': doc['file_type'],
                    'tags': doc['tags'][:5],  # Limit tags
                    'total_chunks': doc['total_chunks'],
                    'cluster': doc_clusters.get(doc['id'], 0),
                    'x': float(x_norm[i]),
                    'y': float(y_norm[i])
                })

            print(f"  Generated {len(coord_data)} document coordinates")

            return {
                'documents': coord_data,
                'method': 'umap' if use_umap else 'tsne',
                'count': len(coord_data)
            }

        except Exception as e:
            print(f"  Error generating coordinates: {e}")
            return {'documents': [], 'method': 'error', 'error': str(e)}

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

                # Get documents in this cluster
                docs = cursor.execute("""
                    SELECT d.doc_id, d.title, d.filename
                    FROM documents d
                    JOIN document_clusters dc ON d.doc_id = dc.doc_id
                    WHERE dc.cluster_id = ?
                    ORDER BY d.title
                    LIMIT 50
                """, (cluster_id,)).fetchall()

                processed_clusters.append({
                    'id': cluster_id,
                    'number': cluster_num,
                    'doc_count': doc_count,
                    'documents': [
                        {
                            'id': doc_id,
                            'title': title,
                            'filename': filename
                        }
                        for doc_id, title, filename in docs
                    ]
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

    def _calculate_document_similarities(self, documents: List[Dict], entities_data: Dict) -> Dict:
        """Calculate document similarities based on entity overlap and tags."""
        print("  Computing similarities...")

        # Build entity-to-documents mapping
        entity_docs = {}
        cursor = self.kb.db_conn.cursor()

        # Get all entity-document associations
        entity_mappings = cursor.execute("""
            SELECT entity_text, doc_id
            FROM document_entities
            WHERE confidence > 0.7
        """).fetchall()

        for entity_text, doc_id in entity_mappings:
            if entity_text not in entity_docs:
                entity_docs[entity_text] = set()
            entity_docs[entity_text].add(doc_id)

        # Calculate similarities for each document
        similarities = {}

        for doc in documents:
            doc_id = doc['id']

            # Get entities for this document
            doc_entities = cursor.execute("""
                SELECT DISTINCT entity_text
                FROM document_entities
                WHERE doc_id = ? AND confidence > 0.7
            """, (doc_id,)).fetchall()

            doc_entity_set = set(e[0] for e in doc_entities)
            doc_tags = set(doc['tags'])

            # Calculate similarity to all other documents
            similar_docs = []

            for other_doc in documents:
                if other_doc['id'] == doc_id:
                    continue  # Skip self

                # Get entities for other document
                other_entities = cursor.execute("""
                    SELECT DISTINCT entity_text
                    FROM document_entities
                    WHERE doc_id = ? AND confidence > 0.7
                """, (other_doc['id'],)).fetchall()

                other_entity_set = set(e[0] for e in other_entities)
                other_tags = set(other_doc['tags'])

                # Calculate entity overlap (Jaccard similarity)
                if len(doc_entity_set) > 0 or len(other_entity_set) > 0:
                    entity_intersection = len(doc_entity_set & other_entity_set)
                    entity_union = len(doc_entity_set | other_entity_set)
                    entity_similarity = entity_intersection / entity_union if entity_union > 0 else 0
                else:
                    entity_similarity = 0

                # Calculate tag overlap
                if len(doc_tags) > 0 or len(other_tags) > 0:
                    tag_intersection = len(doc_tags & other_tags)
                    tag_union = len(doc_tags | other_tags)
                    tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0
                else:
                    tag_similarity = 0

                # Combined similarity score (weighted)
                combined_score = (entity_similarity * 0.7) + (tag_similarity * 0.3)

                if combined_score > 0.1:  # Only include if somewhat similar
                    similar_docs.append({
                        'id': other_doc['id'],
                        'title': other_doc['title'],
                        'filename': re.sub(r'[^\w\-]', '_', other_doc['id']) + '.html',
                        'score': round(combined_score, 3),
                        'common_entities': len(doc_entity_set & other_entity_set),
                        'common_tags': len(doc_tags & other_tags)
                    })

            # Sort by similarity score and take top 10
            similar_docs.sort(key=lambda x: x['score'], reverse=True)
            similarities[doc_id] = similar_docs[:10]

        print(f"  Computed similarities for {len(similarities)} documents")
        return similarities

    def _save_json(self, filename: str, data: Any):
        """Save data as JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filename}")

    def _generate_html_pages(self, documents: List[Dict]):
        """Generate HTML pages for all documents (parallelized)."""
        # Generate index page
        self._generate_index_html()

        # Generate browser pages
        self._generate_documents_browser_html()
        self._generate_chunks_browser_html()

        # Generate document pages in parallel
        print(f"  Generating {len(documents)} document pages in parallel...")
        max_workers = min(multiprocessing.cpu_count() * 2, 8)  # Limit to 8 workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all document generation tasks
            futures = [executor.submit(self._generate_doc_html, doc) for doc in documents]

            # Wait for completion and show progress
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()  # Raise any exceptions
                    completed += 1
                    if completed % 10 == 0 or completed == len(documents):
                        print(f"    Progress: {completed}/{len(documents)} documents")
                except Exception as e:
                    print(f"    Error generating document: {e}")

        # Generate entity pages
        self._generate_entities_html()

        # Generate knowledge graph page
        self._generate_knowledge_graph_html()

        # Generate similarity map page
        self._generate_similarity_map_html()

        # Generate topics page
        self._generate_topics_html()

        # Generate timeline page
        self._generate_timeline_html()

        # Generate file viewer page
        self._generate_file_viewer_html()

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

{self._get_main_nav()}

        <div class="search-section">
            <input type="text" id="search-input" placeholder="Search the knowledge base..." autocomplete="off">
            <div id="search-results" class="search-results"></div>
        </div>

        <main>
{self._get_unified_about_box()}

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

            <section id="popular-articles" class="popular-articles-section">
                <!-- Will be populated by JavaScript from articles.json -->
            </section>

            <section class="tag-cloud-section">
                <h2>🏷️ Explore by Topic</h2>
                <div class="tag-cloud-controls">
                    <button class="tag-cloud-control-btn active" data-sort="count">Most Popular</button>
                    <button class="tag-cloud-control-btn" data-sort="alpha">Alphabetical</button>
                    <button class="tag-cloud-control-btn" data-sort="random">Shuffle</button>
                </div>
                <div class="tag-cloud" id="tag-cloud">
                    <div class="tag-cloud-loading">Loading tags...</div>
                </div>
                <div class="tag-cloud-stats" id="tag-cloud-stats"></div>
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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
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

        <nav class="breadcrumbs">
            <a href="../index.html">🏠 Home</a>
            <span class="separator">›</span>
            <a href="../documents.html">Documents</a>
            <span class="separator">›</span>
            <span class="current">{title_escaped}</span>
        </nav>

        <main class="doc-with-sidebar">
            <article class="document doc-main-content">
                <div class="doc-header">
                    <h1>{title_escaped}</h1>
                    <div class="doc-meta">
                        <span class="meta-item">📄 {html.escape(doc['file_type'])}</span>
                        <span class="meta-item">📊 {doc['total_chunks']} chunks</span>
                        {f'<span class="meta-item">📑 {doc["total_pages"]} pages</span>' if doc['total_pages'] else ''}
                    </div>
                    {f'<div class="doc-tags">{tags_html}</div>' if tags_html else ''}
                    {f'<div class="doc-url"><a href="{html.escape(doc["source_url"])}" target="_blank">🔗 Source URL</a></div>' if doc.get('source_url') else ''}
                    {f'<div class="doc-url"><a href="../viewer.html?file={doc["file_path_in_wiki"]}&name={html.escape(doc["filename"])}&type={doc["file_type"]}" target="_blank" style="background: var(--accent-color); color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; display: inline-block; margin-top: 10px;">📄 View Source File</a></div>' if doc.get('file_path_in_wiki') else ''}
                </div>

                <div class="chunks-container">
                    {''.join(chunks_html)}
                </div>
            </article>

            <aside class="related-docs-sidebar" id="related-docs-sidebar" data-doc-id="{doc_id}">
                <h3>Related Documents</h3>
                <div class="loading-related">Loading related documents...</div>
            </aside>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
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
        html_template = """<!DOCTYPE html>
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

{NAV}

        <main>
{ABOUT}

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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="assets/js/entities.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace('{NAV}', self._get_main_nav('entities'))
        html_content = html_content.replace('{ABOUT}', self._get_unified_about_box())

        filepath = self.output_dir / "entities.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: entities.html")

    def _generate_knowledge_graph_html(self):
        """Generate knowledge graph visualization page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .graph-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
            margin: 20px 0;
            height: calc(100vh - 200px);
        }

        .graph-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .graph-controls h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .control-group {
            margin: 20px 0;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-color);
        }

        .control-group input[type="text"],
        .control-group input[type="range"] {
            width: 100%;
            padding: 8px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: inherit;
        }

        .control-group input[type="text"]:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        .filter-checkboxes {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }

        .filter-checkboxes label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: normal;
            cursor: pointer;
            padding: 6px;
            border-radius: 6px;
            transition: background 0.2s;
        }

        .filter-checkboxes label:hover {
            background: var(--bg-color);
        }

        .filter-checkboxes input[type="checkbox"] {
            cursor: pointer;
        }

        .type-legend {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 10px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid var(--border-color);
        }

        #graph-canvas {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }

        #graph-svg {
            width: 100%;
            height: 100%;
            cursor: move;
        }

        .graph-info {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .graph-info h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .info-empty {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }

        .node-info {
            animation: fadeIn 0.3s;
        }

        .node-info h4 {
            color: var(--accent-color);
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }

        .node-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }

        .stat-item {
            background: var(--bg-color);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .connections-list {
            margin-top: 15px;
        }

        .connections-list h5 {
            margin: 10px 0;
            color: var(--secondary-color);
        }

        .connection-item {
            padding: 8px;
            margin: 5px 0;
            background: var(--bg-color);
            border-radius: 6px;
            border-left: 3px solid var(--accent-color);
            cursor: pointer;
            transition: all 0.2s;
        }

        .connection-item:hover {
            background: var(--border-color);
            transform: translateX(4px);
        }

        .connection-name {
            font-weight: 600;
            color: var(--text-color);
        }

        .connection-strength {
            font-size: 0.85em;
            color: var(--text-muted);
        }

        .graph-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }

        .graph-stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .graph-stat-card .stat-value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .graph-stat-card .stat-label {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-top: 8px;
        }

        .loading-message {
            text-align: center;
            padding: 60px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        .zoom-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 10;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-size: 1.5em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        /* Node and edge styles */
        .node {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.2);
        }

        .node.highlighted {
            stroke: var(--accent-color);
            stroke-width: 4px;
        }

        .node.dimmed {
            opacity: 0.3;
        }

        .link {
            stroke: var(--border-color);
            stroke-opacity: 0.6;
            transition: all 0.3s;
        }

        .link.highlighted {
            stroke: var(--accent-color);
            stroke-width: 3px;
            stroke-opacity: 1;
        }

        .link.dimmed {
            opacity: 0.1;
        }

        .node-label {
            font-size: 11px;
            pointer-events: none;
            text-anchor: middle;
            fill: var(--text-color);
            font-weight: 600;
        }

        .node-label.hidden {
            display: none;
        }

        @media (max-width: 1200px) {
            .graph-container {
                grid-template-columns: 1fr;
                height: auto;
            }

            .graph-info {
                order: -1;
            }

            #graph-canvas {
                height: 600px;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🕸️ Knowledge Graph</h1>
            <p class="subtitle">Explore entity relationships and connections</p>
        </header>

{NAV}

{ABOUT}

        <div class="graph-stats" id="graph-stats">
            <div class="graph-stat-card">
                <div class="stat-value" id="total-nodes">-</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="graph-stat-card">
                <div class="stat-value" id="total-edges">-</div>
                <div class="stat-label">Connections</div>
            </div>
            <div class="graph-stat-card">
                <div class="stat-value" id="total-types">-</div>
                <div class="stat-label">Entity Types</div>
            </div>
        </div>

        <div class="graph-container">
            <div class="graph-controls">
                <h3>Controls</h3>

                <div class="control-group">
                    <label for="search-node">🔍 Search Entity</label>
                    <input type="text" id="search-node" placeholder="Type entity name...">
                </div>

                <div class="control-group">
                    <label>🎨 Filter by Type</label>
                    <div class="filter-checkboxes" id="type-filters"></div>
                </div>

                <div class="control-group">
                    <label for="min-connections">Minimum Connections: <span id="min-connections-value">0</span></label>
                    <input type="range" id="min-connections" min="0" max="20" value="0" step="1">
                </div>

                <div class="control-group">
                    <label>
                        <input type="checkbox" id="show-labels" checked> Show Labels
                    </label>
                </div>

                <div class="control-group">
                    <h3>Legend</h3>
                    <div class="type-legend" id="type-legend"></div>
                </div>
            </div>

            <div id="graph-canvas">
                <div class="loading-message">Loading knowledge graph...</div>
                <div class="zoom-controls">
                    <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
                    <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
                    <button class="zoom-btn" id="zoom-reset" title="Reset View">⟲</button>
                </div>
                <svg id="graph-svg"></svg>
            </div>

            <div class="graph-info">
                <h3>Node Details</h3>
                <div class="info-empty" id="info-empty">
                    Click on a node to view details
                </div>
                <div class="node-info" id="node-info" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="assets/js/enhancements.js"></script>
    <script>
        // Knowledge Graph Visualization using D3.js
        let graphData = null;
        let simulation = null;
        let currentTransform = d3.zoomIdentity;
        let selectedNode = null;

        // Color scheme for entity types
        const typeColors = {
            'HARDWARE': '#e74c3c',
            'SOFTWARE': '#3498db',
            'PERSON': '#2ecc71',
            'ORGANIZATION': '#f39c12',
            'CONCEPT': '#9b59b6',
            'MUSIC': '#1abc9c',
            'GRAPHICS': '#e67e22',
            'GAME': '#16a085',
            'UNKNOWN': '#95a5a6'
        };

        async function loadGraph() {
            try {
                const response = await fetch('assets/data/graph.json');
                graphData = await response.json();

                // Update stats
                document.getElementById('total-nodes').textContent = graphData.stats.total_nodes.toLocaleString();
                document.getElementById('total-edges').textContent = graphData.stats.total_edges.toLocaleString();
                document.getElementById('total-types').textContent = graphData.stats.node_types;

                // Build type filters
                const types = [...new Set(graphData.nodes.map(n => n.type))].sort();
                buildTypeFilters(types);
                buildTypeLegend(types);

                // Initialize graph
                initializeGraph();
            } catch (error) {
                console.error('Error loading graph:', error);
                document.querySelector('.loading-message').textContent = 'Error loading graph data';
            }
        }

        function buildTypeFilters(types) {
            const container = document.getElementById('type-filters');
            container.innerHTML = types.map(type => `
                <label>
                    <input type="checkbox" class="type-filter" value="${type}" checked>
                    <span style="color: ${typeColors[type] || typeColors.UNKNOWN}">${type}</span>
                </label>
            `).join('');

            // Add event listeners
            container.querySelectorAll('.type-filter').forEach(checkbox => {
                checkbox.addEventListener('change', updateGraph);
            });
        }

        function buildTypeLegend(types) {
            const container = document.getElementById('type-legend');
            container.innerHTML = types.map(type => `
                <div class="legend-item">
                    <div class="legend-color" style="background: ${typeColors[type] || typeColors.UNKNOWN}"></div>
                    <span>${type}</span>
                </div>
            `).join('');
        }

        function initializeGraph() {
            const svg = d3.select('#graph-svg');
            const container = document.getElementById('graph-canvas');
            const width = container.clientWidth;
            const height = container.clientHeight;

            svg.attr('width', width).attr('height', height);

            // Clear loading message
            document.querySelector('.loading-message').style.display = 'none';

            // Create zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    currentTransform = event.transform;
                    g.attr('transform', currentTransform);
                });

            svg.call(zoom);

            // Create container group
            const g = svg.append('g');

            // Create force simulation
            simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.edges)
                    .id(d => d.id)
                    .distance(d => 100 / (d.weight || 1)))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.value) * 3 + 10));

            // Draw edges
            const link = g.append('g')
                .selectAll('line')
                .data(graphData.edges)
                .join('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.value || 1));

            // Draw nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(graphData.nodes)
                .join('circle')
                .attr('class', 'node')
                .attr('r', d => Math.sqrt(d.value) * 3 + 5)
                .attr('fill', d => typeColors[d.type] || typeColors.UNKNOWN)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('click', (event, d) => {
                    event.stopPropagation();
                    selectNode(d, node, link);
                })
                .on('mouseover', function(event, d) {
                    d3.select(this).style('cursor', 'pointer');
                });

            // Add labels
            const labels = g.append('g')
                .selectAll('text')
                .data(graphData.nodes)
                .join('text')
                .attr('class', 'node-label')
                .attr('dy', -15)
                .text(d => d.label);

            // Update positions on each tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });

            // Store references for updates
            window.graphElements = { node, link, labels, g, svg, zoom };

            // Setup controls
            setupControls();
        }

        function selectNode(d, nodeSelection, linkSelection) {
            selectedNode = d;

            // Highlight connected nodes
            const connectedNodeIds = new Set();
            connectedNodeIds.add(d.id);

            const connectedEdges = graphData.edges.filter(e =>
                e.source.id === d.id || e.target.id === d.id
            );

            connectedEdges.forEach(e => {
                connectedNodeIds.add(e.source.id);
                connectedNodeIds.add(e.target.id);
            });

            // Update node styles
            nodeSelection
                .classed('highlighted', n => n.id === d.id)
                .classed('dimmed', n => !connectedNodeIds.has(n.id));

            // Update link styles
            linkSelection
                .classed('highlighted', e => e.source.id === d.id || e.target.id === d.id)
                .classed('dimmed', e => e.source.id !== d.id && e.target.id !== d.id);

            // Show node info
            showNodeInfo(d, connectedEdges);
        }

        function showNodeInfo(node, edges) {
            document.getElementById('info-empty').style.display = 'none';
            const infoDiv = document.getElementById('node-info');
            infoDiv.style.display = 'block';

            const connections = edges.map(e => ({
                node: e.source.id === node.id ? e.target : e.source,
                weight: e.weight,
                doc_count: e.doc_count
            })).sort((a, b) => b.weight - a.weight);

            infoDiv.innerHTML = `
                <h4>${escapeHtml(node.label)}</h4>
                <p style="color: ${typeColors[node.type]}; font-weight: 600;">${node.type}</p>

                <div class="node-stats">
                    <div class="stat-item">
                        <div class="stat-value">${node.count}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${connections.length}</div>
                        <div class="stat-label">Connections</div>
                    </div>
                </div>

                ${connections.length > 0 ? `
                    <div class="connections-list">
                        <h5>Connected Entities</h5>
                        ${connections.slice(0, 10).map(c => `
                            <div class="connection-item" onclick="focusOnNode('${c.node.id}')">
                                <div class="connection-name">${escapeHtml(c.node.label)}</div>
                                <div class="connection-strength">Strength: ${c.weight} • ${c.doc_count} shared docs</div>
                            </div>
                        `).join('')}
                        ${connections.length > 10 ? `<p style="text-align: center; color: var(--text-muted); margin-top: 10px;">... and ${connections.length - 10} more</p>` : ''}
                    </div>
                ` : ''}
            `;
        }

        function focusOnNode(nodeId) {
            const node = graphData.nodes.find(n => n.id === nodeId);
            if (node) {
                selectNode(node, window.graphElements.node, window.graphElements.link);

                // Center on node
                const svg = window.graphElements.svg;
                const g = window.graphElements.g;
                const zoom = window.graphElements.zoom;

                const width = svg.node().clientWidth;
                const height = svg.node().clientHeight;

                const scale = 1.5;
                const x = -node.x * scale + width / 2;
                const y = -node.y * scale + height / 2;

                svg.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
            }
        }

        function updateGraph() {
            const activeTypes = Array.from(document.querySelectorAll('.type-filter:checked'))
                .map(cb => cb.value);

            const minConnections = parseInt(document.getElementById('min-connections').value);
            const showLabels = document.getElementById('show-labels').checked;

            const { node, link, labels } = window.graphElements;

            // Filter nodes
            node.style('display', d => {
                const connections = graphData.edges.filter(e =>
                    e.source.id === d.id || e.target.id === d.id
                ).length;

                return activeTypes.includes(d.type) && connections >= minConnections ? null : 'none';
            });

            // Filter links
            link.style('display', d => {
                const sourceVisible = activeTypes.includes(d.source.type);
                const targetVisible = activeTypes.includes(d.target.type);
                return sourceVisible && targetVisible ? null : 'none';
            });

            // Toggle labels
            labels.classed('hidden', !showLabels);
        }

        function setupControls() {
            // Search
            document.getElementById('search-node').addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                if (query.length < 2) return;

                const matches = graphData.nodes.filter(n =>
                    n.label.toLowerCase().includes(query)
                );

                if (matches.length > 0) {
                    focusOnNode(matches[0].id);
                }
            });

            // Min connections slider
            document.getElementById('min-connections').addEventListener('input', (e) => {
                document.getElementById('min-connections-value').textContent = e.target.value;
                updateGraph();
            });

            // Show labels toggle
            document.getElementById('show-labels').addEventListener('change', updateGraph);

            // Zoom controls
            const { svg, zoom } = window.graphElements;

            document.getElementById('zoom-in').addEventListener('click', () => {
                svg.transition().call(zoom.scaleBy, 1.3);
            });

            document.getElementById('zoom-out').addEventListener('click', () => {
                svg.transition().call(zoom.scaleBy, 0.7);
            });

            document.getElementById('zoom-reset').addEventListener('click', () => {
                svg.transition().call(zoom.transform, d3.zoomIdentity);
            });

            // Click background to deselect
            svg.on('click', () => {
                window.graphElements.node.classed('highlighted', false).classed('dimmed', false);
                window.graphElements.link.classed('highlighted', false).classed('dimmed', false);
                document.getElementById('node-info').style.display = 'none';
                document.getElementById('info-empty').style.display = 'block';
            });
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize on load
        loadGraph();
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace('{NAV}', self._get_main_nav('knowledge-graph'))
        html_content = html_content.replace('{ABOUT}', self._get_unified_about_box())

        filepath = self.output_dir / "knowledge-graph.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: knowledge-graph.html")

    def _generate_similarity_map_html(self):
        """Generate document similarity map visualization page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Similarity Map - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .map-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
            margin: 20px 0;
            height: calc(100vh - 200px);
        }

        .map-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .map-controls h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .control-group {
            margin: 20px 0;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-color);
        }

        .control-group input[type="text"],
        .control-group select {
            width: 100%;
            padding: 8px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: inherit;
        }

        .control-group input[type="text"]:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        #canvas-container {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
            cursor: grab;
        }

        #canvas-container.grabbing {
            cursor: grabbing;
        }

        #similarity-canvas {
            display: block;
            width: 100%;
            height: 100%;
        }

        .map-info {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .map-info h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .info-empty {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }

        .doc-info {
            animation: fadeIn 0.3s;
        }

        .doc-info h4 {
            color: var(--accent-color);
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }

        .doc-meta {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }

        .meta-item {
            background: var(--bg-color);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .meta-value {
            font-size: 1.5em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .meta-label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .doc-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 15px 0;
        }

        .tag {
            display: inline-block;
            padding: 4px 10px;
            background: var(--accent-color);
            color: white;
            border-radius: 12px;
            font-size: 0.85em;
        }

        .map-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }

        .stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-card .label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }

        .loading-map {
            text-align: center;
            padding: 60px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        .zoom-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 10;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-size: 1.5em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .cluster-legend {
            margin-top: 15px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
            padding: 4px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .legend-item:hover {
            background: var(--bg-color);
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid var(--border-color);
        }

        @media (max-width: 1200px) {
            .map-container {
                grid-template-columns: 1fr;
                height: auto;
            }

            .map-info {
                order: -1;
            }

            #canvas-container {
                height: 600px;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🗺️ Document Similarity Map</h1>
            <p class="subtitle">Explore documents in 2D semantic space</p>
        </header>

{NAV}

{ABOUT}

        <div class="map-stats">
            <div class="stat-card">
                <div class="value" id="total-docs">-</div>
                <div class="label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-clusters">-</div>
                <div class="label">Clusters</div>
            </div>
            <div class="stat-card">
                <div class="value" id="reduction-method">-</div>
                <div class="label">Method</div>
            </div>
        </div>

        <div class="map-container">
            <div class="map-controls">
                <h3>Controls</h3>

                <div class="control-group">
                    <label for="search-doc">🔍 Search Document</label>
                    <input type="text" id="search-doc" placeholder="Type document title...">
                </div>

                <div class="control-group">
                    <label for="cluster-filter">🎨 Filter by Cluster</label>
                    <select id="cluster-filter">
                        <option value="all">All Clusters</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="file-type-filter">📁 Filter by Type</label>
                    <select id="file-type-filter">
                        <option value="all">All Types</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>
                        <input type="checkbox" id="show-labels" checked> Show Labels
                    </label>
                </div>

                <div class="cluster-legend">
                    <h3>Cluster Legend</h3>
                    <div id="legend-items"></div>
                </div>
            </div>

            <div id="canvas-container">
                <div class="loading-map">Loading similarity map...</div>
                <div class="zoom-controls">
                    <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
                    <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
                    <button class="zoom-btn" id="zoom-reset" title="Reset View">⟲</button>
                </div>
                <canvas id="similarity-canvas"></canvas>
            </div>

            <div class="map-info">
                <h3>Document Details</h3>
                <div class="info-empty" id="info-empty">
                    Hover over a point to view details<br>
                    Click to navigate to document
                </div>
                <div class="doc-info" id="doc-info" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script src="assets/js/enhancements.js"></script>
    <script>
        let documentsData = [];
        let canvas, ctx;
        let scale = 1;
        let offsetX = 0, offsetY = 0;
        let isDragging = false;
        let dragStartX, dragStartY;
        let hoveredDoc = null;
        let selectedCluster = 'all';
        let selectedFileType = 'all';
        let showLabels = true;
        let searchQuery = '';

        const clusterColors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#16a085', '#d35400', '#8e44ad',
            '#c0392b', '#27ae60', '#2980b9', '#f1c40f', '#95a5a6'
        ];

        async function loadMap() {
            try {
                const response = await fetch('assets/data/coordinates.json');
                const data = await response.json();

                if (!data.documents || data.documents.length === 0) {
                    document.querySelector('.loading-map').textContent = 'No coordinate data available';
                    return;
                }

                documentsData = data.documents;

                // Update stats
                const clusters = new Set(documentsData.map(d => d.cluster));
                document.getElementById('total-docs').textContent = documentsData.length;
                document.getElementById('total-clusters').textContent = clusters.size;
                document.getElementById('reduction-method').textContent = data.method.toUpperCase();

                // Build filters
                buildFilters();

                // Initialize canvas
                initCanvas();

                // Hide loading
                document.querySelector('.loading-map').style.display = 'none';
            } catch (error) {
                console.error('Error loading map:', error);
                document.querySelector('.loading-map').textContent = 'Error loading similarity map';
            }
        }

        function buildFilters() {
            const clusters = [...new Set(documentsData.map(d => d.cluster))].sort((a, b) => a - b);
            const fileTypes = [...new Set(documentsData.map(d => d.file_type))].sort();

            const clusterSelect = document.getElementById('cluster-filter');
            clusters.forEach(cluster => {
                const option = document.createElement('option');
                option.value = cluster;
                option.textContent = `Cluster ${cluster}`;
                clusterSelect.appendChild(option);
            });

            const fileTypeSelect = document.getElementById('file-type-filter');
            fileTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type.toUpperCase();
                fileTypeSelect.appendChild(option);
            });

            // Build legend
            const legendContainer = document.getElementById('legend-items');
            clusters.forEach(cluster => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${clusterColors[cluster % clusterColors.length]}"></div>
                    <span>Cluster ${cluster}</span>
                `;
                item.onclick = () => {
                    clusterSelect.value = cluster;
                    selectedCluster = cluster.toString();
                    renderCanvas();
                };
                legendContainer.appendChild(item);
            });

            // Setup event listeners
            document.getElementById('search-doc').addEventListener('input', (e) => {
                searchQuery = e.target.value.toLowerCase();
                renderCanvas();
            });

            clusterSelect.addEventListener('change', (e) => {
                selectedCluster = e.target.value;
                renderCanvas();
            });

            fileTypeSelect.addEventListener('change', (e) => {
                selectedFileType = e.target.value;
                renderCanvas();
            });

            document.getElementById('show-labels').addEventListener('change', (e) => {
                showLabels = e.target.checked;
                renderCanvas();
            });

            document.getElementById('zoom-in').addEventListener('click', () => {
                scale *= 1.2;
                renderCanvas();
            });

            document.getElementById('zoom-out').addEventListener('click', () => {
                scale /= 1.2;
                renderCanvas();
            });

            document.getElementById('zoom-reset').addEventListener('click', () => {
                scale = 1;
                offsetX = 0;
                offsetY = 0;
                renderCanvas();
            });
        }

        function initCanvas() {
            canvas = document.getElementById('similarity-canvas');
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            ctx = canvas.getContext('2d');

            // Mouse events
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseUp);
            canvas.addEventListener('wheel', onWheel);
            canvas.addEventListener('click', onClick);

            // Render
            renderCanvas();
        }

        function renderCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Filter documents
            const filtered = documentsData.filter(doc => {
                if (selectedCluster !== 'all' && doc.cluster !== parseInt(selectedCluster)) return false;
                if (selectedFileType !== 'all' && doc.file_type !== selectedFileType) return false;
                if (searchQuery && !doc.title.toLowerCase().includes(searchQuery)) return false;
                return true;
            });

            // Draw documents
            filtered.forEach(doc => {
                const x = (doc.x * scale) + offsetX + canvas.width / 2;
                const y = (doc.y * scale) + offsetY + canvas.height / 2;

                const color = clusterColors[doc.cluster % clusterColors.length];
                const isHovered = hoveredDoc && hoveredDoc.id === doc.id;
                const isSearchMatch = searchQuery && doc.title.toLowerCase().includes(searchQuery);

                // Draw point
                ctx.beginPath();
                ctx.arc(x, y, isHovered ? 8 : isSearchMatch ? 6 : 4, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = isHovered ? '#fff' : color;
                ctx.lineWidth = isHovered ? 3 : 1;
                ctx.stroke();

                // Draw label
                if (showLabels && (isHovered || isSearchMatch)) {
                    ctx.fillStyle = 'var(--text-color)';
                    ctx.font = '12px sans-serif';
                    ctx.fillText(doc.title.substring(0, 30), x + 10, y - 5);
                }
            });
        }

        function onMouseDown(e) {
            isDragging = true;
            dragStartX = e.clientX - offsetX;
            dragStartY = e.clientY - offsetY;
            document.getElementById('canvas-container').classList.add('grabbing');
        }

        function onMouseMove(e) {
            if (isDragging) {
                offsetX = e.clientX - dragStartX;
                offsetY = e.clientY - dragStartY;
                renderCanvas();
            } else {
                // Check hover
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                hoveredDoc = null;
                for (const doc of documentsData) {
                    const x = (doc.x * scale) + offsetX + canvas.width / 2;
                    const y = (doc.y * scale) + offsetY + canvas.height / 2;
                    const dist = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2);

                    if (dist < 10) {
                        hoveredDoc = doc;
                        showDocInfo(doc);
                        break;
                    }
                }

                if (!hoveredDoc) {
                    hideDocInfo();
                }

                renderCanvas();
            }
        }

        function onMouseUp() {
            isDragging = false;
            document.getElementById('canvas-container').classList.remove('grabbing');
        }

        function onWheel(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= delta;
            renderCanvas();
        }

        function onClick(e) {
            if (hoveredDoc) {
                window.location.href = `docs/${hoveredDoc.filename}`;
            }
        }

        function showDocInfo(doc) {
            document.getElementById('info-empty').style.display = 'none';
            const infoDiv = document.getElementById('doc-info');
            infoDiv.style.display = 'block';

            const color = clusterColors[doc.cluster % clusterColors.length];
            const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

            infoDiv.innerHTML = `
                <h4>${escapeHtml(doc.title)}</h4>
                <div class="doc-meta">
                    <div class="meta-item">
                        <div class="meta-value" style="color: ${color}">${doc.cluster}</div>
                        <div class="meta-label">Cluster</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-value">${doc.total_chunks}</div>
                        <div class="meta-label">Chunks</div>
                    </div>
                </div>
                <div class="doc-tags">${tags}</div>
                <p style="font-size: 0.9em; color: var(--text-muted);">Type: ${doc.file_type.toUpperCase()}</p>
                <p style="font-size: 0.9em; color: var(--text-muted); margin-top: 10px;">Click to open document</p>
            `;
        }

        function hideDocInfo() {
            document.getElementById('doc-info').style.display = 'none';
            document.getElementById('info-empty').style.display = 'block';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load map on page load
        loadMap();

        // Resize handler
        window.addEventListener('resize', () => {
            if (canvas) {
                const container = document.getElementById('canvas-container');
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
                renderCanvas();
            }
        });
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("similarity-map"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "similarity-map.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: similarity-map.html")

    def _generate_topics_html(self):
        """Generate topics browser page."""
        html_template = """<!DOCTYPE html>
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

{NAV}

        <main>
{ABOUT}

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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="assets/js/topics.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("topics"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "topics.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: topics.html")

    def _generate_timeline_html(self):
        """Generate interactive timeline page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Timeline - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .timeline-wrapper {
            margin: 30px 0;
        }

        .timeline-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: space-between;
        }

        .timeline-filters {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 8px 16px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }

        .filter-btn:hover {
            border-color: var(--accent-color);
            background: var(--accent-color);
            color: white;
        }

        .filter-btn.active {
            background: var(--accent-color);
            border-color: var(--accent-color);
            color: white;
        }

        .zoom-controls {
            display: flex;
            gap: 10px;
        }

        .zoom-btn {
            padding: 8px 12px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            cursor: pointer;
            transition: all 0.3s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .zoom-btn.active {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .timeline-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-card .label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }

        .timeline-container {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 40px 20px;
            position: relative;
            overflow-x: auto;
            overflow-y: visible;
            height: calc(100vh - 400px);
            min-height: 500px;
        }

        .timeline-scroll {
            position: relative;
            min-width: 100%;
            width: max-content;
            padding: 60px 20px;
        }

        .timeline-axis {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-color), var(--secondary-color));
            border-radius: 2px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .timeline-year-markers {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            transform: translateY(-50%);
        }

        .year-marker {
            position: absolute;
            width: 2px;
            height: 20px;
            background: var(--border-color);
            transform: translateX(-50%);
        }

        .year-label {
            position: absolute;
            transform: translate(-50%, 30px);
            font-weight: 700;
            color: var(--secondary-color);
            font-size: 1.1em;
            white-space: nowrap;
        }

        .timeline-events {
            position: relative;
            min-height: 400px;
        }

        .timeline-event {
            position: absolute;
            width: 280px;
            cursor: pointer;
            animation: fadeInUp 0.5s ease-out;
            transition: all 0.3s;
        }

        .timeline-event.top {
            bottom: calc(50% + 30px);
        }

        .timeline-event.bottom {
            top: calc(50% + 30px);
        }

        .event-card {
            background: var(--bg-color);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
            position: relative;
        }

        .timeline-event:hover .event-card {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            border-color: var(--accent-color);
        }

        .event-dot {
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: white;
            border: 4px solid;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }

        .timeline-event.top .event-dot {
            top: 100%;
            margin-top: 14px;
        }

        .timeline-event.bottom .event-dot {
            bottom: 100%;
            margin-bottom: 14px;
        }

        .timeline-event:hover .event-dot {
            transform: translateX(-50%) scale(1.3);
        }

        .event-connector {
            position: absolute;
            width: 2px;
            background: var(--border-color);
            left: 50%;
            transform: translateX(-50%);
        }

        .timeline-event.top .event-connector {
            top: 100%;
            height: 30px;
        }

        .timeline-event.bottom .event-connector {
            bottom: 100%;
            height: 30px;
        }

        .event-type-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .event-type-hardware { background: #e74c3c; color: white; }
        .event-type-software { background: #3498db; color: white; }
        .event-type-music { background: #1abc9c; color: white; }
        .event-type-game { background: #16a085; color: white; }
        .event-type-company { background: #f39c12; color: white; }
        .event-type-release { background: #9b59b6; color: white; }
        .event-type-demo { background: #e67e22; color: white; }
        .event-type-magazine { background: #95a5a6; color: white; }

        .event-dot.hardware { border-color: #e74c3c; }
        .event-dot.software { border-color: #3498db; }
        .event-dot.music { border-color: #1abc9c; }
        .event-dot.game { border-color: #16a085; }
        .event-dot.company { border-color: #f39c12; }
        .event-dot.release { border-color: #9b59b6; }
        .event-dot.demo { border-color: #e67e22; }
        .event-dot.magazine { border-color: #95a5a6; }

        .event-title {
            font-weight: 700;
            font-size: 1.1em;
            color: var(--text-color);
            margin-bottom: 6px;
            line-height: 1.3;
        }

        .event-date {
            color: var(--accent-color);
            font-weight: 600;
            font-size: 0.9em;
            margin-bottom: 8px;
        }

        .event-description {
            font-size: 0.9em;
            color: var(--text-muted);
            line-height: 1.4;
            margin-bottom: 8px;
        }

        .event-meta {
            font-size: 0.8em;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 8px;
            margin-top: 8px;
        }

        .event-icon {
            font-size: 1.5em;
            margin-right: 8px;
        }

        .empty-timeline {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .loading-timeline {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 768px) {
            .timeline-controls {
                flex-direction: column;
                align-items: stretch;
            }

            .timeline-filters,
            .zoom-controls {
                justify-content: center;
            }

            .timeline-event {
                width: 240px;
            }

            .timeline-stats {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* Modal for event details */
        .event-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            padding: 20px;
            overflow-y: auto;
        }

        .event-modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 30px;
            max-width: 600px;
            width: 100%;
            position: relative;
            animation: modalSlideIn 0.3s ease-out;
        }

        @keyframes modalSlideIn {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .modal-close {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 32px;
            height: 32px;
            border: 2px solid var(--border-color);
            border-radius: 50%;
            background: var(--bg-color);
            color: var(--text-color);
            font-size: 1.2em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }

        .modal-close:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .modal-event-title {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--secondary-color);
            margin-bottom: 15px;
        }

        .modal-event-date {
            font-size: 1.2em;
            color: var(--accent-color);
            font-weight: 600;
            margin-bottom: 15px;
        }

        .modal-event-description {
            font-size: 1.05em;
            line-height: 1.6;
            color: var(--text-color);
            margin-bottom: 20px;
        }

        .modal-event-meta {
            background: var(--bg-color);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .modal-event-meta p {
            margin: 5px 0;
            font-size: 0.95em;
        }

        .modal-related-docs {
            margin-top: 20px;
        }

        .modal-related-docs h3 {
            color: var(--secondary-color);
            margin-bottom: 10px;
        }

        .related-doc-link {
            display: block;
            padding: 10px;
            margin: 5px 0;
            background: var(--bg-color);
            border-radius: 6px;
            border-left: 3px solid var(--accent-color);
            color: var(--text-color);
            text-decoration: none;
            transition: all 0.2s;
        }

        .related-doc-link:hover {
            background: var(--border-color);
            transform: translateX(4px);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📅 Interactive Timeline</h1>
            <p class="subtitle">Explore Commodore 64 history chronologically</p>
        </header>

{NAV}

{ABOUT}

        <div class="timeline-stats" id="timeline-stats">
            <div class="stat-card">
                <div class="value" id="total-events">-</div>
                <div class="label">Total Events</div>
            </div>
            <div class="stat-card">
                <div class="value" id="date-range">-</div>
                <div class="label">Year Range</div>
            </div>
            <div class="stat-card">
                <div class="value" id="event-types">-</div>
                <div class="label">Event Types</div>
            </div>
        </div>

        <div class="timeline-wrapper">
            <div class="timeline-controls">
                <div class="timeline-filters" id="type-filters">
                    <button class="filter-btn active" data-type="all">All Events</button>
                </div>
                <div class="zoom-controls">
                    <button class="zoom-btn" data-zoom="compact">Compact</button>
                    <button class="zoom-btn active" data-zoom="normal">Normal</button>
                    <button class="zoom-btn" data-zoom="expanded">Expanded</button>
                </div>
            </div>

            <div class="timeline-container">
                <div class="loading-timeline">Loading timeline events...</div>
                <div class="timeline-scroll" id="timeline-scroll" style="display: none;">
                    <div class="timeline-axis"></div>
                    <div class="timeline-year-markers" id="year-markers"></div>
                    <div class="timeline-events" id="timeline-events"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Event Detail Modal -->
    <div class="event-modal" id="event-modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeEventModal()">✕</button>
            <div id="modal-event-content"></div>
        </div>
    </div>

    <script src="assets/js/enhancements.js"></script>
    <script>
        let timelineEvents = [];
        let currentZoom = 'normal';
        let activeFilters = new Set(['all']);

        // Event type icons
        const typeIcons = {
            'hardware': '🖥️',
            'software': '💾',
            'music': '🎵',
            'game': '🎮',
            'company': '🏢',
            'release': '🚀',
            'demo': '✨',
            'magazine': '📰',
            'default': '📅'
        };

        async function loadTimeline() {
            try {
                const response = await fetch('assets/data/events.json');
                timelineEvents = await response.json();

                if (timelineEvents.length === 0) {
                    document.querySelector('.loading-timeline').textContent = 'No events found';
                    return;
                }

                // Sort by year
                timelineEvents.sort((a, b) => a.year - b.year);

                // Update stats
                updateStats();

                // Build type filters
                buildTypeFilters();

                // Render timeline
                renderTimeline();

                // Hide loading, show timeline
                document.querySelector('.loading-timeline').style.display = 'none';
                document.getElementById('timeline-scroll').style.display = 'block';
            } catch (error) {
                console.error('Error loading timeline:', error);
                document.querySelector('.loading-timeline').textContent = 'Error loading timeline';
            }
        }

        function updateStats() {
            const years = timelineEvents.map(e => e.year);
            const minYear = Math.min(...years);
            const maxYear = Math.max(...years);
            const types = [...new Set(timelineEvents.map(e => e.type))];

            document.getElementById('total-events').textContent = timelineEvents.length;
            document.getElementById('date-range').textContent = `${minYear}-${maxYear}`;
            document.getElementById('event-types').textContent = types.length;
        }

        function buildTypeFilters() {
            const types = [...new Set(timelineEvents.map(e => e.type))].sort();
            const container = document.getElementById('type-filters');

            types.forEach(type => {
                const btn = document.createElement('button');
                btn.className = 'filter-btn';
                btn.dataset.type = type;
                btn.innerHTML = `${typeIcons[type] || typeIcons.default} ${capitalize(type)}`;
                btn.onclick = () => toggleFilter(type, btn);
                container.appendChild(btn);
            });

            // Setup zoom controls
            document.querySelectorAll('.zoom-btn').forEach(btn => {
                btn.onclick = () => setZoom(btn.dataset.zoom);
            });
        }

        function toggleFilter(type, btn) {
            if (type === 'all') {
                activeFilters.clear();
                activeFilters.add('all');
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            } else {
                activeFilters.delete('all');
                document.querySelector('.filter-btn[data-type="all"]').classList.remove('active');

                if (activeFilters.has(type)) {
                    activeFilters.delete(type);
                    btn.classList.remove('active');
                } else {
                    activeFilters.add(type);
                    btn.classList.add('active');
                }

                if (activeFilters.size === 0) {
                    activeFilters.add('all');
                    document.querySelector('.filter-btn[data-type="all"]').classList.add('active');
                }
            }

            renderTimeline();
        }

        function setZoom(zoom) {
            currentZoom = zoom;
            document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
            document.querySelector(`.zoom-btn[data-zoom="${zoom}"]`).classList.add('active');
            renderTimeline();
        }

        function renderTimeline() {
            const filtered = activeFilters.has('all')
                ? timelineEvents
                : timelineEvents.filter(e => activeFilters.has(e.type));

            if (filtered.length === 0) {
                document.getElementById('timeline-events').innerHTML = '<div class="empty-timeline">No events match the selected filters</div>';
                return;
            }

            const years = filtered.map(e => e.year);
            const minYear = Math.min(...years);
            const maxYear = Math.max(...years);
            const yearRange = maxYear - minYear || 1;

            // Calculate spacing based on zoom
            const spacing = {
                'compact': 150,
                'normal': 250,
                'expanded': 400
            }[currentZoom];

            const totalWidth = yearRange * spacing + 200;

            // Render year markers
            const markersHtml = [];
            for (let year = minYear; year <= maxYear; year++) {
                const left = ((year - minYear) / yearRange) * (totalWidth - 200) + 100;
                markersHtml.push(`
                    <div class="year-marker" style="left: ${left}px;"></div>
                    <div class="year-label" style="left: ${left}px;">${year}</div>
                `);
            }
            document.getElementById('year-markers').innerHTML = markersHtml.join('');

            // Render events
            const eventsHtml = filtered.map((event, index) => {
                const left = ((event.year - minYear) / yearRange) * (totalWidth - 200) + 100;
                const position = index % 2 === 0 ? 'top' : 'bottom';
                const icon = typeIcons[event.type] || typeIcons.default;

                return `
                    <div class="timeline-event ${position}" style="left: ${left}px;" onclick="showEventModal(${JSON.stringify(event).replace(/"/g, '&quot;')})">
                        <div class="event-card">
                            <div class="event-type-badge event-type-${event.type}">${icon} ${capitalize(event.type)}</div>
                            <div class="event-title">${escapeHtml(event.title)}</div>
                            <div class="event-date">${formatDate(event.date_normalized, event.year)}</div>
                            <div class="event-description">${escapeHtml(truncate(event.description, 100))}</div>
                        </div>
                        <div class="event-connector"></div>
                        <div class="event-dot ${event.type}"></div>
                    </div>
                `;
            }).join('');

            document.getElementById('timeline-events').innerHTML = eventsHtml;

            // Set scroll width
            document.getElementById('timeline-scroll').style.width = totalWidth + 'px';
        }

        function showEventModal(event) {
            const modal = document.getElementById('event-modal');
            const content = document.getElementById('modal-event-content');

            const icon = typeIcons[event.type] || typeIcons.default;

            content.innerHTML = `
                <div class="event-type-badge event-type-${event.type}">${icon} ${capitalize(event.type)}</div>
                <h2 class="modal-event-title">${escapeHtml(event.title)}</h2>
                <div class="modal-event-date">${formatDate(event.date_normalized, event.year)}</div>
                <div class="modal-event-description">${escapeHtml(event.description)}</div>
                ${event.confidence ? `
                    <div class="modal-event-meta">
                        <p><strong>Event ID:</strong> ${event.event_id}</p>
                        <p><strong>Confidence:</strong> ${(event.confidence * 100).toFixed(0)}%</p>
                    </div>
                ` : ''}
            `;

            modal.classList.add('active');
        }

        function closeEventModal() {
            document.getElementById('event-modal').classList.remove('active');
        }

        // Close modal on background click
        document.getElementById('event-modal').addEventListener('click', (e) => {
            if (e.target.id === 'event-modal') {
                closeEventModal();
            }
        });

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeEventModal();
            }
        });

        function formatDate(dateStr, year) {
            if (dateStr && dateStr !== 'None') {
                const date = new Date(dateStr);
                return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
            }
            return year;
        }

        function capitalize(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        }

        function truncate(str, length) {
            if (!str) return '';
            return str.length > length ? str.substring(0, length) + '...' : str;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load timeline on page load
        loadTimeline();
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("timeline"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "timeline.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: timeline.html")

    def _generate_documents_browser_html(self):
        """Generate documents browser page."""
        html_template = """<!DOCTYPE html>
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

{NAV}

        <main>
{ABOUT}

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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="assets/js/documents.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("documents"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "documents.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: documents.html")

    def _generate_chunks_browser_html(self):
        """Generate chunks browser page."""
        html_template = """<!DOCTYPE html>
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

{NAV}

        <main>
{ABOUT}

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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="assets/js/chunks.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("chunks"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "chunks.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: chunks.html")

    def _generate_file_viewer_html(self):
        """Generate universal file viewer page using standard HTML5 components."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Viewer - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .viewer-container {
            width: 100%;
            height: calc(100vh - 250px);
            min-height: 600px;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            background: white;
            overflow: auto;
            margin: 20px 0;
        }
        .viewer-container iframe,
        .viewer-container embed,
        .viewer-container object {
            width: 100%;
            height: 100%;
            border: none;
        }
        .text-viewer {
            padding: 30px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #2d3748;
            background: white;
        }
        .markdown-viewer {
            padding: 30px;
            max-width: 900px;
            margin: 0 auto;
            background: white;
            color: #2d3748;
            line-height: 1.6;
        }
        .html-viewer {
            width: 100%;
            height: 100%;
        }
        .viewer-controls {
            display: flex;
            gap: 15px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 12px;
            margin-bottom: 20px;
            align-items: center;
            justify-content: space-between;
        }
        .viewer-controls .file-info {
            flex: 1;
        }
        .viewer-controls .file-name {
            font-weight: 600;
            color: var(--text-color);
            font-size: 1.1em;
        }
        .viewer-controls .file-type {
            color: var(--text-muted);
            font-size: 0.9em;
            margin-top: 4px;
        }
        .download-btn {
            padding: 10px 20px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: background 0.3s;
        }
        .download-btn:hover {
            background: var(--secondary-color);
        }
        .error-message {
            padding: 40px;
            text-align: center;
            color: #e53e3e;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle"><a href="documents.html">← Back to Documents</a></p>
        </header>

{NAV}

{ABOUT}

        <main>
            <div class="viewer-controls">
                <div class="file-info">
                    <div class="file-name" id="file-name">Loading...</div>
                    <div class="file-type" id="file-type"></div>
                </div>
                <a id="download-link" href="#" download class="download-btn">Download File</a>
            </div>

            <div class="viewer-container" id="viewer-container">
                <div style="text-align: center; padding: 40px; color: var(--text-muted);">
                    Loading file...
                </div>
            </div>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.0/marked.min.js"></script>
    <script>
        // Get file path from URL parameter
        const urlParams = new URLSearchParams(window.location.search);
        const filePath = urlParams.get('file');
        const fileName = urlParams.get('name') || 'Document';
        const fileType = urlParams.get('type') || 'unknown';

        // Update UI
        document.getElementById('file-name').textContent = fileName;
        document.getElementById('file-type').textContent = `Type: ${fileType.toUpperCase()}`;

        if (filePath) {
            document.getElementById('download-link').href = filePath;
            document.getElementById('download-link').download = fileName;

            const container = document.getElementById('viewer-container');

            // Display based on file type
            if (fileType === 'pdf') {
                // Use browser's native PDF viewer via iframe or embed
                container.innerHTML = `
                    <iframe src="${filePath}" type="application/pdf"></iframe>
                `;
            } else if (fileType === 'html') {
                // Display HTML in iframe
                container.innerHTML = `
                    <iframe src="${filePath}" class="html-viewer"></iframe>
                `;
            } else if (fileType === 'markdown' || fileType === 'md') {
                // Fetch and render markdown
                fetch(filePath)
                    .then(response => response.text())
                    .then(text => {
                        container.innerHTML = `<div class="markdown-viewer">${marked.parse(text)}</div>`;
                    })
                    .catch(error => {
                        container.innerHTML = `<div class="error-message">Error loading markdown file: ${error.message}</div>`;
                    });
            } else {
                // Display as text
                fetch(filePath)
                    .then(response => response.text())
                    .then(text => {
                        const escaped = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        container.innerHTML = `<div class="text-viewer">${escaped}</div>`;
                    })
                    .catch(error => {
                        container.innerHTML = `<div class="error-message">Error loading file: ${error.message}</div>`;
                    });
            }
        } else {
            document.getElementById('viewer-container').innerHTML =
                '<div class="error-message">No file specified</div>';
        }
    </script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav())
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box())

        filepath = self.output_dir / "viewer.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: viewer.html")

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

/* ===== SYNTAX HIGHLIGHTING ===== */
.asm-opcode {
    color: #ff79c6;
    font-weight: bold;
}

.asm-hex {
    color: #50fa7b;
}

.asm-binary {
    color: #8be9fd;
}

.asm-number {
    color: #bd93f9;
}

.asm-comment {
    color: #6272a4;
    font-style: italic;
}

.asm-label {
    color: #f1fa8c;
    font-weight: bold;
}

.asm-directive {
    color: #ffb86c;
}

[data-theme="dark"] .asm-opcode { color: #ff79c6; }
[data-theme="dark"] .asm-hex { color: #50fa7b; }
[data-theme="dark"] .asm-comment { color: #6272a4; }
[data-theme="dark"] .asm-label { color: #f1fa8c; }

[data-theme="c64"] .asm-opcode { color: #ffeb3b; }
[data-theme="c64"] .asm-hex { color: #00e676; }
[data-theme="c64"] .asm-comment { color: #9fa8da; }
[data-theme="c64"] .asm-label { color: #ffc107; }

/* ===== BOOKMARKS SYSTEM ===== */
.bookmark-btn {
    display: inline-block;
    margin: 20px 0;
    padding: 10px 20px;
    background: var(--bg-color);
    color: var(--text-color);
    border: 2px solid var(--border-color);
    border-radius: 25px;
    cursor: pointer;
    font-size: 1em;
    transition: all 0.3s;
}

.bookmark-btn:hover {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
    transform: scale(1.05);
}

.bookmark-btn.bookmarked {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

.bookmark-notification {
    position: fixed;
    bottom: 100px;
    right: 30px;
    background: var(--accent-color);
    color: white;
    padding: 15px 25px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10000;
    opacity: 0;
    transform: translateY(20px);
    transition: all 0.3s;
}

.bookmark-notification.show {
    opacity: 1;
    transform: translateY(0);
}

.bookmarks-panel {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.7);
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.3s;
}

.bookmarks-content {
    background: var(--card-bg);
    border-radius: 12px;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.bookmarks-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 2px solid var(--border-color);
}

.bookmarks-header h2 {
    margin: 0;
    color: var(--text-color);
}

.close-btn {
    background: none;
    border: none;
    font-size: 2em;
    color: var(--primary-color);
    cursor: pointer;
    padding: 0;
    width: 40px;
    height: 40px;
    line-height: 1;
}

.close-btn:hover {
    color: var(--accent-color);
}

.bookmarks-list {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.bookmark-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px;
    margin: 10px 0;
    background: var(--bg-color);
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
    transition: all 0.3s;
}

.bookmark-item:hover {
    transform: translateX(5px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.bookmark-item a {
    flex: 1;
    color: var(--text-color);
    text-decoration: none;
    font-weight: 500;
}

.bookmark-item a:hover {
    color: var(--accent-color);
}

.remove-bookmark {
    background: none;
    border: none;
    font-size: 1.2em;
    cursor: pointer;
    padding: 5px 10px;
    opacity: 0.6;
    transition: all 0.3s;
}

.remove-bookmark:hover {
    opacity: 1;
    transform: scale(1.2);
}

.bookmarks-footer {
    padding: 20px;
    border-top: 2px solid var(--border-color);
    text-align: center;
}

.bookmarks-footer button {
    background: var(--accent-color);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1em;
    transition: all 0.3s;
}

.bookmarks-footer button:hover {
    background: var(--secondary-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* ===== AI CHATBOT ===== */
.chat-widget {
    position: fixed;
    bottom: 30px;
    right: 100px;
    z-index: 9999;
}

.chat-toggle {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
    width: 85px;
    height: 85px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: 3px solid rgba(255,255,255,0.3);
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    transition: all 0.3s;
    animation: pulse 2s ease-in-out infinite;
}

.bot-icon {
    font-size: 2.5em;
    line-height: 1;
}

.bot-label {
    font-size: 0.75em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.chat-toggle:hover {
    transform: scale(1.1) rotate(5deg);
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    animation: none;
}

@keyframes pulse {
    0%, 100% {
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    50% {
        box-shadow: 0 6px 20px rgba(0,0,0,0.4), 0 0 20px rgba(88,80,236,0.6);
    }
}

.chat-container {
    position: absolute;
    bottom: 80px;
    right: 0;
    width: 400px;
    max-height: 600px;
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    display: none;
    flex-direction: column;
    overflow: hidden;
}

.chat-container.open {
    display: flex;
    animation: slideUp 0.3s;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.chat-header {
    padding: 15px 20px;
    background: var(--accent-color);
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-header h3 {
    margin: 0;
    font-size: 1.1em;
}

.chat-close {
    background: none;
    border: none;
    color: white;
    font-size: 2em;
    cursor: pointer;
    padding: 0;
    width: 30px;
    height: 30px;
    line-height: 1;
}

.chat-close:hover {
    opacity: 0.8;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    background: var(--bg-color);
    max-height: 400px;
}

.chat-message {
    margin: 15px 0;
    display: flex;
    flex-direction: column;
}

.chat-message.user {
    align-items: flex-end;
}

.chat-message.bot {
    align-items: flex-start;
}

.message-content {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
}

.chat-message.user .message-content {
    background: var(--accent-color);
    color: white;
    border-bottom-right-radius: 4px;
}

.chat-message.bot .message-content {
    background: var(--card-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-bottom-left-radius: 4px;
}

.message-content ul {
    margin: 10px 0;
    padding-left: 20px;
}

.message-content code {
    background: var(--code-bg);
    color: var(--code-color);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
}

.message-content pre {
    background: var(--code-bg);
    color: var(--code-color);
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 10px 0;
}

.message-sources {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border-color);
    font-size: 0.9em;
}

.message-sources ul {
    margin: 5px 0;
    padding-left: 20px;
}

.message-sources a {
    color: var(--accent-color);
    text-decoration: none;
}

.message-sources a:hover {
    text-decoration: underline;
}

.typing-dots {
    display: flex;
    gap: 4px;
    padding: 8px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    background: var(--primary-color);
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

.typing-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: translateY(0);
    }
    30% {
        opacity: 1;
        transform: translateY(-10px);
    }
}

.chat-input-container {
    display: flex;
    padding: 15px;
    background: var(--card-bg);
    border-top: 1px solid var(--border-color);
}

#chat-input {
    flex: 1;
    padding: 10px 15px;
    border: 1px solid var(--border-color);
    border-radius: 20px;
    background: var(--bg-color);
    color: var(--text-color);
    font-size: 0.95em;
}

#chat-input:focus {
    outline: none;
    border-color: var(--accent-color);
}

#chat-send {
    margin-left: 10px;
    padding: 10px 20px;
    background: var(--accent-color);
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 0.95em;
    transition: all 0.3s;
}

#chat-send:hover {
    background: var(--secondary-color);
    transform: scale(1.05);
}

.chat-examples {
    padding: 10px 15px;
    background: var(--bg-color);
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.chat-examples small {
    color: var(--primary-color);
    margin-right: 5px;
}

.example-btn {
    padding: 6px 12px;
    background: var(--card-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 15px;
    cursor: pointer;
    font-size: 0.85em;
    transition: all 0.3s;
}

.example-btn:hover {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

/* Mobile responsive */
@media (max-width: 768px) {
    .chat-container {
        width: 90vw;
        max-width: 90vw;
        right: 5vw;
    }

    .chat-widget {
        right: 20px;
    }
}

/* ===== EXPLANATION BOXES ===== */
.explanation-box {
    background: linear-gradient(135deg, rgba(88,80,236,0.1) 0%, rgba(146,64,230,0.1) 100%);
    border: 2px solid var(--border-color);
    border-left: 5px solid var(--accent-color);
    border-radius: 12px;
    padding: 20px 25px;
    margin: 20px 0 30px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.explanation-box h3 {
    color: var(--accent-color);
    margin: 0 0 12px 0;
    font-size: 1.1em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.explanation-box p {
    margin: 0 0 10px 0;
    line-height: 1.6;
    color: var(--text-color);
}

.explanation-box p:last-child {
    margin-bottom: 0;
}

.explanation-box ul {
    margin: 10px 0;
    padding-left: 25px;
}

.explanation-box li {
    margin: 6px 0;
    line-height: 1.5;
}

.explanation-box code {
    background: var(--bg-color);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
}

/* ===== AUTO TABLE OF CONTENTS ===== */
.auto-toc {
    background: var(--bg-color);
    border: 2px solid var(--border-color);
    border-left: 4px solid var(--accent-color);
    border-radius: 8px;
    padding: 20px;
    margin: 30px 0;
    max-width: 100%;
}

.toc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.toc-header h3 {
    margin: 0;
    color: var(--text-color);
    font-size: 1.2em;
}

.toc-toggle {
    background: none;
    border: none;
    font-size: 1.5em;
    color: var(--primary-color);
    cursor: pointer;
    padding: 5px 10px;
    transition: all 0.3s;
}

.toc-toggle:hover {
    color: var(--accent-color);
}

.auto-toc.collapsed .toc-nav {
    display: none;
}

.auto-toc.collapsed .toc-toggle::before {
    content: '+';
}

.toc-nav ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.toc-nav li {
    padding: 8px 0;
    border-left: 3px solid transparent;
    padding-left: 15px;
    transition: all 0.3s;
}

.toc-nav li.toc-h3 {
    padding-left: 30px;
    font-size: 0.95em;
}

.toc-nav li.active {
    border-left-color: var(--accent-color);
    background: var(--bg-color);
    margin-left: -5px;
    padding-left: 20px;
}

.toc-nav a {
    color: var(--text-color);
    text-decoration: none;
    transition: all 0.3s;
}

.toc-nav a:hover {
    color: var(--accent-color);
}

.toc-highlight {
    animation: highlightSection 2s;
}

@keyframes highlightSection {
    0%, 100% { background: transparent; }
    50% { background: var(--accent-color); opacity: 0.1; }
}

/* ===== SEARCH AUTOCOMPLETE ===== */
.search-autocomplete {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-top: none;
    border-radius: 0 0 8px 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    max-height: 400px;
    overflow-y: auto;
    z-index: 1000;
    display: none;
}

.autocomplete-item {
    padding: 12px 15px;
    cursor: pointer;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-color);
    transition: all 0.2s;
}

.autocomplete-item:last-child {
    border-bottom: none;
}

.autocomplete-item:hover,
.autocomplete-item.active {
    background: var(--accent-color);
    color: white;
}

.autocomplete-item strong {
    font-weight: bold;
    color: inherit;
}

/* Make search input container relative for autocomplete positioning */
.search-section {
    position: relative;
}

/* ===== POPULAR ARTICLES SECTION ===== */
.popular-articles-section {
    margin: 40px 0;
}

.popular-articles-section h2 {
    color: var(--text-color);
    font-size: 1.8em;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.popular-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.popular-card {
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    transition: all 0.3s;
    overflow: hidden;
}

.popular-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    border-color: var(--accent-color);
}

.popular-rank {
    position: absolute;
    top: -10px;
    right: -10px;
    background: var(--accent-color);
    color: white;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.2em;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.popular-card h3 {
    margin: 0 0 15px 0;
    font-size: 1.3em;
}

.popular-card h3 a {
    color: var(--text-color);
    text-decoration: none;
    transition: color 0.3s;
}

.popular-card h3 a:hover {
    color: var(--accent-color);
}

.popular-meta {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    font-size: 0.9em;
    color: var(--primary-color);
    margin-top: 10px;
}

.popular-meta span {
    display: flex;
    align-items: center;
    gap: 5px;
}

.doc-count {
    font-weight: 600;
}

.related-count {
    opacity: 0.8;
}

/* Popular articles responsiveness */
@media (max-width: 768px) {
    .popular-grid {
        grid-template-columns: 1fr;
    }

    .popular-rank {
        width: 40px;
        height: 40px;
        font-size: 1em;
    }
}

/* ===============================================
   BREADCRUMBS NAVIGATION
   =============================================== */

.breadcrumbs {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 0;
    margin: 20px 0;
    font-size: 0.9em;
    color: var(--primary-color);
    flex-wrap: wrap;
}

.breadcrumbs a {
    color: var(--accent-color);
    text-decoration: none;
    transition: color 0.3s;
}

.breadcrumbs a:hover {
    color: var(--secondary-color);
    text-decoration: underline;
}

.breadcrumbs .separator {
    color: var(--border-color);
    user-select: none;
}

.breadcrumbs .current {
    color: var(--text-color);
    font-weight: 500;
}

/* ===============================================
   PRINT-FRIENDLY STYLES
   =============================================== */

@media print {
    /* Reset colors for print */
    * {
        background: white !important;
        color: black !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }

    /* Hide interactive elements */
    .main-nav,
    .search-section,
    .theme-switcher,
    .dark-mode-toggle,
    .copy-button,
    .back-to-top,
    .bookmark-btn,
    .chatbot-widget,
    .chatbot-button,
    footer,
    button,
    .search-autocomplete,
    .popular-articles-section {
        display: none !important;
    }

    /* Optimize layout */
    body {
        font-size: 12pt;
        line-height: 1.5;
        margin: 0;
        padding: 0;
    }

    .container {
        max-width: 100%;
        margin: 0;
        padding: 0;
    }

    /* Header styling */
    header {
        border-bottom: 2px solid #000;
        padding: 10px 0;
        margin-bottom: 20px;
    }

    header h1 {
        font-size: 18pt;
        margin: 0;
    }

    .subtitle {
        font-size: 10pt;
    }

    /* Breadcrumbs for print */
    .breadcrumbs {
        font-size: 9pt;
        margin: 10px 0;
        padding: 5px 0;
        border-bottom: 1px solid #ccc;
    }

    /* Article content */
    article, main {
        page-break-inside: avoid;
    }

    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        page-break-inside: avoid;
    }

    h1 {
        font-size: 16pt;
    }

    h2 {
        font-size: 14pt;
        margin-top: 12pt;
    }

    h3 {
        font-size: 12pt;
        margin-top: 10pt;
    }

    /* Code blocks */
    pre, code {
        border: 1px solid #ccc;
        background: #f5f5f5 !important;
        page-break-inside: avoid;
        font-family: 'Courier New', monospace;
        font-size: 9pt;
    }

    pre {
        padding: 10px;
        margin: 10px 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    /* Links */
    a {
        text-decoration: underline;
        color: #000 !important;
    }

    a[href]:after {
        content: " (" attr(href) ")";
        font-size: 8pt;
        color: #666 !important;
    }

    /* Don't show URLs for internal links */
    a[href^="#"]:after,
    a[href^="javascript:"]:after {
        content: "";
    }

    /* Tables */
    table {
        border-collapse: collapse;
        width: 100%;
        page-break-inside: avoid;
    }

    th, td {
        border: 1px solid #000;
        padding: 4px 8px;
        text-align: left;
    }

    th {
        background: #e0e0e0 !important;
        font-weight: bold;
    }

    /* Images */
    img {
        max-width: 100%;
        page-break-inside: avoid;
    }

    /* Chunks */
    .chunk {
        page-break-inside: avoid;
        border: 1px solid #ccc;
        padding: 10px;
        margin: 10px 0;
    }

    .chunk-header {
        font-weight: bold;
        border-bottom: 1px solid #000;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }

    /* Cards */
    .doc-card, .entity-card, .topic-card {
        border: 1px solid #ccc;
        padding: 10px;
        margin: 10px 0;
        page-break-inside: avoid;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        page-break-inside: avoid;
    }

    .stat-card {
        border: 1px solid #000;
        padding: 10px;
        text-align: center;
    }

    /* Page breaks */
    .page-break {
        page-break-before: always;
    }

    /* Print URL and date */
    @page {
        margin: 2cm;
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
        }
    }

    /* TOC for print */
    .auto-toc {
        border: 2px solid #000;
        padding: 10px;
        margin: 20px 0;
        page-break-inside: avoid;
    }

    .toc-toggle {
        display: none !important;
    }
}

/* ===============================================
   RELATED DOCUMENTS SIDEBAR
   =============================================== */

.related-docs-sidebar {
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin: 30px 0;
    position: sticky;
    top: 20px;
    max-height: calc(100vh - 40px);
    overflow-y: auto;
}

.related-docs-sidebar h3 {
    margin: 0 0 15px 0;
    color: var(--secondary-color);
    font-size: 1.2em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.related-docs-sidebar h3::before {
    content: "🔗";
    font-size: 1.2em;
}

.related-doc-item {
    padding: 12px;
    margin: 8px 0;
    background: var(--bg-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    transition: all 0.3s;
    position: relative;
}

.related-doc-item:hover {
    border-color: var(--accent-color);
    transform: translateX(4px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.related-doc-item a {
    color: var(--text-color);
    text-decoration: none;
    font-weight: 500;
    display: block;
    margin-bottom: 6px;
}

.related-doc-item a:hover {
    color: var(--accent-color);
}

.related-doc-score {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85em;
    color: var(--primary-color);
    margin-top: 6px;
}

.similarity-bar {
    flex: 1;
    height: 4px;
    background: var(--border-color);
    border-radius: 2px;
    overflow: hidden;
}

.similarity-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-color), #68d391);
    border-radius: 2px;
    transition: width 0.5s ease-out;
}

.related-doc-meta {
    display: flex;
    gap: 10px;
    font-size: 0.75em;
    color: var(--primary-color);
    margin-top: 4px;
}

.related-doc-meta span {
    display: flex;
    align-items: center;
    gap: 3px;
}

.no-related-docs {
    text-align: center;
    padding: 20px;
    color: var(--primary-color);
    font-style: italic;
}

.loading-related {
    text-align: center;
    padding: 20px;
    color: var(--primary-color);
}

.loading-related::after {
    content: "";
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-top-color: var(--accent-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 10px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Two-column layout for docs with sidebar */
.doc-with-sidebar {
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 30px;
    align-items: start;
}

.doc-main-content {
    min-width: 0; /* Prevent grid blowout */
}

/* ===============================================
   TAG CLOUD
   =============================================== */

.tag-cloud-section {
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 12px;
    padding: 30px;
    margin: 40px 0;
}

.tag-cloud-section h2 {
    margin: 0 0 20px 0;
    color: var(--secondary-color);
    text-align: center;
    font-size: 1.8em;
}

.tag-cloud-controls {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.tag-cloud-control-btn {
    padding: 8px 16px;
    background: var(--bg-color);
    border: 2px solid var(--border-color);
    border-radius: 6px;
    color: var(--text-color);
    cursor: pointer;
    font-size: 0.9em;
    transition: all 0.3s;
}

.tag-cloud-control-btn:hover {
    border-color: var(--accent-color);
    background: var(--accent-color);
    color: white;
}

.tag-cloud-control-btn.active {
    background: var(--accent-color);
    border-color: var(--accent-color);
    color: white;
}

.tag-cloud {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 10px;
    padding: 20px;
    min-height: 200px;
}

.tag-cloud-item {
    display: inline-block;
    padding: 6px 14px;
    background: var(--bg-color);
    border: 2px solid var(--border-color);
    border-radius: 20px;
    color: var(--text-color);
    text-decoration: none;
    transition: all 0.3s;
    cursor: pointer;
    font-weight: 500;
    position: relative;
    overflow: hidden;
}

.tag-cloud-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent-color), transparent);
    opacity: 0.3;
    transition: left 0.5s;
}

.tag-cloud-item:hover::before {
    left: 100%;
}

.tag-cloud-item:hover {
    border-color: var(--accent-color);
    transform: scale(1.05) translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    background: var(--accent-color);
    color: white;
}

/* Size variations based on frequency */
.tag-cloud-item.size-xs {
    font-size: 0.75em;
    padding: 4px 10px;
}

.tag-cloud-item.size-sm {
    font-size: 0.85em;
    padding: 5px 12px;
}

.tag-cloud-item.size-md {
    font-size: 1em;
    padding: 6px 14px;
}

.tag-cloud-item.size-lg {
    font-size: 1.2em;
    padding: 8px 16px;
}

.tag-cloud-item.size-xl {
    font-size: 1.4em;
    padding: 10px 20px;
    font-weight: 600;
}

.tag-cloud-item.size-xxl {
    font-size: 1.8em;
    padding: 12px 24px;
    font-weight: 700;
}

/* Tag count badge */
.tag-count {
    display: inline-block;
    margin-left: 6px;
    padding: 2px 6px;
    background: var(--accent-color);
    color: white;
    border-radius: 10px;
    font-size: 0.75em;
    font-weight: 600;
    opacity: 0;
    transition: opacity 0.3s;
}

.tag-cloud-item:hover .tag-count {
    opacity: 1;
}

.tag-cloud-loading {
    text-align: center;
    padding: 40px;
    color: var(--primary-color);
    font-size: 1.1em;
}

.tag-cloud-empty {
    text-align: center;
    padding: 40px;
    color: var(--primary-color);
    font-style: italic;
}

.tag-cloud-stats {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 2px solid var(--border-color);
    font-size: 0.9em;
    color: var(--primary-color);
}

.tag-cloud-stats span {
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ===============================================
   MOBILE RESPONSIVENESS
   =============================================== */

/* Tablet and below (768px) */
@media (max-width: 768px) {
    /* Base layout */
    .container {
        padding: 10px;
    }

    /* Header */
    header h1 {
        font-size: 1.8em;
    }

    .subtitle {
        font-size: 1em;
    }

    /* Navigation */
    .main-nav {
        flex-direction: column;
        gap: 5px;
    }

    .main-nav a {
        width: 100%;
        text-align: center;
        padding: 12px;
    }

    /* Breadcrumbs */
    .breadcrumbs {
        font-size: 0.85em;
        padding: 8px 0;
    }

    /* Search */
    .search-section {
        margin: 15px 0;
    }

    #search-input {
        font-size: 16px; /* Prevent zoom on iOS */
        padding: 12px;
    }

    /* Stats grid */
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }

    .stat-card {
        padding: 15px 10px;
    }

    .stat-card h3 {
        font-size: 1.5em;
    }

    /* Document cards */
    .doc-grid {
        grid-template-columns: 1fr;
    }

    .doc-card {
        padding: 15px;
    }

    /* Category grid */
    .category-grid {
        grid-template-columns: 1fr;
    }

    /* Article content */
    .article-content h1 {
        font-size: 1.8em;
    }

    .article-content h2 {
        font-size: 1.4em;
    }

    .article-content h3 {
        font-size: 1.2em;
    }

    /* Code blocks */
    pre {
        font-size: 0.85em;
        padding: 10px;
        overflow-x: auto;
    }

    /* Tables */
    table {
        font-size: 0.9em;
    }

    th, td {
        padding: 8px 4px;
    }

    /* Chatbot */
    .chatbot-button {
        bottom: 15px;
        right: 15px;
        width: 50px;
        height: 50px;
        font-size: 1.3em;
    }

    .chatbot-widget {
        width: 100%;
        height: 100%;
        max-height: 100vh;
        border-radius: 0;
        bottom: 0;
        right: 0;
    }

    .chatbot-header {
        border-radius: 0;
    }

    /* Auto TOC */
    .auto-toc {
        padding: 15px;
        margin: 15px 0;
    }

    .toc-nav ul {
        font-size: 0.9em;
    }

    /* Entity cards */
    .entity-grid {
        grid-template-columns: 1fr;
    }

    /* Topic cards */
    .topic-grid {
        grid-template-columns: 1fr;
    }

    /* Timeline */
    .timeline-event {
        padding: 15px;
    }

    /* Bookmark button */
    .bookmark-btn {
        padding: 6px 10px;
        font-size: 0.9em;
    }

    /* Reading progress */
    .reading-progress-container {
        height: 4px;
    }

    /* Popular articles */
    .popular-grid {
        grid-template-columns: 1fr;
    }

    /* Search results */
    .search-results {
        max-height: 60vh;
    }

    .result-card {
        padding: 12px;
    }

    /* Related docs sidebar - stack on mobile */
    .doc-with-sidebar {
        grid-template-columns: 1fr;
    }

    .related-docs-sidebar {
        position: relative;
        top: 0;
        max-height: none;
        margin: 20px 0;
    }
}

/* Mobile phones (480px) */
@media (max-width: 480px) {
    /* Header */
    header {
        padding: 20px 0 10px;
    }

    header h1 {
        font-size: 1.5em;
    }

    .subtitle {
        font-size: 0.9em;
    }

    /* Stats grid - single column on small phones */
    .stats-grid {
        grid-template-columns: 1fr;
    }

    /* Navigation */
    .main-nav a {
        padding: 10px;
        font-size: 0.95em;
    }

    /* Breadcrumbs */
    .breadcrumbs {
        font-size: 0.8em;
        gap: 5px;
    }

    .breadcrumbs .separator {
        font-size: 0.9em;
    }

    /* Article meta */
    .article-meta,
    .doc-meta {
        flex-direction: column;
        gap: 5px;
        font-size: 0.85em;
    }

    /* Code blocks - smaller on phones */
    pre {
        font-size: 0.75em;
        padding: 8px;
    }

    /* Chunks */
    .chunk {
        padding: 10px;
    }

    .chunk-header {
        font-size: 0.9em;
    }

    /* Tags */
    .tag {
        font-size: 0.75em;
        padding: 3px 8px;
    }

    /* Search autocomplete */
    .autocomplete-item {
        padding: 10px;
        font-size: 0.9em;
    }

    /* Copy button */
    .copy-button {
        padding: 4px 8px;
        font-size: 0.8em;
        top: 5px;
        right: 5px;
    }

    /* Chatbot - full screen on phones */
    .chatbot-messages {
        max-height: calc(100vh - 180px);
    }

    /* Back to top button */
    .back-to-top {
        bottom: 15px;
        right: 15px;
        width: 45px;
        height: 45px;
        font-size: 1.2em;
    }

    /* Popular article rank badges */
    .popular-rank {
        width: 35px;
        height: 35px;
        font-size: 0.9em;
        top: -8px;
        right: -8px;
    }
}

/* Large screens (1200px+) */
@media (min-width: 1200px) {
    .container {
        max-width: 1400px;
    }

    .stats-grid {
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
    }

    .doc-grid,
    .category-grid,
    .entity-grid,
    .topic-grid {
        grid-template-columns: repeat(3, 1fr);
    }

    .popular-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

/* Touch device optimizations */
@media (hover: none) and (pointer: coarse) {
    /* Larger tap targets for touch */
    a, button, .clickable {
        min-height: 44px;
        min-width: 44px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .main-nav a {
        padding: 14px;
    }

    /* Remove hover effects on touch devices */
    .doc-card:hover,
    .entity-card:hover,
    .topic-card:hover,
    .category-card:hover {
        transform: none;
    }

    /* Highlight on tap instead */
    .doc-card:active,
    .entity-card:active,
    .topic-card:active,
    .category-card:active {
        transform: scale(0.98);
        opacity: 0.9;
    }

    /* Disable smooth scroll on touch for better performance */
    html {
        scroll-behavior: auto;
    }

    /* Larger scroll bars for touch */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
}

/* Landscape orientation on phones */
@media (max-width: 768px) and (orientation: landscape) {
    .chatbot-widget {
        max-height: 90vh;
    }

    .search-results {
        max-height: 50vh;
    }

    header h1 {
        font-size: 1.6em;
    }
}

/* High DPI screens */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    /* Sharper borders and text */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
}

/* Reduced motion for accessibility */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }

    html {
        scroll-behavior: auto !important;
    }
}

/* Dark mode preference detection */
@media (prefers-color-scheme: dark) {
    /* Auto dark mode if user hasn't set preference */
    body:not([data-theme]) {
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
}

/* ===============================================
   LAZY LOADING IMAGES
   =============================================== */

img[loading="lazy"] {
    opacity: 0;
    transition: opacity 0.3s;
}

img[loading="lazy"].loaded {
    opacity: 1;
}

/* Placeholder for lazy images */
.lazy-placeholder {
    background: linear-gradient(
        90deg,
        var(--border-color) 0%,
        var(--bg-color) 50%,
        var(--border-color) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    min-height: 200px;
}

@keyframes shimmer {
    0% {
        background-position: 200% 0;
    }
    100% {
        background-position: -200% 0;
    }
}

/* Ensure images don't break layout while loading */
img {
    max-width: 100%;
    height: auto;
    display: block;
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
        generateTagCloud();
        setupTagCloudControls();
    });
} else {
    loadNavigation();
    loadDocuments();
    generateTagCloud();
    setupTagCloudControls();
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

// ===== TAG CLOUD =====
async function generateTagCloud() {
    const tagCloudElement = document.getElementById('tag-cloud');
    if (!tagCloudElement) return;

    try {
        // Load navigation data (contains tags)
        const response = await fetch('assets/data/navigation.json');
        const nav = await response.json();

        // Get tags with document counts
        const tagData = Object.keys(nav.by_tags).map(tag => ({
            tag: tag,
            count: nav.by_tags[tag].length
        }));

        // Sort by count descending
        tagData.sort((a, b) => b.count - a.count);

        // Calculate size classes
        const maxCount = Math.max(...tagData.map(t => t.count));
        const minCount = Math.min(...tagData.map(t => t.count));

        // Function to determine size class
        function getSizeClass(count) {
            const ratio = (count - minCount) / (maxCount - minCount);
            if (ratio > 0.8) return 'size-xxl';
            if (ratio > 0.6) return 'size-xl';
            if (ratio > 0.4) return 'size-lg';
            if (ratio > 0.2) return 'size-md';
            if (ratio > 0.1) return 'size-sm';
            return 'size-xs';
        }

        // Generate HTML
        const html = tagData.map((item, index) => {
            const sizeClass = getSizeClass(item.count);
            return `
                <span class="tag-cloud-item ${sizeClass}"
                      data-tag="${escapeHtml(item.tag)}"
                      data-count="${item.count}"
                      style="animation: fadeIn 0.3s ease-out ${index * 0.02}s both">
                    ${escapeHtml(item.tag)}
                    <span class="tag-count">${item.count}</span>
                </span>
            `;
        }).join('');

        tagCloudElement.innerHTML = html;

        // Add click handlers
        document.querySelectorAll('.tag-cloud-item').forEach(item => {
            item.onclick = () => {
                const tag = item.getAttribute('data-tag');
                // Navigate to documents page with tag filter
                window.location.href = `documents.html#tag-${tag}`;
            };
        });

        // Add stats
        const statsContainer = document.getElementById('tag-cloud-stats');
        if (statsContainer) {
            const totalTags = tagData.length;
            const totalDocs = tagData.reduce((sum, t) => sum + t.count, 0);
            const avgDocs = Math.round(totalDocs / totalTags);

            statsContainer.innerHTML = `
                <span>📊 ${totalTags} tags</span>
                <span>📚 ${totalDocs} total document-tag mappings</span>
                <span>📈 ${avgDocs} avg docs per tag</span>
            `;
        }

    } catch (error) {
        console.error('Failed to generate tag cloud:', error);
        tagCloudElement.innerHTML = '<div class="tag-cloud-empty">Unable to load tag cloud</div>';
    }
}

function setupTagCloudControls() {
    const controls = document.querySelectorAll('.tag-cloud-control-btn');
    const tagCloud = document.getElementById('tag-cloud');
    if (!tagCloud) return;

    controls.forEach(btn => {
        btn.onclick = async () => {
            // Update active state
            controls.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const sortBy = btn.getAttribute('data-sort');

            // Re-fetch and re-sort
            try {
                const response = await fetch('assets/data/navigation.json');
                const nav = await response.json();

                let tagData = Object.keys(nav.by_tags).map(tag => ({
                    tag: tag,
                    count: nav.by_tags[tag].length
                }));

                // Sort based on selected option
                if (sortBy === 'alpha') {
                    tagData.sort((a, b) => a.tag.localeCompare(b.tag));
                } else if (sortBy === 'count') {
                    tagData.sort((a, b) => b.count - a.count);
                } else if (sortBy === 'random') {
                    tagData = tagData.sort(() => Math.random() - 0.5);
                }

                // Calculate size classes
                const maxCount = Math.max(...tagData.map(t => t.count));
                const minCount = Math.min(...tagData.map(t => t.count));

                function getSizeClass(count) {
                    const ratio = (count - minCount) / (maxCount - minCount);
                    if (ratio > 0.8) return 'size-xxl';
                    if (ratio > 0.6) return 'size-xl';
                    if (ratio > 0.4) return 'size-lg';
                    if (ratio > 0.2) return 'size-md';
                    if (ratio > 0.1) return 'size-sm';
                    return 'size-xs';
                }

                // Re-render
                const html = tagData.map((item, index) => {
                    const sizeClass = getSizeClass(item.count);
                    return `
                        <span class="tag-cloud-item ${sizeClass}"
                              data-tag="${escapeHtml(item.tag)}"
                              data-count="${item.count}"
                              style="animation: fadeIn 0.3s ease-out ${index * 0.02}s both">
                            ${escapeHtml(item.tag)}
                            <span class="tag-count">${item.count}</span>
                        </span>
                    `;
                }).join('');

                tagCloud.innerHTML = html;

                // Re-add click handlers
                document.querySelectorAll('.tag-cloud-item').forEach(item => {
                    item.onclick = () => {
                        const tag = item.getAttribute('data-tag');
                        window.location.href = `documents.html#tag-${tag}`;
                    };
                });

            } catch (error) {
                console.error('Failed to re-sort tag cloud:', error);
            }
        };
    });
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

            // Create document list
            let docsList = '';
            if (cluster.documents && cluster.documents.length > 0) {
                const displayDocs = cluster.documents.slice(0, 10); // Show first 10
                docsList = '<div style="margin-top: 12px;"><ul style="list-style: none; padding: 0; font-size: 0.9em;">';
                for (const doc of displayDocs) {
                    const safeFilename = doc.id.replace(/[^\\w\\-]/g, '_') + '.html';
                    docsList += `<li style="margin: 4px 0;"><a href="docs/${safeFilename}" style="color: var(--accent-color); text-decoration: none;">${escapeHtml(doc.title)}</a></li>`;
                }
                if (cluster.documents.length > 10) {
                    docsList += `<li style="margin: 8px 0; font-style: italic; color: var(--text-muted);">...and ${cluster.documents.length - 10} more</li>`;
                }
                docsList += '</ul></div>';
            }

            card.innerHTML = `
                <h3>Cluster ${cluster.number}</h3>
                <div class="doc-card-meta">${cluster.doc_count} documents</div>
                ${docsList}
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

// ===== SYNTAX HIGHLIGHTING =====
function highlightSyntax() {
    document.querySelectorAll('pre code').forEach(block => {
        // Skip if already highlighted
        if (block.classList.contains('highlighted')) return;

        let code = block.textContent;

        // 6502/6510 Assembly syntax highlighting

        // 1. Highlight comments (;)
        code = code.replace(/;.*/g, match => `<span class="asm-comment">${match}</span>`);

        // 2. Highlight hex values ($xx, $xxxx)
        code = code.replace(/\$[0-9A-Fa-f]+/g, match => `<span class="asm-hex">${match}</span>`);

        // 3. Highlight binary values (%)
        code = code.replace(/%[01]+/g, match => `<span class="asm-binary">${match}</span>`);

        // 4. Highlight decimal numbers
        code = code.replace(/\b\d+\b/g, match => `<span class="asm-number">${match}</span>`);

        // 5. Highlight opcodes (all 56 6502 instructions)
        const opcodes = 'ADC|AND|ASL|BCC|BCS|BEQ|BIT|BMI|BNE|BPL|BRK|BVC|BVS|CLC|CLD|CLI|CLV|CMP|CPX|CPY|DEC|DEX|DEY|EOR|INC|INX|INY|JMP|JSR|LDA|LDX|LDY|LSR|NOP|ORA|PHA|PHP|PLA|PLP|ROL|ROR|RTI|RTS|SBC|SEC|SED|SEI|STA|STX|STY|TAX|TAY|TSX|TXA|TXS|TYA';
        const opcodeRegex = new RegExp(`\\b(${opcodes})\\b`, 'gi');
        code = code.replace(opcodeRegex, match => `<span class="asm-opcode">${match.toUpperCase()}</span>`);

        // 6. Highlight labels (word followed by :)
        code = code.replace(/^(\w+):/gm, match => `<span class="asm-label">${match}</span>`);

        // 7. Highlight directives (.byte, .word, etc.)
        code = code.replace(/\.\w+/g, match => `<span class="asm-directive">${match}</span>`);

        block.innerHTML = code;
        block.classList.add('highlighted');
    });
}

// ===== AUTO-GENERATED TABLE OF CONTENTS =====
function generateTableOfContents() {
    // Only generate TOC for article pages or long documents
    const article = document.querySelector('.article-content, main');
    if (!article) return;

    // Find all headings
    const headings = article.querySelectorAll('h2, h3');
    if (headings.length < 3) return; // Skip if too few headings

    // Create TOC container
    const toc = document.createElement('div');
    toc.className = 'auto-toc';
    toc.innerHTML = `
        <div class="toc-header">
            <h3>📑 Table of Contents</h3>
            <button class="toc-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">−</button>
        </div>
        <nav class="toc-nav"></nav>
    `;

    const nav = toc.querySelector('.toc-nav');
    const tocList = document.createElement('ul');

    // Generate TOC entries
    headings.forEach((heading, index) => {
        // Create unique ID for heading
        const id = heading.id || `toc-section-${index}`;
        heading.id = id;

        // Create TOC item
        const li = document.createElement('li');
        li.className = `toc-${heading.tagName.toLowerCase()}`;

        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        link.onclick = (e) => {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Highlight the clicked section temporarily
            heading.classList.add('toc-highlight');
            setTimeout(() => heading.classList.remove('toc-highlight'), 2000);
        };

        li.appendChild(link);
        tocList.appendChild(li);
    });

    nav.appendChild(tocList);

    // Insert TOC at the beginning of the article
    const insertPoint = article.querySelector('h1, .article-meta');
    if (insertPoint && insertPoint.nextSibling) {
        insertPoint.parentNode.insertBefore(toc, insertPoint.nextSibling);
    } else {
        article.insertBefore(toc, article.firstChild);
    }

    // Add active section highlighting on scroll
    window.addEventListener('scroll', () => updateActiveTocSection(headings, tocList));
}

function updateActiveTocSection(headings, tocList) {
    const scrollPos = window.pageYOffset + 100;

    let activeIndex = 0;
    headings.forEach((heading, index) => {
        if (heading.offsetTop <= scrollPos) {
            activeIndex = index;
        }
    });

    // Update active class
    const tocItems = tocList.querySelectorAll('li');
    tocItems.forEach((item, index) => {
        item.classList.toggle('active', index === activeIndex);
    });
}

// ===== SEARCH AUTOCOMPLETE =====
function setupSearchAutocomplete() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;

    // Create suggestions container
    const suggestions = document.createElement('div');
    suggestions.className = 'search-autocomplete';
    suggestions.id = 'search-autocomplete';
    searchInput.parentElement.appendChild(suggestions);

    // Popular/common search terms
    const popularSearches = [
        'SID chip', 'VIC-II', 'VIC-II registers', 'sprites', 'sprite multiplexing',
        'raster interrupts', 'raster bars', 'music tracker', 'music editor',
        '6502 assembly', '6510 assembly', 'memory map', 'zero page',
        'CIA timer', 'CIA chip', 'joystick input', 'keyboard input',
        'screen RAM', 'color RAM', 'character set', 'bitmap mode',
        'sound effects', 'SID music', 'border color', 'background color',
        'BASIC programming', 'machine code', 'assembler', 'VICE emulator'
    ];

    let currentFocus = -1;

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        suggestions.innerHTML = '';
        currentFocus = -1;

        if (query.length < 2) {
            suggestions.style.display = 'none';
            return;
        }

        // Filter matching suggestions
        const matches = popularSearches
            .filter(term => term.toLowerCase().includes(query))
            .slice(0, 8);

        if (matches.length === 0) {
            suggestions.style.display = 'none';
            return;
        }

        // Display suggestions
        matches.forEach((term, index) => {
            const div = document.createElement('div');
            div.className = 'autocomplete-item';
            div.setAttribute('data-index', index);

            // Highlight matching part
            const termLower = term.toLowerCase();
            const startIdx = termLower.indexOf(query);
            const endIdx = startIdx + query.length;

            const before = term.substring(0, startIdx);
            const match = term.substring(startIdx, endIdx);
            const after = term.substring(endIdx);

            div.innerHTML = `${before}<strong>${match}</strong>${after}`;

            div.onclick = () => selectSuggestion(term);
            suggestions.appendChild(div);
        });

        suggestions.style.display = 'block';
    });

    function selectSuggestion(term) {
        searchInput.value = term;
        suggestions.style.display = 'none';
        // Trigger search
        const event = new KeyboardEvent('keyup', { key: 'Enter' });
        searchInput.dispatchEvent(event);
    }

    // Keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
        const items = suggestions.querySelectorAll('.autocomplete-item');
        if (items.length === 0) return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            currentFocus++;
            if (currentFocus >= items.length) currentFocus = 0;
            setActive(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            currentFocus--;
            if (currentFocus < 0) currentFocus = items.length - 1;
            setActive(items);
        } else if (e.key === 'Enter' && currentFocus > -1) {
            e.preventDefault();
            items[currentFocus].click();
        } else if (e.key === 'Escape') {
            suggestions.style.display = 'none';
            currentFocus = -1;
        }
    });

    function setActive(items) {
        items.forEach((item, index) => {
            item.classList.toggle('active', index === currentFocus);
        });
        if (currentFocus >= 0) {
            searchInput.value = items[currentFocus].textContent;
        }
    }

    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (e.target !== searchInput) {
            suggestions.style.display = 'none';
        }
    });
}

// ===== POPULAR ARTICLES SECTION =====
function displayPopularArticles() {
    const popularContainer = document.getElementById('popular-articles');
    if (!popularContainer) return;

    // This data should be generated by wiki_export.py
    // For now, we'll use the articles.json if available
    fetch('articles.json')
        .then(response => response.json())
        .then(data => {
            if (!data || !data.articles) return;

            // Sort by doc_count (most referenced)
            const popular = data.articles
                .sort((a, b) => b.doc_count - a.doc_count)
                .slice(0, 6);

            popularContainer.innerHTML = `
                <h2>🔥 Most Referenced Topics</h2>
                <div class="popular-grid">
                    ${popular.map((article, index) => `
                        <div class="popular-card">
                            <div class="popular-rank">#${index + 1}</div>
                            <h3><a href="articles/${article.filename}">${article.title}</a></h3>
                            <div class="popular-meta">
                                <span class="doc-count">📚 ${article.doc_count} references</span>
                                <span class="related-count">🔗 ${article.related_count || 0} related</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        })
        .catch(err => {
            console.error('Failed to load popular articles:', err);
        });
}

// ===== LAZY LOADING IMAGES =====
function setupLazyLoading() {
    // Check if IntersectionObserver is supported
    if ('IntersectionObserver' in window) {
        setupIntersectionObserver();
    } else {
        // Fallback for older browsers - load all images immediately
        loadAllImages();
    }
}

function setupIntersectionObserver() {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                loadImage(img);
                observer.unobserve(img);
            }
        });
    }, {
        // Start loading when image is 50px from viewport
        rootMargin: '50px',
        threshold: 0.01
    });

    // Find all images with loading="lazy" attribute
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        imageObserver.observe(img);
    });

    // Also observe dynamically added images
    const mutationObserver = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeName === 'IMG' && node.getAttribute('loading') === 'lazy') {
                    imageObserver.observe(node);
                } else if (node.querySelectorAll) {
                    const lazyImgs = node.querySelectorAll('img[loading="lazy"]');
                    lazyImgs.forEach(img => imageObserver.observe(img));
                }
            });
        });
    });

    mutationObserver.observe(document.body, {
        childList: true,
        subtree: true
    });

    console.log(`🖼️ Lazy loading enabled for ${lazyImages.length} images`);
}

function loadImage(img) {
    // Check if we have a data-src attribute (common lazy loading pattern)
    const src = img.getAttribute('data-src') || img.getAttribute('src');

    if (!src) return;

    // Create a new image to preload
    const tempImg = new Image();

    tempImg.onload = () => {
        img.src = src;
        img.classList.add('loaded');

        // Remove placeholder if it exists
        const placeholder = img.previousElementSibling;
        if (placeholder && placeholder.classList.contains('lazy-placeholder')) {
            placeholder.remove();
        }
    };

    tempImg.onerror = () => {
        console.error('Failed to load image:', src);
        img.classList.add('error');
    };

    // Start loading
    if (img.getAttribute('data-src')) {
        tempImg.src = img.getAttribute('data-src');
    } else {
        tempImg.src = src;
    }
}

function loadAllImages() {
    // Fallback for browsers without IntersectionObserver
    const lazyImages = document.querySelectorAll('img[loading="lazy"]');
    lazyImages.forEach(img => {
        const src = img.getAttribute('data-src') || img.getAttribute('src');
        if (src) {
            img.src = src;
            img.classList.add('loaded');
        }
    });

    console.log(`🖼️ Loaded ${lazyImages.length} images (fallback mode)`);
}

// Helper function to add lazy loading to dynamically created images
function makeLazyImage(src, alt = '', className = '') {
    const img = document.createElement('img');
    img.setAttribute('data-src', src);
    img.setAttribute('loading', 'lazy');
    img.alt = alt;
    if (className) img.className = className;

    // Add placeholder
    const placeholder = document.createElement('div');
    placeholder.className = 'lazy-placeholder';

    return { placeholder, img };
}

// Export for use in other scripts
window.lazyLoadingUtils = {
    makeLazyImage,
    loadImage
};

// ===== RELATED DOCUMENTS SIDEBAR =====
async function loadRelatedDocuments() {
    const sidebar = document.getElementById('related-docs-sidebar');
    if (!sidebar) return;

    // Get current document ID from page
    const currentDocId = sidebar.getAttribute('data-doc-id');
    if (!currentDocId) {
        sidebar.innerHTML = '<div class="no-related-docs">No document ID found</div>';
        return;
    }

    try {
        // Load similarities data
        const response = await fetch('../assets/data/similarities.json');
        const similarities = await response.json();

        // Get related docs for current document
        const relatedDocs = similarities[currentDocId] || [];

        if (relatedDocs.length === 0) {
            sidebar.innerHTML = '<div class="no-related-docs">No related documents found</div>';
            return;
        }

        // Build HTML for related docs
        const html = relatedDocs.slice(0, 5).map((doc, index) => {
            const scorePercent = Math.round(doc.score * 100);

            return `
                <div class="related-doc-item" style="animation: fadeIn 0.3s ease-out ${index * 0.1}s both">
                    <a href="${doc.filename}">${escapeHtml(doc.title)}</a>
                    <div class="related-doc-score">
                        <span>${scorePercent}%</span>
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: ${scorePercent}%"></div>
                        </div>
                    </div>
                    <div class="related-doc-meta">
                        ${doc.common_entities > 0 ? `<span>🏷️ ${doc.common_entities} shared topics</span>` : ''}
                        ${doc.common_tags > 0 ? `<span>📁 ${doc.common_tags} shared tags</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        sidebar.innerHTML = html;

    } catch (error) {
        console.error('Failed to load related documents:', error);
        sidebar.innerHTML = '<div class="no-related-docs">Unable to load related documents</div>';
    }
}

// Fade-in animation for related docs
const fadeInStyle = document.createElement('style');
fadeInStyle.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(fadeInStyle);

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== BOOKMARKS SYSTEM =====
class BookmarkManager {
    constructor() {
        this.bookmarks = JSON.parse(localStorage.getItem('c64-bookmarks') || '[]');
        this.init();
    }

    init() {
        // Add bookmark button to articles and documents
        this.addBookmarkButtons();
        // Show bookmark indicator if current page is bookmarked
        this.updateBookmarkStatus();
    }

    addBookmarkButtons() {
        // Add to article pages
        const article = document.querySelector('.article-content h1');
        if (article) {
            const pageId = this.getCurrentPageId();
            const button = this.createBookmarkButton(pageId);
            article.parentElement.insertBefore(button, article.nextSibling);
        }

        // Add to document pages
        const docTitle = document.querySelector('main h1');
        if (docTitle && !article) {
            const pageId = this.getCurrentPageId();
            const button = this.createBookmarkButton(pageId);
            docTitle.parentElement.insertBefore(button, docTitle.nextSibling);
        }
    }

    createBookmarkButton(pageId) {
        const button = document.createElement('button');
        button.className = 'bookmark-btn';
        button.setAttribute('data-page-id', pageId);

        const isBookmarked = this.isBookmarked(pageId);
        button.innerHTML = isBookmarked ? '⭐ Bookmarked' : '☆ Bookmark';
        button.classList.toggle('bookmarked', isBookmarked);

        button.onclick = (e) => {
            e.preventDefault();
            this.toggleBookmark(pageId);
        };

        return button;
    }

    getCurrentPageId() {
        const path = window.location.pathname;
        return path.split('/').pop().replace('.html', '') || 'index';
    }

    getCurrentPageTitle() {
        return document.querySelector('h1')?.textContent || 'Untitled';
    }

    getCurrentPageUrl() {
        return window.location.pathname;
    }

    isBookmarked(pageId) {
        return this.bookmarks.some(b => b.id === pageId);
    }

    toggleBookmark(pageId) {
        if (this.isBookmarked(pageId)) {
            this.removeBookmark(pageId);
        } else {
            this.addBookmark(pageId);
        }
    }

    addBookmark(pageId) {
        const bookmark = {
            id: pageId,
            title: this.getCurrentPageTitle(),
            url: this.getCurrentPageUrl(),
            date: Date.now()
        };

        this.bookmarks.push(bookmark);
        this.save();
        this.updateUI(pageId, true);
        this.showNotification('Bookmark added!');
    }

    removeBookmark(pageId) {
        this.bookmarks = this.bookmarks.filter(b => b.id !== pageId);
        this.save();
        this.updateUI(pageId, false);
        this.showNotification('Bookmark removed');
    }

    save() {
        localStorage.setItem('c64-bookmarks', JSON.stringify(this.bookmarks));
    }

    updateUI(pageId, isBookmarked) {
        const button = document.querySelector(`[data-page-id="${pageId}"]`);
        if (button) {
            button.innerHTML = isBookmarked ? '⭐ Bookmarked' : '☆ Bookmark';
            button.classList.toggle('bookmarked', isBookmarked);
        }
    }

    updateBookmarkStatus() {
        const pageId = this.getCurrentPageId();
        const button = document.querySelector(`[data-page-id="${pageId}"]`);
        if (button) {
            const isBookmarked = this.isBookmarked(pageId);
            button.classList.toggle('bookmarked', isBookmarked);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'bookmark-notification';
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => notification.classList.add('show'), 10);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    getBookmarks() {
        return this.bookmarks;
    }

    exportBookmarks() {
        const data = JSON.stringify(this.bookmarks, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'c64-bookmarks.json';
        a.click();
    }
}

// ===== AI CHATBOT =====
class AIChatbot {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.init();
    }

    init() {
        this.createChatWidget();
        this.loadHistory();
    }

    createChatWidget() {
        const widget = document.createElement('div');
        widget.className = 'chat-widget';
        widget.innerHTML = `
            <button class="chat-toggle" id="chat-toggle" title="Ask AI Assistant">
                <span class="bot-icon">🤖</span>
                <span class="bot-label">Ask AI</span>
            </button>
            <div class="chat-container" id="chat-container">
                <div class="chat-header">
                    <h3>🤖 C64 AI Assistant</h3>
                    <button class="chat-close" id="chat-close">×</button>
                </div>
                <div class="chat-messages" id="chat-messages">
                    <div class="chat-message bot">
                        <div class="message-content">
                            👋 Hi! I'm your C64 knowledge assistant. Ask me anything about:
                            <ul>
                                <li>6502/6510 assembly programming</li>
                                <li>VIC-II graphics & sprites</li>
                                <li>SID sound & music</li>
                                <li>Hardware (CIA, memory map, I/O)</li>
                                <li>Development tools & techniques</li>
                            </ul>
                            Try: "How do I change the border color?" or "Explain sprite multiplexing"
                        </div>
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" id="chat-input" placeholder="Ask about the C64..." autocomplete="off">
                    <button id="chat-send">Send</button>
                </div>
                <div class="chat-examples">
                    <small>Quick examples:</small>
                    <button class="example-btn" onclick="chatbot.askExample('How do I play a sound on the SID?')">🎵 Play sound</button>
                    <button class="example-btn" onclick="chatbot.askExample('Show me sprite example code')">🎮 Sprite code</button>
                </div>
            </div>
        `;

        document.body.appendChild(widget);
        this.attachEventListeners();
    }

    attachEventListeners() {
        document.getElementById('chat-toggle').onclick = () => this.toggle();
        document.getElementById('chat-close').onclick = () => this.toggle();
        document.getElementById('chat-send').onclick = () => this.sendMessage();

        const input = document.getElementById('chat-input');
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
    }

    toggle() {
        this.isOpen = !this.isOpen;
        document.getElementById('chat-container').classList.toggle('open', this.isOpen);

        if (this.isOpen) {
            document.getElementById('chat-input').focus();
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const question = input.value.trim();

        if (!question) return;

        // Add user message
        this.addMessage('user', question);
        input.value = '';

        // Show typing indicator
        this.showTyping();

        // Get answer (simulate for now, would use KB search + LLM in production)
        const answer = await this.getAnswer(question);

        // Remove typing indicator and add bot response
        this.hideTyping();
        this.addMessage('bot', answer.text, answer.sources);

        // Save to history
        this.saveHistory();
    }

    addMessage(role, content, sources = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;

        let messageHTML = `<div class="message-content">${this.formatMessage(content)}</div>`;

        if (sources && sources.length > 0) {
            messageHTML += '<div class="message-sources"><strong>Sources:</strong><ul>';
            sources.forEach(source => {
                messageHTML += `<li><a href="${source.url}">${source.title}</a></li>`;
            });
            messageHTML += '</ul></div>';
        }

        messageDiv.innerHTML = messageHTML;
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store message
        this.messages.push({ role, content, sources, timestamp: Date.now() });
    }

    formatMessage(text) {
        // Convert code blocks
        text = text.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Convert inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Convert line breaks
        text = text.replace(/\\n/g, '<br>');

        return text;
    }

    showTyping() {
        const messagesContainer = document.getElementById('chat-messages');
        const typing = document.createElement('div');
        typing.className = 'chat-message bot typing';
        typing.id = 'typing-indicator';
        typing.innerHTML = '<div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
        messagesContainer.appendChild(typing);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTyping() {
        const typing = document.getElementById('typing-indicator');
        if (typing) typing.remove();
    }

    async getAnswer(question) {
        // Simple pattern matching for demo (would use actual KB search + LLM in production)
        const answers = this.getPatternAnswers();

        const lowerQ = question.toLowerCase();

        // Check for pattern matches
        for (const [pattern, answer] of Object.entries(answers)) {
            if (lowerQ.includes(pattern)) {
                return {
                    text: answer.text,
                    sources: answer.sources || []
                };
            }
        }

        // Default response with search suggestion
        return {
            text: `I'll help you with that! While I'm in demo mode, I can provide information about C64 topics.

Try these resources:
• Search the wiki for "${question}"
• Browse the Articles section for comprehensive guides
• Check the Memory Map for hardware details

For the full AI assistant, we'd integrate:
1. Semantic search through the knowledge base
2. LLM API (Claude/GPT) for natural responses
3. Code generation from examples
4. Citation of source documents`,
            sources: [
                { title: 'Search Results', url: 'index.html?q=' + encodeURIComponent(question) },
                { title: 'Articles', url: 'articles.html' }
            ]
        };
    }

    getPatternAnswers() {
        return {
            'border color': {
                text: `To change the border color on the C64, write to address $D020 (53280 decimal).

**In BASIC:**
\`\`\`
POKE 53280, 0   ; Black border
POKE 53280, 1   ; White border
\`\`\`

**In Assembly:**
\`\`\`
LDA #$00        ; Black
STA $D020       ; Write to border color register
\`\`\`

Color values range from 0-15. The VIC-II chip controls this register.`,
                sources: [
                    { title: 'VIC-II Article', url: 'articles/VIC-II.html' },
                    { title: 'Color Article', url: 'articles/Color.html' }
                ]
            },
            'sprite': {
                text: `**Sprites on the C64:**

The VIC-II chip provides 8 hardware sprites (24x21 pixels each).

**Basic sprite setup:**
\`\`\`
; Enable sprite 0
LDA #$01
STA $D015       ; Sprite enable register

; Set X position
LDA #100
STA $D000       ; Sprite 0 X position

; Set Y position
LDA #50
STA $D001       ; Sprite 0 Y position

; Set sprite pointer (shape)
LDA #13
STA $07F8       ; Sprite 0 pointer
\`\`\`

Sprite data is 63 bytes (24x21 bits + 1 padding).`,
                sources: [
                    { title: 'Sprite Article', url: 'articles/Sprite.html' },
                    { title: 'VIC-II Article', url: 'articles/VIC-II.html' }
                ]
            },
            'sound': {
                text: `**Playing sounds on the SID chip:**

The SID chip ($D400-$D7FF) has 3 independent voices.

**Simple beep on Voice 1:**
\`\`\`
LDA #$0F
STA $D418       ; Max volume

LDA #$20
STA $D405       ; Attack/Decay
STA $D406       ; Sustain/Release

LDA #$10
STA $D401       ; Frequency high

LDA #$00
STA $D400       ; Frequency low

LDA #$11
STA $D404       ; Triangle wave + gate on
\`\`\``,
                sources: [
                    { title: 'SID Article', url: 'articles/SID.html' },
                    { title: 'Sound Article', url: 'articles/Sound.html' }
                ]
            },
            'memory map': {
                text: `**C64 Memory Map:**

\`\`\`
$0000-$00FF  Zero Page (256 bytes)
$0100-$01FF  Stack (256 bytes)
$0200-$03FF  OS/BASIC workspace
$0400-$07FF  Screen RAM (1000 bytes)
$0800-$9FFF  BASIC program & variables
$A000-$BFFF  BASIC ROM (8K)
$C000-$CFFF  RAM (4K)
$D000-$DFFF  I/O area (4K)
  $D000-$D3FF  VIC-II registers
  $D400-$D7FF  SID registers
  $DC00-$DCFF  CIA #1 registers
  $DD00-$DDFF  CIA #2 registers
$E000-$FFFF  Kernal ROM (8K)
\`\`\`

Total: 64K addressable memory with bank switching.`,
                sources: [
                    { title: 'Memory Article', url: 'articles/Memory.html' }
                ]
            }
        };
    }

    askExample(question) {
        document.getElementById('chat-input').value = question;
        this.sendMessage();
    }

    saveHistory() {
        localStorage.setItem('c64-chat-history', JSON.stringify(this.messages.slice(-50))); // Keep last 50
    }

    loadHistory() {
        const history = localStorage.getItem('c64-chat-history');
        if (history) {
            try {
                this.messages = JSON.parse(history);
                // Could restore messages to UI here if desired
            } catch (e) {
                console.error('Failed to load chat history:', e);
            }
        }
    }

    clearHistory() {
        this.messages = [];
        localStorage.removeItem('c64-chat-history');
        document.getElementById('chat-messages').innerHTML = '';
    }
}

// Global instances
let bookmarkManager;
let chatbot;

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
        // Alt+B = Toggle bookmarks view
        if (e.altKey && e.key === 'b') {
            e.preventDefault();
            showBookmarksPanel();
        }
        // Alt+C = Toggle chatbot
        if (e.altKey && e.key === 'c') {
            e.preventDefault();
            if (chatbot) chatbot.toggle();
        }
    });
}

function showBookmarksPanel() {
    if (!bookmarkManager) return;

    const bookmarks = bookmarkManager.getBookmarks();

    if (bookmarks.length === 0) {
        alert('No bookmarks yet! Use the ⭐ Bookmark button on articles to save them.');
        return;
    }

    const panel = document.createElement('div');
    panel.className = 'bookmarks-panel';
    panel.innerHTML = `
        <div class="bookmarks-content">
            <div class="bookmarks-header">
                <h2>📚 Your Bookmarks (${bookmarks.length})</h2>
                <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="bookmarks-list">
                ${bookmarks.map(b => `
                    <div class="bookmark-item">
                        <a href="${b.url}">${b.title}</a>
                        <button class="remove-bookmark" onclick="bookmarkManager.removeBookmark('${b.id}'); this.parentElement.remove();">🗑️</button>
                    </div>
                `).join('')}
            </div>
            <div class="bookmarks-footer">
                <button onclick="bookmarkManager.exportBookmarks()">💾 Export Bookmarks</button>
            </div>
        </div>
    `;

    document.body.appendChild(panel);
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

    // NEW FEATURES
    highlightSyntax();  // Syntax highlighting for code blocks
    // generateTableOfContents();  // Auto TOC for articles - DISABLED per user request
    setupSearchAutocomplete();  // Search suggestions
    displayPopularArticles();  // Popular articles on homepage
    bookmarkManager = new BookmarkManager();  // Bookmarks system
    chatbot = new AIChatbot();  // AI chatbot
    setupLazyLoading();  // Lazy load images for performance
    loadRelatedDocuments();  // Load related documents sidebar

    console.log('✅ Wiki enhancements loaded');
    console.log('💡 Keyboard shortcuts:');
    console.log('   Alt+T = Toggle theme');
    console.log('   Alt+H = Go to home');
    console.log('   Alt+B = View bookmarks');
    console.log('   Alt+C = Toggle AI chatbot');
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
        """Generate articles for major entities and topics (parallelized)."""
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

        # Collect all article tasks
        article_tasks = []
        for category, keywords in article_topics.items():
            for keyword in keywords:
                article_tasks.append((keyword, category))

        articles_generated = []
        max_workers = min(multiprocessing.cpu_count() * 2, 8)

        print(f"  Generating {len(article_tasks)} articles in parallel with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create partial function with entities_data
            create_func = partial(self._generate_single_article, entities_data=entities_data)

            # Submit all article generation tasks
            futures = {executor.submit(create_func, keyword, category): (keyword, category)
                      for keyword, category in article_tasks}

            # Wait for completion and collect results
            completed = 0
            for future in as_completed(futures):
                try:
                    article = future.result()
                    if article:
                        articles_generated.append(article)
                        completed += 1
                        if completed % 5 == 0 or completed == len(article_tasks):
                            print(f"    Progress: {completed}/{len(article_tasks)} articles")
                except Exception as e:
                    keyword, category = futures[future]
                    print(f"    Error generating article '{keyword}': {e}")

        # Generate articles browser page
        self._generate_articles_browser_html(articles_generated)

        # Save articles index
        articles_index_path = self.data_dir / "articles.json"
        with open(articles_index_path, 'w', encoding='utf-8') as f:
            json.dump(articles_generated, f, indent=2)
        print(f"  Saved: articles.json ({len(articles_generated)} articles)")

        self.stats['articles'] = len(articles_generated)
        return articles_generated

    def _generate_single_article(self, keyword: str, category: str, entities_data: Dict) -> Dict:
        """Generate a single article (helper for parallel processing)."""
        # Find matching entities
        matching_entities = self._find_entities_for_article(entities_data, keyword)

        if matching_entities:
            return self._create_article(keyword, category, matching_entities, entities_data)
        return None

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

        html_template = """<!DOCTYPE html>
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

        <nav class="breadcrumbs">
            <a href="../index.html">🏠 Home</a>
            <span class="separator">›</span>
            <a href="../articles.html">Articles</a>
            <span class="separator">›</span>
            <span class="current">{title_escaped}</span>
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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
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

{self._get_main_nav('articles')}

{self._get_unified_about_box()}

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
            <p>TDZ C64 Knowledge Base v2.23.16</p>
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
