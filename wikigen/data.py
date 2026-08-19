"""Everything that writes a JSON data file the exported wiki reads.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""

from typing import Dict, List
import re


class DataExportMixin:
    """Everything that writes a JSON data file the exported wiki reads."""

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

    def _export_search_index(self, documents_data: List[Dict], entities_data: Dict, articles_data: List[Dict]) -> Dict:
        """Export comprehensive search index for Fuse.js."""
        print("  Building search index...")
        search_items = []

        # Index articles
        for article in articles_data:
            search_items.append({
                'type': 'article',
                'title': article['title'],
                'category': article['category'],
                'url': f"articles/{article['filename']}",
                'description': f"{article['entity_count']} entities, {article['doc_count']} document references",
                'tags': [article['category'], 'article'],
                'relevance': article['doc_count']  # For sorting by importance
            })

        # Index documents
        for doc in documents_data:
            # Combine first few chunks for searchable content preview
            preview_chunks = doc.get('chunks', [])[:3]
            content_preview = ' '.join([chunk['content'][:200] for chunk in preview_chunks])[:500]

            search_items.append({
                'type': 'document',
                'title': doc['title'],
                'category': doc['file_type'].upper(),
                'url': f"docs/{re.sub(r'[^\\w\\-]', '_', doc['id'])}.html",
                'description': content_preview,
                'tags': doc.get('tags', []) + [doc['file_type'], 'document'],
                'relevance': doc.get('total_chunks', 0)
            })

        # Index entities (top entities by document count)
        for entity_type, entities in entities_data.items():
            # Only index top 20 entities per type to avoid bloat
            for entity in entities[:20]:
                search_items.append({
                    'type': 'entity',
                    'title': entity['text'],
                    'category': entity_type,
                    'url': f"entities.html?q={entity['text']}",
                    'description': f"{entity['doc_count']} document references",
                    'tags': [entity_type, 'entity'],
                    'relevance': entity['doc_count']
                })

        # Sort by relevance (most referenced first)
        search_items.sort(key=lambda x: x['relevance'], reverse=True)

        print(f"  Indexed {len(search_items)} items ({len(articles_data)} articles, {len(documents_data)} documents, {sum(min(20, len(entities)) for entities in entities_data.values())} entities)")

        return {
            'items': search_items,
            'stats': {
                'total': len(search_items),
                'articles': len(articles_data),
                'documents': len(documents_data),
                'entities': sum(min(20, len(entities)) for entities in entities_data.values())
            }
        }

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

    def _generate_readme(self):
        """
        Generate wiki/README.md from the actual output directory rather than
        hand-maintained prose. Must run after every HTML page and data file
        has been written, so the page/file lists and total size describe what
        export() actually produced (see GitHub issue #7 - the previous
        hand-written README drifted to claiming 4 pages/8 data files/~137 MB
        against a real export of 13 pages/14 files/202 MB).
        """
        html_pages = sorted(p.name for p in self.output_dir.glob('*.html'))
        data_files = sorted(self.data_dir.glob('*.json'))
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())

        page_list = '\n'.join(f"- `{name}`" for name in html_pages)
        data_list = '\n'.join(
            f"- `{f.name}` ({self._human_size(f.stat().st_size)})" for f in data_files
        )

        readme = f"""# TDZ C64 Knowledge Base - Wiki Export

Generated by `wiki_export.py` on {self.export_time} (server version {self.version}).
This file is regenerated on every export - edits made directly to it will be
overwritten the next time `wiki_export.py` runs.

## Statistics

- Documents: {self.stats['documents']}
- Chunks: {self.stats['chunks']}
- Entities: {self.stats['entities']}
- Topics: {self.stats['topics']}
- Clusters: {self.stats['clusters']}
- Events: {self.stats['events']}
- Articles: {self.stats.get('articles', 0)}

## Pages ({len(html_pages)})

{page_list}

## Data Files ({len(data_files)})

{data_list}

**Total Size:** {self._human_size(total_size)}

## Usage

Open `index.html` in a browser to view locally, or upload this directory to
any static host (GitHub Pages, Netlify, Vercel, etc.) - everything here is
static HTML/JS/JSON with no server-side component required.

Regenerate with:
```
python wiki_export.py --output {self.output_dir.name}
```
"""
        with open(self.output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme)
        print(f"  Saved: README.md ({len(html_pages)} pages, {len(data_files)} data files, {self._human_size(total_size)})")
