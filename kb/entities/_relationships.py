"""Entity relationships, the relationship graph, and analytics for EntitiesMixin.

Split out of kb/entities.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from datetime import datetime
from datetime import timedelta
from typing import Optional


class _RelationshipsMixin:

    def export_relationships(self, format: str = 'csv',
                            min_strength: float = 0.0,
                            entity_types: Optional[list] = None,
                            output_path: Optional[str] = None) -> str:
        """
        Export entity relationships to CSV or JSON format.

        Args:
            format: 'csv' or 'json'
            min_strength: Minimum relationship strength (0.0-1.0)
            entity_types: Filter by entity types (None = all types)
            output_path: Optional file path to write to (if None, returns string)

        Returns:
            Exported data as string (or writes to file if output_path provided)

        Example CSV:
            entity1,entity1_type,entity2,entity2_type,strength,doc_count
            VIC-II,hardware,sprite,concept,0.85,12
            SID,hardware,sound,concept,0.78,9

        Example JSON:
            [
                {
                    "entity1": "VIC-II",
                    "entity1_type": "hardware",
                    "entity2": "sprite",
                    "entity2_type": "concept",
                    "strength": 0.85,
                    "doc_count": 12
                },
                ...
            ]
        """
        import csv
        import json
        from io import StringIO

        cursor = self.db_conn.cursor()

        # Build query with filters
        query = """
            SELECT entity1_text, entity1_type,
                   entity2_text, entity2_type,
                   strength, doc_count
            FROM entity_relationships
            WHERE strength >= ?
        """
        params = [min_strength]

        if entity_types:
            placeholders = ','.join(['?'] * len(entity_types))
            query += f" AND (entity1_type IN ({placeholders}) OR entity2_type IN ({placeholders}))"
            params.extend(entity_types * 2)  # Add for both entity1_type and entity2_type

        query += """
            ORDER BY strength DESC, doc_count DESC
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if format.lower() == 'csv':
            output = StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(['entity1', 'entity1_type', 'entity2', 'entity2_type',
                           'strength', 'doc_count'])

            # Write data
            for row in rows:
                writer.writerow([
                    row[0],  # entity1_text
                    row[1],  # entity1_type
                    row[2],  # entity2_text
                    row[3],  # entity2_type
                    f"{row[4]:.3f}",  # strength
                    row[5]   # doc_count
                ])

            result = output.getvalue()
            output.close()

        elif format.lower() == 'json':
            relationships = []
            for row in rows:
                relationships.append({
                    'entity1': row[0],
                    'entity1_type': row[1],
                    'entity2': row[2],
                    'entity2_type': row[3],
                    'strength': round(row[4], 3),
                    'doc_count': row[5]
                })

            result = json.dumps(relationships, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

        # Write to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            self.logger.info(f"Exported {len(rows)} relationships to {output_path}")

        return result

    def extract_entity_relationships(self, doc_id: str, min_confidence: float = 0.6,
                                   force_regenerate: bool = False) -> dict:
        """
        Extract entity co-occurrence relationships from a document.

        This method analyzes how entities appear together in document chunks
        to identify related concepts, hardware, instructions, etc.

        Args:
            doc_id: Document ID to extract relationships from
            min_confidence: Minimum confidence threshold for entities (default: 0.6)
            force_regenerate: If True, regenerate relationships even if they exist

        Returns:
            {
                'doc_id': str,
                'relationship_count': int,
                'relationships': [
                    {
                        'entity1': str,
                        'entity1_type': str,
                        'entity2': str,
                        'entity2_type': str,
                        'strength': float,
                        'context': str
                    },
                    ...
                ]
            }

        Examples:
            # Extract relationships from a document
            result = kb.extract_entity_relationships('doc-id-123')

            # Force regeneration with higher confidence
            result = kb.extract_entity_relationships('doc-id-123',
                min_confidence=0.7, force_regenerate=True)
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        self.logger.info(f"Extracting entity relationships from document {doc_id}")

        # Get all entities for this document
        entity_result = self.get_entities(doc_id, min_confidence=min_confidence)

        if not entity_result['entities']:
            self.logger.info(f"No entities found for document {doc_id}")
            return {
                'doc_id': doc_id,
                'relationship_count': 0,
                'relationships': []
            }

        # Get all chunks for this document
        chunks = self._get_chunks_db(doc_id)

        if not chunks:
            self.logger.info(f"No chunks found for document {doc_id}")
            return {
                'doc_id': doc_id,
                'relationship_count': 0,
                'relationships': []
            }

        # Build entity -> type mapping from the entities list
        entity_map = {e['entity_text']: e['entity_type'] for e in entity_result['entities']}

        # Find co-occurrences with enhanced distance-based strength calculation
        relationships = {}  # (entity1, entity2) -> strength
        relationship_contexts = {}  # (entity1, entity2) -> context

        for chunk in chunks:
            content = chunk.content
            # Find which entities appear in this chunk and their positions
            entity_positions = {}
            for entity in entity_map.keys():
                pos = content.find(entity)
                if pos != -1:
                    entity_positions[entity] = pos

            entities_in_chunk = list(entity_positions.keys())

            # Create pairs of co-occurring entities with distance-based scoring
            for i, e1 in enumerate(entities_in_chunk):
                for e2 in entities_in_chunk[i+1:]:
                    # Sort entities alphabetically to avoid duplicates (A,B) vs (B,A)
                    pair = tuple(sorted([e1, e2]))

                    # Calculate distance-based strength
                    pos1 = entity_positions[e1]
                    pos2 = entity_positions[e2]
                    distance = abs(pos2 - pos1)

                    # Distance-based weight: closer = stronger
                    # Uses exponential decay: weight = e^(-distance/decay_factor)
                    import math
                    decay_factor = 500  # Characters - tune this for sensitivity
                    distance_weight = math.exp(-distance / decay_factor)

                    # Base strength from co-occurrence + distance weight
                    # This gives 1.0 for same position, ~0.61 for 500 chars apart, ~0.37 for 1000 chars
                    strength_increment = 1.0 * distance_weight

                    # Add to cumulative strength
                    relationships[pair] = relationships.get(pair, 0) + strength_increment

                    # Store context (first occurrence)
                    if pair not in relationship_contexts:
                        # Extract context around both entities
                        start_idx = max(0, min(pos1, pos2) - 50)
                        end_idx = min(len(content), max(pos1, pos2) + max(len(e1), len(e2)) + 50)
                        context = content[start_idx:end_idx].strip()
                        relationship_contexts[pair] = context

        # Enhanced normalization with logarithmic scaling
        # This prevents a few very high counts from dominating
        if relationships:
            max_strength = max(relationships.values())
            # Use log scaling for better distribution
            normalized_relationships = {}
            for pair, strength in relationships.items():
                # Log-scale normalization: more even distribution of scores
                # Formula: log(1 + strength) / log(1 + max_strength)
                import math
                normalized_relationships[pair] = math.log(1 + strength) / math.log(1 + max_strength)
        else:
            normalized_relationships = {}

        # Store relationships in database
        cursor = self.db_conn.cursor()
        from datetime import datetime
        now = datetime.now().isoformat()

        relationship_list = []
        for (e1, e2), strength in normalized_relationships.items():
            e1_type = entity_map[e1]
            e2_type = entity_map[e2]
            context = relationship_contexts[(e1, e2)]

            # Check if relationship already exists
            cursor.execute("""
                SELECT strength, doc_count FROM entity_relationships
                WHERE entity1_text = ? AND entity2_text = ? AND relationship_type = 'co-occurrence'
            """, (e1, e2))

            existing = cursor.fetchone()

            if existing:
                # Update existing relationship
                old_strength, doc_count = existing
                new_strength = (old_strength * doc_count + strength) / (doc_count + 1)
                new_doc_count = doc_count + 1

                cursor.execute("""
                    UPDATE entity_relationships
                    SET strength = ?, doc_count = ?, last_updated = ?, context_sample = ?
                    WHERE entity1_text = ? AND entity2_text = ? AND relationship_type = 'co-occurrence'
                """, (new_strength, new_doc_count, now, context, e1, e2))
            else:
                # Insert new relationship
                cursor.execute("""
                    INSERT INTO entity_relationships
                    (entity1_text, entity1_type, entity2_text, entity2_type,
                     relationship_type, strength, doc_count, first_seen_doc,
                     context_sample, last_updated)
                    VALUES (?, ?, ?, ?, 'co-occurrence', ?, 1, ?, ?, ?)
                """, (e1, e1_type, e2, e2_type, strength, doc_id, context, now))

            relationship_list.append({
                'entity1': e1,
                'entity1_type': e1_type,
                'entity2': e2,
                'entity2_type': e2_type,
                'strength': strength,
                'context': context
            })

        self.db_conn.commit()

        self.logger.info(f"Extracted {len(relationship_list)} relationships from {doc_id}")

        return {
            'doc_id': doc_id,
            'relationship_count': len(relationship_list),
            'relationships': sorted(relationship_list, key=lambda x: x['strength'], reverse=True)
        }

    def get_entity_relationships(self, entity_text: str,
                                relationship_type: Optional[str] = None,
                                min_strength: float = 0.0,
                                max_results: int = 20) -> list:
        """
        Get all relationships for a given entity.

        Args:
            entity_text: The entity to find relationships for
            relationship_type: Filter by relationship type (default: all types)
            min_strength: Minimum relationship strength (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            List of relationships:
            [
                {
                    'related_entity': str,
                    'related_type': str,
                    'relationship_type': str,
                    'strength': float,
                    'doc_count': int,
                    'context_sample': str
                },
                ...
            ]

        Examples:
            # Find entities related to VIC-II
            relationships = kb.get_entity_relationships('VIC-II')

            # Find strong co-occurrences only
            relationships = kb.get_entity_relationships('VIC-II',
                relationship_type='co-occurrence', min_strength=0.5)
        """
        cursor = self.db_conn.cursor()

        # Build query
        query = """
            SELECT entity1_text, entity1_type, entity2_text, entity2_type,
                   relationship_type, strength, doc_count, context_sample
            FROM entity_relationships
            WHERE (entity1_text = ? OR entity2_text = ?)
              AND strength >= ?
        """
        params = [entity_text, entity_text, min_strength]

        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)

        query += " ORDER BY strength DESC, doc_count DESC LIMIT ?"
        params.append(max_results)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            e1_text, e1_type, e2_text, e2_type, rel_type, strength, doc_count, context = row

            # Determine which is the "other" entity
            if e1_text == entity_text:
                related_entity = e2_text
                related_type = e2_type
            else:
                related_entity = e1_text
                related_type = e1_type

            results.append({
                'related_entity': related_entity,
                'related_type': related_type,
                'relationship_type': rel_type,
                'strength': strength,
                'doc_count': doc_count,
                'context_sample': context
            })

        return results

    def find_related_entities(self, entity_text: str, max_results: int = 10) -> list:
        """
        Discover entities related to a given entity (convenience method).

        This is a simplified version of get_entity_relationships() optimized
        for entity discovery and exploration.

        Args:
            entity_text: The entity to find related entities for
            max_results: Maximum number of related entities to return

        Returns:
            List of related entities with strength scores

        Examples:
            # Discover entities related to SID chip
            related = kb.find_related_entities('SID')
        """
        return self.get_entity_relationships(
            entity_text=entity_text,
            relationship_type='co-occurrence',
            min_strength=0.3,
            max_results=max_results
        )

    def search_by_entity_pair(self, entity1: str, entity2: str,
                             max_results: int = 10) -> list:
        """
        Find documents that contain both entities.

        Args:
            entity1: First entity to search for
            entity2: Second entity to search for
            max_results: Maximum number of documents to return

        Returns:
            List of documents containing both entities:
            [
                {
                    'doc_id': str,
                    'title': str,
                    'entity1_count': int,
                    'entity2_count': int,
                    'contexts': [str, ...]  # Snippets showing both entities
                },
                ...
            ]

        Examples:
            # Find docs about VIC-II and raster interrupts
            docs = kb.search_by_entity_pair('VIC-II', 'raster interrupt')
        """
        cursor = self.db_conn.cursor()

        # Find documents containing both entities
        query = """
            SELECT e1.doc_id,
                   COUNT(DISTINCT e1.entity_id) as entity1_count,
                   COUNT(DISTINCT e2.entity_id) as entity2_count
            FROM document_entities e1
            JOIN document_entities e2 ON e1.doc_id = e2.doc_id
            WHERE e1.entity_text = ? AND e2.entity_text = ?
            GROUP BY e1.doc_id
            ORDER BY entity1_count + entity2_count DESC
            LIMIT ?
        """

        cursor.execute(query, (entity1, entity2, max_results))
        rows = cursor.fetchall()

        results = []
        for doc_id, e1_count, e2_count in rows:
            doc = self.documents.get(doc_id)
            if not doc:
                continue

            # Get context snippets showing both entities
            chunks = self._get_chunks_db(doc_id)
            contexts = []

            for chunk in chunks:
                if entity1 in chunk.content and entity2 in chunk.content:
                    # Extract snippet around both entities
                    idx1 = chunk.content.find(entity1)
                    idx2 = chunk.content.find(entity2)
                    start = max(0, min(idx1, idx2) - 50)
                    end = min(len(chunk.content), max(idx1 + len(entity1), idx2 + len(entity2)) + 50)
                    context = chunk.content[start:end].strip()
                    contexts.append(context)

            results.append({
                'doc_id': doc_id,
                'title': doc.title,
                'entity1_count': e1_count,
                'entity2_count': e2_count,
                'contexts': contexts[:3]  # Return top 3 contexts
            })

        return results

    def extract_relationships_bulk(self, min_confidence: float = 0.6,
                                   max_docs: Optional[int] = None,
                                   skip_existing: bool = False) -> dict:
        """
        Bulk extract entity relationships for multiple documents.

        Args:
            min_confidence: Minimum confidence threshold for entities
            max_docs: Maximum number of documents to process (None = all)
            skip_existing: If True, skip documents that already have relationships

        Returns:
            {
                'processed': int,
                'failed': int,
                'skipped': int,
                'total_relationships': int,
                'results': [list of individual results]
            }

        Examples:
            # Extract relationships for all documents
            results = kb.extract_relationships_bulk()

            # Extract for documents with entities only
            results = kb.extract_relationships_bulk(skip_existing=True)
        """
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_relationships': 0,
            'results': []
        }

        # Get documents with entities
        docs_to_process = []
        cursor = self.db_conn.cursor()

        for doc_id in self.documents.keys():
            cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
            entity_count = cursor.fetchone()[0]

            if entity_count > 0:
                docs_to_process.append(doc_id)

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Extracting entity relationships for {len(docs_to_process)} documents")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'status': 'unknown',
                'relationship_count': 0,
                'error': None
            }

            try:
                self.logger.info(f"[{i}/{len(docs_to_process)}] Extracting relationships from {doc_id}")

                result = self.extract_entity_relationships(doc_id, min_confidence=min_confidence)

                doc_results['status'] = 'success'
                doc_results['relationship_count'] = result['relationship_count']

                results['processed'] += 1
                results['total_relationships'] += result['relationship_count']
                results['results'].append(doc_results)

            except Exception as e:
                self.logger.error(f"[{i}/{len(docs_to_process)}] Failed to extract relationships from {doc_id}: {e}")
                doc_results['status'] = 'failed'
                doc_results['error'] = str(e)
                results['failed'] += 1
                results['results'].append(doc_results)

        self.logger.info(f"Bulk relationship extraction complete: processed={results['processed']}, "
                        f"failed={results['failed']}, total_relationships={results['total_relationships']}")

        return results

    def add_relationship(self, from_doc_id: str, to_doc_id: str,
                        relationship_type: str = "related", note: str = "") -> dict:
        """
        Add a relationship between two documents.

        Args:
            from_doc_id: Source document ID
            to_doc_id: Target document ID
            relationship_type: Type of relationship (e.g., 'related', 'references', 'prerequisite', 'sequel')
            note: Optional note about the relationship

        Returns:
            Dictionary with relationship details

        Examples:
            # Mark document as related
            kb.add_relationship("doc1", "doc2", "related", "Both cover VIC-II graphics")

            # Mark as prerequisite
            kb.add_relationship("basic_guide", "advanced_guide", "prerequisite", "Read basic first")
        """
        # Validate documents exist
        if from_doc_id not in self.documents:
            raise ValueError(f"Source document not found: {from_doc_id}")
        if to_doc_id not in self.documents:
            raise ValueError(f"Target document not found: {to_doc_id}")

        # Create relationships table if it doesn't exist
        cursor = self.db_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_doc_id TEXT NOT NULL,
                to_doc_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (from_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                FOREIGN KEY (to_doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                UNIQUE(from_doc_id, to_doc_id, relationship_type)
            )
        """)

        # Check if relationship already exists
        cursor.execute("""
            SELECT id FROM document_relationships
            WHERE from_doc_id = ? AND to_doc_id = ? AND relationship_type = ?
        """, (from_doc_id, to_doc_id, relationship_type))

        if cursor.fetchone():
            raise ValueError(f"Relationship already exists: {from_doc_id} -> {to_doc_id} ({relationship_type})")

        # Insert relationship
        created_at = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO document_relationships (from_doc_id, to_doc_id, relationship_type, note, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (from_doc_id, to_doc_id, relationship_type, note, created_at))

        self.db_conn.commit()

        self.logger.info(f"Added relationship: {from_doc_id} -> {to_doc_id} ({relationship_type})")

        return {
            'from_doc_id': from_doc_id,
            'to_doc_id': to_doc_id,
            'relationship_type': relationship_type,
            'note': note,
            'created_at': created_at
        }

    def remove_relationship(self, from_doc_id: str, to_doc_id: str,
                           relationship_type: Optional[str] = None) -> bool:
        """
        Remove a relationship between two documents.

        Args:
            from_doc_id: Source document ID
            to_doc_id: Target document ID
            relationship_type: Optional specific relationship type to remove

        Returns:
            True if relationship was removed, False if not found
        """
        cursor = self.db_conn.cursor()

        if relationship_type:
            cursor.execute("""
                DELETE FROM document_relationships
                WHERE from_doc_id = ? AND to_doc_id = ? AND relationship_type = ?
            """, (from_doc_id, to_doc_id, relationship_type))
        else:
            cursor.execute("""
                DELETE FROM document_relationships
                WHERE from_doc_id = ? AND to_doc_id = ?
            """, (from_doc_id, to_doc_id))

        self.db_conn.commit()
        removed = cursor.rowcount > 0

        if removed:
            self.logger.info(f"Removed relationship: {from_doc_id} -> {to_doc_id}")

        return removed

    def get_entity_analytics(self, time_range_days: int = 30) -> dict:
        """
        Get comprehensive entity analytics for dashboard visualization.

        Provides aggregate statistics and trends for entities and relationships
        across the knowledge base.

        Args:
            time_range_days: Number of days to include in timeline (default: 30)

        Returns:
            {
                'entity_distribution': {                    # Entities by type
                    'hardware': 45,
                    'memory_address': 120,
                    'instruction': 89,
                    ...
                },
                'top_entities': [                           # Most common entities
                    {
                        'entity_text': 'VIC-II',
                        'entity_type': 'hardware',
                        'doc_count': 15,
                        'total_occurrences': 127,
                        'avg_confidence': 0.92
                    },
                    ...
                ],
                'relationship_stats': {                     # Relationship statistics
                    'total': 234,
                    'avg_strength': 0.75,
                    'by_type': {'references': 120, 'related_to': 114}
                },
                'top_relationships': [                      # Strongest relationships
                    {
                        'entity1': 'VIC-II',
                        'entity2': 'sprite',
                        'strength': 0.95,
                        'doc_count': 12
                    },
                    ...
                ],
                'extraction_timeline': [                    # Entities extracted over time
                    {'date': '2024-12-01', 'count': 45},
                    {'date': '2024-12-02', 'count': 67},
                    ...
                ]
            }
        """
        cursor = self.db_conn.cursor()
        analytics = {}

        # 1. Entity distribution by type
        cursor.execute("""
            SELECT entity_type, COUNT(DISTINCT entity_text) as count
            FROM document_entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        analytics['entity_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}

        # 2. Top entities (most common across documents)
        cursor.execute("""
            SELECT
                entity_text,
                entity_type,
                COUNT(DISTINCT doc_id) as doc_count,
                SUM(occurrence_count) as total_occurrences,
                AVG(confidence) as avg_confidence
            FROM document_entities
            GROUP BY entity_text, entity_type
            HAVING doc_count >= 2
            ORDER BY doc_count DESC, total_occurrences DESC
            LIMIT 50
        """)

        analytics['top_entities'] = []
        for row in cursor.fetchall():
            analytics['top_entities'].append({
                'entity_text': row[0],
                'entity_type': row[1],
                'doc_count': row[2],
                'total_occurrences': row[3],
                'avg_confidence': round(row[4], 2)
            })

        # 3. Relationship statistics
        cursor.execute("""
            SELECT COUNT(*) FROM entity_relationships
        """)
        total_relationships = cursor.fetchone()[0]

        cursor.execute("""
            SELECT AVG(strength) FROM entity_relationships
        """)
        avg_strength = cursor.fetchone()[0] or 0.0

        cursor.execute("""
            SELECT relationship_type, COUNT(*) as count
            FROM entity_relationships
            GROUP BY relationship_type
            ORDER BY count DESC
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        analytics['relationship_stats'] = {
            'total': total_relationships,
            'avg_strength': round(avg_strength, 2),
            'by_type': by_type
        }

        # 4. Top relationships (strongest entity pairs)
        cursor.execute("""
            SELECT
                r.entity1_text,
                r.entity1_type,
                r.entity2_text,
                r.entity2_type,
                r.strength,
                r.doc_count
            FROM entity_relationships r
            WHERE r.doc_count >= 1
            ORDER BY r.strength DESC, r.doc_count DESC
            LIMIT 50
        """)

        analytics['top_relationships'] = []
        for row in cursor.fetchall():
            analytics['top_relationships'].append({
                'entity1': row[0],
                'entity1_type': row[1],
                'entity2': row[2],
                'entity2_type': row[3],
                'strength': round(row[4], 2) if row[4] else 0.0,
                'doc_count': row[5]
            })

        # 5. Extraction timeline (entities extracted per day)
        from datetime import datetime, timedelta
        start_date = datetime.now() - timedelta(days=time_range_days)
        start_date_str = start_date.strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT
                DATE(generated_at) as extraction_date,
                COUNT(DISTINCT entity_text) as count
            FROM document_entities
            WHERE generated_at >= ?
            GROUP BY DATE(generated_at)
            ORDER BY extraction_date ASC
        """, (start_date_str,))

        analytics['extraction_timeline'] = []
        for row in cursor.fetchall():
            analytics['extraction_timeline'].append({
                'date': row[0],
                'count': row[1]
            })

        # 6. Overall statistics
        cursor.execute("""
            SELECT
                COUNT(DISTINCT entity_text) as unique_entities,
                COUNT(*) as total_entity_instances,
                COUNT(DISTINCT doc_id) as docs_with_entities
            FROM document_entities
        """)
        row = cursor.fetchone()

        analytics['overall'] = {
            'unique_entities': row[0],
            'total_instances': row[1],
            'docs_with_entities': row[2],
            'avg_entities_per_doc': round(row[1] / row[2], 1) if row[2] > 0 else 0
        }

        return analytics

    def get_relationships(self, doc_id: str, direction: str = "both") -> list[dict]:
        """
        Get all relationships for a document.

        Args:
            doc_id: Document ID
            direction: 'outgoing', 'incoming', or 'both' (default)

        Returns:
            List of relationship dictionaries
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        cursor = self.db_conn.cursor()
        relationships = []

        # Get outgoing relationships (this doc -> others)
        if direction in ["outgoing", "both"]:
            cursor.execute("""
                SELECT from_doc_id, to_doc_id, relationship_type, note, created_at
                FROM document_relationships
                WHERE from_doc_id = ?
            """, (doc_id,))

            for row in cursor.fetchall():
                relationships.append({
                    'direction': 'outgoing',
                    'from_doc_id': row[0],
                    'to_doc_id': row[1],
                    'relationship_type': row[2],
                    'note': row[3],
                    'created_at': row[4],
                    'related_doc_id': row[1]  # For convenience
                })

        # Get incoming relationships (others -> this doc)
        if direction in ["incoming", "both"]:
            cursor.execute("""
                SELECT from_doc_id, to_doc_id, relationship_type, note, created_at
                FROM document_relationships
                WHERE to_doc_id = ?
            """, (doc_id,))

            for row in cursor.fetchall():
                relationships.append({
                    'direction': 'incoming',
                    'from_doc_id': row[0],
                    'to_doc_id': row[1],
                    'relationship_type': row[2],
                    'note': row[3],
                    'created_at': row[4],
                    'related_doc_id': row[0]  # For convenience
                })

        return relationships

    def get_related_documents(self, doc_id: str, relationship_type: Optional[str] = None) -> list[dict]:
        """
        Get all documents related to a given document with full metadata.

        Args:
            doc_id: Document ID
            relationship_type: Optional filter by relationship type

        Returns:
            List of related documents with relationship info
        """
        relationships = self.get_relationships(doc_id)

        if relationship_type:
            relationships = [r for r in relationships if r['relationship_type'] == relationship_type]

        related_docs = []
        for rel in relationships:
            related_doc_id = rel['related_doc_id']
            if related_doc_id in self.documents:
                doc_meta = self.documents[related_doc_id]
                related_docs.append({
                    'doc_id': doc_meta.doc_id,
                    'title': doc_meta.title,
                    'filename': doc_meta.filename,
                    'tags': doc_meta.tags,
                    'relationship_type': rel['relationship_type'],
                    'relationship_direction': rel['direction'],
                    'note': rel['note'],
                    'created_at': rel['created_at']
                })

        return related_docs

    def get_relationship_graph(self, tags: Optional[list[str]] = None,
                              relationship_types: Optional[list[str]] = None) -> dict:
        """
        Get relationship graph data for visualization.

        Args:
            tags: Optional list of tags to filter documents
            relationship_types: Optional list of relationship types to include

        Returns:
            Dictionary with 'nodes' and 'edges' for graph visualization
        """
        cursor = self.db_conn.cursor()

        # Build WHERE clauses for filtering
        where_clauses = []
        params = []

        if relationship_types:
            placeholders = ','.join('?' * len(relationship_types))
            where_clauses.append(f"relationship_type IN ({placeholders})")
            params.extend(relationship_types)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get all relationships
        query = f"""
            SELECT from_doc_id, to_doc_id, relationship_type, note
            FROM document_relationships
            {where_sql}
        """

        cursor.execute(query, params)
        relationships = cursor.fetchall()

        # Build nodes and edges
        nodes = {}
        edges = []

        for from_id, to_id, rel_type, note in relationships:
            # Check if documents match tag filter
            if tags:
                from_doc = self.documents.get(from_id)
                to_doc = self.documents.get(to_id)
                if not from_doc or not to_doc:
                    continue
                if not any(tag in from_doc.tags for tag in tags) and \
                   not any(tag in to_doc.tags for tag in tags):
                    continue

            # Add nodes if not already present
            if from_id not in nodes:
                doc = self.documents.get(from_id)
                if doc:
                    nodes[from_id] = {
                        'id': from_id,
                        'label': doc.title,
                        'title': f"{doc.title}\n{doc.filename}\nTags: {', '.join(doc.tags)}",
                        'tags': doc.tags,
                        'chunks': doc.total_chunks
                    }

            if to_id not in nodes:
                doc = self.documents.get(to_id)
                if doc:
                    nodes[to_id] = {
                        'id': to_id,
                        'label': doc.title,
                        'title': f"{doc.title}\n{doc.filename}\nTags: {', '.join(doc.tags)}",
                        'tags': doc.tags,
                        'chunks': doc.total_chunks
                    }

            # Add edge
            if from_id in nodes and to_id in nodes:
                edges.append({
                    'from': from_id,
                    'to': to_id,
                    'type': rel_type,
                    'label': rel_type,
                    'title': note if note else rel_type
                })

        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'relationship_types': list(set(e['type'] for e in edges))
            }
        }
