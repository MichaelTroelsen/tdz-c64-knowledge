"""Natural-language query translation, faceted search, similarity/comparison, and tables/code search for SearchMixin.

Split out of kb/search.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from features import faiss
from typing import Optional
import json
import numpy as np
import time


class _QueryMixin:

    def translate_nl_query(self, query: str, confidence_threshold: float = 0.7) -> dict:
        """
        Translate natural language query to structured search parameters using LLM.

        Parses natural language queries and extracts:
        - Search terms (keywords to search for)
        - Entity mentions (hardware, addresses, people, etc.)
        - Facet filters (hardware types, memory ranges, etc.)
        - Recommended search mode (keyword, semantic, hybrid)

        Args:
            query: Natural language query (e.g., "find information about sprites on the VIC-II chip")
            confidence_threshold: Minimum confidence for entity extraction (0.0-1.0, default: 0.7)

        Returns:
            {
                'original_query': str,                    # Original query
                'search_terms': [str],                    # Extracted keywords
                'facet_filters': {                        # Mapped facet filters
                    'hardware': ['VIC-II'],
                    'registers': ['$D000-$D3FF'],
                    ...
                },
                'search_mode': 'hybrid',                  # Recommended mode: 'keyword', 'semantic', 'hybrid'
                'confidence': 0.85,                       # Overall confidence (0.0-1.0)
                'entities_found': [                       # Extracted entities
                    {'text': 'VIC-II', 'type': 'hardware', 'confidence': 0.95},
                    ...
                ],
                'reasoning': str                          # Explanation of translation
            }

        Example:
            >>> kb.translate_nl_query("find sprite information")
            {
                'search_terms': ['sprite', 'information'],
                'facet_filters': {'hardware': ['VIC-II']},
                'search_mode': 'hybrid',
                'entities_found': [{'text': 'sprite', 'type': 'graphics', 'confidence': 0.8}],
                ...
            }
        """
        self.logger.info(f"Translating natural language query: '{query}'")
        start_time = time.time()

        # Get LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found")

        llm_client = get_llm_client()
        if not llm_client:
            # Fallback: Basic keyword extraction without LLM
            self.logger.warning("LLM not configured, using basic keyword extraction")
            words = query.lower().split()
            search_terms = [w for w in words if len(w) > 3]
            return {
                'original_query': query,
                'search_terms': search_terms,
                'facet_filters': {},
                'search_mode': 'keyword',
                'confidence': 0.5,
                'entities_found': [],
                'reasoning': 'LLM not available - basic keyword extraction used',
                'suggested_query': ' '.join(search_terms) or query
            }

        # Build structured prompt for LLM
        prompt = f"""You are a Commodore 64 technical documentation query parser.

User Query: "{query}"

Parse this query and extract structured information in JSON format:

{{
  "search_terms": ["keyword1", "keyword2"],        // Main keywords to search for
  "entities": [                                     // Detected technical entities
    {{"text": "VIC-II", "type": "hardware", "confidence": 0.95}},
    {{"text": "$D000", "type": "memory_address", "confidence": 0.90}}
  ],
  "search_mode": "hybrid",                         // "keyword", "semantic", or "hybrid"
  "confidence": 0.85,                              // Overall confidence (0.0-1.0)
  "reasoning": "Detected VIC-II chip mention..."  // Brief explanation
}}

Entity Types:
- hardware: Chip names (VIC-II, SID, CIA, 6502, 6526, 6581, etc.)
- memory_address: Memory addresses ($D000, $D020, 53280, 0xD020, etc.)
- instruction: Assembly instructions (LDA, STA, JMP, JSR, etc.)
- person: People mentioned (Bob Yannes, Jack Tramiel, etc.)
- register: Hardware registers (sprite registers, color registers, etc.)
- graphics: Graphics concepts (sprites, bitmap, character mode, etc.)
- audio: Sound concepts (voices, waveforms, ADSR, etc.)

Search Mode Guidelines:
- "keyword": For specific technical terms, exact matches
- "semantic": For conceptual questions, "how does X work", explanations
- "hybrid": For mixed queries with both specific terms and concepts

Return ONLY valid JSON, no additional text.
"""

        try:
            # Call LLM with low temperature for deterministic parsing
            response_text = llm_client.call(prompt, temperature=0.3, max_tokens=512)

            # Parse JSON response
            # Handle potential markdown code blocks
            if '```json' in response_text:
                # Extract JSON from code block
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                # Extract from generic code block
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                response_text = response_text[json_start:json_end].strip()

            parsed = json.loads(response_text)

            # Map entities to facet filters
            facet_filters = {}
            entities_found = parsed.get('entities', [])

            for entity in entities_found:
                if entity['confidence'] < confidence_threshold:
                    continue

                entity_type = entity['type']
                entity_text = entity['text']

                # Map entity types to facet categories
                if entity_type == 'hardware':
                    if 'hardware' not in facet_filters:
                        facet_filters['hardware'] = []
                    facet_filters['hardware'].append(entity_text)

                elif entity_type in ['memory_address', 'register']:
                    if 'registers' not in facet_filters:
                        facet_filters['registers'] = []
                    facet_filters['registers'].append(entity_text)

                elif entity_type == 'instruction':
                    if 'instructions' not in facet_filters:
                        facet_filters['instructions'] = []
                    facet_filters['instructions'].append(entity_text)

            # Build result
            result = {
                'original_query': query,
                'search_terms': parsed.get('search_terms', []),
                'facet_filters': facet_filters,
                'search_mode': parsed.get('search_mode', 'hybrid'),
                'confidence': parsed.get('confidence', 0.7),
                'entities_found': entities_found,
                'reasoning': parsed.get('reasoning', ''),
                'suggested_query': ' '.join(parsed.get('search_terms', [])) or query  # Reformulated query for searching
            }

            elapsed = (time.time() - start_time) * 1000
            self.logger.info(f"Query translation completed in {elapsed:.2f}ms - mode: {result['search_mode']}, "
                           f"entities: {len(entities_found)}, confidence: {result['confidence']:.2f}")

            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}")
            self.logger.error(f"Raw response: {response_text}")

            # Fallback to basic keyword extraction
            words = query.lower().split()
            search_terms = [w for w in words if len(w) > 3]
            return {
                'original_query': query,
                'search_terms': search_terms,
                'facet_filters': {},
                'search_mode': 'keyword',
                'confidence': 0.5,
                'entities_found': [],
                'reasoning': f'LLM response parsing failed: {str(e)}',
                'suggested_query': ' '.join(search_terms) or query
            }

        except Exception as e:
            self.logger.error(f"Query translation error: {e}")

            # Fallback to basic keyword extraction
            words = query.lower().split()
            search_terms = [w for w in words if len(w) > 3]
            return {
                'original_query': query,
                'search_terms': search_terms,
                'facet_filters': {},
                'search_mode': 'keyword',
                'confidence': 0.5,
                'entities_found': [],
                'reasoning': f'Translation error: {str(e)}',
                'suggested_query': ' '.join(search_terms) or query
            }

    def faceted_search(self, query: str, facet_filters: Optional[dict[str, list[str]]] = None,
                      max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """
        Perform search with faceted filtering.

        Args:
            query: Search query
            facet_filters: Dict of facet_type -> list of values to filter by.
                          Example: {'hardware': ['SID', 'VIC-II'], 'instruction': ['LDA', 'STA']}
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of search results filtered by facets, with facets included
        """
        # Check cache first
        start_time = time.time()
        if self._faceted_cache is not None:
            # Convert facet_filters dict to tuple for hashable cache key
            facet_filters_key = None
            if facet_filters:
                facet_filters_key = tuple(sorted((k, tuple(sorted(v))) for k, v in facet_filters.items()))

            cache_key = self._cache_key('faceted_search',
                                       query=query,
                                       facet_filters=facet_filters_key,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            if cache_key in self._faceted_cache:
                results = self._faceted_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Faceted cache HIT for query: '{query}' ({len(results)} results, {elapsed_ms:.2f}ms)")
                return results

        self.logger.info(f"Faceted search query: '{query}' (facets={facet_filters}, max_results={max_results}, tags={tags})")

        # First get regular search results (request more for filtering)
        results = self.search(query, max_results * 3, tags)

        # If no facet filters specified, just return regular results with facets added
        if not facet_filters:
            # Add facets to each result
            for result in results:
                result['facets'] = self._get_document_facets(result['doc_id'])
            final_results = results[:max_results]
            elapsed = (time.time() - start_time) * 1000
            self.logger.info(f"Faceted search (no filters) completed: {len(final_results)} results in {elapsed:.2f}ms")

            # Store in cache
            if self._faceted_cache is not None:
                facet_filters_key = None
                cache_key = self._cache_key('faceted_search',
                                           query=query,
                                           facet_filters=facet_filters_key,
                                           max_results=max_results,
                                           tags=tuple(sorted(tags)) if tags else None)
                self._faceted_cache[cache_key] = final_results

            return final_results

        # Filter results by facets
        filtered_results = []
        for result in results:
            # Get document facets
            doc_facets = self._get_document_facets(result['doc_id'])

            # Check if document matches all facet filters
            matches = True
            for facet_type, required_values in facet_filters.items():
                doc_values = doc_facets.get(facet_type, set())
                # Document must have at least one of the required values for this facet type
                if not any(val in doc_values for val in required_values):
                    matches = False
                    break

            if matches:
                result['facets'] = doc_facets
                filtered_results.append(result)

                if len(filtered_results) >= max_results:
                    break

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Faceted search completed: {len(filtered_results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'faceted', len(filtered_results), elapsed, tags)

        # Store in cache
        if self._faceted_cache is not None:
            facet_filters_key = tuple(sorted((k, tuple(sorted(v))) for k, v in facet_filters.items()))
            cache_key = self._cache_key('faceted_search',
                                       query=query,
                                       facet_filters=facet_filters_key,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            self._faceted_cache[cache_key] = filtered_results

        return filtered_results

    def _get_document_facets(self, doc_id: str) -> dict[str, set[str]]:
        """Get all facets for a document from the database."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT facet_type, facet_value
            FROM document_facets
            WHERE doc_id = ?
        """, (doc_id,))

        facets = {'hardware': set(), 'instruction': set(), 'register': set()}
        for row in cursor.fetchall():
            facet_type, facet_value = row
            if facet_type in facets:
                facets[facet_type].add(facet_value)

        return facets

    def find_by_reference(self, ref_type: str, ref_value: str, max_results: int = 10) -> list[dict]:
        """
        Find documents by cross-reference type and value.

        Args:
            ref_type: Type of reference ('memory_address', 'register_offset', 'page_reference')
            ref_value: The reference value to search for (e.g., '$D020', 'VIC+0', '156')
            max_results: Maximum number of results to return

        Returns:
            List of results with document info, chunk info, and context
        """
        self.logger.info(f"Finding documents by reference: {ref_type}={ref_value}")
        start_time = time.time()

        cursor = self.db_conn.cursor()

        # Query cross_references table
        cursor.execute("""
            SELECT
                xr.doc_id,
                xr.chunk_id,
                xr.ref_type,
                xr.ref_value,
                xr.context,
                d.filename,
                d.title,
                d.tags
            FROM cross_references xr
            JOIN documents d ON xr.doc_id = d.doc_id
            WHERE xr.ref_type = ? AND xr.ref_value = ?
            ORDER BY d.title, xr.chunk_id
            LIMIT ?
        """, (ref_type, ref_value, max_results))

        results = []
        for row in cursor.fetchall():
            doc_id, chunk_id, ref_type, ref_value, context, filename, title, tags_json = row
            results.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'filename': filename,
                'title': title,
                'ref_type': ref_type,
                'ref_value': ref_value,
                'context': context,
                'tags': json.loads(tags_json) if tags_json else []
            })

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Found {len(results)} references in {elapsed:.2f}ms")

        return results

    def find_similar_documents(self, doc_id: str, chunk_id: Optional[int] = None,
                               max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """
        Find documents similar to the given document or chunk.

        Args:
            doc_id: Document ID to find similar documents for
            chunk_id: Optional chunk ID (if None, uses all chunks from document)
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by

        Returns:
            List of similar documents with similarity scores
        """
        # Check cache first
        if self._similar_cache is not None:
            cache_key = self._cache_key('find_similar',
                                       doc_id=doc_id,
                                       chunk_id=chunk_id,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            if cache_key in self._similar_cache:
                results = self._similar_cache[cache_key]
                self.logger.debug(f"Cache hit for find_similar: doc_id={doc_id}, chunk_id={chunk_id}")
                return results

        # Prefer semantic search if available
        if self.use_semantic and self.embeddings_index is not None:
            results = self._find_similar_semantic(doc_id, chunk_id, max_results, tags)
        else:
            # Fall back to TF-IDF similarity
            results = self._find_similar_tfidf(doc_id, chunk_id, max_results, tags)

        # Store in cache
        if self._similar_cache is not None:
            cache_key = self._cache_key('find_similar',
                                       doc_id=doc_id,
                                       chunk_id=chunk_id,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None)
            self._similar_cache[cache_key] = results

        return results

    def _find_similar_semantic(self, doc_id: str, chunk_id: Optional[int],
                               max_results: int, tags: Optional[list[str]]) -> list[dict]:
        """Find similar documents using semantic embeddings."""
        if not self.use_semantic or self.embeddings_model is None:
            raise RuntimeError("Semantic search not available")

        # Build embeddings index if not yet built
        if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
            self._build_embeddings()
            if self.embeddings_index is None:
                return []

        # Get target embedding(s)
        if chunk_id is not None:
            # Find specific chunk's embedding
            try:
                target_idx = self.embeddings_doc_map.index((doc_id, chunk_id))
                target_embedding = self.embeddings_index.reconstruct(target_idx)
                target_embedding = target_embedding.reshape(1, -1)
            except ValueError:
                self.logger.error(f"Chunk not found in embeddings: {doc_id}, {chunk_id}")
                return []
        else:
            # Average all chunk embeddings for this document
            doc_indices = [i for i, (d, c) in enumerate(self.embeddings_doc_map) if d == doc_id]
            if not doc_indices:
                self.logger.error(f"Document not found in embeddings: {doc_id}")
                return []

            embeddings = np.array([self.embeddings_index.reconstruct(i) for i in doc_indices])
            target_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(target_embedding)

        # Search for similar chunks (get more for filtering)
        k = min(max_results * 10 * self._passage_fanout(), len(self.embeddings_doc_map))
        scores, indices = self.embeddings_index.search(target_embedding, k)

        # Build results, aggregating by document
        doc_scores = {}  # doc_id -> max similarity score
        doc_chunks = {}  # doc_id -> list of matching chunks

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.embeddings_doc_map):
                continue

            found_doc_id, found_chunk_id = self.embeddings_doc_map[idx]

            # Skip the source document/chunk
            if found_doc_id == doc_id:
                if chunk_id is None or found_chunk_id == chunk_id:
                    continue

            found_doc = self.documents.get(found_doc_id)

            # Exclude superseded card versions - a retracted card shouldn't
            # surface as "similar" content either.
            if found_doc and found_doc.superseded_by:
                continue

            # Filter by tags if specified
            if tags:
                if found_doc and not any(t in found_doc.tags for t in tags):
                    continue

            # Track best score per document
            if found_doc_id not in doc_scores or score > doc_scores[found_doc_id]:
                doc_scores[found_doc_id] = float(score)
                doc_chunks[found_doc_id] = found_chunk_id

        # Sort documents by similarity score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]

        # Build result list with document info
        results = []
        for found_doc_id, similarity in sorted_docs:
            doc = self.documents.get(found_doc_id)
            if not doc:
                continue

            best_chunk_id = doc_chunks[found_doc_id]
            chunk = self.get_chunk(found_doc_id, best_chunk_id)

            results.append({
                'doc_id': found_doc_id,
                'filename': doc.filename,
                'title': doc.title,
                'chunk_id': best_chunk_id,
                'similarity': similarity,
                'snippet': chunk.content[:300] + "..." if chunk else "",
                'total_chunks': doc.total_chunks,
                'tags': doc.tags
            })

        return results

    def _find_similar_tfidf(self, doc_id: str, chunk_id: Optional[int],
                            max_results: int, tags: Optional[list[str]]) -> list[dict]:
        """Find similar documents using TF-IDF (fallback when semantic search unavailable)."""
        from collections import Counter
        import math

        # Get target document chunks
        target_chunks = self._get_chunks_db(doc_id)
        if not target_chunks:
            return []

        # If specific chunk requested, use only that chunk
        if chunk_id is not None:
            target_chunks = [c for c in target_chunks if c.chunk_id == chunk_id]
            if not target_chunks:
                return []

        # Build target document term vector
        target_terms = []
        for chunk in target_chunks:
            words = chunk.content.lower().split()
            target_terms.extend(words)

        target_tf = Counter(target_terms)
        target_length = math.sqrt(sum(count**2 for count in target_tf.values()))

        # Calculate similarity with all other documents
        doc_similarities = []

        for other_doc_id, other_doc in self.documents.items():
            # Skip source document
            if other_doc_id == doc_id:
                continue

            # Filter by tags
            if tags and not any(t in other_doc.tags for t in tags):
                continue

            # Get chunks for comparison document
            other_chunks = self._get_chunks_db(other_doc_id)
            if not other_chunks:
                continue

            # Build term vector for other document
            other_terms = []
            for chunk in other_chunks:
                words = chunk.content.lower().split()
                other_terms.extend(words)

            other_tf = Counter(other_terms)
            other_length = math.sqrt(sum(count**2 for count in other_tf.values()))

            # Calculate cosine similarity
            dot_product = sum(target_tf[term] * other_tf[term] for term in target_tf if term in other_tf)

            if target_length > 0 and other_length > 0:
                similarity = dot_product / (target_length * other_length)
            else:
                similarity = 0.0

            if similarity > 0:
                # Find best matching chunk
                best_chunk = other_chunks[0] if other_chunks else None

                doc_similarities.append({
                    'doc_id': other_doc_id,
                    'filename': other_doc.filename,
                    'title': other_doc.title,
                    'chunk_id': best_chunk.chunk_id if best_chunk else 0,
                    'similarity': similarity,
                    'snippet': best_chunk.content[:300] + "..." if best_chunk else "",
                    'total_chunks': other_doc.total_chunks,
                    'tags': other_doc.tags
                })

        # Sort by similarity and return top results
        doc_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return doc_similarities[:max_results]

    def compare_documents(self, doc_id_1: str, doc_id_2: str,
                         comparison_type: str = 'full') -> dict:
        """
        Compare two documents side-by-side with similarity scoring and diff analysis.

        Args:
            doc_id_1: First document ID
            doc_id_2: Second document ID
            comparison_type: Type of comparison ('full', 'metadata', 'content')

        Returns:
            Dictionary with comparison results:
            {
                'similarity_score': float,         # Cosine similarity 0.0-1.0
                'metadata_diff': {                 # Metadata differences
                    'title': [title1, title2],
                    'filename': [filename1, filename2],
                    'tags': {
                        'common': [...],
                        'only_in_doc1': [...],
                        'only_in_doc2': [...]
                    },
                    'file_type': [type1, type2],
                    'total_pages': [pages1, pages2]
                },
                'chunk_count': [count1, count2],   # Number of chunks in each doc
                'content_diff': [                  # Unified diff lines (limited)
                    'diff line 1',
                    'diff line 2',
                    ...
                ],
                'entity_comparison': {             # Entity comparison
                    'common_entities': [           # Entities in both docs
                        {'text': str, 'type': str},
                        ...
                    ],
                    'unique_to_doc1': [...],       # Entities only in doc1
                    'unique_to_doc2': [...],       # Entities only in doc2
                    'total_doc1': int,
                    'total_doc2': int
                },
                'summary': str                     # Human-readable summary
            }

        Raises:
            ValueError: If either document not found

        Examples:
            >>> kb = KnowledgeBase()
            >>> result = kb.compare_documents('doc1', 'doc2')
            >>> print(f"Similarity: {result['similarity_score']:.1%}")
            >>> print(f"Common tags: {result['metadata_diff']['tags']['common']}")
        """
        from collections import Counter
        import math
        import difflib

        # Validate both documents exist
        if doc_id_1 not in self.documents:
            raise ValueError(f"Document not found: {doc_id_1}")
        if doc_id_2 not in self.documents:
            raise ValueError(f"Document not found: {doc_id_2}")

        doc1 = self.documents[doc_id_1]
        doc2 = self.documents[doc_id_2]

        # 1. Calculate Cosine Similarity using TF-IDF
        similarity_score = 0.0
        if comparison_type in ['full', 'content']:
            # Get chunks for both documents
            chunks1 = self._get_chunks_db(doc_id_1)
            chunks2 = self._get_chunks_db(doc_id_2)

            if chunks1 and chunks2:
                # Build term vectors
                terms1 = []
                for chunk in chunks1:
                    words = chunk.content.lower().split()
                    terms1.extend(words)

                terms2 = []
                for chunk in chunks2:
                    words = chunk.content.lower().split()
                    terms2.extend(words)

                tf1 = Counter(terms1)
                tf2 = Counter(terms2)

                length1 = math.sqrt(sum(count**2 for count in tf1.values()))
                length2 = math.sqrt(sum(count**2 for count in tf2.values()))

                # Calculate cosine similarity
                dot_product = sum(tf1[term] * tf2[term] for term in tf1 if term in tf2)

                if length1 > 0 and length2 > 0:
                    similarity_score = dot_product / (length1 * length2)

        # 2. Metadata Comparison
        metadata_diff = {
            'title': [doc1.title, doc2.title],
            'filename': [doc1.filename, doc2.filename],
            'file_type': [doc1.file_type, doc2.file_type],
            'total_pages': [doc1.total_pages, doc2.total_pages],
            'tags': {
                'common': list(set(doc1.tags) & set(doc2.tags)),
                'only_in_doc1': list(set(doc1.tags) - set(doc2.tags)),
                'only_in_doc2': list(set(doc2.tags) - set(doc1.tags))
            }
        }

        # 3. Chunk Count
        chunk_count = [doc1.total_chunks, doc2.total_chunks]

        # 4. Content Diff (limited to avoid huge output)
        content_diff = []
        if comparison_type in ['full', 'content']:
            # Get first few chunks for comparison
            chunks1 = self._get_chunks_db(doc_id_1)
            chunks2 = self._get_chunks_db(doc_id_2)

            if chunks1 and chunks2:
                # Concatenate first 5 chunks from each doc
                text1_sample = '\n\n'.join(c.content for c in chunks1[:5])
                text2_sample = '\n\n'.join(c.content for c in chunks2[:5])

                # Generate unified diff (limit to first 100 lines)
                diff_lines = list(difflib.unified_diff(
                    text1_sample.splitlines(keepends=True),
                    text2_sample.splitlines(keepends=True),
                    fromfile=doc1.filename,
                    tofile=doc2.filename,
                    lineterm=''
                ))
                content_diff = diff_lines[:100]  # Limit to 100 lines

        # 5. Entity Comparison
        entity_comparison = {
            'common_entities': [],
            'unique_to_doc1': [],
            'unique_to_doc2': [],
            'total_doc1': 0,
            'total_doc2': 0
        }

        if comparison_type in ['full', 'metadata']:
            cursor = self.db_conn.cursor()

            # Get entities for doc1
            cursor.execute("""
                SELECT entity_text, entity_type
                FROM document_entities
                WHERE doc_id = ?
            """, (doc_id_1,))
            entities1 = {(row[0], row[1]) for row in cursor.fetchall()}
            entity_comparison['total_doc1'] = len(entities1)

            # Get entities for doc2
            cursor.execute("""
                SELECT entity_text, entity_type
                FROM document_entities
                WHERE doc_id = ?
            """, (doc_id_2,))
            entities2 = {(row[0], row[1]) for row in cursor.fetchall()}
            entity_comparison['total_doc2'] = len(entities2)

            # Find common and unique entities
            common = entities1 & entities2
            unique1 = entities1 - entities2
            unique2 = entities2 - entities1

            entity_comparison['common_entities'] = [
                {'text': e[0], 'type': e[1]} for e in sorted(common)
            ]
            entity_comparison['unique_to_doc1'] = [
                {'text': e[0], 'type': e[1]} for e in sorted(unique1)
            ]
            entity_comparison['unique_to_doc2'] = [
                {'text': e[0], 'type': e[1]} for e in sorted(unique2)
            ]

        # 6. Generate Summary
        summary_parts = []
        summary_parts.append(f"Similarity: {similarity_score:.1%}")

        if metadata_diff['title'][0] == metadata_diff['title'][1]:
            summary_parts.append("Same title")
        else:
            summary_parts.append("Different titles")

        common_tags = len(metadata_diff['tags']['common'])
        if common_tags > 0:
            summary_parts.append(f"{common_tags} common tag(s)")

        if entity_comparison['total_doc1'] > 0 and entity_comparison['total_doc2'] > 0:
            common_ent = len(entity_comparison['common_entities'])
            summary_parts.append(f"{common_ent} common entit{'y' if common_ent == 1 else 'ies'}")

        summary = " | ".join(summary_parts)

        return {
            'similarity_score': round(similarity_score, 4),
            'metadata_diff': metadata_diff,
            'chunk_count': chunk_count,
            'content_diff': content_diff,
            'entity_comparison': entity_comparison,
            'summary': summary
        }

    def search_tables(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None) -> list[dict]:
        """Search for tables in documents using FTS5.

        Returns a list of table dictionaries with structure:
        {
            'doc_id': str,
            'doc_title': str,
            'table_id': int,
            'page': int,
            'markdown': str,
            'row_count': int,
            'col_count': int,
            'score': float
        }
        """
        cursor = self.db_conn.cursor()

        # Check if tables_fts exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='tables_fts'
        """)
        if not cursor.fetchone():
            self.logger.warning("tables_fts index not found, no tables to search")
            return []

        # Build FTS5 query
        # Escape special characters in query
        fts_query = query.replace('"', '""')
        fts_query = f'"{fts_query}"' if ' ' in fts_query else fts_query

        # Search tables_fts
        sql = """
            SELECT t.doc_id, d.title, t.table_id, t.page, t.markdown, t.row_count, t.col_count,
                   tables_fts.rank as score
            FROM tables_fts
            JOIN document_tables t ON tables_fts.doc_id = t.doc_id AND tables_fts.table_id = t.table_id
            JOIN documents d ON t.doc_id = d.doc_id
            WHERE tables_fts MATCH ?
        """

        # Add tag filtering if specified
        if tags:
            tag_conditions = " OR ".join(["d.tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_conditions})"

        sql += " ORDER BY score DESC LIMIT ?"

        # Execute query
        params = [fts_query]
        if tags:
            params.extend([f'%"{tag}"%' for tag in tags])
        params.append(max_results)

        cursor.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'doc_title': row[1],
                'table_id': row[2],
                'page': row[3],
                'markdown': row[4],
                'row_count': row[5],
                'col_count': row[6],
                'score': abs(row[7])  # FTS5 rank is negative, take absolute value
            })

        self.logger.info(f"Table search for '{query}' returned {len(results)} results")
        return results

    def search_code(self, query: str, max_results: int = 5, block_type: Optional[str] = None,
                    tags: Optional[list[str]] = None) -> list[dict]:
        """Search for code blocks in documents using FTS5.

        Args:
            query: Search query
            max_results: Maximum number of results
            block_type: Filter by code type ('basic', 'assembly', 'hex')
            tags: Filter by document tags

        Returns a list of code block dictionaries with structure:
        {
            'doc_id': str,
            'doc_title': str,
            'block_id': int,
            'page': int,
            'block_type': str,
            'code': str,
            'line_count': int,
            'score': float
        }
        """
        cursor = self.db_conn.cursor()

        # Check if code_fts exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='code_fts'
        """)
        if not cursor.fetchone():
            self.logger.warning("code_fts index not found, no code blocks to search")
            return []

        # Build FTS5 query
        # Escape special characters in query
        fts_query = query.replace('"', '""')
        fts_query = f'"{fts_query}"' if ' ' in fts_query else fts_query

        # Search code_fts
        sql = """
            SELECT c.doc_id, d.title, c.block_id, c.page, c.block_type, c.code, c.line_count,
                   code_fts.rank as score
            FROM code_fts
            JOIN document_code_blocks c ON code_fts.doc_id = c.doc_id AND code_fts.block_id = c.block_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE code_fts MATCH ?
        """

        # Add block type filtering if specified
        if block_type:
            sql += " AND c.block_type = ?"

        # Add tag filtering if specified
        if tags:
            tag_conditions = " OR ".join(["d.tags LIKE ?" for _ in tags])
            sql += f" AND ({tag_conditions})"

        sql += " ORDER BY score DESC LIMIT ?"

        # Execute query
        params = [fts_query]
        if block_type:
            params.append(block_type)
        if tags:
            params.extend([f'%"{tag}"%' for tag in tags])
        params.append(max_results)

        cursor.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            results.append({
                'doc_id': row[0],
                'doc_title': row[1],
                'block_id': row[2],
                'page': row[3],
                'block_type': row[4],
                'code': row[5],
                'line_count': row[6],
                'score': abs(row[7])  # FTS5 rank is negative, take absolute value
            })

        self.logger.info(f"Code search for '{query}' returned {len(results)} results (type={block_type or 'all'})")
        return results
