"""Lexical retrieval (BM25/FTS5/simple), suggestions, export, plus embeddings and reranking for SearchMixin.

Split out of kb/search.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from features import BM25Okapi
from features import BM25_SUPPORT
from features import CrossEncoder
from features import FUZZY_SUPPORT
from features import SentenceTransformer
from features import _ensure_nltk
from features import faiss
from features import fuzz
from models import DocumentChunk
from text_utils import _content_terms
from text_utils import _filter_snippet_terms
from typing import Optional
from util import _atomic_write_bytes
from util import _cross_process_lock
from util import _network_timeout
import hashlib
import json
import os
import re
import time


class _RetrievalMixin:

    def _build_bm25_index(self):
        """Build BM25 index from chunks for fast searching (lazy loading from database)."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()
        load_time = 0.0

        if not BM25_SUPPORT:
            self.logger.info("BM25 index not built (no support)")
            return

        # Lazy load all chunks from database if not already in memory
        if not self.chunks:
            self.logger.info("Loading all chunks from database for BM25 index")
            load_start = time.time()
            self.chunks = self._get_chunks_db()
            load_time = time.time() - load_start
            self.logger.info(f"Loaded {len(self.chunks)} chunks in {load_time:.2f}s")

        if not self.chunks:
            self.logger.info("BM25 index not built (no chunks)")
            return

        # Tokenize all chunk content with preprocessing if enabled
        # Use parallel processing for faster tokenization
        self.logger.info(f"Tokenizing {len(self.chunks)} chunks...")
        tokenize_start = time.time()

        if self.use_preprocessing and len(self.chunks) > 100:
            # Parallel tokenization for large datasets
            tokenized_corpus = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tokenization tasks
                future_to_idx = {
                    executor.submit(self._preprocess_text, chunk.content): idx
                    for idx, chunk in enumerate(self.chunks)
                }

                # Collect results in order
                results = [None] * len(self.chunks)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()

                tokenized_corpus = results
        else:
            # Sequential for small datasets or no preprocessing
            tokenized_corpus = [self._preprocess_text(chunk.content) for chunk in self.chunks]

        tokenize_time = time.time() - tokenize_start
        self.logger.info(f"Tokenization completed in {tokenize_time:.2f}s")

        # Build BM25 index
        index_start = time.time()
        self.bm25 = BM25Okapi(tokenized_corpus)
        index_time = time.time() - index_start

        total_time = time.time() - start_time
        preprocessing_status = "with preprocessing" if self.use_preprocessing else "without preprocessing"
        self.logger.info(f"Built BM25 index with {len(self.chunks)} chunks ({preprocessing_status}) - Total: {total_time:.2f}s (load: {load_time:.2f}s, tokenize: {tokenize_time:.2f}s, index: {index_time:.2f}s)")

    def _fuzzy_match_terms(self, query_terms: list[str], content: str) -> tuple[bool, float]:
        """
        Check if query terms fuzzy match content using rapidfuzz.

        Args:
            query_terms: List of query terms to match
            content: Content to search in

        Returns:
            Tuple of (match_found, average_similarity_score)
        """
        if not self.use_fuzzy:
            # Fuzzy search disabled, do exact matching
            content_lower = content.lower()
            matches = sum(1 for term in query_terms if term.lower() in content_lower)
            return (matches > 0, matches / len(query_terms) if query_terms else 0.0)

        # Split content into words for fuzzy matching
        content_words = content.lower().split()

        match_scores = []
        for query_term in query_terms:
            query_term_lower = query_term.lower()

            # Check for exact match first (fastest)
            if query_term_lower in content.lower():
                match_scores.append(100.0)
                continue

            # Try fuzzy matching against content words
            best_score = 0.0
            for content_word in content_words:
                score = fuzz.ratio(query_term_lower, content_word)
                if score > best_score:
                    best_score = score
                if score >= self.fuzzy_threshold:
                    break  # Found a good enough match

            match_scores.append(best_score)

        # Consider it a match if at least one term meets the threshold
        match_found = any(score >= self.fuzzy_threshold for score in match_scores)
        avg_score = sum(match_scores) / len(match_scores) if match_scores else 0.0

        return (match_found, avg_score)

    def _preprocess_text(self, text: str) -> list[str]:
        """Preprocess text for searching: tokenize, lowercase, remove stopwords, stem.

        Args:
            text: The text to preprocess

        Returns:
            List of processed tokens
        """
        if not self.use_preprocessing:
            # No preprocessing - just lowercase and split
            return text.lower().split()

        # Build the stemmer/stopwords on first use (deferred from __init__ so
        # that importing nltk does not delay the MCP handshake).
        if not self._preprocessing_ready:
            with self._lock:
                if not self._preprocessing_ready:
                    try:
                        nltk_parts = _ensure_nltk()
                        if nltk_parts is None:
                            raise ImportError("nltk unavailable")
                        PorterStemmer, stopwords, _ = nltk_parts
                        self.stemmer = PorterStemmer()
                        self.stop_words = set(stopwords.words('english'))
                    except Exception as e:
                        # Degrade to no preprocessing rather than failing the search
                        self.logger.warning(f"Query preprocessing unavailable, disabling: {e}")
                        self.use_preprocessing = False
                        self.stemmer = None
                        self.stop_words = set()
                    self._preprocessing_ready = True
            if not self.use_preprocessing:
                return text.lower().split()

        # Tokenize and lowercase
        try:
            word_tokenize = _ensure_nltk()[2]
            tokens = word_tokenize(text.lower())
        except Exception:
            # Fallback if tokenization fails
            tokens = text.lower().split()

        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            # Keep alphanumeric tokens and hyphenated words (like VIC-II, 6502)
            # Remove pure punctuation tokens
            if token.isalnum() or ('-' in token and any(c.isalnum() for c in token)):
                # Remove stopwords (but keep technical terms with hyphens)
                if token not in self.stop_words:
                    # Apply stemming only to pure alphanumeric tokens
                    # Don't stem technical terms with hyphens/numbers
                    if self.stemmer and token.isalpha():
                        stemmed = self.stemmer.stem(token)
                        processed_tokens.append(stemmed)
                    else:
                        processed_tokens.append(token)

        return processed_tokens

    def build_suggestion_dictionary(self, rebuild: bool = False):
        """
        Build autocomplete suggestion dictionary from all documents.

        Args:
            rebuild: If True, clear existing suggestions and rebuild from scratch
        """
        self.logger.info("Building query suggestion dictionary...")
        start_time = time.time()

        cursor = self.db_conn.cursor()

        # Clear existing if rebuilding
        if rebuild:
            cursor.execute("DELETE FROM query_suggestions")
            self.db_conn.commit()

        # Extract terms from all chunks
        from collections import defaultdict
        terms = defaultdict(int)

        chunks = self._get_chunks_db()
        for chunk in chunks:
            text = chunk.content

            # Extract technical terms (ALL CAPS, 2+ chars)
            tech_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)?\b', text)  # VIC-II, SID, CIA
            for term in tech_terms:
                terms[(term, 'hardware')] += 1

            # Extract memory addresses
            addresses = re.findall(r'\$[0-9A-Fa-f]{4}', text)
            for addr in addresses:
                terms[(addr.upper(), 'register')] += 1

            # Extract 6502 instructions
            instructions = re.findall(
                r'\b(?:LDA|STA|LDX|STX|LDY|STY|TAX|TAY|TXA|TYA|TSX|TXS|'
                r'ADC|SBC|AND|ORA|EOR|INC|DEC|INX|INY|DEX|DEY|'
                r'CMP|CPX|CPY|ASL|LSR|ROL|ROR|BIT|NOP|'
                r'JMP|JSR|RTS|RTI|BEQ|BNE|BCC|BCS|BMI|BPL|BVC|BVS|'
                r'CLC|SEC|CLI|SEI|CLD|SED|CLV|PHA|PLA|PHP|PLP)\b',
                text, re.IGNORECASE
            )
            for instr in instructions:
                terms[(instr.upper(), 'instruction')] += 1

            # Extract common technical phrases (2-3 words)
            # Look for capitalized phrases like "Sprite Multiplexing", "Sound Interface Device"
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            for phrase in phrases:
                if len(phrase) > 5:  # Avoid very short phrases
                    terms[(phrase, 'concept')] += 1

        # Store top N terms (limit to avoid bloat)
        top_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:2000]

        for (term, category), freq in top_terms:
            cursor.execute("""
                INSERT INTO query_suggestions (term, frequency, category)
                VALUES (?, ?, ?)
            """, (term, freq, category))

        self.db_conn.commit()

        elapsed = time.time() - start_time
        self.logger.info(f"Built suggestion dictionary with {len(top_terms)} terms in {elapsed:.2f}s")

    def get_query_suggestions(self, partial: str, max_suggestions: int = 5,
                             category: Optional[str] = None) -> list[dict]:
        """
        Get autocomplete suggestions for partial query.

        Args:
            partial: Partial query string (e.g., "VIC")
            max_suggestions: Maximum number of suggestions to return
            category: Optional category filter ('hardware', 'register', 'instruction', 'concept')

        Returns:
            List of suggestion dicts with 'term', 'frequency', and 'category'
        """
        if not partial or len(partial) < 2:
            return []

        cursor = self.db_conn.cursor()

        # Escape special FTS5 characters by quoting the query
        # FTS5 special chars: $ * " and others need to be quoted
        escaped_partial = f'"{partial}"*'

        # Use FTS5 prefix matching
        if category:
            cursor.execute("""
                SELECT term, frequency, category
                FROM query_suggestions
                WHERE term MATCH ? AND category = ?
                ORDER BY rank, frequency DESC
                LIMIT ?
            """, (escaped_partial, category, max_suggestions))
        else:
            cursor.execute("""
                SELECT term, frequency, category
                FROM query_suggestions
                WHERE term MATCH ?
                ORDER BY rank, frequency DESC
                LIMIT ?
            """, (escaped_partial, max_suggestions))

        results = []
        for row in cursor.fetchall():
            results.append({
                'term': row[0],
                'frequency': row[1],
                'category': row[2]
            })

        return results

    def _update_suggestions_for_chunks(self, chunks: list[DocumentChunk]):
        """
        Incrementally update query suggestions with terms from new chunks.

        Args:
            chunks: List of newly added chunks
        """
        from collections import defaultdict
        terms = defaultdict(int)

        # Extract terms from new chunks
        for chunk in chunks:
            text = chunk.content

            # Extract technical terms (ALL CAPS, 2+ chars)
            tech_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)?\b', text)
            for term in tech_terms:
                terms[(term, 'hardware')] += 1

            # Extract memory addresses
            addresses = re.findall(r'\$[0-9A-Fa-f]{4}', text)
            for addr in addresses:
                terms[(addr.upper(), 'register')] += 1

            # Extract 6502 instructions
            instructions = re.findall(
                r'\b(?:LDA|STA|LDX|STX|LDY|STY|TAX|TAY|TXA|TYA|TSX|TXS|'
                r'ADC|SBC|AND|ORA|EOR|INC|DEC|INX|INY|DEX|DEY|'
                r'CMP|CPX|CPY|ASL|LSR|ROL|ROR|BIT|NOP|'
                r'JMP|JSR|RTS|RTI|BEQ|BNE|BCC|BCS|BMI|BPL|BVC|BVS|'
                r'CLC|SEC|CLI|SEI|CLD|SED|CLV|PHA|PLA|PHP|PLP)\b',
                text, re.IGNORECASE
            )
            for instr in instructions:
                terms[(instr.upper(), 'instruction')] += 1

            # Extract common technical phrases (2-3 words)
            phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
            for phrase in phrases:
                if len(phrase) > 5:
                    terms[(phrase, 'concept')] += 1

        if not terms:
            return

        cursor = self.db_conn.cursor()

        # Update or insert terms (upsert logic)
        for (term, category), freq in terms.items():
            # Check if term already exists
            cursor.execute("""
                SELECT frequency FROM query_suggestions
                WHERE term = ? AND category = ?
            """, (term, category))

            existing = cursor.fetchone()
            if existing:
                # Update frequency
                new_freq = existing[0] + freq
                cursor.execute("""
                    DELETE FROM query_suggestions WHERE term = ? AND category = ?
                """, (term, category))
                cursor.execute("""
                    INSERT INTO query_suggestions (term, frequency, category)
                    VALUES (?, ?, ?)
                """, (term, new_freq, category))
            else:
                # Insert new term
                cursor.execute("""
                    INSERT INTO query_suggestions (term, frequency, category)
                    VALUES (?, ?, ?)
                """, (term, freq, category))

        self.db_conn.commit()
        self.logger.debug(f"Updated suggestion dictionary with {len(terms)} terms")

    def export_search_results(self, results: list[dict], format: str = 'markdown',
                             query: Optional[str] = None) -> str:
        """
        Export search results to various formats.

        Args:
            results: List of search result dicts
            format: Output format ('markdown', 'json', 'html')
            query: Optional query string to include in export

        Returns:
            Formatted string in requested format
        """
        if format == 'markdown':
            return self._export_markdown(results, query)
        elif format == 'json':
            return self._export_json(results, query)
        elif format == 'html':
            return self._export_html(results, query)
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'markdown', 'json', or 'html'.")

    def _export_markdown(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as Markdown."""
        output = "# Search Results\n\n"

        if query:
            output += f"**Query:** {query}\n"
        output += f"**Results:** {len(results)}\n"
        output += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        output += "---\n\n"

        for i, result in enumerate(results, 1):
            output += f"## {i}. {result.get('title', 'Untitled')}\n\n"

            # Add score if present
            if 'score' in result:
                output += f"**Score:** {result['score']:.3f}\n"
            elif 'similarity' in result:
                output += f"**Similarity:** {result['similarity']:.3f}\n"

            # Add metadata
            if 'filename' in result:
                output += f"**File:** {result['filename']}\n"
            if 'page' in result and result['page']:
                output += f"**Page:** {result['page']}\n"
            if 'doc_id' in result:
                output += f"**Doc ID:** {result['doc_id']}\n"
            if 'chunk_id' in result:
                output += f"**Chunk:** {result['chunk_id']}\n"

            output += "\n"

            # Add snippet/content
            if 'snippet' in result:
                output += f"### Excerpt\n\n{result['snippet']}\n\n"
            elif 'context' in result:
                output += f"### Context\n\n{result['context']}\n\n"

            # Add tags if present
            if 'tags' in result and result['tags']:
                tags = result['tags'] if isinstance(result['tags'], list) else []
                if tags:
                    output += f"**Tags:** {', '.join(tags)}\n\n"

            output += "---\n\n"

        return output

    def _export_json(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as JSON."""
        export_data = {
            'query': query,
            'result_count': len(results),
            'generated_at': datetime.now().isoformat(),
            'results': results
        }
        return json.dumps(export_data, indent=2)

    def _export_html(self, results: list[dict], query: Optional[str] = None) -> str:
        """Export results as HTML."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .meta { color: #7f8c8d; margin-bottom: 20px; }
        .result {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .result h2 { color: #2c3e50; margin-top: 0; }
        .result-meta { color: #7f8c8d; font-size: 0.9em; margin: 10px 0; }
        .snippet {
            background: white;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .tags { margin-top: 10px; }
        .tag {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <h1>🔍 Search Results</h1>
"""

        if query:
            html += f"    <div class='meta'><strong>Query:</strong> {query}</div>\n"
        html += f"    <div class='meta'><strong>Results:</strong> {len(results)}</div>\n"
        html += f"    <div class='meta'><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>\n\n"

        for i, result in enumerate(results, 1):
            html += "    <div class='result'>\n"
            html += f"        <h2>{i}. {result.get('title', 'Untitled')}</h2>\n"

            html += "        <div class='result-meta'>\n"
            if 'score' in result:
                html += f"            <strong>Score:</strong> {result['score']:.3f}<br>\n"
            elif 'similarity' in result:
                html += f"            <strong>Similarity:</strong> {result['similarity']:.3f}<br>\n"
            if 'filename' in result:
                html += f"            <strong>File:</strong> {result['filename']}<br>\n"
            if 'page' in result and result['page']:
                html += f"            <strong>Page:</strong> {result['page']}<br>\n"
            html += "        </div>\n"

            if 'snippet' in result:
                html += f"        <div class='snippet'>{result['snippet']}</div>\n"
            elif 'context' in result:
                html += f"        <div class='snippet'>{result['context']}</div>\n"

            if 'tags' in result and result['tags']:
                tags = result['tags'] if isinstance(result['tags'], list) else []
                if tags:
                    html += "        <div class='tags'>\n"
                    for tag in tags:
                        html += f"            <span class='tag'>{tag}</span>\n"
                    html += "        </div>\n"

            html += "    </div>\n\n"

        html += """</body>
</html>"""

        return html

    def search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None,
              include_superseded: bool = False) -> list[dict]:
        """Search the knowledge base using BM25 ranking or simple term frequency.

        Superseded card versions are excluded by default so retracted claims
        don't keep answering searches; pass include_superseded=True to see them.
        """
        self._sync_documents_if_needed()
        start_time = time.time()

        # Check cache first
        if self._search_cache is not None:
            cache_key = self._cache_key('search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       include_superseded=include_superseded)
            if cache_key in self._search_cache:
                results = self._search_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Cache hit for query: '{query}' ({len(results)} results, {elapsed_ms:.2f}ms)")
                return results

        self.logger.info(f"Search query: '{query}' (max_results={max_results}, tags={tags})")

        # Extract phrase queries (text in quotes)
        phrase_pattern = r'"([^"]*)"'
        phrases = re.findall(phrase_pattern, query)
        # Remove phrases from query to get regular terms
        query_without_phrases = re.sub(phrase_pattern, '', query)

        # Preprocess query terms (tokenize, remove stopwords, stem)
        query_terms_list = self._preprocess_text(query_without_phrases)
        query_terms = set(query_terms_list)
        query_terms = {term for term in query_terms if term}  # Remove empty strings

        # Check search backend preference (in priority order)
        # FTS5 can be enabled with environment variable: USE_FTS5=1
        # BM25 can be disabled with environment variable: USE_BM25=0
        use_fts5 = os.environ.get('USE_FTS5', '0') == '1'
        use_bm25 = os.environ.get('USE_BM25', '1') == '1'

        if use_fts5 and self._fts5_available():
            # Use SQLite FTS5 full-text search
            results = self._search_fts5(query, query_terms, phrases, tags, max_results, include_superseded)
            if results is None:
                # A backend failure, not an empty result set. Collapsing the
                # two is what let a one-character syntax bug divert every
                # question-shaped query into a ~265s BM25 index build.
                self.logger.warning(
                    f"FTS5 backend failed for query {query!r}; falling back to BM25/simple")
                results = []
            # Fall back to BM25/simple if FTS5 returns no results
            if not results:
                if use_bm25 and BM25_SUPPORT:
                    if self.bm25 is None:
                        self._build_bm25_index()
                    if self.bm25 is not None:
                        results = self._search_bm25(query, query_terms, phrases, tags, max_results, include_superseded)
                    else:
                        results = self._search_simple(query_terms, phrases, tags, max_results, include_superseded)
                else:
                    results = self._search_simple(query_terms, phrases, tags, max_results, include_superseded)
        elif use_bm25 and BM25_SUPPORT:
            # Build BM25 index if not already built
            if self.bm25 is None:
                self._build_bm25_index()

            if self.bm25 is not None:
                results = self._search_bm25(query, query_terms, phrases, tags, max_results, include_superseded)
            else:
                results = self._search_simple(query_terms, phrases, tags, max_results, include_superseded)
        else:
            results = self._search_simple(query_terms, phrases, tags, max_results, include_superseded)

        # Store in cache
        if self._search_cache is not None:
            cache_key = self._cache_key('search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       include_superseded=include_superseded)
            self._search_cache[cache_key] = results

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Search completed: {len(results)} results in {elapsed_ms:.2f}ms")

        # Log search for analytics
        search_mode = 'fts5' if (use_fts5 and self._fts5_available() and results) else ('bm25' if use_bm25 else 'simple')
        self._log_search(query, search_mode, len(results), elapsed_ms, tags)

        return results

    def _search_bm25(self, query: str, query_terms: set, phrases: list, tags: Optional[list[str]], max_results: int,
                     include_superseded: bool = False) -> list[dict]:
        """Search using BM25 algorithm."""
        # Preprocess query for BM25
        tokenized_query = self._preprocess_text(query)

        # Get BM25 scores for all chunks
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Build results with scores
        results = []
        for idx, chunk in enumerate(self.chunks):
            doc = self.documents.get(chunk.doc_id)

            # Orphaned chunk (parent doc removed) - never searchable, deleted
            # is not the same as superseded.
            if doc is None:
                continue

            # Exclude superseded card versions by default
            if not include_superseded and doc.superseded_by:
                continue

            # Filter by tags if specified
            if tags:
                if doc and not any(t in doc.tags for t in tags):
                    continue

            score = bm25_scores[idx]

            # Boost score for phrase matches
            if phrases:
                content_lower = chunk.content.lower()
                for phrase in phrases:
                    if phrase.lower() in content_lower:
                        score *= 2  # 2x boost for phrase match

            # Combine query_terms and phrases for snippet extraction
            all_terms = _filter_snippet_terms(query_terms | {p.lower() for p in phrases})
            snippet = self._extract_snippet(chunk.content, all_terms)
            results.append({
                'doc_id': chunk.doc_id,
                'filename': chunk.filename,
                'title': chunk.title,
                'chunk_id': chunk.chunk_id,
                'score': float(score),
                'snippet': snippet,
                'word_count': chunk.word_count
            })

        # Sort by score and return top results
        # BM25 scores can be negative for very small documents
        # Accept any non-zero score (positive or moderately negative)
        # Filter only exact zeros (true non-matches)
        results = [r for r in results if abs(r['score']) > 0.0001]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def _search_simple(self, query_terms: set, phrases: list, tags: Optional[list[str]], max_results: int,
                       include_superseded: bool = False) -> list[dict]:
        """Simple term frequency search with fuzzy matching support (fallback when BM25 not available)."""
        results = []

        for chunk in self.chunks:
            doc = self.documents.get(chunk.doc_id)

            # Orphaned chunk (parent doc removed) - never searchable, deleted
            # is not the same as superseded.
            if doc is None:
                continue

            # Exclude superseded card versions by default
            if not include_superseded and doc.superseded_by:
                continue

            # Filter by tags if specified
            if tags:
                if not any(t in doc.tags for t in tags):
                    continue

            content_lower = chunk.content.lower()

            # Score based on term frequency (exact matches)
            score = 0
            for term in query_terms:
                # Exact word match (higher score)
                score += len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower)) * 2
                # Partial match
                score += content_lower.count(term)

            # If fuzzy search is enabled and no exact matches, try fuzzy matching
            if self.use_fuzzy and score == 0:
                match_found, fuzzy_score = self._fuzzy_match_terms(list(query_terms), chunk.content)
                if match_found:
                    # Use fuzzy score (scaled down since it's less reliable than exact match)
                    score = fuzzy_score / 10.0

            # Boost score for phrase matches
            for phrase in phrases:
                if phrase.lower() in content_lower:
                    score += len(phrase.split()) * 10  # High boost for phrase match

            if score > 0:
                # Combine query_terms and phrases for snippet extraction
                all_terms = _filter_snippet_terms(query_terms | {p.lower() for p in phrases})
                snippet = self._extract_snippet(chunk.content, all_terms)
                results.append({
                    'doc_id': chunk.doc_id,
                    'filename': chunk.filename,
                    'title': chunk.title,
                    'chunk_id': chunk.chunk_id,
                    'score': score,
                    'snippet': snippet,
                    'word_count': chunk.word_count
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def _fts5_match_expressions(self, query: str, phrases: list) -> list[str]:
        """Build safe FTS5 MATCH expressions for user text, most precise first.

        The old builder passed the raw query string through with only hyphen
        and colon words quoted. A query is user prose, not FTS5 syntax: a
        trailing '?' - which is to say every natural-language question - is a
        syntax error in operator position. FTS5 raised, the caller read the
        empty return as "no matches", and fell through to BM25, whose index
        costs ~265s to build on first use and ~4s per query after. FTS5 was
        effectively dead for question-shaped input.

        Returns the AND form first (precise) and the OR form second (recall).
        Both execute in ~0ms, so trying both in turn is still orders of
        magnitude cheaper than one BM25 build.
        """
        def quote(text: str) -> str:
            # FTS5 string literal: an embedded double quote is escaped by
            # doubling it. Quoting also stops a bare token like 'AND' or 'OR'
            # in the user's prose being read as an operator.
            return '"%s"' % text.replace('"', '""')

        tokens = [quote(phrase) for phrase in phrases if phrase.strip()]

        # Drop the quoted phrases before tokenising, so their words are not
        # also required individually on top of the phrase match.
        remainder = re.sub(r'"[^"]*"', ' ', query)
        tokens += [quote(word) for word in self._FTS5_TOKEN_RE.findall(remainder)]

        if not tokens:
            return []
        if len(tokens) == 1:
            return tokens
        return [' AND '.join(tokens), ' OR '.join(tokens)]

    def _search_fts5(self, query: str, query_terms: set, phrases: list,
                     tags: Optional[list[str]], max_results: int,
                     include_superseded: bool = False) -> Optional[list[dict]]:
        """Search using SQLite FTS5 full-text search.

        Returns [] when the query genuinely matches nothing, and None when the
        FTS5 backend itself failed. The caller needs to tell those apart: the
        old code returned [] for both, so a malformed MATCH expression was
        indistinguishable from an empty corpus hit and silently triggered the
        expensive BM25 fallback.
        """
        cursor = self.db_conn.cursor()

        expressions = self._fts5_match_expressions(query, phrases)
        if not expressions:
            # Nothing searchable in the input (punctuation only) - that is an
            # empty result, not a backend failure.
            return []

        try:
            # Execute FTS5 search with BM25 ranking, narrowest expression first
            rows = []
            for expression in expressions:
                cursor.execute("""
                SELECT
                    c.doc_id,
                    c.chunk_id,
                    c.content,
                    c.word_count,
                    c.page,
                    d.filename,
                    d.title,
                    fts.rank as score
                FROM chunks_fts5 fts
                JOIN chunks c ON c.rowid = fts.rowid
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE chunks_fts5 MATCH ?
                ORDER BY rank
                LIMIT ?
                """, (expression, max_results * 2))  # Get 2x for tag filtering
                rows = cursor.fetchall()
                if rows:
                    break

            results = []
            for row in rows:
                doc_id, chunk_id, content, word_count, page, filename, title, score = row
                doc = self.documents.get(doc_id)

                # Orphaned chunk (parent doc removed) - never searchable,
                # deleted is not the same as superseded.
                if doc is None:
                    continue

                # Exclude superseded card versions by default
                if not include_superseded and doc.superseded_by:
                    continue

                # Filter by tags if specified
                if tags:
                    if not any(t in doc.tags for t in tags):
                        continue

                # Extract snippet with highlighting
                snippet = self._extract_snippet(
                    content, _filter_snippet_terms(query_terms | {p.lower() for p in phrases}))

                results.append({
                    'doc_id': doc_id,
                    'filename': filename,
                    'title': title,
                    'chunk_id': chunk_id,
                    'score': abs(score),  # FTS5 returns negative scores (lower is better)
                    'snippet': snippet,
                    'word_count': word_count
                })

                if len(results) >= max_results:
                    break

            return results

        except Exception:
            self.logger.exception(f"FTS5 search failed for query {query!r}")
            return None

    def _extract_snippet(self, content: str, query_terms: set, snippet_size: int = 300,
                         highlight: bool = True) -> str:
        """
        Extract a relevant snippet from content with highlighted search terms.
        Enhanced to extract complete sentences and find regions with high term density.

        highlight=False returns the window as plain prose. The ** markers and
        leading/trailing ellipses are display furniture; feeding them to a
        model that scores the text (see rerank) measurably degrades it.
        """
        content_lower = content.lower()

        # Calculate term density across content in sliding windows
        window_size = snippet_size
        best_score = 0
        best_pos = 0

        # Slide through content and score each window by term matches
        for i in range(0, max(1, len(content) - window_size + 1), 50):  # Step by 50 chars
            window = content_lower[i:i + window_size]
            score = sum(window.count(term) for term in query_terms if term)
            if score > best_score:
                best_score = score
                best_pos = i

        # If no matches found, use beginning
        if best_score == 0:
            best_pos = 0

        # Expand to complete sentences
        # Find sentence boundaries (., !, ?, or newline followed by capital letter or code)
        sentence_pattern = r'[.!?\n][\s\n]+'

        # Find start of sentence
        start = best_pos
        # Look backwards for sentence start
        for match in re.finditer(sentence_pattern, content[:best_pos]):
            start = match.end()

        # If we're too far from best_pos, adjust
        if best_pos - start > snippet_size // 2:
            start = max(0, best_pos - snippet_size // 3)

        # Find end of sentence
        end = min(len(content), start + snippet_size)
        matches = list(re.finditer(sentence_pattern, content[start:]))
        for match in matches:
            potential_end = start + match.end()
            if potential_end - start >= snippet_size * 0.8:  # At least 80% of desired size
                end = potential_end
                break

        # If we couldn't find sentence end, use hard cutoff
        if end - start < snippet_size * 0.5:
            end = min(len(content), start + snippet_size)

        # Sentence-alignment above can walk `start` back to the nearest prior
        # boundary and then extend by a fixed snippet_size, which can shift
        # the window just far enough that it no longer covers the FULL span
        # that won the density search - best_pos is only that window's left
        # edge, so a match sitting near its right edge is exactly what gets
        # clipped. Found live: a claim about a SID register value lost its
        # own citation's supporting text this way, by 43 characters, because
        # the nearest sentence boundary sat close enough to pass the "too
        # far" check but not close enough to leave room for what came after
        # it.
        #
        # Compare DENSITY (score per character), not raw score - a shorter,
        # cleanly-bounded window naturally has a lower raw count than the
        # full-size window even when it is just as good throughout (uniform/
        # repetitive text), and that shortening is exactly the sentence-
        # boundary search's intended outcome once it clears the "at least 80%
        # of desired size" threshold. Comparing raw scores would force every
        # such window back out to the full window_size, silently defeating
        # every caller's max_chars/token budget. Verified empirically against
        # both failure classes before picking the threshold: a legitimately
        # shortened window in uniform content scored a 1.0 density ratio; the
        # live SID case that motivated this fix scored 0.14.
        if best_score > 0:
            aligned_len = end - start
            aligned_score = sum(content_lower[start:end].count(term) for term in query_terms if term)
            best_density = best_score / window_size
            aligned_density = aligned_score / max(1, aligned_len)
            if aligned_density < best_density * 0.5:
                start = best_pos
                end = min(len(content), best_pos + window_size)

        snippet = content[start:end].strip()

        # Preserve code blocks (lines starting with spaces/tabs)
        # Don't break in the middle of code
        lines = snippet.split('\n')
        if lines and (lines[0].startswith('    ') or lines[0].startswith('\t')):
            # Start of snippet is code, find complete code block
            code_end = 0
            for i, line in enumerate(lines):
                if line and not line[0].isspace():
                    code_end = i
                    break
            if code_end > 0:
                snippet = '\n'.join(lines[:code_end])

        if not highlight:
            return snippet

        # Highlight matching terms (case-insensitive, whole words)
        for term in query_terms:
            if len(term) >= 2:  # Only highlight terms with 2+ characters
                # Use word boundary for whole word matching when possible
                pattern = re.compile(f'\\b({re.escape(term)})\\b', re.IGNORECASE)
                snippet = pattern.sub(r'**\1**', snippet)

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def _embedding_passages(self, text: str) -> list[str]:
        """Split one chunk into windows that fit inside the encoder's input limit.

        all-MiniLM-L6-v2 truncates at 256 word-pieces (~190 English words), but
        chunks are 1500 words, so encoding a chunk whole discarded ~87% of it
        before the model ever saw it - semantic search was effectively matching
        on chunk openings only. Each chunk is therefore indexed as several
        overlapping windows; semantic_search max-pools them back to one score
        per chunk, so nothing downstream sees the extra vectors.

        EMBEDDING_WINDOW_WORDS=0 restores the old whole-chunk behaviour.
        """
        window = int(os.getenv('EMBEDDING_WINDOW_WORDS', '200'))
        if window <= 0:
            return [text] if text.strip() else []

        words = text.split()
        if not words:
            return []
        if len(words) <= window:
            return [text]

        overlap = int(os.getenv('EMBEDDING_WINDOW_OVERLAP', '40'))
        overlap = max(0, min(overlap, window - 1))
        stride = window - overlap

        passages = []
        for start in range(0, len(words), stride):
            passages.append(' '.join(words[start:start + window]))
            if start + window >= len(words):
                break
        return passages

    def _expand_chunks_to_passages(self, chunks: list) -> tuple[list[str], list[tuple]]:
        """Flatten chunks into (texts, doc_map) where several texts may share a
        (doc_id, chunk_id) key. The map keeps its historical 2-tuple shape so
        every existing consumer keeps working unchanged."""
        texts: list[str] = []
        doc_map: list[tuple] = []
        for chunk in chunks:
            for passage in self._embedding_passages(chunk.content):
                texts.append(passage)
                doc_map.append((chunk.doc_id, chunk.chunk_id))
        return texts, doc_map

    def _passage_fanout(self) -> int:
        """Average vectors per chunk in the live index, used to scale FAISS k.

        Cached against the map length so a search doesn't rescan ~90k entries.
        """
        n = len(self.embeddings_doc_map)
        if n == 0:
            return 1
        cached = getattr(self, '_fanout_cache', None)
        if cached is not None and cached[0] == n:
            return cached[1]
        distinct = len({tuple(entry) for entry in self.embeddings_doc_map})
        fanout = max(1, round(n / max(1, distinct)))
        self._fanout_cache = (n, fanout)
        return fanout

    def _ensure_embeddings_loaded(self):
        """
        Lazy load embeddings model and index on first use.

        This significantly improves startup time by deferring the ~2.5 second
        model loading until semantic search is actually needed.
        """
        if not self.use_semantic or self._embeddings_loaded:
            return

        try:
            # Load the sentence transformer model (this is the slow part)
            model_name = os.getenv('SEMANTIC_MODEL', 'all-MiniLM-L6-v2')
            self.logger.info(f"Lazy loading embeddings model: {model_name} (first semantic search)")
            try:
                # Fast path: model should already be cached on disk. In
                # theory local_files_only=True means no network at all, but
                # some sentence-transformers/huggingface_hub versions still
                # issue a revision/etag check even with that flag set, and a
                # blocked/filtered connection to that check hangs forever
                # with no timeout of its own (see issue #14 - a single
                # add_document call hung 28+ minutes here with zero CPU
                # usage, despite the model already being fully cached).
                # Bound it with the same socket timeout as the fallback path
                # below so this can never hang the caller indefinitely.
                with _network_timeout():
                    self.embeddings_model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                # Not cached yet - fetch it, but bound the wait. Without
                # this, an unreachable Hugging Face Hub (offline machine,
                # filtered egress) can block this call indefinitely, taking
                # every semantic_search call down with it.
                with _network_timeout():
                    self.embeddings_model = SentenceTransformer(model_name)

            # Load the pre-computed embeddings index
            self._load_embeddings()

            self._embeddings_loaded = True
            self.logger.info("Embeddings model and index loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to lazy load embeddings: {e}")
            self.use_semantic = False
            self._embeddings_loaded = False

    def _load_embeddings(self):
        """Load FAISS embeddings index from disk (acquires the cross-process lock)."""
        if not self.use_semantic:
            return
        with _cross_process_lock(self.embeddings_lock_file):
            self._load_embeddings_locked()

    def _load_embeddings_locked(self):
        """Load FAISS embeddings index from disk.

        Caller must already hold embeddings_lock_file - this does no locking
        of its own so it can be called from inside a read-modify-write
        section (see _add_chunks_to_embeddings, _build_embeddings,
        _remove_doc_embeddings) without deadlocking on re-entry.
        """
        try:
            if self.embeddings_file.exists() and self.embeddings_map_file.exists():
                self.embeddings_index = faiss.read_index(str(self.embeddings_file))
                with open(self.embeddings_map_file, 'r') as f:
                    # json gives lists; _find_similar_semantic does a
                    # .index((doc_id, chunk_id)) lookup that never matches a
                    # list, so normalise back to tuples on the way in.
                    self.embeddings_doc_map = [tuple(e) for e in json.load(f)]
                self.logger.info(f"Loaded embeddings index with {len(self.embeddings_doc_map)} vectors")
            else:
                self.embeddings_index = None
                self.embeddings_doc_map = []
                self.logger.info("No existing embeddings found, will build on first use")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            self.embeddings_index = None
            self.embeddings_doc_map = []

    def _save_embeddings_locked(self):
        """Write the FAISS index and its doc-id map to disk, atomically.

        Caller must already hold embeddings_lock_file. Each file is written
        to a temp path and moved into place with os.replace, so a crash or a
        concurrent reader never observes a torn/partial file - and because
        both files are rewritten while the lock is held, a reader taking the
        same lock (see _load_embeddings) never observes one half of the pair
        updated and not the other.
        """
        if not self.use_semantic or self.embeddings_index is None:
            return
        try:
            index_bytes = faiss.serialize_index(self.embeddings_index).tobytes()
            _atomic_write_bytes(self.embeddings_file, index_bytes)
            map_bytes = json.dumps(self.embeddings_doc_map).encode('utf-8')
            _atomic_write_bytes(self.embeddings_map_file, map_bytes)
            self.logger.info(f"Saved embeddings index with {len(self.embeddings_doc_map)} vectors")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")

    def _remove_doc_embeddings(self, doc_id: str):
        """Surgically remove one document's vectors from the shared index.

        remove_document() used to just null self.embeddings_index in memory
        and leave the on-disk files untouched. The next add_document() would
        then see a null index, build a FRESH index containing only its own
        new chunks, and overwrite the full-corpus file with it - silently
        destroying every other document's embeddings. This instead reloads
        the latest committed state under the cross-process lock, drops only
        this document's rows, and saves the result back - so the rest of the
        corpus's embeddings survive a removal.
        """
        if not self.use_semantic:
            return
        with _cross_process_lock(self.embeddings_lock_file):
            self._load_embeddings_locked()
            if self.embeddings_index is None or not self.embeddings_doc_map:
                return

            keep_positions = [i for i, (d, _) in enumerate(self.embeddings_doc_map) if d != doc_id]
            if len(keep_positions) == len(self.embeddings_doc_map):
                return  # this document had no embedded chunks - nothing to do

            if keep_positions:
                vectors = self.embeddings_index.reconstruct_n(0, self.embeddings_index.ntotal)
                kept_vectors = vectors[keep_positions]
                new_index = faiss.IndexFlatIP(self.embeddings_index.d)
                new_index.add(kept_vectors)
                self.embeddings_index = new_index
                self.embeddings_doc_map = [self.embeddings_doc_map[i] for i in keep_positions]
                self._fanout_cache = None
                self._save_embeddings_locked()
            else:
                # Nothing left at all - clear in-memory state and remove the
                # now-empty artifacts so a stale file can't be misread later.
                self.embeddings_index = None
                self.embeddings_doc_map = []
                self.embeddings_file.unlink(missing_ok=True)
                self.embeddings_map_file.unlink(missing_ok=True)
            self.logger.info(f"Removed embeddings for {doc_id}: {len(keep_positions)} vectors remain")

    def _build_embeddings(self):
        """Build FAISS index from all document chunks."""
        if not self.use_semantic:
            return

        # Ensure model is loaded
        self._ensure_embeddings_loaded()

        if self.embeddings_model is None:
            return

        self.logger.info("Building embeddings index for all chunks...")
        start_time = time.time()

        # Load all chunks
        chunks = self._get_chunks_db()
        if not chunks:
            self.logger.warning("No chunks to embed")
            return

        # Generate embeddings (CPU-bound; deliberately outside the lock below
        # so concurrent agent processes don't serialise behind each other's
        # encoding work, only behind the actual shared-file write).
        texts, doc_map = self._expand_chunks_to_passages(chunks)
        if not texts:
            self.logger.warning("No passages to embed")
            return
        self.logger.info(f"Encoding {len(texts)} passages from {len(chunks)} chunks")
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        # This rebuild is authoritative - it re-derives from every chunk in
        # the database, not from a possibly-stale in-memory copy - so it just
        # needs the write itself serialised against other processes, not a
        # reload-before-modify.
        with _cross_process_lock(self.embeddings_lock_file):
            dimension = embeddings.shape[1]
            self.embeddings_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.embeddings_index.add(embeddings)
            self.embeddings_doc_map = doc_map
            self._fanout_cache = None
            self._save_embeddings_locked()

        # Invalidate similarity cache since embeddings changed
        if self._similar_cache is not None:
            self._similar_cache.clear()
            self.logger.info("Similarity cache invalidated after rebuilding embeddings")

        elapsed = time.time() - start_time
        self.logger.info(f"Built embeddings index in {elapsed:.2f}s")

    def _add_chunks_to_embeddings(self, chunks: list[DocumentChunk]):
        """
        Incrementally add new chunks to the FAISS embeddings index with batched processing.

        For large documents, processes chunks in batches to reduce memory usage and
        improve CPU cache utilization. Configurable via EMBEDDING_BATCH_SIZE env var.

        Args:
            chunks: List of DocumentChunk objects to add to embeddings
        """
        if not self.use_semantic or self.embeddings_model is None:
            return

        if not chunks:
            self.logger.debug("No chunks to add to embeddings")
            return

        self.logger.info(f"Adding {len(chunks)} chunks to embeddings index...")
        start_time = time.time()

        # Get batch size from environment (default: 32 for optimal memory/performance trade-off)
        batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))

        # Encode every batch up front - CPU-bound work with no shared state,
        # so it happens outside the lock and doesn't serialise concurrent
        # agent processes against each other's model inference time.
        # Batch over passages, not chunks: one 1500-word chunk expands to ~9
        # encoder-sized windows, so batching by chunk would hand the model
        # batches ~9x larger than EMBEDDING_BATCH_SIZE intends.
        all_texts, all_keys = self._expand_chunks_to_passages(chunks)
        if not all_texts:
            self.logger.debug("No passages to add to embeddings")
            return

        batch_vectors = []
        total_passages = len(all_texts)
        for batch_start in range(0, total_passages, batch_size):
            batch_end = min(batch_start + batch_size, total_passages)
            texts = all_texts[batch_start:batch_end]
            embeddings = self.embeddings_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            batch_vectors.append((embeddings, all_keys[batch_start:batch_end]))
            if total_passages > batch_size:
                self.logger.debug(f"Encoded batch {batch_start+1}-{batch_end}/{total_passages}")

        # Reload the latest committed state under the lock before appending -
        # not the possibly-stale copy this process loaded at startup or built
        # up in memory - then append and save as one atomic critical section.
        # Without this, two concurrent add_document calls each append to
        # their OWN in-memory copy of an older index and the second save
        # silently discards the first agent's new vectors entirely.
        with _cross_process_lock(self.embeddings_lock_file):
            self._load_embeddings_locked()
            if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
                self.logger.info("Creating new embeddings index")
                dimension = batch_vectors[0][0].shape[1]
                self.embeddings_index = faiss.IndexFlatIP(dimension)
                self.embeddings_doc_map = []

            for embeddings, batch_keys in batch_vectors:
                self.embeddings_index.add(embeddings)
                self.embeddings_doc_map.extend(batch_keys)

            self._fanout_cache = None
            self._save_embeddings_locked()

        # Invalidate similarity cache since embeddings changed
        if self._similar_cache is not None:
            self._similar_cache.clear()
            self.logger.debug("Similarity cache invalidated after adding chunks")

        elapsed = time.time() - start_time
        self.logger.info(f"Added {len(chunks)} chunks to embeddings index in {elapsed:.2f}s")

    def _ensure_reranker_loaded(self):
        """Lazy load the cross-encoder on first rerank, mirroring the
        bi-encoder path in _ensure_embeddings_loaded (including its network
        deadline - see issue #14 for what an unbounded HF fetch does)."""
        if not self.use_reranker or self._reranker_loaded:
            return

        model_name = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        max_length = int(os.getenv('RERANK_MAX_LENGTH', '512'))
        try:
            self.logger.info(f"Lazy loading reranker model: {model_name}")
            try:
                with _network_timeout():
                    self.reranker_model = CrossEncoder(model_name, max_length=max_length,
                                                       local_files_only=True)
            except Exception:
                with _network_timeout():
                    self.reranker_model = CrossEncoder(model_name, max_length=max_length)
            self._reranker_loaded = True
            self.logger.info("Reranker model loaded successfully")
        except Exception as e:
            # Degrade to the first-stage ranking rather than failing the search.
            self.logger.error(f"Failed to load reranker, continuing without it: {e}")
            self.use_reranker = False
            self.reranker_model = None
            self._reranker_loaded = False

    def _ensure_nli_loaded(self):
        """Lazy load the local NLI entailment cross-encoder, mirroring
        _ensure_reranker_loaded exactly (same network deadline, same
        local-then-remote load order, same degrade-and-disable on failure).

        Also resolves (contradiction_idx, entailment_idx, neutral_idx) from
        the model's own config.id2label rather than assuming the class
        order - a different NLI checkpoint than the default is one env var
        away (NLI_MODEL), and a wrong assumption here would silently invert
        every verdict instead of raising anything.
        """
        if not self.use_nli_verification or self._nli_loaded:
            return

        model_name = os.getenv('NLI_MODEL', 'cross-encoder/nli-deberta-v3-base')
        max_length = int(os.getenv('NLI_MAX_LENGTH', '512'))
        try:
            self.logger.info(f"Lazy loading NLI verification model: {model_name}")
            try:
                with _network_timeout():
                    self.nli_model = CrossEncoder(model_name, max_length=max_length,
                                                  local_files_only=True)
            except Exception:
                with _network_timeout():
                    self.nli_model = CrossEncoder(model_name, max_length=max_length)

            id2label = {int(k): str(v).strip().lower()
                       for k, v in self.nli_model.model.config.id2label.items()}
            by_label = {v: k for k, v in id2label.items()}
            if {'contradiction', 'entailment', 'neutral'} <= by_label.keys():
                self._nli_label_indices = (
                    by_label['contradiction'], by_label['entailment'], by_label['neutral'])
            else:
                # Standard order for the sentence-transformers NLI cross-encoders
                # (nli-deberta-v3-*, nli-distilroberta-base, ...) - used only if
                # a swapped-in model's labels don't match the expected strings.
                self.logger.warning(
                    f"NLI model {model_name} has unrecognised labels {id2label} - "
                    "assuming the standard (contradiction, entailment, neutral) order")
                self._nli_label_indices = (0, 1, 2)

            self._nli_loaded = True
            self.logger.info(f"NLI verification model loaded successfully (labels: {id2label})")
        except Exception as e:
            # Degrade to the LLM-based check (or the plain heuristic) rather
            # than failing answer_question.
            self.logger.error(f"Failed to load NLI model, continuing without it: {e}")
            self.use_nli_verification = False
            self.nli_model = None
            self._nli_loaded = False

    def _rerank_passage(self, result: dict, query_terms: set, max_chars: int) -> str:
        """Pick the text the cross-encoder should score for one candidate.

        The cross-encoder truncates at RERANK_MAX_LENGTH word-pieces, so
        handing it a 1500-word chunk re-creates the bi-encoder's original
        defect one stage later: the tail is never seen. Score the densest
        query-term window of the real chunk instead, and only fall back to the
        display snippet if the chunk has gone.
        """
        try:
            chunk = self.get_chunk(result['doc_id'], result['chunk_id'])
        except Exception:
            self.logger.exception(
                f"Failed to load chunk {result['doc_id']}/{result['chunk_id']} for reranking")
            chunk = None

        if not chunk or not chunk.content:
            return result.get('snippet', '')
        if len(chunk.content) <= max_chars:
            return chunk.content
        if query_terms:
            return self._extract_snippet(chunk.content, query_terms,
                                         snippet_size=max_chars, highlight=False)
        return chunk.content[:max_chars]

    def rerank(self, query: str, results: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """Reorder first-stage results with a cross-encoder, best first.

        Bi-encoder retrieval embeds query and passage independently, so it can
        only measure whether they land near each other in vector space. A
        cross-encoder reads both together and scores the pair directly, which
        is far more accurate but far too slow to run over the whole corpus -
        hence retrieve-many-then-rerank-few.

        Returns results unchanged (trimmed to top_k) when reranking is off or
        the model failed to load, so callers never need to branch on it.
        """
        if top_k is None:
            top_k = len(results)

        if not results or not self.use_reranker:
            return results[:top_k]

        self._ensure_reranker_loaded()
        if self.reranker_model is None:
            return results[:top_k]

        start_time = time.time()
        max_chars = int(os.getenv('RERANK_PASSAGE_CHARS', '1400'))
        query_terms = _content_terms(query)

        try:
            passages = [self._rerank_passage(r, query_terms, max_chars) for r in results]
            scores = self.reranker_model.predict(
                [(query, passage) for passage in passages],
                show_progress_bar=False,
            )
        except Exception as e:
            self.logger.exception(f"Reranking failed, keeping first-stage order: {e}")
            return results[:top_k]

        reranked = []
        for result, score in zip(results, scores):
            enriched = dict(result)
            enriched['rerank_score'] = float(score)
            # Keep the first-stage score visible; callers and the eval harness
            # both read 'score', so overwriting it would hide the retrieval
            # signal that put the candidate here in the first place.
            enriched['retrieval_score'] = result.get('score')
            reranked.append(enriched)

        reranked.sort(key=lambda r: r['rerank_score'], reverse=True)

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Reranked {len(results)} candidates to {top_k} in {elapsed_ms:.0f}ms")
        return reranked[:top_k]

    def _rerank_depth(self, max_results: int) -> int:
        """How many candidates to retrieve before reranking.

        A reranker can only promote what the first stage returned, so the
        depth is the ceiling on what it can fix.
        """
        depth = int(os.getenv('RERANK_CANDIDATES', '30'))
        return max(depth, max_results)

    def semantic_search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None,
                        include_superseded: bool = False,
                        rerank: Optional[bool] = None) -> list[dict]:
        """
        Perform semantic search using embeddings and vector similarity.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by
            include_superseded: If False (default), excludes superseded card versions
            rerank: Rerank with the cross-encoder. None (default) follows
                USE_RERANKER; True/False override it per call.

        Returns:
            List of search results with scores
        """
        self._sync_documents_if_needed()

        if not self.use_semantic:
            raise RuntimeError("Semantic search not available. Enable with USE_SEMANTIC_SEARCH=1")

        # Lazy load embeddings model on first use (saves ~2.5s on startup)
        self._ensure_embeddings_loaded()

        if self.embeddings_model is None:
            raise RuntimeError("Failed to load embeddings model")

        # Build embeddings index if not yet built
        if self.embeddings_index is None or len(self.embeddings_doc_map) == 0:
            self._build_embeddings()
            if self.embeddings_index is None:
                return []

        do_rerank = self.use_reranker if rerank is None else (rerank and self.use_reranker)
        # Retrieve deeper than the caller asked for when a reranker will sort
        # the candidates back down.
        retrieve_n = self._rerank_depth(max_results) if do_rerank else max_results

        # Check cache first
        start_time = time.time()
        if self._semantic_cache is not None:
            cache_key = self._cache_key('semantic_search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       include_superseded=include_superseded,
                                       rerank=do_rerank)
            if cache_key in self._semantic_cache:
                results = self._semantic_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Semantic cache HIT for query: '{query}' ({len(results)} results, {elapsed_ms:.2f}ms)")
                return results

        self.logger.info(f"Semantic search query: '{query}' (max_results={max_results}, tags={tags})")

        # Check embedding cache first (expensive operation to avoid)
        cache_key = hashlib.md5(query.encode('utf-8')).hexdigest()
        query_embedding = None

        if self._embedding_cache is not None:
            query_embedding = self._embedding_cache.get(cache_key)
            if query_embedding is not None:
                self.logger.debug(f"Embedding cache HIT for query: '{query[:50]}'")

        # Encode query if not cached
        if query_embedding is None:
            self.logger.debug(f"Embedding cache MISS for query: '{query[:50]}'")
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            # Cache the normalized embedding
            if self._embedding_cache is not None:
                self._embedding_cache[cache_key] = query_embedding
                self.logger.debug(f"Cached embedding for query: '{query[:50]}'")
        else:
            # Embedding was cached and already normalized
            pass

        # Search in FAISS index (get more results for filtering)
        # Each chunk now occupies several index positions, so ask for
        # proportionally more neighbours or a single verbose chunk could fill
        # the whole result window on its own.
        k = min(retrieve_n * 5 * self._passage_fanout(), len(self.embeddings_doc_map))
        scores, indices = self.embeddings_index.search(query_embedding, k)

        # Build results
        results = []
        seen_chunks = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.embeddings_doc_map):
                continue

            doc_id, chunk_id = self.embeddings_doc_map[idx]

            # FAISS returns hits in descending score order, so the first window
            # seen for a chunk is its best one - skipping the rest max-pools the
            # windows back to one result per chunk.
            if (doc_id, chunk_id) in seen_chunks:
                continue
            seen_chunks.add((doc_id, chunk_id))

            doc = self.documents.get(doc_id)

            # Exclude superseded card versions by default
            if not include_superseded and doc and doc.superseded_by:
                continue

            # Filter by tags if specified
            if tags:
                if doc and not any(t in doc.tags for t in tags):
                    continue

            # Get chunk
            chunk = self.get_chunk(doc_id, chunk_id)
            if not chunk:
                continue

            # Extract snippet (highlight query terms)
            query_terms = _content_terms(query)
            snippet = self._extract_snippet(chunk.content, query_terms)

            doc = self.documents.get(doc_id)
            results.append({
                'doc_id': doc_id,
                'filename': chunk.filename,
                'title': chunk.title,
                'chunk_id': chunk_id,
                'score': float(score),
                'snippet': snippet,
                'word_count': chunk.word_count,
                'similarity': float(score)  # Cosine similarity score
            })

            if len(results) >= retrieve_n:
                break

        if do_rerank:
            results = self.rerank(query, results, max_results)

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Semantic search completed: {len(results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'semantic', len(results), elapsed, tags)

        # Store in cache
        if self._semantic_cache is not None:
            cache_key = self._cache_key('semantic_search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       include_superseded=include_superseded,
                                       rerank=do_rerank)
            self._semantic_cache[cache_key] = results

        return results

    @staticmethod
    def _rrf_k() -> int:
        try:
            return max(1, int(os.getenv('RRF_K', '60')))
        except ValueError:
            return 60

    def _fuse_rankings(self, fts_results: list[dict], semantic_results: list[dict],
                       semantic_weight: float, max_results: int) -> list[dict]:
        """Combine two rankings into one ordered result list.

        Reciprocal Rank Fusion by default: a result's score is the sum of
        weight/(K + rank) over the rankers that returned it, so appearing in
        both lists beats ranking first in one. Only ordinal position is used,
        which is the point - bm25() magnitudes and cosine similarities are not
        on a common scale, and any attempt to put them on one has to invent a
        normalisation that changes with the batch.

        Set HYBRID_FUSION=weighted for the previous max-normalised score blend.
        """
        semantic_weight = min(1.0, max(0.0, semantic_weight))
        fts_weight = 1.0 - semantic_weight
        legacy = os.getenv('HYBRID_FUSION', 'rrf').lower() == 'weighted'

        def base(r: dict) -> dict:
            return {
                'doc_id': r['doc_id'],
                'filename': r['filename'],
                'title': r['title'],
                'chunk_id': r['chunk_id'],
                'snippet': r['snippet'],
                'word_count': r['word_count'],
            }

        merged: dict = {}
        for arm, arm_results, weight in (('fts', fts_results, fts_weight),
                                         ('semantic', semantic_results, semantic_weight)):
            for rank, r in enumerate(arm_results, 1):
                key = (r['doc_id'], r['chunk_id'])
                item = merged.setdefault(key, {**base(r), 'score': 0.0,
                                               'fts_score': 0.0, 'semantic_score': 0.0,
                                               'fts_rank': None, 'semantic_rank': None})
                item[f'{arm}_rank'] = rank
                # Keep each arm's own score for callers that display it.
                item[f'{arm}_score'] = float(r.get('similarity', r.get('score', 0.0)))

        if legacy:
            # Max-normalise within each arm, then blend. Retained so a bad RRF
            # rollout can be reverted without a code change.
            max_fts = max((abs(r.get('score', 0.0)) for r in fts_results), default=0.0)
            for item in merged.values():
                fts_norm = (abs(item['fts_score']) / max_fts) if max_fts > 0 else 0.0
                item['score'] = fts_weight * fts_norm + semantic_weight * item['semantic_score']
        else:
            k = self._rrf_k()
            for item in merged.values():
                score = 0.0
                if item['fts_rank'] is not None:
                    score += fts_weight / (k + item['fts_rank'])
                if item['semantic_rank'] is not None:
                    score += semantic_weight / (k + item['semantic_rank'])
                item['score'] = score

        results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
        return results[:max_results]

    def hybrid_search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None,
                     semantic_weight: float = 0.7,
                     rerank: Optional[bool] = None) -> list[dict]:
        """
        Perform hybrid search combining FTS5 keyword search and semantic search.

        Fuses the two rankings with Reciprocal Rank Fusion: each result scores
        sum(weight / (RRF_K + rank)) over the rankers that returned it.

        RRF replaced a weighted sum of max-normalised scores, which was fragile
        for two reasons. The arms are not comparable - FTS5 returns bm25()
        values with no fixed scale while semantic returns cosine similarity in
        0-1 - so dividing by the batch maximum made a result's contribution
        depend on how strong its neighbours happened to be, not on how good it
        was. And the blend needed a hand-tuned semantic_weight that silently
        went stale: it sat at 0.3 (70% keyword) from a period when the keyword
        arm was secretly BM25 rather than FTS5, which measured 10 points of
        recall@5 below pure semantic search. Ranks carry no scale to drift.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by
            semantic_weight: Relative weight of the semantic ranking (0.0-1.0).
                Default 0.7. Measured on the 40-question eval set, 0.7 matches
                pure semantic recall@5 (97.5%) while beating its MRR (0.884 vs
                0.822); equal weighting scores 92.5%/0.790, because the FTS5
                arm is the weaker of the two. 1.0 is semantic only, 0.0 keyword
                only. The 0.5-0.85 range is a plateau, not a knife-edge fit.

        Returns:
            List of search results with combined scores

        Env:
            HYBRID_FUSION=weighted restores the legacy normalised-score blend.
            RRF_K (default 60) damps how much the top ranks dominate.
            USE_RERANKER=1 sorts the fused list with a cross-encoder.
        """
        self._sync_documents_if_needed()

        if not self.use_semantic or self.embeddings_model is None:
            # Fall back to regular search if semantic not available
            self.logger.warning("Semantic search not available, falling back to FTS5/BM25")
            return self.search(query, max_results, tags)

        do_rerank = self.use_reranker if rerank is None else (rerank and self.use_reranker)
        retrieve_n = self._rerank_depth(max_results) if do_rerank else max_results

        # Check cache first
        start_time = time.time()
        if self._hybrid_cache is not None:
            cache_key = self._cache_key('hybrid_search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       semantic_weight=semantic_weight,
                                       rerank=do_rerank)
            if cache_key in self._hybrid_cache:
                results = self._hybrid_cache[cache_key]
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"Hybrid cache HIT for query: '{query}' ({len(results)} results, {elapsed_ms:.2f}ms)")
                return results

        self.logger.info(f"Hybrid search query: '{query}' (max_results={max_results}, tags={tags}, semantic_weight={semantic_weight})")

        # Run both search methods in parallel for better performance
        # (FTS5 and semantic are independent operations)
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both searches concurrently
            fts_future = executor.submit(self.search, query, retrieve_n * 2, tags)
            semantic_future = executor.submit(self.semantic_search, query, retrieve_n * 2, tags,
                                              False, False)

            # Wait for both to complete
            fts_results = fts_future.result()
            semantic_results = semantic_future.result()

        self.logger.debug(f"Parallel search completed: FTS={len(fts_results)}, Semantic={len(semantic_results)}")

        results = self._fuse_rankings(fts_results, semantic_results,
                                      semantic_weight, retrieve_n)

        if do_rerank:
            results = self.rerank(query, results, max_results)

        elapsed = (time.time() - start_time) * 1000
        self.logger.info(f"Hybrid search completed: {len(results)} results in {elapsed:.2f}ms")

        # Log search for analytics
        self._log_search(query, 'hybrid', len(results), elapsed, tags)

        # Store in cache
        if self._hybrid_cache is not None:
            cache_key = self._cache_key('hybrid_search',
                                       query=query,
                                       max_results=max_results,
                                       tags=tuple(sorted(tags)) if tags else None,
                                       semantic_weight=semantic_weight,
                                       rerank=do_rerank)
            self._hybrid_cache[cache_key] = results

        return results

    def fuzzy_search(self, query: str, max_results: int = 5, tags: Optional[list[str]] = None,
                     similarity_threshold: int = 80) -> list[dict]:
        """
        Search with typo tolerance using fuzzy string matching.

        Handles misspellings and variations:
        - "VIC2" → finds "VIC-II"
        - "asembly" → finds "assembly"
        - "6052" → finds "6502"

        Args:
            query: Search query (may contain typos)
            max_results: Maximum number of results to return
            tags: Optional list of tags to filter by
            similarity_threshold: Minimum similarity score (0-100, default: 80)

        Returns:
            List of search results, potentially with corrected query terms
        """
        start_time = time.time()

        # Try exact search first (fast path)
        exact_results = self.search(query, max_results, tags)
        if len(exact_results) >= max_results:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Fuzzy search (exact match): '{query}' ({len(exact_results)} results, {elapsed_ms:.2f}ms)")
            return exact_results

        # Build vocabulary from indexed terms if not already built
        if not hasattr(self, '_search_vocabulary') or self._search_vocabulary is None:
            self._build_search_vocabulary()

        # If vocabulary is still empty, fall back to exact search
        if not self._search_vocabulary:
            return exact_results

        # Attempt fuzzy matching on query terms
        if not FUZZY_SUPPORT:
            self.logger.warning("rapidfuzz not available, falling back to exact search")
            return exact_results

        from rapidfuzz import process

        # Split query into terms and correct each one
        query_terms = query.split()
        corrected_terms = []
        corrections_made = []

        for term in query_terms:
            # Find best match in vocabulary using fuzzy string matching
            best_match = process.extractOne(
                term,
                self._search_vocabulary,
                score_cutoff=similarity_threshold
            )

            if best_match:
                corrected_term = best_match[0]
                score = best_match[1]
                corrected_terms.append(corrected_term)

                if corrected_term.lower() != term.lower():
                    corrections_made.append({
                        'original': term,
                        'corrected': corrected_term,
                        'similarity': score
                    })
            else:
                # No match found, keep original term
                corrected_terms.append(term)

        # Build corrected query
        corrected_query = ' '.join(corrected_terms)

        # Log corrections if any were made
        if corrections_made:
            self.logger.info(f"Fuzzy search corrections: {corrections_made}")

        # Search with corrected query
        results = self.search(corrected_query, max_results, tags)

        # Add correction metadata to results
        if corrections_made and results:
            for result in results:
                result['fuzzy_corrections'] = corrections_made
                result['corrected_query'] = corrected_query

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Fuzzy search: '{query}' → '{corrected_query}' ({len(results)} results, {elapsed_ms:.2f}ms)")

        return results

    def _build_search_vocabulary(self):
        """Extract all unique terms from indexed content for fuzzy matching."""
        vocabulary = set()

        # Add known C64 technical terms
        known_terms = [
            'VIC-II', 'VIC2', 'VIC', 'SID', 'CIA', '6502', '6581', '6567', '6569', '6526',
            'sprite', 'raster', 'screen', 'memory', 'register', 'address',
            'assembly', 'BASIC', 'machine', 'code', 'opcode', 'instruction',
            'bit', 'byte', 'word', 'pointer', 'variable', 'string', 'loop',
            'interrupt', 'timer', 'trigger', 'control', 'port', 'peripheral',
            'sound', 'music', 'voice', 'envelope', 'frequency', 'amplitude',
            'color', 'palette', 'graphics', 'pixel', 'bitmap', 'character',
            'kernel', 'ROM', 'RAM', 'disk', 'tape', 'storage',
            'program', 'subroutine', 'jump', 'branch', 'call', 'return',
            'accumulator', 'index', 'stack', 'status', 'flag', 'carry'
        ]
        vocabulary.update(known_terms)

        # Extract terms from chunks (limited to avoid overhead)
        try:
            chunks_sample = self.chunks[:min(1000, len(self.chunks))]  # Sample first 1000 chunks

            for chunk in chunks_sample:
                # Extract words (lowercase, alphanumeric + hyphen)
                words = re.findall(r'\b[a-z0-9-]{3,}\b', chunk.content.lower())
                vocabulary.update(words)
        except Exception as e:
            self.logger.warning(f"Failed to build search vocabulary: {e}")

        self._search_vocabulary = sorted(list(vocabulary))
        self.logger.debug(f"Built search vocabulary with {len(self._search_vocabulary)} terms")

    def search_within_results(self, previous_results: list[dict], refinement_query: str,
                             max_results: int = 5) -> list[dict]:
        """
        Search within a previous result set to refine results.

        Useful for progressive search refinement:
        1. results = search("VIC-II")  # 50 results
        2. refined = search_within_results(results, "sprite collision")  # 8 results

        Args:
            previous_results: Results from a previous search
            refinement_query: Query to refine within the previous results
            max_results: Maximum number of refined results

        Returns:
            Filtered and re-ranked results from the previous search set
        """
        start_time = time.time()

        if not previous_results:
            self.logger.warning("search_within_results called with empty previous results")
            return []

        # Extract unique doc_ids from previous results
        doc_ids = list(set([r['doc_id'] for r in previous_results]))
        self.logger.info(f"Searching within {len(doc_ids)} documents ({len(previous_results)} chunks) for: '{refinement_query}'")

        # Build refinement terms
        refinement_terms = self._preprocess_text(refinement_query)
        if not refinement_terms:
            self.logger.warning(f"No valid terms in refinement query: '{refinement_query}'")
            return previous_results[:max_results]

        # Score each previous result against refinement query
        scored_results = []

        for result in previous_results:
            # Get the full chunk content for better matching
            try:
                chunk = next((c for c in self.chunks
                            if c.doc_id == result['doc_id'] and c.chunk_id == result['chunk_id']),
                           None)

                if not chunk:
                    # Use snippet from search result if chunk not found
                    content = result.get('snippet', '')
                else:
                    content = chunk.content
            except Exception as e:
                self.logger.debug(f"Failed to get chunk content: {e}")
                content = result.get('snippet', '')

            # Calculate relevance score for refinement terms
            relevance_score = 0
            content_lower = content.lower()

            for term in refinement_terms:
                # Count occurrences and weight by position
                occurrences = content_lower.count(term)
                relevance_score += occurrences

                # Boost score if term appears near beginning (30% of content)
                if content_lower[:int(len(content) * 0.3)].count(term) > 0:
                    relevance_score += 2

            # Only include results with at least some match
            if relevance_score > 0:
                scored_results.append({
                    **result,
                    'refinement_score': relevance_score,
                    'original_score': result.get('score', 0)
                })

        # Sort by refinement score (descending), then by original score
        scored_results.sort(key=lambda x: (x['refinement_score'], x['original_score']), reverse=True)

        # Limit results
        refined_results = scored_results[:max_results]

        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info(f"Search within results: '{refinement_query}' ({len(refined_results)}/{len(previous_results)} results, {elapsed_ms:.2f}ms)")

        return refined_results
