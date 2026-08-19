"""RAG answer_question path and answer-grounding verification for SearchMixin.

Split out of kb/search.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from text_utils import _SENTENCE_BOUNDARY_RE
from text_utils import _content_terms
from typing import Optional
import os
import re
import time


class _RagMixin:

    def answer_question(self, question: str, max_context_chunks: int = 5,
                       force_search_mode: Optional[str] = None) -> dict:
        """
        Answer a question using RAG (Retrieval-Augmented Generation).

        Retrieves relevant documentation and uses LLM to synthesize an answer
        with citations to source material.

        Args:
            question: The question to answer about C64 documentation
            max_context_chunks: Maximum number of documentation chunks to use for context
            force_search_mode: Override search mode ('keyword', 'semantic', 'hybrid', or None for auto)

        Returns:
            Dictionary with answer, sources, confidence, and metadata:
            {
                'answer': str,              # Generated answer text
                'sources': list[dict],      # Citations with metadata
                'confidence': float,        # 0.0-1.0, from claim verification when available
                'unverified_claims': list,  # Claim sentences whose cited source didn't support them
                'checked_claims': int,      # How many claims the verification pass actually checked
                'search_results': list,     # Top-N search results used
                'reasoning': str,           # Explanation of how answer was derived
                'model': str,               # LLM model used
                'error': str or None        # Error message if applicable
            }
        """
        start_time = time.time()

        if not question or len(question.strip()) < 3:
            return {
                'answer': None,
                'sources': [],
                'confidence': 0.0,
                'search_results': [],
                'reasoning': 'Invalid question: too short',
                'model': 'N/A',
                'error': 'Question must be at least 3 characters long'
            }

        self.logger.info(f"Answering question: '{question}'")

        try:
            # Step 1: Translate natural language query to understand intent
            translation = self.translate_nl_query(question, confidence_threshold=0.7)
            query_confidence = translation.get('confidence', 0.5)

            # Step 2: Choose search strategy
            if force_search_mode:
                search_mode = force_search_mode
            else:
                search_mode = translation.get('search_mode', 'hybrid')

            # Step 3: Retrieve relevant context
            if search_mode == 'semantic' and self.use_semantic:
                results = self.semantic_search(question, max_context_chunks * 2)
            elif search_mode == 'hybrid' and self.use_semantic:
                results = self.hybrid_search(question, max_context_chunks * 2)
            else:
                # The keyword path has no reranking of its own, so apply it
                # here; the semantic and hybrid paths already reranked.
                results = self.search(question, self._rerank_depth(max_context_chunks * 2)
                                      if self.use_reranker else max_context_chunks * 2)
                results = self.rerank(question, results, max_context_chunks * 2)

            if not results:
                self.logger.warning(f"No search results found for question: {question}")
                return {
                    'answer': f"I could not find relevant documentation to answer: {question}",
                    'sources': [],
                    'confidence': 0.1,
                    'search_results': [],
                    'reasoning': "Search found no results. Try searching directly with search_docs.",
                    'model': 'N/A',
                    'error': 'No relevant documents found'
                }

            # Step 4: Build context from top results
            context = self._build_rag_context(results[:max_context_chunks], question)

            # Step 5: Try to generate answer with LLM
            from llm_integration import get_llm_client

            llm_client = get_llm_client()

            if llm_client:
                # Generate answer using LLM with context
                answer_result = self._generate_answer_with_llm(
                    question=question,
                    context=context,
                    search_results=results[:max_context_chunks],
                    llm_client=llm_client
                )

                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.info(f"Question answering completed in {elapsed_ms:.2f}ms")

                return {
                    'answer': answer_result['text'],
                    'sources': answer_result['citations'],
                    'confidence': answer_result.get('confidence', query_confidence),
                    'unverified_claims': answer_result.get('unverified_claims', []),
                    'checked_claims': answer_result.get('checked_claims', 0),
                    'search_results': results[:max_context_chunks],
                    'reasoning': f"Question mode: {search_mode}. Retrieved {len(results[:max_context_chunks])} sources.",
                    'model': llm_client.provider.model,
                    'error': None
                }
            else:
                # Fallback: Return structured search summary without LLM
                self.logger.warning("LLM not configured, returning search-based fallback")
                return self._generate_summary_fallback(
                    question=question,
                    results=results[:max_context_chunks],
                    query_confidence=query_confidence,
                    search_mode=search_mode
                )

        except Exception as e:
            self.logger.error(f"Error answering question: {e}", exc_info=True)
            return {
                'answer': None,
                'sources': [],
                'confidence': 0.0,
                'search_results': [],
                'reasoning': 'An error occurred during answer generation',
                'model': 'N/A',
                'error': str(e)
            }

    def _build_rag_context(self, search_results: list[dict], question: str = "",
                           max_tokens: Optional[int] = None) -> str:
        """
        Build documentation context from search results for LLM prompt.

        The search results carry a ~300-char display snippet, which used to be
        the entire evidence the generator saw - roughly 1.5KB of text against a
        4000-token budget that stayed 90% empty. A C64 register table or timing
        diagram does not survive a 300-char window, so the answer was being
        written from fragments. Each source is now re-excerpted from its full
        chunk at a per-source share of the budget, centred on the densest
        question-term region of that chunk.

        Args:
            search_results: Search results with snippets and metadata
            question: Original question, used to centre each excerpt
            max_tokens: Context budget (~4 chars per token). Defaults to
                RAG_CONTEXT_TOKENS, or 8000.

        Returns:
            Formatted documentation context string with source citations
        """
        if max_tokens is None:
            max_tokens = int(os.getenv('RAG_CONTEXT_TOKENS', '8000'))

        CHARS_PER_TOKEN = 4
        SECTION_OVERHEAD = 200

        if not search_results:
            return ""

        # Split the budget evenly rather than spending it all on source 1 and
        # dropping the rest: citation breadth is the point of retrieving N.
        n = len(search_results)
        body_tokens = max(0, max_tokens - SECTION_OVERHEAD * n)
        per_source_chars = max(400, (body_tokens // n) * CHARS_PER_TOKEN)

        query_terms = _content_terms(question)

        context = ""
        for i, result in enumerate(search_results, 1):
            # Build section header
            header = f"## Source {i}: {result['title']}\n"
            header += f"Document: {result['filename']} (ID: {result['doc_id']})\n"
            header += f"Chunk {result['chunk_id']}"

            if result.get('page'):
                header += f", Page {result['page']}"

            header += f" - Relevance: {result['score']:.2f}\n\n"

            # Prefer a wide excerpt of the real chunk; fall back to the display
            # snippet if the chunk has since been removed from the database.
            excerpt = ""
            try:
                chunk = self.get_chunk(result['doc_id'], result['chunk_id'])
            except Exception:
                self.logger.exception(
                    f"Failed to load chunk {result['doc_id']}/{result['chunk_id']} for RAG context")
                chunk = None

            if chunk and chunk.content:
                if len(chunk.content) <= per_source_chars:
                    excerpt = chunk.content
                elif query_terms:
                    excerpt = self._extract_snippet(chunk.content, query_terms,
                                                    snippet_size=per_source_chars)
                else:
                    excerpt = chunk.content[:per_source_chars] + "..."

            if not excerpt:
                excerpt = result.get('snippet', '')

            context += header + excerpt + "\n\n"

        return context

    def _generate_answer_with_llm(self, question: str, context: str,
                                 search_results: list[dict],
                                 llm_client) -> dict:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: The user's question
            context: Formatted documentation context
            search_results: Original search results for citation tracking
            llm_client: LLMClient instance

        Returns:
            Dictionary with answer text and extracted citations
        """
        # Build prompt
        prompt = f"""You are a Commodore 64 documentation expert assistant.

Answer the following question based on the provided documentation excerpts. Be accurate and specific about technical details.

QUESTION: {question}

DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Provide a clear, accurate answer based on the documentation
2. If the answer requires multiple sources, synthesize information from them
3. Be specific about memory addresses (like $D000), register names, and technical specifications
4. Reference which documentation sections you're using (e.g., "Source 1", "Source 2")
5. If the documentation is incomplete or you're unsure, state that explicitly
6. Use correct technical terminology for C64 concepts

Please provide your answer now:"""

        try:
            # Call LLM with low temperature for accuracy
            answer_text = llm_client.call(
                prompt,
                temperature=0.3,        # Deterministic, fact-based responses
                max_tokens=2048         # Enough for detailed answers
            )

            # Extract citations from answer
            citations = self._extract_citations(answer_text, search_results)

            if os.getenv('USE_ANSWER_VERIFICATION', '1') == '1':
                verification = self._verify_answer_grounding(
                    question, answer_text, citations, search_results, llm_client)
            else:
                verification = {
                    'confidence': 0.85 if citations else 0.70,
                    'unverified_claims': [],
                    'checked_claims': 0,
                }

            return {
                'text': answer_text,
                'citations': citations,
                'confidence': verification['confidence'],
                'unverified_claims': verification['unverified_claims'],
                'checked_claims': verification['checked_claims'],
            }

        except Exception as e:
            self.logger.error(f"LLM answer generation failed: {e}")
            raise

    def _extract_claims_for_verification(self, answer_text: str, search_results: list[dict]) -> list[dict]:
        """Split an answer into sentence-sized, citation-bearing claims.

        A sentence with no "Source N" reference has nothing to check it
        against, so it is dropped rather than treated as an unverified claim
        - only sentences that actually cite something are checkable at all.
        """
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(answer_text) if s.strip()]
        claims = []
        for sentence in sentences:
            source_indices = sorted({
                int(m) for m in re.findall(r'[Ss]ource\s+(\d+)', sentence)
                if 0 <= int(m) - 1 < len(search_results)
            })
            if source_indices:
                claims.append({'text': sentence, 'sources': source_indices})
        return claims

    def _passages_for_claims(self, claims: list[dict], search_results: list[dict],
                             question: str) -> dict[tuple[int, int], str]:
        """One windowed excerpt per (claim, cited source) pair - not one
        shared window per source.

        Found live: a single 1500-word chunk can hold several distinct
        facts scattered across it (a SID register-reference chunk covering
        both waveform/noise behavior AND the base-address/register table,
        in different regions). Windowing once per source - using the whole
        question's terms, shared across every claim that cites it - picks
        ONE region; a claim about a fact sitting outside that region gets
        judged against text that never mentions it, and a true, correctly
        cited claim comes back "not_mentioned". Windowing per claim, using
        that claim's own words, tracks the fact actually being checked
        instead: the claim and its real supporting text share vocabulary,
        which is exactly what the density-scoring window needs to land in
        the right place.

        Question terms are unioned in rather than dropped, as a fallback for
        pronoun-heavy claim sentences ("It also has this feature.") whose own
        words are too generic to pull the window anywhere meaningful.
        """
        question_terms = _content_terms(question)
        # 2000 was the original default; live testing found it too large -
        # at that size the density search can pick an entirely wrong region
        # of a long chunk (breadth of repeated common terms beating a single
        # occurrence of the actual fact - see _extract_snippet). 800 stayed
        # within the chunk's actual fact-bearing sentence in every case
        # tested against a real ~8700-char chunk with two separate cited facts.
        max_chars = int(os.getenv('VERIFY_PASSAGE_CHARS', '800'))
        passages = {}
        for i, claim in enumerate(claims, 1):
            claim_terms = _content_terms(claim['text']) | question_terms
            for idx in claim['sources']:
                passages[(i, idx)] = self._rerank_passage(search_results[idx - 1], claim_terms, max_chars)
        return passages

    def _verify_claims_llm(self, claims: list[dict], passages: dict[tuple[int, int], str],
                           llm_client) -> Optional[dict[int, str]]:
        """Ask the LLM once (not once per claim) whether each cited passage
        supports its claim. Returns None - not a raised exception - on any
        failure, so the caller can degrade instead of crashing
        answer_question."""
        claim_blocks = []
        for i, claim in enumerate(claims, 1):
            cited_text = "\n---\n".join(passages[(i, idx)] for idx in claim['sources'])
            claim_blocks.append(f"CLAIM {i}: {claim['text']}\nCITED SOURCE(S):\n{cited_text}\n")

        prompt = f"""You are checking whether claims in a generated answer are actually supported by the documentation passages they cite.

For each numbered claim below, decide whether its cited source(s) support it, contradict it, or don't mention it at all.

{"".join(claim_blocks)}
Respond with JSON only, in this exact shape:
{{"verifications": [{{"claim": 1, "verdict": "supported"}}, {{"claim": 2, "verdict": "not_mentioned"}}]}}

verdict must be exactly one of: "supported", "contradicted", "not_mentioned". Include exactly one entry per claim, in claim order."""

        try:
            response = llm_client.call_json(prompt, temperature=0.0, max_tokens=1024)
            verdicts = {}
            for entry in response.get('verifications', []):
                idx = int(entry.get('claim'))
                verdicts[idx] = str(entry.get('verdict', '')).strip().lower()
            return verdicts
        except Exception as e:
            self.logger.warning(f"LLM-based grounding check failed: {e}")
            return None

    def _verify_claims_nli(self, claims: list[dict], passages: dict[tuple[int, int], str]) -> Optional[dict[int, str]]:
        """Score each claim against its cited passage(s) with a local NLI
        entailment cross-encoder instead of a second LLM call.

        Near-zero marginal cost per call once the model is loaded, at the
        cost of a large one-time model download and load. Premise is the
        cited passage(s), hypothesis is the claim - i.e. "does the evidence
        entail this claim", the same direction _verify_claims_llm's prompt
        asks the model to judge.

        Returns None - not a raised exception - on any failure (model
        unavailable, predict() error), so the caller can degrade to the
        LLM-based check or the plain heuristic exactly like an LLM failure
        does.
        """
        self._ensure_nli_loaded()
        if self.nli_model is None:
            return None

        order = list(range(1, len(claims) + 1))
        pairs = [
            ("\n---\n".join(passages[(i, idx)] for idx in claims[i - 1]['sources']), claims[i - 1]['text'])
            for i in order
        ]

        try:
            scores = self.nli_model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        except Exception as e:
            self.logger.warning(f"NLI-based grounding check failed: {e}")
            return None

        contra_idx, entail_idx, _neutral_idx = self._nli_label_indices
        verdicts = {}
        for i, row in zip(order, scores):
            top = int(row.argmax())
            if top == entail_idx:
                verdicts[i] = 'supported'
            elif top == contra_idx:
                verdicts[i] = 'contradicted'
            else:
                verdicts[i] = 'not_mentioned'
        return verdicts

    def _verify_answer_grounding(self, question: str, answer_text: str, citations: list[dict],
                                 search_results: list[dict], llm_client) -> dict:
        """
        Check whether the answer's cited claims are actually supported by the
        passages they cite, instead of trusting the model's self-report.

        The confidence this replaces was a hardcoded 0.85/0.70 keyed on
        whether the literal string "Source N" appeared anywhere in the
        output - which measures that the model claimed to cite something,
        not that the citation holds up. A wrong answer naming "Source 2"
        looked identical, in the response, to a right one.

        Splits the answer into sentence-sized claims and checks each against
        its cited source, via a local NLI cross-encoder (USE_NLI_VERIFICATION=1)
        or, by default, one extra LLM call. Falls back to the old
        citation-presence heuristic - with an empty unverified_claims list,
        since the check never ran - if there is nothing citable to check, or
        if verification fails: NLI unavailable/erroring degrades to the LLM
        check when an llm_client exists, and either backend failing degrades
        to the heuristic. A broken verifier must never take down
        answer_question itself.

        Returns:
            {'confidence': float, 'unverified_claims': list[str], 'checked_claims': int}
        """
        fallback = {
            'confidence': 0.85 if citations else 0.70,
            'unverified_claims': [],
            'checked_claims': 0,
        }

        claims = self._extract_claims_for_verification(answer_text, search_results)
        if not claims:
            return fallback

        passages = self._passages_for_claims(claims, search_results, question)

        verdicts = None
        if self.use_nli_verification:
            verdicts = self._verify_claims_nli(claims, passages)
        if verdicts is None:
            verdicts = self._verify_claims_llm(claims, passages, llm_client)
        if verdicts is None:
            return fallback

        unverified = []
        supported = 0
        for i, claim in enumerate(claims, 1):
            if verdicts.get(i) == 'supported':
                supported += 1
            else:
                unverified.append(claim['text'])

        return {
            'confidence': supported / len(claims),
            'unverified_claims': unverified,
            'checked_claims': len(claims),
        }

    def _extract_citations(self, answer_text: str, search_results: list[dict]) -> list[dict]:
        """
        Extract citations from LLM-generated answer.

        Looks for "Source N" references and maps them to search results.

        Args:
            answer_text: The LLM-generated answer text
            search_results: List of search results to cite

        Returns:
            List of citation dictionaries with full metadata
        """
        citations = []
        seen = set()

        # Find all "Source X" references (case-insensitive)
        source_refs = re.findall(r'[Ss]ource\s+(\d+)', answer_text)

        for ref_str in source_refs:
            try:
                idx = int(ref_str) - 1  # Convert to 0-based index

                if 0 <= idx < len(search_results):
                    result = search_results[idx]
                    key = (result['doc_id'], result['chunk_id'])

                    # Avoid duplicates
                    if key not in seen:
                        citations.append({
                            'doc_id': result['doc_id'],
                            'filename': result['filename'],
                            'title': result['title'],
                            'chunk_id': result['chunk_id'],
                            'page': result.get('page'),
                            'score': result.get('score', 0.0)
                        })
                        seen.add(key)
            except (ValueError, IndexError, KeyError):
                # Skip invalid references
                pass

        return citations

    def _generate_summary_fallback(self, question: str, results: list[dict],
                                   query_confidence: float, search_mode: str) -> dict:
        """
        Fallback response when LLM is unavailable.

        Returns structured summary of search results without LLM-generated answer.

        Args:
            question: The user's question
            results: Search results found
            query_confidence: Confidence from query translation
            search_mode: The search mode that was used

        Returns:
            Response dictionary with search-based answer fallback
        """
        # Build summary from search results
        summary = f"# Search Results for: {question}\n\n"
        summary += f"Found {len(results)} relevant documentation section(s):\n\n"

        sources = []
        for i, result in enumerate(results, 1):
            summary += f"## {i}. {result['title']}\n"
            summary += f"**File**: {result['filename']} | **Chunk**: {result['chunk_id']}"
            if result.get('page'):
                summary += f" | **Page**: {result['page']}"
            summary += f" | **Relevance**: {result['score']:.2f}\n\n"
            summary += f"{result.get('snippet', '')}\n\n"

            sources.append({
                'doc_id': result['doc_id'],
                'filename': result['filename'],
                'title': result['title'],
                'chunk_id': result['chunk_id'],
                'page': result.get('page'),
                'score': result.get('score', 0.0)
            })

        summary += "\n*Note: LLM not configured. Showing search results instead of synthesized answer.*"
        summary += "\n*To enable LLM-based answers, set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variables.*"

        return {
            'answer': summary,
            'sources': sources,
            'confidence': query_confidence * 0.7,  # Reduce confidence for fallback
            'search_results': results,
            'reasoning': f"LLM unavailable. Returned {search_mode} search results.",
            'model': 'fallback-search',
            'error': 'LLM not configured'
        }
