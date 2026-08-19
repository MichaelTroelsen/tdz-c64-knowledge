"""Tagging and summarisation for IngestMixin.

Split out of kb/ingest.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from datetime import datetime
from models import DocumentNotFoundError
from typing import Optional
import json
import os
import re


class _TaggingMixin:

    def suggest_tags(self, doc_id: str, confidence_threshold: float = 0.6) -> list[dict]:
        """
        Suggest tags for a document based on content analysis.

        Uses heuristic-based tag suggestion (no LLM required):
        - Detects hardware components (SID, VIC-II, CIA, 6502)
        - Identifies programming topics (assembly, BASIC, machine code)
        - Recognizes document types (reference, tutorial, guide)
        - Extracts memory addresses and registers

        Args:
            doc_id: Document ID to analyze
            confidence_threshold: Minimum confidence for suggestions (0.0-1.0)

        Returns:
            List of suggested tags with confidence scores:
            [
                {'tag': 'sid-programming', 'confidence': 0.95, 'category': 'hardware'},
                {'tag': 'assembly', 'confidence': 0.85, 'category': 'programming'},
                ...
            ]

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        suggested_tags = []

        # Get sample of document content (first 3 chunks to avoid overhead)
        try:
            chunks = self._get_chunks_db(doc_id)
            sample_text = '\n'.join([c['content'] for c in chunks[:3]]) if chunks else ''
        except Exception as e:
            self.logger.warning(f"Failed to get chunks for {doc_id}: {e}")
            sample_text = ''

        if not sample_text:
            return suggested_tags

        text_lower = sample_text.lower()

        # Hardware detection
        hardware_patterns = {
            'sid-chip': (r'\bsid\b|\b6581\b', 0.9),
            'vic-ii': (r'\bvic-?ii\b|\bvic\s*2\b|\b6569\b|\b6567\b', 0.9),
            'cia': (r'\bcia\b|\b6526\b', 0.85),
            '6502-processor': (r'\b6502\b|\b6510\b', 0.9),
            'joystick': (r'\bjoystick\b|\bcontroller\b', 0.7),
            'disk-drive': (r'\bdisk\s*drive\b|\b1541\b|\b1571\b', 0.8),
        }

        for tag, (pattern, confidence) in hardware_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if confidence >= confidence_threshold:
                    suggested_tags.append({
                        'tag': tag,
                        'confidence': confidence,
                        'category': 'hardware'
                    })

        # Programming topic detection
        programming_patterns = {
            'assembly': (r'\bassembly\b|\bmachine\s*code\b|\basync\b', 0.85),
            'basic': (r'\bbasic\b|\bprogram\s*line\b|\bline\s*numbers\b', 0.8),
            'graphics': (r'\bgraphics\b|\bsprite\b|\bbitmap\b|\bcharacter\s*set\b', 0.85),
            'sound-music': (r'\bsound\b|\bmusic\b|\baudio\b|\benvelop\b|\bsynthesis\b', 0.8),
            'interrupts': (r'\binterrupt\b|\birq\b|\bnmi\b', 0.9),
            'memory-management': (r'\bmemory\s*map\b|\bmemory\s*address\b|\b\$[0-9a-f]{4}\b', 0.75),
        }

        for tag, (pattern, confidence) in programming_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if confidence >= confidence_threshold:
                    suggested_tags.append({
                        'tag': tag,
                        'confidence': confidence,
                        'category': 'programming'
                    })

        # Document type detection
        doc_type_patterns = {
            'reference': (r'\breference\b|\bopcode\s*table\b|\binstruction\s*set\b', 0.85),
            'tutorial': (r'\btutorial\b|\bhow\s*to\b|\bguide\b|\blearn\b', 0.75),
            'specification': (r'\bspecification\b|\bspec\b|\bdatasheet\b|\bmanual\b', 0.9),
            'code-example': (r'\bexample\b|\bcode\s*sample\b|\broutine\b', 0.7),
        }

        for tag, (pattern, confidence) in doc_type_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if confidence >= confidence_threshold:
                    suggested_tags.append({
                        'tag': tag,
                        'confidence': confidence,
                        'category': 'document-type'
                    })

        # Difficulty level detection
        difficulty_patterns = {
            'beginner': (r'\bbeginning\b|\bstarter\b|\bintroduction\b|\bfundamentals?\b', 0.75),
            'intermediate': (r'\bintermediate\b|\badvanced-beginner\b', 0.7),
            'advanced': (r'\badvanced\b|\bexpert\b|\bdeep-dive\b', 0.75),
        }

        for tag, (pattern, confidence) in difficulty_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if confidence >= confidence_threshold:
                    suggested_tags.append({
                        'tag': tag,
                        'confidence': confidence,
                        'category': 'difficulty'
                    })

        # Sort by confidence (descending)
        suggested_tags.sort(key=lambda x: x['confidence'], reverse=True)

        self.logger.info(f"Suggested {len(suggested_tags)} tags for document {doc_id[:12]}")

        return suggested_tags

    def add_tags_to_document(self, doc_id: str, new_tags: list[str],
                            merge: bool = True) -> list[str]:
        """
        Add tags to a document, optionally merging with existing tags.

        Args:
            doc_id: Document ID
            new_tags: Tags to add
            merge: If True, merge with existing tags. If False, replace tags.

        Returns:
            Updated list of all tags for the document

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        # Normalize and deduplicate tags
        new_tags = list(set([t.lower().replace(' ', '-') for t in new_tags if t]))

        if merge:
            # Merge with existing tags
            existing_tags = set(self.documents[doc_id].tags or [])
            all_tags = list(existing_tags | set(new_tags))
        else:
            all_tags = new_tags

        # Update document
        self.update_document_tags(doc_id, all_tags)

        return all_tags

    def get_tags_by_category(self) -> dict[str, list[dict]]:
        """
        Get all tags organized by category for easier browsing.

        Returns:
            Dictionary with categories as keys and tag lists as values:
            {
                'hardware': [
                    {'tag': 'sid-chip', 'count': 15, 'documents': ['doc1', 'doc2', ...]},
                    ...
                ],
                'programming': [...],
                ...
            }
        """
        # Categorize known tags
        tag_categories = {
            'hardware': ['sid-chip', 'vic-ii', 'cia', '6502-processor', 'joystick', 'disk-drive'],
            'programming': ['assembly', 'basic', 'graphics', 'sound-music', 'interrupts', 'memory-management'],
            'document-type': ['reference', 'tutorial', 'specification', 'code-example'],
            'difficulty': ['beginner', 'intermediate', 'advanced'],
        }

        result = {}

        # For each category, count tag usage
        for category, known_tags in tag_categories.items():
            result[category] = []

            for tag in known_tags:
                # Count documents with this tag
                doc_ids = [doc_id for doc_id, doc in self.documents.items()
                          if tag in (doc.tags or [])]

                if doc_ids:  # Only include tags that are actually used
                    result[category].append({
                        'tag': tag,
                        'count': len(doc_ids),
                        'documents': doc_ids[:10]  # Show first 10 docs
                    })

            # Sort by count (descending)
            result[category].sort(key=lambda x: x['count'], reverse=True)

        # Add custom/user-defined tags that don't fit in categories
        all_known_tags = set()
        for tags_list in tag_categories.values():
            all_known_tags.update(tags_list)

        custom_tags = {}
        for doc in self.documents.values():
            for tag in (doc.tags or []):
                if tag not in all_known_tags:
                    if tag not in custom_tags:
                        custom_tags[tag] = []
                    custom_tags[tag].append(doc.doc_id)

        if custom_tags:
            result['custom'] = [
                {
                    'tag': tag,
                    'count': len(doc_ids),
                    'documents': doc_ids[:10]
                }
                for tag, doc_ids in sorted(custom_tags.items(),
                                         key=lambda x: len(x[1]),
                                         reverse=True)
            ]

        return result

    def _call_llm(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """
        Call LLM with a prompt (helper method for LLM operations).

        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLM response text

        Raises:
            ValueError: If LLM not available or call fails
        """
        # Check if LLM client is available
        if not hasattr(self, 'llm_client') or self.llm_client is None:
            # Try to initialize it
            try:
                from llm_integration import LLMClient
                self.llm_client = LLMClient()
            except Exception as e:
                raise ValueError(f"LLM client not available: {e}")

        try:
            response = self.llm_client.call(prompt, max_tokens=max_tokens, temperature=temperature)
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise ValueError(f"LLM call failed: {e}")

    def summarize_document(self, doc_id: str,
                          max_length: int = 500,
                          style: str = "technical") -> str:
        """
        Generate an AI summary of a document.

        Args:
            doc_id: Document ID to summarize
            max_length: Maximum summary length in words
            style: Summary style (technical, simple, or detailed)

        Returns:
            Summary text

        Raises:
            ValueError: If document not found or LLM not available
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        # Check if LLM client is available
        if not hasattr(self, 'llm_client') or self.llm_client is None:
            # Try to initialize it
            try:
                from llm_integration import LLMClient
                self.llm_client = LLMClient()
            except Exception as e:
                raise ValueError(f"LLM client not available: {e}")

        doc = self.documents[doc_id]

        # Get document content (first 10 chunks to keep context reasonable)
        chunks = self._get_chunks_db(doc_id)
        content_chunks = chunks[:10] if len(chunks) > 10 else chunks
        content = '\n\n'.join([chunk.content for chunk in content_chunks])

        # Truncate content if too long (max 20000 chars)
        if len(content) > 20000:
            content = content[:20000] + "..."

        # Build prompt based on style
        style_prompts = {
            "technical": "Provide a concise technical summary focusing on key concepts, technologies, and implementation details.",
            "simple": "Provide a simple, easy-to-understand summary suitable for beginners.",
            "detailed": "Provide a comprehensive detailed summary covering all major topics and subtopics."
        }

        style_instruction = style_prompts.get(style, style_prompts["technical"])

        prompt = f"""Summarize the following document in approximately {max_length} words.

Document Title: {doc.title}

{style_instruction}

Document Content:
{content}

Summary:"""

        try:
            summary = self.llm_client.call(prompt, max_tokens=max_length * 2, temperature=0.3)
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            raise ValueError(f"Failed to generate summary: {e}")

    def update_tags_bulk(self, doc_ids: Optional[list[str]] = None,
                         existing_tags: Optional[list[str]] = None,
                         add_tags: Optional[list[str]] = None,
                         remove_tags: Optional[list[str]] = None,
                         replace_tags: Optional[list[str]] = None) -> dict:
        """
        Update tags for multiple documents in bulk.

        Args:
            doc_ids: List of document IDs to update (if None, uses existing_tags to find docs)
            existing_tags: Find documents with any of these tags (alternative to doc_ids)
            add_tags: Tags to add to the documents
            remove_tags: Tags to remove from the documents
            replace_tags: Replace all tags with these tags

        Returns:
            Dictionary with lists of updated and failed document IDs

        Examples:
            # Add 'assembly' tag to specific documents
            kb.update_tags_bulk(doc_ids=['doc1', 'doc2'], add_tags=['assembly'])

            # Remove 'draft' tag from all documents that have it
            kb.update_tags_bulk(existing_tags=['draft'], remove_tags=['draft'])

            # Replace all tags with 'archive' for specific documents
            kb.update_tags_bulk(doc_ids=['doc1', 'doc2'], replace_tags=['archive'])

            # Add 'reviewed' and remove 'draft' for documents with 'pending' tag
            kb.update_tags_bulk(existing_tags=['pending'], add_tags=['reviewed'], remove_tags=['draft'])
        """
        if not doc_ids and not existing_tags:
            raise ValueError("Must provide either doc_ids or existing_tags")

        if not add_tags and not remove_tags and not replace_tags:
            raise ValueError("Must provide at least one of: add_tags, remove_tags, replace_tags")

        results = {
            'updated': [],
            'failed': []
        }

        # Collect doc_ids to update
        ids_to_update = set()

        if doc_ids:
            ids_to_update.update(doc_ids)

        if existing_tags:
            # Find all documents with any of the specified tags
            for doc_id, doc in self.documents.items():
                if any(tag in doc.tags for tag in existing_tags):
                    ids_to_update.add(doc_id)

        self.logger.info(f"Bulk tag update: updating {len(ids_to_update)} documents")

        for doc_id in ids_to_update:
            # Tracked outside the try so the handler can restore it: doc.tags is
            # mutated in memory before the UPDATE lands.
            old_tags = None
            try:
                if doc_id not in self.documents:
                    results['failed'].append({
                        'doc_id': doc_id,
                        'error': 'Document not found'
                    })
                    continue

                doc = self.documents[doc_id]
                old_tags = doc.tags.copy()

                # Apply tag operations
                if replace_tags is not None:
                    doc.tags = replace_tags.copy()
                else:
                    if add_tags:
                        # Add tags (avoiding duplicates)
                        for tag in add_tags:
                            if tag not in doc.tags:
                                doc.tags.append(tag)

                    if remove_tags:
                        # Remove tags
                        doc.tags = [tag for tag in doc.tags if tag not in remove_tags]

                # Update in database
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    UPDATE documents
                    SET tags = ?
                    WHERE doc_id = ?
                """, (json.dumps(doc.tags), doc_id))
                self.db_conn.commit()

                results['updated'].append({
                    'doc_id': doc_id,
                    'old_tags': old_tags,
                    'new_tags': doc.tags
                })

                self.logger.debug(f"Updated tags for {doc_id}: {old_tags} -> {doc.tags}")

            except Exception as e:
                # Restore the pre-mutation tags. Without this, a failed write
                # left this process reporting tags that were never persisted -
                # they silently reverted on the next restart.
                doc = self.documents.get(doc_id)
                if doc is not None and old_tags is not None:
                    try:
                        self.db_conn.rollback()
                    except Exception:
                        pass
                    doc.tags = old_tags
                results['failed'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })
                self.logger.error(f"Failed to update tags for {doc_id}: {e}")

        self.logger.info(f"Bulk tag update complete: {len(results['updated'])} updated, "
                        f"{len(results['failed'])} failed")

        return results

    def auto_tag_document(self, doc_id: str, confidence_threshold: float = 0.7,
                         max_tags: int = 10, append: bool = True) -> dict:
        """
        Generate tags automatically using LLM analysis.

        Args:
            doc_id: Document to tag
            confidence_threshold: Minimum confidence to accept tag (0.0-1.0)
            max_tags: Maximum number of tags to suggest
            append: If True, append to existing tags; if False, replace

        Returns:
            {
                'doc_id': str,
                'suggested_tags': [{'tag': str, 'confidence': float}, ...],
                'applied_tags': [str],
                'skipped_tags': [str],  # Below confidence threshold
                'existing_tags': [str],
                'new_tags': [str]  # Final tag list
            }

        Example:
            result = kb.auto_tag_document('doc123', confidence_threshold=0.7)
            # {
            #     'suggested_tags': [
            #         {'tag': 'sid-programming', 'confidence': 0.95},
            #         {'tag': 'assembly', 'confidence': 0.88},
            #         {'tag': 'beginner', 'confidence': 0.65}  # Below threshold
            #     ],
            #     'applied_tags': ['sid-programming', 'assembly'],
            #     'skipped_tags': ['beginner'],
            #     ...
            # }
        """
        # Import LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Auto-tagging requires LLM integration.")

        # Get LLM client
        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key.")

        # Get document
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Get sample text (first 3 chunks for analysis)
        chunks = self._get_chunks_db(doc_id)
        sample_chunks = chunks[:3] if len(chunks) > 3 else chunks
        sample_text = "\n\n".join([c.content for c in sample_chunks])

        # Limit text size (first 3000 chars)
        if len(sample_text) > 3000:
            sample_text = sample_text[:3000] + "..."

        # Build prompt
        prompt = f"""Analyze this Commodore 64 technical documentation and suggest relevant tags.

Consider these categories:
1. Hardware components (sid, vic-ii, cia, 6502, memory, cartridge, disk-drive, etc.)
2. Programming topics (assembly, basic, machine-code, graphics, sound, sprites, etc.)
3. Document type (tutorial, reference, manual, guide, example, etc.)
4. Difficulty level (beginner, intermediate, advanced, expert)
5. Content area (programming, hardware, history, repair, modification, etc.)

Document title: {doc.title}
Document filename: {doc.filename}

Sample text:
{sample_text}

Return a JSON object with this structure:
{{
    "tags": [
        {{"tag": "sid-programming", "confidence": 0.95, "reason": "Document extensively discusses SID chip programming"}},
        {{"tag": "assembly", "confidence": 0.88, "reason": "Contains assembly code examples"}}
    ]
}}

Important:
- Use lowercase with hyphens (e.g., "sid-programming" not "SID Programming")
- Provide {max_tags} or fewer tags
- Include confidence score (0.0-1.0) for each tag
- Brief reason for each tag suggestion
- Return ONLY the JSON, no other text"""

        # Call LLM
        self.logger.info(f"Auto-tagging document {doc_id} ({doc.title})")

        try:
            response = llm_client.call_json(prompt, max_tokens=1024, temperature=0.3)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise ValueError(f"Failed to generate tags: {e}")

        # Parse response
        suggested_tags = response.get('tags', [])

        # Filter by confidence
        high_confidence_tags = [
            t for t in suggested_tags
            if t['confidence'] >= confidence_threshold
        ]

        low_confidence_tags = [
            t for t in suggested_tags
            if t['confidence'] < confidence_threshold
        ]

        # Extract tag names
        applied_tag_names = [t['tag'] for t in high_confidence_tags]
        skipped_tag_names = [t['tag'] for t in low_confidence_tags]

        # Get existing tags
        existing_tags = doc.tags.copy()

        # Apply tags
        if append:
            # Add new tags to existing (avoid duplicates)
            new_tags = existing_tags.copy()
            for tag in applied_tag_names:
                if tag not in new_tags:
                    new_tags.append(tag)
        else:
            # Replace all tags
            new_tags = applied_tag_names

        # Update document
        doc.tags = new_tags

        # Update in database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            UPDATE documents
            SET tags = ?
            WHERE doc_id = ?
        """, (json.dumps(new_tags), doc_id))
        self.db_conn.commit()

        result = {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'suggested_tags': suggested_tags,
            'applied_tags': applied_tag_names,
            'skipped_tags': skipped_tag_names,
            'existing_tags': existing_tags,
            'new_tags': new_tags,
            'confidence_threshold': confidence_threshold
        }

        self.logger.info(f"Auto-tagged {doc_id}: applied {len(applied_tag_names)} tags, "
                        f"skipped {len(skipped_tag_names)} low-confidence tags")

        return result

    def auto_tag_all_documents(self, confidence_threshold: float = 0.7,
                               max_tags: int = 10, append: bool = True,
                               skip_tagged: bool = True, max_docs: Optional[int] = None) -> dict:
        """
        Bulk auto-tag all documents using LLM.

        Args:
            confidence_threshold: Minimum confidence to accept tag (0.0-1.0)
            max_tags: Maximum tags per document
            append: If True, append to existing tags; if False, replace
            skip_tagged: If True, skip documents that already have tags
            max_docs: Maximum number of documents to process (None = all)

        Returns:
            {
                'processed': int,
                'skipped': int,
                'failed': int,
                'total_tags_added': int,
                'results': [list of individual results]
            }

        Example:
            results = kb.auto_tag_all_documents(
                confidence_threshold=0.7,
                skip_tagged=True,
                max_docs=10
            )
        """
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'total_tags_added': 0,
            'results': []
        }

        # Get documents to process
        docs_to_process = []

        for doc_id, doc in self.documents.items():
            # Skip if already has tags (optional)
            if skip_tagged and doc.tags:
                results['skipped'] += 1
                continue

            docs_to_process.append(doc_id)

            # Limit number of documents
            if max_docs and len(docs_to_process) >= max_docs:
                break

        self.logger.info(f"Auto-tagging {len(docs_to_process)} documents "
                        f"(skipped {results['skipped']} already tagged)")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            try:
                self.logger.info(f"Auto-tagging {i}/{len(docs_to_process)}: {doc_id}")

                result = self.auto_tag_document(
                    doc_id,
                    confidence_threshold=confidence_threshold,
                    max_tags=max_tags,
                    append=append
                )

                # Count new tags added
                tags_added = len(set(result['new_tags']) - set(result['existing_tags']))
                results['total_tags_added'] += tags_added

                results['processed'] += 1
                results['results'].append(result)

            except Exception as e:
                results['failed'] += 1
                self.logger.error(f"Failed to auto-tag {doc_id}: {e}")
                results['results'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })

        self.logger.info(f"Auto-tagging complete: processed={results['processed']}, "
                        f"failed={results['failed']}, tags_added={results['total_tags_added']}")

        return results

    def generate_summary(self, doc_id: str, summary_type: str = 'brief',
                        force_regenerate: bool = False) -> str:
        """
        Generate an AI-powered summary of a document.

        Args:
            doc_id: Document ID to summarize
            summary_type: Type of summary ('brief', 'detailed', 'bullet')
                - 'brief': 1-2 paragraph overview (200-300 words)
                - 'detailed': Comprehensive summary with key points (500-800 words)
                - 'bullet': Bullet-point summary of main topics
            force_regenerate: If True, regenerate even if cached summary exists

        Returns:
            Summary text as a string

        Raises:
            ValueError: If document not found or LLM not configured
            DocumentNotFoundError: If document doesn't exist

        Examples:
            # Generate brief summary
            summary = kb.generate_summary('doc123', 'brief')

            # Get detailed summary
            summary = kb.generate_summary('doc456', 'detailed')

            # Force regeneration (bypass cache)
            summary = kb.generate_summary('doc789', 'brief', force_regenerate=True)
        """
        # Validate document exists
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Check for cached summary
        if not force_regenerate:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT summary_text FROM document_summaries
                WHERE doc_id = ? AND summary_type = ?
            """, (doc_id, summary_type))
            result = cursor.fetchone()
            if result:
                self.logger.debug(f"Using cached summary for {doc_id} ({summary_type})")
                return result[0]

        # Import LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Summarization requires LLM integration.")

        # Get LLM client
        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key.")

        # Get document content
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            raise ValueError(f"No content found for document: {doc_id}")

        # For brief summaries, use first 5 chunks; for detailed, use more
        if summary_type == 'brief':
            sample_chunks = chunks[:5]
            word_limit = 300
            length_guidance = "1-2 paragraphs, approximately 200-300 words"
        elif summary_type == 'detailed':
            sample_chunks = chunks[:15] if len(chunks) > 15 else chunks
            word_limit = 800
            length_guidance = "3-5 paragraphs with detailed explanations, approximately 500-800 words"
        elif summary_type == 'bullet':
            sample_chunks = chunks[:10]
            word_limit = 400
            length_guidance = "8-12 bullet points covering main topics"
        else:
            raise ValueError(f"Invalid summary type: {summary_type}. Must be 'brief', 'detailed', or 'bullet'.")

        # Join content
        content = "\n\n".join([c.content for c in sample_chunks])

        # Limit content size to first 10k chars to control API costs
        if len(content) > 10000:
            content = content[:10000] + "..."

        # Build prompt based on summary type
        if summary_type == 'bullet':
            prompt = f"""Create a bullet-point summary of this Commodore 64 technical documentation.

Document Title: {doc.title}
Document Type: {doc.file_type}

Content:
{content}

Create a concise bullet-point summary with 8-12 main topics. Each bullet should be clear and informative.
Return ONLY the bullet points, one per line, starting with "- ". No introduction or explanation needed."""

        else:
            prompt = f"""Create a {summary_type} summary of this Commodore 64 technical documentation.

Document Title: {doc.title}
Document Type: {doc.file_type}

Content:
{content}

Write a {summary_type} summary that is {length_guidance}.
Focus on:
- Key concepts and main topics
- Technical details relevant to programmers
- Important procedures or examples
- Practical applications

Return ONLY the summary text, no preamble."""

        # Call LLM
        self.logger.info(f"Generating {summary_type} summary for {doc_id} ({doc.title})")

        try:
            summary_text = llm_client.call(prompt, max_tokens=word_limit + 200, temperature=0.4)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise ValueError(f"Failed to generate summary: {e}")

        # Clean up summary text
        if not summary_text or not summary_text.strip():
            raise ValueError("LLM returned empty summary")

        summary_text = summary_text.strip()

        # Store summary in database
        cursor = self.db_conn.cursor()
        try:
            # Get model name from LLM client
            model = os.getenv('LLM_MODEL', 'unknown')

            cursor.execute("""
                INSERT OR REPLACE INTO document_summaries
                (doc_id, summary_type, summary_text, generated_at, model, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, summary_type, summary_text, datetime.now().isoformat(),
                  model, len(summary_text.split())))

            self.db_conn.commit()
            self.logger.info(f"Saved {summary_type} summary for {doc_id}")

        except Exception as e:
            self.logger.exception("Failed to save summary to database")
            # Return summary even if save failed
            pass

        return summary_text

    def generate_summary_all(self, summary_types: Optional[list[str]] = None,
                            force_regenerate: bool = False,
                            max_docs: Optional[int] = None) -> dict:
        """
        Bulk generate summaries for all documents.

        Args:
            summary_types: List of summary types to generate (['brief'], ['brief', 'detailed'], etc.)
                          Default: ['brief']
            force_regenerate: If True, regenerate all summaries
            max_docs: Maximum number of documents to process (None = all)

        Returns:
            {
                'processed': int,
                'failed': int,
                'total_summaries': int,
                'by_type': {'brief': int, 'detailed': int, 'bullet': int},
                'results': [list of individual results]
            }

        Example:
            results = kb.generate_summary_all(
                summary_types=['brief', 'detailed'],
                max_docs=50
            )
        """
        if summary_types is None:
            summary_types = ['brief']

        results = {
            'processed': 0,
            'failed': 0,
            'total_summaries': 0,
            'by_type': {st: 0 for st in summary_types},
            'results': []
        }

        # Get documents to process
        docs_to_process = list(self.documents.keys())

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Generating summaries for {len(docs_to_process)} documents "
                        f"(types: {', '.join(summary_types)})")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'summaries': {}
            }

            for summary_type in summary_types:
                try:
                    self.logger.info(f"[{i}/{len(docs_to_process)}] {doc_id} ({summary_type})")

                    summary = self.generate_summary(
                        doc_id,
                        summary_type=summary_type,
                        force_regenerate=force_regenerate
                    )

                    doc_results['summaries'][summary_type] = {
                        'success': True,
                        'length': len(summary),
                        'word_count': len(summary.split())
                    }

                    results['total_summaries'] += 1
                    results['by_type'][summary_type] += 1

                except Exception as e:
                    results['failed'] += 1
                    self.logger.error(f"Failed to summarize {doc_id} ({summary_type}): {e}")
                    doc_results['summaries'][summary_type] = {
                        'success': False,
                        'error': str(e)
                    }

            results['processed'] += 1
            results['results'].append(doc_results)

        self.logger.info(f"Summary generation complete: processed={results['processed']}, "
                        f"failed={results['failed']}, total_summaries={results['total_summaries']}")

        return results

    def get_summary(self, doc_id: str, summary_type: str = 'brief') -> Optional[str]:
        """
        Retrieve a cached summary without regenerating.

        Args:
            doc_id: Document ID
            summary_type: Type of summary ('brief', 'detailed', 'bullet')

        Returns:
            Summary text if it exists, None otherwise
        """
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT summary_text FROM document_summaries
            WHERE doc_id = ? AND summary_type = ?
        """, (doc_id, summary_type))
        result = cursor.fetchone()
        return result[0] if result else None
