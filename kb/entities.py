"""Entity extraction, the extraction-job queue, and entity relationships.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Optional
import os
import queue
import re


class EntitiesMixin:

    def _normalize_entity_text(self, text: str, entity_type: str) -> str:
        """
        Normalize entity text for consistent matching and deduplication.

        Args:
            text: Raw entity text
            entity_type: Entity type (hardware, memory_address, etc.)

        Returns:
            Normalized entity text
        """
        normalized = text.strip()

        if entity_type == 'hardware':
            # Normalize hardware names
            # VIC II, VIC 2, VIC-II → VIC-II
            normalized = re.sub(r'VIC\s*-?\s*II|VIC\s+2', 'VIC-II', normalized, flags=re.IGNORECASE)
            # CIA 1, CIA-1 → CIA1
            normalized = re.sub(r'CIA\s*-?\s*([12])', r'CIA\1', normalized, flags=re.IGNORECASE)
            # Preserve standard forms
            if normalized.lower() == 'sid':
                normalized = 'SID'
            if normalized.lower() == 'cia':
                normalized = 'CIA'
            if normalized.lower() == 'pla':
                normalized = 'PLA'
            if normalized.lower() == 'c64':
                normalized = 'C64'
            if normalized.lower() == 'c128':
                normalized = 'C128'
            if 'vic-ii' in normalized.lower():
                normalized = 'VIC-II'

        elif entity_type == 'memory_address':
            # Normalize memory addresses to uppercase $ format
            if normalized.startswith('$'):
                normalized = normalized.upper()
            elif normalized.startswith('0x') or normalized.startswith('&H'):
                # Convert 0xD000 → $D000
                hex_part = normalized[2:] if normalized.startswith('0x') else normalized[2:]
                normalized = f'${hex_part.upper()}'
            elif normalized.isdigit():
                # Keep decimal as-is for now
                pass

        elif entity_type == 'instruction':
            # Instructions are always uppercase
            normalized = normalized.upper().split()[0]  # Take just the mnemonic

        elif entity_type == 'concept':
            # Normalize concept capitalization
            concept_map = {
                'sprite': 'sprite',
                'sprites': 'sprite',  # Singular form
                'raster interrupt': 'raster interrupt',
                'raster interrupts': 'raster interrupt',
                'irq': 'IRQ',
                'nmi': 'NMI',
                'dma': 'DMA',
                'bitmap mode': 'bitmap mode',
                'character mode': 'character mode',
                'multicolor mode': 'multicolor mode',
                'hires mode': 'hires mode',
            }
            lower = normalized.lower()
            normalized = concept_map.get(lower, normalized)

        return normalized

    def _extract_entities_regex(self, text: str) -> list[dict]:
        """
        Extract C64-specific entities using regex patterns (fast, no LLM needed).

        This supplements LLM-based extraction with high-confidence pattern matching
        for well-known C64 entities.

        Returns:
            List of entities with text, type, and confidence
        """
        entities = []

        # Hardware patterns (high confidence)
        hardware_patterns = [
            (r'\bVIC-?II\b', 'VIC-II', 0.98),
            (r'\bVIC\s*2\b', 'VIC-II', 0.95),
            (r'\b6569\b', '6569', 0.98),
            (r'\b6567\b', '6567', 0.98),
            (r'\bSID\b', 'SID', 0.98),
            (r'\b6581\b', '6581', 0.98),
            (r'\b8580\b', '8580', 0.98),
            (r'\bCIA(?:\s*[12])?\b', None, 0.98),  # CIA, CIA1, CIA2
            (r'\b6526\b', '6526', 0.98),
            (r'\b6502\b', '6502', 0.98),
            (r'\b6510\b', '6510', 0.98),
            (r'\bPLA\b', 'PLA', 0.95),
            (r'\b(?:C-?)?64(?:\s*C)?\b', 'C64', 0.90),
            (r'\bC-?128\b', 'C128', 0.95),
            (r'\bVIC-?20\b', 'VIC-20', 0.95),
            (r'\b1541\b', '1541', 0.95),
            (r'\b1571\b', '1571', 0.95),
            (r'\b1581\b', '1581', 0.95),
        ]

        for pattern, entity_name, confidence in hardware_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                # Get context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace('\n', ' ')

                entities.append({
                    'entity_text': entity_name if entity_name else matched_text,
                    'entity_type': 'hardware',
                    'confidence': confidence,
                    'context': context,
                    'source': 'regex'
                })

        # Memory addresses (very high confidence)
        # Matches $D000, $d020, 53280 (decimal), 0xD000
        addr_patterns = [
            (r'\$[0-9A-Fa-f]{4}\b', 0.99),  # $D000
            (r'\b(?:0x|&H)[0-9A-Fa-f]{4}\b', 0.98),  # 0xD000 or &HD000
            (r'\b(?:53|54|55|56|57|58|59)\d{3}\b', 0.85),  # Decimal (53280-59999 range)
        ]

        for pattern, confidence in addr_patterns:
            for match in re.finditer(pattern, text):
                matched_text = match.group(0)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace('\n', ' ')

                entities.append({
                    'entity_text': matched_text.upper() if matched_text.startswith('$') else matched_text,
                    'entity_type': 'memory_address',
                    'confidence': confidence,
                    'context': context,
                    'source': 'regex'
                })

        # 6502 instructions (high confidence)
        instructions = [
            'LDA', 'STA', 'LDX', 'STX', 'LDY', 'STY',
            'TAX', 'TAY', 'TXA', 'TYA', 'TSX', 'TXS',
            'PHA', 'PLA', 'PHP', 'PLP',
            'AND', 'ORA', 'EOR',
            'ADC', 'SBC', 'CMP', 'CPX', 'CPY',
            'INC', 'INX', 'INY', 'DEC', 'DEX', 'DEY',
            'ASL', 'LSR', 'ROL', 'ROR',
            'JMP', 'JSR', 'RTS', 'RTI',
            'BCC', 'BCS', 'BEQ', 'BNE', 'BMI', 'BPL', 'BVC', 'BVS',
            'CLC', 'CLD', 'CLI', 'CLV', 'SEC', 'SED', 'SEI',
            'BRK', 'NOP', 'BIT'
        ]

        for instruction in instructions:
            # Match instruction at word boundary or with addressing mode
            pattern = r'\b' + instruction + r'\b(?:\s+[#$%@]?[\w,()]+)?'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0).strip()
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end].replace('\n', ' ')

                # Higher confidence if followed by addressing mode
                has_operand = len(matched_text) > len(instruction)
                conf = 0.95 if has_operand else 0.85

                entities.append({
                    'entity_text': instruction.upper(),
                    'entity_type': 'instruction',
                    'confidence': conf,
                    'context': context,
                    'source': 'regex'
                })

        # Common C64 concepts (medium-high confidence)
        concept_patterns = [
            (r'\bsprite[s]?\b', 'sprite', 0.90),
            (r'\braster\s+interrupt[s]?\b', 'raster interrupt', 0.95),
            (r'\b(?:IRQ|NMI)\b', None, 0.92),
            (r'\bDMA\b', 'DMA', 0.90),
            (r'\bbitmap\s+mode\b', 'bitmap mode', 0.92),
            (r'\bcharacter\s+mode\b', 'character mode', 0.92),
            (r'\bmulti-?color\s+mode\b', 'multicolor mode', 0.92),
            (r'\bhi-?res\s+mode\b', 'hires mode', 0.90),
            (r'\bborder\s+color\b', 'border color', 0.88),
            (r'\bbackground\s+color\b', 'background color', 0.88),
            (r'\bcolor\s+RAM\b', 'color RAM', 0.90),
            (r'\bscreen\s+memory\b', 'screen memory', 0.90),
            (r'\bcharacter\s+set\b', 'character set', 0.88),
            (r'\bkernel\s+ROM\b', 'kernel ROM', 0.92),
            (r'\bBASIC\s+ROM\b', 'BASIC ROM', 0.92),
        ]

        for pattern, entity_name, confidence in concept_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace('\n', ' ')

                entities.append({
                    'entity_text': entity_name if entity_name else matched_text,
                    'entity_type': 'concept',
                    'confidence': confidence,
                    'context': context,
                    'source': 'regex'
                })

        return entities

    def extract_entities(self, doc_id: str,
                        confidence_threshold: float = 0.6,
                        force_regenerate: bool = False) -> dict:
        """
        Extract named entities from a document using LLM analysis.

        Args:
            doc_id: Document ID to extract entities from
            confidence_threshold: Minimum confidence to include entity (0.0-1.0, default: 0.6)
            force_regenerate: If True, extract even if entities already exist

        Returns:
            {
                'doc_id': str,
                'doc_title': str,
                'entities': [
                    {
                        'entity_text': 'VIC-II',
                        'entity_type': 'hardware',
                        'confidence': 0.95,
                        'context': '...snippet...',
                        'occurrence_count': 5
                    },
                    ...
                ],
                'entity_count': 42,
                'types': {'hardware': 10, 'memory_address': 8, ...}
            }

        Example:
            result = kb.extract_entities('my-doc-id', confidence_threshold=0.7)
            for entity in result['entities']:
                print(f"{entity['entity_type']}: {entity['entity_text']} ({entity['confidence']})")
        """
        # Validate document exists
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        self.logger.info(f"Extracting entities from document {doc_id} ({doc.title})")

        # Check in-memory cache first (fastest)
        cache_key = f"{doc_id}:{confidence_threshold}"
        if not force_regenerate and self._entity_cache is not None:
            cached_result = self._entity_cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Entity cache HIT for document: {doc_id}")
                return cached_result

        # Check if entities already exist in database
        if not force_regenerate:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                self.logger.info(f"Document {doc_id} already has {existing_count} entities (use force_regenerate=True to re-extract)")
                # Return existing entities from database
                result = self.get_entities(doc_id)

                # Cache in memory for faster future access
                if self._entity_cache is not None:
                    self._entity_cache[cache_key] = result
                    self.logger.debug(f"Cached entity result from database for document: {doc_id}")

                return result

        # Get LLM client
        try:
            from llm_integration import get_llm_client
        except ImportError:
            raise ImportError("llm_integration module not found. Install required dependencies.")

        llm_client = get_llm_client()
        if not llm_client:
            raise ValueError("LLM not configured. Set LLM_PROVIDER and appropriate API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)")

        # Get document chunks for sampling
        chunks = self._get_chunks_db(doc_id)
        if not chunks:
            raise ValueError(f"No chunks found for document {doc_id}")

        # Sample first 5 chunks (balance between coverage and cost)
        sample_chunks = chunks[:5] if len(chunks) > 5 else chunks
        sample_text = "\n\n".join([c.content for c in sample_chunks])

        # Limit to 5000 characters for cost control
        if len(sample_text) > 5000:
            sample_text = sample_text[:5000] + "..."

        # Build prompt for entity extraction
        prompt = f"""Extract named entities from this Commodore 64 technical documentation.

Document Title: {doc.title}

Content:
{sample_text}

Extract entities in these categories:
1. hardware - Chip names and components (SID, VIC-II, CIA, 6502, 6526, 6581, etc.)
2. memory_address - Memory addresses in any format ($D000, $D020, 53280, 0xD020, etc.)
3. instruction - Assembly instructions (LDA, STA, JMP, JSR, RTS, BRK, etc.)
4. person - People mentioned (Bob Yannes, Jack Tramiel, etc.)
5. company - Organizations (Commodore, MOS Technology, etc.)
6. product - Hardware/software products (VIC-20, C128, 1541, etc.)
7. concept - Technical concepts (sprite, raster interrupt, IRQ, DMA, etc.)

For each entity found, provide:
- entity_text: The entity name as it appears in the document
- entity_type: One of the categories above (lowercase with underscores)
- confidence: How confident you are this is a valid entity (0.0-1.0)
- context: Brief surrounding text showing how the entity is used (max 100 chars)

Return ONLY valid JSON in this exact format:
{{
    "entities": [
        {{
            "entity_text": "VIC-II",
            "entity_type": "hardware",
            "confidence": 0.95,
            "context": "The VIC-II chip controls all graphics"
        }},
        {{
            "entity_text": "$D020",
            "entity_type": "memory_address",
            "confidence": 0.98,
            "context": "Border color is controlled by $D020"
        }}
    ]
}}

Important:
- Extract 20-50 entities maximum
- Include confidence scores (0.0-1.0)
- Provide brief context snippets
- Preserve original capitalization/format
- Return ONLY JSON, no other text
"""

        # Extract entities using regex patterns first (fast, high-confidence)
        self.logger.info("Extracting entities using regex patterns")
        regex_entities = self._extract_entities_regex(sample_text)
        self.logger.info(f"Regex extraction found {len(regex_entities)} entities")

        # Call LLM for additional entity extraction
        self.logger.info(f"Calling LLM for entity extraction ({len(sample_text)} chars)")
        llm_entities = []
        try:
            response = llm_client.call_json(prompt, max_tokens=2048, temperature=0.3)
            llm_entities = response.get('entities', [])
            self.logger.info(f"LLM returned {len(llm_entities)} entities")
        except Exception as e:
            self.logger.warning(f"LLM call failed, using regex-only extraction: {e}")
            # Continue with regex entities only

        # Combine regex and LLM entities
        all_entities = regex_entities + llm_entities

        # Filter by confidence threshold
        filtered_entities = [
            e for e in all_entities
            if e.get('confidence', 0) >= confidence_threshold
        ]

        # Enhanced deduplication with entity normalization
        entity_map = {}
        for entity in filtered_entities:
            # Normalize entity text for better matching
            normalized_text = self._normalize_entity_text(entity['entity_text'], entity['entity_type'])
            key = (normalized_text.lower(), entity['entity_type'])

            if key in entity_map:
                entity_map[key]['occurrence_count'] += 1

                # Combine confidences: prefer regex (higher quality), boost if both sources agree
                current_source = entity.get('source', 'llm')
                existing_source = entity_map[key].get('source', 'llm')

                if current_source == 'regex' and existing_source == 'llm':
                    # Regex trumps LLM (higher precision)
                    entity_map[key]['confidence'] = entity['confidence']
                    entity_map[key]['source'] = 'regex'
                elif current_source == 'llm' and existing_source == 'regex':
                    # Both sources agree - boost confidence slightly
                    entity_map[key]['confidence'] = min(0.99, entity_map[key]['confidence'] * 1.05)
                    entity_map[key]['source'] = 'both'
                else:
                    # Keep highest confidence
                    if entity['confidence'] > entity_map[key]['confidence']:
                        entity_map[key]['confidence'] = entity['confidence']
                        entity_map[key]['context'] = entity.get('context', '')
            else:
                entity_map[key] = {
                    'entity_text': normalized_text,  # Use normalized form
                    'entity_type': entity['entity_type'],
                    'confidence': entity['confidence'],
                    'context': entity.get('context', ''),
                    'occurrence_count': 1,
                    'source': entity.get('source', 'llm')
                }

        unique_entities = list(entity_map.values())

        # Store in database
        cursor = self.db_conn.cursor()
        try:
            # Check if document still exists (might have been removed during background processing)
            doc_exists = cursor.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()

            if not doc_exists:
                self.logger.warning(
                    f"Document {doc_id} no longer exists, skipping entity storage "
                    f"(likely removed during background extraction)"
                )
                # Return entities even though storage was skipped
                result = {
                    'doc_id': doc_id,
                    'doc_title': 'Document Removed',
                    'entity_count': len(unique_entities),
                    'entities_by_type': {},
                    'entities': unique_entities
                }
                return result

            # Delete existing entities for this document
            cursor.execute("DELETE FROM document_entities WHERE doc_id = ?", (doc_id,))

            # Insert new entities
            from datetime import datetime
            generated_at = datetime.now().isoformat()
            model = llm_client.model if hasattr(llm_client, 'model') else 'unknown'

            for i, entity in enumerate(unique_entities, 1):
                cursor.execute("""
                    INSERT INTO document_entities
                    (doc_id, entity_id, entity_text, entity_type, confidence, context,
                     occurrence_count, generated_at, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    i,
                    entity['entity_text'],
                    entity['entity_type'],
                    entity['confidence'],
                    entity.get('context', ''),
                    entity.get('occurrence_count', 1),
                    generated_at,
                    model
                ))

            self.db_conn.commit()
            self.logger.info(f"Stored {len(unique_entities)} entities for document {doc_id}")

        except Exception as e:
            self.db_conn.rollback()
            self.logger.exception("Failed to store entities in database")
            # Return entities even if storage failed
            pass

        # Build result
        types = {}
        for entity in unique_entities:
            entity_type = entity['entity_type']
            types[entity_type] = types.get(entity_type, 0) + 1

        result = {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'entities': unique_entities,
            'entity_count': len(unique_entities),
            'types': types
        }

        # Cache result for future access (expensive LLM operation)
        if self._entity_cache is not None:
            self._entity_cache[cache_key] = result
            self.logger.debug(f"Cached entity extraction result for document: {doc_id}")

        return result

    def _extraction_worker_loop(self):
        """Background worker that processes entity extraction jobs from the queue."""
        self.logger.info("Entity extraction worker started")

        while not self._extraction_shutdown.is_set():
            try:
                # Wait for a job with timeout to allow checking shutdown flag
                try:
                    job = self._extraction_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # task_done() must run on every path out of here. It used to
                # sit after the work, so a failure *before* the inner
                # try/except - the status='running' UPDATE hitting a locked
                # DB, say - skipped it, and any later queue.join() then
                # waited forever on a job nothing would retry.
                try:
                    self._process_extraction_job(job)
                except Exception as e:
                    self.logger.error(
                        f"Extraction job {job.get('job_id')} aborted: {e}", exc_info=True
                    )
                finally:
                    self._extraction_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in extraction worker loop: {e}", exc_info=True)

        self.logger.info("Entity extraction worker stopped")

    def _process_extraction_job(self, job: dict):
        """Run one queued job and record its outcome in extraction_jobs.

        Handles both job types the queue carries: 'entities' (LLM/regex entity
        extraction) and 'figures' (OCR of embedded PDF images).
        """
        doc_id = job['doc_id']
        job_id = job['job_id']
        confidence_threshold = job['confidence_threshold']
        job_type = job.get('job_type', 'entities')

        cursor = self.db_conn.cursor()

        # Claim the job atomically. Every Claude Code session runs its own
        # server process against this shared database, so startup recovery
        # (see _recover_extraction_jobs) can re-enqueue the same 'queued' row
        # in several of them at once; the WHERE status='queued' guard makes
        # exactly one process win the claim instead of both doing the work.
        cursor.execute("""
            UPDATE extraction_jobs
            SET status = 'running', started_at = ?
            WHERE job_id = ? AND status = 'queued'
        """, (datetime.now(timezone.utc).isoformat(), job_id))
        self.db_conn.commit()
        if cursor.rowcount == 0:
            self.logger.debug(f"Extraction job {job_id} already claimed elsewhere; skipping")
            return

        self.logger.info(f"Processing {job_type} job {job_id} for document {doc_id}")

        try:
            if job_type == 'figures':
                result = self.extract_document_figures(doc_id)
                # entities_extracted doubles as this job type's unit count -
                # figures whose OCR actually yielded text.
                produced = result.get('figures_with_text', 0)
                summary = (
                    f"{result.get('figures_found', 0)} figure(s), "
                    f"{produced} with text"
                )
            else:
                result = self.extract_entities(
                    doc_id=doc_id,
                    confidence_threshold=confidence_threshold,
                    force_regenerate=False
                )
                produced = result.get('entity_count', 0)
                summary = f"{produced} entities extracted"

            cursor.execute("""
                UPDATE extraction_jobs
                SET status = 'completed',
                    completed_at = ?,
                    entities_extracted = ?
                WHERE job_id = ?
            """, (
                datetime.now(timezone.utc).isoformat(),
                produced,
                job_id
            ))
            self.db_conn.commit()

            self.logger.info(f"Completed {job_type} job {job_id}: {summary}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Extraction job {job_id} failed: {error_msg}")

            cursor.execute("""
                UPDATE extraction_jobs
                SET status = 'failed',
                    completed_at = ?,
                    error_message = ?
                WHERE job_id = ?
            """, (datetime.now(timezone.utc).isoformat(), error_msg, job_id))
            self.db_conn.commit()

    def _recover_extraction_jobs(self):
        """Re-queue extraction jobs orphaned by a process that exited mid-flight.

        extraction_jobs rows outlive the in-memory queue that drives them, so
        a 'queued' row whose process died was never retried by anyone - it sat
        at 'queued' forever while get_extraction_status kept reporting it as
        pending work that would never actually happen.
        """
        if os.getenv('TDZ_RECOVER_EXTRACTION_JOBS', '1') != '1':
            return

        stale_minutes = int(os.getenv('TDZ_EXTRACTION_STALE_MINUTES', '60'))
        limit = int(os.getenv('TDZ_EXTRACTION_RECOVER_LIMIT', '100'))

        try:
            cursor = self.db_conn.cursor()
            now = datetime.now(timezone.utc)

            # A 'running' row can legitimately belong to a live sibling
            # process, so only reap ones old enough that no plausible
            # extraction could still be working on them.
            cutoff = (now - timedelta(minutes=stale_minutes)).isoformat()
            cursor.execute("""
                UPDATE extraction_jobs
                SET status = 'failed', completed_at = ?, error_message = ?
                WHERE status = 'running' AND (started_at IS NULL OR started_at < ?)
            """, (
                now.isoformat(),
                f"Interrupted: no progress for over {stale_minutes} minutes (process likely exited)",
                cutoff,
            ))
            reaped = cursor.rowcount
            self.db_conn.commit()

            # Bounded: a data dir with a large backlog of stale 'queued' rows
            # would otherwise kick off that entire backlog (potentially paid
            # LLM calls) on every single session start.
            cursor.execute("""
                SELECT job_id, doc_id, confidence_threshold,
                       COALESCE(job_type, 'entities')
                FROM extraction_jobs
                WHERE status = 'queued' ORDER BY queued_at LIMIT ?
            """, (limit,))

            requeued = 0
            for job_id, doc_id, confidence_threshold, job_type in cursor.fetchall():
                if doc_id not in self.documents:
                    continue  # document is gone; the FK cascade clears the row
                self._extraction_queue.put({
                    'job_id': job_id,
                    'doc_id': doc_id,
                    'confidence_threshold': confidence_threshold,
                    'job_type': job_type,
                })
                requeued += 1

            if reaped or requeued:
                self.logger.info(
                    f"Extraction job recovery: re-queued {requeued}, "
                    f"marked {reaped} stale 'running' job(s) failed"
                )
        except Exception as e:
            # Never block startup on recovery - the server is fully usable
            # without it.
            self.logger.warning(f"Extraction job recovery skipped: {e}")

    def queue_entity_extraction(self, doc_id: str,
                                confidence_threshold: float = 0.6,
                                skip_if_exists: bool = True) -> dict:
        """
        Queue a document for background entity extraction.

        Args:
            doc_id: Document ID to extract entities from
            confidence_threshold: Minimum confidence threshold (0.0-1.0, default: 0.6)
            skip_if_exists: If True, skip if entities already exist or job is queued

        Returns:
            {
                'queued': bool,          # True if job was queued
                'job_id': int,           # Job ID if queued
                'reason': str,           # Reason if not queued
                'existing_job_id': int   # Existing job ID if already queued
            }
        """
        # Validate document exists
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        # Check if entities already exist
        if skip_if_exists:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                return {
                    'queued': False,
                    'reason': f"Document already has {existing_count} entities",
                    'existing_entities': existing_count
                }

            # Check if job already queued or running. Scoped to entity jobs:
            # the same table now also holds 'figures' jobs, and a pending
            # figure OCR must not look like pending entity extraction.
            cursor.execute("""
                SELECT job_id, status FROM extraction_jobs
                WHERE doc_id = ? AND status IN ('queued', 'running')
                  AND COALESCE(job_type, 'entities') = 'entities'
                ORDER BY queued_at DESC
                LIMIT 1
            """, (doc_id,))
            existing_job = cursor.fetchone()
            if existing_job:
                return {
                    'queued': False,
                    'reason': f"Extraction already {existing_job[1]} for this document",
                    'existing_job_id': existing_job[0]
                }

        # Create extraction job record
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO extraction_jobs (doc_id, status, confidence_threshold, queued_at)
            VALUES (?, 'queued', ?, ?)
        """, (doc_id, confidence_threshold, datetime.now(timezone.utc).isoformat()))
        self.db_conn.commit()

        job_id = cursor.lastrowid

        # Add job to queue
        self._extraction_queue.put({
            'job_id': job_id,
            'doc_id': doc_id,
            'confidence_threshold': confidence_threshold,
            'job_type': 'entities',
        })

        self.logger.info(f"Queued entity extraction job {job_id} for document {doc_id}")

        return {
            'queued': True,
            'job_id': job_id
        }

    def get_extraction_status(self, doc_id: str) -> dict:
        """
        Get the entity extraction status for a document.

        Args:
            doc_id: Document ID

        Returns:
            {
                'doc_id': str,
                'has_entities': bool,
                'entity_count': int,
                'jobs': [list of job dicts with status, queued_at, etc.]
            }
        """
        # Check if entities exist
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (doc_id,))
        entity_count = cursor.fetchone()[0]

        # Get extraction jobs for this document. job_type is surfaced because
        # this table now carries figure-OCR jobs too, and a caller seeing a
        # bare 'running' row would otherwise assume it was entity work.
        cursor.execute("""
            SELECT job_id, status, confidence_threshold, queued_at,
                   started_at, completed_at, error_message, entities_extracted,
                   COALESCE(job_type, 'entities')
            FROM extraction_jobs
            WHERE doc_id = ?
            ORDER BY queued_at DESC
        """, (doc_id,))

        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'job_id': row[0],
                'status': row[1],
                'confidence_threshold': row[2],
                'queued_at': row[3],
                'started_at': row[4],
                'completed_at': row[5],
                'error_message': row[6],
                'entities_extracted': row[7],
                'job_type': row[8]
            })

        return {
            'doc_id': doc_id,
            'has_entities': entity_count > 0,
            'entity_count': entity_count,
            'jobs': jobs
        }

    def get_all_extraction_jobs(self, status_filter: Optional[str] = None,
                                limit: int = 100) -> list[dict]:
        """
        Get all entity extraction jobs.

        Args:
            status_filter: Optional status filter ('queued', 'running', 'completed', 'failed')
            limit: Maximum number of jobs to return (default: 100)

        Returns:
            List of job dicts with doc_id, status, timestamps, etc.
        """
        cursor = self.db_conn.cursor()

        if status_filter:
            cursor.execute("""
                SELECT j.job_id, j.doc_id, d.title, j.status, j.confidence_threshold,
                       j.queued_at, j.started_at, j.completed_at, j.error_message,
                       j.entities_extracted
                FROM extraction_jobs j
                LEFT JOIN documents d ON j.doc_id = d.doc_id
                WHERE j.status = ?
                ORDER BY j.queued_at DESC
                LIMIT ?
            """, (status_filter, limit))
        else:
            cursor.execute("""
                SELECT j.job_id, j.doc_id, d.title, j.status, j.confidence_threshold,
                       j.queued_at, j.started_at, j.completed_at, j.error_message,
                       j.entities_extracted
                FROM extraction_jobs j
                LEFT JOIN documents d ON j.doc_id = d.doc_id
                ORDER BY j.queued_at DESC
                LIMIT ?
            """, (limit,))

        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                'job_id': row[0],
                'doc_id': row[1],
                'doc_title': row[2],
                'status': row[3],
                'confidence_threshold': row[4],
                'queued_at': row[5],
                'started_at': row[6],
                'completed_at': row[7],
                'error_message': row[8],
                'entities_extracted': row[9]
            })

        return jobs

    def get_entities(self, doc_id: str,
                    entity_types: Optional[list[str]] = None,
                    min_confidence: float = 0.0) -> dict:
        """
        Get all entities for a specific document.

        Args:
            doc_id: Document ID
            entity_types: Optional list of entity types to filter by
                         (e.g., ['hardware', 'memory_address'])
            min_confidence: Minimum confidence threshold (default: 0.0)

        Returns:
            {
                'doc_id': str,
                'doc_title': str,
                'entities': [list of entity dicts],
                'entity_count': int,
                'types': {'hardware': count, ...}
            }

        Example:
            # Get all entities
            result = kb.get_entities('my-doc-id')

            # Get only hardware entities with high confidence
            result = kb.get_entities('my-doc-id',
                                    entity_types=['hardware'],
                                    min_confidence=0.8)
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        cursor = self.db_conn.cursor()

        # Build query with optional filters
        query = """
            SELECT entity_text, entity_type, confidence, context, occurrence_count
            FROM document_entities
            WHERE doc_id = ? AND confidence >= ?
        """
        params = [doc_id, min_confidence]

        if entity_types:
            placeholders = ','.join(['?'] * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += " ORDER BY entity_type, confidence DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        entities = []
        types = {}
        for row in rows:
            entity = {
                'entity_text': row[0],
                'entity_type': row[1],
                'confidence': row[2],
                'context': row[3],
                'occurrence_count': row[4]
            }
            entities.append(entity)

            entity_type = row[1]
            types[entity_type] = types.get(entity_type, 0) + 1

        return {
            'doc_id': doc_id,
            'doc_title': doc.title,
            'entities': entities,
            'entity_count': len(entities),
            'types': types
        }

    def search_entities(self, query: str,
                       entity_types: Optional[list[str]] = None,
                       min_confidence: float = 0.0,
                       max_results: int = 20) -> dict:
        """
        Search for entities across all documents using full-text search.

        Args:
            query: Search query (e.g., "VIC-II", "sprite", "$D000")
            entity_types: Filter by entity types (e.g., ['hardware', 'memory_address'])
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of results to return

        Returns:
            Dictionary with search results grouped by document:
            {
                'query': str,
                'total_matches': int,
                'documents': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'matches': [
                            {
                                'entity_text': str,
                                'entity_type': str,
                                'confidence': float,
                                'context': str,
                                'occurrence_count': int
                            },
                            ...
                        ],
                        'match_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Search for VIC-II chip mentions
            results = kb.search_entities('VIC-II')

            # Search for memory addresses only
            results = kb.search_entities('$D0', entity_types=['memory_address'])

            # Search with confidence threshold
            results = kb.search_entities('sprite', min_confidence=0.7)
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Build FTS5 query
        # Escape special FTS5 characters and wrap in quotes for literal search
        escaped_query = query.strip().replace('"', '""')
        fts_query = f'"{escaped_query}"'

        # Build WHERE clause for filtering
        where_clauses = []
        params = []

        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            where_clauses.append(f"e.entity_type IN ({placeholders})")
            params.extend(entity_types)

        if min_confidence > 0.0:
            where_clauses.append("e.confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND " + " AND ".join(where_clauses) if where_clauses else ""

        # Search entities_fts and join with document_entities for full data
        query_sql = f"""
            SELECT e.doc_id, e.entity_text, e.entity_type, e.confidence,
                   e.context, e.occurrence_count
            FROM entities_fts fts
            JOIN document_entities e ON fts.rowid = e.rowid
            WHERE entities_fts MATCH ?{where_clause}
            ORDER BY rank
            LIMIT ?
        """

        # Execute search
        cursor = self.db_conn.cursor()
        cursor.execute(query_sql, [fts_query] + params + [max_results * 10])  # Get extra for grouping
        rows = cursor.fetchall()

        # Group results by document
        doc_matches = {}
        for row in rows:
            doc_id, entity_text, entity_type, confidence, context, occurrence_count = row

            if doc_id not in doc_matches:
                doc_matches[doc_id] = []

            doc_matches[doc_id].append({
                'entity_text': entity_text,
                'entity_type': entity_type,
                'confidence': confidence,
                'context': context or '',
                'occurrence_count': occurrence_count
            })

        # Build result list with document titles
        documents = []
        for doc_id, matches in list(doc_matches.items())[:max_results]:
            doc = self.documents.get(doc_id)
            if doc:
                documents.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'matches': matches,
                    'match_count': len(matches)
                })

        return {
            'query': query,
            'total_matches': sum(len(matches) for matches in doc_matches.values()),
            'documents': documents
        }

    def find_docs_by_entity(self, entity_text: str,
                           entity_type: Optional[str] = None,
                           min_confidence: float = 0.0,
                           max_results: int = 20) -> dict:
        """
        Find all documents that contain a specific entity.

        Args:
            entity_text: Exact entity text to search for (e.g., "VIC-II", "$D000")
            entity_type: Optional entity type filter (e.g., 'hardware', 'memory_address')
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_results: Maximum number of documents to return

        Returns:
            Dictionary with documents containing the entity:
            {
                'entity_text': str,
                'entity_type': str or None,
                'total_documents': int,
                'documents': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'entity_type': str,
                        'confidence': float,
                        'context': str,
                        'occurrence_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Find all documents mentioning VIC-II
            results = kb.find_docs_by_entity('VIC-II')

            # Find documents with $D000 memory address
            results = kb.find_docs_by_entity('$D000', entity_type='memory_address')

            # Find with confidence threshold
            results = kb.find_docs_by_entity('sprite', min_confidence=0.7)
        """
        if not entity_text or not entity_text.strip():
            raise ValueError("Entity text cannot be empty")

        # Build WHERE clause
        where_clauses = ["e.entity_text = ?"]
        params = [entity_text.strip()]

        if entity_type:
            where_clauses.append("e.entity_type = ?")
            params.append(entity_type)

        if min_confidence > 0.0:
            where_clauses.append("e.confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND ".join(where_clauses)

        # Query database
        query_sql = f"""
            SELECT e.doc_id, e.entity_type, e.confidence, e.context,
                   e.occurrence_count
            FROM document_entities e
            WHERE {where_clause}
            ORDER BY e.confidence DESC, e.occurrence_count DESC
            LIMIT ?
        """

        cursor = self.db_conn.cursor()
        cursor.execute(query_sql, params + [max_results])
        rows = cursor.fetchall()

        # Build result list with document titles
        documents = []
        for row in rows:
            doc_id, ent_type, confidence, context, occurrence_count = row
            doc = self.documents.get(doc_id)
            if doc:
                documents.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'entity_type': ent_type,
                    'confidence': confidence,
                    'context': context or '',
                    'occurrence_count': occurrence_count
                })

        return {
            'entity_text': entity_text.strip(),
            'entity_type': entity_type,
            'total_documents': len(documents),
            'documents': documents
        }

    def get_entity_stats(self, entity_type: Optional[str] = None) -> dict:
        """
        Get statistics about extracted entities in the knowledge base.

        Args:
            entity_type: Optional filter by entity type (e.g., 'hardware', 'memory_address')

        Returns:
            Dictionary with entity statistics:
            {
                'total_entities': int,
                'total_documents_with_entities': int,
                'by_type': {
                    'hardware': int,
                    'memory_address': int,
                    ...
                },
                'top_entities': [
                    {
                        'entity_text': str,
                        'entity_type': str,
                        'document_count': int,
                        'total_occurrences': int,
                        'avg_confidence': float
                    },
                    ...
                ],
                'documents_with_most_entities': [
                    {
                        'doc_id': str,
                        'doc_title': str,
                        'entity_count': int
                    },
                    ...
                ]
            }

        Examples:
            # Get overall statistics
            stats = kb.get_entity_stats()

            # Get statistics for hardware entities only
            stats = kb.get_entity_stats(entity_type='hardware')
        """
        cursor = self.db_conn.cursor()

        # Total entities
        if entity_type:
            cursor.execute(
                "SELECT COUNT(*) FROM document_entities WHERE entity_type = ?",
                (entity_type,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM document_entities")
        total_entities = cursor.fetchone()[0]

        # Total documents with entities
        if entity_type:
            cursor.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM document_entities WHERE entity_type = ?",
                (entity_type,)
            )
        else:
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM document_entities")
        total_docs = cursor.fetchone()[0]

        # Breakdown by type
        by_type = {}
        if entity_type:
            by_type[entity_type] = total_entities
        else:
            cursor.execute("""
                SELECT entity_type, COUNT(*)
                FROM document_entities
                GROUP BY entity_type
                ORDER BY COUNT(*) DESC
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

        # Top entities by document count
        type_filter = "WHERE entity_type = ?" if entity_type else ""
        params = [entity_type] if entity_type else []

        cursor.execute(f"""
            SELECT entity_text, entity_type,
                   COUNT(DISTINCT doc_id) as doc_count,
                   SUM(occurrence_count) as total_occurrences,
                   AVG(confidence) as avg_confidence
            FROM document_entities
            {type_filter}
            GROUP BY entity_text, entity_type
            ORDER BY doc_count DESC, total_occurrences DESC
            LIMIT 20
        """, params)

        top_entities = [
            {
                'entity_text': row[0],
                'entity_type': row[1],
                'document_count': row[2],
                'total_occurrences': row[3],
                'avg_confidence': round(row[4], 3)
            }
            for row in cursor.fetchall()
        ]

        # Documents with most entities
        cursor.execute(f"""
            SELECT doc_id, COUNT(*) as entity_count
            FROM document_entities
            {type_filter}
            GROUP BY doc_id
            ORDER BY entity_count DESC
            LIMIT 10
        """, params)

        docs_with_most = []
        for row in cursor.fetchall():
            doc_id = row[0]
            entity_count = row[1]
            doc = self.documents.get(doc_id)
            if doc:
                docs_with_most.append({
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'entity_count': entity_count
                })

        return {
            'total_entities': total_entities,
            'total_documents_with_entities': total_docs,
            'by_type': by_type,
            'top_entities': top_entities,
            'documents_with_most_entities': docs_with_most
        }

    def export_entities(self, format: str = 'csv',
                       entity_types: Optional[list] = None,
                       min_confidence: float = 0.0,
                       output_path: Optional[str] = None) -> str:
        """
        Export entities to CSV or JSON format.

        Args:
            format: 'csv' or 'json'
            entity_types: Filter by entity types (None = all types)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            output_path: Optional file path to write to (if None, returns string)

        Returns:
            Exported data as string (or writes to file if output_path provided)

        Example CSV:
            entity_text,entity_type,confidence,doc_count,occurrence_count,first_seen_doc
            VIC-II,hardware,0.95,15,87,89d0943d6009
            SID,hardware,0.92,12,56,89d0943d6009

        Example JSON:
            [
                {
                    "entity_text": "VIC-II",
                    "entity_type": "hardware",
                    "confidence": 0.95,
                    "doc_count": 15,
                    "occurrence_count": 87,
                    "first_seen_doc": "89d0943d6009"
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
            SELECT entity_text, entity_type,
                   AVG(confidence) as avg_confidence,
                   COUNT(DISTINCT doc_id) as doc_count,
                   SUM(occurrence_count) as total_occurrences,
                   MIN(doc_id) as first_seen_doc
            FROM document_entities
            WHERE confidence >= ?
        """
        params = [min_confidence]

        if entity_types:
            placeholders = ','.join(['?'] * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += """
            GROUP BY entity_text, entity_type
            ORDER BY doc_count DESC, total_occurrences DESC
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if format.lower() == 'csv':
            output = StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(['entity_text', 'entity_type', 'avg_confidence',
                           'doc_count', 'total_occurrences', 'first_seen_doc'])

            # Write data
            for row in rows:
                writer.writerow([
                    row[0],  # entity_text
                    row[1],  # entity_type
                    f"{row[2]:.3f}",  # avg_confidence
                    row[3],  # doc_count
                    row[4],  # total_occurrences
                    row[5][:12]  # first_seen_doc (truncated)
                ])

            result = output.getvalue()
            output.close()

        elif format.lower() == 'json':
            entities = []
            for row in rows:
                entities.append({
                    'entity_text': row[0],
                    'entity_type': row[1],
                    'avg_confidence': round(row[2], 3),
                    'doc_count': row[3],
                    'total_occurrences': row[4],
                    'first_seen_doc': row[5][:12]
                })

            result = json.dumps(entities, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")

        # Write to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            self.logger.info(f"Exported {len(rows)} entities to {output_path}")

        return result

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

    def extract_entities_bulk(self, confidence_threshold: float = 0.6,
                             force_regenerate: bool = False,
                             max_docs: Optional[int] = None,
                             skip_existing: bool = True) -> dict:
        """
        Bulk extract entities for multiple documents.

        Args:
            confidence_threshold: Minimum confidence threshold (0.0-1.0, default: 0.6)
            force_regenerate: If True, re-extract even if entities already exist
            max_docs: Maximum number of documents to process (None = all)
            skip_existing: If True, skip documents that already have entities (unless force_regenerate)

        Returns:
            {
                'processed': int,
                'failed': int,
                'skipped': int,
                'total_entities': int,
                'by_type': {'hardware': int, 'memory_address': int, ...},
                'results': [list of individual results]
            }

        Examples:
            # Extract entities for all documents
            results = kb.extract_entities_bulk()

            # Extract for first 10 documents only
            results = kb.extract_entities_bulk(max_docs=10)

            # Force re-extraction for all documents
            results = kb.extract_entities_bulk(force_regenerate=True)
        """
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_entities': 0,
            'by_type': {},
            'results': []
        }

        # Get documents to process
        docs_to_process = list(self.documents.keys())

        if max_docs:
            docs_to_process = docs_to_process[:max_docs]

        self.logger.info(f"Extracting entities for {len(docs_to_process)} documents "
                        f"(confidence threshold: {confidence_threshold})")

        # Process each document
        for i, doc_id in enumerate(docs_to_process, 1):
            doc_results = {
                'doc_id': doc_id,
                'title': self.documents[doc_id].title,
                'status': 'unknown',
                'entity_count': 0,
                'error': None
            }

            try:
                # Check if entities already exist (unless force_regenerate)
                if skip_existing and not force_regenerate:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM document_entities WHERE doc_id = ?",
                        (doc_id,)
                    )
                    existing_count = cursor.fetchone()[0]

                    if existing_count > 0:
                        self.logger.info(f"[{i}/{len(docs_to_process)}] Skipping {doc_id} (already has {existing_count} entities)")
                        doc_results['status'] = 'skipped'
                        doc_results['entity_count'] = existing_count
                        results['skipped'] += 1
                        results['results'].append(doc_results)
                        continue

                # Extract entities
                self.logger.info(f"[{i}/{len(docs_to_process)}] Extracting entities from {doc_id}")

                result = self.extract_entities(
                    doc_id,
                    confidence_threshold=confidence_threshold,
                    force_regenerate=force_regenerate
                )

                doc_results['status'] = 'success'
                doc_results['entity_count'] = result['entity_count']

                # Update counts
                results['processed'] += 1
                results['total_entities'] += result['entity_count']

                # Update by_type counts
                for entity_type, count in result['types'].items():
                    results['by_type'][entity_type] = results['by_type'].get(entity_type, 0) + count

                results['results'].append(doc_results)

            except Exception as e:
                self.logger.error(f"[{i}/{len(docs_to_process)}] Failed to extract entities from {doc_id}: {e}")
                doc_results['status'] = 'failed'
                doc_results['error'] = str(e)
                results['failed'] += 1
                results['results'].append(doc_results)

        self.logger.info(f"Bulk entity extraction complete: processed={results['processed']}, "
                        f"failed={results['failed']}, skipped={results['skipped']}, "
                        f"total_entities={results['total_entities']}")

        return results

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
