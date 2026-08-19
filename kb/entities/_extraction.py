"""Entity extraction and the background extraction-job queue for EntitiesMixin.

Split out of kb/entities.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Optional
import os
import queue
import re


class _ExtractionMixin:

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
