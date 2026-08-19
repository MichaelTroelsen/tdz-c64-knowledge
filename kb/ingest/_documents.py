"""Document lifecycle (add/update/remove/scrape/bulk) for IngestMixin.

Split out of kb/ingest.py by R12 step 4 (module-size reduction). Move, not a
rewrite: every method body below is unchanged from the original.
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
from models import DocumentChunk
from models import DocumentMeta
from models import DocumentNotFoundError
from models import KnowledgeBaseError
from models import ProgressCallback
from models import ProgressUpdate
from models import SecurityError
from pathlib import Path
from text_utils import _expand_brace_pattern
from typing import Optional
from util import USER_AGENT
from util import _retry_on_db_locked
from util import http_get_polite
from util import http_headers
from util import robots_allows
import json
import os
import queue
import re
import sqlite3


class _DocumentsMixin:

    def _find_mdscrape_executable(self) -> Optional[str]:
        """Find mdscrape executable in common locations.

        Returns:
            Path to mdscrape executable, or None if not found
        """
        import shutil

        # Check if mdscrape is in PATH
        mdscrape = shutil.which('mdscrape')
        if mdscrape:
            self.logger.info(f"Found mdscrape in PATH: {mdscrape}")
            return mdscrape

        # Check common Windows/Linux paths
        common_paths = [
            Path(r'C:\Users\mit\claude\mdscrape\mdscrape.exe'),  # User-specified location
            Path(r'C:\Users\mit\claude\mdscrape\mdscrape'),
            Path.home() / 'claude' / 'mdscrape' / 'mdscrape.exe',
            Path.home() / 'claude' / 'mdscrape' / 'mdscrape',
            Path(__file__).parent.parent / 'mdscrape' / 'mdscrape.exe',
            Path(__file__).parent.parent / 'mdscrape' / 'mdscrape',
        ]

        for path in common_paths:
            if path.exists():
                self.logger.info(f"Found mdscrape at: {path}")
                return str(path)

        # Check MDSCRAPE_PATH environment variable
        env_path = os.environ.get('MDSCRAPE_PATH')
        if env_path:
            path = Path(env_path)
            if path.exists():
                self.logger.info(f"Found mdscrape via MDSCRAPE_PATH: {path}")
                return str(path)

        self.logger.warning("mdscrape executable not found. Install from: https://github.com/MichaelTroelsen/mdscrape")
        return None

    def _extract_source_url_from_md(self, md_file: Path) -> Optional[str]:
        """Extract source URL from YAML frontmatter in markdown file.

        Args:
            md_file: Path to markdown file

        Returns:
            Source URL if found, None otherwise
        """
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse YAML frontmatter (between --- delimiters)
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    # Simple YAML parsing for 'source:' or 'url:' field
                    for line in frontmatter.split('\n'):
                        line = line.strip()
                        if line.startswith('source:') or line.startswith('url:'):
                            # Extract URL after colon
                            url = line.split(':', 1)[1].strip().strip('"\'')
                            if url:
                                return url
        except Exception as e:
            self.logger.warning(f"Failed to extract URL from {md_file}: {e}")

        return None

    def _add_scraped_document(self, filepath: str, source_url: str, title: Optional[str],
                              tags: Optional[list[str]], scrape_config: str,
                              scrape_date: str) -> DocumentMeta:
        """Add a scraped markdown document with URL metadata.

        Args:
            filepath: Path to scraped markdown file
            source_url: Original URL that was scraped
            title: Optional title for document
            tags: Optional list of tags
            scrape_config: JSON string with scraping configuration
            scrape_date: ISO timestamp of scrape

        Returns:
            DocumentMeta object for added document
        """
        # First, add document using normal flow
        doc = self.add_document(filepath, title, tags)

        # Compute content hash for change detection
        url_content_hash = self._compute_file_hash(filepath)

        # Update database with URL metadata
        with self._lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET source_url = ?,
                    scrape_date = ?,
                    scrape_config = ?,
                    scrape_status = 'success',
                    url_content_hash = ?
                WHERE doc_id = ?
            """, (source_url, scrape_date, scrape_config, url_content_hash, doc.doc_id))

            self.db_conn.commit()

        # Update in-memory object
        doc.source_url = source_url
        doc.scrape_date = scrape_date
        doc.scrape_config = scrape_config
        doc.scrape_status = 'success'
        doc.url_content_hash = url_content_hash

        # Update in documents dict
        self.documents[doc.doc_id] = doc

        self.logger.info(f"Added scraped document: {doc.title} (from {source_url})")
        return doc

    def _is_path_allowed(self, filepath: str) -> bool:
        """
        Check if a file path is within allowed directories.

        Args:
            filepath: Path to check

        Returns:
            True if path is allowed (or no restrictions configured), False otherwise
        """
        # No restrictions if allowed_dirs not configured
        if not self.allowed_dirs:
            return True

        # Resolve to absolute path to prevent path traversal
        try:
            resolved_path = Path(filepath).resolve()
        except (OSError, ValueError):
            # Invalid path
            return False

        # Check if path is within any allowed directory
        return any(
            resolved_path.is_relative_to(allowed_dir)
            for allowed_dir in self.allowed_dirs
        )

    def get_document_by_card_id(self, card_id: str, include_superseded: bool = False) -> Optional[DocumentMeta]:
        """Resolve a card's logical id to its document.

        Returns the live (non-superseded) card by default - by construction
        there is at most one, since add_document/update_document refuse to
        create a second live document for the same card_id. With
        include_superseded=True, falls back to the most recently indexed
        superseded version if no live one exists.
        """
        matches = [d for d in self.documents.values() if d.card_id == card_id]
        live = [d for d in matches if not d.superseded_by]
        if live:
            return live[0]
        if include_superseded and matches:
            return sorted(matches, key=lambda d: d.indexed_at, reverse=True)[0]
        return None

    def _rebuild_entity_relationships(self) -> None:
        """Recompute entity_relationships from scratch across all live documents.

        entity_relationships has no per-document attribution - it's a running
        aggregate keyed only on (entity1_text, entity2_text, relationship_type)
        with no doc_id column - so a superseded document's contribution can't
        be surgically subtracted. The only correct fix is to wipe the table
        and rebuild it from whatever document_entities currently holds for
        live documents.
        """
        cursor = self.db_conn.cursor()
        cursor.execute("DELETE FROM entity_relationships")
        self.db_conn.commit()

        cursor.execute("SELECT DISTINCT doc_id FROM document_entities")
        doc_ids = [row[0] for row in cursor.fetchall()]
        for doc_id in doc_ids:
            doc = self.documents.get(doc_id)
            if doc is None or doc.superseded_by:
                continue
            try:
                self.extract_entity_relationships(doc_id, force_regenerate=True)
            except Exception as e:
                self.logger.warning(f"Relationship rebuild failed for {doc_id}: {e}")

    def _mark_superseded(self, old_doc_id: str, new_doc_id: str) -> None:
        """Mark old_doc_id as superseded by new_doc_id and purge its contribution
        from derived artifacts that would otherwise keep answering with retracted
        content (entities, entity relationships, cached graph artifacts).

        Chunks/embeddings for old_doc_id are deliberately left in place - the
        card's prior content stays retrievable by doc_id for history/audit -
        but old_doc_id is excluded from default search results and from
        get_document_by_card_id once superseded_by is set.
        """
        if old_doc_id not in self.documents or old_doc_id == new_doc_id:
            return

        cursor = self.db_conn.cursor()
        cursor.execute(
            "UPDATE documents SET superseded_by = ? WHERE doc_id = ?",
            (new_doc_id, old_doc_id)
        )
        self.db_conn.commit()
        self.documents[old_doc_id].superseded_by = new_doc_id

        # Purge stale entities contributed by the retracted content, then
        # rebuild the globally-aggregated relationship table from what's left
        # so stale co-occurrence edges can't survive it.
        cursor.execute("SELECT COUNT(*) FROM document_entities WHERE doc_id = ?", (old_doc_id,))
        had_entities = cursor.fetchone()[0] > 0
        if had_entities:
            cursor.execute("DELETE FROM document_entities WHERE doc_id = ?", (old_doc_id,))
            self.db_conn.commit()
            if self._entity_cache is not None:
                self._entity_cache.clear()
            try:
                self.extract_entities(new_doc_id, confidence_threshold=0.6, force_regenerate=True)
            except Exception as e:
                self.logger.warning(f"Entity re-extraction for {new_doc_id} failed after supersede: {e}")
            self._rebuild_entity_relationships()

        # Drop cached graph artifacts so they can't keep answering with
        # retracted claims - build_knowledge_graph rebuilds from
        # document_entities/entity_relationships on every call, but other
        # code paths may read these caches directly.
        for table in ("graph_cache", "graph_metrics", "graph_paths"):
            try:
                cursor.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        self.db_conn.commit()

        # Invalidate search caches - the old doc is now excluded from default results.
        self._invalidate_caches()

        self.logger.info(f"Marked {old_doc_id} as superseded by {new_doc_id}")

    def add_document(self, filepath: str, title: Optional[str] = None, tags: Optional[list[str]] = None,
                     progress_callback: ProgressCallback = None, replace: bool = False) -> DocumentMeta:
        """Add a document to the knowledge base.

        Args:
            filepath: Path to the document file
            title: Optional title for the document
            tags: Optional list of tags
            progress_callback: Optional callback for progress updates
            replace: If the file is a knowledge card (has a ```json id``` block)
                and a live card with the same id already exists, add_document
                refuses by default (raises KnowledgeBaseError naming the
                existing doc id). Pass replace=True to supersede it instead.
                Non-card documents (no json id block) are never affected by
                this check and are always created, matching prior behavior.
        """
        # Report progress: Start
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=0,
                total=4,
                message="Starting document ingestion",
                item=filepath
            ))

        # Resolve to absolute path to prevent path traversal
        resolved_path = Path(filepath).resolve()

        # Security: Validate path is within allowed directories
        if not self._is_path_allowed(filepath):
            self.logger.error(f"Security violation: Path outside allowed directories: {resolved_path}")
            raise SecurityError(
                f"Path outside allowed directories. File must be within: {self.allowed_dirs}"
            )

        filepath = str(resolved_path)
        self.logger.info(f"Adding document: {filepath}")

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise DocumentNotFoundError(f"File not found: {filepath}")

        filename = os.path.basename(filepath)

        # Extract text based on file type
        text, file_type, total_pages, pdf_metadata = self._extract_text_for_file(filepath)

        # Report progress: Text extraction complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=1,
                total=4,
                message=f"Text extraction complete ({len(text)} characters)",
                item=filename
            ))

        # Extract tables from PDFs
        tables = []
        if file_type == 'pdf':
            tables = self._extract_tables(filepath)
            if tables:
                self.logger.info(f"Extracted {len(tables)} tables from PDF")

        # Detect code blocks in text
        code_blocks = self._detect_code_blocks(text)
        if code_blocks:
            self.logger.info(f"Detected {len(code_blocks)} code blocks")

        # Extract facets for faceted search
        facets = self._extract_facets(text)
        facet_count = sum(len(values) for values in facets.values())
        if facet_count > 0:
            self.logger.info(f"Extracted {facet_count} facets ({len(facets['hardware'])} hardware, {len(facets['instruction'])} instructions, {len(facets['register'])} registers)")

        # Generate content-based doc_id for deduplication
        doc_id = self._generate_doc_id(filepath, text)

        # Parse the card's logical identity, if this is a knowledge card
        card_id = self._extract_card_id(text)

        # Thread-safe duplicate check
        superseded_doc_id = None
        with self._lock:
            # Check for duplicate content
            if doc_id in self.documents:
                existing_doc = self.documents[doc_id]
                self.logger.warning(f"Duplicate content detected: {filepath}")
                self.logger.warning(f"  Matches existing document: {existing_doc.filepath}")
                self.logger.info(f"Skipping duplicate - returning existing document {doc_id}")
                return existing_doc

            # Card-identity guard: refuse to silently fork a card that already
            # exists under the same logical id. This is the fix for the
            # "two live documents both claim id: X" failure mode - callers
            # must explicitly opt into replacing via replace=True or
            # update_document().
            if card_id:
                existing_card = self.get_document_by_card_id(card_id, include_superseded=False)
                if existing_card and existing_card.doc_id != doc_id:
                    if not replace:
                        raise KnowledgeBaseError(
                            f"Card '{card_id}' already exists as document {existing_card.doc_id} "
                            f"('{existing_card.title}'). Use update_document() to replace it, "
                            f"or pass replace=true to add_document()."
                        )
                    superseded_doc_id = existing_card.doc_id

        # Create chunks
        text_chunks = self._chunk_text(text)
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Estimate page number for PDFs based on PAGE BREAK markers
            page_num = None
            if file_type == 'pdf' and '--- PAGE BREAK ---' in text:
                # Count PAGE BREAK markers before this chunk
                chunk_start_pos = text.find(chunk_text[:100])  # Find chunk in full text
                if chunk_start_pos >= 0:
                    page_breaks_before = text[:chunk_start_pos].count('--- PAGE BREAK ---')
                    page_num = page_breaks_before + 1  # Pages are 1-indexed

            chunk = DocumentChunk(
                doc_id=doc_id,
                filename=filename,
                title=title or filename,
                chunk_id=i,
                page=page_num,
                content=chunk_text,
                word_count=len(chunk_text.split())
            )
            chunks.append(chunk)

        # Report progress: Chunking complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=2,
                total=4,
                message=f"Created {len(chunks)} chunks",
                item=filename
            ))

        # Compute file modification time and content hash for update detection
        file_mtime = os.path.getmtime(resolved_path)
        file_hash = self._compute_file_hash(resolved_path)

        # Extract cross-references for content linking
        cross_refs = self._extract_cross_references(chunks, doc_id)
        if cross_refs:
            self.logger.info(f"Extracted {len(cross_refs)} cross-references")

        # Create metadata
        doc_meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            title=title or filename,
            filepath=filepath,
            file_type=file_type,
            total_pages=total_pages,
            total_chunks=len(chunks),
            indexed_at=datetime.now().isoformat(),
            tags=tags or [],
            author=pdf_metadata.get('author'),
            subject=pdf_metadata.get('subject'),
            creator=pdf_metadata.get('creator'),
            creation_date=pdf_metadata.get('creation_date'),
            file_mtime=file_mtime,
            file_hash=file_hash,
            card_id=card_id
        )

        # Thread-safe database insertion and cache invalidation
        with self._lock:
            # Add to database (with tables, code blocks, facets, and cross-references)
            _retry_on_db_locked(
                self._add_document_db, doc_meta, chunks,
                tables=tables, code_blocks=code_blocks, facets=facets, cross_refs=cross_refs
            )
            self.documents[doc_id] = doc_meta

            # Report progress: Database insertion complete
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation="add_document",
                    current=3,
                    total=4,
                    message="Stored in database",
                    item=filename
                ))

            # Invalidate BM25 index (will be rebuilt on next search)
            self.bm25 = None

            # Incrementally add chunks to embeddings (faster than full rebuild).
            # Must load the model first: _add_chunks_to_embeddings silently
            # no-ops when embeddings_model is None, which it is for every
            # process that hasn't yet run a semantic search - i.e. every
            # ingest-only session. Without this, newly-added documents are
            # never embedded and there is no error to notice.
            if self.use_semantic:
                self._ensure_embeddings_loaded()
                if self.embeddings_model is not None:
                    self._add_chunks_to_embeddings(chunks)
                else:
                    self.logger.warning(
                        f"Skipping embeddings for {doc_id}: embeddings model unavailable"
                    )

            # Update query suggestions with new terms
            self._update_suggestions_for_chunks(chunks)

            # Invalidate search caches
            self._invalidate_caches()

        # If this replaces an existing card, retire the old one and refresh
        # everything derived from its (now-retracted) content.
        if superseded_doc_id:
            self._mark_superseded(superseded_doc_id, doc_id)

        self.logger.info(f"Successfully indexed document {doc_id}: {filename} ({len(chunks)} chunks)")

        # Report progress: Complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_document",
                current=4,
                total=4,
                message="Document indexed successfully",
                item=filename
            ))

        # Auto-queue entity extraction (configurable via environment variable)
        auto_extract = os.getenv('AUTO_EXTRACT_ENTITIES', '1') == '1'
        if auto_extract:
            try:
                result = self.queue_entity_extraction(
                    doc_id=doc_meta.doc_id,
                    confidence_threshold=0.6,
                    skip_if_exists=True
                )
                if result.get('queued'):
                    self.logger.info(f"Auto-queued entity extraction job {result['job_id']} for document {doc_meta.doc_id}")
                else:
                    self.logger.debug(f"Entity extraction not queued: {result.get('reason', 'unknown')}")
            except Exception as e:
                # Don't fail document ingestion if extraction queueing fails
                self.logger.warning(f"Failed to auto-queue entity extraction: {e}")

        return doc_meta

    def scrape_url(self, url: str, title: Optional[str] = None, tags: Optional[list[str]] = None,
                   follow_links: bool = True, same_domain_only: bool = True,
                   max_pages: int = 50, depth: int = 3, limit: Optional[str] = None,
                   threads: int = 3, delay: int = 500, selector: Optional[str] = None,
                   progress_callback: ProgressCallback = None) -> dict:
        """Scrape a URL using mdscrape and add resulting documents to knowledge base.

        Supports recursive scraping of entire websites by following links.

        Args:
            url: Starting URL to scrape (e.g., http://www.sidmusic.org/sid/)
            title: Optional base title for scraped documents
            tags: Optional list of tags (domain name auto-added)
            follow_links: Follow links to scrape sub-pages (default: True)
            same_domain_only: Only follow links on the same domain (default: True)
            max_pages: Maximum number of pages to scrape (default: 50)
            depth: Maximum crawl depth - how many link levels to follow (default: 3)
            limit: Advanced: Limit scraping to URLs with this prefix (overrides same_domain_only)
            threads: Number of concurrent threads (default: 3 - these sources are
                small volunteer-run sites; a high thread count invites a ban)
            delay: Delay between requests in ms (default: 500)
            selector: CSS selector for main content (optional)
            progress_callback: Optional callback for progress updates

        Examples:
            # Scrape single page only
            kb.scrape_url("http://example.com/page.html", follow_links=False)

            # Scrape entire site (stay on same domain, max 3 levels deep)
            kb.scrape_url("http://www.sidmusic.org/sid/", follow_links=True, same_domain_only=True, depth=3)

            # Scrape specific section (limit to /sid/ prefix)
            kb.scrape_url("http://www.sidmusic.org/sid/", limit="http://www.sidmusic.org/sid/")

        Returns:
            Dictionary with scraping results:
            {
                'status': 'success' | 'partial' | 'failed',
                'url': original_url,
                'output_dir': path_to_scraped_files,
                'files_scraped': count,
                'docs_added': count,
                'docs_updated': count,
                'docs_failed': count,
                'pages_scraped': list_of_urls,
                'error': error_message (if failed),
                'doc_ids': [list of added doc_ids]
            }
        """
        import subprocess
        from urllib.parse import urlparse
        from datetime import datetime

        # 1. Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
            if parsed.scheme not in ['http', 'https']:
                raise ValueError(f"Only HTTP/HTTPS URLs supported: {url}")
        except Exception as e:
            return {
                'status': 'failed',
                'url': url,
                'error': f"Invalid URL: {str(e)}"
            }

        # 2. Extract domain for auto-tagging
        domain = parsed.netloc.replace('www.', '')
        if tags is None:
            tags = []
        tags = list(tags) + [domain, 'scraped']

        # 3. Setup output directory in scraped_docs
        scraped_base = self.data_dir / "scraped_docs"
        scraped_base.mkdir(exist_ok=True)

        # Use domain + timestamp for unique output dir
        safe_domain = domain.replace('.', '_').replace(':', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = scraped_base / f"{safe_domain}_{timestamp}"

        # 3a. Handle follow_links and same_domain_only parameters
        if not follow_links:
            # Don't follow any links - just scrape the single page
            depth = 1
            self.logger.info("follow_links=False: Scraping single page only (depth=1)")

        # 3a-bis. Respect robots.txt before touching the site at all.
        if not robots_allows(url):
            self.logger.warning(f"robots.txt disallows scraping {url}")
            return {
                'status': 'failed',
                'url': url,
                'files_scraped': 0,
                'docs_added': 0,
                'docs_updated': 0,
                'docs_failed': 0,
                'error': (
                    "robots.txt on this host disallows fetching this URL. "
                    "Set TDZ_RESPECT_ROBOTS=0 to override if you have permission "
                    "(e.g. your own mirror)."
                ),
            }

        # 3b. Detect and handle HTML frames
        frame_urls = self._detect_and_extract_frames(url)
        if frame_urls:
            self.logger.info(f"Detected {len(frame_urls)} frame(s), will scrape each individually")

            # Scrape each frame source recursively
            all_doc_ids = []
            all_files_scraped = 0
            all_docs_added = 0

            for frame_url in frame_urls:
                self.logger.info(f"Scraping frame: {frame_url}")

                # For frames, use the parent directory as limit if same_domain_only
                # This allows following links from the frame
                frame_limit = limit
                if same_domain_only and limit is None:
                    # Use the parent directory of the original URL
                    frame_limit = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rsplit('/', 1)[0]}"
                    if not frame_limit.endswith('/'):
                        frame_limit += '/'

                frame_result = self.scrape_url(
                    url=frame_url,
                    title=title,
                    tags=tags,
                    follow_links=follow_links,
                    same_domain_only=False,  # Disable auto-limit for frames
                    max_pages=max_pages,
                    depth=depth,
                    limit=frame_limit,  # Use parent directory as limit
                    threads=threads,
                    delay=delay,
                    selector=selector,
                    progress_callback=progress_callback
                )

                if frame_result['status'] == 'success':
                    all_doc_ids.extend(frame_result.get('doc_ids', []))
                    all_files_scraped += frame_result.get('files_scraped', 0)
                    all_docs_added += frame_result.get('docs_added', 0)

            # Return combined results from all frames
            return {
                'status': 'success',
                'url': url,
                'frames_detected': len(frame_urls),
                'files_scraped': all_files_scraped,
                'docs_added': all_docs_added,
                'docs_updated': 0,
                'docs_failed': 0,
                'doc_ids': all_doc_ids,
                'message': f'Scraped {len(frame_urls)} frames with {all_docs_added} total documents'
            }

        if same_domain_only and limit is None:
            # Automatically set limit to base domain URL to stay on same domain
            # Extract base URL (scheme + netloc + path up to last /)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # If URL has a path, use it as the limit prefix
            if parsed.path and parsed.path != '/':
                # Get the directory part of the path (not the file)
                path_parts = parsed.path.rstrip('/').split('/')
                if path_parts:
                    # Use the full path as prefix to stay within that section
                    base_path = '/'.join(path_parts)
                    limit = f"{base_url}{base_path}"
                else:
                    limit = base_url
            else:
                limit = base_url

            self.logger.info(f"same_domain_only=True: Limiting to URLs starting with '{limit}'")

        # 4. Build mdscrape command
        mdscrape_path = self._find_mdscrape_executable()
        if not mdscrape_path:
            return {
                'status': 'failed',
                'url': url,
                'error': 'mdscrape executable not found. Set MDSCRAPE_PATH or install from: https://github.com/MichaelTroelsen/mdscrape'
            }

        cmd = [
            mdscrape_path,
            url,
            '--output', str(output_dir),
            '--depth', str(depth),
            '--threads', str(threads),
            '--delay', str(delay),
            # mdscrape otherwise announces itself as the generic "mdscrape/1.0",
            # which tells a site operator nothing about who is crawling them.
            '--user-agent', USER_AGENT,
        ]

        if limit:
            cmd.extend(['--limit', limit])
        if selector:
            cmd.extend(['--selector', selector])
        # Note: max_pages is a UI parameter only - mdscrape doesn't support it yet
        # Use depth to control crawl scope instead

        # 5. Store scrape config
        scrape_config = {
            'url': url,
            'follow_links': follow_links,
            'same_domain_only': same_domain_only,
            'max_pages': max_pages,
            'depth': depth,
            'limit': limit,
            'threads': threads,
            'delay': delay,
            'selector': selector,
            'timestamp': datetime.now().isoformat()
        }
        scrape_config_json = json.dumps(scrape_config)

        # 6. Execute mdscrape
        self.logger.info(f"Scraping URL: {url}")
        self.logger.info(f"Command: {' '.join(cmd)}")

        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="scrape_url",
                current=0,
                total=100,
                message="Starting web scraping",
                item=url
            ))

        try:
            # Execute with real-time output streaming
            import time
            import re
            from threading import Thread, Event
            from queue import Queue, Empty

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            # Queues for capturing output
            stdout_queue = Queue()
            stderr_queue = Queue()

            # Helper to read output in background thread
            def enqueue_output(stream, queue, stop_event):
                try:
                    for line in iter(stream.readline, ''):
                        if stop_event.is_set():
                            break
                        queue.put(line)
                except Exception:
                    pass
                finally:
                    stream.close()

            stop_event = Event()
            stdout_thread = Thread(target=enqueue_output, args=(process.stdout, stdout_queue, stop_event))
            stderr_thread = Thread(target=enqueue_output, args=(process.stderr, stderr_queue, stop_event))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Track progress
            pages_scraped = 0
            current_url = None
            last_update_time = time.time()
            timeout_warned = False
            stdout_lines = []
            stderr_lines = []

            # This loop used to have no wall-clock deadline at all: it only
            # LOGGED a warning after 60s of no progress and never actually
            # terminated the process, so a stalled or hung mdscrape process
            # blocked this call (and, per issue #12's investigation, the
            # whole asyncio event loop) indefinitely. scrape_start_time and
            # stop_reason below give it a real, enforced deadline.
            scrape_start_time = time.time()
            scrape_timeout_s = float(os.getenv('SCRAPE_TIMEOUT_S', '3600'))
            stop_reason = None  # 'max_pages' | 'timeout' | None (ran to completion)

            # Process output in real-time
            while process.poll() is None:
                # Check stdout
                try:
                    line = stdout_queue.get(timeout=0.1)
                    stdout_lines.append(line)

                    # Parse mdscrape output for current URL
                    # mdscrape typically outputs: "Scraping: https://example.com/page"
                    url_match = re.search(r'(?:Scraping|Processing|Fetching)[:\s]+(\S+)', line, re.IGNORECASE)
                    if url_match:
                        current_url = url_match.group(1)
                        pages_scraped += 1
                        last_update_time = time.time()
                        timeout_warned = False

                        # Update progress
                        self.logger.info(f"[{pages_scraped}/{max_pages}] Scraping: {current_url}")

                        if progress_callback:
                            progress_callback(ProgressUpdate(
                                operation="scrape_url",
                                current=min(pages_scraped, max_pages),
                                total=max_pages,
                                message=f"Scraping page {pages_scraped}/{max_pages}",
                                item=current_url
                            ))

                        # mdscrape has no concept of max_pages itself (see the
                        # comment above where the command is built) - this is
                        # the actual enforcement. Reaching the requested cap
                        # is success, not an error, so it's tracked separately
                        # from the timeout case below.
                        if pages_scraped >= max_pages:
                            stop_reason = 'max_pages'
                            self.logger.info(f"Reached max_pages={max_pages}, stopping crawl")
                            process.terminate()
                            break

                except Empty:
                    pass

                # Check stderr
                try:
                    line = stderr_queue.get_nowait()
                    stderr_lines.append(line)
                    # Log errors but don't stop
                    if 'error' in line.lower() and 'image' not in line.lower():
                        self.logger.warning(f"Scrape warning: {line.strip()}")
                except Empty:
                    pass

                # No-progress warning stays informational (mdscrape can go
                # quiet on a slow page and still recover), but the overall
                # wall-clock deadline below is a hard stop - previously
                # nothing in this loop ever terminated the process, so a
                # truly hung mdscrape run blocked this call forever.
                time_since_update = time.time() - last_update_time
                if time_since_update > 60 and not timeout_warned:
                    timeout_warned = True
                    warning_msg = f"⚠️ No progress for {int(time_since_update)} seconds"
                    if current_url:
                        warning_msg += f" (current: {current_url})"
                    self.logger.warning(warning_msg)

                    if progress_callback:
                        progress_callback(ProgressUpdate(
                            operation="scrape_url",
                            current=pages_scraped,
                            total=max_pages,
                            message=f"⚠️ Page taking longer than 60s...",
                            item=current_url or "unknown"
                        ))

                if time.time() - scrape_start_time > scrape_timeout_s:
                    stop_reason = 'timeout'
                    self.logger.error(
                        f"Scraping exceeded {scrape_timeout_s:.0f}s wall-clock limit "
                        f"({pages_scraped} pages scraped so far), terminating"
                    )
                    process.terminate()
                    break

            # A deliberate stop (page cap or timeout) needs its own
            # terminate-then-kill sequence; a process that already exited on
            # its own just needs reaping, which is what the original
            # unconditional wait(timeout=60) here provided.
            if stop_reason is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process did not exit after terminate(), killing it")
                    process.kill()
                    process.wait(timeout=10)
            else:
                process.wait(timeout=60)

            # Stop background threads
            stop_event.set()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            # Collect remaining output
            while not stdout_queue.empty():
                try:
                    stdout_lines.append(stdout_queue.get_nowait())
                except Empty:
                    break

            while not stderr_queue.empty():
                try:
                    stderr_lines.append(stderr_queue.get_nowait())
                except Empty:
                    break

            # Check for errors but don't fail if files were scraped
            stdout_output = ''.join(stdout_lines)
            stderr_output = ''.join(stderr_lines)

            if stop_reason is not None:
                # We deliberately terminated the process ourselves, so a
                # nonzero returncode here is expected (the classic
                # image-error/generic-error classification below is for
                # mdscrape's OWN exit status and would just be noise).
                self.logger.info(
                    f"Scraping stopped by {stop_reason} after {pages_scraped} pages "
                    f"(not a failure - proceeding with what was scraped)"
                )
            elif process.returncode != 0:
                error_msg = stderr_output or stdout_output or "Unknown error"

                # Count image-related errors (not critical failures)
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico']
                error_lines = error_msg.split('\n')
                image_errors = sum(1 for line in error_lines
                                 if any(ext in line.lower() for ext in image_extensions))
                total_errors = len([line for line in error_lines if 'Error:' in line or 'error' in line.lower()])

                # If all errors are image-related, treat as warning not failure
                if image_errors > 0 and total_errors > 0 and image_errors == total_errors:
                    self.logger.warning(f"Scraping completed with {image_errors} image-related errors (expected)")
                else:
                    # Log full error but continue - we'll check if any files were scraped
                    self.logger.warning(f"Scraping completed with errors: {error_msg[:500]}...")
            else:
                self.logger.info(f"[OK] Scraping completed successfully ({pages_scraped} pages)")

        except subprocess.TimeoutExpired:
            # The loop above now enforces scrape_timeout_s itself and
            # terminates/kills the process well before this can fire from
            # the crawl running long - reaching here means the process
            # didn't die even after kill(), which is the genuinely
            # exceptional case worth surfacing distinctly.
            self.logger.error("Scrape process did not exit even after being killed")
            return {
                'status': 'failed',
                'url': url,
                'error': 'Scrape process did not exit even after being killed'
            }
        except Exception as e:
            self.logger.error(f"Scraping error: {e}")
            return {
                'status': 'failed',
                'url': url,
                'error': f"Scraping error: {str(e)}"
            }

        # 7. Find all generated markdown files
        if not output_dir.exists():
            return {
                'status': 'failed',
                'url': url,
                'error': f"Output directory not created: {output_dir}",
                'stop_reason': stop_reason,
            }

        md_files = list(output_dir.rglob('*.md'))

        if not md_files:
            return {
                'status': 'failed',
                'url': url,
                'error': f"No markdown files generated in {output_dir}",
                'stop_reason': stop_reason,
            }

        self.logger.info(f"Found {len(md_files)} markdown files to process")

        # 8. Add each file to knowledge base
        added_docs = []
        failed_docs = []
        scrape_date = datetime.now().isoformat()

        for i, md_file in enumerate(md_files):
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation="scrape_url",
                    current=i,
                    total=len(md_files),
                    message="Adding scraped document",
                    item=md_file.name
                ))

            try:
                # Extract source URL from frontmatter
                source_url_for_file = self._extract_source_url_from_md(md_file)
                if not source_url_for_file:
                    source_url_for_file = url  # Fallback to base URL

                # Generate title from domain + page path
                if title:
                    # Use provided base title
                    doc_title = title
                else:
                    # Extract page name from URL path
                    parsed_source = urlparse(source_url_for_file)
                    page_path = parsed_source.path.strip('/')

                    # If it's the index/root, use domain name
                    if not page_path or page_path.lower() in ['index', 'index.html', 'index.htm']:
                        page_name = "Home"
                    else:
                        # Use the last part of the path as page name
                        page_name = page_path.split('/')[-1]
                        # Remove file extensions
                        page_name = page_name.replace('.html', '').replace('.htm', '').replace('.php', '')
                        # Clean up formatting
                        page_name = page_name.replace('_', ' ').replace('-', ' ').title()

                    # Combine domain + page name
                    doc_title = f"{domain} - {page_name}"

                # Add document with URL metadata
                doc = self._add_scraped_document(
                    filepath=str(md_file),
                    source_url=source_url_for_file,
                    title=doc_title,
                    tags=tags,
                    scrape_config=scrape_config_json,
                    scrape_date=scrape_date
                )
                added_docs.append(doc.doc_id)
                self.logger.info(f"Added: {doc.title} ({doc.doc_id})")

            except Exception as e:
                self.logger.error(f"Failed to add {md_file}: {e}")
                failed_docs.append(str(md_file))

        # 9. Return results
        status = 'success' if not failed_docs else ('partial' if added_docs else 'failed')
        if stop_reason == 'timeout':
            # A deliberate max_pages cap is a normal, successful outcome, but
            # hitting the wall-clock timeout means the crawl was cut short
            # before it necessarily finished what was asked - the caller
            # should be able to tell "got everything requested" apart from
            # "ran out of time partway through".
            status = 'partial' if added_docs else 'failed'

        result_dict = {
            'status': status,
            'url': url,
            'output_dir': str(output_dir),
            'files_scraped': len(md_files),
            'docs_added': len(added_docs),
            'docs_updated': 0,
            'docs_failed': len(failed_docs),
            'doc_ids': added_docs,
            'stop_reason': stop_reason,
        }

        if failed_docs:
            result_dict['error'] = f"{len(failed_docs)} files failed to add"
        if stop_reason == 'timeout':
            result_dict['error'] = (
                result_dict.get('error', '') +
                (' ' if result_dict.get('error') else '') +
                f"Crawl exceeded the {scrape_timeout_s:.0f}s time limit and was stopped early "
                f"after {pages_scraped} pages - results may be incomplete."
            ).strip()

        self.logger.info(f"Scraping complete: {status} - Added {len(added_docs)}/{len(md_files)} documents")

        return result_dict

    def rescrape_document(self, doc_id: str, progress_callback: ProgressCallback = None) -> dict:
        """Re-scrape an existing URL-sourced document.

        Args:
            doc_id: Document ID to re-scrape
            progress_callback: Optional progress callback

        Returns:
            Dictionary with re-scrape results (same format as scrape_url)
        """
        # Get document metadata
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]

        # Check if document has source URL
        if not doc.source_url:
            raise ValueError(f"Document is not URL-sourced: {doc_id}")

        self.logger.info(f"Re-scraping document: {doc.title} (from {doc.source_url})")

        # Parse original scrape config
        scrape_config = {}
        if doc.scrape_config:
            try:
                scrape_config = json.loads(doc.scrape_config)
            except Exception as e:
                self.logger.warning(f"Failed to parse scrape config: {e}")

        # Scrape BEFORE touching the existing document. This used to remove
        # the old document first, unconditionally, with no rollback - a
        # dead/renamed page or a hung/failed scrape then permanently
        # destroyed the only copy of the content. Scraping first means a
        # failure just leaves the original document exactly as it was.
        # depth also no longer silently falls back to 50 (a much deeper
        # crawl than scrape_url's own default of 3) when the stored config
        # predates that field being recorded.
        result = self.scrape_url(
            url=doc.source_url,
            title=doc.title,
            tags=doc.tags,
            depth=scrape_config.get('depth', 3),
            limit=scrape_config.get('limit'),
            threads=scrape_config.get('threads', 10),
            delay=scrape_config.get('delay', 100),
            selector=scrape_config.get('selector'),
            progress_callback=progress_callback
        )

        # Add rescrape metadata to result
        result['rescrape'] = True
        result['old_doc_id'] = doc_id

        if result.get('status') == 'failed' or not result.get('doc_ids'):
            self.logger.warning(
                f"Re-scrape of {doc_id} produced no documents (status={result.get('status')}); "
                "keeping the original document unchanged"
            )
            result['old_doc_kept'] = True
            self.logger.info(f"Re-scrape complete: {result['status']}")
            return result

        if doc_id in result['doc_ids']:
            # Content-hash dedup (see add_document) matched the existing
            # document byte-for-byte - there is nothing new to swap in.
            self.logger.info(f"Re-scrape of {doc_id} found identical content; nothing to replace")
            result['old_doc_kept'] = True
            self.logger.info(f"Re-scrape complete: {result['status']}")
            return result

        # Only now, with a confirmed successful scrape of different content
        # in hand, is it safe to retire the old version.
        self.logger.info(f"Removing superseded document version: {doc_id}")
        self.remove_document(doc_id)
        result['old_doc_kept'] = False

        self.logger.info(f"Re-scrape complete: {result['status']}")
        return result

    def _discover_urls(self, base_url: str, config: Optional[dict] = None, max_pages: int = 100) -> set:
        """Discover all URLs at a website by crawling.

        Args:
            base_url: Starting URL to crawl
            config: Optional scrape configuration with depth, limit, etc.
            max_pages: Maximum number of pages to crawl (default: 100)

        Returns:
            Set of discovered URLs
        """
        from urllib.parse import urljoin, urlparse
        from bs4 import BeautifulSoup
        import requests

        discovered = set()
        to_visit = {base_url}
        visited = set()

        # Extract config parameters
        if config:
            depth = min(config.get('depth', 3), 5)  # Cap at 5 to prevent excessive crawling
            limit = config.get('limit')
            same_domain_only = config.get('same_domain_only', True)
            follow_links = config.get('follow_links', True)
        else:
            depth = 3
            limit = None
            same_domain_only = True
            follow_links = True

        # If follow_links is False, just check the single URL
        if not follow_links:
            depth = 1
            self.logger.info("follow_links=False, checking single URL only")

        base_parsed = urlparse(base_url)
        base_domain = base_parsed.netloc

        self.logger.info(f"URL Discovery: depth={depth}, same_domain={same_domain_only}, limit={limit}, max_pages={max_pages}")

        # Crawl with depth tracking
        current_depth = 0
        total_fetched = 0

        while to_visit and current_depth < depth and total_fetched < max_pages:
            current_level = to_visit.copy()
            to_visit.clear()

            self.logger.info(f"Crawl depth {current_depth + 1}/{depth}: {len(current_level)} URLs to check")

            for url in current_level:
                if url in visited:
                    continue

                if total_fetched >= max_pages:
                    self.logger.info(f"Reached max_pages limit ({max_pages})")
                    break

                visited.add(url)

                if not robots_allows(url):
                    self.logger.info(f"robots.txt disallows, skipping: {url}")
                    continue

                total_fetched += 1

                try:
                    # Fetch page with timeout
                    self.logger.debug(f"Fetching: {url}")
                    response = http_get_polite(url, timeout=15)

                    if response.status_code != 200:
                        self.logger.debug(f"Non-200 status ({response.status_code}): {url}")
                        continue

                    # Add to discovered (successful fetch)
                    discovered.add(url)
                    self.logger.debug(f"Discovered: {url}")

                    # Parse HTML to find links (only if following links and not at max depth)
                    if follow_links and current_depth < depth - 1:
                        if 'text/html' in response.headers.get('Content-Type', ''):
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Find all links
                            links_found = 0
                            for link in soup.find_all('a', href=True):
                                href = link['href']

                                # Convert relative URLs to absolute
                                absolute_url = urljoin(url, href)

                                # Remove fragments
                                absolute_url = absolute_url.split('#')[0]

                                # Skip if already visited or queued
                                if absolute_url in visited or absolute_url in to_visit:
                                    continue

                                # Parse the discovered URL
                                link_parsed = urlparse(absolute_url)

                                # Apply same_domain_only filter
                                if same_domain_only and link_parsed.netloc != base_domain:
                                    continue

                                # Apply limit filter
                                if limit and not absolute_url.startswith(limit):
                                    continue

                                # Skip non-HTTP(S) URLs
                                if link_parsed.scheme not in ['http', 'https']:
                                    continue

                                # Add to next level
                                to_visit.add(absolute_url)
                                links_found += 1

                            if links_found > 0:
                                self.logger.debug(f"Found {links_found} links on {url}")

                except requests.Timeout:
                    self.logger.warning(f"Timeout fetching {url}")
                    continue
                except Exception as e:
                    self.logger.debug(f"Error discovering {url}: {e}")
                    continue

            current_depth += 1

        self.logger.info(f"Discovery complete: {len(discovered)} URLs found (visited {total_fetched} pages at depth {current_depth})")
        return discovered

    def check_url_updates(self, auto_rescrape: bool = False, check_structure: bool = True) -> dict:
        """Check all URL-sourced documents for updates.

        Args:
            auto_rescrape: If True, automatically re-scrape changed URLs
            check_structure: If True, check for new/missing sub-pages (slower but comprehensive)

        Returns:
            Dictionary with update information:
            {
                'unchanged': [list of docs with no changes],
                'changed': [list of docs with updates available],
                'failed': [list of docs where check failed],
                'rescraped': [list of doc_ids that were re-scraped],
                'new_pages': [list of newly discovered URLs not in database],
                'missing_pages': [list of docs in database but no longer accessible],
                'scrape_sessions': [list of detected scrape sessions checked]
            }
        """
        from datetime import datetime, timezone
        import json

        results = {
            'unchanged': [],
            'changed': [],
            'failed': [],
            'rescraped': [],
            'new_pages': [],
            'missing_pages': [],
            'scrape_sessions': []
        }

        # Find all URL-sourced documents
        url_docs = [doc for doc in self.documents.values() if doc.source_url]

        if not url_docs:
            self.logger.info("No URL-sourced documents to check")
            return results

        self.logger.info(f"Checking {len(url_docs)} URL-sourced documents for updates")

        # Group documents by scrape session (same base URL)
        scrape_sessions = {}
        for doc in url_docs:
            # Parse scrape config to get base URL
            if doc.scrape_config:
                try:
                    config = json.loads(doc.scrape_config)
                    base_url = config.get('url', doc.source_url)
                except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
                    base_url = doc.source_url
            else:
                base_url = doc.source_url

            # Group by base URL
            if base_url not in scrape_sessions:
                scrape_sessions[base_url] = {
                    'base_url': base_url,
                    'config': config if doc.scrape_config else None,
                    'docs': []
                }
            scrape_sessions[base_url]['docs'].append(doc)

        self.logger.info(f"Found {len(scrape_sessions)} scrape sessions to check")

        # Check each scrape session
        for session_key, session in scrape_sessions.items():
            session_result = {
                'base_url': session['base_url'],
                'docs_count': len(session['docs']),
                'changed': 0,
                'unchanged': 0,
                'new': 0,
                'missing': 0
            }
            results['scrape_sessions'].append(session_result)

            # Get stored URLs for this session
            stored_urls = {doc.source_url: doc for doc in session['docs']}

            # If check_structure is enabled, discover current URLs
            discovered_urls = set()
            if check_structure:
                try:
                    self.logger.info(f"Discovering URLs for: {session['base_url']}")
                    discovered_urls = self._discover_urls(
                        session['base_url'],
                        session['config']
                    )
                    self.logger.info(f"Discovered {len(discovered_urls)} URLs")
                except Exception as e:
                    self.logger.error(f"Failed to discover URLs for {session['base_url']}: {e}")

            # Check each stored document
            for doc in session['docs']:
                try:
                    import requests

                    # Try HEAD request first (faster)
                    response = requests.head(
                        doc.source_url, timeout=10, allow_redirects=True,
                        headers=http_headers(),
                    )

                    # Update last_checked timestamp
                    with self._lock:
                        cursor = self.db_conn.cursor()
                        cursor.execute("""
                            UPDATE documents
                            SET url_last_checked = ?
                            WHERE doc_id = ?
                        """, (datetime.now().isoformat(), doc.doc_id))
                        self.db_conn.commit()

                    # Check if page still exists
                    if response.status_code == 404:
                        self.logger.warning(f"Page no longer exists: {doc.source_url}")
                        results['missing_pages'].append({
                            'doc_id': doc.doc_id,
                            'title': doc.title,
                            'url': doc.source_url
                        })
                        session_result['missing'] += 1
                        continue

                    # Check Last-Modified header if available
                    page_changed = False
                    if 'Last-Modified' in response.headers:
                        from email.utils import parsedate_to_datetime
                        last_modified = parsedate_to_datetime(response.headers['Last-Modified'])

                        if doc.scrape_date:
                            scrape_dt = datetime.fromisoformat(doc.scrape_date)
                            # Ensure both datetimes are timezone-aware for comparison
                            if scrape_dt.tzinfo is None:
                                scrape_dt = scrape_dt.replace(tzinfo=timezone.utc)
                            if last_modified > scrape_dt:
                                page_changed = True
                                self.logger.info(f"Update available: {doc.title} ({doc.source_url})")
                                results['changed'].append({
                                    'doc_id': doc.doc_id,
                                    'title': doc.title,
                                    'url': doc.source_url,
                                    'last_modified': last_modified.isoformat(),
                                    'scraped_date': doc.scrape_date,
                                    'reason': 'content_modified'
                                })
                                session_result['changed'] += 1

                                # Auto-rescrape if requested
                                if auto_rescrape:
                                    self.logger.info(f"Auto-rescaping: {doc.title}")
                                    try:
                                        rescrape_result = self.rescrape_document(doc.doc_id)
                                        if rescrape_result['status'] == 'success':
                                            results['rescraped'].append(doc.doc_id)
                                    except Exception as e:
                                        self.logger.error(f"Auto-rescrape failed: {e}")

                    if not page_changed:
                        # No change detected
                        results['unchanged'].append({
                            'doc_id': doc.doc_id,
                            'title': doc.title,
                            'url': doc.source_url
                        })
                        session_result['unchanged'] += 1

                except Exception as e:
                    self.logger.error(f"Failed to check {doc.source_url}: {e}")
                    results['failed'].append({
                        'doc_id': doc.doc_id,
                        'title': doc.title,
                        'url': doc.source_url,
                        'error': str(e)
                    })

            # Check for new pages
            if check_structure and discovered_urls:
                new_urls = discovered_urls - set(stored_urls.keys())
                if new_urls:
                    self.logger.info(f"Found {len(new_urls)} new pages for {session['base_url']}")
                    for new_url in new_urls:
                        results['new_pages'].append({
                            'url': new_url,
                            'base_url': session['base_url'],
                            'scrape_config': session['config']
                        })
                        session_result['new'] += 1

                # Check for missing pages (in database but not discovered)
                if discovered_urls:  # Only if discovery was successful
                    missing_urls = set(stored_urls.keys()) - discovered_urls
                    # Filter out already detected 404s
                    existing_missing = {p['url'] for p in results['missing_pages']}
                    for missing_url in missing_urls:
                        if missing_url not in existing_missing:
                            doc = stored_urls[missing_url]
                            self.logger.warning(f"Page not discovered during crawl: {missing_url}")
                            results['missing_pages'].append({
                                'doc_id': doc.doc_id,
                                'title': doc.title,
                                'url': missing_url,
                                'reason': 'not_discovered'
                            })
                            session_result['missing'] += 1

        self.logger.info(
            f"Update check complete: {len(results['unchanged'])} unchanged, "
            f"{len(results['changed'])} changed, {len(results['new_pages'])} new pages, "
            f"{len(results['missing_pages'])} missing pages, {len(results['failed'])} failed"
        )

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        self.logger.info(f"Removing document: {doc_id}")

        if doc_id not in self.documents:
            self.logger.warning(f"Document not found for removal: {doc_id}")
            return False

        filename = self.documents[doc_id].filename

        # Remove from database (chunks cascade automatically)
        success = _retry_on_db_locked(self._remove_document_db, doc_id)

        if success:
            # Remove from in-memory index
            del self.documents[doc_id]

            # Prune this doc's chunks from the in-memory chunk cache. Without
            # this, self.chunks keeps serving the deleted content forever in
            # a long-running process: _build_bm25_index() only reloads from
            # the database when self.chunks is empty, so invalidating
            # self.bm25 alone doesn't pick up the DB-level cascade delete.
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

            # Invalidate BM25 index (will be rebuilt on next search)
            self.bm25 = None

            # Remove this document's vectors from the shared embeddings index
            # in place. Nulling the in-memory index here (the old behaviour)
            # left the on-disk .faiss/.json files untouched but out of sync
            # with self.embeddings_index; the next add_document would then
            # see a "no index" state, build a fresh index from only its own
            # new chunks, and overwrite the full-corpus file with it -
            # silently destroying every other document's embeddings.
            if self.use_semantic:
                self._remove_doc_embeddings(doc_id)

            # Invalidate search caches
            self._invalidate_caches()

            self.logger.info(f"Successfully removed document {doc_id}: {filename}")

        return success

    def needs_reindex(self, filepath: str, doc_id: str) -> bool:
        """
        Check if a document needs re-indexing based on file modification time and content hash.

        Args:
            filepath: Path to the document file
            doc_id: Document ID to check

        Returns:
            True if the document needs re-indexing, False otherwise
        """
        doc = self.documents.get(doc_id)
        if not doc:
            return True  # Document doesn't exist, needs indexing

        # If no mtime/hash stored, can't check - assume needs reindex
        if doc.file_mtime is None or doc.file_hash is None:
            self.logger.info(f"Document {doc_id} has no update detection data, assuming needs reindex")
            return True

        # Quick check: modification time
        try:
            current_mtime = os.path.getmtime(filepath)
            if current_mtime <= doc.file_mtime:
                # File hasn't been modified since last index
                return False
        except OSError:
            # File doesn't exist or can't be accessed
            self.logger.warning(f"Cannot access file: {filepath}")
            return False

        # File was modified - do deep check with content hash
        try:
            current_hash = self._compute_file_hash(filepath)
            if current_hash == doc.file_hash:
                # Content is same despite mtime change (e.g., touched)
                self.logger.info(f"File mtime changed but content unchanged: {filepath}")
                return False
            else:
                # Content has actually changed
                self.logger.info(f"File content changed: {filepath}")
                return True
        except Exception as e:
            self.logger.error(f"Error computing hash for {filepath}: {e}")
            return False

    def _reindex_document_if_changed(self, filepath: str, title: Optional[str] = None, tags: Optional[list[str]] = None) -> DocumentMeta:
        """
        Re-index a document (matched by filepath) if its file content has changed
        since it was indexed, or add it if it doesn't exist yet. Used by the bulk
        directory-scan path (add_documents_bulk / check_for_updates) to catch
        files that were edited on disk. Not card-aware - it does a full
        remove-then-add (fresh content-hash doc_id) rather than an in-place
        update, so it does not preserve doc_id or history. For card documents,
        prefer update_document(card_id_or_doc_id, filepath) instead.

        Args:
            filepath: Path to the document file
            title: Optional title (if not provided, uses filename)
            tags: Optional list of tags

        Returns:
            DocumentMeta for the document (existing or newly indexed)
        """
        # Find existing doc by filepath
        existing_doc = None
        for doc in self.documents.values():
            if doc.filepath == filepath:
                existing_doc = doc
                break

        if not existing_doc:
            # Document doesn't exist, add it
            self.logger.info(f"Document not found, adding: {filepath}")
            return self.add_document(filepath, title, tags)

        if not self.needs_reindex(filepath, existing_doc.doc_id):
            # Document unchanged
            self.logger.info(f"Document unchanged, skipping reindex: {filepath}")
            return existing_doc

        # Document has changed, re-index it
        self.logger.info(f"Document changed, re-indexing: {filepath}")
        self.remove_document(existing_doc.doc_id)
        return self.add_document(filepath, title, tags)

    def update_document(self, card_id_or_doc_id: str, filepath: str,
                        title: Optional[str] = None, tags: Optional[list[str]] = None) -> DocumentMeta:
        """
        Replace an existing card's content (and all derived artifacts) at a
        stable logical identity.

        Resolves card_id_or_doc_id to an existing LIVE document (first as an
        exact doc_id, then as a card's logical id via get_document_by_card_id),
        ingests filepath as the new content, and retires the old document
        (marks it superseded_by the new doc, purges its stale entities, and
        rebuilds entity_relationships) so exactly one live document answers
        for that card afterwards.

        This is a whole-file replace, not a merge: the new file's content
        entirely replaces the old card's content. If the new file declares a
        different (or no) card id than the document being updated, the update
        is refused - update_document does not change a card's identity.

        Re-running the same file through update_document is idempotent: since
        doc_id is content-derived, ingesting identical content resolves to the
        same doc_id as the card already live, so no second supersede happens.

        Args:
            card_id_or_doc_id: The card's logical id (from its json block) or
                an exact doc_id of an existing, live document.
            filepath: Path to the new content to replace it with.
            title: Optional new title (defaults to the existing document's title).
            tags: Optional new tags (defaults to the existing document's tags).

        Returns:
            DocumentMeta for the (new or unchanged) live document.

        Raises:
            DocumentNotFoundError: If card_id_or_doc_id does not resolve to a
                live document. Use add_document to create a new card.
            KnowledgeBaseError: If the new file's declared card id conflicts
                with the document being updated.
        """
        old_doc = self.documents.get(card_id_or_doc_id)
        if old_doc is None or old_doc.superseded_by:
            old_doc = self.get_document_by_card_id(card_id_or_doc_id, include_superseded=False)

        if old_doc is None:
            raise DocumentNotFoundError(
                f"No live document or card found for '{card_id_or_doc_id}'. "
                f"Use add_document() to create a new card."
            )

        # Peek the new file's declared identity BEFORE ingesting anything.
        # add_document(replace=True) will supersede whatever live document
        # currently owns the incoming card_id - so if we didn't validate
        # first, a mismatched file could silently supersede an unrelated
        # card instead of (or in addition to) old_doc.
        resolved_path = Path(filepath).resolve()
        if not self._is_path_allowed(filepath):
            raise SecurityError(
                f"Path outside allowed directories. File must be within: {self.allowed_dirs}"
            )
        if not os.path.exists(str(resolved_path)):
            raise DocumentNotFoundError(f"File not found: {filepath}")
        peek_text, _, _, _ = self._extract_text_for_file(str(resolved_path))
        new_card_id = self._extract_card_id(peek_text)

        if old_doc.card_id and new_card_id != old_doc.card_id:
            raise KnowledgeBaseError(
                f"Refusing update: {filepath} declares card id {new_card_id!r}, "
                f"but you're updating card {old_doc.card_id!r} (doc {old_doc.doc_id}). "
                f"update_document() does not change a card's identity - fix the "
                f"file's id or use add_document() to create a separate card."
            )

        if new_card_id:
            colliding = self.get_document_by_card_id(new_card_id, include_superseded=False)
            if colliding and colliding.doc_id != old_doc.doc_id:
                raise KnowledgeBaseError(
                    f"Refusing update: {filepath} declares card id {new_card_id!r}, "
                    f"which already belongs to a different live document "
                    f"{colliding.doc_id} ('{colliding.title}'). update_document() "
                    f"will not supersede a document other than the one you asked "
                    f"to update ({old_doc.doc_id})."
                )

        resolved_title = title if title is not None else old_doc.title
        resolved_tags = tags if tags is not None else old_doc.tags

        new_doc = self.add_document(filepath, resolved_title, resolved_tags, replace=True)

        if new_doc.doc_id == old_doc.doc_id:
            # Identical content - add_document's own dedupe short-circuited.
            return new_doc

        # add_document(replace=True) already superseded old_doc for us when
        # old_doc.card_id was set (its own card-identity lookup resolves to
        # exactly old_doc, guaranteed by the checks above). Cover the case
        # where old_doc has no card_id (generic content replace) by
        # superseding explicitly here.
        if not old_doc.card_id and not old_doc.superseded_by:
            self._mark_superseded(old_doc.doc_id, new_doc.doc_id)

        return new_doc

    def update_document_title(self, doc_id: str, title: str) -> None:
        """
        Update the title of a document.

        Args:
            doc_id: Document ID
            title: New title for the document

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        old_title = doc.title

        # DB first, memory second: mutating in-memory before the write meant a
        # failed UPDATE left this process serving a title that was never saved.
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "UPDATE documents SET title = ? WHERE doc_id = ?",
                (title, doc_id)
            )
            self.db_conn.commit()
        except Exception:
            self.db_conn.rollback()
            doc.title = old_title
            self.logger.exception(f"Failed to update title for {doc_id}; rolled back")
            raise

        doc.title = title

        self.logger.info(f"Updated title for document {doc_id[:12]}: {title}")

    def repoint_document(self, doc_id: str, new_filepath: str, force: bool = False) -> dict:
        """Point a document at a relocated source file, without reingesting.

        `uploads/` is permanent storage (docs/ARCHITECTURE.md, Data Storage
        Model), so a document whose filepath no longer resolves is a broken
        record rather than expected housekeeping. This repairs the record. It
        deliberately does not re-extract, re-chunk or re-embed: the content is
        unchanged, which is the whole reason to re-point instead of re-adding.

        The candidate file is verified against the document's recorded MD5
        before anything is written. Without that check this method would
        cheerfully bind a document's indexed text to an unrelated file - a
        worse state than the missing filepath it set out to fix, and a silent
        one. Pass force=True to accept a mismatch deliberately.

        Returns a summary dict. Raises DocumentNotFoundError for an unknown
        doc_id, SecurityError if the new path is outside ALLOWED_DOCS_DIRS,
        and KnowledgeBaseError if the path is not a file or its content does
        not match.
        """
        doc = self.documents.get(doc_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document not found: {doc_id}")

        resolved = Path(new_filepath).resolve()
        if not resolved.is_file():
            raise KnowledgeBaseError(
                f"Cannot re-point {doc_id}: {resolved} is not a file on disk"
            )

        # Same whitelist that gates add_document. Re-pointing must not become
        # a way to index a path ingestion would have refused.
        if not self._is_path_allowed(str(resolved)):
            raise SecurityError(
                f"Path outside allowed directories. File must be within: {self.allowed_dirs}"
            )

        new_hash = self._compute_file_hash(str(resolved))
        hash_verified = False
        if doc.file_hash:
            if doc.file_hash == new_hash:
                hash_verified = True
            elif not force:
                raise KnowledgeBaseError(
                    f"Content mismatch: {resolved} hashes to {new_hash[:12]} but "
                    f"document {doc_id} recorded {doc.file_hash[:12]}. That is a "
                    f"different file; pass force=True to re-point anyway."
                )

        old_filepath = doc.filepath

        # DB first, memory second - the same ordering and the same reason as
        # update_document_title: mutating in-memory before the write means a
        # failed UPDATE leaves this process serving a path that was never saved.
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "UPDATE documents SET filepath = ?, file_hash = ? WHERE doc_id = ?",
                (str(resolved), new_hash, doc_id)
            )
            self.db_conn.commit()
        except Exception:
            self.db_conn.rollback()
            self.logger.exception(f"Failed to re-point {doc_id}; rolled back")
            raise

        doc.filepath = str(resolved)
        doc.file_hash = new_hash

        # filename is left alone on purpose: it is the document's identity in
        # search results and citations, and a moved file is still the same
        # document even when it has been renamed on disk.
        self.logger.info(
            f"Re-pointed document {doc_id[:12]}: {old_filepath} -> {resolved}"
        )
        return {
            "doc_id": doc_id,
            "old_filepath": old_filepath,
            "new_filepath": str(resolved),
            "hash_verified": hash_verified,
            "forced": bool(force and not hash_verified),
        }

    def update_document_tags(self, doc_id: str, tags: list[str]) -> None:
        """
        Update the tags for a document.

        Args:
            doc_id: Document ID
            tags: New list of tags (replaces existing tags)

        Raises:
            ValueError: If document not found
        """
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc = self.documents[doc_id]
        old_tags = doc.tags

        # Write the DB first, then mutate memory only once it committed -
        # otherwise a failed write leaves this process reporting tags that were
        # never persisted.
        #
        # The column holds a JSON array: every reader parses it with
        # json.loads (see _load_documents / _reload_documents). This method
        # used to write ','.join(tags), which is not valid JSON, so a single
        # call poisoned the row and the next document reload - i.e. every new
        # session - died with JSONDecodeError and loaded no documents at all.
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "UPDATE documents SET tags = ? WHERE doc_id = ?",
                (json.dumps(tags or []), doc_id)
            )
            self.db_conn.commit()
        except Exception:
            self.db_conn.rollback()
            doc.tags = old_tags
            self.logger.exception(f"Failed to update tags for {doc_id}; rolled back")
            raise

        doc.tags = tags

        self.logger.info(f"Updated tags for document {doc_id[:12]}: {tags}")

    def check_all_updates(self, auto_update: bool = False) -> dict:
        """
        Check all indexed documents for updates.

        Args:
            auto_update: If True, automatically re-index changed documents

        Returns:
            Dictionary with lists of unchanged, changed, and missing documents
        """
        results = {
            'unchanged': [],
            'changed': [],
            'missing': [],
            'updated': []  # Only populated if auto_update=True
        }

        for doc_id, doc in list(self.documents.items()):
            filepath = doc.filepath

            # Check if file still exists
            if not os.path.exists(filepath):
                results['missing'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })
                continue

            # Check if needs reindex
            if self.needs_reindex(filepath, doc_id):
                results['changed'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })

                if auto_update:
                    try:
                        updated_doc = self._reindex_document_if_changed(filepath, doc.title, doc.tags)
                        results['updated'].append({
                            'doc_id': updated_doc.doc_id,
                            'filepath': filepath,
                            'title': updated_doc.title
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to update {filepath}: {e}")
            else:
                results['unchanged'].append({
                    'doc_id': doc_id,
                    'filepath': filepath,
                    'title': doc.title
                })

        return results

    def add_documents_bulk(self, directory: str, pattern: str = "**/*.{pdf,txt,md,html,htm,xlsx,xls}",
                           tags: Optional[list[str]] = None, recursive: bool = True,
                           skip_duplicates: bool = True, progress_callback: ProgressCallback = None) -> dict:
        """
        Add multiple documents from a directory matching a glob pattern.

        Args:
            directory: Directory to search for documents
            pattern: Glob pattern (default: **/*.{pdf,txt})
            tags: Tags to apply to all documents
            recursive: Search subdirectories (default: True)
            skip_duplicates: Skip files with duplicate content (default: True)
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with lists of added, skipped, and failed documents
        """
        from pathlib import Path

        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        # Find matching files. Brace alternation must be expanded by hand -
        # pathlib.glob treats "{pdf,txt}" as a literal extension and matches
        # nothing, which made the default pattern a silent no-op.
        search_pattern = pattern if recursive else pattern.replace('**/', '')
        files = []
        seen_paths = set()
        for expanded in _expand_brace_pattern(search_pattern):
            for match in dir_path.glob(expanded):
                # Dedupe: overlapping alternatives can match the same file.
                if match not in seen_paths:
                    seen_paths.add(match)
                    files.append(match)

        results = {
            'added': [],
            'skipped': [],
            'failed': []
        }

        self.logger.info(f"Bulk add: found {len(files)} files matching pattern '{pattern}' in {directory}")

        # Report progress: Start
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_documents_bulk",
                current=0,
                total=len(files),
                message=f"Starting bulk add of {len(files)} files",
                item=directory
            ))

        # Get worker count (configurable via environment variable, default to CPU count)
        max_workers = int(os.getenv('PARALLEL_WORKERS', str(os.cpu_count() or 4)))
        self.logger.info(f"Using {max_workers} workers for parallel processing")

        # Process files in parallel using ThreadPoolExecutor
        def process_file(file_path):
            """Process a single file and return result."""
            if not file_path.is_file():
                return None

            try:
                # Generate title from filename
                title = file_path.stem
                doc = self.add_document(str(file_path), title=title, tags=tags)

                return {
                    'status': 'added',
                    'doc_id': doc.doc_id,
                    'filepath': str(file_path),
                    'title': title,
                    'chunks': doc.total_chunks
                }

            except Exception as e:
                return {
                    'status': 'failed',
                    'filepath': str(file_path),
                    'error': str(e)
                }

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(process_file, fp): fp for fp in files}

            # Process completed tasks as they finish
            completed = 0
            seen_doc_ids = set()

            for future in as_completed(future_to_file):
                completed += 1
                file_path = future_to_file[future]

                # Report progress: Processing file
                if progress_callback:
                    progress_callback(ProgressUpdate(
                        operation="add_documents_bulk",
                        current=completed,
                        total=len(files),
                        message=f"Processing file {completed}/{len(files)}",
                        item=str(file_path.name)
                    ))

                try:
                    result = future.result()
                    if result is None:
                        continue

                    if result['status'] == 'added':
                        # Check for duplicates
                        if skip_duplicates and result['doc_id'] in seen_doc_ids:
                            results['skipped'].append({
                                'filepath': result['filepath'],
                                'reason': 'duplicate content',
                                'doc_id': result['doc_id']
                            })
                        else:
                            seen_doc_ids.add(result['doc_id'])
                            results['added'].append({
                                'doc_id': result['doc_id'],
                                'filepath': result['filepath'],
                                'title': result['title'],
                                'chunks': result['chunks']
                            })
                    elif result['status'] == 'failed':
                        results['failed'].append({
                            'filepath': result['filepath'],
                            'error': result['error']
                        })
                        self.logger.error(f"Failed to add {result['filepath']}: {result['error']}")

                except Exception as e:
                    results['failed'].append({
                        'filepath': str(file_path),
                        'error': str(e)
                    })
                    self.logger.error(f"Failed to process {file_path}: {e}")

        self.logger.info(f"Bulk add complete: {len(results['added'])} added, "
                        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed")

        # Report progress: Complete
        if progress_callback:
            progress_callback(ProgressUpdate(
                operation="add_documents_bulk",
                current=len(files),
                total=len(files),
                message=f"Bulk add complete: {len(results['added'])} added, "
                        f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
            ))

        return results

    def remove_documents_bulk(self, doc_ids: Optional[list[str]] = None,
                              tags: Optional[list[str]] = None) -> dict:
        """
        Remove multiple documents by doc IDs or tags.

        Args:
            doc_ids: List of document IDs to remove
            tags: Remove all documents with any of these tags

        Returns:
            Dictionary with lists of removed and failed document IDs
        """
        if not doc_ids and not tags:
            raise ValueError("Must provide either doc_ids or tags")

        results = {
            'removed': [],
            'failed': []
        }

        # Collect doc_ids to remove
        ids_to_remove = set()

        if doc_ids:
            ids_to_remove.update(doc_ids)

        if tags:
            # Find all documents with any of the specified tags
            for doc_id, doc in self.documents.items():
                if any(tag in doc.tags for tag in tags):
                    ids_to_remove.add(doc_id)

        self.logger.info(f"Bulk remove: removing {len(ids_to_remove)} documents")

        for doc_id in ids_to_remove:
            try:
                if self.remove_document(doc_id):
                    results['removed'].append(doc_id)
                else:
                    results['failed'].append({
                        'doc_id': doc_id,
                        'error': 'Document not found'
                    })
            except Exception as e:
                results['failed'].append({
                    'doc_id': doc_id,
                    'error': str(e)
                })
                self.logger.error(f"Failed to remove {doc_id}: {e}")

        self.logger.info(f"Bulk remove complete: {len(results['removed'])} removed, "
                        f"{len(results['failed'])} failed")

        return results

    def export_documents_bulk(self, doc_ids: Optional[list[str]] = None,
                              tags: Optional[list[str]] = None,
                              format: str = 'json') -> str:
        """
        Export metadata for multiple documents.

        Args:
            doc_ids: List of document IDs to export (if None, uses tags or exports all)
            tags: Export documents with any of these tags
            format: Export format ('json', 'csv', or 'markdown')

        Returns:
            Exported data as a string

        Examples:
            # Export all documents as JSON
            data = kb.export_documents_bulk(format='json')

            # Export documents with 'reference' tag as CSV
            data = kb.export_documents_bulk(tags=['reference'], format='csv')

            # Export specific documents as Markdown
            data = kb.export_documents_bulk(doc_ids=['doc1', 'doc2'], format='markdown')
        """
        # Collect docs to export
        docs_to_export = []

        if doc_ids:
            # Export specific documents
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    docs_to_export.append(self.documents[doc_id])
        elif tags:
            # Export documents with specified tags
            for doc in self.documents.values():
                if any(tag in doc.tags for tag in tags):
                    docs_to_export.append(doc)
        else:
            # Export all documents
            docs_to_export = list(self.documents.values())

        self.logger.info(f"Bulk export: exporting {len(docs_to_export)} documents as {format}")

        # Format the output
        if format == 'json':
            export_data = []
            for doc in docs_to_export:
                export_data.append({
                    'doc_id': doc.doc_id,
                    'filename': doc.filename,
                    'title': doc.title,
                    'filepath': doc.filepath,
                    'file_type': doc.file_type,
                    'total_pages': doc.total_pages,
                    'total_chunks': doc.total_chunks,
                    'indexed_at': doc.indexed_at,
                    'tags': doc.tags,
                    'author': doc.author,
                    'subject': doc.subject,
                    'creator': doc.creator,
                    'creation_date': doc.creation_date
                })
            return json.dumps(export_data, indent=2)

        elif format == 'csv':
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(['doc_id', 'filename', 'title', 'filepath', 'file_type',
                           'total_pages', 'total_chunks', 'indexed_at', 'tags',
                           'author', 'subject', 'creator', 'creation_date'])

            # Write data
            for doc in docs_to_export:
                writer.writerow([
                    doc.doc_id,
                    doc.filename,
                    doc.title,
                    doc.filepath,
                    doc.file_type,
                    doc.total_pages,
                    doc.total_chunks,
                    doc.indexed_at,
                    ', '.join(doc.tags),
                    doc.author or '',
                    doc.subject or '',
                    doc.creator or '',
                    doc.creation_date or ''
                ])

            return output.getvalue()

        elif format == 'markdown':
            lines = []
            lines.append("# Document Export")
            lines.append(f"\n**Total Documents:** {len(docs_to_export)}")
            lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append("---\n")

            for i, doc in enumerate(docs_to_export, 1):
                lines.append(f"## {i}. {doc.title}")
                lines.append(f"- **ID:** `{doc.doc_id}`")
                lines.append(f"- **Filename:** {doc.filename}")
                lines.append(f"- **Type:** {doc.file_type}")
                lines.append(f"- **Pages:** {doc.total_pages}")
                lines.append(f"- **Chunks:** {doc.total_chunks}")
                lines.append(f"- **Tags:** {', '.join(doc.tags) if doc.tags else 'None'}")
                if doc.author:
                    lines.append(f"- **Author:** {doc.author}")
                if doc.subject:
                    lines.append(f"- **Subject:** {doc.subject}")
                lines.append(f"- **Indexed:** {doc.indexed_at}")
                lines.append(f"- **Path:** `{doc.filepath}`")
                lines.append("")

            return '\n'.join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'markdown'")
