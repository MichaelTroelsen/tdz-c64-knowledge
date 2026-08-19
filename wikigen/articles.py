"""Generated wiki articles and the article browser.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import multiprocessing
from functools import partial
import re


class ArticlesMixin:
    """Generated wiki articles and the article browser."""

    def _generate_articles(self, entities_data: Dict):
        """Generate articles for major entities and topics (parallelized)."""
        print("\n[10/10] Generating articles...")

        # Create articles directory
        articles_dir = self.output_dir / "articles"
        articles_dir.mkdir(exist_ok=True)

        # Define major topics to generate articles for
        article_topics = {
            'HARDWARE': ['SID', 'VIC-II', 'VIC', 'VIC-20', 'CIA', '6510', '6502', '1541', 'Joystick', 'Keyboard', 'Cartridge', 'User Port', 'Datasette'],
            'MUSIC': ['Music', 'Sound', 'Composer', 'Editor', 'Tracker', 'ADSR', 'Waveform'],
            'GRAPHICS': ['Sprite', 'Bitmap', 'Graphics', 'Color', 'Screen', 'Character', 'Raster', 'Multicolor'],
            'PROGRAMMING': ['Assembly', 'BASIC', 'Kernal', 'ROM', 'Memory', 'Interrupt', 'DMA', 'IRQ', 'NMI', 'LDA', 'STA', 'JMP', 'JSR', 'RTS', 'PETSCII', 'Stack', 'Zero Page'],
            'TOOLS': ['Assembler', 'Editor', 'Debugger', 'Monitor', 'Emulator', 'Compiler']
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

        # Fallback: Try to generate article from search results if no entities found
        return self._create_article_from_search(keyword, category, entities_data)

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

    def _create_article_from_search(self, keyword: str, category: str, all_entities: Dict) -> Dict:
        """Create article from search results when no entities are found (fallback)."""
        try:
            # Search knowledge base for the keyword
            search_results = self.kb.search(keyword, max_results=20)

            if not search_results:
                return None  # No results found

            # Create synthetic entity from search results
            synthetic_entity = {
                'type': 'concept',
                'text': keyword,
                'doc_count': len(search_results),
                'confidence': 0.85,  # Synthetic confidence
                'documents': []
            }

            # Build document list from search results
            # Group results by document ID to avoid duplicates
            doc_ids_seen = set()
            for result in search_results:
                doc_id = result.get('doc_id', '')
                if doc_id and doc_id not in doc_ids_seen:
                    doc_ids_seen.add(doc_id)
                    # kb.documents is a dict {doc_id: DocumentMeta}
                    doc = self.kb.documents.get(doc_id)
                    if doc:
                        synthetic_entity['documents'].append({
                            'id': doc.doc_id,
                            'title': doc.title,
                            'filename': re.sub(r'[^\w\-]', '_', doc.doc_id) + '.html'
                        })

            # If we have at least 3 documents, create the article
            if len(synthetic_entity['documents']) >= 3:
                return self._create_article(keyword, category, [synthetic_entity], all_entities)

            return None  # Not enough content

        except Exception as e:
            print(f"    Warning: Failed to create article from search for '{keyword}': {e}")
            return None

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

    def _generate_article_description(self, title: str, category: str, entity: Dict, related_topics: List[Dict]) -> str:
        """Generate AI-powered article description."""
        try:
            # Build context from entity and related topics
            context_parts = [
                f"Entity: {title}",
                f"Category: {category}",
                f"Type: {entity.get('entity_type', 'unknown')}",
                f"Appears in {entity['doc_count']} documents",
            ]

            # Add related topics for context
            if related_topics:
                related_names = [r['text'] for r in related_topics[:5]]
                context_parts.append(f"Related topics: {', '.join(related_names)}")

            context = '\n'.join(context_parts)

            # Create prompt for article description
            prompt = f"""Write a comprehensive 5-6 paragraph technical article for a Commodore 64 knowledge base about "{title}".

Context:
{context}

The article should include:

1. **Introduction** (1 paragraph): Define what {title} is in the context of the Commodore 64, its manufacturer/origin, and its primary purpose.

2. **Technical Architecture** (2 paragraphs): Describe the technical design, internal components, registers, memory mapping, or programming model. Include specific technical details like register addresses, bit layouts, or memory ranges where applicable.

3. **Features and Capabilities** (1-2 paragraphs): Explain the main features, what it can do, common use cases, and programming techniques. Mention specific capabilities, modes, or special functions.

4. **Historical Context and Significance** (1 paragraph): Discuss its importance in C64 development, common applications in games/demos/music, and why it matters to C64 programmers today.

Write in a technical but accessible style for programmers and C64 enthusiasts. Include specific technical details like memory addresses (e.g., $D000-$D3FF), register names, and technical specifications. Be factual and comprehensive.

Write ONLY the article content, no title or introduction phrase."""

            # Try to use LLM if available
            description = self.kb._call_llm(prompt, max_tokens=1000, temperature=0.5)

            if description and len(description.strip()) > 50:
                return description.strip()
            else:
                # Fallback description
                return self._generate_fallback_description(title, category, entity)

        except Exception as e:
            print(f"  Warning: Could not generate AI description for {title}: {e}")
            return self._generate_fallback_description(title, category, entity)

    def _generate_technical_specs(self, title: str, category: str, entity: Dict, code_examples: List[Dict]) -> str:
        """Generate technical specifications section with tables and visual elements."""
        specs_html = ""

        # Add category-specific technical specifications
        if category == 'HARDWARE':
            # Try to extract technical specs from code examples
            specs_html = """
            <div class="tech-specs">
                <h2>Technical Specifications</h2>
                <div class="spec-grid">
            """

            # Add hardware-specific specs based on the entity
            if 'SID' in title.upper():
                specs_html += """
                    <div class="spec-card">
                        <h3>Memory Map</h3>
                        <table class="specs-table">
                            <tr><th>Register</th><th>Address</th><th>Function</th></tr>
                            <tr><td>Voice 1 Frequency</td><td>$D400-$D401</td><td>16-bit frequency control</td></tr>
                            <tr><td>Voice 1 Pulse Width</td><td>$D402-$D403</td><td>12-bit pulse width</td></tr>
                            <tr><td>Voice 1 Control</td><td>$D404</td><td>Waveform and gate control</td></tr>
                            <tr><td>Voice 1 Attack/Decay</td><td>$D405</td><td>Envelope timing</td></tr>
                            <tr><td>Voice 1 Sustain/Release</td><td>$D406</td><td>Envelope levels</td></tr>
                        </table>
                    </div>
                    <div class="spec-card">
                        <h3>Audio Features</h3>
                        <ul class="feature-list">
                            <li>3 independent voices</li>
                            <li>4 waveforms: Triangle, Sawtooth, Pulse, Noise</li>
                            <li>ADSR envelope generator per voice</li>
                            <li>Multi-mode filter (low-pass, high-pass, band-pass)</li>
                            <li>Ring modulation</li>
                            <li>Oscillator sync</li>
                        </ul>
                    </div>
                """
            elif 'VIC' in title.upper():
                specs_html += """
                    <div class="spec-card">
                        <h3>Memory Map</h3>
                        <table class="specs-table">
                            <tr><th>Register</th><th>Address</th><th>Function</th></tr>
                            <tr><td>Sprite 0 X</td><td>$D000</td><td>Horizontal position</td></tr>
                            <tr><td>Sprite 0 Y</td><td>$D001</td><td>Vertical position</td></tr>
                            <tr><td>Border Color</td><td>$D020</td><td>Border color register</td></tr>
                            <tr><td>Background Color</td><td>$D021</td><td>Background color 0</td></tr>
                            <tr><td>Sprite Enable</td><td>$D015</td><td>Enable sprites 0-7</td></tr>
                        </table>
                    </div>
                    <div class="spec-card">
                        <h3>Display Modes</h3>
                        <ul class="feature-list">
                            <li>Standard Text Mode: 40×25 characters</li>
                            <li>Standard Bitmap Mode: 320×200 pixels</li>
                            <li>Multicolor Text Mode: 40×25 with 4 colors per char</li>
                            <li>Multicolor Bitmap Mode: 160×200 pixels, 4 colors</li>
                            <li>8 hardware sprites (24×21 pixels each)</li>
                            <li>16 colors</li>
                        </ul>
                    </div>
                """
            elif 'SPRITE' in title.upper():
                specs_html += """
                    <div class="spec-card">
                        <h3>Sprite Specifications</h3>
                        <table class="specs-table">
                            <tr><th>Property</th><th>Value</th></tr>
                            <tr><td>Dimensions</td><td>24×21 pixels</td></tr>
                            <tr><td>Number Available</td><td>8 sprites (0-7)</td></tr>
                            <tr><td>Colors</td><td>1 color + transparent (2 in multicolor mode)</td></tr>
                            <tr><td>Position Range</td><td>0-511 horizontal, 0-255 vertical</td></tr>
                            <tr><td>Sprite Data</td><td>63 bytes per sprite</td></tr>
                        </table>
                    </div>
                    <div class="spec-card">
                        <h3>Sprite Features</h3>
                        <ul class="feature-list">
                            <li>Individual enable/disable control</li>
                            <li>Horizontal and vertical expansion (2x)</li>
                            <li>Sprite-sprite collision detection</li>
                            <li>Sprite-background collision detection</li>
                            <li>Priority control (front/behind background)</li>
                            <li>Multicolor mode (3 colors + transparent)</li>
                        </ul>
                    </div>
                """

            specs_html += """
                </div>
            </div>
            """

        return specs_html

    def _generate_fallback_description(self, title: str, category: str, entity: Dict) -> str:
        """Generate a basic description when AI is not available."""
        entity_type = entity.get('entity_type', 'component')
        doc_count = entity['doc_count']

        descriptions = {
            'HARDWARE': f"{title} is a hardware component of the Commodore 64 computer system. This {entity_type} is referenced in {doc_count} documents in the knowledge base, indicating its importance in C64 programming and hardware documentation.",
            'MUSIC': f"{title} relates to music and sound capabilities of the Commodore 64. This {entity_type} appears in {doc_count} documents covering SID chip programming, music composition tools, and audio features.",
            'GRAPHICS': f"{title} is related to the graphics capabilities of the Commodore 64. This {entity_type} is documented in {doc_count} sources covering VIC-II programming, display modes, and visual effects.",
            'PROGRAMMING': f"{title} is a programming concept or tool for the Commodore 64. This {entity_type} appears in {doc_count} documents covering assembly language, BASIC programming, and system routines.",
            'TOOLS': f"{title} is a development tool or utility for Commodore 64 programming. This {entity_type} is referenced in {doc_count} documents covering development environments, assemblers, and productivity tools.",
        }

        return descriptions.get(category, f"{title} is documented in {doc_count} sources in the C64 knowledge base.")

    def _extract_code_examples(self, entity: Dict, max_examples: int = 5) -> List[Dict]:
        """Extract code examples from documents mentioning this entity."""
        examples = []

        # Boilerplate patterns to skip (front matter, copyright pages)
        skip_patterns = [
            'copyright', 'page break', 'table of contents', 'all rights reserved',
            'printed in', 'published by', 'library of congress', 'isbn',
            'reproduction', 'permission', 'trademark'
        ]

        # Strong code indicators for C64 content
        code_indicators = [
            'lda', 'sta', 'ldx', 'stx', 'ldy', 'sty', 'jsr', 'jmp', 'rts', 'rti',
            'and', 'ora', 'eor', 'asl', 'lsr', 'rol', 'ror', 'inc', 'dec',
            'beq', 'bne', 'bcc', 'bcs', 'bmi', 'bpl', 'bvc', 'bvs',
            '$d020', '$d021', '$d000', '$d400', '$dc00', '$dd00',
            'vic-ii', 'sid chip', 'cia', '6510', '6502', 'kernal'
        ]

        # Get chunks from documents
        # Use a separate connection for thread safety (articles are generated in parallel)
        import sqlite3
        conn = sqlite3.connect(self.kb.db_file)
        try:
            cursor = conn.cursor()

            for doc in entity['documents'][:max_examples * 2]:  # Check more docs to find good examples
                # Fetch chunks, skip first 3 (usually front matter in PDFs)
                chunks = cursor.execute("""
                    SELECT content, page, chunk_id
                    FROM chunks
                    WHERE doc_id = ?
                    ORDER BY chunk_id
                """, (doc['id'],)).fetchall()

                best_chunk = None
                best_score = 0

                for content, page, chunk_id in chunks:
                    # Skip first 3 chunks (front matter)
                    if chunk_id < 3:
                        continue

                    content_lower = content.lower()

                    # Skip boilerplate content
                    if any(pattern in content_lower for pattern in skip_patterns):
                        continue

                    # Score this chunk based on code density
                    score = sum(1 for indicator in code_indicators if indicator in content_lower)

                    # Boost score if chunk has hex addresses or assembly instructions
                    if '$' in content and any(x in content_lower for x in ['lda', 'sta', 'jsr']):
                        score += 5

                    # Track best chunk for this document
                    if score > best_score:
                        best_score = score
                        best_chunk = (content, page)

                # Add best chunk if it's good enough (score > 2)
                if best_chunk and best_score > 2:
                    content, page = best_chunk
                    examples.append({
                        'doc_title': doc['title'],
                        'doc_id': doc['id'],
                        'doc_filename': doc['filename'],
                        'content': content[:500] + '...' if len(content) > 500 else content,
                        'page': page
                    })

                # Stop once we have enough good examples
                if len(examples) >= max_examples:
                    break
        finally:
            conn.close()

        return examples

    def _calculate_reading_time(self, content: str) -> int:
        """Calculate estimated reading time in minutes."""
        words = len(content.split())
        # Average reading speed: 200 words per minute
        minutes = max(1, round(words / 200))
        return minutes

    def _build_related_articles_sidebar(self, current_title: str, related_entities: List[Dict]) -> str:
        """Build sidebar HTML showing related articles."""
        # Load articles.json to find matching articles
        articles_json_path = self.data_dir / "articles.json"

        if not articles_json_path.exists():
            return ''  # No articles.json yet

        try:
            with open(articles_json_path, 'r', encoding='utf-8') as f:
                all_articles = json.load(f)
        except Exception as e:
            print(f"    Warning: Could not load articles.json: {e}")
            return ''

        # Match related entities to articles
        related_articles = []
        for rel_entity in related_entities[:10]:  # Top 10 related entities
            entity_name = rel_entity['text']

            # Find article with matching title
            for article in all_articles:
                if article['title'].lower() == entity_name.lower():
                    # Don't include the current article
                    if article['title'].lower() != current_title.lower():
                        related_articles.append({
                            'title': article['title'],
                            'filename': article['filename'],
                            'category': article['category'],
                            'doc_count': article['doc_count'],
                            'overlap_count': rel_entity.get('overlap_count', 0),
                            'overlap_ratio': rel_entity.get('overlap_ratio', 0)
                        })
                    break

        # Sort by overlap ratio (most related first)
        related_articles.sort(key=lambda x: x['overlap_ratio'], reverse=True)

        # Build sidebar HTML
        if not related_articles:
            return f"""
            <aside class="related-articles-sidebar">
                <h2>🔗 Related Articles</h2>
                <div class="related-articles-empty">
                    No related articles found yet.
                </div>
            </aside>
            """

        # Build article items
        items_html = ''
        for article in related_articles[:8]:  # Show max 8 related articles
            overlap_pct = int(article['overlap_ratio'] * 100)
            items_html += f"""
            <div class="related-article-item">
                <a href="{article['filename']}">{html.escape(article['title'])}</a>
                <div class="related-article-meta">
                    📚 {article['doc_count']} docs • {overlap_pct}% overlap
                </div>
                <span class="related-article-category">{html.escape(article['category'])}</span>
            </div>
            """

        sidebar_html = f"""
        <aside class="related-articles-sidebar">
            <h2>🔗 Related Articles</h2>
            {items_html}
        </aside>
        """

        return sidebar_html

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
        # Generate AI description
        ai_description = self._generate_article_description(title, category, main_entity, related)

        overview_html = f"""
        <div class="article-overview">
            <h2>Overview</h2>
            <p><strong>Category:</strong> {html.escape(category)}</p>
            <p><strong>Referenced in:</strong> {main_entity['doc_count']} documents</p>
            <p><strong>Entity Type:</strong> {html.escape(main_entity['type'])}</p>
            <p><strong>Confidence:</strong> {(main_entity['confidence'] * 100):.0f}%</p>
        </div>

        <div class="article-description">
            {html.escape(ai_description).replace(chr(10), '<br><br>')}
        </div>
        """

        # Generate technical specifications section
        tech_specs_html = self._generate_technical_specs(title, category, main_entity, code_examples)

        # Generate memory map diagrams
        print(f"    Generating diagrams for {title}...")
        diagrams = self._generate_memory_map_diagrams(title, category)

        # Extract images from PDFs (if any)
        # images = self._extract_images_from_pdfs(title, main_entity, max_images=6)

        # Combine diagrams and images
        all_images = diagrams  # + images

        # Build image gallery section
        images_html = ''
        if all_images:
            images_html = '<div class="article-section image-gallery-section">'
            images_html += '<h2>Diagrams & Visual Reference</h2>'
            images_html += '<div class="image-gallery">'
            for img in all_images:
                # Handle both diagrams and extracted images
                if 'title' in img:  # Diagram
                    images_html += f'''
                    <div class="gallery-item">
                        <a href="{img['path']}" target="_blank">
                            <img src="{img['path']}" alt="{html.escape(img['title'])}" loading="lazy">
                        </a>
                        <div class="image-caption">
                            <strong>{html.escape(img['title'])}</strong><br>
                            <small>{html.escape(img['description'])}</small>
                        </div>
                    </div>
                    '''
                else:  # Extracted image
                    images_html += f'''
                    <div class="gallery-item">
                        <a href="{img['path']}" target="_blank">
                            <img src="{img['path']}" alt="{html.escape(title)}" loading="lazy">
                        </a>
                        <div class="image-caption">
                            From: {html.escape(img['source_doc'])}<br>
                            <small>Page {img['source_page']} • {img['width']}×{img['height']}px</small>
                        </div>
                    </div>
                    '''
            images_html += '</div></div>'

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

        # Build Related Articles sidebar
        related_articles_html = self._build_related_articles_sidebar(title, related)

        html_template = f"""<!DOCTYPE html>
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
        .article-description {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            line-height: 1.8;
            font-size: 1.05em;
            color: var(--text-color);
            border-left: 4px solid var(--secondary-color);
        }}
        .article-description p {{
            margin: 15px 0;
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
        .tech-specs {{
            margin: 30px 0;
            padding: 25px;
            background: var(--card-bg);
            border-radius: 12px;
            border-left: 4px solid var(--accent-color);
        }}
        .tech-specs h2 {{
            color: var(--secondary-color);
            margin-top: 0;
            margin-bottom: 20px;
        }}
        .spec-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .spec-card {{
            background: var(--bg-color);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
        }}
        .spec-card h3 {{
            color: var(--primary-color);
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        .specs-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .specs-table th {{
            background: var(--primary-color);
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }}
        .specs-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        .specs-table tr:last-child td {{
            border-bottom: none;
        }}
        .specs-table tr:nth-child(even) {{
            background: var(--bg-color);
        }}
        .feature-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .feature-list li {{
            padding: 8px 0 8px 25px;
            position: relative;
            line-height: 1.6;
        }}
        .feature-list li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: var(--accent-color);
            font-weight: bold;
            font-size: 1.2em;
        }}
        .image-gallery-section {{
            margin: 30px 0;
        }}
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .gallery-item {{
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .gallery-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        .gallery-item a {{
            display: block;
            text-decoration: none;
        }}
        .gallery-item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: white;
            padding: 10px;
        }}
        .image-caption {{
            padding: 12px;
            font-size: 0.85em;
            color: var(--text-color);
            border-top: 1px solid var(--border-color);
            line-height: 1.4;
        }}
        .image-caption small {{
            color: var(--text-muted);
            font-size: 0.9em;
        }}
        /* Two-column layout for article + sidebar */
        .article-layout {{
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 30px;
            align-items: start;
        }}
        @media (max-width: 1024px) {{
            .article-layout {{
                grid-template-columns: 1fr;
            }}
            .related-articles-sidebar {{
                order: -1; /* Show sidebar above content on mobile */
            }}
        }}
        /* Related Articles Sidebar */
        .related-articles-sidebar {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            position: sticky;
            top: 20px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}
        .related-articles-sidebar h2 {{
            margin: 0 0 15px 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}
        .related-article-item {{
            margin: 12px 0;
            padding: 12px;
            background: var(--bg-color);
            border-radius: 8px;
            border-left: 3px solid var(--accent-color);
            transition: all 0.2s;
        }}
        .related-article-item:hover {{
            transform: translateX(5px);
            border-left-color: var(--primary-color);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .related-article-item a {{
            color: var(--text-color);
            text-decoration: none;
            font-weight: 600;
            display: block;
        }}
        .related-article-item a:hover {{
            color: var(--primary-color);
        }}
        .related-article-meta {{
            font-size: 0.75em;
            color: var(--text-muted);
            margin-top: 5px;
        }}
        .related-article-category {{
            display: inline-block;
            padding: 2px 8px;
            background: var(--accent-color);
            color: white;
            border-radius: 10px;
            font-size: 0.7em;
            margin-top: 5px;
        }}
        .related-articles-empty {{
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 20px;
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

        <main class="article-layout">
            <article class="article-content">
                <span class="article-category">{html.escape(category)}</span>
                <h1>{title_escaped}</h1>

                <div class="article-meta">
                    <span class="reading-time">⏱️ {reading_time} min read</span>
                    <span class="word-count">📄 ~{word_count} words</span>
                    <span class="doc-count">📚 {main_entity['doc_count']} documents</span>
                </div>

                {overview_html}
                {tech_specs_html}
                {images_html}
                {entities_html}
                {related_html}
                {code_html}
                {docs_html}
            </article>

            {related_articles_html}
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="../assets/js/enhancements.js"></script>
</body>
</html>
"""
        return html_template

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

{self._get_unified_about_box("home")}

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

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "articles.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: articles.html")
