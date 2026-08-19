"""The core HTML pages: index, per-document, entities, topics, timeline.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import multiprocessing
import re
import version
from wikigen.urls import safe_external_url, url_query_value


class PagesMixin:
    """The core HTML pages: index, per-document, entities, topics, timeline."""

    def _generate_html_pages(self, documents: List[Dict]):
        """Generate HTML pages for all documents (parallelized)."""
        # Generate index page
        self._generate_index_html()

        # Generate browser pages
        self._generate_documents_browser_html()
        self._generate_chunks_browser_html()

        # Generate document pages in parallel
        print(f"  Generating {len(documents)} document pages in parallel...")
        max_workers = min(multiprocessing.cpu_count() * 2, 8)  # Limit to 8 workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all document generation tasks
            futures = [executor.submit(self._generate_doc_html, doc) for doc in documents]

            # Wait for completion and show progress
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()  # Raise any exceptions
                    completed += 1
                    if completed % 10 == 0 or completed == len(documents):
                        print(f"    Progress: {completed}/{len(documents)} documents")
                except Exception as e:
                    print(f"    Error generating document: {e}")

        # Generate entity pages
        self._generate_entities_html()

        # Generate knowledge graph page
        self._generate_knowledge_graph_html()

        # Generate similarity map page
        self._generate_similarity_map_html()

        # Generate topics page
        self._generate_topics_html()

        # Generate timeline page
        self._generate_timeline_html()

        # Generate file viewer page
        self._generate_file_viewer_html()

        # Generate settings page
        self._generate_settings_html()

    def _generate_index_html(self):
        """Generate main index page."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDZ C64 Knowledge Base - Wiki</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Comprehensive Commodore 64 Documentation</p>
            <p style="color: var(--text-muted); font-size: 0.9em; margin-top: -10px;">Version {version.__version__}</p>
        </header>

{self._get_main_nav()}

        <div class="search-section">
            <input type="text" id="search-input" placeholder="Search the knowledge base..." autocomplete="off">
            <div id="search-results" class="search-results"></div>
        </div>

        <main>
{self._get_unified_about_box("home")}

            <section class="stats-grid">
                <div class="stat-card">
                    <h3>{self.stats['documents']}</h3>
                    <p>Documents</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['chunks']}</h3>
                    <p>Text Chunks</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['entities']}</h3>
                    <p>Entities</p>
                </div>
                <div class="stat-card">
                    <h3>{self.stats['topics']}</h3>
                    <p>Topics</p>
                </div>
            </section>

            <section id="popular-articles" class="popular-articles-section">
                <!-- Will be populated by JavaScript from articles.json -->
            </section>

            <section class="tag-cloud-section">
                <h2>🏷️ Explore by Topic</h2>
                <div class="tag-cloud-controls">
                    <button class="tag-cloud-control-btn active" data-sort="count">Most Popular</button>
                    <button class="tag-cloud-control-btn" data-sort="alpha">Alphabetical</button>
                    <button class="tag-cloud-control-btn" data-sort="random">Shuffle</button>
                </div>
                <div class="tag-cloud" id="tag-cloud">
                    <div class="tag-cloud-loading">Loading tags...</div>
                </div>
                <div class="tag-cloud-stats" id="tag-cloud-stats"></div>
            </section>

            <section class="recent-docs">
                <h2>Browse Documents</h2>
                <div class="doc-grid" id="doc-list">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>

            <section class="browse-section">
                <h2>Browse by Category</h2>
                <div class="category-grid" id="category-list">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>
        </main>

        <footer>
            <p>Exported: {self.stats['export_date']}</p>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/main.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.output_dir / "index.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  Generated: index.html")

    def _generate_doc_html(self, doc: Dict):
        """Generate HTML page for a single document."""
        doc_id = doc['id']
        safe_filename = re.sub(r'[^\w\-]', '_', doc_id) + '.html'

        # Escape HTML in content
        title_escaped = html.escape(doc['title'])

        # Build chunks HTML
        chunks_html = []
        for chunk in doc['chunks']:
            content_escaped = html.escape(chunk['content'])
            page_info = f" (Page {chunk['page']})" if chunk['page'] else ""
            chunks_html.append(f"""
                <div class="chunk">
                    <div class="chunk-header">Chunk {chunk['id']}{page_info}</div>
                    <div class="chunk-content">{content_escaped}</div>
                </div>
            """)

        # Build tags HTML
        tags_html = ' '.join(f'<span class="tag">{html.escape(tag)}</span>' for tag in doc['tags'])

        # Scraped source URLs are untrusted input - drop any non-http(s) scheme
        # rather than emitting it as a clickable link (see safe_external_url).
        safe_source_url = safe_external_url(doc.get('source_url'))
        if doc.get('source_url') and not safe_source_url:
            print(f"  [WARN] Dropping unsafe source_url on '{doc['title']}': {doc['source_url']!r}")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_escaped} - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="../assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle"><a href="../index.html">← Back to Home</a></p>
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
            <a href="../documents.html">Documents</a>
            <span class="separator">›</span>
            <span class="current">{title_escaped}</span>
        </nav>

        <main class="doc-with-sidebar">
            <article class="document doc-main-content">
                <div class="doc-header">
                    <h1>{title_escaped}</h1>
                    <div class="doc-meta">
                        <span class="meta-item">📄 {html.escape(doc['file_type'])}</span>
                        <span class="meta-item">📊 {doc['total_chunks']} chunks</span>
                        {f'<span class="meta-item">📑 {doc["total_pages"]} pages</span>' if doc['total_pages'] else ''}
                    </div>
                    {f'<div class="doc-tags">{tags_html}</div>' if tags_html else ''}
                    {f'<div class="doc-url"><a href="{safe_source_url}" target="_blank" rel="noopener noreferrer">🔗 Source URL</a></div>' if safe_source_url else ''}
                    {f'<div class="doc-url"><a href="../viewer.html?file={url_query_value(doc["file_path_in_wiki"])}&amp;name={url_query_value(doc["filename"])}&amp;type={url_query_value(doc["file_type"])}" target="_blank" style="background: var(--accent-color); color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; display: inline-block; margin-top: 10px;">📄 View Source File</a></div>' if doc.get('file_path_in_wiki') else ''}
                </div>

                <div class="chunks-container">
                    {''.join(chunks_html)}
                </div>
            </article>

            <aside class="related-docs-sidebar" id="related-docs-sidebar" data-doc-id="{doc_id}">
                <h3>Related Documents</h3>
                <div class="loading-related">Loading related documents...</div>
            </aside>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="../assets/js/main.js"></script>
    <script src="../assets/js/enhancements.js"></script>
</body>
</html>
"""
        filepath = self.docs_dir / safe_filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_entities_html(self):
        """Generate entities browser page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entities - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Entity Browser - Click any entity to see related documents</p>
        </header>

{NAV}

        <main>
{ABOUT}

            <!-- Filter and Navigation Controls -->
            <div class="entity-controls">
                <div class="filter-section">
                    <input type="text" id="entity-filter" placeholder="🔍 Search entities..." autocomplete="off">
                </div>

                <div class="entity-type-nav" id="entity-type-nav">
                    <!-- Will be populated by JavaScript -->
                </div>

                <div class="entity-stats" id="entity-stats">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>

            <!-- Entities Container -->
            <div id="entities-container">
                <!-- Will be populated by JavaScript -->
            </div>

            <!-- Back to Top Button -->
            <button id="back-to-top" class="back-to-top" title="Back to top">
                ↑ Top
            </button>
        </main>

        <!-- Entity Details Modal -->
        <div id="entity-modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 id="modal-title"></h2>
                    <span class="modal-close">&times;</span>
                </div>
                <div class="modal-body" id="modal-body">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>
        </div>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/entities.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace('{NAV}', self._get_main_nav('entities'))
        html_content = html_content.replace('{ABOUT}', self._get_unified_about_box('entities'))

        filepath = self.output_dir / "entities.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: entities.html")

    def _generate_topics_html(self):
        """Generate topics browser page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topics & Clusters - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Topics & Clusters</p>
        </header>

{NAV}

        <main>
{ABOUT}

            <section>
                <h2>Topic Models</h2>
                <div id="topics-container">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>

            <section>
                <h2>Document Clusters</h2>
                <div id="clusters-container">
                    <!-- Will be populated by JavaScript -->
                </div>
            </section>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/topics.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("topics"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("topics"))

        filepath = self.output_dir / "topics.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: topics.html")

    def _generate_timeline_html(self):
        """Generate interactive timeline page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Timeline - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .timeline-wrapper {
            margin: 30px 0;
        }

        .timeline-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: space-between;
        }

        .timeline-filters {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 8px 16px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9em;
        }

        .filter-btn:hover {
            border-color: var(--accent-color);
            background: var(--accent-color);
            color: white;
        }

        .filter-btn.active {
            background: var(--accent-color);
            border-color: var(--accent-color);
            color: white;
        }

        .zoom-controls {
            display: flex;
            gap: 10px;
        }

        .zoom-btn {
            padding: 8px 12px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            cursor: pointer;
            transition: all 0.3s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .zoom-btn.active {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .timeline-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-card .label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }

        .timeline-container {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 40px 20px;
            position: relative;
            overflow-x: auto;
            overflow-y: visible;
            height: calc(100vh - 400px);
            min-height: 500px;
        }

        .timeline-scroll {
            position: relative;
            min-width: 100%;
            width: max-content;
            padding: 60px 20px;
        }

        .timeline-axis {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-color), var(--secondary-color));
            border-radius: 2px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .timeline-year-markers {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            transform: translateY(-50%);
        }

        .year-marker {
            position: absolute;
            width: 2px;
            height: 20px;
            background: var(--border-color);
            transform: translateX(-50%);
        }

        .year-label {
            position: absolute;
            transform: translate(-50%, 30px);
            font-weight: 700;
            color: var(--secondary-color);
            font-size: 1.1em;
            white-space: nowrap;
        }

        .timeline-events {
            position: relative;
            min-height: 400px;
        }

        .timeline-event {
            position: absolute;
            width: 280px;
            cursor: pointer;
            animation: fadeInUp 0.5s ease-out;
            transition: all 0.3s;
        }

        .timeline-event.top {
            bottom: calc(50% + 30px);
        }

        .timeline-event.bottom {
            top: calc(50% + 30px);
        }

        .event-card {
            background: var(--bg-color);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
            position: relative;
        }

        .timeline-event:hover .event-card {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            border-color: var(--accent-color);
        }

        .event-dot {
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: white;
            border: 4px solid;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }

        .timeline-event.top .event-dot {
            top: 100%;
            margin-top: 14px;
        }

        .timeline-event.bottom .event-dot {
            bottom: 100%;
            margin-bottom: 14px;
        }

        .timeline-event:hover .event-dot {
            transform: translateX(-50%) scale(1.3);
        }

        .event-connector {
            position: absolute;
            width: 2px;
            background: var(--border-color);
            left: 50%;
            transform: translateX(-50%);
        }

        .timeline-event.top .event-connector {
            top: 100%;
            height: 30px;
        }

        .timeline-event.bottom .event-connector {
            bottom: 100%;
            height: 30px;
        }

        .event-type-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .event-type-hardware { background: #e74c3c; color: white; }
        .event-type-software { background: #3498db; color: white; }
        .event-type-music { background: #1abc9c; color: white; }
        .event-type-game { background: #16a085; color: white; }
        .event-type-company { background: #f39c12; color: white; }
        .event-type-release { background: #9b59b6; color: white; }
        .event-type-demo { background: #e67e22; color: white; }
        .event-type-magazine { background: #95a5a6; color: white; }

        .event-dot.hardware { border-color: #e74c3c; }
        .event-dot.software { border-color: #3498db; }
        .event-dot.music { border-color: #1abc9c; }
        .event-dot.game { border-color: #16a085; }
        .event-dot.company { border-color: #f39c12; }
        .event-dot.release { border-color: #9b59b6; }
        .event-dot.demo { border-color: #e67e22; }
        .event-dot.magazine { border-color: #95a5a6; }

        .event-title {
            font-weight: 700;
            font-size: 1.1em;
            color: var(--text-color);
            margin-bottom: 6px;
            line-height: 1.3;
        }

        .event-date {
            color: var(--accent-color);
            font-weight: 600;
            font-size: 0.9em;
            margin-bottom: 8px;
        }

        .event-description {
            font-size: 0.9em;
            color: var(--text-muted);
            line-height: 1.4;
            margin-bottom: 8px;
        }

        .event-meta {
            font-size: 0.8em;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 8px;
            margin-top: 8px;
        }

        .event-icon {
            font-size: 1.5em;
            margin-right: 8px;
        }

        .empty-timeline {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .loading-timeline {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 768px) {
            .timeline-controls {
                flex-direction: column;
                align-items: stretch;
            }

            .timeline-filters,
            .zoom-controls {
                justify-content: center;
            }

            .timeline-event {
                width: 240px;
            }

            .timeline-stats {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* Modal for event details */
        .event-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            padding: 20px;
            overflow-y: auto;
        }

        .event-modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 30px;
            max-width: 600px;
            width: 100%;
            position: relative;
            animation: modalSlideIn 0.3s ease-out;
        }

        @keyframes modalSlideIn {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .modal-close {
            position: absolute;
            top: 15px;
            right: 15px;
            width: 32px;
            height: 32px;
            border: 2px solid var(--border-color);
            border-radius: 50%;
            background: var(--bg-color);
            color: var(--text-color);
            font-size: 1.2em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }

        .modal-close:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .modal-event-title {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--secondary-color);
            margin-bottom: 15px;
        }

        .modal-event-date {
            font-size: 1.2em;
            color: var(--accent-color);
            font-weight: 600;
            margin-bottom: 15px;
        }

        .modal-event-description {
            font-size: 1.05em;
            line-height: 1.6;
            color: var(--text-color);
            margin-bottom: 20px;
        }

        .modal-event-meta {
            background: var(--bg-color);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .modal-event-meta p {
            margin: 5px 0;
            font-size: 0.95em;
        }

        .modal-related-docs {
            margin-top: 20px;
        }

        .modal-related-docs h3 {
            color: var(--secondary-color);
            margin-bottom: 10px;
        }

        .related-doc-link {
            display: block;
            padding: 10px;
            margin: 5px 0;
            background: var(--bg-color);
            border-radius: 6px;
            border-left: 3px solid var(--accent-color);
            color: var(--text-color);
            text-decoration: none;
            transition: all 0.2s;
        }

        .related-doc-link:hover {
            background: var(--border-color);
            transform: translateX(4px);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📅 Interactive Timeline</h1>
            <p class="subtitle">Explore Commodore 64 history chronologically</p>
        </header>

{NAV}

{ABOUT}

        <div class="timeline-stats" id="timeline-stats">
            <div class="stat-card">
                <div class="value" id="total-events">-</div>
                <div class="label">Total Events</div>
            </div>
            <div class="stat-card">
                <div class="value" id="date-range">-</div>
                <div class="label">Year Range</div>
            </div>
            <div class="stat-card">
                <div class="value" id="event-types">-</div>
                <div class="label">Event Types</div>
            </div>
        </div>

        <div class="timeline-wrapper">
            <div class="timeline-controls">
                <div class="timeline-filters" id="type-filters">
                    <button class="filter-btn active" data-type="all">All Events</button>
                </div>
                <div class="zoom-controls">
                    <button class="zoom-btn" data-zoom="compact">Compact</button>
                    <button class="zoom-btn active" data-zoom="normal">Normal</button>
                    <button class="zoom-btn" data-zoom="expanded">Expanded</button>
                </div>
            </div>

            <div class="timeline-container">
                <div class="loading-timeline">Loading timeline events...</div>
                <div class="timeline-scroll" id="timeline-scroll" style="display: none;">
                    <div class="timeline-axis"></div>
                    <div class="timeline-year-markers" id="year-markers"></div>
                    <div class="timeline-events" id="timeline-events"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Event Detail Modal -->
    <div class="event-modal" id="event-modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeEventModal()">✕</button>
            <div id="modal-event-content"></div>
        </div>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
    <script>
        let timelineEvents = [];
        let currentZoom = 'normal';
        let activeFilters = new Set(['all']);

        // Event type icons
        const typeIcons = {
            'hardware': '🖥️',
            'software': '💾',
            'music': '🎵',
            'game': '🎮',
            'company': '🏢',
            'release': '🚀',
            'demo': '✨',
            'magazine': '📰',
            'default': '📅'
        };

        async function loadTimeline() {
            try {
                const response = await fetch('assets/data/events.json');
                timelineEvents = await response.json();

                if (timelineEvents.length === 0) {
                    document.querySelector('.loading-timeline').textContent = 'No events found';
                    return;
                }

                // Sort by year
                timelineEvents.sort((a, b) => a.year - b.year);

                // Update stats
                updateStats();

                // Build type filters
                buildTypeFilters();

                // Render timeline
                renderTimeline();

                // Hide loading, show timeline
                document.querySelector('.loading-timeline').style.display = 'none';
                document.getElementById('timeline-scroll').style.display = 'block';
            } catch (error) {
                console.error('Error loading timeline:', error);
                document.querySelector('.loading-timeline').textContent = 'Error loading timeline';
            }
        }

        function updateStats() {
            const years = timelineEvents.map(e => e.year);
            const minYear = Math.min(...years);
            const maxYear = Math.max(...years);
            const types = [...new Set(timelineEvents.map(e => e.type))];

            document.getElementById('total-events').textContent = timelineEvents.length;
            document.getElementById('date-range').textContent = `${minYear}-${maxYear}`;
            document.getElementById('event-types').textContent = types.length;
        }

        function buildTypeFilters() {
            const types = [...new Set(timelineEvents.map(e => e.type))].sort();
            const container = document.getElementById('type-filters');

            types.forEach(type => {
                const btn = document.createElement('button');
                btn.className = 'filter-btn';
                btn.dataset.type = type;
                btn.innerHTML = `${typeIcons[type] || typeIcons.default} ${capitalize(type)}`;
                btn.onclick = () => toggleFilter(type, btn);
                container.appendChild(btn);
            });

            // Setup zoom controls
            document.querySelectorAll('.zoom-btn').forEach(btn => {
                btn.onclick = () => setZoom(btn.dataset.zoom);
            });
        }

        function toggleFilter(type, btn) {
            if (type === 'all') {
                activeFilters.clear();
                activeFilters.add('all');
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            } else {
                activeFilters.delete('all');
                document.querySelector('.filter-btn[data-type="all"]').classList.remove('active');

                if (activeFilters.has(type)) {
                    activeFilters.delete(type);
                    btn.classList.remove('active');
                } else {
                    activeFilters.add(type);
                    btn.classList.add('active');
                }

                if (activeFilters.size === 0) {
                    activeFilters.add('all');
                    document.querySelector('.filter-btn[data-type="all"]').classList.add('active');
                }
            }

            renderTimeline();
        }

        function setZoom(zoom) {
            currentZoom = zoom;
            document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
            document.querySelector(`.zoom-btn[data-zoom="${zoom}"]`).classList.add('active');
            renderTimeline();
        }

        function renderTimeline() {
            const filtered = activeFilters.has('all')
                ? timelineEvents
                : timelineEvents.filter(e => activeFilters.has(e.type));

            if (filtered.length === 0) {
                document.getElementById('timeline-events').innerHTML = '<div class="empty-timeline">No events match the selected filters</div>';
                return;
            }

            const years = filtered.map(e => e.year);
            const minYear = Math.min(...years);
            const maxYear = Math.max(...years);
            const yearRange = maxYear - minYear || 1;

            // Calculate spacing based on zoom
            const spacing = {
                'compact': 150,
                'normal': 250,
                'expanded': 400
            }[currentZoom];

            const totalWidth = yearRange * spacing + 200;

            // Render year markers
            const markersHtml = [];
            for (let year = minYear; year <= maxYear; year++) {
                const left = ((year - minYear) / yearRange) * (totalWidth - 200) + 100;
                markersHtml.push(`
                    <div class="year-marker" style="left: ${left}px;"></div>
                    <div class="year-label" style="left: ${left}px;">${year}</div>
                `);
            }
            document.getElementById('year-markers').innerHTML = markersHtml.join('');

            // Render events
            const eventsHtml = filtered.map((event, index) => {
                const left = ((event.year - minYear) / yearRange) * (totalWidth - 200) + 100;
                const position = index % 2 === 0 ? 'top' : 'bottom';
                const icon = typeIcons[event.type] || typeIcons.default;

                return `
                    <div class="timeline-event ${position}" style="left: ${left}px;" onclick="showEventModal(${JSON.stringify(event).replace(/"/g, '&quot;')})">
                        <div class="event-card">
                            <div class="event-type-badge event-type-${event.type}">${icon} ${capitalize(event.type)}</div>
                            <div class="event-title">${escapeHtml(event.title)}</div>
                            <div class="event-date">${formatDate(event.date_normalized, event.year)}</div>
                            <div class="event-description">${escapeHtml(truncate(event.description, 100))}</div>
                        </div>
                        <div class="event-connector"></div>
                        <div class="event-dot ${event.type}"></div>
                    </div>
                `;
            }).join('');

            document.getElementById('timeline-events').innerHTML = eventsHtml;

            // Set scroll width
            document.getElementById('timeline-scroll').style.width = totalWidth + 'px';
        }

        function showEventModal(event) {
            const modal = document.getElementById('event-modal');
            const content = document.getElementById('modal-event-content');

            const icon = typeIcons[event.type] || typeIcons.default;

            content.innerHTML = `
                <div class="event-type-badge event-type-${event.type}">${icon} ${capitalize(event.type)}</div>
                <h2 class="modal-event-title">${escapeHtml(event.title)}</h2>
                <div class="modal-event-date">${formatDate(event.date_normalized, event.year)}</div>
                <div class="modal-event-description">${escapeHtml(event.description)}</div>
                ${event.confidence ? `
                    <div class="modal-event-meta">
                        <p><strong>Event ID:</strong> ${event.event_id}</p>
                        <p><strong>Confidence:</strong> ${(event.confidence * 100).toFixed(0)}%</p>
                    </div>
                ` : ''}
            `;

            modal.classList.add('active');
        }

        function closeEventModal() {
            document.getElementById('event-modal').classList.remove('active');
        }

        // Close modal on background click
        document.getElementById('event-modal').addEventListener('click', (e) => {
            if (e.target.id === 'event-modal') {
                closeEventModal();
            }
        });

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeEventModal();
            }
        });

        function formatDate(dateStr, year) {
            if (dateStr && dateStr !== 'None') {
                const date = new Date(dateStr);
                return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
            }
            return year;
        }

        function capitalize(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        }

        function truncate(str, length) {
            if (!str) return '';
            return str.length > length ? str.substring(0, length) + '...' : str;
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load timeline on page load
        loadTimeline();
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("timeline"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("entities"))

        filepath = self.output_dir / "timeline.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: timeline.html")
