#!/usr/bin/env python3
"""
Wiki Export Script

Exports the TDZ C64 Knowledge Base to a static HTML/JavaScript wiki.

Features:
- Exports all documents, chunks, entities, topics, clusters, events
- Generates search index for client-side search
- Creates navigation structure
- Builds interactive visualizations
- Produces fully static site (no server needed)

Usage:
    python wiki_export.py --output wiki/
"""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re


from wikigen import (ArticlesMixin, BrowsersMixin, DataExportMixin, DiagramsMixin,
                     PagesMixin, VisualizationsMixin)
# Re-exported so `from wiki_export import safe_external_url` keeps working.
from wikigen.urls import safe_external_url, url_query_value, _SAFE_URL_SCHEMES  # noqa: F401


class WikiExporter(DataExportMixin, PagesMixin, VisualizationsMixin,
                   BrowsersMixin, ArticlesMixin, DiagramsMixin):
    """Exports knowledge base to static HTML wiki."""

    def __init__(self, kb: KnowledgeBase, output_dir: str):
        from version import __version__

        self.kb = kb
        self.output_dir = Path(output_dir)
        self.docs_dir = self.output_dir / "docs"
        self.assets_dir = self.output_dir / "assets"
        self.data_dir = self.assets_dir / "data"
        self.files_dir = self.output_dir / "files"  # Directory for actual source files

        # Version and export time
        self.version = __version__
        self.export_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Statistics
        self.stats = {
            'documents': 0,
            'chunks': 0,
            'entities': 0,
            'topics': 0,
            'clusters': 0,
            'events': 0,
            'export_date': datetime.now().isoformat()
        }

    def _get_unified_about_box(self, page: str = 'home') -> str:
        """Generate page-specific about box HTML."""
        about_content = {
            'home': """
                <h3>📚 About This Knowledge Base</h3>
                <p>
                    The <strong>TDZ C64 Knowledge Base</strong> contains {stats[documents]} documents, {stats[chunks]} text chunks,
                    and {stats[entities]} extracted entities. All content is searchable and interconnected for easy exploration.
                </p>
                <p>
                    <strong>Quick Start:</strong> Use the search bar above, browse documents below, or explore entities,
                    topics, and visualizations through the navigation menu.
                </p>
            """,
            'documents': """
                <h3>📄 About Documents</h3>
                <p>
                    Browse all {stats[documents]} documents in the knowledge base. Each document has been processed to extract
                    text, identify entities, and generate searchable chunks. Click any document to view its full content,
                    metadata, and extracted information.
                </p>
                <p>
                    <strong>Features:</strong> Filter by tags • Sort by date or title • View source files •
                    See related entities and topics
                </p>
            """,
            'chunks': """
                <h3>📝 About Text Chunks</h3>
                <p>
                    This page displays all {stats[chunks]} text chunks extracted from documents. Chunks are searchable
                    segments of text (typically 1500 words with 200-word overlap) that enable precise content discovery.
                </p>
                <p>
                    <strong>Use Cases:</strong> Find specific technical details • Locate code examples •
                    Search across document boundaries
                </p>
            """,
            'entities': """
                <h3>🏷️ About Entities</h3>
                <p>
                    The knowledge base has identified {stats[entities]} entities across documents, including hardware
                    components (VIC-II, SID, CIA), memory addresses, instructions, and concepts. Each entity links
                    to all documents where it appears.
                </p>
                <p>
                    <strong>Entity Types:</strong> Hardware • Memory Addresses • Instructions • People • Companies •
                    Products • Concepts
                </p>
            """,
            'knowledge-graph': """
                <h3>🕸️ About Knowledge Graph</h3>
                <p>
                    This interactive graph visualizes relationships between entities in the knowledge base.
                    Nodes represent entities, and edges show their connections based on co-occurrence in documents.
                </p>
                <p>
                    <strong>Interactions:</strong> Drag nodes to explore • Click to see details •
                    Search for specific entities • Filter by entity type
                </p>
            """,
            'similarity-map': """
                <h3>🗺️ About Similarity Map</h3>
                <p>
                    This 2D visualization maps documents in semantic space based on their content similarity.
                    Documents positioned close together share related topics and concepts.
                </p>
                <p>
                    <strong>Interactions:</strong> Hover over points to see document details • Click to navigate •
                    Zoom and pan to explore • Filter by cluster or file type
                </p>
            """,
            'topics': """
                <h3>📊 About Topics</h3>
                <p>
                    Automatically discovered topics and document clusters from the knowledge base. Topics are identified
                    using machine learning algorithms (LDA, NMF, BERTopic) to find common themes across documents.
                </p>
                <p>
                    <strong>Features:</strong> View topic keywords • See documents per topic •
                    Explore clustering results • Understand knowledge base structure
                </p>
            """,
            'timeline': """
                <h3>📅 About Timeline</h3>
                <p>
                    A chronological view of C64 history and documentation events. Events are automatically extracted
                    from documents based on dates and historical references.
                </p>
                <p>
                    <strong>Features:</strong> Filter by date range • View event details •
                    See related documents • Explore historical context
                </p>
            """,
            'articles': """
                <h3>📰 About Articles</h3>
                <p>
                    AI-generated articles based on extracted entities and document analysis. Each article aggregates
                    information from multiple sources to provide comprehensive coverage of C64 topics.
                </p>
                <p>
                    <strong>Categories:</strong> Hardware components • Programming concepts • Memory addresses •
                    Software tools • Historical context
                </p>
            """,
            'viewer': """
                <h3>👁️ About File Viewer</h3>
                <p>
                    View source documents in their original format. Supports PDF (browser native), HTML (iframe),
                    Markdown (rendered), and plain text files.
                </p>
                <p>
                    <strong>Supported Formats:</strong> PDF • HTML • Markdown • Plain Text
                </p>
            """
        }

        content = about_content.get(page, about_content['home'])
        return f"""
            <div class="explanation-box">
                {content.format(stats=self.stats)}
            </div>
"""

    def _get_main_nav(self, active_page: str = '') -> str:
        """Generate consistent main navigation HTML with logo, search, and theme switcher."""
        pages = [
            ('articles', 'Articles'),
            ('documents', 'Documents'),
            ('chunks', 'Chunks'),
            ('entities', 'Entities'),
            ('knowledge-graph', 'Knowledge Graph'),
            ('similarity-map', 'Similarity Map'),
            ('topics', 'Topics'),
            ('timeline', 'Timeline')
        ]

        nav_items = []
        for page_key, display_name in pages:
            active_class = ' class="active"' if page_key == active_page else ''
            page_file = page_key + '.html'
            nav_items.append(f'            <a href="{page_file}"{active_class}>{display_name}</a>')

        return f"""    <nav class="main-nav">
        <div class="nav-left">
            <a href="index.html" class="nav-logo">📚 TDZ C64 KB</a>
        </div>
        <div class="nav-center">
{chr(10).join(nav_items)}
        </div>
        <div class="nav-right">
            <div class="search-container">
                <input type="search" id="nav-search" class="nav-search" placeholder="🔍 Search..." autocomplete="off" />
                <div id="nav-search-results" class="search-results"></div>
            </div>
            <button class="theme-switcher" id="theme-toggle" aria-label="Toggle theme">🌙</button>
        </div>
    </nav>"""

    def export(self):
        """Main export function."""
        print("=" * 60)
        print("TDZ C64 Knowledge Base - Wiki Export")
        print("=" * 60)
        print(f"\nOutput directory: {self.output_dir}")

        # Create directory structure
        print("\n[1/7] Creating directory structure...")
        self._create_directories()

        # Export data
        print("[2/7] Exporting documents...")
        documents_data = self._export_documents()
        self._copy_source_files(documents_data)

        print("[3/7] Exporting entities...")
        entities_data = self._export_entities()
        graph_data = self._export_graph()

        print("  Exporting document coordinates...")
        coordinates_data = self._export_document_coordinates(documents_data)

        print("[4/7] Exporting topics and clusters...")
        topics_data = self._export_topics()
        clusters_data = self._export_clusters()

        print("[5/7] Exporting events...")
        events_data = self._export_events()

        print("[6/9] Generating navigation...")
        navigation = self._build_navigation(documents_data)

        # Export chunks data
        print("[8/9] Exporting chunks...")
        chunks_data = self._export_chunks()

        # Copy PDFs before saving documents.json (so we know which PDFs are available)
        print("[9/10] Copying PDF files...")
        copied_pdf_ids = self._copy_pdfs(documents_data)

        # Add PDF availability info to documents data
        for doc in documents_data:
            doc['pdf_available'] = doc['id'] in copied_pdf_ids if doc['file_type'].lower() == 'pdf' else False

        # Save data files
        print("\nSaving data files...")
        self._save_json('documents.json', documents_data)
        self._save_json('entities.json', entities_data)
        self._save_json('graph.json', graph_data)
        self._save_json('coordinates.json', coordinates_data)
        self._save_json('topics.json', topics_data)
        self._save_json('clusters.json', clusters_data)
        self._save_json('events.json', events_data)
        self._save_json('navigation.json', navigation)
        self._save_json('chunks.json', chunks_data)
        self._save_json('stats.json', self.stats)

        # Calculate document similarities
        print("\nCalculating document similarities...")
        similarities = self._calculate_document_similarities(documents_data, entities_data)
        self._save_json('similarities.json', similarities)

        # Generate HTML pages (with PDF availability info already set)
        print("\nGenerating HTML pages...")
        self._generate_html_pages(documents_data)

        # Generate articles
        articles_data = self._generate_articles(entities_data)

        # Build comprehensive search index (after articles are generated)
        print("\n[7/9] Building comprehensive search index...")
        search_index_data = self._export_search_index(documents_data, entities_data, articles_data)
        self._save_json('search.json', search_index_data)

        # Copy static assets
        print("\nCopying static assets...")
        self._copy_static_assets()

        # Generate README last, after every HTML page and data file exists,
        # so its counts/sizes are read from the actual output rather than
        # hand-maintained prose that inevitably drifts from what export()
        # produces (see GitHub issue #7).
        print("\nGenerating wiki/README.md...")
        self._generate_readme()

        # Print summary
        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  Documents: {self.stats['documents']}")
        print(f"  Chunks: {self.stats['chunks']}")
        print(f"  Entities: {self.stats['entities']}")
        print(f"  Topics: {self.stats['topics']}")
        print(f"  Clusters: {self.stats['clusters']}")
        print(f"  Events: {self.stats['events']}")
        print(f"  Articles: {self.stats.get('articles', 0)}")
        print(f"\nWiki location: {self.output_dir.absolute()}")
        print(f"Open: {(self.output_dir / 'index.html').absolute()}")
        print("\n" + "=" * 60)

    def _create_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
        (self.assets_dir / "css").mkdir(exist_ok=True)
        (self.assets_dir / "js").mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Clear stale data files from a previous export before writing this
        # one. export() only ever *writes* the current set of data files - it
        # never removed files left behind by an older run, so a file dropped
        # or renamed between versions (e.g. search-index.json -> search.json)
        # would linger indefinitely and get served alongside the current
        # export (see GitHub issue #7 - a 68 MB orphan from 2026-01-04 was
        # still sitting next to a 2026-07-19 export).
        for stale in self.data_dir.glob('*.json'):
            stale.unlink()

        # Same problem, different shape: root-level HTML pages this version
        # of export() no longer generates (e.g. pdf-viewer.html/test_viewer.html
        # from a much older layout) otherwise linger forever too.
        current_pages = {
            'index.html', 'articles.html', 'chunks.html', 'documents.html',
            'entities.html', 'knowledge-graph.html', 'settings.html',
            'similarity-map.html', 'timeline.html', 'topics.html', 'viewer.html',
        }
        for stale in self.output_dir.glob('*.html'):
            if stale.name not in current_pages:
                stale.unlink()

        (self.output_dir / "lib").mkdir(exist_ok=True)
        self.files_dir.mkdir(exist_ok=True)  # For actual source files


    def _copy_source_files(self, documents_data: List[Dict]):
        """Copy source files to the files directory for direct viewing."""
        print("  Copying source files to wiki...")
        copied_count = 0

        for doc in documents_data:
            filepath = doc.get('filepath')
            if not filepath or not os.path.exists(filepath):
                continue

            # Create a safe filename
            doc_id = doc['id']
            file_ext = Path(filepath).suffix
            safe_filename = re.sub(r'[^\w\-]', '_', doc_id) + file_ext
            dest_path = self.files_dir / safe_filename

            try:
                shutil.copy2(filepath, dest_path)
                # Store the relative path for linking
                doc['file_path_in_wiki'] = f"files/{safe_filename}"
                copied_count += 1
            except Exception as e:
                print(f"    Warning: Could not copy {filepath}: {e}")

        print(f"  Copied {copied_count} source files")












    def _save_json(self, filename: str, data: Any):
        """Save data as JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filename}")

    @staticmethod
    def _human_size(num_bytes: int) -> str:
        """Format a byte count as a human-readable size string."""
        size = float(num_bytes)
        for unit in ('B', 'KB', 'MB', 'GB'):
            if size < 1024 or unit == 'GB':
                return f"{size:.1f} {unit}" if unit != 'B' else f"{int(size)} {unit}"
            size /= 1024
        return f"{size:.1f} GB"














    def _copy_pdfs(self, documents: List[Dict]) -> set:
        """Copy PDF files to wiki directory.

        Returns:
            set: Set of document IDs for PDFs that were successfully copied
        """
        pdfs_dir = self.output_dir / "pdfs"
        pdfs_dir.mkdir(exist_ok=True)

        pdf_count = 0
        copied_pdf_ids = set()

        for doc in documents:
            if doc['file_type'].lower() == 'pdf':
                # Try to find the original PDF file
                doc_meta = self.kb.documents.get(doc['id'])
                if doc_meta and hasattr(doc_meta, 'filepath'):
                    source_path = Path(doc_meta.filepath)
                    if source_path.exists() and source_path.suffix.lower() == '.pdf':
                        dest_filename = re.sub(r'[^\w\-]', '_', doc['id']) + '.pdf'
                        dest_path = pdfs_dir / dest_filename
                        try:
                            shutil.copy2(source_path, dest_path)
                            pdf_count += 1
                            copied_pdf_ids.add(doc['id'])
                            # Set file_path_in_wiki so "View Source" button appears
                            doc['file_path_in_wiki'] = f"pdfs/{dest_filename}"
                        except Exception as e:
                            print(f"  Warning: Could not copy {source_path.name}: {e}")

        print(f"  Copied {pdf_count} PDF files")
        return copied_pdf_ids

    def _copy_static_assets(self):
        """Copy CSS, JS, and library files."""
        self._create_css()
        self._create_javascript()
        self._download_libraries()

    # ---- Emitted CSS/JS live in wiki_assets/, not in this file --------------
    # These two methods held 5,587 lines of triple-quoted CSS and JavaScript:
    # data, not Python. They are now files under wiki_assets/, read verbatim
    # and written with the SAME open(..., 'w', encoding='utf-8') call as
    # before, so the emitted bytes are unchanged - that call's platform
    # newline translation is what gives the exported wiki its CRLF endings,
    # and it still does. The assets are stored with LF and read with
    # newline='' so nothing translates twice; .gitattributes pins them to LF
    # because a CRLF checkout would translate a second time and corrupt the
    # output.
    _ASSET_ROOT = Path(__file__).resolve().parent / "wiki_assets"

    _JS_FILES = ('main.js', 'search.js', 'entities.js', 'topics.js',
                 'timeline.js', 'documents.js', 'chunks.js', 'pdf-viewer.js',
                 'enhancements.js')

    @classmethod
    def _read_asset(cls, *parts) -> str:
        """Read a shipped asset verbatim, without newline translation."""
        with open(cls._ASSET_ROOT.joinpath(*parts), encoding='utf-8', newline='') as f:
            return f.read()

    def _create_css(self):
        """Create CSS stylesheet."""
        css_path = self.assets_dir / "css" / "style.css"
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(self._read_asset("css", "style.css"))
        print("  Created: assets/css/style.css")

    def _create_javascript(self):
        """Create JavaScript files."""
        js_dir = self.assets_dir / "js"
        for filename in self._JS_FILES:
            filepath = js_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._read_asset("js", filename))
            print(f"  Created: assets/js/{filename}")

    # Every third-party script the exported wiki loads. All of them are
    # vendored into lib/ rather than referenced from a CDN, so the exported
    # wiki works offline and doesn't change behaviour when an upstream CDN
    # moves or rewrites a version. d3 (knowledge graph) and marked (file
    # viewer) used to be the exceptions - they were loaded straight from a
    # CDN, which silently broke those two pages with no network.
    _JS_LIBRARIES = (
        # (display name, url, local filename, what breaks without it)
        ("Fuse.js", "https://cdn.jsdelivr.net/npm/fuse.js@7.0.0/dist/fuse.min.js",
         "fuse.min.js", "search functionality will be limited"),
        ("PDF.js", "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js",
         "pdf.min.js", "PDF viewing will not work"),
        ("PDF.js worker", "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js",
         "pdf.worker.min.js", "PDF viewing will not work"),
        ("D3.js", "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
         "d3.v7.min.js", "the knowledge graph page will not render"),
        ("Marked", "https://cdn.jsdelivr.net/npm/marked@11.1.0/marked.min.js",
         "marked.min.js", "markdown files will render as plain text"),
    )

    def _download_libraries(self):
        """Vendor every third-party JS library into lib/ for offline use."""
        import urllib.request

        lib_dir = self.output_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)

        for name, url, filename, consequence in self._JS_LIBRARIES:
            path = lib_dir / filename

            # An export re-run shouldn't re-download what's already vendored,
            # but a previous run's fallback stub must not be mistaken for the
            # real library (stubs are a few hundred bytes at most).
            if path.exists() and path.stat().st_size > 4096:
                print(f"  Already vendored: lib/{filename}")
                continue

            try:
                print(f"  Downloading {name}...")
                urllib.request.urlretrieve(url, path)
                print(f"  Downloaded: lib/{filename}")
            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")
                print(f"  Consequence: {consequence}")
                path.write_text(
                    f"// {name} could not be downloaded automatically.\n"
                    f"// Download it from: {url}\n"
                    f"// and save it as lib/{filename}\n\n"
                    f"console.warn('{name} not loaded - {consequence}');\n",
                    encoding='utf-8',
                )


















def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Export KB to static HTML wiki')
    parser.add_argument('--output', default='wiki', help='Output directory (default: wiki/)')
    parser.add_argument('--data-dir', help='Knowledge base data directory')

    args = parser.parse_args()

    # Initialize KB
    data_dir = args.data_dir or os.path.expanduser('~/.tdz-c64-knowledge')
    print(f"Loading knowledge base from: {data_dir}")
    kb = KnowledgeBase(data_dir)

    # Export
    exporter = WikiExporter(kb, args.output)
    exporter.export()


if __name__ == '__main__':
    main()
