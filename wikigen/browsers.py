"""The browse/inspect pages: documents, chunks, file viewer, settings.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""

import html


class BrowsersMixin:
    """The browse/inspect pages: documents, chunks, file viewer, settings."""

    def _generate_documents_browser_html(self):
        """Generate documents browser page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documents - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Browse All Documents</p>
        </header>

{NAV}

        <main>
{ABOUT}

            <div class="browser-controls">
                <input type="text" id="doc-search" placeholder="🔍 Search documents..." autocomplete="off">

                <div class="filter-buttons">
                    <button class="filter-btn active" data-type="all">All Types</button>
                    <button class="filter-btn" data-type="pdf">PDF</button>
                    <button class="filter-btn" data-type="text">Text</button>
                    <button class="filter-btn" data-type="html">HTML</button>
                    <button class="filter-btn" data-type="markdown">Markdown</button>
                </div>

                <div class="sort-controls">
                    <label>Sort by:</label>
                    <select id="sort-select">
                        <option value="title">Title (A-Z)</option>
                        <option value="title-desc">Title (Z-A)</option>
                        <option value="chunks">Chunks (Most)</option>
                        <option value="chunks-asc">Chunks (Least)</option>
                        <option value="date">Date Added</option>
                    </select>
                </div>
            </div>

            <div id="documents-grid" class="documents-grid">
                <!-- Populated by JavaScript -->
            </div>

            <button id="back-to-top" class="back-to-top">↑ Top</button>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/documents.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("documents"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("documents"))

        filepath = self.output_dir / "documents.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: documents.html")

    def _generate_chunks_browser_html(self):
        """Generate chunks browser page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chunks - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Browse Text Chunks</p>
        </header>

{NAV}

        <main>
{ABOUT}

            <div class="browser-controls">
                <input type="text" id="chunk-search" placeholder="🔍 Search chunks..." autocomplete="off">

                <div class="chunk-stats" id="chunk-stats">
                    <!-- Populated by JavaScript -->
                </div>
            </div>

            <div id="chunks-list" class="chunks-list">
                <!-- Populated by JavaScript -->
            </div>

            <div class="pagination" id="pagination">
                <!-- Populated by JavaScript -->
            </div>

            <button id="back-to-top" class="back-to-top">↑ Top</button>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/chunks.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("chunks"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("chunks"))

        filepath = self.output_dir / "chunks.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: chunks.html")

    def _generate_file_viewer_html(self):
        """Generate universal file viewer page using standard HTML5 components."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Viewer - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .viewer-container {
            width: 100%;
            height: calc(100vh - 250px);
            min-height: 600px;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            background: white;
            overflow: auto;
            margin: 20px 0;
        }
        .viewer-container iframe,
        .viewer-container embed,
        .viewer-container object {
            width: 100%;
            height: 100%;
            border: none;
        }
        .text-viewer {
            padding: 30px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #2d3748;
            background: white;
        }
        .markdown-viewer {
            padding: 30px;
            max-width: 900px;
            margin: 0 auto;
            background: white;
            color: #2d3748;
            line-height: 1.6;
        }
        .html-viewer {
            width: 100%;
            height: 100%;
        }
        .viewer-controls {
            display: flex;
            gap: 15px;
            padding: 20px;
            background: var(--card-bg);
            border-radius: 12px;
            margin-bottom: 20px;
            align-items: center;
            justify-content: space-between;
        }
        .viewer-controls .file-info {
            flex: 1;
        }
        .viewer-controls .file-name {
            font-weight: 600;
            color: var(--text-color);
            font-size: 1.1em;
        }
        .viewer-controls .file-type {
            color: var(--text-muted);
            font-size: 0.9em;
            margin-top: 4px;
        }
        .download-btn {
            padding: 10px 20px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: background 0.3s;
        }
        .download-btn:hover {
            background: var(--secondary-color);
        }
        .error-message {
            padding: 40px;
            text-align: center;
            color: #e53e3e;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle"><a href="documents.html">← Back to Documents</a></p>
        </header>

{NAV}

{ABOUT}

        <main>
            <div class="viewer-controls">
                <div class="file-info">
                    <div class="file-name" id="file-name">Loading...</div>
                    <div class="file-type" id="file-type"></div>
                </div>
                <a id="download-link" href="#" download class="download-btn">Download File</a>
            </div>

            <div class="viewer-container" id="viewer-container">
                <div style="text-align: center; padding: 40px; color: var(--text-muted);">
                    Loading file...
                </div>
            </div>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v2.23.16</p>
        </footer>
    </div>

    <script src="lib/marked.min.js"></script>
    <script>
        // Get file path from URL parameter
        const urlParams = new URLSearchParams(window.location.search);
        const filePath = urlParams.get('file');
        const fileName = urlParams.get('name') || 'Document';
        const fileType = urlParams.get('type') || 'unknown';

        // Update UI
        document.getElementById('file-name').textContent = fileName;
        document.getElementById('file-type').textContent = `Type: ${fileType.toUpperCase()}`;

        if (filePath) {
            document.getElementById('download-link').href = filePath;
            document.getElementById('download-link').download = fileName;

            const container = document.getElementById('viewer-container');

            // Display based on file type
            if (fileType === 'pdf') {
                // Use browser's native PDF viewer via iframe or embed
                container.innerHTML = `
                    <iframe src="${filePath}" type="application/pdf"></iframe>
                `;
            } else if (fileType === 'html') {
                // Display HTML in iframe
                container.innerHTML = `
                    <iframe src="${filePath}" class="html-viewer"></iframe>
                `;
            } else if (fileType === 'markdown' || fileType === 'md') {
                // Fetch and render markdown
                fetch(filePath)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.text();
                    })
                    .then(text => {
                        container.innerHTML = `<div class="markdown-viewer">${marked.parse(text)}</div>`;
                    })
                    .catch(error => {
                        container.innerHTML = `<div class="error-message">Error loading markdown file from "${filePath}": ${error.message}</div>`;
                    });
            } else {
                // Display as text
                fetch(filePath)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.text();
                    })
                    .then(text => {
                        const escaped = text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        container.innerHTML = `<div class="text-viewer">${escaped}</div>`;
                    })
                    .catch(error => {
                        container.innerHTML = `<div class="error-message">Error loading file from "${filePath}": ${error.message}</div>`;
                    });
            }
        } else {
            document.getElementById('viewer-container').innerHTML =
                '<div class="error-message">No file specified</div>';
        }
    </script>
    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav())
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("viewer"))

        filepath = self.output_dir / "viewer.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: viewer.html")

    def _generate_settings_html(self):
        """Generate settings/configuration information page."""
        import os
        from pathlib import Path

        # Get paths and configuration
        data_dir = Path(self.kb.data_dir).resolve()
        db_file = Path(self.kb.db_file).resolve()
        wiki_dir = Path(self.output_dir).resolve()

        # Get database size
        db_size_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0

        # Get wiki directory size
        wiki_size_mb = sum(f.stat().st_size for f in wiki_dir.rglob('*') if f.is_file()) / (1024 * 1024)

        # Environment variables
        env_vars = {
            'TDZ_DATA_DIR': os.getenv('TDZ_DATA_DIR', 'Not set (using default)'),
            'USE_FTS5': os.getenv('USE_FTS5', 'Not set'),
            'USE_SEMANTIC_SEARCH': os.getenv('USE_SEMANTIC_SEARCH', 'Not set'),
            'LLM_PROVIDER': os.getenv('LLM_PROVIDER', 'Not set'),
        }

        # Build environment variables HTML
        env_html = '\n'.join([
            f'<tr><td class="setting-key">{key}</td><td class="setting-value">{value}</td></tr>'
            for key, value in env_vars.items()
        ])

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .settings-section {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            border-left: 4px solid var(--accent-color);
        }}
        .settings-section h2 {{
            margin-top: 0;
            color: var(--secondary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}
        .settings-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .settings-table td {{
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
        }}
        .setting-key {{
            font-weight: 600;
            color: var(--text-color);
            width: 30%;
        }}
        .setting-value {{
            font-family: 'Courier New', monospace;
            color: var(--text-muted);
            word-break: break-all;
        }}
        .path-link {{
            color: var(--accent-color);
            text-decoration: none;
        }}
        .path-link:hover {{
            text-decoration: underline;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: var(--bg-color);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent-color);
        }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 0.9em;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 TDZ C64 Knowledge Base</h1>
            <p class="subtitle">Settings & Configuration</p>
        </header>

        {{NAV}}

        <main>
            <div class="settings-section">
                <h2>📊 Knowledge Base Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value">{self.stats['documents']}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.stats['chunks']}</div>
                        <div class="stat-label">Chunks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.stats['entities']}</div>
                        <div class="stat-label">Entities</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{db_size_mb:.1f} MB</div>
                        <div class="stat-label">Database Size</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{wiki_size_mb:.1f} MB</div>
                        <div class="stat-label">Wiki Size</div>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <h2>📁 File Paths</h2>
                <table class="settings-table">
                    <tr>
                        <td class="setting-key">Data Directory</td>
                        <td class="setting-value">{html.escape(str(data_dir))}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Database File</td>
                        <td class="setting-value">{html.escape(str(db_file))}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Wiki Directory</td>
                        <td class="setting-value">{html.escape(str(wiki_dir))}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Documents JSON</td>
                        <td class="setting-value">{html.escape(str(wiki_dir / 'assets' / 'data' / 'documents.json'))}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Entities JSON</td>
                        <td class="setting-value">{html.escape(str(wiki_dir / 'assets' / 'data' / 'entities.json'))}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Articles JSON</td>
                        <td class="setting-value">{html.escape(str(wiki_dir / 'assets' / 'data' / 'articles.json'))}</td>
                    </tr>
                </table>
            </div>

            <div class="settings-section">
                <h2>⚙️ Environment Variables</h2>
                <table class="settings-table">
                    {env_html}
                </table>
            </div>

            <div class="settings-section">
                <h2>🔧 Wiki Export Information</h2>
                <table class="settings-table">
                    <tr>
                        <td class="setting-key">Export Version</td>
                        <td class="setting-value">v{self.version}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Export Date</td>
                        <td class="setting-value">{self.export_time}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Python Version</td>
                        <td class="setting-value">{os.sys.version.split()[0]}</td>
                    </tr>
                </table>
            </div>

            <div class="settings-section">
                <h2>📚 Features Enabled</h2>
                <table class="settings-table">
                    <tr>
                        <td class="setting-key">FTS5 Search</td>
                        <td class="setting-value">{'✅ Enabled' if os.getenv('USE_FTS5') == '1' else '❌ Disabled'}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Semantic Search</td>
                        <td class="setting-value">{'✅ Enabled' if os.getenv('USE_SEMANTIC_SEARCH') == '1' else '❌ Disabled'}</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Entity Extraction</td>
                        <td class="setting-value">✅ Enabled</td>
                    </tr>
                    <tr>
                        <td class="setting-key">Article Generation</td>
                        <td class="setting-value">✅ Enabled</td>
                    </tr>
                </table>
            </div>
        </main>

        <footer>
            <p>TDZ C64 Knowledge Base v{self.version}</p>
        </footer>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav())

        filepath = self.output_dir / "settings.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: settings.html")
