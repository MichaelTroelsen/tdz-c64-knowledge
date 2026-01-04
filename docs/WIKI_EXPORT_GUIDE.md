# Wiki Export Guide

## Overview

The TDZ C64 Knowledge Base can export its entire contents as a **fully static HTML/JavaScript wiki** that requires no server and works completely offline in your browser.

## Quick Start

### Export the Wiki

```bash
python wiki_export.py --output wiki
```

This creates a `wiki/` directory with:
- ✅ All 215+ documents as individual HTML pages
- ✅ Client-side search functionality
- ✅ Entity browser
- ✅ Topic and cluster visualizations
- ✅ Timeline of C64 events
- ✅ Complete navigation structure

### View the Wiki

**Option 1: Quick Start (Windows)**
```cmd
start-wiki.bat
```
This automatically starts the web server and opens your browser!

**Option 2: Direct File Access**
```
Open wiki/index.html in your browser
```

**Option 3: Local Web Server (Manual)**
```bash
cd wiki
python -m http.server 8080
# Open http://localhost:8080
```

**Option 4: Deploy Online**
Upload the `wiki/` directory to any static hosting (GitHub Pages, Netlify, Vercel, etc.)

## Features

### 🔍 Full-Text Search
- Client-side fuzzy search using Fuse.js
- Searches across all 215 documents
- Instant results as you type
- No server required

### 📚 Document Browser
- 215 individual document pages
- Organized by tags and file types
- Full content display with chunks
- Source URLs preserved

### 🏷️ Entity Browser
- 1,181 extracted entities
- Grouped by type (Hardware, Instruction, Concept, Company)
- Document count for each entity
- Filterable interface

### 📊 Topics & Clusters
- 14 discovered topics (LDA, NMF, BERTopic)
- 15 document clusters (K-Means, DBSCAN, HDBSCAN)
- Visual representation
- Algorithm comparison

### 📅 Historical Timeline
- 15+ C64 historical events
- Chronologically ordered
- Event types and confidence scores
- Interactive display

## Command-Line Options

```bash
python wiki_export.py [OPTIONS]

Options:
  --output DIR       Output directory (default: wiki/)
  --data-dir DIR     Knowledge base data directory
                     (default: ~/.tdz-c64-knowledge)

Examples:
  # Export to default wiki/ directory
  python wiki_export.py

  # Export to custom directory
  python wiki_export.py --output c64-docs

  # Use custom data directory
  python wiki_export.py --data-dir /path/to/kb
```

## What Gets Exported

### Documents
- All 215 documents from the knowledge base
- Complete content organized by chunks
- Metadata: title, file type, page count, tags
- Source URLs (if available)

### Entities
- 1,181 unique entities across 5 types:
  - Hardware (703): VIC-II, SID, 6510, CIA, etc.
  - Instruction (641): LDA, STA, JMP, RTS, etc.
  - Concept (497): Sprites, interrupts, raster, etc.
  - Company (345): Commodore, MOS, CBM, etc.
  - Article (19): Technical articles and papers

### Topics
- LDA topics: Statistical topic modeling
- NMF topics: Non-negative matrix factorization
- BERTopic: Neural topic modeling

### Clusters
- K-Means: Centroid-based clustering
- DBSCAN: Density-based clustering
- HDBSCAN: Hierarchical density clustering

### Timeline
- Historical C64 events
- Release dates
- Innovation milestones
- Development history

## Export Statistics

Typical export includes:
```
Documents:   215 HTML pages
Chunks:      6,107 text segments
Entities:    1,181 extracted entities
Topics:      14 discovered topics
Clusters:    15 document groups
Events:      15+ timeline events
Total Size:  ~137 MB
```

## File Structure

```
wiki/
├── index.html              # Main page with search
├── entities.html           # Entity browser
├── topics.html            # Topics & clusters
├── timeline.html          # Historical timeline
├── README.md              # Wiki usage guide
├── docs/                  # Document pages (215 files)
│   ├── <doc-id>.html
│   └── ...
├── assets/
│   ├── css/
│   │   └── style.css      # Complete styling
│   ├── js/
│   │   ├── main.js        # Core functionality
│   │   ├── search.js      # Search engine
│   │   ├── entities.js    # Entity browser
│   │   ├── topics.js      # Topics/clusters
│   │   └── timeline.js    # Timeline display
│   └── data/              # JSON data files
│       ├── documents.json      # 68 MB
│       ├── search-index.json   # 68 MB
│       ├── entities.json       # 111 KB
│       ├── topics.json         # 3 KB
│       ├── clusters.json       # 1 KB
│       ├── events.json         # 7 KB
│       ├── navigation.json     # 131 KB
│       └── stats.json          # Statistics
└── lib/
    └── fuse.min.js        # Search library
```

## Technology Stack

- **HTML5** - Semantic markup
- **CSS3** - Responsive design with CSS Grid and Flexbox
- **JavaScript (ES6)** - Modern async/await, modules
- **Fuse.js** - Client-side fuzzy search
- **No frameworks** - Pure HTML/CSS/JS (lightweight)

## Performance

- **Export Time:** ~30 seconds for 215 documents
- **Load Time:** 1-3 seconds (depends on browser)
- **Search:** Instant (client-side)
- **Offline:** 100% - works without internet

## Use Cases

### 1. Offline Documentation
Export and carry on USB drive or laptop for offline access

### 2. Static Website
Deploy to GitHub Pages or any static host for public access

### 3. Archive
Create timestamped archives of knowledge base state

### 4. Sharing
Share complete documentation as a single folder

### 5. Distribution
Package with C64 emulators or development tools

## Deployment Options

### GitHub Pages
```bash
# In your repository
git checkout --orphan gh-pages
cp -r wiki/* .
git add .
git commit -m "Deploy wiki"
git push origin gh-pages
```

### Netlify
```bash
# Drag and drop wiki/ folder to Netlify
# Or use CLI:
netlify deploy --dir=wiki --prod
```

### Vercel
```bash
vercel --prod wiki/
```

### AWS S3
```bash
aws s3 sync wiki/ s3://your-bucket-name/ --acl public-read
```

## Customization

### Change Colors
Edit `wiki/assets/css/style.css`:
```css
:root {
    --primary-color: #4a5568;
    --accent-color: #4299e1;
    /* Modify these variables */
}
```

### Modify Search Settings
Edit `wiki/assets/js/search.js`:
```javascript
fuse = new Fuse(searchIndex, {
    threshold: 0.3,     // Lower = stricter matching
    minMatchCharLength: 2,
    includeScore: true
});
```

### Add Analytics
Add tracking code to HTML templates in `wiki_export.py` before regenerating

## Troubleshooting

### Search Not Working
- **Cause:** Browser restricting local file access
- **Solution:** Use local web server (Option 2 above)

### Large File Size
- **Cause:** All documents included in JSON
- **Solution:** Normal - enables offline search

### Slow Loading
- **Cause:** Large JSON files (68 MB each)
- **Solution:** Use web server with compression enabled

### Missing Documents
- **Cause:** Export failed or incomplete
- **Solution:** Re-run export with verbose logging

## Advanced Usage

### Incremental Updates
```bash
# Export only updated content
python wiki_export.py --output wiki --incremental
```

### Filtered Export
```bash
# Export specific tags (modify script)
python wiki_export.py --tags "hardware,programming"
```

### Custom Templates
Modify the HTML generation methods in `wiki_export.py`:
- `_generate_index_html()` - Main page
- `_generate_doc_html()` - Document pages
- `_generate_entities_html()` - Entity browser
- `_generate_topics_html()` - Topics page
- `_generate_timeline_html()` - Timeline

## Integration

### With Documentation Sites
Embed wiki as subdirectory in existing documentation

### With Emulators
Bundle with C64 emulators for context-sensitive help

### With Development Tools
Integrate with IDEs for inline documentation

## Maintenance

### Regular Exports
```bash
# Create dated archives
python wiki_export.py --output wiki-$(date +%Y%m%d)
```

### Automated Deployment
```bash
#!/bin/bash
# Export and deploy
python wiki_export.py --output wiki
cd wiki
git add .
git commit -m "Update wiki $(date)"
git push
```

## Resources

- **Wiki README:** `wiki/README.md` - User guide for exported wiki
- **Source Code:** `wiki_export.py` - Export script
- **Examples:** See `wiki/` directory after export

## Support

For issues or feature requests:
- GitHub: https://github.com/MichaelTroelsen/tdz-c64-knowledge
- Issues: https://github.com/MichaelTroelsen/tdz-c64-knowledge/issues

---

**TDZ C64 Knowledge Base v2.23.15**
Built with [Claude Code](https://claude.com/claude-code)
