# Web Scraping Guide - Recursive Sub-Page Scraping
**TDZ C64 Knowledge Base v2.17.0+**

## 🎯 Overview

The web scraping feature now supports **recursive scraping of entire websites** by following links to sub-pages. This is perfect for scraping documentation sites like:
- http://www.sidmusic.org/sid/ (SID music resources)
- http://unusedino.de/ec64/ (C64 documentation)
- http://www.zimmers.net/anonftp/pub/cbm/ (Commodore archives)

---

## 🚀 Quick Start

### Example: Scrape sidmusic.org

#### Using MCP Tool (Recommended)
```json
{
  "tool": "scrape_url",
  "arguments": {
    "url": "http://www.sidmusic.org/sid/",
    "follow_links": true,
    "same_domain_only": true,
    "max_pages": 50,
    "depth": 3
  }
}
```

#### Using Streamlit GUI
1. Navigate to "📄 Add Documents" page
2. Expand "➕ Add New URL to Scrape"
3. Enter URL: `http://www.sidmusic.org/sid/`
4. Configure options:
   - ✅ **Follow Links** (checked)
   - ✅ **Same Domain Only** (checked)
   - **Max Pages**: 50
   - **Max Depth**: 3
5. Click "🚀 Start Scraping"

---

## 📋 Parameter Reference

### Simple Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **follow_links** | boolean | `true` | Follow links to scrape sub-pages. Set to `false` to scrape only the single page. |
| **same_domain_only** | boolean | `true` | Only follow links on the same domain. Prevents scraping external sites. |
| **max_pages** | integer | `50` | Maximum number of pages to scrape. Range: 1-500. |

### Advanced Controls

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **depth** | integer | `3` | How many link levels to follow. `1`=single page, `2`=linked pages, `3`=two levels deep. Range: 1-10. |
| **limit** | string | (auto) | Advanced: Only scrape URLs with this prefix. Overrides `same_domain_only`. |
| **threads** | integer | `10` | Number of concurrent download threads. Range: 1-20. |
| **delay** | integer | `100` | Delay between requests in milliseconds. Range: 0-5000. |
| **selector** | string | (auto) | CSS selector for main content (e.g., `article.main-content`). Auto-detected if omitted. |

---

## 💡 Common Use Cases

### 1. Scrape Entire Site (Stay on Domain)
**Goal:** Scrape all pages on sidmusic.org/sid/

```python
kb.scrape_url(
    "http://www.sidmusic.org/sid/",
    follow_links=True,
    same_domain_only=True,
    max_pages=50,
    depth=3
)
```

**What happens:**
- Starts at http://www.sidmusic.org/sid/
- Follows all links starting with http://www.sidmusic.org/sid/
- Scrapes up to 50 pages
- Stops after 3 levels of link depth

---

### 2. Scrape Specific Section Only
**Goal:** Only scrape the `/sid/technical/` section

```python
kb.scrape_url(
    "http://www.sidmusic.org/sid/technical/",
    follow_links=True,
    same_domain_only=True,
    max_pages=20,
    depth=2
)
```

**What happens:**
- Starts at http://www.sidmusic.org/sid/technical/
- Automatically limits to URLs starting with that prefix
- Only scrapes pages within `/sid/technical/` directory
- Max 20 pages, 2 levels deep

---

### 3. Scrape Single Page Only
**Goal:** Just scrape one page, don't follow any links

```python
kb.scrape_url(
    "http://www.sidmusic.org/sid/overview.html",
    follow_links=False
)
```

**What happens:**
- Scrapes only the single page
- Ignores all links
- Depth automatically set to 1

---

### 4. Scrape with No Domain Restrictions
**Goal:** Follow links even to external sites (use carefully!)

```python
kb.scrape_url(
    "http://example.com/c64-links.html",
    follow_links=True,
    same_domain_only=False,
    max_pages=10,
    depth=2
)
```

**Warning:** This can scrape external sites! Use a low `max_pages` limit.

---

### 5. Manual URL Limit (Advanced)
**Goal:** Scrape only specific URL patterns

```python
kb.scrape_url(
    "http://www.sidmusic.org/sid/",
    follow_links=True,
    limit="http://www.sidmusic.org/sid/technical",
    max_pages=30
)
```

**What happens:**
- Only scrapes URLs starting with the exact prefix
- More precise than `same_domain_only`
- Overrides `same_domain_only` setting

---

## 🎓 How It Works

### URL Filtering Logic

```
1. Start at: http://www.sidmusic.org/sid/index.html
                              ↓
2. Extract all links on page:
   - /sid/technical.html  ✅ (same domain)
   - /sid/links.html      ✅ (same domain)
   - http://example.com/  ❌ (external, same_domain_only=true)
                              ↓
3. Follow allowed links (up to max_pages):
   - http://www.sidmusic.org/sid/technical.html
   - http://www.sidmusic.org/sid/links.html
                              ↓
4. Repeat for each page (up to depth levels)
```

### Depth Levels Explained

```
Depth=1: Only scrape the starting page
┌─────────────┐
│ index.html  │ ← Start (scraped)
└─────────────┘

Depth=2: Scrape starting page + directly linked pages
┌─────────────┐
│ index.html  │ ← Start (scraped)
└──────┬──────┘
       ├─→ page1.html ← Linked (scraped)
       ├─→ page2.html ← Linked (scraped)
       └─→ page3.html ← Linked (scraped)

Depth=3: Scrape starting page + linked pages + their links
┌─────────────┐
│ index.html  │ ← Start (scraped)
└──────┬──────┘
       ├─→ page1.html ← Level 1 (scraped)
       │   ├─→ subpage1.html ← Level 2 (scraped)
       │   └─→ subpage2.html ← Level 2 (scraped)
       └─→ page2.html ← Level 1 (scraped)
           └─→ subpage3.html ← Level 2 (scraped)
```

---

## ⚙️ Best Practices

### 1. Start Small, Scale Up
```python
# First attempt: Small scope
kb.scrape_url(url, max_pages=10, depth=2)

# If successful: Increase limits
kb.scrape_url(url, max_pages=50, depth=3)
```

### 2. Be Respectful to Servers
```python
# Good: Reasonable delay for large scrapes
kb.scrape_url(url, delay=200, threads=5)

# Bad: Hammering server (might get blocked)
kb.scrape_url(url, delay=0, threads=20)  # Don't do this!
```

### 3. Use Appropriate Depth
```
- depth=1: Single page (documentation page)
- depth=2: Main page + immediate links (small docs site)
- depth=3: Main + links + sublinks (medium docs site)
- depth=4+: Large, multi-level sites (use carefully)
```

### 4. Monitor Progress
- Watch the Streamlit progress indicator
- Check logs for scraped URLs
- Verify `files_scraped` count in results

---

## 🔍 Troubleshooting

### Problem: Too Many Pages Scraped
**Solution:** Reduce `max_pages` or `depth`
```python
kb.scrape_url(url, max_pages=20, depth=2)  # More conservative
```

### Problem: Not Scraping Sub-Pages
**Check:**
1. Is `follow_links=True`?
2. Is `depth > 1`?
3. Are links on the same domain (if `same_domain_only=True`)?

### Problem: Scraping External Sites Unintentionally
**Solution:** Enable `same_domain_only`
```python
kb.scrape_url(url, same_domain_only=True)  # Stay on domain
```

### Problem: Missing Content
**Solution:** Try a CSS selector for main content
```python
kb.scrape_url(url, selector="article.content")
```

### Problem: HTML Frames (Frameset Pages)
**Automatic Detection:** Frame-based websites are automatically detected and handled!

**What are frames?**
- Obsolete HTML technology from the 1990s
- Uses `<frameset>` and `<frame>` tags instead of regular links
- Examples: sidmusic.org, some old documentation sites

**How it works:**
```
1. Detects <frameset> page → Extracts frame sources
2. Scrapes each frame individually with link following
3. Combines all results into single response
```

**Example:**
```python
# This automatically detects and handles frames!
result = kb.scrape_url("http://www.sidmusic.org/sid/")
# ✅ Detects 2 frames (main.html + menu.html)
# ✅ Scrapes both frames and their sub-pages
# ✅ Returns 18+ documents

print(result['frames_detected'])  # 2
print(result['files_scraped'])    # 18
```

**No special configuration needed** - frame detection is automatic!

---

## 📊 Example Results

### Scraping sidmusic.org/sid/

**Input:**
```python
result = kb.scrape_url(
    "http://www.sidmusic.org/sid/",
    follow_links=True,
    same_domain_only=True,
    max_pages=50,
    depth=3
)
```

**Expected Output:**
```python
{
    'status': 'success',
    'url': 'http://www.sidmusic.org/sid/',
    'frames_detected': 2,          # ✨ Automatically detected frames!
    'files_scraped': 18,
    'docs_added': 18,
    'docs_updated': 0,
    'docs_failed': 0,
    'doc_ids': [
        '87d8c0535be1',  # main.html
        '44397d40fabb',  # rhubbard.html (Rob Hubbard bio)
        '400639f0011e',  # sidcomp.html (composers)
        '1699a4bf6738',  # sidlinks.html (links)
        '1df0eddccf56',  # sidplay.html (SID player)
        '052b43e312d7',  # sidproj.html (projects)
        'ead4aa10e9b5',  # sidtech.html (technical docs)
        '1d50f177a191',  # sidtunes.html (SID tunes)
        '21fd50a0e9cd',  # top100.html (charts)
        # ... more doc IDs
    ],
    'message': 'Scraped 2 frames with 18 total documents'
}
```

**What You Get:**
- 18 new documents in knowledge base
- **Automatic frame detection and handling** (2 frames: main.html + menu.html)
- Pages include: technical docs, composers, SID player, projects, charts, links
- All pages tagged with `sidmusic.org` and `scraped`
- Full-text search across all scraped content
- Entity extraction queued for each document

---

## 🎯 Real-World Example: Complete Workflow

### Scenario: Build SID Music Knowledge Base

**Step 1: Scrape Main Site**
```python
kb.scrape_url(
    "http://www.sidmusic.org/sid/",
    title="SID Music Documentation",
    tags=["sid", "music"],
    follow_links=True,
    same_domain_only=True,
    max_pages=30,
    depth=3
)
```

**Step 2: Search the Scraped Content**
```python
results = kb.search("SID chip filters", max_results=5)
```

**Step 3: Extract Entities**
```python
kb.extract_entities_bulk(confidence_threshold=0.7)
```

**Step 4: Ask Questions (with RAG)**
```python
answer = kb.answer_question("How do I program the SID filter?")
```

---

## 🔄 Updating Scraped Content

### Check for Updates
```python
updates = kb.check_url_updates()
# Returns list of scraped documents that have changed
```

### Re-scrape Updated Documents
```python
for doc in updates['modified']:
    kb.rescrape_document(doc['doc_id'])
```

---

## 📝 Notes

### Automatic Features
- Domain name automatically added as tag
- URLs logged for update tracking
- Duplicate content detection
- Content-based deduplication (MD5 hashing)

### Limitations
- Requires `mdscrape` executable (see MDSCRAPE_PATH in docs)
- JavaScript-heavy sites may not scrape well
- Some sites block automated scraping (robots.txt)

### Performance
- Concurrent downloads (default: 10 threads)
- Typical speed: 2-5 pages per second
- Large sites (100+ pages) may take 5-10 minutes

---

## 🆘 Support

### Common Issues
1. **"mdscrape executable not found"**
   - Set `MDSCRAPE_PATH` environment variable
   - Download from: https://github.com/MichaelTroelsen/mdscrape

2. **"Scraping timeout"**
   - Reduce `max_pages` or `depth`
   - Increase `delay` between requests

3. **"No files scraped"**
   - Check if URL is accessible
   - Try with `selector` for main content
   - Verify site allows scraping (check robots.txt)

---

**Happy Scraping!** 🌐✨
