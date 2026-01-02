# Web Monitoring Guide

Complete guide to monitoring scraped websites for updates, new pages, and missing content.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Check Modes](#check-modes)
- [Use Cases](#use-cases)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Scheduled Monitoring](#scheduled-monitoring)
- [Understanding Results](#understanding-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The TDZ C64 Knowledge Base provides comprehensive URL monitoring to track changes in scraped documentation websites. This helps you:

- **Detect content updates**: Know when pages have been modified
- **Discover new pages**: Find new documentation added to websites
- **Identify missing content**: Detect removed or relocated pages
- **Maintain freshness**: Keep your knowledge base synchronized with source sites

**Key Features:**
- Two check modes: Quick (fast) and Full (comprehensive)
- Scrape session grouping for organized monitoring
- Automatic structure discovery with website crawling
- Configurable depth and page limits
- Detailed progress logging

## Quick Start

### Quick Check (Fast)
Check if any content has been modified:

```python
from server import KnowledgeBase
import os

kb = KnowledgeBase(os.path.expanduser("~/.tdz-c64-knowledge"))
results = kb.check_url_updates(check_structure=False)

print(f"Unchanged: {len(results['unchanged'])}")
print(f"Changed: {len(results['changed'])}")
```

**Speed**: ~1 second per site
**Method**: Last-Modified HTTP headers

### Full Check (Comprehensive)
Discover new and missing pages:

```python
results = kb.check_url_updates(check_structure=True)

print(f"New pages: {len(results['new_pages'])}")
print(f"Missing pages: {len(results['missing_pages'])}")
```

**Speed**: ~10-60 seconds per site
**Method**: Website crawling + HTTP headers

## Check Modes

### Quick Mode (`check_structure=False`)

**What it does:**
- Sends HEAD requests to all scraped URLs
- Compares Last-Modified headers with scrape dates
- Detects 404 errors for removed pages

**Advantages:**
- Fast: ~1 second per site
- Low bandwidth usage
- No server load

**Limitations:**
- Doesn't discover new pages
- Relies on Last-Modified headers (not all sites provide them)
- Won't detect new sub-pages added to the site

**When to use:**
- Daily monitoring of stable documentation
- Bandwidth-constrained environments
- Sites with reliable Last-Modified headers

### Full Mode (`check_structure=True`)

**What it does:**
- Crawls websites using BeautifulSoup
- Discovers all current pages (up to depth 5, max 100 pages)
- Compares discovered URLs with database
- Detects new pages, missing pages, and modifications

**Advantages:**
- Discovers new documentation pages
- Identifies removed or relocated content
- Provides complete site structure analysis
- Groups findings by scrape session

**Limitations:**
- Slower: ~10-60 seconds per site
- Higher bandwidth usage
- May trigger rate limiting on some servers

**When to use:**
- Weekly/monthly comprehensive checks
- After major site updates
- When documentation structure changes frequently
- Initial validation of scrape completeness

## Use Cases

### Use Case 1: Daily Content Monitoring

Check if documentation has been updated overnight:

```python
#!/usr/bin/env python3
"""daily_check.py - Quick daily monitoring"""

from server import KnowledgeBase
import os
from datetime import datetime

kb = KnowledgeBase(os.path.expanduser("~/.tdz-c64-knowledge"))

print(f"Daily check: {datetime.now()}")
results = kb.check_url_updates(
    check_structure=False,  # Quick mode
    auto_rescrape=False     # Manual review first
)

if results['changed']:
    print(f"\nWARNING: {len(results['changed'])} pages updated!")
    for page in results['changed']:
        print(f"  - {page['title']}: {page['url']}")
        print(f"    Last modified: {page['last_modified']}")
```

**Schedule**: Run daily via cron/Task Scheduler

### Use Case 2: Weekly Structure Discovery

Find new pages added to documentation sites:

```python
#!/usr/bin/env python3
"""weekly_discovery.py - Find new pages"""

from server import KnowledgeBase
import os

kb = KnowledgeBase(os.path.expanduser("~/.tdz-c64-knowledge"))

print("Weekly structure check...")
results = kb.check_url_updates(
    check_structure=True,   # Full mode
    auto_rescrape=False
)

if results['new_pages']:
    print(f"\nNEW PAGES FOUND: {len(results['new_pages'])}")

    # Group by site
    by_site = {}
    for page in results['new_pages']:
        site = page['base_url']
        if site not in by_site:
            by_site[site] = []
        by_site[site].append(page['url'])

    # Display grouped results
    for site, urls in by_site.items():
        print(f"\n{site}:")
        print(f"  {len(urls)} new pages:")
        for url in urls[:5]:
            print(f"    - {url}")
        if len(urls) > 5:
            print(f"    ... and {len(urls) - 5} more")

if results['missing_pages']:
    print(f"\nMISSING PAGES: {len(results['missing_pages'])}")
    for page in results['missing_pages'][:5]:
        print(f"  - {page['title']}: {page['url']}")
```

**Schedule**: Run weekly on Sunday night

### Use Case 3: Auto-Rescrape Changed Content

Automatically update changed pages:

```python
results = kb.check_url_updates(
    check_structure=False,
    auto_rescrape=True  # Automatically re-scrape
)

print(f"Auto-rescraped {len(results['rescraped'])} documents")
```

**When to use**: Trusted documentation sources that rarely change structure

## CLI Usage

### Quick Command-Line Check

```bash
# Quick check
.venv/Scripts/python.exe -c "
from server import KnowledgeBase
import os
kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))
r = kb.check_url_updates(check_structure=False)
print(f'Changed: {len(r[\"changed\"])}, Failed: {len(r[\"failed\"])}')
"

# Full check
.venv/Scripts/python.exe run_quick_url_check.py
```

### Using Provided Scripts

**Quick Check** (included):
```bash
.venv/Scripts/python.exe run_quick_url_check.py
```

**Full Check** (create as needed):
```bash
.venv/Scripts/python.exe run_full_url_check.py
```

## Python API

### Method Signature

```python
def check_url_updates(
    self,
    auto_rescrape: bool = False,
    check_structure: bool = True
) -> dict:
    """Check all URL-sourced documents for updates.

    Args:
        auto_rescrape: If True, automatically re-scrape changed URLs
        check_structure: If True, check for new/missing sub-pages

    Returns:
        Dictionary with update information
    """
```

### Return Structure

```python
{
    # Pages that haven't changed
    'unchanged': [
        {'doc_id': 'abc', 'title': 'Page 1', 'url': 'https://...'}
    ],

    # Pages with updates available
    'changed': [
        {
            'doc_id': 'def',
            'title': 'Updated Page',
            'url': 'https://...',
            'last_modified': '2025-12-22T10:30:00Z',
            'scraped_date': '2025-12-15T08:00:00Z',
            'reason': 'content_modified'
        }
    ],

    # URLs discovered but not in database
    'new_pages': [
        {
            'url': 'https://example.com/new-doc.html',
            'base_url': 'https://example.com',
            'scrape_config': {...}  # Original scrape parameters
        }
    ],

    # Pages in database but not discoverable
    'missing_pages': [
        {
            'doc_id': 'xyz',
            'title': 'Removed Page',
            'url': 'https://...',
            'reason': '404'  # or 'not_discovered'
        }
    ],

    # URLs that failed to check
    'failed': [
        {
            'doc_id': 'mno',
            'title': 'Timeout Page',
            'url': 'https://...',
            'error': 'Connection timeout'
        }
    ],

    # Documents that were automatically re-scraped
    'rescraped': ['doc_id1', 'doc_id2'],

    # Statistics per scrape session
    'scrape_sessions': [
        {
            'base_url': 'https://example.com',
            'docs_count': 15,
            'unchanged': 13,
            'changed': 1,
            'new': 2,
            'missing': 1
        }
    ]
}
```

## Scheduled Monitoring

### Windows Task Scheduler

Create a batch file `monitor_urls.bat`:

```bat
@echo off
cd /d C:\path\to\tdz-c64-knowledge
.venv\Scripts\python.exe run_quick_url_check.py >> logs\url_check.log 2>&1
```

Schedule in Task Scheduler:
- **Trigger**: Daily at 2:00 AM
- **Action**: Run `monitor_urls.bat`
- **Conditions**: Only if network available

### Linux Cron

Add to crontab:

```bash
# Daily quick check at 2 AM
0 2 * * * cd /path/to/tdz-c64-knowledge && .venv/bin/python run_quick_url_check.py >> logs/url_check.log 2>&1

# Weekly full check on Sundays at 3 AM
0 3 * * 0 cd /path/to/tdz-c64-knowledge && .venv/bin/python run_full_url_check.py >> logs/url_full_check.log 2>&1
```

### Python Scheduler

Using `schedule` library:

```python
import schedule
import time
from server import KnowledgeBase
import os

kb = KnowledgeBase(os.path.expanduser("~/.tdz-c64-knowledge"))

def daily_quick_check():
    results = kb.check_url_updates(check_structure=False)
    if results['changed']:
        print(f"ALERT: {len(results['changed'])} pages updated!")
        # Send notification email/Slack/etc.

def weekly_full_check():
    results = kb.check_url_updates(check_structure=True)
    if results['new_pages'] or results['missing_pages']:
        print(f"New: {len(results['new_pages'])}, Missing: {len(results['missing_pages'])}")

# Schedule jobs
schedule.every().day.at("02:00").do(daily_quick_check)
schedule.every().sunday.at("03:00").do(weekly_full_check)

# Run forever
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Understanding Results

### Unchanged Pages
Pages where:
- HTTP status is 200 (OK)
- Last-Modified header is older than scrape date
- Content hash matches (if available)

**Action**: None required

### Changed Pages
Pages where:
- Last-Modified header is newer than scrape date
- Content appears to have been updated

**Actions**:
- Review changes manually
- Re-scrape specific pages: `kb.rescrape_document(doc_id)`
- Enable auto-rescrape for automatic updates

### New Pages
URLs discovered during crawling that aren't in the database.

**Common causes**:
- New documentation added since last scrape
- Previously undiscovered pages (deeper than original scrape depth)
- Restructured site with new URLs

**Actions**:
- Review new pages to determine relevance
- Scrape individually: `kb.scrape_url(new_url)`
- Re-scrape entire site with increased depth

### Missing Pages
Pages in database but not discoverable during crawl.

**Common causes**:
- Page removed or deleted (404)
- Page moved to new URL (301/302)
- Changed site structure (links broken)
- Pages beyond crawl depth/limit

**Actions**:
- Check if page moved (manual visit)
- Remove from database if permanently deleted
- Update scrape configuration if structure changed

### Failed Checks
URLs that couldn't be checked due to errors.

**Common causes**:
- Network timeout
- DNS failure
- SSL certificate issues
- Server temporarily down

**Actions**:
- Retry later
- Check server status
- Verify URL is still valid

## Best Practices

### 1. Start with Quick Checks
Run quick checks frequently (daily) to catch content updates early.

### 2. Schedule Full Checks Periodically
Run comprehensive structure discovery weekly or monthly.

### 3. Monitor Failure Patterns
If specific sites consistently fail, investigate:
- Are they blocking automated requests?
- Do they have rate limiting?
- Is the URL still valid?

### 4. Group by Importance
Prioritize monitoring for:
- Official documentation (highest priority)
- Community wikis (medium priority)
- Personal blogs (lower priority)

### 5. Use Auto-Rescrape Carefully
Only enable `auto_rescrape` for:
- Trusted, stable documentation
- Sites with reliable Last-Modified headers
- Low-traffic periods to avoid rate limiting

### 6. Review New Pages Before Scraping
Not all discovered pages may be relevant. Review before bulk-scraping.

### 7. Archive Missing Pages
Before deleting missing pages from database, consider:
- Do you have the only copy?
- Is the content valuable?
- Archive to separate location if important

## Troubleshooting

### Problem: Too Many New Pages Discovered

**Cause**: Crawl depth may be discovering unrelated pages

**Solution**:
```python
# Limit discovery to specific URL prefix
results = kb.check_url_updates(check_structure=True)
# Filter new_pages by URL pattern before scraping
relevant_pages = [
    p for p in results['new_pages']
    if '/docs/' in p['url']  # Only documentation pages
]
```

### Problem: Slow Full Checks

**Cause**: Many scrape sessions or large sites

**Solutions**:
- Run during off-peak hours
- Reduce max_pages limit in `_discover_urls()`
- Split monitoring across multiple scripts
- Check only specific sites instead of all

### Problem: False Missing Pages

**Cause**: Pages exist but not discoverable within depth/limit

**Solution**:
```python
# Check page manually
import requests
response = requests.head(missing_url)
if response.status_code == 200:
    print("Page exists but not discoverable - check scrape config")
```

### Problem: Rate Limiting

**Symptoms**: 429 errors, timeouts, or IP blocks

**Solutions**:
- Increase delay between requests
- Reduce concurrent threads
- Respect robots.txt
- Contact site owner if necessary
- Use VPN or proxy rotation

### Problem: No Changes Detected

**Cause**: Site doesn't provide Last-Modified headers

**Solution**:
- Use full structure discovery mode
- Compare content hashes (requires custom implementation)
- Check manually for visible updates

## Advanced Usage

### Custom Filtering

Filter results by domain:

```python
results = kb.check_url_updates(check_structure=True)

# Filter new pages by domain
c64_wiki_pages = [
    p for p in results['new_pages']
    if 'c64-wiki.com' in p['url']
]
```

### Notification Integration

Send alerts when changes detected:

```python
def send_notification(results):
    if results['changed'] or results['new_pages']:
        # Email
        import smtplib
        # ... send email

        # Slack
        import requests
        requests.post(webhook_url, json={
            'text': f"Changes detected: {len(results['changed'])} updated, {len(results['new_pages'])} new"
        })

results = kb.check_url_updates()
send_notification(results)
```

### Export Results

Save monitoring results for historical analysis:

```python
import json
from datetime import datetime

results = kb.check_url_updates(check_structure=True)

# Save with timestamp
filename = f"url_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

---

## See Also

- [WEB_SCRAPING_GUIDE.md](WEB_SCRAPING_GUIDE.md) - How to scrape websites
- [README.md](README.md) - Complete feature documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical implementation details

## Support

For issues or questions:
- GitHub Issues: https://github.com/MichaelTroelsen/tdz-c64-knowledge/issues
- Documentation: See README.md and other guides
