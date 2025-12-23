# Usage Examples - TDZ C64 Knowledge Base v2.21.0

Practical examples for using features in v2.21.0 and earlier versions.

## Hybrid Search Examples

### Example 1: Balanced Search (Default)
```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")

# Default: 70% FTS5 keyword, 30% semantic
results = kb.hybrid_search("SID sound programming", max_results=5)

for r in results:
    print(f"Title: {r['title']}")
    print(f"Hybrid Score: {r['score']:.3f}")
    print(f"  ↳ FTS: {r['fts_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
    print(f"Snippet: {r['snippet'][:100]}...\n")

kb.close()
```

**Output:**
```
Title: Programming the SID Chip
Hybrid Score: 0.847
  ↳ FTS: 0.950, Semantic: 0.450
Snippet: The **SID** chip (6581/8580) provides three-voice **sound** synthesis...

Title: Advanced Audio Techniques
Hybrid Score: 0.723
  ↳ FTS: 0.380, Semantic: 0.920
Snippet: This chapter covers **sound** synthesis and music **programming**...
```

### Example 2: Keyword-Focused Search
```python
# 90% keyword precision, 10% semantic
# Use for technical terms, register addresses
results = kb.hybrid_search("$D400 SID register",
                          max_results=5,
                          semantic_weight=0.1)
```

**Best for:** Technical documentation, exact register addresses, specific commands

### Example 3: Concept-Focused Search
```python
# 30% keyword, 70% semantic
# Use for understanding concepts, finding related content
results = kb.hybrid_search("how to create moving graphics",
                          max_results=5,
                          semantic_weight=0.7)
```

**Best for:** Learning, tutorials, conceptual understanding

### Example 4: With Tag Filtering
```python
# Search only in assembly programming documents
results = kb.hybrid_search("sprite multiplexing",
                          max_results=5,
                          tags=["assembly", "reference"],
                          semantic_weight=0.3)
```

## Health Check Examples

### Example 1: Basic Health Check
```python
from server import KnowledgeBase

kb = KnowledgeBase("~/.tdz-c64-knowledge")
health = kb.health_check()

print(f"Status: {health['status'].upper()}")
print(f"Message: {health['message']}\n")

# Metrics
print("Knowledge Base Metrics:")
print(f"  Documents: {health['metrics']['documents']:,}")
print(f"  Chunks: {health['metrics']['chunks']:,}")
print(f"  Total Words: {health['metrics']['total_words']:,}\n")

# Database
print("Database Health:")
print(f"  Integrity: {health['database']['integrity']}")
print(f"  Size: {health['database']['size_mb']} MB")
print(f"  Free Disk Space: {health['database']['disk_free_gb']} GB\n")

# Features
print("Search Features:")
for feature, status in health['features'].items():
    icon = "✓" if status else "✗"
    print(f"  {icon} {feature}: {status}")

# Issues
if health['issues']:
    print(f"\n⚠ Issues Detected ({len(health['issues'])}):")
    for issue in health['issues']:
        print(f"  - {issue}")
else:
    print("\n✓ No issues detected")

kb.close()
```

**Output:**
```
Status: HEALTHY
Message: All systems operational

Knowledge Base Metrics:
  Documents: 145
  Chunks: 4,665
  Total Words: 6,870,642

Database Health:
  Integrity: ok
  Size: 45.23 MB
  Free Disk Space: 125.5 GB

Search Features:
  ✓ fts5_enabled: True
  ✓ fts5_available: True
  ✓ semantic_search_enabled: True
  ✓ semantic_search_available: True
  ✓ bm25_enabled: True
  ✓ embeddings_count: 2347

✓ No issues detected
```

### Example 2: Automated Health Monitoring
```python
import schedule
import time
from server import KnowledgeBase

def check_system_health():
    kb = KnowledgeBase("~/.tdz-c64-knowledge")
    health = kb.health_check()

    if health['status'] != 'healthy':
        # Send alert
        print(f"⚠ ALERT: System status is {health['status']}")
        print(f"Issues: {', '.join(health['issues'])}")
        # Send email, Slack notification, etc.
    else:
        print("✓ System healthy")

    kb.close()

# Check health every hour
schedule.every().hour.do(check_system_health)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Enhanced Snippets Examples

The enhanced snippet extraction is automatic - you don't need to do anything different!

### Example: Comparing Old vs New Snippets

**Old Snippet Extraction (v1.0.0):**
```
"...chip controls all graphics and vi..."
```
❌ Cut mid-word ("vi" instead of "video")
❌ No sentence boundaries

**New Snippet Extraction (v2.0.0):**
```
"The VIC-II chip controls all graphics and video output on the Commodore 64.
It has 47 registers mapped to memory locations $D000-$D02E."
```
✅ Complete sentences
✅ Natural boundaries
✅ More context

### Example: Code Preservation

**Search for assembly code:**
```python
results = kb.search("LDA #$00 STA $D020")
```

**Old snippet:**
```
"...background. To change it:
    LDA #$00
    STA $D..."
```
❌ Code block broken mid-instruction

**New snippet:**
```
"To change the border and background colors:
    LDA #$00      ; Load color
    STA $D020     ; Set border
    STA $D021     ; Set background"
```
✅ Complete code block preserved
✅ Comments included
✅ Proper context

## MCP Integration Examples

### Via Claude Desktop Chat

**Hybrid Search:**
```
User: "Use hybrid search to find information about sprite multiplexing
       with semantic weight 0.4"

Claude: [Calls hybrid_search tool with semantic_weight=0.4]
        Found 5 results for 'sprite multiplexing':

        Result 1: "Advanced Sprite Techniques" (hybrid=0.92)
        - FTS: 0.95, Semantic: 0.85
        ...
```

**Health Check:**
```
User: "Check the health of the C64 knowledge base"

Claude: [Calls health_check tool]
        System Health Check
        ==================================================

        Status: HEALTHY
        Message: All systems operational

        Metrics:
          documents: 145
          chunks: 4,665
          total_words: 6,870,642
        ...
```

## Python API Complete Example

```python
#!/usr/bin/env python3
"""
Complete example showing all v2.0.0 features
"""
from server import KnowledgeBase
import os

# Enable all features
os.environ['USE_FTS5'] = '1'
os.environ['USE_SEMANTIC_SEARCH'] = '1'

# Initialize
kb = KnowledgeBase("~/.tdz-c64-knowledge")

# 1. Health check first
print("=== System Health ===")
health = kb.health_check()
print(f"Status: {health['status']}")
print(f"Documents: {health['metrics']['documents']}")
print()

# 2. Hybrid search (best results)
print("=== Hybrid Search ===")
results = kb.hybrid_search("SID sound programming", max_results=3)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']} (score={r['score']:.3f})")
print()

# 3. Keyword-focused search
print("=== Keyword-Focused (semantic_weight=0.1) ===")
results = kb.hybrid_search("$D400 register", max_results=3, semantic_weight=0.1)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']} (FTS={r['fts_score']:.3f})")
print()

# 4. Concept-focused search
print("=== Concept-Focused (semantic_weight=0.7) ===")
results = kb.hybrid_search("creating music", max_results=3, semantic_weight=0.7)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['title']} (Semantic={r['semantic_score']:.3f})")
print()

# Cleanup
kb.close()
print("Done!")
```

## Tips and Best Practices

### When to Use Each Search Mode

| Use Case | Search Mode | semantic_weight |
|----------|-------------|-----------------|
| General search | Hybrid | 0.3 (default) |
| Technical docs | Hybrid | 0.1 (keyword-focused) |
| Learning/concepts | Hybrid | 0.6-0.7 (semantic-focused) |
| Exact register/address | FTS5 only | N/A |
| "How do I..." questions | Semantic only | N/A |

### Performance Tuning

```python
# Maximum precision (slower but comprehensive)
results = kb.hybrid_search(query, max_results=20, semantic_weight=0.5)

# Fast keyword search
results = kb.search(query, max_results=5)  # FTS5: 50-140ms

# Fast semantic search
results = kb.semantic_search(query, max_results=5)  # 12-25ms

# Balanced hybrid (recommended)
results = kb.hybrid_search(query, max_results=5)  # 60-180ms
```

### Health Monitoring Best Practices

```python
# Check before heavy operations
health = kb.health_check()
if health['database']['disk_free_gb'] < 1:
    print("Warning: Low disk space!")
    # Take action

# Verify features are available
if not health['features']['fts5_available']:
    print("FTS5 not available, falling back to BM25")

# Monitor embeddings
if health['features']['embeddings_count'] != expected_count:
    print("Embeddings may need rebuilding")
```

## v2.21.0 Features (New!)

### Health Check with Lazy-Loaded Embeddings

The health check now correctly detects embeddings files on disk, even when using lazy loading (default behavior).

```python
import os
from server import KnowledgeBase

# Enable semantic search
os.environ['USE_SEMANTIC_SEARCH'] = '1'

kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

# Run health check
health = kb.health_check()

print(f"Status: {health['status']}")
print(f"Semantic Available: {health['features']['semantic_search_available']}")

# Check embeddings info (works even if not loaded yet)
if health['features'].get('embeddings_size_mb'):
    print(f"Embeddings: {health['features']['embeddings_count']} vectors")
    print(f"Size: {health['features']['embeddings_size_mb']} MB")

kb.close()
```

**Output (with lazy-loaded embeddings):**
```
Status: healthy
Semantic Available: True
Embeddings: 2612 vectors
Size: 3.83 MB
```

### URL Scraping with WordPress Gallery Sites

Improved error handling for sites with image galleries (v2.21.1).

```python
from server import KnowledgeBase

kb = KnowledgeBase()

# Scrape WordPress site with galleries
result = kb.scrape_url(
    url="https://www.nightfallcrew.com/",
    follow_links=True,
    depth=2,
    max_pages=50
)

# Image errors are now warnings, not failures
if result['status'] == 'success':
    print(f"✓ Scraped {result['docs_added']} documents")
    print(f"  (Image gallery errors handled gracefully)")
elif result['status'] == 'partial':
    print(f"⚠ Partial success: {result['docs_added']} of {result['files_scraped']}")

kb.close()
```

### Admin GUI URL Monitoring

View and monitor scraped URL-sourced documents without errors (fixed in v2.21.1).

```python
# Launch the admin GUI
# Command: streamlit run admin_gui.py

# Navigate to "URL Monitoring" tab
# - View all scraped sites grouped by base URL
# - See document counts per site
# - Check for updates automatically
# - No more AttributeError crashes!
```

### Monitoring and Anomaly Detection

The v2.21.0 anomaly detection system provides intelligent change detection:

```python
from server import KnowledgeBase

kb = KnowledgeBase()

# Check for URL updates (with anomaly detection)
result = kb.check_url_updates()

print(f"Checked: {result['checked']} documents")
print(f"Updated: {result['updated']} documents")
print(f"Failed: {result['failed']} documents")

# Anomaly detection runs automatically
# - ML-based baseline learning
# - 1500x faster than previous implementation
# - Detects significant content changes
# - Reduces false positives

kb.close()
```

## See Also

- **CHANGELOG.md** - Detailed feature documentation
- **USER_GUIDE.md** - Complete user guide
- **CLAUDE.md** - Developer documentation
