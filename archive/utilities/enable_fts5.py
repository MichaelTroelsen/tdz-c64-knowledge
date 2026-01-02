#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enable and test FTS5 full-text search."""

import os
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("FTS5 Full-Text Search Setup")
print("=" * 70)

# Step 1: Check if FTS5 is available in SQLite
print("\n1. Checking SQLite FTS5 Support...")
import sqlite3

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
try:
    cursor.execute("CREATE VIRTUAL TABLE test USING fts5(content)")
    print("   ✓ FTS5 is available in SQLite")
    cursor.execute("DROP TABLE test")
    fts5_available = True
except sqlite3.OperationalError as e:
    print(f"   ✗ FTS5 not available: {e}")
    fts5_available = False
finally:
    conn.close()

if not fts5_available:
    print("\n✗ Cannot enable FTS5 - not supported by your SQLite version")
    sys.exit(1)

# Step 2: Enable FTS5 environment variable
print("\n2. Enabling FTS5...")
os.environ['USE_FTS5'] = '1'
os.environ['USE_OCR'] = '1'
poppler_path = Path(__file__).parent / "poppler-25.12.0" / "Library" / "bin"
os.environ['POPPLER_PATH'] = str(poppler_path)

print("   ✓ Environment variables set:")
print(f"     USE_FTS5 = {os.environ['USE_FTS5']}")

# Import after setting environment
from server import KnowledgeBase

# Step 3: Initialize knowledge base with FTS5
print("\n3. Initializing Knowledge Base with FTS5...")
data_dir = os.path.expanduser("~/.tdz-c64-knowledge")
kb = KnowledgeBase(data_dir)
print(f"   ✓ Knowledge Base initialized")

# Step 4: Check if FTS5 table exists
print("\n4. Checking FTS5 Table Status...")
cursor = kb.db_conn.cursor()
cursor.execute("""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='chunks_fts'
""")
fts_table = cursor.fetchone()

if fts_table:
    print("   ✓ FTS5 table 'chunks_fts' exists")

    # Check row count
    cursor.execute("SELECT COUNT(*) FROM chunks_fts")
    fts_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunks_count = cursor.fetchone()[0]

    print(f"   FTS5 index: {fts_count:,} entries")
    print(f"   Chunks table: {chunks_count:,} entries")

    if fts_count == chunks_count:
        print("   ✓ FTS5 index is complete and synchronized")
    else:
        print("   ⚠ FTS5 index needs rebuilding")
else:
    print("   ⚠ FTS5 table does not exist - will be created on first search")

# Step 5: Performance test
print("\n5. Performance Comparison Test...")
print("-" * 70)

test_queries = [
    "SID chip sound synthesis",
    "VIC-II sprite registers",
    "6502 assembly instructions",
    "BASIC programming commands",
]

print("\nTesting with FTS5 ENABLED (USE_FTS5=1):")
print("-" * 70)

fts5_times = []
for query in test_queries:
    start = time.time()
    results = kb.search(query, max_results=5)
    elapsed = time.time() - start
    fts5_times.append(elapsed)

    print(f"Query: '{query}'")
    print(f"  Time: {elapsed*1000:.1f}ms")
    print(f"  Results: {len(results)}")
    if results:
        # Handle different result formats
        if isinstance(results[0], dict) and 'title' in results[0]:
            print(f"  Top result: {results[0]['title'][:50]}...")
        elif isinstance(results[0], dict) and 'document' in results[0]:
            print(f"  Top result: {results[0]['document']['title'][:50]}...")
    print()

avg_fts5 = sum(fts5_times) / len(fts5_times)
print(f"Average FTS5 search time: {avg_fts5*1000:.1f}ms")

# Step 6: Test with FTS5 disabled for comparison
print("\n" + "-" * 70)
print("Testing with FTS5 DISABLED (for comparison):")
print("-" * 70)

# Temporarily disable FTS5
kb.use_fts5 = False

bm25_times = []
for query in test_queries:
    start = time.time()
    results = kb.search(query, max_results=5)
    elapsed = time.time() - start
    bm25_times.append(elapsed)

    print(f"Query: '{query}'")
    print(f"  Time: {elapsed*1000:.1f}ms (BM25 fallback)")
    print(f"  Results: {len(results)}")
    print()

avg_bm25 = sum(bm25_times) / len(bm25_times)
print(f"Average BM25 search time: {avg_bm25*1000:.1f}ms")

# Step 7: Summary
print("\n" + "=" * 70)
print("Performance Summary")
print("=" * 70)
print(f"FTS5 average:  {avg_fts5*1000:>8.1f}ms")
print(f"BM25 average:  {avg_bm25*1000:>8.1f}ms")
print(f"Speed improvement: {avg_bm25/avg_fts5:.1f}x faster with FTS5")
print()

if avg_fts5 < avg_bm25:
    print("✓ FTS5 is working and providing faster search!")
else:
    print("⚠ FTS5 may not be providing expected performance improvement")

# Close KB
kb.close()

print("\n" + "=" * 70)
print("✓ FTS5 Setup Complete!")
print("=" * 70)
print("\nTo use FTS5 permanently, set this environment variable:")
print("  USE_FTS5=1")
print("\nOr add to your MCP configuration:")
print('  "env": {')
print('    "USE_FTS5": "1",')
print('    "USE_OCR": "1",')
print(f'    "POPPLER_PATH": "{poppler_path}"')
print('  }')
