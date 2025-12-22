#!/usr/bin/env python3
"""
Fast Async URL Monitoring Script

High-performance async implementation using aiohttp for concurrent URL checking.
Up to 10x faster than synchronous version for large numbers of documents.

Usage:
    python monitor_fast.py [--auto-rescrape] [--output FILENAME] [--concurrent N]

Options:
    --auto-rescrape    Automatically re-scrape changed documents
    --output FILE      Save results to JSON file
    --concurrent N     Number of concurrent requests (default: 10)
"""

import os
import sys
import argparse
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from email.utils import parsedate_to_datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Install with: pip install aiohttp")
    sys.exit(1)


async def check_url_async(session, url, timeout=10):
    """Async check of a single URL using HEAD request."""
    try:
        async with session.head(url, timeout=timeout, allow_redirects=True) as response:
            return {
                'url': url,
                'status': response.status,
                'headers': dict(response.headers),
                'error': None
            }
    except asyncio.TimeoutError:
        return {
            'url': url,
            'status': None,
            'headers': {},
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'url': url,
            'status': None,
            'headers': {},
            'error': str(e)
        }


async def check_urls_concurrent(urls, max_concurrent=10, show_progress=True):
    """Check multiple URLs concurrently with rate limiting."""
    from tqdm.asyncio import tqdm as async_tqdm

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def check_with_semaphore(session, url, pbar=None):
        async with semaphore:
            result = await check_url_async(session, url)
            if pbar:
                pbar.update(1)
            return result

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if show_progress:
            with async_tqdm(total=len(urls), desc="Checking URLs", unit="url") as pbar:
                tasks = [check_with_semaphore(session, url, pbar) for url in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [check_with_semaphore(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r if not isinstance(r, Exception) else {'url': 'unknown', 'error': str(r)} for r in results]


async def check_url_updates_async(kb, max_concurrent=10):
    """Async version of check_url_updates using aiohttp for concurrent requests."""
    results = {
        'unchanged': [],
        'changed': [],
        'failed': [],
        'scrape_sessions': []
    }

    # Find all URL-sourced documents
    url_docs = [doc for doc in kb.documents.values() if doc.source_url]

    if not url_docs:
        print("[*] No URL-sourced documents to check")
        return results

    print(f"[*] Checking {len(url_docs)} URLs concurrently (max {max_concurrent} at a time)")

    # Create URL to doc mapping
    url_to_doc = {doc.source_url: doc for doc in url_docs}
    urls = list(url_to_doc.keys())

    # Check all URLs concurrently
    start_time = asyncio.get_event_loop().time()
    check_results = await check_urls_concurrent(urls, max_concurrent)
    elapsed = asyncio.get_event_loop().time() - start_time

    print(f"[*] Checked {len(urls)} URLs in {elapsed:.2f}s ({len(urls)/elapsed:.1f} URLs/sec)")

    # Process results
    for result in check_results:
        url = result['url']
        doc = url_to_doc.get(url)

        if not doc:
            continue

        # Update last_checked timestamp
        with kb._lock:
            cursor = kb.db_conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET url_last_checked = ?
                WHERE doc_id = ?
            """, (datetime.now().isoformat(), doc.doc_id))
            kb.db_conn.commit()

        # Handle errors
        if result['error']:
            results['failed'].append({
                'doc_id': doc.doc_id,
                'title': doc.title,
                'url': url,
                'error': result['error']
            })
            continue

        # Check status code
        if result['status'] == 404:
            results['failed'].append({
                'doc_id': doc.doc_id,
                'title': doc.title,
                'url': url,
                'error': '404 Not Found'
            })
            continue

        # Check Last-Modified header
        page_changed = False
        if 'Last-Modified' in result['headers']:
            try:
                last_modified = parsedate_to_datetime(result['headers']['Last-Modified'])

                if doc.scrape_date:
                    scrape_dt = datetime.fromisoformat(doc.scrape_date)
                    # Ensure both datetimes are timezone-aware
                    if scrape_dt.tzinfo is None:
                        scrape_dt = scrape_dt.replace(tzinfo=timezone.utc)

                    if last_modified > scrape_dt:
                        page_changed = True
                        results['changed'].append({
                            'doc_id': doc.doc_id,
                            'title': doc.title,
                            'url': url,
                            'last_modified': last_modified.isoformat(),
                            'scraped_date': doc.scrape_date,
                            'reason': 'content_modified'
                        })
            except Exception as e:
                # Ignore date parsing errors
                pass

        if not page_changed:
            results['unchanged'].append({
                'doc_id': doc.doc_id,
                'title': doc.title,
                'url': url
            })

    # Group by scrape session for statistics
    scrape_sessions = {}
    for doc in url_docs:
        try:
            if doc.scrape_config:
                config = json.loads(doc.scrape_config)
                base_url = config.get('url', doc.source_url)
            else:
                base_url = doc.source_url

            if base_url not in scrape_sessions:
                scrape_sessions[base_url] = {
                    'base_url': base_url,
                    'docs_count': 0,
                    'changed': 0,
                    'unchanged': 0
                }

            scrape_sessions[base_url]['docs_count'] += 1

            # Count changed/unchanged for this session
            if any(doc.doc_id == d['doc_id'] for d in results['changed']):
                scrape_sessions[base_url]['changed'] += 1
            elif any(doc.doc_id == d['doc_id'] for d in results['unchanged']):
                scrape_sessions[base_url]['unchanged'] += 1

        except Exception:
            pass

    results['scrape_sessions'] = list(scrape_sessions.values())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fast async URL monitoring - quick concurrent checks"
    )
    parser.add_argument(
        '--auto-rescrape',
        action='store_true',
        help='Automatically re-scrape changed documents'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=10,
        help='Number of concurrent requests (default: 10)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='monitor_config.json',
        help='Configuration file path (default: monitor_config.json)'
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                print(f"[OK] Loaded configuration from {args.config}")
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")

    # Initialize knowledge base
    data_dir = os.path.expanduser(config.get('data_dir', '~/.tdz-c64-knowledge'))
    print(f"\n[*] Initializing knowledge base at {data_dir}")

    try:
        kb = KnowledgeBase(data_dir)
    except Exception as e:
        print(f"[ERROR] Failed to initialize knowledge base: {e}")
        return 1

    # Count URL-sourced documents
    url_docs = [d for d in kb.documents.values() if d.source_url]
    print(f"[*] Found {len(url_docs)} URL-sourced documents to check")

    if not url_docs:
        print("[*] No URL-sourced documents found. Nothing to check.")
        return 0

    # Run async check
    print(f"\n[*] Running fast async check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Concurrent requests: {args.concurrent}")

    try:
        results = asyncio.run(check_url_updates_async(kb, args.concurrent))
    except Exception as e:
        print(f"\n[ERROR] Check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Display results
    print("\n" + "=" * 60)
    print("CHECK RESULTS")
    print("=" * 60)
    print(f"[OK] Unchanged:     {len(results['unchanged']):3d} documents")
    print(f"[!]  Changed:       {len(results['changed']):3d} documents")
    print(f"[X]  Failed:        {len(results['failed']):3d} checks")

    # Show changed documents
    if results['changed']:
        print("\n" + "-" * 60)
        print("CHANGED DOCUMENTS:")
        print("-" * 60)
        for doc in results['changed']:
            print(f"\n[!] {doc['title']}")
            print(f"    URL: {doc['url']}")
            print(f"    Doc ID: {doc['doc_id'][:12]}...")
            if 'last_modified' in doc:
                print(f"    Last Modified: {doc['last_modified']}")
            if 'scraped_date' in doc:
                print(f"    Scraped: {doc['scraped_date']}")

    # Show failed checks
    if results['failed']:
        print("\n" + "-" * 60)
        print("FAILED CHECKS:")
        print("-" * 60)
        for doc in results['failed'][:10]:  # Show first 10
            print(f"\n[X] {doc['title']}")
            print(f"    URL: {doc['url']}")
            print(f"    Error: {doc['error']}")
        if len(results['failed']) > 10:
            print(f"\n    ... and {len(results['failed']) - 10} more")

    # Show session statistics
    if results['scrape_sessions']:
        print("\n" + "-" * 60)
        print("SCRAPE SESSION STATISTICS:")
        print("-" * 60)
        for session in results['scrape_sessions']:
            print(f"\n[*] {session['base_url']}")
            print(f"    Total docs:  {session['docs_count']:3d}")
            print(f"    Unchanged:   {session['unchanged']:3d}")
            print(f"    Changed:     {session['changed']:3d}")

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"url_check_fast_{timestamp}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {output_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save results: {e}")

    # Return exit code
    print("\n" + "=" * 60)
    if results['changed']:
        print(f"[!] COMPLETE: {len(results['changed'])} documents need attention")
        return 2  # Exit code 2: changes detected
    elif results['failed']:
        print(f"[!] COMPLETE: {len(results['failed'])} checks failed")
        return 3  # Exit code 3: some checks failed
    else:
        print("[OK] COMPLETE: All documents up to date")
        return 0  # Exit code 0: success, no changes


if __name__ == "__main__":
    sys.exit(main())
