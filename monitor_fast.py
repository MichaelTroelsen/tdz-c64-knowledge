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
import logging
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

# Configure logging
logger = logging.getLogger('monitor_fast')
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

# Add handler if not already added
if not logger.handlers:
    logger.addHandler(console_handler)


async def check_url_async(session, url, timeout=10, max_retries=3):
    """Async check of a single URL using HEAD request with retry logic.

    Args:
        session: aiohttp ClientSession
        url: URL to check
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Dictionary with check results
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            async with session.head(url, timeout=timeout, allow_redirects=True) as response:
                return {
                    'url': url,
                    'status': response.status,
                    'headers': dict(response.headers),
                    'error': None,
                    'attempts': attempt + 1
                }
        except asyncio.TimeoutError as e:
            last_error = 'Timeout'
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                backoff = 2 ** attempt
                logger.debug(f"Timeout for {url}, retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(backoff)
                continue
        except aiohttp.ClientConnectorError as e:
            last_error = f'Connection error: {str(e)}'
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.debug(f"Connection error for {url}, retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(backoff)
                continue
        except aiohttp.ServerTimeoutError as e:
            last_error = 'Server timeout'
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.debug(f"Server timeout for {url}, retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(backoff)
                continue
        except Exception as e:
            # Don't retry on other errors (DNS, SSL, etc.)
            return {
                'url': url,
                'status': None,
                'headers': {},
                'error': str(e),
                'attempts': attempt + 1
            }

    # All retries exhausted
    return {
        'url': url,
        'status': None,
        'headers': {},
        'error': last_error,
        'attempts': max_retries
    }


async def check_urls_concurrent(urls, max_concurrent=10, show_progress=True, adaptive=True):
    """Check multiple URLs concurrently with connection pooling and adaptive concurrency.

    Args:
        urls: List of URLs to check
        max_concurrent: Maximum concurrent requests (default: 10)
        show_progress: Show progress bar (default: True)
        adaptive: Enable adaptive concurrency based on response times (default: True)

    Returns:
        List of check results
    """
    from tqdm.asyncio import tqdm as async_tqdm
    import time

    # Adaptive concurrency tracking
    response_times = []
    current_concurrency = max_concurrent if not adaptive else min(5, max_concurrent)

    semaphore = asyncio.Semaphore(current_concurrency)
    results = []

    async def check_with_semaphore(session, url, pbar=None):
        async with semaphore:
            start_time = time.time()
            result = await check_url_async(session, url)
            elapsed = time.time() - start_time

            # Track response times for adaptive concurrency
            if adaptive:
                response_times.append(elapsed)

            if pbar:
                pbar.update(1)
            return result

    # Connection pooling configuration
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,  # Total connection limit
        limit_per_host=5,  # Per-host connection limit
        ttl_dns_cache=300,  # DNS cache TTL (5 minutes)
        enable_cleanup_closed=True,  # Clean up closed connections
        force_close=False,  # Reuse connections (keep-alive)
    )

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={'Connection': 'keep-alive'}  # Enable HTTP keep-alive
    ) as session:
        if show_progress:
            with async_tqdm(total=len(urls), desc="Checking URLs", unit="url") as pbar:
                # Process in batches for adaptive concurrency
                if adaptive and len(urls) > 20:
                    # Process first batch to measure performance
                    batch_size = min(10, len(urls))
                    first_batch = urls[:batch_size]
                    remaining = urls[batch_size:]

                    tasks = [check_with_semaphore(session, url, pbar) for url in first_batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    results.extend(batch_results)

                    # Adjust concurrency based on response times
                    if response_times:
                        avg_time = sum(response_times) / len(response_times)
                        old_concurrency = current_concurrency

                        if avg_time < 0.5:  # Fast responses
                            current_concurrency = min(max_concurrent, current_concurrency * 2)
                            logger.info(f"Fast responses (avg {avg_time:.2f}s), increasing concurrency: {old_concurrency} → {current_concurrency}")
                        elif avg_time > 2.0:  # Slow responses
                            current_concurrency = max(5, current_concurrency // 2)
                            logger.info(f"Slow responses (avg {avg_time:.2f}s), decreasing concurrency: {old_concurrency} → {current_concurrency}")
                        else:
                            logger.debug(f"Response times normal (avg {avg_time:.2f}s), maintaining concurrency: {current_concurrency}")

                        # Update semaphore with new concurrency
                        semaphore = asyncio.Semaphore(current_concurrency)
                        if show_progress:
                            pbar.set_description(f"Checking URLs (concurrency: {current_concurrency})")

                    # Process remaining URLs
                    if remaining:
                        tasks = [check_with_semaphore(session, url, pbar) for url in remaining]
                        remaining_results = await asyncio.gather(*tasks, return_exceptions=True)
                        results.extend(remaining_results)
                else:
                    # Non-adaptive or small batch - process all at once
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
