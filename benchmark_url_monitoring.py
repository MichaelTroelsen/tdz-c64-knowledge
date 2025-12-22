#!/usr/bin/env python3
"""
Benchmark script for URL monitoring performance

Measures performance of:
- URL discovery (crawling websites)
- URL checking (Last-Modified headers)
- Structure discovery vs quick checks
- Concurrent vs sequential operations

Usage:
    python benchmark_url_monitoring.py [--output FILENAME]
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase


class URLMonitorBenchmark:
    """Benchmark suite for URL monitoring operations."""

    def __init__(self, kb):
        self.kb = kb
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'documents_total': len(kb.documents),
                'url_documents': len([d for d in kb.documents.values() if d.source_url])
            },
            'benchmarks': {}
        }

    def benchmark_quick_check(self):
        """Benchmark quick URL check (Last-Modified only)."""
        print("\n[1/4] Benchmarking Quick Check...")
        print("      (Last-Modified headers only)")

        url_docs = [d for d in self.kb.documents.values() if d.source_url]
        if not url_docs:
            print("      [SKIP] No URL documents found")
            return

        start_time = time.time()
        results = self.kb.check_url_updates(
            auto_rescrape=False,
            check_structure=False  # Quick mode
        )
        elapsed = time.time() - start_time

        docs_checked = len(results['unchanged']) + len(results['changed']) + len(results['failed'])
        avg_per_doc = (elapsed / docs_checked) if docs_checked > 0 else 0

        self.results['benchmarks']['quick_check'] = {
            'total_time': elapsed,
            'documents_checked': docs_checked,
            'avg_per_document': avg_per_doc,
            'unchanged': len(results['unchanged']),
            'changed': len(results['changed']),
            'failed': len(results['failed'])
        }

        print(f"      Total time: {elapsed:.2f}s")
        print(f"      Documents: {docs_checked}")
        print(f"      Avg per doc: {avg_per_doc:.3f}s")
        print(f"      Throughput: {docs_checked/elapsed:.1f} docs/sec")

    def benchmark_full_check(self):
        """Benchmark full check with structure discovery."""
        print("\n[2/4] Benchmarking Full Check...")
        print("      (With structure discovery)")

        url_docs = [d for d in self.kb.documents.values() if d.source_url]
        if not url_docs:
            print("      [SKIP] No URL documents found")
            return

        # Only test on a subset to avoid long runtime
        test_limit = min(3, len(url_docs))
        print(f"      Testing on {test_limit} scrape sessions (limited for speed)")

        start_time = time.time()
        results = self.kb.check_url_updates(
            auto_rescrape=False,
            check_structure=True  # Full mode with discovery
        )
        elapsed = time.time() - start_time

        sessions_checked = len(results.get('scrape_sessions', []))
        avg_per_session = (elapsed / sessions_checked) if sessions_checked > 0 else 0

        self.results['benchmarks']['full_check'] = {
            'total_time': elapsed,
            'sessions_checked': sessions_checked,
            'avg_per_session': avg_per_session,
            'new_pages': len(results.get('new_pages', [])),
            'missing_pages': len(results.get('missing_pages', [])),
            'urls_discovered': sum(session.get('docs_count', 0)
                                  for session in results.get('scrape_sessions', []))
        }

        print(f"      Total time: {elapsed:.2f}s")
        print(f"      Sessions: {sessions_checked}")
        print(f"      Avg per session: {avg_per_session:.2f}s")
        print(f"      New pages: {len(results.get('new_pages', []))}")
        print(f"      Missing: {len(results.get('missing_pages', []))}")

    def benchmark_url_discovery(self):
        """Benchmark URL discovery (crawling) performance."""
        print("\n[3/4] Benchmarking URL Discovery...")
        print("      (Website crawling)")

        # Get first URL document to test discovery
        url_docs = [d for d in self.kb.documents.values() if d.source_url and d.scrape_config]
        if not url_docs:
            print("      [SKIP] No URL documents with scrape config found")
            return

        test_doc = url_docs[0]

        # Parse scrape_config (stored as JSON string in database)
        import json as json_module
        if isinstance(test_doc.scrape_config, str):
            config = json_module.loads(test_doc.scrape_config)
        else:
            config = test_doc.scrape_config

        base_url = config.get('base_url', test_doc.source_url)

        print(f"      Testing: {base_url}")
        print(f"      Depth: {config.get('depth', 3)}, Max pages: 100")

        start_time = time.time()
        discovered = self.kb._discover_urls(base_url, config, max_pages=100)
        elapsed = time.time() - start_time

        self.results['benchmarks']['url_discovery'] = {
            'total_time': elapsed,
            'base_url': base_url,
            'urls_discovered': len(discovered),
            'discovery_rate': len(discovered) / elapsed if elapsed > 0 else 0
        }

        print(f"      Total time: {elapsed:.2f}s")
        print(f"      URLs found: {len(discovered)}")
        print(f"      Rate: {len(discovered)/elapsed:.1f} URLs/sec")

    def benchmark_http_requests(self):
        """Benchmark individual HTTP request performance."""
        print("\n[4/4] Benchmarking HTTP Requests...")
        print("      (Individual HEAD requests)")

        url_docs = [d for d in self.kb.documents.values() if d.source_url]
        if not url_docs:
            print("      [SKIP] No URL documents found")
            return

        # Test first 10 URLs
        test_urls = [d.source_url for d in url_docs[:10]]
        print(f"      Testing {len(test_urls)} URLs")

        import requests
        times = []

        for i, url in enumerate(test_urls):
            try:
                start = time.time()
                response = requests.head(url, timeout=10, allow_redirects=True)
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"      [{i+1}/{len(test_urls)}] {elapsed:.3f}s - {response.status_code} - {url[:60]}")
            except Exception as e:
                print(f"      [{i+1}/{len(test_urls)}] FAILED - {str(e)[:40]}")

        if times:
            self.results['benchmarks']['http_requests'] = {
                'requests_tested': len(times),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'median_time': statistics.median(times),
                'total_time': sum(times)
            }

            print(f"\n      Statistics:")
            print(f"      Avg: {statistics.mean(times):.3f}s")
            print(f"      Min: {min(times):.3f}s")
            print(f"      Max: {max(times):.3f}s")
            print(f"      Median: {statistics.median(times):.3f}s")

    def run_all(self):
        """Run all benchmarks."""
        print("=" * 60)
        print("URL Monitoring Performance Benchmark")
        print("=" * 60)
        print(f"Knowledge Base: {len(self.kb.documents)} documents")
        print(f"URL Documents: {self.results['system_info']['url_documents']}")

        self.benchmark_quick_check()
        self.benchmark_http_requests()
        self.benchmark_url_discovery()
        # Skip full check by default as it's slow
        # self.benchmark_full_check()

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if 'quick_check' in self.results['benchmarks']:
            qc = self.results['benchmarks']['quick_check']
            print(f"\nQuick Check:")
            print(f"  Time: {qc['total_time']:.2f}s for {qc['documents_checked']} docs")
            print(f"  Speed: {qc['avg_per_document']:.3f}s per document")

        if 'http_requests' in self.results['benchmarks']:
            hr = self.results['benchmarks']['http_requests']
            print(f"\nHTTP Requests:")
            print(f"  Avg: {hr['avg_time']:.3f}s")
            print(f"  Range: {hr['min_time']:.3f}s - {hr['max_time']:.3f}s")

        if 'url_discovery' in self.results['benchmarks']:
            ud = self.results['benchmarks']['url_discovery']
            print(f"\nURL Discovery:")
            print(f"  Time: {ud['total_time']:.2f}s")
            print(f"  Found: {ud['urls_discovered']} URLs")
            print(f"  Rate: {ud['discovery_rate']:.1f} URLs/sec")

        if 'full_check' in self.results['benchmarks']:
            fc = self.results['benchmarks']['full_check']
            print(f"\nFull Check:")
            print(f"  Time: {fc['total_time']:.2f}s for {fc['sessions_checked']} sessions")
            print(f"  Avg: {fc['avg_per_session']:.2f}s per session")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark URL monitoring performance"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output file for benchmark results (default: benchmark_results.json)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='~/.tdz-c64-knowledge',
        help='Knowledge base data directory'
    )

    args = parser.parse_args()

    # Initialize knowledge base
    data_dir = os.path.expanduser(args.data_dir)
    print(f"Initializing knowledge base at {data_dir}...")

    try:
        kb = KnowledgeBase(data_dir)
    except Exception as e:
        print(f"ERROR: Failed to initialize knowledge base: {e}")
        return 1

    # Run benchmarks
    benchmark = URLMonitorBenchmark(kb)
    results = benchmark.run_all()
    benchmark.print_summary()

    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {args.output}")
    except Exception as e:
        print(f"[WARNING] Failed to save results: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
