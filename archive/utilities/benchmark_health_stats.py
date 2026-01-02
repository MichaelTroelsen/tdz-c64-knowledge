#!/usr/bin/env python3
"""
Benchmark health_check() and get_stats() Performance

Tests the optimized versions with and without caching.
"""

import time
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase


def benchmark_health_check(kb: KnowledgeBase, runs: int = 10):
    """Benchmark health_check() with different modes."""
    print("=" * 70)
    print("Benchmarking health_check()")
    print("=" * 70)

    # Test 1: Quick check, no cache (fresh)
    times_quick_nocache = []
    for i in range(runs):
        start = time.time()
        result = kb.health_check(quick_check=True, use_cache=False)
        elapsed = (time.time() - start) * 1000
        times_quick_nocache.append(elapsed)

    avg_quick_nocache = sum(times_quick_nocache) / len(times_quick_nocache)
    print(f"\n1. Quick check (no cache): {avg_quick_nocache:.2f}ms avg")
    print(f"   Min: {min(times_quick_nocache):.2f}ms, Max: {max(times_quick_nocache):.2f}ms")

    # Test 2: Quick check, with cache (first call)
    start = time.time()
    result = kb.health_check(quick_check=True, use_cache=True)
    first_cached = (time.time() - start) * 1000
    print(f"\n2. Quick check (first cached call): {first_cached:.2f}ms")

    # Test 3: Quick check, with cache (subsequent calls)
    times_quick_cached = []
    for i in range(runs):
        start = time.time()
        result = kb.health_check(quick_check=True, use_cache=True)
        elapsed = (time.time() - start) * 1000
        times_quick_cached.append(elapsed)

    avg_quick_cached = sum(times_quick_cached) / len(times_quick_cached)
    print(f"\n3. Quick check (cached): {avg_quick_cached:.2f}ms avg")
    print(f"   Min: {min(times_quick_cached):.2f}ms, Max: {max(times_quick_cached):.2f}ms")

    # Calculate speedup
    speedup = avg_quick_nocache / avg_quick_cached
    pct_faster = ((avg_quick_nocache - avg_quick_cached) / avg_quick_nocache) * 100
    print(f"\n   Speedup: {speedup:.1f}x faster ({pct_faster:.1f}% improvement)")

    # Test 4: Full check (expensive operations)
    print(f"\n4. Full check (with integrity & orphan checks):")
    start = time.time()
    result = kb.health_check(quick_check=False, use_cache=False)
    full_check = (time.time() - start) * 1000
    print(f"   Time: {full_check:.2f}ms")
    print(f"   Slower than quick check by: {(full_check - avg_quick_nocache):.2f}ms")

    return {
        'quick_nocache_avg': avg_quick_nocache,
        'quick_cached_avg': avg_quick_cached,
        'full_check': full_check,
        'speedup': speedup,
        'pct_faster': pct_faster
    }


def benchmark_get_stats(kb: KnowledgeBase, runs: int = 50):
    """Benchmark get_stats() with different modes."""
    print("\n" + "=" * 70)
    print("Benchmarking get_stats()")
    print("=" * 70)

    # Test 1: No cache (fresh)
    times_nocache = []
    for i in range(runs):
        start = time.time()
        result = kb.get_stats(use_cache=False)
        elapsed = (time.time() - start) * 1000
        times_nocache.append(elapsed)

    avg_nocache = sum(times_nocache) / len(times_nocache)
    print(f"\n1. get_stats (no cache): {avg_nocache:.2f}ms avg")
    print(f"   Min: {min(times_nocache):.2f}ms, Max: {max(times_nocache):.2f}ms")

    # Test 2: With cache (first call)
    start = time.time()
    result = kb.get_stats(use_cache=True)
    first_cached = (time.time() - start) * 1000
    print(f"\n2. get_stats (first cached call): {first_cached:.2f}ms")

    # Test 3: With cache (subsequent calls)
    times_cached = []
    for i in range(runs):
        start = time.time()
        result = kb.get_stats(use_cache=True)
        elapsed = (time.time() - start) * 1000
        times_cached.append(elapsed)

    avg_cached = sum(times_cached) / len(times_cached)
    print(f"\n3. get_stats (cached): {avg_cached:.2f}ms avg")
    print(f"   Min: {min(times_cached):.2f}ms, Max: {max(times_cached):.2f}ms")

    # Calculate speedup
    speedup = avg_nocache / avg_cached
    pct_faster = ((avg_nocache - avg_cached) / avg_nocache) * 100
    print(f"\n   Speedup: {speedup:.1f}x faster ({pct_faster:.1f}% improvement)")

    return {
        'nocache_avg': avg_nocache,
        'cached_avg': avg_cached,
        'speedup': speedup,
        'pct_faster': pct_faster
    }


def main():
    """Run benchmarks."""
    print("Initializing KnowledgeBase...")
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    doc_count = len(kb.documents)
    print(f"Documents: {doc_count}")
    print()

    # Run benchmarks
    health_results = benchmark_health_check(kb, runs=10)
    stats_results = benchmark_get_stats(kb, runs=50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nhealth_check() optimization:")
    print(f"  Without cache: {health_results['quick_nocache_avg']:.2f}ms")
    print(f"  With cache:    {health_results['quick_cached_avg']:.2f}ms")
    print(f"  Improvement:   {health_results['speedup']:.1f}x faster ({health_results['pct_faster']:.1f}%)")

    print(f"\nget_stats() optimization:")
    print(f"  Without cache: {stats_results['nocache_avg']:.2f}ms")
    print(f"  With cache:    {stats_results['cached_avg']:.2f}ms")
    print(f"  Improvement:   {stats_results['speedup']:.1f}x faster ({stats_results['pct_faster']:.1f}%)")

    print("\nNotes:")
    print("- health_check() quick mode skips expensive integrity and orphan checks")
    print("- Both methods use 5-minute TTL cache for health, 1-minute for stats")
    print("- Caching provides near-instant responses for repeated calls")
    print("- Production deployments should use quick_check=True for health endpoints")

    kb.close()


if __name__ == '__main__':
    main()
