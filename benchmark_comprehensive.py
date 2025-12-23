#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for TDZ C64 Knowledge Base

Tests all major operations and documents performance baselines.
"""

import time
import os
import sys
import json
import statistics
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase


class PerformanceBenchmark:
    """Comprehensive performance testing suite."""

    def __init__(self, data_dir: str = None):
        """Initialize benchmark with knowledge base."""
        if data_dir is None:
            data_dir = os.path.expanduser('~/.tdz-c64-knowledge')

        print("=" * 70)
        print("TDZ C64 Knowledge Base - Comprehensive Performance Benchmark")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print()

        # Initialize KB
        start = time.time()
        self.kb = KnowledgeBase(data_dir)
        init_time = (time.time() - start) * 1000

        # Get system info from health check
        health = self.kb.health_check()

        self.system_info = {
            'timestamp': datetime.now().isoformat(),
            'database_path': str(self.kb.db_file),
            'database_size_mb': health['database'].get('size_mb', 0),
            'documents': health['metrics'].get('documents', len(self.kb.documents)),
            'initialization_time_ms': init_time,
            'fts5_enabled': health['features'].get('fts5_enabled', False),
            'semantic_enabled': health['features'].get('semantic_search_enabled', False),
            'bm25_enabled': health['features'].get('bm25_enabled', False),
        }

        print(f"Database: {self.system_info['database_size_mb']:.2f} MB")
        print(f"Documents: {self.system_info['documents']}")
        print(f"Initialization: {init_time:.2f} ms")
        print()

        self.results = {}

    def benchmark_fts5_search(self):
        """Benchmark FTS5 full-text search."""
        print("1. FTS5 Search Performance")
        print("-" * 70)

        if not self.system_info['fts5_enabled']:
            print("  SKIPPED: FTS5 not enabled")
            self.results['fts5_search'] = {'skipped': True}
            return

        queries = [
            'VIC-II sprite',
            'SID music',
            'raster interrupt',
            'memory map',
            'assembly programming',
            'character set',
            '6502 instruction',
            'color RAM',
        ]

        times = []
        result_counts = []

        for query in queries:
            start = time.time()
            try:
                results = self.kb.search(query, max_results=10)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                result_counts.append(len(results))
                print(f"  {query:25s} - {elapsed:7.2f} ms ({len(results):2d} results)")
            except Exception as e:
                print(f"  {query:25s} - ERROR: {e}")
                times.append(0)
                result_counts.append(0)

        avg_time = statistics.mean(times) if times else 0
        median_time = statistics.median(times) if times else 0

        print()
        print(f"  Average: {avg_time:.2f} ms")
        print(f"  Median:  {median_time:.2f} ms")
        print(f"  Min:     {min(times):.2f} ms")
        print(f"  Max:     {max(times):.2f} ms")
        print()

        self.results['fts5_search'] = {
            'queries': len(queries),
            'avg_time_ms': avg_time,
            'median_time_ms': median_time,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'all_times_ms': times,
            'avg_results': statistics.mean(result_counts),
        }

    def benchmark_semantic_search(self):
        """Benchmark semantic search with embeddings."""
        print("2. Semantic Search Performance")
        print("-" * 70)

        if not self.system_info['semantic_enabled']:
            print("  SKIPPED: Semantic search not enabled")
            self.results['semantic_search'] = {'skipped': True}
            return

        queries = [
            'how to program sprites',
            'sound synthesis techniques',
            'graphics display modes',
            'memory organization',
        ]

        times = []
        result_counts = []
        first_query_time = None

        for i, query in enumerate(queries):
            start = time.time()
            try:
                results = self.kb.semantic_search(query, max_results=10)
                elapsed = (time.time() - start) * 1000

                if i == 0:
                    first_query_time = elapsed
                    print(f"  {query:30s} - {elapsed:7.2f} ms (first query - model loading)")
                else:
                    times.append(elapsed)
                    result_counts.append(len(results))
                    print(f"  {query:30s} - {elapsed:7.2f} ms ({len(results):2d} results)")
            except Exception as e:
                print(f"  {query:30s} - ERROR: {e}")
                if i > 0:
                    times.append(0)
                    result_counts.append(0)

        if times:
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)

            print()
            print(f"  First query (with loading): {first_query_time:.2f} ms")
            print(f"  Average (subsequent):       {avg_time:.2f} ms")
            print(f"  Median:                     {median_time:.2f} ms")
            print()

            self.results['semantic_search'] = {
                'queries': len(queries),
                'first_query_ms': first_query_time,
                'avg_time_ms': avg_time,
                'median_time_ms': median_time,
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'all_times_ms': times,
                'avg_results': statistics.mean(result_counts) if result_counts else 0,
            }
        else:
            print("  No valid queries completed")
            self.results['semantic_search'] = {'error': 'No queries completed'}

    def benchmark_hybrid_search(self):
        """Benchmark hybrid search (FTS5 + semantic)."""
        print("3. Hybrid Search Performance")
        print("-" * 70)

        if not (self.system_info['fts5_enabled'] and self.system_info['semantic_enabled']):
            print("  SKIPPED: Requires both FTS5 and semantic search")
            self.results['hybrid_search'] = {'skipped': True}
            return

        queries = [
            ('VIC-II programming', 0.3),  # Balanced
            ('sprite techniques', 0.5),   # Equal weight
            ('SID register', 0.1),        # Keyword-focused
            ('display concepts', 0.7),    # Semantic-focused
        ]

        times = []
        result_counts = []

        for query, weight in queries:
            start = time.time()
            try:
                results = self.kb.hybrid_search(query, max_results=10, semantic_weight=weight)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                result_counts.append(len(results))
                print(f"  {query:25s} (w={weight}) - {elapsed:7.2f} ms ({len(results):2d} results)")
            except Exception as e:
                print(f"  {query:25s} (w={weight}) - ERROR: {e}")
                times.append(0)
                result_counts.append(0)

        if times:
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)

            print()
            print(f"  Average: {avg_time:.2f} ms")
            print(f"  Median:  {median_time:.2f} ms")
            print()

            self.results['hybrid_search'] = {
                'queries': len(queries),
                'avg_time_ms': avg_time,
                'median_time_ms': median_time,
                'min_time_ms': min(times),
                'max_time_ms': max(times),
                'all_times_ms': times,
                'avg_results': statistics.mean(result_counts) if result_counts else 0,
            }
        else:
            self.results['hybrid_search'] = {'error': 'No queries completed'}

    def benchmark_document_operations(self):
        """Benchmark document retrieval operations."""
        print("4. Document Operations Performance")
        print("-" * 70)

        # Get a sample document ID
        if not self.kb.documents:
            print("  SKIPPED: No documents in database")
            self.results['document_ops'] = {'skipped': True}
            return

        doc_ids = list(self.kb.documents.keys())[:10]

        # Test get_document
        times = []
        for doc_id in doc_ids:
            start = time.time()
            try:
                doc = self.kb.get_document(doc_id)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                print(f"  ERROR retrieving {doc_id}: {e}")

        avg_get = statistics.mean(times) if times else 0
        print(f"  get_document() avg:  {avg_get:.2f} ms ({len(times)} docs)")

        # Test list_documents
        start = time.time()
        docs = self.kb.list_documents()
        list_time = (time.time() - start) * 1000
        print(f"  list_documents():    {list_time:.2f} ms ({len(docs)} docs)")

        # Test get_stats
        start = time.time()
        stats = self.kb.get_stats()
        stats_time = (time.time() - start) * 1000
        print(f"  get_stats():         {stats_time:.2f} ms")

        print()

        self.results['document_ops'] = {
            'get_document_avg_ms': avg_get,
            'list_documents_ms': list_time,
            'get_stats_ms': stats_time,
        }

    def benchmark_health_check(self):
        """Benchmark health check operation."""
        print("5. Health Check Performance")
        print("-" * 70)

        times = []
        for i in range(5):
            start = time.time()
            health = self.kb.health_check()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time = statistics.mean(times)
        print(f"  health_check() avg:  {avg_time:.2f} ms (5 runs)")
        print(f"  Status:              {health['status']}")
        print(f"  Features available:  {sum(1 for v in health['features'].values() if v and isinstance(v, bool))}")
        print()

        self.results['health_check'] = {
            'avg_time_ms': avg_time,
            'status': health['status'],
            'features': health['features'],
        }

    def benchmark_entity_extraction(self):
        """Benchmark entity extraction (regex only, no LLM)."""
        print("6. Entity Extraction Performance (Regex)")
        print("-" * 70)

        test_texts = [
            "The VIC-II chip at $D000 controls graphics.",
            "SID programming with LDA and STA instructions.",
            "Sprites and raster interrupts for C64.",
            "CIA1 chip handles keyboard at $DC00.",
        ]

        times = []
        entity_counts = []

        for text in test_texts:
            start = time.time()
            entities = self.kb._extract_entities_regex(text)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            entity_counts.append(len(entities))

        avg_time = statistics.mean(times)
        avg_entities = statistics.mean(entity_counts)

        print(f"  _extract_entities_regex() avg: {avg_time:.2f} ms")
        print(f"  Average entities found:         {avg_entities:.1f}")
        print(f"  Min time: {min(times):.2f} ms, Max time: {max(times):.2f} ms")
        print()

        self.results['entity_extraction_regex'] = {
            'avg_time_ms': avg_time,
            'avg_entities_found': avg_entities,
            'min_time_ms': min(times),
            'max_time_ms': max(times),
        }

    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        self.benchmark_fts5_search()
        self.benchmark_semantic_search()
        self.benchmark_hybrid_search()
        self.benchmark_document_operations()
        self.benchmark_health_check()
        self.benchmark_entity_extraction()

    def save_results(self, output_file: str = None):
        """Save benchmark results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'benchmark_results_{timestamp}.json'

        results = {
            'system_info': self.system_info,
            'benchmarks': self.results,
            'completed': datetime.now().isoformat(),
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("=" * 70)
        print(f"Results saved to: {output_file}")
        print("=" * 70)

        return output_file

    def print_summary(self):
        """Print summary of benchmark results."""
        print()
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(f"\nSystem:")
        print(f"  Documents:     {self.system_info['documents']}")
        print(f"  Database:      {self.system_info['database_size_mb']:.2f} MB")
        print(f"  Init time:     {self.system_info['initialization_time_ms']:.2f} ms")

        print(f"\nSearch Performance:")
        if 'fts5_search' in self.results and not self.results['fts5_search'].get('skipped'):
            print(f"  FTS5:          {self.results['fts5_search']['avg_time_ms']:.2f} ms avg")
        if 'semantic_search' in self.results and not self.results['semantic_search'].get('skipped'):
            print(f"  Semantic:      {self.results['semantic_search']['avg_time_ms']:.2f} ms avg")
        if 'hybrid_search' in self.results and not self.results['hybrid_search'].get('skipped'):
            print(f"  Hybrid:        {self.results['hybrid_search']['avg_time_ms']:.2f} ms avg")

        print(f"\nOperations:")
        if 'document_ops' in self.results:
            ops = self.results['document_ops']
            print(f"  Get document:  {ops.get('get_document_avg_ms', 0):.2f} ms avg")
            print(f"  List docs:     {ops.get('list_documents_ms', 0):.2f} ms")
            print(f"  Stats:         {ops.get('kb_stats_ms', 0):.2f} ms")

        if 'health_check' in self.results:
            print(f"  Health check:  {self.results['health_check']['avg_time_ms']:.2f} ms avg")

        if 'entity_extraction_regex' in self.results:
            print(f"  Entity regex:  {self.results['entity_extraction_regex']['avg_time_ms']:.2f} ms avg")

        print()

    def close(self):
        """Close knowledge base connection."""
        self.kb.close()


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive performance benchmark')
    parser.add_argument('--data-dir', help='Knowledge base data directory')
    parser.add_argument('--output', help='Output JSON file', default=None)

    args = parser.parse_args()

    # Run benchmark
    benchmark = PerformanceBenchmark(data_dir=args.data_dir)

    try:
        benchmark.run_all_benchmarks()
        benchmark.print_summary()
        output_file = benchmark.save_results(args.output)

        print(f"\nBenchmark completed successfully!")
        print(f"  Results: {output_file}")

    finally:
        benchmark.close()


if __name__ == '__main__':
    main()
