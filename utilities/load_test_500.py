#!/usr/bin/env python3
"""
Load Test for TDZ C64 Knowledge Base with 500+ Documents

Tests system performance and scalability with a large document corpus.
"""

import time
import os
import sys
import json
import statistics
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase


class LoadTest500:
    """Load testing with 500+ documents."""

    def __init__(self, data_dir: str = None, target_docs: int = 500):
        """Initialize load test."""
        if data_dir is None:
            data_dir = os.path.expanduser('~/.tdz-c64-knowledge')

        self.data_dir = data_dir
        self.target_docs = target_docs
        self.temp_docs = []
        self.results = {}

        print("=" * 70)
        print(f"TDZ C64 Knowledge Base - Load Test ({target_docs}+ Documents)")
        print("=" * 70)
        print(f"Started: {datetime.now().isoformat()}")
        print()

        # Initialize KB
        start = time.time()
        self.kb = KnowledgeBase(data_dir)
        init_time = (time.time() - start) * 1000

        self.initial_doc_count = len(self.kb.documents)
        print(f"Current documents: {self.initial_doc_count}")
        print(f"Target documents:  {self.target_docs}")
        print(f"Need to add:       {max(0, self.target_docs - self.initial_doc_count)}")
        print(f"Initialization:    {init_time:.2f} ms")
        print()

        # Get baseline benchmark if available
        self.baseline = None
        if Path('benchmark_results.json').exists():
            with open('benchmark_results.json', 'r') as f:
                self.baseline = json.load(f)
                print(f"Loaded baseline: {self.baseline['system_info']['documents']} docs")
                print()

    def generate_test_documents(self):
        """Generate synthetic C64 documentation for testing."""
        print("1. Generating Test Documents")
        print("-" * 70)

        docs_needed = max(0, self.target_docs - self.initial_doc_count)
        if docs_needed == 0:
            print(f"  Already have {self.initial_doc_count} documents, skipping generation")
            self.results['doc_generation'] = {'skipped': True}
            return

        # C64 topics for synthetic documents
        topics = [
            ("VIC-II Graphics", "The VIC-II chip controls all graphics output including sprites, character sets, and bitmap modes."),
            ("SID Audio", "The SID (Sound Interface Device) provides three-voice sound synthesis with programmable waveforms."),
            ("CIA Chips", "The CIA chips handle I/O operations including keyboard, joystick, and timers."),
            ("6502 CPU", "The 6502 processor runs at 1MHz and has a 16-bit address space."),
            ("Memory Map", "The C64 memory map includes 64KB RAM, ROM, and memory-mapped I/O."),
            ("BASIC Programming", "Commodore BASIC 2.0 provides a simple programming environment."),
            ("Assembly Language", "6502 assembly programming gives direct hardware control."),
            ("Sprites", "Hardware sprites allow smooth movement of up to 8 objects on screen."),
            ("Raster Interrupts", "Raster interrupts enable advanced graphics effects and timing."),
            ("Character Sets", "Custom character sets allow user-defined graphics."),
        ]

        # Create load test directory in project directory (allowed path)
        project_dir = Path(__file__).parent
        load_test_dir = project_dir / "load_test_docs"
        load_test_dir.mkdir(exist_ok=True)

        # Generate documents
        generated = []
        start = time.time()

        for i in range(docs_needed):
            topic_idx = i % len(topics)
            topic_name, topic_desc = topics[topic_idx]

            # Create synthetic content
            content = f"""# {topic_name} - Reference Document {i+1}

## Overview
{topic_desc}

## Technical Details
This is a synthetic test document created for load testing purposes. It contains
information about {topic_name.lower()} and related C64 programming concepts.

### Memory Addresses
The relevant memory addresses for {topic_name.lower()} are located in the range
$D000-$D400. Key registers include:
- Register 0: Control register ($D000)
- Register 1: Data register ($D001)
- Register 2: Status register ($D002)

### Code Example
```basic
10 POKE 53280,0
20 POKE 53281,0
30 PRINT "TEST DOCUMENT {i+1}"
```

### Assembly Example
```assembly
LDA #$00
STA $D020
STA $D021
RTS
```

## Programming Techniques
Common programming techniques for {topic_name.lower()} include initialization,
data transfer, and interrupt handling. This section contains detailed examples
and best practices for C64 development.

## Additional Content
{chr(10).join([f"Line {j}: More detailed information about {topic_name.lower()} feature {j}." for j in range(20)])}

## References
- Commodore 64 Programmer's Reference Guide
- C64 Hardware Manual
- {topic_name} Technical Specifications
"""

            # Create file in allowed directory
            test_file = load_test_dir / f"loadtest_{i+1:04d}.md"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # Add to KB
            try:
                doc_id = self.kb.add_document(
                    str(test_file),
                    title=f"Load Test: {topic_name} #{i+1}",
                    tags=['load-test', topic_name.lower().replace(' ', '-')]
                )
                generated.append(doc_id)
                self.temp_docs.append(str(test_file))
            except Exception as e:
                print(f"  Error adding document {i+1}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{docs_needed} documents...")

        elapsed = (time.time() - start) * 1000
        docs_per_sec = len(generated) / (elapsed / 1000) if elapsed > 0 else 0

        print(f"  Generated {len(generated)} documents in {elapsed:.2f} ms")
        print(f"  Rate: {docs_per_sec:.2f} docs/sec")
        print(f"  Total documents now: {len(self.kb.documents)}")
        print()

        self.results['doc_generation'] = {
            'docs_generated': len(generated),
            'time_ms': elapsed,
            'docs_per_sec': docs_per_sec,
            'total_docs': len(self.kb.documents)
        }

    def test_search_performance(self):
        """Test search performance with large document set."""
        print("2. Search Performance at Scale")
        print("-" * 70)

        queries = [
            'VIC-II sprite',
            'SID music',
            'raster interrupt',
            'memory map',
            'assembly programming',
            'character set',
            '6502 instruction',
            'color RAM',
            'BASIC programming',
            'CIA timer',
        ]

        # Test FTS5 search
        print("  FTS5 Search:")
        fts5_times = []
        for query in queries:
            start = time.time()
            results = self.kb.search(query, max_results=10)
            elapsed = (time.time() - start) * 1000
            fts5_times.append(elapsed)

        fts5_avg = statistics.mean(fts5_times)
        print(f"    Average: {fts5_avg:.2f} ms")
        print(f"    Median:  {statistics.median(fts5_times):.2f} ms")
        print(f"    Min:     {min(fts5_times):.2f} ms")
        print(f"    Max:     {max(fts5_times):.2f} ms")

        # Compare to baseline
        if self.baseline and 'fts5_search' in self.baseline['benchmarks']:
            baseline_avg = self.baseline['benchmarks']['fts5_search']['avg_time_ms']
            diff_pct = ((fts5_avg - baseline_avg) / baseline_avg) * 100
            print(f"    Baseline: {baseline_avg:.2f} ms")
            print(f"    Difference: {diff_pct:+.1f}%")

        print()

        # Test semantic search if enabled
        semantic_times = []
        if self.kb.use_semantic:
            print("  Semantic Search:")
            semantic_queries = queries[:4]  # Fewer queries for semantic

            # First query (model loading)
            start = time.time()
            results = self.kb.semantic_search(semantic_queries[0], max_results=10)
            first_query_time = (time.time() - start) * 1000

            # Subsequent queries
            for query in semantic_queries[1:]:
                start = time.time()
                results = self.kb.semantic_search(query, max_results=10)
                elapsed = (time.time() - start) * 1000
                semantic_times.append(elapsed)

            semantic_avg = statistics.mean(semantic_times)
            print(f"    First query: {first_query_time:.2f} ms")
            print(f"    Average: {semantic_avg:.2f} ms")
            print(f"    Median:  {statistics.median(semantic_times):.2f} ms")

            # Compare to baseline
            if self.baseline and 'semantic_search' in self.baseline['benchmarks']:
                baseline_avg = self.baseline['benchmarks']['semantic_search']['avg_time_ms']
                diff_pct = ((semantic_avg - baseline_avg) / baseline_avg) * 100
                print(f"    Baseline: {baseline_avg:.2f} ms")
                print(f"    Difference: {diff_pct:+.1f}%")

            print()

        # Test hybrid search if enabled
        hybrid_times = []
        if self.kb.use_semantic:
            print("  Hybrid Search:")
            hybrid_queries = queries[:4]

            for query in hybrid_queries:
                start = time.time()
                results = self.kb.hybrid_search(query, max_results=10, semantic_weight=0.3)
                elapsed = (time.time() - start) * 1000
                hybrid_times.append(elapsed)

            hybrid_avg = statistics.mean(hybrid_times)
            print(f"    Average: {hybrid_avg:.2f} ms")
            print(f"    Median:  {statistics.median(hybrid_times):.2f} ms")

            # Compare to baseline
            if self.baseline and 'hybrid_search' in self.baseline['benchmarks']:
                baseline_avg = self.baseline['benchmarks']['hybrid_search']['avg_time_ms']
                diff_pct = ((hybrid_avg - baseline_avg) / baseline_avg) * 100
                print(f"    Baseline: {baseline_avg:.2f} ms")
                print(f"    Difference: {diff_pct:+.1f}%")

            print()

        self.results['search_performance'] = {
            'fts5': {
                'avg_ms': fts5_avg,
                'median_ms': statistics.median(fts5_times),
                'min_ms': min(fts5_times),
                'max_ms': max(fts5_times),
            }
        }

        if semantic_times:
            self.results['search_performance']['semantic'] = {
                'avg_ms': semantic_avg,
                'median_ms': statistics.median(semantic_times),
            }

        if hybrid_times:
            self.results['search_performance']['hybrid'] = {
                'avg_ms': hybrid_avg,
                'median_ms': statistics.median(hybrid_times),
            }

    def test_concurrent_searches(self):
        """Test concurrent search operations."""
        print("3. Concurrent Search Performance")
        print("-" * 70)

        queries = [
            'VIC-II', 'SID', 'CIA', 'sprite', 'raster',
            'memory', 'BASIC', 'assembly', '6502', 'character'
        ]

        for num_workers in [2, 5, 10]:
            print(f"  Testing with {num_workers} concurrent workers:")

            start = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for query in queries:
                    future = executor.submit(self.kb.search, query, 10)
                    futures.append(future)

                results = []
                for future in as_completed(futures):
                    results.append(future.result())

            elapsed = (time.time() - start) * 1000
            throughput = len(queries) / (elapsed / 1000)

            print(f"    Total time: {elapsed:.2f} ms")
            print(f"    Throughput: {throughput:.2f} queries/sec")
            print()

            if 'concurrent_search' not in self.results:
                self.results['concurrent_search'] = {}

            self.results['concurrent_search'][f'{num_workers}_workers'] = {
                'total_time_ms': elapsed,
                'queries': len(queries),
                'throughput_qps': throughput
            }

    def test_memory_usage(self):
        """Test memory usage with large document set."""
        print("4. Memory Usage Analysis")
        print("-" * 70)

        process = psutil.Process()
        mem_info = process.memory_info()

        print(f"  RSS (Resident Set Size): {mem_info.rss / (1024**2):.2f} MB")
        print(f"  VMS (Virtual Memory):    {mem_info.vms / (1024**2):.2f} MB")
        print(f"  Documents in memory:     {len(self.kb.documents)}")

        # Calculate average memory per document
        mem_per_doc = mem_info.rss / len(self.kb.documents) / 1024  # KB per doc
        print(f"  Avg memory per doc:      {mem_per_doc:.2f} KB")
        print()

        self.results['memory_usage'] = {
            'rss_mb': mem_info.rss / (1024**2),
            'vms_mb': mem_info.vms / (1024**2),
            'documents': len(self.kb.documents),
            'kb_per_document': mem_per_doc
        }

    def test_database_size(self):
        """Check database size and growth."""
        print("5. Database Size Analysis")
        print("-" * 70)

        db_file = Path(self.data_dir) / "knowledge_base.db"
        if db_file.exists():
            db_size_mb = db_file.stat().st_size / (1024 * 1024)
            size_per_doc = db_size_mb / len(self.kb.documents)

            print(f"  Database size:      {db_size_mb:.2f} MB")
            print(f"  Documents:          {len(self.kb.documents)}")
            print(f"  MB per document:    {size_per_doc:.3f} MB")

            # Compare to baseline
            if self.baseline:
                baseline_size = self.baseline['system_info']['database_size_mb']
                baseline_docs = self.baseline['system_info']['documents']
                growth_pct = ((db_size_mb - baseline_size) / baseline_size) * 100
                doc_growth_pct = ((len(self.kb.documents) - baseline_docs) / baseline_docs) * 100

                print(f"  Baseline size:      {baseline_size:.2f} MB ({baseline_docs} docs)")
                print(f"  Size growth:        {growth_pct:+.1f}%")
                print(f"  Document growth:    {doc_growth_pct:+.1f}%")

            print()

            self.results['database_size'] = {
                'size_mb': db_size_mb,
                'documents': len(self.kb.documents),
                'mb_per_document': size_per_doc
            }

    def cleanup_test_documents(self):
        """Remove test documents added during load test."""
        print("6. Cleanup (Optional)")
        print("-" * 70)

        if not self.results.get('doc_generation', {}).get('docs_generated'):
            print("  No test documents to clean up")
            return

        project_dir = Path(__file__).parent
        print(f"  Test documents can be cleaned up by:")
        print(f"  1. Remove from KB: python cli.py remove --tag load-test")
        print(f"  2. Delete files: Remove {project_dir / 'load_test_docs'} directory")
        print()

        # Note: Files remain on disk for manual cleanup
        # This allows re-running tests or manual inspection

    def run_all_tests(self):
        """Run all load tests."""
        self.generate_test_documents()
        self.test_search_performance()
        self.test_concurrent_searches()
        self.test_memory_usage()
        self.test_database_size()
        self.cleanup_test_documents()

    def save_results(self, output_file: str = None):
        """Save load test results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'load_test_results_{timestamp}.json'

        results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'target_documents': self.target_docs,
                'initial_documents': self.initial_doc_count,
                'final_documents': len(self.kb.documents),
            },
            'results': self.results,
            'baseline_comparison': self.baseline is not None
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("=" * 70)
        print(f"Results saved to: {output_file}")
        print("=" * 70)

        return output_file

    def print_summary(self):
        """Print summary of load test results."""
        print()
        print("=" * 70)
        print("LOAD TEST SUMMARY")
        print("=" * 70)

        print(f"\nDocuments:")
        print(f"  Initial:  {self.initial_doc_count}")
        print(f"  Final:    {len(self.kb.documents)}")
        if 'doc_generation' in self.results:
            gen = self.results['doc_generation']
            if not gen.get('skipped'):
                print(f"  Added:    {gen['docs_generated']} ({gen['docs_per_sec']:.2f} docs/sec)")

        if 'search_performance' in self.results:
            print(f"\nSearch Performance:")
            sp = self.results['search_performance']
            if 'fts5' in sp:
                print(f"  FTS5:     {sp['fts5']['avg_ms']:.2f} ms avg")
            if 'semantic' in sp:
                print(f"  Semantic: {sp['semantic']['avg_ms']:.2f} ms avg")
            if 'hybrid' in sp:
                print(f"  Hybrid:   {sp['hybrid']['avg_ms']:.2f} ms avg")

        if 'concurrent_search' in self.results:
            print(f"\nConcurrent Throughput:")
            for workers, data in self.results['concurrent_search'].items():
                print(f"  {workers}: {data['throughput_qps']:.2f} queries/sec")

        if 'memory_usage' in self.results:
            mem = self.results['memory_usage']
            print(f"\nMemory Usage:")
            print(f"  RSS:      {mem['rss_mb']:.2f} MB")
            print(f"  Per doc:  {mem['kb_per_document']:.2f} KB")

        if 'database_size' in self.results:
            db = self.results['database_size']
            print(f"\nDatabase:")
            print(f"  Size:     {db['size_mb']:.2f} MB")
            print(f"  Per doc:  {db['mb_per_document']:.3f} MB")

        print()

    def close(self):
        """Close knowledge base connection."""
        self.kb.close()


def main():
    """Main load test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Load test with 500+ documents')
    parser.add_argument('--target', type=int, default=500, help='Target document count')
    parser.add_argument('--data-dir', help='Knowledge base data directory')
    parser.add_argument('--output', help='Output JSON file', default=None)

    args = parser.parse_args()

    # Run load test
    load_test = LoadTest500(data_dir=args.data_dir, target_docs=args.target)

    try:
        load_test.run_all_tests()
        load_test.print_summary()
        output_file = load_test.save_results(args.output)

        print(f"\nLoad test completed successfully!")
        print(f"  Results: {output_file}")

    finally:
        load_test.close()


if __name__ == '__main__':
    main()
