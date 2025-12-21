#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - Performance Benchmark Suite

Comprehensive benchmarking for:
- Search operations (FTS5, semantic, hybrid)
- Document ingestion pipeline
- Entity extraction
- Large dataset handling
- Concurrent operations
"""

import os
import time
import tempfile
import statistics
from pathlib import Path
from typing import List, Dict, Any, Callable
import json

from server import KnowledgeBase


class Colors:
    """Terminal colors for output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


class BenchmarkResult:
    """Store benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.iterations = 0
        self.total_time = 0.0
        self.metadata: Dict[str, Any] = {}

    def add_time(self, duration: float):
        """Add a timing measurement."""
        self.times.append(duration)
        self.iterations += 1
        self.total_time += duration

    def get_stats(self) -> Dict[str, float]:
        """Calculate statistics."""
        if not self.times:
            return {}

        return {
            'min': min(self.times),
            'max': max(self.times),
            'mean': statistics.mean(self.times),
            'median': statistics.median(self.times),
            'stdev': statistics.stdev(self.times) if len(self.times) > 1 else 0.0,
            'total': self.total_time,
            'iterations': self.iterations
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        if not stats:
            print(f"{Colors.RED}No data collected{Colors.END}")
            return

        print(f"{Colors.BOLD}{self.name}{Colors.END}")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Mean:       {Colors.GREEN}{stats['mean']*1000:.2f}ms{Colors.END}")
        print(f"  Median:     {stats['median']*1000:.2f}ms")
        print(f"  Min:        {stats['min']*1000:.2f}ms")
        print(f"  Max:        {stats['max']*1000:.2f}ms")
        if stats['stdev'] > 0:
            print(f"  Std Dev:    {stats['stdev']*1000:.2f}ms")
        print(f"  Total:      {stats['total']:.2f}s")

        # Print metadata
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
        print()


class PerformanceBenchmark:
    """Main benchmark suite."""

    def __init__(self, data_dir: str = None):
        """Initialize benchmark suite."""
        self.data_dir = data_dir or tempfile.mkdtemp(prefix="kb_benchmark_")
        self.kb: KnowledgeBase = None
        self.results: List[BenchmarkResult] = []
        self.sample_docs: List[str] = []

    def setup(self):
        """Setup benchmark environment."""
        print(f"{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}TDZ C64 Knowledge Base - Performance Benchmark{Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

        print(f"Data directory: {self.data_dir}\n")

        # Disable security restrictions for benchmarking
        os.environ["ALLOWED_DOCS_DIRS"] = ""

        # Initialize KB
        print("Initializing KnowledgeBase...")
        start = time.time()
        self.kb = KnowledgeBase(self.data_dir)
        init_time = time.time() - start
        print(f"  {Colors.GREEN}Initialized in {init_time:.2f}s{Colors.END}\n")

    def teardown(self):
        """Cleanup benchmark environment."""
        if self.kb:
            self.kb.close()

        # Cleanup temp files
        for doc_path in self.sample_docs:
            if os.path.exists(doc_path):
                os.unlink(doc_path)

    def create_sample_document(self, size: str = "small") -> str:
        """Create a sample document for testing.

        Args:
            size: Document size (small, medium, large)
        """
        content_map = {
            "small": "The VIC-II chip controls video output on the Commodore 64.\n" * 10,
            "medium": "The VIC-II chip controls video output on the Commodore 64.\n" * 100,
            "large": "The VIC-II chip controls video output on the Commodore 64.\n" * 1000
        }

        content = content_map.get(size, content_map["small"])

        # Add some variety
        content += "\nThe SID chip provides audio synthesis.\n" * (len(content_map[size].split('\n')) // 2)
        content += "\nMemory addresses $D000-$D3FF control hardware registers.\n" * (len(content_map[size].split('\n')) // 3)

        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write(content)
        f.close()

        self.sample_docs.append(f.name)
        return f.name

    def benchmark(self, name: str, func: Callable, iterations: int = 10, **kwargs) -> BenchmarkResult:
        """Run a benchmark test.

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            **kwargs: Arguments to pass to func
        """
        result = BenchmarkResult(name)

        print(f"Running: {name} ({iterations} iterations)...")

        for i in range(iterations):
            start = time.time()
            try:
                output = func(**kwargs)
                duration = time.time() - start
                result.add_time(duration)

                # Store metadata from first iteration
                if i == 0 and isinstance(output, dict):
                    result.metadata.update(output)
                elif i == 0 and isinstance(output, list):
                    result.metadata['result_count'] = len(output)
            except Exception as e:
                print(f"  {Colors.RED}Error in iteration {i+1}: {e}{Colors.END}")
                break

        self.results.append(result)
        return result

    # ========== Document Ingestion Benchmarks ==========

    def benchmark_ingestion_small(self):
        """Benchmark ingestion of small documents."""
        doc_path = self.create_sample_document("small")

        def ingest():
            doc = self.kb.add_document(doc_path, title="Small Test Doc", tags=["test"])
            return {"doc_id": doc.doc_id, "chunks": doc.total_chunks}

        self.benchmark("Document Ingestion (Small)", ingest, iterations=20)

    def benchmark_ingestion_medium(self):
        """Benchmark ingestion of medium documents."""
        doc_path = self.create_sample_document("medium")

        def ingest():
            doc = self.kb.add_document(doc_path, title="Medium Test Doc", tags=["test"])
            return {"doc_id": doc.doc_id, "chunks": doc.total_chunks}

        self.benchmark("Document Ingestion (Medium)", ingest, iterations=10)

    def benchmark_ingestion_large(self):
        """Benchmark ingestion of large documents."""
        doc_path = self.create_sample_document("large")

        def ingest():
            doc = self.kb.add_document(doc_path, title="Large Test Doc", tags=["test"])
            return {"doc_id": doc.doc_id, "chunks": doc.total_chunks}

        self.benchmark("Document Ingestion (Large)", ingest, iterations=5)

    # ========== Search Benchmarks ==========

    def benchmark_search_fts5(self):
        """Benchmark FTS5 search."""
        # Add some documents first
        self._ensure_documents(10)

        def search():
            results = self.kb.search("VIC-II chip", max_results=10)
            return {"results": len(results)}

        self.benchmark("Search (FTS5)", search, iterations=100)

    def benchmark_search_semantic(self):
        """Benchmark semantic search."""
        if not self.kb.use_semantic:
            print(f"  {Colors.YELLOW}Skipped: Semantic search not enabled{Colors.END}\n")
            return

        self._ensure_documents(10)

        def search():
            results = self.kb.semantic_search("video output control", max_results=10)
            return {"results": len(results)}

        self.benchmark("Search (Semantic)", search, iterations=50)

    def benchmark_search_hybrid(self):
        """Benchmark hybrid search."""
        if not self.kb.use_semantic:
            print(f"  {Colors.YELLOW}Skipped: Semantic search not enabled{Colors.END}\n")
            return

        self._ensure_documents(10)

        def search():
            results = self.kb.hybrid_search("chip registers", max_results=10)
            return {"results": len(results)}

        self.benchmark("Search (Hybrid)", search, iterations=50)

    def benchmark_search_complex_query(self):
        """Benchmark complex multi-term search."""
        self._ensure_documents(10)

        def search():
            results = self.kb.search("VIC-II SID memory address register", max_results=10)
            return {"results": len(results)}

        self.benchmark("Search (Complex Query)", search, iterations=100)

    # ========== Entity Extraction Benchmarks ==========

    def benchmark_entity_extraction(self):
        """Benchmark entity extraction."""
        if not hasattr(self.kb, 'extract_entities'):
            print(f"  {Colors.YELLOW}Skipped: Entity extraction not available{Colors.END}\n")
            return

        docs = self._ensure_documents(5)
        doc_id = docs[0]

        def extract():
            self.kb.extract_entities(doc_id, confidence_threshold=0.7)
            return {}

        self.benchmark("Entity Extraction", extract, iterations=10)

    # ========== Large Dataset Benchmarks ==========

    def benchmark_large_dataset_search(self):
        """Benchmark search with large dataset."""
        print(f"\n{Colors.CYAN}Building large dataset (100 documents)...{Colors.END}")
        self._ensure_documents(100)
        print(f"{Colors.GREEN}Dataset ready{Colors.END}\n")

        def search():
            results = self.kb.search("VIC-II", max_results=50)
            return {"results": len(results)}

        self.benchmark("Search (100 docs)", search, iterations=50)

    def benchmark_large_result_set(self):
        """Benchmark search returning many results."""
        self._ensure_documents(50)

        def search():
            results = self.kb.search("the", max_results=100)  # Common word
            return {"results": len(results)}

        self.benchmark("Search (Large Result Set)", search, iterations=20)

    # ========== Helper Methods ==========

    def _ensure_documents(self, count: int) -> List[str]:
        """Ensure KB has at least 'count' documents."""
        current_count = len(self.kb.documents)
        doc_ids = []

        if current_count < count:
            needed = count - current_count
            print(f"  Adding {needed} documents...")

            for i in range(needed):
                doc_path = self.create_sample_document("small")
                doc = self.kb.add_document(
                    doc_path,
                    title=f"Benchmark Doc {current_count + i + 1}",
                    tags=["benchmark", "test"]
                )
                doc_ids.append(doc.doc_id)
        else:
            doc_ids = list(self.kb.documents.keys())[:count]

        return doc_ids

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}Benchmark Results{Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

        for result in self.results:
            result.print_stats()

        # Overall summary
        total_tests = len(self.results)
        total_time = sum(r.total_time for r in self.results)

        print(f"{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total benchmarks: {total_tests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON file."""
        data = {
            "timestamp": time.time(),
            "data_dir": self.data_dir,
            "total_documents": len(self.kb.documents) if self.kb else 0,
            "results": [
                {
                    "name": r.name,
                    "stats": r.get_stats(),
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"{Colors.GREEN}Results saved to {output_file}{Colors.END}")

    def run_all(self):
        """Run all benchmarks."""
        try:
            self.setup()

            # Document ingestion
            print(f"\n{Colors.BOLD}Document Ingestion Benchmarks{Colors.END}")
            print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
            self.benchmark_ingestion_small()
            self.benchmark_ingestion_medium()
            self.benchmark_ingestion_large()

            # Search
            print(f"\n{Colors.BOLD}Search Benchmarks{Colors.END}")
            print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
            self.benchmark_search_fts5()
            self.benchmark_search_semantic()
            self.benchmark_search_hybrid()
            self.benchmark_search_complex_query()

            # Entity extraction
            print(f"\n{Colors.BOLD}Entity Extraction Benchmarks{Colors.END}")
            print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
            self.benchmark_entity_extraction()

            # Large datasets
            print(f"\n{Colors.BOLD}Large Dataset Benchmarks{Colors.END}")
            print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
            self.benchmark_large_dataset_search()
            self.benchmark_large_result_set()

            # Summary
            self.print_summary()
            self.save_results()

        finally:
            self.teardown()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="TDZ C64 Knowledge Base Benchmarks")
    parser.add_argument("--data-dir", help="Data directory for benchmarks")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(data_dir=args.data_dir)
    benchmark.run_all()


if __name__ == "__main__":
    main()
