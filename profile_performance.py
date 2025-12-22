#!/usr/bin/env python3
"""
Performance profiling script for TDZ C64 Knowledge Base.

Profiles key operations:
- Startup time
- Search operations (FTS5, semantic, hybrid)
- Document operations (add, retrieve, delete)
- Database queries
- Memory usage
"""

import os
import time
import tempfile
import tracemalloc
from pathlib import Path
from server import KnowledgeBase

# Use production data directory
DATA_DIR = os.path.expanduser("~/.tdz-c64-knowledge")


class PerformanceProfiler:
    """Profile KB performance."""

    def __init__(self):
        self.results = {}

    def time_operation(self, name, operation, *args, **kwargs):
        """Time a single operation."""
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        self.results[name] = elapsed
        return result, elapsed

    def profile_startup(self):
        """Profile KB initialization time."""
        print("Profiling startup time...")

        # Track memory
        tracemalloc.start()

        start = time.perf_counter()
        kb = KnowledgeBase(DATA_DIR)
        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.results['startup_time_ms'] = elapsed
        self.results['startup_memory_mb'] = peak / 1024 / 1024
        self.results['num_documents'] = len(kb.documents)

        print(f"  OK: Startup: {elapsed:.2f}ms")
        print(f"  OK: Memory: {peak/1024/1024:.2f}MB")
        print(f"  OK: Documents loaded: {len(kb.documents)}")

        return kb

    def profile_search(self, kb):
        """Profile search operations."""
        print("\nProfiling search operations...")

        queries = [
            "VIC-II sprites",
            "SID chip sound",
            "memory map",
            "BASIC programming",
            "graphics modes"
        ]

        # FTS5 Search
        fts5_times = []
        for query in queries:
            start = time.perf_counter()
            results = kb.search(query, max_results=10)
            elapsed = (time.perf_counter() - start) * 1000
            fts5_times.append(elapsed)

        avg_fts5 = sum(fts5_times) / len(fts5_times)
        self.results['avg_fts5_search_ms'] = avg_fts5
        print(f"  OK: FTS5 search: {avg_fts5:.2f}ms avg")

        # Semantic Search (if enabled)
        if kb.use_semantic:
            semantic_times = []
            for query in queries:
                start = time.perf_counter()
                results = kb.semantic_search(query, max_results=10)
                elapsed = (time.perf_counter() - start) * 1000
                semantic_times.append(elapsed)

            avg_semantic = sum(semantic_times) / len(semantic_times)
            self.results['avg_semantic_search_ms'] = avg_semantic
            print(f"  OK: Semantic search: {avg_semantic:.2f}ms avg")

            # Hybrid Search
            hybrid_times = []
            for query in queries:
                start = time.perf_counter()
                results = kb.hybrid_search(query, max_results=10)
                elapsed = (time.perf_counter() - start) * 1000
                hybrid_times.append(elapsed)

            avg_hybrid = sum(hybrid_times) / len(hybrid_times)
            self.results['avg_hybrid_search_ms'] = avg_hybrid
            print(f"  OK: Hybrid search: {avg_hybrid:.2f}ms avg")
        else:
            print(f"  WARN: Semantic search disabled")

    def profile_document_ops(self, kb):
        """Profile document operations."""
        print("\nProfiling document operations...")

        # Create test document
        test_content = "This is a test document for performance profiling. " * 100
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Add document
            start = time.perf_counter()
            doc = kb.add_document(temp_path, title="Performance Test Doc")
            add_time = (time.perf_counter() - start) * 1000
            self.results['add_document_ms'] = add_time
            print(f"  OK: Add document: {add_time:.2f}ms")

            # Retrieve chunks
            start = time.perf_counter()
            chunks = kb.get_chunks(doc.doc_id)
            retrieve_time = (time.perf_counter() - start) * 1000
            self.results['retrieve_chunks_ms'] = retrieve_time
            print(f"  OK: Retrieve chunks: {retrieve_time:.2f}ms ({len(chunks)} chunks)")

            # Delete document
            start = time.perf_counter()
            kb.remove_document(doc.doc_id)
            delete_time = (time.perf_counter() - start) * 1000
            self.results['delete_document_ms'] = delete_time
            print(f"  OK: Delete document: {delete_time:.2f}ms")

        finally:
            os.unlink(temp_path)

    def profile_database_queries(self, kb):
        """Profile common database queries."""
        print("\nProfiling database queries...")

        cursor = kb.db_conn.cursor()

        # Count queries
        queries = {
            'count_documents': "SELECT COUNT(*) FROM documents",
            'count_chunks': "SELECT COUNT(*) FROM chunks",
            'count_entities': "SELECT COUNT(DISTINCT entity_text) FROM document_entities",
            'list_all_docs': "SELECT * FROM documents",
            'get_doc_by_id': f"SELECT * FROM documents LIMIT 1"
        }

        for name, query in queries.items():
            start = time.perf_counter()
            cursor.execute(query)
            result = cursor.fetchall()
            elapsed = (time.perf_counter() - start) * 1000
            self.results[f'query_{name}_ms'] = elapsed
            print(f"  OK: {name}: {elapsed:.2f}ms")

    def profile_entity_operations(self, kb):
        """Profile entity extraction and search."""
        print("\nProfiling entity operations...")

        # Get a sample document
        if kb.documents:
            doc = kb.documents[0]

            # Check if entities already exist
            start = time.perf_counter()
            entities = kb.get_entities(doc.doc_id)
            elapsed = (time.perf_counter() - start) * 1000
            self.results['get_entities_ms'] = elapsed
            print(f"  OK: Get entities: {elapsed:.2f}ms ({len(entities)} entities)")

            # Search entities
            if entities:
                entity_text = entities[0]['entity_text']
                start = time.perf_counter()
                results = kb.search_entities(entity_text)
                elapsed = (time.perf_counter() - start) * 1000
                self.results['search_entities_ms'] = elapsed
                print(f"  OK: Search entities: {elapsed:.2f}ms")

    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        # Startup
        print(f"\nStartup:")
        print(f"  Time: {self.results.get('startup_time_ms', 0):.2f}ms")
        print(f"  Memory: {self.results.get('startup_memory_mb', 0):.2f}MB")
        print(f"  Documents: {self.results.get('num_documents', 0)}")

        # Search
        print(f"\nSearch (average):")
        print(f"  FTS5: {self.results.get('avg_fts5_search_ms', 0):.2f}ms")
        if 'avg_semantic_search_ms' in self.results:
            print(f"  Semantic: {self.results.get('avg_semantic_search_ms', 0):.2f}ms")
            print(f"  Hybrid: {self.results.get('avg_hybrid_search_ms', 0):.2f}ms")

        # Document operations
        print(f"\nDocument operations:")
        print(f"  Add: {self.results.get('add_document_ms', 0):.2f}ms")
        print(f"  Retrieve chunks: {self.results.get('retrieve_chunks_ms', 0):.2f}ms")
        print(f"  Delete: {self.results.get('delete_document_ms', 0):.2f}ms")

        # Database queries
        print(f"\nDatabase queries:")
        for key, value in sorted(self.results.items()):
            if key.startswith('query_'):
                name = key.replace('query_', '').replace('_ms', '')
                print(f"  {name}: {value:.2f}ms")

        # Entity operations
        print(f"\nEntity operations:")
        if 'get_entities_ms' in self.results:
            print(f"  Get entities: {self.results.get('get_entities_ms', 0):.2f}ms")
        if 'search_entities_ms' in self.results:
            print(f"  Search entities: {self.results.get('search_entities_ms', 0):.2f}ms")

        print("\n" + "="*60)

    def save_results(self, filename="performance_baseline.json"):
        """Save results to JSON file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


def main():
    """Run performance profiling."""
    print("="*60)
    print("TDZ C64 Knowledge Base - Performance Profiler")
    print("="*60)

    profiler = PerformanceProfiler()

    # Profile startup
    kb = profiler.profile_startup()

    # Profile search operations
    profiler.profile_search(kb)

    # Profile document operations
    profiler.profile_document_ops(kb)

    # Profile database queries
    profiler.profile_database_queries(kb)

    # Profile entity operations
    profiler.profile_entity_operations(kb)

    # Print summary
    profiler.print_summary()

    # Save results
    profiler.save_results()

    # Cleanup
    kb.close()


if __name__ == "__main__":
    main()
