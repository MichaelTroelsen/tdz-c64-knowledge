#!/usr/bin/env python3
"""
Load Testing Script for Anomaly Detection

Tests system performance with large-scale document monitoring (1000+ documents).
Validates that batch optimizations scale effectively.
"""

import os
import sys
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Disable heavy features
os.environ['USE_SEMANTIC_SEARCH'] = '0'
os.environ['USE_FTS5'] = '1'

from server import KnowledgeBase
from anomaly_detector import AnomalyDetector, CheckResult


class LoadTester:
    """Load testing for anomaly detection system."""

    def __init__(self, num_docs=1000, verbose=True):
        """Initialize load tester.

        Args:
            num_docs: Number of documents to test with
            verbose: Print detailed progress
        """
        self.num_docs = num_docs
        self.verbose = verbose
        self.test_dir = None
        self.kb = None
        self.detector = None
        self.doc_ids = []

    def setup(self):
        """Set up test environment."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"LOAD TEST: {self.num_docs} Documents")
            print(f"{'='*70}\n")

        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'knowledge_base.db')

        if self.verbose:
            print(f"[*] Creating test database at {self.test_dir}")

        self.kb = KnowledgeBase(self.test_dir)

        # Run migration
        from migration_v2_21_0 import run_migration
        run_migration(self.db_path, dry_run=False)

        self.detector = AnomalyDetector(self.kb)

    def teardown(self):
        """Clean up test environment."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_documents(self):
        """Create test documents in database."""
        if self.verbose:
            print(f"\n[Phase 1] Creating {self.num_docs} documents...")

        start = time.time()

        self.doc_ids = [f'load_test_{i:06d}' for i in range(self.num_docs)]

        # Batch insert documents
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            doc_data = [
                (doc_id, f'{doc_id}.txt', f'Load Test Document {doc_id}',
                 f'/tmp/{doc_id}.txt', 'text', 0, 0,
                 datetime.now().isoformat(), '', f'http://example.com/{doc_id}')
                for doc_id in self.doc_ids
            ]

            cursor.executemany("""
                INSERT INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages,
                 total_chunks, indexed_at, tags, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, doc_data)

            self.kb.db_conn.commit()

        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Created {self.num_docs} documents in {elapsed:.2f}s")
            print(f"     Throughput: {self.num_docs/elapsed:.0f} docs/sec")

        return elapsed

    def test_batch_recording(self, num_checks=1000):
        """Test batch check recording performance.

        Args:
            num_checks: Number of checks to record
        """
        if self.verbose:
            print(f"\n[Phase 2] Batch recording {num_checks} checks...")

        # Create checks for subset of documents
        doc_subset = self.doc_ids[:num_checks]
        checks = [
            CheckResult(
                doc_id=doc_id,
                status='unchanged',
                response_time=1.0 + (i % 10) * 0.1,
                http_status=200
            )
            for i, doc_id in enumerate(doc_subset)
        ]

        start = time.time()
        self.detector.record_checks_batch(checks)
        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Recorded {num_checks} checks in {elapsed:.3f}s")
            print(f"     Throughput: {num_checks/elapsed:.0f} checks/sec")

        # Performance target: > 1000 checks/sec
        checks_per_sec = num_checks / elapsed
        status = "PASS" if checks_per_sec > 1000 else "WARN"
        if self.verbose:
            print(f"     [{status}] Target: >1000 checks/sec")

        return elapsed, checks_per_sec

    def test_baseline_updates(self, num_docs=1000):
        """Test baseline update performance.

        Args:
            num_docs: Number of baselines to update
        """
        if self.verbose:
            print(f"\n[Phase 3] Updating {num_docs} baselines...")

        doc_subset = self.doc_ids[:num_docs]

        start = time.time()
        self.detector.update_baselines_batch(doc_subset)
        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Updated {num_docs} baselines in {elapsed:.3f}s")
            print(f"     Throughput: {num_docs/elapsed:.0f} updates/sec")

        # Performance target: > 200 updates/sec
        updates_per_sec = num_docs / elapsed
        status = "PASS" if updates_per_sec > 200 else "WARN"
        if self.verbose:
            print(f"     [{status}] Target: >200 updates/sec")

        return elapsed, updates_per_sec

    def test_score_calculation(self, num_scores=1000):
        """Test anomaly score calculation performance.

        Args:
            num_scores: Number of scores to calculate
        """
        if self.verbose:
            print(f"\n[Phase 4] Calculating {num_scores} anomaly scores...")

        # Build some baseline data first
        subset = self.doc_ids[:num_scores]
        checks = [
            CheckResult(doc_id=doc_id, status='unchanged',
                       response_time=1.0, http_status=200)
            for doc_id in subset
        ]
        self.detector.record_checks_batch(checks)

        # Now calculate scores
        test_check = CheckResult(
            doc_id=subset[0],
            status='unchanged',
            response_time=1.0,
            http_status=200
        )

        start = time.time()
        scores = []
        for doc_id in subset:
            score = self.detector.calculate_anomaly_score(doc_id, test_check)
            scores.append(score)
        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Calculated {num_scores} scores in {elapsed:.3f}s")
            print(f"     Throughput: {num_scores/elapsed:.0f} scores/sec")

        # Performance target: > 500 scores/sec
        scores_per_sec = num_scores / elapsed
        status = "PASS" if scores_per_sec > 500 else "WARN"
        if self.verbose:
            print(f"     [{status}] Target: >500 scores/sec")

        return elapsed, scores_per_sec

    def test_full_monitoring_cycle(self, num_docs=1000):
        """Test complete monitoring cycle.

        Args:
            num_docs: Number of documents to monitor
        """
        if self.verbose:
            print(f"\n[Phase 5] Full monitoring cycle ({num_docs} documents)...")

        doc_subset = self.doc_ids[:num_docs]

        start_total = time.time()

        # Step 1: Record checks (batch)
        start = time.time()
        checks = [
            CheckResult(doc_id=doc_id, status='unchanged',
                       response_time=1.0 + (i % 10) * 0.05, http_status=200)
            for i, doc_id in enumerate(doc_subset)
        ]
        self.detector.record_checks_batch(checks)
        record_time = time.time() - start

        # Step 2: Calculate scores
        start = time.time()
        scores = []
        for doc_id, check in zip(doc_subset, checks):
            score = self.detector.calculate_anomaly_score(doc_id, check)
            scores.append(score)
        score_time = time.time() - start

        total_time = time.time() - start_total
        throughput = num_docs / total_time

        if self.verbose:
            print(f"[OK] Completed monitoring cycle in {total_time:.3f}s")
            print(f"     Record checks: {record_time:.3f}s")
            print(f"     Calculate scores: {score_time:.3f}s")
            print(f"     Throughput: {throughput:.0f} docs/sec")

        # Performance target: > 50 docs/sec for full cycle
        status = "PASS" if throughput > 50 else "WARN"
        if self.verbose:
            print(f"     [{status}] Target: >50 docs/sec full cycle")

        return total_time, throughput

    def test_query_performance(self):
        """Test query performance on large dataset."""
        if self.verbose:
            print(f"\n[Phase 6] Query performance tests...")

        # Test 1: Get all baselines
        start = time.time()
        baselines = self.detector.get_all_baselines()
        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Retrieved {len(baselines)} baselines in {elapsed:.3f}s")

        # Test 2: Query history for specific document
        if self.doc_ids:
            start = time.time()
            history = self.detector.get_history(self.doc_ids[0], days=30)
            elapsed = time.time() - start

            if self.verbose:
                print(f"[OK] Retrieved history in {elapsed:.3f}s")

        # Test 3: Query anomalies
        start = time.time()
        anomalies = self.detector.get_anomalies(min_severity='moderate', days=7)
        elapsed = time.time() - start

        if self.verbose:
            print(f"[OK] Retrieved anomalies in {elapsed:.3f}s")

    def run_all_tests(self):
        """Run complete load test suite."""
        try:
            self.setup()

            results = {}

            # Phase 1: Document creation
            results['document_creation'] = self.create_documents()

            # Phase 2: Batch recording
            elapsed, throughput = self.test_batch_recording(min(1000, self.num_docs))
            results['batch_recording'] = {'time': elapsed, 'throughput': throughput}

            # Phase 3: Baseline updates
            elapsed, throughput = self.test_baseline_updates(min(1000, self.num_docs))
            results['baseline_updates'] = {'time': elapsed, 'throughput': throughput}

            # Phase 4: Score calculation
            elapsed, throughput = self.test_score_calculation(min(1000, self.num_docs))
            results['score_calculation'] = {'time': elapsed, 'throughput': throughput}

            # Phase 5: Full cycle
            elapsed, throughput = self.test_full_monitoring_cycle(min(1000, self.num_docs))
            results['full_cycle'] = {'time': elapsed, 'throughput': throughput}

            # Phase 6: Queries
            self.test_query_performance()

            # Summary
            if self.verbose:
                self.print_summary(results)

            return results

        finally:
            self.teardown()

    def print_summary(self, results):
        """Print load test summary."""
        print(f"\n{'='*70}")
        print("LOAD TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Documents tested: {self.num_docs}")
        print(f"\nPerformance Results:")
        print(f"  Batch recording:     {results['batch_recording']['throughput']:.0f} checks/sec")
        print(f"  Baseline updates:    {results['baseline_updates']['throughput']:.0f} updates/sec")
        print(f"  Score calculation:   {results['score_calculation']['throughput']:.0f} scores/sec")
        print(f"  Full cycle:          {results['full_cycle']['throughput']:.0f} docs/sec")
        print(f"\nPerformance Targets:")
        print(f"  Batch recording:     >1000 checks/sec")
        print(f"  Baseline updates:    >200 updates/sec")
        print(f"  Score calculation:   >500 scores/sec")
        print(f"  Full cycle:          >50 docs/sec")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Load test anomaly detection system')
    parser.add_argument('--docs', type=int, default=1000,
                       help='Number of documents to test (default: 1000)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    args = parser.parse_args()

    tester = LoadTester(num_docs=args.docs, verbose=not args.quiet)
    results = tester.run_all_tests()

    # Exit code based on performance targets
    batch_ok = results['batch_recording']['throughput'] > 1000
    baseline_ok = results['baseline_updates']['throughput'] > 200
    score_ok = results['score_calculation']['throughput'] > 500
    cycle_ok = results['full_cycle']['throughput'] > 50

    if all([batch_ok, baseline_ok, score_ok, cycle_ok]):
        print("[PASS] All performance targets met")
        return 0
    else:
        print("[WARN] Some performance targets not met")
        return 1


if __name__ == "__main__":
    sys.exit(main())
