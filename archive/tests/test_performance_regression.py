#!/usr/bin/env python3
"""
Performance Regression Tests

Validates that performance optimizations remain effective across releases.
These tests set performance baselines and fail if performance degrades.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Disable heavy features for performance testing
os.environ['USE_SEMANTIC_SEARCH'] = '0'
os.environ['USE_FTS5'] = '1'

from server import KnowledgeBase
from anomaly_detector import AnomalyDetector, CheckResult


class TestPerformanceRegression:
    """Performance regression tests for anomaly detection."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.test_dir, 'knowledge_base.db')
        cls.kb = KnowledgeBase(cls.test_dir)

        # Run migration
        from migration_v2_21_0 import run_migration
        run_migration(cls.db_path, dry_run=False)

        # Create test documents
        cls.test_doc_ids = [f'perf_doc_{i:04d}' for i in range(100)]
        with cls.kb._lock:
            cursor = cls.kb.db_conn.cursor()
            for doc_id in cls.test_doc_ids:
                cursor.execute("""
                    INSERT INTO documents
                    (doc_id, filename, title, filepath, file_type, total_pages,
                     total_chunks, indexed_at, tags, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, f'{doc_id}.txt', f'Performance Test {doc_id}',
                      f'/tmp/{doc_id}.txt', 'text', 0, 0,
                      datetime.now().isoformat(), '', f'http://example.com/{doc_id}'))
            cls.kb.db_conn.commit()

    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_batch_record_performance(self):
        """Test that batch recording meets performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Create 100 checks
        checks = [
            CheckResult(
                doc_id=doc_id,
                status='unchanged',
                response_time=1.0,
                http_status=200
            )
            for doc_id in self.test_doc_ids
        ]

        # Measure batch recording time
        start = time.time()
        detector.record_checks_batch(checks)
        elapsed = time.time() - start

        # Performance baseline: 100 checks in < 0.1 seconds
        assert elapsed < 0.1, f"Batch recording took {elapsed:.3f}s, baseline is 0.1s"
        print(f"\n[PASS] Batch recording: {elapsed:.3f}s ({len(checks)/elapsed:.0f} checks/sec)")

    def test_baseline_update_performance(self):
        """Test that baseline updates meet performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Update baselines for 100 documents
        start = time.time()
        detector.update_baselines_batch(self.test_doc_ids)
        elapsed = time.time() - start

        # Performance baseline: 100 baseline updates in < 0.5 seconds
        assert elapsed < 0.5, f"Baseline updates took {elapsed:.3f}s, baseline is 0.5s"
        print(f"[PASS] Baseline updates: {elapsed:.3f}s ({len(self.test_doc_ids)/elapsed:.0f} updates/sec)")

    def test_anomaly_score_calculation_performance(self):
        """Test that anomaly scoring meets performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Build some baseline data first
        initial_checks = [
            CheckResult(doc_id=doc_id, status='unchanged', response_time=1.0, http_status=200)
            for doc_id in self.test_doc_ids[:10]
        ]
        detector.record_checks_batch(initial_checks)

        # Measure scoring time
        check = CheckResult(
            doc_id=self.test_doc_ids[0],
            status='unchanged',
            response_time=1.0,
            http_status=200
        )

        start = time.time()
        for _ in range(100):
            detector.calculate_anomaly_score(self.test_doc_ids[0], check)
        elapsed = time.time() - start

        # Performance baseline: 100 scores in < 0.2 seconds
        assert elapsed < 0.2, f"Anomaly scoring took {elapsed:.3f}s, baseline is 0.2s"
        print(f"[PASS] Anomaly scoring: {elapsed:.3f}s ({100/elapsed:.0f} scores/sec)")

    def test_get_baseline_performance(self):
        """Test that baseline retrieval meets performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Measure baseline retrieval time
        start = time.time()
        for doc_id in self.test_doc_ids[:50]:
            detector.get_baseline(doc_id)
        elapsed = time.time() - start

        # Performance baseline: 50 retrievals in < 0.1 seconds
        assert elapsed < 0.1, f"Baseline retrieval took {elapsed:.3f}s, baseline is 0.1s"
        print(f"[PASS] Baseline retrieval: {elapsed:.3f}s ({50/elapsed:.0f} retrievals/sec)")

    def test_get_all_baselines_performance(self):
        """Test that bulk baseline retrieval meets performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Measure bulk retrieval time
        start = time.time()
        baselines = detector.get_all_baselines()
        elapsed = time.time() - start

        # Performance baseline: All baselines in < 0.1 seconds
        assert elapsed < 0.1, f"Bulk baseline retrieval took {elapsed:.3f}s, baseline is 0.1s"
        assert len(baselines) >= 100, f"Expected >= 100 baselines, got {len(baselines)}"
        print(f"[PASS] Bulk baseline retrieval: {elapsed:.3f}s ({len(baselines)} baselines)")

    def test_history_query_performance(self):
        """Test that history queries meet performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Add some history
        checks = [
            CheckResult(doc_id=self.test_doc_ids[0], status='unchanged',
                       response_time=1.0, http_status=200)
            for _ in range(100)
        ]
        detector.record_checks_batch(checks)

        # Measure history query time
        start = time.time()
        history = detector.get_history(self.test_doc_ids[0], days=30)
        elapsed = time.time() - start

        # Performance baseline: History query in < 0.05 seconds
        assert elapsed < 0.05, f"History query took {elapsed:.3f}s, baseline is 0.05s"
        assert len(history) >= 100, f"Expected >= 100 history entries, got {len(history)}"
        print(f"[PASS] History query: {elapsed:.3f}s ({len(history)} entries)")

    def test_full_monitoring_cycle_performance(self):
        """Test complete monitoring cycle meets performance baseline."""
        detector = AnomalyDetector(self.kb)

        # Simulate full monitoring cycle for 50 documents
        doc_subset = self.test_doc_ids[:50]

        start_total = time.time()

        # Phase 1: Record checks (batch)
        start_phase = time.time()
        checks = [
            CheckResult(doc_id=doc_id, status='unchanged', response_time=1.0, http_status=200)
            for doc_id in doc_subset
        ]
        detector.record_checks_batch(checks)
        phase1_time = time.time() - start_phase

        # Phase 2: Calculate anomaly scores
        start_phase = time.time()
        scores = []
        for doc_id, check in zip(doc_subset, checks):
            score = detector.calculate_anomaly_score(doc_id, check)
            scores.append(score)
        phase2_time = time.time() - start_phase

        total_time = time.time() - start_total

        # Performance baseline: 50 documents in < 1.0 seconds total
        assert total_time < 1.0, f"Full cycle took {total_time:.3f}s, baseline is 1.0s"

        print(f"\n[PASS] Full monitoring cycle (50 docs):")
        print(f"  Phase 1 (record batch): {phase1_time:.3f}s")
        print(f"  Phase 2 (calculate scores): {phase2_time:.3f}s")
        print(f"  Total: {total_time:.3f}s ({len(doc_subset)/total_time:.1f} docs/sec)")


def run_tests():
    """Run performance regression tests."""
    import pytest

    print("\n" + "=" * 70)
    print("PERFORMANCE REGRESSION TESTS")
    print("=" * 70)

    exit_code = pytest.main([__file__, '-v', '--tb=short', '-s'])

    print("\n" + "=" * 70)
    print("PERFORMANCE BASELINES:")
    print("  Batch recording (100 checks):     < 0.1s")
    print("  Baseline updates (100 docs):      < 0.5s")
    print("  Anomaly scoring (100 scores):     < 0.2s")
    print("  Baseline retrieval (50 queries):  < 0.1s")
    print("  Bulk baseline retrieval (100+):   < 0.1s")
    print("  History query (100+ entries):     < 0.05s")
    print("  Full monitoring cycle (50 docs):  < 1.0s")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
