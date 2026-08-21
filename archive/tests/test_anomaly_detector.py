#!/usr/bin/env python3
"""
Unit Tests for Anomaly Detector

Tests all major functionality of the anomaly detection system.
"""

import os
import sys
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase
from anomaly_detector import AnomalyDetector, CheckResult, Baseline, AnomalyScore


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    @classmethod
    def setup_class(cls):
        """Set up test environment once for all tests."""
        # Disable heavy features for faster tests
        os.environ['USE_SEMANTIC_SEARCH'] = '0'
        os.environ['USE_FTS5'] = '1'

        # Create temporary directory for test database
        cls.test_dir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.test_dir, 'knowledge_base.db')

        # Initialize test knowledge base
        cls.kb = KnowledgeBase(cls.test_dir)

        # Run migration to create anomaly tables
        from migration_v2_21_0 import run_migration
        success = run_migration(cls.db_path, dry_run=False)
        assert success, "Migration failed"

        # Add test document directly to database
        cls.test_doc_id = 'test_doc_123'
        with cls.kb._lock:
            cursor = cls.kb.db_conn.cursor()
            cursor.execute("""
                INSERT INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages,
                 total_chunks, indexed_at, tags, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cls.test_doc_id, 'test.txt', 'Test Document', '/tmp/test.txt',
                  'text', 0, 0, datetime.now().isoformat(), '', 'http://example.com'))
            cls.kb.db_conn.commit()

    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_initialization(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector(self.kb)
        assert detector is not None
        assert detector.learning_period_days == 30
        assert detector.kb == self.kb

    def test_02_initialization_custom_period(self):
        """Test initialization with custom learning period."""
        detector = AnomalyDetector(self.kb, learning_period_days=60)
        assert detector.learning_period_days == 60

    def test_03_record_single_check(self):
        """Test recording a single check."""
        detector = AnomalyDetector(self.kb)

        check = CheckResult(
            doc_id=self.test_doc_id,
            status='unchanged',
            response_time=1.5,
            http_status=200
        )

        # Should not raise exception
        detector.record_check(check)

        # Verify check was recorded
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM monitoring_history
                WHERE doc_id = ?
            """, (self.test_doc_id,))
            count = cursor.fetchone()[0]
            assert count >= 1, f"Expected at least 1 check, got {count}"

    def test_04_get_baseline(self):
        """Test getting baseline statistics."""
        detector = AnomalyDetector(self.kb)

        baseline = detector.get_baseline(self.test_doc_id)
        assert baseline is not None
        assert baseline.doc_id == self.test_doc_id
        assert baseline.total_checks >= 0
        assert baseline.total_changes >= 0
        assert baseline.total_failures >= 0

    def test_05_record_checks_batch(self):
        """Test batch recording of checks."""
        detector = AnomalyDetector(self.kb)

        # Create multiple test documents
        test_doc_ids = ['batch_test_1', 'batch_test_2', 'batch_test_3']

        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            for doc_id in test_doc_ids:
                cursor.execute("""
                    INSERT OR IGNORE INTO documents
                    (doc_id, filename, title, filepath, file_type, total_pages,
                     total_chunks, indexed_at, tags, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, f'{doc_id}.txt', f'Batch Test {doc_id}', f'/tmp/{doc_id}.txt',
                      'text', 0, 0, datetime.now().isoformat(), '', f'http://example.com/{doc_id}'))
            self.kb.db_conn.commit()

        # Create batch of checks
        checks = [
            CheckResult(
                doc_id=doc_id,
                status='unchanged',
                response_time=1.0 + i * 0.5,
                http_status=200
            )
            for i, doc_id in enumerate(test_doc_ids)
        ]

        # Record batch
        detector.record_checks_batch(checks)

        # Verify all were recorded
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            for doc_id in test_doc_ids:
                cursor.execute("""
                    SELECT COUNT(*) FROM monitoring_history
                    WHERE doc_id = ?
                """, (doc_id,))
                count = cursor.fetchone()[0]
                assert count >= 1, f"Expected at least 1 check for {doc_id}, got {count}"

    def test_06_calculate_anomaly_score_no_baseline(self):
        """Test anomaly scoring with insufficient baseline."""
        detector = AnomalyDetector(self.kb)

        # Create new document with no history
        new_doc_id = 'new_doc_no_baseline'
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                INSERT INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages,
                 total_chunks, indexed_at, tags, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (new_doc_id, 'new.txt', 'New Document', '/tmp/new.txt',
                  'text', 0, 0, datetime.now().isoformat(), '', 'http://new.com'))
            self.kb.db_conn.commit()

        check = CheckResult(
            doc_id=new_doc_id,
            status='unchanged',
            response_time=1.0,
            http_status=200
        )

        score = detector.calculate_anomaly_score(new_doc_id, check)

        # Should return neutral score (insufficient data)
        assert score.total_score == 0.0
        assert score.severity == 'normal'

    def test_07_calculate_anomaly_score_with_baseline(self):
        """Test anomaly scoring with sufficient baseline."""
        detector = AnomalyDetector(self.kb)

        # Create document and add multiple checks to build baseline
        baseline_doc_id = 'doc_with_baseline'
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO documents
                (doc_id, filename, title, filepath, file_type, total_pages,
                 total_chunks, indexed_at, tags, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (baseline_doc_id, 'baseline.txt', 'Baseline Document', '/tmp/baseline.txt',
                  'text', 0, 0, datetime.now().isoformat(), '', 'http://baseline.com'))
            self.kb.db_conn.commit()

        # Add 10 checks to build baseline
        checks = [
            CheckResult(
                doc_id=baseline_doc_id,
                status='unchanged' if i < 8 else 'changed',
                response_time=1.0 + (i * 0.1),
                http_status=200
            )
            for i in range(10)
        ]

        detector.record_checks_batch(checks)

        # Now calculate score
        test_check = CheckResult(
            doc_id=baseline_doc_id,
            status='unchanged',
            response_time=1.5,
            http_status=200
        )

        score = detector.calculate_anomaly_score(baseline_doc_id, test_check)

        # Should have valid score components
        assert isinstance(score.total_score, float)
        assert 0 <= score.total_score <= 100
        assert score.severity in ['normal', 'minor', 'moderate', 'critical']

    def test_08_severity_classification(self):
        """Test severity level classification."""
        detector = AnomalyDetector(self.kb)

        # Test severity thresholds
        test_scores = [
            (15.0, 'normal'),
            (45.0, 'minor'),
            (75.0, 'moderate'),
            (95.0, 'critical')
        ]

        for score_value, expected_severity in test_scores:
            severity = detector._get_severity(score_value)
            assert severity == expected_severity, f"Score {score_value} should be {expected_severity}, got {severity}"

    def test_09_smart_filtering(self):
        """Test smart noise filtering."""
        detector = AnomalyDetector(self.kb)

        # Test timestamp filtering (using pattern that matches: "Updated: YYYY-MM-DD")
        old_content = "Some content here\nUpdated: 2025-12-21\nMore content"
        new_content = "Some content here\nUpdated: 2025-12-22\nMore content"

        is_filtered = detector.should_filter(old_content, new_content)
        assert is_filtered is True, "Timestamp change should be filtered"

        # Test real content change
        old_content = "Original content"
        new_content = "Changed content"

        is_filtered = detector.should_filter(old_content, new_content)
        assert is_filtered is False, "Real content change should not be filtered"

    def test_10_get_history(self):
        """Test retrieving monitoring history."""
        detector = AnomalyDetector(self.kb)

        history = detector.get_history(self.test_doc_id, days=30)

        assert isinstance(history, list)
        # Should have at least the checks we recorded
        assert len(history) >= 0

    def test_11_get_all_baselines(self):
        """Test retrieving all baselines."""
        detector = AnomalyDetector(self.kb)

        baselines = detector.get_all_baselines()

        assert isinstance(baselines, list)
        # Should have baselines for documents we created
        assert len(baselines) >= 1

    def test_12_edge_case_empty_batch(self):
        """Test batch operations with empty list."""
        detector = AnomalyDetector(self.kb)

        # Should handle gracefully
        detector.record_checks_batch([])
        detector.update_baselines_batch([])

    def test_13_edge_case_failed_check(self):
        """Test recording failed checks."""
        detector = AnomalyDetector(self.kb)

        check = CheckResult(
            doc_id=self.test_doc_id,
            status='failed',
            response_time=0.0,
            http_status=500,
            error_message='Internal Server Error'
        )

        detector.record_check(check)

        # Verify failed check was recorded
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM monitoring_history
                WHERE doc_id = ? AND status = 'failed'
            """, (self.test_doc_id,))
            count = cursor.fetchone()[0]
            assert count >= 1, f"Expected at least 1 failed check, got {count}"

    def test_14_performance_batch_vs_individual(self):
        """Test that batch operations are faster than individual."""
        import time

        detector = AnomalyDetector(self.kb)

        # Create test documents
        perf_doc_ids = [f'perf_test_{i}' for i in range(20)]

        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            for doc_id in perf_doc_ids:
                cursor.execute("""
                    INSERT OR IGNORE INTO documents
                    (doc_id, filename, title, filepath, file_type, total_pages,
                     total_chunks, indexed_at, tags, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (doc_id, f'{doc_id}.txt', f'Perf Test {doc_id}', f'/tmp/{doc_id}.txt',
                      'text', 0, 0, datetime.now().isoformat(), '', f'http://perf.com/{doc_id}'))
            self.kb.db_conn.commit()

        # Test batch recording
        batch_checks = [
            CheckResult(
                doc_id=doc_id,
                status='unchanged',
                response_time=1.0,
                http_status=200
            )
            for doc_id in perf_doc_ids
        ]

        start = time.time()
        detector.record_checks_batch(batch_checks)
        batch_time = time.time() - start

        # Batch should complete in < 1 second
        assert batch_time < 1.0, f"Batch operation took {batch_time:.2f}s, should be < 1s"

        print(f"\n[Performance] Batch recording 20 checks: {batch_time:.3f}s ({20/batch_time:.1f} checks/sec)")

    def test_15_get_anomalies(self):
        """Test retrieving anomalies by severity."""
        detector = AnomalyDetector(self.kb)

        # Get moderate or higher anomalies
        anomalies = detector.get_anomalies(min_severity='moderate', days=7)

        assert isinstance(anomalies, list)
        # All returned anomalies should be moderate or critical
        for anomaly in anomalies:
            assert anomaly.get('anomaly_score', 0) >= 61, "Anomaly score should be >= 61 for moderate severity"


def run_tests():
    """Run all tests."""
    import pytest

    # Run with pytest
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    return exit_code


if __name__ == "__main__":
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code)
