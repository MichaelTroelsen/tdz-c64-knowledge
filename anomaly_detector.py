#!/usr/bin/env python3
"""
Anomaly Detection for URL Monitoring

Tracks monitoring history, learns baseline patterns, and scores deviations.

Key Features:
- Historical tracking of all monitoring checks
- Baseline learning (30-day window)
- Anomaly scoring based on frequency, magnitude, and performance
- Smart filtering to suppress noise (timestamps, counters, ads)
- Severity classification (Normal, Minor, Moderate, Critical)

Usage:
    from anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(kb)
    detector.record_check(doc_id, status, response_time=1.2)
    score = detector.calculate_anomaly_score(doc_id, current_check)
"""

import re
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a monitoring check."""
    doc_id: str
    status: str  # 'unchanged', 'changed', 'failed'
    change_type: Optional[str] = None  # 'content', 'structure', 'metadata'
    response_time: Optional[float] = None  # Seconds
    content_hash: Optional[str] = None
    http_status: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class Baseline:
    """Baseline statistics for a document."""
    doc_id: str
    avg_update_interval_hours: float
    avg_response_time_ms: float
    avg_change_magnitude: float
    total_checks: int
    total_changes: int
    total_failures: int
    last_updated: str


@dataclass
class AnomalyScore:
    """Anomaly score with components."""
    total_score: float  # 0-100
    frequency_score: float  # 0-100
    magnitude_score: float  # 0-100
    performance_score: float  # 0-100
    severity: str  # 'normal', 'minor', 'moderate', 'critical'
    components: Dict[str, float]  # Detailed breakdown


class AnomalyDetector:
    """Anomaly detection for URL monitoring."""

    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        'normal': (0, 30),
        'minor': (31, 60),
        'moderate': (61, 85),
        'critical': (86, 100)
    }

    # Component weights
    WEIGHTS = {
        'frequency': 0.4,
        'magnitude': 0.4,
        'performance': 0.2
    }

    def __init__(self, kb, learning_period_days: int = 30):
        """Initialize anomaly detector.

        Args:
            kb: KnowledgeBase instance
            learning_period_days: Days of history to use for baseline (default: 30)
        """
        self.kb = kb
        self.learning_period_days = learning_period_days
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        """Verify anomaly detection tables exist."""
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='monitoring_history'
            """)
            if not cursor.fetchone():
                raise RuntimeError(
                    "Anomaly detection tables not found. "
                    "Run migration_v2_21_0.py first."
                )

    def record_check(self, check: CheckResult):
        """Record a monitoring check result.

        Args:
            check: CheckResult instance
        """
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                INSERT INTO monitoring_history
                (doc_id, check_date, status, change_type, response_time,
                 content_hash, http_status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                check.doc_id,
                datetime.now().isoformat(),
                check.status,
                check.change_type,
                check.response_time,
                check.content_hash,
                check.http_status,
                check.error_message
            ))
            self.kb.db_conn.commit()

            # Update baseline
            self._update_baseline(check.doc_id)

    def _update_baseline(self, doc_id: str):
        """Update baseline statistics for a document.

        Args:
            doc_id: Document ID
        """
        cutoff_date = (datetime.now() - timedelta(days=self.learning_period_days)).isoformat()

        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()

            # Get historical data
            cursor.execute("""
                SELECT
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN status = 'changed' THEN 1 ELSE 0 END) as total_changes,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as total_failures,
                    AVG(response_time) as avg_response_time
                FROM monitoring_history
                WHERE doc_id = ? AND check_date >= ?
            """, (doc_id, cutoff_date))

            row = cursor.fetchone()
            total_checks = row[0] or 0
            total_changes = row[1] or 0
            total_failures = row[2] or 0
            avg_response_time = row[3] or 0.0

            # Calculate average update interval
            cursor.execute("""
                SELECT check_date
                FROM monitoring_history
                WHERE doc_id = ? AND status = 'changed' AND check_date >= ?
                ORDER BY check_date ASC
            """, (doc_id, cutoff_date))

            change_dates = [datetime.fromisoformat(row[0]) for row in cursor.fetchall()]
            if len(change_dates) >= 2:
                intervals = [
                    (change_dates[i+1] - change_dates[i]).total_seconds() / 3600
                    for i in range(len(change_dates) - 1)
                ]
                avg_update_interval = sum(intervals) / len(intervals)
            else:
                avg_update_interval = 0.0

            # Calculate average change magnitude (placeholder for now)
            avg_change_magnitude = 0.0

            # Update or insert baseline
            cursor.execute("""
                INSERT OR REPLACE INTO anomaly_baselines
                (doc_id, avg_update_interval_hours, avg_response_time_ms,
                 avg_change_magnitude, total_checks, total_changes, total_failures,
                 last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                avg_update_interval,
                avg_response_time * 1000,  # Convert to ms
                avg_change_magnitude,
                total_checks,
                total_changes,
                total_failures,
                datetime.now().isoformat()
            ))

            self.kb.db_conn.commit()

    def get_baseline(self, doc_id: str) -> Optional[Baseline]:
        """Get baseline statistics for a document.

        Args:
            doc_id: Document ID

        Returns:
            Baseline instance or None if not found
        """
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT
                    doc_id, avg_update_interval_hours, avg_response_time_ms,
                    avg_change_magnitude, total_checks, total_changes,
                    total_failures, last_updated
                FROM anomaly_baselines
                WHERE doc_id = ?
            """, (doc_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return Baseline(*row)

    def calculate_anomaly_score(self, doc_id: str, check: CheckResult) -> AnomalyScore:
        """Calculate anomaly score for a check result.

        Args:
            doc_id: Document ID
            check: CheckResult instance

        Returns:
            AnomalyScore instance
        """
        baseline = self.get_baseline(doc_id)

        # No baseline yet - return neutral score
        if not baseline or baseline.total_checks < 5:
            return AnomalyScore(
                total_score=0.0,
                frequency_score=0.0,
                magnitude_score=0.0,
                performance_score=0.0,
                severity='normal',
                components={}
            )

        # Calculate component scores
        frequency_score = self._calculate_frequency_score(doc_id, baseline)
        magnitude_score = self._calculate_magnitude_score(doc_id, check, baseline)
        performance_score = self._calculate_performance_score(check, baseline)

        # Weighted total
        total_score = (
            frequency_score * self.WEIGHTS['frequency'] +
            magnitude_score * self.WEIGHTS['magnitude'] +
            performance_score * self.WEIGHTS['performance']
        )

        # Determine severity
        severity = self._get_severity(total_score)

        return AnomalyScore(
            total_score=total_score,
            frequency_score=frequency_score,
            magnitude_score=magnitude_score,
            performance_score=performance_score,
            severity=severity,
            components={
                'frequency': frequency_score,
                'magnitude': magnitude_score,
                'performance': performance_score,
                'baseline_checks': baseline.total_checks,
                'baseline_changes': baseline.total_changes
            }
        )

    def _calculate_frequency_score(self, doc_id: str, baseline: Baseline) -> float:
        """Calculate frequency anomaly score.

        Detects when a site updates much more or less frequently than usual.

        Args:
            doc_id: Document ID
            baseline: Baseline statistics

        Returns:
            Score 0-100
        """
        if baseline.total_changes < 2:
            return 0.0

        # Get time since last change
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT check_date
                FROM monitoring_history
                WHERE doc_id = ? AND status = 'changed'
                ORDER BY check_date DESC
                LIMIT 1
            """, (doc_id,))

            row = cursor.fetchone()
            if not row:
                return 0.0

            last_change = datetime.fromisoformat(row[0])
            hours_since_change = (datetime.now() - last_change).total_seconds() / 3600

        # Compare to baseline
        if baseline.avg_update_interval_hours == 0:
            return 0.0

        deviation = abs(hours_since_change - baseline.avg_update_interval_hours)
        relative_deviation = deviation / baseline.avg_update_interval_hours

        # Score: 0-100 based on relative deviation
        # 0% deviation = 0 score
        # 100% deviation = 50 score
        # 200%+ deviation = 100 score
        score = min(100, relative_deviation * 50)

        return score

    def _calculate_magnitude_score(self, doc_id: str, check: CheckResult, baseline: Baseline) -> float:
        """Calculate change magnitude anomaly score.

        Detects when changes are unusually large or small.

        Args:
            doc_id: Document ID
            check: Current check result
            baseline: Baseline statistics

        Returns:
            Score 0-100
        """
        # For now, return 0 - will implement when we have content diff capability
        # Future: Compare content hashes, calculate edit distance
        return 0.0

    def _calculate_performance_score(self, check: CheckResult, baseline: Baseline) -> float:
        """Calculate performance anomaly score.

        Detects when response times degrade significantly.

        Args:
            check: Current check result
            baseline: Baseline statistics

        Returns:
            Score 0-100
        """
        if not check.response_time or baseline.avg_response_time_ms == 0:
            return 0.0

        current_ms = check.response_time * 1000
        avg_ms = baseline.avg_response_time_ms

        # Calculate relative deviation
        deviation = abs(current_ms - avg_ms)
        relative_deviation = deviation / avg_ms

        # Score: 0-100 based on relative deviation
        # 0-50% deviation = 0-25 score
        # 50-100% deviation = 25-50 score
        # 100-200% deviation = 50-100 score
        # 200%+ deviation = 100 score
        if relative_deviation <= 0.5:
            score = relative_deviation * 50
        elif relative_deviation <= 1.0:
            score = 25 + (relative_deviation - 0.5) * 50
        elif relative_deviation <= 2.0:
            score = 50 + (relative_deviation - 1.0) * 50
        else:
            score = 100

        return min(100, score)

    def _get_severity(self, score: float) -> str:
        """Get severity level from score.

        Args:
            score: Anomaly score 0-100

        Returns:
            Severity string ('normal', 'minor', 'moderate', 'critical')
        """
        for severity, (low, high) in self.SEVERITY_THRESHOLDS.items():
            if low <= score <= high:
                return severity
        return 'critical'

    def should_filter(self, old_content: str, new_content: str) -> bool:
        """Check if change should be filtered as noise.

        Args:
            old_content: Previous content
            new_content: New content

        Returns:
            True if change is noise and should be filtered
        """
        if not old_content or not new_content:
            return False

        # Get ignore patterns
        patterns = self._get_ignore_patterns()

        # Remove ignored patterns from both versions
        old_cleaned = old_content
        new_cleaned = new_content

        for pattern_regex in patterns:
            try:
                old_cleaned = re.sub(pattern_regex, '', old_cleaned, flags=re.DOTALL)
                new_cleaned = re.sub(pattern_regex, '', new_cleaned, flags=re.DOTALL)
            except re.error:
                # Skip invalid regex
                continue

        # If content is identical after filtering, it's noise
        return old_cleaned.strip() == new_cleaned.strip()

    def _get_ignore_patterns(self) -> List[str]:
        """Get enabled ignore patterns from database.

        Returns:
            List of regex patterns
        """
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT pattern_regex
                FROM anomaly_patterns
                WHERE enabled = 1
            """)
            return [row[0] for row in cursor.fetchall()]

    def get_history(self, doc_id: str, days: int = 30) -> List[Dict]:
        """Get monitoring history for a document.

        Args:
            doc_id: Document ID
            days: Number of days to retrieve (default: 30)

        Returns:
            List of history records
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT
                    id, doc_id, check_date, status, change_type,
                    response_time, content_hash, anomaly_score,
                    http_status, error_message
                FROM monitoring_history
                WHERE doc_id = ? AND check_date >= ?
                ORDER BY check_date DESC
            """, (doc_id, cutoff_date))

            columns = ['id', 'doc_id', 'check_date', 'status', 'change_type',
                      'response_time', 'content_hash', 'anomaly_score',
                      'http_status', 'error_message']

            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_all_baselines(self) -> List[Baseline]:
        """Get all baselines.

        Returns:
            List of Baseline instances
        """
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT
                    doc_id, avg_update_interval_hours, avg_response_time_ms,
                    avg_change_magnitude, total_checks, total_changes,
                    total_failures, last_updated
                FROM anomaly_baselines
                ORDER BY last_updated DESC
            """)

            return [Baseline(*row) for row in cursor.fetchall()]

    def update_anomaly_score(self, doc_id: str, check: CheckResult):
        """Calculate and store anomaly score for a check.

        Args:
            doc_id: Document ID
            check: CheckResult instance
        """
        score = self.calculate_anomaly_score(doc_id, check)

        # Update the most recent history record with the score
        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                UPDATE monitoring_history
                SET anomaly_score = ?
                WHERE doc_id = ?
                  AND check_date = (
                      SELECT MAX(check_date)
                      FROM monitoring_history
                      WHERE doc_id = ?
                  )
            """, (score.total_score, doc_id, doc_id))
            self.kb.db_conn.commit()

    def get_anomalies(self, min_severity: str = 'moderate', days: int = 7) -> List[Dict]:
        """Get recent anomalies above a severity threshold.

        Args:
            min_severity: Minimum severity ('minor', 'moderate', 'critical')
            days: Number of days to check (default: 7)

        Returns:
            List of anomaly records with document info
        """
        # Convert severity to minimum score
        severity_mins = {
            'minor': 31,
            'moderate': 61,
            'critical': 86
        }
        min_score = severity_mins.get(min_severity, 61)

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with self.kb._lock:
            cursor = self.kb.db_conn.cursor()
            cursor.execute("""
                SELECT
                    h.doc_id,
                    d.title,
                    d.source_url,
                    h.check_date,
                    h.status,
                    h.anomaly_score,
                    h.response_time,
                    h.error_message
                FROM monitoring_history h
                JOIN documents d ON h.doc_id = d.doc_id
                WHERE h.check_date >= ?
                  AND h.anomaly_score >= ?
                ORDER BY h.anomaly_score DESC, h.check_date DESC
            """, (cutoff_date, min_score))

            columns = ['doc_id', 'title', 'url', 'check_date', 'status',
                      'anomaly_score', 'response_time', 'error_message']

            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def calculate_content_hash(content: str) -> str:
    """Calculate hash of content for change detection.

    Args:
        content: Content string

    Returns:
        MD5 hash hex string
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()
