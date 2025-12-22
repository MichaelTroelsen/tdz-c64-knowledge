#!/usr/bin/env python3
"""
Database Migration for v2.21.0: Anomaly Detection

Adds tables for historical monitoring data and anomaly detection.

Usage:
    python migration_v2_21_0.py [--data-dir PATH] [--dry-run]

Options:
    --data-dir PATH    Database directory (default: ~/.tdz-c64-knowledge)
    --dry-run          Show SQL without executing
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

# SQL statements for migration
MIGRATION_SQL = {
    'monitoring_history': """
        CREATE TABLE IF NOT EXISTS monitoring_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            check_date TEXT NOT NULL,
            status TEXT NOT NULL,
            change_type TEXT,
            response_time REAL,
            content_hash TEXT,
            anomaly_score REAL,
            http_status INTEGER,
            error_message TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    """,

    'idx_history_doc': """
        CREATE INDEX IF NOT EXISTS idx_history_doc
        ON monitoring_history(doc_id)
    """,

    'idx_history_date': """
        CREATE INDEX IF NOT EXISTS idx_history_date
        ON monitoring_history(check_date)
    """,

    'idx_history_status': """
        CREATE INDEX IF NOT EXISTS idx_history_status
        ON monitoring_history(status)
    """,

    'idx_history_doc_date': """
        CREATE INDEX IF NOT EXISTS idx_history_doc_date
        ON monitoring_history(doc_id, check_date DESC)
    """,

    'anomaly_patterns': """
        CREATE TABLE IF NOT EXISTS anomaly_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            pattern_regex TEXT NOT NULL,
            description TEXT,
            enabled INTEGER DEFAULT 1,
            created_date TEXT NOT NULL
        )
    """,

    'idx_patterns_type': """
        CREATE INDEX IF NOT EXISTS idx_patterns_type
        ON anomaly_patterns(pattern_type)
    """,

    'anomaly_baselines': """
        CREATE TABLE IF NOT EXISTS anomaly_baselines (
            doc_id TEXT PRIMARY KEY,
            avg_update_interval_hours REAL,
            avg_response_time_ms REAL,
            avg_change_magnitude REAL,
            total_checks INTEGER DEFAULT 0,
            total_changes INTEGER DEFAULT 0,
            total_failures INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        )
    """
}

# Default ignore patterns
DEFAULT_PATTERNS = [
    ('timestamp', r'Updated:\s*\d{4}-\d{2}-\d{2}', 'Timestamp in "Updated: YYYY-MM-DD" format'),
    ('timestamp', r'\d{1,2}/\d{1,2}/\d{4}', 'Date in MM/DD/YYYY format'),
    ('timestamp', r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 'ISO timestamp'),
    ('counter', r'Views?:\s*\d+', 'View counter'),
    ('counter', r'\d+\s*views?', 'View counter (alternate)'),
    ('counter', r'Visitors?:\s*\d+', 'Visitor counter'),
    ('ad', r'<div[^>]*class=["\']ad["\'][^>]*>.*?</div>', 'Advertisement div'),
    ('ad', r'<ins[^>]*class=["\']adsbygoogle["\'][^>]*>.*?</ins>', 'Google AdSense'),
    ('tracking', r'_ga=[\w.-]+', 'Google Analytics parameter'),
    ('tracking', r'utm_[\w]+=[^&]+', 'UTM tracking parameters'),
]


def check_migration_needed(db_path: str) -> tuple[bool, list[str]]:
    """Check if migration is needed.

    Args:
        db_path: Path to database file

    Returns:
        Tuple of (needs_migration, missing_tables)
    """
    if not os.path.exists(db_path):
        return False, []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check which tables exist
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN (
            'monitoring_history',
            'anomaly_patterns',
            'anomaly_baselines'
        )
    """)
    existing_tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    expected_tables = {'monitoring_history', 'anomaly_patterns', 'anomaly_baselines'}
    missing_tables = expected_tables - existing_tables

    return len(missing_tables) > 0, list(missing_tables)


def run_migration(db_path: str, dry_run: bool = False) -> bool:
    """Run database migration.

    Args:
        db_path: Path to database file
        dry_run: If True, print SQL without executing

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return False

    print(f"\n{'='*60}")
    print(f"DATABASE MIGRATION: v2.21.0")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    # Check if migration needed
    needs_migration, missing_tables = check_migration_needed(db_path)

    if not needs_migration:
        print("[OK] Database is already up to date. No migration needed.")
        return True

    print(f"[*] Missing tables: {', '.join(missing_tables)}")
    print(f"[*] Starting migration...\n")

    if dry_run:
        print("-- SQL Statements (DRY RUN):\n")
        for name, sql in MIGRATION_SQL.items():
            print(f"-- {name}")
            print(sql.strip())
            print()
        print("-- Default patterns would be inserted into anomaly_patterns")
        return True

    # Execute migration
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables and indexes
        print("[*] Creating tables and indexes...")
        for name, sql in MIGRATION_SQL.items():
            print(f"    - {name}")
            cursor.execute(sql)

        # Insert default patterns
        print(f"[*] Inserting {len(DEFAULT_PATTERNS)} default ignore patterns...")
        current_date = datetime.now().isoformat()
        cursor.executemany(
            """
            INSERT INTO anomaly_patterns
            (pattern_type, pattern_regex, description, enabled, created_date)
            VALUES (?, ?, ?, 1, ?)
            """,
            [(ptype, regex, desc, current_date) for ptype, regex, desc in DEFAULT_PATTERNS]
        )

        # Initialize baselines for existing URL-sourced documents
        print("[*] Initializing baselines for existing documents...")
        cursor.execute("""
            INSERT OR IGNORE INTO anomaly_baselines
            (doc_id, avg_update_interval_hours, avg_response_time_ms,
             avg_change_magnitude, total_checks, total_changes, total_failures, last_updated)
            SELECT
                doc_id,
                0.0,  -- Will be calculated during checks
                0.0,
                0.0,
                0,
                0,
                0,
                ?
            FROM documents
            WHERE source_url IS NOT NULL
        """, (current_date,))

        baseline_count = cursor.rowcount

        conn.commit()
        conn.close()

        print(f"\n{'='*60}")
        print("[OK] Migration completed successfully!")
        print(f"{'='*60}")
        print(f"[*] Tables created: {len(MIGRATION_SQL) - len([k for k in MIGRATION_SQL if k.startswith('idx')])}")
        print(f"[*] Indexes created: {len([k for k in MIGRATION_SQL if k.startswith('idx')])}")
        print(f"[*] Default patterns: {len(DEFAULT_PATTERNS)}")
        print(f"[*] Baselines initialized: {baseline_count}")
        print()

        return True

    except sqlite3.Error as e:
        print(f"\n[ERROR] Migration failed: {e}")
        return False


def verify_migration(db_path: str):
    """Verify migration was successful.

    Args:
        db_path: Path to database file
    """
    if not os.path.exists(db_path):
        print("[ERROR] Database not found")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n{'='*60}")
    print("MIGRATION VERIFICATION")
    print(f"{'='*60}\n")

    # Check tables
    print("[*] Tables:")
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name LIKE '%anomaly%' OR name LIKE '%history%'
        ORDER BY name
    """)
    for row in cursor.fetchall():
        print(f"    [OK] {row[0]}")

    # Check indexes
    print("\n[*] Indexes:")
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND (name LIKE '%history%' OR name LIKE '%pattern%')
        ORDER BY name
    """)
    for row in cursor.fetchall():
        print(f"    [OK] {row[0]}")

    # Count patterns
    print("\n[*] Default Patterns:")
    cursor.execute("SELECT COUNT(*) FROM anomaly_patterns")
    pattern_count = cursor.fetchone()[0]
    print(f"    - Total: {pattern_count}")

    cursor.execute("SELECT pattern_type, COUNT(*) FROM anomaly_patterns GROUP BY pattern_type")
    for ptype, count in cursor.fetchall():
        print(f"    - {ptype}: {count}")

    # Count baselines
    print("\n[*] Baselines:")
    cursor.execute("SELECT COUNT(*) FROM anomaly_baselines")
    baseline_count = cursor.fetchone()[0]
    print(f"    - Initialized: {baseline_count}")

    conn.close()
    print(f"\n{'='*60}")
    print("[OK] Verification complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate database to v2.21.0 (anomaly detection)"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='~/.tdz-c64-knowledge',
        help='Database directory (default: ~/.tdz-c64-knowledge)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show SQL without executing'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify migration after completion'
    )

    args = parser.parse_args()

    # Expand path
    data_dir = os.path.expanduser(args.data_dir)
    db_path = os.path.join(data_dir, 'knowledge_base.db')

    # Run migration
    success = run_migration(db_path, args.dry_run)

    if not success:
        return 1

    # Verify if requested
    if args.verify and not args.dry_run:
        verify_migration(db_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
