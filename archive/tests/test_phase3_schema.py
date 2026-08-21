"""Test script for Phase 3 Task 3.1 Database Schema"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase


def test_phase3_tables():
    """Test that Phase 3 tables are created correctly."""
    print("\n" + "=" * 60)
    print("Testing Phase 3 Database Schema Creation")
    print("=" * 60)

    # Initialize KnowledgeBase (creates database)
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)
    cursor = kb.db_conn.cursor()

    # Check events table
    print("\n1. Checking 'events' table...")
    result = cursor.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='events'
    """).fetchone()

    if result:
        print("   [OK] 'events' table exists")
        # Verify columns
        columns = cursor.execute("PRAGMA table_info(events)").fetchall()
        expected_columns = ['event_id', 'event_type', 'title', 'description',
                          'date_extracted', 'date_normalized', 'year', 'month',
                          'day', 'confidence', 'entities', 'metadata', 'created_date']
        actual_columns = [col[1] for col in columns]

        for col in expected_columns:
            if col in actual_columns:
                print(f"   [OK] Column '{col}' exists")
            else:
                print(f"   [ERROR] Column '{col}' missing")
                return False
    else:
        print("   [ERROR] 'events' table not found")
        return False

    # Check document_events table
    print("\n2. Checking 'document_events' table...")
    result = cursor.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='document_events'
    """).fetchone()

    if result:
        print("   [OK] 'document_events' table exists")
        # Verify foreign keys
        fkeys = cursor.execute("PRAGMA foreign_key_list(document_events)").fetchall()
        if len(fkeys) >= 2:
            print(f"   [OK] Foreign keys configured ({len(fkeys)} keys)")
        else:
            print(f"   [ERROR] Expected 2 foreign keys, found {len(fkeys)}")
            return False
    else:
        print("   [ERROR] 'document_events' table not found")
        return False

    # Check timeline_entries table
    print("\n3. Checking 'timeline_entries' table...")
    result = cursor.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='timeline_entries'
    """).fetchone()

    if result:
        print("   [OK] 'timeline_entries' table exists")
        columns = cursor.execute("PRAGMA table_info(timeline_entries)").fetchall()
        expected_columns = ['entry_id', 'event_id', 'display_date', 'sort_order',
                          'category', 'importance', 'created_date']
        actual_columns = [col[1] for col in columns]

        for col in expected_columns:
            if col in actual_columns:
                print(f"   [OK] Column '{col}' exists")
            else:
                print(f"   [ERROR] Column '{col}' missing")
                return False
    else:
        print("   [ERROR] 'timeline_entries' table not found")
        return False

    # Check indexes
    print("\n4. Checking Phase 3 indexes...")
    indexes = cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND (
            name LIKE 'idx_events_%' OR
            name LIKE 'idx_document_events_%' OR
            name LIKE 'idx_timeline_entries_%'
        )
        ORDER BY name
    """).fetchall()

    expected_indexes = [
        'idx_events_type', 'idx_events_year', 'idx_events_date',
        'idx_document_events_doc', 'idx_document_events_event',
        'idx_timeline_entries_event', 'idx_timeline_entries_sort',
        'idx_timeline_entries_category'
    ]

    actual_indexes = [idx[0] for idx in indexes]

    for idx in expected_indexes:
        if idx in actual_indexes:
            print(f"   [OK] Index '{idx}' exists")
        else:
            print(f"   [ERROR] Index '{idx}' missing")
            return False

    # Test basic operations
    print("\n5. Testing basic insert/query operations...")
    try:
        # Insert test event
        import uuid
        from datetime import datetime

        event_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO events (event_id, event_type, title, description,
                              date_extracted, date_normalized, year, month, day,
                              confidence, entities, metadata, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, 'release', 'Test Event', 'Test description',
              '1982', '1982-01-01', 1982, 1, 1, 0.9, '[]', '{}',
              datetime.utcnow().isoformat()))

        kb.db_conn.commit()
        print("   [OK] Insert into 'events' successful")

        # Query the event
        result = cursor.execute("""
            SELECT title, year FROM events WHERE event_id = ?
        """, (event_id,)).fetchone()

        if result and result[0] == 'Test Event' and result[1] == 1982:
            print("   [OK] Query from 'events' successful")
        else:
            print("   [ERROR] Query failed or returned wrong data")
            return False

        # Insert timeline entry
        entry_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO timeline_entries (entry_id, event_id, display_date,
                                         sort_order, category, importance, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (entry_id, event_id, '1982-01-01', 1982_01_01, 'release', 3,
              datetime.utcnow().isoformat()))

        kb.db_conn.commit()
        print("   [OK] Insert into 'timeline_entries' successful")

        # Query with JOIN
        result = cursor.execute("""
            SELECT e.title, t.display_date
            FROM events e
            JOIN timeline_entries t ON e.event_id = t.event_id
            WHERE e.event_id = ?
        """, (event_id,)).fetchone()

        if result and result[0] == 'Test Event':
            print("   [OK] JOIN query successful")
        else:
            print("   [ERROR] JOIN query failed")
            return False

        # Clean up test data
        cursor.execute("DELETE FROM timeline_entries WHERE entry_id = ?", (entry_id,))
        cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        kb.db_conn.commit()
        print("   [OK] Cleanup successful")

    except Exception as e:
        print(f"   [ERROR] Database operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        kb.db_conn.close()

    print("\n" + "=" * 60)
    print("[OK] ALL SCHEMA TESTS PASSED")
    print("=" * 60)
    print("\nPhase 3 Database Schema Summary:")
    print("  - 3 new tables created (events, document_events, timeline_entries)")
    print("  - 8 indexes created for query optimization")
    print("  - Foreign key relationships configured")
    print("  - Basic CRUD operations working")
    print("\nTask 3.1: Database Schema - COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_phase3_tables()
    sys.exit(0 if success else 1)
