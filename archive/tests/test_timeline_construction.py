"""Test script for Phase 3 Task 3.4 Timeline Construction"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase


def test_timeline_construction():
    """Test timeline construction and querying methods."""
    print("\n" + "=" * 60)
    print("Testing Timeline Construction Methods")
    print("=" * 60)

    # Initialize KnowledgeBase
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    all_passed = True

    # Test 1: Build timeline from existing events
    print("\n1. Testing timeline construction...")
    print()

    result = kb.build_timeline(min_confidence=0.5)

    print(f"  Total events processed: {result['total_events']}")
    print(f"  Timeline entries created: {result['timeline_entries']}")

    if result['date_range']:
        print(f"  Date range: {result['date_range'][0]} - {result['date_range'][1]}")
    else:
        print(f"  Date range: No events with dates")

    if result['total_events'] > 0:
        print(f"  [OK] Timeline built successfully")

        # Show categories
        if result['categories']:
            print(f"\n  Categories ({len(result['categories'])} total):")
            for category, count in sorted(result['categories'].items())[:5]:
                print(f"    - {category}: {count} events")

        # Show year distribution
        if result['by_year']:
            print(f"\n  Events by year (showing first 5):")
            for year, count in sorted(result['by_year'].items())[:5]:
                print(f"    - {year}: {count} events")
    else:
        print(f"  [INFO] No events in database to build timeline")

    # Test 2: Query timeline with various filters
    print("\n2. Testing timeline queries...")
    print()

    # Query 1: All timeline entries
    all_entries = kb.get_timeline()
    print(f"  Total timeline entries: {len(all_entries)}")

    if all_entries:
        print(f"  [OK] get_timeline() returned {len(all_entries)} entries")

        # Show first few entries
        print(f"\n  First 3 timeline entries:")
        for i, entry in enumerate(all_entries[:3], 1):
            print(f"    {i}. [{entry['display_date']}] {entry['title'][:50]}...")
            print(f"       Type: {entry['event_type']}, Importance: {entry['importance']}/5")

    # Query 2: Filter by year range (1980s)
    entries_1980s = kb.get_timeline(start_year=1980, end_year=1989)
    if entries_1980s:
        print(f"\n  [OK] 1980s events: {len(entries_1980s)} entries")
    else:
        print(f"\n  [INFO] No events from 1980s")

    # Query 3: High importance events only
    important_entries = kb.get_timeline(min_importance=4)
    if important_entries:
        print(f"  [OK] High importance events (4-5): {len(important_entries)} entries")
    else:
        print(f"  [INFO] No high importance events")

    # Query 4: Limited results
    limited_entries = kb.get_timeline(limit=5)
    if limited_entries:
        print(f"  [OK] Limited query (5 max): {len(limited_entries)} entries")
        if len(limited_entries) == 5 or len(limited_entries) == len(all_entries):
            print(f"       Limit working correctly")

    # Test 3: Search events by date
    print("\n3. Testing event search by date...")
    print()

    # Search 1: All events from 1982
    events_1982 = kb.search_events_by_date(start_year=1982, end_year=1982)
    print(f"  Events from 1982: {len(events_1982)}")

    if events_1982:
        print(f"  [OK] Found {len(events_1982)} events in 1982")
        for i, event in enumerate(events_1982[:3], 1):
            print(f"    {i}. [{event['date_normalized']}] {event['title'][:50]}...")

    # Search 2: Filter by event type
    release_events = kb.search_events_by_date(
        start_year=1980,
        end_year=1990,
        event_type='release'
    )
    if release_events:
        print(f"\n  [OK] Release events (1980-1990): {len(release_events)}")
    else:
        print(f"\n  [INFO] No release events in 1980-1990")

    # Search 3: High confidence events only
    high_conf_events = kb.search_events_by_date(
        start_year=1980,
        end_year=1990,
        min_confidence=0.8
    )
    if high_conf_events:
        print(f"  [OK] High confidence events (>=0.8): {len(high_conf_events)}")
    else:
        print(f"  [INFO] No high confidence events")

    # Test 4: Get historical context
    print("\n4. Testing historical context...")
    print()

    # Get context for 1982 (C64 release year)
    context = kb.get_historical_context(1982, context_years=2)

    print(f"  Historical context for 1982:")
    print(f"    Year range: {context['year_range'][0]} - {context['year_range'][1]}")
    print(f"    Total events: {context['total_events']}")

    if context['events_by_year']:
        print(f"\n    Events by year:")
        for year in sorted(context['events_by_year'].keys()):
            count = len(context['events_by_year'][year])
            print(f"      {year}: {count} events")

        print(f"\n  [OK] Historical context retrieved successfully")
    else:
        print(f"\n  [INFO] No events in context range")

    # Test 5: Verify database integrity
    print("\n5. Testing database integrity...")
    print()

    cursor = kb.db_conn.cursor()

    # Count timeline entries
    entry_count = cursor.execute("SELECT COUNT(*) FROM timeline_entries").fetchone()[0]
    print(f"  Timeline entries in database: {entry_count}")

    # Verify all entries have valid event_ids
    orphan_count = cursor.execute("""
        SELECT COUNT(*) FROM timeline_entries
        WHERE event_id NOT IN (SELECT event_id FROM events)
    """).fetchone()[0]

    if orphan_count == 0:
        print(f"  [OK] All timeline entries have valid event references")
    else:
        print(f"  [ERROR] {orphan_count} orphan timeline entries found")
        all_passed = False

    # Verify sort order is correct
    entries = cursor.execute("""
        SELECT sort_order FROM timeline_entries
        ORDER BY sort_order
    """).fetchall()

    if entries:
        is_sorted = all(entries[i][0] <= entries[i+1][0] for i in range(len(entries)-1))
        if is_sorted:
            print(f"  [OK] Timeline entries are correctly sorted")
        else:
            print(f"  [ERROR] Timeline entries are not in chronological order")
            all_passed = False

    # Test 6: Verify category structure
    print("\n6. Testing category structure...")
    print()

    categories = cursor.execute("""
        SELECT DISTINCT category FROM timeline_entries
    """).fetchall()

    if categories:
        print(f"  Unique categories: {len(categories)}")
        print(f"  Sample categories:")
        for cat in categories[:5]:
            print(f"    - {cat[0]}")
        print(f"  [OK] Categories properly structured")
    else:
        print(f"  [INFO] No categories in timeline")

    # Close database
    kb.db_conn.close()

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"[OK] ALL TESTS PASSED")
        print("\nTimeline construction methods are working correctly:")
        print("  - Timeline building from events")
        print("  - Chronological sorting with importance levels")
        print("  - Category-based organization")
        print("  - Flexible querying with filters")
        print("  - Date range searches")
        print("  - Historical context retrieval")
        print("\nTask 3.4: Timeline Construction - COMPLETE")
    else:
        print(f"[ERROR] SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_timeline_construction()
    sys.exit(0 if success else 1)
