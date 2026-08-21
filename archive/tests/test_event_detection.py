"""Test script for Phase 3 Task 3.3 Event Detection"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase


def test_event_detection_patterns():
    """Test event detection from C64-related text samples."""
    print("\n" + "=" * 60)
    print("Testing Event Detection Methods")
    print("=" * 60)

    # Initialize KnowledgeBase
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Test samples with different event types
    test_cases = [
        {
            'text': "The Commodore 64 was released in August 1982 by Commodore International.",
            'expected_types': ['release'],
            'expected_min_events': 1,
            'description': "Product release event"
        },
        {
            'text': "Commodore Business Machines was founded in 1954 by Jack Tramiel.",
            'expected_types': ['milestone'],
            'expected_min_events': 1,
            'description': "Company milestone event"
        },
        {
            'text': "The SID chip was the first programmable sound generator in a home computer, developed in 1981.",
            'expected_types': ['innovation'],
            'expected_min_events': 1,
            'description': "Technical innovation event"
        },
        {
            'text': "The demo scene competition was held at the conference in 1985.",
            'expected_types': ['cultural'],
            'expected_min_events': 1,
            'description': "Cultural event"
        },
        {
            'text': "The C64 was introduced at CES in January 1982 and later released in August 1982.",
            'expected_types': ['release'],
            'expected_min_events': 2,  # "introduced" and "released"
            'description': "Multiple events in one sentence"
        },
        {
            'text': "Version 2 of KERNAL was updated in 1983 with bug fixes.",
            'expected_types': ['update'],
            'expected_min_events': 1,
            'description': "Update/version event"
        }
    ]

    all_passed = True
    passed_count = 0
    failed_count = 0

    print("\n1. Testing event pattern detection...")
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"  Text: \"{test_case['text']}\"")

        # Detect events
        events = kb.detect_events_in_text(test_case['text'])

        # Check event count
        if len(events) >= test_case['expected_min_events']:
            print(f"  [OK] Found {len(events)} event(s)")
        else:
            print(f"  [ERROR] Expected at least {test_case['expected_min_events']} events, found {len(events)}")
            all_passed = False
            failed_count += 1
            print()
            continue

        # Check event types
        found_types = [e['type'] for e in events]
        expected_types_set = set(test_case['expected_types'])
        found_types_set = set(found_types)

        if expected_types_set.issubset(found_types_set):
            print(f"  [OK] Event types: {', '.join(set(found_types))}")
        else:
            print(f"  [ERROR] Expected types {test_case['expected_types']}, found {found_types}")
            all_passed = False
            failed_count += 1
            print()
            continue

        # Show detected events
        for j, event in enumerate(events, 1):
            date_str = event['date_info']['text'] if event['date_info'] else 'No date'
            entities_str = ', '.join(event['entities']) if event['entities'] else 'None'
            print(f"    Event {j}:")
            print(f"      Type: {event['type']}")
            print(f"      Trigger: '{event['trigger_word']}'")
            print(f"      Date: {date_str}")
            print(f"      Confidence: {event['confidence']:.2f}")
            print(f"      Entities: {entities_str}")
            print(f"      Title: {event['title'][:60]}...")

        passed_count += 1
        print()

    # Test with real C64 document
    print("\n2. Testing with real C64 documentation...")
    print()

    if kb.documents:
        # Get a document
        doc_id = list(kb.documents.keys())[0]
        doc = kb.documents[doc_id]

        print(f"  Testing with: {doc.title}")

        # Extract events from the document (limiting to high confidence)
        result = kb.extract_document_events(doc_id, min_confidence=0.7)

        print(f"  [OK] Total events detected: {result['event_count']}")
        print(f"  [OK] Events with confidence >= 0.7: {result['filtered_count']}")
        print(f"  [OK] Events stored to database: {result['stored_count']}")

        # Show a few example events
        if result['events']:
            print(f"\n  Sample events:")
            for i, event in enumerate(result['events'][:3], 1):
                date_str = event['date_info']['text'] if event['date_info'] else 'No date'
                print(f"    {i}. [{event['type']}] {event['title'][:60]}...")
                print(f"       Date: {date_str}, Confidence: {event['confidence']:.2f}")

        # Verify events were stored to database
        cursor = kb.db_conn.cursor()
        db_count = cursor.execute("""
            SELECT COUNT(*) FROM events
            WHERE event_id IN (
                SELECT event_id FROM document_events WHERE doc_id = ?
            )
        """, (doc_id,)).fetchone()[0]

        if db_count >= result['stored_count']:
            print(f"\n  [OK] Database verification: {db_count} events found for document (>= {result['stored_count']} expected)")
        else:
            print(f"\n  [ERROR] Database mismatch: expected at least {result['stored_count']}, found {db_count}")
            all_passed = False

    else:
        print("  [INFO] No documents in database, skipping real document test")

    # Test event querying
    print("\n3. Testing event database queries...")
    print()

    cursor = kb.db_conn.cursor()

    # Count events by type
    type_counts = cursor.execute("""
        SELECT event_type, COUNT(*) as count
        FROM events
        GROUP BY event_type
        ORDER BY count DESC
    """).fetchall()

    if type_counts:
        print("  [OK] Events by type:")
        for event_type, count in type_counts:
            print(f"       {event_type}: {count} events")
    else:
        print("  [INFO] No events in database yet")

    # Count events by year
    year_counts = cursor.execute("""
        SELECT year, COUNT(*) as count
        FROM events
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """).fetchall()

    if year_counts:
        print(f"\n  [OK] Events by year (showing first 5):")
        for year, count in year_counts[:5]:
            print(f"       {year}: {count} events")
    else:
        print("\n  [INFO] No events with year information")

    # Close database
    kb.db_conn.close()

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"[OK] ALL TESTS PASSED ({passed_count}/{len(test_cases)})")
        print("\nEvent detection methods are working correctly:")
        print("  - Pattern-based event detection")
        print("  - 5 event types supported (release, milestone, innovation, cultural, update)")
        print("  - Date association and confidence scoring")
        print("  - Entity extraction from context")
        print("  - Database storage with document mapping")
        print("\nTask 3.3: Event Detection - COMPLETE")
    else:
        print(f"[ERROR] SOME TESTS FAILED")
        print(f"  Passed: {passed_count}/{len(test_cases)}")
        print(f"  Failed: {failed_count}/{len(test_cases)}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_event_detection_patterns()
    sys.exit(0 if success else 1)
