"""Test script for Phase 3 Task 3.2 Date/Time Extraction"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase


def test_date_extraction():
    """Test date extraction from C64-related text samples."""
    print("\n" + "=" * 60)
    print("Testing Date Extraction Methods")
    print("=" * 60)

    # Initialize KnowledgeBase
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Test samples with expected results
    test_cases = [
        {
            'text': "The Commodore 64 was released in August 1982.",
            'expected_count': 1,
            'expected_types': ['month_year'],
            'expected_years': [1982],
            'description': "Month-Year format"
        },
        {
            'text': "The C64 was popular from 1982 to 1994.",
            'expected_count': 2,  # "1982 to 1994" range + individual years
            'expected_types': ['range'],
            'expected_years': [1982],
            'description': "Year range"
        },
        {
            'text': "On January 15, 1982, the C64 was unveiled at CES.",
            'expected_count': 1,
            'expected_types': ['full_date'],
            'expected_years': [1982],
            'description': "Full date (Month Day, Year)"
        },
        {
            'text': "The SID chip was revolutionary throughout the 1980s.",
            'expected_count': 1,
            'expected_types': ['decade'],
            'expected_years': [1980],
            'description': "Decade reference"
        },
        {
            'text': "In the early 80s, many games were released on cassette.",
            'expected_count': 1,
            'expected_types': ['decade'],
            'expected_years': [1980],
            'description': "Decade with prefix (early 80s)"
        },
        {
            'text': "The C64 launched in 1982 and was discontinued in 1994.",
            'expected_count': 2,
            'expected_types': ['year', 'year'],
            'expected_years': [1982, 1994],
            'description': "Multiple years"
        },
        {
            'text': "The VIC-20 (1981) preceded the C64 (1982).",
            'expected_count': 2,
            'expected_types': ['year', 'year'],
            'expected_years': [1981, 1982],
            'description': "Years in parentheses"
        },
        {
            'text': "Production ran from 1982-1994.",
            'expected_count': 1,
            'expected_types': ['range'],
            'expected_years': [1982],
            'description': "Year range with hyphen"
        }
    ]

    all_passed = True
    passed_count = 0
    failed_count = 0

    print("\n1. Testing date extraction patterns...")
    print()

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print(f"  Text: \"{test_case['text']}\"")

        # Extract dates
        dates = kb.extract_dates_from_text(test_case['text'])

        # Check count
        if len(dates) >= test_case['expected_count']:
            print(f"  [OK] Found {len(dates)} date(s)")
        else:
            print(f"  [ERROR] Expected {test_case['expected_count']} dates, found {len(dates)}")
            all_passed = False
            failed_count += 1
            continue

        # Check types
        found_types = [d['type'] for d in dates]
        expected_types_set = set(test_case['expected_types'])
        found_types_set = set(found_types)

        if expected_types_set.issubset(found_types_set):
            print(f"  [OK] Date types: {', '.join(found_types)}")
        else:
            print(f"  [ERROR] Expected types {test_case['expected_types']}, found {found_types}")
            all_passed = False
            failed_count += 1
            continue

        # Check years
        found_years = [d['year'] for d in dates if d.get('year')]
        if all(year in found_years for year in test_case['expected_years']):
            print(f"  [OK] Years: {found_years}")
        else:
            print(f"  [ERROR] Expected years {test_case['expected_years']}, found {found_years}")
            all_passed = False
            failed_count += 1
            continue

        # Show extracted details
        for j, date in enumerate(dates, 1):
            print(f"    Date {j}: '{date['text']}' -> type={date['type']}, "
                  f"year={date.get('year')}, month={date.get('month')}, day={date.get('day')}")

        passed_count += 1
        print()

    # Test date normalization
    print("\n2. Testing date normalization...")
    print()

    normalization_tests = [
        ({'year': 1982, 'month': 8, 'day': 15}, ('1982-08-15', 1982, 8, 15)),
        ({'year': 1982, 'month': 8, 'day': None}, ('1982-08', 1982, 8, 0)),
        ({'year': 1982, 'month': None, 'day': None}, ('1982', 1982, 0, 0)),
        ({'year': 1980, 'month': None, 'day': None, 'type': 'decade'}, ('1980', 1980, 0, 0))
    ]

    for date_dict, expected in normalization_tests:
        result = kb.normalize_date(date_dict)

        if result == expected:
            print(f"  [OK] {date_dict} -> {result[0]}")
        else:
            print(f"  [ERROR] {date_dict}")
            print(f"         Expected: {expected}")
            print(f"         Got: {result}")
            all_passed = False
            failed_count += 1

    print()

    # Test with real C64 document if available
    print("\n3. Testing with real C64 documentation...")
    print()

    if kb.documents:
        # Get first document
        doc_id = list(kb.documents.keys())[0]
        doc = kb.documents[doc_id]

        print(f"  Testing with: {doc.title}")

        # Load chunks
        chunks = kb._get_chunks_db(doc_id)

        # Extract dates from first few chunks
        total_dates = []
        for chunk in chunks[:5]:  # Test first 5 chunks
            dates = kb.extract_dates_from_text(chunk.content)
            total_dates.extend(dates)

        if total_dates:
            print(f"  [OK] Found {len(total_dates)} date references in first 5 chunks")
            # Show a few examples
            for date in total_dates[:3]:
                iso, year, month, day = kb.normalize_date(date)
                print(f"       '{date['text']}' -> {iso}")
        else:
            print(f"  [INFO] No dates found in sample chunks (this may be normal)")
    else:
        print("  [INFO] No documents in database, skipping real document test")

    # Close database
    kb.db_conn.close()

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"[OK] ALL TESTS PASSED ({passed_count}/{len(test_cases)})")
        print("\nDate extraction methods are working correctly:")
        print("  - Regex-based pattern matching")
        print("  - Multiple date formats supported")
        print("  - Date normalization to ISO format")
        print("\nTask 3.2: Date/Time Extraction - COMPLETE")
    else:
        print(f"[ERROR] SOME TESTS FAILED")
        print(f"  Passed: {passed_count}/{len(test_cases)}")
        print(f"  Failed: {failed_count}/{len(test_cases)}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_date_extraction()
    sys.exit(0 if success else 1)
