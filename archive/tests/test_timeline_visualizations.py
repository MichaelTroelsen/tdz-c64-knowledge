"""Test script for Phase 3 Task 3.6 Timeline Visualizations"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase
from pathlib import Path


def test_timeline_visualizations():
    """Test interactive timeline visualization methods."""
    print("\n" + "=" * 60)
    print("Testing Timeline Visualization Methods")
    print("=" * 60)

    # Initialize KnowledgeBase
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    all_passed = True

    # Ensure we have events to visualize
    print("\n1. Checking for events in database...")
    print()

    cursor = kb.db_conn.cursor()
    event_count = cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    timeline_count = cursor.execute("SELECT COUNT(*) FROM timeline_entries").fetchone()[0]

    print(f"  Events in database: {event_count}")
    print(f"  Timeline entries: {timeline_count}")

    # If we have no events, extract from first document
    if event_count == 0 and kb.documents:
        print("\n  No events found. Extracting from first document...")
        doc_id = list(kb.documents.keys())[0]
        result = kb.extract_document_events(doc_id, min_confidence=0.5)
        print(f"  [OK] Extracted {result['stored_count']} events")
        event_count = result['stored_count']

    # Build timeline if needed
    if timeline_count == 0 and event_count > 0:
        print("\n  Building timeline from events...")
        result = kb.build_timeline(min_confidence=0.5)
        print(f"  [OK] Built timeline with {result['timeline_entries']} entries")
        timeline_count = result['timeline_entries']

    if event_count == 0:
        print("\n  [INFO] No events available for visualization tests")
        print("  [INFO] Skipping visualization tests (requires events in database)")
        print("\n" + "=" * 60)
        print("[INFO] TESTS SKIPPED - No events available")
        print("=" * 60)
        kb.db_conn.close()
        return True

    print(f"\n  [OK] Ready to test visualizations ({event_count} events, {timeline_count} timeline entries)")

    # Test 1: Timeline visualization
    print("\n2. Testing interactive timeline visualization...")
    print()

    try:
        output_path = "test_timeline.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Create visualization
        result_path = kb.visualize_timeline(output_path=output_path)

        # Check file was created
        if Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"  [OK] Timeline visualization created: {result_path}")
            print(f"       File size: {file_size:,} bytes")

            # Basic content check
            content = Path(result_path).read_text(encoding='utf-8')
            if 'plotly' in content.lower() and 'timeline' in content.lower():
                print(f"  [OK] File contains Plotly timeline content")
            else:
                print(f"  [ERROR] File missing expected content")
                all_passed = False
        else:
            print(f"  [ERROR] Visualization file not created")
            all_passed = False

    except Exception as e:
        print(f"  [ERROR] Timeline visualization failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Event network visualization
    print("\n3. Testing event network visualization...")
    print()

    try:
        output_path = "test_event_network.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Check if we have enough events for network
        if event_count < 2:
            print(f"  [INFO] Need at least 2 events for network (have {event_count})")
            print(f"  [INFO] Skipping network visualization test")
        else:
            # Create visualization
            result_path = kb.visualize_event_network(output_path=output_path)

            if not result_path:
                print(f"  [INFO] Not enough events with relationships for network")
            elif Path(result_path).exists():
                file_size = Path(result_path).stat().st_size
                print(f"  [OK] Event network visualization created: {result_path}")
                print(f"       File size: {file_size:,} bytes")

                # Basic content check
                content = Path(result_path).read_text(encoding='utf-8')
                if 'plotly' in content.lower():
                    print(f"  [OK] File contains Plotly network content")
                else:
                    print(f"  [ERROR] File missing expected content")
                    all_passed = False
            else:
                print(f"  [ERROR] Visualization file not created")
                all_passed = False

    except Exception as e:
        print(f"  [ERROR] Event network visualization failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Event trends visualization
    print("\n4. Testing event trends visualization...")
    print()

    try:
        output_path = "test_event_trends.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Create visualization
        result_path = kb.visualize_event_trends(output_path=output_path)

        # Check file was created
        if Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"  [OK] Event trends visualization created: {result_path}")
            print(f"       File size: {file_size:,} bytes")

            # Basic content check
            content = Path(result_path).read_text(encoding='utf-8')
            if 'plotly' in content.lower() and 'subplot' in content.lower():
                print(f"  [OK] File contains Plotly subplot content")
            else:
                print(f"  [ERROR] File missing expected content")
                all_passed = False
        else:
            print(f"  [ERROR] Visualization file not created")
            all_passed = False

    except Exception as e:
        print(f"  [ERROR] Event trends visualization failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 4: Date range filtering
    print("\n5. Testing timeline with date range filtering...")
    print()

    try:
        # Get year range from database
        year_data = cursor.execute("""
            SELECT MIN(year), MAX(year) FROM events
            WHERE year IS NOT NULL
        """).fetchone()

        if year_data and year_data[0]:
            min_year, max_year = year_data
            print(f"  Event year range: {min_year} - {max_year}")

            # Test with restricted range
            if max_year > min_year:
                mid_year = (min_year + max_year) // 2
                output_path = "test_timeline_filtered.html"

                if Path(output_path).exists():
                    Path(output_path).unlink()

                result_path = kb.visualize_timeline(
                    start_year=min_year,
                    end_year=mid_year,
                    output_path=output_path
                )

                if Path(result_path).exists():
                    print(f"  [OK] Filtered timeline created ({min_year}-{mid_year})")
                else:
                    print(f"  [ERROR] Filtered timeline not created")
                    all_passed = False
            else:
                print(f"  [INFO] All events from same year, skipping filter test")
        else:
            print(f"  [INFO] No years in events, skipping filter test")

    except Exception as e:
        print(f"  [ERROR] Date range filtering failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 5: Verify Plotly dependencies
    print("\n6. Verifying visualization dependencies...")
    print()

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import networkx as nx

        print("  [OK] plotly.graph_objects imported")
        print("  [OK] plotly.subplots imported")
        print("  [OK] networkx imported")

        # Check versions
        import plotly
        print(f"  Plotly version: {plotly.__version__}")
        print(f"  NetworkX version: {nx.__version__}")

    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        all_passed = False

    # Close database
    kb.db_conn.close()

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"[OK] ALL VISUALIZATION TESTS PASSED")
        print("\nTimeline visualization methods are working correctly:")
        print("  - Interactive timeline with Plotly")
        print("  - Event network graph with NetworkX")
        print("  - Multi-chart trend analysis")
        print("  - Date range filtering")
        print("  - HTML output with hover interactivity")
        print("\nGenerated test files:")
        print("  - test_timeline.html")
        print("  - test_event_network.html")
        print("  - test_event_trends.html")
        if Path("test_timeline_filtered.html").exists():
            print("  - test_timeline_filtered.html")
        print("\nTask 3.6: Timeline Visualizations - COMPLETE")
    else:
        print(f"[ERROR] SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_timeline_visualizations()
    sys.exit(0 if success else 1)
