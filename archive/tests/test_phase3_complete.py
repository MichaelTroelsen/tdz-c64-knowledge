"""Comprehensive Phase 3 Test Suite - Temporal Analysis & Advanced Visualizations

This test suite runs all Phase 3 tests in sequence to validate the complete
temporal analysis implementation.

Tests:
1. Database Schema (3.1)
2. Date/Time Extraction (3.2)
3. Event Detection (3.3)
4. Timeline Construction (3.4)
5. Timeline Visualizations (3.6)
6. Advanced Graph Visualizations (3.7)

Note: MCP Tools (3.5) are tested implicitly through the CLI/API.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime


def run_test(test_file: str, test_name: str) -> tuple[bool, float]:
    """
    Run a single test file and return success status and duration.

    Args:
        test_file: Path to test file
        test_name: Human-readable test name

    Returns:
        (success, duration_seconds)
    """
    print(f"\n{'=' * 70}")
    print(f"Running: {test_name}")
    print(f"File: {test_file}")
    print(f"{'=' * 70}")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)

        success = result.returncode == 0

        if success:
            print(f"\n[OK] {test_name} PASSED ({duration:.2f}s)")
        else:
            print(f"\n[FAIL] {test_name} FAILED ({duration:.2f}s)")
            print(f"Exit code: {result.returncode}")

        return success, duration

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n[TIMEOUT] {test_name} TIMEOUT ({duration:.2f}s)")
        return False, duration

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n[ERROR] {test_name} ERROR: {e} ({duration:.2f}s)")
        return False, duration


def main():
    """Run complete Phase 3 test suite."""
    print("\n" + "=" * 70)
    print("TDZ C64 Knowledge Base - Phase 3 Complete Test Suite")
    print("Temporal Analysis & Advanced Visualizations")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define all tests
    tests = [
        ("test_phase3_schema.py", "Task 3.1: Database Schema"),
        ("test_date_extraction.py", "Task 3.2: Date/Time Extraction"),
        ("test_event_detection.py", "Task 3.3: Event Detection"),
        ("test_timeline_construction.py", "Task 3.4: Timeline Construction"),
        ("test_timeline_visualizations.py", "Task 3.6: Timeline Visualizations"),
        ("test_advanced_visualizations.py", "Task 3.7: Advanced Graph Visualizations"),
    ]

    # Check that all test files exist
    missing_files = []
    for test_file, _ in tests:
        if not Path(test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print("\n[ERROR] Missing test files:")
        for f in missing_files:
            print(f"  - {f}")
        return 1

    # Run all tests
    results = []
    total_duration = 0.0

    for test_file, test_name in tests:
        success, duration = run_test(test_file, test_name)
        results.append((test_name, success, duration))
        total_duration += duration

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3 TEST SUITE SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\nTests Run: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")

    print("\nDetailed Results:")
    for test_name, success, duration in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} - {test_name} ({duration:.2f}s)")

    # Check for generated visualizations
    print("\nGenerated Visualization Files:")
    viz_files = [
        "test_timeline.html",
        "test_event_network.html",
        "test_event_trends.html",
        "test_timeline_filtered.html",
        "test_knowledge_graph_3d.html",
        "test_hierarchical_bundling.html",
        "test_topic_flow.html",
        "test_topic_flow_yearly.html"
    ]

    generated = 0
    for viz_file in viz_files:
        if Path(viz_file).exists():
            size = Path(viz_file).stat().st_size
            print(f"  [+] {viz_file} ({size:,} bytes)")
            generated += 1
        else:
            print(f"  [-] {viz_file} (not generated)")

    print(f"\nVisualization Files: {generated}/{len(viz_files)}")

    # Overall status
    print("\n" + "=" * 70)
    if passed == total:
        print("[OK] ALL PHASE 3 TESTS PASSED")
        print("\nPhase 3: Temporal Analysis & Advanced Visualizations - COMPLETE")
        print("\nImplemented Features:")
        print("  [+] Database schema with events, timeline, and mapping tables")
        print("  [+] Date/time extraction (8 formats supported)")
        print("  [+] Event detection (5 event types)")
        print("  [+] Timeline construction with chronological sorting")
        print("  [+] 4 MCP tools for timeline access")
        print("  [+] Interactive timeline visualizations (Plotly)")
        print("  [+] Event network graphs (NetworkX + Plotly)")
        print("  [+] Trend analysis charts (3 subplots)")
        print("  [+] 3D knowledge graph visualization")
        print("  [+] Hierarchical edge bundling (circular layout)")
        print("  [+] Sankey diagrams for topic flow")
        print("\nDocumentation:")
        print("  - See PHASE3_TEMPORAL_ANALYSIS.md for complete details")
        print("  - See test files for usage examples")
        exit_code = 0
    else:
        print("[FAIL] SOME PHASE 3 TESTS FAILED")
        print("\nPlease review failed tests above")
        exit_code = 1

    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
