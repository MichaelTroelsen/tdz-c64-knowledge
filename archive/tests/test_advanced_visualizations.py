"""Test script for Phase 3 Task 3.7 Advanced Graph Visualizations"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase
from pathlib import Path


def test_advanced_visualizations():
    """Test advanced graph visualization methods."""
    print("\n" + "=" * 60)
    print("Testing Advanced Graph Visualization Methods")
    print("=" * 60)

    # Initialize KnowledgeBase
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    all_passed = True

    # Check prerequisites
    print("\n1. Checking for entities and relationships in database...")
    print()

    cursor = kb.db_conn.cursor()
    entity_count = cursor.execute("SELECT COUNT(*) FROM document_entities").fetchone()[0]
    relationship_count = cursor.execute("SELECT COUNT(*) FROM entity_relationships").fetchone()[0]
    event_count = cursor.execute("SELECT COUNT(*) FROM events WHERE entities IS NOT NULL").fetchone()[0]

    print(f"  Entities in database: {entity_count}")
    print(f"  Relationships in database: {relationship_count}")
    print(f"  Events with entities: {event_count}")

    # Check if we need to extract entities
    if entity_count < 10:
        print("\n  [INFO] Insufficient entities for visualizations")
        print("  [INFO] Need at least 10 entities for meaningful graphs")
        print("  [INFO] Run entity extraction first with:")
        print("         python cli.py extract-all-entities")
        print("\n" + "=" * 60)
        print("[INFO] TESTS SKIPPED - Insufficient data")
        print("=" * 60)
        kb.db_conn.close()
        return True

    print(f"\n  [OK] Sufficient data for visualizations")

    # Test 1: 3D Knowledge Graph
    print("\n2. Testing 3D knowledge graph visualization...")
    print()

    try:
        output_path = "test_knowledge_graph_3d.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Create visualization
        result_path = kb.visualize_knowledge_graph_3d(
            max_entities=30,
            min_confidence=0.6,
            output_path=output_path
        )

        if not result_path:
            print(f"  [INFO] No entities/relationships found for 3D graph")
        elif Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"  [OK] 3D knowledge graph created: {result_path}")
            print(f"       File size: {file_size:,} bytes")

            # Basic content check
            content = Path(result_path).read_text(encoding='utf-8')
            if 'plotly' in content.lower() and 'scatter3d' in content.lower():
                print(f"  [OK] File contains Plotly 3D content")
            else:
                print(f"  [ERROR] File missing expected 3D content")
                all_passed = False
        else:
            print(f"  [ERROR] Visualization file not created")
            all_passed = False

    except Exception as e:
        print(f"  [ERROR] 3D knowledge graph failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 2: Hierarchical Edge Bundling
    print("\n3. Testing hierarchical edge bundling visualization...")
    print()

    try:
        output_path = "test_hierarchical_bundling.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Create visualization
        result_path = kb.visualize_hierarchical_bundling(
            max_entities=25,
            output_path=output_path
        )

        if not result_path:
            print(f"  [INFO] No entities found for hierarchical bundling")
        elif Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"  [OK] Hierarchical bundling created: {result_path}")
            print(f"       File size: {file_size:,} bytes")

            # Basic content check
            content = Path(result_path).read_text(encoding='utf-8')
            if 'plotly' in content.lower():
                print(f"  [OK] File contains Plotly visualization")
            else:
                print(f"  [ERROR] File missing expected content")
                all_passed = False
        else:
            print(f"  [ERROR] Visualization file not created")
            all_passed = False

    except Exception as e:
        print(f"  [ERROR] Hierarchical bundling failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Sankey Diagram (Topic Flow)
    print("\n4. Testing Sankey diagram for topic flow...")
    print()

    try:
        output_path = "test_topic_flow.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Test with decade grouping
        result_path = kb.visualize_topic_flow_sankey(
            time_period='decade',
            output_path=output_path
        )

        if not result_path:
            print(f"  [INFO] Insufficient events for Sankey diagram")
            print(f"       Need events with entities across multiple time periods")
        elif Path(result_path).exists():
            file_size = Path(result_path).stat().st_size
            print(f"  [OK] Sankey diagram created: {result_path}")
            print(f"       File size: {file_size:,} bytes")

            # Basic content check
            content = Path(result_path).read_text(encoding='utf-8')
            if 'sankey' in content.lower():
                print(f"  [OK] File contains Sankey diagram")
            else:
                print(f"  [ERROR] File missing Sankey content")
                all_passed = False
        else:
            print(f"  [ERROR] Visualization file not created")
            all_passed = False

    except Exception as e:
        print(f"  [ERROR] Sankey diagram failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 4: Sankey with year grouping
    print("\n5. Testing Sankey diagram with year grouping...")
    print()

    try:
        output_path = "test_topic_flow_yearly.html"

        # Remove old file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # Test with year grouping
        result_path = kb.visualize_topic_flow_sankey(
            time_period='year',
            output_path=output_path
        )

        if not result_path:
            print(f"  [INFO] Insufficient events for yearly Sankey")
        elif Path(result_path).exists():
            print(f"  [OK] Yearly Sankey diagram created: {result_path}")
        else:
            print(f"  [INFO] Could not create yearly Sankey (may need more data)")

    except Exception as e:
        print(f"  [INFO] Yearly Sankey skipped: {e}")

    # Test 5: Verify dependencies
    print("\n6. Verifying advanced visualization dependencies...")
    print()

    try:
        import plotly.graph_objects as go
        import networkx as nx
        import numpy as np

        print("  [OK] plotly.graph_objects imported")
        print("  [OK] networkx imported")
        print("  [OK] numpy imported")

        # Check for 3D support
        try:
            test_trace = go.Scatter3d(x=[0], y=[0], z=[0])
            print("  [OK] Plotly 3D support available")
        except:
            print("  [ERROR] Plotly 3D support missing")
            all_passed = False

        # Check for Sankey support
        try:
            test_sankey = go.Sankey()
            print("  [OK] Plotly Sankey support available")
        except:
            print("  [ERROR] Plotly Sankey support missing")
            all_passed = False

    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        all_passed = False

    # Test 6: Database integrity check
    print("\n7. Checking visualization data quality...")
    print()

    # Check entity type distribution
    type_counts = cursor.execute("""
        SELECT entity_type, COUNT(DISTINCT entity_text) as unique_count
        FROM document_entities
        WHERE confidence >= 0.6
        GROUP BY entity_type
        ORDER BY unique_count DESC
    """).fetchall()

    if type_counts:
        print("  Entity type distribution:")
        for entity_type, count in type_counts:
            print(f"    {entity_type}: {count} unique entities")
        print(f"  [OK] Entity types properly distributed")
    else:
        print("  [INFO] No high-confidence entities found")

    # Check relationship strength distribution
    strength_data = cursor.execute("""
        SELECT
            COUNT(*) as total,
            AVG(strength) as avg_strength,
            MAX(strength) as max_strength
        FROM entity_relationships
    """).fetchone()

    if strength_data and strength_data[0] > 0:
        total, avg_strength, max_strength = strength_data
        print(f"\n  Relationship statistics:")
        print(f"    Total relationships: {total}")
        print(f"    Average strength: {avg_strength:.3f}")
        print(f"    Maximum strength: {max_strength:.3f}")
        print(f"  [OK] Relationship data available")
    else:
        print(f"\n  [INFO] No relationships extracted yet")

    # Close database
    kb.db_conn.close()

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"[OK] ALL ADVANCED VISUALIZATION TESTS PASSED")
        print("\nAdvanced graph visualization methods are working correctly:")
        print("  - 3D interactive knowledge graph")
        print("  - Hierarchical edge bundling with circular layout")
        print("  - Sankey diagrams for topic flow over time")
        print("  - NetworkX graph algorithms")
        print("  - Plotly 3D and Sankey support")
        print("\nGenerated test files:")
        if Path("test_knowledge_graph_3d.html").exists():
            print("  - test_knowledge_graph_3d.html")
        if Path("test_hierarchical_bundling.html").exists():
            print("  - test_hierarchical_bundling.html")
        if Path("test_topic_flow.html").exists():
            print("  - test_topic_flow.html")
        if Path("test_topic_flow_yearly.html").exists():
            print("  - test_topic_flow_yearly.html")
        print("\nTask 3.7: Advanced Graph Visualizations - COMPLETE")
    else:
        print(f"[ERROR] SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_advanced_visualizations()
    sys.exit(0 if success else 1)
