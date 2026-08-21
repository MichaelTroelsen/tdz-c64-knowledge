"""Comprehensive test suite for Phase 2: Discovery - Topic Modeling & Clustering"""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_phase2_complete():
    """Run complete Phase 2 test suite."""

    print("\n" + "="*80)
    print("PHASE 2: TOPIC MODELING & CLUSTERING - COMPREHENSIVE TEST SUITE")
    print("="*80)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))
    cursor = kb.db_conn.cursor()

    print(f"\n[INIT] KnowledgeBase initialized")
    print(f"  Documents: {len(kb.documents)}")

    all_passed = True

    # ========================================================================
    # PART 1: DATABASE SCHEMA VERIFICATION
    # ========================================================================
    print("\n" + "="*80)
    print("PART 1: Database Schema Verification")
    print("="*80)

    # Check topics table
    topics_schema = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='topics'"
    ).fetchone()
    if topics_schema:
        print("\n[OK] topics table exists")
        # Count topics
        topic_counts = cursor.execute("""
            SELECT model_type, COUNT(*)
            FROM topics
            GROUP BY model_type
        """).fetchall()
        for model_type, count in topic_counts:
            print(f"  - {model_type.upper()}: {count} topics")
    else:
        print("\n[ERROR] topics table missing")
        all_passed = False

    # Check document_topics table
    dt_schema = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='document_topics'"
    ).fetchone()
    if dt_schema:
        print("\n[OK] document_topics table exists")
        assignment_count = cursor.execute(
            "SELECT COUNT(*) FROM document_topics"
        ).fetchone()[0]
        print(f"  - Total assignments: {assignment_count}")
    else:
        print("\n[ERROR] document_topics table missing")
        all_passed = False

    # Check clusters table
    clusters_schema = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='clusters'"
    ).fetchone()
    if clusters_schema:
        print("\n[OK] clusters table exists")
        cluster_counts = cursor.execute("""
            SELECT algorithm, COUNT(*)
            FROM clusters
            GROUP BY algorithm
        """).fetchall()
        for algorithm, count in cluster_counts:
            print(f"  - {algorithm.upper()}: {count} clusters")
    else:
        print("\n[ERROR] clusters table missing")
        all_passed = False

    # Check document_clusters table
    dc_schema = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='document_clusters'"
    ).fetchone()
    if dc_schema:
        print("\n[OK] document_clusters table exists")
        assignment_count = cursor.execute(
            "SELECT COUNT(*) FROM document_clusters"
        ).fetchone()[0]
        print(f"  - Total assignments: {assignment_count}")
    else:
        print("\n[ERROR] document_clusters table missing")
        all_passed = False

    # ========================================================================
    # PART 2: TOPIC MODELING METHODS
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Topic Modeling Methods")
    print("="*80)

    # Test LDA
    print("\n[TEST] LDA Topic Modeling")
    try:
        result = kb.train_lda_model(num_topics=3, max_iter=50, random_state=42)
        if 'error' not in result:
            print(f"  [OK] LDA trained: {result['num_topics']} topics")
            print(f"  Perplexity: {result['perplexity']:.2f}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] LDA failed: {e}")
        all_passed = False

    # Test NMF
    print("\n[TEST] NMF Topic Modeling")
    try:
        result = kb.train_nmf_model(num_topics=3, max_iter=100, random_state=42)
        if 'error' not in result:
            print(f"  [OK] NMF trained: {result['num_topics']} topics")
            print(f"  Reconstruction error: {result['reconstruction_error']:.4f}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] NMF failed: {e}")
        all_passed = False

    # Test BERTopic
    print("\n[TEST] BERTopic Modeling")
    try:
        result = kb.train_bertopic_model(num_topics=3, min_cluster_size=5, random_state=42)
        if 'error' not in result:
            print(f"  [OK] BERTopic trained: {result['num_topics']} topics")
            print(f"  Documents assigned: {result['num_assignments']}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] BERTopic failed: {e}")
        all_passed = False

    # ========================================================================
    # PART 3: CLUSTERING METHODS
    # ========================================================================
    print("\n" + "="*80)
    print("PART 3: Clustering Methods")
    print("="*80)

    # Test K-Means
    print("\n[TEST] K-Means Clustering")
    try:
        result = kb.cluster_documents_kmeans(num_clusters=5, random_state=42)
        if 'error' not in result:
            print(f"  [OK] K-Means: {result['num_clusters']} clusters")
            print(f"  Silhouette: {result['silhouette_score']:.3f}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] K-Means failed: {e}")
        all_passed = False

    # Test DBSCAN
    print("\n[TEST] DBSCAN Clustering")
    try:
        result = kb.cluster_documents_dbscan(eps=0.5, min_samples=5)
        if 'error' not in result:
            print(f"  [OK] DBSCAN: {result['num_clusters']} clusters")
            print(f"  Outliers: {result['num_outliers']}")
            print(f"  Silhouette: {result['silhouette_score']:.3f}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] DBSCAN failed: {e}")
        all_passed = False

    # Test HDBSCAN
    print("\n[TEST] HDBSCAN Clustering")
    try:
        result = kb.cluster_documents_hdbscan(min_cluster_size=5)
        if 'error' not in result:
            print(f"  [OK] HDBSCAN: {result['num_clusters']} clusters")
            print(f"  Outliers: {result['num_outliers']}")
        else:
            print(f"  [ERROR] {result['error']}")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] HDBSCAN failed: {e}")
        all_passed = False

    # ========================================================================
    # PART 4: VISUALIZATION METHODS
    # ========================================================================
    print("\n" + "="*80)
    print("PART 4: Visualization Methods")
    print("="*80)

    # Test Topic Word Cloud
    print("\n[TEST] Topic Word Cloud")
    topic = cursor.execute(
        "SELECT topic_id FROM topics WHERE model_type = 'lda' LIMIT 1"
    ).fetchone()
    if topic:
        try:
            result = kb.generate_topic_wordcloud(
                topic_id=topic[0],
                output_path='phase2_test_wordcloud.png'
            )
            if 'error' not in result and os.path.exists(result['output_path']):
                print(f"  [OK] Word cloud generated: {result['output_path']}")
                os.remove(result['output_path'])  # Cleanup
            else:
                print(f"  [ERROR] Word cloud failed")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_passed = False
    else:
        print("  [SKIP] No topics available")

    # Test Cluster Scatter Plot
    print("\n[TEST] Cluster Scatter Plot")
    try:
        result = kb.visualize_cluster_scatter(
            algorithm='kmeans',
            output_path='phase2_test_scatter.png'
        )
        if 'error' not in result and os.path.exists(result['output_path']):
            print(f"  [OK] Scatter plot generated: {result['output_path']}")
            os.remove(result['output_path'])  # Cleanup
        else:
            print(f"  [ERROR] Scatter plot failed")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    # Test Topic Heatmap
    print("\n[TEST] Topic Heatmap")
    try:
        result = kb.generate_topic_heatmap(
            model_type='nmf',
            output_path='phase2_test_heatmap.png',
            max_topics=3,
            max_documents=20
        )
        if 'error' not in result and os.path.exists(result['output_path']):
            print(f"  [OK] Heatmap generated: {result['output_path']}")
            os.remove(result['output_path'])  # Cleanup
        else:
            print(f"  [ERROR] Heatmap failed")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    # Test Cluster Distribution
    print("\n[TEST] Cluster Distribution")
    try:
        result = kb.visualize_cluster_distribution(
            algorithm='dbscan',
            output_path='phase2_test_distribution.png'
        )
        if 'error' not in result and os.path.exists(result['output_path']):
            print(f"  [OK] Distribution chart generated: {result['output_path']}")
            os.remove(result['output_path'])  # Cleanup
        else:
            print(f"  [ERROR] Distribution chart failed")
            all_passed = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        all_passed = False

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] ALL PHASE 2 TESTS PASSED")
        print("="*80)
        print("\nPhase 2 Implementation Complete:")
        print("  + Database schema (4 tables)")
        print("  + Topic modeling (LDA, NMF, BERTopic)")
        print("  + Clustering (K-Means, DBSCAN, HDBSCAN)")
        print("  + Visualizations (word clouds, scatter plots, heatmaps, distributions)")
        print("  + 12 MCP tools (8 topic/cluster + 4 visualization)")
        print("\nReady for production use!")
    else:
        print("[FAILURE] SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review errors above and fix issues.")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    success = test_phase2_complete()
    sys.exit(0 if success else 1)
