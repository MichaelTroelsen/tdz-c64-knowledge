"""Test visualization implementations."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_all_visualizations():
    """Test all visualization methods."""
    print("\n" + "="*60)
    print("Testing Visualizations")
    print("="*60)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

    print(f"\n[OK] KnowledgeBase initialized")
    print(f"  Total documents: {len(kb.documents)}")

    cursor = kb.db_conn.cursor()

    # Test 1: Topic Word Cloud
    print("\n" + "-"*60)
    print("Test 1: Topic Word Cloud")
    print("-"*60)

    # Get first LDA topic
    topic = cursor.execute(
        "SELECT topic_id, topic_number FROM topics WHERE model_type = 'lda' LIMIT 1"
    ).fetchone()

    if topic:
        topic_id, topic_num = topic
        print(f"  Using Topic {topic_num} (ID: {topic_id[:12]}...)")

        result = kb.generate_topic_wordcloud(
            topic_id=topic_id,
            output_path='test_wordcloud.png'
        )

        if 'error' in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  [OK] Word cloud generated")
            print(f"    Output: {result['output_path']}")
            print(f"    Words: {result['num_words']}")
    else:
        print("  [SKIP] No LDA topics found")

    # Test 2: Cluster Scatter Plot
    print("\n" + "-"*60)
    print("Test 2: Cluster Scatter Plot")
    print("-"*60)

    # Check if we have DBSCAN clusters
    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'dbscan'"
    ).fetchone()[0]

    if cluster_count > 0:
        print(f"  Using DBSCAN clusters (count: {cluster_count})")

        result = kb.visualize_cluster_scatter(
            algorithm='dbscan',
            output_path='test_cluster_scatter.png'
        )

        if 'error' in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  [OK] Scatter plot generated")
            print(f"    Output: {result['output_path']}")
            print(f"    Clusters: {result['num_clusters']}")
            print(f"    Documents: {result['num_documents']}")
    else:
        print("  [SKIP] No DBSCAN clusters found")

    # Test 3: Topic Heatmap
    print("\n" + "-"*60)
    print("Test 3: Topic Heatmap")
    print("-"*60)

    # Check if we have NMF topics
    topic_count = cursor.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type = 'nmf'"
    ).fetchone()[0]

    if topic_count > 0:
        print(f"  Using NMF topics (count: {topic_count})")

        result = kb.generate_topic_heatmap(
            model_type='nmf',
            output_path='test_topic_heatmap.png',
            max_topics=10,
            max_documents=30
        )

        if 'error' in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  [OK] Heatmap generated")
            print(f"    Output: {result['output_path']}")
            print(f"    Topics: {result['num_topics']}")
            print(f"    Documents: {result['num_documents']}")
    else:
        print("  [SKIP] No NMF topics found")

    # Test 4: Cluster Distribution
    print("\n" + "-"*60)
    print("Test 4: Cluster Distribution")
    print("-"*60)

    # Check if we have K-Means clusters
    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'kmeans'"
    ).fetchone()[0]

    if cluster_count > 0:
        print(f"  Using K-Means clusters (count: {cluster_count})")

        result = kb.visualize_cluster_distribution(
            algorithm='kmeans',
            output_path='test_cluster_distribution.png'
        )

        if 'error' in result:
            print(f"  [ERROR] {result['error']}")
        else:
            print(f"  [OK] Distribution chart generated")
            print(f"    Output: {result['output_path']}")
            print(f"    Clusters: {result['num_clusters']}")
            print(f"    Cluster sizes: {result['cluster_sizes']}")
    else:
        print("  [SKIP] No K-Means clusters found")

    print("\n" + "="*60)
    print("[SUCCESS] Visualization Tests Complete!")
    print("="*60)
    print("\nGenerated files:")
    for filename in ['test_wordcloud.png', 'test_cluster_scatter.png',
                     'test_topic_heatmap.png', 'test_cluster_distribution.png']:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  - {filename} ({size:,} bytes)")

if __name__ == "__main__":
    test_all_visualizations()
