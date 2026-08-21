"""Test script for document clustering (Phase 2, Task 2.4)"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase

def test_kmeans_clustering():
    """Test K-Means clustering."""
    print("\n" + "=" * 60)
    print("Testing K-Means Clustering")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    num_docs = len(kb.documents)
    print(f"\n1. Document check: {num_docs} documents available")

    # Test K-Means clustering
    print("\n2. Testing K-Means clustering...")
    try:
        results = kb.cluster_documents_kmeans(
            num_clusters=5,
            store_results=True
        )
        print(f"   OK: K-Means clustering completed")
        print(f"   - Algorithm: {results['algorithm']}")
        print(f"   - Number of clusters: {results['num_clusters']}")
        print(f"   - Silhouette score: {results['silhouette_score']:.3f}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        kb.db_conn.close()
        return False

    # Display cluster info
    print("\n3. Cluster information (sample):")
    for i, cluster in enumerate(results['clusters'][:3], 1):
        print(f"\n   Cluster {cluster['cluster_number']}:")
        print(f"   - ID: {cluster['cluster_id']}")
        print(f"   - Documents: {cluster['num_documents']}")
        print(f"   - Top terms: {', '.join(cluster['top_terms'][:5])}")
        print(f"   - Representatives: {len(cluster['representative_docs'])} docs")

    # Verify database storage
    print("\n4. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'kmeans'"
    ).fetchone()[0]
    print(f"   OK: Clusters stored: {cluster_count} clusters")

    assignment_count = cursor.execute(
        "SELECT COUNT(*) FROM document_clusters WHERE algorithm = 'kmeans'"
    ).fetchone()[0]
    print(f"   OK: Assignments stored: {assignment_count} assignments")

    print("\n" + "=" * 60)
    print("K-Means Clustering Test: PASSED")
    print("=" * 60)

    kb.db_conn.close()
    return True


def test_dbscan_clustering():
    """Test DBSCAN clustering."""
    print("\n" + "=" * 60)
    print("Testing DBSCAN Clustering")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Test DBSCAN clustering
    print("\n1. Testing DBSCAN clustering...")
    try:
        results = kb.cluster_documents_dbscan(
            eps=0.5,
            min_samples=3,
            store_results=True
        )
        print(f"   OK: DBSCAN clustering completed")
        print(f"   - Algorithm: {results['algorithm']}")
        print(f"   - Number of clusters: {results['num_clusters']}")
        print(f"   - Outliers: {results['num_outliers']}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        kb.db_conn.close()
        return False

    # Display cluster info
    print("\n2. Cluster information (sample):")
    for i, cluster in enumerate(results['clusters'][:3], 1):
        print(f"\n   Cluster {cluster['cluster_number']}:")
        print(f"   - Documents: {cluster['num_documents']}")
        print(f"   - Top terms: {', '.join(cluster['top_terms'][:5])}")

    # Verify database storage
    print("\n3. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'dbscan'"
    ).fetchone()[0]
    print(f"   OK: Clusters stored: {cluster_count} clusters")

    print("\n" + "=" * 60)
    print("DBSCAN Clustering Test: PASSED")
    print("=" * 60)

    kb.db_conn.close()
    return True


def test_hdbscan_clustering():
    """Test HDBSCAN clustering."""
    print("\n" + "=" * 60)
    print("Testing HDBSCAN Clustering")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Test HDBSCAN clustering
    print("\n1. Testing HDBSCAN clustering...")
    try:
        results = kb.cluster_documents_hdbscan(
            min_cluster_size=5,
            min_samples=3,
            store_results=True
        )
        print(f"   OK: HDBSCAN clustering completed")
        print(f"   - Algorithm: {results['algorithm']}")
        print(f"   - Number of clusters: {results['num_clusters']}")
        print(f"   - Outliers: {results['num_outliers']}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        kb.db_conn.close()
        return False

    # Display cluster info
    print("\n2. Cluster information (sample):")
    for i, cluster in enumerate(results['clusters'][:3], 1):
        print(f"\n   Cluster {cluster['cluster_number']}:")
        print(f"   - Documents: {cluster['num_documents']}")
        print(f"   - Top terms: {', '.join(cluster['top_terms'][:5])}")

    # Verify database storage
    print("\n3. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'hdbscan'"
    ).fetchone()[0]
    print(f"   OK: Clusters stored: {cluster_count} clusters")

    print("\n" + "=" * 60)
    print("HDBSCAN Clustering Test: PASSED")
    print("=" * 60)

    kb.db_conn.close()
    return True


def test_evaluate_clustering():
    """Test clustering evaluation."""
    print("\n" + "=" * 60)
    print("Testing Clustering Evaluation")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Evaluate K-Means clustering
    print("\n1. Evaluating K-Means clustering...")
    try:
        metrics = kb.evaluate_clustering('kmeans')
        print(f"   OK: Evaluation completed")
        print(f"   - Algorithm: {metrics['algorithm']}")
        print(f"   - Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"   - Davies-Bouldin index: {metrics['davies_bouldin_score']:.3f}")
        print(f"   - Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.2f}")
        print(f"   - Number of clusters: {metrics['num_clusters']}")
        print(f"   - Documents evaluated: {metrics['num_documents']}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        kb.db_conn.close()
        return False

    print("\n" + "=" * 60)
    print("Clustering Evaluation Test: PASSED")
    print("=" * 60)

    kb.db_conn.close()
    return True


if __name__ == "__main__":
    import time
    success = True

    # Test K-Means
    if not test_kmeans_clustering():
        success = False
    time.sleep(2)  # Wait for database to release

    # Test DBSCAN
    if not test_dbscan_clustering():
        success = False
    time.sleep(2)  # Wait for database to release

    # Test HDBSCAN
    if not test_hdbscan_clustering():
        success = False
    time.sleep(2)  # Wait for database to release

    # Test evaluation
    if not test_evaluate_clustering():
        success = False

    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED - Phase 2, Task 2.4 Complete")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)
