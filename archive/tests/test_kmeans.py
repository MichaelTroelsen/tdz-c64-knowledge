"""Test K-Means clustering implementation."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_kmeans_clustering():
    """Test K-Means clustering."""
    print("\n" + "="*60)
    print("Testing K-Means Clustering")
    print("="*60)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

    print(f"\n[OK] KnowledgeBase initialized")
    print(f"  Total documents: {len(kb.documents)}")

    # Run K-Means clustering with 5 clusters
    print("\n[RUN] Running K-Means clustering (5 clusters)...")

    result = kb.cluster_documents_kmeans(num_clusters=5, random_state=42)

    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return result

    print(f"\n[OK] K-Means clustering complete!")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Clusters: {result['num_clusters']}")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Assignments: {result['num_assignments']}")
    print(f"  Silhouette score: {result['silhouette_score']:.3f}")

    # Verify database storage
    cursor = kb.db_conn.cursor()

    # Check clusters table
    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'kmeans'"
    ).fetchone()[0]
    print(f"\n[OK] Database verification:")
    print(f"  Clusters stored: {cluster_count}")

    # Check document_clusters table
    assignment_count = cursor.execute(
        """SELECT COUNT(*) FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'kmeans'"""
    ).fetchone()[0]
    print(f"  Assignments stored: {assignment_count}")

    # Sample cluster assignments
    print(f"\n[CLUSTERS] Sample cluster assignments:")
    print("-" * 60)
    sample_assignments = cursor.execute(
        """SELECT c.cluster_number, COUNT(*) as doc_count
           FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'kmeans'
           GROUP BY c.cluster_number
           ORDER BY c.cluster_number"""
    ).fetchall()

    for cluster_num, doc_count in sample_assignments:
        print(f"  Cluster {cluster_num}: {doc_count} documents")

    print("\n" + "="*60)
    print("[SUCCESS] K-Means Test Complete!")
    print("="*60)

    return result

if __name__ == "__main__":
    test_kmeans_clustering()
