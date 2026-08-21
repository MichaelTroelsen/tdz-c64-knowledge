"""Test DBSCAN clustering implementation."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_dbscan_clustering():
    """Test DBSCAN clustering."""
    print("\n" + "="*60)
    print("Testing DBSCAN Clustering")
    print("="*60)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

    print(f"\n[OK] KnowledgeBase initialized")
    print(f"  Total documents: {len(kb.documents)}")

    # Run DBSCAN clustering
    print("\n[RUN] Running DBSCAN clustering (eps=0.5, min_samples=5)...")

    result = kb.cluster_documents_dbscan(eps=0.5, min_samples=5)

    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return result

    print(f"\n[OK] DBSCAN clustering complete!")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Clusters found: {result['num_clusters']}")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Assignments: {result['num_assignments']}")
    print(f"  Outliers: {result['num_outliers']}")
    print(f"  Silhouette score: {result['silhouette_score']:.3f}")

    # Verify database storage
    cursor = kb.db_conn.cursor()

    # Check clusters table
    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'dbscan'"
    ).fetchone()[0]
    print(f"\n[OK] Database verification:")
    print(f"  Clusters stored: {cluster_count}")

    # Check document_clusters table
    assignment_count = cursor.execute(
        """SELECT COUNT(*) FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'dbscan'"""
    ).fetchone()[0]
    print(f"  Assignments stored: {assignment_count}")

    # Sample cluster assignments
    print(f"\n[CLUSTERS] Sample cluster assignments:")
    print("-" * 60)
    sample_assignments = cursor.execute(
        """SELECT c.cluster_number, COUNT(*) as doc_count
           FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'dbscan'
           GROUP BY c.cluster_number
           ORDER BY doc_count DESC
           LIMIT 10"""
    ).fetchall()

    for cluster_num, doc_count in sample_assignments:
        print(f"  Cluster {cluster_num}: {doc_count} documents")

    print("\n" + "="*60)
    print("[SUCCESS] DBSCAN Test Complete!")
    print("="*60)

    return result

if __name__ == "__main__":
    test_dbscan_clustering()
