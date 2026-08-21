"""Test HDBSCAN clustering implementation."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_hdbscan_clustering():
    """Test HDBSCAN clustering."""
    print("\n" + "="*60)
    print("Testing HDBSCAN Clustering")
    print("="*60)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

    print(f"\n[OK] KnowledgeBase initialized")
    print(f"  Total documents: {len(kb.documents)}")

    # Run HDBSCAN clustering
    print("\n[RUN] Running HDBSCAN clustering (min_cluster_size=5)...")

    result = kb.cluster_documents_hdbscan(min_cluster_size=5)

    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return result

    print(f"\n[OK] HDBSCAN clustering complete!")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Clusters found: {result['num_clusters']}")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Assignments: {result['num_assignments']}")
    print(f"  Outliers: {result['num_outliers']}")

    if result['cluster_persistence']:
        print(f"  Cluster persistence scores: {len(result['cluster_persistence'])} clusters")

    # Verify database storage
    cursor = kb.db_conn.cursor()

    # Check clusters table
    cluster_count = cursor.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm = 'hdbscan'"
    ).fetchone()[0]
    print(f"\n[OK] Database verification:")
    print(f"  Clusters stored: {cluster_count}")

    # Check document_clusters table
    assignment_count = cursor.execute(
        """SELECT COUNT(*) FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'hdbscan'"""
    ).fetchone()[0]
    print(f"  Assignments stored: {assignment_count}")

    # Sample cluster assignments
    print(f"\n[CLUSTERS] Sample cluster assignments:")
    print("-" * 60)
    sample_assignments = cursor.execute(
        """SELECT c.cluster_number, COUNT(*) as doc_count
           FROM document_clusters dc
           JOIN clusters c ON dc.cluster_id = c.cluster_id
           WHERE c.algorithm = 'hdbscan'
           GROUP BY c.cluster_number
           ORDER BY doc_count DESC
           LIMIT 10"""
    ).fetchall()

    for cluster_num, doc_count in sample_assignments:
        print(f"  Cluster {cluster_num}: {doc_count} documents")

    print("\n" + "="*60)
    print("[SUCCESS] HDBSCAN Test Complete!")
    print("="*60)

    return result

if __name__ == "__main__":
    test_hdbscan_clustering()
