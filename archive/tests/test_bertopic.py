"""Test BERTopic topic modeling implementation."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os

def test_bertopic_model():
    """Test BERTopic topic model training."""
    print("\n" + "="*60)
    print("Testing BERTopic Topic Modeling")
    print("="*60)

    # Initialize KB
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

    print(f"\n[OK] KnowledgeBase initialized")
    print(f"  Total documents: {len(kb.documents)}")

    # Train BERTopic model with 5 topics
    print("\n[RUN] Training BERTopic model (5 topics)...")
    print("  Note: This requires documents to have embeddings")

    result = kb.train_bertopic_model(
        num_topics=5,
        min_cluster_size=5,
        random_state=42
    )

    if 'error' in result:
        print(f"\n[ERROR] {result['error']}")
        return result

    print(f"\n[OK] BERTopic training complete!")
    print(f"  Model type: {result['model_type']}")
    print(f"  Topics discovered: {result['num_topics']}")
    print(f"  Documents processed: {result['num_documents']}")
    print(f"  Document-topic assignments: {result['num_assignments']}")
    print(f"  Outliers: {result['outliers']}")

    # Display discovered topics
    print(f"\n[TOPICS] Discovered Topics:")
    print("-" * 60)
    for i, topic in enumerate(result['topics'], 1):
        topic_num = topic['topic_number']
        words = topic['words'][:10]
        weights_dict = topic['weights']

        print(f"\nTopic {topic_num}:")
        word_str = ", ".join([f"{w}({weights_dict.get(w, 0):.3f})" for w in words])
        print(f"  {word_str}")

    # Verify database storage
    cursor = kb.db_conn.cursor()

    # Check topics table
    topic_count = cursor.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type = 'bertopic'"
    ).fetchone()[0]
    print(f"\n[OK] Database verification:")
    print(f"  Topics stored: {topic_count}")

    # Check document_topics table
    assignment_count = cursor.execute(
        """SELECT COUNT(*) FROM document_topics dt
           JOIN topics t ON dt.topic_id = t.topic_id
           WHERE t.model_type = 'bertopic'"""
    ).fetchone()[0]
    print(f"  Assignments stored: {assignment_count}")

    # Sample document assignments
    print(f"\n[DOCS] Sample document-topic assignments:")
    print("-" * 60)
    sample_assignments = cursor.execute(
        """SELECT d.title, t.topic_number, dt.probability
           FROM document_topics dt
           JOIN documents d ON dt.doc_id = d.doc_id
           JOIN topics t ON dt.topic_id = t.topic_id
           WHERE t.model_type = 'bertopic'
           ORDER BY dt.probability DESC
           LIMIT 10"""
    ).fetchall()

    for title, topic_num, prob in sample_assignments:
        print(f"  Topic {topic_num}: {title[:50]}... (prob={prob:.3f})")

    print("\n" + "="*60)
    print("[SUCCESS] BERTopic Test Complete!")
    print("="*60)

    return result

if __name__ == "__main__":
    test_bertopic_model()
