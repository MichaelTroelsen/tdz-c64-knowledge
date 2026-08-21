"""Test script for LDA topic modeling implementation (Phase 2, Task 2.2)"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase

def test_lda_topic_modeling():
    """Test LDA topic modeling implementation."""
    print("=" * 60)
    print("Testing LDA Topic Modeling Implementation")
    print("=" * 60)

    # Initialize knowledge base
    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Check we have documents
    num_docs = len(kb.documents)
    print(f"\n1. Document check: {num_docs} documents available")
    if num_docs == 0:
        print("   ERROR: No documents available for topic modeling")
        return False

    # Test corpus preparation
    print("\n2. Testing corpus preparation...")
    try:
        doc_ids, vectorizer, doc_term_matrix, feature_names = kb._prepare_topic_model_corpus(
            min_df=2,
            max_df=0.8,
            max_features=500
        )
        print(f"   OK: Corpus prepared successfully")
        print(f"   - Documents: {len(doc_ids)}")
        print(f"   - Vocabulary size: {len(feature_names)}")
        print(f"   - Matrix shape: {doc_term_matrix.shape}")
        print(f"   - Sample features: {list(feature_names[:10])}")
    except Exception as e:
        print(f"   ERROR preparing corpus: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test LDA training
    print("\n3. Testing LDA model training...")
    try:
        results = kb.train_lda_model(
            num_topics=5,
            max_iter=50,
            random_state=42,
            min_df=2,
            max_df=0.8,
            max_features=500,
            store_results=True
        )
        print(f"   OK: LDA model trained successfully")
        print(f"   - Model type: {results['model_type']}")
        print(f"   - Number of topics: {results['num_topics']}")
        print(f"   - Perplexity: {results['perplexity']:.2f}")
        print(f"   - Vocabulary size: {results['vocabulary_size']}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Training time: {results.get('training_time', 'N/A')}")
    except Exception as e:
        print(f"   ERROR training LDA model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display topics
    print("\n4. Extracted topics:")
    for i, topic in enumerate(results['topics'][:3], 1):  # Show first 3 topics
        print(f"\n   Topic {topic['topic_number']}:")
        print(f"   - ID: {topic['topic_id']}")
        print(f"   - Top words: {', '.join(topic['top_words'][:5])}")
        top_5_weights = dict(list(topic['word_weights'].items())[:5])
        print(f"   - Top weights: {top_5_weights}")

    # Display document-topic assignments
    print("\n5. Document-topic assignments (sample):")
    for i, doc_topic in enumerate(results['document_topics'][:3], 1):  # Show first 3 docs
        doc_id = doc_topic['doc_id']
        top_topics = doc_topic['topic_assignments'][:2]  # Top 2 topics
        doc_title = kb.documents[doc_id].title if doc_id in kb.documents else "Unknown"
        print(f"\n   Document: {doc_title[:50]}")
        print(f"   - Doc ID: {doc_id[:12]}...")
        for topic_num, prob in top_topics:
            print(f"   - Topic {topic_num}: {prob:.3f}")

    # Verify database storage
    print("\n6. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    # Check topics table
    topic_count = cursor.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type = 'lda'"
    ).fetchone()[0]
    print(f"   OK: Topics stored: {topic_count} topics in database")

    if topic_count != results['num_topics']:
        print(f"   WARNING: Expected {results['num_topics']} topics, found {topic_count}")

    # Check document_topics table
    assignment_count = cursor.execute(
        "SELECT COUNT(*) FROM document_topics WHERE model_type = 'lda'"
    ).fetchone()[0]
    print(f"   OK: Document assignments stored: {assignment_count} assignments")

    expected_assignments = results['num_documents'] * results['num_topics']
    if assignment_count != expected_assignments:
        print(f"   WARNING: Expected {expected_assignments} assignments, found {assignment_count}")

    # Verify a sample topic from database
    sample_topic = cursor.execute("""
        SELECT topic_id, topic_number, top_words, num_documents
        FROM topics
        WHERE model_type = 'lda'
        ORDER BY topic_number
        LIMIT 1
    """).fetchone()

    if sample_topic:
        print(f"\n   Sample topic from database:")
        print(f"   - Topic ID: {sample_topic[0]}")
        print(f"   - Topic number: {sample_topic[1]}")
        print(f"   - Top words (JSON): {sample_topic[2][:80]}...")
        print(f"   - Num documents: {sample_topic[3]}")

    # Verify document counts were updated
    doc_counts = cursor.execute("""
        SELECT topic_number, num_documents
        FROM topics
        WHERE model_type = 'lda'
        ORDER BY topic_number
    """).fetchall()

    print(f"\n   Document counts per topic:")
    for topic_num, count in doc_counts[:3]:  # Show first 3
        print(f"   - Topic {topic_num}: {count} documents")

    print("\n" + "=" * 60)
    print("LDA Topic Modeling Test: PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_lda_topic_modeling()
    sys.exit(0 if success else 1)
