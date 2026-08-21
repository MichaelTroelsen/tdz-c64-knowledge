"""Test script for NMF, BERTopic, and model comparison (Phase 2, Task 2.3)"""

import sys
import os
sys.path.insert(0, '.')

from server import KnowledgeBase

def test_nmf_model():
    """Test NMF topic modeling."""
    print("\n" + "=" * 60)
    print("Testing NMF Topic Modeling")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Check we have documents
    num_docs = len(kb.documents)
    print(f"\n1. Document check: {num_docs} documents available")
    if num_docs == 0:
        print("   ERROR: No documents available")
        return False

    # Test NMF training
    print("\n2. Testing NMF model training...")
    try:
        results = kb.train_nmf_model(
            num_topics=5,
            max_iter=200,
            random_state=42,
            min_df=2,
            max_df=0.8,
            max_features=500,
            store_results=True
        )
        print(f"   OK: NMF model trained successfully")
        print(f"   - Model type: {results['model_type']}")
        print(f"   - Number of topics: {results['num_topics']}")
        print(f"   - Reconstruction error: {results['reconstruction_error']:.2f}")
        print(f"   - Vocabulary size: {results['vocabulary_size']}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
    except Exception as e:
        print(f"   ERROR training NMF model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display topics
    print("\n3. Extracted topics:")
    for i, topic in enumerate(results['topics'][:3], 1):
        print(f"\n   Topic {topic['topic_number']}:")
        print(f"   - ID: {topic['topic_id']}")
        print(f"   - Top words: {', '.join(topic['top_words'][:5])}")
        top_5_weights = dict(list(topic['word_weights'].items())[:5])
        print(f"   - Top weights: {top_5_weights}")

    # Display document-topic assignments
    print("\n4. Document-topic assignments (sample):")
    for i, doc_topic in enumerate(results['document_topics'][:3], 1):
        doc_id = doc_topic['doc_id']
        top_topics = doc_topic['topic_assignments'][:2]
        doc_title = kb.documents[doc_id].title if doc_id in kb.documents else "Unknown"
        print(f"\n   Document: {doc_title[:50]}")
        print(f"   - Doc ID: {doc_id[:12]}...")
        for topic_num, prob in top_topics:
            print(f"   - Topic {topic_num}: {prob:.3f}")

    # Verify database storage
    print("\n5. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    topic_count = cursor.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type = 'nmf'"
    ).fetchone()[0]
    print(f"   OK: Topics stored: {topic_count} topics")

    assignment_count = cursor.execute(
        "SELECT COUNT(*) FROM document_topics WHERE model_type = 'nmf'"
    ).fetchone()[0]
    print(f"   OK: Document assignments stored: {assignment_count} assignments")

    print("\n" + "=" * 60)
    print("NMF Topic Modeling Test: PASSED")
    print("=" * 60)

    # Close database connection
    kb.db_conn.close()
    return True


def test_bertopic_model():
    """Test BERTopic modeling."""
    print("\n" + "=" * 60)
    print("Testing BERTopic Modeling")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Check we have documents
    num_docs = len(kb.documents)
    print(f"\n1. Document check: {num_docs} documents available")

    # Test BERTopic training
    print("\n2. Testing BERTopic model training...")
    print("   (This may take a while - using embeddings + UMAP + HDBSCAN)")
    try:
        results = kb.train_bertopic_model(
            num_topics=5,
            min_topic_size=3,
            store_results=True
        )
        print(f"   OK: BERTopic model trained successfully")
        print(f"   - Model type: {results['model_type']}")
        print(f"   - Number of topics: {results['num_topics']}")
        print(f"   - Documents processed: {results['num_documents']}")
        print(f"   - Outliers: {results['outliers']} documents")
        print(f"   - Training time: {results['training_time']:.2f} seconds")
    except Exception as e:
        print(f"   ERROR training BERTopic model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display topics
    print("\n3. Extracted topics:")
    for i, topic in enumerate(results['topics'][:3], 1):
        print(f"\n   Topic {topic['topic_number']}:")
        print(f"   - ID: {topic['topic_id']}")
        print(f"   - Top words: {', '.join(topic['top_words'][:5])}")

    # Display document-topic assignments (non-outliers)
    print("\n4. Document-topic assignments (sample, excluding outliers):")
    non_outlier_docs = [
        dt for dt in results['document_topics']
        if dt['topic_assignments'] and dt['topic_assignments'][0][0] != -1
    ]
    for i, doc_topic in enumerate(non_outlier_docs[:3], 1):
        doc_id = doc_topic['doc_id']
        top_topics = doc_topic['topic_assignments'][:2]
        doc_title = kb.documents[doc_id].title if doc_id in kb.documents else "Unknown"
        print(f"\n   Document: {doc_title[:50]}")
        print(f"   - Doc ID: {doc_id[:12]}...")
        for topic_num, prob in top_topics:
            if topic_num != -1:
                print(f"   - Topic {topic_num}: {prob:.3f}")

    # Verify database storage
    print("\n5. Verifying database storage...")
    cursor = kb.db_conn.cursor()

    topic_count = cursor.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type = 'bertopic'"
    ).fetchone()[0]
    print(f"   OK: Topics stored: {topic_count} topics")

    assignment_count = cursor.execute(
        "SELECT COUNT(*) FROM document_topics WHERE model_type = 'bertopic'"
    ).fetchone()[0]
    print(f"   OK: Document assignments stored: {assignment_count} assignments")

    print("\n" + "=" * 60)
    print("BERTopic Modeling Test: PASSED")
    print("=" * 60)

    # Close database connection
    kb.db_conn.close()
    return True


def test_model_comparison():
    """Test topic model comparison."""
    print("\n" + "=" * 60)
    print("Testing Topic Model Comparison")
    print("=" * 60)

    data_dir = os.path.expanduser('~/.tdz-c64-knowledge')
    kb = KnowledgeBase(data_dir)

    # Run comparison
    print("\n1. Comparing topic models...")
    try:
        comparison = kb.compare_topic_models()
        print(f"   OK: Comparison completed")
        print(f"   - Total models found: {comparison['total_models']}")
    except Exception as e:
        print(f"   ERROR comparing models: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display model statistics
    print("\n2. Model statistics:")
    for model_type, stats in comparison['models'].items():
        if stats:
            print(f"\n   {model_type.upper()}:")
            print(f"   - Topics: {stats['num_topics']}")
            print(f"   - Documents covered: {stats['documents_covered']}")
            print(f"   - Avg docs per topic: {stats['avg_docs_per_topic']}")
            print(f"   - Vocabulary diversity: {stats['vocabulary_diversity']} unique words")
            print(f"   - Top topic: {stats['top_topics'][0]['top_words'][:5] if stats['top_topics'] else 'N/A'}")
        else:
            print(f"\n   {model_type.upper()}: Not trained")

    # Display comparison results
    print("\n3. Comparison results:")
    comp = comparison['comparison']
    if 'most_topics' in comp:
        print(f"\n   Most topics:")
        print(f"   - Model: {comp['most_topics']['model'].upper()}")
        print(f"   - Count: {comp['most_topics']['count']}")

    if 'best_coverage' in comp:
        print(f"\n   Best coverage:")
        print(f"   - Model: {comp['best_coverage']['model'].upper()}")
        print(f"   - Documents: {comp['best_coverage']['documents']}")

    if 'most_diverse_vocabulary' in comp:
        print(f"\n   Most diverse vocabulary:")
        print(f"   - Model: {comp['most_diverse_vocabulary']['model'].upper()}")
        print(f"   - Unique words: {comp['most_diverse_vocabulary']['unique_words']}")

    if 'recommended' in comp:
        print(f"\n   Recommended model:")
        print(f"   - Model: {comp['recommended']['model'].upper()}")
        print(f"   - Score: {comp['recommended']['score']}")
        print(f"   - Reason: {comp['recommended']['reason']}")

    print("\n" + "=" * 60)
    print("Topic Model Comparison Test: PASSED")
    print("=" * 60)

    # Close database connection
    kb.db_conn.close()
    return True


if __name__ == "__main__":
    success = True

    # Test NMF
    if not test_nmf_model():
        success = False

    # Test BERTopic
    if not test_bertopic_model():
        success = False

    # Test comparison
    if not test_model_comparison():
        success = False

    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED - Phase 2, Task 2.3 Complete")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)
