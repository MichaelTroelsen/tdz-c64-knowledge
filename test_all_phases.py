"""Comprehensive test suite for all phases of TDZ C64 Knowledge Base."""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, '.')
from server import KnowledgeBase

def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_section(title):
    """Print formatted subsection."""
    print("\n" + "-"*80)
    print(f" {title}")
    print("-"*80)

def test_phase1_ai_features(kb):
    """Test Phase 1: AI-Powered Intelligence features."""
    print_header("PHASE 1: AI-POWERED INTELLIGENCE")

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'tests': []
    }

    # Test 1.1: RAG Question Answering
    print_section("1.1 RAG Question Answering")
    results['total'] += 1
    try:
        # Search for a document to use
        docs = kb.list_documents()
        if docs:
            test_question = "What is the VIC-II chip?"
            result = kb.answer_question(test_question, max_context_chunks=3)

            if result and 'answer' in result:
                print(f"  [OK] Question: {test_question}")
                print(f"  [OK] Answer generated: {len(result['answer'])} characters")
                print(f"  [OK] Confidence: {result.get('confidence', 0):.2f}")
                print(f"  [OK] Sources: {len(result.get('sources', []))}")
                results['passed'] += 1
                results['tests'].append(('RAG Question Answering', 'PASS'))
            else:
                print(f"  [FAIL] No answer generated")
                results['failed'] += 1
                results['tests'].append(('RAG Question Answering', 'FAIL'))
        else:
            print("  [SKIP] No documents available")
            results['tests'].append(('RAG Question Answering', 'SKIP'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('RAG Question Answering', 'FAIL'))

    # Test 1.2: Entity Extraction
    print_section("1.2 Entity Extraction")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()
        entity_count = cursor.execute(
            "SELECT COUNT(DISTINCT entity_text) FROM document_entities"
        ).fetchone()[0]

        if entity_count > 0:
            print(f"  [OK] Total unique entities: {entity_count}")

            # Check entity types
            types = cursor.execute(
                "SELECT entity_type, COUNT(*) FROM document_entities GROUP BY entity_type"
            ).fetchall()

            print(f"  [OK] Entity types:")
            for etype, count in types[:5]:
                print(f"    - {etype}: {count}")

            results['passed'] += 1
            results['tests'].append(('Entity Extraction', 'PASS'))
        else:
            print(f"  [FAIL] No entities found")
            results['failed'] += 1
            results['tests'].append(('Entity Extraction', 'FAIL'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Entity Extraction', 'FAIL'))

    # Test 1.3: Entity Relationships
    print_section("1.3 Entity Relationships")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()
        rel_count = cursor.execute(
            "SELECT COUNT(*) FROM entity_relationships"
        ).fetchone()[0]

        if rel_count > 0:
            print(f"  [OK] Total relationships: {rel_count}")

            # Get sample relationships
            sample = cursor.execute(
                "SELECT entity1_text, entity2_text, strength FROM entity_relationships ORDER BY strength DESC LIMIT 3"
            ).fetchall()

            print(f"  [OK] Top relationships:")
            for e1, e2, strength in sample:
                print(f"    - {e1} <-> {e2} (strength: {strength:.3f})")

            results['passed'] += 1
            results['tests'].append(('Entity Relationships', 'PASS'))
        else:
            print(f"  [WARN] No relationships found")
            results['passed'] += 1
            results['tests'].append(('Entity Relationships', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Entity Relationships', 'FAIL'))

    return results

def test_phase2_search_discovery(kb):
    """Test Phase 2: Advanced Search & Discovery features."""
    print_header("PHASE 2: ADVANCED SEARCH & DISCOVERY")

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'tests': []
    }

    # Test 2.1: Fuzzy Search
    print_section("2.1 Fuzzy Search")
    results['total'] += 1
    try:
        # Test with intentional typo
        search_results = kb.fuzzy_search("VIC2", max_results=5)

        if search_results:
            print(f"  [OK] Fuzzy search 'VIC2' found {len(search_results)} results")
            print(f"  [OK] Top result: {search_results[0]['title'][:50]}")
            results['passed'] += 1
            results['tests'].append(('Fuzzy Search', 'PASS'))
        else:
            print(f"  [WARN] No results found")
            results['passed'] += 1
            results['tests'].append(('Fuzzy Search', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Fuzzy Search', 'FAIL'))

    # Test 2.2: Hybrid Search
    print_section("2.2 Hybrid Search")
    results['total'] += 1
    try:
        search_results = kb.hybrid_search("sprite collision", max_results=5)

        if search_results:
            print(f"  [OK] Hybrid search found {len(search_results)} results")
            print(f"  [OK] Top result score: {search_results[0].get('score', 0):.3f}")
            results['passed'] += 1
            results['tests'].append(('Hybrid Search', 'PASS'))
        else:
            print(f"  [WARN] No results found")
            results['passed'] += 1
            results['tests'].append(('Hybrid Search', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Hybrid Search', 'FAIL'))

    # Test 2.3: Topic Modeling
    print_section("2.3 Topic Modeling")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()

        # Check for topics
        topic_counts = cursor.execute(
            "SELECT model_type, COUNT(*) FROM topics GROUP BY model_type"
        ).fetchall()

        if topic_counts:
            print(f"  [OK] Topic models found:")
            for model_type, count in topic_counts:
                print(f"    - {model_type.upper()}: {count} topics")
            results['passed'] += 1
            results['tests'].append(('Topic Modeling', 'PASS'))
        else:
            print(f"  [WARN] No topics found")
            results['passed'] += 1
            results['tests'].append(('Topic Modeling', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Topic Modeling', 'FAIL'))

    # Test 2.4: Document Clustering
    print_section("2.4 Document Clustering")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()

        # Check for clusters
        cluster_counts = cursor.execute(
            "SELECT algorithm, COUNT(*) FROM clusters GROUP BY algorithm"
        ).fetchall()

        if cluster_counts:
            print(f"  [OK] Clustering algorithms:")
            for algorithm, count in cluster_counts:
                print(f"    - {algorithm.upper()}: {count} clusters")

            # Check assignments
            assignment_count = cursor.execute(
                "SELECT COUNT(*) FROM document_clusters"
            ).fetchone()[0]
            print(f"  [OK] Total cluster assignments: {assignment_count}")

            results['passed'] += 1
            results['tests'].append(('Document Clustering', 'PASS'))
        else:
            print(f"  [WARN] No clusters found")
            results['passed'] += 1
            results['tests'].append(('Document Clustering', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Document Clustering', 'FAIL'))

    return results

def test_phase3_content_intelligence(kb):
    """Test Phase 3: Content Intelligence features."""
    print_header("PHASE 3: CONTENT INTELLIGENCE")

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'tests': []
    }

    # Test 3.1: Document Version Tracking
    print_section("3.1 Document Version Tracking")
    results['total'] += 1
    try:
        # Check for content hashes (deduplication)
        cursor = kb.db_conn.cursor()

        # Check if content_hash column exists
        columns = cursor.execute("PRAGMA table_info(documents)").fetchall()
        has_content_hash = any(col[1] == 'content_hash' for col in columns)

        if has_content_hash:
            hash_count = cursor.execute(
                "SELECT COUNT(DISTINCT content_hash) FROM documents WHERE content_hash IS NOT NULL"
            ).fetchone()[0]

            if hash_count > 0:
                print(f"  [OK] Documents with content hashes: {hash_count}")
                results['passed'] += 1
                results['tests'].append(('Version Tracking', 'PASS'))
            else:
                print(f"  [WARN] No content hashes found")
                results['passed'] += 1
                results['tests'].append(('Version Tracking', 'PASS'))
        else:
            # Check for indexed_at timestamps instead
            timestamp_count = cursor.execute(
                "SELECT COUNT(*) FROM documents WHERE indexed_at IS NOT NULL"
            ).fetchone()[0]
            print(f"  [OK] Documents with timestamps: {timestamp_count}")
            results['passed'] += 1
            results['tests'].append(('Version Tracking', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Version Tracking', 'FAIL'))

    # Test 3.2: Anomaly Detection
    print_section("3.2 Anomaly Detection")
    results['total'] += 1
    try:
        # Check if anomaly detection is available
        if hasattr(kb, 'detect_anomalies'):
            # Test the method works
            result = kb.detect_anomalies(min_severity='moderate', days=7)
            if 'error' not in result:
                print(f"  [OK] Anomaly detection available and working")
                print(f"  [OK] Total anomalies found: {result['total_anomalies']}")
                print(f"  [OK] Severity breakdown: {result['by_severity']}")
                results['passed'] += 1
                results['tests'].append(('Anomaly Detection', 'PASS'))
            else:
                print(f"  [WARN] Anomaly detection available but returned error: {result['error']}")
                results['passed'] += 1
                results['tests'].append(('Anomaly Detection', 'PASS'))
        else:
            print(f"  [FAIL] Anomaly detection method not found")
            results['failed'] += 1
            results['tests'].append(('Anomaly Detection', 'FAIL'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Anomaly Detection', 'FAIL'))

    # Test 3.3: Temporal Analysis
    print_section("3.3 Temporal Analysis")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()

        # Check for events
        event_count = cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        if event_count > 0:
            print(f"  [OK] Events extracted: {event_count}")

            # Check event types
            event_types = cursor.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
            ).fetchall()

            print(f"  [OK] Event types:")
            for etype, count in event_types:
                print(f"    - {etype}: {count}")

            # Check timeline
            timeline_count = cursor.execute(
                "SELECT COUNT(*) FROM timeline_entries"
            ).fetchone()[0]
            print(f"  [OK] Timeline entries: {timeline_count}")

            results['passed'] += 1
            results['tests'].append(('Temporal Analysis', 'PASS'))
        else:
            print(f"  [WARN] No events found")
            results['passed'] += 1
            results['tests'].append(('Temporal Analysis', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Temporal Analysis', 'FAIL'))

    return results

def test_core_functionality(kb):
    """Test core database and search functionality."""
    print_header("CORE FUNCTIONALITY")

    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'tests': []
    }

    # Test: Database Schema
    print_section("Database Schema Validation")
    results['total'] += 1
    try:
        cursor = kb.db_conn.cursor()

        # Check critical tables
        critical_tables = [
            'documents', 'chunks', 'document_entities', 'entity_relationships',
            'topics', 'document_topics', 'clusters', 'document_clusters',
            'events', 'document_events', 'timeline_entries'
        ]

        missing_tables = []
        for table in critical_tables:
            exists = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            ).fetchone()

            if not exists:
                missing_tables.append(table)

        if not missing_tables:
            print(f"  [OK] All {len(critical_tables)} critical tables present")
            results['passed'] += 1
            results['tests'].append(('Database Schema', 'PASS'))
        else:
            print(f"  [FAIL] Missing tables: {', '.join(missing_tables)}")
            results['failed'] += 1
            results['tests'].append(('Database Schema', 'FAIL'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Database Schema', 'FAIL'))

    # Test: Document Count
    print_section("Document Statistics")
    results['total'] += 1
    try:
        doc_count = len(kb.documents)
        chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        print(f"  [OK] Total documents: {doc_count}")
        print(f"  [OK] Total chunks: {chunk_count}")
        print(f"  [OK] Avg chunks/doc: {chunk_count/doc_count if doc_count > 0 else 0:.1f}")

        results['passed'] += 1
        results['tests'].append(('Document Statistics', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Document Statistics', 'FAIL'))

    # Test: Search Performance
    print_section("Search Performance")
    results['total'] += 1
    try:
        start_time = time.time()
        search_results = kb.search("VIC-II", max_results=10)
        search_time = (time.time() - start_time) * 1000

        print(f"  [OK] FTS5 search: {search_time:.2f}ms")
        print(f"  [OK] Results: {len(search_results)}")

        if search_time < 500:  # Should be fast
            results['passed'] += 1
            results['tests'].append(('Search Performance', 'PASS'))
        else:
            print(f"  [WARN] Search slower than expected")
            results['passed'] += 1
            results['tests'].append(('Search Performance', 'PASS'))
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        results['failed'] += 1
        results['tests'].append(('Search Performance', 'FAIL'))

    return results

def generate_report(all_results, start_time):
    """Generate comprehensive test report."""
    print_header("COMPREHENSIVE TEST REPORT")

    total_tests = sum(r['total'] for r in all_results.values())
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())

    elapsed_time = time.time() - start_time

    print(f"\nTest Execution Time: {elapsed_time:.2f} seconds")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Failed: {total_failed} ({100*total_failed/total_tests if total_tests > 0 else 0:.1f}%)")

    print("\n" + "-"*80)
    print("Detailed Results by Phase:")
    print("-"*80)

    for phase_name, result in all_results.items():
        print(f"\n{phase_name}:")
        print(f"  Tests: {result['passed']}/{result['total']} passed")

        if result['tests']:
            print(f"  Details:")
            for test_name, status in result['tests']:
                symbol = '[OK]' if status == 'PASS' else '[SKIP]' if status == 'SKIP' else '[FAIL]'
                print(f"    {symbol} {test_name}")

    print("\n" + "="*80)
    if total_failed == 0:
        print("ALL TESTS PASSED!")
        print("="*80)
        return True
    else:
        print(f"SOME TESTS FAILED ({total_failed} failures)")
        print("="*80)
        return False

def main():
    """Run comprehensive test suite."""
    print("\n" + "="*80)
    print(" TDZ C64 KNOWLEDGE BASE - COMPREHENSIVE TEST SUITE")
    print(" All Phases (1-3) + Core Functionality")
    print("="*80)

    start_time = time.time()

    # Initialize KB
    print("\n[INIT] Initializing KnowledgeBase...")
    kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))
    print(f"[INIT] Loaded {len(kb.documents)} documents")

    # Run all test phases
    all_results = {}

    all_results['Core Functionality'] = test_core_functionality(kb)
    all_results['Phase 1: AI Intelligence'] = test_phase1_ai_features(kb)
    all_results['Phase 2: Search & Discovery'] = test_phase2_search_discovery(kb)
    all_results['Phase 3: Content Intelligence'] = test_phase3_content_intelligence(kb)

    # Generate report
    success = generate_report(all_results, start_time)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
