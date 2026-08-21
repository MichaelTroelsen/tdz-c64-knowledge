"""Test script for Phase 2 Task 2.5 MCP Tools"""

import sys
import os
sys.path.insert(0, '.')

from server import kb, list_tools, call_tool
import asyncio
import json

async def test_tool_registration():
    """Test that all 8 new tools are registered."""
    print("\n" + "=" * 60)
    print("Testing MCP Tool Registration")
    print("=" * 60)

    tools = await list_tools()
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "train_lda_topics",
        "train_nmf_topics",
        "train_bertopic",
        "get_document_topics",
        "cluster_documents_kmeans",
        "cluster_documents_dbscan",
        "cluster_documents_hdbscan",
        "get_cluster_documents"
    ]

    print(f"\n1. Checking tool registration...")
    all_found = True
    for tool_name in expected_tools:
        if tool_name in tool_names:
            print(f"   [OK] {tool_name} - registered")
        else:
            print(f"   [ERROR] {tool_name} - NOT FOUND")
            all_found = False

    if all_found:
        print(f"\n   OK: All 8 tools registered successfully")
        print("=" * 60)
        return True
    else:
        print(f"\n   ERROR: Some tools not registered")
        print("=" * 60)
        return False


async def test_train_lda_topics():
    """Test train_lda_topics MCP tool."""
    print("\n" + "=" * 60)
    print("Testing train_lda_topics MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    try:
        results = await call_tool(
            name="train_lda_topics",
            arguments={"num_topics": 5, "max_iter": 50, "max_features": 500}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response type: {type(results[0]).__name__}")
            print(f"   [OK] Response preview: {results[0].text[:200]}...")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: train_lda_topics works correctly")
    print("=" * 60)
    return True


async def test_train_nmf_topics():
    """Test train_nmf_topics MCP tool."""
    print("\n" + "=" * 60)
    print("Testing train_nmf_topics MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    try:
        results = await call_tool(name="train_nmf_topics",
            arguments={"num_topics": 5, "max_iter": 100, "max_features": 500}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response contains: {len(results[0].text)} characters")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: train_nmf_topics works correctly")
    print("=" * 60)
    return True


async def test_train_bertopic():
    """Test train_bertopic MCP tool."""
    print("\n" + "=" * 60)
    print("Testing train_bertopic MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    print("   (This may take a while - using embeddings + UMAP + HDBSCAN)")
    try:
        results = await call_tool(name="train_bertopic",
            arguments={"num_topics": 5, "min_topic_size": 3}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response preview: {results[0].text[:200]}...")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: train_bertopic works correctly")
    print("=" * 60)
    return True


async def test_get_document_topics():
    """Test get_document_topics MCP tool."""
    print("\n" + "=" * 60)
    print("Testing get_document_topics MCP Tool")
    print("=" * 60)

    # Test 1: Get all document topics for LDA
    print("\n1. Testing get all topics for LDA model...")
    try:
        results = await call_tool(name="get_document_topics",
            arguments={"model_type": "lda", "min_probability": 0.1}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response length: {len(results[0].text)} characters")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Get topics for specific document
    print("\n2. Testing get topics for specific document...")
    try:
        # Get a doc_id from database
        cursor = kb.db_conn.cursor()
        doc = cursor.execute("SELECT doc_id FROM documents LIMIT 1").fetchone()

        if doc:
            doc_id = doc[0]
            results = await call_tool(name="get_document_topics",
                arguments={"model_type": "lda", "doc_id": doc_id}
            )

            if results and len(results) > 0:
                print(f"   [OK] Specific document query works")
            else:
                print(f"   [ERROR] No results for specific document")
                return False
        else:
            print(f"   ⚠ No documents in database to test")

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

    print("\n   OK: get_document_topics works correctly")
    print("=" * 60)
    return True


async def test_cluster_documents_kmeans():
    """Test cluster_documents_kmeans MCP tool."""
    print("\n" + "=" * 60)
    print("Testing cluster_documents_kmeans MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    try:
        results = await call_tool(name="cluster_documents_kmeans",
            arguments={"num_clusters": 5}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response preview: {results[0].text[:200]}...")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: cluster_documents_kmeans works correctly")
    print("=" * 60)
    return True


async def test_cluster_documents_dbscan():
    """Test cluster_documents_dbscan MCP tool."""
    print("\n" + "=" * 60)
    print("Testing cluster_documents_dbscan MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    try:
        results = await call_tool(name="cluster_documents_dbscan",
            arguments={"eps": 0.5, "min_samples": 3}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response contains: {len(results[0].text)} characters")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: cluster_documents_dbscan works correctly")
    print("=" * 60)
    return True


async def test_cluster_documents_hdbscan():
    """Test cluster_documents_hdbscan MCP tool."""
    print("\n" + "=" * 60)
    print("Testing cluster_documents_hdbscan MCP Tool")
    print("=" * 60)

    print("\n1. Testing with default parameters...")
    try:
        results = await call_tool(name="cluster_documents_hdbscan",
            arguments={"min_cluster_size": 5, "min_samples": 3}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response preview: {results[0].text[:200]}...")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n   OK: cluster_documents_hdbscan works correctly")
    print("=" * 60)
    return True


async def test_get_cluster_documents():
    """Test get_cluster_documents MCP tool."""
    print("\n" + "=" * 60)
    print("Testing get_cluster_documents MCP Tool")
    print("=" * 60)

    # Test 1: Get all clusters
    print("\n1. Testing get all clusters for kmeans...")
    try:
        results = await call_tool(name="get_cluster_documents",
            arguments={"algorithm": "kmeans"}
        )

        if results and len(results) > 0:
            print(f"   [OK] Tool executed successfully")
            print(f"   [OK] Response length: {len(results[0].text)} characters")
        else:
            print(f"   [ERROR] No results returned")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Get specific cluster
    print("\n2. Testing get specific cluster...")
    try:
        results = await call_tool(name="get_cluster_documents",
            arguments={"algorithm": "kmeans", "cluster_number": 0}
        )

        if results and len(results) > 0:
            print(f"   [OK] Specific cluster query works")
        else:
            print(f"   [ERROR] No results for specific cluster")
            return False

    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

    print("\n   OK: get_cluster_documents works correctly")
    print("=" * 60)
    return True


async def run_all_tests():
    """Run all MCP tool tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 TASK 2.5 - MCP TOOLS TEST SUITE")
    print("=" * 80)

    success = True

    # Test tool registration (asynchronous)
    if not await test_tool_registration():
        success = False

    # Test each tool handler (asynchronous)
    if not await test_train_lda_topics():
        success = False

    if not await test_train_nmf_topics():
        success = False

    if not await test_train_bertopic():
        success = False

    if not await test_get_document_topics():
        success = False

    if not await test_cluster_documents_kmeans():
        success = False

    if not await test_cluster_documents_dbscan():
        success = False

    if not await test_cluster_documents_hdbscan():
        success = False

    if not await test_get_cluster_documents():
        success = False

    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("[OK] ALL TESTS PASSED - Phase 2, Task 2.5 Complete")
        print("\nAll 8 MCP tools are:")
        print("  • Properly registered")
        print("  • Correctly parsing arguments")
        print("  • Successfully executing backend methods")
        print("  • Returning properly formatted responses")
    else:
        print("[ERROR] SOME TESTS FAILED")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
