#!/usr/bin/env python3
"""
Script to enable semantic search and build embeddings index.
"""
import os
import sys
from pathlib import Path

# Set environment variable to enable semantic search
os.environ['USE_SEMANTIC_SEARCH'] = '1'
os.environ['SEMANTIC_MODEL'] = 'all-MiniLM-L6-v2'  # Fast, efficient model

# Import server after setting env vars
from server import KnowledgeBase

def main():
    print("=" * 70)
    print("TDZ C64 Knowledge - Semantic Search Enablement")
    print("=" * 70)
    print()

    # Initialize knowledge base with semantic search enabled
    data_dir = Path(os.getenv('TDZ_DATA_DIR', Path.home() / '.tdz-c64-knowledge'))
    print(f"Data directory: {data_dir}")
    print()

    print("Initializing knowledge base with semantic search...")
    kb = KnowledgeBase(str(data_dir))

    if not kb.use_semantic:
        print("ERROR: Semantic search failed to initialize!")
        print("Check that sentence-transformers and faiss-cpu are installed:")
        print("  pip install sentence-transformers faiss-cpu")
        return 1

    print(f"[OK] Semantic search enabled")
    print(f"[OK] Model: {kb.embeddings_model}")
    print()

    # Check if embeddings index exists
    if kb.embeddings_index is None:
        print("No embeddings index found. Building index...")
        print(f"This will process {len(kb.documents)} documents with {len(kb._get_chunks_db())} chunks.")
        print("This may take several minutes...")
        print()

        # Build embeddings
        kb._build_embeddings()
        print()
        print("[OK] Embeddings index built and saved")
    else:
        print(f"[OK] Embeddings index loaded ({kb.embeddings_index.ntotal} vectors)")

    print()
    print("=" * 70)
    print("Semantic Search Test")
    print("=" * 70)
    print()

    # Test semantic search
    test_queries = [
        "How do I create graphics on the screen?",
        "Playing sounds and music",
        "Programming assembly language",
    ]

    for query in test_queries:
        print(f"Query: \"{query}\"")
        results = kb.semantic_search(query, max_results=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['filename']} (score: {result['score']:.3f})")
            snippet = result['snippet'][:100] + "..." if len(result['snippet']) > 100 else result['snippet']
            print(f"     {snippet}")
        print()

    print("=" * 70)
    print("Success! Semantic search is ready to use.")
    print()
    print("To enable semantic search in the MCP server, set environment variable:")
    print("  USE_SEMANTIC_SEARCH=1")
    print()
    print("Add this to your Claude Code MCP configuration:")
    print('  "env": {')
    print('    "USE_SEMANTIC_SEARCH": "1"')
    print('  }')
    print("=" * 70)

    kb.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
