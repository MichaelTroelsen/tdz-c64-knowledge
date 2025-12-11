#!/usr/bin/env python3
"""
Debug BM25 scoring to understand why tests are failing.
"""

import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase
from rank_bm25 import BM25Okapi

def test_bm25_scoring():
    """Test BM25 with sample data."""

    # Create sample corpus (similar to test data)
    sample_texts = [
        "VIC-II Graphics Chip The VIC-II chip controls all graphics",
        "The SID chip handles sound generation with 3 voices",
        "Sprites are movable objects on screen controlled by VIC-II",
        "Memory map shows VIC-II registers at $D000-$D02E",
    ]

    print("=" * 60)
    print("BM25 Debug Test")
    print("=" * 60)

    # Tokenize corpus
    tokenized_corpus = [text.lower().split() for text in sample_texts]
    print(f"\nCorpus ({len(sample_texts)} documents):")
    for i, tokens in enumerate(tokenized_corpus):
        print(f"  Doc {i}: {tokens[:10]}...")  # First 10 tokens

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Test queries
    test_queries = [
        "VIC-II",
        "VIC-II chip",
        "graphics",
        "sprite",
        "SID sound",
    ]

    print("\n" + "=" * 60)
    print("Query Results")
    print("=" * 60)

    for query in test_queries:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        print(f"\nQuery: '{query}'")
        print(f"Tokenized: {tokenized_query}")
        print(f"Scores: {scores}")
        print(f"Max score: {max(scores):.6f}")
        print(f"Non-zero scores: {sum(1 for s in scores if s > 0)}")

        # Show top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        print("Top 3 results:")
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                print(f"  {rank}. Doc {idx} (score={scores[idx]:.6f}): {sample_texts[idx][:50]}...")

    print("\n" + "=" * 60)
    print("Testing with actual KnowledgeBase")
    print("=" * 60)

    # Create temp directory and KB
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(tmpdir)

        # Add a test document
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("""
VIC-II Graphics Chip

The VIC-II chip controls all graphics and video output on the Commodore 64.
It has 47 registers mapped to memory locations $D000-$D02E.

The SID chip handles sound generation with 3 voices.
Sprites are movable objects controlled by the VIC-II.
        """)

        kb.add_document(str(test_file), "Test Doc", ["test"])

        print(f"\nAdded document with {len(kb.chunks)} chunks")
        print(f"BM25 index: {kb.bm25 is not None}")

        if kb.bm25 is not None:
            # Test search with BM25
            test_query = "VIC-II"
            tokenized_query = test_query.lower().split()
            scores = kb.bm25.get_scores(tokenized_query)

            print(f"\nQuery: '{test_query}'")
            print(f"BM25 Scores: {scores}")
            print(f"Max score: {max(scores) if scores.size > 0 else 0:.6f}")
            print(f"Score > 0: {sum(1 for s in scores if s > 0)}")
            print(f"Score > 0.0001: {sum(1 for s in scores if s > 0.0001)}")


if __name__ == "__main__":
    test_bm25_scoring()
