#!/usr/bin/env python3
"""
Retrieval evaluation harness.

Measures whether search actually surfaces the chunk that contains the answer,
so ranking changes can be judged by a number instead of by impression.

Ground truth is answerability, not hand-labelled document IDs: a retrieved
chunk counts as relevant when its text satisfies the question's `must_contain`
spec (AND of ORs - every group must match, any variant within a group will do).
This keeps the question set portable across re-ingests, where doc_ids change.

The obvious limitation: a lexical relevance judgment mildly flatters keyword
search over semantic search, because a chunk that paraphrases the answer
without using the literal term scores as a miss. It is still the right measure
for the question this harness exists to answer - "did retrieval reach the
passage holding the fact?" - and the same bias applies to every arm of a
comparison, so deltas remain meaningful.

Usage:
    # Score the live index
    python eval_retrieval.py

    # Compare two index files (A/B a re-embedding change)
    python eval_retrieval.py --compare \\
        --index-a ~/.tdz-c64-knowledge/embeddings.faiss.bak \\
        --map-a   ~/.tdz-c64-knowledge/embeddings_map.json.bak \\
        --label-a "whole-chunk" --label-b "windowed"

    # Validate the question set against the corpus (no search, fast)
    python eval_retrieval.py --validate
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_QUESTIONS = Path(__file__).parent / "eval" / "retrieval_questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: bad JSON - {e}")
    return questions


def is_relevant(text: str, must_contain: list[list[str]]) -> bool:
    """AND of ORs: every group must be satisfied by at least one variant."""
    if not text:
        return False
    lowered = text.lower()
    return all(any(v.lower() in lowered for v in group) for group in must_contain)


# ---------------------------------------------------------------------------
# Question-set validation
# ---------------------------------------------------------------------------

def validate_questions(questions: list[dict], data_dir: Path) -> int:
    """Check every question is answerable from the corpus.

    A question with no matching chunk caps recall below 1.0 for reasons that
    have nothing to do with the retriever, which would quietly poison every
    later comparison.
    """
    db = data_dir / "knowledge_base.db"
    if not db.exists():
        raise SystemExit(f"database not found: {db}")

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT doc_id, chunk_id, content FROM chunks").fetchall()
    conn.close()
    print(f"scanning {len(rows)} chunks for {len(questions)} questions\n")

    unanswerable = 0
    for q in questions:
        matches = sum(1 for _, _, content in rows if is_relevant(content, q["must_contain"]))
        if matches == 0:
            unanswerable += 1
            print(f"  UNANSWERABLE  {q['id']:<24} {q['must_contain']}")
        else:
            print(f"  ok ({matches:>5} chunks)  {q['id']}")

    print()
    if unanswerable:
        print(f"{unanswerable} of {len(questions)} questions have no supporting chunk - "
              f"fix or drop them before trusting any score.")
    else:
        print(f"all {len(questions)} questions are answerable from the corpus.")
    return unanswerable


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def build_kb(data_dir: Path, index_file: Path | None, map_file: Path | None):
    os.environ.setdefault("TDZ_DATA_DIR", str(data_dir))
    os.environ["USE_SEMANTIC_SEARCH"] = "1"
    os.environ["AUTO_EXTRACT_ENTITIES"] = "0"
    # Without this, search() silently drops to the pure-Python BM25 fallback,
    # which takes ~2 minutes per query on this corpus and would make any
    # keyword or hybrid measurement meaningless. The MCP server config sets it.
    os.environ.setdefault("USE_FTS5", "1")

    from server import KnowledgeBase

    kb = KnowledgeBase(str(data_dir))

    if index_file or map_file:
        # Repoint before the lazy loader fires, so it reads the alternate pair.
        if index_file:
            kb.embeddings_file = Path(index_file).expanduser()
        if map_file:
            kb.embeddings_map_file = Path(map_file).expanduser()
        for missing in (kb.embeddings_file, kb.embeddings_map_file):
            if not missing.exists():
                raise SystemExit(f"index file not found: {missing}")

    kb._ensure_embeddings_loaded()
    if kb.embeddings_index is None:
        raise SystemExit("no embeddings index loaded - is USE_SEMANTIC_SEARCH working?")
    return kb


def run_search(kb, question: str, mode: str, k: int) -> list[dict]:
    if mode == "semantic":
        return kb.semantic_search(question, max_results=k)
    if mode == "hybrid":
        return kb.hybrid_search(question, max_results=k)
    if mode == "keyword":
        return kb.search(question, max_results=k)
    raise SystemExit(f"unknown mode: {mode}")


def score(kb, questions: list[dict], mode: str, k: int, verbose: bool = False) -> dict:
    """Recall@k, MRR@k and mean latency over the question set.

    Relevance is judged against the full chunk text, not the returned snippet:
    the snippet is a ~300-char display excerpt and a chunk can hold the answer
    in a part the snippet never shows.
    """
    hits = 0
    reciprocal_ranks = []
    latencies = []
    per_question = []

    for q in questions:
        start = time.time()
        try:
            results = run_search(kb, q["question"], mode, k)
        except Exception as e:
            print(f"  ERROR {q['id']}: {e}", file=sys.stderr)
            results = []
        latencies.append((time.time() - start) * 1000)

        rank = None
        for i, r in enumerate(results, 1):
            chunk = kb.get_chunk(r["doc_id"], r["chunk_id"])
            if chunk and is_relevant(chunk.content, q["must_contain"]):
                rank = i
                break

        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        per_question.append((q["id"], rank))
        if verbose:
            print(f"  {'hit @' + str(rank) if rank else 'MISS   ':<9} {q['id']}")

    n = len(questions) or 1
    return {
        "mode": mode,
        "k": k,
        "n": len(questions),
        "recall": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "median_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
        "per_question": per_question,
    }


def print_summary(res: dict, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\n{res['mode']}@{res['k']}{tag}: "
          f"recall={res['recall']:.1%}  MRR={res['mrr']:.3f}  "
          f"median={res['median_ms']:.0f}ms  (n={res['n']})")


def print_comparison(a: dict, b: dict, label_a: str, label_b: str) -> None:
    print(f"\n{'':<26} {label_a:>14} {label_b:>14} {'delta':>10}")
    print("-" * 68)
    for name, key, fmt in (("recall@%d" % a["k"], "recall", "{:.1%}"),
                           ("MRR@%d" % a["k"], "mrr", "{:.3f}"),
                           ("median latency (ms)", "median_ms", "{:.0f}")):
        va, vb = a[key], b[key]
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{name:<26} {fmt.format(va):>14} {fmt.format(vb):>14} "
              f"{sign + fmt.format(delta):>10}")

    ranks_a = dict(a["per_question"])
    ranks_b = dict(b["per_question"])
    gained = [q for q in ranks_b if ranks_b[q] and not ranks_a.get(q)]
    lost = [q for q in ranks_b if ranks_a.get(q) and not ranks_b[q]]

    print()
    if gained:
        print(f"newly found ({len(gained)}): {', '.join(sorted(gained))}")
    if lost:
        print(f"REGRESSED ({len(lost)}): {', '.join(sorted(lost))}")
    if not gained and not lost:
        print("no change in which questions are answered.")


def main() -> int:
    default_dir = os.getenv("TDZ_DATA_DIR") or str(Path.home() / ".tdz-c64-knowledge")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument("--mode", default="semantic",
                    choices=["semantic", "hybrid", "keyword"])
    ap.add_argument("-k", type=int, default=5, help="cutoff for recall@k / MRR@k")
    ap.add_argument("--verbose", action="store_true", help="per-question hit/miss")
    ap.add_argument("--validate", action="store_true",
                    help="check every question is answerable, then exit")
    ap.add_argument("--compare", action="store_true", help="A/B two index files")
    ap.add_argument("--index-a", type=Path, help="baseline .faiss (with --compare)")
    ap.add_argument("--map-a", type=Path, help="baseline map json (with --compare)")
    ap.add_argument("--index-b", type=Path, help="candidate .faiss (default: live)")
    ap.add_argument("--map-b", type=Path, help="candidate map json (default: live)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")

    if args.validate:
        return 1 if validate_questions(questions, args.data_dir) else 0

    if args.compare:
        if not args.index_a:
            raise SystemExit("--compare needs at least --index-a")
        # Separate processes would be cleaner, but a fresh KnowledgeBase per arm
        # is enough: the search caches are per-instance.
        kb_a = build_kb(args.data_dir, args.index_a, args.map_a)
        res_a = score(kb_a, questions, args.mode, args.k, args.verbose)
        vectors_a = kb_a.embeddings_index.ntotal
        kb_a.close()

        kb_b = build_kb(args.data_dir, args.index_b, args.map_b)
        res_b = score(kb_b, questions, args.mode, args.k, args.verbose)
        vectors_b = kb_b.embeddings_index.ntotal
        kb_b.close()

        print(f"\n{args.label_a}: {vectors_a} vectors    "
              f"{args.label_b}: {vectors_b} vectors")
        print_comparison(res_a, res_b, args.label_a, args.label_b)
        return 0

    kb = build_kb(args.data_dir, args.index_b, args.map_b)
    res = score(kb, questions, args.mode, args.k, args.verbose)
    print_summary(res)
    kb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
