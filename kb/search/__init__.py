"""SearchMixin assembled from its sub-mixins (R12 step 4).

kb/search.py grew to 3,565 lines / 58 methods, over the ~2,500-line module
target. It is split here into three sub-mixins by responsibility - lexical
retrieval plus embeddings/reranking, the RAG answer_question/verification
path, and query translation/similarity/comparison - each its own module,
still under the target.

The public name `SearchMixin` (and `from kb import SearchMixin`) is
unchanged: this package's __init__ composes it from the sub-mixins below,
so server.py needs no edit.

Splitting a mixin into pieces and recombining by inheritance means a
method defined in two pieces would silently shadow one of them, and the
method count would still look right. Guard it explicitly, the same way
mcp_tools/handlers.py guards its handler-dict merge: assert no method name
appears twice across the sub-mixins, and that the union equals the
original 58-name set.

The sub-modules are imported normally. Note for anyone patching these
symbols in a test: a bare name inside a method resolves against the
module that DEFINES the method, so CrossEncoder and SentenceTransformer
must be patched at kb.search._retrieval, not kb.search. Patching the
package would bind a name nothing reads and the test would pass while
exercising the real model - test_groundedness.py and test_mcp_startup.py
target _retrieval for exactly this reason.
"""
from ._query import _QueryMixin
from ._rag import _RagMixin
from ._retrieval import _RetrievalMixin

_SUB_MIXINS = [_RetrievalMixin, _RagMixin, _QueryMixin]

_EXPECTED_METHODS = frozenset({
    "_build_bm25_index", "_fuzzy_match_terms", "_preprocess_text",
    "build_suggestion_dictionary", "get_query_suggestions",
    "_update_suggestions_for_chunks", "export_search_results",
    "_export_markdown", "_export_json", "_export_html", "search",
    "_search_bm25", "_search_simple", "_fts5_match_expressions",
    "_search_fts5", "_extract_snippet", "_embedding_passages",
    "_expand_chunks_to_passages", "_passage_fanout",
    "_ensure_embeddings_loaded", "_load_embeddings",
    "_load_embeddings_locked", "_save_embeddings_locked",
    "_remove_doc_embeddings", "_build_embeddings",
    "_add_chunks_to_embeddings", "_ensure_reranker_loaded",
    "_ensure_nli_loaded", "_rerank_passage", "rerank", "_rerank_depth",
    "semantic_search", "_rrf_k", "_fuse_rankings", "hybrid_search",
    "fuzzy_search", "_build_search_vocabulary", "search_within_results",
    "answer_question", "_build_rag_context", "_generate_answer_with_llm",
    "_extract_claims_for_verification", "_passages_for_claims",
    "_verify_claims_llm", "_verify_claims_nli", "_verify_answer_grounding",
    "_extract_citations", "_generate_summary_fallback",
    "translate_nl_query", "faceted_search", "_get_document_facets",
    "find_by_reference", "find_similar_documents", "_find_similar_semantic",
    "_find_similar_tfidf", "compare_documents", "search_tables",
    "search_code",
})

_seen: dict = {}
for _cls in _SUB_MIXINS:
    for _name in vars(_cls):
        if _name.startswith("__"):
            continue
        if _name in _seen:
            raise RuntimeError(
                f"duplicate method {_name!r} defined in both "
                f"{_seen[_name].__name__} and {_cls.__name__} - this would "
                "silently shadow one of them"
            )
        _seen[_name] = _cls

_actual = frozenset(_seen)
if _actual != _EXPECTED_METHODS:
    _missing = _EXPECTED_METHODS - _actual
    _extra = _actual - _EXPECTED_METHODS
    raise RuntimeError(
        f"SearchMixin sub-mixin method set changed unexpectedly: "
        f"missing={sorted(_missing)} extra={sorted(_extra)}"
    )


class SearchMixin(_RetrievalMixin, _RagMixin, _QueryMixin):
    pass
