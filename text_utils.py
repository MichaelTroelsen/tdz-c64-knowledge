#!/usr/bin/env python3
"""
TDZ C64 Knowledge - snippet/term-matching text helpers

Small, stdlib-only (re only) text utilities used by KnowledgeBase's snippet
extraction, answer-grounding, and bulk-ingest glob handling. Split out of
server.py alongside models.py/features.py so that KnowledgeBase no longer
depends on anything defined in server.py itself - a prerequisite for
splitting that 253-method class into mixin modules without a circular
import.
"""

import re


def _expand_brace_pattern(pattern: str) -> list[str]:
    """Expand a shell-style brace glob into the list of plain globs it means.

    `pathlib.Path.glob` does NOT support brace alternation, so a pattern like
    "**/*.{pdf,txt}" matches literally nothing - it looks for a file whose
    extension is the 9-character string "{pdf,txt}". Bulk ingest shipped that
    as its default and so silently found 0 files while reporting success.

    >>> _expand_brace_pattern("**/*.{pdf,txt}")
    ['**/*.pdf', '**/*.txt']
    >>> _expand_brace_pattern("**/*.md")
    ['**/*.md']
    """
    start = pattern.find('{')
    if start == -1:
        return [pattern]
    end = pattern.find('}', start)
    if end == -1:
        return [pattern]  # unbalanced - treat as a literal, don't guess
    prefix, alts, suffix = pattern[:start], pattern[start + 1:end], pattern[end + 1:]
    expanded = []
    for alt in alts.split(','):
        # Recurse so patterns with more than one brace group also work.
        expanded.extend(_expand_brace_pattern(prefix + alt.strip() + suffix))
    return expanded


# Shared with _verify_answer_grounding so claim-splitting and snippet-boundary
# splitting agree on what a "sentence" is.
_SENTENCE_BOUNDARY_RE = re.compile(r'[.!?\n][\s\n]+')

# _extract_snippet's density scoring is a raw, unweighted substring count with
# no stopword or length filtering. Found live: a claim about "three ...
# voices" lost its correct window to an unrelated ring-modulation/SYNC-bit
# region because the bare digit "3" and function words like "is"/"as"/"the"
# substring-matched repeatedly there ("voice 3", "oscillator 3", "REG 3"),
# out-scoring the sentence that actually states the fact. _content_terms
# filters those out before a term set is used for windowing.
_LIGHTWEIGHT_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'and', 'or', 'but',
    'if', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'as', 'by', 'it', 'its', 'this', 'that',
    'these', 'those', 'we', 'you', 'he', 'she', 'they', 'i', 'has', 'have', 'had', 'do', 'does',
    'did', 'not', 'no', 'so', 'than', 'then', 'also', 'what', 'how', 'many', 'which', 'who',
    'whom', 'from', 'into', 'about', 'can', 'will', 'would', 'could', 'should', 'may', 'might',
})


def _filter_snippet_terms(terms, min_len: int = 3) -> set:
    """Drop stopwords and anything shorter than min_len from an already-
    tokenized term set, before it is used for _extract_snippet's density
    scoring.

    Deliberately separate from each search backend's own query-term
    extraction (FTS5 has its own tokenizer; search()/BM25's terms already go
    through _preprocess_text's NLTK stopword removal when enabled) - this is
    a second, narrower filter applied only to what gets handed to
    _extract_snippet for windowing/highlighting, so it never changes what a
    query actually matches. It matters even after NLTK stopword removal
    because NLTK's English stopword list has no notion of digits: a bare "3"
    survives that pass but still substring-matches inside unrelated numbers
    scattered through a document ("voice 3", "REG 3"), which is exactly the
    live-found defect this exists to prevent.
    """
    return {t for t in terms if len(t) >= min_len and t not in _LIGHTWEIGHT_STOPWORDS}


def _content_terms(text: str, min_len: int = 3) -> set:
    """Tokenize raw text and apply _filter_snippet_terms - for callers that
    only have a text string (a claim, a question, a rerank query), not an
    already-tokenized term set."""
    return _filter_snippet_terms(re.split(r'\W+', text.lower()), min_len)
