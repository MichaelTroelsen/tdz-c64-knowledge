# Scoping: groundedness/faithfulness check for `answer_question`

**Status: both tiers built.** Tier 1 (LLM-based check) shipped first; Tier 2
(local NLI cross-encoder, `USE_NLI_VERIFICATION=1`) shipped after, ahead of
the "only if Tier 1 proves too slow" trigger below - built on request rather
than after that measurement. `_verify_answer_grounding` now dispatches to
NLI when enabled, cascading to the LLM check if the NLI model is unavailable
or errors, and to the plain citation-presence heuristic if both fail. Label
indices for the NLI model are resolved from the loaded model's own
`config.id2label` rather than hardcoded (see `_ensure_nli_loaded`), verified
against `cross-encoder/nli-deberta-v3-base`'s actual HuggingFace config
before writing any of this code.

## Problem, precisely

`_generate_answer_with_llm` (server.py:17452) has no verification step. `confidence`
is not calibrated against anything — it's a hardcoded binary:

```python
if citations:
    confidence = 0.85   # "Higher confidence if LLM cited sources"
else:
    confidence = 0.70
```

(server.py:17497-17500). "Cited sources" means `_extract_citations` (server.py:17513)
found the literal string `Source N` in the model's output — a self-report of citing,
not evidence the cited passage supports the claim next to it. Nothing checks that the
answer's claims are entailed by the retrieved chunks rather than drawn from the model's
own C64 knowledge, which for a 40-year-old, extensively-documented platform is
substantial. A wrong answer that cites "Source 2" looks identical, in the API response,
to a right one.

## Proposed design

Add a verification pass between generation and return, in `answer_question`
(server.py:16928) right after `_generate_answer_with_llm` returns:

1. **Split the answer into checkable units.** Sentence-split `answer_result['text']`
   (reuse the existing sentence-boundary regex already in `_extract_snippet`,
   server.py:16184, rather than adding a new one). Skip sentences with no citation
   marker attached — nothing to verify against.

2. **Entailment check per unit**, one of two implementation tiers:
   - **Tier 1 (cheap, ship first):** one more `llm_client.call_json` per answer (not
     per sentence) asking the model to return, for each numbered claim, whether the
     cited source passage supports/contradicts/doesn't-mention it. Reuses the existing
     `LLMClient`/`extract_json` plumbing (llm_integration.py:242-267) — no new
     dependency. Cost: one extra LLM call per `answer_question` invocation, roughly
     doubling generation-side latency and token cost. This is the one to build first;
     it needs no new infrastructure.
   - **Tier 2 (cheaper per-call, more infra):** a local NLI model (e.g.
     `cross-encoder/nli-deberta-v3-base`, same lazy-import pattern as the reranker —
     see `CrossEncoder` at server.py:441 and `use_reranker`/`_ensure_reranker_loaded`
     at server.py:743,16606) scoring (claim, cited-passage) pairs for
     entailment/neutral/contradiction. Near-zero marginal cost per call once the model
     is loaded, but adds a second lazy-loaded model, its own `USE_*` env var, and its
     own test file — proportionally the same shape of work as the reranker did today.
     Worth doing only if Tier 1's extra LLM call proves too slow/expensive in practice.

3. **Replace the confidence heuristic** with the check's output: e.g.
   `confidence = supported_claims / total_checked_claims`, clamped, with a floor when
   `total_checked_claims == 0` (no citations — keep today's 0.70 fallback for that
   case, since there's nothing to verify).

4. **Surface unsupported claims**, don't just downgrade confidence silently. Add
   `answer_result['unverified_claims']: list[str]` to the return dict so callers (the
   MCP tool response, the REST endpoint) can show *what* wasn't grounded, not just a
   number. This is what actually earns user trust — a bare confidence score is exactly
   the kind of "looks rigorous, might not be" number this fix is meant to replace.

## What this does NOT need to solve

- **Multi-hop / query decomposition** — separate problem, separate scope.
- **Refusal calibration** ("say I don't know") — falls out of step 3 for free once
  confidence is real: `answer_question` can already return its "could not find
  relevant documentation" fallback path (server.py:16995) when retrieval is empty;
  extend that threshold check to also fire on a low post-verification confidence.
  One-line addition once steps 1-4 exist, not separate scope.
- **Conversational/multi-turn memory** — out of scope, unrelated to groundedness.

## Test surface (mirrors today's pattern: `test_retrieval_quality.py`, `test_rerank.py`)

A new `test_groundedness.py`:
- unsupported claim in a synthetic answer is flagged (mock `llm_client.call_json`
  to return a fixed verdict — no live API needed, same mocking style already used
  for LLM-touching tests elsewhere in this codebase)
- fully-supported answer keeps high confidence
- zero-citation answer falls back to the existing 0.70, doesn't crash the check
- verification-call failure (`llm_client` raises) doesn't take down `answer_question` —
  falls back to the pre-existing heuristic rather than erroring the whole call, same
  defensive shape as `rerank()`'s `try/except` fallback (server.py:16674-16686)

## Size estimate

Tier 1: roughly the same scope as today's RRF-fusion change (~1 new method, one call
site edit, one test file, no new env-gated model) — half a day including tests.
Tier 2 adds the lazy-loaded-model overhead the reranker already paid for, so only
worth it if Tier 1's added latency turns out to matter in practice.
