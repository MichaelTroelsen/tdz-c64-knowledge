# Scoping: groundedness/faithfulness check for `answer_question`

**Status: both tiers built, and live-tested.** Tier 1 (LLM-based check) shipped first;
Tier 2 (local NLI cross-encoder, `USE_NLI_VERIFICATION=1`) shipped after, ahead of the
"only if Tier 1 proves too slow" trigger below - built on request rather than after that
measurement. `_verify_answer_grounding` now dispatches to NLI when enabled, cascading to
the LLM check if the NLI model is unavailable or errors, and to the plain citation-presence
heuristic if both fail. Label indices for the NLI model are resolved from the loaded
model's own `config.id2label` rather than hardcoded (see `_ensure_nli_loaded`), verified
against `cross-encoder/nli-deberta-v3-base`'s actual HuggingFace config before writing any
of this code.

**Live-testing round (after both tiers shipped):** ran real `answer_question()` calls
against the real corpus with a real Anthropic API key and a real downloaded NLI model,
rather than trusting the mocked test suite alone. Found and fixed five real bugs no mock
could have caught, since mocks always hand-construct clean inputs:

1. `checked_claims` was computed but never forwarded through `answer_question`'s
   top-level return dict - callers always saw `None`.
2. `_passages_for_claims` shared one windowed excerpt per cited *source* across every
   claim that cited it. A source cited for two different facts (a chunk's base-address
   register table and its voice-count intro, 2,495 characters apart) only got windowed
   once; whichever fact won the shared window, the other claim was judged against text
   that never mentioned it - a genuinely well-cited claim came back "not_mentioned".
   Now windows per (claim, source) pair.
3. The underlying `_extract_snippet` density scorer is a raw, unweighted substring count
   with no stopword/length filtering. A claim's term set including the bare digit "3"
   matched inside unrelated numbers scattered through the chunk ("voice 3", "REG 3"),
   stealing the window from the sentence that actually said "three voices". New
   `_content_terms()` helper filters stopwords and sub-3-character tokens - applied to
   all three `_extract_snippet`-windowing call sites (verification, reranking,
   `_build_rag_context`), not just the one that surfaced it.
4. Even with (3) fixed, `_extract_snippet`'s sentence-boundary alignment could still clip
   a correctly-found match by walking the window's start back to the nearest prior
   sentence boundary and re-extending by a fixed length - landing short of content near
   the *original* winning window's far edge. Fixed by comparing density (score per
   character) between the aligned window and the original, only overriding when density
   dropped substantially (empirically: 1.0 for a legitimate shorter/uniform-content
   trim, 0.14 for the real failure).
5. `VERIFY_PASSAGE_CHARS`'s original default (2000) was large enough that (4)'s guard
   couldn't help - at that size the density search picks an entirely wrong region of a
   long chunk, not merely clips the right one. Reduced default to 800.

**Known remaining limitation, found live and deliberately not chased further:** (4)'s
density-ratio guard catches gross content loss, not surgical loss of one specific/rare
term when most of a multi-term match set still clusters together nearby. A live claim
citing a table-of-contents-style appendix listing showed this pattern - the claim was very
plausibly accurate, but the windowing likely didn't reach the exact confirming line, and
the check flagged it unverified anyway. A full fix would need TF-IDF-style term weighting
or a semantic (embedding-based) passage-selection approach instead of substring-density
counting - out of scope for this round.

Full account, including the exact numbers behind each fix: the "Fix five bugs found by
live-testing Tier 1/2 answer grounding" commit, and `docs/ARCHITECTURE.md`'s "RAG Answer
Grounding" and "Enhanced Snippet Extraction" sections.

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
