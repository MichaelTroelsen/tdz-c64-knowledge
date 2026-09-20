# Why 46 searches logged a start and never logged a completion

**Task:** `forty-six-searches-logged-started-and-never-completed`
**Run:** `/runqueue --until-blocked`, cycle 3, main lane, 2026-09-20
**HEAD:** `721ea1d` on branch `doc-id-strip-page-break-markers`
**Depends on:** `peer-telemetry-writes-invalidate-every-search-cache` (`done`) — the
fix is in the working tree and this diagnosis was run against it.

---

## Answer

**The searches did not complete because their PROCESS DIED mid-search. It was not
client cancellation, and it was not a server-side block.**

The finding that settles it is an argument from the concurrency model rather than
from the timestamps, and it points the opposite way to what the plan recorded:

`call_tool` (`server.py`) dispatches every tool through
`await asyncio.to_thread(_call_tool_impl, name, arguments)`. A client cancellation
cancels the *awaiting coroutine*. It does **not** kill the OS worker thread — Python
has no mechanism to do so. So a cancelled search keeps running on its thread and
still reaches `self.logger.info("Search completed: ...")`.

**A cancelled request would therefore still produce a completion line.** The absence
of one rules cancellation out as the cause, rather than merely making it less likely.

---

## The recount, done at run time

Re-counted against the live `server.log`, not carried from the plan:

| line | count |
|---|---|
| `Search query:` | 1899 |
| `Search completed:` | 1878 |
| **never completed** | **46** |
| `cancelled` lines (all dates) | 25 |

The plan recorded 46 and it is still 46 — no new ones have appeared since.

### The gap is real, not a logging artifact

This was the first hypothesis worth killing, because it would have made the whole
task moot. It does not survive:

- `Search query:` has exactly **one** producer: `kb/search/_retrieval.py`, inside
  `search()`.
- `Search completed:` has exactly **one** producer: the same method.
- Between them there is **no early return**. The cache-hit branch returns *above*
  the start line, so a cache hit produces **neither** line and cannot contribute to
  the diff.

So every unmatched start is a call to `search()` that genuinely never reached its
own last statement.

### No exception ever propagated

`call_tool`'s `except` calls `kb.logger.exception(f"MCP tool {name!r} raised")`.

    grep -c "MCP tool .* raised" server.log   ->  0
    grep -c "Traceback" server.log            -> 34

All 34 tracebacks are `_add_document_db` failures, unrelated to search. **No search
tool call has ever raised.** With an exception ruled out and a normal return ruled
out, the only remaining exit from `search()` is the process ceasing to exist.

### The cancellations do not line up

Of the 25 `cancelled` lines in the entire log, **23 are from 2026-08-08**. Only
**two** fall on 2026-09-20 — Requests 5 and 6, at 17:34:57 and 17:34:59. Two
cancellations cannot account for 46 non-completions, and per the argument above they
would not have produced one anyway.

### What directly follows an unmatched start

For three of the last five unmatched searches, the next log line is a fresh
`KnowledgeBase` startup banner:

    18:13:50,648  Search query: 'self modifying code speed optimisation 6502 inner loop'
    18:14:54,512  ============================================================
    18:14:54,513  TDZ C64 Knowledge Base v2.24.0

That is a new process starting where the previous one was mid-search.

**This evidence is weaker than it looks, and the weakness is worth stating.** See
*A defect in the evidence itself* below — adjacency in this file does not prove
same-process causation. It is corroboration for the process-death conclusion, not
its basis. The basis is the thread-cancellation argument at the top.

---

## Does the gap survive the `kb/core.py` change?

**The cache-invalidation mechanism does not survive it. The non-completions were
never caused by it.** These are two separate questions and the plan conflated them.

Reproduction against the **live** 459 MB database with the fixed `kb/core.py`: warm
the caches and the BM25 index, then issue nine searches, each immediately preceded by
a peer connection committing a row to `mcp_call_log`.

    PHASE 1: warm
      chunks loaded: 9318 | bm25: True | search cache entries: 1
      Built BM25 index with 9318 chunks - Total: 95.10s (tokenize: 92.45s)

    PHASE 2: peer telemetry write, then a search
      before (chunks, bm25, cache): (9318, True, 1)
      after  (chunks, bm25, cache): (9318, True, 2)
      survived: True | 5 results | 182 ms

    PHASE 3: 8 searches, each preceded by a peer telemetry commit
      'rotozoom'                              1 hits   145.5 ms
      'raster interrupt stable timing'        5 hits     7.2 ms
      'SID filter cutoff resonance'           5 hits   177.2 ms
      'sprite multiplexer'                    5 hits   171.4 ms
      'rotozoom'                              1 hits     6.4 ms
      'VIC-II bad line'                       5 hits   174.4 ms
      'self modifying code 6502'              5 hits   161.5 ms
      'rotozoom texture mapping optimisation' 5 hits   172.8 ms

    PHASE 4: chunks: 9318 | bm25 built: True
             searches that failed to complete: 0

Before the fix every one of those nine commits would have cleared `self.chunks`, set
`self.bm25` to `None` and flushed the search cache. After it, the chunk set and the
index survive all nine, and the cache **grows** (1 → 2) instead of emptying. The
repeated queries drop to single-digit milliseconds (145.5 → 6.4, and 7.2) because the
cache is now allowed to hold.

**The BM25 rebuild cost is now measured rather than estimated: 95.10 s, of which
92.45 s is tokenising 9318 chunks.** That is what each peer telemetry write was
throwing away. Earlier records in this repo put it at "~265 s"; 95 s is the measured
figure on this machine at this corpus size, and it should replace the estimate.

Zero of the nine searches failed to complete.

---

## A defect in the evidence itself

`kb/core.py` configures logging with

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

There is **no `%(process)d`**, and every `server.py` process appends to the same
`server.log`. Six server processes were alive during this investigation. Their lines
interleave in one file with nothing to tell them apart.

This is why the adjacency analysis above is corroboration rather than proof, and it
is the single change that would make the next investigation of this class cheap:
**add the pid to the log format.** Without it, "what happened next" is unanswerable
whenever more than one server is running — which, given the server is registered in
`~/.claude.json` at user scope, is essentially always.

---

## What was NOT established

Stated plainly rather than papered over:

- **Why** those processes died. Nothing in this repo's logs records a process exit
  reason, and the processes in question are long gone. The behaviour is not currently
  reproducing.
- **Which client** was driving them. The task
  `a-client-spawns-a-new-server-process-per-tool-call` was cancelled by human decision
  on exactly this ground; that decision and its evidence are in `decisions.jsonl`.
- Whether the 23 cancellations on 2026-08-08 share this cause. They are outside this
  task's window and were not investigated.

## Recommendation

One change, small and decidable now: **put `%(process)d` in the log format.** Every
other question in this class is unanswerable without it, and it costs one line.
