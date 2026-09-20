# Why 46 searches logged a start and never logged a completion

**Task:** `forty-six-searches-logged-started-and-never-completed`
**First written:** 2026-09-20, `/runqueue --until-blocked` cycle 3
**CORRECTED:** 2026-09-20, same day, after a live reproduction refuted it
**HEAD when corrected:** `d38ec79` on branch `doc-id-strip-page-break-markers`

---

## Correction notice

**The first version of this document said the cause was PROCESS DEATH. That was
wrong**, and the correction is kept in full rather than quietly replaced, because
how it was wrong is more useful than the wrong answer.

It argued: `call_tool` dispatches through `asyncio.to_thread`, so a client
cancellation cancels the awaiting coroutine and cannot kill the OS worker thread;
a cancelled search would therefore still reach its `Search completed:` line, so
the absence of that line rules cancellation out and leaves process death.

The reasoning about threads was correct. **The conclusion did not follow**, because
it assumed the only way to miss the log line is to stop existing. There is a third
way: the thread can still be alive and simply never get there.

What settled it was a live reproduction on a process that was **demonstrably
alive** — it answered a `health_check` seconds earlier — exhibiting the exact
signature. No process death required.

---

## Answer

**The first search in any cold process performs a heavyweight lazy import inside
the un-instrumented window between the two log lines. That is where these searches
disappear.**

Stack dump of a live, blocked server (`py-spy dump`, pid 71816):

    search                (kb/search/_retrieval.py:608)
      _preprocess_text    (kb/search/_retrieval.py:169)
        _ensure_nltk      (features.py:94)
          import nltk -> nltk.collocations -> nltk.metrics.association
            -> scipy.stats -> scipy.spatial -> scipy.linalg -> scipy.linalg.blas
              -> create_module            <-- blocked here

`search()` logs `Search query:` at `_retrieval.py:599` and `Search completed:` at
`:661`. The import chain above is entered at `:608`. Anything that stalls in there
produces exactly one line and never the second.

This is not an accident of configuration. `CLAUDE.md` defers `nltk`,
`sentence-transformers`, `torch` and `transformers` off the module level *on
purpose*, to protect the 30 s MCP initialize handshake budget. That deferral is
correct. But the cost does not disappear — it moves into the first search, where
nothing bounds it and nothing logs it.

---

## The evidence, from one shared log

Both outcomes appear in `server.log` minutes apart:

    19:55:12,761  Search query: 'raster interrupt stable timing' (max_results=3)
                  ... no completion line, ever ...
    20:00:27,093  Search query: 'raster interrupt stable timing' (max_results=3)
    20:00:27,818  Search completed: 3 results in 724.65ms
    20:00:27,934  Search query: 'sprite multiplexer' (max_results=3)
    20:00:28,008  Search completed: 3 results in 74.04ms
    20:01:01,681  Request 4 cancelled - duplicate response suppressed

Same query, same corpus, same code. The 19:55 call was the stdio server's first
search; the 20:00 calls went to a separate HTTP server process that had already
been running. **724 ms against nine minutes and counting.**

### The cancellation line is the decisive one

`20:01:01` is the 19:55 request being cancelled — I stopped it by hand. **No
completion line followed it.**

That is a direct test of the original argument, and it fails. The claim was that a
cancelled request still logs its completion because the worker thread survives
cancellation. The thread did survive. It still logged nothing, because it was
blocked in a kernel wait and could not reach the statement. "The thread keeps
running" is true and useless when the thread is not running.

### The process was idle, not slow

Measured while blocked:

| sample | CPU total | working set | stack |
|---|---|---|---|
| 19:59:58 | 1.2 s | 158 MB | identical |
| 20:00:03 | 1.2 s | 158 MB | identical |

Zero CPU accumulated over five seconds, no memory growth, byte-identical stack
across repeated samples 4 s apart. A kernel wait, not computation.

### Recount

Re-counted at run time against the live log: **1899** `Search query:`, **1878**
`Search completed:`, **46** never completed. `Search query:` and
`Search completed:` have exactly one producer each, both inside `search()`, with
no early return between them — the cache-hit branch returns *above* the start
line, so a cache hit emits neither and cannot inflate the diff. The gap is real.

---

## What is still NOT known, and it matters

**Why that particular process's DLL load wedged is unexplained.** Three
measurements argue it is not inherent to the import:

- the identical first search on a different fresh process: **0.815 s**
- `import nltk` + `nltk.collocations` in a fresh interpreter: **1.1 s**
- the same import driven through `asyncio.to_thread`, mirroring what `server.py`
  does: **0.9 s** — which refutes the obvious "imports deadlock on worker threads"
  hypothesis outright

Only one thread in the blocked process was importing, so it is not a two-thread
import deadlock either, and every other thread was idle, so nothing held the GIL.
Something stalled inside `create_module` — a C-extension `LoadLibrary` — and this
document does not claim to know what.

So the honest statement is two-part: **the window is the defect and is fully
established; the trigger that wedged one process inside it is not.** A fix that
bounds or instruments the window is worth doing regardless of the trigger, which
is what makes the unknown tolerable rather than blocking.

Also not established: why the historical processes died, and which client drove
them. The task `a-client-spawns-a-new-server-process-per-tool-call` was cancelled
by human decision on that ground.

---

## Recommendations

Two, in order of value:

1. **Instrument the window.** Log immediately before and after `_preprocess_text`
   on the first call, or warm `_ensure_nltk` at startup behind an explicit flag so
   the cost is paid where the 30 s budget can see it. Today a search that dies in
   there is indistinguishable in the log from one that was never dispatched.

2. **Put `%(process)d` in the log format.** `kb/core.py` configures
   `format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'` and every
   `server.py` process appends to the same `server.log`. Six were alive during this
   investigation. Without a pid, "what happened next" is unanswerable whenever more
   than one server runs — and since the server is registered in `~/.claude.json` at
   user scope, that is essentially always. This is tracked as
   `log-format-carries-no-pid-so-concurrent-servers-are-indistinguishable`.

That second point is also why the first version of this document reached the wrong
conclusion. Its process-death claim rested partly on observing a fresh startup
banner immediately after several unmatched starts. In a single-pid-less log shared
by six processes, adjacency proves nothing about causation — a caveat the original
did state, then leaned on anyway.
