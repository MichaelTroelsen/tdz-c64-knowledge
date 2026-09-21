# Issue #17, re-scoped against what the code actually does

**Date:** 2026-09-21
**Head at time of writing:** 609f764
**Outcome:** `amend-the-issue`

Issue #17 makes two requests. The first is a straightforward configuration
change and stands as written. The second rests on a false premise, and this
document replaces it with the defect the reporter actually hit.

---

## Request 1 stands: add `C:/Users/mit/claude/brain` to the allowlist

`add_documents_bulk` genuinely does reject the brain tree, and genuinely did
fail all 153 files with "outside allowed directories". Nothing below touches
that half of the issue. It is a real request against real behaviour, and it is
still open.

## Request 2 does not stand: `check_updates` never sees a path

> "Make `check_updates` fail fast — return an error — when it encounters a
> disallowed path, instead of hanging."

`check_updates` cannot encounter a path, because nothing hands it one.

**The call chain, re-confirmed rather than assumed.** The MCP tool named
`check_updates` is registered in `mcp_tools/admin.py:776` and dispatched to
`handle_check_updates` (`mcp_tools/admin.py:264`), which calls
`kb.check_all_updates(auto_update)` — one boolean argument, and nothing else.

`check_all_updates` is at `kb/ingest/_documents.py:2107`. Its full signature is
`(self, auto_update: bool = False)`. It takes **no directory, no pattern, and
no path of any kind**. It iterates `self.documents` — the documents already in
the database — and for each one calls `os.path.exists(doc.filepath)` and
`self.needs_reindex(...)`. There is no allowlist check anywhere in it, and no
network call at any point.

That last part matters for a second reason: the two tools are easy to confuse.
`kb.check_url_updates` — the one that *does* go to the network — is reached by
a **different** tool, `check_url_updates`, registered in
`mcp_tools/documents.py:444` and handled at `mcp_tools/documents.py:216`. A
reader skimming for "the update checker" can land on either. The issue's
request is worded as though `check_updates` were the crawling one; it is not.

So there is no code path in which `check_updates` "encounters a disallowed
path", and therefore none in which it could fail fast on one. Implementing
request 2 as written would mean adding an allowlist check to a method that has
no path to check.

---

## What the 1800-second stall actually was: a long run, not a hang

This is the distinction the issue does not draw, and it is the one that
changes what should be built.

**Measured over the live corpus** (`C:\Users\mit\.tdz-c64-knowledge\knowledge_base.db`,
1,138 documents):

```
documents: 1138
loop elapsed: 2.369s
  missing   220
  changed   323  (of which 0 had no mtime/hash recorded)
  unchanged 595
  files content-hashed because mtime moved: 348
```

**2.37 seconds.** The read-only half of `check_updates` — everything that runs
when `auto_update=False` — completes in under three seconds against the real
corpus, including the 348 content hashes it computes for files whose mtime
moved. It is not slow, and it does not hang.

The issue reports `check_updates` being called **with `auto_update=true`**.
That flag changes the method from a scan into a whole-corpus re-index:

```python
if auto_update:
    updated_doc = self._reindex_document_if_changed(filepath, doc.title, doc.tags)
```

`_reindex_document_if_changed` (`kb/ingest/_documents.py:1804`) does a full
remove-then-add per document — its own docstring says so — which means
extraction, chunking and re-indexing for each one. **323 documents qualified**
on the measurement above. At even a few seconds apiece that is well past 1800
seconds, and PDFs are not a few seconds apiece.

`check_updates` is also in `_LONG_RUNNING_TOOLS` (`server.py:294`), the 23-tool
set exempted from the 600-second dispatch bound precisely because a whole-corpus
pass legitimately runs for minutes to hours. So it is unbounded **by design**,
and was entirely unbounded at the time of the report — the bounds in `server.py`
landed in 7fa124b, after issue #17 was filed on 2026-09-19.

**Therefore the reporter did not observe a hang.** They observed a correctly
functioning whole-corpus re-index of 323 documents, with no progress reporting
and no completion estimate, and aborted it at the 30-minute mark because
nothing distinguished it from a hang. That is a real and serious defect — but
it is a **reporting** defect, not a liveness one, and the fix request in the
issue would not have addressed it.

### The hang mechanism that was ruled out

This session found a genuine unbounded wait in the MCP server (issue #22): once
the nltk warm-up starts, a thread holding the Windows loader lock blocks the
next tool that calls `Thread.start()` until client traffic arrives. It is a real
candidate for "a tool that never returns", so it was checked rather than assumed
irrelevant.

It does not apply here. The `auto_update` branch reaches `add_document`, which
is not the `ThreadPoolExecutor` path — that belongs to `add_documents_bulk`
(`kb/ingest/_documents.py:2221-2248`), a different tool. The one thread the
KnowledgeBase starts, `EntityExtractionWorker` (`kb/core.py:306`), is started in
the **constructor**, before any tool call, so it is not a victim of a lock taken
later. No `Thread.start()` occurs inside the `check_updates` reindex loop.

`add_documents_bulk`, which *does* use a thread pool, is a separate matter and
is already listed among the five tools affected in #22.

---

## What should replace request 2

1. **`check_updates` with `auto_update=true` must report progress.** A tool
   whose normal successful runtime is half an hour, that emits nothing until it
   finishes, is indistinguishable from a wedged one — and the caller's only
   available response is the one this reporter took: abort, and file it as a
   hang. `add_documents_bulk` already accepts a `progress_callback`; this path
   has no equivalent.

2. **The tool description should say what `auto_update=true` costs.** The
   scheduled job that filed this issue called it expecting a check and got a
   re-index. A caller should be able to tell from the schema that the flag turns
   a three-second scan into a multi-hour rebuild.

3. **Separately worth deciding, but out of this document's scope:** 220 of the
   1,138 documents are `missing` — their recorded `filepath` no longer exists.
   That is the long-standing re-point problem already tracked in `TO-DOS.md`,
   and it is why the "changed" set is large. It is not a bug in `check_updates`.

## What should NOT be built

An allowlist check inside `check_all_updates`. There is no path for it to
check, and adding one would mean inventing an argument the tool does not have
in order to satisfy a request written against the wrong tool.

---

## Method note

The figures were taken twice, by two independent routes, and they agree.

First by **transcribing** `check_all_updates`'s loop
(`kb/ingest/_documents.py:2124-2158`) and `needs_reindex` (`:1758`) against the
live `documents` table opened read-only (`mode=ro`). That indirection was not a
preference: a concurrent agent in the same cycle held `rw:kb/core.py`, and
importing the `KnowledgeBase` would have read a file another lane was mid-way
through editing.

Then, once that lock was released, by **invoking the real method**:

```
ctor 0.11s  documents=1138
check_all_updates(auto_update=False): 2.66s
  unchanged  595
  changed    323
  missing    220
  updated      0
```

Identical counts, 2.66s against the transcription's 2.37s. The conclusion does
not rest on the reproduction.

`auto_update=True` was **not** run against the live database, per the task's own
constraint: it re-indexes every changed document and there is no git rollback
for that file. Its cost is reasoned from the code above, not measured.
