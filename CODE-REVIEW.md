# Code Review Findings — Full Toolchain

Reviewed: `server.py` (MCP server + KnowledgeBase), `rest_server.py`, `cli.py`,
`admin_gui.py`, `wiki_export.py` (incl. generated wiki JS: search, PDF viewer,
chatbot, bookmarks), `llm_integration.py`, `anomaly_detector.py`, tests,
packaging, repo hygiene.

**How to use this file:** work top-down. Each finding has file:line anchors, a
concrete failure scenario, and a fix. P0 items are ordered — **fix R1 before R5/R7**
(they interact). Check items off as they land. Line numbers are as of commit
`1025be0`; re-locate by the quoted identifiers if the file has shifted.

Legend: `[ ]` open · `[x]` done

---

## P0 — Correctness & security

### [x] R1. Shared SQLite connection used from multiple threads without serialization

**Where:** `server.py:665` (`sqlite3.connect(..., check_same_thread=False)`),
`server.py:7321` (`_extraction_worker_loop` uses `self.db_conn` directly),
`server.py:21079` (`add_documents_bulk` runs on an `asyncio.to_thread` worker),
`server.py:1862` (`_log_mcp_call` commits on the event-loop thread).
`self._lock` exists (`server.py:473`) but guards only 7 call sites vs ~188 uses
of `self.db_conn`.

**Failure scenario:** `add_documents_bulk` runs in a worker thread and is
mid-transaction inserting chunks. A trivial `kb_stats` call finishes on the
event-loop thread and its `_log_mcp_call` executes `self.db_conn.commit()`
(`server.py:1881`) — committing the half-finished ingest transaction. If the
ingest then errors and rolls back, it rolls back nothing; the DB is left with a
document row whose chunk set is partial. The extraction worker's own
`UPDATE ... commit` (`server.py:7346`) can do the same at any moment. This is
latent today and becomes frequent once R5/R7 move more work onto threads.

**Fix:** Give each thread its own connection. Wrap connection access in a
helper, e.g. `self._local = threading.local()` and
`def _conn(self): conn = getattr(self._local, 'conn', None); if conn is None: conn = self._make_conn(); self._local.conn = conn; return conn`
where `_make_conn()` applies the same PRAGMAs as `_init_database`
(foreign_keys, busy_timeout; WAL is a DB-file property, set once). Then replace
`self.db_conn` with `self._conn()` mechanically (it's ~188 sites but a safe
find/replace once the helper exists), and have `close()` close all known
connections. WAL mode (already enabled) makes multi-connection concurrency
safe. Alternative minimal fix if the above is too invasive in one pass: a
single `threading.RLock` acquired around every transaction (BEGIN→commit), but
per-thread connections are the correct end state and also fix R7 cleanly.

**Verify:** add a test that runs `add_documents_bulk` in a thread while
hammering `kb_stats`/`_log_mcp_call` from another; assert no
`cannot commit`/`no transaction is active` errors and chunk counts stay
consistent (pattern exists in `test_mcp_startup.py` for cross-process; this is
the in-process analog).

### [x] R2. `Path.cwd()` is always on the ingestion whitelist — defeats `ALLOWED_DOCS_DIRS`

**Where:** `server.py:549-554` (`default_allowed_dirs` includes `Path.cwd()`),
enforced at `server.py:3856` (`_is_path_allowed`) / `server.py:4074`.

**Failure scenario:** the server is registered at user scope in Claude Code
(it is — see `~/.claude.json`), so it is spawned with the cwd of *whatever
project the user happens to be in*. That entire project tree silently becomes
ingestible, making the whitelist decorative. The path-traversal protection log
line (`server.py:570`) claims protection it isn't providing.

**Fix:** drop `Path.cwd()` from the defaults. Gate it behind an explicit
opt-in env var (`TDZ_ALLOW_CWD=1`) for the CLI/dev workflow that relied on it.
Also `.resolve()` the default dirs the same way user dirs are resolved
(`server.py:558`) so `is_relative_to` can't be defeated by case/symlink
differences. Update `.env.example`/README to mention the flag.

### [x] R3. REST upload endpoint writes to a directory that is not on the whitelist

**Where:** `rest_server.py:648` (`uploads_dir = data_dir / "uploads"`), comment
at `rest_server.py:646` claims it's "allowed by security check" — it isn't:
defaults are `scraped_docs`, `downloads`, `temp`, cwd (`server.py:549-553`).

**Failure scenario:** `POST /api/v1/documents` → `kb.add_document(tmp_path)`
raises `SecurityError` → 500 "Upload failed" for every upload, *unless* the
REST server happens to be started from a cwd that covers `uploads/` (which is
why a stray `uploads/` exists in the repo root — it worked by accident via the
cwd hole R2 closes). Fixing R2 alone breaks this endpoint outright.

**Fix:** add `self.data_dir / "uploads"` to `default_allowed_dirs` in
`server.py` (it's server-controlled, safe), or write uploads into the existing
allowed `data_dir / "temp"`. Fix the comment. Add a REST test for the upload
endpoint (none exist — see R17).

### [x] R4. REST server: unauthenticated-by-default, bound to 0.0.0.0, CORS wildcard + credentials

**Where:** `rest_server.py:915` (`host="0.0.0.0"`), `rest_server.py:95-99`
(no API keys configured → all requests allowed), `rest_server.py:82-88`
(`allow_origins=['*']` together with `allow_credentials=True`),
`rest_server.py:128-138` (500 handler returns raw `str(exc)`).

**Failure scenario:** default launch (`run_rest_api.bat`) exposes full
read/write KB control (add/remove documents, scrape arbitrary URLs → SSRF from
the host) to the whole LAN with no auth. The wildcard-origin+credentials combo
is an invalid CORS configuration browsers reject inconsistently. `str(exc)`
leaks internal paths in error responses.

**Fix:** default `host` to `127.0.0.1`, read host/port from env
(`TDZ_REST_HOST`/`TDZ_REST_PORT`); refuse to bind non-loopback when `API_KEYS`
is empty (hard error with a clear message, overridable via
`TDZ_REST_ALLOW_INSECURE=1`); when CORS origins is `*`, set
`allow_credentials=False`; make the generic 500 handler return a generic
message and log the detail server-side instead.

### [x] R5. Almost every heavy MCP tool still blocks the event loop (only `add_documents_bulk` was fixed)

**Where:** `server.py:21079` is the only `asyncio.to_thread` in the 3,300-line
dispatch. Direct synchronous calls include: `add_document` (`server.py:20559`,
OCR of a scanned PDF can take minutes), `scrape_url` (`server.py:20605`, an
entire-site crawl can run for hours), `update_document`, `rescrape_document`,
`extract_entities` / `extract_entities_bulk`, `summarize_all`, `auto_tag_all`,
`train_lda_topics`/`train_nmf_topics`/`train_bertopic`, clustering tools,
`create_backup`/`restore_backup`, `check_url_updates`.

**Failure scenario:** exactly issue #13 again (see the comment at
`server.py:21069-21078`): while `scrape_url` crawls, every other request on
the session — including trivial `kb_stats` — queues behind it; the session
appears hung and MCP keep-alives stall. Commit `af95e77` fixed one tool; the
class of bug remains.

**Fix:** *after R1 lands* (per-thread connections make this safe), route the
dispatch through `asyncio.to_thread` uniformly. Cleanest: in `call_tool`
(`server.py:23461`), if `name` is in a `HEAVY_TOOLS` set, run
`await asyncio.to_thread(_dispatch_sync, name, arguments)`; or simpler and
uniformly correct: run the *entire* `_call_tool_impl` body via `to_thread`
(it is fully synchronous code — nothing in it needs the loop). Keep
`_log_mcp_call` where it is.

### [x] R6. Packaging is broken: missing dependency, wheel omits required modules, dead console script

**Where:** `pyproject.toml`.
- `server.py:31` does `from dotenv import load_dotenv` unconditionally, but
  `python-dotenv` is not in `[project.dependencies]` → fresh
  `pip install tdz-c64-knowledge` crashes on import.
- `[tool.hatch.build.targets.wheel] only-include = ["server.py", "cli.py"]`
  (`pyproject.toml:66`) omits `version.py`, `anomaly_detector.py`,
  `llm_integration.py`, `rest_models.py`, `rest_server.py` — `server.py:35`
  (`from version import ...`) makes the installed wheel unimportable.
- `[project.scripts] tdz-c64-knowledge = "server:main"` points at an
  `async def main` (`server.py:23513`); the entry point calls it, gets a
  coroutine, never awaits it → the installed command silently does nothing.

**Fix:** add `python-dotenv>=1.0.0` to dependencies; extend `only-include`
with the five modules above (or restructure into a package, see R12); add a
sync wrapper `def cli_main(): asyncio.run(main())` and point the script at
`server:cli_main`. **Verify:** `pip install .` into a scratch venv, run
`tdz-c64-knowledge --help`/handshake, and `python -c "import server"`.

---

## P1 — Reliability & performance

### [x] R7. Importing `server` instantiates a full KnowledgeBase as a module side effect

**Where:** `server.py:17994` (`kb = KnowledgeBase(DATA_DIR)` at module level).
Consumers: `rest_server.py:24`, `admin_gui.py:36`, `cli.py:15`,
`wiki_export.py:20`, all test files — every one executes it on import.

**Failure scenario:** the REST server builds **two** KnowledgeBase instances
(module-level one + its own at `rest_server.py:55`): two DB connections, two
background extraction-worker threads, documents loaded twice. Same for the
GUI, CLI, and wiki export — `python cli.py stats` pays double startup, and the
orphan extraction worker of the module-level KB can process jobs concurrently
with the "real" instance's worker (compounding R1). It also makes unit tests
touch the real production DB just by importing.

**Fix:** make it lazy. Replace the module-level construction with
`kb: KnowledgeBase | None = None` plus `def get_kb(): global kb; ...` — the
MCP handlers call `get_kb()` (or `main()` assigns `kb` before `server.run`).
`_call_tool_impl` references module-global `kb` throughout, so initializing it
in `main()` keeps every handler working unchanged while imports become free.

### [x] R8. REST endpoints are `async def` calling synchronous KB methods

**Where:** all of `rest_server.py` (e.g. search at ~231, scrape at ~583,
upload at 631).

**Failure scenario:** same as R5 but for uvicorn: one long scrape/upload
freezes every other HTTP request, including `/health`, because the coroutine
blocks the single event loop.

**Fix:** cheapest correct fix in FastAPI: declare the handlers as plain `def`
(FastAPI then runs them on its threadpool automatically). Requires R1
(per-thread connections) first. Alternatively `await asyncio.to_thread(...)`
per call.

### [x] R9. Extraction worker robustness gaps

**Where:** `server.py:7321-7392`.
- `task_done()` (`server.py:7387`) is skipped if the *outer* try raises (e.g.
  the status-UPDATE at 7341 hits a locked DB) → any future `queue.join()`
  hangs; the job also stays 'running' forever with no retry/timeout.
- Jobs queued but not yet started are lost on restart with status stuck at
  'queued' (rows persist in `extraction_jobs`, nothing re-enqueues them on
  startup).

**Fix:** move `task_done()` into a `finally` on the per-job try; on startup,
re-enqueue rows with status `queued` and mark stale `running` rows (older than
N minutes) as `failed: interrupted by restart`. Consider a max-runtime guard.

### [x] R10. Graph cache uses pickle from the database

**Where:** `server.py:12061` (`G = pickle.loads(row[0])` in
`_load_cached_graph`), written by `_cache_graph` (`server.py:11983`).

**Failure scenario:** anyone who can write the `knowledge_base.db` file (it's
shared across processes, restored from backups, and lives in a user directory)
gets arbitrary code execution in every server process that loads a cached
graph. Backup/restore (`restore_from_backup`, `server.py:17850`) makes
foreign DB files a realistic input.

**Fix:** serialize with `networkx.node_link_data(G)` → JSON, load with
`node_link_graph`. Bump the cache-table schema/version so stale pickle rows
are discarded rather than parsed.

### [x] R11. Scraper politeness & enforcement (carried from TO-DOS.md, partially done)

**Where:** `server.py:4348` (`scrape_url`), defaults `threads=10, delay=100`
(`server.py:4351`); client-side `max_pages` stop exists now
(`server.py:4648-4655`) but robots.txt, User-Agent, and retry/backoff are
still absent; `_discover_urls` (`server.py:4984`) fetches with no UA either.

**Fix:** default `threads=2, delay=1000` for new domains; send a UA like
`tdz-c64-knowledge/<version> (+repo URL)`; fetch and honor `robots.txt`
(`urllib.robotparser`) before crawling a domain; exponential backoff on
429/5xx. Keep the TO-DOS.md entry in sync (it stays authoritative for the
HVSC/DeepSID ingestion item, which is a feature, not a review fix).

---

## P2 — Architecture & maintainability

### [ ] R12. `server.py` is a 23,538-line god file; `KnowledgeBase` has ~250 methods

**Where:** `server.py:442-17987` (one class, ~17.5k lines), `list_tools()`
~2,170 lines (`server.py:17998`), `_call_tool_impl` ~3,290 lines of
`elif name ==` chain (`server.py:20175`).

**Why it matters:** every change risks unrelated breakage, review is
impossible, merge conflicts are constant, and the file defeats editor/CI
tooling. This is the single biggest tax on future work.

**Fix (incremental, mechanical — do in this order):**
1. Extract the MCP layer: move `list_tools`/`call_tool`/dispatch into
   `mcp_tools/` with a registry — each tool one entry
   `{"schema": Tool(...), "handler": fn}`; `list_tools` returns
   `[t.schema for t in REGISTRY]`, dispatch becomes a dict lookup (also kills
   the elif chain and makes R5's `to_thread` wrapping trivial).
2. Split `KnowledgeBase` by domain into mixins or collaborating classes:
   ingest/extraction (`_extract_*`, `add_document*`, scrape), search
   (fts5/bm25/semantic/hybrid/fuzzy/faceted), entities+graph, topics+cluster,
   timeline/events, viz, admin (backup/health/stats). Keep `KnowledgeBase`
   as the facade so callers don't change.
3. Move helpers (`_LazyModule`, locks, retry, atomic write) into `util.py`.
Target: no file over ~2,500 lines. Do it as several PRs, tests green between
each.

### [x] R13. `version.py` is 1,871 lines because the full changelog lives in a Python string

**Where:** `version.py:77` (`VERSION_HISTORY = """..."""`, ~75KB) — imported
by `server.py` at startup on every session.

**Fix:** move the history to `CHANGELOG.md`; keep `version.py` to
`__version__`, `__build_date__`, and the couple of helper functions. If any
tool surfaces the history at runtime, read the file lazily.

### [x] R14. LLM plumbing is duplicated and fragile

**Where:** `llm_integration.py` (providers) vs `server.py:5837` (`_call_llm`
lazily builds another client) vs `_generate_answer_with_llm`
(`server.py:16074`). A new `anthropic.Anthropic()` client is constructed per
call (`llm_integration.py:49`); no request timeout or retry policy;
`call_json` strips markdown fences by line-slicing (`llm_integration.py:172`,
breaks on trailing text).

**Fix:** one client instance cached on `LLMClient`; pass
`timeout=` and `max_retries=` to the SDK constructor; make `_call_llm` the
single entry point that delegates to `LLMClient`; harden `call_json` with a
regex fence extraction and a single retry-on-parse-failure.

### [ ] R15. Broad exception swallowing

**Where:** 161 `except Exception`/bare `except` in `server.py`, 62 in
`admin_gui.py`, ~234 handlers end in `pass`-like suppression patterns
project-wide.

**Fix (targeted, not wholesale):** in DB-transaction paths, catch, `rollback()`,
and re-raise (`_add_document_db`, `_remove_document_db`, bulk ops); in tool
handlers keep the catch-all but always log with `self.logger.exception` so
stack traces land in `server.log`. Leave cosmetic ones alone.

### [x] R16. Wiki export: 13k lines of HTML/CSS/JS inside Python strings; two escaping gaps; CDN deps break offline use

**Where:** `wiki_export.py:4353` (`_create_css` — ~3,100 lines of CSS in a
string), `:7510` (`_create_javascript` — ~1,800 lines incl. `BookmarkManager`
and `AIChatbot` JS classes), doc pages at `:1254`.
- Escaping: `doc["file_path_in_wiki"]` is interpolated into an `href`
  unescaped (`wiki_export.py:1321`); `source_url` is HTML-escaped but not
  scheme-validated (`wiki_export.py:1320`) — a scraped page whose recorded
  source URL is `javascript:...` becomes a live XSS link in the generated
  wiki.
- Offline: `knowledge-graph.html` loads d3 from CDN (`wiki_export.py:1858`)
  and the file viewer loads `marked` from CDN (`:3999`), while fuse/pdf.js are
  vendored via `_download_libraries` (`:9941`) — so two pages break without
  internet even though the wiki is otherwise self-contained.

**Fix:** validate `source_url` scheme (`http/https` only) and
`html.escape`/`urllib.parse.quote` the viewer href params; vendor d3 and
marked exactly like fuse.js. Longer term, move static CSS/JS out of Python
strings into `wiki_assets/` files copied at export (template only the truly
dynamic pages) — this alone removes ~6k lines from the Python file.

### [x] R17. Test coverage is concentrated in 4 niches; the REST API and the dispatch layer have none

**Where:** 51 tests total: cards (`test_card_updates.py`), startup/concurrency
(`test_mcp_startup.py`), embeddings (`test_embeddings_reliability.py`),
scrape (`test_scrape_reliability.py`), pdf viewer (`test_pdf_viewer.py`).
Zero coverage of: all 18 REST endpoints, the 87-tool MCP dispatch, search
correctness (FTS5/BM25/hybrid ranking), entity extraction, backup/restore.

**Fix (highest value first):**
1. A dispatch smoke test: for every tool name in the registry, call
   `call_tool(name, minimal_args)` against a tiny fixture KB and assert a
   `TextContent` comes back (catches typo'd `arguments.get` keys and broken
   handlers across the whole surface cheaply).
2. FastAPI `TestClient` tests: auth on/off, upload (guards R3), search,
   remove.
3. A search-correctness test with a 3-doc fixture asserting known-hit ranking
   for FTS5 and BM25 paths.

### [ ] R18. `admin_gui.py` is a 5,336-line single script

**Where:** `admin_gui.py` — one file with 62 broad exception handlers and
inline HTML via `unsafe_allow_html` (spot-checked: interpolations are
internal values, no user-content injection found).

**Fix:** split into Streamlit multipage layout (`pages/` directory — the
mechanical split Streamlit natively supports). Low urgency; do after R12 so
imports stabilize first.

---

## P3 — Hygiene & polish

### [x] R19. Repo-root clutter

Stray dev/debug scripts and outputs in the root: `check_files.py`,
`check_tfidf.py`, `debug_nmf.py`, `migration_v2_21_0.py` (one-shot, done),
`test_graph.json`, `benchmark_results.json`, `load_test_results.json`,
`url_check_quick_results.json`, `start-all.log`, `readme.txt`, scraped-site
folders (`www.sidmusic.org_sid/`, `unusedino.de_ec64_technical_aay_c64/`),
`uploads/`, `wiki_test/`. Most are untracked but pollute every directory
listing and glob. **Fix:** move keepers to `archive/utilities/`, delete the
rest, extend `.gitignore` (scraped-site dirs, `uploads/`, `*_results.json`).
The scraped-site folders belong under `TDZ_DATA_DIR/scraped_docs`, not the
repo.

### [ ] R20. `_network_timeout` mutates process-wide socket default

**Where:** `server.py:128` — `socket.setdefaulttimeout` affects every socket
in the process; if two threads overlap, the inner `finally` restores the
outer's temporary value. Benign today (rare paths), racy after R5 adds
threading. **Fix:** note-and-accept, or scope timeouts per-library (nltk
downloader and huggingface accept explicit timeouts via env/config in current
versions — verify before switching).

### [x] R21. `health_check` should report worker liveness

**Where:** `server.py:17266`. It checks DB and disk but not whether the
extraction worker thread is alive (it can die only via daemon teardown, but
after R9's changes a liveness flag is cheap). **Fix:** include
`extraction_worker_alive: self._extraction_worker.is_alive()` and queue depth.

### [x] R22. Documentation drift

Tool counts disagree (CLAUDE.md says 87 tools; older docs/handoffs say 59;
`whats-next.md` is a stale one-off handoff that should be deleted or archived).
`DOC-AUDIT.md` already tracks doc/code drift — fold these into it rather than
duplicating, then delete `whats-next.md` (its task completed).

---

## Bugs found by the new tests (not visible from static review)

The R17 dispatch smoke test (`test_mcp_tool_dispatch.py`) calls all 87 MCP
tools with schema-derived minimal args against a fixture KB. On first run it
caught two real, previously-undetected production bugs, both now fixed:

- **`add_documents_bulk` crashed on every successful call.** The dispatch
  handler's success-output formatting read `doc['filename']`
  (`server.py`, `elif name == "add_documents_bulk"` block), but
  `KnowledgeBase.add_documents_bulk`'s `results['added']` entries only ever
  contained `doc_id`/`filepath`/`title`/`chunks` — never `filename`. Any real
  (non-duplicate) call to this tool raised `KeyError: 'filename'` after the
  underlying ingest had already succeeded, so the tool always reported
  failure even when it worked. Fixed by deriving the display name from
  `filepath` instead.
- **`health_check` raised `AttributeError` under the default configuration.**
  It unconditionally read `self.embeddings_file`/`embeddings_map_file`,
  which only exist as attributes when `use_semantic` is `True` — and
  `USE_SEMANTIC_SEARCH` defaults to `0` (off). Every fresh install running
  `health_check` with default settings hit this (caught internally and
  surfaced as a "Health check error" issue, so not a crash, but a
  permanently-degraded health report for semantic feature detection). Fixed
  by gating the check on `self.use_semantic`.

Both are the kind of bug R17 exists to catch: correct in isolation, broken
only in the combination of "real success path" × "actual default config" —
exactly what a static read-through misses and a cheap dispatch smoke test
catches immediately.

## Notes from the R1 / R5 + R8 concurrency work

- **`db_conn` is now a property, not an attribute.** `KnowledgeBase.db_conn`
  returns a thread-local `sqlite3.Connection`, created on first touch by
  `_make_conn()` and registered in `self._all_conns` so `close()` can close
  connections whose owning thread has already retired. All ~190 existing
  `self.db_conn.execute(...)` sites, and the external readers in
  `rest_server.py` / `admin_gui.py` / `wiki_export.py`, were left untouched.
  Because the property has no setter, anything that used to do
  `self.db_conn = ...` had to change: `_init_database()` no longer connects at
  all (the first statement inside `_init_database_locked` triggers the lazy
  open) and `close()` routes through the new `_close_all_conns()`.
- **`restore_backup()` calls `_init_database()` a second time**, after the .db
  file on disk has been replaced. With one shared connection that just
  reconnected; with per-thread connections every stale connection has to be
  dropped, so `_init_database()` now begins with `_close_all_conns()` (which
  also resets `_thread_local`, so threads already parked in a pool re-open
  against the new file rather than reusing a closed connection).
- **The pre-existing `self._lock` sites were deliberately left alone.** Two of
  the five (`add_document`'s duplicate check and its insert/cache-invalidation
  block) protect `self.documents` and the BM25/embeddings caches, i.e. genuine
  shared in-memory state, and must stay. A third guards the lazy nltk
  `_preprocessing_ready` init. The remaining two (`add_document_with_url`,
  `check_url_updates`) are now redundant for DB purposes but are short
  single-statement writes, so removing them would be churn with no benefit.
- **The per-tool `asyncio.to_thread` in the `add_documents_bulk` handler was
  removed rather than left double-wrapped.** `_call_tool_impl` is now a plain
  `def` — it could not stay `async def` and still be handed to
  `asyncio.to_thread`, since that would pass an un-awaited coroutine to a
  worker thread. `call_tool()` is unchanged apart from
  `await asyncio.to_thread(_call_tool_impl, name, arguments)`; the
  timing/`_log_mcp_call` block still runs on the event-loop thread (it is a
  single INSERT on that thread's own connection).
- **Falsification check.** Both fixes were verified to actually be load-bearing,
  not just green: emulating the pre-fix shared connection (forcing `_make_conn`
  to return one cached connection per instance) makes
  `test_a_commit_on_one_thread_cannot_commit_another_threads_transaction` fail
  on *both* of its assertions — the two threads get the same connection object,
  and the deliberately rolled-back sentinel row survives in `mcp_call_log`
  (count 1 instead of 0), which is exactly the R1 data-corruption mechanism.
  Emulating the pre-fix inline dispatch makes
  `test_a_slow_non_bulk_tool_also_does_not_block_the_event_loop` fail with
  `kb_stats` taking the full 3.0s of the slow tool instead of returning
  immediately.
- **Known residual (accepted, not a regression):** a connection is opened per
  thread that touches the DB and is only closed by `close()` /
  `_init_database()`. In practice the thread population is bounded (the
  asyncio default executor and `add_documents_bulk`'s internal
  `ThreadPoolExecutor` both reuse threads), so this is a small fixed cost, not
  a leak that grows with request count.

## Bugs found while fixing R9 / R11 / R16

Two more latent bugs surfaced during this round, both fixed:

- **Generated wiki JS contained literal control characters.** In the
  `enhancements_js` literal in `wiki_export.py`, `\b` sat inside a *non-raw*
  Python string, so Python interpreted it as backspace (0x08) rather than
  passing a JavaScript word-boundary through. The emitted
  `assets/js/enhancements.js` carried `/\x08REM\s+.*/gi` and
  `/\x08\d+\x08/g`, so BASIC `REM`-comment and assembly-number syntax
  highlighting silently never matched anything. Verified present in the
  shipped `wiki/assets/js/enhancements.js` before the fix. All 45 backslash
  escapes in that literal are now explicit; the emitted JS was diffed
  before/after and changed in exactly those two lines. `test_wiki_safety.py`
  now asserts the emitted JS contains no control characters.
- **R9 recovery found real stranded work on the live database.** First run
  against the production data dir logged
  `re-queued 27, marked 33 stale 'running' job(s) failed` — 60 extraction
  jobs had been sitting permanently in limbo, reported by
  `get_extraction_status` as pending work that nothing would ever pick up.

Also note, for whoever picks up R12/R13: `version.py:102` has the same
invalid-escape warning, inside the `VERSION_HISTORY` string. Moving that
changelog out to `CHANGELOG.md` (R13) removes it for free.

## Bugs found while fixing R13 / R14 / R19 / R22

Four more, all fixed. The last two are the most serious findings of the whole
review after R1:

- **Cluster centroids were written and read in incompatible formats.**
  `_store_clusters_to_db` wrote them with `pickle.dumps`, while
  `visualize_cluster_dendrogram` read them back with
  `np.frombuffer(blob, dtype=np.float32)`. Those do not round-trip: verified
  that `np.frombuffer` on a pickled 4-element float32 array raises
  `ValueError: buffer size must be a multiple of element size` (the pickle was
  143 bytes, not a multiple of 4), so dendrogram visualization crashed
  outright whenever centroids had been stored — and would have returned
  silent garbage in the cases where the pickle length happened to divide by 4.
  Now stored as a raw `float32` buffer matching the reader, which also removed
  the last `pickle` call from the codebase.
- **A fresh database never got four of its tables.** `topics`,
  `document_topics`, `clusters` and `document_clusters` were created only
  inside `_init_database_locked`'s `else:` (existing-database) branch, so a
  brand-new install lacked them entirely and every topic-modelling and
  clustering tool failed with `no such table`. Confirmed empirically: a fresh
  data dir had 56 tables with all four missing, while the live 799-document
  database (which had been through the migration path) had them. Moved into a
  new always-run `_migrate_topics_clusters_schema()`; a fresh DB now has 60
  tables. `test_serialization.py` asserts the full expected table set on a
  fresh database so schema-creation and migration cannot drift apart again.
  Note this is why the R17 dispatch smoke test did not catch it: those handlers
  catch their own exceptions and return an `Error: ...` message, which the
  smoke test accepts as a handled outcome.
- **`validate_docs.py` was reporting false failures and hiding real ones.**
  Its README tool pattern (`^#{3,4}\s+([a-z_]+)$`) matched none of README's
  actual `**tool_name**` markup, so it reported 0 documented tools and all 92
  as missing; it looked for `CHANGELOG.md`/`QUICKSTART.md` in the repo root
  when both live under `docs/`; and its version-section regex terminated on
  any `##`, including the section's own `### Added` subheading, so every
  feature list came back empty while still printing a green tick. Fixed, plus
  the pass/fail logic now flags *stale* README tool references (real drift)
  rather than demanding all 92 tools appear in a README that says it documents
  a curated subset.
- **Version drift:** `docs/QUICKSTART.md` claimed 2.23.1 against 2.24.0
  everywhere else — found only once the validator above actually worked.

**Correction to R19:** `migration_v2_21_0.py` is **not** spent one-shot
clutter and was left in the repo root. `anomaly_detector.py:106` actively
directs users to run it, and the live database still lacks the anomaly tables
it creates ("Anomaly detection tables not found. Run migration_v2_21_0.py
first."). Only `whats-next.md` (completed handoff) and `readme.txt` (superseded
by README.md, its table markup collapsed into unreadable run-on text) were
removed, plus `performance_phase2_results.json` untracked (it matches the
existing `*_results.json` ignore rule but predated it).

## Suggested implementation order

1. **R6** (packaging) — small, self-contained, unblocks clean installs.
2. **R1** (per-thread connections) — prerequisite for R5/R8.
3. **R2 + R3** together (whitelist changes interact with upload path).
4. **R5, R7, R8** (event-loop + import side effect) — one PR, plus R9.
5. **R4** (REST hardening) and **R10** (pickle→JSON).
6. **R17** tests (lock in the above).
7. **R12/R13/R16** refactors as separate incremental PRs.
8. **R11, R14, R15, R18–R22** opportunistically.
