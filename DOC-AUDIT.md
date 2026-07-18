# Documentation Audit — tdz-c64-knowledge

**Audited:** 2026-07-18 · **Commit:** 982c125 · **Branch:** master
**⚠ Working tree dirty:** 4 uncommitted changes — findings describe the working tree, which may not match `HEAD`.
**Findings:** 3 P0 · 5 P1 · 4 P2 · 3 P3 — all HIGH confidence
**Filed:** 14 findings grouped into 6 issues — [#3](../../issues/3) · [#4](../../issues/4) · [#5](../../issues/5) · [#6](../../issues/6) · [#7](../../issues/7) · [#8](../../issues/8)
**Fixed in this session:** P0-2 (`.env`) — commit `748ad9d`, not pushed
**Withheld from issues:** P0-2 — filing it publicly would have advertised the trap and the file to watch

| Finding | Disposition |
|---|---|
| P0-1 CI failing / archived tests | [#3](../../issues/3) |
| P0-2 `.env` tracked in public repo | **fixed locally**, commit `748ad9d` |
| P0-3 version 1.0.0 vs 2.24.0 | [#4](../../issues/4) |
| P1-1 documented test commands fail | [#3](../../issues/3) — same root cause |
| P1-2 REST endpoints 27→18 | [#5](../../issues/5) |
| P1-3 MCP tool count 62→87 | [#5](../../issues/5) |
| P1-4 dead doc links | [#6](../../issues/6) |
| P1-5 missing `enable_semantic_search.py` | [#6](../../issues/6) |
| P2-1 DB tables 12+→22 | [#5](../../issues/5) |
| P2-2 wiki README vs actual export | [#7](../../issues/7) |
| P2-3 stale `search-index.json` | [#7](../../issues/7) |
| P2-4 `docs/README.md` currency claim | [#8](../../issues/8) |
| P3-1 absolute paths in tracked `.env` | resolved by the P0-2 fix |
| P3-2 ROADMAP self-contradiction | [#8](../../issues/8) |
| P3-3 ARCHITECTURE vs ROADMAP | [#8](../../issues/8) |

## Scope — read what, and what was not read

**Read in full (8 files, 112 KB):** `README.md`, `CLAUDE.md`, `CONTEXT.md`, `docs/README.md`, `wiki/README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/QUICKSTART.md`

**Not read (31 files):** the remaining `docs/*.md`. Claims *about* them (link targets, referenced counts) were verified; their contents were not. A follow-up pass would likely find more.

**Not verified:** ARCHITECTURE.md contains ~24 citations of the form `server.py ~line 4350`. These were not checked — `server.py` has been heavily edited since, and confirming each would require reading a very large file. Treat all such line references as suspect until checked.

---

## Ground truth

| Fact | Actual value | Source | Confidence |
|---|---|---|---|
| Code version | **2.24.0** | `version.py:13` | HIGH |
| Packaged version | **1.0.0** | `pyproject.toml:3` | HIGH |
| MCP tools | **87 unique** (97 `Tool(` calls) | `rg` over `server.py` | HIGH |
| REST endpoints | **18** (8 GET, 8 POST, 1 PUT, 1 DELETE) | `rest_server.py` decorators, positive control run | HIGH |
| DB tables | **22** | `CREATE TABLE` in `server.py` | HIGH |
| Tests in root | **2** (`test_card_updates.py`, `test_pdf_viewer.py`) | `ls` | HIGH |
| Tests in archive | **35** | `find archive -name 'test_*.py'` | HIGH |
| `test_server.py` | only at `archive/tests/test_server.py` | `find` | HIGH |
| CI status | **failing on every run since 2026-01-04** | `gh run list` | HIGH |
| `.env` tracked in git | **yes**, committed in v2.13.0 and v2.14.0 | `git ls-files`, `git log` | HIGH |

---

## Findings

### P0-1 · CI has been failing for over six months while the README advertises it

**Locations:** `.github/workflows/ci.yml:37`, `README.md:517`
**Claim:** `README.md:517` — "GitHub Actions workflow tests on Python 3.10/3.11/3.12 across Windows/Linux/macOS with Ruff code quality checks."
**Actual:** The matrix is configured exactly as described (`ci.yml:18-19`), but line 37 runs `pytest test_server.py`, and that file lives at `archive/tests/test_server.py`. Every job fails at the test step.

```
$ gh run list --limit 8
2026-07-16  CI/CD Pipeline  -> failure
2026-07-16  CI/CD Pipeline  -> failure
2026-07-16  CI/CD Pipeline  -> failure
2026-01-10  CI/CD Pipeline  -> failure
2026-01-06  CI/CD Pipeline  -> failure
2026-01-05  CI/CD Pipeline  -> failure
2026-01-04  CI/CD Pipeline  -> failure
2026-01-04  CI/CD Pipeline  -> failure
```

**Confidence:** HIGH — failure observed via the GitHub API; the cause is a single unambiguous path in `ci.yml`.
**Consequence:** 9 jobs burn on every push and have never passed since at least January. The README's claim creates unearned confidence that the code is tested on three platforms. This is the finding the project can least afford, because it silently removes the safety net every other claim rests on.
**Fix:** Decide whether the archived suite is live. If yes, move `archive/tests/` back and point CI at it. If no, remove the testing claim from `README.md:517` and disable or fix the workflow. Do not leave a red badge and a green sentence.

---

### P0-2 · `.env` is tracked in a public repo, and the docs will eventually put a key in it

**Locations:** `.gitignore` (no `.env` entry), `.env:58`, `.env.example:55`, `docs/QUICKSTART.md:107`
**Actual state:**

```
$ git ls-files --error-unmatch .env
.env
$ git log --oneline --all -- .env
96d13c4 Release v2.14.0: UI/UX Improvements & Configuration Enhancements
0f6bf95 Release v2.13.0: AI-Powered Document Summarization Feature
$ git check-ignore -v .env      # no output — not ignored
$ rg -n 'env' .gitignore
11:# Virtual environment
12:.venv/
13:venv/
15:env/
```

`.gitignore` covers `.venv/`, `venv/` and `env/` — but **not** `.env`.

**No credential is exposed today.** The committed file contains only configuration (`TDZ_DATA_DIR`, `USE_SEMANTIC_SEARCH`, `SEMANTIC_MODEL`, `LLM_PROVIDER`, `LLM_MODEL`, `POPPLER_PATH` and similar). `ANTHROPIC_API_KEY` sits commented out at `.env:58`. Verified without reading values.

**Confidence:** HIGH on every component — tracking, commit history, gitignore contents, absence of current secrets.
**Consequence:** This is a loaded trap, not a present breach. The repo is **public**; `.env` is tracked; `.env.example:55` shows `ANTHROPIC_API_KEY=sk-ant-api03-...`; and enabling the LLM features requires filling that key in. The moment line 58 is uncommented and populated, the next `git add -A` publishes a live Anthropic key to a public repository.

`docs/QUICKSTART.md:107` uses `set ANTHROPIC_API_KEY=...` — a shell variable, which is the safe path. The `.env` route is the dangerous one and is the one `.env.example` advertises.

**Fix:**
1. Add `.env` to `.gitignore`.
2. `git rm --cached .env` so it stops being tracked.
3. Keep `.env.example` tracked as the template.

**This finding must not be filed as a public GitHub issue.** Doing so would advertise the trap and the exact file to watch.

---

### P0-3 · The package ships as version 1.0.0 while the code is 2.24.0

**Locations:** `pyproject.toml:3`, `version.py:13`, `README.md:3`, `docs/README.md:3`, `wiki/README.md:4`, `CONTEXT.md:23`

| Source | Version |
|---|---|
| `pyproject.toml:3` — **what pip installs** | **1.0.0** |
| `version.py:13` — what the code reports | **2.24.0** |
| `README.md:3` badge | 2.23.15 |
| `wiki/README.md:4` | 2.23.15 |
| `CONTEXT.md:23` | 2.23.14 |
| `docs/README.md:3` | 2.23.1 |

**Confidence:** HIGH — all six values read directly.
**Consequence:** Five different numbers, and the authoritative one is the furthest from reality. Anyone installing the package gets metadata claiming 1.0.0 — over a year of releases invisible to any tooling that reads package metadata.
**Fix:** Make `version.py` canonical; have `pyproject.toml` read from it (`dynamic = ["version"]`), and generate the badge from it. Five hand-maintained copies will drift again.

---

### P1-1 · Every documented test command fails

**Locations:** `CLAUDE.md:37,106,107,108`, `README.md:497,500,503`
**Claims:** `pytest test_server.py -v`, `pytest test_server.py test_wiki_export.py -v`, `pytest test_wiki_export.py -v`, plus a coverage variant.
**Actual:** Both files are in `archive/tests/`. The repo root contains only `test_card_updates.py` and `test_pdf_viewer.py`.
**Confidence:** HIGH — `find` located both under `archive/tests/`; neither exists at the documented path.
**Consequence:** Seven commands across two files, none of which run. `README.md:508` further describes "`test_wiki_export.py` - Wiki generation features (16 tests)" — a test suite documented in detail that a reader cannot execute.
**Fix:** Same decision as P0-1. The two are one problem: the test suite was archived and nothing downstream was updated.

---

### P1-2 · REST endpoint count overstated by 50%

**Locations:** `README.md:73,535,580`, `CLAUDE.md:22`, `CONTEXT.md:17`, `docs/README.md:36,113`, `docs/QUICKSTART.md:153,262`
**Claim:** "27 endpoints" — stated in **9 places** across 5 files.
**Actual:** 18.

```
$ rg -o '@app\.[a-z]+' rest_server.py | sort | uniq -c
      1 @app.delete
      2 @app.exception
      8 @app.get
      8 @app.post
      1 @app.put
```
8 + 8 + 1 + 1 = 18 route decorators. (`@app.exception` is a handler, not an endpoint.)

**Confidence:** HIGH — pattern verified by positive control; the decorator breakdown is shown rather than a bare count.
**Fix:** Correct to 18, or better, generate the number. Nine hand-written copies is how it reached 27 in the first place.

---

### P1-3 · MCP tool count understated everywhere, and no two documents agree

**Locations:** `README.md:276` ("62 MCP tools"), `CONTEXT.md:62` ("59 tools"), `CLAUDE.md:21` ("50+ tools"), `docs/README.md:37` ("50+ MCP tools"), `docs/ARCHITECTURE.md:363` ("MCP Tools (5 tools)")
**Actual:** **87 unique tool names**, from 97 `Tool(` registrations in `server.py`.
**Confidence:** HIGH on 87 unique names. The 97-vs-87 gap is unexplained and may indicate duplicate registrations — worth investigating separately.
**Consequence:** Four different numbers, all too low. `CONTEXT.md:64` additionally labels a category "Documents (6)" and then lists 8 tools; `CONTEXT.md:66` says "AI & Analytics (14)" and lists 15. The counts are not being derived from anything.
**Fix:** Generate the count and the category listing from the tool registry at build time.

---

### P1-4 · Dead documentation links in the two most-read files

**Locations:** `README.md:29,70,272,524,528`, `CLAUDE.md:7-11`, `docs/README.md:24,26,141`

Files moved into `docs/` without updating inbound links:

| Link | Written as | Actually at |
|---|---|---|
| QUICKSTART.md | root | `docs/QUICKSTART.md` |
| ARCHITECTURE.md | root | `docs/ARCHITECTURE.md` |
| CHANGELOG.md | root | `docs/CHANGELOG.md` |
| PHASE3_TEMPORAL_ANALYSIS.md | root | `docs/PHASE3_TEMPORAL_ANALYSIS.md` |
| WIKI_EXPORT_GUIDE.md | root | `docs/WIKI_EXPORT_GUIDE.md` |

`docs/README.md` has the mirror-image error — it links `../ARCHITECTURE.md`, `../CHANGELOG.md`, `../QUICKSTART.md`, but those files sit in `docs/` alongside it, so the `../` prefix points at nothing.

`CLAUDE.md:11` lists `FUTURE_IMPROVEMENTS.md` as the roadmap; that file is at `archive/historical-docs/FUTURE_IMPROVEMENTS.md` while the live roadmap is `docs/ROADMAP.md`. The pointer aims at the archived one.

**Confidence:** HIGH — each target resolved with `find`, with a fallback so relative references are not miscounted as missing.

---

### P1-5 · QUICKSTART instructs running a script that does not exist

**Location:** `docs/QUICKSTART.md:91` — `.venv\Scripts\python.exe enable_semantic_search.py`
**Actual:** No `enable_semantic_search.py` anywhere in the repo.
**Confidence:** HIGH — `find`, repo-wide.
**Consequence:** Semantic search is a headline feature; this is the documented way to turn it on, in the file called "Quick Start". Also `docs/QUICKSTART.md:157` links `README_REST_API.md`, which does not exist, and `:230` links `IMPROVEMENTS.md`, which is at `archive/historical-docs/`.

---

### P2-1 · Database table count wrong in every document, in both directions

**Locations:** `README.md:444` ("12+ tables"), `CLAUDE.md:76` ("12+ tables"), `CONTEXT.md:18` ("16 tables"), `CONTEXT.md:48` ("12+ tables"), `docs/ARCHITECTURE.md:32` ("four main tables", then lists six)
**Actual:** **22** — `chunks, clusters, cross_references, document_clusters, document_code_blocks, document_entities, document_events, document_facets, document_relationships, document_summaries, document_tables, document_topics, documents, entity_relationships, events, extraction_jobs, graph_cache, graph_metrics, graph_paths, search_log, timeline_entries, topics`
**Confidence:** HIGH. `CONTEXT.md` contradicts itself 30 lines apart (16 at :18, 12+ at :48).

---

### P2-2 · `wiki/README.md` describes an export that no longer matches

**Location:** `wiki/README.md:88-91, 115, 128-134, 163`

| Claim | Line | Actual |
|---|---|---|
| 4 main pages | 88-91 | **13** HTML pages |
| 8 data files | 128-134 | **14** |
| "~137 MB" | 163 | **202 MB** |

Undocumented data files include `chunks.json` (72.8 MB — the largest file in the export), `articles.json`, `coordinates.json`, `graph.json`, `search.json`, `similarities.json`.
**Confidence:** HIGH — `ls`, `du`.
**Fix:** Generate this README from `wiki_export.py` using `stats.json` and a real file scan. A hand-written description of generated output cannot stay true.

---

### P2-3 · The wiki's search index is six days older than the content it indexes

**Location:** `wiki/assets/data/`

```
2026-01-04   68,134,735   search-index.json
2026-01-10   68,598,034   documents.json
2026-01-10   72,807,365   chunks.json
2026-01-10      ...       (all 12 others)
```

**Confidence:** HIGH on the timestamp discrepancy; **MEDIUM** on the consequence — whether search is degraded or the file is simply orphaned was not determined.
**Consequence:** Either the wiki's search silently misses six days of content, or `search-index.json` is 68 MB of dead weight (note `search.json` also exists, dated 2026-01-10). This is a code finding, not a documentation one.
**Fix:** Determine which file the wiki actually loads, then regenerate or delete the other.

---

### P2-4 · `docs/README.md` claims a currency guarantee it does not meet

**Location:** `docs/README.md:133,136` — "**Versioned** - Each doc shows the version it applies to" and "**Up-to-date** - Updated with each release"
**Actual:** The file is stamped 2.23.1 (`:3`) and "Last Updated: 2026-01-03" (`:4`) while the code is at 2.24.0. It also indexes roughly 25 of the 39 files in `docs/`.
**Confidence:** HIGH.
**Consequence:** A documentation index that asserts its own freshness while being six months stale is worse than one making no claim — it discourages the reader from checking.

---

### P3-1 · Absolute paths containing the username are published in the tracked `.env`

**Location:** `.env`, committed
**Actual:** 3 lines contain `C:\Users\<username>\...` paths (`TDZ_DATA_DIR`, `ALLOWED_DOCS_DIRS`, `POPPLER_PATH`). `docs/ARCHITECTURE.md:142` likewise hardcodes `C:\Users\mit\claude\mdscrape`.
**Confidence:** HIGH.
**Consequence:** Minor privacy exposure, already public. Resolved as a side effect of P0-2's `git rm --cached`.

---

### P3-2 · ROADMAP contradicts itself on what is finished

**Location:** `docs/ROADMAP.md`

| Feature | Marked proposed | Marked complete |
|---|---|---|
| Smart Auto-Tagging | `:160,163` "Proposed" | `:960` "✅ Smart Auto-Tagging (v2.23.0)" |
| REST API Server | `:831` "Proposed" (Phase 5) | `:983` "Complete REST API (v2.18.0+)" |

**Confidence:** HIGH that the document contradicts itself. Which state is true was not determined for auto-tagging; REST is clearly built (18 endpoints exist).

---

### P3-3 · ARCHITECTURE and ROADMAP disagree on delivered features

**Locations:** `docs/ARCHITECTURE.md:630,631` vs `docs/ROADMAP.md:964,966` vs `docs/QUICKSTART.md:241`
**Claim:** ARCHITECTURE lists "Fuzzy search / typo tolerance" and "Multi-language support" under **Future Enhancements**. ROADMAP marks both ✅ COMPLETE. QUICKSTART marks fuzzy search ✅.
**Actual:** Fuzzy search is implemented — `USE_FUZZY_SEARCH` and `FUZZY_THRESHOLD` are live env vars in `.env`, and `fuzzy_search` is among the 87 registered tools. So ARCHITECTURE is the stale one.
**Confidence:** HIGH for fuzzy search. **Multi-language support was not verified** — no finding is made on it.

---

## Duplicated facts

| Fact | Copies | Agree? | Canonical source should be |
|---|---|---|---|
| Version | 6 locations | **no** — 4 distinct values | `version.py`, with `pyproject.toml` dynamic |
| REST endpoint count | 9 locations | yes — all wrong (27 vs 18) | generated from `rest_server.py` |
| MCP tool count | 5 locations | **no** — 62/59/50+/50+/5 | generated from the tool registry |
| DB table count | 5 locations | **no** — 12+/16/four | generated from schema |
| Doc index | 3 competing indexes | **no** — 2 have dead links | `docs/README.md` only |

---

## Verified clean

- All 8 documented batch files exist: `setup.bat`, `run.bat`, `tdz.bat`, `start-all.bat`, `run_rest_api.bat`, `add-pdfs.bat`, `start-wiki.bat` — `ls`, 7/7
- All 8 documented CLI commands exist in `cli.py`, including `show-relationships` and `translate-query` which `ARCHITECTURE.md` omits — `rg`, 8/8
- No exposed credentials in any tracked file — secrets scan across the repo, plus targeted check of `.env` at `HEAD`
- Chunking parameters (1500 words / 200 overlap) match between docs and `.env`
- `docs/ROADMAP.md` is the live roadmap and exists where `docs/README.md:84` says it does

---

## Unverifiable

| Claim | Location | Why |
|---|---|---|
| "480x faster (50ms vs 24s)" | `README.md:34` +5 more | No benchmark script in the repo; the only trace is in `archive/`. Methodology, hardware and dataset unstated. |
| "5,712 queries/sec (10 workers)" | `README.md:80` | Same — `load_test_results.json` exists in root but its provenance and date are undocumented. |
| "3400+ docs/second" anomaly detection | `README.md:57` | Same. |
| "90%+ coverage" | `CONTEXT.md:105` | Cannot be measured while the test suite does not run (P0-1). |
| Multi-language support | `ROADMAP.md:966` | Not checked. |
| ~24 `server.py ~line NNNN` citations | `docs/ARCHITECTURE.md` | Not checked — see Scope. |

---

## Structural observations

**One habit produces most of these findings.** Five facts — version, endpoint count, tool count, table count, doc index — are each hand-maintained in 3–9 places. Every one has drifted. The counts are never wrong by a little: 27 vs 18, 62 vs 87, 12 vs 22. They are not being derived from anything, so they cannot self-correct. Generating them is a smaller job than the next five audits.

**Archiving was done without a reverse lookup.** `test_server.py`, `test_wiki_export.py`, `IMPROVEMENTS.md`, `FUTURE_IMPROVEMENTS.md` were moved to `archive/`, and nothing that referenced them was updated — CI, two test-command blocks, and a roadmap pointer all still aim at the old paths. A single `rg` for each filename before moving would have caught all of it.

**The project documents itself as more finished than it is, and the mechanism is uniform:** completion is asserted in prose (Phase 1/2/3 ✅ 100%, "production-ready", "90%+ coverage", a CI badge sentence) while the artifacts that would substantiate it are archived, unrun, or absent. Nothing here looks like an attempt to mislead — it looks like documentation written at the moment of completion and never revisited. That is exactly what an audit is for.

**The comparison worth drawing:** `sid-reference-project` documents its own limitations rigorously and is the better project for it. This repo has no "Known limitations" section in any of the eight files read. Adding one would do more for its credibility than any single fix below.

---

## Recommended order

1. **P0-2** — `.gitignore` + `git rm --cached .env`. Smallest fix, largest downside avoided. Do this first.
2. **P0-1 / P1-1** — decide the fate of `archive/tests/`. One decision resolves both.
3. **P0-3** — make `version.py` canonical; `pyproject.toml` dynamic.
4. **P1-4 / P1-5** — repair dead links and the missing `enable_semantic_search.py` reference.
5. **P1-2 / P1-3 / P2-1** — generate the three counts instead of writing them.
6. **P2-3** — resolve the stale `search-index.json` (code, not docs).

<!-- Regenerated on each audit. Git holds the history. -->
