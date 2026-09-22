"""Dispatch smoke test: every registered MCP tool must RETURN, over a real
stdio session, within a generous per-call bound - or the test names it and
fails, instead of hanging the test runner forever.

This is a full rewrite (see issue #18 / the search-hang investigation). The
old version called `asyncio.run(server_module.call_tool(...))` directly, in
the SAME process, with NO timeout around the call at all. Two things made
that version worthless for the defect it now exists to catch:

  1. NO TIMEOUT: a genuinely hanging handler hung the test runner, not the
     test. `pytest` would sit forever instead of reporting a failure. Every
     call here goes through MCPSession.call_tool(), which reads the
     subprocess's stdout via a dedicated background reader thread and pulls
     from a queue with a deadline - it is asserted to fail, never waited on.
  2. FRESH PROCESS PER CALL: calling the dispatcher in-process (or spawning a
     fresh server per tool) does not reproduce the class of hang this test
     exists to catch. The real bug (issue #18) was a first-search-only stall
     caused by a deferred, BLAS-linked import touching stdin on a LONG-LIVED
     stdio session - it does not reproduce in a fresh interpreter. This test
     therefore spawns exactly ONE server.py subprocess and drives all tool
     calls through that single long-lived MCP stdio session, the same way a
     real client does.

This is intentionally NOT a correctness test for each tool's business logic.
A tool coming back as a structured `isError: true` CallToolResult (bad
arguments, a missing file, an empty/untrained model) is an ACCEPTED outcome
here - the low-level MCP SDK wraps even an unhandled exception raised inside
a handler into an isError CallToolResult before it ever reaches this test
(see mcp.server.lowlevel.server.Server.call_tool's `except Exception`). The
only property this test checks is that the call RETURNS at all, within the
bound. Do not mistake a green run here for tool correctness.

Populated data dir: built fresh under pytest's tmp_path via a throwaway
in-process KnowledgeBase (one short text document is added, then the KB is
closed before the subprocess ever opens the same directory). This is NOT a
copy of, and never opens, the user's real
C:\\Users\\mit\\.tdz-c64-knowledge\\knowledge_base.db. That matters twice
over: many of the 95 tools mutate state (add_document, remove_document,
restore_backup, bulk operations, ...), and calling all of them with minimal
arguments against a real data dir would damage it.
"""
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import server as server_module
from mcp_tools.schemas import TOOL_SCHEMAS

REPO = Path(__file__).parent
SERVER = REPO / "server.py"

# Prefer the project venv so this measures what Claude Code actually launches
# (the bare `python` on PATH here is 3.14 without the project's deps).
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

HANDSHAKE_BUDGET_S = 15.0
LIST_TOOLS_BUDGET_S = 15.0
# Generous: comfortably above the 5s deadline _ensure_nltk() now enforces
# internally (features.py, TDZ_NLTK_IMPORT_TIMEOUT_S), with margin for a
# slow CI box, without letting a single genuinely-hung tool eat minutes.
TOOL_CALL_BUDGET_S = 20.0


def _env(data_dir, allowed_docs_dir):
    env = dict(os.environ)
    env["TDZ_DATA_DIR"] = str(data_dir)
    env["ALLOWED_DOCS_DIRS"] = str(allowed_docs_dir)
    env["AUTO_EXTRACT_ENTITIES"] = "0"
    env["USE_SEMANTIC_SEARCH"] = "0"  # keep this smoke test fast and deterministic
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class MCPSession:
    """Minimal MCP stdio client: spawn a server subprocess and speak
    JSON-RPC to it over one long-lived session.

    Exactly ONE background thread ever reads the subprocess's stdout (started
    once, in __enter__) and pushes whole lines onto a queue. Every request
    method pulls from that queue with a deadline until it sees its own
    request id, ignoring stray/late lines from an earlier call that timed
    out. This is deliberate: spawning a NEW reader thread per call and having
    each one race to call `.readline()` on the same buffered pipe risks one
    thread's blocking syscall permanently holding the pipe's internal lock,
    which would silently defeat every later call's own timeout. One thread,
    one queue, no shared-lock hazard.
    """

    def __init__(self, data_dir, allowed_docs_dir, argv=None):
        self.data_dir = data_dir
        self.allowed_docs_dir = allowed_docs_dir
        self.argv = argv if argv is not None else [PYTHON, str(SERVER)]
        self.proc = None
        self._queue = queue.Queue()
        self._reader_thread = None
        self._stderr_thread = None
        self._stderr_chunks = []
        self._next_id = 1

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_env(self.data_dir, self.allowed_docs_dir),
            cwd=str(REPO),
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        # Drain stderr concurrently too, for the life of the session - not
        # just once at teardown. Measured live: leaving stderr undrained
        # while the session ran ~30 tool calls reproduced a genuine
        # subprocess pipe deadlock (both this process and the server showed
        # threads parked in the Windows EventPairLow wait state - the
        # classic signature of a blocked synchronous pipe read/write) once
        # the server had logged enough to fill stderr's OS pipe buffer
        # (default 64KB) - the server then blocked on its own next write to
        # stderr, mid-dispatch, and never produced another reply again. This
        # is the standard subprocess.PIPE gotcha: stdout and stderr must both
        # be drained concurrently, or whichever fills first can wedge the
        # child. A background thread here, like the stdout reader above,
        # fixes it at the source instead of masking it with a bigger budget.
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        return False

    def _reader_loop(self):
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                if not raw:
                    break
                self._queue.put(raw)
        except (ValueError, OSError):
            pass  # pipe torn down under us during teardown - fine, we're done

    def _stderr_loop(self):
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                if not raw:
                    break
                self._stderr_chunks.append(raw)
        except (ValueError, OSError):
            pass  # pipe torn down under us during teardown - fine, we're done

    def stderr_text(self):
        """Whatever stderr this session's subprocess has produced so far,
        drained continuously by _stderr_loop rather than read in one blocking
        call - see the comment in __enter__ for why a one-shot read here,
        while the process is still alive, previously deadlocked the whole
        session."""
        return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    def _alloc_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def _send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _await_id(self, req_id, timeout):
        """Pull lines from the queue until one matches req_id, or time out."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = self._queue.get(timeout=remaining)
            except queue.Empty:
                return None
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == req_id:
                return msg
            # else: a stray/late reply for an earlier timed-out call - keep waiting

    def initialize(self, timeout):
        """Perform the MCP initialize handshake. Returns (response, elapsed)."""
        start = time.monotonic()
        req_id = self._alloc_id()
        self._send({
            "jsonrpc": "2.0", "id": req_id, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest-dispatch", "version": "1.0"},
            },
        })
        response = self._await_id(req_id, timeout)
        elapsed = time.monotonic() - start
        if response is None:
            return None, elapsed
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return response, elapsed

    def list_tools(self, timeout):
        req_id = self._alloc_id()
        self._send({"jsonrpc": "2.0", "id": req_id, "method": "tools/list", "params": {}})
        return self._await_id(req_id, timeout)

    def call_tool(self, name, arguments, timeout):
        """Returns (response_or_None, elapsed_seconds). None means: no
        matching reply arrived within `timeout` - the caller must treat that
        as a named failure, never wait on it further."""
        start = time.monotonic()
        req_id = self._alloc_id()
        self._send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        response = self._await_id(req_id, timeout)
        return response, time.monotonic() - start


# ---------------------------------------------------------------------------
# Minimal-valid-argument construction (ported from the old in-process test)
# ---------------------------------------------------------------------------

def _special_args(tool_name: str, fixture: dict) -> dict:
    """Tools whose required args need a specific, meaningfully-typed value to
    reach their dispatch code at all (a generic string/int placeholder would
    either KeyError inside a nested lookup unrelated to dispatch correctness,
    or the tool would trivially no-op)."""
    doc_id = fixture["doc_id"]
    tmp = fixture["tmp_path"]
    return {
        "get_chunk": {"doc_id": doc_id, "chunk_id": 0},
        "get_document": {"doc_id": doc_id},
        "update_document": {"card_id_or_doc_id": doc_id, "filepath": fixture["doc_path"]},
        "get_document_by_card_id": {"card_id": doc_id},
        "find_similar": {"doc_id": doc_id},
        "compare_documents": {"doc_id_1": doc_id, "doc_id_2": doc_id},
        "add_document": {"filepath": fixture["doc_path"]},
        "add_documents_bulk": {"directory": str(tmp)},
        "scrape_url": {"url": "http://127.0.0.1:9/does-not-resolve"},  # must fail fast, not hang
        "search_within_results": {
            "previous_results": [{"doc_id": doc_id, "chunk_id": 0, "title": "t", "snippet": "s", "score": 1.0}],
            "refinement_query": "test",
        },
        "export_results": {"results": [{"doc_id": doc_id, "chunk_id": 0, "title": "t", "snippet": "s", "score": 1.0}]},
        "find_by_reference": {"ref_type": "register", "ref_value": "$D020"},
        "create_backup": {"dest_dir": str(tmp / "backup")},
        "restore_backup": {"backup_path": str(tmp / "does-not-exist")},
        "extract_document_events": {"doc_id": doc_id},
        "get_historical_context": {"year": 1982},
        "summarize_document": {"doc_id": doc_id},
        "auto_tag_document": {"doc_id": doc_id},
        "suggest_tags": {"doc_id": doc_id},
        "extract_entities": {"doc_id": doc_id},
        "generate_summary": {"doc_id": doc_id},
        "get_summary": {"doc_id": doc_id},
        "get_document_topics": {"doc_id": doc_id},
        "get_cluster_documents": {"doc_id": doc_id},
        "generate_topic_wordcloud": {"topic_id": 0, "output_path": str(tmp / "wc.png")},
        "visualize_cluster_scatter": {"algorithm": "kmeans", "output_path": str(tmp / "scatter.png")},
        "generate_topic_heatmap": {"model_type": "lda", "output_path": str(tmp / "heatmap.png")},
        "visualize_cluster_distribution": {"algorithm": "kmeans", "output_path": str(tmp / "dist.png")},
        "train_lda_topics": {"model_type": "lda"},
        "cluster_documents_kmeans": {"algorithm": "kmeans"},
        "cluster_documents_dbscan": {"algorithm": "dbscan"},
        "cluster_documents_hdbscan": {"algorithm": "hdbscan"},
    }.get(tool_name, {})


def _placeholder_for(prop_schema: dict):
    if "enum" in prop_schema and prop_schema["enum"]:
        return prop_schema["enum"][0]
    ptype = prop_schema.get("type")
    if ptype == "string":
        return prop_schema.get("default", "test")
    if ptype == "integer":
        return prop_schema.get("default", 1)
    if ptype == "number":
        return prop_schema.get("default", 1.0)
    if ptype == "boolean":
        return prop_schema.get("default", False)
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return "test"


def _build_args(tool: dict, fixture: dict) -> dict:
    """`tool` is the JSON-decoded tools/list entry: {"name": ..., "inputSchema": {...}}."""
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    args = dict(_special_args(tool["name"], fixture))
    for key in required:
        if key in args:
            continue
        args[key] = _placeholder_for(props.get(key, {}))
    return args


# ---------------------------------------------------------------------------
# Fixture: a small, throwaway, populated data dir (never the live database)
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TDZ_DATA_DIR", str(data_dir))
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(tmp_path))
    monkeypatch.setenv("AUTO_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "0")

    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "The VIC-II chip handles sprites and raster interrupts. "
        "The SID chip at $D400 handles sound synthesis.",
        encoding="utf-8",
    )

    # Built and closed BEFORE the subprocess ever touches this directory, so
    # there is no concurrent-writer overlap between this in-process KB and
    # the long-lived server subprocess that will open the same files.
    kb = server_module.KnowledgeBase(str(data_dir))
    try:
        doc = kb.add_document(str(doc_path))
        doc_id = doc.doc_id
    finally:
        kb.close()

    return {
        "data_dir": data_dir,
        "tmp_path": tmp_path,
        "doc_path": str(doc_path),
        "doc_id": doc_id,
    }


@pytest.mark.skipif(not os.getenv('TDZ_RUN_FULL_DISPATCH_TEST'),
                    reason="spawns a real server subprocess and drives all 95 tools over "
                           "one long-lived session (~100s); set TDZ_RUN_FULL_DISPATCH_TEST=1. "
                           "Skipped by default only because of that runtime - it is a sweep, "
                           "not a per-commit guard. The hang it used to reproduce (scrape_url, "
                           "add_deepsid_document, add_deepsid_folder and add_documents_bulk each "
                           "burning the full budget and never replying) was NOT in the "
                           "robots.txt/outbound-network path this reason used to blame: a "
                           "background native import - features.py's nltk warm-up - held the "
                           "Windows loader lock, so every Thread.start() in the process blocked, "
                           "and that import in turn only advanced when the client sent more "
                           "traffic. Fixed in server.py by never leaving a read parked on stdin, "
                           "and pinned per-commit by "
                           "test_thread_starting_tools_return_while_the_nltk_warmup_is_in_flight "
                           "below, which is NOT gated on this flag.")
def test_every_tool_returns_within_bound_over_a_long_lived_stdio_session(populated_data_dir):
    data_dir = populated_data_dir["data_dir"]
    tmp = populated_data_dir["tmp_path"]

    with MCPSession(data_dir, tmp) as session:
        response, elapsed = session.initialize(HANDSHAKE_BUDGET_S)
        assert response is not None, (
            f"MCP handshake did not complete within {HANDSHAKE_BUDGET_S}s"
        )

        tools_response = session.list_tools(LIST_TOOLS_BUDGET_S)
        assert tools_response is not None, (
            f"tools/list did not respond within {LIST_TOOLS_BUDGET_S}s"
        )
        tools = tools_response["result"]["tools"]

        # Read live, not hardcoded, so this test cannot silently stop
        # covering a tool added after it was written.
        expected_count = len(TOOL_SCHEMAS)
        assert len(tools) == expected_count, (
            f"server reported {len(tools)} tools but TOOL_SCHEMAS currently "
            f"has {expected_count} - list_tools() and the schema registry "
            "have drifted apart"
        )

        # A single genuinely-hung handler on this long-lived session can wedge
        # EVERY later call too (observed: once one handler never returns, the
        # rest of the session's dispatch stops responding as well - a real
        # property of the server under stdio, not a flaw in this test). Waiting
        # out the full generous budget for every one of the ~90 remaining tools
        # in that case would take upwards of half an hour. Once several
        # consecutive tools have each burned the full budget with no reply,
        # assume the session is wedged and fall back to a much shorter budget
        # for the rest - every remaining tool is still attempted and still
        # named individually on failure, just without paying the full price
        # per tool once the pattern is already established.
        CONSECUTIVE_TIMEOUTS_BEFORE_DEGRADING = 3
        DEGRADED_BUDGET_S = 2.0

        failures = []
        covered = 0
        consecutive_timeouts = 0
        degraded = False
        for tool in tools:
            covered += 1
            args = _build_args(tool, populated_data_dir)
            budget = DEGRADED_BUDGET_S if degraded else TOOL_CALL_BUDGET_S
            response, elapsed = session.call_tool(tool["name"], args, budget)
            if response is None:
                consecutive_timeouts += 1
                degraded_note = (
                    " [reduced timeout - an earlier hang already appears to "
                    "have wedged the whole session]" if degraded else ""
                )
                failures.append(
                    f"{tool['name']}: no response within {budget}s "
                    f"(args={args}) - HUNG rather than returning{degraded_note}"
                )
                print(f"[{covered}/{expected_count}] {tool['name']}: TIMEOUT after {elapsed:.2f}s", flush=True)
                if not degraded and consecutive_timeouts >= CONSECUTIVE_TIMEOUTS_BEFORE_DEGRADING:
                    degraded = True
                if session.proc.poll() is not None:
                    failures.append(
                        f"{tool['name']}: server process exited "
                        f"(code {session.proc.returncode}) while awaiting this "
                        "call - aborting remaining tools, session is dead"
                    )
                    break
                continue
            consecutive_timeouts = 0
            # A JSON-RPC "error" (bad method/params at the protocol level) or
            # a CallToolResult with isError=true both still count as
            # RETURNING - see module docstring. Only "no response at all"
            # (hang) is a failure here.

        assert covered == expected_count, (
            f"attempted {covered} of {expected_count} tools - coverage "
            "assertion itself is broken"
        )
        assert not failures, "Tools that did not return within the bound:\n" + "\n".join(failures)


def test_thread_starting_tools_return_while_the_nltk_warmup_is_in_flight(populated_data_dir):
    """The loader-lock regression, isolated: two tool calls, no extra traffic.

    search_docs starts features.py's one-shot nltk warm-up on a background
    thread. That import loads native extensions, and on Windows a native
    load holds the process-wide loader lock - while it is held, NO new
    thread can start anywhere in the process, because a new thread cannot
    run its DLL_THREAD_ATTACH callbacks first. threading.Thread.start()
    then sits in _started.wait() with no deadline of its own.

    Both tools below start a thread on the way to their answer (scrape_url
    spawns two readers for mdscrape's output; add_documents_bulk takes the
    cross-process lock, which runs a heartbeat thread), and both are in
    server._LONG_RUNNING_TOOLS, so _run_bounded deliberately gives them no
    timeout either. So while that import ran, they did not return - ever,
    not slowly: the import itself only advanced when the client sent
    another message, and the client was waiting for the reply to the call
    that was stuck.

    Measured at the head this test was written against (2026-09-21, full
    dispatch run, TDZ_RUN_FULL_DISPATCH_TEST=1): scrape_url, then
    add_deepsid_document, add_deepsid_folder and add_documents_bulk, all
    four at 20.00s = the budget, having never replied. With server.py's
    polling stdin reader in place: 4.81s and ~0.6s respectively.

    This test is deliberately NOT gated on TDZ_RUN_FULL_DISPATCH_TEST - it
    costs about ten seconds, against the ~100s of the full sweep, and it is
    the one that must run on every commit. Its assertion is the DISPATCH
    property (does the call come back), not the diagnosis; the warm-up's
    own timing line is printed alongside as corroboration, not asserted,
    so a machine without nltk installed still tests something real.

    NO artificial traffic is sent. Adding any would hide the defect: one
    unrelated client message is exactly what unsticks the stalled import.
    """
    data_dir = populated_data_dir["data_dir"]
    tmp = populated_data_dir["tmp_path"]

    # Both start a thread before they can answer; neither needs the network
    # to succeed, and neither is allowed to need it - scrape_url is pointed
    # at a closed localhost port on purpose. (A connect to 127.0.0.1:9 is
    # NOT instant on Windows: it costs ~2s of SYN retry, twice over here
    # with the robots.txt fetch, which is why the budget is the same
    # generous TOOL_CALL_BUDGET_S the sweep uses and not something tight.)
    thread_starting_calls = [
        ("scrape_url", {"url": "http://127.0.0.1:9/does-not-resolve"}),
        ("add_documents_bulk", {"directory": str(tmp)}),
    ]

    failures = []
    with MCPSession(data_dir, tmp) as session:
        response, elapsed = session.initialize(HANDSHAKE_BUDGET_S)
        assert response is not None, (
            f"MCP handshake did not complete within {HANDSHAKE_BUDGET_S}s"
        )

        # The trigger. It must come back too: before the fix its own reply
        # arrived only because _ensure_nltk gives up on its 5s deadline and
        # degrades - so a slow answer here is itself the defect's shadow.
        response, search_elapsed = session.call_tool(
            "search_docs", {"query": "VIC-II sprites"}, TOOL_CALL_BUDGET_S
        )
        assert response is not None, (
            f"search_docs did not return within {TOOL_CALL_BUDGET_S}s - the "
            "warm-up trigger itself is wedged, so the rest of this test "
            "would be measuring the wrong thing"
        )
        print(f"search_docs (starts the warm-up): {search_elapsed:.2f}s", flush=True)

        for name, args in thread_starting_calls:
            response, elapsed = session.call_tool(name, args, TOOL_CALL_BUDGET_S)
            print(f"{name}: {'returned' if response else 'TIMEOUT'} after {elapsed:.2f}s", flush=True)
            if response is None:
                failures.append(
                    f"{name}: no response within {TOOL_CALL_BUDGET_S}s "
                    f"(args={args}) - HUNG rather than returning, "
                    f"{elapsed:.2f}s after the nltk warm-up was started by "
                    "search_docs and with no other client traffic sent"
                )

        # Diagnostic only. features.py logs this from the warm-up thread
        # itself; if the stall is back the line is simply absent, because
        # the import never finished.
        warmup = [ln for ln in session.stderr_text().splitlines()
                  if "nltk warm-up returned" in ln or "stdin reader" in ln]
        print("\n".join(warmup) or "(no warm-up/stdin-reader line on stderr)", flush=True)

    assert not failures, (
        "Thread-starting tools wedged behind the nltk warm-up's loader "
        "lock:\n" + "\n".join(failures)
    )


def test_a_permanently_blocked_handler_fails_naming_the_tool_not_the_suite(populated_data_dir):
    """SABOTAGE CHECK for the property the test above depends on: that a
    handler which never returns is reported as a NAMED failure within the
    bound, rather than hanging the test process.

    Monkeypatching the real `server_module` in THIS process would have no
    effect on the separate long-lived subprocess the dispatch test above
    talks to, so the sabotage here patches the target subprocess's own copy
    of `server.HANDLERS` before it starts serving: the subprocess is launched
    as `python -c <script>` (instead of `python server.py`) where the script
    imports server, overwrites one entry of server.HANDLERS with a function
    that blocks forever, and only then calls server.main().
    """
    target = "health_check"  # takes {} - no required args, so no fixture data needed

    import mcp_tools.admin as admin_mod
    original_source = inspect.getsource(admin_mod.handle_health_check)
    source_file = inspect.getsourcefile(admin_mod.handle_health_check)
    _, first_lineno = inspect.getsourcelines(admin_mod.handle_health_check)
    # ascii-safe: the real source contains non-ASCII checkmark glyphs
    # (U+2713/U+2717) that crash a plain print() under a cp1252 console -
    # printing the encoded/escaped form instead still proves we located and
    # are about to shadow the real region on disk, without that hazard.
    print(
        f"--- about to shadow handle_{target}, defined at "
        f"{source_file}:{first_lineno} ---\n"
        f"{original_source.encode('ascii', errors='backslashreplace').decode('ascii')}"
    )
    assert f"def handle_{target}(" in original_source  # prove we found the real region

    sabotage_script = (
        "import sys, time, asyncio\n"
        "import server\n"
        "def _wedge(kb, name, arguments):\n"
        "    time.sleep(10 ** 6)\n"
        "    return []\n"
        f"server.HANDLERS[{target!r}] = _wedge\n"
        f"print('SABOTAGE-ACTIVE ' + repr(server.HANDLERS[{target!r}]), "
        "file=sys.stderr, flush=True)\n"
        "asyncio.run(server.main())\n"
    )

    data_dir = populated_data_dir["data_dir"]
    tmp = populated_data_dir["tmp_path"]
    failures = []
    with MCPSession(data_dir, tmp, argv=[PYTHON, "-c", sabotage_script]) as session:
        response, elapsed = session.initialize(HANDSHAKE_BUDGET_S)
        assert response is not None, (
            "sabotaged server did not even complete the handshake - the "
            "sabotage script itself is broken, not the property under test"
        )

        response, elapsed = session.call_tool(target, {}, timeout=8.0)
        if response is None:
            failures.append(
                f"{target}: no response within 8.0s - HUNG rather than returning"
            )

    # stderr_text() does a blocking read() to EOF on the subprocess's stderr
    # pipe. Called while the sabotaged process is still alive (still inside
    # the `with` block above), that read never sees EOF - the process is
    # deliberately wedged and never exits on its own - and hangs forever,
    # long past this test's own 8.0s call_tool bound. __exit__ (on the `with`
    # block above) kills the process first; only THEN does its stderr pipe
    # reach EOF and this read return. This ordering is why the docstring on
    # stderr_text() says "only meaningful after the process has been
    # killed/exited" - that was a requirement the call site below did not
    # honour before this fix. Confirmed by direct measurement: with the read
    # inside the `with` block, this single test hung for 4+ real minutes
    # (killed manually) instead of finishing in a few seconds.
    stderr_text = session.stderr_text()

    assert failures, (
        f"expected the sabotaged {target!r} handler to be reported as a "
        "named failure by the timeout machinery - instead it returned, so "
        "the sabotage never took effect"
    )
    assert target in failures[0], f"failure did not name the tool: {failures}"

    # Prove the mutation actually landed in the subprocess, not just that we
    # asked it to.
    assert "SABOTAGE-ACTIVE" in stderr_text, (
        "the sabotage marker never appeared on the subprocess's stderr - the "
        "monkeypatch may not have landed before the server started serving "
        f"requests. stderr was:\n{stderr_text}"
    )
