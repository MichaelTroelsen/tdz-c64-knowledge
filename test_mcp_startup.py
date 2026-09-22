"""Regression tests for MCP server startup and multi-session connectivity.

These guard the failure that made the server unusable from Claude Code: the
server imported sentence-transformers, torch and nltk eagerly at module load,
costing ~16s of an ~18s startup. A single session just barely fit inside the
MCP client's 30s handshake timeout; as soon as a second or third Claude Code
session started a server against the same database, contention pushed the
handshake past 30s and every session failed to connect. Reconnecting made it
worse, because each retry spawned another competing process.

The fixes under test:
  1. Heavy optional dependencies are imported on first use, not at module load.
  2. The SQLite database runs in WAL mode so concurrent server processes do
     not serialise behind a single exclusive writer lock.
  3. stdout carries only JSON-RPC - any stray print would corrupt the stream.
  4. Those same first-use imports (nltk corpora, the sentence-transformers
     model) fetch data over the network with no timeout of their own. On a
     blocked or filtered connection they hung forever instead of degrading,
     taking every search_docs/semantic_search call down with them (issue #12).
     _network_timeout() bounds those fetches so they fail fast and fall back.
  5. Two sessions starting at once against a BRAND-NEW data directory (no
     database file yet) raced on schema creation: sqlite3.connect() creates
     the physical .db file immediately even with zero tables, so a
     file-existence check is not a reliable "has the schema been created"
     signal once two processes are involved. This crashed one process
     outright with "table already exists" or "no such table: documents"
     depending on timing. _init_database() now serialises schema creation
     and WAL activation behind a cross-process lock, and every CREATE
     statement is idempotent (IF NOT EXISTS) so a racing process safely
     no-ops through whatever a peer already committed.

Run with:  pytest test_mcp_startup.py -v
"""

import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parent
SERVER = REPO / "server.py"

# Prefer the project venv so the test measures what Claude Code actually launches.
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

# The MCP client gives a server 30s to complete the initialize handshake.
# Budget well under that so the test fails while there is still headroom,
# rather than only once users are already being disconnected.
HANDSHAKE_BUDGET_S = 15.0
CONCURRENT_SESSIONS = 4
CONCURRENT_BUDGET_S = 25.0

# Modules that must NOT be imported during startup. Each costs seconds.
FORBIDDEN_AT_STARTUP = ["sentence_transformers", "torch", "transformers", "nltk", "markitdown", "magika", "onnxruntime"]


def _env(data_dir):
    env = dict(os.environ)
    env["TDZ_DATA_DIR"] = str(data_dir)
    env["USE_FTS5"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep optional features enabled: the point is that enabling them must not
    # force their heavy imports at startup.
    env["USE_SEMANTIC_SEARCH"] = "1"
    env["USE_QUERY_PREPROCESSING"] = "1"
    return env


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Isolated data dir so tests never touch the user's real knowledge base."""
    return tmp_path_factory.mktemp("tdz_data")


class MCPSession:
    """Minimal MCP stdio client: spawn server.py and speak JSON-RPC to it."""

    def __init__(self, data_dir, client_name="pytest", extra_env=None):
        self.data_dir = data_dir
        self.client_name = client_name
        self.extra_env = dict(extra_env or {})
        self.proc = None
        self.stderr = b""

    def __enter__(self):
        self.proc = subprocess.Popen(
            [PYTHON, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**_env(self.data_dir), **self.extra_env},
            cwd=str(REPO),
        )
        # Drain stderr for the life of the session. Without this the server
        # blocks on its own stderr write as soon as the pipe fills, mid-startup,
        # and the handshake simply never arrives - a hang with no diagnostic,
        # indistinguishable from a slow import. The buffer is far smaller than
        # the 64KB usually quoted: measured 2026-09-21, the server wrote 3911
        # bytes of startup logging and fitted, then 4181 bytes (the same lines
        # with a pid added to the format) and wedged, so the effective limit
        # here is 4096. That is 185 bytes of headroom, which is no headroom at
        # all - adding one log line, or one field to the format, was enough to
        # make every session in this file fail to connect.
        self._stderr_chunks = []
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True, name="mcpsession-stderr",
        )
        self._stderr_thread.start()
        return self

    def _stderr_loop(self):
        for line in iter(self.proc.stderr.readline, b""):
            self._stderr_chunks.append(line)

    def stderr_text(self):
        """What the server logged. Safe to call while the session is alive,
        because the reader thread owns the pipe - reading it directly here
        would block until the process exits."""
        return b"".join(self._stderr_chunks).decode("utf-8", "replace")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        return False

    def _send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def _readline(self, timeout):
        """Read one line from stdout, or return None if it takes too long."""
        box = []
        t = threading.Thread(target=lambda: box.append(self.proc.stdout.readline()), daemon=True)
        t.start()
        t.join(timeout)
        return box[0] if box else None

    def initialize(self, timeout):
        """Perform the MCP initialize handshake. Returns (response, elapsed)."""
        start = time.time()
        self._send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": "1.0"},
            },
        })
        line = self._readline(timeout)
        elapsed = time.time() - start
        if line is None:
            return None, elapsed
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return json.loads(line), elapsed

    def list_tools(self, timeout=30):
        self._send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        line = self._readline(timeout)
        return json.loads(line) if line else None

    def call_tool_frames(self, name, arguments=None, progress_token=None,
                         req_id=7, timeout=60):
        """Call a tool and return EVERY frame read until its response arrives.

        Deliberately returns the raw frames in wire order rather than just the
        result: notifications/progress is a separate JSON-RPC frame the server
        pushes mid-call, and the only honest proof a client can observe it is
        to read it off stdout BEFORE the response with the matching id.
        """
        params = {"name": name, "arguments": dict(arguments or {})}
        if progress_token is not None:
            params["_meta"] = {"progressToken": progress_token}
        self._send({"jsonrpc": "2.0", "id": req_id,
                    "method": "tools/call", "params": params})
        frames = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._readline(max(0.5, deadline - time.time()))
            if not line:
                break
            frame = json.loads(line)
            frames.append(frame)
            if frame.get("id") == req_id and ("result" in frame or "error" in frame):
                break
        return frames


def test_handshake_completes_within_budget(data_dir):
    """A single session must connect well inside the client's 30s timeout."""
    with MCPSession(data_dir) as s:
        response, elapsed = s.initialize(timeout=HANDSHAKE_BUDGET_S)

    assert response is not None, (
        f"MCP handshake did not complete within {HANDSHAKE_BUDGET_S}s. "
        "This is the failure that stops Claude Code sessions connecting."
    )
    assert "result" in response, f"initialize returned an error: {response}"
    assert elapsed < HANDSHAKE_BUDGET_S, f"handshake took {elapsed:.1f}s"


# The MCP client owns the server's stderr pipe under the stdio transport, and
# the server blocks on its own write the moment that pipe is full. The
# effective buffer measured here is 4096 bytes, not the 64KB usually quoted,
# and before daf145c/this change the INFO startup banner alone was 3911 of
# them: adding a pid to the log format wedged every session in this file.
# The fix is that stderr carries WARNING and above by default while
# server.log keeps INFO. The ceiling below is a quarter of the buffer - room
# for a handful of genuine warnings, and a loud failure the moment INFO
# leaks back onto the pipe.
STARTUP_STDERR_BUDGET_BYTES = 1024


def _startup_stderr_bytes(data_dir, extra_env=None):
    with MCPSession(data_dir, client_name="stderr-budget", extra_env=extra_env) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"
        # Let the post-handshake log lines (if any) land before we measure.
        time.sleep(0.5)
        text = s.stderr_text()
    return len(text.encode("utf-8")), text


def test_startup_writes_well_under_the_stderr_pipe_buffer(data_dir):
    """A client that does not drain stderr must still get a handshake."""
    n, text = _startup_stderr_bytes(data_dir)
    assert n < STARTUP_STDERR_BUDGET_BYTES, (
        f"server wrote {n} bytes to stderr before/at startup, budget is "
        f"{STARTUP_STDERR_BUDGET_BYTES} (pipe buffer is 4096). An undrained "
        f"client would wedge. stderr was:\n{text[-1500:]}"
    )
    assert " - INFO - " not in text, (
        "INFO-level lines reached stderr; they belong in server.log only. "
        f"stderr was:\n{text[-1500:]}"
    )


def test_stderr_log_level_knob_restores_info_on_the_console(data_dir):
    """The default is a policy, not a lost capability: TDZ_STDERR_LOG_LEVEL=INFO
    puts the startup trace back on stderr for someone running the server by
    hand. This also proves the budget test above passes because of the level
    split, not because the server went quiet."""
    n, text = _startup_stderr_bytes(data_dir, {"TDZ_STDERR_LOG_LEVEL": "INFO"})
    assert " - INFO - " in text, f"INFO did not reach stderr with the knob set:\n{text[-1500:]}"
    assert n > STARTUP_STDERR_BUDGET_BYTES, (
        f"only {n} bytes with INFO enabled - the budget test is not "
        "discriminating anything if INFO output fits under it"
    )


def test_serverinfo_reports_project_version(data_dir):
    """serverInfo.version must be the project version, not the installed mcp SDK's."""
    from version import __version__

    with MCPSession(data_dir) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)

    assert response is not None, "handshake failed"
    assert "result" in response, f"initialize returned an error: {response}"
    server_info = response["result"]["serverInfo"]
    assert server_info["version"] == __version__, (
        f"serverInfo.version={server_info['version']!r} != version.__version__={__version__!r}"
    )


def test_tools_list_after_handshake(data_dir):
    """The server must actually serve its tool list, not just handshake."""
    with MCPSession(data_dir) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"
        tools = s.list_tools()

    assert tools is not None, "tools/list timed out"
    assert "result" in tools, f"tools/list returned an error: {tools}"
    names = [t["name"] for t in tools["result"]["tools"]]
    assert len(names) > 50, f"expected the full tool set, got {len(names)}"
    assert "search_docs" in names
    assert len(names) == len(set(names)), "duplicate tool names registered"


def test_concurrent_sessions_all_connect(data_dir):
    """Several Claude Code sessions starting at once must all connect.

    This is the regression that broke real usage: startup was slow enough that
    concurrent sessions pushed each other past the client's handshake timeout.
    """
    results = {}

    def connect(i):
        # Spawning several subprocesses from concurrent threads at once can hit
        # a transient Windows pipe/handle error (observed: OSError(22) from
        # stdin.write) that has nothing to do with the handshake-timeout race
        # this test targets. One retry with a fresh subprocess absorbs that
        # OS-level noise without masking a real handshake failure - a second
        # occurrence still fails the test.
        last_exc = None
        for attempt in range(2):
            try:
                with MCPSession(data_dir, client_name=f"session-{i}") as s:
                    response, elapsed = s.initialize(timeout=CONCURRENT_BUDGET_S)
                    results[i] = (response is not None and "result" in (response or {}), elapsed)
                return
            except OSError as e:
                last_exc = e
        results[i] = (False, f"exception: {last_exc!r}")  # pragma: no cover - diagnostic path

    threads = [threading.Thread(target=connect, args=(i,)) for i in range(CONCURRENT_SESSIONS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = {i: v for i, v in results.items() if not v[0]}
    assert not failed, (
        f"{len(failed)}/{CONCURRENT_SESSIONS} concurrent sessions failed to connect "
        f"within {CONCURRENT_BUDGET_S}s: {failed}"
    )
    assert len(results) == CONCURRENT_SESSIONS


def test_heavy_modules_not_imported_at_startup(data_dir):
    """Guard the root cause directly: no multi-second imports during startup.

    Fails loudly if someone reintroduces a top-level `import nltk` or
    `from sentence_transformers import ...`, which is what caused the outage.
    """
    probe = (
        "import sys, json; import server; "
        f"print(json.dumps([m for m in {FORBIDDEN_AT_STARTUP!r} if m in sys.modules]))"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, f"importing server failed: {out.stderr.decode(errors='replace')[-2000:]}"

    leaked = json.loads(out.stdout.decode().strip().splitlines()[-1])
    assert leaked == [], (
        f"These heavy modules were imported at startup: {leaked}. "
        "They must be imported on first use - importing them eagerly adds "
        "~16s to startup and breaks MCP handshakes for concurrent sessions."
    )


def test_importing_server_does_not_construct_a_knowledgebase(data_dir):
    """`import server` must not build a KnowledgeBase as a side effect.

    rest_server.py, admin_gui.py, cli.py and wiki_export.py all do
    `from server import KnowledgeBase`, which - like any import - executes
    the whole module regardless of which name is requested. A module-level
    `kb = KnowledgeBase(DATA_DIR)` used to make every one of those imports
    build a full second KnowledgeBase (DB connection, background extraction
    worker thread, document load) in addition to the instance the consumer
    then built for itself. server.get_kb() defers that cost to the one
    process that actually calls it (real usage: main(), before serving any
    MCP requests).
    """
    probe = "import server; print('KB_IS_NONE', server.kb is None)"
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("KB_IS_NONE")][-1]
    assert line == "KB_IS_NONE True", (
        f"importing server built a KnowledgeBase eagerly: {line!r}. "
        "Every module-level consumer of `from server import KnowledgeBase` "
        "now pays that cost redundantly."
    )


def test_optional_features_still_enabled(data_dir):
    """Lazy importing must not silently disable semantic search or NLTK."""
    probe = "import server; print('FLAGS', server.SEMANTIC_SUPPORT, server.NLTK_SUPPORT)"
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("FLAGS")][-1]
    assert line == "FLAGS True True", (
        f"optional feature detection regressed: {line!r}. Detection must not "
        "depend on the modules having been imported."
    )


def test_lazy_dependencies_work_on_first_use(data_dir):
    """The deferred imports must actually resolve when a feature is used."""
    probe = (
        "import server, numpy as np;"
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR']);"
        "toks = kb._preprocess_text('The VIC-II sprite collision registers');"
        "idx = server.faiss.IndexFlatIP(4);"
        "v = np.array([[1,0,0,0]], dtype='float32');"
        "server.faiss.normalize_L2(v); idx.add(v);"
        "print('LAZY', 'sprite' in toks, 'the' not in toks, idx.ntotal)"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=300,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-3000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("LAZY")][-1]
    assert line == "LAZY True True 1", (
        f"lazy dependency resolution failed: {line!r}. Expected nltk stemming/"
        "stopword removal and the faiss proxy to work on first use."
    )


def test_stdout_carries_only_jsonrpc(data_dir):
    """Any non-JSON byte on stdout corrupts the MCP stream and kills the client.

    Logging goes to stderr; this guards against a stray print() reaching stdout.
    """
    with MCPSession(data_dir) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"
        tools = s.list_tools()
        assert tools is not None and "result" in tools

    # Every line the server emitted on stdout parsed as JSON-RPC above; if a
    # warning had been printed to stdout it would have been read as the
    # handshake response and failed to parse.
    assert response.get("jsonrpc") == "2.0"
    assert "protocolVersion" in response["result"]


def test_log_format_includes_pid_so_concurrent_servers_are_distinguishable(tmp_path):
    """Every MCP session spawns its own server.py, and they all append to one
    shared server.log in TDZ_DATA_DIR. Without a pid in the format, two
    interleaved processes' lines cannot be told apart, and a diagnosis read
    off the log can attribute one process's line to another.

    Configures the real logging setup (kb.core.CoreMixin.__init__) in a
    subprocess and asserts the FORMATTED log line contains that subprocess's
    actual pid - not just that the format string literal contains the pid
    placeholder, which would pass even if a different format were the one
    actually installed.
    """
    log_dir = tmp_path / "logtest"
    log_dir.mkdir()
    probe = "\n".join([
        "import os",
        "import kb.core",
        "try:",
        f"    kb.core.CoreMixin({str(log_dir)!r})",
        "except Exception:",
        "    pass  # CoreMixin is not standalone; the logging setup at the",
        "          # top of __init__ has already run by the time later",
        "          # mixin-only attributes are missing.",
        "print('PID', os.getpid())",
    ])
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(log_dir), cwd=str(REPO), timeout=60,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    stdout = out.stdout.decode(errors="replace")
    pid_line = [ln for ln in stdout.splitlines() if ln.startswith("PID")][-1]
    pid = pid_line.split()[1]

    log_text = (log_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    assert re.search(rf"\b{re.escape(pid)}\b", log_text), (
        f"server.log does not contain this process's pid ({pid}) anywhere, "
        "so the configured logging format is not actually emitting it. "
        f"log tail: {log_text[-800:]!r}"
    )


def test_database_uses_wal_mode(data_dir):
    """WAL keeps concurrent server processes from blocking each other.

    In the default 'delete' journal mode a writer takes an exclusive lock on
    the whole file, stalling readers in every other Claude Code session.
    """
    import sqlite3

    with MCPSession(data_dir) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"

    db = Path(data_dir) / "knowledge_base.db"
    assert db.exists(), "server did not create its database"

    # proc.kill() is TerminateProcess on Windows - abrupt, no orderly sqlite3
    # close - and proc.wait() only confirms the process exited, not that
    # Windows has finished releasing its handle on the WAL/SHM mmap. Opening
    # the file in that gap can transiently raise "disk I/O error"; retry
    # rather than treating it as a real WAL-mode regression.
    for attempt in range(5):
        try:
            conn = sqlite3.connect(str(db))
            try:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            finally:
                conn.close()
            break
        except sqlite3.OperationalError:
            if attempt == 4:
                raise
            time.sleep(0.2)
    assert str(mode).lower() == "wal", f"journal_mode is {mode!r}, expected 'wal'"


_FRESH_START_WORKER = """
import os, sys, time
import server

filename = sys.argv[1]
barrier_dir = sys.argv[2]
nworkers = int(sys.argv[3])

kb = server.get_kb()  # lazily constructs and caches the module-level singleton

path = os.path.join(os.environ['TDZ_DATA_DIR'], filename)
with open(path, 'w') as f:
    f.write('Concurrent fresh-start regression test document.')

# Barrier: make both processes reach add_document at (approximately) the
# same moment, so their schema-initialisation windows actually overlap -
# the scenario that used to crash one process outright.
open(os.path.join(barrier_dir, filename + '.ready'), 'w').close()
deadline = time.time() + 60
while time.time() < deadline:
    if len([f for f in os.listdir(barrier_dir) if f.endswith('.ready')]) >= nworkers:
        break
    time.sleep(0.05)
else:
    raise SystemExit('barrier timed out')

doc = kb.add_document(path)
print(f"ADDED {doc.doc_id}")
"""


def test_concurrent_fresh_start_does_not_crash_on_schema_creation(tmp_path):
    """Two sessions starting at once against a brand-new data dir must not crash.

    Regression for a startup race distinct from WAL mode itself: sqlite3.
    connect() creates the physical .db file immediately even with zero
    tables, so a file-existence check alone can't tell "schema not created
    yet" from "another process is creating it right now". Before the fix,
    this reliably crashed one of the two processes with either
    "table already exists" or "no such table: documents" depending on exact
    timing - reproduced directly during development, roughly 1 run in 5-10
    against a fresh temp directory.
    """
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    data_dir = tmp_path / "data"  # deliberately NOT created - must not exist yet

    env = dict(os.environ)
    env["TDZ_DATA_DIR"] = str(data_dir)
    env["ALLOWED_DOCS_DIRS"] = str(data_dir)
    env["AUTO_EXTRACT_ENTITIES"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"

    logs = []
    procs = []
    for name in ("docA.txt", "docB.txt"):
        out_f = open(tmp_path / f"{name}.out", "wb")
        err_f = open(tmp_path / f"{name}.err", "wb")
        logs.append((name, out_f, err_f))
        procs.append(subprocess.Popen(
            [PYTHON, "-c", _FRESH_START_WORKER, name, str(barrier_dir), "2"],
            env=env, cwd=str(REPO), stdout=out_f, stderr=err_f,
        ))

    results = []
    for (name, out_f, err_f), proc in zip(logs, procs):
        rc = proc.wait(timeout=180)
        out_f.close()
        err_f.close()
        results.append((name, rc))

    failures = []
    for name, rc in results:
        if rc != 0:
            err_text = (tmp_path / f"{name}.err").read_text(errors="replace")
            failures.append(f"{name} exited {rc}:\n{err_text[-2000:]}")
    assert not failures, "\n\n".join(failures)


def test_server_exits_when_client_disconnects(data_dir):
    """A disconnected server must exit, not linger holding a DB connection.

    Orphaned processes accumulate across reconnect attempts and each one keeps
    a database connection and background worker alive, so a session that
    failed to connect makes the next attempt worse rather than better.
    """
    with MCPSession(data_dir) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"

        s.proc.stdin.close()  # simulate the client going away
        try:
            s.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "server was still running 20s after its client disconnected; "
                "orphaned processes pile up on every reconnect attempt"
            )
        assert s.proc.returncode == 0, f"unclean exit: rc={s.proc.returncode}"


def test_busy_timeout_is_generous(data_dir):
    """A 5s busy timeout is too short when several sessions contend."""
    probe = (
        "import server;"
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR']);"
        "print('BUSY', kb.db_conn.execute('PRAGMA busy_timeout').fetchone()[0])"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("BUSY")][-1]
    timeout_ms = int(line.split()[1])
    assert timeout_ms >= 15000, f"busy_timeout is only {timeout_ms}ms"


# --- Issue #12: search_docs/semantic_search hung forever on a blocked network ---
#
# Root cause: _preprocess_text() (called by every search_docs query) and
# _ensure_embeddings_loaded() (called by every semantic_search query) each
# lazily fetch a network resource - NLTK corpora and the sentence-transformers
# model - on first use. Neither fetch had a timeout, so a filtered/unreachable
# network blocked the call forever instead of raising. health_check never
# touches either path, which is why it kept responding while every search hung.
#
# These tests simulate "network present but never responds" with a real local
# socket that accepts a connection and then never sends anything back, which
# is exactly the shape of a silently-dropped/firewalled connection. They fail
# (by hanging, tripping pytest's own timeout) if the bounding fix regresses.

def test_network_timeout_bounds_a_stalled_connection(data_dir):
    """_network_timeout() must bound a socket that never responds.

    Unit-level check of the helper itself: a socket created *inside* the
    context (matching how nltk/requests open their connection during the
    call) must pick up the temporary default timeout and raise rather than
    block forever, and the previous default must be restored on exit.
    """
    probe = (
        "import server, socket, time\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "start = time.time()\n"
        "raised = False\n"
        "try:\n"
        "    with server._network_timeout(1.0):\n"
        "        client = socket.socket()\n"
        "        client.connect(('127.0.0.1', port))\n"
        "        client.recv(1)\n"
        "except socket.timeout:\n"
        "    raised = True\n"
        "elapsed = time.time() - start\n"
        "print('RESULT', raised, elapsed < 10, socket.getdefaulttimeout() is None)\n"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT True True True", (
        f"_network_timeout did not bound the stalled connection: {line!r}"
    )


def test_ensure_nltk_degrades_instead_of_hanging_on_a_stalled_download(data_dir):
    """The exact issue #12 failure for search_docs: nltk.download() hangs.

    Simulates missing corpora (stopwords.words() raising LookupError, as it
    does the first time NLTK data isn't on disk) plus a download that connects
    but never responds. Before the fix this blocked forever; now it must
    return within the configured network timeout and leave the KB usable
    (falling back to unpreprocessed search) rather than wedging every future
    search behind the same doomed download.
    """
    probe = (
        # _ensure_nltk (and the _nltk_ready flag it reads/writes via `global`)
        # live in features.py, not server.py - server.py only re-imports the
        # function itself, so the readiness flag must be poked on the module
        # that actually owns it or this setup/assertion silently no-ops.
        "import server, features, socket, nltk.corpus, time\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def boom(*a, **k):\n"
        "    raise LookupError('simulated: corpus not installed')\n"
        "nltk.corpus.stopwords.words = boom\n"
        "def fake_download(*a, **k):\n"
        "    c = socket.socket()\n"
        "    c.connect(('127.0.0.1', port))\n"
        "    c.recv(1)  # never arrives - simulates a filtered/dead connection\n"
        "nltk.download = fake_download\n"
        "features._nltk_ready = False\n"
        "start = time.time()\n"
        "result = server._ensure_nltk()\n"
        "elapsed = time.time() - start\n"
        "print('RESULT', elapsed < 10, result is not None, features._nltk_ready)\n"
    )
    env = _env(data_dir)
    env["NETWORK_FETCH_TIMEOUT_S"] = "2"
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=env, cwd=str(REPO), timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT True True True", (
        f"_ensure_nltk did not degrade within the network timeout: {line!r}. "
        "This is the exact hang reported in issue #12 for search_docs."
    )


# --- A search over a LONG-LIVED stdio session must actually return ---
#
# Distinct from issue #12 above, and it does not reproduce in a fresh process:
# a one-shot interpreter answered the identical query in 0.8s while a
# long-lived stdio server left it unanswered for 90s. Measured cause
# (2026-09-20): _preprocess_text's deferred `import nltk` pulls in
# nltk.collocations -> scipy.stats -> scipy.linalg, a chain of C-extension DLL
# loads that each touch this process's stdin. Windows serialises operations on
# a synchronous file object, and the stdio transport always has a reader thread
# parked in a read on stdin that only completes when the client sends its next
# message - which it never does, because it is waiting for this search. Poking
# that wedged server with one extra request per second let the search finish
# after exactly 11 of them.
#
# So this test must NOT write anything else to stdin while it waits: every
# extra byte advances the very import whose stall it is trying to catch. And
# it must fail on a deadline rather than hang, which _readline() gives it.
FIRST_SEARCH_BUDGET_S = 60.0


def test_first_search_returns_over_a_long_lived_stdio_session(data_dir):
    """search_docs must answer on a session that stays open - see above."""
    with MCPSession(data_dir, client_name="first-search") as s:
        # The drain this test used to start for itself now lives in
        # MCPSession.__enter__, because every session needed it, not just this
        # one. Two readers on one pipe would steal lines from each other.
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"

        start = time.time()
        s._send({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {
                "name": "search_docs",
                "arguments": {"query": "raster interrupt stable timing", "max_results": 3},
            },
        })
        line = s._readline(FIRST_SEARCH_BUDGET_S)
        elapsed = time.time() - start

    assert line is not None, (
        f"search_docs never answered over a long-lived stdio session "
        f"({FIRST_SEARCH_BUDGET_S}s). The deferred nltk import is waiting on "
        f"stdin behind the transport's own pending read, and the client that "
        f"would release it is blocked on this reply."
    )
    payload = json.loads(line)
    assert payload.get("id") == 7, payload
    assert "result" in payload, payload
    assert elapsed < FIRST_SEARCH_BUDGET_S, f"search took {elapsed:.1f}s"


def test_ensure_nltk_returns_on_a_deadline_instead_of_waiting_for_the_import(data_dir):
    """The bound itself: a one-time import that never finishes must not be
    waited on forever. Stands in for the stdin stall above without needing a
    live stdio session, and fails fast if the deadline is ever removed."""
    probe = chr(10).join([
        "import features, threading, time",
        # An import that can never complete - the shape the stdin stall has
        # from the caller side.
        "features._nltk_import_worker = lambda: threading.Event().wait()",
        "start = time.time()",
        "raised = ''",
        "try:",
        "    features._ensure_nltk(timeout=2.0)",
        "except TimeoutError:",
        "    raised = 'TimeoutError'",
        "print('RESULT', raised, time.time() - start < 10)",
    ])
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=120,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT TimeoutError True", (
        f"_ensure_nltk did not return on its deadline: {line!r}. Without that "
        "bound, every search_docs/semantic_search/fuzzy_search/answer_question "
        "call inherits an unbounded wait on the first one."
    )


def test_ensure_embeddings_degrades_instead_of_hanging_when_offline(data_dir):
    """The exact issue #12 failure for semantic_search: model load hangs.

    Simulates a model that isn't cached locally (SentenceTransformer raises
    with local_files_only=True) and a network fetch that connects but never
    responds. Must return within the network timeout and disable semantic
    search rather than hanging every future semantic_search call.
    """
    probe = (
        "import server, socket, time\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def fake_st(model_name, local_files_only=False):\n"
        "    if local_files_only:\n"
        "        raise OSError('simulated: model not cached locally')\n"
        "    c = socket.socket()\n"
        "    c.connect(('127.0.0.1', port))\n"
        "    c.recv(1)  # never arrives - simulates a filtered/dead connection\n"
        "import kb.search._retrieval as _r; _r.SentenceTransformer = fake_st\n"
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR'])\n"
        "assert kb.use_semantic, 'test setup: semantic search should be enabled'\n"
        "start = time.time()\n"
        "kb._ensure_embeddings_loaded()\n"
        "elapsed = time.time() - start\n"
        "print('RESULT', elapsed < 10, kb.use_semantic is False, kb._embeddings_loaded is False)\n"
    )
    env = _env(data_dir)
    env["NETWORK_FETCH_TIMEOUT_S"] = "2"
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=env, cwd=str(REPO), timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT True True True", (
        f"_ensure_embeddings_loaded did not degrade within the network timeout: {line!r}. "
        "This is the exact hang reported in issue #12 for semantic_search."
    )


def test_ensure_embeddings_degrades_when_cached_path_itself_hangs(data_dir):
    """Issue #14: the model is already fully cached locally, but the
    local_files_only=True call itself hangs.

    Observed in production: a single add_document call (887-page PDF) fully
    extracted, chunked, and committed to the database in ~4 minutes, then the
    process hung for 28+ minutes with zero further CPU usage while "lazy
    loading embeddings model" - despite the model already being present in
    the huggingface cache. Some sentence-transformers/huggingface_hub
    versions still issue a revision/etag check even with local_files_only=
    True, and a silently dropped/filtered connection to that check has no
    timeout of its own. Every add_document call in a fresh process hits this
    path (server.py ~line 4243), so this is not a rare edge case.

    Must return within the network timeout and disable semantic search
    rather than hanging the caller (and therefore every other request on the
    same single-threaded MCP session) indefinitely.
    """
    probe = (
        "import server, socket, time\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def fake_st(model_name, local_files_only=False):\n"
        "    # Hangs the same way regardless of local_files_only - simulates\n"
        "    # a version where the 'cached' path still reaches out to the\n"
        "    # network and that connection is silently dropped/filtered.\n"
        "    c = socket.socket()\n"
        "    c.connect(('127.0.0.1', port))\n"
        "    c.recv(1)  # never arrives\n"
        "    raise AssertionError('unreachable')\n"
        "import kb.search._retrieval as _r; _r.SentenceTransformer = fake_st\n"
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR'])\n"
        "assert kb.use_semantic, 'test setup: semantic search should be enabled'\n"
        "start = time.time()\n"
        "kb._ensure_embeddings_loaded()\n"
        "elapsed = time.time() - start\n"
        "print('RESULT', elapsed < 10, kb.use_semantic is False, kb._embeddings_loaded is False)\n"
    )
    env = _env(data_dir)
    env["NETWORK_FETCH_TIMEOUT_S"] = "2"
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=env, cwd=str(REPO), timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT True True True", (
        f"_ensure_embeddings_loaded did not degrade when the local_files_only=True "
        f"path itself hung: {line!r}. This is the exact hang reported in issue #14."
    )


def test_add_documents_bulk_does_not_block_the_event_loop(data_dir, tmp_path):
    """Issue #13: add_documents_bulk blocked every other MCP request.

    The call_tool handler invoked kb.add_documents_bulk(...) directly (no
    await/to_thread). That method uses a ThreadPoolExecutor internally, but
    its driving loop - a plain `for future in as_completed(...)` - is a
    blocking call that ran on the asyncio event loop thread itself. Observed
    in production: a 29-file bulk ingest of large PDFs blocked the entire
    session for over 30 minutes, during which even a trivial, read-only
    kb_stats call on the same session queued behind it and eventually timed
    out client-side without ever being serviced.

    Simulates a bulk add that takes a few seconds (monkeypatched) and asserts
    a concurrent kb_stats call on the same event loop still completes well
    before the bulk operation does, proving the loop was not blocked.

    The per-tool asyncio.to_thread that originally fixed this is gone: call_tool
    now runs the whole _call_tool_impl dispatch on a worker thread, so this test
    (and the scrape_url one below) exercise that single generic mechanism.
    """
    bulk_dir = tmp_path / "bulkdir"
    bulk_dir.mkdir()
    (bulk_dir / "a.txt").write_text("hello world")

    probe = (
        "import asyncio, sys, time, server\n"
        "def slow_bulk(self, *a, **kw):\n"
        "    time.sleep(3)\n"
        "    return {'added': [], 'skipped': [], 'failed': []}\n"
        "server.KnowledgeBase.add_documents_bulk = slow_bulk\n"
        "async def main():\n"
        "    t0 = time.time()\n"
        "    bulk_task = asyncio.create_task(server.call_tool(\n"
        "        'add_documents_bulk',\n"
        "        {'directory': sys.argv[1], 'pattern': '*.txt', 'recursive': False},\n"
        "    ))\n"
        "    await asyncio.sleep(0)  # let bulk_task start and reach the blocking call\n"
        "    await server.call_tool('kb_stats', {})\n"
        "    kb_elapsed = time.time() - t0\n"
        "    await bulk_task\n"
        "    print('RESULT', kb_elapsed < 1.5, kb_elapsed)\n"
        "asyncio.run(main())\n"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe, str(bulk_dir)],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=30,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line.startswith("RESULT True"), (
        f"kb_stats did not run concurrently with add_documents_bulk: {line!r}. "
        "This is the exact hang reported in issue #13."
    )


# ---------------------------------------------------------------------------
# A stand-in blocker that only sleeps proves nothing.
#
# The two tests below replace the one that monkeypatched a handler to
# time.sleep(3) and asserted an unrelated kb_stats still answered. time.sleep
# in a worker thread holds no lock, touches no shared structure and releases
# the GIL, so it is precisely the one kind of blocker that CANNOT reproduce
# the production failure - a tool wedged *inside* the shared-cache path, with
# KnowledgeBase._lock in its hand. Measured on this tree 2026-09-21:
#
#   stand-in = time.sleep(3)          -> a concurrent search_docs answered in
#                                        ~0.2s. The test was green and pinned
#                                        nothing.
#   stand-in = holds kb._lock forever -> the same search_docs did not answer
#                                        in 20s with the default 600s bound.
#
# kb._lock (kb/core.py) is what the search path itself takes: search_docs ->
# _preprocess_text -> `with self._lock` while the stemmer/stopwords are built
# on first use (kb/search/_retrieval.py:281). It also guards the ingest side
# (4 sites in kb/ingest/_documents.py). Holding it is therefore not a
# contrived wedge; it is the state a real slow ingest holds.
#
# Note that the sibling issue-#13 test above still uses a plain sleep. That is
# deliberate: it reproduces one specific reported incident (a bulk ingest that
# was CPU-bound on the event loop thread) and its assertion is about the loop,
# not about shared state. The generic claim - "a slow tool does not take the
# session down" - is pinned here instead.
# ---------------------------------------------------------------------------

# Short enough to keep the test quick, far above a healthy kb_stats/search
# (~100-300ms). The shipped default is 600s.
LOCKED_WEDGE_BOUND_S = "3"

_SHARED_LOCK_WEDGE_PROBE = """
import asyncio, os, sys, threading, time
import server

BOUND = float(os.environ["TDZ_TOOL_TIMEOUT_S"])
CEILING = BOUND * 6 + 10
held = threading.Event()

def hold_the_kb_lock(kb, name, arguments):
    # NOT a sleep. This is what a real slow ingest looks like from the rest of
    # the process: the shared KnowledgeBase lock is taken and not given back.
    with kb._lock:
        held.set()
        threading.Event().wait()
    raise AssertionError("unreachable")

server.HANDLERS["scrape_url"] = hold_the_kb_lock

async def main():
    server._tool_call_lock = None          # stdio transport: no serialisation
    kb = server.get_kb()
    t0 = time.time()
    wedged = asyncio.create_task(server.call_tool(
        "scrape_url", {"url": "https://example.invalid/", "max_pages": 1}))
    for _ in range(200):                   # bounded wait for the lock to be taken
        if held.is_set():
            break
        await asyncio.sleep(0.05)

    # (a) a tool that needs nothing the wedge holds must still answer fast:
    #     that is the event loop being free, which is what the old sleeping
    #     stand-in tested and all it tested.
    s0 = time.time()
    try:
        stats = await asyncio.wait_for(server.call_tool("kb_stats", {}), CEILING)
        stats_ok = bool(stats) and not stats[0].text.startswith("Error")
    except (asyncio.TimeoutError, TimeoutError):
        stats_ok = False
    stats_s = time.time() - s0

    # (b) a tool that DOES need it - search_docs takes kb._lock in
    #     _preprocess_text - cannot answer until the dispatch bound fires.
    #     It must still come back, as an error, rather than going silent.
    q0 = time.time()
    try:
        later = await asyncio.wait_for(
            server.call_tool("search_docs", {"query": "raster interrupt"}), CEILING)
        search_returned = bool(later)
        search_errored = search_returned and later[0].text.startswith("Error")
    except (asyncio.TimeoutError, TimeoutError):
        search_returned = search_errored = False
    search_s = time.time() - q0

    print("RESULT", held.is_set(), kb.use_preprocessing, stats_ok,
          round(stats_s, 2), search_returned, search_errored,
          round(search_s, 2), round(time.time() - t0, 2), flush=True)
    sys.stdout.flush()
    # The abandoned workers are live non-daemon threads; a normal exit would
    # block in the executor join asyncio.run performs on the way out.
    os._exit(0)

asyncio.run(main())
"""


def test_a_tool_holding_the_shared_kb_lock_does_not_take_the_session_down(data_dir):
    """R5: a slow non-bulk tool must not freeze the session - measured with a
    blocker that actually holds what the search path needs.

    Two separate properties, and the sleeping stand-in could only ever show
    the first:

      (a) the event loop stays free, so a tool that touches none of the
          wedged state (kb_stats) answers in its usual ~200ms;
      (b) a tool that DOES contend for the wedged state (search_docs, via
          `with self._lock` in _preprocess_text) is genuinely blocked - and
          still comes back, bounded, as an error rather than silence.

    (b) is the half with teeth. If it ever passes *fast*, the stand-in has
    stopped holding anything and the test is back to proving nothing, so the
    "was actually blocked" check below is an assertion, not a comment.
    """
    env = _env(data_dir)
    env["TDZ_TOOL_TIMEOUT_S"] = LOCKED_WEDGE_BOUND_S
    bound = float(LOCKED_WEDGE_BOUND_S)
    out = subprocess.run(
        [PYTHON, "-c", _SHARED_LOCK_WEDGE_PROBE],
        capture_output=True, env=env, cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    lines = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")]
    assert lines, (
        "the shared-lock wedge probe produced no RESULT line; stderr:\n"
        + out.stderr.decode(errors="replace")[-2000:]
    )
    p = lines[-1].split()
    held, preproc = p[1] == "True", p[2] == "True"
    stats_ok, stats_s = p[3] == "True", float(p[4])
    search_returned, search_errored, search_s = p[5] == "True", p[6] == "True", float(p[7])
    raw = lines[-1]

    assert held, f"the stand-in never acquired kb._lock: {raw!r}"
    assert preproc, (
        f"query preprocessing is off on this install ({raw!r}), so search_docs "
        "never reaches `with self._lock` and this stand-in holds nothing the "
        "search path wants. Needs nltk (USE_QUERY_PREPROCESSING=1)."
    )

    # (a) the loop is free.
    assert stats_ok and stats_s < 1.5, (
        f"kb_stats queued behind a wedged scrape_url: {raw!r} - the dispatch is "
        "back on the event loop thread"
    )

    # (b) the stand-in is NOT benign. A sleeping stand-in answers here in
    # ~0.2s; this assertion is what rejects it.
    assert search_s >= bound * 0.8, (
        f"search_docs answered in {search_s:.2f}s while the stand-in supposedly "
        f"held kb._lock ({raw!r}). Either the search path stopped taking that "
        "lock or the stand-in stopped holding it - either way this test is "
        "benign again and pins nothing. Do not 'fix' it by relaxing this bound."
    )
    # ... and being blocked must still end in an answer, not silence.
    assert search_returned and search_errored, (
        f"search_docs blocked on kb._lock and never came back: {raw!r}. One "
        "wedged tool has silenced a later one - the dispatch bound "
        "(TDZ_TOOL_TIMEOUT_S) is not firing."
    )


# ---------------------------------------------------------------------------
# "One client per process" is not "no concurrency".
#
# The stdio dispatch takes no lock at all (_tool_call_lock is set only by the
# HTTP transport, server.py:655), while the entity-extraction worker is started
# in the KnowledgeBase constructor (kb/core.py, daemon thread
# "EntityExtractionWorker" running _extraction_worker_loop) and is therefore
# already running before the first tool call arrives. Two threads, one set of
# caches, one self.documents, from boot - and nothing in this suite exercised
# that pairing.
#
# The worker's real job body calls an LLM (extract_entities raises
# "LLM not configured" immediately without one), so a stock test environment
# drains any queue in milliseconds and would overlap nothing. The probe
# therefore substitutes a job body that does the shared-state work the real one
# does - _get_chunks_db, iterating self.documents, writing self._entity_cache
# (kb/entities/_extraction.py:507) - for the duration of a job, and runs it
# through the REAL worker thread and the REAL queue. The overlap is then
# measured, not assumed: a run where no search coincided with a busy worker
# fails instead of quietly passing.
# ---------------------------------------------------------------------------

_WORKER_CONCURRENCY_PROBE = """
import asyncio, os, sys, time
import server
from pathlib import Path

CORPUS = sys.argv[1]
JOB_WORK_S = 0.4
SEARCHES = 12
PER_CALL_CEILING = 20.0

def busy_extract(self, doc_id, confidence_threshold=0.6, force_regenerate=False):
    end = time.time() + JOB_WORK_S
    n = 0
    while time.time() < end:
        chunks = self._get_chunks_db(doc_id)
        for _did, meta in list(self.documents.items()):
            n += len(meta.tags)
        if self._entity_cache is not None:
            self._entity_cache[f"{doc_id}:{confidence_threshold}"] = {
                'entity_count': len(chunks), 'entities': []}
        n += len(chunks)
    return {'entity_count': n, 'entities': []}

server.KnowledgeBase.extract_entities = busy_extract

async def main():
    kb = server.get_kb()
    docs = [kb.add_document(str(p)).doc_id for p in sorted(Path(CORPUS).glob("*.txt"))]
    # Distinct thresholds: the entity cache is keyed "doc_id:threshold", so
    # equal ones would be served from cache and the worker would idle.
    for r in range(5):
        for d in docs:
            kb.queue_entity_extraction(d, confidence_threshold=0.30 + r * 0.01,
                                       skip_if_exists=False)

    worst, overlapped, bad = 0.0, 0, 0
    for _i in range(SEARCHES):
        busy = kb._extraction_queue.unfinished_tasks > 0
        t0 = time.time()
        try:
            out = await asyncio.wait_for(
                server.call_tool("search_docs", {"query": "VIC-II raster interrupt"}),
                PER_CALL_CEILING)
            worst = max(worst, time.time() - t0)
            if busy:
                overlapped += 1
            if not out or out[0].text.startswith("Error"):
                bad += 1
        except (asyncio.TimeoutError, TimeoutError):
            bad += 1
            break
        await asyncio.sleep(0.05)

    print("RESULT", len(docs), overlapped, bad, round(worst, 2),
          kb._extraction_worker.is_alive(), flush=True)
    sys.stdout.flush()
    os._exit(0)

asyncio.run(main())
"""


def test_search_keeps_answering_while_the_extraction_worker_runs(tmp_path):
    """Tool calls overlap the background entity-extraction worker from boot.

    Not a hypothetical: the worker thread starts in the KnowledgeBase
    constructor, so every stdio session has a second thread walking
    self.documents, the chunk cache and the entity TTLCache while tool calls
    run with no lock between them. Repeated search_docs calls must keep
    answering, bounded, while that worker is actively processing jobs.

    Fails, never hangs: every call has its own asyncio.wait_for ceiling and the
    probe prints a RESULT line.
    """
    corpus = tmp_path / "worker_corpus"
    corpus.mkdir()
    words = ("VIC-II raster interrupt sprite SID register Commodore CIA timer "
             "bitmap charset border sync NMI IRQ kernal basic chip MOS").split()
    rng = random.Random(6510)
    for i in range(6):
        (corpus / f"d{i}.txt").write_text(
            " ".join(rng.choice(words) for _ in range(25000)), encoding="utf-8")

    # Its own data dir, not the module-scoped one: this test ingests a corpus,
    # and tests later in the file (health_check's bm25 lazy-build case) assert
    # on a fresh, empty KB.
    own_data = tmp_path / "worker_data"
    own_data.mkdir()
    env = _env(own_data)
    env["ALLOWED_DOCS_DIRS"] = str(corpus)
    # queue_entity_extraction now declines up front with no LLM configured
    # (kb/entities/_extraction.py), so without a stub nothing would ever be
    # queued for the worker to be busy with. The probe patches
    # extract_entities itself to busy_extract, so this credential is never
    # actually used to call out - it only has to make get_llm_client() truthy
    # so queue_entity_extraction lets the job through.
    env["LLM_PROVIDER"] = "anthropic"
    env["ANTHROPIC_API_KEY"] = "test-key"
    out = subprocess.run(
        [PYTHON, "-c", _WORKER_CONCURRENCY_PROBE, str(corpus)],
        capture_output=True, env=env, cwd=str(REPO), timeout=300,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    lines = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")]
    assert lines, (
        "the worker-concurrency probe produced no RESULT line; stderr:\n"
        + out.stderr.decode(errors="replace")[-2000:]
    )
    p = lines[-1].split()
    n_docs, overlapped, bad, worst, alive = (
        int(p[1]), int(p[2]), int(p[3]), float(p[4]), p[5] == "True")
    raw = lines[-1]

    assert n_docs == 6, f"corpus did not ingest: {raw!r}"
    assert alive, f"the extraction worker thread died during the run: {raw!r}"
    assert overlapped >= 8, (
        f"only {overlapped}/12 searches were issued while the extraction queue "
        f"had unfinished work ({raw!r}); the worker idled through the run, so "
        "this test exercised no concurrency at all"
    )
    assert bad == 0, (
        f"{bad} of 12 search_docs calls timed out or errored while the entity "
        f"extraction worker was running: {raw!r}"
    )
    assert worst < 10.0, f"a search took {worst:.2f}s behind the worker: {raw!r}"


def test_bulk_add_default_pattern_actually_matches_files(data_dir, tmp_path):
    """The default bulk-ingest glob must not be a silent no-op.

    pathlib.Path.glob has no brace alternation, so the shipped default
    "**/*.{pdf,txt,md,...}" matched literally zero files while
    add_documents_bulk cheerfully reported "0 added, 0 skipped, 0 failed".
    Verified against the repo before the fix: the brace pattern matched 0
    files where plain "**/*.md" matched 702.
    """
    probe = (
        "import server, json, inspect\n"
        "pat = inspect.signature(server.KnowledgeBase.add_documents_bulk)"
        ".parameters['pattern'].default\n"
        "expanded = server._expand_brace_pattern(pat)\n"
        "print('EXPANDED', json.dumps(expanded))\n"
    )
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("EXPANDED")][-1]
    expanded = json.loads(line[len("EXPANDED "):])
    assert len(expanded) > 1, f"default pattern did not expand: {expanded}"
    assert not any("{" in p for p in expanded), f"brace left unexpanded: {expanded}"
    assert "**/*.pdf" in expanded and "**/*.txt" in expanded, expanded

    # And prove the expansion actually finds real files on disk.
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "sub" / "b.txt").write_text("VIC-II sprite notes")
    (tmp_path / "sub" / "c.md").write_text("# SID register map")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")

    probe2 = (
        "import server, sys, json\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "pat = '**/*.{pdf,xlsx,xls,html,htm,txt,md,asm,bas,inc,s,sid,psid,rsid,zip,docx,pptx,epub,csv,json,xml}'\n"
        "found = set()\n"
        "for p in server._expand_brace_pattern(pat):\n"
        "    found |= {f.name for f in root.glob(p)}\n"
        "print('FOUND', json.dumps(sorted(found)))\n"
    )
    out2 = subprocess.run(
        [PYTHON, "-c", probe2, str(tmp_path)],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=180,
    )
    assert out2.returncode == 0, out2.stderr.decode(errors="replace")[-2000:]
    line2 = [ln for ln in out2.stdout.decode().splitlines() if ln.startswith("FOUND")][-1]
    found = json.loads(line2[len("FOUND "):])
    assert found == ["a.pdf", "b.txt", "c.md"], (
        f"expanded default pattern found {found}; expected the pdf/txt/md files "
        "and not the .bin"
    )


def test_health_check_bm25_not_built_is_not_reported_as_an_issue(data_dir):
    """bm25_index_built:False before any search is expected, not a failure.

    Regression for the confusing report in issue #12: the text output showed
    a plain ✗ next to bm25_index_built as if it were a failed check. The
    lazy-build case must read as informational instead, and must not appear
    in the itemised Issues list (unlike a genuine problem such as semantic
    search being enabled with no embeddings built - which a fresh, empty KB
    correctly *does* still flag, so this test doesn't assert a clean report,
    only that bm25 specifically isn't misreported as one).
    """
    with MCPSession(data_dir, client_name="health-check-probe") as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None, "handshake failed"
        s._send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "health_check", "arguments": {}},
        })
        line = s._readline(HANDSHAKE_BUDGET_S)

    assert line is not None, "health_check timed out"
    result = json.loads(line)
    assert "result" in result, f"health_check returned an error: {result}"
    text = result["result"]["content"][0]["text"]
    assert "bm25_index_built: False" in text, "test setup: expected a fresh KB with no BM25 index yet"
    assert "✗ bm25_index_built" not in text, f"still reports the lazy build as a failure:\n{text}"
    assert "ⓘ bm25_index_built" in text
    assert "bm25" not in text.split("Issues")[-1].lower(), "bm25 lazy-build leaked into the Issues list"


# ---------------------------------------------------------------------------
# In-process thread safety of the SQLite connection (R1)
#
# The rest of this file drives real server subprocesses, because the startup
# failures it guards are cross-process. The two tests below are the in-process
# analog: one KnowledgeBase, several threads, one database.
# ---------------------------------------------------------------------------

@pytest.fixture
def inproc_kb(tmp_path):
    """A KnowledgeBase on a throwaway data dir, safe to write from threads."""
    from server import KnowledgeBase

    docs = tmp_path / "docs"
    docs.mkdir()
    saved = {k: os.environ.get(k) for k in ("ALLOWED_DOCS_DIRS", "AUTO_EXTRACT_ENTITIES")}
    os.environ["ALLOWED_DOCS_DIRS"] = str(docs)
    # Offline and deterministic: no background LLM extraction jobs.
    os.environ["AUTO_EXTRACT_ENTITIES"] = "0"
    kb = KnowledgeBase(str(tmp_path / "data"))
    try:
        yield kb, docs
    finally:
        kb.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_a_commit_on_one_thread_cannot_commit_another_threads_transaction(inproc_kb):
    """R1: sqlite3 transactions are per-connection, so threads need own connections.

    KnowledgeBase used to open one connection in __init__ and share it across
    the event-loop thread, the extraction worker and every asyncio.to_thread /
    ThreadPoolExecutor worker. Because a transaction belongs to the connection,
    not the caller, a bare `self.db_conn.commit()` on any thread - e.g. the one
    at the end of _log_mcp_call, which runs for *every* MCP tool call - landed
    whatever transaction another thread had only half-built. The victim's later
    rollback() then rolled back nothing and left partial data committed.

    Reproduced deliberately without lock contention: the second thread only
    commits, it does not write, so nothing here can be explained away as
    SQLITE_BUSY timing.
    """
    kb, _ = inproc_kb
    inserted = threading.Event()
    outsider_committed = threading.Event()
    conn_ids = {}
    errors = []

    def victim():
        try:
            conn_ids["victim"] = id(kb.db_conn)
            kb.db_conn.execute(
                "INSERT INTO mcp_call_log (tool_name, called_at, duration_ms, success) "
                "VALUES ('SENTINEL_UNCOMMITTED', ?, 0, 1)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"),),
            )
            inserted.set()
            assert outsider_committed.wait(30), "outsider thread never committed"
            kb.db_conn.rollback()
        except Exception as e:  # pragma: no cover - surfaced via `errors`
            errors.append(repr(e))
            inserted.set()

    t = threading.Thread(target=victim, daemon=True)
    t.start()
    assert inserted.wait(30), "victim thread never reached its INSERT"

    conn_ids["main"] = id(kb.db_conn)
    kb.db_conn.commit()  # exactly what _log_mcp_call does on the event-loop thread
    outsider_committed.set()
    t.join(timeout=30)
    assert not t.is_alive(), "victim thread hung"
    assert not errors, errors

    assert conn_ids["main"] != conn_ids["victim"], (
        "both threads got the same sqlite3.Connection object - db_conn is not "
        "thread-local, so every thread still shares one transaction scope"
    )
    row = kb.db_conn.execute(
        "SELECT COUNT(*) FROM mcp_call_log WHERE tool_name = 'SENTINEL_UNCOMMITTED'"
    ).fetchone()
    assert row[0] == 0, (
        "a rolled-back INSERT survived: another thread's commit() committed this "
        "thread's open transaction"
    )


def test_bulk_ingest_thread_and_call_logging_thread_do_not_corrupt_each_other(inproc_kb):
    """R1 under realistic load: ingest on a worker thread, call logging on the caller's.

    This is the shape R5 made routine - the whole MCP dispatch now runs on an
    asyncio.to_thread worker while _log_mcp_call and get_stats keep running on
    the event-loop thread. Asserts the ingest's writes all land intact and that
    _log_mcp_call never has to swallow a database error (it deliberately never
    raises, so its failures are only visible in the log).
    """
    kb, docs = inproc_kb
    n_docs = 12
    for i in range(n_docs):
        (docs / f"card_{i:02d}.md").write_text(
            f"# Doc {i}\n\n" + f"VIC-II sprite collision register notes for doc {i}. " * 60,
            encoding="utf-8",
        )

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.WARNING)
            self.records = []

        def emit(self, record):
            self.records.append(record.getMessage())

    capture = _Capture()
    logging.getLogger("server").addHandler(capture)

    bulk_result = {}
    bulk_error = []
    done = threading.Event()

    def ingest():
        try:
            bulk_result.update(
                kb.add_documents_bulk(str(docs), "**/*.md", ["concurrency-test"], True, True)
            )
        except Exception as e:  # pragma: no cover - surfaced via `bulk_error`
            bulk_error.append(repr(e))
        finally:
            done.set()

    t = threading.Thread(target=ingest, daemon=True)
    hammer_errors = []
    hammer_rounds = 0
    try:
        t.start()
        while not done.wait(0.01):
            try:
                kb._log_mcp_call("kb_stats", 1.0, True, None, {})
                kb.get_stats(use_cache=False)
                hammer_rounds += 1
            except Exception as e:
                hammer_errors.append(repr(e))
        t.join(timeout=300)
    finally:
        logging.getLogger("server").removeHandler(capture)

    assert not t.is_alive(), "bulk ingest thread hung"
    assert not bulk_error, bulk_error
    assert not hammer_errors, f"concurrent _log_mcp_call/get_stats raised: {hammer_errors}"
    assert hammer_rounds > 0, "test setup: ingest finished before any concurrent call ran"

    db_errors = [
        m for m in capture.records
        if "Failed to log MCP call" in m
        or any(marker in m.lower() for marker in ("cannot commit", "no transaction is active"))
    ]
    assert not db_errors, f"database errors were swallowed during concurrent access: {db_errors}"

    assert len(bulk_result.get("added", [])) == n_docs, bulk_result
    assert not bulk_result.get("failed"), bulk_result["failed"]

    cur = kb.db_conn.cursor()
    assert cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == n_docs
    assert len(kb.documents) == n_docs
    # Every document's chunks must be fully present: the old shared connection
    # let an unrelated commit land a half-written chunk set.
    mismatched = cur.execute(
        """SELECT d.doc_id, d.total_chunks, COUNT(c.chunk_id)
             FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id
           HAVING d.total_chunks != COUNT(c.chunk_id)"""
    ).fetchall()
    assert not mismatched, f"documents with a partial chunk set: {mismatched}"
    orphans = cur.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
    ).fetchone()[0]
    assert orphans == 0, f"{orphans} orphaned chunks"


# --- R9: extraction jobs orphaned by a killed process ---
#
# extraction_jobs rows outlive the in-memory queue that drives them. A row left
# at 'queued' when its process died was never retried by anything, and a row
# left at 'running' stayed 'running' forever - both showed up in
# get_extraction_status as pending/in-flight work that would never happen.


def _extraction_job_row(kb, job_id):
    return kb.db_conn.execute(
        "SELECT status, error_message FROM extraction_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()


def test_orphaned_queued_job_is_requeued_on_startup(inproc_kb, monkeypatch):
    """A 'queued' row from a dead process must be picked up by the next one."""
    from server import KnowledgeBase

    kb, docs = inproc_kb
    (docs / "orphan.md").write_text("VIC-II sprite notes.", encoding="utf-8")
    doc = kb.add_document(str(docs / "orphan.md"))

    # Simulate the dead process: a committed 'queued' row that no live
    # in-memory queue references.
    kb.db_conn.execute(
        "INSERT INTO extraction_jobs (doc_id, status, confidence_threshold, queued_at) "
        "VALUES (?, 'queued', 0.6, ?)",
        (doc.doc_id, time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    kb.db_conn.commit()
    job_id = kb.db_conn.execute("SELECT MAX(job_id) FROM extraction_jobs").fetchone()[0]

    # Keep the successor's worker from actually consuming the job, so the
    # assertion is specifically "recovery enqueued it", not "it ran".
    monkeypatch.setattr(KnowledgeBase, "_extraction_worker_loop", lambda self: None)
    successor = KnowledgeBase(str(kb.data_dir))
    try:
        queued = []
        while not successor._extraction_queue.empty():
            queued.append(successor._extraction_queue.get_nowait())
    finally:
        successor.close()

    assert any(j["job_id"] == job_id for j in queued), (
        f"orphaned 'queued' job {job_id} was not re-queued on startup; got {queued}"
    )


def test_stale_running_job_is_failed_not_left_running_forever(inproc_kb, monkeypatch):
    """A 'running' row older than the stale window is a dead process's leftover."""
    from server import KnowledgeBase

    kb, docs = inproc_kb
    (docs / "stale.md").write_text("SID filter notes.", encoding="utf-8")
    doc = kb.add_document(str(docs / "stale.md"))

    long_ago = "2020-01-01T00:00:00+00:00"
    kb.db_conn.execute(
        "INSERT INTO extraction_jobs (doc_id, status, confidence_threshold, queued_at, started_at) "
        "VALUES (?, 'running', 0.6, ?, ?)",
        (doc.doc_id, long_ago, long_ago),
    )
    kb.db_conn.commit()
    job_id = kb.db_conn.execute("SELECT MAX(job_id) FROM extraction_jobs").fetchone()[0]

    monkeypatch.setattr(KnowledgeBase, "_extraction_worker_loop", lambda self: None)
    successor = KnowledgeBase(str(kb.data_dir))
    try:
        status, error_message = _extraction_job_row(successor, job_id)
    finally:
        successor.close()

    assert status == "failed", f"stale 'running' job was left at {status!r}"
    assert "Interrupted" in (error_message or ""), error_message


def test_a_recent_running_job_is_left_alone(inproc_kb, monkeypatch):
    """Don't reap a 'running' row that a live sibling process may still own."""
    from server import KnowledgeBase

    kb, docs = inproc_kb
    (docs / "fresh.md").write_text("CIA timer notes.", encoding="utf-8")
    doc = kb.add_document(str(docs / "fresh.md"))

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    kb.db_conn.execute(
        "INSERT INTO extraction_jobs (doc_id, status, confidence_threshold, queued_at, started_at) "
        "VALUES (?, 'running', 0.6, ?, ?)",
        (doc.doc_id, now, now),
    )
    kb.db_conn.commit()
    job_id = kb.db_conn.execute("SELECT MAX(job_id) FROM extraction_jobs").fetchone()[0]

    monkeypatch.setattr(KnowledgeBase, "_extraction_worker_loop", lambda self: None)
    successor = KnowledgeBase(str(kb.data_dir))
    try:
        status, _ = _extraction_job_row(successor, job_id)
    finally:
        successor.close()

    assert status == "running", (
        f"a just-started job was reaped as stale (status={status!r}) - this would "
        "yank jobs out from under a concurrently running sibling process"
    )


def test_worker_death_is_reported_by_health_check(inproc_kb):
    """R21: a dead worker silently strands every queued job; health must say so."""
    kb, _ = inproc_kb

    # use_cache=False throughout: health_check memoises for 5 minutes, so a
    # cached "healthy" would mask the liveness change this test is about.
    healthy = kb.health_check(quick_check=True, use_cache=False)
    assert healthy['features']['extraction_worker_alive'] is True
    assert 'extraction_queue_depth' in healthy['features']

    # Stop the worker the same way close() does, without tearing down the DB.
    kb._extraction_shutdown.set()
    kb._extraction_worker.join(timeout=15)
    assert not kb._extraction_worker.is_alive(), "worker did not stop"

    dead = kb.health_check(quick_check=True, use_cache=False)
    assert dead['features']['extraction_worker_alive'] is False
    assert dead['status'] == 'warning'
    assert any('extraction worker' in i.lower() for i in dead['issues']), dead['issues']


# ---------------------------------------------------------------------------
# HTTP transport (issue #16). stdio ties the client and the ~8 GB TDZ_DATA_DIR
# to one machine; --transport http lets one hosted instance serve both. These
# guard the two things that make that safe rather than merely working: the
# handshake really completes over HTTP, and the server refuses to expose an
# unauthenticated knowledge base to a network.
# ---------------------------------------------------------------------------

def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_port(port, proc, timeout=180):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


class HTTPServerProc:
    """Spawn server.py on the streamable-HTTP transport and wait for the port."""

    def __init__(self, data_dir, api_keys=None, host="127.0.0.1"):
        self.data_dir = data_dir
        self.api_keys = api_keys
        self.host = host
        self.port = _free_port()
        self.proc = None

    def __enter__(self):
        env = _env(self.data_dir)
        env["TDZ_MCP_TRANSPORT"] = "http"
        env["TDZ_MCP_HOST"] = self.host
        env["TDZ_MCP_PORT"] = str(self.port)
        env["TDZ_API_KEYS"] = self.api_keys or ""
        self.proc = subprocess.Popen(
            [PYTHON, str(SERVER)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(REPO),
        )
        # Drain the pipe from a thread: uvicorn and the KB both log here, and
        # a full pipe buffer would block the server before it ever binds.
        # It also means a failure can show what the server actually said,
        # which a dead pipe cannot.
        self._output = []
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        if not _wait_for_port(self.port, self.proc):
            raise AssertionError(
                f"server never listened on {self.port}: {self.output[-2000:]}"
            )
        return self

    def _drain(self):
        for line in iter(self.proc.stdout.readline, b""):
            self._output.append(line)

    @property
    def output(self):
        return b"".join(self._output).decode("utf-8", "replace")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _http_initialize(url, headers=None):
    """Complete a real MCP handshake over HTTP and return (serverInfo, tool count)."""
    import asyncio

    import httpx

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    # create_mcp_http_client is private, and that is a deliberate trade rather
    # than an oversight. streamable_http_client is not a rename of
    # streamablehttp_client: it dropped headers/timeout in favour of a
    # caller-built client, and this helper is what applies the defaults the
    # transport expects - notably follow_redirects, which matters here because
    # POST /mcp answers 307 to /mcp/. Hand-rolling an httpx.AsyncClient would
    # risk silently missing that. If an mcp bump moves this helper, the import
    # breaks loudly, which is the failure mode to prefer.
    from mcp.shared._httpx_utils import create_mcp_http_client

    async def go():
        # The SDK closes only a client it created itself ("if not
        # client_provided"), so a caller-supplied one is owned here.
        async with create_mcp_http_client(
            headers=headers, timeout=httpx.Timeout(30)
        ) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (r, w, _):
                async with ClientSession(r, w) as session:
                    init = await session.initialize()
                    tools = await session.list_tools()
                    return init.serverInfo.name, len(tools.tools)

    return asyncio.run(go())


def test_http_transport_completes_initialize(data_dir):
    """A real MCP client must reach a usable session over HTTP, not just a socket."""
    with HTTPServerProc(data_dir) as srv:
        name, tool_count = _http_initialize(srv.url)
    assert name == "tdz-c64-knowledge"
    # The whole tool surface must survive the transport swap - server.run() is
    # transport-agnostic, so a drop here means the registration path broke.
    assert tool_count > 50, tool_count


def test_http_transport_bare_mcp_path_does_not_redirect(data_dir):
    """The startup log advertises the bare '/mcp' path. A client that doesn't
    follow redirects, or that downgrades POST to GET across one, must still
    get a working response there - not a 307 to '/mcp/'. Both the bare and
    the trailing-slash form must answer 200 with no redirect hop."""
    import http.client
    from urllib.parse import urlsplit

    body = (
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{'
        b'"protocolVersion":"2025-06-18","capabilities":{},'
        b'"clientInfo":{"name":"redirect-probe","version":"0"}}}'
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    with HTTPServerProc(data_dir) as srv:
        parts = urlsplit(srv.url)
        assert parts.path == "/mcp", srv.url  # matches what the startup log advertises
        for path in (parts.path, parts.path + "/"):
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=30)
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                resp.read()
                assert resp.status == 200, (
                    path, resp.status, resp.getheader("Location"), srv.output[-2000:]
                )
            finally:
                conn.close()


def test_http_transport_rejects_a_bad_api_key(data_dir):
    """TDZ_API_KEYS must gate the transport, not just the REST API."""
    import urllib.error
    import urllib.request

    with HTTPServerProc(data_dir, api_keys="right-key") as srv:
        req = urllib.request.Request(
            srv.url, method="POST", data=b'{"jsonrpc":"2.0","id":1,"method":"initialize"}',
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "X-API-Key": "wrong-key",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=30)
        assert excinfo.value.code == 401

        # ...and the correct key still reaches a working session.
        name, _ = _http_initialize(srv.url, headers={"X-API-Key": "right-key"})
        assert name == "tdz-c64-knowledge"


def test_http_transport_refuses_insecure_network_bind(data_dir):
    """Binding off loopback with no keys would serve the whole KB to the LAN."""
    env = _env(data_dir)
    env["TDZ_MCP_TRANSPORT"] = "http"
    env["TDZ_MCP_HOST"] = "0.0.0.0"
    env["TDZ_MCP_PORT"] = str(_free_port())
    env["TDZ_API_KEYS"] = ""
    env.pop("TDZ_MCP_ALLOW_INSECURE", None)

    proc = subprocess.run(
        [PYTHON, str(SERVER)], env=env, cwd=str(REPO),
        capture_output=True, timeout=180,
    )
    combined = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    assert proc.returncode != 0, combined[-1500:]
    assert "Refusing to bind" in combined, combined[-1500:]


def test_stdio_remains_the_default(data_dir):
    """No argument and no env var must still mean stdio - existing client
    configs launch `python server.py` with nothing else and must keep working."""
    import server as server_module

    args = server_module._resolve_transport_config([])
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# issue #18: one wedged tool call must not silence every LATER call
#
# The reported symptom was not "search is slow". It was that health_check,
# kb_stats and list_docs - each timed at 100-200ms in the same session moments
# earlier - all stopped answering once a search call wedged first. That is the
# difference between one broken tool and a dead server, and it is a separate
# defect from whatever wedged the search: the hang is a bug in one handler, the
# silence afterwards is a missing bound in the dispatch.
#
# Measured on this tree before the bound existed (server.py records the same
# numbers next to the fix):
#
#   stdio (no _tool_call_lock) : a later kb_stats answered in 0.21s, but the
#                                wedged call itself NEVER returned
#   HTTP  (_tool_call_lock set): the later kb_stats never returned either
#
# Both transports are exercised below, because _tool_call_lock is set only by
# the HTTP transport and the two paths therefore fail differently.
#
# The wedge is threading.Event().wait() rather than a sleep on purpose: a sleep
# ends by itself and would pass an implementation with no bound at all. It is
# also not the real faiss-under-a-pending-stdin-read wedge, deliberately - over
# a live stdio session, sending the unrelated probe request is itself what
# advances the stalled DLL chain (see the long-lived-session test above), so
# that setup would report "the later call returned" for the wrong reason. An
# in-process wedge has no such feedback path.
#
# Every wait in the probe has a ceiling and the probe prints a RESULT line, so
# a missing bound makes these tests FAIL rather than hang.
# ---------------------------------------------------------------------------

# Short enough to keep the tests quick, long enough that a healthy kb_stats
# (~100-200ms) is nowhere near it. The shipped default is 600s.
WEDGE_BOUND_S = "3"

_WEDGED_CALL_PROBE = """
import asyncio, os, sys, threading, time
import server

BOUND = float(os.environ["TDZ_TOOL_TIMEOUT_S"])
USE_LOCK = sys.argv[1] == "lock"
CEILING = BOUND * 6 + 10

def wedge(kb, name, arguments):
    threading.Event().wait()          # never returns, cannot be cancelled
    raise AssertionError("unreachable")

server.HANDLERS["search_docs"] = wedge

async def main():
    server._tool_call_lock = asyncio.Lock() if USE_LOCK else None
    t0 = time.time()
    wedged = asyncio.create_task(
        server.call_tool("search_docs", {"query": "raster interrupt"}))
    await asyncio.sleep(0.2)          # let it reach the wedge and take the lock

    try:
        later = await asyncio.wait_for(server.call_tool("kb_stats", {}), CEILING)
        later_ok = bool(later) and not later[0].text.startswith("Error")
    except (asyncio.TimeoutError, TimeoutError):
        later_ok = False
    later_s = time.time() - t0

    try:
        out = await asyncio.wait_for(asyncio.shield(wedged), CEILING)
        wedged_err = bool(out) and out[0].text.startswith("Error")
    except (asyncio.TimeoutError, TimeoutError):
        wedged_err = False
    wedged_s = time.time() - t0

    print("RESULT", later_ok, wedged_err, server._orphaned_tool_calls,
          round(later_s, 2), round(wedged_s, 2), flush=True)
    sys.stdout.flush()
    # The abandoned worker is a live non-daemon thread, so a normal exit would
    # block in the executor join asyncio.run performs on the way out.
    os._exit(0)

asyncio.run(main())
"""


def _run_wedge_probe(data_dir, mode):
    env = _env(data_dir)
    env["TDZ_TOOL_TIMEOUT_S"] = WEDGE_BOUND_S
    env["TDZ_TOOL_LOCK_WAIT_S"] = WEDGE_BOUND_S
    out = subprocess.run(
        [PYTHON, "-c", _WEDGED_CALL_PROBE, mode],
        capture_output=True, env=env, cwd=str(REPO), timeout=180,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    lines = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")]
    assert lines, (
        "the wedge probe produced no RESULT line; stderr:\n"
        + out.stderr.decode(errors="replace")[-2000:]
    )
    parts = lines[-1].split()
    return {
        "later_ok": parts[1] == "True",
        "wedged_errored": parts[2] == "True",
        "orphaned": int(parts[3]),
        "later_s": float(parts[4]),
        "wedged_s": float(parts[5]),
        "raw": lines[-1],
    }


def test_a_wedged_tool_call_returns_an_error_instead_of_silence(data_dir):
    """stdio path (_tool_call_lock is None): the wedged call must come back.

    Before the bound this half failed on both transports - the caller waited
    forever and the client eventually gave up with nothing to show for it.
    """
    r = _run_wedge_probe(data_dir, "nolock")
    bound = float(WEDGE_BOUND_S)
    assert r["wedged_errored"], (
        f"a wedged search_docs never returned anything to its caller: {r['raw']!r}. "
        "Without a per-call bound the caller waits on a thread that cannot be "
        "cancelled, and the session just goes quiet (issue #18)."
    )
    assert r["wedged_s"] < bound * 3, f"the bound did not hold: {r['raw']!r}"
    assert r["orphaned"] >= 1, (
        f"the abandoned worker was not recorded: {r['raw']!r} - _orphaned_tool_calls "
        "is how a process learns it is leaking wedged threads."
    )
    assert r["later_ok"], f"an unrelated kb_stats did not succeed: {r['raw']!r}"


def test_a_wedged_tool_call_does_not_silence_later_calls_over_http(data_dir):
    """HTTP path (_tool_call_lock set): THE asymmetry.

    _tool_call_lock is set only by the HTTP transport, so this is the path
    where one wedged call took every later call down with it - the lock was
    held for the whole dispatch and its holder never finished. stdio has no
    such lock and never showed this half of the failure, which is why both
    transports are tested rather than one standing in for the other.
    """
    r = _run_wedge_probe(data_dir, "lock")
    bound = float(WEDGE_BOUND_S)
    assert r["later_ok"], (
        f"an unrelated kb_stats never returned while another call was wedged: "
        f"{r['raw']!r}. It is queued behind _tool_call_lock, whose holder is a "
        "thread that cannot be cancelled - this is what made the server look dead."
    )
    assert r["later_s"] < bound * 3, (
        f"kb_stats returned, but only after {r['later_s']}s: {r['raw']!r}"
    )
    assert r["wedged_errored"], f"the wedged call itself never returned: {r['raw']!r}"


def test_long_running_tools_are_real_tools_and_exempt_from_the_bound(data_dir):
    """The exemption list is the part that can rot silently.

    A flat bound would abort work that is long BY DESIGN - OCR in
    add_document, a whole-site scrape_url, topic training, backup/restore - so
    those names are exempted. A typo in that set would put a long tool back
    under the 600s bound and kill a legitimate hour-long call, and nothing else
    in the suite would notice. Assert the set against the live schema list, and
    assert the two sides of the policy actually differ.
    """
    probe = chr(10).join([
        "import server",
        "from mcp_tools.schemas import TOOL_SCHEMAS",
        "names = {t.name for t in TOOL_SCHEMAS}",
        "unknown = sorted(server._LONG_RUNNING_TOOLS - names)",
        "print('RESULT', unknown,",
        "      server._tool_timeout_s('health_check'),",
        "      server._tool_timeout_s('add_document'),",
        "      server._tool_timeout_s('a_tool_added_tomorrow'))",
    ])
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=120,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT [] 600.0 None 600.0", (
        f"the per-call bound policy is not what server.py documents: {line!r}. "
        "Expected: no unknown names in _LONG_RUNNING_TOOLS; health_check bounded "
        "at the 600s default; add_document exempt (None); and an unrecognised "
        "tool name bounded by DEFAULT - the bound has to be opt-out, or the next "
        "tool somebody adds inherits the unbounded wait this guards against."
    )
# --- The faiss/sentence-transformers load carries the SAME unbounded wait ---
#
# test_ensure_nltk_returns_on_a_deadline_instead_of_waiting_for_the_import
# above bounds the nltk import. _ensure_embeddings_loaded had the identical
# hole: it imports sentence_transformers, and via _load_embeddings first
# touches faiss, and BOTH stall on a pending read of this process's stdin
# exactly the way the scipy/BLAS chain does.
#
# Re-measured 2026-09-21 in fresh child processes, each handed a pipe on stdin
# whose write end the parent KEEPS OPEN (closing it hands the child EOF and
# the stall vanishes - the reproduction is worthless without that):
#
#   no pending read on stdin  : faiss 0.12s, sentence_transformers 3.26s
#   an IDLE extra thread      : faiss 0.12s   (control - a thread is harmless)
#   pending os.read(0, 1)     : BOTH still unfinished at 25s
#   same, one byte at t+6s    : released at the poke (5.60s / 8.42s)
#
# These two cases stand in for that stall without needing a live stdio
# session: the one-time load is replaced by one that can never finish, which
# is what the stall looks like from the caller's side. They FAIL on their own
# deadline rather than hanging if the bound is ever removed.


def test_ensure_embeddings_returns_on_a_deadline_instead_of_waiting_for_the_load(data_dir):
    """The bound itself, and that a missed deadline is not a failure.

    A timeout must NOT latch use_semantic off the way a real load failure
    does (see the two issue #12/#14 cases above): the load is still running
    and the next call should get the real index.
    """
    probe = chr(10).join([
        "import server, threading, time",
        "import kb.search._retrieval as _r",
        # A load that can never complete - the shape the stdin stall has from
        # the caller side. Patched at module scope, which is where the real
        # worker lives (kb/search/__init__.py pins the mixin method set).
        "_r._embeddings_load_worker = lambda kb: threading.Event().wait()",
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR'])",
        "assert kb.use_semantic, 'test setup: semantic search should be enabled'",
        "start = time.time()",
        "raised = ''",
        "try:",
        "    kb._ensure_embeddings_loaded(timeout=2.0, raise_on_timeout=True)",
        "except TimeoutError:",
        "    raised = 'TimeoutError'",
        "bounded = time.time() - start < 10",
        # A second caller must not re-pay the full deadline.
        "start2 = time.time()",
        "kb._ensure_embeddings_loaded()",
        "fast_retry = time.time() - start2 < 1.0",
        "print('RESULT', raised, bounded, kb.use_semantic is True, fast_retry)",
    ])
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=_env(data_dir), cwd=str(REPO), timeout=120,
    )
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    line = [ln for ln in out.stdout.decode().splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT TimeoutError True True True", (
        f"_ensure_embeddings_loaded did not return on its deadline: {line!r}. "
        "Expected, in order: TimeoutError raised; it returned inside 10s; "
        "use_semantic still True (a 'not yet' is not a failure); and a second "
        "caller degraded immediately instead of re-paying the whole deadline. "
        "Without that bound every semantic_search/hybrid_search/answer_question "
        "call inherits an unbounded wait on the first one."
    )


def test_semantic_search_says_the_index_is_loading_rather_than_returning_nothing(data_dir):
    """What semantic_search DOES when the index is not ready yet.

    There is no degraded semantic answer, so it raises rather than inventing
    one. An empty list would be indistinguishable from "no matches" - the one
    answer that is definitely wrong - and handle_semantic_search renders only
    title/doc_id/similarity/snippet, so silently substituting FTS5 hits could
    not be labelled as such and would be read as semantic. The exception
    reaches the user verbatim as "Semantic search error: ...", and the same
    text is on stderr as a warning, which is how the bound is observable in
    the log the way _ensure_nltk's is.
    """
    probe = chr(10).join([
        "import server, threading, time",
        "import kb.search._retrieval as _r",
        "_r._embeddings_load_worker = lambda kb: threading.Event().wait()",
        "kb = server.KnowledgeBase(server.os.environ['TDZ_DATA_DIR'])",
        "assert kb.use_semantic, 'test setup: semantic search should be enabled'",
        "start = time.time()",
        "msg = ''",
        "results = 'NOT-RAISED'",
        "try:",
        "    results = kb.semantic_search('raster interrupt stable timing', 3)",
        "except TimeoutError as e:",
        "    msg = str(e)",
        "print('RESULT', repr(results), 'still loading' in msg,",
        "      'search_docs' in msg, time.time() - start < 10)",
    ])
    env = _env(data_dir)
    # Proves the env knob is what sets the deadline: without it this would
    # wait the full 20s default.
    env["TDZ_EMBEDDINGS_LOAD_TIMEOUT_S"] = "2"
    start = time.time()
    out = subprocess.run(
        [PYTHON, "-c", probe],
        capture_output=True, env=env, cwd=str(REPO), timeout=120,
    )
    elapsed = time.time() - start
    assert out.returncode == 0, out.stderr.decode(errors="replace")[-2000:]
    stdout = out.stdout.decode(errors="replace")
    stderr = out.stderr.decode(errors="replace")
    line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")][-1]
    assert line == "RESULT 'NOT-RAISED' True True True", (
        f"semantic_search did not bound its first call: {line!r} (whole run "
        f"{elapsed:.1f}s). Expected it to raise TimeoutError (so 'results' is "
        "still the sentinel), with a message that says the index is still "
        "loading and names search_docs as the thing to use meanwhile."
    )
    assert "Embeddings warm-up deadline missed" in stderr, (
        "the missed deadline was not logged, so an operator watching the "
        "server log cannot tell a stalled load from a slow one. stderr tail: "
        f"{stderr[-1500:]!r}"
    )


# --- notifications/progress -----------------------------------------------
#
# issue #17's reporter killed a working check_updates at 30 minutes because a
# tool call is one blocking request-to-response: the client saw nothing until
# it returned. These drive a REAL stdio session and read the notification
# frames off the wire, because "the server called send_progress_notification"
# is not evidence any client received anything.


def test_client_receives_a_progress_notification_before_the_tool_result(data_dir):
    """A client that sends a progressToken gets notifications/progress for
    that token, ON THE WIRE, ahead of the response frame."""
    with MCPSession(data_dir, client_name="progress") as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None and "result" in response, response
        frames = s.call_tool_frames("health_check", {}, progress_token="tok-17",
                                    req_id=41, timeout=90)
        stderr = s.stderr_text()

    methods = [f.get("method") or f"response(id={f.get('id')})" for f in frames]
    progress = [
        f for f in frames
        if f.get("method") == "notifications/progress"
        and f.get("params", {}).get("progressToken") == "tok-17"
    ]
    assert progress, (
        "no notifications/progress frame with our token reached the client "
        f"during the call. Frames read, in wire order: {methods}. "
        f"server stderr: {stderr[-1500:]!r}"
    )
    assert "result" in frames[-1] or "error" in frames[-1], frames[-1]
    assert frames.index(progress[0]) < len(frames) - 1, (
        "the progress frame did not precede the result, so it bought the "
        f"caller nothing: {methods}"
    )
    params = progress[0]["params"]
    assert "progress" in params, params
    assert "health_check" in (params.get("message") or ""), (
        f"the notification does not say which tool it is about: {params}"
    )


def test_no_progress_notifications_when_the_client_sends_no_token(data_dir):
    """Per spec, no progressToken means no notifications. A server that
    pushed them anyway would be talking to clients that never asked."""
    with MCPSession(data_dir, client_name="no-progress") as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None and "result" in response, response
        frames = s.call_tool_frames("health_check", {}, progress_token=None,
                                    req_id=42, timeout=90)

    assert len(frames) == 1 and ("result" in frames[0] or "error" in frames[0]), (
        "the server sent unsolicited frames to a client that asked for no "
        f"progress: {[f.get('method') for f in frames]}"
    )


def test_a_real_client_observes_the_heartbeat_repeat_over_stdio(tmp_path):
    """The previous test proves the ack; this proves the REPEAT the ack
    cannot: a client reading real frames off a real stdio session sees more
    than one heartbeat tick for the same call.

    The obstacle named in the gap this closes is real: no tool in the suite
    reliably outlives the 15s default interval, so an honest wire-level test
    either waits 15s+ or shortens the interval for this one session via
    TDZ_PROGRESS_INTERVAL_S (0.15s here) - the shipped default is untouched,
    only this child process's env is. That alone would prove nothing if the
    tool returned in milliseconds regardless of the interval, so this also
    needs a tool that genuinely outlives it. add_document is real,
    already-registered production code (no test-only handler, no server.py
    change) and this test makes ONE call to it outlive the interval by
    holding, from this process, the exact SQLite write lock add_document's
    own commit needs on the SAME knowledge_base.db file - a real resource,
    not a sleep. The child's dispatch stays on a worker thread while the
    heartbeat runs as a separate asyncio task on its event loop (see
    _progress_heartbeat / asyncio.to_thread in server.py), so the block is
    exactly what lets several ticks land on the wire before the call
    proceeds.

    The lock's release is bounded from THIS process on a background thread
    (RELEASE_AFTER_S below), not left to expire via SQLite's own busy
    timeout. Measured with the release left to the busy timeout instead: the
    call took 29.67s and the whole test 30.81s against a 30s frame-read
    timeout - a flake on any slower machine. That duration traced to
    util._retry_on_db_locked (NOT kb/core.py's per-connection
    PRAGMA busy_timeout alone): add_document's write path retries up to 5
    times on a "database is locked" OperationalError, each attempt
    exhausting the full TDZ_DB_BUSY_TIMEOUT_MS=4000 busy-timeout wait before
    raising, plus exponential backoff (0.5+1+2+4=7.5s) between attempts -
    5*4s + 7.5s = 27.5s, matching the ~29.67s measured. Releasing the lock
    from a timed background thread well inside that first 4s busy-timeout
    window means the connection's internal retry simply succeeds without
    ever raising "database is locked", so none of _retry_on_db_locked's
    outer retries fire at all - the stall is now bounded by RELEASE_AFTER_S,
    not by SQLite/retry internals. Measured after the fix: the call itself
    took ~5.0s (RELEASE_AFTER_S=1.0s of deliberate lock hold, plus
    add_document's own extraction/chunking/indexing work, which the lock no
    longer inflates with retry waits) against the 15s frame-read timeout
    below - comfortably inside it, versus the previous ~29.67s call against
    a 30s timeout that left only ~0.3s of headroom.
    """
    import sqlite3

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    docfile = data_dir / "heartbeat-doc.txt"
    docfile.write_text("Heartbeat-over-stdio regression document.", encoding="utf-8")

    extra_env = {
        "TDZ_PROGRESS_INTERVAL_S": "0.15",
        "ALLOWED_DOCS_DIRS": str(data_dir),
        "TDZ_DB_BUSY_TIMEOUT_MS": "4000",
    }

    with MCPSession(data_dir, client_name="heartbeat-wire", extra_env=extra_env) as s:
        response, _ = s.initialize(timeout=HANDSHAKE_BUDGET_S)
        assert response is not None and "result" in response, response

        db_path = Path(data_dir) / "knowledge_base.db"
        assert db_path.exists(), "server did not create its database at handshake"

        # Hold the write lock add_document's own commit needs, from THIS
        # process, on the SAME db file - the tool call stalls on that real
        # lock while the heartbeat (a separate asyncio task, not the blocked
        # worker thread) keeps ticking on the wire. RELEASE_AFTER_S bounds
        # the stall deliberately (see docstring for why this replaced
        # waiting out SQLite's busy timeout): it is long enough for several
        # 0.15s heartbeat ticks to land, short enough to keep the test fast
        # and immune to machine-speed flakes.
        RELEASE_AFTER_S = 1.0
        locker = sqlite3.connect(str(db_path), timeout=1, check_same_thread=False)
        locker.isolation_level = None
        locker.execute("BEGIN IMMEDIATE")

        def _release_lock():
            time.sleep(RELEASE_AFTER_S)
            try:
                locker.execute("ROLLBACK")
            except Exception:
                pass
            finally:
                locker.close()

        releaser = threading.Thread(target=_release_lock, daemon=True)
        releaser.start()
        try:
            start = time.time()
            frames = s.call_tool_frames(
                "add_document", {"filepath": str(docfile)},
                progress_token="tok-heartbeat", req_id=99, timeout=15,
            )
            elapsed = time.time() - start
        finally:
            releaser.join(timeout=5)

        stderr = s.stderr_text()

    methods = [f.get("method") or f"response(id={f.get('id')})" for f in frames]
    progress = [
        f for f in frames
        if f.get("method") == "notifications/progress"
        and f.get("params", {}).get("progressToken") == "tok-heartbeat"
    ]
    ticks = [
        f for f in progress
        if "still running" in (f["params"].get("message") or "")
    ]

    assert len(progress) >= 3, (
        "expected the ack plus at least two heartbeat ticks (the repeat, "
        f"not just one), got {len(progress)} over {elapsed:.2f}s. frames in "
        f"wire order: {methods}. server stderr: {stderr[-1500:]!r}"
    )
    assert len(ticks) >= 2, (
        "expected at least two real heartbeat ticks distinct from the ack, "
        f"got {len(ticks)}: {[p['params'] for p in progress]}"
    )
    progresses = [p["params"]["progress"] for p in progress]
    assert progresses == sorted(progresses) and len(set(progresses)) == len(progresses), (
        f"progress must strictly increase per the MCP spec: {progresses}"
    )
    assert "result" in frames[-1] or "error" in frames[-1], frames[-1]


def test_heartbeat_keeps_ticking_while_a_tool_runs():
    """The ack alone would still leave a 30-minute call looking dead after
    the first second. The heartbeat is what answers 'alive or wedged', so
    tick it directly - no tool in the suite reliably runs long enough to
    observe a second tick through a real session without wasting minutes."""
    import asyncio as _asyncio
    import server as srv

    sent = []

    class _FakeChannel:
        async def asend(self, progress, total=None, message=None):
            sent.append((progress, total, message))
            return True

    async def drive():
        original = srv._PROGRESS_INTERVAL_S
        srv._PROGRESS_INTERVAL_S = 0.05
        try:
            task = _asyncio.create_task(
                srv._progress_heartbeat(_FakeChannel(), "check_updates", time.time())
            )
            await _asyncio.sleep(0.35)
            task.cancel()
        finally:
            srv._PROGRESS_INTERVAL_S = original

    _asyncio.run(drive())

    assert len(sent) >= 2, f"heartbeat produced {len(sent)} ticks: {sent}"
    progresses = [p for p, _, _ in sent]
    assert progresses == sorted(progresses) and len(set(progresses)) == len(progresses), (
        f"progress must strictly increase per the MCP spec: {progresses}"
    )
    assert all("check_updates" in (m or "") for _, _, m in sent), sent
