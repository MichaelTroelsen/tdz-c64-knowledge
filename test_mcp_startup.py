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
import os
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
FORBIDDEN_AT_STARTUP = ["sentence_transformers", "torch", "transformers", "nltk"]


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

    def __init__(self, data_dir, client_name="pytest"):
        self.data_dir = data_dir
        self.client_name = client_name
        self.proc = None
        self.stderr = b""

    def __enter__(self):
        self.proc = subprocess.Popen(
            [PYTHON, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_env(self.data_dir),
            cwd=str(REPO),
        )
        return self

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
        try:
            with MCPSession(data_dir, client_name=f"session-{i}") as s:
                response, elapsed = s.initialize(timeout=CONCURRENT_BUDGET_S)
                results[i] = (response is not None and "result" in (response or {}), elapsed)
        except Exception as e:  # pragma: no cover - diagnostic path
            results[i] = (False, f"exception: {e!r}")

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

    conn = sqlite3.connect(str(db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert str(mode).lower() == "wal", f"journal_mode is {mode!r}, expected 'wal'"


_FRESH_START_WORKER = """
import os, sys, time
import server

filename = sys.argv[1]
barrier_dir = sys.argv[2]
nworkers = int(sys.argv[3])

kb = server.kb  # import already constructed this module-level singleton

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
        "import server, socket, nltk.corpus, time\n"
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
        "server._nltk_ready = False\n"
        "start = time.time()\n"
        "result = server._ensure_nltk()\n"
        "elapsed = time.time() - start\n"
        "print('RESULT', elapsed < 10, result is not None, server._nltk_ready)\n"
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
        "server.SentenceTransformer = fake_st\n"
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
        "server.SentenceTransformer = fake_st\n"
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
        "pat = '**/*.{pdf,txt,md,html,htm,xlsx,xls}'\n"
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
