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
