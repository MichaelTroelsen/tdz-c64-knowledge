#!/usr/bin/env python3
"""
TDZ C64 Knowledge - shared low-level helpers

Self-contained, dependency-light utilities used by server.py: lazy module
loading, outbound-network politeness (User-Agent, robots.txt, backoff),
cross-process/file locking, and SQLite busy-retry. Split out of server.py
in R12 step 3.

This module must stay stdlib-only at module level - it is imported by every
server.py process on every MCP session startup, and the whole point of the
lazy-import machinery in server.py (see _LazyModule below) is to keep that
path under the client's 30s initialize-handshake budget. See CLAUDE.md
"MCP startup performance".
"""

import importlib
import importlib.util
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from version import __version__


class _LazyModule:
    """Proxy that imports the wrapped module on first attribute access."""

    def __init__(self, name: str):
        self._lazy_name = name
        self._lazy_mod = None

    def __getattr__(self, attr):
        if self._lazy_mod is None:
            self._lazy_mod = importlib.import_module(self._lazy_name)
        return getattr(self._lazy_mod, attr)


# How long a one-off network fetch of a lazily-loaded NLP resource (NLTK
# corpora, the sentence-transformers model) may block before giving up.
# These fetches happen on the first real search call, not at startup, so a
# filtered/unreachable network must fail fast here rather than hang the
# calling tool indefinitely - see _ensure_nltk and _ensure_embeddings_loaded.
NETWORK_FETCH_TIMEOUT_S = float(os.environ.get('NETWORK_FETCH_TIMEOUT_S', '15'))

# Identify ourselves on every outbound scrape. The sources this tool targets
# are small volunteer-run retro-computing wikis and archives; an unidentified
# multi-threaded crawler is exactly what gets a client rate-limited or
# IP-banned, and it denies operators any way to contact us or write a rule.
USER_AGENT = os.environ.get(
    'TDZ_USER_AGENT',
    f"tdz-c64-knowledge/{__version__} (+https://github.com/Thordanielz/tdz-c64-knowledge)"
)

# Honour robots.txt unless explicitly disabled (a self-hosted mirror, say).
RESPECT_ROBOTS = os.environ.get('TDZ_RESPECT_ROBOTS', '1') == '1'

_robots_cache: dict[str, Any] = {}
_robots_cache_lock = threading.Lock()


def http_headers(extra: Optional[dict] = None) -> dict:
    """Default headers for any outbound HTTP request this server makes."""
    headers = {'User-Agent': USER_AGENT}
    if extra:
        headers.update(extra)
    return headers


def robots_allows(url: str) -> bool:
    """Whether robots.txt permits USER_AGENT to fetch `url`.

    Fails open: an unreachable or malformed robots.txt must not block a
    scrape the user explicitly asked for. Results are cached per origin so a
    crawl of N pages costs one robots.txt fetch, not N.
    """
    if not RESPECT_ROBOTS:
        return True

    from urllib.parse import urlparse
    from urllib.robotparser import RobotFileParser

    try:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return True

    with _robots_cache_lock:
        parser = _robots_cache.get(origin)
        cached = origin in _robots_cache

    if not cached:
        parser = RobotFileParser()
        parser.set_url(f"{origin}/robots.txt")
        try:
            with _network_timeout(10):
                parser.read()
        except Exception:
            parser = None  # no usable robots.txt -> allow
        with _robots_cache_lock:
            _robots_cache[origin] = parser

    if parser is None:
        return True
    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def http_get_polite(url: str, timeout: float = 15, max_attempts: int = 3,
                    base_delay: float = 1.0, **kwargs):
    """requests.get with our User-Agent and backoff on 429/5xx.

    A bare single-shot GET treats a server's "slow down" (429) and its
    transient 5xx exactly like a hard 404 - it drops the page and moves on,
    hammering the next URL immediately. Backing off instead is both politer
    and the only way those pages actually get fetched.
    """
    import requests

    kwargs.setdefault('headers', http_headers())
    kwargs.setdefault('allow_redirects', True)

    last_exc = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                # Prefer the server's own Retry-After when it sends one.
                retry_after = response.headers.get('Retry-After')
                delay = base_delay * (2 ** attempt)
                if retry_after:
                    try:
                        delay = max(delay, min(float(retry_after), 60.0))
                    except ValueError:
                        pass
                time.sleep(delay)
                continue
            return response
        except requests.RequestException as e:
            last_exc = e
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    if last_exc:
        raise last_exc


# socket.setdefaulttimeout is process-wide, so two threads entering
# _network_timeout at once would interleave: the first to exit restores the
# other's temporary value instead of the real original, leaving the default
# permanently clamped. That is no longer hypothetical - the whole MCP dispatch
# runs on worker threads now - so entry is serialised. An RLock (not Lock)
# because the guarded regions can nest within one thread.
_network_timeout_lock = threading.RLock()


@contextmanager
def _network_timeout(seconds: float = NETWORK_FETCH_TIMEOUT_S):
    """Temporarily cap the process-wide default socket timeout.

    nltk.download() and huggingface_hub's model download path issue plain
    urllib/requests calls with no timeout of their own, so a blocked or
    silently-dropped connection blocks forever instead of raising. Bounding
    the default socket timeout here gives those calls a real deadline; on
    expiry they raise (socket.timeout / URLError / requests exceptions),
    which the existing except-and-degrade logic at each call site handles.

    Every guarded region is a one-time lazy initialisation (NLTK corpora, the
    sentence-transformers model, a robots.txt fetch), so serialising them
    costs nothing and additionally stops two threads racing to download the
    same resource.
    """
    with _network_timeout_lock:
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(seconds)
        try:
            yield
        finally:
            socket.setdefaulttimeout(previous)


# A held lock must keep looking fresh, or `stale_after` cannot tell a crashed
# holder from one that is legitimately still working: a full embeddings rebuild
# that runs past stale_after would otherwise have its lock seized mid-write.
# The holder therefore refreshes the lock file's mtime on this cadence
# (stale_after/4, so four refreshes may be missed before anyone calls it
# stale), clamped so a large stale_after does not mean a lock file that sits
# unrefreshed for minutes and a tiny one does not spin.
_LOCK_HEARTBEAT_MAX_S = 5.0
_LOCK_HEARTBEAT_MIN_S = 0.02


def _lock_identity(lock_path: Path):
    """Fingerprint of the lock file as it exists right now, or None if it is gone.

    Identity is (recorded pid, st_dev, st_ino, st_mtime_ns). The pid alone is
    not enough - one process can be both the robbed holder and the robber - and
    st_ino alone is not enough either, since an inode freed by unlink is often
    handed straight back to the next create on POSIX. Content is read before
    stat on purpose: if the file is replaced mid-read the two halves come from
    different files and match no one, which is the safe direction to fail (we
    decline to delete a file we are unsure about, rather than deleting someone
    else's).
    """
    try:
        with open(str(lock_path), 'rb') as fh:
            owner = fh.read(64)
        st = os.stat(str(lock_path))
    except OSError:
        return None
    return (owner, st.st_dev, st.st_ino, st.st_mtime_ns)


class _HeldLock:
    """The lock file this process currently owns, and the proof that it does."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._guard = threading.Lock()
        self.identity = _lock_identity(lock_path)

    def refresh(self) -> bool:
        """Restamp our lock file's mtime. False once it is no longer ours."""
        with self._guard:
            if self.identity is None:
                return False
            if _lock_identity(self.lock_path) != self.identity:
                self.identity = None  # somebody reclaimed it out from under us
                return False
            try:
                os.utime(str(self.lock_path), None)
            except OSError:
                return False  # keep identity: release may still clean up
            self.identity = _lock_identity(self.lock_path)
            return self.identity is not None

    def release(self):
        """Delete the lock file, but only while it is still demonstrably ours."""
        with self._guard:
            if self.identity is None:
                return
            if _lock_identity(self.lock_path) == self.identity:
                try:
                    os.unlink(str(self.lock_path))
                except OSError:
                    pass
            self.identity = None


def _lock_heartbeat(held: '_HeldLock', stop: threading.Event, interval: float):
    while not stop.wait(interval):
        if not held.refresh():
            return


@contextmanager
def _cross_process_lock(lock_path: Path, timeout: float = 60.0, stale_after: float = 120.0,
                         poll_interval: float = 0.1):
    """Mutual exclusion across separate OS processes sharing a data directory.

    Every Claude Code session runs its own server.py process (see
    test_mcp_startup.py), so `threading.Lock` (self._lock) only serialises
    threads within one of those processes - it does nothing to stop two
    concurrent sessions from both loading the shared embeddings.faiss /
    embeddings_map.json, both appending in memory, and one silently
    clobbering the other's write. This uses atomic exclusive file creation
    (open with O_CREAT|O_EXCL fails if the file already exists) as a real
    cross-process mutex. A lock file older than `stale_after` is assumed to
    belong to a crashed holder and is reclaimed rather than deadlocking every
    future session forever.

    Liveness, not just age: a background thread restamps the lock file's mtime
    every `stale_after`/4 for as long as the lock is held, so `stale_after`
    measures silence rather than elapsed work. A crashed holder stops
    restamping and is still reclaimed on schedule; a slow one - a real
    embeddings rebuild past the 120s default - is not.

    Release is conditional. It deletes the lock file only while that file is
    still demonstrably the one we created (see _lock_identity). Without that
    check, a holder whose lock was reclaimed anyway would, on finishing, delete
    the file its successor is holding and admit a third process into the
    critical section; instead its release becomes a no-op.

    The tradeoff this makes deliberately: a holder that is alive but wedged
    keeps its lock indefinitely, where before it lost it after `stale_after`.
    That cannot deadlock a waiter - waiters still give up after `timeout` and
    raise TimeoutError - and losing the lock mid-write was the worse outcome.
    """
    lock_path = Path(lock_path)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue  # released between our open() attempt and stat()
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_interval)

    held = _HeldLock(lock_path)
    interval = max(_LOCK_HEARTBEAT_MIN_S, min(stale_after / 4.0, _LOCK_HEARTBEAT_MAX_S))
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_lock_heartbeat, args=(held, stop, interval),
        name='cross-process-lock-heartbeat', daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stop.set()
        heartbeat.join(timeout=interval + 5.0)
        held.release()


def _atomic_write_bytes(path: Path, data: bytes):
    """Write a file such that concurrent readers never see a partial write.

    A crash or a concurrent reader mid-write of embeddings.faiss/
    embeddings_map.json previously risked a torn file - or worse, a FAISS
    index and its doc-id map from two different points in time, since they
    were written as two separate non-atomic files. Writing to a temp file
    first and using os.replace (atomic on the same filesystem, including
    Windows) means any given file is always either fully the old version or
    fully the new one.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.tmp-{os.getpid()}')
    tmp.write_bytes(data)
    os.replace(str(tmp), str(path))


_RETRYABLE_DB_ERRORS = ('locked', 'busy', 'no such table')


def _retry_on_db_locked(fn, *args, max_attempts: int = 5, base_delay: float = 0.5, **kwargs):
    """Retry a DB write a few times if it hits transient SQLite contention.

    `PRAGMA busy_timeout` already makes a connection retry internally while
    waiting for another connection's write lock, but that budget can still be
    exhausted when two agent processes both start a transaction at almost the
    same instant (observed directly: two fresh processes both calling
    add_document() the moment they started up). Retrying the whole write from
    the top - not just re-waiting on the same lock - is the standard pattern
    for this class of transient SQLITE_BUSY condition.

    Also retries "no such table" - observed directly as a residual startup
    race even after serialising schema creation/WAL-activation behind a
    cross-process lock (see _init_database): two brand-new connections
    enabling WAL mode at almost the same instant on a not-yet-existent file
    can leave one connection with a transiently stale view of the schema for
    its very next statement. A short retry resolves that once the view
    catches up; a genuinely missing table (a real bug, not a race) still
    fails after max_attempts rather than being silently hidden forever.

    Callers like _add_document_db catch the raw sqlite3.OperationalError
    themselves and re-raise a KnowledgeBaseError whose message includes the
    original text, so this checks the message rather than the exception type
    to still recognise a retryable condition underneath that wrapping.
    """
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if not any(marker in msg for marker in _RETRYABLE_DB_ERRORS):
                raise
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
