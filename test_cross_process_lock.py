"""Direct tests for util._cross_process_lock - the only real mutex between
concurrent server.py sessions that share a data directory.

Why this file exists: every Claude Code session runs its own server.py
process, so `threading.Lock` only serialises threads inside one of them.
`_cross_process_lock` (util.py:190) is what actually stops two sessions from
corrupting shared state, and it guards five call sites whose state has no
git-diff rollback behind it:

  * kb/core.py:389                       schema_init.lock  (schema creation/WAL)
  * kb/search/_retrieval.py:1139,1203,   embeddings_lock_file (the shared FAISS
    1265,1332                            index and its id-map, both in data_dir)

test_embeddings_reliability.py::test_concurrent_agents_do_not_lose_each_others_embeddings
already exercises this indirectly, through KnowledgeBase. These tests hit the
primitive itself.

Real OS processes, not threads: threads share one process's file-descriptor
table and its Python-level state, so a thread-based test can pass while the
cross-process exclusion property does not hold. test_mcp_startup.py:172 is the
in-repo precedent for spawning real subprocesses on Windows.

Run with:  pytest test_cross_process_lock.py -v
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from util import _cross_process_lock

REPO = Path(__file__).parent

# Prefer the project venv, matching test_mcp_startup.py / test_embeddings_reliability.py.
_VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

# Contention parameters for the two-process exclusion test. ITERATIONS * HOLD_S
# is the work each process does inside the critical section; if the two really
# serialise, the wall clock of the contended region is ~2x that.
ITERATIONS = 6
HOLD_S = 0.25
SERIAL_WORK_S = ITERATIONS * HOLD_S


# ---------------------------------------------------------------------------
# 1. Two real processes contending for one lock_path must serialise.
# ---------------------------------------------------------------------------

_CONTENDER = """
import os, sys, time
from pathlib import Path

repo = sys.argv[1]
sys.path.insert(0, repo)
from util import _cross_process_lock

lock_path, occupancy, barrier_dir, worker_id = sys.argv[2:6]
iters = int(sys.argv[6])
hold = float(sys.argv[7])

# Barrier: do not start acquiring until BOTH processes are up, so their
# critical sections actually overlap in time. Without this the two could run
# back-to-back and the test would pass vacuously, proving nothing about
# exclusion.
Path(barrier_dir, worker_id + '.ready').write_text(worker_id)
deadline = time.time() + 60
while time.time() < deadline:
    if len(list(Path(barrier_dir).glob('*.ready'))) >= 2:
        break
    time.sleep(0.02)
else:
    print('BARRIER-TIMEOUT', flush=True)
    raise SystemExit(2)

start = time.time()
for i in range(iters):
    with _cross_process_lock(Path(lock_path), timeout=120.0, poll_interval=0.01):
        # Occupancy marker, created with the same O_CREAT|O_EXCL atomic
        # primitive the lock itself uses: if the peer process is ALSO inside
        # the critical section right now, exactly one of us sees FileExistsError.
        # This is the actual exclusion assertion - the timing check in the
        # parent is only corroborating evidence that contention happened.
        try:
            fd = os.open(occupancy, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print('VIOLATION-CONCURRENT-ENTRY ' + worker_id, flush=True)
            raise SystemExit(3)
        os.write(fd, worker_id.encode())
        os.close(fd)
        time.sleep(hold)
        # And nobody may have replaced our marker while we held the lock.
        seen = Path(occupancy).read_text()
        if seen != worker_id:
            print('VIOLATION-MARKER-OVERWRITTEN by ' + seen, flush=True)
            raise SystemExit(4)
        os.unlink(occupancy)

print('OK %s %d %.3f' % (worker_id, iters, time.time() - start), flush=True)
"""


def _spawn_contender(tmp_path, lock, occupancy, barrier_dir, worker_id):
    out = open(tmp_path / (worker_id + ".out"), "wb")
    err = open(tmp_path / (worker_id + ".err"), "wb")
    proc = subprocess.Popen(
        [PYTHON, "-c", _CONTENDER, str(REPO), str(lock), str(occupancy),
         str(barrier_dir), worker_id, str(ITERATIONS), str(HOLD_S)],
        cwd=str(REPO), stdout=out, stderr=err,
    )
    return proc, out, err


def test_two_processes_never_hold_the_same_lock_at_once(tmp_path):
    """Two OS processes hammering one lock_path must never both be inside.

    Each iteration creates an occupancy marker with O_CREAT|O_EXCL while
    holding the lock; a second process inside the critical section at the same
    instant would fail that create (exit 3) or find its marker overwritten
    (exit 4). The elapsed-time assertion then confirms the two processes really
    did contend, so a green result cannot come from them simply never
    overlapping.
    """
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    lock = tmp_path / "embeddings.lock"
    occupancy = tmp_path / "occupied"

    procs = []
    handles = []
    for worker_id in ("A", "B"):
        proc, out, err = _spawn_contender(tmp_path, lock, occupancy, barrier_dir, worker_id)
        procs.append((worker_id, proc))
        handles += [out, err]

    try:
        for _, proc in procs:
            proc.wait(timeout=300)
    finally:
        for _, proc in procs:
            if proc.poll() is None:  # pragma: no cover - only on a hang
                proc.kill()
        for h in handles:
            h.close()

    reports = {}
    for worker_id, proc in procs:
        stdout = (tmp_path / (worker_id + ".out")).read_text(errors="replace").strip()
        stderr = (tmp_path / (worker_id + ".err")).read_text(errors="replace")
        assert proc.returncode == 0, (
            f"worker {worker_id} exited {proc.returncode}: {stdout!r}\n{stderr[-3000:]}"
        )
        assert stdout.startswith("OK "), f"worker {worker_id} said: {stdout!r}"
        reports[worker_id] = float(stdout.split()[-1])

    # Corroboration that the workers actually contended. Serialised, the
    # contended region lasts ~2 * SERIAL_WORK_S and the last finisher sees all
    # of it; if exclusion were broken the two would run in parallel and finish
    # in ~SERIAL_WORK_S. 1.5x sits clear of both.
    slowest = max(reports.values())
    assert slowest >= 1.5 * SERIAL_WORK_S, (
        f"both workers finished in {reports}, but {ITERATIONS} x {HOLD_S}s of "
        f"serialised work should take ~{2 * SERIAL_WORK_S:.1f}s for the last "
        "one out. They did not block on each other, so this test proved nothing."
    )

    # The lock leaves nothing behind.
    assert not lock.exists(), "lock file survived both processes exiting"
    assert not occupancy.exists()


# ---------------------------------------------------------------------------
# 2. A lock file older than stale_after is reclaimed, not deadlocked on.
# ---------------------------------------------------------------------------


def test_stale_lock_is_reclaimed(tmp_path):
    """A crashed holder's lock file must not block every future session forever."""
    lock = tmp_path / "schema_init.lock"
    lock.write_text("999999")  # pid of a process that died holding this
    old = time.time() - 300.0
    os.utime(lock, (old, old))

    start = time.time()
    with _cross_process_lock(lock, timeout=5.0, stale_after=120.0, poll_interval=0.01):
        elapsed = time.time() - start
        assert lock.exists(), "reclaimed lock should be re-created for us"
        assert lock.read_text() == str(os.getpid()), (
            "after reclaiming, the lock file must name US as the holder"
        )
    assert elapsed < 2.0, f"stale reclaim took {elapsed:.2f}s - it should be immediate"
    assert not lock.exists(), "lock file not cleaned up on release"


def test_stale_threshold_is_measured_from_mtime(tmp_path):
    """Just-under-stale is still respected; just-over is reclaimed.

    Deterministic: both cases are set up by stamping mtime, not by sleeping.
    """
    lock = tmp_path / "schema_init.lock"
    lock.write_text("999999")

    just_under = time.time() - 100.0
    os.utime(lock, (just_under, just_under))
    with pytest.raises(TimeoutError):
        with _cross_process_lock(lock, timeout=0.3, stale_after=120.0, poll_interval=0.01):
            pytest.fail("reclaimed a lock that is younger than stale_after")
    assert lock.read_text() == "999999", "younger-than-stale lock was tampered with"

    just_over = time.time() - 140.0
    os.utime(lock, (just_over, just_over))
    with _cross_process_lock(lock, timeout=2.0, stale_after=120.0, poll_interval=0.01):
        assert lock.read_text() == str(os.getpid())


# ---------------------------------------------------------------------------
# 3. A lock file YOUNGER than stale_after must NOT be stolen.
#
#    This is testable deterministically, in both directions:
#      * in-process, by stamping an mtime and giving the waiter a timeout
#        far shorter than stale_after (no sleep-race involved: the only
#        timing assumption is 0.5s < 120s);
#      * cross-process, by having a real second process hold the lock across
#        the whole of the waiter's timeout window, with a ready-marker
#        handshake so the waiter provably starts while the holder is inside.
#
#    That includes a holder that legitimately runs LONGER than stale_after (a
#    large embeddings rebuild): it used to be robbed, because the lock file's
#    mtime was stamped once at acquisition and never refreshed, making a live
#    slow holder indistinguishable from a crashed one. util.py now restamps the
#    mtime from a background thread while the lock is held, so stale_after
#    measures silence rather than elapsed work; the two tests at the end of this
#    section pin that, and the ownership check that backstops it.
# ---------------------------------------------------------------------------


def test_live_holders_lock_is_not_stolen_by_a_waiter(tmp_path):
    """A waiter must time out rather than seize a lock that is still young."""
    lock = tmp_path / "embeddings.lock"
    lock.write_text("12345")  # a peer session, mid-write, mtime = now

    start = time.time()
    with pytest.raises(TimeoutError):
        with _cross_process_lock(lock, timeout=0.5, stale_after=120.0, poll_interval=0.01):
            pytest.fail("stole a lock held by a live peer")
    elapsed = time.time() - start

    assert lock.exists(), "the waiter deleted a lock it never owned"
    assert lock.read_text() == "12345", (
        "the waiter overwrote the live holder's lock file - the holder would "
        "then delete the NEXT holder's lock on release"
    )
    assert 0.5 <= elapsed < 20.0, f"waiter returned after {elapsed:.2f}s, expected ~0.5s"


_SLOW_HOLDER = """
import os, sys, time
from pathlib import Path

repo = sys.argv[1]
sys.path.insert(0, repo)
from util import _cross_process_lock

lock_path, ready_marker = sys.argv[2:4]
hold = float(sys.argv[4])
stale_after = float(sys.argv[5])

with _cross_process_lock(Path(lock_path), timeout=60.0, stale_after=stale_after):
    Path(ready_marker).write_text(str(os.getpid()))
    time.sleep(hold)
print('DONE %d' % os.getpid(), flush=True)
"""


def test_another_process_holding_the_lock_blocks_us_out(tmp_path):
    """A real second process's lock must survive our whole wait window.

    Deterministic despite involving sleeps: the holder announces itself with a
    ready marker before we start waiting, and holds for far longer (6s) than
    our 1s timeout, so 'the holder was still inside' is not a race we hope for.
    """
    lock = tmp_path / "embeddings.lock"
    ready = tmp_path / "holder.ready"
    hold_s = 6.0

    with open(tmp_path / "holder.out", "wb") as out, open(tmp_path / "holder.err", "wb") as err:
        proc = subprocess.Popen(
            [PYTHON, "-c", _SLOW_HOLDER, str(REPO), str(lock), str(ready), str(hold_s), "3600.0"],
            cwd=str(REPO), stdout=out, stderr=err,
        )
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not ready.exists():
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), (
            "holder process never acquired the lock: "
            + (tmp_path / "holder.err").read_text(errors="replace")[-3000:]
        )
        holder_pid = ready.read_text()
        assert lock.read_text() == holder_pid, "lock file should record the holder's pid"

        with pytest.raises(TimeoutError):
            with _cross_process_lock(lock, timeout=1.0, stale_after=3600.0, poll_interval=0.01):
                pytest.fail("acquired a lock another live process is holding")

        assert lock.exists(), "we deleted another process's lock file"
        assert lock.read_text() == holder_pid, "we overwrote another process's lock file"

        assert proc.wait(timeout=120) == 0, (
            (tmp_path / "holder.err").read_text(errors="replace")[-3000:]
        )
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hang
            proc.kill()
            proc.wait(timeout=30)

    assert not lock.exists(), "holder did not release its lock"


def test_holder_slower_than_stale_after_is_not_robbed(tmp_path):
    """A holder still working past stale_after keeps its lock (was: lost it).

    This is the case a real embeddings rebuild hits: the work outlasts the
    120s default. It used to be robbed, because the lock file's mtime was
    written once at acquisition, so `stale_after` was measuring elapsed work
    rather than silence. The holder now restamps that mtime while it holds the
    lock, and a waiter must therefore time out instead of seizing it.

    stale_after is dialled down to 0.25s so the window is reachable in a test.
    The timing assumptions are one-directional and generous: we hold for 1.0s,
    which is 4x stale_after, and the heartbeat runs every stale_after/4 =
    0.0625s, so ~16 refreshes are due in that window and losing most of them to
    scheduler jitter would still not make the lock look stale.
    """
    lock = tmp_path / "embeddings.lock"

    with _cross_process_lock(lock, timeout=5.0, stale_after=0.25, poll_interval=0.01):
        stamped_at_acquisition = lock.stat().st_mtime_ns

        time.sleep(1.0)  # the holder is still working, well past stale_after

        assert lock.stat().st_mtime_ns > stamped_at_acquisition, (
            "the lock file's mtime was never refreshed while held, so a live "
            "slow holder is still indistinguishable from a crashed one"
        )

        with pytest.raises(TimeoutError):
            with _cross_process_lock(lock, timeout=0.3, stale_after=0.25, poll_interval=0.01):
                pytest.fail("robbed a live holder that had simply outrun stale_after")

        assert lock.read_text() == str(os.getpid()), "the waiter overwrote our lock file"

    assert not lock.exists(), "the holder did not release its own lock"


def test_a_real_slow_process_keeps_its_lock_past_stale_after(tmp_path):
    """The same property as above, but between two real OS processes.

    The in-process test proves the heartbeat mechanism; this proves it is a
    property of the mutex as another process sees it, which is the only thing
    the five call sites actually depend on. The holder announces itself before
    we start, holds for 4s with stale_after=0.5s (8x over), and we keep trying
    to take it for ~3s of that - so every acquisition attempt happens while the
    holder is provably inside and provably 'older' than stale_after.
    """
    lock = tmp_path / "embeddings.lock"
    ready = tmp_path / "holder.ready"

    with open(tmp_path / "holder.out", "wb") as out, open(tmp_path / "holder.err", "wb") as err:
        proc = subprocess.Popen(
            [PYTHON, "-c", _SLOW_HOLDER, str(REPO), str(lock), str(ready), "4.0", "0.5"],
            cwd=str(REPO), stdout=out, stderr=err,
        )
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not ready.exists():
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), (
            "holder process never acquired the lock: "
            + (tmp_path / "holder.err").read_text(errors="replace")[-3000:]
        )
        holder_pid = ready.read_text()

        attempts_deadline = time.time() + 3.0
        attempts = 0
        while time.time() < attempts_deadline:
            with pytest.raises(TimeoutError):
                with _cross_process_lock(lock, timeout=0.3, stale_after=0.5, poll_interval=0.01):
                    pytest.fail(
                        "robbed a live peer process that had outrun stale_after - "
                        "its embeddings write is now racing ours"
                    )
            attempts += 1
            assert lock.read_text() == holder_pid, "the holder's lock file was replaced"
        assert attempts >= 3, f"only {attempts} acquisition attempts - test proved little"

        assert proc.wait(timeout=120) == 0, (
            (tmp_path / "holder.err").read_text(errors="replace")[-3000:]
        )
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hang
            proc.kill()
            proc.wait(timeout=30)

    assert not lock.exists(), "holder did not release its lock"


@pytest.mark.parametrize("robber_pid", ["999999", "self"])
def test_release_does_not_delete_a_lock_file_it_no_longer_owns(tmp_path, robber_pid):
    """A robbed holder's release must be a no-op, not a second theft.

    Backstop for the heartbeat above: if a holder does lose its lock anyway -
    stopped process, frozen filesystem, a stale_after tuned below the pause -
    its eventual release used to unlink unconditionally, deleting the file its
    SUCCESSOR was holding and letting a third process straight into the
    critical section. Here the reclaim is performed directly rather than waited
    for, so nothing about this test is a race.

    Parametrised over the robber's pid because pid equality alone cannot decide
    ownership: 'self' is the case where the robber happens to be this same
    process, which a pid check would wave through.
    """
    lock = tmp_path / "embeddings.lock"
    stolen_by = str(os.getpid()) if robber_pid == "self" else robber_pid

    holder = _cross_process_lock(lock, timeout=5.0, stale_after=120.0, poll_interval=0.01)
    holder.__enter__()
    released = False
    try:
        assert lock.exists()

        # Someone else reclaims it - exactly what the stale path does: unlink,
        # then create afresh recording their own pid.
        lock.unlink()
        time.sleep(0.01)
        lock.write_text(stolen_by)

        holder.__exit__(None, None, None)
        released = True

        assert lock.exists(), (
            "the robbed holder's release deleted the lock file its robber now "
            "holds - a third process would walk straight into the critical "
            "section while the second is still inside it"
        )
        assert lock.read_text() == stolen_by, "release tampered with the new holder's lock file"

        # And that surviving lock is honoured: nobody is admitted behind it.
        with pytest.raises(TimeoutError):
            with _cross_process_lock(lock, timeout=0.3, stale_after=120.0, poll_interval=0.01):
                pytest.fail("entered a critical section the robber is still inside")
    finally:
        if not released:  # pragma: no cover - only when an assertion above fails
            holder.__exit__(None, None, None)
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Basic hygiene the five call sites depend on.
# ---------------------------------------------------------------------------


def test_lock_is_released_when_the_body_raises(tmp_path):
    """A failed embeddings write must not leave the lock behind for 120s."""
    lock = tmp_path / "embeddings.lock"

    with pytest.raises(ValueError):
        with _cross_process_lock(lock, timeout=5.0):
            assert lock.exists()
            raise ValueError("simulated failure inside the critical section")

    assert not lock.exists(), "an exception leaked the lock file"
    # And the lock is immediately usable again.
    with _cross_process_lock(lock, timeout=5.0):
        assert lock.read_text() == str(os.getpid())


def test_lock_records_the_holder_pid(tmp_path):
    """The lock file names its holder, which is what makes a stale one diagnosable."""
    lock = tmp_path / "schema_init.lock"
    with _cross_process_lock(lock, timeout=5.0):
        assert lock.read_text() == str(os.getpid())


def test_timeout_error_names_the_lock_path(tmp_path):
    """The failure message must say WHICH lock, since two are in play (schema, embeddings)."""
    lock = tmp_path / "schema_init.lock"
    lock.write_text("999999")
    with pytest.raises(TimeoutError, match="schema_init.lock"):
        with _cross_process_lock(lock, timeout=0.2, stale_after=120.0, poll_interval=0.01):
            pytest.fail("acquired a held lock")


def test_lock_is_not_reentrant_within_one_process(tmp_path):
    """Documented shape of the primitive: it is a file mutex, not an RLock.

    _load_embeddings_locked exists precisely because re-entering would
    deadlock (util.py's docstring for it says so); this pins that down.
    """
    lock = tmp_path / "embeddings.lock"
    with _cross_process_lock(lock, timeout=5.0):
        with pytest.raises(TimeoutError):
            with _cross_process_lock(lock, timeout=0.2, stale_after=120.0, poll_interval=0.01):
                pytest.fail("re-entered a lock this process already holds")
        assert lock.exists()
