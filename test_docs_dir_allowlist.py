"""ALLOWED_DOCS_DIRS guards add_document/scrape_url against arbitrary-path
reads - a path outside it must be rejected, and none of the defaults or the
env var it merges with may silently stop being honoured.

The only prior coverage of this boundary lived in archive/tests/test_security.py,
which pyproject.toml's norecursedirs (archive) excludes from collection - so at
this head nothing collected would notice if the check stopped working. This
file is that missing live test.

kb/core.py builds self.allowed_dirs from four server-controlled defaults under
data_dir (scraped_docs, downloads, temp, uploads), merged with the
comma-separated ALLOWED_DOCS_DIRS environment variable, and - only when
TDZ_ALLOW_CWD=1 - the process's current working directory. cwd is excluded by
default on purpose: this server is commonly registered at Claude Code user
scope and launched with the cwd of whatever project the caller happens to be
in, so unconditionally allowing it would make that entire project tree
ingestible.

server.py's `load_dotenv` loads this repo's (gitignored) .env at import time,
and on this machine .env sets ALLOWED_DOCS_DIRS to real paths on disk. Every
fixture here explicitly deletes/sets ALLOWED_DOCS_DIRS and TDZ_ALLOW_CWD
before constructing a KnowledgeBase, and TDZ_DATA_DIR always points at
tmp_path, so these tests assert on what they configured, not on this
machine's .env or its live database.
"""

from pathlib import Path

import pytest

from models import SecurityError


@pytest.fixture
def make_kb(tmp_path, monkeypatch):
    """Factory for a KnowledgeBase on an isolated data dir, with
    ALLOWED_DOCS_DIRS/TDZ_ALLOW_CWD explicitly cleared unless a test sets
    its own value first. Never touches the live ~/.tdz-c64-knowledge db."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    monkeypatch.delenv("ALLOWED_DOCS_DIRS", raising=False)
    monkeypatch.delenv("TDZ_ALLOW_CWD", raising=False)

    created = []

    def _make():
        from server import KnowledgeBase
        instance = KnowledgeBase(str(tmp_path))
        created.append(instance)
        return instance

    yield _make

    for instance in created:
        instance.close()


def test_default_data_dir_subdirs_are_accepted(make_kb, tmp_path):
    """The four server-controlled defaults are allowed with no env var set
    at all - dropping any one of them from kb/core.py would fail this."""
    kb = make_kb()

    for name in ("scraped_docs", "downloads", "temp", "uploads"):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        f = d / "doc.md"
        f.write_text("content", encoding="utf-8")
        assert kb._is_path_allowed(str(f)) is True, f"default dir {name!r} should be allowed"


def test_path_outside_allowlist_is_rejected(make_kb, tmp_path):
    kb = make_kb()

    outside = tmp_path.parent / "outside-allowlist-boundary"
    outside.mkdir(exist_ok=True)
    f = outside / "secret.md"
    f.write_text("content", encoding="utf-8")

    assert kb._is_path_allowed(str(f)) is False

    with pytest.raises(SecurityError):
        kb.add_document(str(f))


def test_allowed_docs_dirs_env_var_extends_the_defaults(make_kb, tmp_path, monkeypatch):
    """The env var must MERGE with the data_dir defaults, not replace them -
    a test that only proved the env-configured dir is accepted would pass
    even if the four defaults were dropped."""
    extra = tmp_path.parent / "extra-allowed"
    extra.mkdir(exist_ok=True)
    monkeypatch.setenv("ALLOWED_DOCS_DIRS", str(extra))

    kb = make_kb()

    extra_file = extra / "doc.md"
    extra_file.write_text("content", encoding="utf-8")
    assert kb._is_path_allowed(str(extra_file)) is True

    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    default_file = uploads / "doc.md"
    default_file.write_text("content", encoding="utf-8")
    assert kb._is_path_allowed(str(default_file)) is True, (
        "a default data_dir subdir stopped being allowed once ALLOWED_DOCS_DIRS was set"
    )


def test_cwd_is_excluded_by_default(make_kb, tmp_path, monkeypatch):
    """The deliberate decision documented in kb/core.py: cwd is NOT
    auto-allowed, because this server is often launched with the cwd of
    whatever project the caller happens to be in."""
    work_dir = tmp_path.parent / "cwd-sim"
    work_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(work_dir)

    kb = make_kb()

    assert Path.cwd().resolve() not in kb.allowed_dirs

    f = work_dir / "doc.md"
    f.write_text("content", encoding="utf-8")
    assert kb._is_path_allowed(str(f)) is False


def test_tdz_allow_cwd_opts_in_to_cwd(make_kb, tmp_path, monkeypatch):
    """TDZ_ALLOW_CWD=1 is the explicit opt-in the code comment in kb/core.py
    describes for the CLI/dev workflow."""
    work_dir = tmp_path.parent / "cwd-sim-optin"
    work_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(work_dir)
    monkeypatch.setenv("TDZ_ALLOW_CWD", "1")

    kb = make_kb()

    assert Path.cwd().resolve() in kb.allowed_dirs

    f = work_dir / "doc.md"
    f.write_text("content", encoding="utf-8")
    assert kb._is_path_allowed(str(f)) is True


def test_rejection_is_driven_by_allowed_dirs_content_not_vacuous(make_kb, tmp_path):
    """Negative control: widen kb.allowed_dirs IN MEMORY ONLY (kb/core.py is
    never touched) and confirm the exact path rejected above now passes.
    This proves the rejection assertion actually discriminates on
    allowed_dirs's content, rather than passing for some unrelated reason
    (e.g. _is_path_allowed always returning False, or the test path
    accidentally landing inside a default dir)."""
    kb = make_kb()

    outside = tmp_path.parent / "sabotage-target"
    outside.mkdir(exist_ok=True)
    f = outside / "secret.md"
    f.write_text("content", encoding="utf-8")

    before = list(kb.allowed_dirs)
    assert kb._is_path_allowed(str(f)) is False
    print(f"allowed_dirs BEFORE in-memory sabotage: {before}")

    kb.allowed_dirs = list(kb.allowed_dirs) + [outside.resolve()]
    after = list(kb.allowed_dirs)
    print(f"allowed_dirs AFTER in-memory sabotage:  {after}")

    assert kb._is_path_allowed(str(f)) is True, (
        "widening kb.allowed_dirs in memory should flip this path to accepted; "
        "if it doesn't, the rejection test above isn't actually pinned to allowed_dirs"
    )
