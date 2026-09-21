#!/usr/bin/env python3
"""
TDZ C64 Knowledge - optional-dependency detection and lazy-loaded model factories

Whether each optional dependency (nltk, sentence-transformers, faiss, rapidfuzz,
rank-bm25, pypdf, pdfplumber, pytesseract/pdf2image, PyMuPDF, cachetools,
anomaly_detector) is available, plus the lazy factories for the two heavy
sentence-transformers classes. Split out of server.py alongside models.py/
text_utils.py so KnowledgeBase no longer depends on anything defined in
server.py itself - a prerequisite for splitting that 253-method class into
mixin modules without a circular import.

Heavy optional dependencies (nltk, sentence-transformers, faiss) are detected
here but NOT imported. Importing them eagerly cost ~16s of the ~18s startup
time, which pushed MCP client handshakes past their 30s timeout whenever more
than one Claude Code session started a server at once. They are imported on
first actual use via the lazy helpers below - see CLAUDE.md "MCP startup
performance". Keep it that way: do not turn any of the `_module_available`
checks below into a real top-level import.
"""

import logging
import os
import sys
import threading
import time
import importlib.util

from util import _LazyModule, _network_timeout

# Anomaly detection support
try:
    from anomaly_detector import AnomalyDetector, CheckResult, AnomalyScore, Baseline
    ANOMALY_SUPPORT = True
except ImportError:
    ANOMALY_SUPPORT = False
    AnomalyDetector = None
    print("Warning: anomaly_detector not found. Anomaly detection disabled.", file=sys.stderr)

# Caching support
try:
    from cachetools import TTLCache
    CACHE_SUPPORT = True
except ImportError:
    CACHE_SUPPORT = False
    print("Warning: cachetools not installed. Search caching disabled.", file=sys.stderr)

# Optional imports for PDF support
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed. PDF support disabled.", file=sys.stderr)

# PDF table extraction support
try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("Warning: pdfplumber not installed. Table extraction disabled.", file=sys.stderr)

# BM25 search support
try:
    from rank_bm25 import BM25Okapi
    BM25_SUPPORT = True
except ImportError:
    BM25_SUPPORT = False
    print("Warning: rank-bm25 not installed. Using simple search.", file=sys.stderr)


def _module_available(name: str) -> bool:
    """Check whether a module can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# NLTK for query preprocessing (imported lazily - see _ensure_nltk)
NLTK_SUPPORT = _module_available('nltk')
if not NLTK_SUPPORT:
    print("Warning: nltk not installed. Query preprocessing disabled.", file=sys.stderr)

_nltk_ready = False

# Importing nltk pulls in nltk.collocations -> nltk.metrics.association ->
# scipy.stats -> scipy.linalg, which loads a chain of C-extension DLLs. Each
# of those DLL initialisations touches this process's stdin, and Windows
# serialises operations on a synchronous file object. Under the MCP stdio
# transport a reader thread is permanently parked in a read on stdin that only
# completes when the client sends its next message, so every DLL in the chain
# queues behind it - while the client is itself blocked waiting for the reply
# to the search that started the import. Nothing breaks the tie and the search
# never returns.
#
# Measured, 2026-09-20: a search over a long-lived stdio session was still
# unanswered after 90s; feeding that same server one extra request per second
# let it finish after exactly 11 of them. The identical import costs 0.2s in a
# process with no pending read on stdin, and the minimal reproduction is a
# thread blocked in os.read(0, 1) plus `import scipy.linalg.blas` - which
# completes 0.2s after one byte is written to the pipe.
#
# So the import runs once, on its own daemon thread, and callers wait for it
# with a deadline instead of inheriting that unbounded wait. A caller that hits
# the deadline degrades for that one call (see _preprocess_text) while the
# import carries on in the background and lands as soon as any further client
# traffic releases stdin, so the next call gets the real stemmer. Warming it at
# module level instead is not an option: that is exactly the ~16s startup
# regression CLAUDE.md and test_mcp_startup.py's FORBIDDEN_AT_STARTUP forbid.
NLTK_IMPORT_TIMEOUT_S = float(os.environ.get('TDZ_NLTK_IMPORT_TIMEOUT_S', '5'))

# Once one caller has already waited out the full deadline, later callers
# wait only this long. Under the stdin stall the import advances by roughly
# one DLL per client message, so re-paying the full deadline on every search
# would charge it to each of the ~11 messages the chain needs; measured cold
# in a process with no pending read on stdin the whole thing costs 0.64s, so
# a caller that has seen the deadline blown is looking at the stalled case
# and should degrade immediately rather than wait again.
_NLTK_RETRY_WAIT_S = 0.05

_nltk_parts = None
_nltk_done = threading.Event()
_nltk_start_lock = threading.Lock()
_nltk_thread = None
_nltk_deadline_missed = False
_nltk_logger = logging.getLogger(__name__)


def _nltk_import_worker() -> None:
    """Import nltk and fetch its corpora exactly once, off the caller's thread.

    Never raises: a failure leaves _nltk_parts None and _ensure_nltk returns
    None, which callers already treat as "degrade to no preprocessing".
    """
    global _nltk_parts, _nltk_ready
    started = time.monotonic()
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        _nltk_logger.info("nltk import finished in %.2fs", time.monotonic() - started)
        if not _nltk_ready:
            # Ensure NLTK data is available. This can hit the network, which is
            # why it must never run at import time - a slow or offline mirror
            # would stall the MCP handshake instead of just this one call.
            try:
                stopwords.words('english')
            except LookupError:
                try:
                    with _network_timeout():
                        nltk.download('stopwords', quiet=True)
                        nltk.download('punkt', quiet=True)
                        nltk.download('punkt_tab', quiet=True)
                except Exception as e:
                    # Network unreachable/filtered - the caller's except-and-
                    # degrade logic (see _preprocess_text) handles this fine as
                    # long as we don't hang here first. Mark ready regardless so
                    # every subsequent search doesn't re-attempt (and re-wait
                    # out) the same doomed download.
                    print(f"Warning: NLTK data download failed or timed out: {e}", file=sys.stderr)
            _nltk_ready = True
        _nltk_parts = (PorterStemmer, stopwords, word_tokenize)
    except Exception as e:
        _nltk_logger.warning("nltk unavailable, query preprocessing will degrade: %s", e)
    finally:
        _nltk_logger.info("nltk warm-up returned after %.2fs", time.monotonic() - started)
        _nltk_done.set()


def _ensure_nltk(timeout: float | None = None):
    """Start (once) and wait on the deferred nltk import, with a deadline.

    Returns (PorterStemmer_cls, stopwords_mod, word_tokenize_fn), or None if
    nltk is unavailable or its data could not be fetched.

    Raises TimeoutError if the one-time import is still in flight after
    `timeout` seconds (default NLTK_IMPORT_TIMEOUT_S). That is a "not yet",
    not a failure: the caller degrades for this one call and retries on the
    next, because the import does finish - see the comment above for what it
    is waiting on and why the caller must not wait with it.
    """
    global _nltk_thread, _nltk_deadline_missed
    if not NLTK_SUPPORT:
        return None
    if not _nltk_done.is_set():
        with _nltk_start_lock:
            if _nltk_thread is None:
                _nltk_thread = threading.Thread(
                    target=_nltk_import_worker, name='nltk-warmup', daemon=True,
                )
                _nltk_thread.start()
        if timeout is None:
            timeout = _NLTK_RETRY_WAIT_S if _nltk_deadline_missed else NLTK_IMPORT_TIMEOUT_S
        if not _nltk_done.wait(timeout):
            _nltk_deadline_missed = True
            raise TimeoutError(
                f"nltk is still importing after {timeout:.1f}s; degrading this "
                f"call rather than blocking on it (TDZ_NLTK_IMPORT_TIMEOUT_S "
                f"tunes the deadline)"
            )
    return _nltk_parts


# Semantic search support (sentence-transformers + faiss imported lazily)
SEMANTIC_SUPPORT = _module_available('sentence_transformers') and _module_available('faiss')
if not SEMANTIC_SUPPORT:
    print("Warning: sentence-transformers or faiss-cpu not installed. Semantic search disabled.", file=sys.stderr)

# faiss is referenced from many methods; the proxy defers the ~4s import until
# the first embedding operation actually touches it.
faiss = _LazyModule('faiss')


def SentenceTransformer(*args, **kwargs):
    """Lazily import and instantiate the real SentenceTransformer class."""
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    return _SentenceTransformer(*args, **kwargs)


def CrossEncoder(*args, **kwargs):
    """Lazily import and instantiate the real CrossEncoder class.

    Same reasoning as SentenceTransformer above: importing this at module
    level would put the transformers/torch import cost back on every MCP
    handshake, which is exactly what test_mcp_startup.py forbids.
    """
    from sentence_transformers import CrossEncoder as _CrossEncoder
    return _CrossEncoder(*args, **kwargs)


# Fuzzy search support
try:
    from rapidfuzz import fuzz
    FUZZY_SUPPORT = True
except ImportError:
    FUZZY_SUPPORT = False
    print("Warning: rapidfuzz not installed. Fuzzy search disabled.", file=sys.stderr)

# OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False
    print("Warning: pytesseract/pdf2image/Pillow not installed. OCR disabled.", file=sys.stderr)

# Embedded-figure extraction needs PyMuPDF. Detected without importing: fitz
# pulls in a sizeable native library, and only the figure-OCR path uses it,
# so paying that cost at startup would slow every session down for a feature
# most calls never touch (see the startup budget notes in CLAUDE.md).
FIGURE_SUPPORT = _module_available('fitz')
if not FIGURE_SUPPORT:
    print("Warning: PyMuPDF (fitz) not installed. Figure OCR disabled.", file=sys.stderr)

# markitdown fallback extraction (docx/pptx). Detected without importing:
# markitdown imports magika at module level, and magika pulls in onnxruntime,
# so an eager import here would put that cost back on every MCP handshake.
# A future caller must import markitdown inside a function, not at module
# level (see CLAUDE.md startup-budget notes).
MARKITDOWN_SUPPORT = _module_available('markitdown')
if not MARKITDOWN_SUPPORT:
    print("Warning: markitdown not installed. Fallback document extraction disabled.", file=sys.stderr)
