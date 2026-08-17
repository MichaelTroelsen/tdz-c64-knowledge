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

import sys
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


def _ensure_nltk():
    """Import nltk and download its corpora on first use.

    Returns (PorterStemmer_cls, stopwords_mod, word_tokenize_fn) or None if
    nltk is unavailable or its data cannot be fetched.
    """
    global _nltk_ready
    if not NLTK_SUPPORT:
        return None
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
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
                sys.stderr.write(f"Warning: NLTK data download failed or timed out: {e}\n")
        _nltk_ready = True
    return PorterStemmer, stopwords, word_tokenize


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
