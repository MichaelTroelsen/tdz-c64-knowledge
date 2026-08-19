#!/usr/bin/env python3
"""
TDZ C64 Knowledge - MCP Server
A Model Context Protocol server for searching C64 documentation.
"""

import asyncio
import os
import sys
import json
import re
import hashlib
import logging
import time
import sqlite3
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file. Resolve relative to this
# script's own directory, not the launching process's cwd - the MCP client
# can start this server from any project's working directory.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# Import version information
from version import __build_date__, __version__, get_full_version_string

# Self-contained, stdlib-only helpers (lazy module loading, outbound-network
# politeness, cross-process locking, DB busy-retry) - see util.py. Kept as a
# separate module because it must stay import-light: see util.py's own
# docstring and CLAUDE.md "MCP startup performance".
from util import (
    USER_AGENT,
    _LazyModule,
    _atomic_write_bytes,
    _cross_process_lock,
    _network_timeout,
    _retry_on_db_locked,
    http_get_polite,
    http_headers,
    robots_allows,
)

# Pure-data types (dataclasses, exceptions, the progress-callback type alias)
# used throughout KnowledgeBase - see models.py. Split out so the eventual
# mixin-module split of the 253-method class isn't circular: methods need
# these types, but the types must not depend on anything defined inside the
# class's own module.
from models import (
    DocumentChunk,
    DocumentMeta,
    DocumentNotFoundError,
    KnowledgeBaseError,
    ProgressCallback,
    ProgressUpdate,
    SecurityError,
    UnsupportedFileTypeError,
)

# Snippet/term-matching text helpers - see text_utils.py. Same circular-
# import reasoning as models.py above.
from text_utils import (
    _SENTENCE_BOUNDARY_RE,
    _content_terms,
    _expand_brace_pattern,
    _filter_snippet_terms,
)

# Optional-dependency detection and lazy-loaded heavy-model factories - see
# features.py, including the "MCP startup performance" rationale for why
# nltk/sentence-transformers/faiss are detected but never imported here.
from features import (
    ANOMALY_SUPPORT,
    AnomalyDetector,
    BM25_SUPPORT,
    BM25Okapi,
    CACHE_SUPPORT,
    CrossEncoder,
    FIGURE_SUPPORT,
    FUZZY_SUPPORT,
    NLTK_SUPPORT,
    OCR_SUPPORT,
    PDF_SUPPORT,
    PDFPLUMBER_SUPPORT,
    PdfReader,
    SEMANTIC_SUPPORT,
    SentenceTransformer,
    TTLCache,
    _ensure_nltk,
    convert_from_path,
    faiss,
    fuzz,
    pdfplumber,
    pytesseract,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from kb import IngestMixin, FiguresMixin, SearchMixin, EntitiesMixin, GraphMixin, TopicsMixin, TemporalMixin, AdminMixin, CoreMixin
from mcp_tools import HANDLERS, TOOL_SCHEMAS
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)

# numpy is cheap (~0.3s) and used throughout, so it stays eager.
try:
    import numpy as np
except ImportError:
    np = None


# Custom Exceptions that stay module-local: nothing in KnowledgeBase reaches
# for these two directly (see models.py for the exceptions the class does
# use), so they were left out of that move rather than widened into it.
class ChunkNotFoundError(KnowledgeBaseError):
    """Raised when a chunk is not found."""
    pass


class IndexCorruptedError(KnowledgeBaseError):
    """Raised when the index is corrupted."""
    pass


class KnowledgeBase(IngestMixin, FiguresMixin, SearchMixin, EntitiesMixin, GraphMixin, TopicsMixin, TemporalMixin, AdminMixin, CoreMixin):
    """Manages the document index and search."""

    # ------------------------------------------------------------------
    # Figure OCR
    #
    # The ingest-time OCR path above only fires for PDFs detected as
    # *entirely* scanned. A normal text PDF gets its text layer indexed and
    # its embedded figures - memory maps, register tables, schematics, which
    # in C64 documentation is often where the actual reference data lives -
    # are never read at all. This pass extracts those images and OCRs each
    # one, as a background batch over documents already in the KB.
    # ------------------------------------------------------------------

    # Below these dimensions an embedded image is almost always a rule,
    # bullet, gradient or logo fragment - never a figure with readable text.
    FIGURE_MIN_WIDTH = int(os.getenv('TDZ_FIGURE_MIN_WIDTH', '120'))
    FIGURE_MIN_HEIGHT = int(os.getenv('TDZ_FIGURE_MIN_HEIGHT', '120'))
    # OCR on a figure with no text returns whitespace/punctuation noise;
    # storing those rows just dilutes search results.
    FIGURE_MIN_CHARS = int(os.getenv('TDZ_FIGURE_MIN_CHARS', '12'))

    # Typeset manuals draw schematics, memory maps and timing diagrams as
    # vector paths rather than embedding them as bitmaps, so get_images()
    # returns nothing for exactly the pages worth reading. Off by default:
    # rendering page regions costs far more than pulling out a stored bitmap.
    FIGURE_RASTERIZE_PAGES = os.getenv('TDZ_FIGURE_RASTERIZE_PAGES', '0') == '1'
    FIGURE_RASTER_DPI = int(os.getenv('TDZ_FIGURE_RASTER_DPI', '200'))
    # A drawing cluster this close to page-sized is a border or content frame.
    # Rendering it would OCR the body text a second time, duplicating what the
    # PDF's own text layer already put in the chunk index.
    FIGURE_RASTER_MAX_AREA = float(os.getenv('TDZ_FIGURE_RASTER_MAX_AREA', '0.9'))
    # OCR blocks in a tesseract subprocess, so the GIL is not what limits this
    # and a pool scales with cores. Default 1 to keep the previous behaviour.
    FIGURE_OCR_WORKERS = max(1, int(os.getenv('TDZ_FIGURE_OCR_WORKERS', '1')))

    # Words, optionally hyphenated (VIC-II, read-only). Everything else -
    # '?', '*', ':', parentheses, quotes - is punctuation to a user typing a
    # question but syntax to FTS5, so it never reaches the MATCH expression.
    _FTS5_TOKEN_RE = re.compile(r'\w+(?:-\w+)*', re.UNICODE)


# Initialize the MCP server
server = Server("tdz-c64-knowledge")

# Get data directory from environment or use default
DATA_DIR = os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge"))

# The KnowledgeBase is NOT constructed here at module level. Every consumer
# that only wants the class (rest_server.py, admin_gui.py, cli.py,
# wiki_export.py, the test suite) does `from server import KnowledgeBase` /
# `import server`, which executes this entire module regardless of which
# name it asks for - so a module-level `kb = KnowledgeBase(DATA_DIR)` here
# used to build a full KnowledgeBase (DB connection, background extraction
# worker thread, document load) as a side effect of every one of those
# imports, in addition to whatever instance the consumer then built for
# itself. get_kb() defers that cost to the one process that actually acts as
# the MCP server, via main() below.
kb: Optional['KnowledgeBase'] = None


def get_kb() -> 'KnowledgeBase':
    """Return the process-wide KnowledgeBase, constructing it on first call."""
    global kb
    if kb is None:
        kb = KnowledgeBase(DATA_DIR)
    return kb


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    # Schemas live in mcp_tools.schemas (R12); a new list each call so a
    # caller mutating the result cannot corrupt the shared registry.
    return list(TOOL_SCHEMAS)


def _call_tool_impl(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to its handler in mcp_tools.handlers.

    Deliberately synchronous: call_tool() runs this on a worker thread via
    asyncio.to_thread, so nothing reached from here may block the event
    loop and nothing may await.

    Was a 3,400-line `elif name == ...` chain (R12). A dict lookup cannot
    silently shadow a tool the way a mis-ordered elif could.
    """
    handler = HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    return handler(kb, name, arguments)


# Set to an asyncio.Lock only by the HTTP transport. stdio guarantees one
# client per process, so its dispatch path stays unlocked and unchanged.
_tool_call_lock = None


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Time and log every MCP tool invocation to mcp_call_log, then dispatch to
    the real implementation. Kept separate from _call_tool_impl so the
    logging wrapper doesn't get lost in that function's ~3000-line dispatch
    body.

    _call_tool_impl is synchronous and some of its handlers block for minutes
    to hours (OCR in add_document, a whole-site crawl in scrape_url, topic
    training, backup/restore). Awaiting it inline froze every other request on
    the session - including MCP keep-alives - for that entire time, so the
    whole dispatch runs on a worker thread. This is only safe because each
    thread now gets its own SQLite connection (see KnowledgeBase.db_conn);
    with the previously shared connection, concurrent tool calls committed
    each other's half-built transactions.
    """
    get_kb()  # idempotent; normally already initialized by main() before
              # the server starts accepting requests, but callers that
              # invoke call_tool()/_call_tool_impl directly (as some tests
              # do) would otherwise see a None `kb` in every bare reference
              # throughout the dispatch body below.
    start_time = time.time()
    error_message = None
    result = None
    try:
        if _tool_call_lock is None:
            result = await asyncio.to_thread(_call_tool_impl, name, arguments)
        else:
            # Over HTTP one process serves several clients, so two tool calls
            # can now overlap. Thread-local connections keep SQLite safe (see
            # KnowledgeBase.db_conn); everything else the tools touch is shared
            # mutable state. This lock is deliberately global. Audited, and the
            # obvious narrowing does not work:
            #
            #   - A reader/writer split fails because there are almost no
            #     readers. search, semantic_search, faceted_search,
            #     find_similar_documents, the entity queries and even
            #     health_check all WRITE a cache on the way out - 10 distinct
            #     `_cache[cache_key] = ...` sites across kb/. cachetools is not
            #     internally synchronised (TTLCache.__setitem__ takes no lock,
            #     checked against cachetools 6.2.4), so two "read-only" tools
            #     racing on one TTLCache corrupt it.
            #   - self.documents is iterated at 26 sites across kb/ and written
            #     in kb/core.py. Iterating it while another thread inserts
            #     raises "dictionary changed size during iteration" - a crash,
            #     not a subtle wrong answer.
            #   - self.embeddings_index is both mutated (.add) and REPLACED
            #     (= faiss.IndexFlatIP(...)) while searches read it; a faiss
            #     index is not safe for concurrent add and search.
            #
            # Narrowing is possible in principle - a lock per structure, taken
            # at each of those ~36 sites - and it would buy something real: a
            # long scrape_url or add_document would then hold the documents
            # lock only while mutating, not for the whole crawl. It is not
            # worth it yet. There is no concurrency test in the suite to catch
            # a missed site, and one missed site is a corrupted index rather
            # than a slow one. Revisit if multi-client HTTP use makes the
            # serialisation actually painful; write the concurrency tests
            # first.
            async with _tool_call_lock:
                result = await asyncio.to_thread(_call_tool_impl, name, arguments)
        return result
    except Exception as e:
        error_message = str(e)
        # mcp_call_log only stores the message, and the client only sees a
        # protocol-level error - without this the stack trace was lost
        # entirely, leaving nothing in server.log to debug from.
        kb.logger.exception(f"MCP tool {name!r} raised")
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        success = error_message is None
        if success and result:
            first_text = getattr(result[0], 'text', '') if result else ''
            if first_text.startswith('Error'):
                success = False
                error_message = first_text[:300]
                # A handler that caught its own exception and returned an
                # "Error: ..." string logged nothing at all; record at least
                # which tool degraded and why.
                kb.logger.warning(f"MCP tool {name!r} returned an error: {error_message}")
        kb._log_mcp_call(name, duration_ms, success, error_message, arguments)


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    get_kb()
    resources = []
    for doc in kb.list_documents():
        resources.append(Resource(
            uri=f"c64kb://{doc.doc_id}",
            name=doc.title,
            description=f"{doc.file_type.upper()} document with {doc.total_chunks} chunks",
            mimeType="text/plain"
        ))
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    get_kb()
    if uri.startswith("c64kb://"):
        doc_id = uri[8:]
        content = kb.get_document_content(doc_id)
        if content:
            return content
    return f"Resource not found: {uri}"


# ---------------------------------------------------------------------------
# Transport selection
#
# stdio stays the default and its code path is unchanged: every existing MCP
# client config launches this file as a child process and talks over pipes.
# HTTP is opt-in (--transport http / TDZ_MCP_TRANSPORT=http) so one hosted
# instance can serve clients on other machines without replicating the ~8 GB
# TDZ_DATA_DIR. See issue #16.
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')


def _mcp_api_keys() -> list[str]:
    # Same variable rest_server.py already documents in docs/REST_API.md; a
    # second key scheme for the same knowledge base would be worse than none.
    return [k.strip() for k in os.getenv('TDZ_API_KEYS', '').split(',') if k.strip()]


def _resolve_transport_config(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog='server.py',
        description='TDZ C64 knowledge base MCP server.',
    )
    parser.add_argument(
        '--transport', choices=('stdio', 'http'),
        default=os.getenv('TDZ_MCP_TRANSPORT', 'stdio'),
        help='Transport to serve on (default: stdio).',
    )
    parser.add_argument(
        '--host', default=os.getenv('TDZ_MCP_HOST', '127.0.0.1'),
        help='Bind address for --transport http (default: 127.0.0.1).',
    )
    parser.add_argument(
        '--port', type=int, default=int(os.getenv('TDZ_MCP_PORT', '8765')),
        help='Port for --transport http (default: 8765).',
    )
    return parser.parse_args(argv)


class _ApiKeyMiddleware:
    """Reject unauthenticated requests before they reach the MCP session.

    Raw ASGI rather than a Starlette BaseHTTPMiddleware: the streamable-HTTP
    transport holds long-lived SSE responses open and BaseHTTPMiddleware
    buffers the response body.
    """

    def __init__(self, app, api_keys: list[str]):
        self.app = app
        self.api_keys = api_keys

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not self.api_keys:
            await self.app(scope, receive, send)
            return

        import hmac

        headers = {k.lower(): v for k, v in (scope.get('headers') or [])}
        provided = headers.get(b'x-api-key', b'').decode('latin-1')
        if not provided:
            auth = headers.get(b'authorization', b'').decode('latin-1')
            if auth.lower().startswith('bearer '):
                provided = auth[7:].strip()

        # compare_digest against each configured key: a plain `in` test leaks
        # key length and prefix through timing.
        if not any(hmac.compare_digest(provided, key) for key in self.api_keys):
            body = b'{"error": "Invalid or missing API key"}'
            await send({
                'type': 'http.response.start',
                'status': 401,
                'headers': [
                    (b'content-type', b'application/json'),
                    (b'content-length', str(len(body)).encode()),
                    (b'www-authenticate', b'ApiKey'),
                ],
            })
            await send({'type': 'http.response.body', 'body': body})
            return

        await self.app(scope, receive, send)


def _transport_security_settings(host: str, port: int, logger):
    """DNS-rebinding settings for the streamable-HTTP endpoint.

    The SDK treats "protection enabled with an empty allowed_hosts" as reject
    every request, so this cannot simply be switched on. On a loopback bind
    the allow-list is knowable and the attack is real (a browser on this
    machine resolving an attacker domain to 127.0.0.1), so protection is on.
    On a non-loopback bind the Host header is whatever name the client used -
    a LAN IP, a Tailscale name - which this process cannot enumerate, so
    protection stays off unless TDZ_MCP_ALLOWED_HOSTS names them. API-key
    auth, which the bind guard already makes mandatory off loopback, is the
    control there.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    configured = [h.strip() for h in os.getenv('TDZ_MCP_ALLOWED_HOSTS', '').split(',') if h.strip()]

    if '*' in configured:
        logger.warning('TDZ_MCP_ALLOWED_HOSTS=* - DNS rebinding protection disabled.')
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    if not configured and host not in _LOOPBACK_HOSTS:
        logger.warning(
            'Binding to %s with no TDZ_MCP_ALLOWED_HOSTS - DNS rebinding '
            'protection disabled because the Host header clients will send is '
            'not knowable here. Set TDZ_MCP_ALLOWED_HOSTS to the hostnames '
            'clients use to enable it.', host,
        )
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    allowed = set(configured)
    for candidate in (host, *_LOOPBACK_HOSTS):
        if candidate in ('0.0.0.0', '::'):
            continue
        allowed.add(f'{candidate}:{port}')
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(allowed),
        allowed_origins=sorted(f'http://{a}' for a in allowed),
    )


async def _run_http(host: str, port: int) -> None:
    # Imported here, not at module scope: every MCP session spawns this file
    # and the client allows 30s for the initialize handshake, so nothing that
    # only the HTTP path needs may cost import time on the stdio path.
    # See CLAUDE.md, "MCP startup performance".
    import contextlib

    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    global _tool_call_lock

    logger = logging.getLogger(__name__)
    api_keys = _mcp_api_keys()

    # This transport exposes every tool, including add_document and scrape_url
    # (arbitrary-URL fetch from the host - SSRF). Refuse to serve that to a
    # network with no authentication rather than failing open; mirrors
    # rest_server.py's guard so the two behave the same way.
    if host not in _LOOPBACK_HOSTS and not api_keys:
        if os.getenv('TDZ_MCP_ALLOW_INSECURE', '0') != '1':
            raise SystemExit(
                f'Refusing to bind the MCP HTTP transport to {host} with no API '
                'keys configured (TDZ_API_KEYS is unset). This would expose '
                'unauthenticated read/write access to the knowledge base. Set '
                'TDZ_API_KEYS, or set TDZ_MCP_ALLOW_INSECURE=1 to override.'
            )
        logger.warning('TDZ_MCP_ALLOW_INSECURE=1 - serving %s with no authentication.', host)

    # Several clients now share one KnowledgeBase; see call_tool.
    _tool_call_lock = asyncio.Lock()

    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,
        stateless=False,
        security_settings=_transport_security_settings(host, port, logger),
    )

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with session_manager.run():
            yield

    app = _ApiKeyMiddleware(
        Starlette(routes=[Mount('/mcp', app=handle_mcp)], lifespan=lifespan),
        api_keys,
    )

    logger.info(
        'MCP streamable-HTTP transport on http://%s:%d/mcp (auth: %s)',
        host, port, f'{len(api_keys)} API key(s)' if api_keys else 'none',
    )
    await uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='info')).serve()


async def main(argv=None):
    """Run the MCP server."""
    # Log version information
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"Starting {get_full_version_string()}")
    logger.info(f"Build Date: {__build_date__}")
    logger.info("=" * 60)

    args = _resolve_transport_config(argv)

    # Construct the process-wide KnowledgeBase now, before serving any
    # requests, so every tool handler's bare `kb` reference resolves to the
    # same already-initialized instance for the rest of this process's life.
    get_kb()

    try:
        if args.transport == 'http':
            await _run_http(args.host, args.port)
        else:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options()
                )
    finally:
        # Release the DB connection and worker thread whenever serving stops so
        # this process doesn't linger holding a lock. Under stdio that is the
        # client disconnecting; under HTTP it is the server shutting down, NOT
        # an individual client going away - the transport outlives any one
        # client, and closing per-disconnect would pull the KB out from under
        # everyone still connected.
        if kb is not None:
            kb.close()


def cli_main():
    """Synchronous entry point for the `tdz-c64-knowledge` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
