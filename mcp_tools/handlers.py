"""Aggregates the per-domain MCP tool handlers into a single HANDLERS dict.

Originally this module held all 92 `handle_<tool>` functions inline (one
function per branch of server.py's 3,400-line `elif name == ...` chain,
R12). It grew to 3,624 lines, so the functions were moved out into
per-domain modules (search, documents, admin, entities, figures,
knowledge_graph, topics, temporal) - a move, not a rewrite, so each
function's body is still exactly the original branch body, dedented one
level and otherwise unchanged.

This module is now just the aggregation point: `from mcp_tools.handlers
import HANDLERS` (and `from mcp_tools import HANDLERS`, via __init__.py)
keep working unchanged. The per-domain HANDLERS_* dicts are merged here,
with an explicit duplicate-key check - a duplicate tool name defined in
two modules would otherwise silently shadow one of them.
"""
from .admin import HANDLERS_ADMIN
from .documents import HANDLERS_DOCUMENTS
from .entities import HANDLERS_ENTITIES
from .figures import HANDLERS_FIGURES
from .knowledge_graph import HANDLERS_KNOWLEDGE_GRAPH
from .search import HANDLERS_SEARCH
from .temporal import HANDLERS_TEMPORAL
from .topics import HANDLERS_TOPICS

_DOMAIN_HANDLER_DICTS = [
    HANDLERS_SEARCH,
    HANDLERS_DOCUMENTS,
    HANDLERS_ADMIN,
    HANDLERS_ENTITIES,
    HANDLERS_FIGURES,
    HANDLERS_KNOWLEDGE_GRAPH,
    HANDLERS_TOPICS,
    HANDLERS_TEMPORAL,
]

_seen: dict = {}
for _d in _DOMAIN_HANDLER_DICTS:
    for _name in _d:
        if _name in _seen:
            raise RuntimeError(
                f"duplicate tool handler {_name!r} defined in more than one "
                "mcp_tools domain module - this would silently shadow one "
                "of them"
            )
        _seen[_name] = _d[_name]

HANDLERS = _seen
