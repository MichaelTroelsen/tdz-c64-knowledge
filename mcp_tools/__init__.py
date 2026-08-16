"""The MCP protocol layer: tool schemas and their handlers.

Split out of server.py by R12. Nothing in here imports server, so the
KnowledgeBase is always passed in by the caller.
"""
from .handlers import HANDLERS
from .schemas import TOOL_SCHEMAS

__all__ = ["HANDLERS", "TOOL_SCHEMAS"]
