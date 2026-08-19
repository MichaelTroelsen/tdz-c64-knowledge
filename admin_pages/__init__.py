"""Per-page modules for the admin GUI (R18).

admin_gui.py keeps its sidebar radio and looks the selected label up here,
so navigation behaves exactly as before; only the 5,000 lines of page bodies
moved. PAGES maps the radio label to that page's render(kb).
"""
from .dashboard import render as _dashboard
from .documents import render as _documents
from .web_scraping import render as _web_scraping
from .url_monitoring import render as _url_monitoring
from .tag_management import render as _tag_management
from .entity_extraction import render as _entity_extraction
from .relationship_graph import render as _relationship_graph
from .search import render as _search
from .backup_restore import render as _backup_restore
from .entity_analytics import render as _entity_analytics
from .graph_explorer import render as _graph_explorer
from .mcp_monitor import render as _mcp_monitor
from .document_comparison import render as _document_comparison
from .system_analytics import render as _system_analytics
from .archive_search import render as _archive_search
from .settings import render as _settings

PAGES = {
    "📊 Dashboard": _dashboard,
    "📚 Documents": _documents,
    "🌐 Web Scraping": _web_scraping,
    "🌐 URL Monitoring": _url_monitoring,
    "🏷️ Tag Management": _tag_management,
    "🧠 Entity Extraction": _entity_extraction,
    "🔗 Relationship Graph": _relationship_graph,
    "🔍 Search": _search,
    "💾 Backup & Restore": _backup_restore,
    "📈 Entity Analytics": _entity_analytics,
    "🕸️ Graph Explorer": _graph_explorer,
    "🛰️ MCP Monitor": _mcp_monitor,
    "📄 Document Comparison": _document_comparison,
    "📉 System Analytics": _system_analytics,
    "🔍 Archive Search": _archive_search,
    "⚙️ Settings": _settings,
}

__all__ = ["PAGES"]
