#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - Version Information

This file contains version and build information for the project.
"""

# Version number follows Semantic Versioning (MAJOR.MINOR.PATCH)
# MAJOR: Incompatible API changes
# MINOR: Add functionality in a backwards compatible manner
# PATCH: Backwards compatible bug fixes

__version__ = "2.18.0"
__version_info__ = (2, 18, 0)

# Build information
__build_date__ = "2025-12-21"
__author__ = "TDZ Development Team"
__project_name__ = "TDZ C64 Knowledge Base"
__description__ = "MCP server for managing and searching Commodore 64 documentation"

# Feature version tracking
FEATURES = {
    "mcp_server": "2.0.0",
    "semantic_search": "2.0.0",
    "hybrid_search": "2.0.0",
    "fts5_search": "2.0.0",
    "table_extraction": "2.1.0",
    "code_block_detection": "2.1.0",
    "html_support": "2.10.0",
    "excel_support": "2.9.0",
    "gui_file_path_input": "2.11.0",
    "gui_duplicate_detection": "2.11.0",
    "gui_file_viewer": "2.11.0",
    "smart_auto_tagging": "2.12.0",
    "llm_integration": "2.12.0",
    "document_summarization": "2.13.0",
    "ai_summary_caching": "2.13.0",
    "url_scraping": "2.14.0",
    "web_content_ingestion": "2.14.0",
    "mdscrape_integration": "2.14.0",
    "loading_indicators": "2.14.0",
    "dotenv_configuration": "2.14.0",
    "entity_extraction": "2.15.0",
    "entity_relationships": "2.16.0",
    "nl_query_translation": "2.17.0",
    "rest_api_server": "2.18.0",
    "api_authentication": "2.18.0",
    "fastapi_integration": "2.18.0",
    "background_entity_extraction": "2.18.0",
    "entity_analytics_dashboard": "2.18.0",
    "performance_optimizations": "2.18.0",
    "query_embedding_cache": "2.18.0",
    "parallel_hybrid_search": "2.18.0",
}

# Version history
VERSION_HISTORY = """
v2.18.0 (2025-12-21)
  🚀 MAJOR RELEASE: Background Entity Extraction + Performance Optimizations + Analytics Dashboard

  Background Entity Extraction (Phase 2):
  - Zero-delay asynchronous entity extraction with background worker thread
  - Auto-queue on document ingestion (configurable via AUTO_EXTRACT_ENTITIES=1)
  - extraction_jobs table for full job tracking (queued/running/completed/failed)
  - 3 new methods: queue_entity_extraction(), get_extraction_status(), get_all_extraction_jobs()
  - 3 new MCP tools: queue_entity_extraction, get_extraction_status, get_extraction_jobs
  - Users never wait for LLM extraction (previously 3-30 seconds)

  Entity Analytics Dashboard (Sprint 2):
  - get_entity_analytics() method with 6 comprehensive data structures
  - 4-tab interactive GUI: Overview, Top Entities, Relationships, Trends
  - Interactive network graph with pyvis (drag-and-drop, color-coded, 7-type legend)
  - Export buttons for CSV/JSON downloads
  - Real-time stats: 989 unique entities, 128 relationships

  Performance Optimizations (Phase 1):
  - Semantic search 43% faster (14.53ms → 8.31ms) via query embedding cache
  - Hybrid search 22% faster (19.44ms → 15.24ms) via parallel execution
  - Entity extraction 4x faster for cached calls (0.12ms → 0.03ms)
  - Overall 8% faster benchmark time (6.27s → 5.75s)
  - Memory impact: ~6.5MB for all caches
  - PERFORMANCE_IMPROVEMENTS.md with detailed analysis

  REST API Server:
  - FastAPI-based HTTP/REST interface with 27 endpoints
  - API key authentication, CORS middleware
  - OpenAPI/Swagger docs at /api/docs
  - 6 endpoint categories: Health, Search, Documents, URL Scraping, AI, Analytics
  - Complete Pydantic v2 validation
  - README_REST_API.md documentation

  New Environment Variables:
  - AUTO_EXTRACT_ENTITIES=1 (default: enabled)
  - EMBEDDING_CACHE_TTL=3600 (1 hour)
  - ENTITY_CACHE_TTL=86400 (24 hours)

v2.17.0 (2025-12-21)
  - Added Natural Language Query Translation (Sprint 1: Quick Wins)
  - AI-powered query parsing with entity extraction
  - Dual extraction: Regex patterns for C64-specific hardware + LLM for contextual entities
  - Core method: translate_nl_query() with confidence scoring
  - MCP tool: translate_query
  - CLI command: translate-query with formatted output
  - GUI integration: Search page with NL translation toggle and results display
  - Automatic search mode recommendation (keyword/semantic/hybrid)
  - Facet filter generation from detected entities
  - Graceful fallback when LLM unavailable

v2.16.0 (2025-12-21)
  - Added Entity Relationship Tracking
  - Track co-occurrence of entities within documents
  - Database schema: entity_relationships table with 4 indexes
  - Core methods: extract_entity_relationships(), get_entity_relationships(), find_related_entities(), search_by_entity_pair(), extract_relationships_bulk()
  - MCP tools: extract_entity_relationships, get_entity_relationships, find_related_entities, search_entity_pair
  - CLI commands: extract-relationships, extract-all-relationships, show-relationships, search-pair
  - GUI: 4-tab Entity Relationships interface
  - Relationship strength scoring (0.0-1.0) based on co-occurrence frequency
  - Context extraction for relationship examples
  - Incremental updates across multiple documents

v2.15.0 (2025-12-20)
  - Added AI-Powered Named Entity Extraction
  - 7 entity types: hardware, memory_address, instruction, person, company, product, concept
  - Database schema: document_entities table with FTS5 search
  - Core methods: extract_entities(), get_entities(), search_entities(), find_docs_by_entity(), get_entity_stats(), extract_entities_bulk()
  - MCP tools: extract_entities, list_entities, search_entities, entity_stats, extract_entities_bulk
  - CLI commands: extract-entities, extract-all-entities, search-entity, entity-stats
  - Confidence scoring and occurrence counting
  - Full-text search across all entities with filtering

v2.14.0 (2025-12-18)
  - Added URL Scraping & Web Content Ingestion (mdscrape integration)
  - New MCP tools: scrape_url, rescrape_document, check_url_updates
  - Concurrent scraping with configurable threads and depth control
  - Automatic content-based update detection
  - UI/UX improvements: centered loading indicators, progress bars
  - python-dotenv integration for automatic .env configuration
  - Bug fixes: preview slider, warning suppression, security paths
  - Comprehensive test suite for path security validation

v2.13.0 (2025-12-17)
  - Added AI-Powered Document Summarization (Phase 1.2)
  - Three summary types: brief, detailed, bullet-point
  - Intelligent caching with database storage
  - New MCP tools: summarize_document, get_summary, summarize_all
  - New CLI commands: summarize, summarize-all
  - Comprehensive 400+ line feature guide (SUMMARIZATION.md)
  - Works with Anthropic Claude and OpenAI GPT models
  - Bulk summarization for entire knowledge base

v2.12.0 (2025-12-13)
  - Added Smart Auto-Tagging with LLM integration
  - Supports Anthropic Claude and OpenAI GPT models
  - Confidence-based tag filtering and recommendations
  - Bulk auto-tagging for all documents
  - New MCP tools: auto_tag_document, auto_tag_all

v2.11.0 (2025-12-13)
  - Added file path input in GUI (no need for upload)
  - Added duplicate detection with user notifications
  - Enhanced file viewer for MD/TXT files with rendering
  - Improved progress indicators and status messages

v2.10.0 (2024-XX-XX)
  - Added HTML file support (.html, .htm)

v2.9.0 (2024-XX-XX)
  - Added Excel file support (.xlsx, .xls)
  - Enhanced Markdown visibility

v2.1.0 (2024-XX-XX)
  - Added table extraction from PDFs
  - Added code block detection (BASIC/Assembly/Hex)

v2.0.0 (2024-XX-XX)
  - Hybrid search (FTS5 + semantic)
  - Enhanced snippet extraction
  - Health monitoring system
  - SQLite FTS5 full-text search
  - Semantic search with embeddings
"""


def get_version():
    """Get version string."""
    return __version__


def get_version_info():
    """Get version as tuple."""
    return __version_info__


def get_full_version_string():
    """Get full version string with project name."""
    return f"{__project_name__} v{__version__}"


def get_version_dict():
    """Get version information as dictionary."""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build_date": __build_date__,
        "project_name": __project_name__,
        "description": __description__,
        "author": __author__,
        "features": FEATURES,
    }


def print_version_info():
    """Print version information to console."""
    print("=" * 60)
    print(f"{__project_name__}")
    print(f"Version: {__version__}")
    print(f"Build Date: {__build_date__}")
    print(f"Author: {__author__}")
    print("=" * 60)
    print(f"{__description__}")
    print("=" * 60)


if __name__ == "__main__":
    print_version_info()
    print("\nFeatures:")
    for feature, version in FEATURES.items():
        print(f"  - {feature}: {version}")
