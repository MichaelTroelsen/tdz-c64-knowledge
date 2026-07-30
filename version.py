#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - Version Information

This file contains version and build information for the project.
"""

# Version number follows Semantic Versioning (MAJOR.MINOR.PATCH)
# MAJOR: Incompatible API changes
# MINOR: Add functionality in a backwards compatible manner
# PATCH: Backwards compatible bug fixes

__version__ = "2.24.0"
__version_info__ = (2, 24, 0)

# Build information
__build_date__ = "2026-01-10"
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
    "entity_analytics_dashboard": "2.17.0",
    "document_comparison": "2.17.0",
    "entity_export": "2.17.0",
    "relationship_export": "2.17.0",
    "frame_detection": "2.17.1",
    "automatic_frame_scraping": "2.17.1",
    "rest_api": "2.18.0",
    "file_upload_api": "2.18.0",
    "export_api": "2.18.0",
    "lazy_loading_embeddings": "2.19.0",
    "performance_optimizations_phase2": "2.19.0",
    "instant_startup": "2.19.0",
    "enhanced_url_update_checking": "2.20.0",
    "url_structure_discovery": "2.20.0",
    "new_page_detection": "2.20.0",
    "missing_page_detection": "2.20.0",
    "project_directory_security_fix": "2.20.0",
    "c64_specific_entity_patterns": "2.22.0",
    "entity_normalization": "2.22.0",
    "entity_source_tracking": "2.22.0",
    "distance_based_relationship_strength": "2.22.0",
    "comprehensive_performance_benchmarking": "2.22.0",
    "load_testing_infrastructure": "2.22.0",
    "rag_question_answering": "2.23.0",
    "fuzzy_search": "2.23.0",
    "progressive_search_refinement": "2.23.0",
    "smart_document_tagging": "2.23.0",
}



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
