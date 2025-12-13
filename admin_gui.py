#!/usr/bin/env python3
"""
TDZ C64 Knowledge Base - Admin GUI

Streamlit-based web interface for managing the knowledge base.

Usage:
    streamlit run admin_gui.py

Features:
    - Document management (add/remove/list)
    - Multi-mode search (keyword/semantic/hybrid)
    - Backup & restore operations
    - Analytics dashboard
    - Health monitoring
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase

# Page configuration
st.set_page_config(
    page_title="TDZ C64 Knowledge Base Admin",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'kb' not in st.session_state:
    data_dir = os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge"))
    st.session_state.kb = KnowledgeBase(data_dir)
    st.session_state.data_dir = data_dir

kb = st.session_state.kb

# Sidebar navigation
st.sidebar.title("🎮 C64 Knowledge Base")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📚 Documents", "🔍 Search", "💾 Backup & Restore", "📈 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Directory:**\n`{st.session_state.data_dir}`")

# Helper functions
def format_bytes(bytes):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable string."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

# ========== DASHBOARD PAGE ==========
if page == "📊 Dashboard":
    st.title("📊 Knowledge Base Dashboard")

    # Get statistics
    stats = kb.get_stats()
    health = kb.health_check()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Documents", stats['total_documents'])

    with col2:
        st.metric("Chunks", f"{stats['total_chunks']:,}")

    with col3:
        st.metric("Total Words", f"{stats['total_words']:,}")

    with col4:
        status_color = "🟢" if health['status'] == 'healthy' else "🔴"
        st.metric("Status", f"{status_color} {health['status'].upper()}")

    st.markdown("---")

    # Health information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Database Info")
        if health['database']:
            st.write(f"**Size:** {format_bytes(health['database'].get('size_mb', 0) * 1024 * 1024)}")
            st.write(f"**Integrity:** {health['database'].get('integrity', 'Unknown')}")
            st.write(f"**Free Disk Space:** {health['database'].get('disk_free_gb', 0):.2f} GB")

    with col2:
        st.subheader("⚙️ Features")
        if health['features']:
            for feature, enabled in health['features'].items():
                icon = "✅" if enabled else "❌"
                st.write(f"{icon} {feature.replace('_', ' ').title()}")

    st.markdown("---")

    # File types and tags
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 File Types")
        if stats['file_types']:
            for ftype in stats['file_types']:
                st.write(f"• {ftype}")
        else:
            st.info("No documents yet")

    with col2:
        st.subheader("🏷️ Tags")
        if stats['all_tags']:
            for tag in stats['all_tags']:
                st.write(f"• {tag}")
        else:
            st.info("No tags yet")

    # Issues
    if health.get('issues'):
        st.markdown("---")
        st.error("⚠️ **Issues Detected:**")
        for issue in health['issues']:
            st.write(f"• {issue}")

# ========== DOCUMENTS PAGE ==========
elif page == "📚 Documents":
    st.title("📚 Document Management")

    # Add document section
    with st.expander("➕ Add New Document", expanded=False):
        st.subheader("Upload Document")

        uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=['pdf', 'txt'])

        col1, col2 = st.columns(2)
        with col1:
            doc_title = st.text_input("Title (optional)", "")
        with col2:
            doc_tags = st.text_input("Tags (comma-separated)", "")

        if st.button("Add Document") and uploaded_file:
            try:
                # Save uploaded file temporarily
                temp_path = Path(st.session_state.data_dir) / f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Add to knowledge base
                tags = [t.strip() for t in doc_tags.split(',') if t.strip()]
                doc = kb.add_document(str(temp_path), doc_title or None, tags)

                # Clean up temp file
                temp_path.unlink()

                st.success(f"✅ Document added: {doc.title} ({doc.total_chunks} chunks)")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding document: {str(e)}")

    st.markdown("---")

    # List documents
    st.subheader("📋 Document Library")

    docs = kb.list_documents()

    if not docs:
        st.info("No documents in the knowledge base. Add some using the section above!")
    else:
        # Search/filter
        search_query = st.text_input("🔎 Filter documents", "")

        # Filter documents
        filtered_docs = docs
        if search_query:
            filtered_docs = [d for d in docs if search_query.lower() in d.title.lower() or
                           search_query.lower() in d.filename.lower()]

        st.write(f"Showing {len(filtered_docs)} of {len(docs)} documents")

        # Display documents
        for doc in filtered_docs:
            with st.expander(f"📄 {doc.title}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**File:** {doc.filename}")
                    st.write(f"**Type:** {doc.file_type}")
                    st.write(f"**ID:** `{doc.doc_id}`")
                    if doc.total_pages:
                        st.write(f"**Pages:** {doc.total_pages}")
                    st.write(f"**Chunks:** {doc.total_chunks}")
                    if doc.tags:
                        st.write(f"**Tags:** {', '.join(doc.tags)}")
                    st.write(f"**Indexed:** {format_timestamp(doc.indexed_at)}")

                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_{doc.doc_id}"):
                        if kb.remove_document(doc.doc_id):
                            st.success(f"Deleted: {doc.title}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

# ========== SEARCH PAGE ==========
elif page == "🔍 Search":
    st.title("🔍 Search Knowledge Base")

    # Search mode selection
    search_mode = st.radio(
        "Search Mode",
        ["Keyword (FTS5)", "Semantic", "Hybrid"],
        horizontal=True
    )

    # Search input
    query = st.text_input("Enter your search query:", "")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
    with col2:
        if search_mode == "Hybrid":
            semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.3, 0.1)
    with col3:
        tag_filter = st.text_input("Filter by tags (comma-separated)", "")

    # Search button
    if st.button("🔍 Search") and query:
        tags = [t.strip() for t in tag_filter.split(',') if t.strip()] if tag_filter else None

        try:
            # Perform search based on mode
            if search_mode == "Keyword (FTS5)":
                results = kb.search(query, max_results, tags)
            elif search_mode == "Semantic":
                if not kb.use_semantic:
                    st.error("Semantic search is not enabled. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    results = kb.semantic_search(query, max_results, tags)
            else:  # Hybrid
                if not kb.use_semantic:
                    st.error("Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    results = kb.hybrid_search(query, max_results, tags, semantic_weight)

            # Display results
            if not results:
                st.warning("No results found.")
            else:
                st.success(f"Found {len(results)} results")

                # Export options
                col1, col2 = st.columns([3, 1])
                with col2:
                    export_format = st.selectbox("Export as", ["Markdown", "JSON", "HTML"])
                    if st.button("📤 Export Results"):
                        format_map = {"Markdown": "markdown", "JSON": "json", "HTML": "html"}
                        exported = kb.export_search_results(results, format_map[export_format], query)

                        st.download_button(
                            label=f"Download {export_format}",
                            data=exported,
                            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_map[export_format]}",
                            mime="text/plain"
                        )

                # Display each result
                for i, result in enumerate(results, 1):
                    with st.expander(f"**{i}. {result.get('title', 'Untitled')}**", expanded=(i==1)):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**File:** {result.get('filename', 'Unknown')}")
                            st.write(f"**Doc ID:** `{result.get('doc_id', 'Unknown')}`")
                            if result.get('page'):
                                st.write(f"**Page:** {result['page']}")

                        with col2:
                            score_key = 'score' if 'score' in result else 'similarity'
                            if score_key in result:
                                st.metric("Score", f"{result[score_key]:.4f}")

                        # Snippet
                        if 'snippet' in result:
                            st.markdown("**Excerpt:**")
                            st.markdown(result['snippet'])

                        # Tags
                        if result.get('tags'):
                            st.write(f"🏷️ {', '.join(result['tags'])}")

        except Exception as e:
            st.error(f"Search error: {str(e)}")

# ========== BACKUP & RESTORE PAGE ==========
elif page == "💾 Backup & Restore":
    st.title("💾 Backup & Restore")

    col1, col2 = st.columns(2)

    # Create backup
    with col1:
        st.subheader("📦 Create Backup")

        backup_dir = st.text_input("Backup Directory", value=str(Path.home() / "c64kb_backups"))
        compress = st.checkbox("Compress to ZIP", value=True)

        if st.button("🔄 Create Backup"):
            try:
                with st.spinner("Creating backup..."):
                    backup_path = kb.create_backup(backup_dir, compress)
                    st.success(f"✅ Backup created successfully!\n\n**Location:** `{backup_path}`")
            except Exception as e:
                st.error(f"Backup failed: {str(e)}")

    # Restore backup
    with col2:
        st.subheader("♻️ Restore Backup")

        st.warning("⚠️ **Warning:** Restoring will replace the current database. A safety backup will be created automatically.")

        restore_path = st.text_input("Backup Path (file or directory)")
        verify = st.checkbox("Verify backup before restoring", value=True)

        if st.button("⚠️ Restore Backup", type="primary"):
            if not restore_path:
                st.error("Please provide a backup path")
            else:
                try:
                    with st.spinner("Restoring backup..."):
                        result = kb.restore_from_backup(restore_path, verify)
                        st.success(f"✅ Restore completed successfully!\n\n"
                                 f"**Documents restored:** {result['restored_documents']}\n"
                                 f"**Time:** {result['elapsed_seconds']:.2f}s")
                        st.rerun()
                except Exception as e:
                    st.error(f"Restore failed: {str(e)}")

# ========== ANALYTICS PAGE ==========
elif page == "📈 Analytics":
    st.title("📈 Search Analytics")

    # Time range selection
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)

    if st.button("📊 Generate Report"):
        try:
            analytics = kb.get_search_analytics(days=days, limit=100)

            if 'error' in analytics:
                st.error(f"Error: {analytics['error']}")
            else:
                # Overview metrics
                st.subheader("📊 Overview")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Searches", f"{analytics.get('total_searches', 0):,}")
                with col2:
                    st.metric("Unique Queries", f"{analytics.get('unique_queries', 0):,}")
                with col3:
                    st.metric("Avg Results", f"{analytics.get('avg_results', 0):.1f}")
                with col4:
                    st.metric("Avg Time", f"{analytics.get('avg_execution_time_ms', 0):.1f}ms")

                st.markdown("---")

                # Two columns for charts
                col1, col2 = st.columns(2)

                with col1:
                    # Top queries
                    st.subheader("🔝 Top Queries")
                    if analytics.get('top_queries'):
                        df = pd.DataFrame(analytics['top_queries'][:10])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No query data available")

                with col2:
                    # Search modes
                    st.subheader("🔍 Search Mode Usage")
                    if analytics.get('search_modes'):
                        df = pd.DataFrame(analytics['search_modes'])
                        st.bar_chart(df.set_index('mode')['count'])
                    else:
                        st.info("No search mode data available")

                st.markdown("---")

                # Failed searches
                st.subheader("❌ Failed Searches (0 results)")
                if analytics.get('failed_searches'):
                    df = pd.DataFrame(analytics['failed_searches'][:10])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No failed searches")

                # Popular tags
                if analytics.get('popular_tags'):
                    st.markdown("---")
                    st.subheader("🏷️ Popular Tags")
                    df = pd.DataFrame(analytics['popular_tags'][:10])
                    st.bar_chart(df.set_index('tag')['count'])

        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**TDZ C64 Knowledge Base v2.5.0**")
st.sidebar.markdown("Built with ❤️ using Claude Code")
