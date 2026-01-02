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
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import threading
import time

# Suppress Streamlit ScriptRunContext warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# Add parent directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase
from version import __build_date__, get_full_version_string

# Page configuration
st.set_page_config(
    page_title="TDZ C64 Knowledge Base Admin",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background indexing function
def build_bm25_index_background(kb):
    """Build BM25 index in background thread with progress tracking."""
    import warnings
    # Suppress Streamlit ScriptRunContext warnings for background threads
    warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

    try:
        st.session_state.index_status = "building"
        st.session_state.index_start_time = time.time()
        st.session_state.index_progress = 0.0

        # Check if BM25 is enabled via environment variable
        use_bm25 = os.environ.get("USE_BM25", "1") != "0"

        if kb.bm25 is None and use_bm25:
            # Start a progress updater thread
            import threading

            def update_progress():
                """Simulate progress updates during index building."""
                import warnings
                # Suppress Streamlit ScriptRunContext warnings for background threads
                warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

                start = time.time()
                estimated_time = 60  # Estimated 60 seconds based on previous runs

                while st.session_state.get('index_status') == 'building':
                    if st.session_state.get('index_status') == 'cancelled':
                        return

                    elapsed = time.time() - start
                    progress = min(0.95, elapsed / estimated_time)  # Cap at 95% until actually done
                    st.session_state.index_progress = progress
                    time.sleep(0.5)  # Update every 500ms

            # Start progress updater
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()

            # Build the index
            kb._build_bm25_index()

            # Check if cancelled
            if st.session_state.get('index_status') == 'cancelled':
                st.session_state.index_progress = 0.0
                return

        # Complete
        st.session_state.index_progress = 1.0
        elapsed = time.time() - st.session_state.index_start_time
        st.session_state.index_status = "ready"
        st.session_state.index_build_time = elapsed
    except Exception as e:
        st.session_state.index_status = "error"
        st.session_state.index_error = str(e)
        st.session_state.index_progress = 0.0

def trigger_background_reindex():
    """Trigger background reindexing after documents are added/removed."""
    use_bm25 = os.environ.get("USE_BM25", "1") != "0"

    # Only reindex if BM25 is enabled and index was invalidated
    if use_bm25 and st.session_state.kb.bm25 is None:
        # Kill any existing index thread
        if st.session_state.get('index_thread') and st.session_state.index_thread.is_alive():
            # Thread will finish naturally, just start a new one
            pass

        # Start new background indexing thread
        st.session_state.index_status = "starting"
        st.session_state.index_thread = threading.Thread(
            target=build_bm25_index_background,
            args=(st.session_state.kb,),
            daemon=True
        )
        st.session_state.index_thread.start()

# Initialize session state
if 'kb' not in st.session_state:
    data_dir = os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge"))
    st.session_state.kb = KnowledgeBase(data_dir)
    st.session_state.data_dir = data_dir
    st.session_state.index_status = "not_started"
    st.session_state.index_thread = None

    # Start background indexing if BM25 is enabled and index not built
    use_bm25 = os.environ.get("USE_BM25", "1") != "0"
    if use_bm25 and st.session_state.kb.bm25 is None:
        st.session_state.index_status = "starting"
        st.session_state.index_thread = threading.Thread(
            target=build_bm25_index_background,
            args=(st.session_state.kb,),
            daemon=True
        )
        st.session_state.index_thread.start()

kb = st.session_state.kb

# Sidebar navigation
st.sidebar.title("🎮 C64 Knowledge Base")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📚 Documents", "🌐 Web Scraping", "🌐 URL Monitoring", "🏷️ Tag Management", "🧠 Entity Extraction", "🔗 Relationship Graph", "📈 Entity Analytics", "📄 Document Comparison", "🔍 Search", "💾 Backup & Restore", "📉 System Analytics", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Data Directory:**\n`{st.session_state.data_dir}`")

# URL Update Checker
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 URL Update Check")

url_docs = [d for d in kb.documents.values() if d.source_url]
if url_docs:
    st.sidebar.write(f"📊 {len(url_docs)} URL-sourced documents")

    if st.sidebar.button("Check for Updates"):
        with st.spinner("Checking URLs..."):
            try:
                results = kb.check_url_updates(auto_rescrape=False)

                if results['changed']:
                    st.sidebar.warning(f"⚠️ {len(results['changed'])} documents have updates available")
                    for doc in results['changed'][:3]:  # Show first 3
                        st.sidebar.write(f"- {doc['title']}")
                else:
                    st.sidebar.success("✅ All documents up to date")

                if results['failed']:
                    st.sidebar.error(f"❌ {len(results['failed'])} checks failed")

            except Exception as e:
                st.sidebar.error(f"Error checking updates: {str(e)}")
else:
    st.sidebar.info("No URL-sourced documents")

# Show index building status in sidebar
if st.session_state.get('index_status') == 'building':
    st.sidebar.warning("⏳ **Building search index...**\nFirst search will be ready soon.")
elif st.session_state.get('index_status') == 'ready':
    if 'index_build_time' in st.session_state:
        st.sidebar.success(f"✅ **Search index ready**\n(Built in {st.session_state.index_build_time:.1f}s)")
elif st.session_state.get('index_status') == 'error':
    st.sidebar.error(f"❌ **Index error:**\n{st.session_state.get('index_error', 'Unknown')}")

# Show centered loading indicator for index building
if st.session_state.get('index_status') == 'building':
    # Center the content
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
            <style>
            .loading-container {
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                text-align: center;
            }
            .loading-text {
                color: white;
                font-size: 1.5rem;
                font-weight: bold;
                margin-top: 1rem;
            }
            .loading-subtext {
                color: rgba(255,255,255,0.9);
                font-size: 1rem;
                margin-top: 0.5rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .spinner {
                font-size: 3rem;
                animation: spin 1s linear infinite;
            }
            </style>
            <div class="loading-container">
                <div class="spinner">🏃</div>
                <div class="loading-text">Building Search Index</div>
                <div class="loading-subtext">Optimizing search performance with parallel processing...</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Progress bar
        progress = st.session_state.get('index_progress', 0)
        st.progress(progress, text=f"Progress: {int(progress * 100)}%")

        # Stop button
        if st.button("⏹️ Stop", key="stop_index_building", use_container_width=True):
            st.session_state['index_status'] = 'cancelled'
            st.warning("⚠️ Index building cancelled. Search may be slower.")
            st.rerun()

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
    except (ValueError, TypeError, AttributeError):
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
    with st.expander("➕ Add Documents", expanded=False):
        upload_tabs = st.tabs(["📄 Single Upload", "📦 Bulk Upload", "📁 Add by File Path"])

        # Tab 1: Single Upload
        with upload_tabs[0]:
            st.subheader("Upload Single Document")

            uploaded_file = st.file_uploader("Choose a file (PDF, TXT, MD, HTML, or Excel)", type=['pdf', 'txt', 'md', 'html', 'htm', 'xlsx', 'xls'], key="single_upload")

            col1, col2 = st.columns(2)
            with col1:
                doc_title = st.text_input("Title (optional)", "", key="single_title")
            with col2:
                doc_tags = st.text_input("Tags (comma-separated)", "", key="single_tags")

            if st.button("Add Document", key="add_single") and uploaded_file:
                # Show progress
                with st.spinner(f"⏳ Adding document: {uploaded_file.name}..."):
                    try:
                        # Create uploads directory in project directory
                        project_dir = Path(__file__).parent
                        uploads_dir = project_dir / "uploads"
                        uploads_dir.mkdir(exist_ok=True)

                        # Save uploaded file permanently
                        permanent_path = uploads_dir / uploaded_file.name

                        # Handle duplicate filenames by appending a number
                        counter = 1
                        while permanent_path.exists():
                            name_parts = uploaded_file.name.rsplit('.', 1)
                            if len(name_parts) == 2:
                                permanent_path = uploads_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                            else:
                                permanent_path = uploads_dir / f"{uploaded_file.name}_{counter}"
                            counter += 1

                        with open(permanent_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())

                        # Get current number of documents
                        initial_doc_count = len(kb.documents)

                        # Add to knowledge base
                        tags = [t.strip() for t in doc_tags.split(',') if t.strip()]
                        doc = kb.add_document(str(permanent_path), doc_title or None, tags)

                        # Check if this was a duplicate (doc count didn't increase)
                        final_doc_count = len(kb.documents)
                        is_duplicate = (final_doc_count == initial_doc_count)

                        if is_duplicate:
                            st.warning(f"⚠️ **Document already exists in knowledge base**\n\n"
                                     f"**Title:** {doc.title}\n\n"
                                     f"**Document ID:** `{doc.doc_id}`\n\n"
                                     f"**Chunks:** {doc.total_chunks}\n\n"
                                     f"This file (or a file with identical content) has already been indexed.")
                        else:
                            # Trigger background reindexing
                            trigger_background_reindex()

                            st.success(f"✅ **Document added successfully!**\n\n"
                                     f"**Title:** {doc.title}\n\n"
                                     f"**Chunks:** {doc.total_chunks}\n\n"
                                     f"**Document ID:** `{doc.doc_id}`")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ **Error adding document:**\n\n{str(e)}")

        # Tab 2: Bulk Upload
        with upload_tabs[1]:
            st.subheader("Upload Multiple Documents")
            st.write("📌 **Drag and drop multiple files or click to browse**")

            uploaded_files = st.file_uploader(
                "Choose files (PDF, TXT, MD, HTML, or Excel)",
                type=['pdf', 'txt', 'md', 'html', 'htm', 'xlsx', 'xls'],
                accept_multiple_files=True,
                key="bulk_upload"
            )

            bulk_tags = st.text_input("Tags for all documents (comma-separated)", "", key="bulk_tags")

            if st.button("📦 Add All Documents", key="add_bulk") and uploaded_files:
                tags = [t.strip() for t in bulk_tags.split(',') if t.strip()]

                # Create uploads directory in project directory
                project_dir = Path(__file__).parent
                uploads_dir = project_dir / "uploads"
                uploads_dir.mkdir(exist_ok=True)

                progress_bar = st.progress(0)
                status_text = st.empty()

                added = 0
                duplicates = 0
                failed = 0
                duplicate_files = []

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"⏳ Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

                        # Save uploaded file permanently
                        permanent_path = uploads_dir / uploaded_file.name

                        # Handle duplicate filenames by appending a number
                        counter = 1
                        while permanent_path.exists():
                            name_parts = uploaded_file.name.rsplit('.', 1)
                            if len(name_parts) == 2:
                                permanent_path = uploads_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
                            else:
                                permanent_path = uploads_dir / f"{uploaded_file.name}_{counter}"
                            counter += 1

                        with open(permanent_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())

                        # Get current number of documents
                        initial_doc_count = len(kb.documents)

                        # Add to knowledge base (use filename as title)
                        doc = kb.add_document(str(permanent_path), None, tags)

                        # Check if this was a duplicate
                        final_doc_count = len(kb.documents)
                        is_duplicate = (final_doc_count == initial_doc_count)

                        if is_duplicate:
                            duplicates += 1
                            duplicate_files.append(uploaded_file.name)
                        else:
                            added += 1
                    except Exception as e:
                        failed += 1
                        st.warning(f"⚠️ Failed to add {uploaded_file.name}: {str(e)}")

                progress_bar.empty()
                status_text.empty()

                # Trigger background reindexing if any documents were added
                if added > 0:
                    trigger_background_reindex()

                # Show results
                if added > 0:
                    st.success(f"✅ Successfully added {added} new document(s)")
                if duplicates > 0:
                    st.warning(f"⚠️ Skipped {duplicates} duplicate document(s): {', '.join(duplicate_files)}")
                if failed > 0:
                    st.error(f"❌ Failed to add {failed} document(s)")

                if added > 0:
                    st.rerun()

        # Tab 3: Add by File Path
        with upload_tabs[2]:
            st.subheader("Add Document by File Path")
            st.write("📁 **Enter the full path to a file on your system**")

            file_path_input = st.text_input(
                "File Path",
                placeholder="C:\\Users\\username\\Documents\\file.pdf",
                key="file_path_input"
            )

            col1, col2 = st.columns(2)
            with col1:
                path_doc_title = st.text_input("Title (optional)", "", key="path_title")
            with col2:
                path_doc_tags = st.text_input("Tags (comma-separated)", "", key="path_tags")

            if st.button("Add Document from Path", key="add_path"):
                if not file_path_input:
                    st.error("❌ Please enter a file path")
                elif not os.path.exists(file_path_input):
                    st.error(f"❌ File not found: {file_path_input}")
                else:
                    # Show progress
                    with st.spinner(f"⏳ Adding document: {os.path.basename(file_path_input)}..."):
                        try:
                            # Prepare tags
                            tags = [t.strip() for t in path_doc_tags.split(',') if t.strip()]

                            # Get current number of documents
                            initial_doc_count = len(kb.documents)

                            # Add to knowledge base
                            doc = kb.add_document(file_path_input, path_doc_title or None, tags)

                            # Check if this was a duplicate (doc count didn't increase)
                            final_doc_count = len(kb.documents)
                            is_duplicate = (final_doc_count == initial_doc_count)

                            if is_duplicate:
                                st.warning(f"⚠️ **Document already exists in knowledge base**\n\n"
                                         f"**Title:** {doc.title}\n\n"
                                         f"**Document ID:** `{doc.doc_id}`\n\n"
                                         f"**Chunks:** {doc.total_chunks}\n\n"
                                         f"This file (or a file with identical content) has already been indexed.")
                            else:
                                # Trigger background reindexing
                                trigger_background_reindex()

                                st.success(f"✅ **Document added successfully!**\n\n"
                                         f"**Title:** {doc.title}\n\n"
                                         f"**Chunks:** {doc.total_chunks}\n\n"
                                         f"**Document ID:** `{doc.doc_id}`")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ **Error adding document:**\n\n{str(e)}")

    st.markdown("---")

    # List documents
    st.subheader("📋 Document Library")

    docs = kb.list_documents()

    if not docs:
        st.info("No documents in the knowledge base. Add some using the section above!")
    else:
        # Search/filter and sort controls
        col1, col2 = st.columns([3, 1])

        with col1:
            search_query = st.text_input("🔎 Filter documents", "")

        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ["Date Added (Newest)", "Date Added (Oldest)", "Title (A-Z)", "Title (Z-A)", "File Type"],
                key="doc_sort"
            )

        # Filter documents
        filtered_docs = docs
        if search_query:
            filtered_docs = [d for d in docs if search_query.lower() in d.title.lower() or
                           search_query.lower() in d.filename.lower()]

        # Sort documents
        if sort_by == "Date Added (Newest)":
            filtered_docs = sorted(filtered_docs, key=lambda d: d.indexed_at, reverse=True)
        elif sort_by == "Date Added (Oldest)":
            filtered_docs = sorted(filtered_docs, key=lambda d: d.indexed_at)
        elif sort_by == "Title (A-Z)":
            filtered_docs = sorted(filtered_docs, key=lambda d: d.title.lower())
        elif sort_by == "Title (Z-A)":
            filtered_docs = sorted(filtered_docs, key=lambda d: d.title.lower(), reverse=True)
        elif sort_by == "File Type":
            filtered_docs = sorted(filtered_docs, key=lambda d: (d.file_type, d.title.lower()))

        st.write(f"Showing {len(filtered_docs)} of {len(docs)} documents")

        # Display documents
        for doc in filtered_docs:
            # Format indexed date for display
            indexed_date = format_timestamp(doc.indexed_at) if doc.indexed_at else "N/A"
            with st.expander(f"📄 {doc.title} • 📅 {indexed_date}", expanded=False):
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

                    # Show URL metadata for scraped documents
                    if doc.source_url:
                        st.write(f"**Source URL:** {doc.source_url}")
                        if doc.scrape_date:
                            st.write(f"**Scraped:** {format_timestamp(doc.scrape_date)}")
                        if doc.scrape_status:
                            status_emoji = "✅" if doc.scrape_status == "success" else "⚠️"
                            st.write(f"**Scrape Status:** {status_emoji} {doc.scrape_status}")

                with col2:
                    if st.button("👁️ Preview", key=f"preview_{doc.doc_id}"):
                        st.session_state[f"show_preview_{doc.doc_id}"] = not st.session_state.get(f"show_preview_{doc.doc_id}", False)
                        st.rerun()

                    if st.button("🔗 Relationships", key=f"rels_{doc.doc_id}"):
                        st.session_state[f"show_relationships_{doc.doc_id}"] = not st.session_state.get(f"show_relationships_{doc.doc_id}", False)
                        st.rerun()

                    # Show re-scrape button for URL-sourced documents
                    if doc.source_url:
                        if st.button("🔄 Re-scrape", key=f"rescrape_{doc.doc_id}"):
                            with st.spinner(f"Re-scraping {doc.source_url}..."):
                                try:
                                    result = kb.rescrape_document(doc.doc_id)
                                    if result['status'] == 'success':
                                        trigger_background_reindex()
                                        st.success(f"✅ Re-scraped successfully! Added {result['docs_added']} documents")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Re-scrape failed: {result.get('error', 'Unknown')}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")

                    if st.button("🗑️ Delete", key=f"del_{doc.doc_id}"):
                        if kb.remove_document(doc.doc_id):
                            # Trigger background reindexing
                            trigger_background_reindex()
                            st.success(f"Deleted: {doc.title}")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

                # Show preview if enabled
                if st.session_state.get(f"show_preview_{doc.doc_id}", False):
                    st.markdown("---")
                    st.subheader("📖 Document Preview")

                    # Get document content
                    try:
                        full_doc = kb.get_document(doc.doc_id)

                        if full_doc and 'chunks' in full_doc:
                            chunk_count = len(full_doc['chunks'])

                            # Show preview options
                            preview_col1, preview_col2 = st.columns([2, 1])

                            with preview_col1:
                                # Only show slider if there's more than 1 chunk
                                if chunk_count > 1:
                                    preview_chunks = st.slider(
                                        "Number of chunks to preview",
                                        min_value=1,
                                        max_value=min(chunk_count, 10),
                                        value=min(3, chunk_count),
                                        key=f"preview_slider_{doc.doc_id}"
                                    )
                                else:
                                    # Single chunk - no slider needed
                                    preview_chunks = 1
                                    st.info("📄 Single chunk document")

                            with preview_col2:
                                show_metadata = st.checkbox("Show metadata", value=False, key=f"meta_{doc.doc_id}")

                            st.markdown("---")

                            # Display selected chunks
                            for i, chunk in enumerate(full_doc['chunks'][:preview_chunks]):
                                if show_metadata:
                                    st.caption(f"Chunk {chunk['chunk_id']} | Page {chunk.get('page', 'N/A')} | {chunk['word_count']} words")

                                # Display content in a nice container
                                with st.container():
                                    st.markdown(chunk['content'])

                                if i < preview_chunks - 1:
                                    st.markdown("---")

                            # Show total chunks info
                            if chunk_count > preview_chunks:
                                st.info(f"📄 Showing {preview_chunks} of {chunk_count} total chunks")

                            # Export preview option
                            if st.button("📥 Export Full Document", key=f"export_{doc.doc_id}"):
                                # Combine all chunks
                                full_text = "\n\n---\n\n".join([chunk['content'] for chunk in full_doc['chunks']])

                                st.download_button(
                                    label="Download as Text",
                                    data=full_text,
                                    file_name=f"{doc.filename}.txt",
                                    mime="text/plain",
                                    key=f"download_{doc.doc_id}"
                                )
                        else:
                            st.warning("No content available for preview")
                    except Exception as e:
                        st.error(f"Error loading preview: {str(e)}")

                # Show relationships if enabled
                if st.session_state.get(f"show_relationships_{doc.doc_id}", False):
                    st.markdown("---")
                    st.subheader("🔗 Document Relationships")

                    try:
                        # Get all relationships for this document
                        relationships = kb.get_relationships(doc.doc_id, direction="both")

                        # Separate into outgoing and incoming
                        outgoing = [r for r in relationships if r['direction'] == 'outgoing']
                        incoming = [r for r in relationships if r['direction'] == 'incoming']

                        # Display existing relationships in two columns
                        rel_col1, rel_col2 = st.columns(2)

                        with rel_col1:
                            st.write(f"**Outgoing ({len(outgoing)})** - This document links to:")
                            if outgoing:
                                for rel in outgoing:
                                    related_doc = kb.documents.get(rel['related_doc_id'])
                                    if related_doc:
                                        rel_container = st.container()
                                        with rel_container:
                                            col_a, col_b = st.columns([3, 1])
                                            with col_a:
                                                st.caption(f"**{rel['relationship_type']}** → {related_doc.title}")
                                                if rel.get('note'):
                                                    st.caption(f"_\"{rel['note']}\"_")
                                            with col_b:
                                                if st.button("🗑️", key=f"del_out_{doc.doc_id}_{rel['related_doc_id']}_{rel['relationship_type']}"):
                                                    try:
                                                        kb.remove_relationship(doc.doc_id, rel['related_doc_id'], rel['relationship_type'])
                                                        st.success("Relationship removed")
                                                        st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Error: {str(e)}")
                            else:
                                st.info("No outgoing relationships")

                        with rel_col2:
                            st.write(f"**Incoming ({len(incoming)})** - Other documents link here:")
                            if incoming:
                                for rel in incoming:
                                    related_doc = kb.documents.get(rel['related_doc_id'])
                                    if related_doc:
                                        rel_container = st.container()
                                        with rel_container:
                                            col_a, col_b = st.columns([3, 1])
                                            with col_a:
                                                st.caption(f"{related_doc.title} → **{rel['relationship_type']}**")
                                                if rel.get('note'):
                                                    st.caption(f"_\"{rel['note']}\"_")
                                            with col_b:
                                                if st.button("🗑️", key=f"del_in_{doc.doc_id}_{rel['related_doc_id']}_{rel['relationship_type']}"):
                                                    try:
                                                        kb.remove_relationship(rel['related_doc_id'], doc.doc_id, rel['relationship_type'])
                                                        st.success("Relationship removed")
                                                        st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Error: {str(e)}")
                            else:
                                st.info("No incoming relationships")

                        st.markdown("---")

                        # Add new relationship form
                        st.write("**➕ Add New Relationship**")

                        add_col1, add_col2, add_col3 = st.columns([2, 1, 2])

                        with add_col1:
                            # Select target document
                            other_docs = {d.doc_id: d.title for d in docs if d.doc_id != doc.doc_id}
                            if other_docs:
                                target_doc = st.selectbox(
                                    "Link to document:",
                                    options=list(other_docs.keys()),
                                    format_func=lambda x: other_docs[x],
                                    key=f"target_{doc.doc_id}"
                                )
                            else:
                                st.info("No other documents available")
                                target_doc = None

                        with add_col2:
                            rel_type = st.selectbox(
                                "Type:",
                                ["related", "references", "prerequisite", "sequel"],
                                key=f"type_{doc.doc_id}"
                            )

                        with add_col3:
                            rel_note = st.text_input(
                                "Note (optional):",
                                key=f"note_{doc.doc_id}",
                                placeholder="Optional description"
                            )

                        if target_doc:
                            if st.button("➕ Add Relationship", key=f"add_rel_{doc.doc_id}"):
                                try:
                                    kb.add_relationship(doc.doc_id, target_doc, rel_type, rel_note)
                                    st.success(f"Relationship added: {doc.title} → {other_docs[target_doc]}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error adding relationship: {str(e)}")

                    except Exception as e:
                        st.error(f"Error loading relationships: {str(e)}")

        st.markdown("---")

        # Bulk operations section
        with st.expander("⚡ Bulk Operations", expanded=False):
            st.subheader("Bulk Document Management")

            bulk_tabs = st.tabs(["🗑️ Bulk Delete", "🏷️ Bulk Re-tag", "📤 Bulk Export"])

            # Tab 1: Bulk Delete
            with bulk_tabs[0]:
                st.write("**Delete Multiple Documents**")

                delete_method = st.radio(
                    "Select documents by:",
                    ["Document IDs", "Tags"],
                    key="delete_method"
                )

                if delete_method == "Document IDs":
                    # Let user enter document IDs
                    doc_ids_input = st.text_area(
                        "Document IDs (one per line)",
                        height=100,
                        placeholder="Enter document IDs, one per line"
                    )

                    if st.button("🗑️ Delete Selected Documents"):
                        if doc_ids_input.strip():
                            doc_ids = [line.strip() for line in doc_ids_input.split('\n') if line.strip()]

                            with st.spinner(f"Deleting {len(doc_ids)} documents..."):
                                results = kb.remove_documents_bulk(doc_ids=doc_ids)

                            st.success(f"✅ Deleted {len(results['removed'])} documents")

                            if results['failed']:
                                st.warning(f"⚠️ Failed to delete {len(results['failed'])} documents")
                                with st.expander("View errors"):
                                    for failure in results['failed']:
                                        st.text(f"- {failure['doc_id']}: {failure['error']}")

                            st.rerun()
                        else:
                            st.error("Please enter at least one document ID")

                else:  # Tags
                    delete_tags_input = st.text_input(
                        "Tags (comma-separated)",
                        placeholder="e.g., draft, old, archive",
                        key="delete_tags"
                    )

                    if st.button("🗑️ Delete Documents with Tags"):
                        if delete_tags_input.strip():
                            tags = [t.strip() for t in delete_tags_input.split(',') if t.strip()]

                            # Show confirmation
                            matching_docs = [doc for doc in docs if any(tag in doc.tags for tag in tags)]
                            st.warning(f"⚠️ This will delete {len(matching_docs)} documents with tags: {', '.join(tags)}")

                            if st.button("⚠️ Confirm Delete", key="confirm_delete_tags"):
                                with st.spinner(f"Deleting {len(matching_docs)} documents..."):
                                    results = kb.remove_documents_bulk(tags=tags)

                                st.success(f"✅ Deleted {len(results['removed'])} documents")

                                if results['failed']:
                                    st.warning(f"⚠️ Failed to delete {len(results['failed'])} documents")

                                st.rerun()
                        else:
                            st.error("Please enter at least one tag")

            # Tab 2: Bulk Re-tag
            with bulk_tabs[1]:
                st.write("**Update Tags for Multiple Documents**")

                retag_method = st.radio(
                    "Select documents by:",
                    ["Document IDs", "Existing Tags"],
                    key="retag_method"
                )

                if retag_method == "Document IDs":
                    retag_doc_ids = st.text_area(
                        "Document IDs (one per line)",
                        height=100,
                        placeholder="Enter document IDs, one per line",
                        key="retag_doc_ids"
                    )
                    retag_existing_tags = None
                else:
                    retag_doc_ids = None
                    retag_existing_tags_input = st.text_input(
                        "Find documents with tags (comma-separated)",
                        placeholder="e.g., draft, pending",
                        key="retag_existing_tags"
                    )
                    retag_existing_tags = [t.strip() for t in retag_existing_tags_input.split(',') if t.strip()] if retag_existing_tags_input else None

                # Tag operation selection
                operation = st.selectbox(
                    "Operation",
                    ["Add Tags", "Remove Tags", "Replace All Tags"],
                    key="tag_operation"
                )

                if operation == "Add Tags":
                    tags_input = st.text_input("Tags to add (comma-separated)", key="add_tags_input")

                    if st.button("➕ Add Tags"):
                        if (retag_doc_ids and retag_doc_ids.strip()) or retag_existing_tags:
                            tags = [t.strip() for t in tags_input.split(',') if t.strip()]
                            if not tags:
                                st.error("Please enter at least one tag to add")
                            else:
                                doc_ids = [line.strip() for line in retag_doc_ids.split('\n') if line.strip()] if retag_doc_ids else None

                                with st.spinner("Updating tags..."):
                                    results = kb.update_tags_bulk(
                                        doc_ids=doc_ids,
                                        existing_tags=retag_existing_tags,
                                        add_tags=tags
                                    )

                                st.success(f"✅ Updated {len(results['updated'])} documents")

                                if results['failed']:
                                    st.warning(f"⚠️ Failed to update {len(results['failed'])} documents")

                                st.rerun()
                        else:
                            st.error("Please select documents")

                elif operation == "Remove Tags":
                    tags_input = st.text_input("Tags to remove (comma-separated)", key="remove_tags_input")

                    if st.button("➖ Remove Tags"):
                        if (retag_doc_ids and retag_doc_ids.strip()) or retag_existing_tags:
                            tags = [t.strip() for t in tags_input.split(',') if t.strip()]
                            if not tags:
                                st.error("Please enter at least one tag to remove")
                            else:
                                doc_ids = [line.strip() for line in retag_doc_ids.split('\n') if line.strip()] if retag_doc_ids else None

                                with st.spinner("Updating tags..."):
                                    results = kb.update_tags_bulk(
                                        doc_ids=doc_ids,
                                        existing_tags=retag_existing_tags,
                                        remove_tags=tags
                                    )

                                st.success(f"✅ Updated {len(results['updated'])} documents")

                                if results['failed']:
                                    st.warning(f"⚠️ Failed to update {len(results['failed'])} documents")

                                st.rerun()
                        else:
                            st.error("Please select documents")

                else:  # Replace All Tags
                    tags_input = st.text_input("New tags (comma-separated)", key="replace_tags_input")

                    if st.button("🔄 Replace All Tags"):
                        if (retag_doc_ids and retag_doc_ids.strip()) or retag_existing_tags:
                            tags = [t.strip() for t in tags_input.split(',') if t.strip()]
                            doc_ids = [line.strip() for line in retag_doc_ids.split('\n') if line.strip()] if retag_doc_ids else None

                            with st.spinner("Updating tags..."):
                                results = kb.update_tags_bulk(
                                    doc_ids=doc_ids,
                                    existing_tags=retag_existing_tags,
                                    replace_tags=tags
                                )

                            st.success(f"✅ Updated {len(results['updated'])} documents")

                            if results['failed']:
                                st.warning(f"⚠️ Failed to update {len(results['failed'])} documents")

                            st.rerun()
                        else:
                            st.error("Please select documents")

            # Tab 3: Bulk Export
            with bulk_tabs[2]:
                st.write("**Export Document Metadata**")

                export_method = st.radio(
                    "Export:",
                    ["All Documents", "Documents with Tags", "Specific Documents"],
                    key="export_method"
                )

                export_format = st.selectbox(
                    "Format",
                    ["JSON", "CSV", "Markdown"],
                    key="bulk_export_format"
                )

                if export_method == "All Documents":
                    if st.button("📤 Export All"):
                        with st.spinner("Exporting documents..."):
                            export_data = kb.export_documents_bulk(format=export_format.lower())

                        st.download_button(
                            label=f"Download {export_format}",
                            data=export_data,
                            file_name=f"documents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                            mime="text/plain"
                        )

                elif export_method == "Documents with Tags":
                    export_tags_input = st.text_input(
                        "Tags (comma-separated)",
                        placeholder="e.g., reference, c64",
                        key="export_tags"
                    )

                    if st.button("📤 Export by Tags"):
                        if export_tags_input.strip():
                            tags = [t.strip() for t in export_tags_input.split(',') if t.strip()]

                            with st.spinner("Exporting documents..."):
                                export_data = kb.export_documents_bulk(tags=tags, format=export_format.lower())

                            st.download_button(
                                label=f"Download {export_format}",
                                data=export_data,
                                file_name=f"documents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                                mime="text/plain"
                            )
                        else:
                            st.error("Please enter at least one tag")

                else:  # Specific Documents
                    export_doc_ids_input = st.text_area(
                        "Document IDs (one per line)",
                        height=100,
                        placeholder="Enter document IDs, one per line",
                        key="export_doc_ids"
                    )

                    if st.button("📤 Export Selected"):
                        if export_doc_ids_input.strip():
                            doc_ids = [line.strip() for line in export_doc_ids_input.split('\n') if line.strip()]

                            with st.spinner("Exporting documents..."):
                                export_data = kb.export_documents_bulk(doc_ids=doc_ids, format=export_format.lower())

                            st.download_button(
                                label=f"Download {export_format}",
                                data=export_data,
                                file_name=f"documents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                                mime="text/plain"
                            )
                        else:
                            st.error("Please enter at least one document ID")

# ========== WEB SCRAPING PAGE ==========
elif page == "🌐 Web Scraping":
    st.title("🌐 Web Scraping")
    st.write("Scrape documentation websites and convert them to searchable markdown documents.")

    # Add URL section
    with st.expander("➕ Add New URL to Scrape", expanded=True):
        st.subheader("Scrape Documentation Website")

        url_input = st.text_input(
            "📍 Website URL",
            placeholder="https://docs.example.com/api/",
            key="scrape_url_input",
            help="Enter the starting URL to scrape"
        )

        # Simple options (above the fold)
        col1, col2, col3 = st.columns(3)
        with col1:
            follow_links = st.checkbox(
                "Follow Links",
                value=True,
                help="Follow links to scrape sub-pages (uncheck to scrape only the single page)"
            )
        with col2:
            same_domain_only = st.checkbox(
                "Same Domain Only",
                value=True,
                help="Only follow links on the same domain (prevents scraping external sites)"
            )
        with col3:
            max_pages = st.number_input(
                "Max Pages",
                min_value=1,
                max_value=500,
                value=50,
                help="Maximum number of pages to scrape"
            )

        # Configuration options
        with st.expander("⚙️ Advanced Scraping Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                scrape_depth = st.number_input(
                    "Max Depth",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Maximum link depth to follow (1=single page, 2=linked pages, 3=two levels deep)"
                )
                scrape_threads = st.number_input(
                    "Threads",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Number of concurrent download threads for faster scraping"
                )

            with col2:
                scrape_delay = st.number_input(
                    "Delay (ms)",
                    min_value=0,
                    max_value=5000,
                    value=100,
                    help="Delay between requests to avoid overwhelming the server"
                )
                scrape_limit = st.text_input(
                    "Limit URLs (optional)",
                    placeholder="https://docs.example.com/api/",
                    help="Advanced: Only scrape URLs with this prefix (overrides Same Domain Only)"
                )

            scrape_selector = st.text_input(
                "CSS Selector (optional)",
                placeholder="article.main-content",
                help="CSS selector to extract specific content (e.g., main article body)"
            )

        col1, col2 = st.columns(2)
        with col1:
            scrape_title = st.text_input(
                "Base Title (optional)",
                "",
                key="scrape_title",
                help="Base title for scraped documents (will be combined with page titles)"
            )
        with col2:
            scrape_tags = st.text_input(
                "Tags (comma-separated)",
                "",
                key="scrape_tags",
                help="Additional tags (domain name will be auto-added)"
            )

        if st.button("🚀 Start Scraping", key="scrape_url_btn", type="primary"):
            if not url_input:
                st.error("❌ Please enter a URL")
            else:
                # Show centered loading indicator
                loading_container = st.empty()

                with loading_container.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown("""
                            <style>
                            .scraping-container {
                                padding: 2rem;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 15px;
                                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                                text-align: center;
                                margin: 2rem 0;
                            }
                            .scraping-text {
                                color: white;
                                font-size: 1.5rem;
                                font-weight: bold;
                                margin-top: 1rem;
                            }
                            .scraping-subtext {
                                color: rgba(255,255,255,0.9);
                                font-size: 1rem;
                                margin-top: 0.5rem;
                            }
                            @keyframes pulse {
                                0%, 100% { transform: scale(1); }
                                50% { transform: scale(1.1); }
                            }
                            .scraping-icon {
                                font-size: 3rem;
                                animation: pulse 2s ease-in-out infinite;
                            }
                            </style>
                            <div class="scraping-container">
                                <div class="scraping-icon">🌐</div>
                                <div class="scraping-text">Scraping Website</div>
                                <div class="scraping-subtext">Please wait while we fetch and process the content...</div>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"**URL:** {url_input}")
                        progress_bar = st.progress(0)

                try:
                    # Parse tags
                    tags = [t.strip() for t in scrape_tags.split(',') if t.strip()]

                    # Start scraping
                    result = kb.scrape_url(
                        url=url_input,
                        title=scrape_title or None,
                        tags=tags,
                        follow_links=follow_links,
                        same_domain_only=same_domain_only,
                        max_pages=max_pages,
                        depth=scrape_depth,
                        limit=scrape_limit or None,
                        threads=scrape_threads,
                        delay=scrape_delay,
                        selector=scrape_selector or None
                    )

                    # Clear loading indicator
                    loading_container.empty()

                    if result['status'] == 'success':
                        st.success(f"✅ **Scraping complete!**\n\n"
                                 f"**Files scraped:** {result['files_scraped']}\n\n"
                                 f"**Documents added:** {result['docs_added']}\n\n"
                                 f"**Output directory:** `{result['output_dir']}`")

                        # Trigger background reindexing
                        trigger_background_reindex()
                        st.rerun()

                    elif result['status'] == 'partial':
                        st.warning(f"⚠️ **Scraping partially complete**\n\n"
                                 f"**Files scraped:** {result['files_scraped']}\n\n"
                                 f"**Documents added:** {result['docs_added']}\n\n"
                                 f"**Failed:** {result['docs_failed']}\n\n"
                                 f"**Error:** {result.get('error', 'Unknown')}")

                        # Trigger background reindexing
                        if result['docs_added'] > 0:
                            trigger_background_reindex()
                            st.rerun()
                    else:
                        st.error(f"❌ **Scraping failed**\n\n{result.get('error', 'Unknown error')}")

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ **Error during scraping:**\n\n{str(e)}")

    st.markdown("---")

    # List scraped websites
    st.subheader("📚 Scraped Websites")

    # Get all URL-sourced documents
    url_docs = [doc for doc in kb.documents.values() if doc.source_url]

    if not url_docs:
        st.info("No scraped websites yet. Add a URL using the section above!")
    else:
        # Group by domain
        from urllib.parse import urlparse
        websites = {}
        for doc in url_docs:
            domain = urlparse(doc.source_url).netloc
            if domain not in websites:
                websites[domain] = []
            websites[domain].append(doc)

        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Websites", len(websites))
        with col2:
            st.metric("Total Pages", len(url_docs))
        with col3:
            # Count successful scrapes
            successful = sum(1 for doc in url_docs if doc.scrape_status == 'success')
            st.metric("Successful", successful)

        st.markdown("---")

        # Search/filter
        search_query = st.text_input("🔎 Filter websites", "")

        # Filter websites
        filtered_websites = {}
        if search_query:
            for domain, docs in websites.items():
                matching_docs = [d for d in docs if search_query.lower() in d.title.lower() or
                               search_query.lower() in d.source_url.lower()]
                if matching_docs:
                    filtered_websites[domain] = matching_docs
        else:
            filtered_websites = websites

        st.write(f"Showing {len(filtered_websites)} of {len(websites)} websites")

        # Display each website
        for domain, docs in sorted(filtered_websites.items()):
            with st.expander(f"🌐 {domain} ({len(docs)} pages)", expanded=False):
                # Website summary
                st.write(f"**Domain:** {domain}")
                st.write(f"**Pages scraped:** {len(docs)}")

                # Latest scrape date
                latest_scrape = max([doc.scrape_date for doc in docs if doc.scrape_date], default=None)
                if latest_scrape:
                    st.write(f"**Last scraped:** {format_timestamp(latest_scrape)}")

                # Tags summary
                all_tags = set()
                for doc in docs:
                    all_tags.update(doc.tags)
                if all_tags:
                    st.write(f"**Tags:** {', '.join(sorted(all_tags))}")

                st.markdown("---")

                # Actions
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🔄 Re-scrape All", key=f"rescrape_all_{domain}"):
                        with st.spinner(f"Re-scraping all pages from {domain}..."):
                            rescrape_count = 0
                            for doc in docs:
                                try:
                                    result = kb.rescrape_document(doc.doc_id)
                                    if result['status'] == 'success':
                                        rescrape_count += 1
                                except Exception as e:
                                    st.error(f"Error re-scraping {doc.title}: {str(e)}")

                            if rescrape_count > 0:
                                trigger_background_reindex()
                                st.success(f"✅ Re-scraped {rescrape_count} pages")
                                st.rerun()

                with col2:
                    if st.button("📋 View All Pages", key=f"view_pages_{domain}"):
                        st.session_state[f"show_pages_{domain}"] = not st.session_state.get(f"show_pages_{domain}", False)
                        st.rerun()

                with col3:
                    if st.button("🗑️ Delete All", key=f"delete_all_{domain}"):
                        st.session_state[f"confirm_delete_{domain}"] = True
                        st.rerun()

                # Confirm delete
                if st.session_state.get(f"confirm_delete_{domain}", False):
                    st.warning(f"⚠️ This will delete all {len(docs)} pages from {domain}")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Confirm Delete", key=f"confirm_del_{domain}"):
                            for doc in docs:
                                kb.remove_document(doc.doc_id)
                            trigger_background_reindex()
                            st.session_state[f"confirm_delete_{domain}"] = False
                            st.success(f"Deleted all pages from {domain}")
                            st.rerun()
                    with confirm_col2:
                        if st.button("❌ Cancel", key=f"cancel_del_{domain}"):
                            st.session_state[f"confirm_delete_{domain}"] = False
                            st.rerun()

                # Show individual pages
                if st.session_state.get(f"show_pages_{domain}", False):
                    st.markdown("---")
                    st.write("**Individual Pages:**")

                    for doc in sorted(docs, key=lambda d: d.title):
                        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([3, 1, 1, 1, 1])

                        with page_col1:
                            status_emoji = "✅" if doc.scrape_status == "success" else "⚠️"
                            st.write(f"{status_emoji} {doc.title}")
                            st.caption(f"🔗 {doc.source_url}")

                        with page_col2:
                            st.caption(f"{doc.total_chunks} chunks")

                        with page_col3:
                            if st.button("📄", key=f"view_md_{doc.doc_id}", help="View scraped markdown"):
                                st.session_state[f"show_md_{doc.doc_id}"] = not st.session_state.get(f"show_md_{doc.doc_id}", False)
                                st.rerun()

                        with page_col4:
                            if st.button("🔄", key=f"rescrape_{doc.doc_id}", help="Re-scrape this page"):
                                with st.spinner("Re-scraping..."):
                                    try:
                                        result = kb.rescrape_document(doc.doc_id)
                                        if result['status'] == 'success':
                                            trigger_background_reindex()
                                            st.success("✅ Re-scraped successfully")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {result.get('error')}")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")

                        with page_col5:
                            if st.button("🗑️", key=f"delete_{doc.doc_id}", help="Delete this page"):
                                kb.remove_document(doc.doc_id)
                                trigger_background_reindex()
                                st.rerun()

                        # Show markdown content if toggled
                        if st.session_state.get(f"show_md_{doc.doc_id}", False):
                            st.markdown("---")
                            st.subheader("📄 Scraped Markdown Content")

                            try:
                                # Get the full document content
                                full_doc = kb.get_document(doc.doc_id)

                                if full_doc and 'chunks' in full_doc:
                                    # Combine all chunks
                                    markdown_content = "\n\n".join([chunk['content'] for chunk in full_doc['chunks']])

                                    # Show metadata
                                    st.info(f"**Source:** {doc.source_url}\n\n"
                                           f"**Scraped:** {format_timestamp(doc.scrape_date) if doc.scrape_date else 'N/A'}\n\n"
                                           f"**Total chunks:** {len(full_doc['chunks'])}")

                                    # Display markdown in code block for easier viewing
                                    st.code(markdown_content, language="markdown")

                                    # Download button
                                    st.download_button(
                                        label="💾 Download Markdown",
                                        data=markdown_content,
                                        file_name=f"{doc.filename}.md",
                                        mime="text/markdown",
                                        key=f"download_md_{doc.doc_id}"
                                    )
                                else:
                                    st.warning("No markdown content available")
                            except Exception as e:
                                st.error(f"Error loading markdown: {str(e)}")

# ========== URL MONITORING PAGE ==========
elif page == "🌐 URL Monitoring":
    st.title("🌐 URL Monitoring Dashboard")
    st.write("Monitor scraped websites for updates, new pages, and missing content.")

    # Get URL-sourced documents
    url_docs = [doc for doc in kb.documents.values() if doc.source_url]

    if not url_docs:
        st.info("📋 No URL-sourced documents found. Use the **🌐 Web Scraping** page to add websites first.")
    else:
        # Top metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌐 Monitored Sites", len(url_docs))
        with col2:
            # Count unique base URLs
            base_urls = set()
            for doc in url_docs:
                if doc.scrape_config:
                    try:
                        config = json.loads(doc.scrape_config) if isinstance(doc.scrape_config, str) else doc.scrape_config
                        if 'base_url' in config:
                            base_urls.add(config['base_url'])
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
            st.metric("🔗 Unique Sources", len(base_urls))
        with col3:
            # Show last check time if available
            if 'last_url_check' in st.session_state:
                last_check = st.session_state['last_url_check']
                st.metric("🕒 Last Check", last_check.strftime("%H:%M:%S"))
            else:
                st.metric("🕒 Last Check", "Never")

        st.markdown("---")

        # Check controls
        st.subheader("🔍 Run Update Check")

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            check_mode = st.radio(
                "Check Mode:",
                ["Quick (Fast)", "Full (Comprehensive)"],
                help="Quick: Check Last-Modified headers only (~1s/site)\nFull: Discover new/missing pages (~10-60s/site)"
            )

        with col2:
            auto_rescrape = st.checkbox(
                "Auto Re-scrape",
                value=False,
                help="Automatically re-scrape changed documents"
            )

        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("▶️ Run Check", type="primary", use_container_width=True):
                check_structure = (check_mode == "Full (Comprehensive)")

                with st.spinner(f"Running {check_mode.split()[0].lower()} check on {len(url_docs)} documents..."):
                    try:
                        results = kb.check_url_updates(
                            auto_rescrape=auto_rescrape,
                            check_structure=check_structure
                        )

                        st.session_state['last_url_check'] = datetime.now()
                        st.session_state['last_check_results'] = results
                        st.success("✅ Check complete!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during check: {str(e)}")

        st.markdown("---")

        # Display results if available
        if 'last_check_results' in st.session_state:
            results = st.session_state['last_check_results']

            # Results summary
            st.subheader("📊 Check Results")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("✅ Unchanged", len(results.get('unchanged', [])))
            with col2:
                changed_count = len(results.get('changed', []))
                st.metric("🔄 Changed", changed_count, delta=changed_count if changed_count > 0 else None)
            with col3:
                new_count = len(results.get('new_pages', []))
                st.metric("🆕 New Pages", new_count, delta=new_count if new_count > 0 else None)
            with col4:
                missing_count = len(results.get('missing_pages', []))
                st.metric("❌ Missing", missing_count, delta=-missing_count if missing_count > 0 else None)
            with col5:
                failed_count = len(results.get('failed', []))
                st.metric("⚠️ Failed", failed_count)

            # Tabbed results display
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                f"🔄 Changed ({len(results.get('changed', []))})",
                f"🆕 New Pages ({len(results.get('new_pages', []))})",
                f"❌ Missing ({len(results.get('missing_pages', []))})",
                f"✅ Unchanged ({len(results.get('unchanged', []))})",
                f"📈 Sessions ({len(results.get('scrape_sessions', []))})"
            ])

            # Tab 1: Changed documents
            with tab1:
                changed = results.get('changed', [])
                if changed:
                    st.write(f"**{len(changed)} documents have updates available:**")

                    for doc_info in changed:
                        with st.expander(f"🔄 {doc_info['title']}", expanded=False):
                            st.write(f"**URL:** {doc_info['url']}")
                            st.write(f"**Document ID:** `{doc_info['doc_id'][:12]}...`")
                            if 'last_modified' in doc_info:
                                st.write(f"**Last Modified:** {doc_info['last_modified']}")
                            if 'scraped_date' in doc_info:
                                st.write(f"**Scraped Date:** {doc_info['scraped_date']}")
                            if 'reason' in doc_info:
                                st.write(f"**Reason:** {doc_info['reason']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔄 Re-scrape", key=f"rescrape_{doc_info['doc_id']}"):
                                    with st.spinner(f"Re-scraping {doc_info['title']}..."):
                                        try:
                                            new_doc_id = kb.rescrape_document(doc_info['doc_id'])
                                            st.success(f"✅ Re-scraped successfully! New ID: {new_doc_id[:12]}...")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                            with col2:
                                if st.button("🌐 Open URL", key=f"open_{doc_info['doc_id']}"):
                                    st.markdown(f"[Open in browser]({doc_info['url']})")
                else:
                    st.info("✅ No changed documents")

            # Tab 2: New pages
            with tab2:
                new_pages = results.get('new_pages', [])
                if new_pages:
                    st.write(f"**{len(new_pages)} new pages discovered:**")

                    # Group by base_url
                    by_site = {}
                    for page in new_pages:
                        site = page.get('base_url', 'Unknown')
                        if site not in by_site:
                            by_site[site] = []
                        by_site[site].append(page)

                    for site, pages in by_site.items():
                        with st.expander(f"🌐 {site} ({len(pages)} new pages)", expanded=False):
                            for page in pages[:10]:  # Show first 10
                                st.write(f"- {page['url']}")
                            if len(pages) > 10:
                                st.write(f"... and {len(pages) - 10} more")

                            if st.button(f"📥 Scrape All {len(pages)} Pages", key=f"scrape_new_{hash(site)}"):
                                with st.spinner(f"Scraping {len(pages)} new pages from {site}..."):
                                    try:
                                        scraped = 0
                                        progress_bar = st.progress(0)
                                        for i, page in enumerate(pages):
                                            try:
                                                kb.scrape_url(
                                                    page['url'],
                                                    depth=1,  # Single page
                                                    follow_links=False
                                                )
                                                scraped += 1
                                            except Exception as e:
                                                st.warning(f"Failed to scrape {page['url']}: {str(e)}")
                                            progress_bar.progress((i + 1) / len(pages))
                                        st.success(f"✅ Scraped {scraped}/{len(pages)} new pages")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.info("✅ No new pages discovered")

            # Tab 3: Missing pages
            with tab3:
                missing = results.get('missing_pages', [])
                if missing:
                    st.write(f"**{len(missing)} pages missing or inaccessible:**")

                    for doc_info in missing:
                        with st.expander(f"❌ {doc_info['title']}", expanded=False):
                            st.write(f"**URL:** {doc_info['url']}")
                            st.write(f"**Document ID:** `{doc_info['doc_id'][:12]}...`")
                            if 'reason' in doc_info:
                                st.write(f"**Reason:** {doc_info['reason']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🗑️ Remove from DB", key=f"remove_{doc_info['doc_id']}"):
                                    if st.button("⚠️ Confirm Delete", key=f"confirm_remove_{doc_info['doc_id']}"):
                                        try:
                                            kb.remove_document(doc_info['doc_id'])
                                            st.success(f"✅ Removed {doc_info['title']}")
                                            # Clear results to force re-check
                                            if 'last_check_results' in st.session_state:
                                                del st.session_state['last_check_results']
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                            with col2:
                                if st.button("🌐 Check URL", key=f"check_{doc_info['doc_id']}"):
                                    st.markdown(f"[Open in browser]({doc_info['url']})")
                else:
                    st.info("✅ No missing pages")

            # Tab 4: Unchanged documents
            with tab4:
                unchanged = results.get('unchanged', [])
                if unchanged:
                    st.write(f"**{len(unchanged)} documents are up to date:**")

                    # Display as table
                    df_data = []
                    for doc_info in unchanged:
                        df_data.append({
                            'Title': doc_info['title'][:50] + '...' if len(doc_info['title']) > 50 else doc_info['title'],
                            'Doc ID': doc_info['doc_id'][:12] + '...',
                            'URL': doc_info['url']
                        })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.info("No unchanged documents")

            # Tab 5: Scrape sessions
            with tab5:
                sessions = results.get('scrape_sessions', [])
                if sessions:
                    st.write(f"**{len(sessions)} scrape sessions monitored:**")

                    # Display as table
                    df_data = []
                    for session in sessions:
                        df_data.append({
                            'Site': session['base_url'],
                            'Total Docs': session['docs_count'],
                            'Unchanged': session['unchanged'],
                            'Changed': session['changed'],
                            'New': session.get('new', 0),
                            'Missing': session.get('missing', 0)
                        })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

                        # Export option
                        st.markdown("---")
                        if st.button("📥 Export Results as JSON"):
                            import json
                            from datetime import datetime

                            filename = f"url_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            json_str = json.dumps(results, indent=2, default=str)

                            st.download_button(
                                label="💾 Download JSON",
                                data=json_str,
                                file_name=filename,
                                mime="application/json"
                            )
                else:
                    st.info("No scrape sessions found")

        else:
            # No results yet
            st.info("👆 Click 'Run Check' above to check for URL updates")

        # Sites overview table
        st.markdown("---")
        st.subheader("🌐 All Monitored Sites")

        # Group documents by base URL
        by_base_url = {}
        for doc in url_docs:
            # Parse scrape_config from JSON string if it exists
            if doc.scrape_config:
                try:
                    config = json.loads(doc.scrape_config) if isinstance(doc.scrape_config, str) else doc.scrape_config
                    base_url = config.get('base_url', doc.source_url)
                except (json.JSONDecodeError, AttributeError):
                    base_url = doc.source_url
            else:
                base_url = doc.source_url

            if base_url not in by_base_url:
                by_base_url[base_url] = []
            by_base_url[base_url].append(doc)

        # Display grouped sites
        for base_url, docs in sorted(by_base_url.items()):
            with st.expander(f"🌐 {base_url} ({len(docs)} documents)", expanded=False):
                st.write(f"**Base URL:** {base_url}")
                st.write(f"**Documents:** {len(docs)}")

                # Show document list
                for doc in docs[:5]:  # Show first 5
                    st.write(f"- {doc.title}")
                if len(docs) > 5:
                    st.write(f"... and {len(docs) - 5} more")

                # Quick actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Re-scrape All", key=f"rescrape_all_{hash(base_url)}"):
                        with st.spinner(f"Re-scraping {len(docs)} documents..."):
                            try:
                                rescraped = 0
                                progress_bar = st.progress(0)
                                for i, doc in enumerate(docs):
                                    try:
                                        kb.rescrape_document(doc.doc_id)
                                        rescraped += 1
                                    except Exception as e:
                                        st.warning(f"Failed to re-scrape {doc.title}: {str(e)}")
                                    progress_bar.progress((i + 1) / len(docs))
                                st.success(f"✅ Re-scraped {rescraped}/{len(docs)} documents")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                with col2:
                    if st.button("🗑️ Remove All", key=f"remove_all_{hash(base_url)}"):
                        if st.button("⚠️ Confirm Delete All", key=f"confirm_remove_all_{hash(base_url)}"):
                            try:
                                for doc in docs:
                                    kb.remove_document(doc.doc_id)
                                st.success(f"✅ Removed {len(docs)} documents from {base_url}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                with col3:
                    if st.button("📊 View Stats", key=f"stats_{hash(base_url)}"):
                        # Show detailed stats
                        total_chunks = sum(len(kb._get_chunks_db(doc.doc_id)) for doc in docs)
                        total_size = sum(len(doc.content or '') for doc in docs)
                        st.write(f"**Total Chunks:** {total_chunks}")
                        st.write(f"**Total Size:** {total_size:,} chars")

# ========== TAG MANAGEMENT PAGE ==========
elif page == "🏷️ Tag Management":
    st.title("🏷️ Tag Management")

    # Get all tags from documents
    all_tags = {}
    for doc in kb.documents.values():
        for tag in doc.tags:
            if tag not in all_tags:
                all_tags[tag] = []
            all_tags[tag].append(doc.doc_id)

    if not all_tags:
        st.info("No tags found in the knowledge base.")
    else:
        # Statistics
        st.subheader("📊 Tag Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Tags", len(all_tags))
        with col2:
            avg_docs = sum(len(docs) for docs in all_tags.values()) / len(all_tags)
            st.metric("Avg Documents per Tag", f"{avg_docs:.1f}")
        with col3:
            max_tag = max(all_tags.items(), key=lambda x: len(x[1]))
            st.metric("Most Used Tag", f"{max_tag[0]} ({len(max_tag[1])})")

        st.markdown("---")

        # Tag list with operations
        st.subheader("📋 All Tags")

        # Sort options
        sort_by = st.radio("Sort by:", ["Name (A-Z)", "Document Count", "Name (Z-A)"], horizontal=True)

        if sort_by == "Name (A-Z)":
            sorted_tags = sorted(all_tags.items())
        elif sort_by == "Name (Z-A)":
            sorted_tags = sorted(all_tags.items(), reverse=True)
        else:  # Document Count
            sorted_tags = sorted(all_tags.items(), key=lambda x: len(x[1]), reverse=True)

        # Display tags in a table format
        tag_data = []
        for tag, doc_ids in sorted_tags:
            tag_data.append({
                "Tag": tag,
                "Documents": len(doc_ids),
                "Select": False
            })

        # Show as dataframe
        df = pd.DataFrame(tag_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Tag operations
        st.subheader("⚙️ Tag Operations")

        operation_tabs = st.tabs(["🔄 Rename Tag", "🔗 Merge Tags", "🗑️ Delete Tag", "➕ Add to All"])

        # Tab 1: Rename Tag
        with operation_tabs[0]:
            st.write("**Rename a tag across all documents**")

            col1, col2 = st.columns(2)
            with col1:
                old_tag = st.selectbox("Select tag to rename:", list(all_tags.keys()), key="rename_old")
            with col2:
                new_tag = st.text_input("New tag name:", key="rename_new")

            if old_tag and old_tag in all_tags:
                st.info(f"This will rename '{old_tag}' in {len(all_tags[old_tag])} document(s)")

            if st.button("🔄 Rename Tag") and old_tag and new_tag:
                if new_tag.strip():
                    try:
                        # Use update_tags_bulk to remove old and add new
                        results = kb.update_tags_bulk(
                            existing_tags=[old_tag],
                            remove_tags=[old_tag],
                            add_tags=[new_tag.strip()]
                        )

                        st.success(f"✅ Renamed '{old_tag}' to '{new_tag}' in {len(results['updated'])} documents")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error renaming tag: {str(e)}")
                else:
                    st.error("New tag name cannot be empty")

        # Tab 2: Merge Tags
        with operation_tabs[1]:
            st.write("**Merge multiple tags into one**")

            tags_to_merge = st.multiselect(
                "Select tags to merge:",
                list(all_tags.keys()),
                key="merge_tags"
            )

            target_tag = st.text_input("Merge into tag name:", key="merge_target")

            if tags_to_merge:
                total_docs = set()
                for tag in tags_to_merge:
                    total_docs.update(all_tags[tag])
                st.info(f"This will merge {len(tags_to_merge)} tags affecting {len(total_docs)} document(s)")

            if st.button("🔗 Merge Tags") and tags_to_merge and target_tag:
                if target_tag.strip():
                    try:
                        # Remove all source tags and add target tag
                        results = kb.update_tags_bulk(
                            existing_tags=tags_to_merge,
                            remove_tags=tags_to_merge,
                            add_tags=[target_tag.strip()]
                        )

                        st.success(f"✅ Merged {len(tags_to_merge)} tags into '{target_tag}' across {len(results['updated'])} documents")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error merging tags: {str(e)}")
                else:
                    st.error("Target tag name cannot be empty")

        # Tab 3: Delete Tag
        with operation_tabs[2]:
            st.write("**Remove a tag from all documents**")

            tag_to_delete = st.selectbox("Select tag to delete:", list(all_tags.keys()), key="delete_tag")

            if tag_to_delete and tag_to_delete in all_tags:
                st.warning(f"⚠️ This will remove '{tag_to_delete}' from {len(all_tags[tag_to_delete])} document(s)")

            if st.button("🗑️ Delete Tag") and tag_to_delete:
                try:
                    results = kb.update_tags_bulk(
                        existing_tags=[tag_to_delete],
                        remove_tags=[tag_to_delete]
                    )

                    st.success(f"✅ Removed '{tag_to_delete}' from {len(results['updated'])} documents")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting tag: {str(e)}")

        # Tab 4: Add to All
        with operation_tabs[3]:
            st.write("**Add a tag to all documents**")

            new_global_tag = st.text_input("Tag to add to all documents:", key="global_tag")

            total_docs = len(kb.documents)
            if new_global_tag:
                st.info(f"This will add '{new_global_tag}' to all {total_docs} document(s)")

            if st.button("➕ Add to All Documents") and new_global_tag:
                if new_global_tag.strip():
                    try:
                        # Add tag to all documents
                        all_doc_ids = list(kb.documents.keys())
                        results = kb.update_tags_bulk(
                            doc_ids=all_doc_ids,
                            add_tags=[new_global_tag.strip()]
                        )

                        st.success(f"✅ Added '{new_global_tag}' to {len(results['updated'])} documents")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding tag: {str(e)}")
                else:
                    st.error("Tag name cannot be empty")

# ========== ENTITY EXTRACTION PAGE ==========
elif page == "🧠 Entity Extraction":
    st.title("🧠 Named Entity Extraction")
    st.write("Extract and explore technical entities from C64 documentation using AI")

    # Check if LLM is configured
    try:
        llm_available = True
    except Exception:
        llm_available = False
        st.error("⚠️ LLM not configured. Entity extraction requires LLM_PROVIDER and API key.")
        st.info("Set environment variables: LLM_PROVIDER, ANTHROPIC_API_KEY or OPENAI_API_KEY")

    if llm_available:
        # Get entity statistics
        try:
            stats = kb.get_entity_stats()

            # Overall statistics
            st.subheader("📊 Entity Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Entities", f"{stats.get('total_entities', 0):,}")
            with col2:
                st.metric("Documents with Entities", f"{stats.get('documents_with_entities', 0)}/{len(kb.documents)}")
            with col3:
                avg_per_doc = stats.get('total_entities', 0) / max(stats.get('documents_with_entities', 1), 1)
                st.metric("Avg per Document", f"{avg_per_doc:.1f}")
            with col4:
                entity_types = len(stats.get('entities_by_type', {}))
                st.metric("Entity Types", entity_types)

            st.markdown("---")

            # Entity type breakdown
            if stats.get('entities_by_type'):
                st.subheader("📋 Entities by Type")

                type_data = []
                for entity_type, count in stats['entities_by_type'].items():
                    percentage = (count / stats['total_entities'] * 100) if stats['total_entities'] > 0 else 0
                    type_data.append({
                        "Type": entity_type.replace('_', ' ').title(),
                        "Count": count,
                        "Percentage": f"{percentage:.1f}%"
                    })

                df = pd.DataFrame(type_data).sort_values('Count', ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Top entities
            if stats.get('top_entities'):
                st.subheader("🏆 Top Entities (by document count)")

                top_entities = stats['top_entities'][:15]
                entity_data = []
                for ent in top_entities:
                    entity_data.append({
                        "Entity": ent['entity_text'],
                        "Type": ent['entity_type'].replace('_', ' ').title(),
                        "Documents": ent['document_count'],
                        "Avg Confidence": f"{ent['avg_confidence']:.2f}"
                    })

                df = pd.DataFrame(entity_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception:
            st.info("No entities extracted yet. Use the tabs below to start extracting.")

        st.markdown("---")

        # Operation tabs
        tabs = st.tabs([
            "🔍 Extract from Document",
            "📋 View Entities",
            "🔎 Search Entities",
            "⚡ Bulk Extraction"
        ])

        # Tab 1: Extract from single document
        with tabs[0]:
            st.subheader("🔍 Extract Entities from Document")
            st.write("Extract named entities from a single document using AI")

            # Document selection
            doc_options = {f"{doc.title} ({doc_id[:12]}...)": doc_id
                          for doc_id, doc in kb.documents.items()}

            if doc_options:
                selected_doc = st.selectbox(
                    "Select document:",
                    options=list(doc_options.keys()),
                    key="extract_doc_select"
                )

                col1, col2 = st.columns(2)
                with col1:
                    confidence = st.slider(
                        "Confidence threshold:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.05,
                        help="Only extract entities with confidence above this threshold"
                    )
                with col2:
                    force_regen = st.checkbox(
                        "Force regenerate",
                        value=False,
                        help="Re-extract even if entities already exist"
                    )

                if st.button("🚀 Extract Entities", type="primary"):
                    doc_id = doc_options[selected_doc]

                    with st.spinner(f"Extracting entities from {selected_doc}..."):
                        try:
                            result = kb.extract_entities(
                                doc_id=doc_id,
                                confidence_threshold=confidence,
                                force_regenerate=force_regen
                            )

                            st.success(f"✅ Extracted {result['entity_count']} entities!")

                            # Display results grouped by type
                            if result['entities_by_type']:
                                st.markdown("### Extracted Entities")

                                for entity_type, entities in result['entities_by_type'].items():
                                    with st.expander(f"**{entity_type.replace('_', ' ').title()}** ({len(entities)} entities)"):
                                        for i, entity in enumerate(entities[:20], 1):  # Show first 20
                                            st.markdown(f"{i}. **{entity['entity_text']}** (confidence: {entity['confidence']:.2f})")
                                            if entity.get('context'):
                                                st.caption(f"Context: {entity['context'][:100]}...")

                                        if len(entities) > 20:
                                            st.caption(f"... and {len(entities) - 20} more")

                            # Show metadata
                            st.caption(f"Model: {result.get('model', 'N/A')} | Generated: {result.get('generated_at', 'N/A')}")

                        except Exception as e:
                            st.error(f"Error extracting entities: {str(e)}")
            else:
                st.info("No documents available. Add documents first.")

        # Tab 2: View entities for a document
        with tabs[1]:
            st.subheader("📋 View Extracted Entities")
            st.write("Browse entities that have been extracted from a document")

            # Document selection
            doc_options = {f"{doc.title} ({doc_id[:12]}...)": doc_id
                          for doc_id, doc in kb.documents.items()}

            if doc_options:
                selected_doc = st.selectbox(
                    "Select document:",
                    options=list(doc_options.keys()),
                    key="view_doc_select"
                )

                # Entity type filter
                entity_type_filter = st.multiselect(
                    "Filter by entity type (optional):",
                    options=[
                        "hardware", "memory_address", "instruction",
                        "person", "company", "product", "concept"
                    ],
                    key="view_entity_type_filter"
                )

                if st.button("📋 Load Entities"):
                    doc_id = doc_options[selected_doc]

                    try:
                        entities = kb.get_entities(
                            doc_id=doc_id,
                            entity_types=entity_type_filter if entity_type_filter else None
                        )

                        if entities:
                            st.success(f"Found {len(entities)} entities")

                            # Group by type
                            entities_by_type = {}
                            for entity in entities:
                                entity_type = entity['entity_type']
                                if entity_type not in entities_by_type:
                                    entities_by_type[entity_type] = []
                                entities_by_type[entity_type].append(entity)

                            # Display grouped
                            for entity_type, ents in sorted(entities_by_type.items()):
                                with st.expander(f"**{entity_type.replace('_', ' ').title()}** ({len(ents)} entities)", expanded=True):
                                    entity_data = []
                                    for ent in ents:
                                        entity_data.append({
                                            "Entity": ent['entity_text'],
                                            "Confidence": f"{ent['confidence']:.2f}",
                                            "Context": ent.get('context', '')[:80] + "..." if ent.get('context') else ""
                                        })

                                    df = pd.DataFrame(entity_data)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No entities found for this document. Extract entities first.")

                    except Exception as e:
                        st.error(f"Error loading entities: {str(e)}")
            else:
                st.info("No documents available.")

        # Tab 3: Search entities
        with tabs[2]:
            st.subheader("🔎 Search Entities")
            st.write("Search for entities across all documents")

            col1, col2 = st.columns([3, 1])
            with col1:
                search_query = st.text_input(
                    "Search for entity:",
                    placeholder="e.g., VIC-II, SID, LDA, Commodore...",
                    key="entity_search_query"
                )
            with col2:
                max_results = st.number_input(
                    "Max results:",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key="entity_search_max"
                )

            # Entity type filter
            entity_type_filter = st.multiselect(
                "Filter by entity type (optional):",
                options=[
                    "hardware", "memory_address", "instruction",
                    "person", "company", "product", "concept"
                ],
                key="search_entity_type_filter"
            )

            if st.button("🔍 Search") and search_query:
                with st.spinner("Searching entities..."):
                    try:
                        results = kb.search_entities(
                            query=search_query,
                            entity_types=entity_type_filter if entity_type_filter else None,
                            max_results=max_results
                        )

                        if results:
                            st.success(f"Found {len(results)} matching entities across {len(set(r['doc_id'] for r in results))} documents")

                            # Display results
                            for i, result in enumerate(results, 1):
                                doc = kb.documents.get(result['doc_id'])
                                doc_title = doc.title if doc else result['doc_id']

                                with st.expander(f"{i}. **{result['entity_text']}** in *{doc_title}*"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Type", result['entity_type'].replace('_', ' ').title())
                                    with col2:
                                        st.metric("Confidence", f"{result['confidence']:.2f}")
                                    with col3:
                                        st.metric("Occurrences", result.get('occurrence_count', 1))

                                    if result.get('context'):
                                        st.markdown("**Context:**")
                                        st.caption(result['context'])
                        else:
                            st.info(f"No entities found matching '{search_query}'")

                    except Exception as e:
                        st.error(f"Error searching entities: {str(e)}")

        # Tab 4: Bulk extraction
        with tabs[3]:
            st.subheader("⚡ Bulk Entity Extraction")
            st.write("Extract entities from multiple documents at once")

            # Count docs without entities
            docs_without_entities = []
            docs_with_entities = []

            for doc_id, doc in kb.documents.items():
                try:
                    entities = kb.get_entities(doc_id)
                    if entities:
                        docs_with_entities.append(doc_id)
                    else:
                        docs_without_entities.append(doc_id)
                except (KeyError, AttributeError, TypeError):
                    docs_without_entities.append(doc_id)

            st.info(f"📊 {len(docs_with_entities)} documents have entities | {len(docs_without_entities)} documents need extraction")

            col1, col2 = st.columns(2)
            with col1:
                confidence = st.slider(
                    "Confidence threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.05,
                    key="bulk_confidence"
                )
            with col2:
                max_docs = st.number_input(
                    "Max documents (0 = all):",
                    min_value=0,
                    max_value=len(kb.documents),
                    value=0,
                    help="Limit bulk extraction for testing"
                )

            skip_existing = st.checkbox(
                "Skip documents with existing entities",
                value=True,
                help="Only extract from documents without entities"
            )

            force_regen = st.checkbox(
                "Force regenerate all",
                value=False,
                help="Re-extract even if entities exist"
            )

            if st.button("⚡ Start Bulk Extraction", type="primary"):
                with st.spinner("Running bulk entity extraction..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        result = kb.extract_entities_bulk(
                            confidence_threshold=confidence,
                            max_documents=max_docs if max_docs > 0 else None,
                            force_regenerate=force_regen,
                            skip_existing=skip_existing
                        )

                        progress_bar.progress(1.0)

                        st.success("✅ Bulk extraction completed!")

                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Processed", result['processed'])
                        with col2:
                            st.metric("Skipped", result['skipped'])
                        with col3:
                            st.metric("Failed", result['failed'])

                        st.metric("Total Entities Extracted", f"{result['total_entities']:,}")

                        if result['failed_docs']:
                            with st.expander("⚠️ Failed Documents"):
                                for doc_id in result['failed_docs'][:20]:
                                    doc = kb.documents.get(doc_id)
                                    st.write(f"- {doc.title if doc else doc_id}")

                                if len(result['failed_docs']) > 20:
                                    st.caption(f"... and {len(result['failed_docs']) - 20} more")

                        st.caption(f"Processing time: {result.get('processing_time', 'N/A')}")

                        # Offer to refresh stats
                        if st.button("🔄 Refresh Statistics"):
                            st.rerun()

                    except Exception as e:
                        progress_bar.progress(0)
                        st.error(f"Error during bulk extraction: {str(e)}")
                        st.exception(e)

# ========== ENTITY RELATIONSHIPS PAGE ==========
elif page == "🔗 Relationship Graph":
    st.title("🔗 Entity Relationships")
    st.write("Explore how entities (hardware, instructions, concepts) co-occur and relate to each other in documents")

    # Show database statistics
    cursor = kb.db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM entity_relationships")
    total_relationships = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT entity1_text) + COUNT(DISTINCT entity2_text) FROM entity_relationships")
    unique_entities = cursor.fetchone()[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔗 Total Relationships", f"{total_relationships:,}")
    with col2:
        st.metric("🏷️ Unique Entities", f"{unique_entities:,}")

    st.markdown("---")

    # Operation tabs
    tabs = st.tabs([
        "🔍 Extract Relationships",
        "📊 View Relationships",
        "🔎 Search Entity Pair",
        "⚡ Bulk Extraction"
    ])

    # Tab 1: Extract relationships from single document
    with tabs[0]:
        st.subheader("🔍 Extract Entity Relationships from Document")
        st.write("Analyze entity co-occurrence patterns in a document")

        # Document selection
        doc_options = {f"{doc.title} ({doc_id[:12]}...)": doc_id
                      for doc_id, doc in kb.documents.items()}

        if doc_options:
            selected_doc = st.selectbox(
                "Select document:",
                options=list(doc_options.keys()),
                key="extract_rel_doc_select"
            )

            col1, col2 = st.columns(2)
            with col1:
                confidence = st.slider(
                    "Entity confidence threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.05,
                    key="rel_confidence",
                    help="Only use entities with confidence above this threshold"
                )
            with col2:
                force_regen = st.checkbox(
                    "Force regenerate",
                    value=False,
                    key="rel_force_regen",
                    help="Re-extract even if relationships already exist"
                )

            if st.button("🚀 Extract Relationships", type="primary"):
                doc_id = doc_options[selected_doc]

                with st.spinner(f"Extracting relationships from {selected_doc}..."):
                    try:
                        result = kb.extract_entity_relationships(
                            doc_id=doc_id,
                            min_confidence=confidence,
                            force_regenerate=force_regen
                        )

                        st.success(f"✅ Extracted {result['relationship_count']} relationships!")

                        # Display top relationships
                        if result['relationships']:
                            st.markdown("### Top Relationships by Strength")

                            for i, rel in enumerate(result['relationships'][:20], 1):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.markdown(
                                        f"{i}. **{rel['entity1']}** ({rel['entity1_type']}) ↔ "
                                        f"**{rel['entity2']}** ({rel['entity2_type']})"
                                    )
                                    if rel.get('context'):
                                        st.caption(f"Context: {rel['context'][:150]}...")

                                with col2:
                                    st.metric("Strength", f"{rel['strength']:.2f}")

                            if len(result['relationships']) > 20:
                                st.caption(f"... and {len(result['relationships']) - 20} more")

                    except Exception as e:
                        st.error(f"Error extracting relationships: {str(e)}")
        else:
            st.info("No documents available. Add documents first.")

    # Tab 2: View relationships for an entity
    with tabs[1]:
        st.subheader("📊 View Entity Relationships")
        st.write("Show all entities related to a specific entity")

        entity_search = st.text_input(
            "Entity name:",
            placeholder="e.g., VIC-II, SID, sprite, LDA",
            key="entity_rel_search"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_strength = st.slider(
                "Minimum relationship strength:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="min_strength"
            )
        with col2:
            max_results = st.number_input(
                "Max results:",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="max_rel_results"
            )

        if st.button("🔍 Find Relationships", key="find_rel_btn") and entity_search:
            with st.spinner(f"Searching for entities related to '{entity_search}'..."):
                try:
                    relationships = kb.get_entity_relationships(
                        entity_text=entity_search,
                        min_strength=min_strength,
                        max_results=max_results
                    )

                    if relationships:
                        st.success(f"Found {len(relationships)} related entities")

                        # Display results
                        for i, rel in enumerate(relationships, 1):
                            with st.expander(
                                f"{i}. **{rel['related_entity']}** ({rel['related_type']}) - "
                                f"Strength: {rel['strength']:.2f}"
                            ):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Relationship Strength", f"{rel['strength']:.2f}")
                                with col2:
                                    st.metric("Documents", rel['doc_count'])

                                if rel.get('context'):
                                    st.markdown("**Context:**")
                                    st.caption(rel['context'])
                    else:
                        st.info(f"No relationships found for '{entity_search}'")

                except Exception as e:
                    st.error(f"Error finding relationships: {str(e)}")

    # Tab 3: Search by entity pair
    with tabs[2]:
        st.subheader("🔎 Search by Entity Pair")
        st.write("Find documents that contain both entities")

        col1, col2 = st.columns(2)
        with col1:
            entity1 = st.text_input(
                "First entity:",
                placeholder="e.g., VIC-II",
                key="entity1_search"
            )
        with col2:
            entity2 = st.text_input(
                "Second entity:",
                placeholder="e.g., sprite",
                key="entity2_search"
            )

        max_docs = st.number_input(
            "Max documents:",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="max_pair_docs"
        )

        if st.button("🔍 Search Pair", key="search_pair_btn") and entity1 and entity2:
            with st.spinner(f"Searching for documents with '{entity1}' AND '{entity2}'..."):
                try:
                    results = kb.search_by_entity_pair(
                        entity1=entity1,
                        entity2=entity2,
                        max_results=max_docs
                    )

                    if results:
                        st.success(f"Found {len(results)} documents containing both entities")

                        for i, doc_result in enumerate(results, 1):
                            doc = kb.documents.get(doc_result['doc_id'])
                            if doc:
                                with st.expander(
                                    f"{i}. {doc.title} - "
                                    f"'{entity1}': {doc_result['entity1_count']}, "
                                    f"'{entity2}': {doc_result['entity2_count']}"
                                ):
                                    st.markdown(f"**Document ID:** `{doc_result['doc_id']}`")
                                    st.markdown(f"**File:** {doc.file_path}")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(f"'{entity1}' occurrences", doc_result['entity1_count'])
                                    with col2:
                                        st.metric(f"'{entity2}' occurrences", doc_result['entity2_count'])

                                    # Show context samples
                                    if doc_result.get('contexts'):
                                        st.markdown("**Context Samples:**")
                                        for ctx in doc_result['contexts'][:3]:
                                            st.caption(f"• {ctx}")
                    else:
                        st.info(f"No documents found containing both '{entity1}' and '{entity2}'")

                except Exception as e:
                    st.error(f"Error searching entity pair: {str(e)}")

    # Tab 4: Bulk extraction
    with tabs[3]:
        st.subheader("⚡ Bulk Relationship Extraction")
        st.write("Extract entity relationships from multiple documents at once")

        col1, col2 = st.columns(2)
        with col1:
            bulk_confidence = st.slider(
                "Entity confidence threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="bulk_rel_confidence"
            )
        with col2:
            max_docs_bulk = st.number_input(
                "Max documents to process:",
                min_value=1,
                max_value=500,
                value=50,
                step=10,
                key="max_bulk_docs",
                help="Process this many documents"
            )

        skip_existing = st.checkbox(
            "Skip documents with existing relationships",
            value=True,
            key="skip_existing_rels",
            help="Don't re-extract from documents that already have relationships"
        )

        # Count documents that would be processed
        cursor.execute("""
            SELECT COUNT(DISTINCT doc_id)
            FROM document_entities
            WHERE doc_id NOT IN (SELECT DISTINCT first_seen_doc FROM entity_relationships)
        """)
        docs_to_process = cursor.fetchone()[0]

        st.info(f"📊 {docs_to_process} documents with entities have no relationships yet")

        if st.button("🚀 Start Bulk Extraction", type="primary", key="bulk_extract_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Extracting relationships from documents..."):
                try:
                    result = kb.extract_relationships_bulk(
                        min_confidence=bulk_confidence,
                        max_docs=max_docs_bulk if max_docs_bulk else None,
                        skip_existing=skip_existing
                    )

                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.success("✅ Bulk extraction complete!")

                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents Processed", result['processed'])
                    with col2:
                        st.metric("Total Relationships", result['total_relationships'])
                    with col3:
                        if result['processed'] > 0:
                            avg = result['total_relationships'] / result['processed']
                            st.metric("Avg per Document", f"{avg:.1f}")

                    if result.get('failed'):
                        st.warning(f"⚠️ {len(result['failed'])} documents failed")
                        with st.expander("Show failed documents"):
                            for doc_id in result['failed']:
                                doc = kb.documents.get(doc_id)
                                if doc:
                                    st.write(f"- {doc.title} (`{doc_id}`)")

                except Exception as e:
                    st.error(f"Error during bulk extraction: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

# ========== SEARCH PAGE ==========
elif page == "🔍 Search":
    st.title("🔍 Search Knowledge Base")

    # Show index status on search page
    if st.session_state.get('index_status') == 'building':
        st.info("⏳ **Search index is building in the background...**\n\n"
                "You can search now, but the first search may take a moment while the index completes. "
                "The sidebar shows the current status.")
        # Auto-refresh every 2 seconds while building
        time.sleep(2)
        st.rerun()
    elif st.session_state.get('index_status') == 'ready':
        if 'index_build_time' in st.session_state:
            st.success(f"✅ Search index ready! (Built in {st.session_state.index_build_time:.1f}s)")

    # Search mode selection
    search_mode = st.radio(
        "Search Mode",
        ["Keyword (FTS5)", "Semantic", "Hybrid"],
        horizontal=True
    )

    # Natural Language Translation toggle
    use_nl_translation = st.checkbox(
        "🤖 Use Natural Language Translation (AI-powered query parsing)",
        value=False,
        help="Enable AI to parse your natural language query and extract entities, keywords, and optimal search parameters"
    )

    # Use a form to enable Enter key submission
    with st.form(key="search_form", clear_on_submit=False):
        # Search input
        if use_nl_translation:
            query = st.text_area("Enter your natural language question:", "",
                               placeholder="e.g., 'find information about sprites on the VIC-II chip' or 'how does sound work on the C64?'",
                               height=80)
        else:
            query = st.text_input("Enter your search query:", "")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
        with col2:
            if search_mode == "Hybrid":
                semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.3, 0.1)
            else:
                semantic_weight = 0.3  # Default value when not shown
        with col3:
            tag_filter = st.text_input("Filter by tags (comma-separated)", "")

        # Search button (form submit button)
        search_submitted = st.form_submit_button("🔍 Search")

    # Execute search when form is submitted (button click or Enter key)
    if search_submitted and query:
        tags = [t.strip() for t in tag_filter.split(',') if t.strip()] if tag_filter else None

        # Natural Language Translation
        nl_result = None
        if use_nl_translation:
            try:
                with st.spinner("🤖 Translating natural language query..."):
                    nl_result = kb.translate_nl_query(query, confidence_threshold=0.7)

                # Display translation results in an expander
                with st.expander("🔍 Query Translation Results", expanded=True):
                    st.markdown(f"**Original Query:** \"{nl_result['original_query']}\"")
                    st.markdown(f"**Search Mode Recommendation:** `{nl_result['search_mode'].upper()}`")
                    st.markdown(f"**Confidence:** {nl_result['confidence']:.0%}")
                    st.markdown(f"**Reasoning:** {nl_result['reasoning']}")

                    if nl_result.get('search_terms'):
                        st.markdown(f"**Search Terms:** {', '.join(nl_result['search_terms'])}")

                    if nl_result.get('entities_found'):
                        st.markdown(f"**Entities Detected:** {len(nl_result['entities_found'])} found")
                        entity_data = []
                        for e in nl_result['entities_found'][:10]:  # Show top 10
                            entity_data.append({
                                'Entity': e['text'],
                                'Type': e['type'],
                                'Confidence': f"{e['confidence']:.0%}"
                            })
                        st.dataframe(pd.DataFrame(entity_data), use_container_width=True)
                        if len(nl_result['entities_found']) > 10:
                            st.caption(f"... and {len(nl_result['entities_found']) - 10} more entities")

                    if nl_result.get('facet_filters'):
                        st.markdown("**Facet Filters:**")
                        for facet_type, values in nl_result['facet_filters'].items():
                            st.markdown(f"- **{facet_type}:** {', '.join(values)}")

                    # Show if LLM was unavailable
                    if nl_result.get('confidence', 1.0) < 0.6:
                        st.warning("⚠️ LLM may be unavailable - using basic keyword extraction")

                # Override search mode based on translation recommendation
                if nl_result['search_mode'] == 'semantic':
                    search_mode = "Semantic"
                elif nl_result['search_mode'] == 'hybrid':
                    search_mode = "Hybrid"
                else:
                    search_mode = "Keyword (FTS5)"

                # Use search terms from translation if available, otherwise use original query
                if nl_result.get('search_terms'):
                    query = ' '.join(nl_result['search_terms'])
                # else keep original query

            except ValueError as e:
                st.error(f"Translation error: {e}\n\nMake sure LLM_PROVIDER and API key are configured.")
                st.stop()
            except Exception as e:
                st.error(f"Error during query translation: {e}")
                st.stop()

        # Create centered containers for progress bar and status text
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            # Wait for index to be ready if it's still building
            if st.session_state.get('index_status') == 'building':
                status_text.text("⏳ Waiting for search index to finish building...")
                progress_bar.progress(0.1)

                # Wait for the index thread to complete (with timeout)
                if st.session_state.get('index_thread'):
                    start_wait = time.time()
                    while st.session_state.get('index_status') == 'building':
                        time.sleep(0.5)
                        # Auto-refresh the page to update status
                        if time.time() - start_wait > 0.5:  # Check every 0.5 seconds
                            st.rerun()
                        # Timeout after 60 seconds
                        if time.time() - start_wait > 60:
                            st.error("⚠️ Index building timeout. Please try again.")
                            break

            # Update status - preparing search
            status_text.text(f"🔍 Preparing {search_mode.lower()} search...")
            progress_bar.progress(0.2)

            # Perform search based on mode
            if search_mode == "Keyword (FTS5)":
                status_text.text(f"🔎 Searching for '{query}' using FTS5...")
                progress_bar.progress(0.5)
                results = kb.search(query, max_results, tags)
            elif search_mode == "Semantic":
                if not kb.use_semantic:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("Semantic search is not enabled. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    status_text.text(f"🧠 Computing semantic embeddings for '{query}'...")
                    progress_bar.progress(0.5)
                    results = kb.semantic_search(query, max_results, tags)
            else:  # Hybrid
                if not kb.use_semantic:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    status_text.text(f"🔬 Running hybrid search (keyword + semantic) for '{query}'...")
                    progress_bar.progress(0.5)
                    results = kb.hybrid_search(query, max_results, tags, semantic_weight)

            # Update status - processing results
            status_text.text("📊 Processing results...")
            progress_bar.progress(0.9)

            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Search complete!")

            # Clear progress indicators after a brief moment
            import time
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

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
                    # Create unique key for this result using index, doc_id, and chunk_id
                    doc_id = result.get('doc_id', 'unknown')
                    chunk_id = result.get('chunk_id', 'none')
                    unique_key = f"{i}_{doc_id}_{chunk_id}"

                    with st.expander(f"**{i}. {result.get('title', 'Untitled')}**", expanded=(i==1)):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**File:** {result.get('filename', 'Unknown')}")
                            st.write(f"**Doc ID:** `{result.get('doc_id', 'Unknown')}`")
                            if result.get('chunk_id'):
                                st.write(f"**Chunk:** {result['chunk_id']}")
                            if result.get('page'):
                                st.write(f"**Page:** {result['page']}")

                        with col2:
                            score_key = 'score' if 'score' in result else 'similarity'
                            if score_key in result:
                                st.metric("Score", f"{result[score_key]:.4f}")

                            # File action buttons
                            st.markdown("**Actions:**")

                            # Add button to view file
                            if st.button("👁️ View File", key=f"view_file_{unique_key}", use_container_width=True):
                                if doc_id:
                                    st.session_state[f"show_viewer_{unique_key}"] = not st.session_state.get(f"show_viewer_{unique_key}", False)
                                    st.rerun()

                            # Add button to download file
                            if doc_id and doc_id in kb.documents:
                                doc = kb.documents[doc_id]
                                filepath = doc.filepath

                                # Check if file exists
                                if os.path.exists(filepath):
                                    try:
                                        with open(filepath, 'rb') as f:
                                            file_data = f.read()

                                        st.download_button(
                                            label="💾 Download",
                                            data=file_data,
                                            file_name=os.path.basename(filepath),
                                            mime="application/octet-stream",
                                            key=f"download_file_{unique_key}",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.caption(f"❌ Download error: {str(e)}")
                                else:
                                    st.caption("❌ File not found")

                        # File viewer (if toggled)
                        if st.session_state.get(f"show_viewer_{unique_key}", False):
                            st.markdown("---")
                            st.subheader("📄 File Viewer")

                            if doc_id and doc_id in kb.documents:
                                doc = kb.documents[doc_id]
                                filepath = doc.filepath

                                if os.path.exists(filepath):
                                    file_ext = os.path.splitext(filepath)[1].lower()

                                    try:
                                        if file_ext == '.pdf':
                                            # Display PDF using iframe
                                            with open(filepath, 'rb') as f:
                                                pdf_data = f.read()

                                            import base64
                                            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                                            st.markdown(pdf_display, unsafe_allow_html=True)

                                        elif file_ext == '.md':
                                            # Display markdown files with rendering
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()

                                            # Toggle for raw vs rendered view
                                            view_mode = st.radio(
                                                "View Mode",
                                                ["Rendered", "Raw Markdown"],
                                                horizontal=True,
                                                key=f"md_view_mode_{unique_key}"
                                            )

                                            if view_mode == "Rendered":
                                                # Render markdown with proper formatting
                                                st.markdown("### 📄 Rendered Markdown")
                                                with st.container():
                                                    st.markdown(content, unsafe_allow_html=False)
                                            else:
                                                # Show raw markdown
                                                st.markdown("### 📝 Raw Markdown")
                                                st.code(content, language='markdown', line_numbers=True)

                                        elif file_ext == '.txt':
                                            # Display text files in a scrollable container
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()

                                            st.markdown("### 📄 Text File Content")

                                            # Use expander for very long files
                                            line_count = content.count('\n') + 1
                                            if line_count > 50:
                                                st.info(f"📊 File contains {line_count} lines")

                                            # Display in a scrollable code block
                                            st.code(content, language='text', line_numbers=True)

                                        elif file_ext in ['.html', '.htm']:
                                            # Display HTML files
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                html_content = f.read()

                                            st.code(html_content, language='html')

                                        elif file_ext in ['.xlsx', '.xls']:
                                            # Display Excel files
                                            try:
                                                import pandas as pd
                                                df = pd.read_excel(filepath)
                                                st.dataframe(df, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Cannot preview Excel file: {str(e)}")

                                        else:
                                            st.info(f"Preview not available for {file_ext} files. Use the Download button to view externally.")

                                    except Exception as e:
                                        st.error(f"Error displaying file: {str(e)}")
                                else:
                                    st.error(f"❌ File not found: {filepath}")
                            else:
                                st.error("❌ Document not found in knowledge base")

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

# ========== ENTITY ANALYTICS PAGE ==========
elif page == "📈 Entity Analytics":
    st.title("📈 Entity Analytics Dashboard")

    # Get analytics data
    with st.spinner("Loading analytics data..."):
        analytics = kb.get_entity_analytics(time_range_days=365)

    # Summary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Unique Entities",
            f"{analytics['overall']['unique_entities']:,}",
            delta=f"{analytics['overall']['docs_with_entities']} docs"
        )
    with col2:
        st.metric(
            "Total Relationships",
            f"{analytics['relationship_stats']['total']:,}",
            delta=f"{len(analytics['relationship_stats']['by_type'])} types"
        )
    with col3:
        st.metric(
            "Avg Entities/Doc",
            f"{analytics['overall']['avg_entities_per_doc']:.1f}",
            delta=f"{len(analytics['entity_distribution'])} entity types"
        )
    with col4:
        if analytics['relationship_stats']['total'] > 0:
            st.metric(
                "Avg Relationship Strength",
                f"{analytics['relationship_stats']['avg_strength']:.2f}",
                delta="0.0-1.0 scale"
            )
        else:
            st.metric("Avg Relationship Strength", "N/A", delta="No relationships")

    st.markdown("---")

    # Export buttons
    st.subheader("📥 Export Data")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Export Entities (CSV)", key="export_ent_csv"):
            csv_data = kb.export_entities(format='csv', min_confidence=0.0)
            st.download_button(
                label="Download Entities CSV",
                data=csv_data,
                file_name="entities.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export Entities (JSON)", key="export_ent_json"):
            json_data = kb.export_entities(format='json', min_confidence=0.0)
            st.download_button(
                label="Download Entities JSON",
                data=json_data,
                file_name="entities.json",
                mime="application/json"
            )

    with col3:
        if st.button("Export Relationships (CSV)", key="export_rel_csv"):
            csv_data = kb.export_relationships(format='csv', min_strength=0.0)
            st.download_button(
                label="Download Relationships CSV",
                data=csv_data,
                file_name="relationships.csv",
                mime="text/csv"
            )

    with col4:
        if st.button("Export Relationships (JSON)", key="export_rel_json"):
            json_data = kb.export_relationships(format='json', min_strength=0.0)
            st.download_button(
                label="Download Relationships JSON",
                data=json_data,
                file_name="relationships.json",
                mime="application/json"
            )

    st.markdown("---")

    # Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Top Entities", "🔗 Relationships", "📈 Trends"])

    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        st.subheader("Entity Distribution by Type")

        if analytics['entity_distribution']:
            # Prepare data for bar chart
            import pandas as pd
            dist_df = pd.DataFrame([
                {'Type': k, 'Count': v}
                for k, v in analytics['entity_distribution'].items()
            ])

            # Bar chart
            st.bar_chart(dist_df.set_index('Type'))

            # Data table
            st.dataframe(
                dist_df.sort_values('Count', ascending=False).reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("No entities extracted yet. Use the Entity Extraction page to extract entities.")

    # ========== TAB 2: TOP ENTITIES ==========
    with tab2:
        st.subheader("Top 50 Entities")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            entity_types = ['All'] + list(analytics['entity_distribution'].keys())
            selected_type = st.selectbox("Filter by Type", entity_types)
        with col2:
            min_doc_count = st.number_input("Min Documents", min_value=0, value=1)
        with col3:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)

        # Filter entities
        filtered_entities = analytics['top_entities']
        if selected_type != 'All':
            filtered_entities = [e for e in filtered_entities if e['entity_type'] == selected_type]
        filtered_entities = [e for e in filtered_entities if e['doc_count'] >= min_doc_count]
        filtered_entities = [e for e in filtered_entities if e['avg_confidence'] >= min_confidence]

        if filtered_entities:
            # Convert to DataFrame
            import pandas as pd
            entity_df = pd.DataFrame(filtered_entities)
            entity_df['avg_confidence'] = entity_df['avg_confidence'].apply(lambda x: f"{x:.1%}")

            st.dataframe(
                entity_df.rename(columns={
                    'entity_text': 'Entity',
                    'entity_type': 'Type',
                    'doc_count': 'Documents',
                    'avg_confidence': 'Confidence',
                    'total_occurrences': 'Occurrences'
                }),
                use_container_width=True
            )

            st.caption(f"Showing {len(filtered_entities)} entities")
        else:
            st.warning("No entities match the selected filters.")

    # ========== TAB 3: RELATIONSHIPS ==========
    with tab3:
        st.subheader("Entity Relationships")

        if analytics['relationship_stats']['total'] > 0:
            # Relationship type distribution
            st.markdown("**Relationship Types Distribution**")
            if analytics['relationship_stats']['by_type']:
                import pandas as pd
                rel_type_data = [
                    {'Relationship Type': k, 'Count': v}
                    for k, v in analytics['relationship_stats']['by_type'].items()
                ]
                rel_type_df = pd.DataFrame(rel_type_data).sort_values('Count', ascending=False)
                st.dataframe(rel_type_df, use_container_width=True)

            st.markdown("---")

            # Network Graph Visualization
            st.markdown("**🕸️ Interactive Relationship Network**")

            col1, col2, col3 = st.columns(3)
            with col1:
                show_network = st.checkbox("Show Network Graph", value=True)
            with col2:
                max_nodes = st.slider("Max Nodes", 10, 100, 50, 5, help="Limit nodes for better performance")
            with col3:
                graph_min_strength = st.slider("Graph Min Strength", 0.0, 1.0, 0.3, 0.05, help="Filter weak relationships")

            if show_network:
                # Filter relationships for graph
                graph_rels = [
                    r for r in analytics['top_relationships'][:max_nodes]
                    if r['strength'] >= graph_min_strength
                ]

                if graph_rels:
                    try:
                        from pyvis.network import Network
                        import tempfile
                        import streamlit.components.v1 as components

                        # Create network
                        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
                        net.barnes_hut()

                        # Color scheme for entity types
                        type_colors = {
                            'hardware': '#FF6B6B',
                            'memory_address': '#4ECDC4',
                            'instruction': '#45B7D1',
                            'person': '#FFA07A',
                            'company': '#98D8C8',
                            'product': '#F7DC6F',
                            'concept': '#BB8FCE'
                        }

                        # Add nodes and edges
                        added_nodes = set()
                        for rel in graph_rels:
                            entity1 = rel['entity1']
                            entity2 = rel['entity2']
                            type1 = rel['entity1_type']
                            type2 = rel['entity2_type']
                            strength = rel['strength']
                            doc_count = rel['doc_count']

                            # Add entity1 node
                            if entity1 not in added_nodes:
                                net.add_node(
                                    entity1,
                                    label=entity1,
                                    color=type_colors.get(type1, '#CCCCCC'),
                                    title=f"{entity1} ({type1})",
                                    size=20
                                )
                                added_nodes.add(entity1)

                            # Add entity2 node
                            if entity2 not in added_nodes:
                                net.add_node(
                                    entity2,
                                    label=entity2,
                                    color=type_colors.get(type2, '#CCCCCC'),
                                    title=f"{entity2} ({type2})",
                                    size=20
                                )
                                added_nodes.add(entity2)

                            # Add edge
                            edge_width = strength * 5  # Scale edge width by strength
                            net.add_edge(
                                entity1,
                                entity2,
                                value=edge_width,
                                title=f"Strength: {strength:.2f}\nShared docs: {doc_count}",
                                color={'color': f'rgba(255,255,255,{strength})'}
                            )

                        # Generate and display
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                            net.save_graph(f.name)
                            with open(f.name, 'r', encoding='utf-8') as f2:
                                html_content = f2.read()
                            components.html(html_content, height=620)

                        # Legend
                        st.markdown("**Legend:**")
                        legend_cols = st.columns(len(type_colors))
                        for idx, (entity_type, color) in enumerate(type_colors.items()):
                            with legend_cols[idx]:
                                st.markdown(f"<span style='color:{color};'>●</span> {entity_type}", unsafe_allow_html=True)

                        st.caption(f"Showing {len(added_nodes)} nodes and {len(graph_rels)} edges. Drag nodes to explore. Hover for details.")

                    except Exception as e:
                        st.error(f"Failed to create network graph: {e}")
                        st.info("Network graph requires pyvis. Install with: pip install pyvis")
                else:
                    st.info("No relationships meet the minimum strength threshold for visualization.")

            st.markdown("---")

            # Top relationships
            st.markdown("**📊 Top 50 Relationships by Strength**")

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_strength = st.slider("Table Min Strength", 0.0, 1.0, 0.0, 0.05, key="table_strength")
            with col2:
                min_docs = st.number_input("Min Shared Documents", min_value=1, value=1)

            # Filter relationships
            filtered_rels = [
                r for r in analytics['top_relationships']
                if r['strength'] >= min_strength and r['doc_count'] >= min_docs
            ]

            if filtered_rels:
                import pandas as pd
                rel_df = pd.DataFrame(filtered_rels[:50])  # Limit to 50
                rel_df['strength'] = rel_df['strength'].apply(lambda x: f"{x:.2f}")

                st.dataframe(
                    rel_df.rename(columns={
                        'entity1': 'Entity 1',
                        'entity1_type': 'Type 1',
                        'entity2': 'Entity 2',
                        'entity2_type': 'Type 2',
                        'strength': 'Strength',
                        'doc_count': 'Shared Docs'
                    }),
                    use_container_width=True
                )

                st.caption(f"Showing {len(filtered_rels[:50])} relationships")
            else:
                st.warning("No relationships match the selected filters.")

        else:
            st.info("No relationships extracted yet. Use the Relationship Graph page to extract relationships.")

    # ========== TAB 4: TRENDS ==========
    with tab4:
        st.subheader("Entity Extraction Timeline")

        if analytics['extraction_timeline']:
            import pandas as pd
            timeline_df = pd.DataFrame(analytics['extraction_timeline'])

            # Line chart
            st.line_chart(timeline_df.set_index('date'))

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days", len(timeline_df))
            with col2:
                avg_per_day = timeline_df['count'].mean()
                st.metric("Avg Entities/Day", f"{avg_per_day:.1f}")
            with col3:
                max_day = timeline_df.loc[timeline_df['count'].idxmax()]
                st.metric("Peak Day", max_day['date'], delta=f"{max_day['count']} entities")

            # Raw data
            with st.expander("View Raw Data"):
                st.dataframe(timeline_df, use_container_width=True)
        else:
            st.info("No timeline data available. Entity extraction dates are tracked from document creation.")

# ========== DOCUMENT COMPARISON PAGE ==========
elif page == "📄 Document Comparison":
    st.title("📄 Document Comparison")
    st.markdown("Compare two documents side-by-side with similarity scoring, metadata differences, and content analysis.")

    # Document selection
    docs = kb.list_documents()
    if len(docs) < 2:
        st.warning("At least 2 documents are required for comparison.")
    else:
        doc_options = {f"{doc.title} ({doc.doc_id[:8]}...)": doc.doc_id for doc in docs}
        doc_labels = list(doc_options.keys())

        col1, col2 = st.columns(2)
        with col1:
            doc1_label = st.selectbox("📄 First Document", doc_labels, key="doc1")
            doc_id_1 = doc_options[doc1_label]
        with col2:
            doc2_label = st.selectbox("📄 Second Document", doc_labels, index=min(1, len(doc_labels)-1), key="doc2")
            doc_id_2 = doc_options[doc2_label]

        # Comparison type
        comparison_type = st.radio(
            "Comparison Type",
            ['full', 'metadata', 'content'],
            index=0,
            horizontal=True,
            help="Full: All comparisons | Metadata: Only metadata | Content: Only content and entities"
        )

        # Compare button
        if st.button("🔍 Compare Documents", type="primary"):
            if doc_id_1 == doc_id_2:
                st.error("Please select two different documents.")
            else:
                try:
                    with st.spinner("Comparing documents..."):
                        result = kb.compare_documents(doc_id_1, doc_id_2, comparison_type)

                    # Similarity Score Header
                    score = result['similarity_score']
                    st.markdown("---")
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        st.metric("Similarity Score", f"{score:.1%}",
                                 help="Cosine similarity based on TF-IDF (0% = completely different, 100% = identical)")

                    st.info(result['summary'])

                    # Tab layout for different comparison aspects
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Metadata", "🏷️ Tags", "🧠 Entities", "📝 Content Diff"])

                    # ========== TAB 1: METADATA ==========
                    with tab1:
                        st.subheader("Metadata Comparison")
                        meta = result['metadata_diff']

                        # Create comparison table
                        import pandas as pd
                        comparison_data = {
                            "Property": ["Title", "Filename", "File Type", "Total Pages", "Total Chunks"],
                            "Document 1": [
                                meta['title'][0],
                                meta['filename'][0],
                                meta['file_type'][0],
                                meta['total_pages'][0],
                                result['chunk_count'][0]
                            ],
                            "Document 2": [
                                meta['title'][1],
                                meta['filename'][1],
                                meta['file_type'][1],
                                meta['total_pages'][1],
                                result['chunk_count'][1]
                            ]
                        }
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # ========== TAB 2: TAGS ==========
                    with tab2:
                        st.subheader("Tag Comparison")

                        if meta['tags']:
                            tags = meta['tags']

                            if tags['common']:
                                st.markdown("**✅ Common Tags:**")
                                st.write(", ".join(tags['common']))

                            col1, col2 = st.columns(2)
                            with col1:
                                if tags['only_in_doc1']:
                                    st.markdown("**📌 Only in Document 1:**")
                                    st.write(", ".join(tags['only_in_doc1']))
                                else:
                                    st.info("No unique tags in Document 1")

                            with col2:
                                if tags['only_in_doc2']:
                                    st.markdown("**📌 Only in Document 2:**")
                                    st.write(", ".join(tags['only_in_doc2']))
                                else:
                                    st.info("No unique tags in Document 2")
                        else:
                            st.info("No tags found in either document.")

                    # ========== TAB 3: ENTITIES ==========
                    with tab3:
                        st.subheader("Entity Comparison")

                        if result.get('entity_comparison'):
                            ent = result['entity_comparison']

                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Document 1 Entities", ent['total_doc1'])
                            with col2:
                                st.metric("Common Entities", len(ent['common_entities']))
                            with col3:
                                st.metric("Document 2 Entities", ent['total_doc2'])

                            # Common entities
                            if ent['common_entities']:
                                st.markdown("**✅ Common Entities:**")
                                common_df = pd.DataFrame(
                                    [(text, etype) for text, etype in ent['common_entities']],
                                    columns=['Entity', 'Type']
                                )
                                st.dataframe(common_df, use_container_width=True, hide_index=True)

                            # Unique entities in two columns
                            st.markdown("---")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**📌 Unique to Document 1:**")
                                if ent['unique_to_doc1']:
                                    unique1_df = pd.DataFrame(
                                        [(text, etype) for text, etype in ent['unique_to_doc1']],
                                        columns=['Entity', 'Type']
                                    )
                                    st.dataframe(unique1_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No unique entities")

                            with col2:
                                st.markdown("**📌 Unique to Document 2:**")
                                if ent['unique_to_doc2']:
                                    unique2_df = pd.DataFrame(
                                        [(text, etype) for text, etype in ent['unique_to_doc2']],
                                        columns=['Entity', 'Type']
                                    )
                                    st.dataframe(unique2_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No unique entities")
                        else:
                            st.info("No entities extracted for these documents yet. Use Entity Extraction page to extract entities.")

                    # ========== TAB 4: CONTENT DIFF ==========
                    with tab4:
                        st.subheader("Content Differences")

                        if result.get('content_diff') and len(result['content_diff']) > 0:
                            st.markdown(f"**Showing first {min(len(result['content_diff']), 100)} diff lines** (out of {len(result['content_diff'])} total)")

                            # Display diff with syntax highlighting
                            diff_text = "\n".join(result['content_diff'][:100])
                            st.code(diff_text, language="diff")

                            if len(result['content_diff']) > 100:
                                st.info(f"Content diff truncated. Showing 100 of {len(result['content_diff'])} lines.")
                        else:
                            st.info("No content differences found or content comparison not performed.")

                except Exception as e:
                    st.error(f"Error comparing documents: {e}")

# ========== SYSTEM ANALYTICS PAGE ==========
elif page == "📉 System Analytics":
    st.title("📉 Search Analytics")

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

# ========== SETTINGS PAGE ==========
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")
    st.markdown("View MCP configuration, runtime settings, and environment variables.")

    # Helper function to find MCP config files
    def find_mcp_config_files():
        """Find potential MCP configuration files."""
        config_files = []

        # Claude Desktop config (Windows)
        claude_desktop_config = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude_desktop_config.json'
        if claude_desktop_config.exists():
            config_files.append(('Claude Desktop', str(claude_desktop_config)))

        # Claude Code config (.claude/settings.json in project)
        claude_code_config = Path.cwd() / '.claude' / 'settings.json'
        if claude_code_config.exists():
            config_files.append(('Claude Code (Project)', str(claude_code_config)))

        # Claude Code global config
        home = Path.home()
        claude_code_global = home / '.claude' / 'settings.json'
        if claude_code_global.exists():
            config_files.append(('Claude Code (Global)', str(claude_code_global)))

        return config_files

    # Helper function to read MCP config
    def read_mcp_config(config_path):
        """Read and parse MCP configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            return {'error': str(e)}

    # Helper function to find this server in MCP config
    def find_server_config(config):
        """Find tdz-c64-knowledge server configuration in MCP config."""
        if 'mcpServers' in config:
            for server_name, server_config in config['mcpServers'].items():
                if 'tdz-c64-knowledge' in server_name.lower() or 'c64' in server_name.lower():
                    return server_name, server_config
        return None, None

    # Create tabs for different settings sections
    tab1, tab2, tab3, tab4 = st.tabs(["📂 File Paths", "🔧 MCP Configuration", "🌍 Environment Variables", "⚡ Features & Capabilities"])

    # ========== TAB 1: FILE PATHS ==========
    with tab1:
        st.subheader("📂 File Paths")

        # Data directory
        st.markdown("**Data Directory:**")
        st.code(st.session_state.data_dir, language="text")

        # Database path
        db_path = Path(st.session_state.data_dir) / "knowledge_base.db"
        st.markdown("**Database Path:**")
        st.code(str(db_path), language="text")
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            st.caption(f"✅ Exists ({size_mb:.2f} MB)")
        else:
            st.caption("❌ Not found")

        # Embeddings path
        embeddings_path = Path(st.session_state.data_dir) / "embeddings.pkl"
        st.markdown("**Embeddings Path:**")
        st.code(str(embeddings_path), language="text")
        if embeddings_path.exists():
            size_mb = embeddings_path.stat().st_size / (1024 * 1024)
            st.caption(f"✅ Exists ({size_mb:.2f} MB)")
        else:
            st.caption("ℹ️ Not yet created (will be generated on first semantic search)")

        # MCP config files
        st.markdown("**MCP Configuration Files:**")
        config_files = find_mcp_config_files()
        if config_files:
            for config_type, config_path in config_files:
                st.code(f"{config_type}: {config_path}", language="text")
                st.caption(f"✅ Found")
        else:
            st.warning("No MCP configuration files found")
            st.caption("Expected locations:\n"
                      "- Claude Desktop: %APPDATA%\\Claude\\claude_desktop_config.json\n"
                      "- Claude Code: .claude/settings.json")

    # ========== TAB 2: MCP CONFIGURATION ==========
    with tab2:
        st.subheader("🔧 MCP Server Configuration")

        config_files = find_mcp_config_files()

        if not config_files:
            st.warning("No MCP configuration files found")
            st.info("To use this server with Claude Desktop or Claude Code, you need to configure it in the MCP settings.")

            st.markdown("**Example Configuration:**")
            example_config = {
                "mcpServers": {
                    "tdz-c64-knowledge": {
                        "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
                        "args": ["C:\\path\\to\\server.py"],
                        "env": {
                            "TDZ_DATA_DIR": str(st.session_state.data_dir),
                            "USE_FTS5": "1",
                            "USE_SEMANTIC_SEARCH": "1"
                        }
                    }
                }
            }
            st.json(example_config)
        else:
            # Read and display each config file
            for config_type, config_path in config_files:
                with st.expander(f"📄 {config_type} Configuration", expanded=True):
                    st.code(config_path, language="text")

                    config = read_mcp_config(config_path)

                    if 'error' in config:
                        st.error(f"Error reading config: {config['error']}")
                    else:
                        # Find our server in the config
                        server_name, server_config = find_server_config(config)

                        if server_config:
                            st.success(f"✅ Found server configuration: **{server_name}**")

                            # Display server configuration
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Command:**")
                                st.code(server_config.get('command', 'N/A'), language="text")

                                if 'args' in server_config:
                                    st.markdown("**Arguments:**")
                                    for arg in server_config['args']:
                                        st.code(arg, language="text")

                            with col2:
                                if 'env' in server_config:
                                    st.markdown("**Environment Variables:**")
                                    env_df = pd.DataFrame([
                                        {"Variable": k, "Value": v}
                                        for k, v in server_config['env'].items()
                                    ])
                                    st.dataframe(env_df, hide_index=True, use_container_width=True)

                            # Show full JSON
                            st.markdown("**Full Configuration:**")
                            st.json(server_config)
                        else:
                            st.warning("TDZ C64 Knowledge server not found in this configuration file")

                            # Show available servers
                            if 'mcpServers' in config:
                                st.markdown("**Available servers in this config:**")
                                for server_name in config['mcpServers'].keys():
                                    st.write(f"- {server_name}")

    # ========== TAB 3: ENVIRONMENT VARIABLES ==========
    with tab3:
        st.subheader("🌍 Runtime Environment Variables")

        # Key environment variables for this application
        important_vars = [
            'TDZ_DATA_DIR',
            'USE_FTS5',
            'USE_SEMANTIC_SEARCH',
            'USE_BM25',
            'ALLOWED_DOCS_DIRS',
            'REST_API_KEY',
            'MDSCRAPE_PATH',
            'EMBEDDING_CACHE_TTL',
            'ENTITY_CACHE_TTL',
            'MAX_CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'ANTHROPIC_API_KEY',
            'OPENAI_API_KEY'
        ]

        # Build dataframe of environment variables
        env_data = []
        for var in important_vars:
            value = os.environ.get(var, '')

            # Mask sensitive values
            if 'KEY' in var or 'SECRET' in var or 'PASSWORD' in var:
                if value:
                    value = '***' + value[-4:] if len(value) > 4 else '***'

            status = '✅ Set' if os.environ.get(var) else '❌ Not set'
            env_data.append({
                'Variable': var,
                'Value': value or '(not set)',
                'Status': status
            })

        df = pd.DataFrame(env_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Show all environment variables (optional expander)
        with st.expander("🔍 View All Environment Variables"):
            all_env = {k: v for k, v in os.environ.items()}
            # Mask sensitive values
            for key in all_env.keys():
                if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key or 'TOKEN' in key:
                    if all_env[key]:
                        all_env[key] = '***' + all_env[key][-4:] if len(all_env[key]) > 4 else '***'

            st.json(all_env, expanded=False)

    # ========== TAB 4: FEATURES & CAPABILITIES ==========
    with tab4:
        st.subheader("⚡ Features & Capabilities")

        # Run health check to get feature status
        try:
            health = kb.health_check()

            # Display overall status
            status_color = {
                'healthy': '🟢',
                'degraded': '🟡',
                'unhealthy': '🔴'
            }
            st.markdown(f"### {status_color.get(health['status'], '⚪')} System Status: **{health['status'].upper()}**")

            if health.get('issues'):
                st.warning("**Issues:**")
                for issue in health['issues']:
                    st.write(f"- {issue}")

            st.markdown("---")

            # Feature flags in columns
            col1, col2 = st.columns(2)

            features = health.get('features', {})

            with col1:
                st.markdown("**Search Features:**")
                st.write(f"{'✅' if features.get('fts5_enabled') else '❌'} FTS5 Full-Text Search")
                st.write(f"{'✅' if features.get('semantic_search_enabled') else '❌'} Semantic Search (Enabled)")
                st.write(f"{'✅' if features.get('semantic_search_available') else '❌'} Semantic Search (Available)")
                st.write(f"{'✅' if features.get('bm25_available') else '❌'} BM25 Ranking")

                st.markdown("**Document Processing:**")
                st.write(f"{'✅' if features.get('pdf_support') else '❌'} PDF Support")
                st.write(f"{'✅' if features.get('ocr_enabled') else '❌'} OCR (Optical Character Recognition)")

            with col2:
                st.markdown("**AI Features:**")
                st.write(f"{'✅' if features.get('entity_extraction_available') else '❌'} Entity Extraction")
                st.write(f"{'✅' if features.get('relationship_extraction_available') else '❌'} Relationship Mapping")
                st.write(f"{'✅' if features.get('rag_available') else '❌'} RAG Question Answering")

                st.markdown("**Web Features:**")
                st.write(f"{'✅' if features.get('web_scraping_available') else '❌'} Web Scraping")
                st.write(f"{'✅' if features.get('url_monitoring_available') else '❌'} URL Monitoring")

            st.markdown("---")

            # Database statistics
            st.markdown("**Database Statistics:**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.metric("Total Documents", health.get('database', {}).get('total_documents', 0))
                st.metric("Total Chunks", health.get('database', {}).get('total_chunks', 0))

            with stats_col2:
                st.metric("Total Entities", health.get('database', {}).get('total_entities', 0))
                st.metric("Total Relationships", health.get('database', {}).get('total_relationships', 0))

            with stats_col3:
                embeddings_count = features.get('embeddings_count', 0)
                embeddings_size = features.get('embeddings_size_mb', 0)
                st.metric("Embeddings", f"{embeddings_count:,}")
                if embeddings_size > 0:
                    st.caption(f"({embeddings_size:.2f} MB)")

            st.markdown("---")

            # Full health check JSON
            with st.expander("🔍 View Full Health Check Results"):
                st.json(health, expanded=False)

        except Exception as e:
            st.error(f"Error retrieving features and capabilities: {str(e)}")
            st.exception(e)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{get_full_version_string()}**")
st.sidebar.caption(f"Build Date: {__build_date__}")
st.sidebar.markdown("Built with ❤️ using Claude Code")
