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
from datetime import datetime, timezone
from urllib.parse import quote
import pandas as pd
import threading
import time

# Suppress Streamlit ScriptRunContext warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# Add parent directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase
from admin_common import build_bm25_index_background, trigger_background_reindex, format_bytes, format_timestamp
from admin_pages import PAGES
from version import __build_date__, get_full_version_string

# Page configuration
st.set_page_config(
    page_title="TDZ C64 Knowledge Base Admin",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Background indexing function


# Initialize session state
if 'kb' not in st.session_state:
    data_dir = os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge"))
    st.session_state.kb = KnowledgeBase(data_dir)
    st.session_state.data_dir = data_dir
    st.session_state.index_status = "not_started"
    st.session_state.index_thread = None
    st.session_state.quick_added_files = []  # Track quick-added files from Archive Search
    st.session_state.downloaded_files_added = {}  # Track which downloaded files have been added to KB

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
    ["📊 Dashboard", "📚 Documents", "🌐 Web Scraping", "🌐 URL Monitoring", "🏷️ Tag Management", "🧠 Entity Extraction", "🔗 Relationship Graph", "📈 Entity Analytics", "🕸️ Graph Explorer", "📄 Document Comparison", "🔍 Search", "💾 Backup & Restore", "📉 System Analytics", "🛰️ MCP Monitor", "🔍 Archive Search", "⚙️ Settings"]
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


# ========== DASHBOARD PAGE ==========

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{get_full_version_string()}**")
st.sidebar.caption(f"Build Date: {__build_date__}")
st.sidebar.markdown("Built with ❤️ using Claude Code")


# The 16 page bodies live in admin_pages/ (R18). The radio above still
# drives navigation; this is the whole dispatch that used to be a
# 5,000-line if/elif chain.
PAGES[page](kb)
