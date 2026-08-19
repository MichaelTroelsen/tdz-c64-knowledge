"""Helpers shared by admin_gui.py and the per-page modules under admin_pages/.

Extracted so a page module can use them without importing admin_gui, which
would be circular: admin_gui imports the pages to dispatch to them.
"""
import threading
import time

import streamlit as st

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
