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
    ["📊 Dashboard", "📚 Documents", "🏷️ Tag Management", "🔗 Relationship Graph", "🔍 Search", "💾 Backup & Restore", "📈 Analytics"]
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
    with st.expander("➕ Add Documents", expanded=False):
        upload_tabs = st.tabs(["📄 Single Upload", "📦 Bulk Upload"])

        # Tab 1: Single Upload
        with upload_tabs[0]:
            st.subheader("Upload Single Document")

            uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=['pdf', 'txt'], key="single_upload")

            col1, col2 = st.columns(2)
            with col1:
                doc_title = st.text_input("Title (optional)", "", key="single_title")
            with col2:
                doc_tags = st.text_input("Tags (comma-separated)", "", key="single_tags")

            if st.button("Add Document", key="add_single") and uploaded_file:
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

        # Tab 2: Bulk Upload
        with upload_tabs[1]:
            st.subheader("Upload Multiple Documents")
            st.write("📌 **Drag and drop multiple files or click to browse**")

            uploaded_files = st.file_uploader(
                "Choose files (PDF or TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                key="bulk_upload"
            )

            bulk_tags = st.text_input("Tags for all documents (comma-separated)", "", key="bulk_tags")

            if st.button("📦 Add All Documents", key="add_bulk") and uploaded_files:
                tags = [t.strip() for t in bulk_tags.split(',') if t.strip()]

                progress_bar = st.progress(0)
                status_text = st.empty()

                added = 0
                failed = 0

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

                        # Save uploaded file temporarily
                        temp_path = Path(st.session_state.data_dir) / f"temp_{uploaded_file.name}"
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())

                        # Add to knowledge base (use filename as title)
                        doc = kb.add_document(str(temp_path), None, tags)

                        # Clean up temp file
                        temp_path.unlink()

                        added += 1
                    except Exception as e:
                        failed += 1
                        st.warning(f"⚠️ Failed to add {uploaded_file.name}: {str(e)}")

                progress_bar.empty()
                status_text.empty()

                if added > 0:
                    st.success(f"✅ Successfully added {added} document(s)")
                if failed > 0:
                    st.error(f"❌ Failed to add {failed} document(s)")

                if added > 0:
                    st.rerun()

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
                    if st.button(f"👁️ Preview", key=f"preview_{doc.doc_id}"):
                        st.session_state[f"show_preview_{doc.doc_id}"] = not st.session_state.get(f"show_preview_{doc.doc_id}", False)
                        st.rerun()

                    if st.button(f"🔗 Relationships", key=f"rels_{doc.doc_id}"):
                        st.session_state[f"show_relationships_{doc.doc_id}"] = not st.session_state.get(f"show_relationships_{doc.doc_id}", False)
                        st.rerun()

                    if st.button(f"🗑️ Delete", key=f"del_{doc.doc_id}"):
                        if kb.remove_document(doc.doc_id):
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
                            # Show preview options
                            preview_col1, preview_col2 = st.columns([2, 1])

                            with preview_col1:
                                chunk_count = len(full_doc['chunks'])
                                preview_chunks = st.slider(
                                    "Number of chunks to preview",
                                    min_value=1,
                                    max_value=min(chunk_count, 10),
                                    value=min(3, chunk_count),
                                    key=f"preview_slider_{doc.doc_id}"
                                )

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

# ========== RELATIONSHIP GRAPH PAGE ==========
elif page == "🔗 Relationship Graph":
    st.title("🔗 Document Relationship Graph")

    st.write("**Visualize connections between documents**")
    st.write("Explore how documents relate to each other through references, prerequisites, and related content.")

    # Import visualization libraries
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        import tempfile
        import os
    except ImportError:
        st.error("📦 Visualization libraries not installed. Run: `pip install pyvis networkx`")
        st.stop()

    # Filters
    st.markdown("---")
    st.subheader("🎛️ Filters")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        # Get all tags
        all_tags = set()
        for doc in kb.documents.values():
            all_tags.update(doc.tags)

        if all_tags:
            selected_tags = st.multiselect(
                "Filter by tags:",
                options=sorted(list(all_tags)),
                default=None
            )
        else:
            selected_tags = None
            st.info("No tags found")

    with filter_col2:
        selected_rel_types = st.multiselect(
            "Relationship types:",
            options=["related", "references", "prerequisite", "sequel"],
            default=["related", "references", "prerequisite", "sequel"]
        )

    # Get graph data
    try:
        graph_data = kb.get_relationship_graph(
            tags=selected_tags if selected_tags else None,
            relationship_types=selected_rel_types if selected_rel_types else None
        )

        # Display stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📄 Documents", graph_data['stats']['total_nodes'])
        with col2:
            st.metric("🔗 Relationships", graph_data['stats']['total_edges'])
        with col3:
            if graph_data['stats']['relationship_types']:
                st.metric("🏷️ Types", len(graph_data['stats']['relationship_types']))
            else:
                st.metric("🏷️ Types", 0)

        if graph_data['stats']['total_nodes'] == 0:
            st.warning("⚠️ No documents with relationships found. Create relationships in the Documents page.")
        elif graph_data['stats']['total_edges'] == 0:
            st.warning("⚠️ No relationships match the selected filters.")
        else:
            st.markdown("---")

            # Visualization options
            with st.expander("⚙️ Visualization Options", expanded=False):
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    physics_enabled = st.checkbox("Enable physics simulation", value=True)
                    show_arrows = st.checkbox("Show relationship direction", value=True)
                    node_size = st.slider("Node size", 10, 50, 25)

                with viz_col2:
                    layout = st.selectbox(
                        "Layout algorithm:",
                        ["hierarchical", "force_atlas", "barnes_hut", "repulsion"]
                    )
                    edge_color = st.color_picker("Edge color", "#808080")

            # Create network graph
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")

            # Set physics
            if physics_enabled:
                if layout == "hierarchical":
                    net.set_options("""
                    {
                        "physics": {
                            "enabled": true,
                            "hierarchicalRepulsion": {
                                "centralGravity": 0.0,
                                "springLength": 100,
                                "springConstant": 0.01,
                                "nodeDistance": 120,
                                "damping": 0.09
                            },
                            "solver": "hierarchicalRepulsion"
                        },
                        "layout": {
                            "hierarchical": {
                                "enabled": true,
                                "direction": "UD",
                                "sortMethod": "directed"
                            }
                        }
                    }
                    """)
                else:
                    net.toggle_physics(True)
            else:
                net.toggle_physics(False)

            # Add nodes
            node_colors = {
                'related': '#4CAF50',
                'references': '#2196F3',
                'prerequisite': '#FF9800',
                'sequel': '#9C27B0'
            }

            for node in graph_data['nodes']:
                # Color nodes based on their outgoing relationship types
                color = "#808080"  # Default gray
                net.add_node(
                    node['id'],
                    label=node['label'][:50],  # Truncate long titles
                    title=node['title'],
                    size=node_size,
                    color=color
                )

            # Add edges
            for edge in graph_data['edges']:
                color = node_colors.get(edge['type'], edge_color)
                net.add_edge(
                    edge['from'],
                    edge['to'],
                    title=edge['title'],
                    label=edge['label'],
                    color=color,
                    arrows='to' if show_arrows else ''
                )

            # Generate and display graph
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as f:
                html_path = f.name
                net.save_graph(html_path)

            # Read and display
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Clean up
            os.unlink(html_path)

            # Display in streamlit
            st.markdown("### 📊 Interactive Graph")
            st.write("💡 **Tip:** Click and drag nodes to explore. Hover for details. Scroll to zoom.")
            components.html(html_content, height=620, scrolling=False)

            # Legend
            st.markdown("---")
            st.markdown("### 🎨 Legend")
            legend_cols = st.columns(4)

            with legend_cols[0]:
                st.markdown(f"🟢 **related** - General relationship")
            with legend_cols[1]:
                st.markdown(f"🔵 **references** - Cites/references")
            with legend_cols[2]:
                st.markdown(f"🟠 **prerequisite** - Must read first")
            with legend_cols[3]:
                st.markdown(f"🟣 **sequel** - Continuation")

            # Export option
            st.markdown("---")
            if st.button("📥 Export Graph Data as JSON"):
                import json
                graph_json = json.dumps(graph_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=graph_json,
                    file_name=f"relationship_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

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
st.sidebar.markdown("**TDZ C64 Knowledge Base v2.8.0**")
st.sidebar.markdown("Built with ❤️ using Claude Code")
