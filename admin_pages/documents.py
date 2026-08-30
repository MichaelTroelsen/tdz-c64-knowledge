"""Admin GUI page: 📚 Documents.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
from datetime import datetime
from pathlib import Path
import os
import streamlit as st
from admin_common import format_timestamp, trigger_background_reindex


def render(kb):
    st.title("📚 Document Management")
    with st.expander("➕ Add Documents", expanded=False):
        upload_tabs = st.tabs(["📄 Single Upload", "📦 Bulk Upload", "📁 Add by File Path"])

        # Tab 1: Single Upload
        with upload_tabs[0]:
            st.subheader("Upload Single Document")

            uploaded_file = st.file_uploader("Choose a file (PDF, TXT, MD, HTML, Excel, Word, PowerPoint, EPUB, CSV, JSON, XML, SID, or ZIP)", type=['pdf', 'txt', 'md', 'html', 'htm', 'xlsx', 'xls', 'asm', 'bas', 'inc', 's', 'sid', 'psid', 'rsid', 'zip', 'docx', 'pptx', 'epub', 'csv', 'json', 'xml'], key="single_upload")

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
                "Choose files (PDF, TXT, MD, HTML, Excel, Word, PowerPoint, EPUB, CSV, JSON, XML, SID, or ZIP)",
                type=['pdf', 'txt', 'md', 'html', 'htm', 'xlsx', 'xls', 'asm', 'bas', 'inc', 's', 'sid', 'psid', 'rsid', 'zip', 'docx', 'pptx', 'epub', 'csv', 'json', 'xml'],
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
