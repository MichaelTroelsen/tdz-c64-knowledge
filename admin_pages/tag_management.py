"""Admin GUI page: 🏷️ Tag Management.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import pandas as pd
import streamlit as st


def render(kb):
    st.title("🏷️ Tag Management")
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
