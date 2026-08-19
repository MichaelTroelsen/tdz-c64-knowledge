"""Admin GUI page: 📄 Document Comparison.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st


def render(kb):
    st.title("📄 Document Comparison")
    st.markdown("Compare two documents side-by-side with similarity scoring, metadata differences, and content analysis.")
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
