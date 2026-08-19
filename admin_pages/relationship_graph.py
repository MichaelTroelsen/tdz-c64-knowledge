"""Admin GUI page: 🔗 Relationship Graph.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st


def render(kb):
    st.title("🔗 Entity Relationships")
    st.write("Explore how entities (hardware, instructions, concepts) co-occur and relate to each other in documents")
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
    tabs = st.tabs([
        "🔍 Extract Relationships",
        "📊 View Relationships",
        "🔎 Search Entity Pair",
        "⚡ Bulk Extraction"
    ])
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
