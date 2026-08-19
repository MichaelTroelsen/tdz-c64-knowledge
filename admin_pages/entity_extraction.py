"""Admin GUI page: 🧠 Entity Extraction.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import pandas as pd
import streamlit as st


def render(kb):
    st.title("🧠 Named Entity Extraction")
    st.write("Extract and explore technical entities from C64 documentation using AI")
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
