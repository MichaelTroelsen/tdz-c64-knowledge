"""Admin GUI page: 📈 Entity Analytics.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st


def render(kb):
    st.title("📈 Entity Analytics Dashboard")
    with st.spinner("Loading analytics data..."):
        analytics = kb.get_entity_analytics(time_range_days=365)
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Top Entities", "🔗 Relationships", "📈 Trends"])
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
