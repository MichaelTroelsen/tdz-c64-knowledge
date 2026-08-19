"""Admin GUI page: 🕸️ Graph Explorer.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st


def render(kb):
    st.title("🕸️ Knowledge Graph Explorer")
    st.write("Explore the entity relationship graph interactively: filter, color/size by metric, and trace paths between entities.")
    cursor = kb.db_conn.cursor()
    all_entity_types = [row[0] for row in cursor.execute(
        "SELECT DISTINCT entity_type FROM document_entities ORDER BY entity_type"
    ).fetchall()]
    st.markdown("**Filters**")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_types = st.multiselect("Entity Types", all_entity_types, default=[], help="Leave empty to include all types")
    with col2:
        min_occurrences = st.number_input("Min Occurrences", min_value=1, value=2, step=1, help="Minimum times an entity must appear to be included")
    with col3:
        min_strength = st.slider("Min Relationship Strength", 0.0, 1.0, 0.3, 0.05)
    st.markdown("**Display**")
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        color_by = st.selectbox("Color By", ["type", "community"])
    with col5:
        size_by = st.selectbox("Size By", ["pagerank", "occurrences", "degree"])
    with col6:
        max_nodes = st.slider("Max Nodes Shown", 10, 300, 100, 10)
    with col7:
        show_labels = st.checkbox("Show Labels", value=True)
    if st.button("🔄 Generate Graph", type="primary"):
        with st.spinner("Building knowledge graph..."):
            G = kb.build_knowledge_graph(
                entity_types=selected_types or None,
                min_occurrences=min_occurrences,
                min_relationship_strength=min_strength,
                use_cache=False
            )
            st.session_state.graph_explorer_G = G
    if "graph_explorer_G" in st.session_state:
        G = st.session_state.graph_explorer_G

        if G.number_of_nodes() == 0:
            st.info("No entities match the current filters. Try lowering Min Occurrences or widening the entity type selection.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔵 Nodes", f"{G.number_of_nodes():,}")
            with col2:
                st.metric("🔗 Edges", f"{G.number_of_edges():,}")
            with col3:
                import networkx as nx
                density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
                st.metric("📐 Density", f"{density:.4f}")

            with st.spinner("Rendering visualization..."):
                try:
                    import tempfile
                    import streamlit.components.v1 as components

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                        html_path = f.name
                    kb.visualize_knowledge_graph_pyvis(
                        G, output_path=html_path,
                        color_by=color_by, size_by=size_by,
                        show_labels=show_labels, max_nodes=max_nodes
                    )
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=770)

                    color_map = kb._get_color_map_for_graph(G, color_by)
                    if color_map:
                        st.markdown("**Legend:**")
                        legend_cols = st.columns(min(len(color_map), 6))
                        for idx, (key, color) in enumerate(color_map.items()):
                            with legend_cols[idx % len(legend_cols)]:
                                st.markdown(f"<span style='color:{color};'>●</span> {key}", unsafe_allow_html=True)

                    if G.number_of_nodes() > max_nodes:
                        st.caption(f"Showing top {max_nodes} of {G.number_of_nodes()} nodes by degree centrality. Increase 'Max Nodes Shown' to see more.")
                except Exception as e:
                    st.error(f"Failed to render graph: {e}")
                    st.info("Graph visualization requires pyvis. Install with: pip install pyvis")

            st.markdown("---")
            st.markdown(f"**🏆 Top Entities by {size_by.title()}**")
            sample_node = next(iter(G.nodes))
            if size_by == "pagerank" and 'pagerank' in G.nodes[sample_node]:
                ranked = sorted(G.nodes(data=True), key=lambda nd: nd[1].get('pagerank', 0), reverse=True)
            elif size_by == "degree":
                degree_cent = nx.degree_centrality(G)
                ranked = sorted(G.nodes(data=True), key=lambda nd: degree_cent.get(nd[0], 0), reverse=True)
            else:
                ranked = sorted(G.nodes(data=True), key=lambda nd: nd[1].get('occurrences', 0), reverse=True)

            top_rows = []
            for node, attrs in ranked[:10]:
                top_rows.append({
                    "Entity": node,
                    "Type": attrs.get('type', 'unknown'),
                    "Occurrences": attrs.get('occurrences', 0),
                    "Degree": G.degree(node)
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**🧭 Find Path Between Entities**")
            node_list = sorted(G.nodes())
            colp1, colp2, colp3 = st.columns([2, 2, 1])
            with colp1:
                entity1 = st.selectbox("From Entity", node_list, key="path_entity1")
            with colp2:
                entity2 = st.selectbox("To Entity", node_list, key="path_entity2", index=min(1, len(node_list) - 1))
            with colp3:
                st.write("")
                st.write("")
                find_path_clicked = st.button("Find Path")

            if find_path_clicked:
                if entity1 == entity2:
                    st.warning("Choose two different entities.")
                else:
                    path = kb.find_shortest_path(G, entity1, entity2, cache_result=False)
                    if path:
                        st.success(f"Path found ({len(path)} nodes):")
                        st.markdown(" → ".join(f"`{p}`" for p in path))
                        for a, b in zip(path, path[1:]):
                            edge = G.edges[a, b]
                            st.caption(f"`{a}` → `{b}`: strength={edge.get('weight', 0):.2f}, co-occurrences={edge.get('co_occurrences', 0)}")
                    else:
                        st.warning(f"No path exists between '{entity1}' and '{entity2}' with the current filters.")
    else:
        st.info("Set your filters above and click **Generate Graph** to build and explore the knowledge graph.")
