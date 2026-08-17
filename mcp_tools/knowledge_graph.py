"""Knowledge Graph Tool Handlers (v2.24.0 - Phase 1, Task 1.4), plus the
Phase-3 graph-analytics tools (pagerank/communities/centrality/stats)
that operate on the same graph. Split out of handlers.py (R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


def handle_build_knowledge_graph(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    use_cache = arguments.get("use_cache", True)

    try:
        import networkx as nx

        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            use_cache=use_cache
        )

        # Build comprehensive output
        output = "# Knowledge Graph Built\n\n"
        output += f"**Graph Statistics:**\n"
        output += f"- Nodes (entities): {G.number_of_nodes()}\n"
        output += f"- Edges (relationships): {G.number_of_edges()}\n"

        if G.number_of_nodes() > 0:
            density = nx.density(G)
            output += f"- Density: {density:.4f}\n"
            output += f"- Connected components: {nx.number_connected_components(G)}\n"

            # Show sample nodes
            sample_nodes = list(G.nodes())[:10]
            output += f"\n**Sample Entities (first 10):**\n"
            for node in sample_nodes:
                node_data = G.nodes[node]
                output += f"- {node} (type: {node_data.get('type', 'unknown')}, "
                output += f"occurrences: {node_data.get('occurrences', 0)})\n"

            # Show sample edges
            if G.number_of_edges() > 0:
                sample_edges = list(G.edges(data=True))[:10]
                output += f"\n**Sample Relationships (first 10):**\n"
                for e1, e2, data in sample_edges:
                    output += f"- {e1} ↔ {e2} (strength: {data.get('weight', 0):.3f})\n"
        else:
            output += "\n**No entities found matching the criteria.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error building knowledge graph: {str(e)}")]


def handle_compute_graph_metrics(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    store_results = arguments.get("store_results", True)

    try:
        metrics = kb.compute_graph_metrics(
            G=None,  # Will build new graph
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            store_results=store_results
        )

        # Build output
        output = "# Graph Metrics Computed\n\n"

        # Graph statistics
        output += f"**Graph Statistics:**\n"
        stats = metrics['graph_stats']
        output += f"- Nodes: {stats['nodes']}\n"
        output += f"- Edges: {stats['edges']}\n"
        output += f"- Density: {stats['density']:.4f}\n"
        output += f"- Connected components: {stats['connected_components']}\n"
        output += f"- Communities detected: {metrics['num_communities']}\n"
        output += f"- Computed: {stats['computed_at']}\n"

        # Top entities by PageRank
        if metrics['pagerank']:
            output += f"\n**Top 15 Entities by PageRank:**\n"
            top_pr = sorted(metrics['pagerank'].items(), key=lambda x: x[1], reverse=True)[:15]
            for rank, (entity, score) in enumerate(top_pr, 1):
                community = metrics['communities'].get(entity, 'N/A')
                output += f"{rank:2d}. {entity:30s} (PR: {score:.6f}, Community: {community})\n"

        # Top entities by Betweenness
        if metrics['betweenness']:
            output += f"\n**Top 10 Bridge Entities (Betweenness):**\n"
            top_bt = sorted(metrics['betweenness'].items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (entity, score) in enumerate(top_bt, 1):
                output += f"{rank:2d}. {entity:30s} (Betweenness: {score:.6f})\n"

        if store_results:
            output += f"\n**Metrics stored to database for {len(metrics['pagerank'])} entities.**\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error computing graph metrics: {str(e)}")]


def handle_get_entity_metrics(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    metric_types = arguments.get("metric_types")

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        result = kb.get_entity_metrics(entity_text, metric_types)

        if not result['found']:
            return [TextContent(type="text", text=f"No metrics found for entity: '{entity_text}'\n\nThis entity may not be in the knowledge graph, or metrics have not been computed yet. Run 'compute_graph_metrics' first.")]

        # Build output
        output = f"# Entity Metrics: {entity_text}\n\n"
        output += f"**Entity Type:** {result['entity_type']}\n"
        output += f"**Computed:** {result['computed_date']}\n\n"

        output += f"**Metrics:**\n"
        metrics = result['metrics']

        if 'pagerank' in metrics and metrics['pagerank'] is not None:
            output += f"- PageRank: {metrics['pagerank']:.6f} (importance score)\n"

        if 'betweenness' in metrics and metrics['betweenness'] is not None:
            output += f"- Betweenness: {metrics['betweenness']:.6f} (bridge score)\n"

        if 'degree' in metrics and metrics['degree'] is not None:
            output += f"- Degree: {metrics['degree']:.6f} (connection score)\n"

        if 'community' in metrics and metrics['community'] is not None:
            output += f"- Community ID: {metrics['community']}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error retrieving entity metrics: {str(e)}")]


def handle_find_entity_path(kb, name: str, arguments: dict) -> list[TextContent]:
    entity1 = arguments.get("entity1")
    entity2 = arguments.get("entity2")
    max_path_length = arguments.get("max_path_length", 6)
    store_result = arguments.get("store_result", True)

    if not entity1 or not entity2:
        return [TextContent(type="text", text="Error: both entity1 and entity2 are required")]

    try:
        result = kb.find_shortest_path(
            entity1=entity1,
            entity2=entity2,
            G=None,  # Will build graph
            max_path_length=max_path_length,
            store_result=store_result
        )

        if result is None:
            return [TextContent(type="text", text=f"Error: Could not compute path. One or both entities may not exist in the graph.")]

        # Build output
        output = f"# Path: {entity1} → {entity2}\n\n"

        if result['exists']:
            output += f"**Path Found:** {result['length']} edges\n\n"
            output += f"**Path:**\n"
            output += " → ".join(result['path']) + "\n\n"

            if result['relationships']:
                output += f"**Relationship Details:**\n"
                for i, rel in enumerate(result['relationships'], 1):
                    output += f"{i}. {rel['from']} → {rel['to']}\n"
                    output += f"   - Strength: {rel['weight']:.3f}\n"
                    output += f"   - Co-occurrences: {rel['doc_count']} documents\n"

            if store_result:
                output += f"\n**Path stored to database.**\n"
        else:
            output += f"**No path found.**\n\n"
            output += f"The entities '{entity1}' and '{entity2}' are in different connected components "
            output += f"of the knowledge graph (no relationship path exists between them).\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error finding path: {str(e)}")]


def handle_get_entity_community(kb, name: str, arguments: dict) -> list[TextContent]:
    entity_text = arguments.get("entity_text")
    max_members = arguments.get("max_members", 50)

    if not entity_text:
        return [TextContent(type="text", text="Error: entity_text is required")]

    try:
        # First, get the entity's community ID
        entity_metrics = kb.get_entity_metrics(entity_text)

        if not entity_metrics['found']:
            return [TextContent(type="text", text=f"Entity '{entity_text}' not found in graph metrics.\n\nRun 'compute_graph_metrics' first to detect communities.")]

        if 'community' not in entity_metrics['metrics'] or entity_metrics['metrics']['community'] is None:
            return [TextContent(type="text", text=f"No community information for '{entity_text}'.\n\nRun 'compute_graph_metrics' first.")]

        community_id = entity_metrics['metrics']['community']

        # Get all entities in the same community
        cursor = kb.db_conn.cursor()
        rows = cursor.execute("""
            SELECT entity_text, entity_type, pagerank, degree_centrality
            FROM graph_metrics
            WHERE community_id = ?
            ORDER BY pagerank DESC
            LIMIT ?
        """, (community_id, max_members)).fetchall()

        # Build output
        output = f"# Community {community_id}\n\n"
        output += f"**Anchor Entity:** {entity_text} ({entity_metrics['entity_type']})\n"
        output += f"**Total Members:** {len(rows)}\n\n"

        output += f"**Community Members (ranked by PageRank):**\n"
        for i, (ent_text, ent_type, pr, degree) in enumerate(rows, 1):
            marker = " ← anchor" if ent_text == entity_text else ""
            output += f"{i:2d}. {ent_text:30s} (type: {ent_type}, PR: {pr:.6f}){marker}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting community: {str(e)}")]


def handle_get_top_entities(kb, name: str, arguments: dict) -> list[TextContent]:
    metric = arguments.get("metric", "pagerank")
    limit = arguments.get("limit", 10)
    entity_types = arguments.get("entity_types")

    try:
        # Build query based on metric
        metric_column_map = {
            'pagerank': 'pagerank',
            'betweenness': 'betweenness_centrality',
            'degree': 'degree_centrality'
        }

        metric_column = metric_column_map.get(metric, 'pagerank')

        cursor = kb.db_conn.cursor()

        # Build query
        query = f"""
            SELECT entity_text, entity_type, pagerank, betweenness_centrality,
                   degree_centrality, community_id
            FROM graph_metrics
            WHERE {metric_column} IS NOT NULL
        """
        params = []

        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += f" ORDER BY {metric_column} DESC LIMIT ?"
        params.append(limit)

        rows = cursor.execute(query, params).fetchall()

        if not rows:
            return [TextContent(type="text", text=f"No entities found.\n\nRun 'compute_graph_metrics' first to generate metrics.")]

        # Build output
        metric_display = metric.replace('_', ' ').title()
        output = f"# Top {len(rows)} Entities by {metric_display}\n\n"

        if entity_types:
            output += f"**Filtered by types:** {', '.join(entity_types)}\n\n"

        output += f"**Ranking:**\n"
        for rank, (ent_text, ent_type, pr, betw, deg, comm) in enumerate(rows, 1):
            output += f"{rank:2d}. {ent_text:30s}\n"
            output += f"    Type: {ent_type}, Community: {comm}\n"
            output += f"    PageRank: {pr:.6f}, Betweenness: {betw:.6f}, Degree: {deg:.6f}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting top entities: {str(e)}")]


def handle_visualize_graph(kb, name: str, arguments: dict) -> list[TextContent]:
    output_path = arguments.get("output_path", "knowledge_graph.html")
    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    min_relationship_strength = arguments.get("min_relationship_strength", 0.3)
    color_by = arguments.get("color_by", "entity_type")
    size_by = arguments.get("size_by", "pagerank")
    physics_enabled = arguments.get("physics_enabled", True)
    height = arguments.get("height", "800px")
    width = arguments.get("width", "100%")

    try:
        # Generate visualization
        saved_path = kb.visualize_knowledge_graph(
            G=None,  # Will build graph
            output_path=output_path,
            entity_types=entity_types,
            min_occurrences=min_occurrences,
            min_relationship_strength=min_relationship_strength,
            color_by=color_by,
            size_by=size_by,
            physics_enabled=physics_enabled,
            height=height,
            width=width
        )

        if not saved_path:
            return [TextContent(type="text", text="Error: Could not generate visualization (empty graph or error occurred).")]

        # Build output message
        output = "# Knowledge Graph Visualization Generated\n\n"
        output += f"**File saved to:** `{saved_path}`\n\n"

        output += f"**Visualization Settings:**\n"
        output += f"- Nodes colored by: {color_by}\n"
        output += f"- Node size based on: {size_by}\n"
        output += f"- Physics simulation: {'Enabled' if physics_enabled else 'Disabled'}\n"

        if entity_types:
            output += f"- Filtered to types: {', '.join(entity_types)}\n"

        output += f"- Min occurrences: {min_occurrences}\n"
        output += f"- Min relationship strength: {min_relationship_strength}\n\n"

        output += f"**Interactive Features:**\n"
        output += f"- Hover over nodes to see entity details and metrics\n"
        output += f"- Hover over edges to see relationship strength\n"
        output += f"- Click and drag nodes to rearrange\n"
        output += f"- Zoom with mouse wheel\n"
        output += f"- Pan by dragging the background\n\n"

        output += f"Open the HTML file in your web browser to explore the interactive visualization!\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating visualization: {str(e)}")]


def handle_analyze_graph_pagerank(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    top_n = arguments.get("top_n", 20)
    alpha = arguments.get("alpha", 0.85)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate PageRank
        pagerank = kb.analyze_pagerank(G, alpha=alpha)

        # Format results
        output = f"# PageRank Analysis\n\n"
        output += f"**Total entities:** {len(pagerank)}\n"
        output += f"**Damping factor (alpha):** {alpha}\n\n"
        output += f"## Top {top_n} Entities by PageRank\n\n"

        for i, (entity, score) in enumerate(list(pagerank.items())[:top_n], 1):
            node_data = G.nodes[entity]
            output += f"{i}. **{entity}**\n"
            output += f"   - PageRank: {score:.6f}\n"
            output += f"   - Type: {node_data['type']}\n"
            output += f"   - Occurrences: {node_data['occurrences']}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing PageRank: {str(e)}")]


def handle_detect_graph_communities(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    algorithm = arguments.get("algorithm", "louvain")

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Detect communities
        communities = kb.detect_communities(G, algorithm=algorithm)
        num_communities = len(set(communities.values()))

        # Format results
        output = f"# Community Detection\n\n"
        output += f"**Algorithm:** {algorithm}\n"
        output += f"**Total entities:** {len(communities)}\n"
        output += f"**Communities detected:** {num_communities}\n\n"

        # Group entities by community
        community_groups = {}
        for entity, comm_id in communities.items():
            if comm_id not in community_groups:
                community_groups[comm_id] = []
            community_groups[comm_id].append(entity)

        # Show communities sorted by size
        sorted_communities = sorted(community_groups.items(), key=lambda x: len(x[1]), reverse=True)

        output += "## Communities (by size)\n\n"
        for i, (comm_id, members) in enumerate(sorted_communities[:10], 1):
            output += f"### Community {comm_id} ({len(members)} members)\n\n"
            # Show first 10 members
            for entity in members[:10]:
                node_type = G.nodes[entity].get('type', 'unknown')
                output += f"- {entity} (type: {node_type})\n"
            if len(members) > 10:
                output += f"- ... and {len(members) - 10} more\n"
            output += "\n"

        if len(sorted_communities) > 10:
            output += f"\n... and {len(sorted_communities) - 10} more communities\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error detecting communities: {str(e)}")]


def handle_calculate_graph_centrality(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)
    top_n = arguments.get("top_n", 10)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate centrality
        centrality = kb.calculate_centrality(G)

        # Format results
        output = f"# Centrality Analysis\n\n"
        output += f"**Total entities:** {G.number_of_nodes()}\n\n"

        # Show top entities for each centrality measure
        for measure_name, measure_values in centrality.items():
            sorted_values = sorted(measure_values.items(), key=lambda x: x[1], reverse=True)

            output += f"## Top {top_n} by {measure_name.capitalize()} Centrality\n\n"
            output += f"*{measure_name.capitalize()} measures "
            if measure_name == 'betweenness':
                output += "entities that bridge different parts of the graph*\n\n"
            elif measure_name == 'closeness':
                output += "entities that are close to all other entities*\n\n"
            elif measure_name == 'degree':
                output += "entities with many direct connections*\n\n"

            for i, (entity, score) in enumerate(sorted_values[:top_n], 1):
                node_data = G.nodes[entity]
                output += f"{i}. **{entity}**\n"
                output += f"   - Score: {score:.6f}\n"
                output += f"   - Type: {node_data['type']}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating centrality: {str(e)}")]


def handle_get_graph_statistics(kb, name: str, arguments: dict) -> list[TextContent]:
    try:
        import networkx as nx
    except ImportError:
        return [TextContent(type="text", text="NetworkX not installed. Run: pip install networkx>=3.0")]

    entity_types = arguments.get("entity_types")
    min_occurrences = arguments.get("min_occurrences", 2)

    try:
        # Build graph
        G = kb.build_knowledge_graph(
            entity_types=entity_types,
            min_occurrences=min_occurrences
        )

        if G.number_of_nodes() == 0:
            return [TextContent(type="text", text="No entities found for graph construction.")]

        # Calculate statistics
        density = nx.density(G)
        num_components = nx.number_connected_components(G)

        # Get degree distribution
        degrees = [d for _, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        # Format results
        output = f"# Knowledge Graph Statistics\n\n"
        output += f"## Basic Metrics\n\n"
        output += f"- **Nodes (entities):** {G.number_of_nodes()}\n"
        output += f"- **Edges (relationships):** {G.number_of_edges()}\n"
        output += f"- **Density:** {density:.4f} (0 = sparse, 1 = complete)\n"
        output += f"- **Connected components:** {num_components}\n\n"

        # Show largest connected component
        if num_components > 0:
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            output += f"## Connected Components\n\n"
            output += f"- Largest: {len(components[0])} nodes ({len(components[0])/G.number_of_nodes()*100:.1f}%)\n"
            if len(components) > 1:
                output += f"- Second largest: {len(components[1])} nodes\n"
            output += f"- Isolated nodes: {sum(1 for c in components if len(c) == 1)}\n\n"

        # Degree statistics
        output += f"## Degree Distribution\n\n"
        output += f"- Average degree: {avg_degree:.2f}\n"
        output += f"- Maximum degree: {max_degree}\n"
        output += f"- Minimum degree: {min_degree}\n\n"

        # Entity type distribution
        type_counts = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        output += f"## Entity Types\n\n"
        for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            output += f"- {etype}: {count} ({count/G.number_of_nodes()*100:.1f}%)\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting graph statistics: {str(e)}")]


# Phase 2: Visualization Tools


HANDLERS_KNOWLEDGE_GRAPH = {
    "build_knowledge_graph": handle_build_knowledge_graph,
    "compute_graph_metrics": handle_compute_graph_metrics,
    "get_entity_metrics": handle_get_entity_metrics,
    "find_entity_path": handle_find_entity_path,
    "get_entity_community": handle_get_entity_community,
    "get_top_entities": handle_get_top_entities,
    "visualize_graph": handle_visualize_graph,
    "analyze_graph_pagerank": handle_analyze_graph_pagerank,
    "detect_graph_communities": handle_detect_graph_communities,
    "calculate_graph_centrality": handle_calculate_graph_centrality,
    "get_graph_statistics": handle_get_graph_statistics,
}
