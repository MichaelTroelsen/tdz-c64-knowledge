"""Knowledge-graph construction, graph analytics and graph visualisation.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from typing import Dict
from typing import List
from typing import Optional
import json


class GraphMixin:

    def clear_graph_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached graphs from database.

        Args:
            older_than_days: Only clear caches older than this many days (None = clear all)

        Returns:
            Number of caches cleared
        """
        from datetime import datetime, timedelta

        cursor = self.db_conn.cursor()

        if older_than_days is None:
            # Clear all caches
            cursor.execute("DELETE FROM graph_cache")
        else:
            # Clear old caches
            cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
            cursor.execute("""
                DELETE FROM graph_cache WHERE created_date < ?
            """, (cutoff_date,))

        deleted = cursor.rowcount
        self.db_conn.commit()

        self.logger.info(f"Cleared {deleted} cached graphs")
        return deleted

    def compute_graph_metrics(self, G=None, entity_types: Optional[list[str]] = None,
                              min_occurrences: int = 2,
                              min_relationship_strength: float = 0.3,
                              store_results: bool = True) -> dict:
        """
        Compute comprehensive graph analysis metrics.

        Computes:
        - PageRank centrality (importance based on connections)
        - Betweenness centrality (bridge nodes)
        - Degree centrality (connection count)
        - Community detection (Louvain method)

        Args:
            G: Pre-built NetworkX graph (None = build new graph)
            entity_types: Filter to specific entity types
            min_occurrences: Minimum entity occurrences for graph building
            min_relationship_strength: Minimum relationship strength
            store_results: Save metrics to database

        Returns:
            {
                'pagerank': {entity: score, ...},
                'betweenness': {entity: score, ...},
                'degree': {entity: score, ...},
                'communities': {entity: community_id, ...},
                'num_communities': int,
                'graph_stats': {...}
            }
        """
        import networkx as nx
        from datetime import datetime

        # Build or use provided graph
        if G is None:
            self.logger.info("Building knowledge graph for analysis...")
            G = self.build_knowledge_graph(
                entity_types=entity_types,
                min_occurrences=min_occurrences,
                min_relationship_strength=min_relationship_strength
            )

        if G.number_of_nodes() == 0:
            self.logger.warning("Empty graph - no metrics to compute")
            return {
                'pagerank': {},
                'betweenness': {},
                'degree': {},
                'communities': {},
                'num_communities': 0,
                'graph_stats': {}
            }

        self.logger.info(f"Computing graph metrics for {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        results = {}

        # 1. PageRank - Measures entity importance based on connections
        try:
            self.logger.debug("Computing PageRank centrality...")
            pagerank = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-6)
            results['pagerank'] = pagerank
            top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"Top 5 PageRank entities: {[f'{e}({s:.4f})' for e, s in top_pr]}")
        except Exception as e:
            self.logger.error(f"PageRank computation failed: {e}")
            results['pagerank'] = {}

        # 2. Betweenness Centrality - Measures entities that bridge communities
        try:
            self.logger.debug("Computing betweenness centrality...")
            betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
            results['betweenness'] = betweenness
            top_bt = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"Top 5 betweenness entities: {[f'{e}({s:.4f})' for e, s in top_bt]}")
        except Exception as e:
            self.logger.error(f"Betweenness computation failed: {e}")
            results['betweenness'] = {}

        # 3. Degree Centrality - Measures connection count
        try:
            self.logger.debug("Computing degree centrality...")
            degree = nx.degree_centrality(G)
            results['degree'] = degree
            top_deg = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"Top 5 degree entities: {[f'{e}({s:.4f})' for e, s in top_deg]}")
        except Exception as e:
            self.logger.error(f"Degree computation failed: {e}")
            results['degree'] = {}

        # 4. Community Detection - Louvain method
        try:
            self.logger.debug("Detecting communities (Louvain method)...")
            # Import community detection
            try:
                import community.community_louvain as community_louvain
                communities = community_louvain.best_partition(G, weight='weight')
                results['communities'] = communities
                results['num_communities'] = len(set(communities.values()))
                self.logger.info(f"Detected {results['num_communities']} communities")

                # Show community sizes
                comm_sizes = {}
                for entity, comm_id in communities.items():
                    comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1
                top_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.info(f"Top 5 community sizes: {top_comms}")

            except ImportError:
                self.logger.warning("python-louvain not installed - using greedy modularity communities instead")
                from networkx.algorithms import community as nx_community
                communities_list = nx_community.greedy_modularity_communities(G, weight='weight')
                # Convert to dict format
                communities = {}
                for idx, comm in enumerate(communities_list):
                    for entity in comm:
                        communities[entity] = idx
                results['communities'] = communities
                results['num_communities'] = len(communities_list)
                self.logger.info(f"Detected {results['num_communities']} communities (greedy modularity)")

        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            results['communities'] = {}
            results['num_communities'] = 0

        # Graph statistics
        results['graph_stats'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_connected_components(G),
            'computed_at': datetime.now().isoformat()
        }

        # Store results in database
        if store_results and results['pagerank']:
            try:
                self._store_graph_metrics(results)
                self.logger.info("Stored graph metrics to database")
            except Exception as e:
                self.logger.error(f"Failed to store graph metrics: {e}")

        return results

    def _store_graph_path(self, entity1: str, entity2: str, path: list, length: int) -> None:
        """Store computed shortest path to database."""
        from datetime import datetime
        import json
        import hashlib

        cursor = self.db_conn.cursor()

        # Generate path_id
        path_id = hashlib.md5(f"{entity1}_{entity2}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Check if path already exists
        cursor.execute("""
            DELETE FROM graph_paths
            WHERE entity1 = ? AND entity2 = ?
        """, (entity1, entity2))

        # Insert new path
        cursor.execute("""
            INSERT INTO graph_paths
            (path_id, entity1, entity2, path_length, path_nodes, path_weight, computed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            path_id,
            entity1,
            entity2,
            length,
            json.dumps(path),
            None,  # path_weight can be computed later if needed
            datetime.now().isoformat()
        ))

        self.db_conn.commit()
        self.logger.debug(f"Stored path: {entity1} -> {entity2} (length {length})")

    def get_entity_metrics(self, entity_text: str, metric_types: Optional[list[str]] = None) -> dict:
        """
        Retrieve stored graph metrics for an entity.

        Args:
            entity_text: Entity to get metrics for
            metric_types: List of metric types to retrieve (None = all)
                         Options: 'pagerank', 'betweenness', 'degree', 'community'

        Returns:
            {
                'entity': str,
                'entity_type': str,
                'metrics': {
                    'pagerank': float,
                    'betweenness': float,
                    'degree': float,
                    'community': int
                },
                'computed_date': str,
                'found': bool
            }
        """
        cursor = self.db_conn.cursor()

        # Query metrics table
        row = cursor.execute("""
            SELECT entity_type, pagerank, betweenness_centrality,
                   degree_centrality, community_id, computed_date
            FROM graph_metrics
            WHERE entity_text = ?
        """, (entity_text,)).fetchone()

        if not row:
            return {
                'entity': entity_text,
                'entity_type': None,
                'metrics': {},
                'computed_date': None,
                'found': False
            }

        entity_type, pagerank, betweenness, degree, community, computed_date = row

        # Build metrics dict (filter by metric_types if provided)
        all_metrics = {
            'pagerank': pagerank,
            'betweenness': betweenness,
            'degree': degree,
            'community': community
        }

        if metric_types:
            metrics = {k: v for k, v in all_metrics.items() if k in metric_types}
        else:
            metrics = all_metrics

        return {
            'entity': entity_text,
            'entity_type': entity_type,
            'metrics': metrics,
            'computed_date': computed_date,
            'found': True
        }

    def visualize_knowledge_graph(self, G=None,
                                  output_path: str = "knowledge_graph.html",
                                  entity_types: Optional[list[str]] = None,
                                  min_occurrences: int = 2,
                                  min_relationship_strength: float = 0.3,
                                  color_by: str = "entity_type",
                                  size_by: str = "pagerank",
                                  highlight_communities: bool = False,
                                  physics_enabled: bool = True,
                                  height: str = "800px",
                                  width: str = "100%") -> str:
        """
        Generate interactive HTML visualization of the knowledge graph using PyVis.

        Args:
            G: Pre-built NetworkX graph (None = build new graph)
            output_path: Path to save HTML file
            entity_types: Filter to specific entity types
            min_occurrences: Minimum entity occurrences for graph building
            min_relationship_strength: Minimum relationship strength
            color_by: Node coloring scheme: 'entity_type', 'community', or 'uniform'
            size_by: Node sizing metric: 'pagerank', 'degree', 'betweenness', or 'uniform'
            highlight_communities: Add community borders/grouping
            physics_enabled: Enable physics simulation for layout
            height: Visualization height (CSS format)
            width: Visualization width (CSS format)

        Returns:
            Path to generated HTML file

        Examples:
            # Basic visualization
            path = kb.visualize_knowledge_graph()

            # Color by community, size by PageRank
            path = kb.visualize_knowledge_graph(
                color_by='community',
                size_by='pagerank',
                highlight_communities=True
            )

            # Hardware entities only
            path = kb.visualize_knowledge_graph(
                entity_types=['hardware'],
                min_occurrences=3
            )
        """
        from pyvis.network import Network
        import networkx as nx
        from pathlib import Path

        # Build graph if not provided
        if G is None:
            self.logger.info("Building knowledge graph for visualization...")
            G = self.build_knowledge_graph(
                entity_types=entity_types,
                min_occurrences=min_occurrences,
                min_relationship_strength=min_relationship_strength
            )

        if G.number_of_nodes() == 0:
            self.logger.warning("Empty graph - cannot visualize")
            return ""

        self.logger.info(f"Visualizing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Create PyVis network
        net = Network(height=height, width=width, directed=False, notebook=False)

        # Configure physics
        if physics_enabled:
            net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.05)
        else:
            net.toggle_physics(False)

        # Get metrics if using them for sizing or coloring
        metrics = None
        if size_by in ['pagerank', 'degree', 'betweenness'] or color_by == 'community':
            # Try to load from database first
            cursor = self.db_conn.cursor()
            stored_metrics = cursor.execute("SELECT COUNT(*) FROM graph_metrics").fetchone()[0]

            if stored_metrics == 0:
                self.logger.info("Computing graph metrics for visualization...")
                metrics = self.compute_graph_metrics(G=G, store_results=False)
            else:
                # Load from database
                self.logger.info("Loading stored metrics for visualization...")
                metrics = {
                    'pagerank': {},
                    'betweenness': {},
                    'degree': {},
                    'communities': {}
                }
                rows = cursor.execute("""
                    SELECT entity_text, pagerank, betweenness_centrality,
                           degree_centrality, community_id
                    FROM graph_metrics
                """).fetchall()

                for entity, pr, betw, deg, comm in rows:
                    if entity in G.nodes():
                        metrics['pagerank'][entity] = pr
                        metrics['betweenness'][entity] = betw or 0
                        metrics['degree'][entity] = deg
                        metrics['communities'][entity] = comm

        # Define color scheme for entity types
        entity_type_colors = {
            'hardware': '#FF6B6B',      # Red
            'memory_address': '#4ECDC4', # Teal
            'instruction': '#45B7D1',    # Blue
            'person': '#FFA07A',         # Light salmon
            'company': '#98D8C8',        # Mint
            'product': '#F7DC6F',        # Yellow
            'concept': '#BB8FCE',        # Purple
            'unknown': '#95A5A6'         # Gray
        }

        # Generate community colors if needed
        community_colors = {}
        if color_by == 'community' and metrics:
            import random
            random.seed(42)  # Consistent colors
            unique_communities = set(metrics['communities'].values()) if metrics['communities'] else set()
            colors = ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(len(unique_communities))]
            community_colors = dict(zip(sorted(unique_communities), colors))

        # Calculate node sizes if using metric
        node_sizes = {}
        if size_by != 'uniform' and metrics:
            metric_values = metrics.get(size_by, {})
            if metric_values:
                max_val = max(metric_values.values()) if metric_values else 1
                min_val = min(metric_values.values()) if metric_values else 0

                for node in G.nodes():
                    val = metric_values.get(node, 0)
                    # Scale to size range 10-50
                    if max_val > min_val:
                        normalized = (val - min_val) / (max_val - min_val)
                        node_sizes[node] = 10 + (normalized * 40)
                    else:
                        node_sizes[node] = 20

        # Add nodes to PyVis network
        for node in G.nodes(data=True):
            node_id = node[0]
            node_data = node[1]

            # Determine color
            if color_by == 'entity_type':
                entity_type = node_data.get('type', 'unknown')
                color = entity_type_colors.get(entity_type, '#95A5A6')
            elif color_by == 'community' and metrics:
                comm_id = metrics['communities'].get(node_id, 0)
                color = community_colors.get(comm_id, '#95A5A6')
            else:  # uniform
                color = '#4ECDC4'

            # Determine size
            size = node_sizes.get(node_id, 20)

            # Create hover title with entity info
            title = f"<b>{node_id}</b><br>"
            title += f"Type: {node_data.get('type', 'unknown')}<br>"
            title += f"Occurrences: {node_data.get('occurrences', 0)}<br>"

            if metrics:
                if node_id in metrics.get('pagerank', {}):
                    title += f"PageRank: {metrics['pagerank'][node_id]:.6f}<br>"
                if node_id in metrics.get('betweenness', {}):
                    title += f"Betweenness: {metrics['betweenness'][node_id]:.6f}<br>"
                if node_id in metrics.get('degree', {}):
                    title += f"Degree: {metrics['degree'][node_id]:.6f}<br>"
                if node_id in metrics.get('communities', {}):
                    title += f"Community: {metrics['communities'][node_id]}"

            net.add_node(node_id, label=node_id, color=color, size=size, title=title)

        # Add edges to PyVis network
        for edge in G.edges(data=True):
            source = edge[0]
            target = edge[1]
            edge_data = edge[2]

            weight = edge_data.get('weight', 0.5)
            doc_count = edge_data.get('doc_count', 0)

            # Edge width based on weight
            width = 1 + (weight * 3)

            # Edge title
            title = f"Strength: {weight:.3f}<br>Documents: {doc_count}"

            net.add_edge(source, target, value=width, title=title)

        # Generate HTML
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            # Save to data directory if relative path
            output_path_obj = Path(self.data_dir) / output_path

        net.save_graph(str(output_path_obj))

        self.logger.info(f"Visualization saved to: {output_path_obj}")

        return str(output_path_obj)

    def visualize_knowledge_graph_3d(self, max_entities: int = 50,
                                     min_confidence: float = 0.7,
                                     output_path: str = "knowledge_graph_3d.html") -> str:
        """
        Create 3D interactive knowledge graph showing entities, relationships, and documents.

        Args:
            max_entities: Maximum number of entities to include (top by frequency)
            min_confidence: Minimum confidence for entities
            output_path: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        import plotly.graph_objects as go
        from pathlib import Path
        import networkx as nx
        import numpy as np

        cursor = self.db_conn.cursor()

        # Get top entities
        entities_data = cursor.execute("""
            SELECT entity_text, entity_type, COUNT(*) as frequency,
                   AVG(confidence) as avg_confidence
            FROM document_entities
            WHERE confidence >= ?
            GROUP BY entity_text, entity_type
            ORDER BY frequency DESC
            LIMIT ?
        """, (min_confidence, max_entities)).fetchall()

        if not entities_data:
            self.logger.warning("No entities found for 3D knowledge graph")
            return ""

        # Build graph
        G = nx.Graph()

        # Add entity nodes
        entity_map = {}
        for i, (name, entity_type, freq, conf) in enumerate(entities_data):
            G.add_node(f"entity_{i}", label=name, node_type='entity',
                      entity_type=entity_type, frequency=freq, confidence=conf)
            entity_map[name] = f"entity_{i}"

        # Get relationships between entities
        relationships = cursor.execute("""
            SELECT entity1_text, entity2_text, relationship_type, strength, doc_count
            FROM entity_relationships
            WHERE entity1_text IN ({}) AND entity2_text IN ({})
            AND strength >= 0.5
        """.format(','.join('?' * len(entity_map)), ','.join('?' * len(entity_map))),
                                     list(entity_map.keys()) * 2).fetchall()

        # Add edges
        for entity1, entity2, rel_type, strength, co_occur in relationships:
            if entity1 in entity_map and entity2 in entity_map:
                G.add_edge(entity_map[entity1], entity_map[entity2],
                          relationship=rel_type, strength=strength, co_occur=co_occur)

        # Calculate 3D spring layout
        pos_3d = nx.spring_layout(G, dim=3, k=0.5, iterations=50)

        # Extract positions
        x_nodes = [pos_3d[node][0] for node in G.nodes()]
        y_nodes = [pos_3d[node][1] for node in G.nodes()]
        z_nodes = [pos_3d[node][2] for node in G.nodes()]

        # Prepare node data
        node_labels = [G.nodes[node]['label'] for node in G.nodes()]
        node_types = [G.nodes[node]['entity_type'] for node in G.nodes()]
        node_frequencies = [G.nodes[node]['frequency'] for node in G.nodes()]

        # Color map for entity types
        type_color_map = {
            'PERSON': '#FF6B6B',
            'ORG': '#4ECDC4',
            'PRODUCT': '#95E1D3',
            'TECH': '#FFA07A',
            'LOCATION': '#9B59B6'
        }
        node_colors = [type_color_map.get(t, '#95A5A6') for t in node_types]

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []

        for edge in G.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        # Create edge trace
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125, 125, 125, 0.3)', width=1),
            hoverinfo='none',
            name='Relationships'
        )

        # Create node trace
        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=[min(30, freq * 2) for freq in node_frequencies],
                color=node_colors,
                line=dict(color='white', width=0.5),
                opacity=0.8
            ),
            text=node_labels,
            textposition='top center',
            textfont=dict(size=8),
            customdata=list(zip(node_types, node_frequencies)),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Type: %{customdata[0]}<br>'
                'Frequency: %{customdata[1]}<br>'
                '<extra></extra>'
            ),
            name='Entities'
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=dict(text='3D Knowledge Graph - C64 Knowledge Base', font=dict(size=16)),
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor='rgba(240, 240, 240, 0.9)'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=800
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"3D knowledge graph saved to {output_file} ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        return str(output_file)

    def visualize_hierarchical_bundling(self, max_entities: int = 30,
                                       output_path: str = "hierarchical_bundling.html") -> str:
        """
        Create hierarchical edge bundling visualization showing entity relationships.

        Uses circular layout with entities grouped by type, showing relationships
        with curved bundled edges.

        Args:
            max_entities: Maximum number of entities to include
            output_path: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        import plotly.graph_objects as go
        from pathlib import Path
        import numpy as np

        cursor = self.db_conn.cursor()

        # Get top entities grouped by type
        entities_data = cursor.execute("""
            SELECT entity_text, entity_type, COUNT(*) as frequency,
                   AVG(confidence) as avg_confidence
            FROM document_entities
            WHERE confidence >= 0.7
            GROUP BY entity_text, entity_type
            ORDER BY entity_type, frequency DESC
        """).fetchall()

        # Limit and group by type
        entities_by_type = {}
        for name, entity_type, freq, conf in entities_data:
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            if len(entities_by_type[entity_type]) < max_entities // len(set(e[1] for e in entities_data[:max_entities])):
                entities_by_type[entity_type].append((name, freq, conf))

        # Calculate circular positions for each type
        entity_positions = {}
        angle = 0
        total_entities = sum(len(entities) for entities in entities_by_type.values())

        if total_entities == 0:
            self.logger.warning("No entities found for hierarchical bundling")
            return ""

        angle_step = 2 * np.pi / total_entities

        for entity_type, entities in entities_by_type.items():
            for name, freq, conf in entities:
                x = np.cos(angle)
                y = np.sin(angle)
                entity_positions[name] = (x, y, entity_type, freq, conf)
                angle += angle_step

        # Get relationships
        entity_names = list(entity_positions.keys())
        relationships = cursor.execute("""
            SELECT entity1_text, entity2_text, relationship_type, strength, doc_count
            FROM entity_relationships
            WHERE entity1_text IN ({}) AND entity2_text IN ({})
            AND strength >= 0.3
        """.format(','.join('?' * len(entity_names)), ','.join('?' * len(entity_names))),
                                     entity_names * 2).fetchall()

        # Create edge traces with bundling effect
        edge_traces = []

        for entity1, entity2, rel_type, strength, co_occur in relationships:
            if entity1 in entity_positions and entity2 in entity_positions:
                x1, y1, type1, freq1, conf1 = entity_positions[entity1]
                x2, y2, type2, freq2, conf2 = entity_positions[entity2]

                # Create curved path (quadratic Bezier curve through origin)
                t = np.linspace(0, 1, 20)
                # Control point at origin for bundling effect
                cx, cy = 0, 0
                x_curve = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
                y_curve = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2

                # Color and width based on strength
                color = f'rgba(100, 100, 100, {strength * 0.5})'
                width = max(0.5, strength * 3)

                edge_trace = go.Scatter(
                    x=x_curve, y=y_curve,
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(edge_trace)

        # Create node traces by type
        type_color_map = {
            'PERSON': '#FF6B6B',
            'ORG': '#4ECDC4',
            'PRODUCT': '#95E1D3',
            'TECH': '#FFA07A',
            'LOCATION': '#9B59B6'
        }

        node_traces = []
        for entity_type, color in type_color_map.items():
            type_entities = [(name, pos) for name, pos in entity_positions.items() if pos[2] == entity_type]

            if type_entities:
                x_nodes = [pos[0] for name, pos in type_entities]
                y_nodes = [pos[1] for name, pos in type_entities]
                labels = [name for name, pos in type_entities]
                frequencies = [pos[3] for name, pos in type_entities]

                node_trace = go.Scatter(
                    x=x_nodes, y=y_nodes,
                    mode='markers+text',
                    marker=dict(
                        size=[min(20, f * 2) for f in frequencies],
                        color=color,
                        line=dict(color='white', width=1)
                    ),
                    text=labels,
                    textposition='top center',
                    textfont=dict(size=8),
                    name=entity_type,
                    hovertemplate='<b>%{text}</b><br>Type: ' + entity_type + '<br><extra></extra>'
                )
                node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)

        fig.update_layout(
            title=dict(text='Hierarchical Edge Bundling - Entity Relationships', font=dict(size=16)),
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            height=800,
            hovermode='closest'
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Hierarchical bundling visualization saved to {output_file}")
        return str(output_file)

    def build_knowledge_graph(self, entity_types: Optional[List[str]] = None,
                             min_occurrences: int = 2,
                             min_relationship_strength: float = 0.3,
                             use_cache: bool = True) -> 'nx.Graph':
        """
        Build NetworkX knowledge graph from entities and relationships.

        Creates a weighted graph where:
        - Nodes = entities (with type, occurrences, weight attributes)
        - Edges = relationships (with weight=strength, co_occurrences attributes)

        Args:
            entity_types: Filter to specific entity types (None = all types)
            min_occurrences: Minimum entity occurrences to include (default: 2)
            min_relationship_strength: Minimum relationship strength (0.0-1.0, default: 0.3)
            use_cache: Try to load from cache first (default: True)

        Returns:
            NetworkX Graph with entities as nodes and relationships as edges

        Example:
            >>> kb = KnowledgeBase()
            >>> G = kb.build_knowledge_graph(entity_types=['person', 'org'], min_occurrences=5)
            >>> print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        Raises:
            ImportError: If NetworkX is not installed
            ValueError: If parameters are invalid
        """
        import hashlib
        from datetime import datetime

        try:
            import networkx as nx
        except ImportError:
            self.logger.error("NetworkX not installed. Run: pip install networkx")
            raise ImportError("NetworkX required for knowledge graph. Install with: pip install networkx>=3.0")

        # Validate parameters
        if min_occurrences < 1:
            raise ValueError("min_occurrences must be >= 1")
        if not 0.0 <= min_relationship_strength <= 1.0:
            raise ValueError("min_relationship_strength must be between 0.0 and 1.0")

        self.logger.info(f"Building knowledge graph (min_occurrences={min_occurrences}, min_strength={min_relationship_strength})")

        cursor = self.db_conn.cursor()

        # Add nodes from entities
        query = """
            SELECT entity_text, entity_type, COUNT(*) as occurrences
            FROM document_entities
            WHERE confidence >= 0.5
            GROUP BY entity_text, entity_type
            HAVING occurrences >= ?
        """
        params = [min_occurrences]

        if entity_types:
            placeholders = ','.join('?' * len(entity_types))
            query += f" AND entity_type IN ({placeholders})"
            params.extend(entity_types)

        query += " ORDER BY occurrences DESC"

        entities = cursor.execute(query, params).fetchall()

        if not entities:
            self.logger.warning("No entities found matching criteria")
            return nx.Graph()

        # Create graph and add nodes
        G = nx.Graph()
        entity_set = set()

        for entity_text, entity_type, occurrences in entities:
            G.add_node(entity_text,
                      type=entity_type,
                      occurrences=occurrences,
                      weight=occurrences)  # Node weight = occurrence count
            entity_set.add(entity_text)

        self.logger.info(f"Added {G.number_of_nodes()} nodes to graph")

        # Add edges from relationships
        rel_query = """
            SELECT entity1_text, entity2_text, strength, doc_count
            FROM entity_relationships
            WHERE strength >= ?
            ORDER BY strength DESC
        """

        relationships = cursor.execute(rel_query, [min_relationship_strength]).fetchall()

        edge_count = 0
        for e1, e2, strength, doc_count in relationships:
            # Only add edge if both entities are in the graph
            if e1 in entity_set and e2 in entity_set:
                G.add_edge(e1, e2,
                          weight=strength,
                          co_occurrences=doc_count)
                edge_count += 1

        self.logger.info(f"Added {edge_count} edges to graph")

        # Validate graph
        if G.number_of_nodes() == 0:
            self.logger.warning("Graph has no nodes")
            return G

        # Log graph statistics
        density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        num_components = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len) if num_components > 0 else set()

        self.logger.info(f"Graph statistics:")
        self.logger.info(f"  Nodes: {G.number_of_nodes()}")
        self.logger.info(f"  Edges: {G.number_of_edges()}")
        self.logger.info(f"  Density: {density:.4f}")
        self.logger.info(f"  Connected components: {num_components}")
        self.logger.info(f"  Largest component size: {len(largest_cc)}")

        # Cache graph
        if use_cache and G.number_of_nodes() > 0:
            cache_id = self._cache_graph(G)
            self.logger.info(f"Graph cached with ID: {cache_id}")

        return G

    def _cache_graph(self, G: 'nx.Graph') -> str:
        """
        Cache NetworkX graph to database for quick reloading.

        Args:
            G: NetworkX graph to cache

        Returns:
            cache_id: Unique identifier for cached graph

        Example:
            >>> cache_id = kb._cache_graph(G)
            >>> print(f"Cached as: {cache_id}")
        """
        import hashlib
        import networkx as nx
        from datetime import datetime

        # Generate cache ID from graph properties and timestamp
        cache_str = f"{G.number_of_nodes()}_{G.number_of_edges()}_{datetime.now().isoformat()}"
        cache_id = hashlib.sha256(cache_str.encode()).hexdigest()[:16]

        # Serialize graph as JSON (node_link_data), not pickle. graph_cache
        # is a shared SQLite file that outlives this process (copied via
        # create_backup/restore_from_backup) - unpickling arbitrary bytes
        # from it would be remote-code-execution-by-file-tamper. JSON has no
        # such risk and networkx graphs round-trip through it losslessly.
        try:
            graph_data = json.dumps(nx.node_link_data(G)).encode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to serialize graph: {e}")
            raise

        # Store to database (graph_version=2 marks the JSON format; version 1
        # rows, if any remain from before this change, are pickle and are
        # treated as a cache miss on load rather than deserialized).
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO graph_cache
            (cache_id, graph_version, graph_data, node_count, edge_count, created_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cache_id, 2, graph_data, G.number_of_nodes(),
              G.number_of_edges(), datetime.now().isoformat()))

        self.db_conn.commit()

        self.logger.debug(f"Cached graph {cache_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return cache_id

    def _load_cached_graph(self, cache_id: str) -> Optional['nx.Graph']:
        """
        Load cached NetworkX graph from database.

        Args:
            cache_id: Unique identifier of cached graph

        Returns:
            NetworkX graph if found, None otherwise

        Example:
            >>> G = kb._load_cached_graph("abc123def456")
            >>> if G:
            >>>     print(f"Loaded: {G.number_of_nodes()} nodes")
        """
        import networkx as nx
        from datetime import datetime

        cursor = self.db_conn.cursor()
        row = cursor.execute("""
            SELECT graph_data, graph_version FROM graph_cache WHERE cache_id = ?
        """, (cache_id,)).fetchone()

        if not row:
            self.logger.debug(f"Cache miss: {cache_id}")
            return None

        graph_data, graph_version = row
        if graph_version != 2:
            # Pre-existing pickle-format row (graph_version=1) from before
            # the JSON migration - do not unpickle untrusted bytes from a
            # shared database file. Drop it and treat as a miss so the
            # caller rebuilds and re-caches in the current format.
            self.logger.info(f"Discarding legacy pickle graph cache entry: {cache_id}")
            cursor.execute("DELETE FROM graph_cache WHERE cache_id = ?", (cache_id,))
            self.db_conn.commit()
            return None

        # Update last accessed timestamp
        cursor.execute("""
            UPDATE graph_cache SET last_accessed = ? WHERE cache_id = ?
        """, (datetime.now().isoformat(), cache_id))
        self.db_conn.commit()

        # Deserialize graph
        try:
            G = nx.node_link_graph(json.loads(graph_data.decode('utf-8')))
            self.logger.debug(f"Cache hit: {cache_id} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
            return G
        except Exception as e:
            self.logger.error(f"Failed to deserialize graph {cache_id}: {e}")
            return None

    def analyze_pagerank(self, G: 'nx.Graph', alpha: float = 0.85,
                        max_iter: int = 100, store_to_db: bool = True) -> Dict[str, float]:
        """
        Calculate PageRank scores for all entities in knowledge graph.

        PageRank identifies the most "important" entities based on their
        connections to other entities in the graph.

        Args:
            G: NetworkX graph (from build_knowledge_graph)
            alpha: Damping parameter (0.0-1.0, default: 0.85)
            max_iter: Maximum iterations for convergence (default: 100)
            store_to_db: Store results to graph_metrics table (default: True)

        Returns:
            Dict mapping entity_text -> PageRank score (sorted by score descending)

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> pagerank = kb.analyze_pagerank(G)
            >>> top_5 = list(pagerank.items())[:5]
            >>> for entity, score in top_5:
            >>>     print(f"{entity}: {score:.4f}")

        Raises:
            ImportError: If NetworkX is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        if G.number_of_nodes() == 0:
            self.logger.warning("Cannot compute PageRank on empty graph")
            return {}

        self.logger.info(f"Computing PageRank (alpha={alpha}, max_iter={max_iter})")

        # Calculate PageRank with edge weights
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight='weight')

        # Sort by score descending
        sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

        self.logger.info(f"PageRank computed for {len(pagerank)} entities")
        self.logger.info(f"Top entity: {sorted_pr[0][0]} (score: {sorted_pr[0][1]:.6f})")

        # Store to database
        if store_to_db:
            stored = self._store_graph_metrics(pagerank, metric_type='pagerank', G=G)
            self.logger.info(f"Stored PageRank metrics for {stored} entities")

        return dict(sorted_pr)

    def detect_communities(self, G: 'nx.Graph', algorithm: str = 'louvain',
                          store_to_db: bool = True) -> Dict[str, int]:
        """
        Detect communities (clusters) in knowledge graph.

        Communities are groups of entities that are more densely connected
        to each other than to the rest of the graph.

        Args:
            G: NetworkX graph (from build_knowledge_graph)
            algorithm: Detection algorithm:
                - 'louvain': Louvain method (best for large graphs)
                - 'label_propagation': Fast, non-deterministic
                - 'greedy_modularity': Greedy optimization
            store_to_db: Store results to graph_metrics table (default: True)

        Returns:
            Dict mapping entity_text -> community_id

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> communities = kb.detect_communities(G, algorithm='louvain')
            >>> print(f"Found {len(set(communities.values()))} communities")

        Raises:
            ImportError: If NetworkX is not installed
            ValueError: If algorithm is unknown
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        if G.number_of_nodes() == 0:
            self.logger.warning("Cannot detect communities in empty graph")
            return {}

        self.logger.info(f"Detecting communities using {algorithm} algorithm")

        # Run community detection
        if algorithm == 'louvain':
            communities = nx.community.louvain_communities(G, weight='weight')
        elif algorithm == 'label_propagation':
            communities = nx.community.label_propagation_communities(G)
        elif algorithm == 'greedy_modularity':
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'louvain', 'label_propagation', or 'greedy_modularity'")

        # Convert to dict: entity -> community_id
        entity_to_community = {}
        for idx, community in enumerate(communities):
            for entity in community:
                entity_to_community[entity] = idx

        num_communities = len(communities)
        avg_size = len(entity_to_community) / num_communities if num_communities > 0 else 0

        self.logger.info(f"Detected {num_communities} communities (avg size: {avg_size:.1f})")

        # Store to database
        if store_to_db:
            stored = self._store_graph_metrics(entity_to_community, metric_type='community', G=G)
            self.logger.info(f"Stored community assignments for {stored} entities")

        return entity_to_community

    def calculate_centrality(self, G: 'nx.Graph', store_to_db: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Calculate multiple centrality measures for all entities.

        Centrality measures identify important/influential entities:
        - Betweenness: Entities that bridge different parts of the graph
        - Closeness: Entities that are close to all other entities
        - Degree: Entities with many direct connections

        Args:
            G: NetworkX graph (from build_knowledge_graph)
            store_to_db: Store results to graph_metrics table (default: True)

        Returns:
            Dict with keys 'betweenness', 'closeness', 'degree', each mapping entity -> score

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> centrality = kb.calculate_centrality(G)
            >>> top_betweenness = sorted(centrality['betweenness'].items(),
            >>>                          key=lambda x: x[1], reverse=True)[:5]

        Raises:
            ImportError: If NetworkX is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        if G.number_of_nodes() == 0:
            self.logger.warning("Cannot calculate centrality on empty graph")
            return {'betweenness': {}, 'closeness': {}, 'degree': {}}

        self.logger.info("Calculating centrality measures...")

        centrality = {}

        # Betweenness centrality (weighted)
        self.logger.debug("Computing betweenness centrality")
        centrality['betweenness'] = nx.betweenness_centrality(G, weight='weight')

        # Closeness centrality (weighted as distance)
        self.logger.debug("Computing closeness centrality")
        centrality['closeness'] = nx.closeness_centrality(G, distance='weight')

        # Degree centrality (unweighted)
        self.logger.debug("Computing degree centrality")
        centrality['degree'] = nx.degree_centrality(G)

        self.logger.info(f"Computed 3 centrality measures for {G.number_of_nodes()} entities")

        # Store all metrics to database
        if store_to_db:
            for metric_type, values in centrality.items():
                stored = self._store_graph_metrics(values, metric_type=metric_type, G=G)
                self.logger.debug(f"Stored {metric_type} for {stored} entities")

        return centrality

    def find_shortest_path(self, G: 'nx.Graph', entity1: str, entity2: str,
                          cache_result: bool = True) -> Optional[List[str]]:
        """
        Find shortest path between two entities in knowledge graph.

        Args:
            G: NetworkX graph (from build_knowledge_graph)
            entity1: Source entity name
            entity2: Target entity name
            cache_result: Cache path to database (default: True)

        Returns:
            List of entity names forming the path, or None if no path exists

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> path = kb.find_shortest_path(G, "VIC-II", "sprites")
            >>> if path:
            >>>     print(" -> ".join(path))

        Raises:
            ImportError: If NetworkX is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        if entity1 not in G.nodes():
            self.logger.warning(f"Entity not in graph: {entity1}")
            return None

        if entity2 not in G.nodes():
            self.logger.warning(f"Entity not in graph: {entity2}")
            return None

        try:
            # Find shortest path using edge weights
            path = nx.shortest_path(G, entity1, entity2, weight='weight')

            self.logger.info(f"Found path from '{entity1}' to '{entity2}': {len(path)} steps")

            # Cache path to database
            if cache_result:
                self._cache_path(entity1, entity2, path, G=G)

            return path

        except nx.NetworkXNoPath:
            self.logger.info(f"No path found between '{entity1}' and '{entity2}'")
            return None

    def _store_graph_metrics(self, metrics: Dict[str, float], metric_type: str,
                            G: Optional['nx.Graph'] = None) -> int:
        """
        Store graph analysis metrics to database.

        Args:
            metrics: Dict mapping entity_text -> metric_value
            metric_type: Type of metric ('pagerank', 'community', 'betweenness', 'closeness', 'degree')
            G: NetworkX graph (optional, for entity type lookup)

        Returns:
            Number of metrics stored
        """
        from datetime import datetime

        cursor = self.db_conn.cursor()
        stored = 0

        for entity_text, value in metrics.items():
            # Get entity type from graph or database
            entity_type = None
            if G and entity_text in G.nodes():
                entity_type = G.nodes[entity_text].get('type', 'unknown')
            else:
                # Lookup from database
                row = cursor.execute("""
                    SELECT entity_type FROM document_entities
                    WHERE entity_text = ?
                    LIMIT 1
                """, (entity_text,)).fetchone()
                if row:
                    entity_type = row[0]

            if not entity_type:
                entity_type = 'unknown'

            # Generate metric ID
            import hashlib
            metric_id = hashlib.sha256(f"{entity_text}_{metric_type}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

            # Determine which column to update based on metric type
            if metric_type == 'pagerank':
                cursor.execute("""
                    INSERT OR REPLACE INTO graph_metrics
                    (metric_id, entity_text, entity_type, pagerank, computed_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_id, entity_text, entity_type, value, datetime.now().isoformat()))

            elif metric_type == 'community':
                cursor.execute("""
                    INSERT OR REPLACE INTO graph_metrics
                    (metric_id, entity_text, entity_type, community_id, computed_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_id, entity_text, entity_type, int(value), datetime.now().isoformat()))

            elif metric_type == 'betweenness':
                cursor.execute("""
                    INSERT OR REPLACE INTO graph_metrics
                    (metric_id, entity_text, entity_type, betweenness_centrality, computed_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_id, entity_text, entity_type, value, datetime.now().isoformat()))

            elif metric_type == 'closeness':
                cursor.execute("""
                    INSERT OR REPLACE INTO graph_metrics
                    (metric_id, entity_text, entity_type, closeness_centrality, computed_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_id, entity_text, entity_type, value, datetime.now().isoformat()))

            elif metric_type == 'degree':
                cursor.execute("""
                    INSERT OR REPLACE INTO graph_metrics
                    (metric_id, entity_text, entity_type, degree_centrality, computed_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (metric_id, entity_text, entity_type, value, datetime.now().isoformat()))

            stored += 1

        self.db_conn.commit()
        return stored

    def _cache_path(self, entity1: str, entity2: str, path: List[str],
                   G: Optional['nx.Graph'] = None) -> str:
        """
        Cache shortest path to database.

        Args:
            entity1: Source entity
            entity2: Target entity
            path: List of entities forming the path
            G: NetworkX graph (optional, for weight calculation)

        Returns:
            path_id: Unique identifier for cached path
        """
        import hashlib
        import json
        from datetime import datetime

        # Generate path ID
        path_str = f"{entity1}_{entity2}_{len(path)}_{datetime.now().isoformat()}"
        path_id = hashlib.sha256(path_str.encode()).hexdigest()[:16]

        # Calculate path weight if graph provided
        path_weight = None
        if G:
            path_weight = 0.0
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i + 1]):
                    path_weight += G.edges[path[i], path[i + 1]].get('weight', 1.0)

        # Store to database
        cursor = self.db_conn.cursor()
        cursor.execute("""
            INSERT INTO graph_paths
            (path_id, entity1, entity2, path_length, path_nodes, path_weight, computed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (path_id, entity1, entity2, len(path), json.dumps(path),
              path_weight, datetime.now().isoformat()))

        self.db_conn.commit()

        return path_id

    def visualize_knowledge_graph_pyvis(self, G: 'nx.Graph',
                                       output_path: str = "knowledge_graph.html",
                                       color_by: str = "type",
                                       size_by: str = "pagerank",
                                       show_labels: bool = True,
                                       max_nodes: int = 100) -> str:
        """
        Generate interactive HTML visualization of knowledge graph using PyVis.

        Creates an interactive, physics-based graph visualization that can be
        panned, zoomed, and explored in a web browser.

        Args:
            G: NetworkX graph (from build_knowledge_graph)
            output_path: Output HTML file path (default: "knowledge_graph.html")
            color_by: Node coloring attribute:
                - 'type': Color by entity type
                - 'community': Color by community ID
            size_by: Node sizing attribute:
                - 'pagerank': Size by PageRank score
                - 'occurrences': Size by occurrence count
                - 'degree': Size by node degree
            show_labels: Show entity names on nodes (default: True)
            max_nodes: Maximum nodes to display (default: 100)

        Returns:
            Path to generated HTML file

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> pagerank = kb.analyze_pagerank(G)
            >>> communities = kb.detect_communities(G)
            >>> kb.visualize_knowledge_graph_pyvis(G, color_by='community', size_by='pagerank')

        Raises:
            ImportError: If PyVis is not installed
        """
        try:
            from pyvis.network import Network
        except ImportError:
            self.logger.error("PyVis not installed. Run: pip install pyvis")
            raise ImportError("PyVis required for visualization. Install with: pip install pyvis>=0.3.0")

        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        from pathlib import Path

        if G.number_of_nodes() == 0:
            self.logger.warning("Cannot visualize empty graph")
            return ""

        self.logger.info(f"Creating PyVis visualization (color_by={color_by}, size_by={size_by})")

        # Calculate metrics if needed for sizing
        if size_by == "pagerank":
            # Check if PageRank already computed
            sample_node = list(G.nodes())[0]
            if 'pagerank' not in G.nodes[sample_node]:
                self.logger.info("Computing PageRank for node sizing")
                pagerank = self.analyze_pagerank(G, store_to_db=False)
                nx.set_node_attributes(G, pagerank, 'pagerank')

        elif size_by == "degree":
            # Calculate degree centrality
            degree_cent = nx.degree_centrality(G)
            nx.set_node_attributes(G, degree_cent, 'degree')

        # Calculate communities if needed for coloring
        if color_by == "community":
            sample_node = list(G.nodes())[0]
            if 'community' not in G.nodes[sample_node]:
                self.logger.info("Computing communities for node coloring")
                communities = self.detect_communities(G, store_to_db=False)
                nx.set_node_attributes(G, communities, 'community')

        # Filter to top nodes if graph is too large
        if G.number_of_nodes() > max_nodes:
            self.logger.info(f"Filtering graph to top {max_nodes} nodes by degree centrality")
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        # Create PyVis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False
        )

        # Configure physics for better layout
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.001,
            damping=0.09
        )

        # Get color mapping
        color_map = self._get_color_map_for_graph(G, color_by)

        # Add nodes with styling
        for node, attrs in G.nodes(data=True):
            # Determine size
            if size_by == "pagerank":
                size = attrs.get('pagerank', 0.001) * 1000  # Scale PageRank
            elif size_by == "occurrences":
                size = attrs.get('occurrences', 1) * 2
            elif size_by == "degree":
                size = attrs.get('degree', 0.01) * 100
            else:
                size = 10

            # Clamp size to reasonable range
            size = max(10, min(50, size))

            # Determine color
            if color_by == "type":
                color_key = attrs.get('type', 'default')
            elif color_by == "community":
                color_key = attrs.get('community', 0)
            else:
                color_key = 'default'

            color = color_map.get(color_key, "#97c2fc")

            # Create tooltip
            tooltip = self._get_node_tooltip(node, attrs)

            # Add node
            net.add_node(
                node,
                label=node if show_labels else "",
                size=size,
                color=color,
                title=tooltip
            )

        # Add edges
        for e1, e2, attrs in G.edges(data=True):
            weight = attrs.get('weight', 0.5)
            co_occurrences = attrs.get('co_occurrences', 0)

            net.add_edge(
                e1, e2,
                value=weight * 5,  # Scale edge width
                title=f"Strength: {weight:.2f}, Co-occurrences: {co_occurrences}"
            )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_file))

        self.logger.info(f"PyVis visualization saved to {output_file} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return str(output_file)

    def _get_color_map_for_graph(self, G: 'nx.Graph', color_by: str) -> Dict[str, str]:
        """
        Get color mapping for graph nodes.

        Args:
            G: NetworkX graph
            color_by: Attribute to color by ('type' or 'community')

        Returns:
            Dict mapping attribute value -> hex color
        """
        if color_by == "type":
            return {
                'hardware': '#e74c3c',       # Red
                'instruction': '#3498db',    # Blue
                'register': '#2ecc71',       # Green
                'memory_address': '#f39c12', # Orange
                'person': '#9b59b6',         # Purple
                'company': '#1abc9c',        # Teal
                'product': '#e67e22',        # Dark orange
                'org': '#16a085',            # Dark teal
                'tech': '#d35400',           # Dark red-orange
                'location': '#8e44ad',       # Dark purple
                'default': '#95a5a6'         # Gray
            }

        elif color_by == "community":
            # Generate distinct colors for communities
            import colorsys
            try:
                import networkx as nx
            except ImportError:
                return {}

            communities = nx.get_node_attributes(G, 'community')
            if not communities:
                return {}

            num_communities = len(set(communities.values()))
            color_map = {}

            for i in range(num_communities):
                hue = i / num_communities
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
                color_map[i] = hex_color

            return color_map

        return {'default': '#97c2fc'}

    def _get_node_tooltip(self, node: str, attrs: Dict) -> str:
        """
        Generate HTML tooltip for graph node.

        Args:
            node: Node name/text
            attrs: Node attributes dict

        Returns:
            HTML string for tooltip
        """
        lines = [
            f"<b>{node}</b>",
            f"Type: {attrs.get('type', 'unknown')}",
            f"Occurrences: {attrs.get('occurrences', 0)}"
        ]

        if 'pagerank' in attrs:
            lines.append(f"PageRank: {attrs['pagerank']:.6f}")

        if 'community' in attrs:
            lines.append(f"Community: {attrs['community']}")

        if 'betweenness_centrality' in attrs:
            lines.append(f"Betweenness: {attrs['betweenness_centrality']:.4f}")

        if 'degree' in attrs:
            lines.append(f"Degree Centrality: {attrs['degree']:.4f}")

        return "<br>".join(lines)

    def export_graph(self, G: 'nx.Graph', output_path: str,
                    format: str = 'graphml') -> str:
        """
        Export knowledge graph to various formats.

        Args:
            G: NetworkX graph
            output_path: Output file path
            format: Export format:
                - 'graphml': GraphML XML format
                - 'gexf': GEXF XML format (Gephi)
                - 'json': JSON graph format
                - 'gml': GML format

        Returns:
            Path to exported file

        Example:
            >>> G = kb.build_knowledge_graph()
            >>> kb.export_graph(G, "graph.graphml", format="graphml")

        Raises:
            ImportError: If NetworkX is not installed
            ValueError: If format is unknown
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required. Install with: pip install networkx>=3.0")

        from pathlib import Path

        if format not in ['graphml', 'gexf', 'json', 'gml']:
            raise ValueError(f"Unknown format: {format}. Use 'graphml', 'gexf', 'json', or 'gml'")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting graph to {format} format: {output_file}")

        if format == 'graphml':
            nx.write_graphml(G, str(output_file))
        elif format == 'gexf':
            nx.write_gexf(G, str(output_file))
        elif format == 'json':
            from networkx.readwrite import json_graph
            import json
            data = json_graph.node_link_data(G)
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'gml':
            nx.write_gml(G, str(output_file))

        self.logger.info(f"Graph exported to {output_file} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return str(output_file)
