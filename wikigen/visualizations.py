"""The D3-driven knowledge-graph and similarity-map pages.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""



class VisualizationsMixin:
    """The D3-driven knowledge-graph and similarity-map pages."""

    def _generate_knowledge_graph_html(self):
        """Generate knowledge graph visualization page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .graph-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
            margin: 20px 0;
            height: calc(100vh - 200px);
        }

        .graph-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .graph-controls h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .control-group {
            margin: 20px 0;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-color);
        }

        .control-group input[type="text"],
        .control-group input[type="range"] {
            width: 100%;
            padding: 8px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: inherit;
        }

        .control-group input[type="text"]:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        .filter-checkboxes {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 10px;
        }

        .filter-checkboxes label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: normal;
            cursor: pointer;
            padding: 6px;
            border-radius: 6px;
            transition: background 0.2s;
        }

        .filter-checkboxes label:hover {
            background: var(--bg-color);
        }

        .filter-checkboxes input[type="checkbox"] {
            cursor: pointer;
        }

        .type-legend {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 10px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid var(--border-color);
        }

        #graph-canvas {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }

        #graph-svg {
            width: 100%;
            height: 100%;
            cursor: move;
        }

        .graph-info {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .graph-info h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .info-empty {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }

        .node-info {
            animation: fadeIn 0.3s;
        }

        .node-info h4 {
            color: var(--accent-color);
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }

        .node-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }

        .stat-item {
            background: var(--bg-color);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .connections-list {
            margin-top: 15px;
        }

        .connections-list h5 {
            margin: 10px 0;
            color: var(--secondary-color);
        }

        .connection-item {
            padding: 8px;
            margin: 5px 0;
            background: var(--bg-color);
            border-radius: 6px;
            border-left: 3px solid var(--accent-color);
            cursor: pointer;
            transition: all 0.2s;
        }

        .connection-item:hover {
            background: var(--border-color);
            transform: translateX(4px);
        }

        .connection-name {
            font-weight: 600;
            color: var(--text-color);
        }

        .connection-strength {
            font-size: 0.85em;
            color: var(--text-muted);
        }

        .graph-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }

        .graph-stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .graph-stat-card .stat-value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .graph-stat-card .stat-label {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-top: 8px;
        }

        .loading-message {
            text-align: center;
            padding: 60px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        .zoom-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 10;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-size: 1.5em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        /* Node and edge styles */
        .node {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.2);
        }

        .node.highlighted {
            stroke: var(--accent-color);
            stroke-width: 4px;
        }

        .node.dimmed {
            opacity: 0.3;
        }

        .link {
            stroke: var(--border-color);
            stroke-opacity: 0.6;
            transition: all 0.3s;
        }

        .link.highlighted {
            stroke: var(--accent-color);
            stroke-width: 3px;
            stroke-opacity: 1;
        }

        .link.dimmed {
            opacity: 0.1;
        }

        .node-label {
            font-size: 11px;
            pointer-events: none;
            text-anchor: middle;
            fill: var(--text-color);
            font-weight: 600;
        }

        .node-label.hidden {
            display: none;
        }

        @media (max-width: 1200px) {
            .graph-container {
                grid-template-columns: 1fr;
                height: auto;
            }

            .graph-info {
                order: -1;
            }

            #graph-canvas {
                height: 600px;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🕸️ Knowledge Graph</h1>
            <p class="subtitle">Explore entity relationships and connections</p>
        </header>

{NAV}

{ABOUT}

        <div class="graph-stats" id="graph-stats">
            <div class="graph-stat-card">
                <div class="stat-value" id="total-nodes">-</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="graph-stat-card">
                <div class="stat-value" id="total-edges">-</div>
                <div class="stat-label">Connections</div>
            </div>
            <div class="graph-stat-card">
                <div class="stat-value" id="total-types">-</div>
                <div class="stat-label">Entity Types</div>
            </div>
        </div>

        <div class="graph-container">
            <div class="graph-controls">
                <h3>Controls</h3>

                <div class="control-group">
                    <label for="search-node">🔍 Search Entity</label>
                    <input type="text" id="search-node" placeholder="Type entity name...">
                </div>

                <div class="control-group">
                    <label>🎨 Filter by Type</label>
                    <div class="filter-checkboxes" id="type-filters"></div>
                </div>

                <div class="control-group">
                    <label for="min-connections">Minimum Connections: <span id="min-connections-value">0</span></label>
                    <input type="range" id="min-connections" min="0" max="20" value="0" step="1">
                </div>

                <div class="control-group">
                    <label>
                        <input type="checkbox" id="show-labels" checked> Show Labels
                    </label>
                </div>

                <div class="control-group">
                    <h3>Legend</h3>
                    <div class="type-legend" id="type-legend"></div>
                </div>
            </div>

            <div id="graph-canvas">
                <div class="loading-message">Loading knowledge graph...</div>
                <div class="zoom-controls">
                    <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
                    <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
                    <button class="zoom-btn" id="zoom-reset" title="Reset View">⟲</button>
                </div>
                <svg id="graph-svg"></svg>
            </div>

            <div class="graph-info">
                <h3>Node Details</h3>
                <div class="info-empty" id="info-empty">
                    Click on a node to view details
                </div>
                <div class="node-info" id="node-info" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script src="lib/d3.v7.min.js"></script>
    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
    <script>
        // Knowledge Graph Visualization using D3.js
        let graphData = null;
        let simulation = null;
        let currentTransform = d3.zoomIdentity;
        let selectedNode = null;

        // Color scheme for entity types
        const typeColors = {
            'HARDWARE': '#e74c3c',
            'SOFTWARE': '#3498db',
            'PERSON': '#2ecc71',
            'ORGANIZATION': '#f39c12',
            'CONCEPT': '#9b59b6',
            'MUSIC': '#1abc9c',
            'GRAPHICS': '#e67e22',
            'GAME': '#16a085',
            'UNKNOWN': '#95a5a6'
        };

        async function loadGraph() {
            try {
                const response = await fetch('assets/data/graph.json');
                graphData = await response.json();

                // Update stats
                document.getElementById('total-nodes').textContent = graphData.stats.total_nodes.toLocaleString();
                document.getElementById('total-edges').textContent = graphData.stats.total_edges.toLocaleString();
                document.getElementById('total-types').textContent = graphData.stats.node_types;

                // Build type filters
                const types = [...new Set(graphData.nodes.map(n => n.type))].sort();
                buildTypeFilters(types);
                buildTypeLegend(types);

                // Initialize graph
                initializeGraph();
            } catch (error) {
                console.error('Error loading graph:', error);
                document.querySelector('.loading-message').textContent = 'Error loading graph data';
            }
        }

        function buildTypeFilters(types) {
            const container = document.getElementById('type-filters');
            container.innerHTML = types.map(type => `
                <label>
                    <input type="checkbox" class="type-filter" value="${type}" checked>
                    <span style="color: ${typeColors[type] || typeColors.UNKNOWN}">${type}</span>
                </label>
            `).join('');

            // Add event listeners
            container.querySelectorAll('.type-filter').forEach(checkbox => {
                checkbox.addEventListener('change', updateGraph);
            });
        }

        function buildTypeLegend(types) {
            const container = document.getElementById('type-legend');
            container.innerHTML = types.map(type => `
                <div class="legend-item">
                    <div class="legend-color" style="background: ${typeColors[type] || typeColors.UNKNOWN}"></div>
                    <span>${type}</span>
                </div>
            `).join('');
        }

        function initializeGraph() {
            const svg = d3.select('#graph-svg');
            const container = document.getElementById('graph-canvas');
            const width = container.clientWidth;
            const height = container.clientHeight;

            svg.attr('width', width).attr('height', height);

            // Clear loading message
            document.querySelector('.loading-message').style.display = 'none';

            // Create zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on('zoom', (event) => {
                    currentTransform = event.transform;
                    g.attr('transform', currentTransform);
                });

            svg.call(zoom);

            // Create container group
            const g = svg.append('g');

            // Create force simulation
            simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.edges)
                    .id(d => d.id)
                    .distance(d => 100 / (d.weight || 1)))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => Math.sqrt(d.value) * 3 + 10));

            // Draw edges
            const link = g.append('g')
                .selectAll('line')
                .data(graphData.edges)
                .join('line')
                .attr('class', 'link')
                .attr('stroke-width', d => Math.sqrt(d.value || 1));

            // Draw nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(graphData.nodes)
                .join('circle')
                .attr('class', 'node')
                .attr('r', d => Math.sqrt(d.value) * 3 + 5)
                .attr('fill', d => typeColors[d.type] || typeColors.UNKNOWN)
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('click', (event, d) => {
                    event.stopPropagation();
                    selectNode(d, node, link);
                })
                .on('mouseover', function(event, d) {
                    d3.select(this).style('cursor', 'pointer');
                });

            // Add labels
            const labels = g.append('g')
                .selectAll('text')
                .data(graphData.nodes)
                .join('text')
                .attr('class', 'node-label')
                .attr('dy', -15)
                .text(d => d.label);

            // Update positions on each tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });

            // Store references for updates
            window.graphElements = { node, link, labels, g, svg, zoom };

            // Setup controls
            setupControls();
        }

        function selectNode(d, nodeSelection, linkSelection) {
            selectedNode = d;

            // Highlight connected nodes
            const connectedNodeIds = new Set();
            connectedNodeIds.add(d.id);

            const connectedEdges = graphData.edges.filter(e =>
                e.source.id === d.id || e.target.id === d.id
            );

            connectedEdges.forEach(e => {
                connectedNodeIds.add(e.source.id);
                connectedNodeIds.add(e.target.id);
            });

            // Update node styles
            nodeSelection
                .classed('highlighted', n => n.id === d.id)
                .classed('dimmed', n => !connectedNodeIds.has(n.id));

            // Update link styles
            linkSelection
                .classed('highlighted', e => e.source.id === d.id || e.target.id === d.id)
                .classed('dimmed', e => e.source.id !== d.id && e.target.id !== d.id);

            // Show node info
            showNodeInfo(d, connectedEdges);
        }

        function showNodeInfo(node, edges) {
            document.getElementById('info-empty').style.display = 'none';
            const infoDiv = document.getElementById('node-info');
            infoDiv.style.display = 'block';

            const connections = edges.map(e => ({
                node: e.source.id === node.id ? e.target : e.source,
                weight: e.weight,
                doc_count: e.doc_count
            })).sort((a, b) => b.weight - a.weight);

            infoDiv.innerHTML = `
                <h4>${escapeHtml(node.label)}</h4>
                <p style="color: ${typeColors[node.type]}; font-weight: 600;">${node.type}</p>

                <div class="node-stats">
                    <div class="stat-item">
                        <div class="stat-value">${node.count}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${connections.length}</div>
                        <div class="stat-label">Connections</div>
                    </div>
                </div>

                ${connections.length > 0 ? `
                    <div class="connections-list">
                        <h5>Connected Entities</h5>
                        ${connections.slice(0, 10).map(c => `
                            <div class="connection-item" onclick="focusOnNode('${c.node.id}')">
                                <div class="connection-name">${escapeHtml(c.node.label)}</div>
                                <div class="connection-strength">Strength: ${c.weight} • ${c.doc_count} shared docs</div>
                            </div>
                        `).join('')}
                        ${connections.length > 10 ? `<p style="text-align: center; color: var(--text-muted); margin-top: 10px;">... and ${connections.length - 10} more</p>` : ''}
                    </div>
                ` : ''}
            `;
        }

        function focusOnNode(nodeId) {
            const node = graphData.nodes.find(n => n.id === nodeId);
            if (node) {
                selectNode(node, window.graphElements.node, window.graphElements.link);

                // Center on node
                const svg = window.graphElements.svg;
                const g = window.graphElements.g;
                const zoom = window.graphElements.zoom;

                const width = svg.node().clientWidth;
                const height = svg.node().clientHeight;

                const scale = 1.5;
                const x = -node.x * scale + width / 2;
                const y = -node.y * scale + height / 2;

                svg.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
            }
        }

        function updateGraph() {
            const activeTypes = Array.from(document.querySelectorAll('.type-filter:checked'))
                .map(cb => cb.value);

            const minConnections = parseInt(document.getElementById('min-connections').value);
            const showLabels = document.getElementById('show-labels').checked;

            const { node, link, labels } = window.graphElements;

            // Filter nodes
            node.style('display', d => {
                const connections = graphData.edges.filter(e =>
                    e.source.id === d.id || e.target.id === d.id
                ).length;

                return activeTypes.includes(d.type) && connections >= minConnections ? null : 'none';
            });

            // Filter links
            link.style('display', d => {
                const sourceVisible = activeTypes.includes(d.source.type);
                const targetVisible = activeTypes.includes(d.target.type);
                return sourceVisible && targetVisible ? null : 'none';
            });

            // Toggle labels
            labels.classed('hidden', !showLabels);
        }

        function setupControls() {
            // Search
            document.getElementById('search-node').addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                if (query.length < 2) return;

                const matches = graphData.nodes.filter(n =>
                    n.label.toLowerCase().includes(query)
                );

                if (matches.length > 0) {
                    focusOnNode(matches[0].id);
                }
            });

            // Min connections slider
            document.getElementById('min-connections').addEventListener('input', (e) => {
                document.getElementById('min-connections-value').textContent = e.target.value;
                updateGraph();
            });

            // Show labels toggle
            document.getElementById('show-labels').addEventListener('change', updateGraph);

            // Zoom controls
            const { svg, zoom } = window.graphElements;

            document.getElementById('zoom-in').addEventListener('click', () => {
                svg.transition().call(zoom.scaleBy, 1.3);
            });

            document.getElementById('zoom-out').addEventListener('click', () => {
                svg.transition().call(zoom.scaleBy, 0.7);
            });

            document.getElementById('zoom-reset').addEventListener('click', () => {
                svg.transition().call(zoom.transform, d3.zoomIdentity);
            });

            // Click background to deselect
            svg.on('click', () => {
                window.graphElements.node.classed('highlighted', false).classed('dimmed', false);
                window.graphElements.link.classed('highlighted', false).classed('dimmed', false);
                document.getElementById('node-info').style.display = 'none';
                document.getElementById('info-empty').style.display = 'block';
            });
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initialize on load
        loadGraph();
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace('{NAV}', self._get_main_nav('knowledge-graph'))
        html_content = html_content.replace('{ABOUT}', self._get_unified_about_box('knowledge-graph'))

        filepath = self.output_dir / "knowledge-graph.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: knowledge-graph.html")

    def _generate_similarity_map_html(self):
        """Generate document similarity map visualization page."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Similarity Map - TDZ C64 Knowledge Base</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .map-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
            margin: 20px 0;
            height: calc(100vh - 200px);
        }

        .map-controls {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .map-controls h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .control-group {
            margin: 20px 0;
        }

        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--text-color);
        }

        .control-group input[type="text"],
        .control-group select {
            width: 100%;
            padding: 8px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            background: var(--bg-color);
            color: var(--text-color);
            font-family: inherit;
        }

        .control-group input[type="text"]:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        #canvas-container {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
            cursor: grab;
        }

        #canvas-container.grabbing {
            cursor: grabbing;
        }

        #similarity-canvas {
            display: block;
            width: 100%;
            height: 100%;
        }

        .map-info {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            overflow-y: auto;
        }

        .map-info h3 {
            margin-top: 0;
            color: var(--secondary-color);
            font-size: 1.2em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }

        .info-empty {
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }

        .doc-info {
            animation: fadeIn 0.3s;
        }

        .doc-info h4 {
            color: var(--accent-color);
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }

        .doc-meta {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 15px 0;
        }

        .meta-item {
            background: var(--bg-color);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .meta-value {
            font-size: 1.5em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .meta-label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .doc-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 15px 0;
        }

        .tag {
            display: inline-block;
            padding: 4px 10px;
            background: var(--accent-color);
            color: white;
            border-radius: 12px;
            font-size: 0.85em;
        }

        .map-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }

        .stat-card {
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-color);
        }

        .stat-card .label {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }

        .loading-map {
            text-align: center;
            padding: 60px;
            color: var(--text-muted);
            font-size: 1.2em;
        }

        .zoom-controls {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 10;
        }

        .zoom-btn {
            width: 40px;
            height: 40px;
            background: var(--card-bg);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-size: 1.5em;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .zoom-btn:hover {
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }

        .cluster-legend {
            margin-top: 15px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
            padding: 4px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .legend-item:hover {
            background: var(--bg-color);
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid var(--border-color);
        }

        @media (max-width: 1200px) {
            .map-container {
                grid-template-columns: 1fr;
                height: auto;
            }

            .map-info {
                order: -1;
            }

            #canvas-container {
                height: 600px;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🗺️ Document Similarity Map</h1>
            <p class="subtitle">Explore documents in 2D semantic space</p>
        </header>

{NAV}

{ABOUT}

        <div class="map-stats">
            <div class="stat-card">
                <div class="value" id="total-docs">-</div>
                <div class="label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="value" id="total-clusters">-</div>
                <div class="label">Clusters</div>
            </div>
            <div class="stat-card">
                <div class="value" id="reduction-method">-</div>
                <div class="label">Method</div>
            </div>
        </div>

        <div class="map-container">
            <div class="map-controls">
                <h3>Controls</h3>

                <div class="control-group">
                    <label for="search-doc">🔍 Search Document</label>
                    <input type="text" id="search-doc" placeholder="Type document title...">
                </div>

                <div class="control-group">
                    <label for="cluster-filter">🎨 Filter by Cluster</label>
                    <select id="cluster-filter">
                        <option value="all">All Clusters</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="file-type-filter">📁 Filter by Type</label>
                    <select id="file-type-filter">
                        <option value="all">All Types</option>
                    </select>
                </div>

                <div class="control-group">
                    <label>
                        <input type="checkbox" id="show-labels" checked> Show Labels
                    </label>
                </div>

                <div class="cluster-legend">
                    <h3>Cluster Legend</h3>
                    <div id="legend-items"></div>
                </div>
            </div>

            <div id="canvas-container">
                <div class="loading-map">Loading similarity map...</div>
                <div class="zoom-controls">
                    <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
                    <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
                    <button class="zoom-btn" id="zoom-reset" title="Reset View">⟲</button>
                </div>
                <canvas id="similarity-canvas"></canvas>
            </div>

            <div class="map-info">
                <h3>Document Details</h3>
                <div class="info-empty" id="info-empty">
                    Hover over a point to view details<br>
                    Click to navigate to document
                </div>
                <div class="doc-info" id="doc-info" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script src="lib/fuse.min.js"></script>
    <script src="assets/js/search.js"></script>
    <script src="assets/js/enhancements.js"></script>
    <script>
        let documentsData = [];
        let canvas, ctx;
        let scale = 1;
        let offsetX = 0, offsetY = 0;
        let isDragging = false;
        let dragStartX, dragStartY;
        let hoveredDoc = null;
        let selectedCluster = 'all';
        let selectedFileType = 'all';
        let showLabels = true;
        let searchQuery = '';

        const clusterColors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#16a085', '#d35400', '#8e44ad',
            '#c0392b', '#27ae60', '#2980b9', '#f1c40f', '#95a5a6'
        ];

        async function loadMap() {
            try {
                const response = await fetch('assets/data/coordinates.json');
                const data = await response.json();

                if (!data.documents || data.documents.length === 0) {
                    document.querySelector('.loading-map').textContent = 'No coordinate data available';
                    return;
                }

                documentsData = data.documents;

                // Update stats
                const clusters = new Set(documentsData.map(d => d.cluster));
                document.getElementById('total-docs').textContent = documentsData.length;
                document.getElementById('total-clusters').textContent = clusters.size;
                document.getElementById('reduction-method').textContent = data.method.toUpperCase();

                // Build filters
                buildFilters();

                // Initialize canvas
                initCanvas();

                // Hide loading
                document.querySelector('.loading-map').style.display = 'none';
            } catch (error) {
                console.error('Error loading map:', error);
                document.querySelector('.loading-map').textContent = 'Error loading similarity map';
            }
        }

        function buildFilters() {
            const clusters = [...new Set(documentsData.map(d => d.cluster))].sort((a, b) => a - b);
            const fileTypes = [...new Set(documentsData.map(d => d.file_type))].sort();

            const clusterSelect = document.getElementById('cluster-filter');
            clusters.forEach(cluster => {
                const option = document.createElement('option');
                option.value = cluster;
                option.textContent = `Cluster ${cluster}`;
                clusterSelect.appendChild(option);
            });

            const fileTypeSelect = document.getElementById('file-type-filter');
            fileTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type.toUpperCase();
                fileTypeSelect.appendChild(option);
            });

            // Build legend
            const legendContainer = document.getElementById('legend-items');
            clusters.forEach(cluster => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${clusterColors[cluster % clusterColors.length]}"></div>
                    <span>Cluster ${cluster}</span>
                `;
                item.onclick = () => {
                    clusterSelect.value = cluster;
                    selectedCluster = cluster.toString();
                    renderCanvas();
                };
                legendContainer.appendChild(item);
            });

            // Setup event listeners
            document.getElementById('search-doc').addEventListener('input', (e) => {
                searchQuery = e.target.value.toLowerCase();
                renderCanvas();
            });

            clusterSelect.addEventListener('change', (e) => {
                selectedCluster = e.target.value;
                renderCanvas();
            });

            fileTypeSelect.addEventListener('change', (e) => {
                selectedFileType = e.target.value;
                renderCanvas();
            });

            document.getElementById('show-labels').addEventListener('change', (e) => {
                showLabels = e.target.checked;
                renderCanvas();
            });

            document.getElementById('zoom-in').addEventListener('click', () => {
                scale *= 1.2;
                renderCanvas();
            });

            document.getElementById('zoom-out').addEventListener('click', () => {
                scale /= 1.2;
                renderCanvas();
            });

            document.getElementById('zoom-reset').addEventListener('click', () => {
                scale = 1;
                offsetX = 0;
                offsetY = 0;
                renderCanvas();
            });
        }

        function initCanvas() {
            canvas = document.getElementById('similarity-canvas');
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            ctx = canvas.getContext('2d');

            // Mouse events
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseUp);
            canvas.addEventListener('wheel', onWheel);
            canvas.addEventListener('click', onClick);

            // Render
            renderCanvas();
        }

        function renderCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Filter documents
            const filtered = documentsData.filter(doc => {
                if (selectedCluster !== 'all' && doc.cluster !== parseInt(selectedCluster)) return false;
                if (selectedFileType !== 'all' && doc.file_type !== selectedFileType) return false;
                if (searchQuery && !doc.title.toLowerCase().includes(searchQuery)) return false;
                return true;
            });

            // Draw documents
            filtered.forEach(doc => {
                const x = (doc.x * scale) + offsetX + canvas.width / 2;
                const y = (doc.y * scale) + offsetY + canvas.height / 2;

                const color = clusterColors[doc.cluster % clusterColors.length];
                const isHovered = hoveredDoc && hoveredDoc.id === doc.id;
                const isSearchMatch = searchQuery && doc.title.toLowerCase().includes(searchQuery);

                // Draw point
                ctx.beginPath();
                ctx.arc(x, y, isHovered ? 8 : isSearchMatch ? 6 : 4, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = isHovered ? '#fff' : color;
                ctx.lineWidth = isHovered ? 3 : 1;
                ctx.stroke();

                // Draw label
                if (showLabels && (isHovered || isSearchMatch)) {
                    ctx.fillStyle = 'var(--text-color)';
                    ctx.font = '12px sans-serif';
                    ctx.fillText(doc.title.substring(0, 30), x + 10, y - 5);
                }
            });
        }

        function onMouseDown(e) {
            isDragging = true;
            dragStartX = e.clientX - offsetX;
            dragStartY = e.clientY - offsetY;
            document.getElementById('canvas-container').classList.add('grabbing');
        }

        function onMouseMove(e) {
            if (isDragging) {
                offsetX = e.clientX - dragStartX;
                offsetY = e.clientY - dragStartY;
                renderCanvas();
            } else {
                // Check hover
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                hoveredDoc = null;
                for (const doc of documentsData) {
                    const x = (doc.x * scale) + offsetX + canvas.width / 2;
                    const y = (doc.y * scale) + offsetY + canvas.height / 2;
                    const dist = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2);

                    if (dist < 10) {
                        hoveredDoc = doc;
                        showDocInfo(doc);
                        break;
                    }
                }

                if (!hoveredDoc) {
                    hideDocInfo();
                }

                renderCanvas();
            }
        }

        function onMouseUp() {
            isDragging = false;
            document.getElementById('canvas-container').classList.remove('grabbing');
        }

        function onWheel(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= delta;
            renderCanvas();
        }

        function onClick(e) {
            if (hoveredDoc) {
                window.location.href = `docs/${hoveredDoc.filename}`;
            }
        }

        function showDocInfo(doc) {
            document.getElementById('info-empty').style.display = 'none';
            const infoDiv = document.getElementById('doc-info');
            infoDiv.style.display = 'block';

            const color = clusterColors[doc.cluster % clusterColors.length];
            const tags = doc.tags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('');

            infoDiv.innerHTML = `
                <h4>${escapeHtml(doc.title)}</h4>
                <div class="doc-meta">
                    <div class="meta-item">
                        <div class="meta-value" style="color: ${color}">${doc.cluster}</div>
                        <div class="meta-label">Cluster</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-value">${doc.total_chunks}</div>
                        <div class="meta-label">Chunks</div>
                    </div>
                </div>
                <div class="doc-tags">${tags}</div>
                <p style="font-size: 0.9em; color: var(--text-muted);">Type: ${doc.file_type.toUpperCase()}</p>
                <p style="font-size: 0.9em; color: var(--text-muted); margin-top: 10px;">Click to open document</p>
            `;
        }

        function hideDocInfo() {
            document.getElementById('doc-info').style.display = 'none';
            document.getElementById('info-empty').style.display = 'block';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load map on page load
        loadMap();

        // Resize handler
        window.addEventListener('resize', () => {
            if (canvas) {
                const container = document.getElementById('canvas-container');
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
                renderCanvas();
            }
        });
    </script>
</body>
</html>
"""
        # Replace template placeholders with actual content
        html_content = html_template.replace("{NAV}", self._get_main_nav("similarity-map"))
        html_content = html_content.replace("{ABOUT}", self._get_unified_about_box("entities"))

        filepath = self.output_dir / "similarity-map.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"  Generated: similarity-map.html")
