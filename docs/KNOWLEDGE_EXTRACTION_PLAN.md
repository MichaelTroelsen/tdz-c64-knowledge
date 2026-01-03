# Knowledge Extraction Implementation Plan

**Version:** 2.24.0
**Project:** TDZ C64 Knowledge Base - Advanced Analytics
**Scope:** Option C - Full Implementation (All 6 Algorithms, 4 Phases)
**Total Effort:** 64-80 hours
**Created:** 2026-01-03

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1: Foundation](#phase-1-foundation-16-20-hours)
3. [Phase 2: Discovery](#phase-2-discovery-20-24-hours)
4. [Phase 3: Temporal](#phase-3-temporal-16-20-hours)
5. [Phase 4: Integration](#phase-4-integration-12-16-hours)
6. [Dependencies](#dependencies)
7. [Risk Assessment](#risk-assessment)
8. [Testing Strategy](#testing-strategy)
9. [Validation Criteria](#validation-criteria)

---

## Executive Summary

This plan implements 6 advanced knowledge extraction algorithms to unlock deeper insights from the TDZ C64 Knowledge Base (199 documents, 5,044 chunks, 7.4M words).

**Deliverables:**
- Knowledge Graph with PageRank and community detection
- Topic modeling revealing hidden themes (LDA, NMF, BERTopic)
- Document clustering grouping similar content
- Timeline extraction showing C64 history chronologically
- Interactive visualizations (Plotly, PyVis)
- Streamlit analytics dashboard integrating all features

**Success Criteria:**
- All 6 algorithms operational and tested
- 20+ new MCP tools for Claude integration
- Analytics dashboard with 6 tabs
- Comprehensive documentation
- 90%+ test coverage for new code

---

## Phase 1: Foundation (16-20 hours)

**Goal:** Build knowledge graph infrastructure and basic visualizations

**Deliverables:**
- NetworkX knowledge graph from entities/relationships
- Graph analysis algorithms (PageRank, community detection)
- 6 new MCP tools for graph operations
- Basic graph visualization (PyVis)
- Database schema extensions

---

### Task 1.1: Database Schema Extensions (2 hours)

**Objective:** Add tables for storing graph analysis results

**Subtasks:**
1. **Create graph_cache table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS graph_cache (
       cache_id TEXT PRIMARY KEY,
       graph_version INTEGER NOT NULL,
       graph_data BLOB NOT NULL,  -- Pickled NetworkX graph
       node_count INTEGER NOT NULL,
       edge_count INTEGER NOT NULL,
       created_date TEXT NOT NULL,
       last_accessed TEXT
   );
   ```

2. **Create graph_metrics table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS graph_metrics (
       metric_id TEXT PRIMARY KEY,
       entity_text TEXT NOT NULL,
       entity_type TEXT NOT NULL,
       pagerank REAL,
       betweenness_centrality REAL,
       closeness_centrality REAL,
       degree_centrality REAL,
       community_id INTEGER,
       computed_date TEXT NOT NULL,
       FOREIGN KEY (entity_text) REFERENCES entities(text)
   );
   CREATE INDEX idx_graph_metrics_pagerank ON graph_metrics(pagerank DESC);
   CREATE INDEX idx_graph_metrics_community ON graph_metrics(community_id);
   ```

3. **Create graph_paths table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS graph_paths (
       path_id TEXT PRIMARY KEY,
       entity1 TEXT NOT NULL,
       entity2 TEXT NOT NULL,
       path_length INTEGER NOT NULL,
       path_nodes TEXT NOT NULL,  -- JSON array
       path_weight REAL,
       computed_date TEXT NOT NULL
   );
   CREATE INDEX idx_graph_paths_entities ON graph_paths(entity1, entity2);
   ```

4. **Write migration script** (30 min)
   - Create `migrations/migration_v2_24_0.py`
   - Add schema verification
   - Test migration on development DB

**Dependencies:** None (uses existing entities/relationships tables)

**Testing:**
- Verify tables created successfully
- Test migration rollback
- Check indexes created

**Validation:**
```python
# Verify schema
assert 'graph_cache' in [t[0] for t in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
assert 'graph_metrics' in [t[0] for t in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
```

---

### Task 1.2: Knowledge Graph Builder (4 hours)

**Objective:** Implement core graph construction from existing entities/relationships

**Subtasks:**

1. **Implement build_knowledge_graph() method** (2 hours)
   ```python
   def build_knowledge_graph(self, entity_types: Optional[List[str]] = None,
                            min_occurrences: int = 2,
                            min_relationship_strength: float = 0.3) -> nx.Graph:
       """
       Build NetworkX graph from entities and relationships.

       Args:
           entity_types: Filter to specific entity types (None = all)
           min_occurrences: Minimum entity occurrences to include
           min_relationship_strength: Minimum relationship strength

       Returns:
           NetworkX graph with weighted edges
       """
       G = nx.Graph()

       # Add nodes from entities
       query = """
           SELECT text, type, COUNT(*) as occurrences
           FROM document_entities
           GROUP BY text, type
           HAVING occurrences >= ?
       """
       params = [min_occurrences]

       if entity_types:
           placeholders = ','.join('?' * len(entity_types))
           query += f" AND type IN ({placeholders})"
           params.extend(entity_types)

       entities = self.conn.execute(query, params).fetchall()

       for text, etype, count in entities:
           G.add_node(text,
                     type=etype,
                     occurrences=count,
                     weight=count)  # Node weight = occurrence count

       # Add edges from relationships
       rel_query = """
           SELECT entity1_text, entity2_text, strength, co_occurrence_count
           FROM entity_relationships
           WHERE strength >= ?
       """

       relationships = self.conn.execute(rel_query,
                                        [min_relationship_strength]).fetchall()

       for e1, e2, strength, count in relationships:
           if G.has_node(e1) and G.has_node(e2):
               G.add_edge(e1, e2,
                         weight=strength,
                         co_occurrences=count)

       # Cache graph
       self._cache_graph(G)

       return G
   ```

2. **Implement graph caching** (1 hour)
   ```python
   def _cache_graph(self, G: nx.Graph) -> str:
       """Cache graph to database for quick reloading."""
       import pickle

       cache_id = hashlib.sha256(
           f"{G.number_of_nodes()}{G.number_of_edges()}{time.time()}".encode()
       ).hexdigest()[:16]

       graph_data = pickle.dumps(G)

       self.conn.execute("""
           INSERT INTO graph_cache
           (cache_id, graph_version, graph_data, node_count, edge_count, created_date)
           VALUES (?, ?, ?, ?, ?, ?)
       """, (cache_id, 1, graph_data, G.number_of_nodes(),
             G.number_of_edges(), datetime.now().isoformat()))

       self.conn.commit()
       return cache_id

   def _load_cached_graph(self, cache_id: str) -> Optional[nx.Graph]:
       """Load cached graph from database."""
       import pickle

       row = self.conn.execute("""
           SELECT graph_data FROM graph_cache WHERE cache_id = ?
       """, (cache_id,)).fetchone()

       if row:
           # Update last accessed
           self.conn.execute("""
               UPDATE graph_cache SET last_accessed = ? WHERE cache_id = ?
           """, (datetime.now().isoformat(), cache_id))
           self.conn.commit()

           return pickle.loads(row[0])
       return None
   ```

3. **Add error handling and validation** (1 hour)
   - Validate graph is not empty
   - Handle disconnected components
   - Log graph statistics (nodes, edges, density)
   - Add progress reporting for large graphs

**Dependencies:** Task 1.1 (database schema)

**Testing:**
```python
def test_build_knowledge_graph():
    kb = KnowledgeBase()
    G = kb.build_knowledge_graph()

    assert G.number_of_nodes() > 0, "Graph should have nodes"
    assert G.number_of_edges() > 0, "Graph should have edges"

    # Check node attributes
    sample_node = list(G.nodes())[0]
    assert 'type' in G.nodes[sample_node]
    assert 'occurrences' in G.nodes[sample_node]

    # Check edge attributes
    if G.number_of_edges() > 0:
        sample_edge = list(G.edges())[0]
        assert 'weight' in G.edges[sample_edge]
```

**Validation:**
- Graph contains 100+ nodes (entities with min_occurrences=2)
- Graph contains 200+ edges (relationships)
- All nodes have required attributes (type, occurrences)
- All edges have weight attribute

---

### Task 1.3: Graph Analysis Algorithms (4 hours)

**Objective:** Implement PageRank, community detection, centrality measures

**Subtasks:**

1. **Implement PageRank analysis** (1 hour)
   ```python
   def analyze_pagerank(self, G: nx.Graph, alpha: float = 0.85,
                       max_iter: int = 100) -> Dict[str, float]:
       """
       Calculate PageRank for all entities in graph.

       Args:
           G: NetworkX graph
           alpha: Damping parameter (default 0.85)
           max_iter: Maximum iterations

       Returns:
           Dict mapping entity -> PageRank score
       """
       pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter,
                             weight='weight')

       # Store in database
       self._store_graph_metrics(pagerank, metric_type='pagerank')

       # Return top entities
       sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
       return dict(sorted_pr)
   ```

2. **Implement community detection** (1.5 hours)
   ```python
   def detect_communities(self, G: nx.Graph,
                         algorithm: str = 'louvain') -> Dict[str, int]:
       """
       Detect communities in knowledge graph.

       Args:
           G: NetworkX graph
           algorithm: 'louvain', 'label_propagation', or 'greedy_modularity'

       Returns:
           Dict mapping entity -> community_id
       """
       if algorithm == 'louvain':
           communities = nx.community.louvain_communities(G, weight='weight')
       elif algorithm == 'label_propagation':
           communities = nx.community.label_propagation_communities(G)
       elif algorithm == 'greedy_modularity':
           communities = nx.community.greedy_modularity_communities(G,
                                                                    weight='weight')
       else:
           raise ValueError(f"Unknown algorithm: {algorithm}")

       # Convert to dict
       entity_to_community = {}
       for idx, community in enumerate(communities):
           for entity in community:
               entity_to_community[entity] = idx

       # Store in database
       self._store_graph_metrics(entity_to_community,
                                metric_type='community')

       return entity_to_community
   ```

3. **Implement centrality measures** (1 hour)
   ```python
   def calculate_centrality(self, G: nx.Graph) -> Dict[str, Dict[str, float]]:
       """
       Calculate multiple centrality measures.

       Returns:
           Dict with keys: betweenness, closeness, degree
       """
       centrality = {
           'betweenness': nx.betweenness_centrality(G, weight='weight'),
           'closeness': nx.closeness_centrality(G, distance='weight'),
           'degree': nx.degree_centrality(G)
       }

       # Store all metrics
       for metric_type, values in centrality.items():
           self._store_graph_metrics(values, metric_type=metric_type)

       return centrality
   ```

4. **Implement shortest path finding** (30 min)
   ```python
   def find_shortest_path(self, G: nx.Graph, entity1: str,
                         entity2: str) -> Optional[List[str]]:
       """Find shortest path between two entities."""
       try:
           path = nx.shortest_path(G, entity1, entity2, weight='weight')

           # Cache path
           self._cache_path(entity1, entity2, path)

           return path
       except nx.NetworkXNoPath:
           return None
   ```

**Dependencies:** Task 1.2 (graph builder)

**Testing:**
```python
def test_graph_analysis():
    kb = KnowledgeBase()
    G = kb.build_knowledge_graph()

    # Test PageRank
    pr = kb.analyze_pagerank(G)
    assert len(pr) == G.number_of_nodes()
    assert all(0 <= score <= 1 for score in pr.values())

    # Test communities
    communities = kb.detect_communities(G)
    assert len(communities) == G.number_of_nodes()
    assert len(set(communities.values())) >= 2  # At least 2 communities

    # Test centrality
    centrality = kb.calculate_centrality(G)
    assert 'betweenness' in centrality
    assert 'closeness' in centrality
```

**Validation:**
- PageRank scores sum to ~1.0
- All centrality scores in range [0, 1]
- Community detection finds 5-20 communities
- Top PageRank entities are known important concepts (VIC-II, SID, etc.)

---

### Task 1.4: MCP Tool Integration (3 hours)

**Objective:** Add 6 new MCP tools for Claude to use graph analysis

**Tools to implement:**

1. **build_knowledge_graph** (30 min)
   ```python
   {
       "name": "build_knowledge_graph",
       "description": "Build knowledge graph from entities and relationships",
       "inputSchema": {
           "type": "object",
           "properties": {
               "entity_types": {
                   "type": "array",
                   "items": {"type": "string"},
                   "description": "Filter to specific entity types (optional)"
               },
               "min_occurrences": {
                   "type": "integer",
                   "default": 2,
                   "description": "Minimum entity occurrences"
               }
           }
       }
   }
   ```

2. **analyze_graph_pagerank** (30 min)
3. **detect_graph_communities** (30 min)
4. **calculate_graph_centrality** (30 min)
5. **find_entity_path** (30 min)
6. **get_graph_statistics** (30 min)

**Implementation in call_tool():**
```python
elif name == "analyze_graph_pagerank":
    G = self.build_knowledge_graph(
        entity_types=arguments.get("entity_types"),
        min_occurrences=arguments.get("min_occurrences", 2)
    )

    pagerank = self.analyze_pagerank(G, alpha=arguments.get("alpha", 0.85))
    top_n = arguments.get("top_n", 20)

    # Format results
    results = []
    for entity, score in list(pagerank.items())[:top_n]:
        node_data = G.nodes[entity]
        results.append({
            "entity": entity,
            "type": node_data['type'],
            "pagerank": round(score, 6),
            "occurrences": node_data['occurrences']
        })

    return [types.TextContent(
        type="text",
        text=json.dumps(results, indent=2)
    )]
```

**Dependencies:** Task 1.3 (analysis algorithms)

**Testing:**
- Test each MCP tool via call_tool()
- Verify JSON schema validation
- Test with real Claude queries
- Check error handling for invalid inputs

**Validation:**
- All 6 tools callable via MCP
- Tools return valid JSON
- Error messages are helpful

---

### Task 1.5: Basic Graph Visualization (3 hours)

**Objective:** Generate interactive HTML visualizations with PyVis

**Subtasks:**

1. **Implement PyVis graph renderer** (1.5 hours)
   ```python
   def visualize_knowledge_graph(self, G: nx.Graph,
                                 output_path: str = "knowledge_graph.html",
                                 color_by: str = "type",
                                 size_by: str = "pagerank",
                                 show_labels: bool = True,
                                 max_nodes: int = 100) -> str:
       """
       Generate interactive HTML visualization of knowledge graph.

       Args:
           G: NetworkX graph
           output_path: Output HTML file path
           color_by: Node color attribute (type, community)
           size_by: Node size attribute (pagerank, occurrences)
           show_labels: Show node labels
           max_nodes: Maximum nodes to display

       Returns:
           Path to generated HTML file
       """
       from pyvis.network import Network

       # Calculate metrics if needed
       if size_by == "pagerank" and 'pagerank' not in next(iter(G.nodes.values())):
           pagerank = self.analyze_pagerank(G)
           nx.set_node_attributes(G, pagerank, 'pagerank')

       # Filter to top nodes by centrality if too large
       if G.number_of_nodes() > max_nodes:
           centrality = nx.degree_centrality(G)
           top_nodes = sorted(centrality.items(),
                            key=lambda x: x[1],
                            reverse=True)[:max_nodes]
           G = G.subgraph([n for n, _ in top_nodes])

       # Create PyVis network
       net = Network(height="750px", width="100%",
                    bgcolor="#222222", font_color="white",
                    notebook=False)

       # Configure physics
       net.barnes_hut(gravity=-80000, central_gravity=0.3,
                     spring_length=250, spring_strength=0.001)

       # Add nodes with styling
       color_map = self._get_color_map(color_by)

       for node, attrs in G.nodes(data=True):
           size = attrs.get(size_by, 10)
           if size_by == "pagerank":
               size = size * 1000  # Scale PageRank to visible size

           color = color_map.get(attrs.get(color_by, "default"), "#97c2fc")

           net.add_node(node,
                       label=node if show_labels else "",
                       size=max(10, min(50, size)),
                       color=color,
                       title=self._get_node_tooltip(node, attrs))

       # Add edges
       for e1, e2, attrs in G.edges(data=True):
           net.add_edge(e1, e2,
                       value=attrs.get('weight', 1),
                       title=f"Strength: {attrs.get('weight', 0):.2f}")

       # Save
       net.save_graph(output_path)
       return output_path
   ```

2. **Add color scheme helpers** (30 min)
   ```python
   def _get_color_map(self, color_by: str) -> Dict[str, str]:
       """Get color mapping for node attribute."""
       if color_by == "type":
           return {
               'hardware': '#e74c3c',      # Red
               'instruction': '#3498db',   # Blue
               'register': '#2ecc71',      # Green
               'memory_address': '#f39c12', # Orange
               'person': '#9b59b6',        # Purple
               'company': '#1abc9c',       # Teal
               'product': '#e67e22',       # Dark orange
               'default': '#95a5a6'        # Gray
           }
       elif color_by == "community":
           # Generate distinct colors for communities
           import colorsys
           num_communities = len(set(nx.get_node_attributes(G, 'community').values()))
           colors = []
           for i in range(num_communities):
               hue = i / num_communities
               rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
               colors.append('#%02x%02x%02x' % tuple(int(c * 255) for c in rgb))
           return {i: color for i, color in enumerate(colors)}
       return {}

   def _get_node_tooltip(self, node: str, attrs: Dict) -> str:
       """Generate HTML tooltip for node."""
       lines = [
           f"<b>{node}</b>",
           f"Type: {attrs.get('type', 'unknown')}",
           f"Occurrences: {attrs.get('occurrences', 0)}"
       ]
       if 'pagerank' in attrs:
           lines.append(f"PageRank: {attrs['pagerank']:.4f}")
       if 'community' in attrs:
           lines.append(f"Community: {attrs['community']}")
       return "<br>".join(lines)
   ```

3. **Add export formats** (1 hour)
   ```python
   def export_knowledge_graph(self, G: nx.Graph,
                             output_path: str,
                             format: str = "graphml") -> str:
       """
       Export knowledge graph to various formats.

       Formats:
           - graphml: GraphML XML (import to Gephi, yEd)
           - gexf: GEXF XML (Gephi native)
           - json: Node-link JSON
           - gml: Graph Modeling Language
       """
       if format == "graphml":
           nx.write_graphml(G, output_path)
       elif format == "gexf":
           nx.write_gexf(G, output_path)
       elif format == "json":
           from networkx.readwrite import json_graph
           data = json_graph.node_link_data(G)
           with open(output_path, 'w') as f:
               json.dump(data, f, indent=2)
       elif format == "gml":
           nx.write_gml(G, output_path)
       else:
           raise ValueError(f"Unknown format: {format}")

       return output_path
   ```

**Dependencies:** Task 1.3 (graph analysis)

**Testing:**
```python
def test_graph_visualization():
    kb = KnowledgeBase()
    G = kb.build_knowledge_graph()

    # Test HTML generation
    output = kb.visualize_knowledge_graph(G, "test_graph.html")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 10000  # Non-trivial file

    # Test export formats
    for fmt in ['graphml', 'json', 'gexf']:
        output = kb.export_knowledge_graph(G, f"test_graph.{fmt}", format=fmt)
        assert os.path.exists(output)
```

**Validation:**
- HTML visualization opens in browser
- Nodes are colored by type
- Nodes are sized by PageRank
- Edges show relationship strength
- Tooltips show entity details
- Graph is interactive (drag, zoom, pan)

---

### Task 1.6: Documentation and Testing (2 hours)

**Objective:** Document Phase 1 features and achieve 90%+ test coverage

**Subtasks:**

1. **Write comprehensive tests** (1 hour)
   - test_build_knowledge_graph()
   - test_pagerank_analysis()
   - test_community_detection()
   - test_centrality_measures()
   - test_shortest_path()
   - test_graph_visualization()
   - test_mcp_tools()

2. **Document API in ARCHITECTURE.md** (30 min)
   - Add "Knowledge Graph Analysis" section
   - Document all new methods
   - Include usage examples

3. **Update CHANGELOG.md** (15 min)
   - Add v2.24.0 entry for Phase 1

4. **Create user guide** (15 min)
   - Add section to README.md
   - Include example queries for Claude

**Dependencies:** All Phase 1 tasks

**Validation:**
- pytest test_server.py -v shows 90%+ coverage for new code
- Documentation builds without errors
- README examples work

---

### Phase 1 Completion Checklist

- [ ] Database schema extended (3 new tables, 5 indexes)
- [ ] Knowledge graph builder operational
- [ ] PageRank analysis implemented
- [ ] Community detection (3 algorithms)
- [ ] Centrality measures (betweenness, closeness, degree)
- [ ] Shortest path finding
- [ ] 6 new MCP tools added
- [ ] PyVis HTML visualization working
- [ ] Export to GraphML/GEXF/JSON
- [ ] 90%+ test coverage
- [ ] Documentation complete
- [ ] CHANGELOG updated

**Phase 1 Deliverables:**
- `server.py`: +500 lines (graph analysis methods)
- `migrations/migration_v2_24_0.py`: Database migration script
- `test_server.py`: +300 lines (comprehensive tests)
- `knowledge_graph.html`: Example visualization
- Updated: ARCHITECTURE.md, CHANGELOG.md, README.md

---

## Phase 2: Discovery (20-24 hours)

**Goal:** Implement topic modeling and document clustering

**Deliverables:**
- LDA, NMF, BERTopic topic models
- K-Means, DBSCAN, HDBSCAN clustering
- 8 new MCP tools
- Topic/cluster visualizations
- Document similarity search

---

### Task 2.1: Database Schema for Topics/Clusters (2 hours)

**Objective:** Add tables for storing topic models and cluster assignments

**Subtasks:**

1. **Create topics table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS topics (
       topic_id TEXT PRIMARY KEY,
       model_type TEXT NOT NULL,  -- 'lda', 'nmf', 'bertopic'
       topic_number INTEGER NOT NULL,
       top_words TEXT NOT NULL,   -- JSON array
       word_weights TEXT NOT NULL, -- JSON object
       num_documents INTEGER DEFAULT 0,
       coherence_score REAL,
       created_date TEXT NOT NULL,
       UNIQUE(model_type, topic_number)
   );
   CREATE INDEX idx_topics_model ON topics(model_type);
   ```

2. **Create document_topics table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS document_topics (
       assignment_id TEXT PRIMARY KEY,
       doc_id TEXT NOT NULL,
       topic_id TEXT NOT NULL,
       probability REAL NOT NULL,  -- Topic probability
       model_type TEXT NOT NULL,
       assigned_date TEXT NOT NULL,
       FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
       FOREIGN KEY (topic_id) REFERENCES topics(topic_id) ON DELETE CASCADE
   );
   CREATE INDEX idx_doc_topics_doc ON document_topics(doc_id);
   CREATE INDEX idx_doc_topics_topic ON document_topics(topic_id);
   ```

3. **Create clusters table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS clusters (
       cluster_id TEXT PRIMARY KEY,
       algorithm TEXT NOT NULL,  -- 'kmeans', 'dbscan', 'hdbscan'
       cluster_number INTEGER NOT NULL,
       centroid_vector BLOB,  -- Pickled numpy array
       num_documents INTEGER DEFAULT 0,
       representative_docs TEXT,  -- JSON array of doc_ids
       top_terms TEXT,  -- JSON array
       silhouette_score REAL,
       created_date TEXT NOT NULL,
       UNIQUE(algorithm, cluster_number)
   );
   CREATE INDEX idx_clusters_algorithm ON clusters(algorithm);
   ```

4. **Create document_clusters table** (30 min)
   ```sql
   CREATE TABLE IF NOT EXISTS document_clusters (
       assignment_id TEXT PRIMARY KEY,
       doc_id TEXT NOT NULL,
       cluster_id TEXT NOT NULL,
       distance REAL,  -- Distance to centroid
       algorithm TEXT NOT NULL,
       assigned_date TEXT NOT NULL,
       FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
       FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id) ON DELETE CASCADE
   );
   CREATE INDEX idx_doc_clusters_doc ON document_clusters(doc_id);
   CREATE INDEX idx_doc_clusters_cluster ON document_clusters(cluster_id);
   ```

**Dependencies:** Phase 1 complete

**Testing:**
- Verify all tables created
- Test foreign key constraints
- Verify indexes

---

### Task 2.2: Topic Modeling - LDA Implementation (4 hours)

**Objective:** Implement Latent Dirichlet Allocation for topic discovery

**Subtasks:**

1. **Prepare document corpus** (1 hour)
   ```python
   def _prepare_topic_model_corpus(self, min_df: int = 2,
                                   max_df: float = 0.8) -> Tuple:
       """
       Prepare TF-IDF corpus for topic modeling.

       Returns:
           (documents, vectorizer, tfidf_matrix)
       """
       from sklearn.feature_extraction.text import TfidfVectorizer

       # Get all document texts
       docs = []
       doc_ids = []
       for doc_id, doc in self.documents.items():
           chunks = self._get_chunks_db(doc_id)
           full_text = " ".join(chunk.content for chunk in chunks)
           docs.append(full_text)
           doc_ids.append(doc_id)

       # Create TF-IDF vectorizer
       vectorizer = TfidfVectorizer(
           max_df=max_df,
           min_df=min_df,
           stop_words='english',
           max_features=1000,
           ngram_range=(1, 2)
       )

       tfidf_matrix = vectorizer.fit_transform(docs)

       return doc_ids, vectorizer, tfidf_matrix
   ```

2. **Implement LDA model** (2 hours)
   ```python
   def train_lda_model(self, num_topics: int = 10,
                      max_iter: int = 100,
                      random_state: int = 42) -> Dict[str, Any]:
       """
       Train LDA topic model.

       Args:
           num_topics: Number of topics to discover
           max_iter: Maximum iterations
           random_state: Random seed

       Returns:
           Model statistics and topic assignments
       """
       from sklearn.decomposition import LatentDirichletAllocation

       doc_ids, vectorizer, tfidf_matrix = self._prepare_topic_model_corpus()

       # Train LDA
       lda = LatentDirichletAllocation(
           n_components=num_topics,
           max_iter=max_iter,
           learning_method='online',
           random_state=random_state,
           n_jobs=-1
       )

       doc_topic_dist = lda.fit_transform(tfidf_matrix)

       # Extract topics
       feature_names = vectorizer.get_feature_names_out()
       topics = []

       for topic_idx, topic in enumerate(lda.components_):
           top_indices = topic.argsort()[-10:][::-1]
           top_words = [feature_names[i] for i in top_indices]
           word_weights = {feature_names[i]: float(topic[i])
                          for i in top_indices}

           # Store topic
           topic_id = self._store_topic(
               model_type='lda',
               topic_number=topic_idx,
               top_words=top_words,
               word_weights=word_weights
           )

           topics.append({
               'topic_id': topic_id,
               'topic_number': topic_idx,
               'words': top_words,
               'weights': word_weights
           })

       # Assign documents to topics
       for doc_idx, doc_id in enumerate(doc_ids):
           topic_probs = doc_topic_dist[doc_idx]

           # Store top 3 topics for this document
           top_topic_indices = topic_probs.argsort()[-3:][::-1]

           for topic_idx in top_topic_indices:
               if topic_probs[topic_idx] > 0.05:  # Minimum probability
                   self._assign_document_to_topic(
                       doc_id=doc_id,
                       topic_id=topics[topic_idx]['topic_id'],
                       probability=float(topic_probs[topic_idx]),
                       model_type='lda'
                   )

       # Calculate perplexity
       perplexity = lda.perplexity(tfidf_matrix)

       return {
           'model_type': 'lda',
           'num_topics': num_topics,
           'topics': topics,
           'perplexity': perplexity,
           'num_documents': len(doc_ids)
       }
   ```

3. **Implement topic coherence calculation** (1 hour)
   ```python
   def calculate_topic_coherence(self, model_type: str = 'lda') -> float:
       """Calculate topic coherence using C_v metric."""
       # Implementation using gensim's CoherenceModel
       # Returns coherence score (higher = better)
       pass
   ```

**Dependencies:** Task 2.1

**Testing:**
```python
def test_lda_model():
    kb = KnowledgeBase()
    results = kb.train_lda_model(num_topics=5)

    assert results['num_topics'] == 5
    assert len(results['topics']) == 5
    assert results['perplexity'] > 0

    # Check topics stored in DB
    topics = kb.conn.execute(
        "SELECT COUNT(*) FROM topics WHERE model_type='lda'"
    ).fetchone()[0]
    assert topics == 5
```

---

### Task 2.3: Topic Modeling - NMF and BERTopic (4 hours)

**Objective:** Implement NMF and BERTopic as alternative topic models

**Subtasks:**

1. **Implement NMF model** (1.5 hours)
   - Similar to LDA but uses Non-negative Matrix Factorization
   - Often produces more coherent topics

2. **Implement BERTopic** (2 hours)
   - Uses embeddings + UMAP + HDBSCAN
   - State-of-the-art topic modeling
   ```python
   def train_bertopic_model(self, num_topics: int = 10) -> Dict[str, Any]:
       """Train BERTopic model using existing embeddings."""
       from bertopic import BERTopic
       from umap import UMAP
       from hdbscan import HDBSCAN

       # Get documents and embeddings
       doc_ids = []
       documents = []
       embeddings = []

       for doc_id, doc in self.documents.items():
           chunks = self._get_chunks_db(doc_id)
           # Use first chunk embedding as document embedding
           if chunks:
               doc_ids.append(doc_id)
               documents.append(chunks[0].content)
               embeddings.append(chunks[0].embedding)

       embeddings = np.array(embeddings)

       # Configure BERTopic
       umap_model = UMAP(n_neighbors=15, n_components=5,
                        min_dist=0.0, metric='cosine')
       hdbscan_model = HDBSCAN(min_cluster_size=5,
                              metric='euclidean',
                              cluster_selection_method='eom')

       topic_model = BERTopic(
           umap_model=umap_model,
           hdbscan_model=hdbscan_model,
           nr_topics=num_topics
       )

       # Train
       topics, probs = topic_model.fit_transform(documents, embeddings)

       # Store topics and assignments
       # ... similar to LDA implementation

       return {
           'model_type': 'bertopic',
           'num_topics': len(set(topics)) - 1,  # -1 for outlier topic
           'topics': topic_model.get_topics()
       }
   ```

3. **Add topic comparison utility** (30 min)
   ```python
   def compare_topic_models(self) -> Dict[str, Any]:
       """Compare LDA, NMF, BERTopic models."""
       # Calculate coherence for each
       # Return comparison metrics
       pass
   ```

**Dependencies:** Task 2.2

---

### Task 2.4: Document Clustering (5 hours)

**Objective:** Implement K-Means, DBSCAN, HDBSCAN clustering

**Subtasks:**

1. **Implement K-Means clustering** (1.5 hours)
   ```python
   def cluster_documents_kmeans(self, num_clusters: int = 10) -> Dict[str, Any]:
       """
       Cluster documents using K-Means on embeddings.

       Args:
           num_clusters: Number of clusters (K)

       Returns:
           Clustering results with statistics
       """
       from sklearn.cluster import KMeans
       from sklearn.metrics import silhouette_score

       # Get document embeddings
       doc_ids = []
       embeddings = []

       for doc_id in self.documents.keys():
           chunks = self._get_chunks_db(doc_id)
           if chunks and chunks[0].embedding is not None:
               doc_ids.append(doc_id)
               embeddings.append(chunks[0].embedding)

       embeddings = np.array(embeddings)

       # Train K-Means
       kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
       labels = kmeans.fit_predict(embeddings)

       # Calculate silhouette score
       silhouette = silhouette_score(embeddings, labels)

       # Store clusters
       for cluster_num in range(num_clusters):
           cluster_docs = [doc_ids[i] for i, label in enumerate(labels)
                          if label == cluster_num]

           cluster_id = self._store_cluster(
               algorithm='kmeans',
               cluster_number=cluster_num,
               centroid=kmeans.cluster_centers_[cluster_num],
               doc_ids=cluster_docs,
               silhouette=silhouette
           )

           # Assign documents
           for doc_id in cluster_docs:
               doc_idx = doc_ids.index(doc_id)
               distance = np.linalg.norm(
                   embeddings[doc_idx] - kmeans.cluster_centers_[cluster_num]
               )
               self._assign_document_to_cluster(
                   doc_id, cluster_id, distance, 'kmeans'
               )

       return {
           'algorithm': 'kmeans',
           'num_clusters': num_clusters,
           'silhouette_score': silhouette,
           'num_documents': len(doc_ids)
       }
   ```

2. **Implement DBSCAN clustering** (1.5 hours)
   - Density-based clustering (no need to specify K)
   - Finds arbitrary shaped clusters

3. **Implement HDBSCAN clustering** (1.5 hours)
   - Hierarchical DBSCAN (better than DBSCAN)
   - Automatic cluster selection

4. **Add cluster quality metrics** (30 min)
   ```python
   def evaluate_clustering(self, algorithm: str) -> Dict[str, float]:
       """Evaluate clustering quality with multiple metrics."""
       # Silhouette score
       # Davies-Bouldin index
       # Calinski-Harabasz score
       pass
   ```

**Dependencies:** Task 2.1

**Testing:**
```python
def test_kmeans_clustering():
    kb = KnowledgeBase()
    results = kb.cluster_documents_kmeans(num_clusters=5)

    assert results['num_clusters'] == 5
    assert 0 <= results['silhouette_score'] <= 1

    # Check clusters in DB
    clusters = kb.conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE algorithm='kmeans'"
    ).fetchone()[0]
    assert clusters == 5
```

---

### Task 2.5: MCP Tools for Topics/Clusters (3 hours)

**Objective:** Add 8 new MCP tools

**Tools:**
1. train_lda_topics
2. train_nmf_topics
3. train_bertopic
4. get_document_topics
5. cluster_documents_kmeans
6. cluster_documents_dbscan
7. cluster_documents_hdbscan
8. get_cluster_documents

**Implementation:** Similar to Phase 1 MCP tools

---

### Task 2.6: Topic/Cluster Visualizations (4 hours)

**Objective:** Create interactive visualizations for topics and clusters

**Subtasks:**

1. **Topic word clouds** (1 hour)
   ```python
   def visualize_topics_wordcloud(self, model_type: str = 'lda',
                                 output_dir: str = "topic_wordclouds") -> List[str]:
       """Generate word cloud for each topic."""
       from wordcloud import WordCloud
       import matplotlib.pyplot as plt

       topics = self.conn.execute("""
           SELECT topic_number, word_weights FROM topics
           WHERE model_type = ?
           ORDER BY topic_number
       """, (model_type,)).fetchall()

       output_files = []
       for topic_num, word_weights_json in topics:
           word_weights = json.loads(word_weights_json)

           wc = WordCloud(width=800, height=400,
                         background_color='white').generate_from_frequencies(word_weights)

           plt.figure(figsize=(10, 5))
           plt.imshow(wc, interpolation='bilinear')
           plt.axis('off')
           plt.title(f"Topic {topic_num}")

           output_path = f"{output_dir}/topic_{model_type}_{topic_num}.png"
           plt.savefig(output_path, bbox_inches='tight', dpi=150)
           plt.close()

           output_files.append(output_path)

       return output_files
   ```

2. **Topic distribution charts** (1 hour)
   - Bar charts showing topic distribution across corpus
   - Plotly interactive charts

3. **Cluster scatter plots** (1 hour)
   ```python
   def visualize_clusters_2d(self, algorithm: str = 'kmeans',
                            output_path: str = "clusters_2d.html") -> str:
       """Visualize clusters in 2D using UMAP reduction."""
       import plotly.express as px
       from umap import UMAP

       # Get embeddings and labels
       doc_ids, embeddings, labels = self._get_cluster_data(algorithm)

       # Reduce to 2D with UMAP
       reducer = UMAP(n_components=2, random_state=42)
       embedding_2d = reducer.fit_transform(embeddings)

       # Create DataFrame
       df = pd.DataFrame({
           'x': embedding_2d[:, 0],
           'y': embedding_2d[:, 1],
           'cluster': labels,
           'doc_id': doc_ids,
           'title': [self.documents[doc_id].title for doc_id in doc_ids]
       })

       # Plot
       fig = px.scatter(df, x='x', y='y', color='cluster',
                       hover_data=['title'],
                       title=f"Document Clusters ({algorithm})")

       fig.write_html(output_path)
       return output_path
   ```

4. **Dendrogram for hierarchical clustering** (1 hour)

**Dependencies:** Tasks 2.2, 2.3, 2.4

---

### Task 2.7: Documentation and Testing (2-3 hours)

**Dependencies:** All Phase 2 tasks

---

### Phase 2 Completion Checklist

- [ ] Database schema for topics/clusters (4 tables)
- [ ] LDA topic modeling
- [ ] NMF topic modeling
- [ ] BERTopic implementation
- [ ] K-Means clustering
- [ ] DBSCAN clustering
- [ ] HDBSCAN clustering
- [ ] 8 new MCP tools
- [ ] Topic word clouds
- [ ] Cluster visualizations
- [ ] 90%+ test coverage
- [ ] Documentation complete

---

## Phase 3: Temporal (16-20 hours)

**Goal:** Extract timelines and events, implement advanced visualizations

**Deliverables:**
- Date/time extraction from documents
- Event detection and ordering
- Timeline visualizations
- Trend analysis
- Historical C64 timeline

---

### Task 3.1: Database Schema for Events/Timeline (1.5 hours)

**Tables:**
- events
- document_events
- timeline_entries

---

### Task 3.2: Date/Time Extraction (3 hours)

**Objective:** Extract dates and temporal references from documents

**Subtasks:**
1. Regex-based date extraction (1 hour)
2. NLP-based temporal extraction with SpaCy (1.5 hours)
3. Normalize dates to timeline format (30 min)

---

### Task 3.3: Event Detection (3 hours)

**Objective:** Identify significant events in C64 history

**Extraction patterns:**
- Product releases ("released in 1982")
- Company milestones ("founded", "acquired")
- Technical innovations ("first to...", "introduced")
- Cultural events (competitions, demos)

---

### Task 3.4: Timeline Construction (2 hours)

**Objective:** Build chronological timeline from events

---

### Task 3.5: MCP Tools for Timeline (2 hours)

**Tools:**
- extract_document_events
- get_timeline
- search_events_by_date
- get_historical_context

---

### Task 3.6: Timeline Visualizations (4 hours)

**Subtasks:**
1. Interactive timeline (Plotly) (1.5 hours)
2. Event network visualization (1.5 hours)
3. Trend charts (1 hour)

---

### Task 3.7: Advanced Graph Visualizations (3 hours)

**Subtasks:**
1. 3D knowledge graph (1 hour)
2. Hierarchical edge bundling (1 hour)
3. Sankey diagrams for topic flow (1 hour)

---

### Task 3.8: Documentation and Testing (2-3 hours)

---

### Phase 3 Completion Checklist

- [ ] Date/time extraction working
- [ ] Event detection operational
- [ ] Timeline construction
- [ ] 4 new MCP tools
- [ ] Interactive timeline visualization
- [ ] Event network graph
- [ ] Trend analysis
- [ ] Advanced graph visualizations
- [ ] Documentation complete

---

## Phase 4: Integration (12-16 hours)

**Goal:** Create unified Streamlit analytics dashboard

**Deliverables:**
- Multi-tab Streamlit interface
- Knowledge graph explorer
- Topic discovery interface
- Cluster navigator
- Timeline viewer
- Search and export features

---

### Task 4.1: Dashboard Architecture (2 hours)

**Objective:** Design multi-page Streamlit app structure

**Pages:**
1. Overview / Statistics
2. Knowledge Graph Explorer
3. Topic Discovery
4. Document Clusters
5. Timeline & Events
6. Search & Export

---

### Task 4.2: Knowledge Graph Page (3 hours)

**Features:**
- Interactive graph controls
- Entity search
- Path finding interface
- Community filtering
- Export options

---

### Task 4.3: Topic Discovery Page (2 hours)

**Features:**
- Model comparison (LDA vs NMF vs BERTopic)
- Topic explorer with word clouds
- Document browser by topic
- Topic trends over time

---

### Task 4.4: Cluster Navigator Page (2 hours)

**Features:**
- Algorithm comparison
- 2D/3D cluster visualization
- Cluster details panel
- Similar document finder

---

### Task 4.5: Timeline Page (2 hours)

**Features:**
- Interactive timeline
- Date range filtering
- Event search
- Historical context viewer

---

### Task 4.6: Search & Export Page (2 hours)

**Features:**
- Unified search across all features
- Export knowledge graph
- Export topics/clusters
- Export timeline data
- Batch analysis tools

---

### Task 4.7: Testing and Polish (2-3 hours)

**Subtasks:**
1. Integration testing (1 hour)
2. Performance optimization (1 hour)
3. UI/UX polish (1 hour)

---

### Phase 4 Completion Checklist

- [ ] Streamlit dashboard operational
- [ ] 6 dashboard pages complete
- [ ] Knowledge graph explorer
- [ ] Topic discovery interface
- [ ] Cluster navigator
- [ ] Timeline viewer
- [ ] Search and export
- [ ] Documentation complete
- [ ] User guide created

---

## Dependencies

### External Libraries

Install all dependencies:
```cmd
pip install networkx>=3.0 scikit-learn>=1.3.0 plotly>=5.0.0 pyvis>=0.3.0 hdbscan>=0.8.0 bertopic umap-learn wordcloud
```

**Phase 1:**
- networkx >= 3.0
- pyvis >= 0.3.0
- plotly >= 5.0.0

**Phase 2:**
- scikit-learn >= 1.3.0
- hdbscan >= 0.8.0
- bertopic
- umap-learn
- wordcloud

**Phase 3:**
- spacy >= 3.0.0
- python-dateutil

**Phase 4:**
- streamlit (already installed)

### Task Dependencies

**Phase 2 depends on:**
- Phase 1 complete (uses knowledge graph)

**Phase 3 depends on:**
- Phase 1 complete (entity extraction)
- Phase 2 complete (topic models for event context)

**Phase 4 depends on:**
- All phases 1-3 complete

---

## Risk Assessment

### High Risk Items

1. **BERTopic Performance** (Phase 2, Task 2.3)
   - **Risk:** May be slow on 199 documents with embeddings
   - **Mitigation:** Implement caching, use UMAP pre-reduction
   - **Fallback:** Skip BERTopic, use LDA/NMF only

2. **Graph Visualization Size** (Phase 1, Task 1.5)
   - **Risk:** 100+ node graphs may be slow in browser
   - **Mitigation:** Limit to top N nodes, implement filtering
   - **Fallback:** Server-side rendering with static images

3. **Timeline Extraction Accuracy** (Phase 3, Task 3.2)
   - **Risk:** Date extraction may have low precision
   - **Mitigation:** Use multiple extraction methods, manual validation
   - **Fallback:** Focus on explicit dates only

### Medium Risk Items

1. **Community Detection Quality**
   - **Risk:** May produce too many/few communities
   - **Mitigation:** Test multiple algorithms, tune parameters

2. **Topic Coherence**
   - **Risk:** Topics may not be semantically coherent
   - **Mitigation:** Calculate coherence metrics, tune num_topics

3. **Dashboard Performance**
   - **Risk:** Streamlit may be slow with large visualizations
   - **Mitigation:** Lazy loading, caching, pagination

---

## Testing Strategy

### Unit Tests

Each phase requires comprehensive unit tests:

**Phase 1:**
- test_build_knowledge_graph()
- test_pagerank()
- test_community_detection()
- test_graph_visualization()

**Phase 2:**
- test_lda_model()
- test_nmf_model()
- test_kmeans_clustering()
- test_topic_visualization()

**Phase 3:**
- test_date_extraction()
- test_event_detection()
- test_timeline_construction()

**Phase 4:**
- test_dashboard_pages()
- test_integration()

### Integration Tests

- Test full pipeline: extract → analyze → visualize
- Test MCP tool chains
- Test dashboard navigation

### Performance Tests

- Benchmark graph analysis on 100+ nodes
- Measure topic modeling time
- Profile dashboard load times

### Target Metrics

- **Test Coverage:** 90%+ for all new code
- **Performance:** All operations < 10 seconds
- **Reliability:** Zero crashes on valid inputs

---

## Validation Criteria

### Phase 1 Success Criteria

- [ ] Knowledge graph with 100+ nodes, 200+ edges
- [ ] PageRank identifies key entities (VIC-II, SID rank high)
- [ ] Community detection finds 5-20 coherent groups
- [ ] HTML visualization opens and is interactive
- [ ] All 6 MCP tools callable
- [ ] Tests pass with 90%+ coverage

### Phase 2 Success Criteria

- [ ] LDA produces 10 coherent topics
- [ ] Topics align with known C64 domains (graphics, sound, programming)
- [ ] Clustering groups similar documents
- [ ] Silhouette score > 0.3
- [ ] All 8 MCP tools callable
- [ ] Visualizations generated successfully

### Phase 3 Success Criteria

- [ ] Extract 50+ events from documents
- [ ] Timeline spans 1975-1995
- [ ] Key events identified (C64 release in 1982, etc.)
- [ ] Timeline visualization shows chronological flow
- [ ] All 4 MCP tools callable

### Phase 4 Success Criteria

- [ ] Dashboard loads in < 5 seconds
- [ ] All 6 pages functional
- [ ] Knowledge graph explorer shows real-time updates
- [ ] Topic/cluster navigation works
- [ ] Timeline is interactive
- [ ] Export features generate valid files

---

## Implementation Schedule

**Recommended order:**

### Week 1: Foundation
- Days 1-2: Phase 1, Tasks 1.1-1.3 (Database + Graph Building)
- Days 3-4: Phase 1, Tasks 1.4-1.5 (MCP Tools + Visualization)
- Day 5: Phase 1, Task 1.6 (Documentation + Testing)

### Week 2: Discovery Part 1
- Days 1-2: Phase 2, Tasks 2.1-2.2 (Schema + LDA)
- Days 3-4: Phase 2, Task 2.3 (NMF + BERTopic)
- Day 5: Phase 2, Task 2.4 start (K-Means)

### Week 3: Discovery Part 2
- Days 1-2: Phase 2, Task 2.4 complete (DBSCAN + HDBSCAN)
- Days 3-4: Phase 2, Tasks 2.5-2.6 (MCP Tools + Viz)
- Day 5: Phase 2, Task 2.7 (Documentation + Testing)

### Week 4: Temporal
- Days 1-2: Phase 3, Tasks 3.1-3.3 (Schema + Extraction)
- Days 3-4: Phase 3, Tasks 3.4-3.6 (Timeline + MCP Tools + Viz)
- Day 5: Phase 3, Tasks 3.7-3.8 (Advanced Viz + Testing)

### Week 5: Integration
- Days 1-2: Phase 4, Tasks 4.1-4.3 (Architecture + Graph/Topic pages)
- Days 3-4: Phase 4, Tasks 4.4-4.6 (Cluster/Timeline/Export pages)
- Day 5: Phase 4, Task 4.7 (Testing + Polish)

**Total: 5 weeks (~25 working days)**

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Set up development environment**
   ```cmd
   pip install networkx scikit-learn plotly pyvis hdbscan bertopic umap-learn wordcloud
   ```
3. **Create feature branch**
   ```cmd
   git checkout -b feature/knowledge-extraction-v2.24
   ```
4. **Start with Phase 1, Task 1.1**

---

## Appendix: Code Examples

### Example 1: Using Knowledge Graph via MCP

```python
# Claude query: "Show me the most important entities in the C64 knowledge base"

# Claude calls MCP tool:
{
    "tool": "analyze_graph_pagerank",
    "arguments": {
        "top_n": 10
    }
}

# Response:
[
    {"entity": "VIC-II", "type": "hardware", "pagerank": 0.0324, "occurrences": 156},
    {"entity": "SID", "type": "hardware", "pagerank": 0.0298, "occurrences": 142},
    {"entity": "6510", "type": "hardware", "pagerank": 0.0276, "occurrences": 89},
    ...
]
```

### Example 2: Topic Discovery

```python
# Train LDA model
results = kb.train_lda_model(num_topics=10)

# Results:
{
    "model_type": "lda",
    "num_topics": 10,
    "topics": [
        {
            "topic_number": 0,
            "words": ["sprite", "vic", "graphics", "screen", "bitmap"],
            "interpretation": "Graphics Programming"
        },
        {
            "topic_number": 1,
            "words": ["sid", "sound", "music", "frequency", "waveform"],
            "interpretation": "Audio/Music"
        },
        ...
    ]
}
```

### Example 3: Timeline Query

```python
# Get events in 1982
events = kb.get_timeline(start_date="1982-01-01", end_date="1982-12-31")

# Results:
[
    {
        "date": "1982-08",
        "event": "Commodore 64 released",
        "type": "product_release",
        "source_docs": ["doc_id_123", "doc_id_456"]
    },
    ...
]
```

---

**End of Implementation Plan**

This plan provides a complete roadmap for implementing all 6 knowledge extraction algorithms across 4 phases. Each task is broken down into 1-4 hour chunks with specific deliverables, dependencies, and validation criteria.

**Total Effort:** 64-80 hours
**Timeline:** 5 weeks (recommended)
**Risk Level:** Medium (mitigations in place)

Ready to begin implementation with Phase 1, Task 1.1.
