# Advanced Knowledge Extraction Algorithms

**Version:** 2.23.16
**Status:** Design Proposal
**Last Updated:** 2026-01-03

Proposal for extracting deeper insights from the TDZ C64 Knowledge Base using advanced algorithms and making this information accessible through new MCP tools and visualizations.

---

## 📋 Current Capabilities

**Existing Knowledge Extraction:**
- ✅ Entity extraction (50+ entity types, regex + LLM)
- ✅ Entity relationships (distance-based co-occurrence)
- ✅ Semantic embeddings (384-dimensional vectors, FAISS index)
- ✅ Document comparison (metadata, content, full)
- ✅ Faceted search (entity-based filtering)
- ✅ RAG question answering

**Current Statistics:**
- 199 documents
- 5,044 chunks
- 7.4M words
- Semantic embeddings built
- Entity extraction available

---

## 🎯 Proposed Knowledge Extraction Algorithms

### 1. Knowledge Graph Analysis ⭐⭐⭐⭐⭐

**What:** Build and analyze a graph database of entities and their relationships

**Algorithms:**
- **PageRank** - Find most important/central entities
- **Community Detection** - Discover entity clusters (Louvain algorithm)
- **Shortest Path** - Find connections between concepts
- **Centrality Measures** - Identify hub entities (betweenness, closeness)
- **Graph Clustering** - Group related entities

**Example Insights:**
```
Most Central Entities (PageRank):
1. VIC-II (score: 0.234) - Connected to 127 other entities
2. SID chip (score: 0.198) - Connected to 94 other entities
3. 6502 CPU (score: 0.176) - Connected to 82 other entities

Communities Detected:
- Audio/Music: SID, waveforms, filters, ADSR
- Graphics: VIC-II, sprites, rasters, colors
- Programming: 6502, assembly, BASIC, memory
```

**Implementation:**
```python
def build_knowledge_graph(self) -> nx.Graph:
    """Build NetworkX graph from entities and relationships."""
    G = nx.Graph()

    # Add nodes (entities)
    entities = self._get_all_entities()
    for entity in entities:
        G.add_node(entity['text'],
                  type=entity['type'],
                  occurrences=entity['count'])

    # Add edges (relationships)
    relationships = self._get_all_relationships()
    for rel in relationships:
        G.add_edge(rel['entity1'], rel['entity2'],
                  weight=rel['strength'],
                  co_occurrences=rel['count'])

    return G

def analyze_knowledge_graph(self, G: nx.Graph) -> dict:
    """Run graph algorithms to extract insights."""
    return {
        'pagerank': nx.pagerank(G, weight='weight'),
        'communities': nx.community.louvain_communities(G),
        'centrality': nx.betweenness_centrality(G),
        'clustering': nx.clustering(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else None
    }

def find_entity_path(self, entity1: str, entity2: str) -> list:
    """Find shortest path between two entities."""
    G = self.build_knowledge_graph()
    try:
        path = nx.shortest_path(G, entity1, entity2)
        return {
            'path': path,
            'length': len(path) - 1,
            'explanation': self._explain_path(path)
        }
    except nx.NetworkXNoPath:
        return {'path': None, 'message': 'No connection found'}
```

**MCP Tools:**
- `analyze_knowledge_graph` - Get graph statistics
- `find_entity_path` - Find connections between concepts
- `get_central_entities` - List most important entities
- `discover_communities` - Find entity clusters

---

### 2. Topic Modeling ⭐⭐⭐⭐

**What:** Discover hidden topics in the corpus using unsupervised learning

**Algorithms:**
- **LDA (Latent Dirichlet Allocation)** - Probabilistic topic modeling
- **NMF (Non-negative Matrix Factorization)** - Linear algebra approach
- **BERTopic** - Modern topic modeling with transformers

**Example Insights:**
```
Topic 0: Graphics Programming (18% of corpus)
Top words: sprite, VIC-II, raster, screen, color, bitmap
Documents: 34 documents strongly associated

Topic 1: Audio/Music (15% of corpus)
Top words: SID, sound, waveform, frequency, filter, ADSR
Documents: 28 documents strongly associated

Topic 2: Hardware Architecture (12% of corpus)
Top words: memory, address, register, chip, CIA, RAM
Documents: 22 documents strongly associated
```

**Implementation:**
```python
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def discover_topics(self, num_topics: int = 10, method: str = 'lda') -> dict:
    """
    Discover topics in the corpus.

    Args:
        num_topics: Number of topics to extract
        method: 'lda', 'nmf', or 'bertopic'

    Returns:
        {
            'topics': [{'id': 0, 'top_words': [...], 'weight': 0.18}],
            'document_topics': {doc_id: [topic_weights]},
            'topic_evolution': {...}  # How topics change over time
        }
    """
    # Get all document texts
    texts = [self._get_document_text(doc_id)
             for doc_id in self.documents.keys()]

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000,
                                 stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)

    # Apply topic model
    if method == 'lda':
        model = LatentDirichletAllocation(n_components=num_topics,
                                         random_state=42)
    elif method == 'nmf':
        model = NMF(n_components=num_topics, random_state=42)

    doc_topic_matrix = model.fit_transform(doc_term_matrix)

    # Extract topics
    topics = []
    feature_names = vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append({
            'id': topic_idx,
            'top_words': top_words,
            'weight': doc_topic_matrix[:, topic_idx].mean()
        })

    return {
        'topics': topics,
        'document_topics': self._assign_topics(doc_topic_matrix),
        'model': model
    }

def get_topic_timeline(self, topic_id: int) -> dict:
    """Track how a topic evolves over time."""
    # Analyze documents by date, track topic prevalence
    pass
```

**MCP Tools:**
- `discover_topics` - Extract topics from corpus
- `get_document_topics` - Show topics for a document
- `search_by_topic` - Find documents in a topic
- `get_topic_timeline` - Track topic evolution

---

### 3. Document Clustering ⭐⭐⭐⭐

**What:** Group similar documents into clusters

**Algorithms:**
- **K-Means** - Partition documents into K clusters
- **Hierarchical Clustering** - Build dendrogram of document relationships
- **DBSCAN** - Density-based clustering (finds natural clusters)
- **HDBSCAN** - Hierarchical DBSCAN (better for varying densities)

**Example Insights:**
```
Cluster 0: VIC-II Programming Guides (23 documents)
- Average similarity: 0.87
- Key topics: sprites, rasters, interrupts
- Recommended for: Graphics programming

Cluster 1: SID Music Tutorials (18 documents)
- Average similarity: 0.91
- Key topics: waveforms, ADSR, filters
- Recommended for: Music composition

Cluster 2: Memory Maps & Hardware Reference (31 documents)
- Average similarity: 0.76
- Key topics: addresses, registers, I/O
- Recommended for: Hardware understanding
```

**Implementation:**
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def cluster_documents(self, n_clusters: int = 10,
                     method: str = 'kmeans') -> dict:
    """
    Cluster documents by similarity.

    Args:
        n_clusters: Number of clusters (for kmeans, hierarchical)
        method: 'kmeans', 'hierarchical', 'dbscan', 'hdbscan'

    Returns:
        {
            'clusters': [{
                'id': 0,
                'documents': [doc_ids],
                'centroid_topics': [...],
                'avg_similarity': 0.87,
                'size': 23
            }],
            'quality_score': 0.65  # Silhouette score
        }
    """
    # Get embeddings for all documents
    embeddings = self._get_document_embeddings()

    # Apply clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.3, min_samples=2)

    cluster_labels = clusterer.fit_predict(embeddings)

    # Build cluster info
    clusters = self._build_cluster_info(cluster_labels, embeddings)

    # Calculate quality
    quality = silhouette_score(embeddings, cluster_labels)

    return {
        'clusters': clusters,
        'quality_score': quality,
        'method': method
    }

def get_cluster_summary(self, cluster_id: int) -> dict:
    """Get detailed summary of a cluster."""
    return {
        'documents': [...],
        'common_entities': [...],  # Most frequent entities
        'common_topics': [...],     # Topic distribution
        'representative_doc': doc_id,  # Document closest to centroid
        'outliers': [...]            # Documents on cluster edge
    }
```

**MCP Tools:**
- `cluster_documents` - Group similar documents
- `get_cluster_summary` - Detailed cluster analysis
- `find_document_cluster` - Which cluster is a document in?
- `recommend_similar_clusters` - Find related clusters

---

### 4. Timeline & Event Extraction ⭐⭐⭐⭐

**What:** Extract temporal information and build chronological timelines

**Algorithms:**
- **Date/Time Extraction** - Regex + NER for dates
- **Event Detection** - Identify significant events
- **Temporal Ordering** - Order events chronologically
- **Trend Analysis** - Track how topics/entities change over time

**Example Insights:**
```
C64 Historical Timeline:

1982:
- VIC-II chip released (MOS Technology)
- C64 announced at CES
- First units shipped

1983:
- SID chip documented (Bob Yannes)
- BASIC 2.0 released
- 1 million units sold

1985:
- C64C revision released
- SID 8580 variant introduced

Technology Evolution:
- Early focus: Hardware specs, memory maps
- Mid-period: Programming techniques, demos
- Later: Preservation, emulation, modern tools
```

**Implementation:**
```python
import re
from datetime import datetime

def extract_timeline(self) -> dict:
    """
    Extract chronological timeline from documents.

    Returns:
        {
            'events': [{
                'date': '1982-08-01',
                'event': 'C64 released',
                'entities': ['Commodore', 'VIC-II', 'SID'],
                'source_docs': [doc_ids],
                'confidence': 0.95
            }],
            'periods': [{
                'start': '1982',
                'end': '1985',
                'name': 'Early Era',
                'characteristics': [...]
            }]
        }
    """
    events = []

    # Extract dates and events from all documents
    for doc_id, doc in self.documents.items():
        text = self._get_document_text(doc_id)

        # Find dates (multiple patterns)
        dates = self._extract_dates(text)

        # Find events near dates
        for date in dates:
            event_text = self._extract_event_near_date(text, date)
            entities = self._extract_entities_from_text(event_text)

            events.append({
                'date': date,
                'event': event_text,
                'entities': entities,
                'source_doc': doc_id,
                'confidence': self._calculate_confidence(event_text)
            })

    # Sort chronologically and group into periods
    events_sorted = sorted(events, key=lambda x: x['date'])
    periods = self._identify_periods(events_sorted)

    return {
        'events': events_sorted,
        'periods': periods,
        'entity_timeline': self._build_entity_timeline(events_sorted)
    }

def get_entity_timeline(self, entity: str) -> dict:
    """Track an entity through time."""
    return {
        'entity': entity,
        'first_mention': '1982-08',
        'last_mention': '2025-12',
        'mentions_by_year': {
            '1982': 5,
            '1983': 12,
            ...
        },
        'key_events': [...]
    }
```

**MCP Tools:**
- `extract_timeline` - Build chronological timeline
- `get_entity_timeline` - Track entity over time
- `search_by_period` - Find documents from time period
- `get_technology_evolution` - See how tech evolved

---

### 5. Concept Network & Visualization ⭐⭐⭐⭐⭐

**What:** Build interactive concept maps and knowledge visualizations

**Algorithms:**
- **Force-Directed Layout** - Position nodes based on relationships
- **Hierarchical Layout** - Tree structure visualization
- **Topic Heatmaps** - Visualize topic distributions
- **Timeline Visualization** - Interactive chronological view

**Example Visualizations:**

```
1. Interactive Knowledge Graph
   - Nodes: Entities (sized by importance)
   - Edges: Relationships (thickness = strength)
   - Colors: Entity types
   - Interactive: Click to explore, zoom, filter

2. Topic Distribution Heatmap
   - X-axis: Documents
   - Y-axis: Topics
   - Color: Topic strength
   - Reveals document-topic patterns

3. Timeline Visualization
   - X-axis: Time
   - Y-axis: Categories (hardware, software, people)
   - Events plotted chronologically
   - Zoom and filter by entity

4. Cluster Dendrogram
   - Hierarchical tree of document relationships
   - Cut at different levels for varying granularity
   - Interactive branch exploration
```

**Implementation:**
```python
def generate_knowledge_graph_viz(self, format: str = 'html') -> str:
    """
    Generate interactive knowledge graph visualization.

    Args:
        format: 'html', 'json', 'graphml', 'gephi'

    Returns:
        Visualization data in requested format
    """
    import plotly.graph_objects as go
    # or: import networkx as nx; import pyvis

    G = self.build_knowledge_graph()

    # Calculate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Build Plotly figure
    edge_trace = go.Scatter(...)  # Draw edges
    node_trace = go.Scatter(...)  # Draw nodes

    fig = go.Figure(data=[edge_trace, node_trace])

    if format == 'html':
        return fig.to_html()
    elif format == 'json':
        return fig.to_json()

def generate_topic_heatmap(self) -> str:
    """Generate topic distribution heatmap."""
    topics = self.discover_topics()
    # Create heatmap showing document-topic relationships
    pass

def generate_timeline_viz(self) -> str:
    """Generate interactive timeline."""
    timeline = self.extract_timeline()
    # Create timeline visualization with plotly
    pass
```

**Output Formats:**
- HTML (interactive, self-contained)
- JSON (for custom frontends)
- GraphML/GEXF (for Gephi, Cytoscape)
- SVG/PNG (static images)

---

### 6. Advanced Analytics Dashboard ⭐⭐⭐⭐

**What:** Comprehensive analytics interface in Streamlit GUI

**Features:**
- **Knowledge Graph Explorer** - Interactive graph visualization
- **Topic Discovery** - Topic modeling results and trends
- **Document Clusters** - Cluster visualization and navigation
- **Timeline View** - Chronological event browser
- **Entity Analytics** - Deep dive into entity networks
- **Insights Panel** - Automatically generated insights

**Screens:**

```
Tab 1: Knowledge Graph
- Interactive graph with filters
- Search and highlight entities
- Show/hide entity types
- Explore neighborhoods

Tab 2: Topics
- Topic list with top words
- Document-topic heatmap
- Topic evolution over time
- Search by topic

Tab 3: Clusters
- Cluster dendrogram
- Cluster summaries
- Document distribution
- Similarity matrix

Tab 4: Timeline
- Chronological event list
- Filter by entity/period
- Timeline visualization
- Entity evolution tracks

Tab 5: Insights
- Automatically generated insights
- "Hidden patterns discovered"
- Anomalies and outliers
- Recommendations
```

---

## 📊 Implementation Priority

### Phase 1: Foundation (v2.24.0) - 16-20 hours
1. **Knowledge Graph Analysis** - Core infrastructure
   - Build NetworkX graph from existing entities
   - Implement PageRank, community detection
   - Add MCP tools for graph queries
   - Cache graph for performance

2. **Basic Visualizations** - Initial displays
   - Simple knowledge graph HTML export
   - Entity network visualization
   - Add to Streamlit GUI

### Phase 2: Discovery (v2.25.0) - 20-24 hours
3. **Topic Modeling** - Content analysis
   - Implement LDA topic discovery
   - Document-topic assignments
   - Topic search and filtering
   - MCP tools for topic queries

4. **Document Clustering** - Grouping
   - K-means clustering on embeddings
   - Cluster quality metrics
   - Cluster summaries and navigation
   - Recommendation engine

### Phase 3: Temporal (v2.26.0) - 16-20 hours
5. **Timeline Extraction** - Time-based analysis
   - Date/event extraction
   - Chronological ordering
   - Entity timelines
   - Period identification

6. **Advanced Visualizations** - Rich displays
   - Interactive timeline
   - Topic heatmaps
   - Cluster dendrograms
   - Export formats

### Phase 4: Integration (v2.27.0) - 12-16 hours
7. **Analytics Dashboard** - Unified interface
   - Multi-tab Streamlit interface
   - All visualizations integrated
   - Automated insights
   - Export capabilities

---

## 🎯 Benefits

### For Users
- **Discover Hidden Patterns** - Find connections not obvious from reading
- **Understand Topic Structure** - See major themes in documentation
- **Navigate by Similarity** - Find related documents automatically
- **Track Evolution** - See how topics/entities changed over time
- **Visual Exploration** - Interactive graphs and timelines

### For Researchers
- **Historical Analysis** - Track C64 history chronologically
- **Technology Evolution** - See how concepts developed
- **Community Detection** - Find related concepts/technologies
- **Gap Analysis** - Identify under-documented areas

### For Developers
- **API Access** - MCP tools for all analytics
- **Programmable** - Build custom queries and visualizations
- **Export Formats** - Use data in external tools
- **Integration Ready** - Connect to other systems

---

## 🛠️ Technical Requirements

### Dependencies
```bash
pip install networkx>=3.0
pip install scikit-learn>=1.3.0
pip install plotly>=5.0.0
pip install pyvis>=0.3.0
pip install hdbscan>=0.8.0
```

### Database Schema Additions
```sql
-- Store computed graph metrics
CREATE TABLE knowledge_graph_metrics (
    computed_at TEXT,
    metric_type TEXT,
    entity_id TEXT,
    value REAL,
    metadata TEXT  -- JSON
);

-- Store topic models
CREATE TABLE topic_models (
    model_id TEXT PRIMARY KEY,
    created_at TEXT,
    num_topics INTEGER,
    method TEXT,  -- lda, nmf, bertopic
    model_data BLOB,  -- Pickled model
    topics TEXT  -- JSON of topic info
);

-- Store document clusters
CREATE TABLE document_clusters (
    cluster_id INTEGER,
    doc_id TEXT,
    distance_to_centroid REAL,
    cluster_label TEXT,
    created_at TEXT
);

-- Store timeline events
CREATE TABLE timeline_events (
    event_id TEXT PRIMARY KEY,
    event_date TEXT,
    event_text TEXT,
    entities TEXT,  -- JSON array
    source_docs TEXT,  -- JSON array
    confidence REAL
);
```

---

## 🔬 Example Usage

```python
# Knowledge graph analysis
from server import KnowledgeBase

kb = KnowledgeBase()

# Analyze entity relationships
graph_stats = kb.analyze_knowledge_graph()
print(f"Most central entities: {graph_stats['pagerank'][:10]}")
print(f"Communities found: {len(graph_stats['communities'])}")

# Find connection between concepts
path = kb.find_entity_path("VIC-II", "SID chip")
print(f"Connection: {' → '.join(path['path'])}")

# Discover topics
topics = kb.discover_topics(num_topics=10)
for topic in topics['topics']:
    print(f"Topic {topic['id']}: {', '.join(topic['top_words'][:5])}")

# Cluster documents
clusters = kb.cluster_documents(n_clusters=15)
for cluster in clusters['clusters'][:5]:
    print(f"Cluster {cluster['id']}: {cluster['size']} docs")

# Extract timeline
timeline = kb.extract_timeline()
for event in timeline['events'][:10]:
    print(f"{event['date']}: {event['event']}")
```

---

## 📚 Next Steps

1. **Review & Approve** - Confirm scope and priorities
2. **Phase 1 Implementation** - Start with knowledge graph
3. **Testing & Validation** - Verify insights are meaningful
4. **Documentation** - Update guides with new features
5. **Iteration** - Refine based on user feedback

---

**Version:** 2.23.16
**Status:** Design Proposal - Ready for Implementation
**Last Updated:** 2026-01-03

**Questions? Feedback?** This is a comprehensive proposal. We can:
- Implement all phases sequentially
- Start with specific algorithms that interest you most
- Adjust scope based on priorities
- Prototype specific features first
