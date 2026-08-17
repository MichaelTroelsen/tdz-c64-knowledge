"""Topic modelling, document clustering and their visualisations.

Extracted from server.py's KnowledgeBase by R12 step 2. These are mixins,
not standalone classes: every method still expects the attributes
KnowledgeBase.__init__ sets up, and nothing here imports server, so the
class can be assembled in server.py without an import cycle.
"""
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import hashlib


class TopicsMixin:

    def _store_topics_to_db(self, topics: list, model_type: str) -> None:
        """Store topics to database."""
        import json
        from datetime import datetime
        import hashlib

        cursor = self.db_conn.cursor()
        timestamp = datetime.now().isoformat()

        # Clear existing topics for this model type
        cursor.execute("DELETE FROM topics WHERE model_type = ?", (model_type,))

        # Insert new topics
        for topic in topics:
            topic_id = topic['topic_id']

            cursor.execute("""
                INSERT INTO topics
                (topic_id, model_type, topic_number, top_words, word_weights,
                 num_documents, coherence_score, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                topic_id,
                model_type,
                topic['topic_number'],
                json.dumps(topic['top_words']),
                json.dumps(topic['word_weights']),
                0,  # Will be updated when document assignments are stored
                topic.get('coherence_score'),
                timestamp
            ))

        self.db_conn.commit()
        self.logger.debug(f"Stored {len(topics)} {model_type} topics to database")

    def _store_document_topics(self, document_topics: list, model_type: str) -> None:
        """Store document-topic assignments to database."""
        from datetime import datetime
        import hashlib

        cursor = self.db_conn.cursor()
        timestamp = datetime.now().isoformat()

        # Clear existing assignments for this model type
        cursor.execute("DELETE FROM document_topics WHERE model_type = ?", (model_type,))

        # Store new assignments
        for doc_topic in document_topics:
            doc_id = doc_topic['doc_id']

            # Store each topic assignment
            for topic_num, probability in doc_topic['topic_assignments']:
                # Generate assignment ID
                assignment_id = hashlib.md5(
                    f"{doc_id}_{model_type}_{topic_num}".encode()
                ).hexdigest()[:16]

                # Get topic_id
                topic_id = f"{model_type}_topic_{topic_num}"

                cursor.execute("""
                    INSERT INTO document_topics
                    (assignment_id, doc_id, topic_id, probability, model_type, assigned_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    assignment_id,
                    doc_id,
                    topic_id,
                    probability,
                    model_type,
                    timestamp
                ))

        self.db_conn.commit()

        # Update document counts in topics table
        cursor.execute("""
            UPDATE topics
            SET num_documents = (
                SELECT COUNT(DISTINCT doc_id)
                FROM document_topics
                WHERE document_topics.topic_id = topics.topic_id
            )
            WHERE model_type = ?
        """, (model_type,))

        self.db_conn.commit()
        self.logger.debug(f"Stored {len(document_topics)} document-topic assignments to database")

    def compare_topic_models(self) -> dict:
        """
        Compare all topic models (LDA, NMF, BERTopic) stored in the database.

        Provides a comprehensive comparison of different topic modeling approaches
        to help select the best model for the corpus.

        Returns:
            {
                'models': {
                    'lda': model_stats,
                    'nmf': model_stats,
                    'bertopic': model_stats
                },
                'comparison': {
                    'fastest': str,
                    'most_topics': str,
                    'best_coverage': str
                }
            }

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.train_lda_model(num_topics=5)
            >>> kb.train_nmf_model(num_topics=5)
            >>> kb.train_bertopic_model(num_topics=5)
            >>> comparison = kb.compare_topic_models()
            >>> print(f"Fastest model: {comparison['comparison']['fastest']}")
        """
        cursor = self.db_conn.cursor()

        models = {}
        model_types = ['lda', 'nmf', 'bertopic']

        for model_type in model_types:
            # Get topics for this model
            topics = cursor.execute("""
                SELECT COUNT(*), AVG(num_documents)
                FROM topics
                WHERE model_type = ?
            """, (model_type,)).fetchone()

            if topics and topics[0] > 0:
                num_topics = topics[0]
                avg_docs_per_topic = topics[1] or 0

                # Get document coverage
                doc_count = cursor.execute("""
                    SELECT COUNT(DISTINCT doc_id)
                    FROM document_topics
                    WHERE model_type = ?
                """, (model_type,)).fetchone()[0]

                # Get top topics by document count
                top_topics = cursor.execute("""
                    SELECT topic_number, num_documents, top_words
                    FROM topics
                    WHERE model_type = ?
                    ORDER BY num_documents DESC
                    LIMIT 5
                """, (model_type,)).fetchall()

                # Calculate topic diversity (unique words across all topics)
                all_topics_data = cursor.execute("""
                    SELECT top_words
                    FROM topics
                    WHERE model_type = ?
                """, (model_type,)).fetchall()

                unique_words = set()
                for (top_words_json,) in all_topics_data:
                    import json
                    words = json.loads(top_words_json)
                    unique_words.update(words)

                models[model_type] = {
                    'num_topics': num_topics,
                    'documents_covered': doc_count,
                    'avg_docs_per_topic': round(avg_docs_per_topic, 1),
                    'vocabulary_diversity': len(unique_words),
                    'top_topics': [
                        {
                            'topic_number': t[0],
                            'num_documents': t[1],
                            'top_words': json.loads(t[2])[:5]  # First 5 words
                        }
                        for t in top_topics
                    ]
                }
            else:
                models[model_type] = None

        # Calculate comparisons
        comparison = {}

        # Find model with most topics
        valid_models = {k: v for k, v in models.items() if v is not None}
        if valid_models:
            most_topics = max(valid_models.items(),
                            key=lambda x: x[1]['num_topics'])
            comparison['most_topics'] = {
                'model': most_topics[0],
                'count': most_topics[1]['num_topics']
            }

            # Find model with best coverage
            best_coverage = max(valid_models.items(),
                              key=lambda x: x[1]['documents_covered'])
            comparison['best_coverage'] = {
                'model': best_coverage[0],
                'documents': best_coverage[1]['documents_covered']
            }

            # Find model with highest vocabulary diversity
            most_diverse = max(valid_models.items(),
                             key=lambda x: x[1]['vocabulary_diversity'])
            comparison['most_diverse_vocabulary'] = {
                'model': most_diverse[0],
                'unique_words': most_diverse[1]['vocabulary_diversity']
            }

            # Summary recommendation
            if len(valid_models) >= 2:
                # Simple heuristic: prioritize coverage and diversity
                scores = {}
                for model, stats in valid_models.items():
                    score = (stats['documents_covered'] * 0.4 +
                            stats['vocabulary_diversity'] * 0.3 +
                            stats['num_topics'] * 0.3)
                    scores[model] = score

                recommended = max(scores.items(), key=lambda x: x[1])
                comparison['recommended'] = {
                    'model': recommended[0],
                    'score': round(recommended[1], 2),
                    'reason': f"Best balance of coverage, diversity, and topic count"
                }

        return {
            'models': models,
            'comparison': comparison,
            'total_models': len(valid_models)
        }

    def _store_clusters_to_db(self, clusters: list, algorithm: str) -> None:
        """
        Store clusters to database.

        Args:
            clusters: List of cluster dicts with cluster info
            algorithm: Clustering algorithm name (kmeans, dbscan, hdbscan)
        """
        import json
        import numpy as np

        cursor = self.db_conn.cursor()

        # Clear existing clusters for this algorithm
        cursor.execute("DELETE FROM clusters WHERE algorithm = ?", (algorithm,))

        for cluster in clusters:
            # Serialize the centroid as a raw float32 buffer. This MUST match
            # how visualize_cluster_dendrogram reads it back
            # (np.frombuffer(blob, dtype=np.float32)) - it was written with
            # pickle.dumps, and np.frombuffer cannot decode a pickle stream:
            # it raises "buffer size must be a multiple of element size", or
            # silently yields garbage when the pickle length happens to be a
            # multiple of 4. Raw bytes also keep the DB free of pickle.
            centroid_blob = None
            if cluster.get('centroid') is not None:
                centroid_blob = np.asarray(
                    cluster['centroid'], dtype=np.float32
                ).tobytes()

            cursor.execute("""
                INSERT INTO clusters
                (cluster_id, algorithm, cluster_number, centroid_vector,
                 num_documents, representative_docs, top_terms, silhouette_score, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster['cluster_id'],
                algorithm,
                cluster['cluster_number'],
                centroid_blob,
                cluster.get('num_documents', 0),
                json.dumps(cluster.get('representative_docs', [])),
                json.dumps(cluster.get('top_terms', [])),
                cluster.get('silhouette_score'),
                datetime.now().isoformat()
            ))

        self.db_conn.commit()
        self.logger.debug(f"Stored {len(clusters)} clusters to database")

    def _store_document_clusters(self, document_clusters: list, algorithm: str) -> None:
        """
        Store document-cluster assignments to database.

        Args:
            document_clusters: List of {doc_id, cluster_assignments} dicts
            algorithm: Clustering algorithm name
        """
        cursor = self.db_conn.cursor()

        # Clear existing assignments for this algorithm
        cursor.execute("DELETE FROM document_clusters WHERE algorithm = ?", (algorithm,))

        for doc_cluster in document_clusters:
            doc_id = doc_cluster['doc_id']
            cluster_num = doc_cluster['cluster_number']
            distance = doc_cluster.get('distance')

            assignment_id = hashlib.md5(
                f"{doc_id}_{algorithm}_{cluster_num}".encode()
            ).hexdigest()[:16]
            cluster_id = f"{algorithm}_cluster_{cluster_num}"

            cursor.execute("""
                INSERT INTO document_clusters
                (assignment_id, doc_id, cluster_id, distance, algorithm, assigned_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (assignment_id, doc_id, cluster_id, distance,
                  algorithm, datetime.now().isoformat()))

        self.db_conn.commit()

        # Update document counts in clusters table
        cursor.execute("""
            UPDATE clusters
            SET num_documents = (
                SELECT COUNT(DISTINCT doc_id)
                FROM document_clusters
                WHERE document_clusters.cluster_id = clusters.cluster_id
            )
            WHERE algorithm = ?
        """, (algorithm,))
        self.db_conn.commit()
        self.logger.debug(f"Stored {len(document_clusters)} document-cluster assignments to database")

    def _extract_top_terms_from_texts(self, texts: list, top_n: int = 10) -> list:
        """
        Extract top N terms from a list of texts using TF-IDF.

        Args:
            texts: List of text strings
            top_n: Number of top terms to extract

        Returns:
            List of top terms
        """
        if not texts:
            return []

        from sklearn.feature_extraction.text import TfidfVectorizer

        try:
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=1
            )
            vectorizer.fit(texts)
            return list(vectorizer.get_feature_names_out())
        except Exception as e:
            self.logger.warning(f"Could not extract top terms: {e}")
            return []

    def evaluate_clustering(self, algorithm: str) -> dict:
        """
        Evaluate clustering quality with multiple metrics.

        Calculates Silhouette score, Davies-Bouldin index, and
        Calinski-Harabasz score for the specified clustering algorithm.

        Args:
            algorithm: Clustering algorithm name (kmeans, dbscan, hdbscan)

        Returns:
            {
                'algorithm': str,
                'silhouette_score': float (higher is better, -1 to 1),
                'davies_bouldin_score': float (lower is better, ≥0),
                'calinski_harabasz_score': float (higher is better, ≥0),
                'num_clusters': int,
                'num_documents': int
            }

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.cluster_documents_kmeans(num_clusters=5)
            >>> metrics = kb.evaluate_clustering('kmeans')
            >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        import numpy as np

        self.logger.info(f"Evaluating {algorithm} clustering quality...")

        # Get document cluster assignments
        cursor = self.db_conn.cursor()
        assignments = cursor.execute("""
            SELECT doc_id, cluster_id
            FROM document_clusters
            WHERE algorithm = ?
            ORDER BY doc_id
        """, (algorithm,)).fetchall()

        if not assignments:
            raise ValueError(f"No clustering results found for algorithm: {algorithm}")

        # Get documents and generate embeddings
        doc_ids = [doc_id for doc_id, _ in assignments]
        documents = []

        for doc_id in doc_ids:
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                full_text = " ".join(chunk.content for chunk in chunks)
                documents.append(full_text)

        # Generate embeddings
        if not self._embeddings_loaded:
            self._ensure_embeddings_loaded()

        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Get cluster labels
        cluster_ids = [cluster_id for _, cluster_id in assignments]
        unique_cluster_ids = list(set(cluster_ids))
        cluster_to_label = {cid: i for i, cid in enumerate(unique_cluster_ids)}
        labels = np.array([cluster_to_label[cid] for cid in cluster_ids])

        # Calculate metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, labels)

        self.logger.info(f"Silhouette score: {silhouette:.3f}")
        self.logger.info(f"Davies-Bouldin index: {davies_bouldin:.3f}")
        self.logger.info(f"Calinski-Harabasz score: {calinski_harabasz:.2f}")

        return {
            'algorithm': algorithm,
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski_harabasz),
            'num_clusters': len(unique_cluster_ids),
            'num_documents': len(doc_ids)
        }

    def visualize_topics_wordcloud(self, model_type: str = 'lda',
                                   output_dir: str = "topic_wordclouds") -> List[str]:
        """
        Generate word cloud visualizations for each topic.

        Creates PNG images showing word importance in each topic using
        word clouds. Words that are more important in the topic appear
        larger in the visualization.

        Args:
            model_type: Topic model type (lda, nmf, bertopic)
            output_dir: Directory to save word cloud images

        Returns:
            List of output file paths

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.train_lda_model(num_topics=5)
            >>> files = kb.visualize_topics_wordcloud('lda')
            >>> print(f"Created {len(files)} word clouds")
        """
        import json
        import os
        from pathlib import Path

        # Import visualization libraries
        try:
            from wordcloud import WordCloud
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("wordcloud and matplotlib required for visualizations. Install with: pip install wordcloud matplotlib")

        self.logger.info(f"Generating word clouds for {model_type} topics...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get topics from database
        cursor = self.db_conn.cursor()
        topics = cursor.execute("""
            SELECT topic_number, word_weights, top_words
            FROM topics
            WHERE model_type = ?
            ORDER BY topic_number
        """, (model_type,)).fetchall()

        if not topics:
            raise ValueError(f"No topics found for model type: {model_type}")

        output_files = []

        for topic_num, word_weights_json, top_words_json in topics:
            # Parse word weights
            word_weights = json.loads(word_weights_json)

            # Create word cloud
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(word_weights)

            # Create figure
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"{model_type.upper()} Topic {topic_num}", fontsize=16, fontweight='bold')

            # Save image
            output_file = output_path / f"topic_{model_type}_{topic_num}.png"
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            plt.close()

            output_files.append(str(output_file))
            self.logger.info(f"Created word cloud: {output_file}")

        self.logger.info(f"Generated {len(output_files)} word cloud visualizations")
        return output_files

    def visualize_topic_distribution(self, model_type: str = 'lda',
                                     output_path: str = "topic_distribution.html") -> str:
        """
        Create interactive bar chart showing topic distribution across corpus.

        Shows how many documents are primarily associated with each topic,
        with interactive hover information.

        Args:
            model_type: Topic model type (lda, nmf, bertopic)
            output_path: Output HTML file path

        Returns:
            Path to output HTML file

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.train_lda_model(num_topics=5)
            >>> file = kb.visualize_topic_distribution('lda')
            >>> print(f"Chart saved to {file}")
        """
        import json
        from pathlib import Path

        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required for visualizations. Install with: pip install plotly")

        self.logger.info(f"Creating topic distribution chart for {model_type}...")

        # Get topics with document counts
        cursor = self.db_conn.cursor()
        topics = cursor.execute("""
            SELECT topic_number, num_documents, top_words
            FROM topics
            WHERE model_type = ?
            ORDER BY topic_number
        """, (model_type,)).fetchall()

        if not topics:
            raise ValueError(f"No topics found for model type: {model_type}")

        # Parse data
        topic_numbers = []
        doc_counts = []
        top_words_list = []

        for topic_num, num_docs, top_words_json in topics:
            topic_numbers.append(f"Topic {topic_num}")
            doc_counts.append(num_docs)
            top_words = json.loads(top_words_json)
            top_words_list.append(", ".join(top_words[:5]))

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=topic_numbers,
                y=doc_counts,
                text=doc_counts,
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Documents: %{y}<br>Top words: %{customdata}<extra></extra>',
                customdata=top_words_list,
                marker=dict(
                    color=doc_counts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Documents")
                )
            )
        ])

        fig.update_layout(
            title=f"{model_type.upper()} Topic Distribution Across Corpus",
            xaxis_title="Topic",
            yaxis_title="Number of Documents",
            height=500,
            template="plotly_white",
            hovermode='closest'
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Topic distribution chart saved to {output_file}")
        return str(output_file)

    def visualize_clusters_2d(self, algorithm: str = 'kmeans',
                              output_path: str = "clusters_2d.html") -> str:
        """
        Visualize document clusters in 2D using UMAP dimensionality reduction.

        Creates interactive scatter plot showing how documents cluster together
        in semantic space. Uses UMAP to reduce high-dimensional embeddings to 2D.

        Args:
            algorithm: Clustering algorithm (kmeans, dbscan, hdbscan)
            output_path: Output HTML file path

        Returns:
            Path to output HTML file

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.cluster_documents_kmeans(num_clusters=5)
            >>> file = kb.visualize_clusters_2d('kmeans')
            >>> print(f"Cluster visualization saved to {file}")
        """
        import numpy as np
        from pathlib import Path

        try:
            import plotly.express as px
            import pandas as pd
        except ImportError:
            raise ImportError("plotly and pandas required. Install with: pip install plotly pandas")

        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("umap-learn required. Install with: pip install umap-learn")

        self.logger.info(f"Creating 2D cluster visualization for {algorithm}...")

        # Get cluster assignments
        cursor = self.db_conn.cursor()
        assignments = cursor.execute("""
            SELECT dc.doc_id, dc.cluster_id, c.cluster_number
            FROM document_clusters dc
            JOIN clusters c ON dc.cluster_id = c.cluster_id
            WHERE dc.algorithm = ?
            ORDER BY dc.doc_id
        """, (algorithm,)).fetchall()

        if not assignments:
            raise ValueError(f"No clustering results found for algorithm: {algorithm}")

        # Get documents and generate embeddings
        doc_ids = [doc_id for doc_id, _, _ in assignments]
        cluster_labels = [cluster_num for _, _, cluster_num in assignments]
        documents = []
        titles = []

        for doc_id in doc_ids:
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                full_text = " ".join(chunk.content for chunk in chunks)
                documents.append(full_text)
                titles.append(self.documents[doc_id].title if doc_id in self.documents else doc_id[:12])

        # Generate embeddings
        if not self._embeddings_loaded:
            self._ensure_embeddings_loaded()

        self.logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Reduce to 2D with UMAP
        self.logger.info("Reducing embeddings to 2D with UMAP...")
        reducer = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        embedding_2d = reducer.fit_transform(embeddings)

        # Create DataFrame
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': [f"Cluster {label}" for label in cluster_labels],
            'cluster_num': cluster_labels,
            'doc_id': doc_ids,
            'title': titles
        })

        # Create scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['title', 'doc_id'],
            title=f"Document Clusters in 2D ({algorithm.upper()})",
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(
            height=600,
            template="plotly_white",
            hovermode='closest',
            legend=dict(title="Cluster")
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"2D cluster visualization saved to {output_file}")
        return str(output_file)

    def visualize_cluster_dendrogram(self, algorithm: str = 'kmeans',
                                     output_path: str = "cluster_dendrogram.html") -> str:
        """
        Create dendrogram visualization showing hierarchical cluster relationships.

        Builds a hierarchical tree showing how clusters relate to each other
        based on their centroid distances. Useful for understanding cluster
        similarity and structure.

        Args:
            algorithm: Clustering algorithm (kmeans, dbscan, hdbscan)
            output_path: Output HTML file path

        Returns:
            Path to output HTML file

        Examples:
            >>> kb = KnowledgeBase()
            >>> kb.cluster_documents_kmeans(num_clusters=5)
            >>> file = kb.visualize_cluster_dendrogram('kmeans')
            >>> print(f"Dendrogram saved to {file}")
        """
        import numpy as np
        from pathlib import Path

        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
        except ImportError:
            raise ImportError("scipy required. Install with: pip install scipy")

        try:
            import plotly.figure_factory as ff
        except ImportError:
            raise ImportError("plotly required. Install with: pip install plotly")

        self.logger.info(f"Creating dendrogram for {algorithm} clusters...")

        # Get cluster centroids
        cursor = self.db_conn.cursor()
        clusters = cursor.execute("""
            SELECT cluster_number, centroid_vector
            FROM clusters
            WHERE algorithm = ?
            ORDER BY cluster_number
        """, (algorithm,)).fetchall()

        if not clusters:
            raise ValueError(f"No clustering results found for algorithm: {algorithm}")

        # Parse centroids
        cluster_numbers = []
        centroids = []

        for cluster_num, centroid_blob in clusters:
            cluster_numbers.append(cluster_num)
            centroid = np.frombuffer(centroid_blob, dtype=np.float32)
            centroids.append(centroid)

        centroids = np.array(centroids, dtype=np.float64)  # Convert to float64
        self.logger.info(f"Loaded {len(centroids)} cluster centroids with shape {centroids.shape}")

        # Check for and handle non-finite values
        if not np.all(np.isfinite(centroids)):
            self.logger.warning("Found non-finite values in centroids, replacing with zeros")
            centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize centroids to prevent numerical issues
        from sklearn.preprocessing import normalize
        centroids = normalize(centroids)

        # Perform hierarchical clustering on centroids
        self.logger.info("Computing hierarchical clustering...")
        linkage_matrix = linkage(centroids, method='ward')

        # Create labels
        labels = [f"Cluster {num}" for num in cluster_numbers]

        # Create dendrogram using plotly
        fig = ff.create_dendrogram(
            centroids,
            labels=labels,
            linkagefun=lambda x: linkage(x, method='ward')
        )

        fig.update_layout(
            title=f"Cluster Dendrogram ({algorithm.upper()})",
            xaxis_title="Cluster",
            yaxis_title="Distance",
            height=500,
            template="plotly_white"
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Dendrogram saved to {output_file}")
        return str(output_file)

    def visualize_topic_flow_sankey(self, time_period: str = 'decade',
                                   output_path: str = "topic_flow.html") -> str:
        """
        Create Sankey diagram showing flow of topics/entities over time.

        Args:
            time_period: Time grouping ('year' or 'decade')
            output_path: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        import plotly.graph_objects as go
        from pathlib import Path

        cursor = self.db_conn.cursor()

        # Get events with entities
        events_data = cursor.execute("""
            SELECT e.year, e.event_type, e.entities
            FROM events e
            WHERE e.year IS NOT NULL AND e.entities IS NOT NULL
            ORDER BY e.year
        """).fetchall()

        if not events_data:
            self.logger.warning("No events with entities found for Sankey diagram")
            return ""

        # Process data into time periods
        import json

        time_entity_map = {}
        entity_type_map = {}

        for year, event_type, entities_json in events_data:
            # Determine time period
            if time_period == 'decade':
                period = f"{(year // 10) * 10}s"
            else:
                period = str(year)

            # Parse entities
            try:
                entities = json.loads(entities_json)
                for entity in entities:
                    if period not in time_entity_map:
                        time_entity_map[period] = {}

                    if entity not in time_entity_map[period]:
                        time_entity_map[period][entity] = 0
                    time_entity_map[period][entity] += 1

                    # Track entity types
                    if entity not in entity_type_map:
                        entity_type_map[entity] = event_type
            except:
                continue

        if len(time_entity_map) < 2:
            self.logger.warning("Need at least 2 time periods for Sankey diagram")
            return ""

        # Build Sankey data
        nodes = []
        node_map = {}
        node_colors = []

        # Color scheme for time periods
        period_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#9B59B6', '#F7DC6F']

        # Create nodes for each time period and their entities
        color_idx = 0
        for period in sorted(time_entity_map.keys()):
            # Get top entities for this period
            top_entities = sorted(time_entity_map[period].items(),
                                key=lambda x: x[1], reverse=True)[:10]

            base_color = period_colors[color_idx % len(period_colors)]

            for entity, count in top_entities:
                node_label = f"{entity} ({period})"
                node_map[node_label] = len(nodes)
                nodes.append(node_label)
                node_colors.append(base_color)

            color_idx += 1

        # Create links between consecutive periods
        links_source = []
        links_target = []
        links_value = []
        links_color = []

        sorted_periods = sorted(time_entity_map.keys())
        for i in range(len(sorted_periods) - 1):
            period1 = sorted_periods[i]
            period2 = sorted_periods[i + 1]

            # Find entities that appear in both periods
            entities1 = set(time_entity_map[period1].keys())
            entities2 = set(time_entity_map[period2].keys())
            common_entities = entities1.intersection(entities2)

            for entity in common_entities:
                node1 = f"{entity} ({period1})"
                node2 = f"{entity} ({period2})"

                if node1 in node_map and node2 in node_map:
                    # Flow value is minimum of the two counts
                    value = min(time_entity_map[period1][entity],
                              time_entity_map[period2][entity])

                    links_source.append(node_map[node1])
                    links_target.append(node_map[node2])
                    links_value.append(value)
                    links_color.append('rgba(100, 100, 100, 0.3)')

        if not links_source:
            self.logger.warning("No entity flows found between periods")
            return ""

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='white', width=0.5),
                label=nodes,
                color=node_colors
            ),
            link=dict(
                source=links_source,
                target=links_target,
                value=links_value,
                color=links_color
            )
        )])

        fig.update_layout(
            title=dict(text=f'Topic Flow Over Time ({time_period.capitalize()})',
                      font=dict(size=16)),
            font=dict(size=10),
            height=800
        )

        # Save to HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))

        self.logger.info(f"Sankey diagram saved to {output_file} ({len(nodes)} nodes, {len(links_source)} flows)")
        return str(output_file)

    def _prepare_topic_model_corpus(self, min_df: int = 2,
                                   max_df: float = 0.8,
                                   max_features: int = 1000) -> Tuple[List[str], Any, Any]:
        """
        Prepare document corpus for topic modeling.

        Creates TF-IDF matrix from all documents in the knowledge base.

        Args:
            min_df: Minimum document frequency (default: 2)
            max_df: Maximum document frequency as fraction (default: 0.8)
            max_features: Maximum number of features (default: 1000)

        Returns:
            Tuple of (doc_ids, vectorizer, tfidf_matrix)

        Example:
            >>> doc_ids, vectorizer, matrix = kb._prepare_topic_model_corpus()
            >>> print(f"Prepared {len(doc_ids)} documents")
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.logger.info(f"Preparing corpus for topic modeling ({len(self.documents)} documents)")

        # Get all document texts
        docs = []
        doc_ids = []

        for doc_id, doc in self.documents.items():
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                full_text = " ".join(chunk.content for chunk in chunks)
                docs.append(full_text)
                doc_ids.append(doc_id)

        if not docs:
            self.logger.warning("No documents found for topic modeling")
            return [], None, None

        self.logger.info(f"Creating TF-IDF vectorizer (min_df={min_df}, max_df={max_df}, max_features={max_features})")

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            stop_words='english',
            max_features=max_features,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            self.logger.info(f"TF-IDF matrix: {tfidf_matrix.shape[0]} documents × {tfidf_matrix.shape[1]} features")
            return doc_ids, vectorizer, tfidf_matrix
        except Exception as e:
            self.logger.error(f"Failed to create TF-IDF matrix: {e}")
            raise

    def train_lda_model(self, num_topics: int = 10,
                       max_iter: int = 100,
                       random_state: int = 42,
                       min_doc_prob: float = 0.05) -> Dict[str, Any]:
        """
        Train Latent Dirichlet Allocation (LDA) topic model.

        LDA discovers latent topics in document collection by modeling
        documents as mixtures of topics and topics as mixtures of words.

        Args:
            num_topics: Number of topics to discover (default: 10)
            max_iter: Maximum iterations (default: 100)
            random_state: Random seed for reproducibility (default: 42)
            min_doc_prob: Minimum topic probability to assign document (default: 0.05)

        Returns:
            Dict with model statistics:
                - model_type: 'lda'
                - num_topics: Number of topics
                - topics: List of topic dicts with words and weights
                - perplexity: Model perplexity (lower = better)
                - num_documents: Number of documents processed

        Example:
            >>> results = kb.train_lda_model(num_topics=5)
            >>> for topic in results['topics']:
            >>>     print(f"Topic {topic['topic_number']}: {', '.join(topic['words'][:5])}")

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            self.logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            raise ImportError("scikit-learn required for LDA. Install with: pip install scikit-learn")

        import hashlib
        from datetime import datetime

        self.logger.info(f"Training LDA model with {num_topics} topics")

        # Prepare corpus
        doc_ids, vectorizer, tfidf_matrix = self._prepare_topic_model_corpus()

        if not doc_ids:
            return {'error': 'No documents available for topic modeling'}

        # Train LDA
        self.logger.info("Fitting LDA model...")
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=max_iter,
            learning_method='online',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

        doc_topic_dist = lda.fit_transform(tfidf_matrix)

        self.logger.info(f"LDA training complete")

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic_weights in enumerate(lda.components_):
            # Get top 10 words for this topic
            top_indices = topic_weights.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            word_weights = {feature_names[i]: float(topic_weights[i])
                           for i in top_indices}

            # Store topic to database
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

            self.logger.debug(f"Topic {topic_idx}: {', '.join(top_words[:5])}")

        # Assign documents to topics
        assignments = 0
        for doc_idx, doc_id in enumerate(doc_ids):
            topic_probs = doc_topic_dist[doc_idx]

            # Get top 3 topics for this document
            top_topic_indices = topic_probs.argsort()[-3:][::-1]

            for topic_idx in top_topic_indices:
                prob = topic_probs[topic_idx]
                if prob > min_doc_prob:
                    self._assign_document_to_topic(
                        doc_id=doc_id,
                        topic_id=topics[topic_idx]['topic_id'],
                        probability=float(prob),
                        model_type='lda'
                    )
                    assignments += 1

        self.logger.info(f"Created {assignments} document-topic assignments")

        # Calculate perplexity (lower is better)
        perplexity = lda.perplexity(tfidf_matrix)

        self.logger.info(f"LDA model perplexity: {perplexity:.2f}")

        return {
            'model_type': 'lda',
            'num_topics': num_topics,
            'topics': topics,
            'perplexity': perplexity,
            'num_documents': len(doc_ids),
            'num_assignments': assignments
        }

    def _store_topic(self, model_type: str, topic_number: int,
                    top_words: List[str], word_weights: Dict[str, float],
                    coherence_score: Optional[float] = None) -> str:
        """
        Store topic to database.

        Args:
            model_type: Model type ('lda', 'nmf', 'bertopic')
            topic_number: Topic index number
            top_words: List of top words for topic
            word_weights: Dict mapping words to weights
            coherence_score: Optional coherence score

        Returns:
            topic_id: Unique topic identifier
        """
        import hashlib
        import json
        from datetime import datetime

        # Generate topic ID
        topic_str = f"{model_type}_{topic_number}_{datetime.now().isoformat()}"
        topic_id = hashlib.sha256(topic_str.encode()).hexdigest()[:16]

        cursor = self.db_conn.cursor()

        # Store topic
        cursor.execute("""
            INSERT OR REPLACE INTO topics
            (topic_id, model_type, topic_number, top_words, word_weights,
             coherence_score, created_date, num_documents)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            topic_id,
            model_type,
            topic_number,
            json.dumps(top_words),
            json.dumps(word_weights),
            coherence_score,
            datetime.now().isoformat()
        ))

        self.db_conn.commit()

        return topic_id

    def _assign_document_to_topic(self, doc_id: str, topic_id: str,
                                  probability: float, model_type: str):
        """
        Assign document to topic with probability.

        Args:
            doc_id: Document ID
            topic_id: Topic ID
            probability: Topic probability for document (0.0-1.0)
            model_type: Model type ('lda', 'nmf', 'bertopic')
        """
        import hashlib
        from datetime import datetime

        # Generate assignment ID
        assignment_str = f"{doc_id}_{topic_id}_{datetime.now().isoformat()}"
        assignment_id = hashlib.sha256(assignment_str.encode()).hexdigest()[:16]

        cursor = self.db_conn.cursor()

        # Store assignment
        cursor.execute("""
            INSERT OR REPLACE INTO document_topics
            (assignment_id, doc_id, topic_id, probability, model_type, assigned_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            assignment_id,
            doc_id,
            topic_id,
            probability,
            model_type,
            datetime.now().isoformat()
        ))

        # Update topic document count
        cursor.execute("""
            UPDATE topics
            SET num_documents = (
                SELECT COUNT(DISTINCT doc_id)
                FROM document_topics
                WHERE topic_id = ?
            )
            WHERE topic_id = ?
        """, (topic_id, topic_id))

        self.db_conn.commit()

    def train_nmf_model(self, num_topics: int = 10,
                       max_iter: int = 200,
                       random_state: int = 42,
                       min_doc_prob: float = 0.05) -> Dict[str, Any]:
        """
        Train Non-negative Matrix Factorization (NMF) topic model.

        NMF produces topics by factorizing the document-term matrix into
        two non-negative matrices. Often produces more coherent and
        interpretable topics than LDA.

        Args:
            num_topics: Number of topics to discover (default: 10)
            max_iter: Maximum iterations (default: 200)
            random_state: Random seed for reproducibility (default: 42)
            min_doc_prob: Minimum topic weight to assign document (default: 0.05)

        Returns:
            Dict with model statistics:
                - model_type: 'nmf'
                - num_topics: Number of topics
                - topics: List of topic dicts with words and weights
                - reconstruction_error: Model reconstruction error (lower = better)
                - num_documents: Number of documents processed

        Example:
            >>> results = kb.train_nmf_model(num_topics=5)
            >>> for topic in results['topics']:
            >>>     print(f"Topic {topic['topic_number']}: {', '.join(topic['words'][:5])}")

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            from sklearn.decomposition import NMF
        except ImportError:
            self.logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            raise ImportError("scikit-learn required for NMF. Install with: pip install scikit-learn")

        import hashlib
        from datetime import datetime

        self.logger.info(f"Training NMF model with {num_topics} topics")

        # Prepare corpus
        doc_ids, vectorizer, tfidf_matrix = self._prepare_topic_model_corpus()

        if not doc_ids:
            return {'error': 'No documents available for topic modeling'}

        # Train NMF
        self.logger.info("Fitting NMF model...")
        nmf = NMF(
            n_components=num_topics,
            max_iter=max_iter,
            random_state=random_state,
            init='nndsvd',   # Non-negative Double SVD initialization
            verbose=0
        )

        # W = document-topic matrix, H = topic-word matrix
        doc_topic_matrix = nmf.fit_transform(tfidf_matrix)

        self.logger.info(f"NMF training complete")

        # Extract topics from H matrix (topic-word matrix)
        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic_weights in enumerate(nmf.components_):
            # Get top 10 words for this topic
            top_indices = topic_weights.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            word_weights = {feature_names[i]: float(topic_weights[i])
                           for i in top_indices}

            # Store topic to database
            topic_id = self._store_topic(
                model_type='nmf',
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

            self.logger.debug(f"Topic {topic_idx}: {', '.join(top_words[:5])}")

        # Assign documents to topics
        assignments = 0
        for doc_idx, doc_id in enumerate(doc_ids):
            topic_weights = doc_topic_matrix[doc_idx]

            # Normalize to get probabilities
            topic_probs = topic_weights / topic_weights.sum() if topic_weights.sum() > 0 else topic_weights

            # Get top 3 topics for this document
            top_topic_indices = topic_probs.argsort()[-3:][::-1]

            for topic_idx in top_topic_indices:
                prob = topic_probs[topic_idx]
                if prob > min_doc_prob:
                    self._assign_document_to_topic(
                        doc_id=doc_id,
                        topic_id=topics[topic_idx]['topic_id'],
                        probability=float(prob),
                        model_type='nmf'
                    )
                    assignments += 1

        self.logger.info(f"Created {assignments} document-topic assignments")

        # Calculate reconstruction error (lower is better)
        reconstruction_error = nmf.reconstruction_err_

        self.logger.info(f"NMF model reconstruction error: {reconstruction_error:.2f}")

        return {
            'model_type': 'nmf',
            'num_topics': num_topics,
            'topics': topics,
            'reconstruction_error': reconstruction_error,
            'num_documents': len(doc_ids),
            'num_assignments': assignments
        }

    def train_bertopic_model(self, num_topics: int = 10,
                            min_cluster_size: int = 5,
                            random_state: int = 42) -> Dict[str, Any]:
        """
        Train BERTopic model using document embeddings.

        BERTopic is a state-of-the-art topic modeling technique that uses
        embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.
        Produces highly coherent topics.

        Args:
            num_topics: Target number of topics (default: 10)
            min_cluster_size: Minimum documents per topic (default: 5)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Dict with model statistics:
                - model_type: 'bertopic'
                - num_topics: Number of topics discovered
                - topics: List of topic dicts with words and scores
                - num_documents: Number of documents processed

        Example:
            >>> results = kb.train_bertopic_model(num_topics=5)
            >>> for topic in results['topics']:
            >>>     print(f"Topic {topic['topic_number']}: {', '.join(topic['words'][:5])}")

        Raises:
            ImportError: If bertopic, umap-learn, or hdbscan not installed
        """
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
        except ImportError as e:
            missing = str(e).split("'")[1]
            self.logger.error(f"{missing} not installed. Run: pip install bertopic umap-learn hdbscan")
            raise ImportError(f"BERTopic dependencies required. Install with: pip install bertopic umap-learn hdbscan")

        import numpy as np
        from datetime import datetime

        self.logger.info(f"Training BERTopic model (target topics: {num_topics})")

        # Ensure embeddings model is loaded
        if self.embeddings_model is None:
            self._ensure_embeddings_loaded()

        # Get documents and generate/retrieve embeddings
        doc_ids = []
        documents = []

        for doc_id, doc in self.documents.items():
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                doc_ids.append(doc_id)
                # Use first chunk as document representation
                documents.append(chunks[0].content[:500])  # Limit length for performance

        if not doc_ids:
            self.logger.warning("No documents found")
            return {'error': 'No documents available for topic modeling'}

        # Generate embeddings for all documents
        self.logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)

        embeddings = np.array(embeddings)
        self.logger.info(f"Loaded {len(doc_ids)} documents with embeddings (shape: {embeddings.shape})")

        # Configure UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=random_state
        )

        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Create BERTopic model
        self.logger.info("Fitting BERTopic model...")
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=num_topics,
            verbose=False,
            calculate_probabilities=True
        )

        # Train model
        topic_assignments, probs = topic_model.fit_transform(documents, embeddings)

        # Get topic info
        topic_info = topic_model.get_topic_info()
        num_discovered_topics = len(topic_info) - 1  # -1 for outlier topic (-1)

        self.logger.info(f"BERTopic discovered {num_discovered_topics} topics")

        # Extract and store topics
        topics = []
        topic_id_map = {}  # Map BERTopic topic numbers to our topic_ids

        for idx, row in topic_info.iterrows():
            topic_num = row['Topic']
            if topic_num == -1:  # Skip outlier topic
                continue

            # Get top words for this topic
            topic_words = topic_model.get_topic(topic_num)
            if not topic_words:
                continue

            top_words = [word for word, _ in topic_words[:10]]
            word_weights = {word: float(score) for word, score in topic_words[:10]}

            # Store topic to database
            topic_id = self._store_topic(
                model_type='bertopic',
                topic_number=int(topic_num),
                top_words=top_words,
                word_weights=word_weights
            )

            topic_id_map[topic_num] = topic_id

            topics.append({
                'topic_id': topic_id,
                'topic_number': int(topic_num),
                'words': top_words,
                'weights': word_weights,
                'document_count': int(row['Count'])
            })

            self.logger.debug(f"Topic {topic_num}: {', '.join(top_words[:5])}")

        # Assign documents to topics
        assignments = 0
        for doc_idx, (doc_id, topic_num, prob) in enumerate(zip(doc_ids, topic_assignments, probs)):
            if topic_num == -1:  # Skip outlier documents
                continue

            if topic_num in topic_id_map:
                # For BERTopic, prob might be an array, take max
                if isinstance(prob, (list, np.ndarray)):
                    probability = float(np.max(prob))
                else:
                    probability = float(prob)

                self._assign_document_to_topic(
                    doc_id=doc_id,
                    topic_id=topic_id_map[topic_num],
                    probability=probability,
                    model_type='bertopic'
                )
                assignments += 1

        self.logger.info(f"Created {assignments} document-topic assignments")

        # Count outliers (topic -1)
        outlier_count = int(np.sum(np.array(topic_assignments) == -1))

        return {
            'model_type': 'bertopic',
            'num_topics': num_discovered_topics,
            'topics': topics,
            'num_documents': len(doc_ids),
            'num_assignments': assignments,
            'outliers': outlier_count
        }

    def _store_cluster(self, algorithm: str, cluster_number: int,
                      centroid_vector: "Optional[np.ndarray]" = None,
                      silhouette_score: Optional[float] = None) -> str:
        """
        Store cluster to database.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
            cluster_number: Cluster index number
            centroid_vector: Optional cluster centroid (for kmeans)
            silhouette_score: Optional silhouette score

        Returns:
            cluster_id: Unique cluster identifier
        """
        import hashlib
        import json
        from datetime import datetime

        # Generate cluster ID
        cluster_str = f"{algorithm}_{cluster_number}_{datetime.now().isoformat()}"
        cluster_id = hashlib.sha256(cluster_str.encode()).hexdigest()[:16]

        cursor = self.db_conn.cursor()

        # Convert centroid to JSON if provided
        centroid_json = json.dumps(centroid_vector.tolist()) if centroid_vector is not None else None

        # Store cluster
        cursor.execute("""
            INSERT OR REPLACE INTO clusters
            (cluster_id, algorithm, cluster_number, centroid_vector,
             silhouette_score, created_date, num_documents)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            cluster_id,
            algorithm,
            cluster_number,
            centroid_json,
            silhouette_score,
            datetime.now().isoformat()
        ))

        self.db_conn.commit()

        return cluster_id

    def _assign_document_to_cluster(self, doc_id: str, cluster_id: str,
                                    distance: float, algorithm: str):
        """
        Assign document to cluster with distance metric.

        Args:
            doc_id: Document ID
            cluster_id: Cluster ID
            distance: Distance from cluster centroid
            algorithm: Clustering algorithm
        """
        import hashlib
        from datetime import datetime

        cursor = self.db_conn.cursor()

        # Generate assignment ID
        assignment_str = f"{doc_id}_{cluster_id}_{datetime.now().isoformat()}"
        assignment_id = hashlib.sha256(assignment_str.encode()).hexdigest()[:16]

        cursor.execute("""
            INSERT OR REPLACE INTO document_clusters
            (assignment_id, doc_id, cluster_id, distance, algorithm, assigned_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (assignment_id, doc_id, cluster_id, distance, algorithm, datetime.now().isoformat()))

        self.db_conn.commit()

    def cluster_documents_kmeans(self, num_clusters: int = 10,
                                 random_state: int = 42) -> Dict[str, Any]:
        """
        Cluster documents using K-Means on embeddings.

        K-Means is a centroid-based clustering algorithm that partitions
        documents into K clusters based on embedding similarity.

        Args:
            num_clusters: Number of clusters (K) (default: 10)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Dict with clustering statistics:
                - algorithm: 'kmeans'
                - num_clusters: Number of clusters
                - silhouette_score: Clustering quality metric (-1 to 1)
                - num_documents: Number of documents clustered

        Example:
            >>> results = kb.cluster_documents_kmeans(num_clusters=5)
            >>> print(f"Clustered {results['num_documents']} docs into 5 clusters")
            >>> print(f"Silhouette score: {results['silhouette_score']:.3f}")

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            self.logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            raise ImportError("scikit-learn required for K-Means. Install with: pip install scikit-learn")

        import numpy as np

        self.logger.info(f"K-Means clustering with {num_clusters} clusters")

        # Ensure embeddings model is loaded
        if self.embeddings_model is None:
            self._ensure_embeddings_loaded()

        # Get documents and generate embeddings
        doc_ids = []
        documents = []

        for doc_id, doc in self.documents.items():
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                doc_ids.append(doc_id)
                documents.append(chunks[0].content[:500])

        if not doc_ids:
            return {'error': 'No documents available for clustering'}

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Train K-Means
        self.logger.info("Fitting K-Means model...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Calculate silhouette score
        silhouette = silhouette_score(embeddings, labels)
        self.logger.info(f"K-Means silhouette score: {silhouette:.3f}")

        # Store clusters and assign documents
        assignments = 0
        for cluster_num in range(num_clusters):
            # Find documents in this cluster
            cluster_doc_indices = [i for i, label in enumerate(labels) if label == cluster_num]

            # Store cluster
            cluster_id = self._store_cluster(
                algorithm='kmeans',
                cluster_number=cluster_num,
                centroid_vector=kmeans.cluster_centers_[cluster_num],
                silhouette_score=silhouette
            )

            # Assign documents to cluster
            for doc_idx in cluster_doc_indices:
                doc_id = doc_ids[doc_idx]
                distance = np.linalg.norm(embeddings[doc_idx] - kmeans.cluster_centers_[cluster_num])
                self._assign_document_to_cluster(doc_id, cluster_id, float(distance), 'kmeans')
                assignments += 1

        self.logger.info(f"Created {assignments} document-cluster assignments")

        return {
            'algorithm': 'kmeans',
            'num_clusters': num_clusters,
            'silhouette_score': float(silhouette),
            'num_documents': len(doc_ids),
            'num_assignments': assignments
        }

    def cluster_documents_dbscan(self, eps: float = 0.5,
                                 min_samples: int = 5) -> Dict[str, Any]:
        """
        Cluster documents using DBSCAN on embeddings.

        DBSCAN is a density-based clustering algorithm that finds
        arbitrary-shaped clusters and identifies outliers. Does not
        require specifying the number of clusters.

        Args:
            eps: Maximum distance between samples (default: 0.5)
            min_samples: Minimum samples in neighborhood (default: 5)

        Returns:
            Dict with clustering statistics:
                - algorithm: 'dbscan'
                - num_clusters: Number of clusters found
                - silhouette_score: Clustering quality metric
                - num_documents: Number of documents clustered
                - num_outliers: Number of outlier documents

        Example:
            >>> results = kb.cluster_documents_dbscan(eps=0.5, min_samples=5)
            >>> print(f"Found {results['num_clusters']} clusters")
            >>> print(f"Outliers: {results['num_outliers']}")

        Raises:
            ImportError: If scikit-learn is not installed
        """
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import silhouette_score
        except ImportError:
            self.logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            raise ImportError("scikit-learn required for DBSCAN. Install with: pip install scikit-learn")

        import numpy as np

        self.logger.info(f"DBSCAN clustering (eps={eps}, min_samples={min_samples})")

        # Ensure embeddings model is loaded
        if self.embeddings_model is None:
            self._ensure_embeddings_loaded()

        # Get documents and generate embeddings
        doc_ids = []
        documents = []

        for doc_id, doc in self.documents.items():
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                doc_ids.append(doc_id)
                documents.append(chunks[0].content[:500])

        if not doc_ids:
            return {'error': 'No documents available for clustering'}

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Train DBSCAN
        self.logger.info("Fitting DBSCAN model...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        # Count clusters and outliers
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_outliers = int((labels == -1).sum())

        self.logger.info(f"DBSCAN found {num_clusters} clusters and {num_outliers} outliers")

        # Calculate silhouette score (excluding outliers)
        if num_clusters > 1 and num_outliers < len(labels):
            # Filter out outliers for silhouette calculation
            non_outlier_mask = labels != -1
            if non_outlier_mask.sum() > 1:
                silhouette = silhouette_score(
                    embeddings[non_outlier_mask],
                    labels[non_outlier_mask]
                )
            else:
                silhouette = 0.0
        else:
            silhouette = 0.0

        self.logger.info(f"DBSCAN silhouette score: {silhouette:.3f}")

        # Store clusters and assign documents
        assignments = 0
        cluster_centroids = {}

        for cluster_num in unique_labels:
            if cluster_num == -1:
                continue  # Skip outliers

            # Find documents in this cluster
            cluster_doc_indices = [i for i, label in enumerate(labels) if label == cluster_num]

            # Calculate cluster centroid
            cluster_embeddings = embeddings[cluster_doc_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[cluster_num] = centroid

            # Store cluster
            cluster_id = self._store_cluster(
                algorithm='dbscan',
                cluster_number=cluster_num,
                centroid_vector=centroid,
                silhouette_score=silhouette
            )

            # Assign documents to cluster
            for doc_idx in cluster_doc_indices:
                doc_id = doc_ids[doc_idx]
                distance = np.linalg.norm(embeddings[doc_idx] - centroid)
                self._assign_document_to_cluster(doc_id, cluster_id, float(distance), 'dbscan')
                assignments += 1

        self.logger.info(f"Created {assignments} document-cluster assignments")

        return {
            'algorithm': 'dbscan',
            'num_clusters': num_clusters,
            'silhouette_score': float(silhouette),
            'num_documents': len(doc_ids),
            'num_assignments': assignments,
            'num_outliers': num_outliers
        }

    def cluster_documents_hdbscan(self, min_cluster_size: int = 5,
                                  min_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Cluster documents using HDBSCAN on embeddings.

        HDBSCAN (Hierarchical DBSCAN) is an advanced density-based
        clustering algorithm that automatically selects clusters and
        handles varying densities better than DBSCAN.

        Args:
            min_cluster_size: Minimum samples per cluster (default: 5)
            min_samples: Minimum samples in neighborhood (default: None = min_cluster_size)

        Returns:
            Dict with clustering statistics:
                - algorithm: 'hdbscan'
                - num_clusters: Number of clusters found
                - num_documents: Number of documents clustered
                - num_outliers: Number of outlier documents

        Example:
            >>> results = kb.cluster_documents_hdbscan(min_cluster_size=5)
            >>> print(f"Found {results['num_clusters']} clusters")
            >>> print(f"Outliers: {results['num_outliers']}")

        Raises:
            ImportError: If hdbscan is not installed
        """
        try:
            import hdbscan
        except ImportError:
            self.logger.error("hdbscan not installed. Run: pip install hdbscan")
            raise ImportError("hdbscan required for HDBSCAN. Install with: pip install hdbscan")

        import numpy as np

        self.logger.info(f"HDBSCAN clustering (min_cluster_size={min_cluster_size})")

        # Ensure embeddings model is loaded
        if self.embeddings_model is None:
            self._ensure_embeddings_loaded()

        # Get documents and generate embeddings
        doc_ids = []
        documents = []

        for doc_id, doc in self.documents.items():
            chunks = self._get_chunks_db(doc_id)
            if chunks:
                doc_ids.append(doc_id)
                documents.append(chunks[0].content[:500])

        if not doc_ids:
            return {'error': 'No documents available for clustering'}

        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embeddings_model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # Train HDBSCAN
        self.logger.info("Fitting HDBSCAN model...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        labels = clusterer.fit_predict(embeddings)

        # Count clusters and outliers
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_outliers = int((labels == -1).sum())

        self.logger.info(f"HDBSCAN found {num_clusters} clusters and {num_outliers} outliers")

        # Store clusters and assign documents
        assignments = 0
        cluster_centroids = {}

        for cluster_num in unique_labels:
            if cluster_num == -1:
                continue  # Skip outliers

            # Find documents in this cluster
            cluster_doc_indices = [i for i, label in enumerate(labels) if label == cluster_num]

            # Calculate cluster centroid
            cluster_embeddings = embeddings[cluster_doc_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[cluster_num] = centroid

            # Store cluster
            cluster_id = self._store_cluster(
                algorithm='hdbscan',
                cluster_number=cluster_num,
                centroid_vector=centroid,
                silhouette_score=None  # HDBSCAN uses different metrics
            )

            # Assign documents to cluster
            for doc_idx in cluster_doc_indices:
                doc_id = doc_ids[doc_idx]
                distance = np.linalg.norm(embeddings[doc_idx] - centroid)
                self._assign_document_to_cluster(doc_id, cluster_id, float(distance), 'hdbscan')
                assignments += 1

        self.logger.info(f"Created {assignments} document-cluster assignments")

        return {
            'algorithm': 'hdbscan',
            'num_clusters': num_clusters,
            'num_documents': len(doc_ids),
            'num_assignments': assignments,
            'num_outliers': num_outliers,
            'cluster_persistence': clusterer.cluster_persistence_.tolist() if hasattr(clusterer, 'cluster_persistence_') else []
        }

    def generate_topic_wordcloud(self, topic_id: str, output_path: str,
                                 width: int = 800, height: int = 400,
                                 background_color: str = 'white') -> Dict[str, Any]:
        """
        Generate a word cloud visualization for a topic.

        Args:
            topic_id: Topic ID to visualize
            output_path: Path to save the word cloud image
            width: Image width in pixels
            height: Image height in pixels
            background_color: Background color for word cloud

        Returns:
            {
                'topic_id': str,
                'topic_number': int,
                'model_type': str,
                'output_path': str,
                'num_words': int
            }

        Examples:
            # Generate word cloud for a topic
            result = kb.generate_topic_wordcloud(
                'topic-123',
                'topic_wordcloud.png',
                width=1200,
                height=600
            )
        """
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        cursor = self.db_conn.cursor()

        # Get topic details
        topic_row = cursor.execute(
            "SELECT topic_number, model_type, top_words, word_weights FROM topics WHERE topic_id = ?",
            (topic_id,)
        ).fetchone()

        if not topic_row:
            return {'error': f'Topic {topic_id} not found'}

        topic_number, model_type, top_words_json, weights_json = topic_row

        # Parse word weights
        import json
        weights = json.loads(weights_json)

        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap='viridis',
            relative_scaling=0.5
        ).generate_from_frequencies(weights)

        # Save to file
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_number} ({model_type.upper()})', fontsize=16)
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return {
            'topic_id': topic_id,
            'topic_number': topic_number,
            'model_type': model_type,
            'output_path': output_path,
            'num_words': len(weights)
        }

    def visualize_cluster_scatter(self, algorithm: str, output_path: str,
                                  width: int = 1200, height: int = 800,
                                  n_neighbors: int = 15,
                                  min_dist: float = 0.1) -> Dict[str, Any]:
        """
        Generate 2D scatter plot of document clusters using UMAP projection.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
            output_path: Path to save the scatter plot
            width: Image width in pixels
            height: Image height in pixels
            n_neighbors: UMAP n_neighbors parameter (controls local vs global structure)
            min_dist: UMAP min_dist parameter (controls cluster tightness)

        Returns:
            {
                'algorithm': str,
                'num_clusters': int,
                'num_documents': int,
                'output_path': str
            }

        Examples:
            # Visualize DBSCAN clusters
            result = kb.visualize_cluster_scatter('dbscan', 'clusters.png')
        """
        import umap
        import matplotlib.pyplot as plt
        import numpy as np

        cursor = self.db_conn.cursor()

        # Get clusters for this algorithm
        clusters = cursor.execute(
            "SELECT cluster_id, cluster_number FROM clusters WHERE algorithm = ?",
            (algorithm,)
        ).fetchall()

        if not clusters:
            return {'error': f'No clusters found for algorithm {algorithm}'}

        cluster_map = {cid: cnum for cid, cnum in clusters}

        # Get document embeddings and cluster assignments
        doc_data = []
        for doc_id, doc in self.documents.items():
            # Get cluster assignment
            assignment = cursor.execute(
                """SELECT c.cluster_number FROM document_clusters dc
                   JOIN clusters c ON dc.cluster_id = c.cluster_id
                   WHERE dc.doc_id = ? AND c.algorithm = ?""",
                (doc_id, algorithm)
            ).fetchone()

            if assignment:
                cluster_num = assignment[0]
                doc_data.append((doc_id, doc.title, cluster_num))

        if not doc_data:
            return {'error': f'No document assignments found for algorithm {algorithm}'}

        # Extract embeddings for assigned documents
        doc_ids = [d[0] for d in doc_data]
        titles = [d[1] for d in doc_data]
        cluster_labels = np.array([d[2] for d in doc_data])

        # Generate embeddings
        documents = [self.documents[doc_id] for doc_id in doc_ids]
        doc_texts = []
        for doc in documents:
            chunks = self._get_chunks_db(doc.doc_id)[:3]
            chunk_texts = [chunk.content for chunk in chunks]
            doc_text = doc.title + " " + " ".join(chunk_texts)
            doc_texts.append(doc_text)

        # Ensure embeddings model is loaded
        self._ensure_embeddings_loaded()
        embeddings = self.embeddings_model.encode(doc_texts, show_progress_bar=False)

        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            metric='cosine'
        )
        embedding_2d = reducer.fit_transform(embeddings)

        # Create scatter plot
        plt.figure(figsize=(width/100, height/100), dpi=100)

        # Get unique clusters (including outliers as -1)
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_num in enumerate(unique_clusters):
            mask = cluster_labels == cluster_num
            label = f'Outliers' if cluster_num == -1 else f'Cluster {cluster_num}'

            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.6,
                s=50
            )

        plt.title(f'Document Clusters ({algorithm.upper()})', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return {
            'algorithm': algorithm,
            'num_clusters': len(unique_clusters),
            'num_documents': len(doc_data),
            'output_path': output_path
        }

    def generate_topic_heatmap(self, model_type: str, output_path: str,
                               max_topics: int = 20,
                               max_documents: int = 50) -> Dict[str, Any]:
        """
        Generate heatmap showing document-topic probability matrix.

        Args:
            model_type: Topic model type ('lda', 'nmf', 'bertopic')
            output_path: Path to save the heatmap
            max_topics: Maximum number of topics to include
            max_documents: Maximum number of documents to include

        Returns:
            {
                'model_type': str,
                'num_topics': int,
                'num_documents': int,
                'output_path': str
            }

        Examples:
            # Generate heatmap for LDA model
            result = kb.generate_topic_heatmap('lda', 'topic_heatmap.png')
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        cursor = self.db_conn.cursor()

        # Get topics for this model
        topics = cursor.execute(
            "SELECT topic_id, topic_number FROM topics WHERE model_type = ? ORDER BY topic_number LIMIT ?",
            (model_type, max_topics)
        ).fetchall()

        if not topics:
            return {'error': f'No topics found for model type {model_type}'}

        topic_ids = [t[0] for t in topics]
        topic_numbers = [t[1] for t in topics]

        # Get document-topic assignments
        # Get top documents by average probability across topics
        doc_assignments = cursor.execute(
            f"""SELECT dt.doc_id, d.title, AVG(dt.probability) as avg_prob
               FROM document_topics dt
               JOIN documents d ON dt.doc_id = d.doc_id
               JOIN topics t ON dt.topic_id = t.topic_id
               WHERE t.model_type = ?
               GROUP BY dt.doc_id
               ORDER BY avg_prob DESC
               LIMIT ?""",
            (model_type, max_documents)
        ).fetchall()

        if not doc_assignments:
            return {'error': f'No document-topic assignments found for {model_type}'}

        doc_ids = [d[0] for d in doc_assignments]
        doc_titles = [d[1][:40] for d in doc_assignments]  # Truncate titles

        # Build probability matrix
        matrix = np.zeros((len(doc_ids), len(topic_ids)))

        for i, doc_id in enumerate(doc_ids):
            for j, topic_id in enumerate(topic_ids):
                prob = cursor.execute(
                    "SELECT probability FROM document_topics WHERE doc_id = ? AND topic_id = ?",
                    (doc_id, topic_id)
                ).fetchone()
                if prob:
                    matrix[i, j] = prob[0]

        # Create heatmap
        plt.figure(figsize=(12, max(8, len(doc_ids) * 0.3)))

        plt.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Topic Probability')

        # Set ticks
        plt.xticks(range(len(topic_numbers)), [f'T{n}' for n in topic_numbers], rotation=0)
        plt.yticks(range(len(doc_titles)), doc_titles, fontsize=8)

        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Documents', fontsize=12)
        plt.title(f'Document-Topic Heatmap ({model_type.upper()})', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'model_type': model_type,
            'num_topics': len(topic_ids),
            'num_documents': len(doc_ids),
            'output_path': output_path
        }

    def visualize_cluster_distribution(self, algorithm: str, output_path: str,
                                       width: int = 1000, height: int = 600) -> Dict[str, Any]:
        """
        Generate bar chart showing cluster size distribution.

        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
            output_path: Path to save the bar chart
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            {
                'algorithm': str,
                'num_clusters': int,
                'output_path': str,
                'cluster_sizes': dict
            }

        Examples:
            # Visualize cluster distribution
            result = kb.visualize_cluster_distribution('kmeans', 'distribution.png')
        """
        import matplotlib.pyplot as plt
        import numpy as np

        cursor = self.db_conn.cursor()

        # Get cluster sizes
        cluster_sizes = cursor.execute(
            """SELECT c.cluster_number, COUNT(*) as size
               FROM document_clusters dc
               JOIN clusters c ON dc.cluster_id = c.cluster_id
               WHERE c.algorithm = ?
               GROUP BY c.cluster_number
               ORDER BY c.cluster_number""",
            (algorithm,)
        ).fetchall()

        if not cluster_sizes:
            return {'error': f'No clusters found for algorithm {algorithm}'}

        cluster_numbers = [c[0] for c in cluster_sizes]
        sizes = [c[1] for c in cluster_sizes]

        # Create bar chart
        plt.figure(figsize=(width/100, height/100), dpi=100)

        colors = ['red' if cn == -1 else 'steelblue' for cn in cluster_numbers]
        labels = ['Outliers' if cn == -1 else f'Cluster {cn}' for cn in cluster_numbers]

        bars = plt.bar(range(len(cluster_numbers)), sizes, color=colors, alpha=0.7)

        plt.xticks(range(len(cluster_numbers)), labels, rotation=45 if len(cluster_numbers) > 10 else 0)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.xlabel('Cluster', fontsize=12)
        plt.title(f'Cluster Size Distribution ({algorithm.upper()})', fontsize=14)

        # Add value labels on bars
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(size)}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Build cluster sizes dict
        sizes_dict = {f'cluster_{cn}' if cn != -1 else 'outliers': size
                      for cn, size in cluster_sizes}

        return {
            'algorithm': algorithm,
            'num_clusters': len(cluster_numbers),
            'output_path': output_path,
            'cluster_sizes': sizes_dict
        }
