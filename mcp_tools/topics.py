"""Topic modeling and document clustering handlers (LDA/NMF/BERTopic,
k-means/DBSCAN/HDBSCAN), plus their Phase-2 visualization tools
(wordcloud/scatter/heatmap/distribution). Split out of handlers.py
(R12 follow-up).

Bodies are the original branch bodies, dedented one level and otherwise
unchanged - this was a move, not a rewrite.
"""


from mcp.types import TextContent


def handle_train_lda_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    max_iter = arguments.get("max_iter", 100)
    max_features = arguments.get("max_features", 1000)

    try:
        results = kb.train_lda_model(
            num_topics=num_topics,
            max_iter=max_iter,
            max_features=max_features,
            store_results=True
        )

        output = f"# LDA Topic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Vocabulary size: {results['vocabulary_size']}\n"
        output += f"- Perplexity: {results['perplexity']:.2f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training LDA model: {str(e)}")]


def handle_train_nmf_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    max_iter = arguments.get("max_iter", 200)
    max_features = arguments.get("max_features", 1000)

    try:
        results = kb.train_nmf_model(
            num_topics=num_topics,
            max_iter=max_iter,
            max_features=max_features,
            store_results=True
        )

        output = f"# NMF Topic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Vocabulary size: {results['vocabulary_size']}\n"
        output += f"- Reconstruction error: {results['reconstruction_error']:.2f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training NMF model: {str(e)}")]


def handle_train_bertopic(kb, name: str, arguments: dict) -> list[TextContent]:
    num_topics = arguments.get("num_topics", 10)
    min_topic_size = arguments.get("min_topic_size", 5)

    try:
        results = kb.train_bertopic_model(
            num_topics=num_topics,
            min_topic_size=min_topic_size,
            store_results=True
        )

        output = f"# BERTopic Model Trained\n\n"
        output += f"**Model Statistics:**\n"
        output += f"- Number of topics: {results['num_topics']}\n"
        output += f"- Documents processed: {results['num_documents']}\n"
        output += f"- Outliers: {results['outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Top Topics:**\n\n"
        for i, topic in enumerate(results['topics'][:5], 1):
            output += f"{i}. **Topic {topic['topic_number']}:** {', '.join(topic['top_words'][:5])}\n"

        output += f"\nResults stored to database. Use `get_document_topics` to see document assignments.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error training BERTopic model: {str(e)}")]


def handle_get_document_topics(kb, name: str, arguments: dict) -> list[TextContent]:
    model_type = arguments.get("model_type")
    doc_id = arguments.get("doc_id")
    min_probability = arguments.get("min_probability", 0.1)

    try:
        cursor = kb.db_conn.cursor()

        if doc_id:
            # Get topics for specific document
            assignments = cursor.execute("""
                SELECT dt.topic_id, t.topic_number, t.top_words, dt.probability
                FROM document_topics dt
                JOIN topics t ON dt.topic_id = t.topic_id
                WHERE dt.doc_id = ? AND dt.model_type = ? AND dt.probability >= ?
                ORDER BY dt.probability DESC
            """, (doc_id, model_type, min_probability)).fetchall()

            if not assignments:
                return [TextContent(type="text", text=f"No topic assignments found for document {doc_id} with model {model_type}")]

            doc = kb.documents.get(doc_id)
            doc_title = doc.title if doc else "Unknown"

            output = f"# Topic Assignments for: {doc_title}\n\n"
            output += f"**Model:** {model_type}\n\n"

            for topic_id, topic_num, top_words_json, probability in assignments:
                import json
                top_words = json.loads(top_words_json)
                output += f"- **Topic {topic_num}** (prob: {probability:.3f}): {', '.join(top_words[:5])}\n"

        else:
            # Get summary of all documents
            stats = cursor.execute("""
                SELECT COUNT(DISTINCT doc_id), COUNT(*)
                FROM document_topics
                WHERE model_type = ?
            """, (model_type,)).fetchone()

            if not stats or stats[0] == 0:
                return [TextContent(type="text", text=f"No topic assignments found for model {model_type}")]

            num_docs, num_assignments = stats

            output = f"# Topic Assignments Summary\n\n"
            output += f"**Model:** {model_type}\n"
            output += f"**Documents:** {num_docs}\n"
            output += f"**Total assignments:** {num_assignments}\n\n"

            # Get topic distribution
            topic_dist = cursor.execute("""
                SELECT t.topic_number, t.top_words, COUNT(DISTINCT dt.doc_id) as doc_count
                FROM topics t
                LEFT JOIN document_topics dt ON t.topic_id = dt.topic_id
                WHERE t.model_type = ?
                GROUP BY t.topic_number
                ORDER BY doc_count DESC
            """, (model_type,)).fetchall()

            output += f"**Topic Distribution:**\n\n"
            for topic_num, top_words_json, doc_count in topic_dist:
                import json
                top_words = json.loads(top_words_json)
                output += f"- **Topic {topic_num}** ({doc_count} docs): {', '.join(top_words[:5])}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting document topics: {str(e)}")]


def handle_cluster_documents_kmeans(kb, name: str, arguments: dict) -> list[TextContent]:
    num_clusters = arguments.get("num_clusters", 10)

    try:
        results = kb.cluster_documents_kmeans(
            num_clusters=num_clusters,
            store_results=True
        )

        output = f"# K-Means Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Silhouette score: {results['silhouette_score']:.3f}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        output += f"**Cluster Sizes:**\n\n"
        for i, cluster in enumerate(results['clusters'][:10], 1):
            output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_cluster_documents_dbscan(kb, name: str, arguments: dict) -> list[TextContent]:
    eps = arguments.get("eps", 0.5)
    min_samples = arguments.get("min_samples", 3)

    try:
        results = kb.cluster_documents_dbscan(
            eps=eps,
            min_samples=min_samples,
            store_results=True
        )

        output = f"# DBSCAN Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Outliers: {results['num_outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        if results['clusters']:
            output += f"**Cluster Sizes:**\n\n"
            for i, cluster in enumerate(results['clusters'][:10], 1):
                output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_cluster_documents_hdbscan(kb, name: str, arguments: dict) -> list[TextContent]:
    min_cluster_size = arguments.get("min_cluster_size", 5)
    min_samples = arguments.get("min_samples", 3)

    try:
        results = kb.cluster_documents_hdbscan(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            store_results=True
        )

        output = f"# HDBSCAN Clustering Complete\n\n"
        output += f"**Clustering Statistics:**\n"
        output += f"- Number of clusters: {results['num_clusters']}\n"
        output += f"- Documents clustered: {results['num_documents']}\n"
        output += f"- Outliers: {results['num_outliers']}\n"
        output += f"- Training time: {results['training_time']:.2f}s\n\n"

        if results['clusters']:
            output += f"**Cluster Sizes:**\n\n"
            for i, cluster in enumerate(results['clusters'][:10], 1):
                output += f"{i}. **Cluster {cluster['cluster_number']}** ({cluster['num_documents']} docs): {', '.join(cluster['top_terms'][:5])}\n"

        output += f"\nResults stored to database. Use `get_cluster_documents` to see cluster contents.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error clustering documents: {str(e)}")]


def handle_get_cluster_documents(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    cluster_number = arguments.get("cluster_number")

    try:
        cursor = kb.db_conn.cursor()

        if cluster_number is not None:
            # Get documents in specific cluster
            cluster_id = f"{algorithm}_cluster_{cluster_number}"

            # Get cluster info
            cluster_info = cursor.execute("""
                SELECT num_documents, top_terms, silhouette_score
                FROM clusters
                WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()

            if not cluster_info:
                return [TextContent(type="text", text=f"Cluster {cluster_number} not found for algorithm {algorithm}")]

            num_docs, top_terms_json, silhouette = cluster_info
            import json
            top_terms = json.loads(top_terms_json)

            # Get documents
            docs = cursor.execute("""
                SELECT dc.doc_id, dc.distance
                FROM document_clusters dc
                WHERE dc.cluster_id = ?
                ORDER BY dc.distance ASC
                LIMIT 20
            """, (cluster_id,)).fetchall()

            output = f"# Cluster {cluster_number} ({algorithm})\n\n"
            output += f"**Cluster Info:**\n"
            output += f"- Documents: {num_docs}\n"
            output += f"- Top terms: {', '.join(top_terms[:10])}\n"
            if silhouette:
                output += f"- Silhouette score: {silhouette:.3f}\n"
            output += f"\n**Documents (closest to centroid):**\n\n"

            for doc_id, distance in docs:
                doc = kb.documents.get(doc_id)
                if doc:
                    output += f"- {doc.title[:60]} (distance: {distance:.3f})\n"

        else:
            # Get summary of all clusters
            clusters = cursor.execute("""
                SELECT cluster_number, num_documents, top_terms
                FROM clusters
                WHERE algorithm = ?
                ORDER BY num_documents DESC
            """, (algorithm,)).fetchall()

            if not clusters:
                return [TextContent(type="text", text=f"No clusters found for algorithm {algorithm}")]

            output = f"# Clusters ({algorithm})\n\n"
            output += f"**Total clusters:** {len(clusters)}\n\n"

            for cluster_num, num_docs, top_terms_json in clusters[:20]:
                import json
                top_terms = json.loads(top_terms_json)
                output += f"- **Cluster {cluster_num}** ({num_docs} docs): {', '.join(top_terms[:5])}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting cluster documents: {str(e)}")]


# Phase 3: Temporal Analysis Tools


def handle_generate_topic_wordcloud(kb, name: str, arguments: dict) -> list[TextContent]:
    topic_id = arguments.get("topic_id")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 800)
    height = arguments.get("height", 400)
    background_color = arguments.get("background_color", "white")

    try:
        result = kb.generate_topic_wordcloud(
            topic_id=topic_id,
            output_path=output_path,
            width=width,
            height=height,
            background_color=background_color
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Topic Word Cloud Generated\n{'='*60}\n\n"
        output += f"Topic: {result['topic_number']} ({result['model_type'].upper()})\n"
        output += f"Output File: {result['output_path']}\n"
        output += f"Image Size: {width}x{height}px\n"
        output += f"Words Visualized: {result['num_words']}\n\n"
        output += "The word cloud has been saved. Larger words indicate higher importance in the topic.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating word cloud: {str(e)}")]


def handle_visualize_cluster_scatter(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 1200)
    height = arguments.get("height", 800)
    n_neighbors = arguments.get("n_neighbors", 15)
    min_dist = arguments.get("min_dist", 0.1)

    try:
        result = kb.visualize_cluster_scatter(
            algorithm=algorithm,
            output_path=output_path,
            width=width,
            height=height,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Cluster Scatter Plot Generated\n{'='*60}\n\n"
        output += f"Algorithm: {result['algorithm'].upper()}\n"
        output += f"Clusters Visualized: {result['num_clusters']}\n"
        output += f"Documents Plotted: {result['num_documents']}\n"
        output += f"Output File: {result['output_path']}\n"
        output += f"Image Size: {width}x{height}px\n\n"
        output += "The scatter plot shows document clusters in 2D space using UMAP projection.\n"
        output += "Documents in the same cluster appear closer together in the visualization.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating scatter plot: {str(e)}")]


def handle_generate_topic_heatmap(kb, name: str, arguments: dict) -> list[TextContent]:
    model_type = arguments.get("model_type")
    output_path = arguments.get("output_path")
    max_topics = arguments.get("max_topics", 20)
    max_documents = arguments.get("max_documents", 50)

    try:
        result = kb.generate_topic_heatmap(
            model_type=model_type,
            output_path=output_path,
            max_topics=max_topics,
            max_documents=max_documents
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Topic Heatmap Generated\n{'='*60}\n\n"
        output += f"Model Type: {result['model_type'].upper()}\n"
        output += f"Topics: {result['num_topics']}\n"
        output += f"Documents: {result['num_documents']}\n"
        output += f"Output File: {result['output_path']}\n\n"
        output += "The heatmap shows topic-document probability matrix.\n"
        output += "Brighter colors indicate higher topic probability for a document.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating heatmap: {str(e)}")]


def handle_visualize_cluster_distribution(kb, name: str, arguments: dict) -> list[TextContent]:
    algorithm = arguments.get("algorithm")
    output_path = arguments.get("output_path")
    width = arguments.get("width", 1000)
    height = arguments.get("height", 600)

    try:
        result = kb.visualize_cluster_distribution(
            algorithm=algorithm,
            output_path=output_path,
            width=width,
            height=height
        )

        if 'error' in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        output = f"Cluster Distribution Chart Generated\n{'='*60}\n\n"
        output += f"Algorithm: {result['algorithm'].upper()}\n"
        output += f"Clusters: {result['num_clusters']}\n"
        output += f"Output File: {result['output_path']}\n\n"
        output += "Cluster Sizes:\n"

        # Show cluster sizes
        for cluster_name, size in sorted(result['cluster_sizes'].items()):
            if cluster_name == 'outliers':
                output += f"  Outliers: {size} documents\n"
            else:
                cluster_num = cluster_name.replace('cluster_', '')
                output += f"  Cluster {cluster_num}: {size} documents\n"

        output += "\nThe bar chart shows the distribution of documents across clusters.\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating distribution chart: {str(e)}")]


HANDLERS_TOPICS = {
    "train_lda_topics": handle_train_lda_topics,
    "train_nmf_topics": handle_train_nmf_topics,
    "train_bertopic": handle_train_bertopic,
    "get_document_topics": handle_get_document_topics,
    "cluster_documents_kmeans": handle_cluster_documents_kmeans,
    "cluster_documents_dbscan": handle_cluster_documents_dbscan,
    "cluster_documents_hdbscan": handle_cluster_documents_hdbscan,
    "get_cluster_documents": handle_get_cluster_documents,
    "generate_topic_wordcloud": handle_generate_topic_wordcloud,
    "visualize_cluster_scatter": handle_visualize_cluster_scatter,
    "generate_topic_heatmap": handle_generate_topic_heatmap,
    "visualize_cluster_distribution": handle_visualize_cluster_distribution,
}
