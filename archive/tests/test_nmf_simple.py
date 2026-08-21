"""Simple NMF test with fixed parameters."""

import sys
sys.path.insert(0, '.')
from server import KnowledgeBase
import os
from sklearn.decomposition import NMF
import numpy as np

kb = KnowledgeBase(os.path.expanduser('~/.tdz-c64-knowledge'))

print("\n[RUN] Preparing corpus...")
doc_ids, vectorizer, tfidf_matrix = kb._prepare_topic_model_corpus()

print(f"  TF-IDF shape: {tfidf_matrix.shape}")

# Test 1: Minimal NMF with just defaults
print("\n[TEST 1] NMF with defaults...")
nmf1 = NMF(n_components=5, random_state=42)
W1 = nmf1.fit_transform(tfidf_matrix)
print(f"  W sum: {W1.sum():.2f}, H sum: {nmf1.components_.sum():.2f}")

# Test 2: NMF with nndsvd init
print("\n[TEST 2] NMF with nndsvd init...")
nmf2 = NMF(n_components=5, init='nndsvd', random_state=42)
W2 = nmf2.fit_transform(tfidf_matrix)
print(f"  W sum: {W2.sum():.2f}, H sum: {nmf2.components_.sum():.2f}")

# Test 3: Show top words from test 2
print("\n[TEST 3] Top words from successful NMF:")
feature_names = vectorizer.get_feature_names_out()
for topic_idx in range(5):
    topic_weights = nmf2.components_[topic_idx]
    top_indices = topic_weights.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_values = [topic_weights[i] for i in top_indices]
    print(f"\nTopic {topic_idx}:")
    for word, val in zip(top_words[:5], top_values[:5]):
        print(f"  {word}: {val:.4f}")
