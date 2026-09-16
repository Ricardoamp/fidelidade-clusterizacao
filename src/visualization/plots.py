"""Plotting helpers.

UMAP is used here strictly to visualize clusters that were already computed on the raw
RFM features elsewhere (src/models/clustering.py) -- it must never be the space the
clustering algorithm itself operates on (see that module's docstring for why).
"""
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap


def plot_silhouette_scores(df_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in df_results.index:
        ax.plot(df_results.columns, df_results.loc[algo], linestyle="--", marker="o", label=algo)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score x k")
    ax.legend()
    return fig


def plot_cluster_embedding(X, labels, random_state=42):
    """2D UMAP projection of X, colored by pre-computed cluster labels. Display only."""
    reducer = umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="deep", ax=ax)
    ax.set_xlabel("embedding_x")
    ax.set_ylabel("embedding_y")
    ax.set_title("Cluster visualization (UMAP projection, for display only)")
    return fig
