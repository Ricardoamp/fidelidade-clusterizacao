"""Clustering pipeline: scaling, model-selection tuning, and final model training.

Clustering runs directly on the scaled RFM features (not on a UMAP/tree embedding) so
that cluster membership stays interpretable in terms of revenue, recency, frequency and
returns. Any embedding (UMAP/PCA) used elsewhere in this project is for visualization
only and must never feed back into these functions.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from sklearn import cluster, metrics, mixture, preprocessing

from src.features.build_features import RFM_COLUMNS

RANDOM_STATE = 42

# gross_revenue, qtde_products, frequency and qtde_returns are strongly right-skewed by a
# handful of high-volume customers (e.g. std(gross_revenue) > 4x its mean). Left un-transformed,
# GMM's Gaussian cluster assumption breaks down: empirically, Silhouette Score was negative
# across every k. log1p compresses that tail before scaling. recency_days is bounded and not
# heavy-tailed, so it is left as-is.
LOG_COLUMNS = ["gross_revenue", "qtde_products", "frequency", "qtde_returns"]


def scale_features(df, columns=RFM_COLUMNS, log_columns=LOG_COLUMNS):
    """Log1p the heavy-tailed RFM columns, then fit one MinMaxScaler over all columns.

    Returns (X_scaled, scaler). Only one scaler is fit, across all columns at once (rather
    than one throwaway MinMaxScaler per column), so it can be persisted and reapplied to
    new customers later.
    """
    df_t = df.copy()
    for col in log_columns:
        df_t[col] = np.log1p(df_t[col])

    scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(
        scaler.fit_transform(df_t[columns]),
        columns=columns,
        index=df.index,
    )
    return X, scaler


def tune_clustering(X, k_range, random_state=RANDOM_STATE):
    """Compare KMeans, GMM and Hierarchical Clustering by Silhouette Score across k_range.

    random_state is fixed for every k so the comparison (and the k it points to) is
    reproducible across runs.
    """
    results = {"KMeans": [], "GMM": [], "HC": []}

    for k in k_range:
        kmeans = cluster.KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        results["KMeans"].append(metrics.silhouette_score(X, labels))

        gmm = mixture.GaussianMixture(n_components=k, random_state=random_state)
        labels = gmm.fit_predict(X)
        results["GMM"].append(metrics.silhouette_score(X, labels))

        hc_model = hierarchy.linkage(X, "ward")
        labels = hierarchy.fcluster(hc_model, k, criterion="maxclust")
        results["HC"].append(metrics.silhouette_score(X, labels))

    df_results = pd.DataFrame(results).T
    df_results.columns = list(k_range)
    return df_results


def run_dbscan(X, eps, min_samples):
    """Fit DBSCAN and report its Silhouette Score.

    Unlike the legacy notebook (which pasted in hardcoded results from an offline run),
    this always recomputes live so the numbers can't drift out of sync with the data.
    Returns (labels, silhouette, n_clusters); silhouette is NaN when DBSCAN collapses
    everything into fewer than 2 clusters (silhouette_score is undefined there).
    """
    model = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = metrics.silhouette_score(X, labels) if n_clusters > 1 else float("nan")
    return labels, sil, n_clusters


def train_final_model(X, k, random_state=RANDOM_STATE, n_init=300):
    model = mixture.GaussianMixture(n_components=k, n_init=n_init, random_state=random_state)
    model.fit(X)
    return model


def save_artifacts(scaler, model, models_dir="../models"):
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    joblib.dump(model, f"{models_dir}/gmm_model.pkl")


def load_artifacts(models_dir="../models"):
    scaler = joblib.load(f"{models_dir}/scaler.pkl")
    model = joblib.load(f"{models_dir}/gmm_model.pkl")
    return scaler, model
