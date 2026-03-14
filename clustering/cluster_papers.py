import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NOISE_THRESHOLD = 0.40  # If more than 40% noise, fall back to k-means


class PaperClusterer:
    """Cluster paper embeddings using HDBSCAN with k-means fallback."""

    def cluster(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 10,
        method: str = "hdbscan",
    ) -> np.ndarray:
        """
        Cluster embeddings and return cluster label array.

        Labels are integers >= 0 for valid clusters, -1 for noise.
        Falls back to k-means if HDBSCAN produces too many noise points.
        """
        if len(embeddings) == 0:
            return np.array([], dtype=np.int32)

        if len(embeddings) < min_cluster_size * 2:
            logger.warning(
                f"Too few papers ({len(embeddings)}) for min_cluster_size={min_cluster_size}. "
                "Adjusting min_cluster_size."
            )
            min_cluster_size = max(2, len(embeddings) // 4)

        if method == "hdbscan":
            labels = self._hdbscan_cluster(embeddings, min_cluster_size)
            noise_ratio = (labels == -1).sum() / len(labels)
            logger.info(
                f"HDBSCAN: {labels.max() + 1} clusters, "
                f"{(labels == -1).sum()} noise points ({noise_ratio:.1%})"
            )
            if noise_ratio > NOISE_THRESHOLD:
                logger.warning(
                    f"HDBSCAN noise ratio {noise_ratio:.1%} > {NOISE_THRESHOLD:.0%}. "
                    "Falling back to k-means."
                )
                n_clusters = max(2, len(embeddings) // min_cluster_size)
                labels = self._kmeans_cluster(embeddings, n_clusters)
        else:
            n_clusters = max(2, len(embeddings) // min_cluster_size)
            labels = self._kmeans_cluster(embeddings, n_clusters)

        return labels

    def _hdbscan_cluster(self, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """Run HDBSCAN clustering."""
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(1, min_cluster_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings.astype(np.float64))
        return labels.astype(np.int32)

    def _kmeans_cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Run k-means clustering."""
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(n_clusters, len(embeddings))
        logger.info(f"K-means clustering with k={n_clusters}")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=5,
            max_iter=300,
        )
        labels = kmeans.fit_predict(embeddings.astype(np.float32))
        return labels.astype(np.int32)

    def get_cluster_papers(
        self, labels: np.ndarray, papers: list[dict]
    ) -> dict[int, list[dict]]:
        """
        Group papers by cluster label.

        Noise points (label=-1) are excluded from the result.
        """
        assert len(labels) == len(papers), (
            f"Label length {len(labels)} != paper count {len(papers)}"
        )
        clusters: dict[int, list[dict]] = {}
        for label, paper in zip(labels, papers):
            label = int(label)
            if label == -1:
                continue  # Skip noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(paper)

        logger.info(f"Grouped papers into {len(clusters)} clusters (noise excluded)")
        return clusters
