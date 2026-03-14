"""Unit tests for clustering components."""
import numpy as np
import pytest

from clustering.cluster_papers import PaperClusterer
from clustering.cluster_evaluator import ClusterEvaluator


def make_synthetic_embeddings(n_clusters: int = 3, n_per_cluster: int = 20, dim: int = 64, seed: int = 42) -> tuple:
    """Create synthetic embeddings with clear cluster structure."""
    rng = np.random.RandomState(seed)
    embeddings = []
    labels_true = []
    papers = []

    for cluster_id in range(n_clusters):
        # Each cluster centered at a different point
        center = rng.randn(dim) * 5
        cluster_vecs = center + rng.randn(n_per_cluster, dim) * 0.3
        # Normalize to unit sphere
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        cluster_vecs = cluster_vecs / norms

        embeddings.append(cluster_vecs)
        labels_true.extend([cluster_id] * n_per_cluster)

        for i in range(n_per_cluster):
            papers.append({
                "pmid": f"{cluster_id}_{i}",
                "title": f"Paper {cluster_id}-{i} about topic {cluster_id}",
                "abstract": (
                    f"This paper studies mechanism of cluster {cluster_id} pathway. "
                    f"We found that protein X{cluster_id} regulates gene Y{cluster_id} "
                    f"through signaling pathway Z{cluster_id} in cancer cells. "
                    f"The results show significant effects on cell proliferation and apoptosis. "
                    f"Future studies should investigate the downstream targets of this pathway."
                ),
                "year": 2020 + cluster_id,
                "mesh_terms": [f"Topic{cluster_id}", "Cancer", f"Gene{cluster_id}"],
                "concepts": {
                    "genes": [f"GENE{cluster_id}A", f"GENE{cluster_id}B"],
                    "diseases": ["cancer"],
                    "chemicals": [],
                    "mutations": [],
                },
            })

    return np.vstack(embeddings), np.array(labels_true), papers


class TestPaperClusterer:
    """Tests for PaperClusterer."""

    def setup_method(self):
        self.clusterer = PaperClusterer()

    def test_cluster_returns_array(self):
        """Test that cluster returns a numpy array of labels."""
        embeddings, _, _ = make_synthetic_embeddings(n_clusters=3, n_per_cluster=20)
        labels = self.clusterer.cluster(embeddings, min_cluster_size=5)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(embeddings)

    def test_cluster_labels_are_integers(self):
        """Test that all labels are integers."""
        embeddings, _, _ = make_synthetic_embeddings(n_clusters=3, n_per_cluster=15)
        labels = self.clusterer.cluster(embeddings, min_cluster_size=5)
        assert labels.dtype in (np.int32, np.int64, np.int_)
        # Labels should be -1 (noise) or >= 0 (cluster)
        assert all(l >= -1 for l in labels)

    def test_cluster_detects_structure(self):
        """Test that clearly structured data produces multiple clusters."""
        embeddings, true_labels, _ = make_synthetic_embeddings(
            n_clusters=3, n_per_cluster=30, dim=64
        )
        labels = self.clusterer.cluster(embeddings, min_cluster_size=5)
        n_clusters = len(set(labels.tolist()) - {-1})
        # Should detect at least 2 clusters in clearly separated data
        assert n_clusters >= 2, f"Expected >= 2 clusters, got {n_clusters}"

    def test_cluster_handles_small_dataset(self):
        """Test clustering with fewer than min_cluster_size * 2 points."""
        embeddings = np.random.randn(5, 32).astype(np.float32)
        labels = self.clusterer.cluster(embeddings, min_cluster_size=10)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 5

    def test_cluster_handles_empty_input(self):
        """Test clustering with empty input."""
        embeddings = np.empty((0, 64), dtype=np.float32)
        labels = self.clusterer.cluster(embeddings, min_cluster_size=5)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 0

    def test_get_cluster_papers_groups_correctly(self):
        """Test that papers are grouped by cluster label."""
        papers = [{"pmid": str(i)} for i in range(6)]
        labels = np.array([0, 0, 1, 1, -1, 0])  # cluster 0: 3 papers, cluster 1: 2 papers, noise: 1

        result = self.clusterer.get_cluster_papers(labels, papers)

        assert 0 in result
        assert 1 in result
        assert -1 not in result  # Noise excluded
        assert len(result[0]) == 3
        assert len(result[1]) == 2

    def test_get_cluster_papers_excludes_noise(self):
        """Test that noise points (-1) are excluded from cluster groups."""
        papers = [{"pmid": "1"}, {"pmid": "2"}, {"pmid": "3"}]
        labels = np.array([-1, -1, -1])

        result = self.clusterer.get_cluster_papers(labels, papers)
        assert len(result) == 0

    def test_kmeans_fallback(self):
        """Test k-means fallback clustering."""
        embeddings, _, _ = make_synthetic_embeddings(n_clusters=3, n_per_cluster=20)
        labels = self.clusterer._kmeans_cluster(embeddings, n_clusters=3)
        assert isinstance(labels, np.ndarray)
        assert len(set(labels.tolist())) == 3  # Exactly 3 clusters from k-means


class TestClusterEvaluator:
    """Tests for ClusterEvaluator."""

    def setup_method(self):
        self.evaluator = ClusterEvaluator()

    def test_evaluate_returns_list_of_dicts(self):
        """Test that evaluate returns a list of summary dicts."""
        embeddings, labels, papers = make_synthetic_embeddings(n_clusters=3, n_per_cluster=15)
        summaries = self.evaluator.evaluate(embeddings, labels, papers)
        assert isinstance(summaries, list)

    def test_evaluate_summary_has_required_fields(self):
        """Test that each cluster summary has required fields."""
        embeddings, labels, papers = make_synthetic_embeddings(n_clusters=3, n_per_cluster=15)
        summaries = self.evaluator.evaluate(embeddings, labels, papers)

        required_fields = {"cluster_id", "label", "paper_count", "top_terms", "silhouette_score", "top_papers"}
        for summary in summaries:
            assert required_fields.issubset(set(summary.keys())), (
                f"Missing fields: {required_fields - set(summary.keys())}"
            )

    def test_evaluate_silhouette_in_range(self):
        """Test that silhouette scores are in [-1, 1]."""
        embeddings, labels, papers = make_synthetic_embeddings(n_clusters=3, n_per_cluster=15)
        summaries = self.evaluator.evaluate(embeddings, labels, papers)
        for summary in summaries:
            assert -1.0 <= summary["silhouette_score"] <= 1.0, (
                f"Silhouette score out of range: {summary['silhouette_score']}"
            )

    def test_evaluate_paper_counts_correct(self):
        """Test that paper counts in summaries match actual cluster sizes."""
        embeddings, labels, papers = make_synthetic_embeddings(
            n_clusters=2, n_per_cluster=10
        )
        # Force clean labels (no noise for this test)
        clean_labels = labels.copy()

        summaries = self.evaluator.evaluate(embeddings, clean_labels, papers)
        total_from_summaries = sum(s["paper_count"] for s in summaries)
        expected_total = (clean_labels != -1).sum()
        assert total_from_summaries == expected_total

    def test_evaluate_handles_single_cluster(self):
        """Test evaluation with all papers in one cluster."""
        embeddings = np.random.randn(10, 32).astype(np.float32)
        labels = np.zeros(10, dtype=np.int32)
        papers = [{"pmid": str(i), "title": "test", "abstract": "test abstract " * 10,
                   "mesh_terms": [], "concepts": {}} for i in range(10)]

        summaries = self.evaluator.evaluate(embeddings, labels, papers)
        # Should return one summary
        assert len(summaries) == 1
        assert summaries[0]["paper_count"] == 10

    def test_evaluate_excludes_noise(self):
        """Test that noise points are excluded from cluster summaries."""
        embeddings, labels, papers = make_synthetic_embeddings(n_clusters=2, n_per_cluster=10)
        # Add some artificial noise
        labels_with_noise = labels.copy()
        labels_with_noise[:5] = -1

        summaries = self.evaluator.evaluate(embeddings, labels_with_noise, papers)
        total_from_summaries = sum(s["paper_count"] for s in summaries)
        assert total_from_summaries == (labels_with_noise != -1).sum()

    def test_evaluate_generates_labels(self):
        """Test that cluster labels are generated and non-empty."""
        embeddings, labels, papers = make_synthetic_embeddings(n_clusters=2, n_per_cluster=15)
        summaries = self.evaluator.evaluate(embeddings, labels, papers)
        for summary in summaries:
            assert isinstance(summary["label"], str)
            assert len(summary["label"]) > 0
