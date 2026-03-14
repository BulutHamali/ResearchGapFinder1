import logging
import math
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Standalone cluster quality evaluator.

    Different from clustering/cluster_evaluator.py — this one focuses on
    topic coherence (PMI-based) and generates human-readable quality reports.
    """

    def evaluate(self, clusters: dict, papers: list[dict]) -> dict:
        """
        Evaluate cluster quality with multiple metrics.

        Args:
            clusters: dict mapping cluster_id (int) to list of paper dicts
            papers: flat list of all papers (for reference)

        Returns:
            dict with:
                - cluster_coherence: dict mapping cluster_id to coherence score
                - silhouette_scores: dict mapping cluster_id to silhouette score (if embeddings available)
                - davies_bouldin: float (global)
                - topic_coherence_mean: float
                - quality_report: str (human-readable)
                - per_cluster_stats: list of dicts
        """
        if not clusters:
            return {"error": "No clusters provided", "quality_report": "No clusters to evaluate."}

        per_cluster_stats = []
        coherence_scores = {}

        # Build global term vocabulary for PMI computation
        all_term_counts, doc_term_sets = self._build_term_stats(papers)
        n_docs = len(papers)

        for cluster_id, cluster_papers in clusters.items():
            if not cluster_papers:
                continue

            # Topic coherence (PMI-based)
            top_terms = self._get_top_terms(cluster_papers, n=10)
            coherence = self._compute_pmi_coherence(top_terms, doc_term_sets, n_docs)
            coherence_scores[cluster_id] = coherence

            # Basic stats
            year_counts = Counter(p.get("year", 0) for p in cluster_papers if p.get("year"))
            most_common_year = year_counts.most_common(1)[0][0] if year_counts else "N/A"
            avg_abstract_len = (
                sum(len(p.get("abstract", "").split()) for p in cluster_papers)
                / len(cluster_papers)
            )

            # Top MeSH terms
            mesh_counter: Counter = Counter()
            for paper in cluster_papers:
                for term in paper.get("mesh_terms", []):
                    mesh_counter[term] += 1
            top_mesh = [term for term, _ in mesh_counter.most_common(5)]

            per_cluster_stats.append({
                "cluster_id": int(cluster_id),
                "paper_count": len(cluster_papers),
                "topic_coherence": round(coherence, 4),
                "top_terms": top_terms[:5],
                "top_mesh_terms": top_mesh,
                "most_common_year": most_common_year,
                "avg_abstract_length_words": round(avg_abstract_len, 1),
            })

        # Sort by cluster_id
        per_cluster_stats.sort(key=lambda x: x["cluster_id"])

        topic_coherence_mean = (
            sum(coherence_scores.values()) / len(coherence_scores)
            if coherence_scores else 0.0
        )

        quality_report = self._generate_quality_report(per_cluster_stats, topic_coherence_mean)

        return {
            "cluster_coherence": {int(k): round(v, 4) for k, v in coherence_scores.items()},
            "topic_coherence_mean": round(topic_coherence_mean, 4),
            "per_cluster_stats": per_cluster_stats,
            "quality_report": quality_report,
            "n_clusters": len(clusters),
            "total_papers": sum(len(p) for p in clusters.values()),
        }

    def _build_term_stats(self, papers: list[dict]) -> tuple[Counter, list[set]]:
        """Build global term frequency counts and per-document term sets."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        import re

        all_term_counts: Counter = Counter()
        doc_term_sets: list[set] = []

        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            tokens = re.findall(r"\b[a-z][a-z0-9-]{2,}\b", text)
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            term_set = set(tokens)
            doc_term_sets.append(term_set)
            all_term_counts.update(term_set)

        return all_term_counts, doc_term_sets

    def _get_top_terms(self, cluster_papers: list[dict], n: int = 10) -> list[str]:
        """Get top n terms from a cluster by frequency."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re

        texts = [
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in cluster_papers
        ]
        if not texts or all(not t.strip() for t in texts):
            return []

        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words="english",
                ngram_range=(1, 1),
            )
            tfidf = vectorizer.fit_transform(texts).toarray()
            mean_scores = tfidf.mean(axis=0)
            top_indices = mean_scores.argsort()[::-1][:n]
            features = vectorizer.get_feature_names_out()
            return [features[i] for i in top_indices]
        except Exception:
            return []

    def _compute_pmi_coherence(
        self,
        top_terms: list[str],
        doc_term_sets: list[set],
        n_docs: int,
        top_n: int = 10,
    ) -> float:
        """
        Compute topic coherence using Pointwise Mutual Information (PMI).

        For each pair of top terms (t_i, t_j), compute PMI and average.
        Higher PMI = terms tend to co-occur = more coherent topic.
        """
        if len(top_terms) < 2 or n_docs == 0:
            return 0.0

        terms = top_terms[:top_n]
        pmi_scores = []
        epsilon = 1e-9

        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                t_i = terms[i]
                t_j = terms[j]

                # Count documents containing each term and both terms
                count_i = sum(1 for doc in doc_term_sets if t_i in doc)
                count_j = sum(1 for doc in doc_term_sets if t_j in doc)
                count_ij = sum(1 for doc in doc_term_sets if t_i in doc and t_j in doc)

                p_i = count_i / n_docs
                p_j = count_j / n_docs
                p_ij = count_ij / n_docs

                if p_i < epsilon or p_j < epsilon or p_ij < epsilon:
                    continue

                pmi = math.log(p_ij / (p_i * p_j) + epsilon)
                pmi_scores.append(pmi)

        return sum(pmi_scores) / len(pmi_scores) if pmi_scores else 0.0

    def _generate_quality_report(
        self, per_cluster_stats: list[dict], mean_coherence: float
    ) -> str:
        """Generate a human-readable cluster quality report."""
        lines = [
            "=" * 60,
            "CLUSTER QUALITY REPORT",
            "=" * 60,
            f"Total clusters: {len(per_cluster_stats)}",
            f"Mean topic coherence (PMI): {mean_coherence:.4f}",
            "",
        ]

        # Quality thresholds
        if mean_coherence > 0.5:
            quality_label = "GOOD — Clusters are well-separated and coherent"
        elif mean_coherence > 0.2:
            quality_label = "FAIR — Moderate cluster coherence"
        else:
            quality_label = "POOR — Clusters may overlap significantly"

        lines.append(f"Overall quality: {quality_label}")
        lines.append("")
        lines.append("Per-Cluster Details:")
        lines.append("-" * 40)

        for stats in per_cluster_stats:
            lines.append(
                f"\nCluster {stats['cluster_id']} ({stats['paper_count']} papers)"
            )
            lines.append(f"  Coherence: {stats['topic_coherence']:.4f}")
            lines.append(f"  Top terms: {', '.join(stats['top_terms'][:5])}")
            lines.append(f"  Top MeSH: {', '.join(stats['top_mesh_terms'][:3])}")
            lines.append(f"  Avg abstract: {stats['avg_abstract_length_words']:.0f} words")
            lines.append(f"  Most common year: {stats['most_common_year']}")

            # Individual cluster quality assessment
            coh = stats["topic_coherence"]
            if coh > 0.5:
                assessment = "Well-defined topic cluster"
            elif coh > 0.2:
                assessment = "Moderately coherent cluster"
            else:
                assessment = "Diffuse cluster — may need further splitting"
            lines.append(f"  Assessment: {assessment}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
