import logging
import math
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_samples, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """Evaluate cluster quality and generate cluster summaries."""

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        papers: list[dict],
    ) -> list[dict]:
        """
        Evaluate clustering quality and return cluster summary dicts.

        Returns list of dicts with: cluster_id, label, paper_count,
        top_terms, silhouette_score, top_papers.
        """
        assert len(embeddings) == len(labels) == len(papers)

        valid_mask = labels != -1
        if valid_mask.sum() < 2:
            logger.warning("Too few valid (non-noise) points for cluster evaluation")
            return []

        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        valid_papers = [p for p, m in zip(papers, valid_mask) if m]

        unique_labels = sorted(set(valid_labels.tolist()))
        if len(unique_labels) < 2:
            logger.warning("Only one cluster found; silhouette score not meaningful")
            sil_scores_map = {unique_labels[0]: 0.0}
        else:
            # Per-sample silhouette scores
            try:
                sil_samples = silhouette_samples(valid_embeddings, valid_labels, metric="euclidean")
            except Exception as e:
                logger.warning(f"Silhouette computation failed: {e}")
                sil_samples = np.zeros(len(valid_labels))
            sil_scores_map = {}
            for lbl in unique_labels:
                mask = valid_labels == lbl
                sil_scores_map[int(lbl)] = float(np.mean(sil_samples[mask]))

        # Davies-Bouldin score (global metric)
        try:
            if len(unique_labels) >= 2:
                db_score = float(davies_bouldin_score(valid_embeddings, valid_labels))
            else:
                db_score = 0.0
        except Exception as e:
            logger.warning(f"Davies-Bouldin score failed: {e}")
            db_score = 0.0
        logger.info(f"Davies-Bouldin score: {db_score:.4f}")

        # Group papers by cluster
        cluster_papers: dict[int, list[dict]] = {}
        for lbl, paper in zip(valid_labels.tolist(), valid_papers):
            lbl = int(lbl)
            cluster_papers.setdefault(lbl, []).append(paper)

        # TF-IDF on cluster texts for top terms
        all_cluster_texts = []
        cluster_order = sorted(cluster_papers.keys())
        for lbl in cluster_order:
            cluster_text = self._build_cluster_text(cluster_papers[lbl])
            all_cluster_texts.append(cluster_text)

        tfidf_top_terms = self._extract_tfidf_top_terms(all_cluster_texts, cluster_order)

        summaries = []
        for lbl in cluster_order:
            papers_in_cluster = cluster_papers[lbl]
            top_terms = tfidf_top_terms.get(lbl, [])
            # Also incorporate MeSH terms
            mesh_top = self._top_mesh_terms(papers_in_cluster, n=5)
            combined_terms = list(dict.fromkeys(top_terms + mesh_top))[:10]

            label = self._generate_label(combined_terms)
            top_pmids = [p.get("pmid", "") for p in papers_in_cluster[:5] if p.get("pmid")]

            summaries.append({
                "cluster_id": int(lbl),
                "label": label,
                "paper_count": len(papers_in_cluster),
                "top_terms": combined_terms,
                "silhouette_score": round(sil_scores_map.get(int(lbl), 0.0), 4),
                "top_papers": top_pmids,
            })

        logger.info(f"ClusterEvaluator: {len(summaries)} cluster summaries generated")
        return summaries

    def _build_cluster_text(self, papers: list[dict]) -> str:
        """Concatenate titles and abstracts for TF-IDF."""
        parts = []
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            parts.append(f"{title} {abstract}")
        return " ".join(parts)

    def _extract_tfidf_top_terms(
        self, cluster_texts: list[str], cluster_order: list[int], n_terms: int = 10
    ) -> dict[int, list[str]]:
        """Run TF-IDF over cluster texts and extract top terms per cluster."""
        if not cluster_texts or len(cluster_texts) < 2:
            return {lbl: [] for lbl in cluster_order}

        try:
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()

            result = {}
            for i, lbl in enumerate(cluster_order):
                row = tfidf_matrix[i].toarray()[0]
                top_indices = row.argsort()[::-1][:n_terms]
                result[lbl] = [str(feature_names[idx]) for idx in top_indices if row[idx] > 0]
            return result

        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return {lbl: [] for lbl in cluster_order}

    def _top_mesh_terms(self, papers: list[dict], n: int = 5) -> list[str]:
        """Return the most frequent MeSH terms in a cluster."""
        counter: Counter = Counter()
        for paper in papers:
            for term in paper.get("mesh_terms", []):
                counter[term] += 1
        return [term for term, _ in counter.most_common(n)]

    def _generate_label(self, top_terms: list[str]) -> str:
        """Generate a human-readable cluster label from top terms."""
        if not top_terms:
            return "Unknown Cluster"
        # Use the top 3 terms, title-cased
        label_parts = [t.title() for t in top_terms[:3]]
        return " / ".join(label_parts)
