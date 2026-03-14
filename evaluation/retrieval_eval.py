import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate retrieval quality using benchmark queries with known relevant papers."""

    def __init__(self, pubmed_searcher):
        """
        pubmed_searcher: PubMedSearcher instance for executing searches.
        """
        self.pubmed_searcher = pubmed_searcher

    def evaluate(self, benchmark_queries: list[dict]) -> dict:
        """
        Evaluate retrieval performance on benchmark queries.

        Each benchmark_query dict has:
            - query: str
            - expected_pmids: list[str] — ground truth relevant PMIDs
            - expected_concepts: list[str] — expected concepts to appear in results

        Returns:
            dict with aggregated precision@k, recall@k, MRR metrics.
        """
        if not benchmark_queries:
            logger.warning("RetrievalEvaluator: no benchmark queries provided")
            return {}

        results = []
        k_values = [5, 10, 20]

        for i, bq in enumerate(benchmark_queries):
            query = bq.get("query", "")
            expected_pmids = set(str(p) for p in bq.get("expected_pmids", []))
            expected_concepts = bq.get("expected_concepts", [])

            logger.info(f"Evaluating query {i+1}/{len(benchmark_queries)}: '{query[:60]}'")

            try:
                retrieved_papers = self.pubmed_searcher.search(
                    query=query,
                    max_papers=max(k_values),
                    year_range=(2010, 2025),
                    article_types=["research", "review"],
                )
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                results.append(self._empty_query_result(query, k_values))
                continue

            retrieved_pmids = [str(p.get("pmid", "")) for p in retrieved_papers]

            query_metrics = {"query": query}
            # Precision@k and Recall@k
            for k in k_values:
                top_k_pmids = set(retrieved_pmids[:k])
                if not expected_pmids:
                    precision_k = 0.0
                    recall_k = 0.0
                else:
                    hits = top_k_pmids & expected_pmids
                    precision_k = len(hits) / k if k > 0 else 0.0
                    recall_k = len(hits) / len(expected_pmids)
                query_metrics[f"precision@{k}"] = round(precision_k, 4)
                query_metrics[f"recall@{k}"] = round(recall_k, 4)

            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for rank, pmid in enumerate(retrieved_pmids, 1):
                if pmid in expected_pmids:
                    mrr = 1.0 / rank
                    break
            query_metrics["mrr"] = round(mrr, 4)

            # Concept recall: fraction of expected concepts appearing in retrieved abstracts
            if expected_concepts:
                all_text = " ".join(
                    f"{p.get('title', '')} {p.get('abstract', '')}".lower()
                    for p in retrieved_papers[:20]
                )
                found_concepts = sum(
                    1 for c in expected_concepts if c.lower() in all_text
                )
                concept_recall = found_concepts / len(expected_concepts)
            else:
                concept_recall = 0.0
            query_metrics["concept_recall"] = round(concept_recall, 4)

            results.append(query_metrics)

        # Aggregate metrics across all queries
        aggregated = self._aggregate_metrics(results, k_values)
        aggregated["per_query_results"] = results
        logger.info(f"RetrievalEvaluator: evaluated {len(benchmark_queries)} queries")
        return aggregated

    def _aggregate_metrics(self, results: list[dict], k_values: list[int]) -> dict:
        """Compute mean metrics across all query results."""
        agg = {}
        metric_keys = [f"precision@{k}" for k in k_values] + \
                      [f"recall@{k}" for k in k_values] + \
                      ["mrr", "concept_recall"]

        for key in metric_keys:
            values = [r.get(key, 0.0) for r in results if key in r]
            agg[f"mean_{key}"] = round(sum(values) / len(values), 4) if values else 0.0

        return agg

    def _empty_query_result(self, query: str, k_values: list[int]) -> dict:
        """Return zero-value metrics for a failed query."""
        result = {"query": query, "mrr": 0.0, "concept_recall": 0.0}
        for k in k_values:
            result[f"precision@{k}"] = 0.0
            result[f"recall@{k}"] = 0.0
        return result
