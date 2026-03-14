import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RetrospectiveEvaluator:
    """
    Evaluate hypothesis quality by checking if predictions from a cutoff year
    are confirmed by papers published after the cutoff (in validation_year_range).
    """

    def __init__(self, pipeline_runner=None):
        """
        pipeline_runner: callable that takes (query, max_papers, year_range) and returns
        an AnalysisResponse-like object. If None, must be set before calling evaluate().
        """
        self.pipeline_runner = pipeline_runner

    def evaluate(
        self,
        query: str,
        cutoff_year: int,
        validation_year_range: tuple[int, int],
        max_papers: int = 1000,
        llm_preset: str = "balanced",
    ) -> dict:
        """
        Run full pipeline on corpus up to cutoff_year, then check if generated
        hypotheses are confirmed by papers in validation_year_range.

        Returns:
            dict with keys: precision, recall, hit_rate, confirmed_hypotheses,
                           total_hypotheses, validation_papers_checked
        """
        if self.pipeline_runner is None:
            raise RuntimeError("pipeline_runner must be set before calling evaluate()")

        logger.info(
            f"RetrospectiveEvaluator: query='{query}', cutoff={cutoff_year}, "
            f"validation={validation_year_range}"
        )

        # Step 1: Run pipeline on historical corpus (up to cutoff_year)
        historical_start = max(2000, cutoff_year - 10)
        logger.info(f"Running pipeline on historical corpus: {historical_start}-{cutoff_year}")

        try:
            historical_response = self.pipeline_runner(
                query=query,
                max_papers=max_papers,
                year_range=(historical_start, cutoff_year),
                llm_preset=llm_preset,
            )
        except Exception as e:
            logger.error(f"Historical pipeline run failed: {e}")
            return self._empty_result()

        hypotheses = getattr(historical_response, "hypotheses", [])
        if not hypotheses:
            logger.warning("No hypotheses generated from historical corpus")
            return self._empty_result()

        # Step 2: Retrieve validation papers (after cutoff)
        logger.info(f"Retrieving validation papers: {validation_year_range}")
        try:
            validation_response = self.pipeline_runner(
                query=query,
                max_papers=max(200, max_papers // 5),
                year_range=validation_year_range,
                llm_preset="cheap_fast",  # Use cheaper preset for validation
            )
            validation_papers = self._extract_papers(validation_response)
        except Exception as e:
            logger.error(f"Validation retrieval failed: {e}")
            validation_papers = []

        logger.info(f"Retrieved {len(validation_papers)} validation papers")

        # Step 3: Check each hypothesis against validation papers
        confirmed = []
        unconfirmed = []

        for hypothesis in hypotheses:
            is_confirmed, matching_pmids = self._check_hypothesis_confirmed(
                hypothesis.hypothesis, validation_papers
            )
            if is_confirmed:
                confirmed.append({
                    "hypothesis": hypothesis.hypothesis[:200],
                    "confirmed_by": matching_pmids[:3],
                })
            else:
                unconfirmed.append(hypothesis.hypothesis[:200])

        total = len(hypotheses)
        n_confirmed = len(confirmed)
        hit_rate = n_confirmed / total if total > 0 else 0.0

        # Compute precision and recall
        # Precision: of all hypotheses generated, what fraction was confirmed?
        precision = hit_rate

        # Recall: of all validation papers' findings, what fraction did we predict?
        # (Approximate: count unique confirmed PMIDs / total validation papers)
        confirmed_pmids = set()
        for c in confirmed:
            confirmed_pmids.update(c["confirmed_by"])
        recall = len(confirmed_pmids) / len(validation_papers) if validation_papers else 0.0

        result = {
            "precision": round(precision, 4),
            "recall": round(min(1.0, recall), 4),
            "hit_rate": round(hit_rate, 4),
            "confirmed_hypotheses": confirmed,
            "unconfirmed_hypotheses": unconfirmed,
            "total_hypotheses": total,
            "validation_papers_checked": len(validation_papers),
            "cutoff_year": cutoff_year,
            "validation_year_range": validation_year_range,
        }

        logger.info(
            f"RetrospectiveEval: precision={precision:.3f}, recall={recall:.3f}, "
            f"hit_rate={hit_rate:.3f}, confirmed={n_confirmed}/{total}"
        )
        return result

    def _check_hypothesis_confirmed(
        self, hypothesis: str, validation_papers: list[dict]
    ) -> tuple[bool, list[str]]:
        """
        Check if a hypothesis is confirmed by validation papers.

        Uses keyword overlap heuristic: if >50% of key hypothesis terms appear
        in a paper's abstract, consider it a confirmation.
        """
        import re

        # Extract key terms from hypothesis
        key_terms = self._extract_key_terms(hypothesis)
        if not key_terms:
            return False, []

        matching_pmids = []
        threshold = max(2, len(key_terms) // 2)  # At least half the terms must match

        for paper in validation_papers:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            matches = sum(1 for term in key_terms if term.lower() in paper_text)
            if matches >= threshold:
                pmid = paper.get("pmid", "")
                if pmid:
                    matching_pmids.append(pmid)

        is_confirmed = len(matching_pmids) >= 1
        return is_confirmed, matching_pmids

    def _extract_key_terms(self, hypothesis: str) -> list[str]:
        """Extract important key terms from hypothesis text."""
        import re
        # Gene names (all-caps)
        genes = re.findall(r"\b[A-Z][A-Z0-9]{1,8}\b", hypothesis)
        # Biological process terms
        processes = re.findall(
            r"\b(?:phosphorylation|ubiquitination|methylation|acetylation|"
            r"apoptosis|ferroptosis|autophagy|senescence|angiogenesis|"
            r"metastasis|signaling|pathway|expression|inhibition|activation)\b",
            hypothesis, re.IGNORECASE
        )
        # Disease terms
        diseases = re.findall(
            r"\b(?:cancer|carcinoma|tumor|disease|syndrome|diabetes|"
            r"alzheimer|parkinson|fibrosis)\b",
            hypothesis, re.IGNORECASE
        )
        all_terms = list(set(genes[:4] + processes[:3] + diseases[:2]))
        return all_terms

    def _extract_papers(self, response) -> list[dict]:
        """Extract papers from pipeline response (handles different response formats)."""
        if hasattr(response, "papers_retrieved"):
            # It's an AnalysisResponse — we need raw papers
            # In practice, the pipeline runner should return raw papers too
            return []
        if isinstance(response, list):
            return response
        return []

    def _empty_result(self) -> dict:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "confirmed_hypotheses": [],
            "unconfirmed_hypotheses": [],
            "total_hypotheses": 0,
            "validation_papers_checked": 0,
        }
