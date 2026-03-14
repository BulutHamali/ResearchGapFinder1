import logging
from typing import Optional

from llm.reasoner import LLMReasoner

logger = logging.getLogger(__name__)

SCORER_SYSTEM_PROMPT = (
    "You are an expert biomedical researcher evaluating scientific hypotheses. "
    "Rate each dimension objectively based on the evidence provided. "
    "All scores must be floats between 0.0 and 1.0."
)


class HypothesisScorer:
    """Score hypotheses across multiple dimensions."""

    def score(
        self,
        hypothesis: str,
        novelty_evidence: dict,
        cluster_summaries: list[dict],
        reasoner: LLMReasoner,
    ) -> dict:
        """
        Score a hypothesis on novelty, support, feasibility, and impact.

        Returns dict with: novelty_score, support_score, feasibility_score, impact_score.
        All scores are floats clamped to [0.0, 1.0].
        """
        # Novelty score from novelty_checker results
        novelty_score = self._compute_novelty_score(novelty_evidence)

        # Use LLM for support, feasibility, and impact
        llm_scores = self._llm_score(hypothesis, cluster_summaries, novelty_evidence, reasoner)

        support_score = self._clamp(llm_scores.get("support_score", 0.5))
        feasibility_score = self._clamp(llm_scores.get("feasibility_score", 0.5))
        impact_score = self._clamp(llm_scores.get("impact_score", 0.5))

        return {
            "novelty_score": round(novelty_score, 3),
            "support_score": round(support_score, 3),
            "feasibility_score": round(feasibility_score, 3),
            "impact_score": round(impact_score, 3),
        }

    def _compute_novelty_score(self, novelty_evidence: dict) -> float:
        """
        Compute novelty score from novelty_checker results.

        1.0 = no supporting papers (fully novel)
        0.0 = fully established
        """
        if novelty_evidence.get("already_established", False):
            return 0.0

        hint = novelty_evidence.get("novelty_score_hint", 0.7)
        supporting_count = len(novelty_evidence.get("supporting_papers", []))

        # Penalize based on number of supporting papers found
        paper_penalty = min(0.4, supporting_count * 0.02)
        score = hint - paper_penalty
        return self._clamp(score)

    def _llm_score(
        self,
        hypothesis: str,
        cluster_summaries: list[dict],
        novelty_evidence: dict,
        reasoner: LLMReasoner,
    ) -> dict:
        """Use LLM to score support, feasibility, and impact."""
        cluster_text = self._format_clusters_brief(cluster_summaries[:5])
        novelty_text = novelty_evidence.get("novelty_evidence", "No novelty assessment available.")
        supporting_papers = novelty_evidence.get("supporting_papers", [])

        prompt = f"""Evaluate this scientific hypothesis on three dimensions.

Hypothesis:
"{hypothesis}"

Supporting literature context:
{cluster_text}

Novelty assessment:
{novelty_text}
Supporting papers found: {len(supporting_papers)}

Rate each dimension as a float from 0.0 to 1.0:

1. support_score: Strength of indirect evidence from the literature context (0=no support, 1=strong indirect evidence)
2. feasibility_score: How testable is this with standard lab methods like cell lines, mouse models, CRISPR, Western blots? (0=requires impossible technology, 1=immediately testable with standard tools)
3. impact_score: Clinical or mechanistic importance if confirmed (0=trivial, 1=paradigm-shifting)

Return JSON:
{{
  "support_score": <float>,
  "feasibility_score": <float>,
  "impact_score": <float>,
  "brief_reasoning": "<1-2 sentences justifying scores>"
}}"""

        try:
            result = reasoner.complete_json(
                prompt=prompt,
                system=SCORER_SYSTEM_PROMPT,
                max_tokens=512,
                temperature=0.2,
            )
            return result
        except Exception as e:
            logger.warning(f"HypothesisScorer LLM call failed: {e}")
            return {
                "support_score": 0.5,
                "feasibility_score": 0.5,
                "impact_score": 0.5,
            }

    def _format_clusters_brief(self, summaries: list[dict]) -> str:
        """Format cluster summaries briefly for the scoring prompt."""
        if not summaries:
            return "No cluster data available."
        lines = []
        for s in summaries:
            terms = ", ".join(s.get("top_terms", [])[:5])
            lines.append(
                f"Cluster {s.get('cluster_id', '?')} ({s.get('paper_count', 0)} papers): "
                f"{s.get('label', 'N/A')} — Key terms: {terms}"
            )
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a float to [0.0, 1.0]."""
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5
