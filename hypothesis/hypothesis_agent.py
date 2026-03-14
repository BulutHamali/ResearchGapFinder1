import logging
from pathlib import Path
from typing import Optional

from llm.reasoner import LLMReasoner
from schemas.output_schema import ResearchGap, Hypothesis
from hypothesis.novelty_checker import NoveltyChecker
from hypothesis.hypothesis_scorer import HypothesisScorer

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "hypothesis_prompt.txt"

HYPOTHESIS_SYSTEM_PROMPT = (
    "You are a creative yet rigorous biomedical research scientist. "
    "Generate mechanistic, testable hypotheses grounded in the provided evidence. "
    "Be specific: name proteins, pathways, cell types. "
    "Avoid vague or speculative claims that cannot be tested in standard lab settings. "
    "Return only valid JSON."
)

MAX_HYPOTHESES = 5


class HypothesisAgent:
    """Generate, validate, and score research hypotheses from detected gaps."""

    def __init__(
        self,
        reasoner: LLMReasoner,
        novelty_checker: NoveltyChecker,
        scorer: HypothesisScorer,
    ):
        self.reasoner = reasoner
        self.novelty_checker = novelty_checker
        self.scorer = scorer
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Hypothesis prompt not found at {PROMPT_PATH}, using default")
            return self._default_template()

    def _default_template(self) -> str:
        return """Generate mechanistic, testable hypotheses for the following research gaps.

Query: {query}

Research Gaps:
{gaps}

Literature Context (Cluster Summaries):
{cluster_summaries}

Return a JSON array of hypothesis objects."""

    def generate(
        self,
        query: str,
        gaps: list[ResearchGap],
        cluster_summaries: list[dict],
    ) -> list[Hypothesis]:
        """
        Generate hypotheses for each gap, check novelty, score, and filter.

        Returns ranked list of Hypothesis objects (by composite score).
        """
        if not gaps:
            logger.warning("HypothesisAgent: no gaps provided, returning empty list")
            return []

        # Format inputs for prompt
        gaps_text = self._format_gaps(gaps)
        cluster_text = self._format_cluster_summaries(cluster_summaries[:8])

        prompt = self._prompt_template.format(
            query=query,
            gaps=gaps_text,
            cluster_summaries=cluster_text,
        )

        logger.info(f"HypothesisAgent: generating hypotheses for {len(gaps)} gaps")

        try:
            raw_response = self.reasoner.complete_json(
                prompt=prompt,
                system=HYPOTHESIS_SYSTEM_PROMPT,
                max_tokens=None,
                temperature=None,
            )
        except Exception as e:
            logger.error(f"HypothesisAgent LLM call failed: {e}")
            return []

        raw_hypotheses = self._parse_raw_hypotheses(raw_response)
        logger.info(f"HypothesisAgent: {len(raw_hypotheses)} raw hypotheses generated")

        # Novelty check, score, and build Hypothesis objects
        hypotheses: list[Hypothesis] = []
        for raw_h in raw_hypotheses[:MAX_HYPOTHESES + 3]:  # Process extra to account for filtering
            hyp_text = raw_h.get("hypothesis", "").strip()
            if not hyp_text:
                continue

            # Novelty check
            try:
                novelty_result = self.novelty_checker.check(hyp_text)
            except Exception as e:
                logger.warning(f"Novelty check failed for hypothesis: {e}")
                novelty_result = {
                    "already_established": False,
                    "supporting_papers": [],
                    "novelty_evidence": "Novelty check unavailable.",
                    "novelty_score_hint": 0.7,
                }

            # Skip fully established hypotheses
            if novelty_result.get("already_established", False):
                logger.debug(f"Filtering established hypothesis: '{hyp_text[:60]}'")
                continue

            # Score hypothesis
            try:
                scores = self.scorer.score(
                    hypothesis=hyp_text,
                    novelty_evidence=novelty_result,
                    cluster_summaries=cluster_summaries,
                    reasoner=self.reasoner,
                )
            except Exception as e:
                logger.warning(f"Hypothesis scoring failed: {e}")
                scores = {
                    "novelty_score": novelty_result.get("novelty_score_hint", 0.5),
                    "support_score": 0.5,
                    "feasibility_score": 0.5,
                    "impact_score": 0.5,
                }

            reasoning_summary = str(raw_h.get("reasoning_summary", "")).strip()
            if not reasoning_summary:
                reasoning_summary = novelty_result.get("novelty_evidence", "")

            hypothesis = Hypothesis(
                hypothesis=hyp_text,
                novelty_score=scores["novelty_score"],
                support_score=scores["support_score"],
                feasibility_score=scores["feasibility_score"],
                impact_score=scores["impact_score"],
                reasoning_summary=reasoning_summary[:1000],
                already_established=novelty_result.get("already_established", False),
            )
            hypotheses.append(hypothesis)

            if len(hypotheses) >= MAX_HYPOTHESES:
                break

        # Rank by composite score
        hypotheses.sort(
            key=lambda h: (
                h.novelty_score * 0.35
                + h.support_score * 0.25
                + h.feasibility_score * 0.20
                + h.impact_score * 0.20
            ),
            reverse=True,
        )

        logger.info(f"HypothesisAgent: {len(hypotheses)} hypotheses after filtering and ranking")
        return hypotheses

    def _format_gaps(self, gaps: list[ResearchGap]) -> str:
        """Format ResearchGap objects for the prompt."""
        lines = []
        for i, gap in enumerate(gaps, 1):
            snippets = gap.evidence_snippets[:2]
            snippet_text = ""
            if snippets:
                snippet_text = f"\n   Evidence: \"{snippets[0].text[:200]}\""
            lines.append(
                f"{i}. [{gap.type}] {gap.gap}\n"
                f"   Reason underexplored: {gap.reason_underexplored}\n"
                f"   Uncertainty: {gap.uncertainty}"
                f"{snippet_text}"
            )
        return "\n\n".join(lines)

    def _format_cluster_summaries(self, summaries: list[dict]) -> str:
        """Format cluster summaries for the prompt."""
        lines = []
        for s in summaries:
            terms = ", ".join(s.get("top_terms", [])[:6])
            lines.append(
                f"Cluster {s.get('cluster_id', '?')} ({s.get('paper_count', 0)} papers): "
                f"{s.get('label', 'N/A')}\n  Key concepts: {terms}"
            )
        return "\n\n".join(lines) if lines else "No cluster data available."

    def _parse_raw_hypotheses(self, response: object) -> list[dict]:
        """Parse raw LLM response into a list of hypothesis dicts."""
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        elif isinstance(response, dict):
            for key in ["hypotheses", "results", "data"]:
                if key in response and isinstance(response[key], list):
                    return [item for item in response[key] if isinstance(item, dict)]
            return [response]
        return []
