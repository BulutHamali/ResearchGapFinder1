import json
import logging
import os
from pathlib import Path
from typing import Optional

from llm.reasoner import LLMReasoner
from schemas.output_schema import ResearchGap, EvidenceSnippet

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "gap_prompt.txt"

SYSTEM_PROMPT = (
    "You are an expert biomedical researcher and systematic review specialist. "
    "Your task is to identify genuine, actionable research gaps from scientific literature data. "
    "Be rigorous, specific, and evidence-based. Return only valid JSON."
)


class GapAgent:
    """LLM-based research gap detection agent."""

    def __init__(self, reasoner: LLMReasoner):
        self.reasoner = reasoner
        self._prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load the gap detection prompt template."""
        try:
            return PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Gap prompt not found at {PROMPT_PATH}, using default template")
            return self._default_prompt_template()

    def _default_prompt_template(self) -> str:
        return """You are analyzing research literature to identify genuine research gaps.

Query: {query}

Cluster Summaries:
{cluster_summaries}

Statistical Gap Candidates:
{candidate_gaps}

Please identify 3-7 genuine research gaps. Return a JSON array."""

    def detect(
        self,
        query: str,
        cluster_summaries: list[dict],
        candidate_gaps: list[dict],
    ) -> list[ResearchGap]:
        """
        Detect research gaps using LLM reasoning over cluster summaries and statistical candidates.

        Returns list of ResearchGap objects.
        """
        # Format cluster summaries for prompt
        cluster_text = self._format_cluster_summaries(cluster_summaries)

        # Format top statistical candidate gaps
        candidate_text = self._format_candidate_gaps(candidate_gaps[:10])

        prompt = self._prompt_template.format(
            query=query,
            cluster_summaries=cluster_text,
            candidate_gaps=candidate_text,
        )

        logger.info(f"GapAgent: detecting gaps for query='{query[:60]}'")

        try:
            raw_response = self.reasoner.complete_json(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=None,
                temperature=None,
            )
        except Exception as e:
            logger.error(f"GapAgent LLM call failed: {e}")
            return []

        gaps = self._parse_gaps(raw_response)
        logger.info(f"GapAgent: {len(gaps)} research gaps detected")
        return gaps

    def _format_cluster_summaries(self, summaries: list[dict]) -> str:
        """Format cluster summaries as readable text for the prompt."""
        lines = []
        for s in summaries:
            lines.append(
                f"Cluster {s.get('cluster_id', '?')} ({s.get('paper_count', 0)} papers): "
                f"'{s.get('label', 'N/A')}'\n"
                f"  Top terms: {', '.join(s.get('top_terms', [])[:8])}\n"
                f"  Silhouette score: {s.get('silhouette_score', 0):.3f}"
            )
        return "\n\n".join(lines) if lines else "No clusters available."

    def _format_candidate_gaps(self, candidates: list[dict]) -> str:
        """Format statistical gap candidates for the prompt."""
        if not candidates:
            return "No statistical gap candidates detected."
        lines = []
        for i, c in enumerate(candidates, 1):
            snippets = c.get("evidence_snippets", [])
            snippet_text = ""
            if snippets:
                first_snippet = snippets[0]
                snippet_text = f"\n  Evidence: \"{first_snippet.get('text', '')[:200]}\""
            lines.append(
                f"{i}. [{c.get('type', 'unknown')}] {c.get('description', '')[:300]}"
                f"{snippet_text}\n"
                f"   Score: {c.get('statistical_score', 0):.3f}"
            )
        return "\n\n".join(lines)

    def _parse_gaps(self, response: object) -> list[ResearchGap]:
        """Parse LLM response into ResearchGap objects."""
        if isinstance(response, list):
            gap_list = response
        elif isinstance(response, dict):
            # Try common wrapper keys
            for key in ["gaps", "research_gaps", "results", "data"]:
                if key in response and isinstance(response[key], list):
                    gap_list = response[key]
                    break
            else:
                gap_list = [response]
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return []

        gaps = []
        for item in gap_list:
            if not isinstance(item, dict):
                continue
            try:
                # Parse evidence snippets
                raw_snippets = item.get("evidence_snippets", [])
                snippets = []
                for s in raw_snippets:
                    if isinstance(s, dict):
                        snippets.append(EvidenceSnippet(
                            pmid=str(s.get("pmid", "")),
                            text=str(s.get("text", ""))[:500],
                        ))

                # Validate gap type
                gap_type = str(item.get("type", "explicit_gap"))
                valid_types = {"explicit_gap", "implicit_gap", "missing_link", "contradictory_gap"}
                if gap_type not in valid_types:
                    gap_type = "explicit_gap"

                # Validate uncertainty
                uncertainty = str(item.get("uncertainty", "medium")).lower()
                if uncertainty not in {"low", "medium", "high"}:
                    uncertainty = "medium"

                competing = item.get("competing_explanations", [])
                if not isinstance(competing, list):
                    competing = [str(competing)] if competing else []

                gap = ResearchGap(
                    gap=str(item.get("gap", "")).strip(),
                    type=gap_type,
                    evidence_snippets=snippets,
                    reason_underexplored=str(item.get("reason_underexplored", "")).strip(),
                    uncertainty=uncertainty,
                    competing_explanations=[str(c) for c in competing],
                )
                if gap.gap:
                    gaps.append(gap)

            except Exception as e:
                logger.warning(f"Failed to parse gap item: {e}. Item: {str(item)[:200]}")

        return gaps
