import logging
import re
from typing import Optional

from llm.reasoner import LLMReasoner
from schemas.output_schema import Hypothesis, ResearchGap, SuggestedExperiment
from experiments.experiment_ontology import MODALITY_ASSAY_MAP, CONCEPT_TO_MODALITY

logger = logging.getLogger(__name__)

EXPERIMENT_SYSTEM_PROMPT = (
    "You are an expert experimental biologist. "
    "Design specific, feasible experiments to test scientific hypotheses. "
    "Be concrete: specify exact assays, models, and readouts. "
    "Return only valid JSON."
)


class ExperimentGenerator:
    """Generate suggested experiments for validated hypotheses."""

    def __init__(self, reasoner: LLMReasoner):
        self.reasoner = reasoner

    def generate(
        self,
        hypotheses: list[Hypothesis],
        gaps: list[ResearchGap],
    ) -> list[SuggestedExperiment]:
        """
        Generate 2-3 experiments per top hypothesis.

        Maps to experimental modalities and returns SuggestedExperiment objects.
        """
        if not hypotheses:
            logger.warning("ExperimentGenerator: no hypotheses provided")
            return []

        all_experiments: list[SuggestedExperiment] = []

        # Process top 3 hypotheses
        for hypothesis in hypotheses[:3]:
            try:
                experiments = self._generate_for_hypothesis(hypothesis, gaps)
                all_experiments.extend(experiments)
            except Exception as e:
                logger.warning(f"Experiment generation failed for hypothesis: {e}")

        logger.info(f"ExperimentGenerator: {len(all_experiments)} experiments generated")
        return all_experiments

    def _generate_for_hypothesis(
        self, hypothesis: Hypothesis, gaps: list[ResearchGap]
    ) -> list[SuggestedExperiment]:
        """Generate 2-3 experiments for a single hypothesis."""
        # Determine relevant modalities from hypothesis text
        modalities = self._map_hypothesis_to_modalities(hypothesis.hypothesis)

        # Build prompt for LLM
        gap_context = self._format_gap_context(gaps[:3])
        modality_context = self._format_modality_options(modalities)

        prompt = f"""Design 2-3 specific experiments to test this hypothesis.

Hypothesis:
"{hypothesis.hypothesis}"

Context from research gaps:
{gap_context}

Available experimental modalities and assays:
{modality_context}

For each experiment, provide:
- A specific experiment title/description
- The experimental modality (choose from: {', '.join(modalities)})
- Specific assays to use (list 2-4)
- Required models (cell lines, animal models, etc.)
- Complexity (low/medium/high)

Return a JSON array of exactly 2-3 experiment objects with this structure:
[
  {{
    "experiment": "<specific experiment description>",
    "modality": "<modality name>",
    "assays": ["<assay1>", "<assay2>"],
    "required_models": ["<model1>", "<model2>"],
    "complexity": "low|medium|high"
  }}
]"""

        try:
            raw = self.reasoner.complete_json(
                prompt=prompt,
                system=EXPERIMENT_SYSTEM_PROMPT,
                max_tokens=2048,
                temperature=0.3,
            )
        except Exception as e:
            logger.warning(f"ExperimentGenerator LLM call failed: {e}")
            # Fallback: create template experiments from modality map
            return self._fallback_experiments(hypothesis, modalities)

        return self._parse_experiments(raw, modalities)

    def _map_hypothesis_to_modalities(self, hypothesis_text: str) -> list[str]:
        """
        Map hypothesis text to relevant experimental modalities using keyword matching.
        """
        text_lower = hypothesis_text.lower()
        modality_scores: dict[str, int] = {m: 0 for m in MODALITY_ASSAY_MAP.keys()}

        for concept_kw, modality_list in CONCEPT_TO_MODALITY.items():
            if concept_kw.replace("_", " ") in text_lower or concept_kw in text_lower:
                for modality in modality_list:
                    modality_scores[modality] = modality_scores.get(modality, 0) + 1

        # Also check for direct assay-related keywords
        keyword_modality_map = {
            "knock": "genetic_perturbation",
            "crispr": "genetic_perturbation",
            "sirna": "genetic_perturbation",
            "inhibit": "pharmacological",
            "drug": "pharmacological",
            "compound": "pharmacological",
            "rna-seq": "omics_profiling",
            "proteom": "omics_profiling",
            "chip": "omics_profiling",
            "western": "molecular_profiling",
            "elisa": "molecular_profiling",
            "pcr": "molecular_profiling",
            "flow cytom": "cell_biology",
            "imaging": "cell_biology",
            "migration": "cell_biology",
            "mouse": "in_vivo",
            "xenograft": "in_vivo",
            "in vivo": "in_vivo",
            "animal": "in_vivo",
        }

        for kw, modality in keyword_modality_map.items():
            if kw in text_lower:
                modality_scores[modality] = modality_scores.get(modality, 0) + 2

        # Sort by score, return top 3 modalities
        sorted_modalities = sorted(modality_scores.items(), key=lambda x: x[1], reverse=True)
        top_modalities = [m for m, score in sorted_modalities if score > 0][:3]

        # Ensure at least 2 modalities
        if len(top_modalities) < 2:
            top_modalities = ["molecular_profiling", "cell_biology", "genetic_perturbation"]

        return top_modalities

    def _format_gap_context(self, gaps: list[ResearchGap]) -> str:
        """Format gaps as context for the experiment prompt."""
        if not gaps:
            return "No specific gap context."
        lines = []
        for gap in gaps:
            lines.append(f"- {gap.gap}")
        return "\n".join(lines)

    def _format_modality_options(self, modalities: list[str]) -> str:
        """Format relevant modality options for the prompt."""
        lines = []
        for modality in modalities:
            info = MODALITY_ASSAY_MAP.get(modality, {})
            assays = ", ".join(info.get("assays", [])[:4])
            models = ", ".join(info.get("models", [])[:3])
            complexity = info.get("complexity", "medium")
            lines.append(
                f"**{modality}** (complexity: {complexity})\n"
                f"  Assays: {assays}\n"
                f"  Models: {models}"
            )
        return "\n\n".join(lines)

    def _parse_experiments(self, raw: object, default_modalities: list[str]) -> list[SuggestedExperiment]:
        """Parse LLM response into SuggestedExperiment objects."""
        if isinstance(raw, list):
            exp_list = raw
        elif isinstance(raw, dict):
            for key in ["experiments", "results", "data"]:
                if key in raw and isinstance(raw[key], list):
                    exp_list = raw[key]
                    break
            else:
                exp_list = [raw]
        else:
            return []

        experiments = []
        for item in exp_list:
            if not isinstance(item, dict):
                continue
            try:
                modality = str(item.get("modality", default_modalities[0] if default_modalities else "molecular_profiling"))
                # Normalize modality
                if modality not in MODALITY_ASSAY_MAP:
                    modality = self._normalize_modality(modality)

                assays = item.get("assays", [])
                if not isinstance(assays, list):
                    assays = [str(assays)]
                assays = [str(a) for a in assays]

                models = item.get("required_models", [])
                if not isinstance(models, list):
                    models = [str(models)]
                models = [str(m) for m in models]

                complexity = str(item.get("complexity", "medium")).lower()
                if complexity not in {"low", "medium", "high"}:
                    complexity = MODALITY_ASSAY_MAP.get(modality, {}).get("complexity", "medium")

                experiment_text = str(item.get("experiment", "")).strip()
                if not experiment_text:
                    continue

                experiments.append(SuggestedExperiment(
                    experiment=experiment_text[:500],
                    modality=modality,
                    assays=assays[:6],
                    complexity=complexity,
                    required_models=models[:5],
                ))
            except Exception as e:
                logger.warning(f"Failed to parse experiment item: {e}")

        return experiments[:3]

    def _normalize_modality(self, modality: str) -> str:
        """Normalize a free-text modality string to a known key."""
        modality_lower = modality.lower().replace(" ", "_").replace("-", "_")
        # Direct match
        if modality_lower in MODALITY_ASSAY_MAP:
            return modality_lower
        # Partial match
        for key in MODALITY_ASSAY_MAP.keys():
            if key in modality_lower or modality_lower in key:
                return key
        # Keyword matching
        if any(kw in modality_lower for kw in ["genetic", "crispr", "sirna", "knockout"]):
            return "genetic_perturbation"
        if any(kw in modality_lower for kw in ["drug", "inhibit", "pharma", "compound"]):
            return "pharmacological"
        if any(kw in modality_lower for kw in ["rna", "seq", "omic", "proteo"]):
            return "omics_profiling"
        if any(kw in modality_lower for kw in ["western", "elisa", "pcr", "blot", "immuno"]):
            return "molecular_profiling"
        if any(kw in modality_lower for kw in ["cell", "flow", "imaging", "migration"]):
            return "cell_biology"
        if any(kw in modality_lower for kw in ["vivo", "mouse", "animal", "xenograft"]):
            return "in_vivo"
        return "molecular_profiling"  # Default

    def _fallback_experiments(
        self, hypothesis: Hypothesis, modalities: list[str]
    ) -> list[SuggestedExperiment]:
        """Generate template experiments when LLM fails."""
        experiments = []
        for modality in modalities[:2]:
            info = MODALITY_ASSAY_MAP.get(modality, {})
            experiments.append(SuggestedExperiment(
                experiment=f"Test hypothesis '{hypothesis.hypothesis[:100]}...' using {modality} approach",
                modality=modality,
                assays=info.get("assays", ["Western blot", "qRT-PCR"])[:3],
                complexity=info.get("complexity", "medium"),
                required_models=info.get("models", ["cell lines"])[:2],
            ))
        return experiments
