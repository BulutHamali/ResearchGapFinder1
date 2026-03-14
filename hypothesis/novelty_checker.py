import logging
import re
from typing import Optional

from llm.reasoner import LLMReasoner

logger = logging.getLogger(__name__)

NOVELTY_SYSTEM_PROMPT = (
    "You are a biomedical research expert assessing scientific novelty. "
    "Evaluate whether a hypothesis is already established or genuinely novel based on provided literature. "
    "Be conservative: if strong supporting evidence exists, mark as established."
)


class NoveltyChecker:
    """Check whether a hypothesis is already established in the literature."""

    def __init__(self, pubmed_searcher, reasoner: LLMReasoner):
        self.pubmed_searcher = pubmed_searcher
        self.reasoner = reasoner

    def check(self, hypothesis: str) -> dict:
        """
        Check novelty of a hypothesis by searching PubMed and asking LLM.

        Returns dict with:
          - already_established: bool
          - supporting_papers: list of PMIDs
          - novelty_evidence: str explanation
          - novelty_score_hint: float (1.0 = fully novel, 0.0 = fully established)
        """
        logger.info(f"Checking novelty for hypothesis: '{hypothesis[:80]}...'")

        # Extract key search terms from the hypothesis
        search_terms = self._extract_search_terms(hypothesis)
        search_query = " AND ".join(f'"{t}"' for t in search_terms[:3])

        logger.debug(f"Novelty search query: '{search_query}'")

        # Search PubMed (last 5 years, max 50 papers)
        papers = []
        try:
            import datetime
            current_year = datetime.date.today().year
            papers = self.pubmed_searcher.search(
                query=search_query,
                max_papers=50,
                year_range=(current_year - 5, current_year),
                article_types=["research", "review"],
            )
        except Exception as e:
            logger.warning(f"NoveltyChecker PubMed search failed: {e}")

        supporting_pmids = [p["pmid"] for p in papers if p.get("pmid")]

        # Build context for LLM
        paper_summaries = self._format_paper_summaries(papers[:20])

        prompt = f"""You are assessing whether the following hypothesis is already established in scientific literature.

Hypothesis:
"{hypothesis}"

Relevant literature found (up to 20 papers):
{paper_summaries}

Based on the above literature, assess:
1. Is this hypothesis already well-established as scientific fact? (yes/no)
2. What is your confidence level? (0.0 = completely novel, 1.0 = fully established)
3. Provide a brief explanation of your assessment.

Return JSON with this exact structure:
{{
  "already_established": true/false,
  "novelty_score_hint": <float 0.0-1.0, where 1.0 = fully novel>,
  "novelty_evidence": "<explanation of novelty assessment>"
}}"""

        try:
            result = self.reasoner.complete_json(
                prompt=prompt,
                system=NOVELTY_SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.2,
            )
            already_established = bool(result.get("already_established", False))
            novelty_score = float(result.get("novelty_score_hint", 0.7))
            novelty_score = max(0.0, min(1.0, novelty_score))
            novelty_evidence = str(result.get("novelty_evidence", ""))

        except Exception as e:
            logger.warning(f"NoveltyChecker LLM call failed: {e}")
            # Default: assume moderately novel if no papers found, less novel if many found
            already_established = len(papers) > 20
            novelty_score = max(0.1, 1.0 - len(papers) / 50.0)
            novelty_evidence = f"Based on {len(papers)} related papers found. LLM assessment unavailable."

        return {
            "already_established": already_established,
            "supporting_papers": supporting_pmids[:10],
            "novelty_evidence": novelty_evidence,
            "novelty_score_hint": novelty_score,
        }

    def _extract_search_terms(self, hypothesis: str) -> list[str]:
        """
        Extract 2-3 key search terms from a hypothesis string.

        Focuses on biological entities (genes, diseases, mechanisms).
        """
        terms = []

        # Extract capitalized multi-word phrases or gene names
        gene_names = re.findall(r"\b[A-Z][A-Z0-9]{1,8}\b", hypothesis)
        terms.extend(gene_names[:3])

        # Extract disease names (title-cased multi-word phrases)
        disease_pattern = re.compile(
            r"\b(?:cancer|disease|syndrome|carcinoma|tumor|diabetes|hypertension|"
            r"alzheimer|parkinson|infection|fibrosis|inflammation)\b",
            re.IGNORECASE,
        )
        for match in disease_pattern.finditer(hypothesis):
            # Get the word before the disease term for context
            start = max(0, match.start() - 20)
            phrase = hypothesis[start: match.end()].strip()
            terms.append(phrase)
            if len(terms) >= 4:
                break

        # Extract key biological process terms
        process_terms = re.findall(
            r"\b(?:signaling|pathway|expression|regulation|activation|"
            r"inhibition|phosphorylation|transcription|metabolism|apoptosis|"
            r"ferroptosis|autophagy|senescence|angiogenesis|metastasis)\b",
            hypothesis, re.IGNORECASE,
        )
        terms.extend(process_terms[:2])

        # Remove duplicates and overly short terms
        seen: set[str] = set()
        unique_terms = []
        for t in terms:
            t_clean = t.strip()
            if t_clean.lower() not in seen and len(t_clean) > 2:
                seen.add(t_clean.lower())
                unique_terms.append(t_clean)

        # If we couldn't extract specific terms, use the first few meaningful words
        if len(unique_terms) < 2:
            words = [w for w in hypothesis.split() if len(w) > 4]
            unique_terms.extend(words[:3])

        return unique_terms[:3]

    def _format_paper_summaries(self, papers: list[dict]) -> str:
        """Format papers as readable summaries for the LLM."""
        if not papers:
            return "No relevant papers found."

        summaries = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "N/A")[:120]
            abstract = paper.get("abstract", "")[:200]
            year = paper.get("year", "N/A")
            pmid = paper.get("pmid", "N/A")
            summaries.append(f"{i}. [{year}] PMID:{pmid} - {title}\n   {abstract}...")

        return "\n\n".join(summaries)
