import logging
import re
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
PUBTATOR_BATCH_SIZE = 20

# Regex patterns for fallback concept extraction
GENE_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,6}(?:-\d+)?)\b(?=\s+(?:gene|protein|mRNA|expression|mutation|knockout))",
    re.UNICODE,
)
MUTATION_PATTERN = re.compile(
    r"\b([A-Z]\d+[A-Z]|p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}|c\.\d+[ACGT]>[ACGT])\b"
)
CHEMICAL_PATTERN = re.compile(
    r"\b(?:inhibitor|compound|drug|molecule|agent|treatment|therapy|chemotherapy)\b",
    re.IGNORECASE,
)
DISEASE_KEYWORDS = [
    "cancer", "carcinoma", "tumor", "sarcoma", "lymphoma", "leukemia",
    "disease", "syndrome", "disorder", "diabetes", "hypertension",
    "alzheimer", "parkinson", "depression", "anxiety", "infection",
    "sepsis", "fibrosis", "atherosclerosis", "stroke", "infarction",
]


class ConceptExtractor:
    """Extract biomedical concepts from papers using PubTator3 API with fallback."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ResearchGapFinder/1.0"})

    def extract(self, papers: list[dict]) -> list[dict]:
        """
        Add a 'concepts' field to each paper dict.

        Tries PubTator3 API first; falls back to regex extraction.
        """
        # Separate papers with and without PMIDs
        papers_with_pmid = [p for p in papers if p.get("pmid")]
        papers_without_pmid = [p for p in papers if not p.get("pmid")]

        # Build pmid -> paper index for updates
        pmid_to_paper: dict[str, dict] = {p["pmid"]: p for p in papers_with_pmid}

        # Try PubTator in batches
        pmids = list(pmid_to_paper.keys())
        for i in range(0, len(pmids), PUBTATOR_BATCH_SIZE):
            batch_pmids = pmids[i: i + PUBTATOR_BATCH_SIZE]
            try:
                annotations = self._fetch_pubtator(batch_pmids)
                for pmid, concepts in annotations.items():
                    if pmid in pmid_to_paper:
                        pmid_to_paper[pmid]["concepts"] = concepts
            except Exception as e:
                logger.warning(f"PubTator batch failed (PMIDs {batch_pmids[:3]}...): {e}. Using fallback.")
                for pmid in batch_pmids:
                    if pmid in pmid_to_paper:
                        pmid_to_paper[pmid]["concepts"] = self._regex_extract(pmid_to_paper[pmid])

            time.sleep(0.5)  # Rate limit

        # Ensure all papers with PMIDs have concepts
        for pmid, paper in pmid_to_paper.items():
            if "concepts" not in paper:
                paper["concepts"] = self._regex_extract(paper)

        # Papers without PMIDs: use fallback
        for paper in papers_without_pmid:
            paper["concepts"] = self._regex_extract(paper)

        return papers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10), reraise=True)
    def _fetch_pubtator(self, pmids: list[str]) -> dict[str, dict]:
        """Fetch PubTator3 annotations for a batch of PMIDs."""
        params = {"pmids": ",".join(pmids)}
        resp = self._session.get(PUBTATOR_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results: dict[str, dict] = {}
        passages = data if isinstance(data, list) else data.get("PubTator3", [])

        for doc in passages:
            pmid = str(doc.get("id", "")).strip()
            if not pmid:
                continue

            concepts: dict[str, list[str]] = {
                "genes": [],
                "diseases": [],
                "chemicals": [],
                "mutations": [],
                "species": [],
            }

            for passage in doc.get("passages", []):
                for annotation in passage.get("annotations", []):
                    infons = annotation.get("infons", {})
                    ann_type = infons.get("type", "").lower()
                    text = annotation.get("text", "").strip()
                    if not text:
                        continue

                    if ann_type == "gene":
                        if text not in concepts["genes"]:
                            concepts["genes"].append(text)
                    elif ann_type in ("disease", "phenotype"):
                        if text not in concepts["diseases"]:
                            concepts["diseases"].append(text)
                    elif ann_type in ("chemical", "drug"):
                        if text not in concepts["chemicals"]:
                            concepts["chemicals"].append(text)
                    elif ann_type in ("mutation", "variant"):
                        if text not in concepts["mutations"]:
                            concepts["mutations"].append(text)
                    elif ann_type == "species":
                        if text not in concepts["species"]:
                            concepts["species"].append(text)

            results[pmid] = concepts

        return results

    def _regex_extract(self, paper: dict) -> dict:
        """Fallback: extract concepts using regex patterns on title + abstract."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

        genes: list[str] = []
        for match in GENE_PATTERN.finditer(text):
            gene = match.group(1)
            if gene not in genes:
                genes.append(gene)

        mutations: list[str] = []
        for match in MUTATION_PATTERN.finditer(text):
            mut = match.group(1)
            if mut not in mutations:
                mutations.append(mut)

        # Extract diseases by keyword matching
        diseases: list[str] = []
        text_lower = text.lower()
        for keyword in DISEASE_KEYWORDS:
            # Find the phrase containing the keyword
            pattern = re.compile(
                r"(?:\b\w+\s+)?" + re.escape(keyword) + r"(?:\s+\w+\b)?",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                phrase = match.group(0).strip()
                if phrase and phrase.lower() not in [d.lower() for d in diseases]:
                    diseases.append(phrase)
                if len(diseases) >= 5:
                    break
            if len(diseases) >= 5:
                break

        # Use MeSH terms if available
        mesh_terms = paper.get("mesh_terms", [])

        # Simple chemical extraction from MeSH
        chemicals: list[str] = []
        for term in mesh_terms:
            if any(kw in term.lower() for kw in ["inhibitor", "antibody", "receptor", "kinase"]):
                chemicals.append(term)

        return {
            "genes": genes[:10],
            "diseases": diseases[:10],
            "chemicals": chemicals[:10],
            "mutations": mutations[:10],
            "species": [],
        }
