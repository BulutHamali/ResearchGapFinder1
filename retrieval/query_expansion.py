import logging
import re
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Common biomedical aliases: gene symbols, disease synonyms, pathway names
BIOMEDICAL_ALIASES: dict[str, list[str]] = {
    "tp53": ["p53", "tumor protein p53", "TP53"],
    "p53": ["TP53", "tumor protein p53"],
    "brca1": ["BRCA1", "breast cancer type 1 susceptibility protein"],
    "brca2": ["BRCA2", "breast cancer type 2 susceptibility protein"],
    "egfr": ["EGFR", "epidermal growth factor receptor", "HER1", "ERBB1"],
    "her2": ["ERBB2", "HER2", "neu", "HER-2"],
    "kras": ["KRAS", "kirsten ras", "K-Ras"],
    "pten": ["PTEN", "phosphatase and tensin homolog"],
    "vegf": ["VEGF", "vascular endothelial growth factor", "VEGFA"],
    "mtor": ["mTOR", "mechanistic target of rapamycin", "FRAP1"],
    "pi3k": ["PI3K", "phosphoinositide 3-kinase", "PIK3CA"],
    "akt": ["AKT", "protein kinase B", "PKB"],
    "nfkb": ["NF-kB", "nuclear factor kappa B", "NFKB1"],
    "mapk": ["MAPK", "mitogen-activated protein kinase", "ERK"],
    "wnt": ["Wnt", "wingless-type", "WNT signaling"],
    "notch": ["Notch", "NOTCH1", "notch signaling"],
    "alzheimer": ["Alzheimer's disease", "AD", "senile dementia"],
    "parkinson": ["Parkinson's disease", "PD", "parkinsonism"],
    "cancer": ["malignancy", "neoplasm", "tumor", "carcinoma"],
    "breast cancer": ["breast carcinoma", "mammary cancer", "breast neoplasm"],
    "lung cancer": ["lung carcinoma", "pulmonary cancer", "NSCLC", "SCLC"],
    "diabetes": ["diabetes mellitus", "T2DM", "T1DM", "hyperglycemia"],
    "hypertension": ["high blood pressure", "arterial hypertension"],
    "inflammation": ["inflammatory response", "neuroinflammation", "inflammatory signaling"],
    "apoptosis": ["programmed cell death", "cell death", "apoptotic pathway"],
    "autophagy": ["autophagic flux", "autophagosome", "mitophagy"],
    "ferroptosis": ["iron-dependent cell death", "lipid peroxidation cell death"],
    "oxidative stress": ["reactive oxygen species", "ROS", "oxidative damage"],
    "metastasis": ["cancer metastasis", "tumor invasion", "tumor spread"],
    "angiogenesis": ["neovascularization", "blood vessel formation"],
    "senescence": ["cellular senescence", "replicative senescence", "SASP"],
    "epigenetics": ["epigenetic regulation", "DNA methylation", "histone modification"],
    "rna sequencing": ["RNA-seq", "transcriptomics", "gene expression profiling"],
    "crispr": ["CRISPR-Cas9", "gene editing", "genome editing"],
    "immunotherapy": ["immune checkpoint", "PD-1", "PD-L1", "CAR-T"],
}


class QueryExpander:
    """Expands a research query using MeSH terms and biomedical aliases."""

    MESH_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    MESH_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    MAX_EXPANDED_QUERIES = 4

    def __init__(self, email: str = "researcher@example.com"):
        self.email = email
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": f"ResearchGapFinder/1.0 ({email})"})

    def expand(self, query: str) -> list[str]:
        """
        Expand query using MeSH synonyms and biomedical aliases.

        Returns original query plus up to MAX_EXPANDED_QUERIES expanded variants.
        """
        results = [query]
        terms = self._split_query(query)

        expanded_sets: list[set[str]] = []
        for term in terms:
            synonyms: set[str] = set()
            # Local alias lookup (case-insensitive)
            lower_term = term.lower()
            for alias_key, alias_values in BIOMEDICAL_ALIASES.items():
                if alias_key == lower_term or lower_term in [v.lower() for v in alias_values]:
                    synonyms.update(alias_values)
                    synonyms.add(alias_key)

            # MeSH lookup (best effort)
            try:
                mesh_synonyms = self._fetch_mesh_terms(term)
                synonyms.update(mesh_synonyms)
            except Exception as e:
                logger.debug(f"MeSH fetch failed for '{term}': {e}")

            # Remove the original term from synonyms to avoid duplication
            synonyms.discard(term)
            synonyms.discard(term.lower())
            if synonyms:
                expanded_sets.append(synonyms)

        # Build expanded queries by substituting synonyms for individual terms
        for i, term in enumerate(terms):
            if i < len(expanded_sets) and expanded_sets[i]:
                # Build an OR-joined variant with synonyms
                synonym_list = list(expanded_sets[i])[:5]  # limit synonyms per term
                expanded_term = " OR ".join(f'"{s}"' if " " in s else s for s in synonym_list)
                # Replace the original term in query
                expanded_query = re.sub(
                    re.escape(term), f"({expanded_term})", query, flags=re.IGNORECASE, count=1
                )
                if expanded_query not in results:
                    results.append(expanded_query)

            if len(results) >= self.MAX_EXPANDED_QUERIES:
                break

        # Add a broad MeSH-based query if we have room
        if len(results) < self.MAX_EXPANDED_QUERIES and terms:
            broad_terms = []
            for term in terms[:3]:
                lower = term.lower()
                if lower in BIOMEDICAL_ALIASES:
                    broad_terms.append(f'"{BIOMEDICAL_ALIASES[lower][0]}"[MeSH Terms]')
                else:
                    broad_terms.append(f'"{term}"[MeSH Terms]')
            if broad_terms:
                mesh_query = " AND ".join(broad_terms)
                if mesh_query not in results:
                    results.append(mesh_query)

        logger.info(f"Query expansion: '{query}' -> {len(results)} variants")
        return results[: self.MAX_EXPANDED_QUERIES]

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=False)
    def _fetch_mesh_terms(self, term: str) -> list[str]:
        """Fetch MeSH synonyms from NCBI E-utilities."""
        params = {
            "db": "mesh",
            "term": term,
            "retmode": "json",
            "retmax": 5,
            "email": self.email,
        }
        response = self._session.get(self.MESH_ESEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        # Fetch MeSH record for the first result to get entry terms (synonyms)
        fetch_params = {
            "db": "mesh",
            "id": id_list[0],
            "retmode": "xml",
            "email": self.email,
        }
        time.sleep(0.35)  # respect rate limit
        fetch_resp = self._session.get(self.MESH_FETCH_URL, params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()

        xml_text = fetch_resp.text
        # Extract entry terms (synonyms) from MeSH XML
        entry_terms = re.findall(r"<String>(.*?)</String>", xml_text)
        # Remove duplicates and the term itself
        synonyms = [t.strip() for t in entry_terms if t.strip().lower() != term.lower()]
        return synonyms[:8]

    def _split_query(self, query: str) -> list[str]:
        """
        Split a query into individual meaningful terms.

        Handles quoted phrases and single words.
        """
        terms = []
        # Extract quoted phrases first
        quoted = re.findall(r'"([^"]+)"', query)
        terms.extend(quoted)

        # Remove quoted phrases and split remaining
        remaining = re.sub(r'"[^"]+"', "", query)
        # Remove boolean operators
        remaining = re.sub(r'\b(AND|OR|NOT)\b', " ", remaining, flags=re.IGNORECASE)
        # Split on whitespace and punctuation, keep multi-word bio terms
        words = [w.strip() for w in remaining.split() if len(w.strip()) > 2]

        # Try to detect multi-word terms (e.g., "breast cancer", "oxidative stress")
        multi_word_terms = list(BIOMEDICAL_ALIASES.keys())
        multi_word_terms = [t for t in multi_word_terms if " " in t]

        remaining_lower = remaining.lower()
        added_ranges: list[tuple[int, int]] = []
        for mwt in multi_word_terms:
            pos = remaining_lower.find(mwt)
            if pos != -1:
                # Check we haven't already added a term covering this range
                end = pos + len(mwt)
                overlap = any(s <= pos < e or s < end <= e for s, e in added_ranges)
                if not overlap:
                    terms.append(remaining[pos:end])
                    added_ranges.append((pos, end))

        # Add single words that aren't already covered by multi-word terms
        for word in words:
            word_lower = word.lower()
            already_covered = any(word_lower in t.lower() for t in terms)
            if not already_covered and len(word) > 3:
                terms.append(word)

        # De-duplicate preserving order
        seen: set[str] = set()
        unique_terms = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        return unique_terms[:6]  # limit to 6 terms
