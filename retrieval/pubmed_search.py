import logging
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from processing.metadata_parser import MetadataParser
from utils.rate_limit import pubmed_limiter
from utils.caching import get_cache, Cache

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 100


class PubMedSearcher:
    """Search PubMed via NCBI E-utilities and return structured paper dicts."""

    def __init__(self, email: str = "researcher@example.com", api_key: Optional[str] = None, cache: Optional[Cache] = None):
        self.email = email
        self.api_key = api_key
        self.parser = MetadataParser()
        self.cache = cache or get_cache()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": f"ResearchGapFinder/1.0 ({email})"})
        # With API key: 10 req/s, without: 3 req/s
        self._rate_per_sec = 10 if api_key else 3
        self._min_sleep = 1.0 / self._rate_per_sec

    def search(
        self,
        query: str,
        max_papers: int = 2000,
        year_range: tuple[int, int] = (2015, 2025),
        article_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search PubMed and return a list of paper dicts.

        Applies year_range filter and optionally article_type filter.
        De-duplicates by PMID.
        """
        if article_types is None:
            article_types = ["research", "review"]

        cache_key = Cache.make_key("pubmed_search", query, max_papers, year_range, tuple(sorted(article_types)))
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"PubMed cache hit for query: '{query[:60]}'")
            return cached

        # Build date filter
        min_year, max_year = year_range
        date_filter = f"{min_year}/01/01:{max_year}/12/31[dp]"

        # Build article type filter
        type_filter = self._build_type_filter(article_types)
        full_query = f"({query}) AND {date_filter}"
        if type_filter:
            full_query += f" AND {type_filter}"

        logger.info(f"PubMed search: '{full_query[:120]}' (max={max_papers})")

        # Step 1: esearch to get PMIDs
        pmids = self._esearch(full_query, max_papers)
        if not pmids:
            logger.warning("No PMIDs returned from PubMed esearch")
            return []

        logger.info(f"PubMed esearch returned {len(pmids)} PMIDs")

        # Step 2: efetch in batches
        papers = self._efetch_batches(pmids)

        # Step 3: Deduplicate by PMID
        seen: set[str] = set()
        unique_papers = []
        for p in papers:
            if p.get("pmid") and p["pmid"] not in seen:
                seen.add(p["pmid"])
                unique_papers.append(p)

        # Step 4: Filter by year (efetch may include papers outside range if date field varies)
        filtered = [
            p for p in unique_papers
            if p.get("year", 0) == 0 or (min_year <= p.get("year", 0) <= max_year)
        ]

        logger.info(f"PubMed: {len(filtered)} papers after dedup and year filter")
        self.cache.set(cache_key, filtered, ttl=86400)
        return filtered

    def _build_type_filter(self, article_types: list[str]) -> str:
        """Map friendly article type names to PubMed publication type filters."""
        type_map = {
            "research": "Journal Article[pt]",
            "review": "Review[pt]",
            "systematic_review": "Systematic Review[pt]",
            "meta_analysis": "Meta-Analysis[pt]",
            "clinical_trial": "Clinical Trial[pt]",
            "case_report": "Case Reports[pt]",
        }
        filters = []
        for at in article_types:
            mapped = type_map.get(at.lower())
            if mapped:
                filters.append(mapped)
        if not filters:
            return ""
        return "(" + " OR ".join(filters) + ")"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=True)
    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Execute esearch and return a list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10000),
            "retmode": "json",
            "email": self.email,
            "usehistory": "y",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        pubmed_limiter.acquire_sync()
        resp = self._session.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch_batches(self, pmids: list[str]) -> list[dict]:
        """Fetch papers in batches of BATCH_SIZE and parse XML."""
        all_papers = []
        for i in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[i: i + BATCH_SIZE]
            logger.debug(f"Fetching PubMed batch {i // BATCH_SIZE + 1}: {len(batch)} PMIDs")
            papers = self._efetch_batch(batch)
            all_papers.extend(papers)
            # Throttle between batches
            time.sleep(self._min_sleep)
        return all_papers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=True)
    def _efetch_batch(self, pmids: list[str]) -> list[dict]:
        """Fetch a single batch of PMIDs via efetch and parse the XML."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        pubmed_limiter.acquire_sync()
        resp = self._session.get(EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        return self.parser.parse_pubmed_xml(resp.text)
