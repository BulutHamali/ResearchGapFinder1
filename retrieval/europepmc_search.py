import logging
import time
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from processing.metadata_parser import MetadataParser
from utils.caching import get_cache, Cache

logger = logging.getLogger(__name__)

EUROPEPMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PAGE_SIZE = 100


class EuropePMCSearcher:
    """Search EuropePMC REST API and return structured paper dicts."""

    def __init__(self, cache: Optional[Cache] = None):
        self.parser = MetadataParser()
        self.cache = cache or get_cache()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ResearchGapFinder/1.0"})

    def search(
        self,
        query: str,
        max_papers: int = 2000,
        year_range: tuple[int, int] = (2015, 2025),
    ) -> list[dict]:
        """
        Search EuropePMC and return a list of paper dicts.

        Paginates using cursorMark. De-duplicates by PMID.
        """
        cache_key = Cache.make_key("europepmc_search", query, max_papers, year_range)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info(f"EuropePMC cache hit for query: '{query[:60]}'")
            return cached

        min_year, max_year = year_range
        # EuropePMC date filter syntax
        full_query = f"({query}) AND (PUB_YEAR:[{min_year} TO {max_year}])"

        logger.info(f"EuropePMC search: '{full_query[:120]}' (max={max_papers})")

        all_papers: list[dict] = []
        cursor_mark = "*"
        seen_pmids: set[str] = set()

        while len(all_papers) < max_papers:
            batch, next_cursor = self._fetch_page(full_query, cursor_mark, PAGE_SIZE)
            if not batch:
                break

            for paper in batch:
                pmid = paper.get("pmid", "")
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    all_papers.append(paper)
                elif not pmid:
                    # Include papers without PMID (preprints, etc.)
                    all_papers.append(paper)

            if not next_cursor or next_cursor == cursor_mark:
                break
            cursor_mark = next_cursor
            time.sleep(0.5)  # Be polite to EuropePMC

        result = all_papers[:max_papers]
        logger.info(f"EuropePMC: {len(result)} papers retrieved")
        self.cache.set(cache_key, result, ttl=86400)
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15), reraise=True)
    def _fetch_page(self, query: str, cursor_mark: str, page_size: int) -> tuple[list[dict], str]:
        """Fetch a single page of results from EuropePMC."""
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "resultType": "core",
            "sort": "CITED desc",
        }
        resp = self._session.get(EUROPEPMC_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        papers = self.parser.parse_europepmc_json(data)
        next_cursor = data.get("nextCursorMark", "")
        return papers, next_cursor

    def merge_with_pubmed(self, pubmed_papers: list[dict], europepmc_papers: list[dict]) -> list[dict]:
        """
        Merge EuropePMC results with PubMed results, de-duplicating by PMID.

        PubMed results take priority (they have MeSH terms).
        """
        pubmed_pmids: set[str] = {p["pmid"] for p in pubmed_papers if p.get("pmid")}
        merged = list(pubmed_papers)

        for paper in europepmc_papers:
            pmid = paper.get("pmid", "")
            if pmid and pmid in pubmed_pmids:
                continue  # Already have this from PubMed
            merged.append(paper)

        logger.info(
            f"Merged {len(pubmed_papers)} PubMed + {len(europepmc_papers)} EuropePMC = {len(merged)} unique papers"
        )
        return merged
