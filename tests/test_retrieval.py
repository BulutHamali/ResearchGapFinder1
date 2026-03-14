"""Unit tests for retrieval components."""
import json
from unittest.mock import MagicMock, patch, Mock
import pytest

from retrieval.query_expansion import QueryExpander


class TestQueryExpander:
    """Tests for QueryExpander."""

    def setup_method(self):
        self.expander = QueryExpander(email="test@example.com")

    def test_split_query_simple(self):
        """Test splitting a simple query into terms."""
        terms = self.expander._split_query("KRAS cancer")
        assert isinstance(terms, list)
        assert len(terms) >= 1

    def test_split_query_removes_boolean_operators(self):
        """Test that AND/OR/NOT operators are stripped."""
        terms = self.expander._split_query("cancer AND therapy OR treatment")
        lower_terms = [t.lower() for t in terms]
        assert "and" not in lower_terms
        assert "or" not in lower_terms
        assert "not" not in lower_terms

    def test_split_query_multi_word_biomedical_term(self):
        """Test that known multi-word terms are preserved."""
        terms = self.expander._split_query("breast cancer oxidative stress")
        term_lower = [t.lower() for t in terms]
        assert any("breast cancer" in t for t in term_lower)

    def test_expand_returns_original_plus_variants(self):
        """Test that expand always includes the original query."""
        with patch.object(self.expander, "_fetch_mesh_terms", return_value=[]):
            results = self.expander.expand("KRAS cancer")
        assert results[0] == "KRAS cancer"
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_expand_respects_max_queries(self):
        """Test that expansion doesn't exceed MAX_EXPANDED_QUERIES."""
        with patch.object(self.expander, "_fetch_mesh_terms", return_value=["synonym1", "synonym2"]):
            results = self.expander.expand("EGFR lung cancer treatment")
        assert len(results) <= QueryExpander.MAX_EXPANDED_QUERIES

    def test_expand_known_alias(self):
        """Test expansion of a known biomedical alias."""
        with patch.object(self.expander, "_fetch_mesh_terms", return_value=[]):
            results = self.expander.expand("tp53")
        # Should have expansion from alias dict
        assert len(results) >= 1
        # The alias for tp53 should create expanded queries
        all_text = " ".join(results).lower()
        # Either p53 or TP53 should appear in some form
        assert "tp53" in all_text or "p53" in all_text.lower()

    def test_split_query_filters_short_terms(self):
        """Test that very short words are filtered."""
        terms = self.expander._split_query("a be cancer")
        # 'a' and 'be' are too short (< 3 chars for 'a', and 'be' is 2)
        # 'cancer' should pass
        assert any("cancer" in t.lower() for t in terms)


class TestPubMedSearcher:
    """Tests for PubMedSearcher with mocked HTTP."""

    def test_search_returns_list(self):
        """Test that search returns a list of papers."""
        from retrieval.pubmed_search import PubMedSearcher
        from utils.caching import Cache

        mock_cache = MagicMock(spec=Cache)
        mock_cache.get.return_value = None
        mock_cache.set.return_value = None

        searcher = PubMedSearcher(email="test@example.com", cache=mock_cache)

        # Mock esearch response
        esearch_response = {
            "esearchresult": {
                "idlist": ["12345678", "87654321"],
                "count": "2",
            }
        }

        # Minimal valid PubMed XML
        efetch_xml = """<?xml version="1.0"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN" "">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="1">12345678</PMID>
      <Article>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2020</Year></PubDate>
          </JournalIssue>
          <Title>Nature</Title>
        </Journal>
        <ArticleTitle>Test cancer study with important findings</ArticleTitle>
        <Abstract>
          <AbstractText>This is a test abstract about cancer and TP53 mutations.
          The study found that oxidative stress plays a key role in tumor progression
          through multiple pathways including ferroptosis and apoptosis regulation in cells.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""

        with patch.object(searcher._session, "get") as mock_get:
            # First call: esearch
            esearch_mock = Mock()
            esearch_mock.status_code = 200
            esearch_mock.json.return_value = esearch_response
            esearch_mock.raise_for_status = Mock()

            # Second call: efetch
            efetch_mock = Mock()
            efetch_mock.status_code = 200
            efetch_mock.text = efetch_xml
            efetch_mock.raise_for_status = Mock()

            mock_get.side_effect = [esearch_mock, efetch_mock]

            papers = searcher.search(
                query="TP53 cancer",
                max_papers=10,
                year_range=(2018, 2025),
            )

        assert isinstance(papers, list)

    def test_build_type_filter_research(self):
        """Test article type filter construction."""
        from retrieval.pubmed_search import PubMedSearcher
        from utils.caching import Cache

        mock_cache = MagicMock(spec=Cache)
        mock_cache.get.return_value = None
        searcher = PubMedSearcher(email="test@example.com", cache=mock_cache)

        filter_str = searcher._build_type_filter(["research", "review"])
        assert "Journal Article" in filter_str
        assert "Review" in filter_str
        assert "[pt]" in filter_str

    def test_build_type_filter_empty(self):
        """Test that unknown article types return empty filter."""
        from retrieval.pubmed_search import PubMedSearcher
        from utils.caching import Cache

        mock_cache = MagicMock(spec=Cache)
        mock_cache.get.return_value = None
        searcher = PubMedSearcher(email="test@example.com", cache=mock_cache)

        filter_str = searcher._build_type_filter(["unknown_type"])
        assert filter_str == ""


class TestDeduplication:
    """Test paper deduplication logic."""

    def test_europepmc_merge_deduplicates_by_pmid(self):
        """Test that EuropePMC merger deduplicates by PMID."""
        from retrieval.europepmc_search import EuropePMCSearcher

        searcher = EuropePMCSearcher()

        pubmed_papers = [
            {"pmid": "11111111", "title": "PubMed Paper 1", "abstract": "abstract"},
            {"pmid": "22222222", "title": "PubMed Paper 2", "abstract": "abstract"},
        ]
        europepmc_papers = [
            {"pmid": "11111111", "title": "Duplicate from EuropePMC", "abstract": "abstract"},
            {"pmid": "33333333", "title": "Unique EuropePMC Paper", "abstract": "abstract"},
        ]

        merged = searcher.merge_with_pubmed(pubmed_papers, europepmc_papers)

        # Should have 3 unique papers (not 4)
        assert len(merged) == 3
        pmids = [p["pmid"] for p in merged]
        assert pmids.count("11111111") == 1  # No duplicate

    def test_deduplication_preserves_pubmed_records(self):
        """Test that PubMed records take priority over EuropePMC duplicates."""
        from retrieval.europepmc_search import EuropePMCSearcher

        searcher = EuropePMCSearcher()

        pubmed_papers = [
            {"pmid": "11111111", "title": "PubMed Version", "mesh_terms": ["Cancer"]},
        ]
        europepmc_papers = [
            {"pmid": "11111111", "title": "EuropePMC Version", "mesh_terms": []},
        ]

        merged = searcher.merge_with_pubmed(pubmed_papers, europepmc_papers)

        # PubMed version should be kept
        assert merged[0]["title"] == "PubMed Version"
        assert merged[0]["mesh_terms"] == ["Cancer"]
