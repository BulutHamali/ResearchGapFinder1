"""Integration tests for the full analysis pipeline."""
import json
from unittest.mock import MagicMock, patch, Mock, AsyncMock
import numpy as np
import pytest

from schemas.output_schema import (
    AnalysisRequest,
    AnalysisResponse,
    ResearchGap,
    Hypothesis,
    SuggestedExperiment,
    ClusterSummary,
    EvidenceSnippet,
)


# Sample minimal papers for testing
SAMPLE_PAPERS = [
    {
        "pmid": f"1000000{i}",
        "title": f"TP53 mutations in cancer study {i}",
        "abstract": (
            f"This study investigates TP53 mutations in cancer cells. "
            f"We found that TP53 regulates apoptosis and cell cycle arrest. "
            f"The relationship between TP53 and MDM2 remains poorly understood. "
            f"Further investigation is needed to elucidate the exact mechanism. "
            f"Our results suggest oxidative stress plays a key role in tumor suppression "
            f"through the p53-MDM2 axis in various cancer cell lines study {i}."
        ),
        "year": 2020 + (i % 5),
        "journal": "Nature Cancer",
        "authors": ["Smith J", "Jones A"],
        "mesh_terms": ["TP53", "Cancer", "Apoptosis", "Cell Cycle"],
        "article_type": ["Journal Article"],
        "doi": f"10.1000/test.{i}",
        "pmc_id": "",
        "concepts": {
            "genes": ["TP53", "MDM2"],
            "diseases": ["cancer"],
            "chemicals": [],
            "mutations": [],
        },
    }
    for i in range(20)
]


class TestSchemaValidation:
    """Test that all output schemas work correctly."""

    def test_evidence_snippet(self):
        snippet = EvidenceSnippet(pmid="12345678", text="Test evidence text")
        assert snippet.pmid == "12345678"
        assert snippet.text == "Test evidence text"

    def test_research_gap(self):
        gap = ResearchGap(
            gap="The mechanism of TP53 in ferroptosis is unknown",
            type="explicit_gap",
            evidence_snippets=[EvidenceSnippet(pmid="12345", text="remains unknown")],
            reason_underexplored="Technical challenges in measuring ferroptosis",
            uncertainty="medium",
            competing_explanations=["Alternative pathway may explain this"],
        )
        assert gap.type == "explicit_gap"
        assert len(gap.evidence_snippets) == 1

    def test_hypothesis(self):
        hyp = Hypothesis(
            hypothesis="TP53 regulates ferroptosis via GPX4 in cancer cells",
            novelty_score=0.8,
            support_score=0.6,
            feasibility_score=0.9,
            impact_score=0.7,
            reasoning_summary="Based on cluster analysis showing TP53 and GPX4 never co-studied",
            already_established=False,
        )
        assert 0.0 <= hyp.novelty_score <= 1.0
        assert 0.0 <= hyp.support_score <= 1.0
        assert 0.0 <= hyp.feasibility_score <= 1.0
        assert 0.0 <= hyp.impact_score <= 1.0

    def test_suggested_experiment(self):
        exp = SuggestedExperiment(
            experiment="CRISPR knockout of TP53 in MCF7 cells",
            modality="genetic_perturbation",
            assays=["CRISPR-Cas9 knockout", "Western blot", "flow cytometry"],
            complexity="medium",
            required_models=["MCF7 cells", "MDA-MB-231 cells"],
        )
        assert exp.complexity in ("low", "medium", "high")

    def test_cluster_summary(self):
        cs = ClusterSummary(
            cluster_id=0,
            label="TP53 Cancer",
            paper_count=50,
            top_terms=["tp53", "cancer", "apoptosis"],
            silhouette_score=0.45,
        )
        assert cs.cluster_id == 0
        assert cs.paper_count == 50

    def test_analysis_request_defaults(self):
        req = AnalysisRequest(query="TP53 cancer")
        assert req.max_papers == 2000
        assert req.year_range == (2015, 2025)
        assert req.llm_preset == "balanced"

    def test_analysis_response(self):
        resp = AnalysisResponse(
            query="TP53 cancer",
            papers_retrieved=100,
            clusters=[],
            research_gaps=[],
            hypotheses=[],
            suggested_experiments=[],
        )
        assert resp.query == "TP53 cancer"
        assert resp.papers_retrieved == 100


class TestPaperCleaner:
    """Tests for PaperCleaner functionality."""

    def setup_method(self):
        from processing.paper_cleaner import PaperCleaner
        self.cleaner = PaperCleaner()

    def test_clean_removes_html_tags(self):
        paper = {
            "pmid": "1",
            "title": "<b>Test Title</b>",
            "abstract": "<p>This is a test abstract that is long enough to pass the filter. " * 5 + "</p>",
        }
        cleaned = self.cleaner.clean(paper)
        assert "<b>" not in cleaned["title"]
        assert "<p>" not in cleaned["abstract"]

    def test_clean_filters_short_abstract(self):
        paper = {
            "pmid": "1",
            "title": "Test Title",
            "abstract": "Too short.",  # Less than 50 words
        }
        result = self.cleaner.clean(paper)
        assert result is None

    def test_clean_passes_long_abstract(self):
        paper = {
            "pmid": "1",
            "title": "Test Title",
            "abstract": "This is a long enough abstract. " * 10,  # > 50 words
        }
        result = self.cleaner.clean(paper)
        assert result is not None
        assert result["pmid"] == "1"

    def test_clean_removes_copyright(self):
        paper = {
            "pmid": "1",
            "title": "Test Title",
            "abstract": (
                "This study examined cancer cells and their response to treatment. "
                "We found significant results in apoptosis regulation. "
                "The pathway analysis revealed important mechanistic insights. "
                "Our data supports the hypothesis of oxidative stress involvement. "
                "Further validation is needed in clinical settings. "
                "Copyright © 2023 Elsevier. All rights reserved."
            ),
        }
        result = self.cleaner.clean(paper)
        if result:  # Passes length filter
            assert "Copyright" not in result["abstract"] or "©" not in result["abstract"]

    def test_clean_batch_filters_correctly(self):
        papers = [
            {"pmid": "1", "title": "Good Paper", "abstract": "Long abstract. " * 10},
            {"pmid": "2", "title": "Bad Paper", "abstract": "Short."},
        ]
        cleaned = self.cleaner.clean_batch(papers)
        assert len(cleaned) == 1
        assert cleaned[0]["pmid"] == "1"


class TestEmbedder:
    """Tests for Embedder class."""

    def test_embed_returns_correct_shape(self):
        from embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Test text one", "Test text two", "Test text three"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == embedder.dim

    def test_embed_returns_float32(self):
        from embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Test text"]
        embeddings = embedder.embed(texts)
        assert embeddings.dtype == np.float32

    def test_embed_normalized(self):
        from embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        texts = ["Test text for normalization"]
        embeddings = embedder.embed(texts)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_single(self):
        from embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embedding = embedder.embed_single("Single text")
        assert embedding.ndim == 1
        assert len(embedding) == embedder.dim

    def test_embed_empty_list(self):
        from embedding.embedder import Embedder
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embeddings = embedder.embed([])
        assert embeddings.shape == (0, embedder.dim)

    def test_papers_to_texts(self):
        from embedding.embedder import Embedder
        papers = [
            {"title": "Title 1", "abstract": "Abstract 1"},
            {"title": "Title 2", "abstract": ""},
            {"title": "", "abstract": "Abstract 3"},
        ]
        texts = Embedder.papers_to_texts(papers)
        assert len(texts) == 3
        assert "Title 1. Abstract 1" in texts[0]


class TestGapScorer:
    """Tests for statistical gap detection."""

    def setup_method(self):
        from gap_detection.gap_scorer import GapScorer
        self.scorer = GapScorer()

    def test_detect_explicit_gaps_from_text(self):
        """Test that explicit gap phrases are detected."""
        papers = [
            {
                "pmid": "1",
                "abstract": (
                    "The role of TP53 in ferroptosis remains unknown and poorly understood. "
                    "Future studies should investigate this relationship in cancer cells."
                ),
                "concepts": {},
            }
        ]
        gaps = self.scorer._detect_explicit_gaps(papers)
        assert len(gaps) > 0
        assert all(g.get("type") in ("explicit_gap", "contradictory_gap") for g in gaps)

    def test_detect_contradictory_gap(self):
        """Test that contradictory findings are flagged."""
        papers = [
            {
                "pmid": "2",
                "abstract": (
                    "Some studies show that oxidative stress promotes cancer, "
                    "while others found it inhibits tumor growth. "
                    "These conflicting results remain controversial and debated in the field."
                ),
                "concepts": {},
            }
        ]
        gaps = self.scorer._detect_explicit_gaps(papers)
        assert any(g.get("type") == "contradictory_gap" for g in gaps)

    def test_score_returns_list(self):
        """Test that score returns a list."""
        clusters = {0: SAMPLE_PAPERS[:10], 1: SAMPLE_PAPERS[10:20]}
        cluster_summaries = [
            {"cluster_id": 0, "label": "TP53 Cancer", "top_terms": ["tp53", "cancer"]},
            {"cluster_id": 1, "label": "Apoptosis", "top_terms": ["apoptosis", "cell death"]},
        ]
        result = self.scorer.score(clusters, cluster_summaries)
        assert isinstance(result, list)

    def test_score_normalized(self):
        """Test that statistical scores are normalized to [0, 1]."""
        clusters = {0: SAMPLE_PAPERS[:10], 1: SAMPLE_PAPERS[10:20]}
        cluster_summaries = [
            {"cluster_id": 0, "label": "TP53 Cancer", "top_terms": ["tp53", "cancer"]},
            {"cluster_id": 1, "label": "Apoptosis", "top_terms": ["apoptosis", "cell death"]},
        ]
        result = self.scorer.score(clusters, cluster_summaries)
        for gap in result:
            score = gap.get("statistical_score", 0.0)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


class TestVectorStore:
    """Tests for FAISS vector store."""

    def test_add_and_search(self):
        from vector_db.vector_store import VectorStore
        store = VectorStore(dim=32)
        embeddings = np.random.randn(10, 32).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        metadata = [{"pmid": str(i)} for i in range(10)]

        store.add(embeddings, metadata)
        assert store.size == 10

        query = embeddings[0]
        results = store.search(query, k=3)
        assert len(results) == 3
        # Best match should be the query itself (or very similar)
        assert results[0][1] > 0.9  # High similarity score

    def test_size_property(self):
        from vector_db.vector_store import VectorStore
        store = VectorStore(dim=16)
        assert store.size == 0

        vecs = np.random.randn(5, 16).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(vecs, [{"id": i} for i in range(5)])
        assert store.size == 5

    def test_save_and_load(self, tmp_path):
        from vector_db.vector_store import VectorStore
        store = VectorStore(dim=16)
        vecs = np.random.randn(5, 16).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        meta = [{"pmid": str(i)} for i in range(5)]
        store.add(vecs, meta)

        path = str(tmp_path / "test_index")
        store.save(path)

        store2 = VectorStore(dim=16)
        store2.load(path)
        assert store2.size == 5
        assert store2.get_all_metadata() == meta


class TestMiniPipeline:
    """Integration test: small end-to-end pipeline without external APIs."""

    def test_embedding_to_clustering_pipeline(self):
        """Test the embedding -> vector store -> clustering -> evaluation sub-pipeline."""
        from embedding.embedder import Embedder
        from vector_db.vector_store import VectorStore
        from clustering.cluster_papers import PaperClusterer
        from clustering.cluster_evaluator import ClusterEvaluator

        # Use minimal model for speed
        embedder = Embedder(model_name="all-MiniLM-L6-v2")

        # Create 30 papers spread across 3 clear topics
        papers = []
        texts = []
        for topic_id in range(3):
            for j in range(10):
                title = f"Topic {topic_id} paper {j}: {'Cancer' if topic_id==0 else 'Neuroscience' if topic_id==1 else 'Immunology'}"
                abstract = (
                    f"This paper investigates {'cancer cell apoptosis TP53 mutation tumor suppressor MDM2 pathway ' if topic_id==0 else 'neuron synapse brain alzheimer BDNF neurodegeneration dopamine receptor ' if topic_id==1 else 'immune cell T-cell cytokine IL-6 inflammation autoimmune antibody '} "
                    * 5
                )
                papers.append({
                    "pmid": f"tp{topic_id}_{j}",
                    "title": title,
                    "abstract": abstract,
                    "year": 2020,
                    "mesh_terms": [f"MeSH_T{topic_id}"],
                    "concepts": {"genes": [], "diseases": [], "chemicals": [], "mutations": []},
                })
                texts.append(f"{title}. {abstract}")

        embeddings = embedder.embed(texts)
        assert embeddings.shape == (30, embedder.dim)

        vector_store = VectorStore(dim=embedder.dim)
        vector_store.add(embeddings, papers)
        assert vector_store.size == 30

        clusterer = PaperClusterer()
        labels = clusterer.cluster(embeddings, min_cluster_size=3)
        assert len(labels) == 30

        cluster_groups = clusterer.get_cluster_papers(labels, papers)
        assert len(cluster_groups) >= 1  # At least one cluster found

        evaluator = ClusterEvaluator()
        summaries = evaluator.evaluate(embeddings, labels, papers)
        assert isinstance(summaries, list)
        for s in summaries:
            assert "cluster_id" in s
            assert "label" in s
            assert "paper_count" in s
            assert "silhouette_score" in s

    def test_gap_scorer_on_sample_papers(self):
        """Test gap scorer with sample papers."""
        from gap_detection.gap_scorer import GapScorer

        scorer = GapScorer()
        clusters = {
            0: SAMPLE_PAPERS[:10],
            1: SAMPLE_PAPERS[10:20],
        }
        summaries = [
            {"cluster_id": 0, "label": "TP53", "top_terms": ["tp53", "cancer", "apoptosis"]},
            {"cluster_id": 1, "label": "MDM2", "top_terms": ["mdm2", "cancer", "cell cycle"]},
        ]
        gaps = scorer.score(clusters, summaries)
        assert isinstance(gaps, list)
        # Should detect explicit gaps from "remains poorly understood" in sample abstracts
        assert len(gaps) > 0
