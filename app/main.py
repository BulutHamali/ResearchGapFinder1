import asyncio
import json
import logging
import queue as _queue
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings, get_preset
from schemas.output_schema import (
    AnalysisRequest,
    AnalysisResponse,
    ClusterSummary,
)
from retrieval.query_expansion import QueryExpander
from retrieval.pubmed_search import PubMedSearcher
from retrieval.europepmc_search import EuropePMCSearcher
from processing.paper_cleaner import PaperCleaner
from processing.concept_extractor import ConceptExtractor
from embedding.embedder import Embedder
from vector_db.vector_store import VectorStore
from clustering.cluster_papers import PaperClusterer
from clustering.cluster_evaluator import ClusterEvaluator
from gap_detection.gap_scorer import GapScorer
from gap_detection.gap_agent import GapAgent
from hypothesis.novelty_checker import NoveltyChecker
from hypothesis.hypothesis_scorer import HypothesisScorer
from hypothesis.hypothesis_agent import HypothesisAgent
from experiments.experiment_generator import ExperimentGenerator
from llm.reasoner import LLMReasoner
from utils.caching import get_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level shared components (initialized at startup)
_components: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown shared resources."""
    logger.info("Starting ResearchGapFinder application...")

    # Initialize cache
    cache = get_cache(settings.CACHE_DIR)
    _components["cache"] = cache

    # Initialize shared components
    _components["paper_cleaner"] = PaperCleaner()
    _components["concept_extractor"] = ConceptExtractor()
    _components["paper_clusterer"] = PaperClusterer()
    _components["gap_scorer"] = GapScorer()

    logger.info("ResearchGapFinder initialized successfully")
    yield

    # Teardown
    logger.info("Shutting down ResearchGapFinder...")
    if "cache" in _components:
        _components["cache"].close()


app = FastAPI(
    title="ResearchGapFinder",
    description="AI-powered research gap detection and hypothesis generation from biomedical literature",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "cache_dir": settings.CACHE_DIR,
        "default_preset": settings.LLM_PRESET,
    }


def _run_analysis_sync(request: AnalysisRequest) -> AnalysisResponse:
    """Synchronous analysis pipeline, callable from both endpoints."""
    start_time = time.time()
    preset = get_preset(request.llm_preset)
    logger.info(
        f"Analysis started: query='{request.query[:60]}', "
        f"max_papers={request.max_papers}, preset={request.llm_preset}"
    )

    # Override max_papers from preset if not specified differently
    max_papers = min(request.max_papers, preset["max_papers_default"])
    min_cluster_size = preset["min_cluster_size"]

    # 1. Query Expansion
    t0 = time.time()
    expander = QueryExpander(email=settings.PUBMED_EMAIL)
    expanded_queries = expander.expand(request.query)
    logger.info(f"Query expansion: {len(expanded_queries)} variants in {time.time()-t0:.1f}s")

    # 2. PubMed Search
    t0 = time.time()
    cache = _components.get("cache")
    pubmed_searcher = PubMedSearcher(
        email=settings.PUBMED_EMAIL,
        cache=cache,
    )
    all_papers: list[dict] = []
    seen_pmids: set[str] = set()

    for q in expanded_queries:
        papers = pubmed_searcher.search(
            query=q,
            max_papers=max_papers // len(expanded_queries) + 100,
            year_range=request.year_range,
            article_types=request.article_types,
        )
        for p in papers:
            pmid = p.get("pmid", "")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_papers.append(p)
            elif not pmid:
                all_papers.append(p)

        if len(all_papers) >= max_papers:
            break

    all_papers = all_papers[:max_papers]
    logger.info(f"PubMed retrieval: {len(all_papers)} papers in {time.time()-t0:.1f}s")

    # Optionally add EuropePMC results
    if len(all_papers) < max_papers // 2:
        try:
            t0 = time.time()
            epmc_searcher = EuropePMCSearcher(cache=cache)
            epmc_papers = epmc_searcher.search(
                query=request.query,
                max_papers=max_papers - len(all_papers),
                year_range=request.year_range,
            )
            all_papers = epmc_searcher.merge_with_pubmed(all_papers, epmc_papers)
            all_papers = all_papers[:max_papers]
            logger.info(f"EuropePMC added papers, total: {len(all_papers)} in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.warning(f"EuropePMC search failed (non-fatal): {e}")

    if not all_papers:
        raise HTTPException(
            status_code=404,
            detail=f"No papers found for query: '{request.query}'. Try different search terms.",
        )

    # 3. Clean papers
    t0 = time.time()
    cleaner: PaperCleaner = _components["paper_cleaner"]
    cleaned_papers = cleaner.clean_batch(all_papers)
    logger.info(f"Paper cleaning: {len(cleaned_papers)} papers in {time.time()-t0:.1f}s")

    if not cleaned_papers:
        raise HTTPException(
            status_code=422,
            detail="All retrieved papers were filtered out (abstracts too short or missing).",
        )

    # 4. Extract concepts
    t0 = time.time()
    concept_extractor: ConceptExtractor = _components["concept_extractor"]
    papers_with_concepts = concept_extractor.extract(cleaned_papers)
    logger.info(f"Concept extraction: complete in {time.time()-t0:.1f}s")

    # 5. Generate embeddings
    t0 = time.time()
    embedder = Embedder(model_name=preset["embedding_model"])
    texts = embedder.papers_to_texts(papers_with_concepts)
    embeddings = embedder.embed(texts)
    logger.info(f"Embedding: {len(embeddings)} vectors, dim={embedder.dim} in {time.time()-t0:.1f}s")

    # 6. Build vector store
    t0 = time.time()
    vector_store = VectorStore(dim=embedder.dim, index_type="flat")
    vector_store.add(embeddings, papers_with_concepts)
    logger.info(f"Vector store: {vector_store.size} vectors in {time.time()-t0:.1f}s")

    # 7. Cluster papers
    t0 = time.time()
    clusterer: PaperClusterer = _components["paper_clusterer"]
    labels = clusterer.cluster(embeddings, min_cluster_size=min_cluster_size, method="hdbscan")
    cluster_paper_groups = clusterer.get_cluster_papers(labels, papers_with_concepts)
    logger.info(
        f"Clustering: {len(cluster_paper_groups)} clusters in {time.time()-t0:.1f}s"
    )

    # 8. Evaluate clusters
    t0 = time.time()
    cluster_evaluator = ClusterEvaluator()
    cluster_summaries_raw = cluster_evaluator.evaluate(embeddings, labels, papers_with_concepts)
    logger.info(f"Cluster evaluation: complete in {time.time()-t0:.1f}s")

    cluster_summaries_schema = [
        ClusterSummary(
            cluster_id=cs["cluster_id"],
            label=cs["label"],
            paper_count=cs["paper_count"],
            top_terms=cs["top_terms"],
            silhouette_score=cs["silhouette_score"],
        )
        for cs in cluster_summaries_raw
    ]

    # 9. LLM setup
    from app.config import Settings
    preset_settings = settings
    reasoner = LLMReasoner(preset_settings)
    reasoner.model = preset["llm_model"]
    reasoner.default_max_tokens = preset["max_tokens"]
    reasoner.default_temperature = preset["temperature"]

    # 10. Score gaps statistically
    t0 = time.time()
    gap_scorer: GapScorer = _components["gap_scorer"]
    candidate_gaps = gap_scorer.score(cluster_paper_groups, cluster_summaries_raw)
    logger.info(
        f"Statistical gap scoring: {len(candidate_gaps)} candidates in {time.time()-t0:.1f}s"
    )

    # 11. LLM gap detection
    t0 = time.time()
    gap_agent = GapAgent(reasoner=reasoner)
    research_gaps = gap_agent.detect(
        query=request.query,
        cluster_summaries=cluster_summaries_raw,
        candidate_gaps=candidate_gaps,
    )
    logger.info(f"Gap detection: {len(research_gaps)} gaps in {time.time()-t0:.1f}s")

    # 12. Generate hypotheses
    t0 = time.time()
    novelty_checker = NoveltyChecker(pubmed_searcher=pubmed_searcher, reasoner=reasoner)
    hypothesis_scorer = HypothesisScorer()
    hypothesis_agent = HypothesisAgent(
        reasoner=reasoner,
        novelty_checker=novelty_checker,
        scorer=hypothesis_scorer,
    )
    hypotheses = hypothesis_agent.generate(
        query=request.query,
        gaps=research_gaps,
        cluster_summaries=cluster_summaries_raw,
    )
    logger.info(f"Hypothesis generation: {len(hypotheses)} hypotheses in {time.time()-t0:.1f}s")

    # 13. Generate experiments
    t0 = time.time()
    experiment_generator = ExperimentGenerator(reasoner=reasoner)
    experiments = experiment_generator.generate(
        hypotheses=hypotheses,
        gaps=research_gaps,
    )
    logger.info(f"Experiment generation: {len(experiments)} experiments in {time.time()-t0:.1f}s")

    total_time = time.time() - start_time
    logger.info(
        f"Analysis complete: query='{request.query[:60]}', "
        f"{len(research_gaps)} gaps, {len(hypotheses)} hypotheses, "
        f"{len(experiments)} experiments, total={total_time:.1f}s"
    )

    return AnalysisResponse(
        query=request.query,
        papers_retrieved=len(papers_with_concepts),
        clusters=cluster_summaries_schema,
        research_gaps=research_gaps,
        hypotheses=hypotheses,
        suggested_experiments=experiments,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Full research gap analysis pipeline (blocking JSON response)."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _run_analysis_sync, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during analysis: {str(e)}",
        )


@app.post("/analyze-stream")
async def analyze_stream(request: AnalysisRequest):
    """
    Same pipeline as /analyze but streams log lines as NDJSON before the final result.

    Each line is a JSON object with one of these shapes:
      {"type": "log",    "msg": "<timestamp> [LEVEL] message"}
      {"type": "result", "data": { ...AnalysisResponse... }}
      {"type": "error",  "msg": "<error detail>"}
    """
    log_queue: _queue.Queue = _queue.Queue()
    result_holder: dict = {}

    def _worker():
        # Register a logging handler scoped to this thread only
        thread_id = threading.current_thread().ident

        class _ThreadHandler(logging.Handler):
            def emit(self, record):
                if threading.current_thread().ident == thread_id:
                    try:
                        log_queue.put_nowait({"type": "log", "msg": self.format(record)})
                    except Exception:
                        pass

        handler = _ThreadHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(handler)
        try:
            result = _run_analysis_sync(request)
            result_holder["ok"] = jsonable_encoder(result)
        except HTTPException as e:
            result_holder["error"] = e.detail
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            logging.getLogger().removeHandler(handler)
            log_queue.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    async def _stream():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, log_queue.get)
            if item is None:
                if "error" in result_holder:
                    yield json.dumps({"type": "error", "msg": result_holder["error"]}) + "\n"
                else:
                    yield json.dumps({"type": "result", "data": result_holder["ok"]}) + "\n"
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(
        _stream(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
