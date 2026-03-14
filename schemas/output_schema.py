from pydantic import BaseModel, Field
from typing import Optional


class EvidenceSnippet(BaseModel):
    pmid: str
    text: str


class ResearchGap(BaseModel):
    gap: str
    type: str  # explicit_gap, implicit_gap, missing_link, contradictory_gap
    evidence_snippets: list[EvidenceSnippet]
    reason_underexplored: str
    uncertainty: str  # low, medium, high
    competing_explanations: list[str]


class Hypothesis(BaseModel):
    hypothesis: str
    novelty_score: float
    support_score: float
    feasibility_score: float
    impact_score: float
    reasoning_summary: str
    already_established: bool


class SuggestedExperiment(BaseModel):
    experiment: str
    modality: str
    assays: list[str]
    complexity: str  # low, medium, high
    required_models: list[str]


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    paper_count: int
    top_terms: list[str]
    silhouette_score: float


class AnalysisRequest(BaseModel):
    query: str
    max_papers: int = 2000
    year_range: tuple[int, int] = (2015, 2025)
    article_types: list[str] = ["research", "review"]
    llm_preset: str = "balanced"


class AnalysisResponse(BaseModel):
    query: str
    papers_retrieved: int
    clusters: list[ClusterSummary]
    research_gaps: list[ResearchGap]
    hypotheses: list[Hypothesis]
    suggested_experiments: list[SuggestedExperiment]
