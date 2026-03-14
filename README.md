# ResearchGapFinder

**AI-powered system for discovering scientific research gaps and generating testable hypotheses**

ResearchGapFinder is an open-source, end-to-end scientific discovery tool that scans biomedical literature (PubMed, EuropePMC), detects knowledge gaps, and proposes mechanistic, experiment-ready hypotheses — all through a single pipeline with a professional web UI.

---

## Why This Tool

Scientific literature is growing faster than any human can read. Existing tools address pieces of this problem — Elicit and Iris.ai focus on literature mapping and Q&A, while LBD frameworks like SKiM-GPT focus on hypothesis scoring. ResearchGapFinder is different:

- **End-to-end pipeline**: From raw query to validated gaps, scored hypotheses, and specific experimental designs — not just associations or literature summaries.
- **Mechanistic focus**: Gaps and hypotheses target biological mechanisms, not surface-level topic overlaps.
- **Open-source and extensible**: Modular architecture with a model-agnostic LLM interface. No vendor lock-in.
- **Structured and scored outputs**: Every gap and hypothesis includes numeric scores for novelty, support, feasibility, and impact.
- **Professional UI**: Dark-theme Streamlit interface with radar charts, score bars, and cluster visualizations.

---

## Who It's For

- Bioinformatics and computational biology researchers
- Biomedical scientists preparing grant proposals
- Translational medicine teams exploring new therapeutic angles
- Anyone doing early-stage discovery research or literature review

---

## System Architecture

```
User Query
    │
    ▼
Query Expansion Module
    │  (MeSH/UMLS synonym expansion, gene aliases, disease synonyms)
    │
    ▼
Literature Retriever (PubMed / EuropePMC API)
    │  (temporal filters, article type filters, de-duplication)
    │
    ▼
Paper Processor
    │  ├── Abstract cleaning & boilerplate removal
    │  ├── Metadata extraction (PMID, authors, year, journal, DOI)
    │  └── Concept extraction & normalization (PubTator3 API)
    │
    ▼
Embedding Generator
    │  (all-MiniLM-L6-v2 for balanced/cheap_fast · BioLORD for max_quality)
    │
    ▼
Vector Database (FAISS · optional Pinecone)
    │
    ▼
Paper Clustering (HDBSCAN · k-means fallback)
    │  ├── Silhouette score per cluster
    │  ├── Davies-Bouldin index
    │  ├── TF-IDF top terms per cluster
    │  └── Auto-generated cluster labels
    │
    ▼
Research Gap Detector (Statistical Pipeline + LLM)
    │  ├── Explicit gap NLP (phrase detection in abstracts)
    │  ├── Concept co-occurrence analysis (TF-IDF, Jaccard)
    │  ├── ABC missing-link chain detection
    │  └── LLM articulation and rating of candidate gaps (Groq API)
    │
    ▼
Novelty Checker
    │  (Re-query PubMed with hypothesis-derived searches)
    │  (LLM evaluates: "Is this already established?")
    │
    ▼
Hypothesis Generator (LLM reasoning)
    │  ├── Novelty score
    │  ├── Support score (indirect evidence strength)
    │  ├── Feasibility score
    │  └── Impact score
    │
    ▼
Experiment Designer
    │  ├── Mapped to modalities (genetic perturbation, pharmacological, omics)
    │  ├── Standard assay recommendations
    │  └── Complexity tags (low / medium / high)
    │
    ▼
Streamlit UI + JSON Report (with full evidence trails)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly |
| Retrieval | PubMed E-utilities API, EuropePMC REST API |
| Concept extraction | PubTator3 API |
| Embeddings | `all-MiniLM-L6-v2` (default) · `FremyCompany/BioLORD-2023` (max_quality) |
| Vector database | FAISS (default) · Pinecone (optional) |
| Clustering | HDBSCAN, scikit-learn k-means |
| LLM | Groq API — Llama 3.1 8B / Llama 3.3 70B |
| Caching | diskcache (disk-based, SHA-256 keyed) |
| Rate limiting | Async token bucket (3 req/s PubMed, 10 req/s Groq) |
| Deployment | Render (Docker, two services) |

---

## Quality Presets

| Preset | LLM | Embeddings | Max Papers | RAM | Best For |
|--------|-----|-----------|-----------|-----|---------|
| `cheap_fast` | Llama 3.1 8B | MiniLM | 500 | ~150 MB | Quick demos, free tier |
| `balanced` | Llama 3.3 70B | MiniLM | 2,000 | ~150 MB | Free tier, good quality |
| `max_quality` | Llama 3.3 70B | BioLORD | 5,000 | ~1.5 GB | Real research, paid tier |

---

## Gap Detection: How It Works

Gaps are not simply "topics with few papers." The system uses a structured taxonomy:

| Gap Type | Definition | Detection Method |
|----------|-----------|-----------------|
| **Explicit gap** | Authors state "X remains unknown" | NLP phrase extraction from abstracts |
| **Implicit gap** | Expected concept co-occurrence is missing | Association measures across clusters |
| **Missing link** | A→B and B→C exist, but A→C is unexplored | ABC chain analysis on concept graphs |
| **Contradictory gap** | Studies disagree on a mechanism | Stance detection on shared concepts |

Each detected gap includes:
- **Type** (from taxonomy above)
- **Evidence snippets** with PMIDs
- **Competing explanations**
- **Reason for underexploration** (methodological, ethical, or neglect)
- **Uncertainty rating** (low / medium / high)

---

## Hypothesis Scoring

Every generated hypothesis receives four numeric scores (0.0 – 1.0):

| Score | What It Measures |
|-------|-----------------|
| **Novelty** | How little direct support currently exists |
| **Support** | Strength of indirect / mechanistic evidence |
| **Feasibility** | Testability with standard lab methods |
| **Impact** | Clinical importance or pathway centrality |

A composite score (average of all four) is shown in the UI alongside a radar chart.

---

## Example Workflow

**Query:** `TP53 ferroptosis breast cancer`

**Retrieved:** ~100–2,000 papers (with MeSH-expanded synonyms)

**Clusters:**
- Cluster 0 — Tumor suppressor signaling (silhouette: 0.72)
- Cluster 1 — Ferroptosis regulation (silhouette: 0.68)
- Cluster 2 — Breast cancer resistance (silhouette: 0.71)

**Detected gap:**
> Few studies directly examine how TP53 mutation status shapes ferroptosis sensitivity in therapy-resistant breast cancer subtypes.
>
> *Type: Implicit gap · Uncertainty: High · Reason: Methodological — requires isogenic cell line panels*

**Generated hypothesis:**
> Loss or mutation of TP53 may alter ferroptosis sensitivity by rewiring oxidative stress response pathways (particularly GPX4 and SLC7A11) in therapy-resistant breast cancer cells.
>
> Novelty: 0.78 · Support: 0.65 · Feasibility: 0.82 · Impact: 0.85

**Suggested experiments:**

| Experiment | Modality | Assays | Complexity |
|-----------|----------|--------|-----------|
| Compare WT vs mutant TP53 isogenic lines | Genetic perturbation | Cell viability (RSL3/erastin), BODIPY lipid ROS | Medium |
| Measure GPX4/SLC7A11 expression | Molecular profiling | Western blot, qPCR | Low |
| Transcriptomic analysis under ferroptosis induction | Omics profiling | RNA-seq | High |

---

## Project Structure

```
ResearchGapFinder1-1/
├── app/
│   ├── main.py                     # FastAPI app, 13-step pipeline, /analyze endpoint
│   └── config.py                   # Settings, PRESETS dict, get_preset()
├── retrieval/
│   ├── pubmed_search.py            # PubMed E-utilities, batched, cached, rate-limited
│   ├── europepmc_search.py         # EuropePMC REST API with cursorMark pagination
│   └── query_expansion.py          # MeSH API + biomedical alias dictionary
├── processing/
│   ├── paper_cleaner.py            # HTML stripping, boilerplate removal, length filter
│   ├── metadata_parser.py          # PubMed XML + EuropePMC JSON parsing
│   └── concept_extractor.py        # PubTator3 API (genes, diseases, chemicals)
├── embedding/
│   └── embedder.py                 # SentenceTransformers, unit normalization, singleton cache
├── vector_db/
│   └── vector_store.py             # FAISS IndexFlatIP, save/load
├── clustering/
│   ├── cluster_papers.py           # HDBSCAN primary, k-means fallback (>40% noise)
│   └── cluster_evaluator.py        # Silhouette, Davies-Bouldin, TF-IDF labels
├── gap_detection/
│   ├── gap_agent.py                # LLM gap articulation (Groq)
│   ├── gap_scorer.py               # Statistical signals: phrases, co-occurrence, ABC chains
│   └── gap_taxonomy.py             # GapType enum + descriptions
├── hypothesis/
│   ├── hypothesis_agent.py         # LLM hypothesis generation + ranking
│   ├── novelty_checker.py          # PubMed re-query + LLM novelty evaluation
│   └── hypothesis_scorer.py        # 4-dimension scoring
├── experiments/
│   ├── experiment_generator.py     # LLM experiment design
│   └── experiment_ontology.py      # Modality → assay mappings
├── llm/
│   └── reasoner.py                 # LLMReasoner: Groq client, retry, JSON parsing
├── schemas/
│   └── output_schema.py            # Pydantic v2 models for all I/O
├── prompts/
│   ├── gap_prompt.txt
│   └── hypothesis_prompt.txt
├── evaluation/
│   ├── retrospective_eval.py       # Historical corpus evaluation
│   ├── retrieval_eval.py           # Precision/recall benchmarks
│   └── cluster_eval.py             # PMI-based topic coherence
├── utils/
│   ├── rate_limit.py               # Async token bucket rate limiter
│   └── caching.py                  # diskcache with SHA-256 keys
├── tests/
│   ├── test_retrieval.py
│   ├── test_clustering.py
│   └── test_pipeline.py
├── streamlit_app.py                # Professional dark-theme UI
├── Dockerfile.backend              # FastAPI Docker image
├── Dockerfile.frontend             # Streamlit Docker image
├── render.yaml                     # Render deployment blueprint (2 services)
├── requirements.txt
└── .env.example
```

---

## Installation & Local Setup

```bash
git clone https://github.com/BulutHamali/ResearchGapFinder1
cd ResearchGapFinder1
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys in .env
```

**`.env` variables:**
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_key   # optional — FAISS is used by default
PUBMED_EMAIL=you@example.com
LLM_PRESET=balanced
VECTOR_DB=faiss
CACHE_DIR=.cache
```

---

## Running Locally

**Terminal 1 — Backend:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## API Usage

**Endpoint:** `POST /analyze`

**Request:**
```json
{
  "query": "TP53 ferroptosis breast cancer",
  "max_papers": 100,
  "year_range": [2015, 2025],
  "article_types": ["research", "review"],
  "llm_preset": "balanced"
}
```

**Response:**
```json
{
  "query": "TP53 ferroptosis breast cancer",
  "papers_retrieved": 97,
  "clusters": [
    {
      "cluster_id": 0,
      "label": "Tumor Suppressor Signaling",
      "paper_count": 34,
      "top_terms": ["TP53", "apoptosis", "cell cycle"],
      "silhouette_score": 0.72
    }
  ],
  "research_gaps": [
    {
      "gap": "The role of TP53 mutation status in ferroptosis sensitivity in therapy-resistant breast cancer remains unclear",
      "type": "implicit_gap",
      "evidence_snippets": [{"pmid": "38291234", "text": "..."}],
      "reason_underexplored": "methodological",
      "uncertainty": "high",
      "competing_explanations": ["TP53-independent mechanisms may dominate"]
    }
  ],
  "hypotheses": [
    {
      "hypothesis": "Mutant TP53 decreases ferroptosis sensitivity via GPX4 and SLC7A11 rewiring",
      "novelty_score": 0.78,
      "support_score": 0.65,
      "feasibility_score": 0.82,
      "impact_score": 0.85,
      "reasoning_summary": "Based on indirect evidence from clusters 0 and 1...",
      "already_established": false
    }
  ],
  "suggested_experiments": [
    {
      "experiment": "Compare TP53 WT vs mutant isogenic cell lines under RSL3/erastin treatment",
      "modality": "genetic_perturbation",
      "assays": ["cell viability (RSL3/erastin)", "BODIPY lipid ROS"],
      "complexity": "medium",
      "required_models": ["MCF7 isogenic panel"]
    }
  ]
}
```

**Health check:** `GET /health`

---

## Deployment (Render)

The repo includes a `render.yaml` blueprint that deploys two services automatically:

1. **Fork / clone the repo** to your GitHub
2. Go to [render.com](https://render.com) → **New → Blueprint** → connect repo
3. Set the secret environment variables when prompted:
   - `GROQ_API_KEY`
   - `PUBMED_EMAIL`
4. Click **Apply** — both services build and deploy automatically

**Live URLs after deploy:**
- Frontend: `https://researchgapfinder-frontend.onrender.com`
- Backend: `https://researchgapfinder-backend.onrender.com`

> **Note:** Free tier services sleep after 15 min of inactivity. First request after sleep takes ~30–60s.

---

## Token Optimization

LLMs cannot process thousands of papers directly. The system separates concerns:

1. Cluster papers → reduces thousands of papers to N clusters
2. Summarize each cluster with top TF-IDF terms and paper count
3. Send only cluster summaries + statistical gap signals to the LLM

**Example:** 500 papers → 12 clusters → 12 summaries → LLM reasoning

This keeps costs low while preserving the breadth of the literature scan.

---

## Evaluation Strategy

### Intrinsic System Evaluation
- **Retrieval**: Precision/recall against curated benchmark queries
- **Clustering**: Silhouette scores, Davies-Bouldin index, topic coherence
- **Gap detection**: Agreement with human annotators on labeled corpora

### Retrospective Hypothesis Validation
*Can the system, run on a 2015 corpus, predict a link confirmed in 2016–2025 literature?*

- Hide all papers after a cutoff year
- Generate hypotheses from the truncated corpus
- Check if any were confirmed by subsequent publications

---

## Roadmap

### Phase 1 — Core Pipeline ✅
- [x] Literature retrieval with MeSH query expansion
- [x] Concept extraction via PubTator3
- [x] Domain-specific embeddings (MiniLM / BioLORD)
- [x] Quality-validated HDBSCAN clustering
- [x] Structured gap detection with statistical scoring
- [x] Hypothesis generation with novelty checking and 4-dimension scoring
- [x] Experiment design with assay and modality mappings
- [x] FastAPI backend with full pipeline
- [x] Professional Streamlit UI with dark theme
- [x] Render deployment (Docker, two services)

### Phase 2 — Evaluation & Validation
- [ ] Retrospective evaluation benchmark
- [ ] Retrieval precision/recall benchmarks
- [ ] Published case study

### Phase 3 — Advanced Discovery
- [ ] Citation graph analysis
- [ ] ABC chain / AnC discovery models
- [ ] Graph-based community detection on co-mention networks

### Phase 4 — Data Integration
- [ ] GEO/TCGA dataset integration
- [ ] Pathway database integration (KEGG, Reactome)
- [ ] Differential expression checks against hypotheses

### Phase 5 — UI Enhancements
- [ ] Interactive cluster landscape exploration
- [ ] Researcher annotation and feedback loop
- [ ] Hypothesis history and session management

---

## Transparency and Safety

Every gap and hypothesis includes:
- **Supporting paper snippets** with cited sentences and PMIDs
- **Reasoning summary** generated strictly from evidence snippets
- **Confidence calibration** via numeric scores

**Safety guardrails:**
- No direct therapeutic recommendations to patients
- Explicit disclaimers on associative vs. causal findings
- Hard filters preventing over-interpretation of observational results

---

## License

MIT License

---

## Author

**Bulut Hamali**
Bioinformatics · AI for Science · Research Automation
