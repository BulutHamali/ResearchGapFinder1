# ResearchGapFinder

**AI-powered system for discovering scientific research gaps and generating testable hypotheses**

ResearchGapFinder is an open-source, end-to-end scientific discovery tool that scans large biomedical literature collections (e.g., PubMed), detects knowledge gaps, and proposes mechanistic, experiment-ready research hypotheses.

The system combines large-scale literature retrieval, semantic clustering, structured gap detection, hypothesis generation with novelty checking, and actionable experiment design — all in a single pipeline.

---

## Why This Tool

Scientific literature is growing faster than any human can read. Existing tools address pieces of this problem — Elicit and Iris.ai focus on literature mapping and Q&A, while LBD frameworks like SKiM-GPT focus on hypothesis scoring. ResearchGapFinder is different:

- **End-to-end pipeline**: From raw query to validated gaps, scored hypotheses, and specific experimental designs — not just associations or literature summaries.
- **Mechanistic focus**: Gaps and hypotheses target biological mechanisms, not surface-level topic overlaps.
- **Open-source and extensible**: Modular architecture with a model-agnostic LLM interface. No vendor lock-in.
- **Structured and scored outputs**: Every gap and hypothesis includes numeric scores for novelty, support, feasibility, and impact — not just "sounds plausible."

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
    │  ├── Abstract cleaning
    │  ├── Full-text extraction (PMC OA where available)
    │  ├── Metadata extraction
    │  └── Concept extraction & normalization (MeSH/UMLS/PubTator)
    │
    ▼
Embedding Generator
    │  (Domain-specific: BioLORD / SciBERT, with fallback to SentenceTransformers)
    │
    ▼
Vector Database (Pinecone / FAISS)
    │
    ▼
Paper Clustering (HDBSCAN / k-means)
    │  ├── Cluster quality metrics (silhouette, Davies-Bouldin, topic coherence)
    │  ├── Top MeSH/UMLS terms per cluster
    │  └── Inter-cluster citation connectivity analysis
    │
    ▼
Research Gap Detector (Scoring Pipeline + LLM)
    │  ├── Coverage statistics per cluster
    │  ├── Association/interestingness measures (TF-IDF, Jaccard, mutual information)
    │  ├── "Expected but missing" concept pair detection
    │  └── LLM articulation and rating of candidate gaps
    │
    ▼
Novelty Checker
    │  (Re-query PubMed with hypothesis-derived searches; LLM evaluates: "Is this already established?")
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
    │  ├── Mapped to experimental modalities (genetic perturbation, pharmacological, omics)
    │  ├── Standard assay recommendations based on hypothesis elements
    │  └── Cost/complexity tags (low / medium / high)
    │
    ▼
Final Report (structured JSON with evidence trails)
```

### Why This Architecture

LLMs cannot efficiently process thousands of papers. The system separates concerns:

1. **Retrieval** — search and filter thousands of papers with query expansion
2. **Concept extraction** — normalize entities to structured vocabularies (MeSH, UMLS)
3. **Clustering** — detect research themes with quality-validated groupings
4. **Gap detection** — combine statistical signals with LLM reasoning (not a single LLM call)
5. **Novelty checking** — verify hypotheses aren't already established
6. **Hypothesis scoring** — calibrate confidence with numeric scores
7. **Experiment design** — map hypothesis elements to specific assays and models

---

## Gap Detection: How It Works

Gaps are not simply "topics with few papers." The system uses a structured taxonomy:

| Gap Type | Definition | Detection Method |
|----------|-----------|-----------------|
| **Explicit gap** | Authors state "X remains unknown" | NLP extraction from abstracts/discussion |
| **Implicit gap** | Expected concept co-occurrence is missing | Association measures across clusters |
| **Missing link** | A→B and B→C exist, but A→C is unexplored | ABC chain analysis on concept graphs |
| **Contradictory gap** | Studies disagree on a mechanism | Sentiment/stance detection on shared concepts |

Each detected gap includes:

- **Type** (from taxonomy above)
- **Evidence snippets** with PMIDs
- **Competing explanations**
- **Reason for underexploration** (methodological limitation, ethical constraint, or neglect)
- **Uncertainty rating**

The system filters out gaps that are underexplored for good reasons (known toxicity, ethical constraints, technical impossibility).

---

## Hypothesis Scoring

Every generated hypothesis receives four numeric scores:

| Score | What It Measures |
|-------|-----------------|
| **Novelty** | How little direct support currently exists |
| **Support** | Strength of indirect/mechanistic evidence |
| **Feasibility** | Can this be tested with standard lab methods? |
| **Impact** | Does it touch clinically important conditions or key pathways? |

The novelty checker converts each hypothesis into search queries, re-queries the literature, and uses the LLM to assess whether the hypothesis is already established, partially supported, or genuinely novel.

---

## Example Workflow

**User query:**
```
TP53 ferroptosis breast cancer
```

**System retrieves:** ~2,300 papers (with MeSH-expanded synonyms for TP53, ferroptosis, and breast cancer subtypes)

**Clusters (quality-validated):**
- Cluster A — Tumor suppressor signaling (silhouette: 0.72)
- Cluster B — Ferroptosis regulation mechanisms (silhouette: 0.68)
- Cluster C — Breast cancer resistance mechanisms (silhouette: 0.71)

**Detected gap:**
> Few studies directly examine how TP53 mutation status shapes ferroptosis sensitivity in therapy-resistant breast cancer subtypes.
>
> *Type: Implicit gap (expected co-occurrence missing)*
> *Reason for underexploration: Methodological — requires isogenic cell line panels*

**Generated hypothesis:**
> Loss or mutation of TP53 may alter ferroptosis sensitivity by rewiring oxidative stress response pathways (particularly GPX4 and SLC7A11) in therapy-resistant breast cancer cells.
>
> Novelty: 0.78 | Support: 0.65 | Feasibility: 0.82 | Impact: 0.85

**Suggested experiments:**
| Experiment | Modality | Assay | Complexity |
|-----------|----------|-------|-----------|
| Compare WT vs mutant TP53 isogenic lines | Genetic perturbation | Cell viability (RSL3/erastin), BODIPY lipid ROS | Medium |
| Measure GPX4/SLC7A11 expression | Molecular profiling | Western blot, qPCR | Low |
| Transcriptomic analysis under ferroptosis induction | Omics profiling | RNA-seq | High |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI |
| Retrieval | PubMed API, EuropePMC API |
| Concept normalization | PubTator, MeSH/UMLS |
| Embeddings | BioLORD / SciBERT (primary), SentenceTransformers (fallback) |
| Vector database | Pinecone, FAISS |
| Clustering | HDBSCAN, scikit-learn |
| LLM reasoning | Model-agnostic interface (`LLMReasoner` class) |
| Supported LLM backends | Groq API, local open-weights models (Llama 3, Mistral via quantization) |
| Configuration presets | `cheap_fast`, `balanced`, `max_quality` |

---

## Project Structure

```
research-gap-finder/
├── app/
│   ├── main.py
│   └── config.py
├── retrieval/
│   ├── pubmed_search.py
│   ├── europepmc_search.py
│   └── query_expansion.py          # MeSH/UMLS synonym expansion
├── processing/
│   ├── paper_cleaner.py
│   ├── metadata_parser.py
│   └── concept_extractor.py        # PubTator / UMLS normalization
├── embedding/
│   └── embedder.py                 # Domain-specific model support
├── vector_db/
│   └── vector_store.py
├── clustering/
│   ├── cluster_papers.py
│   └── cluster_evaluator.py        # Silhouette, Davies-Bouldin, coherence
├── gap_detection/
│   ├── gap_agent.py
│   ├── gap_scorer.py               # Statistical signals pipeline
│   └── gap_taxonomy.py             # Gap type definitions
├── hypothesis/
│   ├── hypothesis_agent.py
│   ├── novelty_checker.py          # Re-query + LLM evaluation
│   └── hypothesis_scorer.py        # Novelty / support / feasibility / impact
├── experiments/
│   ├── experiment_generator.py
│   └── experiment_ontology.py      # Modality → assay mappings
├── llm/
│   └── reasoner.py                 # Model-agnostic LLM interface
├── schemas/
│   └── output_schema.py
├── prompts/
│   ├── gap_prompt.txt
│   └── hypothesis_prompt.txt
├── evaluation/                     # Validation framework
│   ├── retrospective_eval.py       # Historical corpus evaluation
│   ├── retrieval_eval.py           # Precision/recall benchmarks
│   └── cluster_eval.py             # Coherence metrics
├── utils/
│   ├── rate_limit.py
│   └── caching.py
└── tests/
```

---

## Installation

```bash
git clone https://github.com/yourname/research-gap-finder
cd research-gap-finder
pip install -r requirements.txt
```

**Environment variables:**
```
GROQ_API_KEY=           # Required for Groq backend
PINECONE_API_KEY=       # Required for Pinecone (optional if using FAISS)
PUBMED_EMAIL=           # Required for PubMed API
LLM_PRESET=balanced     # Options: cheap_fast, balanced, max_quality
```

---

## Running the System

Start the API server:
```bash
python app/main.py
```

**Example request:**
```
POST /analyze
```

**Payload:**
```json
{
  "query": "TP53 ferroptosis breast cancer",
  "max_papers": 2000,
  "year_range": [2015, 2025],
  "article_types": ["research", "review"],
  "llm_preset": "balanced"
}
```

---

## Example Output

```json
{
  "research_gaps": [
    {
      "gap": "The role of TP53 mutation status in shaping ferroptosis response in therapy-resistant breast cancer remains unclear",
      "type": "implicit_gap",
      "evidence_snippets": [
        {"pmid": "PMID1", "text": "..."},
        {"pmid": "PMID2", "text": "..."}
      ],
      "reason_underexplored": "methodological",
      "uncertainty": "high",
      "competing_explanations": ["TP53-independent mechanisms may dominate"]
    }
  ],
  "hypotheses": [
    {
      "hypothesis": "Mutant TP53 decreases ferroptosis sensitivity by rewiring oxidative stress response pathways via GPX4 and SLC7A11",
      "novelty_score": 0.78,
      "support_score": 0.65,
      "feasibility_score": 0.82,
      "impact_score": 0.85,
      "reasoning_summary": "Based on indirect evidence from clusters A and B...",
      "already_established": false
    }
  ],
  "suggested_experiments": [
    {
      "experiment": "Compare TP53 wild-type vs mutant isogenic cell lines",
      "modality": "genetic_perturbation",
      "assays": ["cell viability (RSL3/erastin)", "BODIPY lipid ROS"],
      "complexity": "medium",
      "required_models": ["MCF7 isogenic panel"]
    }
  ]
}
```

---

## Token Optimization

To manage LLM token costs:

1. Cluster papers before sending to the LLM
2. Summarize each cluster with top MeSH terms and coverage statistics
3. Send only cluster summaries + statistical signals to the model

**Example:** 2,000 papers → 50 clusters → 50 summaries + coverage stats → LLM reasoning

---

## Evaluation Strategy

Validation is essential for credibility. The system includes three evaluation layers:

### Intrinsic System Evaluation
- **Retrieval**: Precision/recall against curated benchmark queries
- **Clustering**: Topic coherence scores, silhouette scores, human judgments
- **Gap detection**: Agreement with human annotators on labeled corpora

### Retrospective Hypothesis Validation
The key credibility test: *Can this system, run on a 2015 corpus, predict at least one link later confirmed in 2016–2025 literature?*

- Hide all papers after a cutoff year
- Generate hypotheses from the truncated corpus
- Check if any were confirmed by subsequent publications

### Prospective User Study (Planned)
- Provide gaps/hypotheses to practicing researchers
- Measure whether outputs change their experimental plans or literature searches

---

## Transparency and Safety

Every gap and hypothesis includes:
- **Supporting paper snippets** with cited sentences and PMIDs
- **Reasoning summary** generated strictly from evidence snippets (not free-form speculation)
- **Confidence calibration** via numeric scores

**Safety guardrails:**
- No direct therapeutic recommendations to patients
- Explicit disclaimers on associative vs. causal findings
- Hard filters preventing over-interpretation of observational results

---

## Roadmap

### Phase 1 — Core Pipeline (Current)
- [x] Literature retrieval with query expansion
- [x] Concept extraction and normalization
- [x] Domain-specific embeddings
- [x] Quality-validated clustering
- [x] Structured gap detection with scoring pipeline
- [x] Hypothesis generation with novelty checking
- [x] Experiment design with assay mappings

### Phase 2 — Evaluation & Validation
- [ ] Retrospective evaluation benchmark
- [ ] Retrieval precision/recall benchmarks
- [ ] Cluster coherence benchmarks
- [ ] Published case study showing system catches a known-but-delayed discovery

### Phase 3 — Advanced Discovery
- [ ] Citation graph analysis (contradictions, unexplored edges)
- [ ] ABC chain / AnC discovery models for mechanistic hypotheses
- [ ] Graph-based community detection on co-mention networks
- [ ] Novelty detection against existing hypothesis databases

### Phase 4 — Data Integration
- [ ] GEO/TCGA dataset integration (surface relevant datasets for each hypothesis)
- [ ] Pathway database integration (KEGG, Reactome) for network topology validation
- [ ] Basic differential expression checks: "Does existing data support or contradict this hypothesis?"

### Phase 5 — Interactive UI
- [ ] Visual exploration of discovered gaps and cluster landscapes
- [ ] Interactive self-screening of hypothesis quality
- [ ] Researcher annotation and feedback loop

---

## Potential Applications

- Grant proposal preparation
- Systematic literature review acceleration
- Early-stage discovery research
- AI-assisted scientific brainstorming
- Identifying collaboration opportunities across research silos

---

## License

MIT License

---

## Acknowledgments

Inspired by emerging work in AI-assisted scientific discovery, literature-based discovery (LBD), and automated research assistants.

---

## Author

**Bulut Hamali**
Bioinformatics · AI for Science · Research Automation
