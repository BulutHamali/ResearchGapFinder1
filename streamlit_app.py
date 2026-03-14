"""
ResearchGapFinder — Streamlit UI
Calls the FastAPI backend at http://localhost:8000
"""

import json
import time
from datetime import datetime

import httpx
import plotly.graph_objects as go
import streamlit as st

import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchGapFinder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .score-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .gap-card {
        background: #f8f9fb;
        border-left: 4px solid #1a73e8;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .hypothesis-card {
        background: #f0f7ff;
        border-left: 4px solid #34a853;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .experiment-card {
        background: #fff8f0;
        border-left: 4px solid #fa7b17;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 ResearchGapFinder</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-powered discovery of scientific research gaps and testable hypotheses from biomedical literature</p>',
    unsafe_allow_html=True,
)

# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    query = st.text_area(
        "Research Query",
        placeholder="e.g. TP53 ferroptosis breast cancer",
        height=90,
        help="Enter key terms, gene names, diseases, or mechanisms to investigate.",
    )

    preset = st.selectbox(
        "Quality Preset",
        options=["cheap_fast", "balanced", "max_quality"],
        index=1,
        help=(
            "**cheap_fast** — small model, up to 500 papers, fastest\n\n"
            "**balanced** — large model, up to 2,000 papers, recommended\n\n"
            "**max_quality** — large model, up to 5,000 papers, thorough"
        ),
    )

    max_papers = st.slider(
        "Max Papers",
        min_value=50,
        max_value=5000,
        value=500,
        step=50,
        help="Upper bound on papers retrieved. Larger values take longer.",
    )

    col1, col2 = st.columns(2)
    with col1:
        year_start = st.number_input("From Year", min_value=1990, max_value=2025, value=2015)
    with col2:
        year_end = st.number_input("To Year", min_value=1990, max_value=2025, value=2025)

    article_types = st.multiselect(
        "Article Types",
        options=["research", "review", "meta-analysis", "clinical_trial"],
        default=["research", "review"],
    )

    st.divider()
    backend_url = st.text_input(
        "Backend URL",
        value=BACKEND_URL,
        help="URL of the running FastAPI backend.",
    )

    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True, disabled=not query.strip())

    # Backend health
    st.divider()
    if st.button("Check Backend", use_container_width=True):
        try:
            r = httpx.get(f"{backend_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("Backend online ✓")
            else:
                st.error(f"Backend returned {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")


# ── Helper functions ──────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 0.75:
        return "#1e7e34"   # green
    elif score >= 0.5:
        return "#856404"   # amber
    else:
        return "#721c24"   # red


def score_bar(label: str, value: float, color: str = "#1a73e8"):
    st.markdown(f"**{label}** `{value:.2f}`")
    st.progress(value)


def gap_type_badge(gap_type: str) -> str:
    colors = {
        "explicit_gap":     ("🔵", "#cce5ff", "#004085"),
        "implicit_gap":     ("🟡", "#fff3cd", "#856404"),
        "missing_link":     ("🟢", "#d4edda", "#155724"),
        "contradictory_gap":("🔴", "#f8d7da", "#721c24"),
    }
    icon, bg, fg = colors.get(gap_type, ("⚪", "#e2e3e5", "#383d41"))
    label = gap_type.replace("_", " ").title()
    return f'<span class="badge" style="background:{bg};color:{fg};">{icon} {label}</span>'


def complexity_badge(complexity: str) -> str:
    colors = {"low": ("#d4edda", "#155724"), "medium": ("#fff3cd", "#856404"), "high": ("#f8d7da", "#721c24")}
    bg, fg = colors.get(complexity, ("#e2e3e5", "#383d41"))
    return f'<span class="badge" style="background:{bg};color:{fg};">{complexity.upper()}</span>'


def render_gaps(gaps: list):
    if not gaps:
        st.info("No research gaps detected.")
        return
    for i, gap in enumerate(gaps, 1):
        with st.expander(f"Gap {i} — {gap['gap'][:100]}{'…' if len(gap['gap']) > 100 else ''}", expanded=(i == 1)):
            st.markdown(
                f"<div class='gap-card'>"
                f"{gap_type_badge(gap['type'])} "
                f"&nbsp; Uncertainty: <strong>{gap['uncertainty']}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Gap:** {gap['gap']}")

            if gap.get("reason_underexplored"):
                st.markdown(f"**Why underexplored:** {gap['reason_underexplored']}")

            if gap.get("competing_explanations"):
                st.markdown("**Competing explanations:**")
                for exp in gap["competing_explanations"]:
                    st.markdown(f"- {exp}")

            if gap.get("evidence_snippets"):
                st.markdown("**Evidence snippets:**")
                for snip in gap["evidence_snippets"]:
                    pmid = snip.get("pmid", "—")
                    text = snip.get("text", "")
                    st.markdown(f"> {text}  \n> `PMID: {pmid}`")


def render_hypotheses(hypotheses: list):
    if not hypotheses:
        st.info("No hypotheses generated.")
        return
    for i, hyp in enumerate(hypotheses, 1):
        established_tag = "✅ Novel" if not hyp.get("already_established") else "⚠️ May already be established"
        with st.expander(
            f"Hypothesis {i} — {hyp['hypothesis'][:100]}{'…' if len(hyp['hypothesis']) > 100 else ''}",
            expanded=(i == 1),
        ):
            st.markdown(f"**{established_tag}**")
            st.markdown(f"**Hypothesis:** {hyp['hypothesis']}")

            if hyp.get("reasoning_summary"):
                st.markdown(f"**Reasoning:** {hyp['reasoning_summary']}")

            # Scores as a radar / bar chart
            scores = {
                "Novelty": hyp.get("novelty_score", 0),
                "Support": hyp.get("support_score", 0),
                "Feasibility": hyp.get("feasibility_score", 0),
                "Impact": hyp.get("impact_score", 0),
            }

            col_a, col_b = st.columns([1, 1])
            with col_a:
                for label, val in scores.items():
                    c = score_color(val)
                    st.markdown(
                        f"**{label}** &nbsp; "
                        f'<span style="color:{c};font-weight:700;">{val:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    st.progress(val)
            with col_b:
                fig = go.Figure(
                    go.Scatterpolar(
                        r=list(scores.values()),
                        theta=list(scores.keys()),
                        fill="toself",
                        fillcolor="rgba(26, 115, 232, 0.2)",
                        line=dict(color="#1a73e8", width=2),
                    )
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=240,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_experiments(experiments: list):
    if not experiments:
        st.info("No experiments suggested.")
        return
    for i, exp in enumerate(experiments, 1):
        with st.expander(
            f"Experiment {i} — {exp['experiment'][:100]}{'…' if len(exp['experiment']) > 100 else ''}",
            expanded=(i <= 2),
        ):
            modality = exp.get("modality", "—").replace("_", " ").title()
            complexity = exp.get("complexity", "—")
            st.markdown(
                f"**Modality:** {modality} &nbsp;&nbsp; "
                f"**Complexity:** {complexity_badge(complexity)}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Experiment:** {exp['experiment']}")

            assays = exp.get("assays", [])
            if assays:
                st.markdown("**Assays:** " + " · ".join(f"`{a}`" for a in assays))

            models = exp.get("required_models", [])
            if models:
                st.markdown("**Models / Systems:** " + " · ".join(f"`{m}`" for m in models))


def render_clusters(clusters: list):
    if not clusters:
        st.info("No cluster data available.")
        return

    # Bubble chart: x = cluster_id, y = silhouette, size = paper_count
    fig = go.Figure()
    for c in clusters:
        fig.add_trace(
            go.Scatter(
                x=[c["cluster_id"]],
                y=[c["silhouette_score"]],
                mode="markers+text",
                marker=dict(
                    size=max(12, min(60, c["paper_count"] // 3)),
                    color=c["silhouette_score"],
                    colorscale="Blues",
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    line=dict(width=1, color="#1a73e8"),
                ),
                text=[c["label"][:30]],
                textposition="top center",
                name=c["label"],
                hovertemplate=(
                    f"<b>{c['label']}</b><br>"
                    f"Papers: {c['paper_count']}<br>"
                    f"Silhouette: {c['silhouette_score']:.2f}<br>"
                    f"Top terms: {', '.join(c['top_terms'][:5])}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Cluster ID",
        yaxis_title="Silhouette Score",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    rows = []
    for c in clusters:
        rows.append({
            "ID": c["cluster_id"],
            "Label": c["label"],
            "Papers": c["paper_count"],
            "Silhouette": f"{c['silhouette_score']:.2f}",
            "Top Terms": ", ".join(c["top_terms"][:5]),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Main — run analysis ───────────────────────────────────────────────────────

if analyze_btn and query.strip():
    payload = {
        "query": query.strip(),
        "max_papers": max_papers,
        "year_range": [int(year_start), int(year_end)],
        "article_types": article_types,
        "llm_preset": preset,
    }

    status_box = st.empty()
    progress_bar = st.progress(0, text="Starting analysis…")

    steps = [
        (10, "Expanding query with MeSH synonyms…"),
        (25, "Retrieving papers from PubMed…"),
        (40, "Cleaning and extracting concepts…"),
        (55, "Generating embeddings and clustering…"),
        (70, "Detecting research gaps…"),
        (85, "Generating and scoring hypotheses…"),
        (95, "Designing experiments…"),
    ]

    result = None
    error = None

    with st.spinner(""):
        # Animate progress while waiting for the backend
        import threading

        done = threading.Event()

        def animate():
            for pct, msg in steps:
                if done.is_set():
                    break
                progress_bar.progress(pct, text=msg)
                time.sleep(3)

        t = threading.Thread(target=animate, daemon=True)
        t.start()

        try:
            response = httpx.post(
                f"{backend_url}/analyze",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as e:
            error = f"Backend error {e.response.status_code}: {e.response.text}"
        except httpx.ConnectError:
            error = (
                f"Cannot connect to backend at `{backend_url}`. "
                "Make sure the FastAPI server is running (`python app/main.py`)."
            )
        except Exception as e:
            error = str(e)
        finally:
            done.set()

    progress_bar.empty()
    status_box.empty()

    if error:
        st.error(f"**Analysis failed:** {error}")
    elif result:
        # ── Summary metrics ───────────────────────────────────────────────
        st.success(f"Analysis complete — `{result['query']}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Papers Retrieved", result.get("papers_retrieved", 0))
        m2.metric("Clusters Found", len(result.get("clusters", [])))
        m3.metric("Research Gaps", len(result.get("research_gaps", [])))
        m4.metric("Hypotheses", len(result.get("hypotheses", [])))

        # ── Tabs ──────────────────────────────────────────────────────────
        tab_clusters, tab_gaps, tab_hyp, tab_exp, tab_json = st.tabs(
            ["📊 Clusters", "🔍 Research Gaps", "💡 Hypotheses", "🧪 Experiments", "📄 Raw JSON"]
        )

        with tab_clusters:
            st.subheader("Paper Cluster Landscape")
            render_clusters(result.get("clusters", []))

        with tab_gaps:
            st.subheader("Detected Research Gaps")
            render_gaps(result.get("research_gaps", []))

        with tab_hyp:
            st.subheader("Generated Hypotheses")
            render_hypotheses(result.get("hypotheses", []))

        with tab_exp:
            st.subheader("Suggested Experiments")
            render_experiments(result.get("suggested_experiments", []))

        with tab_json:
            st.subheader("Full JSON Output")
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"research_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            st.code(json_str, language="json")

else:
    # ── Welcome screen ────────────────────────────────────────────────────
    st.info(
        "👈 **Enter a research query in the sidebar and click Analyze** to start.\n\n"
        "Example queries:\n"
        "- `TP53 ferroptosis breast cancer`\n"
        "- `BRCA1 DNA repair immunotherapy`\n"
        "- `mTOR autophagy neurodegeneration`"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Gap Detection")
        st.markdown(
            "Identifies explicit, implicit, missing-link, and contradictory gaps "
            "using NLP + statistical signals across clustered literature."
        )
    with col2:
        st.markdown("### 💡 Hypothesis Generation")
        st.markdown(
            "Generates mechanistic, experiment-ready hypotheses scored on "
            "**novelty**, **support**, **feasibility**, and **impact**."
        )
    with col3:
        st.markdown("### 🧪 Experiment Design")
        st.markdown(
            "Maps each hypothesis to specific assays, experimental models, "
            "and complexity tags (low / medium / high)."
        )
