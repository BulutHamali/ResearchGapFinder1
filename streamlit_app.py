"""
ResearchGapFinder — Streamlit UI
"""

import json
import os
import threading
import time
from datetime import datetime

import httpx
import plotly.graph_objects as go
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchGapFinder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* ── Hide default header decoration ── */
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3448; border-radius: 3px; }

/* ── Hero ── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-title span { color: #4f8ef7; }
.hero-sub {
    margin-top: 0.5rem;
    font-size: 1rem;
    color: #6b7a99;
    max-width: 620px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #4f8ef7;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.3rem;
}

/* ── Cards ── */
.card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent-blue  { border-left: 3px solid #4f8ef7; }
.card-accent-green { border-left: 3px solid #3ecf8e; }
.card-accent-amber { border-left: 3px solid #f5a623; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-blue   { background: #1a2d52; color: #4f8ef7; }
.badge-yellow { background: #2d2510; color: #f5a623; }
.badge-green  { background: #0d2b1f; color: #3ecf8e; }
.badge-red    { background: #2d1010; color: #f76f6f; }
.badge-gray   { background: #1e2535; color: #8b97b0; }

/* ── Score bar ── */
.score-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.score-label { font-size: 0.8rem; color: #8b97b0; width: 80px; flex-shrink: 0; }
.score-track { flex: 1; height: 5px; background: #1e2535; border-radius: 3px; overflow: hidden; }
.score-fill  { height: 100%; border-radius: 3px; }
.score-val   { font-size: 0.82rem; font-weight: 700; width: 32px; text-align: right; flex-shrink: 0; }

/* ── Evidence quote ── */
.evidence {
    border-left: 2px solid #2a3448;
    padding: 0.5rem 0 0.5rem 0.9rem;
    margin: 0.4rem 0;
    color: #8b97b0;
    font-size: 0.88rem;
    font-style: italic;
}
.pmid-tag {
    display: inline-block;
    font-size: 0.7rem;
    font-style: normal;
    background: #1e2535;
    color: #4f8ef7;
    padding: 1px 7px;
    border-radius: 4px;
    margin-top: 4px;
    font-family: monospace;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4f8ef7;
    margin: 1rem 0 0.5rem 0;
}

/* ── Welcome feature box ── */
.feature-box {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.4rem;
}
.feature-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.feature-title { font-size: 1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem; }
.feature-desc  { font-size: 0.88rem; color: #6b7a99; line-height: 1.5; }

/* ── Query chip ── */
.query-chip {
    display: inline-block;
    background: #1a2d52;
    color: #4f8ef7;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.82rem;
    font-family: monospace;
    margin: 2px;
    cursor: pointer;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    color: #6b7a99 !important;
    font-size: 0.85rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4f8ef7 !important;
    border-bottom-color: #4f8ef7 !important;
}

/* ── Sidebar input labels ── */
[data-testid="stSidebar"] label { font-size: 0.8rem !important; }
[data-testid="stSidebar"] .stTextArea textarea {
    background: #0f1117 !important;
    border: 1px solid #2a3448 !important;
    color: #c9d1e0 !important;
    font-size: 0.9rem !important;
}

/* ── Success / error banners ── */
.banner-success {
    background: #0d2b1f;
    border: 1px solid #3ecf8e;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #3ecf8e;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.banner-error {
    background: #2d1010;
    border: 1px solid #f76f6f;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #f76f6f;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def score_color_hex(val: float) -> str:
    if val >= 0.75: return "#3ecf8e"
    if val >= 0.5:  return "#f5a623"
    return "#f76f6f"


def score_bar_html(label: str, val: float) -> str:
    color = score_color_hex(val)
    pct   = int(val * 100)
    return f"""
    <div class="score-row">
      <div class="score-label">{label}</div>
      <div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color};"></div></div>
      <div class="score-val" style="color:{color};">{val:.2f}</div>
    </div>"""


def gap_badge(gap_type: str) -> str:
    mapping = {
        "explicit_gap":      ("Explicit Gap",      "badge-blue"),
        "implicit_gap":      ("Implicit Gap",      "badge-yellow"),
        "missing_link":      ("Missing Link",      "badge-green"),
        "contradictory_gap": ("Contradictory Gap", "badge-red"),
    }
    label, cls = mapping.get(gap_type, (gap_type, "badge-gray"))
    return f'<span class="badge {cls}">{label}</span>'


def uncertainty_badge(u: str) -> str:
    cls = {"low": "badge-green", "medium": "badge-yellow", "high": "badge-red"}.get(u, "badge-gray")
    return f'<span class="badge {cls}">Uncertainty: {u}</span>'


def complexity_badge(c: str) -> str:
    cls = {"low": "badge-green", "medium": "badge-yellow", "high": "badge-red"}.get(c, "badge-gray")
    return f'<span class="badge {cls}">{c} complexity</span>'


def novel_badge(established: bool) -> str:
    if not established:
        return '<span class="badge badge-green">Novel</span>'
    return '<span class="badge badge-red">May be established</span>'


def radar_chart(scores: dict) -> go.Figure:
    labels = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(79,142,247,0.12)",
        line=dict(color="#4f8ef7", width=2),
        marker=dict(color="#4f8ef7", size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0f1117",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#6b7a99", size=9), gridcolor="#1e2535"),
            angularaxis=dict(tickfont=dict(color="#c9d1e0", size=11), gridcolor="#1e2535"),
        ),
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=220,
    )
    return fig


def cluster_chart(clusters: list) -> go.Figure:
    fig = go.Figure()
    colors = ["#4f8ef7", "#3ecf8e", "#f5a623", "#f76f6f", "#b48ef7", "#4fc9f7", "#f76fc9"]
    for i, c in enumerate(clusters):
        fig.add_trace(go.Scatter(
            x=[c["cluster_id"]],
            y=[c["silhouette_score"]],
            mode="markers+text",
            marker=dict(
                size=max(14, min(55, c["paper_count"] // 2)),
                color=colors[i % len(colors)],
                opacity=0.85,
                line=dict(width=1.5, color="#0f1117"),
            ),
            text=[f"<b>{c['label'][:28]}</b>"],
            textposition="top center",
            textfont=dict(color="#c9d1e0", size=10),
            hovertemplate=(
                f"<b>{c['label']}</b><br>"
                f"Papers: {c['paper_count']}<br>"
                f"Silhouette: {c['silhouette_score']:.2f}<br>"
                f"Top terms: {', '.join(c['top_terms'][:4])}<extra></extra>"
            ),
        ))
    fig.update_layout(
        paper_bgcolor="#161b27",
        plot_bgcolor="#161b27",
        xaxis=dict(title="Cluster", gridcolor="#1e2535", tickfont=dict(color="#6b7a99"), title_font=dict(color="#6b7a99")),
        yaxis=dict(title="Silhouette Score", range=[0, 1.15], gridcolor="#1e2535", tickfont=dict(color="#6b7a99"), title_font=dict(color="#6b7a99")),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=20, t=20, b=50),
    )
    return fig


# ─── Render sections ──────────────────────────────────────────────────────────

def render_metrics(result: dict):
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="val">{result.get("papers_retrieved", 0)}</div>
        <div class="lbl">Papers Retrieved</div>
      </div>
      <div class="metric-card">
        <div class="val">{len(result.get("clusters", []))}</div>
        <div class="lbl">Clusters</div>
      </div>
      <div class="metric-card">
        <div class="val">{len(result.get("research_gaps", []))}</div>
        <div class="lbl">Research Gaps</div>
      </div>
      <div class="metric-card">
        <div class="val">{len(result.get("hypotheses", []))}</div>
        <div class="lbl">Hypotheses</div>
      </div>
      <div class="metric-card">
        <div class="val">{len(result.get("suggested_experiments", []))}</div>
        <div class="lbl">Experiments</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_clusters(clusters: list):
    if not clusters:
        st.markdown('<p style="color:#6b7a99;">No clusters found.</p>', unsafe_allow_html=True)
        return
    st.plotly_chart(cluster_chart(clusters), use_container_width=True)
    for c in clusters:
        terms = " &nbsp;·&nbsp; ".join(f'<span class="badge badge-gray">{t}</span>' for t in c["top_terms"][:6])
        sil_color = score_color_hex(c["silhouette_score"])
        st.markdown(f"""
        <div class="card card-accent-blue">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="font-weight:700;color:#c9d1e0;">Cluster {c["cluster_id"]} — {c["label"]}</span>
            <span style="font-size:0.78rem;color:{sil_color};font-weight:600;">Silhouette {c["silhouette_score"]:.2f} &nbsp;|&nbsp; {c["paper_count"]} papers</span>
          </div>
          <div>{terms}</div>
        </div>
        """, unsafe_allow_html=True)


def render_gaps(gaps: list):
    if not gaps:
        st.markdown('<p style="color:#6b7a99;">No gaps detected.</p>', unsafe_allow_html=True)
        return
    for i, gap in enumerate(gaps, 1):
        with st.expander(f"Gap {i} — {gap['gap'][:90]}{'…' if len(gap['gap']) > 90 else ''}", expanded=(i == 1)):
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
              {gap_badge(gap.get('type',''))} &nbsp; {uncertainty_badge(gap.get('uncertainty',''))}
            </div>
            <p style="color:#c9d1e0;line-height:1.6;margin-bottom:0.8rem;">{gap['gap']}</p>
            """, unsafe_allow_html=True)

            if gap.get("reason_underexplored"):
                st.markdown(f"""
                <div class="section-heading">Why Underexplored</div>
                <p style="color:#8b97b0;font-size:0.9rem;">{gap['reason_underexplored']}</p>
                """, unsafe_allow_html=True)

            if gap.get("competing_explanations"):
                items = "".join(f'<li style="color:#8b97b0;font-size:0.88rem;margin-bottom:3px;">{e}</li>' for e in gap["competing_explanations"])
                st.markdown(f'<div class="section-heading">Competing Explanations</div><ul style="margin:0;padding-left:1.2rem;">{items}</ul>', unsafe_allow_html=True)

            if gap.get("evidence_snippets"):
                st.markdown('<div class="section-heading">Evidence</div>', unsafe_allow_html=True)
                for snip in gap["evidence_snippets"]:
                    pmid = snip.get("pmid", "—")
                    text = snip.get("text", "")
                    st.markdown(f"""
                    <div class="evidence">
                      {text}<br>
                      <span class="pmid-tag">PMID {pmid}</span>
                    </div>
                    """, unsafe_allow_html=True)


def render_hypotheses(hypotheses: list):
    if not hypotheses:
        st.markdown('<p style="color:#6b7a99;">No hypotheses generated.</p>', unsafe_allow_html=True)
        return
    for i, hyp in enumerate(hypotheses, 1):
        scores = {
            "Novelty":     hyp.get("novelty_score", 0),
            "Support":     hyp.get("support_score", 0),
            "Feasibility": hyp.get("feasibility_score", 0),
            "Impact":      hyp.get("impact_score", 0),
        }
        composite = sum(scores.values()) / len(scores)
        with st.expander(f"Hypothesis {i} — {hyp['hypothesis'][:90]}{'…' if len(hyp['hypothesis']) > 90 else ''}", expanded=(i == 1)):
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
              {novel_badge(hyp.get('already_established', False))}
              &nbsp;
              <span style="font-size:0.78rem;color:#6b7a99;">Composite score: <b style="color:{score_color_hex(composite)};">{composite:.2f}</b></span>
            </div>
            <p style="color:#c9d1e0;line-height:1.6;margin-bottom:1rem;">{hyp['hypothesis']}</p>
            """, unsafe_allow_html=True)

            col_scores, col_radar = st.columns([1, 1])
            with col_scores:
                st.markdown('<div class="section-heading">Scores</div>', unsafe_allow_html=True)
                bars = "".join(score_bar_html(k, v) for k, v in scores.items())
                st.markdown(bars, unsafe_allow_html=True)
            with col_radar:
                st.plotly_chart(radar_chart(scores), use_container_width=True)

            if hyp.get("reasoning_summary"):
                st.markdown(f"""
                <div class="section-heading">Reasoning</div>
                <p style="color:#8b97b0;font-size:0.88rem;line-height:1.6;">{hyp['reasoning_summary']}</p>
                """, unsafe_allow_html=True)


def render_experiments(experiments: list):
    if not experiments:
        st.markdown('<p style="color:#6b7a99;">No experiments suggested.</p>', unsafe_allow_html=True)
        return
    for i, exp in enumerate(experiments, 1):
        modality  = exp.get("modality", "—").replace("_", " ").title()
        complexity = exp.get("complexity", "—")
        assays    = exp.get("assays", [])
        models    = exp.get("required_models", [])
        assay_str = " &nbsp;·&nbsp; ".join(f'<code style="background:#1e2535;color:#c9d1e0;padding:1px 6px;border-radius:4px;font-size:0.8rem;">{a}</code>' for a in assays)
        model_str = " &nbsp;·&nbsp; ".join(f'<code style="background:#1e2535;color:#c9d1e0;padding:1px 6px;border-radius:4px;font-size:0.8rem;">{m}</code>' for m in models)
        with st.expander(f"Experiment {i} — {exp['experiment'][:90]}{'…' if len(exp['experiment']) > 90 else ''}", expanded=(i <= 2)):
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
              <span class="badge badge-blue">{modality}</span>
              &nbsp;{complexity_badge(complexity)}
            </div>
            <p style="color:#c9d1e0;line-height:1.6;margin-bottom:0.8rem;">{exp['experiment']}</p>
            """, unsafe_allow_html=True)
            if assays:
                st.markdown(f'<div class="section-heading">Assays</div><div style="margin-bottom:0.6rem;">{assay_str}</div>', unsafe_allow_html=True)
            if models:
                st.markdown(f'<div class="section-heading">Models / Systems</div><div>{model_str}</div>', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:1.2rem;">🔬 ResearchGapFinder</p>', unsafe_allow_html=True)

    query = st.text_area(
        "Research Query",
        placeholder="e.g. TP53 ferroptosis breast cancer",
        height=85,
        help="Gene names, diseases, pathways, or mechanisms.",
    )

    preset = st.selectbox(
        "Quality Preset",
        options=["cheap_fast", "balanced", "max_quality"],
        index=1,
        format_func=lambda x: {"cheap_fast": "⚡ Cheap & Fast", "balanced": "⚖️ Balanced", "max_quality": "🎯 Max Quality"}[x],
    )

    max_papers = st.slider("Max Papers", 50, 5000, 100, 50,
        help="100 recommended for demos. 500+ for real research.")

    col1, col2 = st.columns(2)
    with col1:
        year_start = st.number_input("From", min_value=1990, max_value=2025, value=2015)
    with col2:
        year_end = st.number_input("To", min_value=1990, max_value=2025, value=2025)

    article_types = st.multiselect(
        "Article Types",
        ["research", "review", "meta-analysis", "clinical_trial"],
        default=["research", "review"],
    )

    st.divider()
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)

    analyze_btn = st.button(
        "Run Analysis →",
        type="primary",
        use_container_width=True,
        disabled=not query.strip(),
    )

    st.divider()
    if st.button("Check Backend", use_container_width=True):
        try:
            r = httpx.get(f"{backend_url}/health", timeout=6)
            if r.status_code == 200:
                st.success("Backend online ✓")
            else:
                st.error(f"Status {r.status_code}")
        except Exception as e:
            st.error(f"Unreachable: {e}")

    st.markdown('<p style="font-size:0.72rem;color:#3a4459;margin-top:1rem;">Built by Bulut Hamali</p>', unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-title">Research<span>Gap</span>Finder</div>
  <div class="hero-sub">
    Scan biomedical literature, detect knowledge gaps, and generate mechanistic hypotheses — end to end.
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Main content ─────────────────────────────────────────────────────────────

if analyze_btn and query.strip():
    payload = {
        "query": query.strip(),
        "max_papers": max_papers,
        "year_range": [int(year_start), int(year_end)],
        "article_types": article_types,
        "llm_preset": preset,
    }

    steps = [
        (8,  "Expanding query with MeSH synonyms…"),
        (20, "Retrieving papers from PubMed…"),
        (38, "Cleaning abstracts and extracting concepts…"),
        (54, "Generating embeddings…"),
        (65, "Clustering papers…"),
        (76, "Detecting research gaps…"),
        (88, "Generating and scoring hypotheses…"),
        (95, "Designing experiments…"),
    ]

    progress_bar = st.progress(0, text="Initialising…")
    result = None
    error  = None
    done   = threading.Event()

    def _animate():
        for pct, msg in steps:
            if done.is_set(): break
            progress_bar.progress(pct, text=msg)
            time.sleep(4)

    threading.Thread(target=_animate, daemon=True).start()

    try:
        response = httpx.post(f"{backend_url}/analyze", json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
    except httpx.HTTPStatusError as e:
        error = f"Backend error {e.response.status_code}: {e.response.text[:300]}"
    except httpx.ConnectError:
        error = f"Cannot connect to backend at `{backend_url}`."
    except Exception as e:
        error = str(e)
    finally:
        done.set()
        progress_bar.empty()

    if error:
        st.markdown(f'<div class="banner-error">⚠ &nbsp;{error}</div>', unsafe_allow_html=True)
    elif result:
        st.markdown(
            f'<div class="banner-success">✓ &nbsp;Analysis complete for &nbsp;<code style="background:transparent;font-size:0.88rem;">{result["query"]}</code></div>',
            unsafe_allow_html=True,
        )
        render_metrics(result)

        tab_clusters, tab_gaps, tab_hyp, tab_exp, tab_json = st.tabs([
            "Clusters", "Research Gaps", "Hypotheses", "Experiments", "Export",
        ])

        with tab_clusters:
            render_clusters(result.get("clusters", []))

        with tab_gaps:
            render_gaps(result.get("research_gaps", []))

        with tab_hyp:
            render_hypotheses(result.get("hypotheses", []))

        with tab_exp:
            render_experiments(result.get("suggested_experiments", []))

        with tab_json:
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "⬇ Download JSON",
                data=json_str,
                file_name=f"rgf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            st.code(json_str, language="json")

else:
    # ─── Welcome screen ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    boxes = [
        ("🔍", "Gap Detection",
         "Identifies explicit, implicit, missing-link, and contradictory gaps using NLP and statistical signals across clustered literature."),
        ("💡", "Hypothesis Generation",
         "Generates mechanistic, experiment-ready hypotheses scored on novelty, support, feasibility, and clinical impact."),
        ("🧪", "Experiment Design",
         "Maps each hypothesis to specific assays, model systems, and complexity tags — ready to take into the lab."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], boxes):
        col.markdown(f"""
        <div class="feature-box">
          <div class="feature-icon">{icon}</div>
          <div class="feature-title">{title}</div>
          <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7a99;font-size:0.85rem;">Example queries to try:</p>', unsafe_allow_html=True)
    st.markdown("""
    <div>
      <span class="query-chip">TP53 ferroptosis breast cancer</span>
      <span class="query-chip">BRCA1 DNA repair immunotherapy</span>
      <span class="query-chip">mTOR autophagy neurodegeneration</span>
      <span class="query-chip">PD-L1 resistance melanoma</span>
    </div>
    """, unsafe_allow_html=True)
