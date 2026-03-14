import logging
import math
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from gap_detection.gap_taxonomy import GapType

logger = logging.getLogger(__name__)

# NLP patterns indicating explicit research gaps
EXPLICIT_GAP_PHRASES = [
    r"remains? unknown",
    r"little is known",
    r"poorly understood",
    r"not well understood",
    r"no studies have",
    r"few studies have",
    r"limited studies",
    r"limited evidence",
    r"future studies should",
    r"warrants? (?:further )?investigation",
    r"need(?:s?) (?:to be )?(?:further )?(?:studied|investigated|explored|examined)",
    r"remains? to be (?:elucidated|determined|investigated|established)",
    r"lack(?:s?) (?:of )?(?:evidence|data|studies|research)",
    r"under(?:studied|investigated|explored|researched)",
    r"(?:has|have) not been (?:studied|investigated|explored|examined)",
    r"(?:is|are) not (?:well )?established",
    r"gap(?:s?) in (?:the )?(?:knowledge|literature|understanding|evidence)",
    r"(?:no|limited) (?:clinical|randomized|prospective) (?:trials?|studies?|data)",
    r"contradictory (?:results?|findings?|evidence|data)",
    r"conflicting (?:results?|findings?|evidence|data)",
    r"controversial",
    r"debated",
    r"inconsistent (?:results?|findings?)",
]

COMPILED_GAP_PHRASES = [
    re.compile(pattern, re.IGNORECASE) for pattern in EXPLICIT_GAP_PHRASES
]

# Phrases indicating contradictory findings
CONTRADICTORY_PHRASES = [
    r"contradictory",
    r"conflicting",
    r"controversial",
    r"debated",
    r"inconsistent",
    r"some studies? (?:show|suggest|report|found)",
    r"while others?",
    r"in contrast",
    r"however,? (?:other|some) (?:studies?|reports?|findings?)",
]

COMPILED_CONTRADICTORY = [
    re.compile(p, re.IGNORECASE) for p in CONTRADICTORY_PHRASES
]


class GapScorer:
    """Score candidate research gaps statistically from cluster data."""

    def score(
        self,
        clusters: dict[int, list[dict]],
        cluster_summaries: list[dict],
    ) -> list[dict]:
        """
        Identify and score candidate gaps from cluster data.

        Returns list of candidate gap dicts with type, concepts, evidence_snippets,
        and statistical_score.
        """
        all_papers = [p for papers in clusters.values() for p in papers]
        if not all_papers:
            return []

        candidate_gaps: list[dict] = []

        # 1. Detect explicit gaps from paper text
        explicit_gaps = self._detect_explicit_gaps(all_papers)
        candidate_gaps.extend(explicit_gaps)

        # 2. TF-IDF analysis to find under-represented concepts across clusters
        tfidf_gaps = self._tfidf_analysis(clusters)
        candidate_gaps.extend(tfidf_gaps)

        # 3. Concept co-occurrence analysis
        cooccurrence_gaps = self._compute_concept_cooccurrence(clusters)
        candidate_gaps.extend(cooccurrence_gaps)

        # 4. Missing link detection via ABC concept chains
        concept_graph = self._build_concept_graph(all_papers)
        missing_link_gaps = self._find_missing_links(clusters, concept_graph)
        candidate_gaps.extend(missing_link_gaps)

        # Normalize scores to [0, 1]
        if candidate_gaps:
            max_score = max(g.get("statistical_score", 0.0) for g in candidate_gaps)
            if max_score > 0:
                for g in candidate_gaps:
                    g["statistical_score"] = round(
                        min(1.0, g.get("statistical_score", 0.0) / max_score), 4
                    )

        # Sort by score descending and deduplicate
        candidate_gaps.sort(key=lambda g: g.get("statistical_score", 0.0), reverse=True)
        candidate_gaps = self._deduplicate_gaps(candidate_gaps)

        logger.info(f"GapScorer: {len(candidate_gaps)} candidate gaps identified")
        return candidate_gaps[:20]  # Return top 20

    def _detect_explicit_gaps(self, papers: list[dict]) -> list[dict]:
        """Extract explicit gap statements from paper abstracts."""
        gaps: list[dict] = []

        for paper in papers:
            abstract = paper.get("abstract", "")
            if not abstract:
                continue

            pmid = paper.get("pmid", "")
            sentences = re.split(r"(?<=[.!?])\s+", abstract)

            for sentence in sentences:
                is_contradictory = any(p.search(sentence) for p in COMPILED_CONTRADICTORY)
                is_explicit_gap = any(p.search(sentence) for p in COMPILED_GAP_PHRASES)

                if is_explicit_gap or is_contradictory:
                    gap_type = GapType.CONTRADICTORY.value if is_contradictory else GapType.EXPLICIT.value
                    # Count how many patterns match to score severity
                    match_count = sum(1 for p in COMPILED_GAP_PHRASES if p.search(sentence))

                    gaps.append({
                        "type": gap_type,
                        "description": sentence.strip(),
                        "concepts": self._extract_concepts_from_text(sentence),
                        "evidence_snippets": [{"pmid": pmid, "text": sentence.strip()}],
                        "statistical_score": float(match_count) * 1.5,
                        "source": "explicit_nlp",
                    })

        # Merge similar explicit gaps
        return self._merge_similar_gaps(gaps)

    def _tfidf_analysis(self, clusters: dict[int, list[dict]]) -> list[dict]:
        """
        Identify concepts prevalent in some clusters but absent in others.

        These represent implicit gaps (expected concept not covered).
        """
        if len(clusters) < 2:
            return []

        gaps: list[dict] = []
        cluster_ids = sorted(clusters.keys())

        # Build per-cluster document
        cluster_texts = []
        for cid in cluster_ids:
            text = " ".join(
                f"{p.get('title', '')} {p.get('abstract', '')}"
                for p in clusters[cid]
            )
            cluster_texts.append(text)

        try:
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_texts).toarray()
            feature_names = vectorizer.get_feature_names_out()

            # Find terms with high variance across clusters
            variances = np.var(tfidf_matrix, axis=0)
            high_var_idx = np.argsort(variances)[::-1][:50]

            for idx in high_var_idx:
                term = feature_names[idx]
                scores_per_cluster = tfidf_matrix[:, idx]
                # Find clusters where term is present and absent
                present_clusters = [cluster_ids[i] for i, s in enumerate(scores_per_cluster) if s > 0.01]
                absent_clusters = [cluster_ids[i] for i, s in enumerate(scores_per_cluster) if s <= 0.01]

                if present_clusters and absent_clusters:
                    gaps.append({
                        "type": GapType.IMPLICIT.value,
                        "description": f"Concept '{term}' is studied in {len(present_clusters)} cluster(s) "
                                       f"but absent in {len(absent_clusters)} cluster(s)",
                        "concepts": [term],
                        "evidence_snippets": [],
                        "statistical_score": float(variances[idx]) * 10,
                        "source": "tfidf",
                        "present_clusters": present_clusters,
                        "absent_clusters": absent_clusters,
                    })

        except Exception as e:
            logger.warning(f"TF-IDF gap analysis failed: {e}")

        return gaps[:10]

    def _compute_concept_cooccurrence(self, clusters: dict[int, list[dict]]) -> list[dict]:
        """
        Find concept pairs that co-occur in some clusters but not others.
        """
        if len(clusters) < 2:
            return []

        # Build co-occurrence per cluster
        cluster_cooccurrence: dict[int, Counter] = {}
        for cid, papers in clusters.items():
            counter: Counter = Counter()
            for paper in papers:
                concepts = paper.get("concepts", {})
                all_concepts = (
                    concepts.get("genes", []) +
                    concepts.get("diseases", []) +
                    concepts.get("chemicals", [])
                )
                all_concepts = [c.lower() for c in all_concepts if len(c) > 2]
                # Count pairs
                for i in range(len(all_concepts)):
                    for j in range(i + 1, len(all_concepts)):
                        pair = tuple(sorted([all_concepts[i], all_concepts[j]]))
                        counter[pair] += 1
            cluster_cooccurrence[cid] = counter

        # Find pairs present in some clusters but not others
        all_pairs: set = set()
        for counter in cluster_cooccurrence.values():
            all_pairs.update(counter.keys())

        gaps = []
        for pair in list(all_pairs)[:100]:  # Limit for performance
            present_in = [cid for cid, counter in cluster_cooccurrence.items() if counter.get(pair, 0) > 0]
            absent_in = [cid for cid in clusters.keys() if cid not in present_in]

            if present_in and absent_in and len(present_in) >= 1:
                concept_a, concept_b = pair
                total_count = sum(cluster_cooccurrence[cid].get(pair, 0) for cid in present_in)
                gaps.append({
                    "type": GapType.IMPLICIT.value,
                    "description": f"Co-occurrence of '{concept_a}' and '{concept_b}' "
                                   f"in {len(present_in)} cluster(s) but unexplored relationship",
                    "concepts": list(pair),
                    "evidence_snippets": [],
                    "statistical_score": float(total_count) * 0.5,
                    "source": "cooccurrence",
                })

        gaps.sort(key=lambda g: g["statistical_score"], reverse=True)
        return gaps[:8]

    def _build_concept_graph(self, papers: list[dict]) -> dict[str, set[str]]:
        """Build a concept relationship graph from paper concepts."""
        graph: dict[str, set[str]] = defaultdict(set)
        for paper in papers:
            concepts = paper.get("concepts", {})
            all_concepts = (
                concepts.get("genes", []) +
                concepts.get("diseases", []) +
                concepts.get("chemicals", [])
            )
            all_concepts = [c.lower() for c in all_concepts if len(c) > 2]
            for i in range(len(all_concepts)):
                for j in range(len(all_concepts)):
                    if i != j:
                        graph[all_concepts[i]].add(all_concepts[j])
        return graph

    def _find_missing_links(
        self, clusters: dict[int, list[dict]], concept_graph: dict[str, set[str]]
    ) -> list[dict]:
        """
        Find ABC missing link patterns: A->B and B->C exist, but A->C is absent.
        """
        gaps = []
        # Get top concepts (most frequent)
        concept_freq: Counter = Counter()
        for papers in clusters.values():
            for paper in papers:
                concepts = paper.get("concepts", {})
                for c in concepts.get("genes", []) + concepts.get("diseases", []):
                    concept_freq[c.lower()] += 1

        top_concepts = [c for c, _ in concept_freq.most_common(30)]

        missing_links_found = set()
        for a in top_concepts[:15]:
            for b in list(concept_graph.get(a, set()))[:10]:
                if b not in concept_freq:
                    continue
                for c in list(concept_graph.get(b, set()))[:10]:
                    if c == a or c not in concept_freq:
                        continue
                    # A->B and B->C exist
                    # Check if A->C is missing
                    if c not in concept_graph.get(a, set()):
                        link_key = tuple(sorted([a, b, c]))
                        if link_key not in missing_links_found:
                            missing_links_found.add(link_key)
                            score = (concept_freq[a] + concept_freq[b] + concept_freq[c]) / 3.0
                            gaps.append({
                                "type": GapType.MISSING_LINK.value,
                                "description": (
                                    f"Potential missing link: '{a}' is related to '{b}', "
                                    f"and '{b}' is related to '{c}', "
                                    f"but the direct '{a}' \u2192 '{c}' relationship is unexplored"
                                ),
                                "concepts": [a, b, c],
                                "evidence_snippets": [],
                                "statistical_score": score,
                                "source": "missing_link",
                            })

        gaps.sort(key=lambda g: g["statistical_score"], reverse=True)
        return gaps[:5]

    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """Simple concept extraction from a text snippet."""
        # Extract capitalized terms (likely gene names or proper nouns)
        caps = re.findall(r"\b[A-Z][A-Z0-9]{1,8}\b", text)
        # Extract important lowercase terms
        important_lower = re.findall(
            r"\b(?:pathway|signaling|expression|mutation|inhibition|activation|"
            r"regulation|function|mechanism|receptor|kinase|transcription)\b",
            text, re.IGNORECASE
        )
        return list(set(caps[:5] + important_lower[:5]))

    def _merge_similar_gaps(self, gaps: list[dict]) -> list[dict]:
        """Merge gaps with very similar descriptions (de-duplicate)."""
        if not gaps:
            return []
        merged: list[dict] = []
        seen_descriptions: set[str] = set()

        for gap in gaps:
            # Use first 80 chars as a fingerprint
            fp = gap["description"][:80].lower().strip()
            if fp not in seen_descriptions:
                seen_descriptions.add(fp)
                merged.append(gap)
            else:
                # Merge evidence snippets
                for existing in merged:
                    if existing["description"][:80].lower().strip() == fp:
                        existing["evidence_snippets"].extend(gap.get("evidence_snippets", []))
                        existing["statistical_score"] = max(
                            existing["statistical_score"], gap["statistical_score"]
                        )
                        break
        return merged

    def _deduplicate_gaps(self, gaps: list[dict]) -> list[dict]:
        """Remove near-duplicate gaps across sources."""
        seen: set[str] = set()
        unique = []
        for gap in gaps:
            key = gap.get("description", "")[:100].lower()
            if key not in seen:
                seen.add(key)
                unique.append(gap)
        return unique
