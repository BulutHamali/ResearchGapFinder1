import html
import logging
import re
import unicodedata
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Common boilerplate phrases to remove from abstracts
BOILERPLATE_PATTERNS = [
    r"copyright\s+©?\s*\d{4}.*",
    r"©\s*\d{4}.*",
    r"all rights reserved.*",
    r"published by elsevier.*",
    r"published by springer.*",
    r"published by wiley.*",
    r"this article is protected by copyright.*",
    r"for correspondence:.*",
    r"electronic supplementary material.*",
    r"supplementary information.*",
    r"the authors declare.*conflict of interest.*",
    r"no conflict of interest.*",
    r"competing interests?:.*",
    r"funding:.*",
    r"acknowledgements?:.*",
    r"this is an open access article.*",
    r"creative commons attribution.*",
    r"pmid:\s*\d+.*",
    r"doi:\s*10\.\S+.*",
]

COMPILED_BOILERPLATE = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in BOILERPLATE_PATTERNS
]

MIN_ABSTRACT_WORDS = 50


class PaperCleaner:
    """Clean and normalize paper metadata for downstream processing."""

    def clean(self, paper: dict) -> Optional[dict]:
        """
        Clean a paper dict. Returns None if the paper should be filtered out.

        Filters papers with abstract shorter than MIN_ABSTRACT_WORDS.
        """
        cleaned = dict(paper)

        # Clean title
        if cleaned.get("title"):
            cleaned["title"] = self._clean_text(cleaned["title"])

        # Clean and validate abstract
        if cleaned.get("abstract"):
            abstract = self._clean_text(cleaned["abstract"])
            abstract = self._remove_boilerplate(abstract)
            cleaned["abstract"] = abstract
        else:
            cleaned["abstract"] = ""

        # Filter short abstracts
        word_count = len(cleaned["abstract"].split())
        if word_count < MIN_ABSTRACT_WORDS:
            logger.debug(
                f"Filtered paper PMID={cleaned.get('pmid', 'unknown')}: "
                f"abstract too short ({word_count} words)"
            )
            return None

        # Clean authors list
        if isinstance(cleaned.get("authors"), list):
            cleaned["authors"] = [
                self._clean_text(a) for a in cleaned["authors"] if a and a.strip()
            ]

        # Clean journal name
        if cleaned.get("journal"):
            cleaned["journal"] = self._clean_text(cleaned["journal"])

        return cleaned

    def clean_batch(self, papers: list[dict]) -> list[dict]:
        """Clean a list of papers, returning only those that pass filters."""
        cleaned_papers = []
        removed = 0
        for paper in papers:
            result = self.clean(paper)
            if result is not None:
                cleaned_papers.append(result)
            else:
                removed += 1
        logger.info(f"PaperCleaner: {len(cleaned_papers)} kept, {removed} filtered out")
        return cleaned_papers

    def _clean_text(self, text: str) -> str:
        """
        Remove HTML tags, normalize whitespace, fix encoding issues.
        """
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML/XML tags
        try:
            soup = BeautifulSoup(text, "lxml")
            text = soup.get_text(separator=" ")
        except Exception:
            # Fallback: simple regex tag removal
            text = re.sub(r"<[^>]+>", " ", text)

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Fix common encoding artifacts
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u00a0", " ")  # non-breaking space

        # Normalize whitespace (collapse multiple spaces, tabs, newlines)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate phrases from abstract text.
        Splits on sentence boundaries and removes boilerplate sentences.
        """
        if not text:
            return ""

        # Apply compiled patterns
        for pattern in COMPILED_BOILERPLATE:
            text = pattern.sub("", text)

        # Remove any remaining sentence fragments starting with common copyright signals
        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered_sentences = []
        for sentence in sentences:
            s_lower = sentence.lower().strip()
            skip = False
            skip_starts = [
                "copyright",
                "© ",
                "all rights reserved",
                "published by",
                "this article",
                "the authors",
                "no competing",
                "conflict of interest",
                "open access",
                "supplementary",
            ]
            for start in skip_starts:
                if s_lower.startswith(start):
                    skip = True
                    break
            if not skip and len(sentence.strip()) > 10:
                filtered_sentences.append(sentence.strip())

        cleaned = " ".join(filtered_sentences)
        # Final whitespace normalization
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
