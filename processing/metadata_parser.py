import logging
import re
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)


class MetadataParser:
    """Parse PubMed XML and EuropePMC JSON responses into standardized paper dicts."""

    def parse_pubmed_xml(self, xml_text: str) -> list[dict]:
        """Parse a PubMed efetch XML response and return a list of paper dicts."""
        papers = []
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except etree.XMLSyntaxError as e:
            logger.error(f"XML parse error: {e}")
            return papers

        for article in root.findall(".//PubmedArticle"):
            paper = self._parse_pubmed_article(article)
            if paper:
                papers.append(paper)

        return papers

    def _parse_pubmed_article(self, article_elem) -> Optional[dict]:
        """Extract fields from a single PubmedArticle element."""
        try:
            medline = article_elem.find("MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text.strip() if pmid_elem is not None else ""
            if not pmid:
                return None

            article = medline.find("Article")
            if article is None:
                return None

            # Title
            title_elem = article.find("ArticleTitle")
            title = self._elem_text(title_elem)

            # Abstract
            abstract_texts = []
            abstract_elem = article.find("Abstract")
            if abstract_elem is not None:
                for text_elem in abstract_elem.findall("AbstractText"):
                    label = text_elem.get("Label", "")
                    text = self._elem_text(text_elem)
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
            abstract = " ".join(abstract_texts)

            # Authors
            authors = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = self._elem_text(author.find("LastName"))
                    fore = self._elem_text(author.find("ForeName"))
                    if last:
                        authors.append(f"{last} {fore}".strip())

            # Journal
            journal_elem = article.find("Journal")
            journal = ""
            if journal_elem is not None:
                journal_title = journal_elem.find("Title")
                journal = self._elem_text(journal_title)

            # Year
            year = self._extract_year(article, medline)

            # DOI
            doi = ""
            for eloc in article.findall("ELocationID"):
                if eloc.get("EIdType") == "doi":
                    doi = eloc.text.strip() if eloc.text else ""
                    break

            # Article type
            article_types = []
            pub_type_list = article.find("PublicationTypeList")
            if pub_type_list is not None:
                for pt in pub_type_list.findall("PublicationType"):
                    if pt.text:
                        article_types.append(pt.text.strip())

            # MeSH terms
            mesh_terms = []
            mesh_heading_list = medline.find("MeshHeadingList")
            if mesh_heading_list is not None:
                for heading in mesh_heading_list.findall("MeshHeading"):
                    descriptor = heading.find("DescriptorName")
                    if descriptor is not None and descriptor.text:
                        mesh_terms.append(descriptor.text.strip())

            # PMC ID from PubmedData
            pmc_id = ""
            pubmed_data = article_elem.find("PubmedData")
            if pubmed_data is not None:
                article_id_list = pubmed_data.find("ArticleIdList")
                if article_id_list is not None:
                    for aid in article_id_list.findall("ArticleId"):
                        if aid.get("IdType") == "pmc":
                            pmc_id = aid.text.strip() if aid.text else ""

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "mesh_terms": mesh_terms,
                "article_type": article_types,
                "pmc_id": pmc_id,
            }

        except Exception as e:
            logger.warning(f"Error parsing PubMed article: {e}")
            return None

    def _elem_text(self, elem) -> str:
        """Safely extract all text from an XML element including sub-elements."""
        if elem is None:
            return ""
        return "".join(elem.itertext()).strip()

    def _extract_year(self, article_elem, medline_elem) -> int:
        """Extract publication year from multiple possible locations."""
        # Try PubDate in Journal
        journal = article_elem.find("Journal")
        if journal is not None:
            journal_issue = journal.find("JournalIssue")
            if journal_issue is not None:
                pub_date = journal_issue.find("PubDate")
                if pub_date is not None:
                    year_elem = pub_date.find("Year")
                    if year_elem is not None and year_elem.text:
                        try:
                            return int(year_elem.text.strip())
                        except ValueError:
                            pass
                    # Try MedlineDate
                    medline_date = pub_date.find("MedlineDate")
                    if medline_date is not None and medline_date.text:
                        year_match = re.search(r"\b(19|20)\d{2}\b", medline_date.text)
                        if year_match:
                            return int(year_match.group(0))

        # Try DateCompleted in MedlineCitation
        for date_tag in ["DateCompleted", "DateRevised"]:
            date_elem = medline_elem.find(date_tag)
            if date_elem is not None:
                year_elem = date_elem.find("Year")
                if year_elem is not None and year_elem.text:
                    try:
                        return int(year_elem.text.strip())
                    except ValueError:
                        pass

        return 0

    def parse_europepmc_json(self, json_data: dict) -> list[dict]:
        """Parse EuropePMC REST API JSON response and return list of paper dicts."""
        papers = []
        results = json_data.get("resultList", {}).get("result", [])

        for result in results:
            paper = self._parse_europepmc_result(result)
            if paper:
                papers.append(paper)

        return papers

    def _parse_europepmc_result(self, result: dict) -> Optional[dict]:
        """Extract fields from a single EuropePMC result entry."""
        try:
            pmid = str(result.get("pmid", "")).strip()
            pmcid = str(result.get("pmcid", "")).strip()
            title = result.get("title", "").strip()
            abstract = result.get("abstractText", "").strip()

            # Authors
            authors = []
            author_string = result.get("authorString", "")
            if author_string:
                authors = [a.strip() for a in author_string.split(",") if a.strip()]

            # Year
            year = 0
            pub_year = result.get("pubYear", "")
            if pub_year:
                try:
                    year = int(str(pub_year).strip())
                except ValueError:
                    pass

            journal = result.get("journalTitle", "").strip()
            doi = result.get("doi", "").strip()

            return {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "journal": journal,
                "doi": doi,
                "mesh_terms": [],  # EuropePMC doesn't return MeSH in basic search
                "article_type": [result.get("pubType", "")],
                "pmc_id": pmcid,
            }

        except Exception as e:
            logger.warning(f"Error parsing EuropePMC result: {e}")
            return None
