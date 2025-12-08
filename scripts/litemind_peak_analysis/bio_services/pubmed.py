import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    """Return text inside an XML element or None (helper)."""
    return elem.text.strip() if elem is not None and elem.text else None

def fetch_pubmed_record(pmid:str) -> str:
    """
    Fetches a PubMed article by PMID via NCBI’s EFetch API and returns a nicely formatted multiline string containing the title, author list, journal, publication year, abstract, and the PMID.
    """
    try:

        record = get_pubmed_record_full(pmid)

        title = record.get("title", "No title available")
        authors = ", ".join(record.get("authors", [])) or "No authors available"
        journal = record.get("journal", "No journal available")
        year = record.get("year", "No year available")
        abstract = record.get("abstract", "No abstract available")

        formatted_record = (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Journal: {journal} ({year})\n"
            f"Abstract: {abstract}\n"
            f"PMID: {pmid}"
        )
        return formatted_record
    except Exception as e:
        return f"Error fetching record for PMID '{pmid}': {e}"

def get_pubmed_record_full(
    pmid: str,
    api_key: Optional[str] = None,
    timeout: int = 10,
) -> Dict[str, object]:
    """
    Retrieve basic bibliographic data from PubMed given a PMID.

    Parameters
    ----------
    pmid : str
        PubMed / Medline identifier (e.g. ``"26378223"``).
    api_key : str, optional
        NCBI API key for higher rate limits (see NCBI docs).  Leave *None*
        for casual use.
    timeout : int, optional
        Seconds to wait before aborting the HTTP request.

    Returns
    -------
    dict
        {
            "title":    Optional[str],
            "authors":  List[str],          # each as "Last F."
            "journal":  Optional[str],
            "abstract": Optional[str],
            "year":     Optional[str],
            "pmid":     str
        }

    Raises
    ------
    requests.HTTPError
        If the network call fails or PMID is invalid.
    """
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(_EUTILS, params=params, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    article = root.find(".//PubmedArticle/MedlineCitation/Article")

    # --- title -------------------------------------------------------------
    title = _text_or_none(article.find("ArticleTitle"))

    # --- abstract (may be multipart) ---------------------------------------
    abstract_elems = article.findall("Abstract/AbstractText")
    abstract = "\n".join(_text_or_none(e) or "" for e in abstract_elems).strip() or None

    # --- journal name ------------------------------------------------------
    journal = _text_or_none(article.find("Journal/Title"))

    # --- publication year --------------------------------------------------
    pub_date = article.find("Journal/JournalIssue/PubDate")
    year = _text_or_none(pub_date.find("Year")) or _text_or_none(pub_date.find("MedlineDate"))

    # --- authors -----------------------------------------------------------
    authors: List[str] = []
    for au in article.findall("AuthorList/Author"):
        last = _text_or_none(au.find("LastName"))
        init = _text_or_none(au.find("Initials"))
        if last and init:
            authors.append(f"{last} {init}.")
        elif last:
            authors.append(last)

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "abstract": abstract,
        "year": year,
        "pmid": pmid,
    }


# ── quick demonstration ──────────────────────────────────────────────
if __name__ == "__main__":
    record = get_pubmed_record_full("26378223")          # example PMID
    for k, v in record.items():
        print(f"{k:8}: {v if k != 'authors' else ', '.join(v)}")
