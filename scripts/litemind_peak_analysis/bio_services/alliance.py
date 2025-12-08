from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def fetch_alliance_expression_summary(gene_name_or_id: str):
    """
    Fetches a gene-level expression summary from the Alliance of Genome Resources REST API (`/api/gene/{gene_id}/expression-summary`).
    Give it a gene ID or symbol e.g. 'RGD:2129' or 'ZFIN:ZDB-GENE-990415-72', or 'fgf8a' and it returns the JSON report of tissues, stages, and evidence counts for that gene.
    """
    BASE = "https://www.alliancegenome.org"
    return fetch_json(
        f"{BASE}/api/gene/{gene_name_or_id}/expression-summary",
        content_type="application/json"
    )


def fetch_alliance_gene_disease(gene_name_or_id: str):
    """
    Fetches gene–disease associations from the Alliance of Genome Resources
    (`/api/gene/{gene}/disease`).
    Give it a gene ID or symbol (e.g. `HGNC:1097`, `ZFIN:ZDB-GENE-990415-8`,
    or `braf`); it returns the JSON list of diseases and
    annotations linked to that gene.
    """
    BASE = "https://www.alliancegenome.org"
    return fetch_json(
        f"{BASE}/api/gene/{gene_name_or_id}/disease",
        content_type="application/json",
    )
