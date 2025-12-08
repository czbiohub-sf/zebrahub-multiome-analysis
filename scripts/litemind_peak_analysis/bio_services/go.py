from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def get_go_enrichment(gene_ids:list) -> dict:
    BASE = "http://api.geneontology.org/api"
    return fetch_json(f"{BASE}/bioentityset/function/enrich",
                      genes=",".join(gene_ids),
                      species="Danio rerio")