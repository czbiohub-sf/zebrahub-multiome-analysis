from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def get_pathways_for_gene(uniprot_id:str):
    BASE = "https://reactome.org/ContentService/data"
    return fetch_json(f"{BASE}/participants/{uniprot_id}")
