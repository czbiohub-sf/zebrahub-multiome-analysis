from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def fetch_zfin_gene_aliases(symbol):
    BASE = "https://zfin.org/api"
    return fetch_json(f"{BASE}/gene/{symbol}/alias")
