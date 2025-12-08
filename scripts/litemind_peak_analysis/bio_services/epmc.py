from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def search_epmc(query:str):
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    return fetch_json(f"{BASE}/search", query=query,
                      pageSize=10,
                      format="json",
                      synonym="true")