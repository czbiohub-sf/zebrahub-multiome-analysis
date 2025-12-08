from scripts.litemind_peak_analysis.bio_services.core_http import fetch_json


def lookup_ensembl_gene(ensembl_id: str, species: str):
    """
    Retrieves detailed metadata for a gene from the Ensembl REST API.
    Give it an Ensembl gene ID (e.g., ENSDARG00000000001) and a species name (e.g., danio_rerio), and it returns a JSON dictionary containing the gene’s symbol, description, coordinates, biotype, canonical transcript, assembly, and other key attributes.
    In the case of zebrafish, it also includes the ZFIN ID (ZDB-GENE-......-..) and gene name.
    """
    BASE = "https://rest.ensembl.org"
    return fetch_json(f"{BASE}/lookup/id/{ensembl_id}",
                      species=species,
                      expand=True,
                      content_type='application/json')