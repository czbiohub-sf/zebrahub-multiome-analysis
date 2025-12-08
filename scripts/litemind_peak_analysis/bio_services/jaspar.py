import json

import requests
from arbol import aprint, asection


def fetch_jaspar_motif_info(motif_id: str) -> str:
    """
    Fetch motif information from the JASPAR transcription factor motif database using its RESTful API.
    Use this function to obtain a pretty-printed JSON string containing the motif's metadata (ID, version, TF name, family, class, taxonomic range, UniProt/PubMed cross-refs, sequence-logo URL), a nucleotide frequency matrix (pfm), and optionally an associated first-order TFFM model.

    Parameters
    ----------
    motif_id : str
        The unique identifier for the motif in the JASPAR database. For example: 'MA0659.2'.

    Returns
    -------
    str
        A JSON formatted string containing the motif information.
    """
    try:
        aprint(f"!! Tool use: Fetching motif info for ID '{motif_id}' from JASPAR...")
        info_dict = jaspar_motif_info(motif_id)
        json_str = json.dumps(info_dict, indent=2, ensure_ascii=False, sort_keys=True)
        with asection(f"Fetched motif info for ID '{motif_id}':"):
            aprint(json_str)

        return json_str
    except Exception as e:
        return f"Error fetching motif info for ID '{motif_id}': {e}"


def jaspar_motif_info(motif_id: str, release: str = "2022") -> dict:
    """
    Fetch motif information from the JASPAR database using its RESTful API.
    Parameters
    ----------
    motif_id : str
        The unique identifier for the motif in the JASPAR database. For example, "MA0659.2".
    release : str
        The release version of the JASPAR database to query. Default is "2022".
    """
    if release=='2022':
        url = f"https://jaspar{release}.genereg.net/api/v1/matrix/{motif_id}/"
    elif release=='2024':
        url = f"https://jaspar.elixir.no/api/v1/matrix/{motif_id}/"
    else:
        raise ValueError(f"Unsupported JASPAR release: {release}. Supported releases are '2022' and '2024'.")
    r   = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
    r.raise_for_status()
    return r.json()