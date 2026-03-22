# %% [markdown]
# # Step 08: Download 1:1:1 Zebrafish–Mouse–Human Orthologs
#
# Strategy:
#   Primary:  pybiomart (Ensembl BioMart) — tries www, useast mirrors
#   Fallback: Stream-filter Ensembl Compara homologies TSV from FTP
#             then join gene names from local GTF files
#
# Output:
#   {SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv
#   Columns: ensembl_gene_id_zf, gene_name_zf,
#            ensembl_gene_id_mm, gene_name_mm,
#            ensembl_gene_id_hs, gene_name_hs
#
# Expected: ~8,000–12,000 complete 1:1:1 triplets
#
# Env: single-cell-base (interactive, ~10 min with FTP fallback)
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 08_download_orthologs.py

# %% Imports
import os, time, io, gzip
import urllib.request
import pandas as pd

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
OUT_DIR = f"{SCRATCH}/orthologs"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = f"{OUT_DIR}/zebrafish_mouse_human_1to1_orthologs.csv"

# Local GTF paths for gene name lookup (BioMart fallback)
ZF_GTF   = "/hpc/reference/sequencing_alignment/alignment_references/zebrafish_genome_GRCz11/genes/genes.gtf.gz"
MOUSE_GTF = "/hpc/reference/sequencing_alignment/gff_files/Mus_musculus.GRCm38.92.gtf.gz"
HUMAN_GTF = "/hpc/reference/sequencing_alignment/gff_files/Homo_sapiens.GRCh37.75.gtf.gz"

# Ensembl Compara FTP (release 113, protein homologies)
# - danio_rerio file: has homo_sapiens but NOT mus_musculus as homology_species
# - mus_musculus file: has danio_rerio as homology_species → flip to get ZF→MM pairs
COMPARA_ZF_URL = (
    "https://ftp.ensembl.org/pub/release-113/tsv/ensembl-compara/homologies/"
    "danio_rerio/Compara.113.protein_default.homologies.tsv"
)
COMPARA_MM_URL = (
    "https://ftp.ensembl.org/pub/release-113/tsv/ensembl-compara/homologies/"
    "mus_musculus/Compara.113.protein_default.homologies.tsv"
)


# %% ── Helper: gene-name lookup from GTF ────────────────────────────────────────
def gene_names_from_gtf(gtf_path: str) -> dict:
    """
    Returns dict: ensembl_gene_id → gene_name
    Parses the gene_id and gene_name attributes from GTF gene records only.
    """
    print(f"  Parsing gene names from {gtf_path} ...")
    t0 = time.time()
    opener = gzip.open if gtf_path.endswith(".gz") else open
    id2name = {}
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            gene_id = gene_name = ""
            for seg in attrs.split(";"):
                seg = seg.strip()
                if seg.startswith("gene_id "):
                    gene_id = seg.split('"')[1] if '"' in seg else seg.split()[-1]
                elif seg.startswith("gene_name "):
                    gene_name = seg.split('"')[1] if '"' in seg else seg.split()[-1]
            if gene_id:
                id2name[gene_id] = gene_name or gene_id
    print(f"    {len(id2name):,} genes in {time.time()-t0:.1f}s")
    return id2name


# %% ── Method A: pybiomart ─────────────────────────────────────────────────────
def try_biomart() -> pd.DataFrame | None:
    try:
        from pybiomart import Server
    except ImportError:
        print("pybiomart not installed; skipping BioMart method.")
        return None

    mirrors = [
        "http://www.ensembl.org",
        "http://useast.ensembl.org",
    ]
    attributes = [
        "ensembl_gene_id", "external_gene_name",
        "mmusculus_homolog_ensembl_gene", "mmusculus_homolog_associated_gene_name",
        "mmusculus_homolog_orthology_type",
        "hsapiens_homolog_ensembl_gene", "hsapiens_homolog_associated_gene_name",
        "hsapiens_homolog_orthology_type",
    ]
    for mirror in mirrors:
        try:
            print(f"  Trying BioMart mirror: {mirror} ...")
            t0 = time.time()
            server  = Server(host=mirror)
            mart    = server["ENSEMBL_MART_ENSEMBL"]
            dataset = mart["drerio_gene_ensembl"]
            print(f"    Connected in {time.time()-t0:.1f}s — querying ...")
            result = dataset.query(attributes=attributes)
            print(f"    Raw rows: {len(result):,}")

            # Normalise column names
            col_map = {}
            for col in result.columns:
                cl = col.lower()
                if "mmusculus" in cl and "orthology_type" in cl:
                    col_map[col] = "mm_orthology_type"
                elif "hsapiens" in cl and "orthology_type" in cl:
                    col_map[col] = "hs_orthology_type"
                elif "mmusculus" in cl and "stable" in cl:
                    col_map[col] = "ensembl_gene_id_mm"
                elif "mmusculus" in cl and ("gene_name" in cl or "associated" in cl):
                    col_map[col] = "gene_name_mm"
                elif "hsapiens" in cl and "stable" in cl:
                    col_map[col] = "ensembl_gene_id_hs"
                elif "hsapiens" in cl and ("gene_name" in cl or "associated" in cl):
                    col_map[col] = "gene_name_hs"
                elif "gene stable id" in cl or col == "ensembl_gene_id":
                    col_map[col] = "ensembl_gene_id_zf"
                elif "gene name" in cl and "stable" not in cl and "mouse" not in cl and "human" not in cl:
                    col_map[col] = "gene_name_zf"
            result = result.rename(columns=col_map)

            one2one = result[
                (result.get("mm_orthology_type", pd.Series()) == "ortholog_one2one") &
                (result.get("hs_orthology_type", pd.Series()) == "ortholog_one2one")
            ].copy()
            one2one = one2one.dropna(subset=["ensembl_gene_id_zf","ensembl_gene_id_mm","ensembl_gene_id_hs"])
            one2one = one2one[
                (one2one["ensembl_gene_id_mm"] != "") &
                (one2one["ensembl_gene_id_hs"] != "")
            ]
            keep = [c for c in ["ensembl_gene_id_zf","gene_name_zf",
                                  "ensembl_gene_id_mm","gene_name_mm",
                                  "ensembl_gene_id_hs","gene_name_hs"] if c in one2one.columns]
            return one2one[keep].drop_duplicates().reset_index(drop=True)

        except Exception as e:
            print(f"    FAILED: {e}")
    return None


# %% ── Method B: stream Ensembl Compara FTP + local GTF gene names ───────────────
def stream_filter_compara(url: str, target_species: str, flip: bool = False) -> list:
    """
    Stream a Compara homologies TSV from FTP and return a list of (query_id, target_id)
    tuples where homology_type == ortholog_one2one and homology_species == target_species.

    If flip=True, the file has target_species as the query (gene_stable_id) and
    source_species as homology_species — so return (homology_id, gene_stable_id).
    """
    t0 = time.time()
    print(f"  Streaming {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Python/urllib"})
    rows = []
    with urllib.request.urlopen(req, timeout=180) as response:
        header = response.readline().decode("utf-8").rstrip("\n").split("\t")
        ci = {name: i for i, name in enumerate(header)}
        i_gid      = ci["gene_stable_id"]
        i_htype    = ci["homology_type"]
        i_hgid     = ci["homology_gene_stable_id"]
        i_hspecies = ci["homology_species"]
        n = 0
        for raw_line in response:
            parts = raw_line.decode("utf-8").rstrip("\n").split("\t")
            if len(parts) <= max(i_gid, i_htype, i_hgid, i_hspecies):
                continue
            if parts[i_htype] != "ortholog_one2one":
                continue
            if parts[i_hspecies] != target_species:
                continue
            if flip:
                rows.append((parts[i_hgid], parts[i_gid]))  # (ZF_id, MM_id)
            else:
                rows.append((parts[i_gid], parts[i_hgid]))  # (ZF_id, HS_id)
            n += 1
            if n % 2_000 == 0:
                print(f"    ... {n:,} pairs so far  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.1f}s — {len(rows):,} 1:1 pairs")
    return rows


def build_from_compara_ftp() -> pd.DataFrame:
    """
    Stream-filter Ensembl Compara homologies from FTP.
    - ZF ↔ HS: from danio_rerio file (homo_sapiens present as homology_species)
    - ZF ↔ MM: from mus_musculus file (danio_rerio present as homology_species, flip=True)
    Then join gene names from local GTF files.
    """
    print(f"\nFalling back to Ensembl Compara FTP ...")

    print("\n  [1/2] ZF ↔ Human 1:1 orthologs (danio_rerio file) ...")
    hs_rows = stream_filter_compara(COMPARA_ZF_URL, target_species="homo_sapiens", flip=False)

    print("\n  [2/2] ZF ↔ Mouse 1:1 orthologs (mus_musculus file, flipped) ...")
    mm_rows = stream_filter_compara(COMPARA_MM_URL, target_species="danio_rerio", flip=True)

    print(f"\n  1:1 orthologs: mouse={len(mm_rows):,}, human={len(hs_rows):,}")

    # Build DataFrames
    mm_df = pd.DataFrame(mm_rows, columns=["ensembl_gene_id_zf", "ensembl_gene_id_mm"])
    hs_df = pd.DataFrame(hs_rows, columns=["ensembl_gene_id_zf", "ensembl_gene_id_hs"])

    # Inner join: keep genes with BOTH mouse and human 1:1 orthologs
    merged = mm_df.merge(hs_df, on="ensembl_gene_id_zf", how="inner")
    print(f"  Complete 1:1:1 triplets (IDs only): {len(merged):,}")

    # ── Add gene names from local GTFs ────────────────────────────────────────
    print("\nLoading gene names from GTFs ...")
    zf_id2name = gene_names_from_gtf(ZF_GTF)
    mm_id2name = gene_names_from_gtf(MOUSE_GTF)
    hs_id2name = gene_names_from_gtf(HUMAN_GTF)

    merged["gene_name_zf"] = merged["ensembl_gene_id_zf"].map(zf_id2name).fillna("")
    merged["gene_name_mm"] = merged["ensembl_gene_id_mm"].map(mm_id2name).fillna("")
    merged["gene_name_hs"] = merged["ensembl_gene_id_hs"].map(hs_id2name).fillna("")

    # Reorder columns
    merged = merged[[
        "ensembl_gene_id_zf", "gene_name_zf",
        "ensembl_gene_id_mm", "gene_name_mm",
        "ensembl_gene_id_hs", "gene_name_hs",
    ]].drop_duplicates().reset_index(drop=True)

    return merged


# %% ── Run: try BioMart first, then FTP fallback ─────────────────────────────────
print("="*60)
print("Attempting BioMart ...")
one2one = try_biomart()

if one2one is None or len(one2one) < 100:
    print("\nBioMart unavailable or returned too few rows; using FTP fallback ...")
    one2one = build_from_compara_ftp()

print(f"\nFinal 1:1:1 triplets: {len(one2one):,}")
print(f"Sample:\n{one2one.head(10).to_string()}")

# %% ── Spot-check known orthologs ────────────────────────────────────────────────
print("\nSpot-check known orthologs:")
checks = ["gata4", "sox17", "myod1", "tp53", "kdm5b"]
for gene in checks:
    hit = one2one[one2one["gene_name_zf"].str.lower() == gene.lower()]
    if len(hit):
        row = hit.iloc[0]
        print(f"  {gene}: mm={row.get('gene_name_mm','?')}  hs={row.get('gene_name_hs','?')}")
    else:
        print(f"  {gene}: NOT FOUND in triplet table")

# %% ── Save ──────────────────────────────────────────────────────────────────────
one2one.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print("Done.")
