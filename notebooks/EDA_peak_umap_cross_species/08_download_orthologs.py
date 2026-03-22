# %% [markdown]
# # Step 08: Download 1:1:1 Zebrafish–Mouse–Human Orthologs from Ensembl BioMart
#
# Uses pybiomart to query the zebrafish Ensembl dataset (drerio_gene_ensembl)
# for 1:1:1 ortholog triplets with mouse (mmusculus) and human (hsapiens).
#
# Output:
#   {SCRATCH}/orthologs/zebrafish_mouse_human_1to1_orthologs.csv
#   Columns: ensembl_gene_id_zf, gene_name_zf,
#            ensembl_gene_id_mm, gene_name_mm,
#            ensembl_gene_id_hs, gene_name_hs
#
# Expected: ~8,000–12,000 complete 1:1:1 triplets
#
# Env: single-cell-base (interactive, <5 min)
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 08_download_orthologs.py

# %% Imports
import os, time
import pandas as pd

try:
    from pybiomart import Server
except ImportError:
    raise ImportError("pybiomart not found. Install with: pip install pybiomart")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
OUT_DIR = f"{SCRATCH}/orthologs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = f"{OUT_DIR}/zebrafish_mouse_human_1to1_orthologs.csv"

# %% ── Connect to Ensembl BioMart ────────────────────────────────────────────────
print("Connecting to Ensembl BioMart ...")
t0 = time.time()

server   = Server(host="http://www.ensembl.org")
mart     = server["ENSEMBL_MART_ENSEMBL"]
dataset  = mart["drerio_gene_ensembl"]

print(f"Connected in {time.time()-t0:.1f}s")

# %% ── Query: zebrafish gene IDs + mouse + human homologs ─────────────────────────
print("Querying BioMart for ortholog attributes ...")
t1 = time.time()

attributes = [
    # Zebrafish gene
    "ensembl_gene_id",
    "external_gene_name",
    # Mouse homolog
    "mmusculus_homolog_ensembl_gene",
    "mmusculus_homolog_associated_gene_name",
    "mmusculus_homolog_orthology_type",
    # Human homolog
    "hsapiens_homolog_ensembl_gene",
    "hsapiens_homolog_associated_gene_name",
    "hsapiens_homolog_orthology_type",
]

result = dataset.query(attributes=attributes)
print(f"  Raw query: {len(result):,} rows   ({time.time()-t1:.1f}s)")
print(result.head(3))

# %% ── Filter 1:1:1 orthologs ────────────────────────────────────────────────────
print("\nFiltering 1:1:1 orthologs ...")

# Column names as returned by pybiomart (may vary slightly by version)
# Typical names:
#   'Gene stable ID', 'Gene name',
#   'Mouse gene stable ID', 'Mouse gene name', 'Mouse homology type',
#   'Human gene stable ID', 'Human gene name', 'Human homology type'
print("Column names:", result.columns.tolist())

# Normalise column names regardless of pybiomart version
col_map = {}
for col in result.columns:
    cl = col.lower()
    if "mmusculus" in cl and "orthology_type" in cl:
        col_map[col] = "mm_orthology_type"
    elif "hsapiens" in cl and "orthology_type" in cl:
        col_map[col] = "hs_orthology_type"
    elif "mmusculus" in cl and "gene" in cl and "stable" in cl:
        col_map[col] = "ensembl_gene_id_mm"
    elif "mmusculus" in cl and ("gene_name" in cl or "associated" in cl):
        col_map[col] = "gene_name_mm"
    elif "hsapiens" in cl and "gene" in cl and "stable" in cl:
        col_map[col] = "ensembl_gene_id_hs"
    elif "hsapiens" in cl and ("gene_name" in cl or "associated" in cl):
        col_map[col] = "gene_name_hs"
    elif "gene stable id" in cl or col == "ensembl_gene_id":
        col_map[col] = "ensembl_gene_id_zf"
    elif "gene name" in cl and "gene stable" not in cl and "mouse" not in cl and "human" not in cl:
        col_map[col] = "gene_name_zf"

# Fallback: positional rename if pattern didn't catch everything
expected_order = [
    "ensembl_gene_id_zf", "gene_name_zf",
    "ensembl_gene_id_mm", "gene_name_mm", "mm_orthology_type",
    "ensembl_gene_id_hs", "gene_name_hs", "hs_orthology_type",
]
if len(col_map) < 8:
    print("  WARNING: auto column mapping incomplete, using positional rename")
    result.columns = expected_order[: len(result.columns)]
    col_map = {c: c for c in result.columns}

result = result.rename(columns=col_map)
print(f"  Renamed columns: {result.columns.tolist()}")

# Filter 1:1 both
one2one = result[
    (result["mm_orthology_type"] == "ortholog_one2one") &
    (result["hs_orthology_type"] == "ortholog_one2one")
].copy()

print(f"  1:1:1 before dropping empties: {len(one2one):,}")

# Drop rows with missing IDs
one2one = one2one.dropna(subset=["ensembl_gene_id_zf", "ensembl_gene_id_mm", "ensembl_gene_id_hs"])
one2one = one2one[
    (one2one["ensembl_gene_id_mm"] != "") &
    (one2one["ensembl_gene_id_hs"] != "")
]

print(f"  1:1:1 after dropping empties: {len(one2one):,}")

# Keep only needed columns
keep_cols = [
    "ensembl_gene_id_zf", "gene_name_zf",
    "ensembl_gene_id_mm", "gene_name_mm",
    "ensembl_gene_id_hs", "gene_name_hs",
]
keep_cols = [c for c in keep_cols if c in one2one.columns]
one2one = one2one[keep_cols].drop_duplicates().reset_index(drop=True)

print(f"  Final 1:1:1 triplets: {len(one2one):,}")
print(f"\nSample:\n{one2one.head(10).to_string()}")

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
