"""
Compute MHB marker genes from the standalone Zebrahub scRNA-seq atlas
(ZF_atlas_v01) and compare against the chromatin-derived MHB markers
computed from the V3 peak parts list.

Why a separate scRNA atlas: the Zebrahub multiome RNA channel has lower
sensitivity than the standalone scRNA dataset, which has cleaner
expression-level marker calls.

Output: notebooks/EDA_peak_parts_list/outputs/V3/marker_gene_queries/
  mhb_rna_markers_zebrahub_v01.csv     # ranked RNA markers
  mhb_chromatin_vs_rna_overlap.csv     # joined comparison
  mhb_marker_overlap_summary.txt       # human-readable summary
"""

import os, re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

REPO   = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
ATLAS  = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/data/annotated_data/ZF_atlas_v01/ZF_atlas_v01.h5ad"
OUTDIR = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/marker_gene_queries"
os.makedirs(OUTDIR, exist_ok=True)

# Pre-existing chromatin-based MHB markers (from peak parts list)
CHROMATIN_CSV = f"{OUTDIR}/mhb_marker_genes_named.csv"

# ─── 1. Load Zebrahub atlas, define MHB cells ─────────────────────────
print("Loading ZF_atlas_v01 ...")
a = ad.read_h5ad(ATLAS)
print(f"  shape: {a.shape}")

# MHB cells = any Midbrain_* annotation with isthmic markers
# (pax2a, pax5, en1/en2 are canonical MHB / isthmic organizer markers)
mhb_clusters = [c for c in a.obs["cell_annotation"].cat.categories
                if c.startswith("Midbrain_")
                and any(m in c.lower() for m in ("pax2a", "pax5", "en2", "en1"))]
print(f"\nMHB clusters used:")
for c in mhb_clusters:
    n = (a.obs["cell_annotation"] == c).sum()
    print(f"  {c}: {n} cells")

mhb_mask = a.obs["cell_annotation"].isin(mhb_clusters)
n_mhb   = mhb_mask.sum()
n_other = (~mhb_mask).sum()
print(f"\nTotal MHB cells: {n_mhb}    Non-MHB cells: {n_other}")

# ─── 2. Normalize → run rank_genes_groups (Wilcoxon) ──────────────────
a.obs["mhb_group"] = np.where(mhb_mask, "MHB", "other")
a.obs["mhb_group"] = a.obs["mhb_group"].astype("category")

# Standard scanpy normalization
print("\nNormalizing (counts → log1p(CP10K)) ...")
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)

print("Running rank_genes_groups (Wilcoxon, MHB vs other) ...")
sc.tl.rank_genes_groups(a, groupby="mhb_group", reference="other",
                        groups=["MHB"], method="wilcoxon")

# Extract result
res = a.uns["rank_genes_groups"]
rna_markers = pd.DataFrame({
    "gene":      [g for g in res["names"]["MHB"]],
    "log2fc":    [v for v in res["logfoldchanges"]["MHB"]],
    "pval":      [v for v in res["pvals"]["MHB"]],
    "pval_adj":  [v for v in res["pvals_adj"]["MHB"]],
    "score":     [v for v in res["scores"]["MHB"]],
})

# Mean expression in MHB vs other (for sanity)
mhb_X    = a[a.obs["mhb_group"] == "MHB"].X
other_X  = a[a.obs["mhb_group"] == "other"].X
mhb_mean   = np.asarray(mhb_X.mean(axis=0)).ravel()
other_mean = np.asarray(other_X.mean(axis=0)).ravel()
gene_to_idx = {g: i for i, g in enumerate(a.var_names)}
rna_markers["mean_MHB"]   = [mhb_mean[gene_to_idx[g]]   if g in gene_to_idx else np.nan for g in rna_markers["gene"]]
rna_markers["mean_other"] = [other_mean[gene_to_idx[g]] if g in gene_to_idx else np.nan for g in rna_markers["gene"]]

# Filter: significant + reasonably specific + meaningfully expressed
sig = (rna_markers["pval_adj"] < 1e-5) & (rna_markers["log2fc"] > 1) & (rna_markers["mean_MHB"] > 0.1)
rna_markers_filt = rna_markers[sig].sort_values("score", ascending=False).reset_index(drop=True)

rna_markers.to_csv(f"{OUTDIR}/mhb_rna_markers_zebrahub_v01.csv", index=False)
print(f"\nRNA markers (significant): {len(rna_markers_filt)} / {len(rna_markers)}")
print(f"Saved: {OUTDIR}/mhb_rna_markers_zebrahub_v01.csv")

# ─── 3. Cross-reference with chromatin-based markers ──────────────────
print(f"\nLoading chromatin-based MHB markers from {CHROMATIN_CSV} ...")
chrom_markers = pd.read_csv(CHROMATIN_CSV)
chrom_markers = chrom_markers.rename(columns={
    "gene": "gene", "best_z": "atac_best_z",
    "median_z": "atac_median_z", "n_peaks": "atac_n_peaks",
})
print(f"  chromatin markers: {len(chrom_markers)}")

# Strip Ensembl-style IDs from RNA list for cleaner joining
ENSEMBL = re.compile(r"^(BX|CR|CT|CU|AL|CABZ|si:|zgc:)|^[A-Z]{2}\d", re.I)
rna_named = rna_markers_filt[~rna_markers_filt["gene"].str.match(ENSEMBL, na=False)].copy()

# Outer join — keep all RNA + all chromatin genes, mark which sources support each
merged = chrom_markers.merge(
    rna_named[["gene", "log2fc", "pval_adj", "score", "mean_MHB", "mean_other"]],
    on="gene", how="outer", indicator=True,
)
merged["in_chromatin"] = merged["atac_best_z"].notna()
merged["in_rna"]       = merged["log2fc"].notna()
merged.drop(columns=["_merge"], inplace=True)

# Sort: shared > RNA-only > chromatin-only, then by combined evidence
def sort_key(row):
    if row["in_chromatin"] and row["in_rna"]:
        # Combined rank — average of best ATAC and Wilcoxon score
        return (-3, -(row.get("atac_best_z", 0) + row.get("score", 0) / 50))
    if row["in_rna"]:
        return (-2, -row.get("score", 0))
    return (-1, -row.get("atac_best_z", 0))
merged["_sort"] = merged.apply(sort_key, axis=1)
merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

merged.to_csv(f"{OUTDIR}/mhb_chromatin_vs_rna_overlap.csv", index=False)
print(f"Saved: {OUTDIR}/mhb_chromatin_vs_rna_overlap.csv")

# ─── 4. Summary ───────────────────────────────────────────────────────
both    = merged[merged["in_chromatin"] & merged["in_rna"]]
rna_only   = merged[~merged["in_chromatin"] & merged["in_rna"]]
chrom_only = merged[merged["in_chromatin"] & ~merged["in_rna"]]

summary_lines = [
    "MHB MARKER GENES — Chromatin (V3 ATAC parts list) vs RNA (Zebrahub v01 atlas)",
    "=" * 78,
    "",
    f"MHB cells used: {n_mhb} (from clusters: {len(mhb_clusters)})",
    f"  Clusters: {', '.join(mhb_clusters)}",
    "",
    f"Chromatin-based markers (named, ATAC V3 parts list):  {len(chrom_markers):>5}",
    f"RNA-based markers (named, padj<1e-5, log2fc>1):       {len(rna_named):>5}",
    "",
    f"  ⊕ Shared (both ATAC + RNA support):  {len(both):>5}  ({100*len(both)/max(len(rna_named),1):.0f}% of RNA)",
    f"  • RNA-only:                          {len(rna_only):>5}",
    f"  • ATAC-only:                         {len(chrom_only):>5}",
    "",
    "─" * 78,
    "TOP 30 SHARED MARKERS (sorted by combined ATAC + RNA evidence):",
    "─" * 78,
    f"{'Gene':<14} {'ATAC z':>7} {'n peaks':>8} {'RNA log2fc':>11} {'RNA padj':>10}",
]
for _, r in both.head(30).iterrows():
    summary_lines.append(
        f"{str(r['gene']):<14} {r['atac_best_z']:>7.2f} {int(r['atac_n_peaks']):>8d} "
        f"{r['log2fc']:>11.2f} {r['pval_adj']:>10.2e}"
    )

summary_lines += ["", "─" * 78,
                  "TOP 20 RNA-ONLY MARKERS (high RNA but no specific MHB peak):",
                  "─" * 78,
                  f"{'Gene':<14} {'log2fc':>7} {'mean_MHB':>9} {'pval_adj':>10}"]
for _, r in rna_only.head(20).iterrows():
    summary_lines.append(
        f"{str(r['gene']):<14} {r['log2fc']:>7.2f} {r['mean_MHB']:>9.3f} {r['pval_adj']:>10.2e}"
    )

summary_lines += ["", "─" * 78,
                  "TOP 20 ATAC-ONLY MARKERS (specific peak but no RNA marker call):",
                  "─" * 78,
                  f"{'Gene':<14} {'ATAC z':>7} {'n peaks':>8}"]
for _, r in chrom_only.head(20).iterrows():
    summary_lines.append(
        f"{str(r['gene']):<14} {r['atac_best_z']:>7.2f} {int(r['atac_n_peaks']):>8d}"
    )

text = "\n".join(summary_lines)
with open(f"{OUTDIR}/mhb_marker_overlap_summary.txt", "w") as f:
    f.write(text + "\n")
print()
print(text)
print(f"\nSaved: {OUTDIR}/mhb_marker_overlap_summary.txt")
