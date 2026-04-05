# %% [markdown]
# # Step 04: Marker Gene Validation for Parts List Peaks
#
# For each of the 32 cell types, find the top 20 most specific peaks
# (across all reliable timepoints) and check:
#   (1) What fraction have a linked/associated gene?
#   (2) Are those genes known marker genes for that cell type?
#
# Uses the project's curated marker gene tables:
#   data/table_marker_genes/marker_genes_15somites.csv
#   data/table_marker_genes/marker_genes_30somites.csv
# Plus a hand-curated zebrafish marker gene dictionary per cell type.
#
# Outputs:
#   outputs/marker_validation_per_celltype_v2.csv    — per-cell-type stats
#   outputs/top20_peaks_per_celltype_v2.csv          — all top-20 peaks with gene+marker info
#   figures/peak_parts_list/marker_recall_barplot_v2.pdf
#   figures/peak_parts_list/gene_link_coverage_barplot_v2.pdf
#   figures/peak_parts_list/top_examples_detail_v2.pdf

# %% Imports
import os, re
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print(f"anndata {ad.__version__}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"
FIG_DIR = f"{REPO}/figures/peak_parts_list"
os.makedirs(FIG_DIR, exist_ok=True)

SPEC_H5AD    = f"{OUTDIR}/specificity_matrix_v2.h5ad"
MARKER_15    = f"{BASE}/data/table_marker_genes/marker_genes_15somites.csv"
MARKER_30    = f"{BASE}/data/table_marker_genes/marker_genes_30somites.csv"

sns.set(style="whitegrid", context="paper")
plt.rcParams["pdf.fonttype"] = 42

TOP_N = 20   # top peaks per cell type to evaluate

# Uncharacterized gene name patterns to exclude from "named gene" filter
UNCHARACTERIZED_PREFIXES = ("cr", "bx", "cu", "al", "si:", "zgc:", "cabz",
                             "si:ch", "si:dkey", "si:dkeyp", "si:cab")

def is_named_gene(gene_str):
    """Return True if the gene name looks like a real/named gene."""
    if not gene_str or str(gene_str).lower() in ("nan", "none", ""):
        return False
    g = str(gene_str).lower()
    return not any(g.startswith(p) for p in UNCHARACTERIZED_PREFIXES)

# %% ── Build marker gene dictionary from curated tables ─────────────────────
print("Loading curated marker gene tables ...")

def load_marker_csv(path, marker_cols):
    df = pd.read_csv(path, index_col=0)
    # Normalize column names (strip BOM, whitespace)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    markers_by_annot = {}
    for _, row in df.iterrows():
        annot = str(row.get("annotation_human_readable", "")).strip().lower()
        if not annot or annot == "nan":
            continue
        genes = set()
        for col in marker_cols:
            if col in row.index:
                g = str(row[col]).strip().lower()
                if g and g != "nan":
                    genes.add(g)
        if annot not in markers_by_annot:
            markers_by_annot[annot] = set()
        markers_by_annot[annot].update(genes)
    return markers_by_annot

m15 = load_marker_csv(MARKER_15, ["marker genes", "Unnamed: 5", "Unnamed: 6"])
m30 = load_marker_csv(MARKER_30, ["marker genes", "Unnamed: 7", "Unnamed: 8",
                                    "Unnamed: 9", "Unnamed: 10", "Unnamed: 11"])

# Merge both tables
markers_from_csv = {}
for d in [m15, m30]:
    for k, v in d.items():
        markers_from_csv.setdefault(k, set()).update(v)

# %% ── Hand-curated marker gene dictionary (from zebrafish developmental biology) ──
# Keys match the 32 cell type names in adata.var["annotation_ML_coarse"]
MARKER_GENES = {
    "NMPs": {
        "tbxta", "sox2", "cdx4", "nkx1.2lb", "msgn1", "wnt3a", "cdx1a",
    },
    "PSM": {
        "msgn1", "tbx6l", "tbx16", "ripply1", "ripply2", "mesp-b", "hes6",
        "nrarp-a", "her1", "her7",
    },
    "differentiating_neurons": {
        "elavl3", "elavl4", "neurod1", "neurod4", "neurog1", "snap25a",
        "stmn1b", "tubb5",
    },
    "endocrine_pancreas": {
        "ins", "gcga", "sst2", "isl1", "pax6b", "neurod1", "mnx2b",
        "ptf1a",
    },
    "endoderm": {
        "foxa2", "foxa3", "sox17", "hhex", "cldn15la", "gata5", "gata6",
        "sox32", "casanova",
    },
    "enteric_neurons": {
        "phox2bb", "phox2ba", "ret", "gfra1a", "sox10",
    },
    "epidermis": {
        "krt4", "krt18", "tp63", "grhl3", "foxi3a", "foxi3b", "tpma",
        "krt17", "cdh1",
    },
    "fast_muscle": {
        "myhz1.1", "myhz1.2", "myhz2", "tnni2a.4", "mylpfa", "smyd1b",
        "tnnc2", "myod1", "myog",
    },
    "floor_plate": {
        "shha", "shhb", "foxa2", "ptch1", "ptch2", "nkx2.2a", "arx",
    },
    "hatching_gland": {
        "hce1", "hce2", "ctsl1a", "sgpl1", "he1a", "he1b",
    },
    "heart_myocardium": {
        "gata4", "gata5", "gata6", "tbx5a", "nkx2.5", "myl7", "myh6",
        "tnnt2a", "hand2", "hand1l", "tbx20", "nppa", "vmhcl",
    },
    "hemangioblasts": {
        "tal1", "lmo2", "gfi1aa", "fli1a", "kdrl", "etv2", "gata1a",
    },
    "hematopoietic_vasculature": {
        "gata1a", "klf1", "hbae1.1", "hbae3", "runx1", "tal1", "lmo2",
        "kdrl", "cdh5", "fli1a", "etv2",
    },
    "hindbrain": {
        "hoxb1a", "hoxb2a", "hoxb3a", "krox20", "egr2b", "vhnf1",
        "gbx2", "mafba", "mafbb",
    },
    "lateral_plate_mesoderm": {
        "gata4", "hand2", "tbx5a", "fli1a", "scl", "nkx2.3",
        "lhx9", "irx3a",
    },
    "midbrain_hindbrain_boundary": {
        "en2a", "en2b", "wnt1", "fgf8a", "pax2a", "pax5",
        "her5", "gbx2",
    },
    "muscle": {
        "myhz1.1", "desma", "tnnc1b", "acta1a", "mylpfa", "myod1",
        "myog", "myf5", "mrf4",
    },
    "neural": {
        "sox2", "sox3", "pax3", "zic2a", "zic3", "sox19a", "sox19b",
        "notch1a", "her2", "her4.1",
    },
    "neural_crest": {
        "sox10", "foxd3", "snai1b", "twist1b", "tfap2a", "tfap2c",
        "crestin", "ednrab", "dlx2a", "sox9b", "tfec",
    },
    "neural_floor_plate": {
        "shha", "foxa2", "nkx2.2a", "olig2", "ptch1", "arx",
    },
    "neural_optic": {
        "rx3", "rx1", "vsx2", "six3b", "six6b", "pax6a", "lhx2a",
        "prss56", "vax2",
    },
    "neural_posterior": {
        "cdx4", "cdx1a", "evx1", "evx2", "hoxc9a", "hoxd9a",
    },
    "neural_telencephalon": {
        "emx2", "emx3", "dlx2a", "dlx5a", "lhx2a", "arxa",
        "foxg1a", "tbr1b",
    },
    "neurons": {
        "elavl3", "snap25a", "syp", "syt1a", "syt1b", "nrxn3a",
        "stmn1b", "tubb5",
    },
    "notochord": {
        "col2a1a", "tbxta", "noto", "shha", "col8a1a", "ntla",
        "plcl2b",
    },
    "optic_cup": {
        "rx3", "vsx2", "six3b", "pax6a", "atoh7", "crx", "prss56",
        "lhx2a",
    },
    "pharyngeal_arches": {
        "dlx2a", "dlx3b", "dlx4b", "dlx5a", "hand2", "edn1", "tfap2a",
        "nkx3.2", "barx1",
    },
    "primordial_germ_cells": {
        "nanos3", "dazl", "dnd1", "tdrd7", "ddx4", "piwil1", "dazap2",
    },
    "pronephros": {
        "pax2a", "pax8", "lhx1a", "gata3", "cdh17", "hnf1ba", "hnf4a",
        "slc12a1",
    },
    "somites": {
        "meox1", "paraxis", "myf5", "myog", "tbx24", "lrrc17",
        "mespaa", "mespab",
    },
    "spinal_cord": {
        "dbx2", "evx1", "sim1a", "olig2", "nkx6.1", "pax3a", "en1a",
        "pax2a",
    },
    "tail_bud": {
        "tbxta", "cdx4", "msgn1", "wnt3a", "sfrp1a", "nkx1.2lb",
        "mespaa",
    },
}

# Enrich with CSV-based markers (fuzzy match on annotation_human_readable → cell type)
ANNOT_TO_CT = {
    "somite": "somites",
    "optic vesicle": "neural_optic",
    "optic cup": "optic_cup",
    "paraxial mesoderm": "PSM",
    "neural crest": "neural_crest",
    "endoderm": "endoderm",
    "neural": "neural",
    "epidermis": "epidermis",
    "heart": "heart_myocardium",
    "notochord": "notochord",
    "floor plate": "floor_plate",
    "hindbrain": "hindbrain",
    "pronephros": "pronephros",
    "muscle": "muscle",
    "spinal cord": "spinal_cord",
    "hematopoietic": "hematopoietic_vasculature",
    "hatching gland": "hatching_gland",
}
for annot, ct in ANNOT_TO_CT.items():
    for csv_key, genes in markers_from_csv.items():
        if annot in csv_key and ct in MARKER_GENES:
            MARKER_GENES[ct].update(genes)

print(f"Marker gene dictionary: {len(MARKER_GENES)} cell types")
print(f"  Median markers per cell type: {np.median([len(v) for v in MARKER_GENES.values()]):.0f}")

# %% Load specificity matrix
import time
print("\nLoading specificity matrix ...", flush=True)
t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

Z = np.array(Z_adata.X)  # (640830, 190)
obs = Z_adata.obs.copy()
var = Z_adata.var.copy()

# Parse condition → celltype, timepoint
_tp_re = re.compile(r'_(\d+somites)$')
var["celltype_name"] = var.index.to_series().apply(
    lambda c: _tp_re.sub('', c))
var["timepoint_name"] = var.index.to_series().apply(
    lambda c: m.group(1) if (m := _tp_re.search(c)) else "")

CELLTYPES = sorted(var["celltype_name"].unique())
print(f"\n{len(CELLTYPES)} cell types found")

# %% ── For each cell type: find top-N peaks using max z across reliable timepoints ──
# Strategy: for each peak, take the MAX z-score across all reliable timepoints
# of that cell type. This gives the "best-case" z-score per (peak, celltype).

print("\nComputing per-celltype top peaks ...", flush=True)

def get_gene(row):
    """Return the best available gene name for a peak."""
    for col in ["linked_gene", "associated_gene", "nearest_gene"]:
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                return val, col
    return None, None

records = []          # top-N peaks per cell type
reverse_records = []  # reverse lookup: marker gene → best associated peak z-score

# Pre-build fast lookup arrays (avoid iterrows inside loop)
var_ct_arr       = var["celltype_name"].values      # (190,) str array
var_reliable_arr = var["reliable"].values            # (190,) bool array
var_names_arr    = np.array(var.index.tolist())      # (190,) str array

for ct in CELLTYPES:
    ct_mask      = var_ct_arr == ct
    reliable_idx = np.where(ct_mask & var_reliable_arr)[0]
    if len(reliable_idx) == 0:
        reliable_idx = np.where(ct_mask)[0]
    if len(reliable_idx) == 0:
        continue

    # Best representative timepoint = most cells
    n_cells_ct = var["n_cells"].values[reliable_idx]
    best_tp_idx = reliable_idx[np.argmax(n_cells_ct)]  # single column index
    z_ct = Z[:, best_tp_idx]          # (640830,) — one condition
    condition_name = var_names_arr[best_tp_idx]

    known_markers = MARKER_GENES.get(ct, set())

    # ── Approach 1: Top-N peaks overall (ranked by z), report named genes ──
    top_idx_all = np.argsort(z_ct)[-TOP_N*5:][::-1]  # get 5× more to filter
    count = 0
    for peak_pos in top_idx_all:
        if count >= TOP_N:
            break
        peak_id  = obs.index[peak_pos]
        peak_obs = obs.iloc[peak_pos]
        gene, gene_source = get_gene(peak_obs)
        gene_lower = gene.lower() if gene else None

        # Named gene filter: skip uncharacterized
        has_named_gene = is_named_gene(gene)

        # Marker match
        is_marker = False
        if gene_lower:
            is_marker = gene_lower in known_markers
            if not is_marker:
                is_marker = any(m in gene_lower or gene_lower in m
                                for m in known_markers if len(m) >= 4)

        records.append({
            "celltype":         ct,
            "rank":             count + 1,
            "peak_id":          peak_id,
            "chrom":            str(peak_obs.get("chrom", "")),
            "start":            peak_obs.get("start", ""),
            "end":              peak_obs.get("end", ""),
            "peak_type":        str(peak_obs.get("peak_type", "")),
            "zscore":           float(z_ct[peak_pos]),
            "condition":        condition_name,
            "gene":             gene,
            "gene_source":      gene_source,
            "has_named_gene":   has_named_gene,
            "is_known_marker":  is_marker,
            "leiden_coarse":    str(peak_obs.get("leiden_coarse", "")),
        })
        count += 1

    # ── Approach 2: Reverse lookup — find peaks near known markers ──────────
    for marker_gene in known_markers:
        # find all peaks associated with this gene (any source)
        for col in ["linked_gene", "associated_gene", "nearest_gene"]:
            if col not in obs.columns:
                continue
            match_mask = obs[col].astype(str).str.lower() == marker_gene
            if match_mask.any():
                matched_zscores = z_ct[match_mask.values]
                matched_peak_ids = obs.index[match_mask.values]
                best_pos = np.argmax(matched_zscores)
                reverse_records.append({
                    "celltype":      ct,
                    "condition":     condition_name,
                    "marker_gene":   marker_gene,
                    "peak_id":       matched_peak_ids[best_pos],
                    "gene_source":   col,
                    "zscore":        float(matched_zscores[best_pos]),
                    "n_peaks_found": int(match_mask.sum()),
                })
                break  # use first source found

top_df = pd.DataFrame(records)
rev_df = pd.DataFrame(reverse_records)

print(f"  Top-N table: {len(top_df)} rows")
print(f"  Reverse lookup: {len(rev_df)} marker-peak pairs")

out_path = f"{OUTDIR}/top{TOP_N}_peaks_per_celltype.csv"
top_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

if not rev_df.empty:
    rev_path = f"{OUTDIR}/marker_gene_reverse_lookup_v2.csv"
    rev_df.sort_values(["celltype","zscore"], ascending=[True,False]).to_csv(rev_path, index=False)
    print(f"  Saved: {rev_path}")

# %% ── Compute per-cell-type summary statistics ───────────────────────────────
print("\nComputing per-cell-type summary stats ...", flush=True)

summary_rows = []
for ct in CELLTYPES:
    sub = top_df[top_df["celltype"] == ct]
    if sub.empty:
        continue

    n_with_gene    = sub["gene"].notna().sum()
    n_named        = sub["has_named_gene"].sum()
    n_marker       = sub["is_known_marker"].sum()
    n_linked       = (sub["gene_source"] == "linked_gene").sum()
    n_known_total  = len(MARKER_GENES.get(ct, set()))
    top1_z         = sub.iloc[0]["zscore"] if not sub.empty else 0

    # Reverse lookup stats
    if not rev_df.empty:
        rev_ct = rev_df[rev_df["celltype"] == ct]
        n_markers_with_peak = len(rev_ct)
        n_markers_z2   = (rev_ct["zscore"] >= 2).sum() if not rev_ct.empty else 0
        n_markers_z4   = (rev_ct["zscore"] >= 4).sum() if not rev_ct.empty else 0
        top_marker_hits = rev_ct[rev_ct["zscore"] >= 2]["marker_gene"].tolist()
    else:
        n_markers_with_peak = n_markers_z2 = n_markers_z4 = 0
        top_marker_hits = []

    summary_rows.append({
        "celltype":                ct,
        "n_curated_markers":       n_known_total,
        "top20_with_any_gene":     n_with_gene,
        "top20_with_named_gene":   n_named,
        "top20_pct_named":         round(100 * n_named / TOP_N, 1),
        "top20_linked":            n_linked,
        "top20_marker_hits":       n_marker,
        "top1_zscore":             round(top1_z, 2),
        # Reverse lookup
        "markers_found_in_dataset": n_markers_with_peak,
        "markers_z_ge_2":          n_markers_z2,
        "markers_z_ge_4":          n_markers_z4,
        "marker_recall_pct":       round(100 * n_markers_z2 / max(n_known_total, 1), 1),
        "top_marker_genes_z2":     ", ".join(top_marker_hits[:8]),
        "top20_named_genes":       ", ".join(g for g in sub[sub["has_named_gene"]]["gene"].tolist()),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("marker_recall_pct", ascending=False)
sum_path = f"{OUTDIR}/marker_validation_per_celltype_v2.csv"
summary_df.to_csv(sum_path, index=False)
print(f"  Saved: {sum_path}")

print("\n── Top 10 cell types by marker recall (reverse lookup z ≥ 2) ───────")
print(summary_df[["celltype","n_curated_markers","markers_z_ge_2","markers_z_ge_4",
                   "marker_recall_pct","top20_pct_named","top1_zscore",
                   "top_marker_genes_z2"]].head(10).to_string(index=False))
print("\n── Named genes in top-20 peaks (first 8 cell types) ────────────────")
for _, r in summary_df.head(8).iterrows():
    print(f"  {r['celltype']}: {r['top20_named_genes']}")

# %% ── Figure 1: Named gene coverage (sorted) ───────────────────────────────
print("\nFig 1: Named gene coverage ...", flush=True)

df_s1 = summary_df.sort_values("top20_pct_named", ascending=True)
colors = ["#2c7bb6" if p >= 50 else "#f39c12" if p >= 25 else "#95a5a6"
          for p in df_s1["top20_pct_named"]]

fig, ax = plt.subplots(figsize=(7, 10))
ax.barh(range(len(df_s1)), df_s1["top20_pct_named"], color=colors)
ax.set_yticks(range(len(df_s1)))
ax.set_yticklabels([c.replace("_", " ") for c in df_s1["celltype"]], fontsize=8)
ax.set_xlabel(f"% of top-{TOP_N} specific peaks with a named (characterized) gene")
ax.set_title("Named gene coverage in top-20 specific peaks", fontsize=10)
ax.axvline(50, color="gray", lw=0.8, linestyle="--")
ax.set_xlim(0, 105)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/gene_link_coverage_barplot_v2.{ext}", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/gene_link_coverage_barplot_v2.{{pdf,png}}")

# %% ── Figure 2: Marker recall (reverse lookup) ──────────────────────────────
print("Fig 2: Marker recall (reverse lookup) ...", flush=True)

df_s2 = summary_df.sort_values("marker_recall_pct", ascending=True)
colors2 = ["#2ecc71" if p >= 50 else "#f39c12" if p >= 20 else "#95a5a6"
           for p in df_s2["marker_recall_pct"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# Left: marker recall %
ax = axes[0]
ax.barh(range(len(df_s2)), df_s2["marker_recall_pct"], color=colors2)
ax.set_yticks(range(len(df_s2)))
ax.set_yticklabels([c.replace("_", " ") for c in df_s2["celltype"]], fontsize=8)
ax.set_xlabel("% of known markers with z ≥ 2 associated peak")
ax.set_title("Marker gene recall\n(reverse lookup: peaks near known markers)", fontsize=9)
ax.axvline(50, color="gray", lw=0.8, linestyle="--")

# Right: absolute count z≥2 vs z≥4
ax2 = axes[1]
df_s2b = summary_df.sort_values("markers_z_ge_2", ascending=True)
ax2.barh(range(len(df_s2b)), df_s2b["markers_z_ge_2"],
         color="#3498db", label="z ≥ 2", alpha=0.9)
ax2.barh(range(len(df_s2b)), df_s2b["markers_z_ge_4"],
         color="#e74c3c", label="z ≥ 4", alpha=0.9)
ax2.set_yticks(range(len(df_s2b)))
ax2.set_yticklabels([c.replace("_", " ") for c in df_s2b["celltype"]], fontsize=8)
ax2.set_xlabel("# known marker genes found (with peak at z threshold)")
ax2.set_title("Absolute marker recovery", fontsize=9)
ax2.legend(fontsize=8)

plt.suptitle("Marker gene validation via reverse lookup", fontsize=11, y=1.01)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/marker_recall_barplot_v2.{ext}", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/marker_recall_barplot_v2.{{pdf,png}}")

# %% ── Figure 3: Detailed table panels for top 6 cell types ─────────────────
print("Fig 3: Detail panels for top examples ...", flush=True)

TOP_CT = summary_df.head(6)["celltype"].tolist()

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.ravel()

for ax, ct in zip(axes, TOP_CT):
    sub = top_df[top_df["celltype"] == ct].head(TOP_N)
    known_markers = MARKER_GENES.get(ct, set())

    # Table data
    rows_data = []
    for _, r in sub.iterrows():
        gene = r["gene"] or "—"
        marker_flag = "★" if r["is_known_marker"] else ""
        rows_data.append([
            int(r["rank"]),
            f"{r['chrom']}:{int(r['start'])//1000}k",
            r["peak_type"][:5] if r["peak_type"] else "—",
            f"{r['zscore']:.1f}",
            f"{gene}{marker_flag}"[:20],
        ])

    col_labels = ["#", "Coords", "Type", "Z", "Gene (★=marker)"]
    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Highlight marker rows
    for row_idx, (_, r) in enumerate(sub.iterrows(), 1):
        if r["is_known_marker"]:
            for col_idx in range(len(col_labels)):
                table[(row_idx, col_idx)].set_facecolor("#d5f5d5")

    ax.axis("off")
    n_markers = sub["is_known_marker"].sum()
    n_gene    = sub["gene"].notna().sum()
    ax.set_title(f"{ct.replace('_',' ')}\n"
                 f"{n_gene}/{TOP_N} have genes · {n_markers}/{TOP_N} are known markers",
                 fontsize=9, pad=2)

fig.suptitle(f"Top-{TOP_N} specific peaks per cell type — ★ = known marker gene",
             fontsize=11, y=1.01)
plt.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIG_DIR}/top_examples_detail_v2.{ext}", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/top_examples_detail_v2.{{pdf,png}}")

# %% ── Print final highlight: best biological examples ───────────────────────
print("\n" + "="*65)
print("BIOLOGICAL HIGHLIGHTS — cell types with richest marker-peak evidence")
print("="*65)
for _, row in summary_df.head(8).iterrows():
    ct = row["celltype"]
    print(f"\n{ct.replace('_',' ').upper()}")
    print(f"  Named gene coverage: {row['top20_pct_named']:.0f}% of top-{TOP_N} peaks have named gene")
    print(f"  Marker recall      : {row['markers_z_ge_2']} markers found at z≥2 ({row['marker_recall_pct']}%)")
    print(f"  Marker genes found : {row['top_marker_genes_z2']}")
    print(f"  Top z-score        : {row['top1_zscore']}")

print("\nDone.")
