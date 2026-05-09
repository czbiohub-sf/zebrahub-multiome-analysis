# %% [markdown]
# # Step 07: Cardiac Regulatory Hierarchy
#
# Tests the hypothesis that cardiac cis-regulatory elements show a
# temporal ordering matching the known transcriptional hierarchy:
#
#   GATA4 / TBX5a   (pioneer TFs, cardiac fate commitment)
#       ↓
#   NKX2.5 / HAND2  (homeodomain TFs, chamber specification)
#       ↓
#   MEF2C / TEAD    (structural gene activators)
#       ↓
#   MYL7 / TNNT2A / MYH6  (sarcomere / contractile apparatus)
#
# If this hierarchy is real, peaks near upstream TFs should be MOST SPECIFIC
# at EARLIER timepoints than peaks near downstream structural genes.
#
# Method:
#   1. For each cardiac marker gene, find ALL associated peaks in the
#      640K peak universe (linked_gene / associated_gene / nearest_gene)
#   2. Extract temporal z-score profiles (V2) across heart timepoints
#   3. Compute "peak specificity timepoint" per gene
#   4. Visualize the temporal ordering
#   5. Characterize motifs via leiden_coarse cluster-level maelstrom scores
#
# Outputs:
#   figures/peak_parts_list/cardiac_hierarchy_temporal_order.pdf
#   figures/peak_parts_list/cardiac_hierarchy_profiles.pdf
#   figures/peak_parts_list/cardiac_hierarchy_motifs.pdf
#   outputs/cardiac_hierarchy_peak_table.csv

# %% Imports
import os, re, gc, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

print(f"anndata {ad.__version__}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs"
FIG_DIR = f"{REPO}/figures/peak_parts_list"
OBJ_DIR = f"{BASE}/data/annotated_data/objects_v2"
os.makedirs(FIG_DIR, exist_ok=True)

SPEC_H5AD   = f"{OUTDIR}/specificity_matrix_v2.h5ad"
MOTIF_CSV   = f"{OBJ_DIR}/leiden_by_motifs_maelstrom.csv"
MOTIF_INFO  = f"{OBJ_DIR}/info_cisBP_v2_danio_rerio_motif_factors.csv"

sns.set(style="whitegrid", context="paper")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

TIMEPOINTS_ORDERED = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]
TP_INT = [0, 5, 10, 15, 20, 30]   # integer somite stage for x-axis

# %% Cardiac regulatory hierarchy — gene groups
# Three tiers of the zebrafish cardiac GRN, from Bruneau 2013 and Stainier 1993
CARDIAC_HIERARCHY = {
    "Tier 1 — Pioneer TFs\n(cardiac fate)": {
        "genes":  ["gata4", "gata5", "gata6", "tbx5a", "hand2"],
        "color":  "#c0392b",   # red
        "marker": "D",
    },
    "Tier 2 — Homeodomain TFs\n(chamber specification)": {
        "genes":  ["nkx2.5", "tbx20", "hand1l", "mef2a", "mef2cb"],
        "color":  "#e67e22",   # orange
        "marker": "s",
    },
    "Tier 3 — Structural / Contractile\n(maturation)": {
        "genes":  ["myl7", "tnnt2a", "myh6", "myh7", "tnni1b", "vmhcl", "nppa"],
        "color":  "#2980b9",   # blue
        "marker": "o",
    },
}

# Canonical motif families to track
MOTIF_FAMILIES = {
    "GATA":    ["M10577_2.00", "M02017_2.00", "M09022_2.00", "M05137_2.00"],
    "TBX":     ["M09441_2.00", "M05803_2.00", "M09430_2.00", "M06453_2.00"],
    "NKX/HMX": ["M06375_2.00", "M09563_2.00", "M00294_2.00", "M05897_2.00"],
    "HAND":    ["M02929_2.00"],
    "MEF2":    ["M10937_2.00", "M00684_2.00"],
    "TEAD":    ["M08175_2.00"],
}

# %% Load cluster-level motif scores
print("Loading cluster motif scores ...", flush=True)
motif_df   = pd.read_csv(MOTIF_CSV, index_col=0)   # (n_clusters × n_motifs)
motif_info = pd.read_csv(MOTIF_INFO, index_col=0)

# Build motif-family → cluster score mapping
def cluster_motif_family_score(cluster_id, family_motifs):
    """Max score across family motifs for a given cluster."""
    try:
        cid = int(cluster_id)
        if cid not in motif_df.index:
            return 0.0
        scores = [float(motif_df.loc[cid, m]) for m in family_motifs
                  if m in motif_df.columns]
        return max(scores) if scores else 0.0
    except (ValueError, TypeError):
        return 0.0

print(f"  Motif clusters: {motif_df.shape[0]}  Motifs: {motif_df.shape[1]}")

# %% Load V2 specificity matrix
print("Loading specificity matrix V2 ...", flush=True)
t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

Z   = np.array(Z_adata.X)    # (640830, 190)
obs = Z_adata.obs.copy()
var = Z_adata.var.copy()

_tp_re = re.compile(r'_(\d+somites)$')
var["celltype_name"]  = var.index.to_series().apply(lambda c: _tp_re.sub('', c))
var["timepoint_name"] = var.index.to_series().apply(
    lambda c: m.group(1) if (m := _tp_re.search(c)) else "")

# Heart timepoint indices (ordered)
ct = "heart_myocardium"
tp_order   = [tp for tp in TIMEPOINTS_ORDERED if f"{ct}_{tp}" in var.index]
tp_indices = [int(np.where(var.index == f"{ct}_{tp}")[0][0]) for tp in tp_order]
tp_reliable = [bool(var.iloc[j]["reliable"]) for j in tp_indices]
tp_n_cells  = [int(var.iloc[j]["n_cells"])   for j in tp_indices]
tp_int_vals = [TP_INT[TIMEPOINTS_ORDERED.index(tp)] for tp in tp_order]

Z_heart = Z[:, tp_indices]   # (640830, 6)
print(f"  Heart timepoints: {tp_order}")
print(f"  n_cells: {tp_n_cells}")

# %% Find peaks for each cardiac gene — search all three gene annotation columns
print("\nSearching peaks for each cardiac gene ...", flush=True)
GENE_COLS = ["linked_gene", "associated_gene", "nearest_gene"]

all_genes = []
for tier_name, tier_info in CARDIAC_HIERARCHY.items():
    all_genes.extend(tier_info["genes"])

# Pre-lowercase all gene columns for fast matching
col_lower = {}
for col in GENE_COLS:
    if col in obs.columns:
        col_lower[col] = obs[col].astype(str).str.lower().values

gene_records = []

for tier_name, tier_info in CARDIAC_HIERARCHY.items():
    tier_num = list(CARDIAC_HIERARCHY.keys()).index(tier_name) + 1
    for gene in tier_info["genes"]:
        gene_l = gene.lower()

        # Find all peaks mentioning this gene in any annotation column
        gene_mask = np.zeros(len(obs), dtype=bool)
        for col, vals in col_lower.items():
            gene_mask |= (vals == gene_l)

        n_peaks = gene_mask.sum()
        if n_peaks == 0:
            print(f"  {gene}: 0 peaks found")
            continue

        peak_positions = np.where(gene_mask)[0]
        z_heart_peaks  = Z_heart[peak_positions, :]   # (n_peaks, n_tps)

        # For each peak: max z across heart timepoints (reliable only)
        rel_mask = [r for r, rel in enumerate(tp_reliable) if rel]
        if rel_mask:
            max_z_per_peak = z_heart_peaks[:, rel_mask].max(axis=1)
        else:
            max_z_per_peak = z_heart_peaks.max(axis=1)

        # Filter: keep only peaks with heart specificity z >= 1.5
        specific_mask = max_z_per_peak >= 1.5
        n_specific = specific_mask.sum()

        if n_specific == 0:
            print(f"  {gene}: {n_peaks} peaks found, 0 with z>=1.5")
            continue

        # Best peak per gene (highest max z)
        best_local = np.argmax(max_z_per_peak[specific_mask])
        specific_positions = peak_positions[specific_mask]
        best_pos   = specific_positions[best_local]
        best_z_profile = z_heart_peaks[specific_mask, :][best_local, :]   # (n_tps,)

        # Peak timepoint = timepoint with highest z (reliable only)
        if rel_mask:
            peak_tp_local = rel_mask[int(np.argmax(best_z_profile[rel_mask]))]
        else:
            peak_tp_local = int(np.argmax(best_z_profile))
        peak_tp_name  = tp_order[peak_tp_local]
        peak_tp_int   = tp_int_vals[peak_tp_local]
        peak_max_z    = float(best_z_profile[peak_tp_local])

        # Mean z-score profile across all specific peaks for this gene
        mean_z_profile = z_heart_peaks[specific_mask, :].mean(axis=0)

        # leiden_coarse cluster of the best peak
        leiden_c = str(obs.iloc[best_pos].get("leiden_coarse", ""))
        peak_type = str(obs.iloc[best_pos].get("peak_type", ""))

        # Motif family scores for this peak's cluster
        motif_scores = {fam: cluster_motif_family_score(leiden_c, mids)
                        for fam, mids in MOTIF_FAMILIES.items()}

        gene_records.append({
            "tier":          tier_num,
            "tier_name":     tier_name,
            "gene":          gene,
            "n_peaks_total": n_peaks,
            "n_peaks_spec":  n_specific,
            "peak_tp":       peak_tp_name,
            "peak_tp_int":   peak_tp_int,
            "peak_max_z":    round(peak_max_z, 2),
            "peak_type":     peak_type,
            "leiden_coarse": leiden_c,
            "z_profile":     best_z_profile.tolist(),
            "mean_z_profile":mean_z_profile.tolist(),
            **{f"motif_{k}": v for k, v in motif_scores.items()},
        })
        print(f"  {gene:12s} tier{tier_num}: {n_specific:3d} specific peaks  "
              f"best_z={peak_max_z:.2f}  peak_tp={peak_tp_name}")

gene_df = pd.DataFrame(gene_records)
gene_df.to_csv(f"{OUTDIR}/cardiac_hierarchy_peak_table.csv", index=False)
print(f"\nSaved: {OUTDIR}/cardiac_hierarchy_peak_table.csv  ({len(gene_df)} genes)")

# %% ── Figure 1: Temporal ordering lollipop plot ──────────────────────────────
print("\nFig 1: Temporal ordering lollipop ...", flush=True)

# Sort genes by tier then peak_tp_int
gene_df_sorted = gene_df.sort_values(["tier", "peak_tp_int"]).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(gene_df_sorted) * 0.55)),
                         gridspec_kw={"width_ratios": [3, 2]})

# Left: lollipop — gene vs peak timepoint
ax = axes[0]
tier_colors = {1: "#c0392b", 2: "#e67e22", 3: "#2980b9"}
tier_markers = {1: "D", 2: "s", 3: "o"}

for i, row in gene_df_sorted.iterrows():
    col = tier_colors[row["tier"]]
    mk  = tier_markers[row["tier"]]
    # Stem
    ax.plot([0, row["peak_tp_int"]], [i, i], color=col, lw=1.2, alpha=0.5)
    # Dot sized by z-score
    ax.scatter(row["peak_tp_int"], i, s=row["peak_max_z"] * 25,
               color=col, marker=mk, zorder=5, edgecolors="white", lw=0.5)
    ax.text(row["peak_tp_int"] + 0.5, i, f"  {row['gene']} (z={row['peak_max_z']:.1f})",
            va="center", fontsize=7.5, color=col)

# Tier separators and labels (use positional row indices)
tier_row_ranges = {}
for tier, grp in gene_df_sorted.groupby("tier", sort=True):
    rows = [i for i, (_, r) in enumerate(gene_df_sorted.iterrows()) if r["tier"] == tier]
    tier_row_ranges[tier] = rows

tier_list = sorted(tier_row_ranges.keys())
for t_idx, tier in enumerate(tier_list[:-1]):
    sep_y = max(tier_row_ranges[tier]) + 0.5
    ax.axhline(sep_y, color="gray", lw=0.5, linestyle="--")

for tier in tier_list:
    rows = tier_row_ranges[tier]
    mid_y = np.mean(rows)
    tier_label = [t.split("\n")[0] for t in CARDIAC_HIERARCHY.keys()][tier - 1]
    ax.text(-1, mid_y, tier_label, va="center", ha="right",
            fontsize=7, color=tier_colors[tier], fontweight="bold")

# Unreliable timepoint markers
for tp_int, rel in zip(tp_int_vals, tp_reliable):
    if not rel:
        ax.axvline(tp_int, color="#dddddd", lw=8, alpha=0.5, zorder=0)

ax.set_xticks(tp_int_vals)
ax.set_xticklabels([f"{t}s" for t in tp_int_vals], fontsize=8)
ax.set_yticks([])
ax.set_xlabel("Peak specificity timepoint (somite stage)")
ax.set_xlim(-3, 35)
ax.set_ylim(-0.7, len(gene_df_sorted) - 0.3)
ax.set_title("When does each gene's regulatory\npeaks reach max heart specificity?",
             fontsize=10)

# Add arrow showing hierarchy direction
ax.annotate("", xy=(28, len(gene_df_sorted) - 0.5), xytext=(28, -0.5),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.text(29.5, len(gene_df_sorted) / 2, "Downstream →\n(maturation)",
        va="center", fontsize=6.5, color="gray", rotation=90)

# Right: motif heatmap per gene × motif family
ax2 = axes[1]
motif_cols = [f"motif_{fam}" for fam in MOTIF_FAMILIES.keys()]
motif_matrix = gene_df_sorted[motif_cols].values.astype(float)
motif_labels = list(MOTIF_FAMILIES.keys())

im = ax2.imshow(motif_matrix, aspect="auto", cmap="RdYlBu_r",
                vmin=-2, vmax=3, interpolation="nearest")
ax2.set_xticks(range(len(motif_labels)))
ax2.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
ax2.set_yticks(range(len(gene_df_sorted)))
ax2.set_yticklabels(gene_df_sorted["gene"], fontsize=7.5)
plt.colorbar(im, ax=ax2, label="Cluster maelstrom z-score", fraction=0.05)
ax2.set_title("TF motif enrichment\nin peak's chromatin cluster", fontsize=9)

# Tier separators on motif heatmap
for tier in tier_list[:-1]:
    sep_row = max(tier_row_ranges[tier]) + 0.5
    ax2.axhline(sep_row, color="white", lw=2)

fig.suptitle("Heart Myocardium — Cardiac Regulatory Hierarchy\n"
             "Dot size = max specificity z-score  |  Gray = unreliable timepoint",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_temporal_order.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_temporal_order.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/cardiac_hierarchy_temporal_order.{{pdf,png}}")

# %% ── Figure 2: Z-score temporal profiles per tier ───────────────────────────
print("Fig 2: Temporal profiles per tier ...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
xpos = np.array(tp_int_vals)

for ax, (tier_name, tier_info) in zip(axes, CARDIAC_HIERARCHY.items()):
    tier_num = list(CARDIAC_HIERARCHY.keys()).index(tier_name) + 1
    tier_color = tier_info["color"]
    tier_genes = tier_info["genes"]

    # Collect z-profiles for genes in this tier that were found
    found = gene_df[gene_df["tier"] == tier_num]
    n_found = len(found)

    # Plot individual gene profiles
    all_profiles = []
    for _, row in found.iterrows():
        profile = np.array(row["z_profile"])
        all_profiles.append(profile)
        ax.plot(xpos, profile, color=tier_color, alpha=0.35, lw=1.2)
        # Label the gene at its peak
        best_tp_i = int(np.argmax(profile))
        ax.text(xpos[best_tp_i], profile[best_tp_i] + 0.05,
                row["gene"], fontsize=6, ha="center", color=tier_color, alpha=0.8)

    # Mean profile
    if all_profiles:
        mean_profile = np.mean(all_profiles, axis=0)
        ax.plot(xpos, mean_profile, color=tier_color, lw=2.5,
                label=f"Mean (n={n_found})", zorder=5)

    # Reference lines
    ax.axhline(2, color="#f39c12", lw=0.8, linestyle="--", alpha=0.6, label="z=2")
    ax.axhline(0, color="black",   lw=0.5)

    # Gray bands for unreliable timepoints
    for tp_i, rel in zip(tp_int_vals, tp_reliable):
        if not rel:
            ax.axvspan(tp_i - 2, tp_i + 2, color="#eeeeee", alpha=0.7, zorder=0)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{t}s" for t in tp_int_vals], fontsize=8)
    ax.set_xlabel("Somite stage")
    ax.set_ylabel("Heart specificity z-score (V2)")
    short_name = tier_name.split("—")[1].strip().split("\n")[0]
    ax.set_title(f"Tier {tier_num}: {short_name}", fontsize=9, color=tier_color)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=-0.5)

fig.suptitle("Heart Myocardium — Temporal Specificity Profiles by Regulatory Tier\n"
             "(peaks near each gene; gray = unreliable timepoint)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_profiles.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_profiles.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/cardiac_hierarchy_profiles.{{pdf,png}}")

# %% ── Figure 3: Violin/box — peak timepoint distribution per tier ─────────────
print("Fig 3: Peak timepoint distribution per tier ...", flush=True)

fig, ax = plt.subplots(figsize=(8, 5))

tier_colors_list = ["#c0392b", "#e67e22", "#2980b9"]
tier_labels_short = ["Tier 1\nPioneer TFs\n(gata4/tbx5a)",
                      "Tier 2\nHomeodomain TFs\n(nkx2.5/hand2)",
                      "Tier 3\nStructural genes\n(myl7/tnnt2a)"]

for tier_num in [1, 2, 3]:
    tier_data = gene_df[gene_df["tier"] == tier_num]["peak_tp_int"].values
    x_jitter  = np.random.default_rng(tier_num).uniform(-0.15, 0.15, len(tier_data))
    col = tier_colors_list[tier_num - 1]
    ax.scatter(np.full(len(tier_data), tier_num - 1) + x_jitter, tier_data,
               color=col, s=80, zorder=5, edgecolors="white", lw=0.8,
               label=tier_labels_short[tier_num - 1])
    if len(tier_data) > 0:
        ax.plot([tier_num - 1 - 0.2, tier_num - 1 + 0.2],
                [np.mean(tier_data), np.mean(tier_data)],
                color=col, lw=3, zorder=6)
        ax.errorbar(tier_num - 1, np.mean(tier_data),
                    yerr=np.std(tier_data) if len(tier_data) > 1 else 0,
                    color=col, lw=1.5, capsize=5, zorder=6)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(tier_labels_short, fontsize=8)
ax.set_yticks(tp_int_vals)
ax.set_yticklabels([f"{t}s" for t in tp_int_vals], fontsize=8)
ax.set_ylabel("Timepoint of peak heart specificity (somite stage)")
ax.set_title("Cardiac regulatory hierarchy — temporal ordering of peak specificity\n"
             "Thick bar = mean  |  Error bar = ±1 SD  |  Each dot = one gene's best peak",
             fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Annotate with gene names
for _, row in gene_df.iterrows():
    tier_num = row["tier"]
    x_base   = tier_num - 1
    ax.annotate(row["gene"], (x_base + 0.22, row["peak_tp_int"]),
                fontsize=6, va="center", color=tier_colors_list[tier_num - 1], alpha=0.9)

plt.tight_layout()
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_timing.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIG_DIR}/cardiac_hierarchy_timing.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  Saved: {FIG_DIR}/cardiac_hierarchy_timing.{{pdf,png}}")

# %% Print summary
print("\n" + "=" * 65)
print("CARDIAC REGULATORY HIERARCHY SUMMARY")
print("=" * 65)
for tier_num in [1, 2, 3]:
    tier_data = gene_df[gene_df["tier"] == tier_num]
    tier_name = [t for t in CARDIAC_HIERARCHY.keys()][tier_num - 1]
    print(f"\n{tier_name}")
    for _, row in tier_data.sort_values("peak_tp_int").iterrows():
        print(f"  {row['gene']:12s}  peak_tp={row['peak_tp']:12s}  "
              f"max_z={row['peak_max_z']:.2f}  "
              f"n_specific_peaks={row['n_peaks_spec']}")
    if not tier_data.empty:
        mean_tp = tier_data["peak_tp_int"].mean()
        print(f"  → Mean peak timepoint: {mean_tp:.1f} somites")

print("\nDone.")
