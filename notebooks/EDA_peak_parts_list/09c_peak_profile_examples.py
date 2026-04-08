# %% [markdown]
# # Script 09c: Peak Profile Bar Grid Examples
#
# For top 5 V3-specific peaks per celltype, generate two-panel figures:
#   Panel A: Celltype accessibility bar plot (mean across timepoints, all 31 celltypes)
#   Panel B: Temporal accessibility bar plot (within the target celltype)
#
# Uses the same color palettes as notebooks/Fig_peak_umap/09_annotate_peak_umap_celltype_timepoints.py
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

# %% Imports
import os, re, time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Publication figure settings (exact pattern from 01_EDA_annotate_peak_umap.py) ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset all rcParams to defaults
_mpl.rcParams['font.family'] = 'Arial'      # 2. explicit Arial font
_mpl.rcParams["pdf.fonttype"] = 42          # 3. editable text in Illustrator
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper") # 4. seaborn (after fonttype)
_mpl.rcParams["savefig.dpi"]  = 300         # 5. DPI re-set after sns.set()
# ────────────────────────────────────────────────────────────────────────────────

print("=== Script 09c: Peak Profile Bar Grid Examples ===")
print(f"Start: {time.strftime('%c')}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/peak_profiles/per_celltype"
os.makedirs(OUTDIR, exist_ok=True)

MASTER_H5AD = (f"{BASE}/data/annotated_data/objects_v2/"
               "peaks_by_ct_tp_master_anno.h5ad")
V3_PEAKS    = (f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/"
               "V3_celltype_level_top_peaks.csv")

# %% Color palettes (from notebook 09)
cell_type_color_dict = {
    'NMPs': '#8dd3c7',
    'PSM': '#008080',
    'differentiating_neurons': '#bebada',
    'endocrine_pancreas': '#fb8072',
    'endoderm': '#80b1d3',
    'enteric_neurons': '#fdb462',
    'epidermis': '#b3de69',
    'fast_muscle': '#df4b9b',
    'floor_plate': '#d9d9d9',
    'hatching_gland': '#bc80bd',
    'heart_myocardium': '#ccebc5',
    'hemangioblasts': '#ffed6f',
    'hematopoietic_vasculature': '#e41a1c',
    'hindbrain': '#377eb8',
    'lateral_plate_mesoderm': '#4daf4a',
    'midbrain_hindbrain_boundary': '#984ea3',
    'muscle': '#ff7f00',
    'neural': '#e6ab02',
    'neural_crest': '#a65628',
    'neural_floor_plate': '#66a61e',
    'neural_optic': '#999999',
    'neural_posterior': '#393b7f',
    'neural_telencephalon': '#fdcdac',
    'neurons': '#cbd5e8',
    'notochord': '#f4cae4',
    'optic_cup': '#c0c000',
    'pharyngeal_arches': '#fff2ae',
    'primordial_germ_cells': '#f1e2cc',
    'pronephros': '#cccccc',
    'somites': '#1b9e77',
    'spinal_cord': '#d95f02',
    'tail_bud': '#7570b3',
}

# Celltype display order (lineage-grouped, from notebook 09)
CELLTYPE_ORDER = [
    'neural', 'neural_optic', 'neural_posterior', 'neural_telencephalon',
    'neurons', 'hindbrain', 'midbrain_hindbrain_boundary', 'optic_cup',
    'spinal_cord', 'differentiating_neurons', 'floor_plate', 'neural_floor_plate',
    'enteric_neurons', 'neural_crest',
    'somites', 'fast_muscle', 'muscle', 'PSM', 'NMPs', 'tail_bud', 'notochord',
    'lateral_plate_mesoderm', 'heart_myocardium', 'hematopoietic_vasculature',
    'hemangioblasts', 'pharyngeal_arches', 'pronephros', 'hatching_gland',
    'endoderm', 'endocrine_pancreas',
    'epidermis',
]

TIMEPOINT_ORDER = ['0somites', '5somites', '10somites',
                   '15somites', '20somites', '30somites']
TP_INT = {tp: int(tp.replace('somites', '')) for tp in TIMEPOINT_ORDER}
TP_LABELS = [tp.replace('somites', 's') for tp in TIMEPOINT_ORDER]

# Timepoint colors (viridis, from notebook 09)
n_tp = len(TIMEPOINT_ORDER)
_viridis = plt.cm.viridis(np.linspace(0, 1, n_tp))
timepoint_colors = dict(zip(TIMEPOINT_ORDER, _viridis))

# Lineage boundaries for separator lines on celltype bar plot
LINEAGE_BOUNDARIES = [14, 21, 28, 30]  # CNS|NC, paraxial|lateral, lateral|endoderm, endo|epiderm

# %% Configuration
FOCAL_CELLTYPES = [
    "fast_muscle", "heart_myocardium", "neural_crest",
    "PSM", "notochord", "epidermis", "hemangioblasts",
]
TOP_N = 5
MIN_CELLS = 20

# %% Load master h5ad
print(f"\nLoading {MASTER_H5AD.split('/')[-1]} ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)  # (640830, 190)
obs = adata.obs
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)")

# %% Parse conditions
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in adata.var_names],
    columns=["celltype", "timepoint"],
    index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS

# %% Build celltype-level mean matrix (for celltype bar plot)
reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()
celltype_mapping = {col: parse_condition(col)[0] for col in adata.var_names}

reliable_celltypes = sorted(set(
    celltype_mapping[col] for col in reliable_groups
    if celltype_mapping[col] != "primordial_germ_cells"
))

ct_mean = {}  # peak_idx → {celltype: mean_accessibility}
# Precompute column indices per celltype
ct_col_indices = {}
for ct in reliable_celltypes:
    cols = [col for col, c in celltype_mapping.items()
            if c == ct and col in reliable_groups]
    ct_col_indices[ct] = [list(adata.var_names).index(c) for c in cols]

# Precompute column indices per (celltype, timepoint) for temporal profiles
ct_tp_col = {}  # (ct, tp) → col_idx in M
for col in adata.var_names:
    ct, tp = parse_condition(col)
    if col in reliable_groups:
        ct_tp_col[(ct, tp)] = list(adata.var_names).index(col)

# %% Load V3 top peaks
print("Loading V3 top peaks ...", flush=True)
top_peaks_df = pd.read_csv(V3_PEAKS, index_col=0)
print(f"  {len(top_peaks_df)} peaks across {top_peaks_df['celltype'].nunique()} celltypes")

# %% Helper: get peak data
def get_peak_profile(peak_idx):
    """Get celltype-level mean accessibility and temporal profile for a peak."""
    row = M[peak_idx]

    # Celltype-level means
    ct_vals = {}
    for ct in reliable_celltypes:
        ct_vals[ct] = np.mean(row[ct_col_indices[ct]])

    return ct_vals

def get_peak_temporal(peak_idx, target_ct):
    """Get temporal accessibility for a peak within target celltype."""
    row = M[peak_idx]
    temporal = {}
    for tp in TIMEPOINT_ORDER:
        key = (target_ct, tp)
        if key in ct_tp_col:
            temporal[tp] = row[ct_tp_col[key]]
    return temporal

# %% Classify temporal trend
def classify_temporal_trend(temporal_vals):
    """Classify the temporal pattern of a peak."""
    tps = sorted(temporal_vals.keys(), key=lambda x: TP_INT[x])
    vals = [temporal_vals[tp] for tp in tps]
    if len(vals) < 3:
        return "insufficient_data", 0.0

    xs = np.array([TP_INT[tp] for tp in tps], dtype=float)
    ys = np.array(vals, dtype=float)

    # Linear regression
    mean_x, mean_y = xs.mean(), ys.mean()
    ss_xy = ((xs - mean_x) * (ys - mean_y)).sum()
    ss_xx = ((xs - mean_x) ** 2).sum()
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    ss_yy = ((ys - mean_y) ** 2).sum()
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0

    # Coefficient of variation
    cv = ys.std() / ys.mean() if ys.mean() > 0 else 0

    if cv < 0.15:
        pattern = "constitutive"
    elif slope > 0 and r_sq > 0.5:
        pattern = "increasing"
    elif slope < 0 and r_sq > 0.5:
        pattern = "decreasing"
    elif ys.argmax() > 0 and ys.argmax() < len(ys) - 1:
        pattern = "transient_peak"
    else:
        pattern = "variable"

    return pattern, r_sq

# %% Generate figures
print("\nGenerating peak profile figures ...", flush=True)

summary_rows = []

for focal_ct in FOCAL_CELLTYPES:
    ct_peaks = top_peaks_df[top_peaks_df["celltype"] == focal_ct]
    if len(ct_peaks) == 0:
        print(f"  {focal_ct}: no peaks — skipping")
        continue

    ct_dir = f"{OUTDIR}/{focal_ct}"
    os.makedirs(ct_dir, exist_ok=True)

    # Take top N
    ct_top = ct_peaks.nlargest(TOP_N, "V3_zscore")

    for rank, (peak_id, row) in enumerate(ct_top.iterrows(), 1):
        # Get peak row index in adata
        peak_iloc = obs.index.get_loc(peak_id)

        ct_vals = get_peak_profile(peak_iloc)
        tp_vals = get_peak_temporal(peak_iloc, focal_ct)

        # Classify temporal trend
        pattern, r_sq = classify_temporal_trend(tp_vals)

        # Gene annotations
        nearest = str(row["nearest_gene"]) if pd.notna(row["nearest_gene"]) else ""
        linked = str(row["linked_gene"]) if pd.notna(row["linked_gene"]) else ""
        gene_label = linked if linked and linked != "nan" else nearest
        zscore = row["V3_zscore"]
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])

        # ── Create two-panel figure ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5),
                                        gridspec_kw={"width_ratios": [3, 1.2]})

        # Panel A: Celltype bar plot
        ct_order_present = [ct for ct in CELLTYPE_ORDER if ct in ct_vals]
        x_vals = [ct_vals[ct] for ct in ct_order_present]
        colors = [cell_type_color_dict.get(ct, "#cccccc") for ct in ct_order_present]
        # Highlight target celltype with edge
        edgecolors = ["black" if ct == focal_ct else "none" for ct in ct_order_present]
        linewidths = [2.0 if ct == focal_ct else 0.5 for ct in ct_order_present]

        bars = ax1.bar(range(len(ct_order_present)), x_vals, color=colors,
                       edgecolor=edgecolors, linewidth=linewidths)
        ax1.set_xticks(range(len(ct_order_present)))
        ax1.set_xticklabels(ct_order_present, rotation=90, fontsize=7)
        ax1.set_ylabel("Mean log-norm accessibility")
        ax1.set_title(f"Celltype profile — {gene_label or peak_id}\n"
                       f"chr{chrom}:{start}-{end}  |  V3 z={zscore:.1f}  |  "
                       f"top celltype: {focal_ct}",
                       fontsize=10)
        ax1.grid(axis="y", alpha=0.3)

        # Lineage separator lines
        for boundary in LINEAGE_BOUNDARIES:
            if boundary < len(ct_order_present):
                ax1.axvline(boundary - 0.5, color="gray", ls="--", lw=0.7, alpha=0.5)

        # Panel B: Temporal bar plot within target celltype
        tp_present = [tp for tp in TIMEPOINT_ORDER if tp in tp_vals]
        tp_x = list(range(len(tp_present)))
        tp_y = [tp_vals[tp] for tp in tp_present]
        tp_colors = [timepoint_colors[tp] for tp in tp_present]
        tp_labels = [tp.replace("somites", "s") for tp in tp_present]

        ax2.bar(tp_x, tp_y, color=tp_colors, edgecolor="none", width=0.7)
        ax2.set_xticks(tp_x)
        ax2.set_xticklabels(tp_labels, fontsize=9)
        ax2.set_ylabel("Log-norm accessibility")
        ax2.set_xlabel(f"{focal_ct} timepoints")
        ax2.set_title(f"Temporal profile — {pattern}\n(R²={r_sq:.2f})", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fname = f"{focal_ct}_rank{rank}_{gene_label or peak_id}"
        fname = fname.replace("/", "_").replace(" ", "_")
        fig.savefig(f"{ct_dir}/{fname}.pdf")
        fig.savefig(f"{ct_dir}/{fname}.png", dpi=300)
        plt.close(fig)

        # Summary row
        summary_rows.append({
            "celltype": focal_ct,
            "rank": rank,
            "peak_id": peak_id,
            "coords": f"chr{chrom}:{start}-{end}",
            "V3_zscore": zscore,
            "nearest_gene": nearest,
            "linked_gene": linked,
            "temporal_pattern": pattern,
            "temporal_R2": r_sq,
            "max_timepoint": max(tp_vals, key=tp_vals.get) if tp_vals else "",
            "min_timepoint": min(tp_vals, key=tp_vals.get) if tp_vals else "",
            "accessibility_range": f"{min(tp_vals.values()):.1f}-{max(tp_vals.values()):.1f}" if tp_vals else "",
        })

    print(f"  {focal_ct}: {min(TOP_N, len(ct_top))} peaks plotted")

# %% Save summary table
summary_df = pd.DataFrame(summary_rows)
summary_path = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_peak_profile_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary: {summary_path}")

# %% Print summary table
print("\n" + "=" * 120)
print("PEAK PROFILE SUMMARY — Top 5 V3-specific peaks per celltype")
print("=" * 120)
print(f"{'Celltype':<25} {'Rank':>4} {'Gene':<15} {'V3_z':>6} {'Pattern':<15} {'R²':>5} "
      f"{'Accessibility Range':<20} {'Coords'}")
print("-" * 120)

for _, r in summary_df.iterrows():
    gene = r["linked_gene"] if r["linked_gene"] and str(r["linked_gene"]) != "nan" else r["nearest_gene"]
    gene = str(gene)[:14] if pd.notna(gene) else ""
    print(f"{r['celltype']:<25} {r['rank']:>4} {gene:<15} {r['V3_zscore']:>6.1f} "
          f"{r['temporal_pattern']:<15} {r['temporal_R2']:>5.2f} "
          f"{r['accessibility_range']:<20} {r['coords']}")

# %% Highlight interesting temporal patterns
print("\n" + "=" * 80)
print("INTERESTING TEMPORAL PATTERNS")
print("=" * 80)

for pattern_type in ["increasing", "decreasing", "transient_peak", "constitutive"]:
    subset = summary_df[summary_df["temporal_pattern"] == pattern_type]
    if len(subset) > 0:
        print(f"\n{pattern_type.upper()} ({len(subset)} peaks):")
        for _, r in subset.iterrows():
            gene = r["linked_gene"] if str(r["linked_gene"]) != "nan" else r["nearest_gene"]
            gene = str(gene) if pd.notna(gene) else "NA"
            print(f"  {r['celltype']:<22} rank {r['rank']}  {gene:<15} "
                  f"z={r['V3_zscore']:.1f}  R²={r['temporal_R2']:.2f}  "
                  f"range={r['accessibility_range']}")

print(f"\nDone.")
print(f"Figures saved to: {OUTDIR}/")
print(f"End: {time.strftime('%c')}")
