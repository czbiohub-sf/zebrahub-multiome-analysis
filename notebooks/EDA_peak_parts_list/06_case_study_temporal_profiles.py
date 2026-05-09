# %% [markdown]
# # Step 06: Case Study — Temporal Accessibility and Specificity Profiles
#
# For 6 highlighted cell types, generate "parts list case study" figures
# demonstrating that top specific peaks are:
#   (1) ACCESSIBLE — log_norm above global mean in the target cell type
#   (2) SPECIFIC   — high z-score profile, distinct from all other conditions
#   (3) ANNOTATED  — near known marker genes
#   (4) REGULATORY — distal / proximal enhancers (non-promoter peak types)
#   (5) MOTIF-RICH — their chromatin cluster harbors relevant TF binding motifs
#
# For each cell type:
#   Panel A: Temporal accessibility line plot (log_norm across timepoints)
#   Panel B: Temporal specificity line plot (z-score across timepoints)
#   Panel C: Peak type distribution for top-100 specific peaks (bar)
#   Panel D: Top TF motifs from leiden_coarse cluster (horizontal bar)
#   + Summary markdown table per cell type
#
# Inputs:
#   outputs/specificity_matrix_v2.h5ad          — z-scores
#   outputs/marker_gene_reverse_lookup_v2.csv   — marker peaks
#   peaks_by_ct_tp_master_anno.h5ad             — raw log_norm accessibility
#   leiden_by_motifs_maelstrom.csv              — cluster × motif enrichment
#   info_cisBP_v2_danio_rerio_motif_factors.csv — motif → TF name

# %% Imports
import os, re, gc, time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
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
FIG_DIR = f"{REPO}/figures/peak_parts_list/case_studies"
OBJ_DIR = f"{BASE}/data/annotated_data/objects_v2"
os.makedirs(FIG_DIR, exist_ok=True)

SPEC_H5AD   = f"{OUTDIR}/specificity_matrix_v2.h5ad"
REV_CSV     = f"{OUTDIR}/marker_gene_reverse_lookup_v2.csv"
MASTER_H5AD = f"{OBJ_DIR}/peaks_by_ct_tp_master_anno.h5ad"
MOTIF_CSV   = f"{OBJ_DIR}/leiden_by_motifs_maelstrom.csv"
MOTIF_INFO  = f"{OBJ_DIR}/info_cisBP_v2_danio_rerio_motif_factors.csv"

sns.set(style="whitegrid", context="paper")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

TIMEPOINTS_ORDERED = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]
TOP_PEAKS_PER_CT   = 5     # marker-gene-associated peaks to profile
TOP_N_SPECIFIC     = 100   # for peak-type distribution chart

TARGET_CELLTYPES = [
    "PSM",
    "heart_myocardium",
    "neural_crest",
    "fast_muscle",
    "primordial_germ_cells",
    "hemangioblasts",
]

# Peak-type classification (derive from obs peak_type column)
PROMOTER_TYPES = {"promoter", "tss", "promoter-tss"}

def classify_peak(peak_type_str):
    pt = str(peak_type_str).lower().strip()
    if pt in PROMOTER_TYPES or "promoter" in pt:
        return "promoter"
    elif pt in ("intronic",):
        return "intronic enhancer"
    elif pt in ("intergenic", "distal", ""):
        return "distal enhancer"
    elif pt in ("exonic",):
        return "exonic"
    else:
        return pt

# %% Load motif data
print("Loading motif data ...", flush=True)
motif_df   = pd.read_csv(MOTIF_CSV, index_col=0)   # (n_clusters × n_motifs)
motif_info = pd.read_csv(MOTIF_INFO, index_col=0)  # motif → TF names

def motif_label(motif_id):
    if motif_id in motif_info.index:
        for col in ["indirect", "factors"]:
            if col in motif_info.columns:
                tfs = str(motif_info.loc[motif_id, col])
                if tfs not in ("nan", ""):
                    return tfs.split(",")[0].strip().split("_")[0].strip()
    return motif_id

def top_motifs_for_cluster(cluster_id, n=3):
    """Return top n motif TF names for a given leiden_coarse cluster."""
    try:
        cid = int(cluster_id)
        if cid in motif_df.index:
            top = motif_df.loc[cid].nlargest(n)
            return [(motif_label(m), float(z)) for m, z in top.items() if z > 0.5]
    except (ValueError, TypeError):
        pass
    return []

# %% Load specificity matrix (V2)
print("Loading specificity matrix V2 ...", flush=True)
t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

Z   = np.array(Z_adata.X)     # (640830, 190) float32
obs = Z_adata.obs.copy()
var = Z_adata.var.copy()

_tp_re = re.compile(r'_(\d+somites)$')
var["celltype_name"]  = var.index.to_series().apply(lambda c: _tp_re.sub('', c))
var["timepoint_name"] = var.index.to_series().apply(
    lambda c: m.group(1) if (m := _tp_re.search(c)) else "")

# Global mean per peak (used as accessibility reference baseline)
global_mean_z = Z.mean(axis=1)   # (640830,) mean z-score across all conditions

# %% Load raw log_norm (backed mode for memory efficiency)
print("Loading master h5ad (backed) ...", flush=True)
t0 = time.time()
adata_raw = ad.read_h5ad(MASTER_H5AD, backed="r")
print(f"  Shape: {adata_raw.shape}  ({time.time()-t0:.1f}s)")

# Compute global mean log_norm per peak across all 190 conditions
# Do this in chunks to avoid loading the full matrix
print("Computing global mean log_norm per peak ...", flush=True)
raw_layer = adata_raw.layers["log_norm"]
if sp.issparse(raw_layer):
    global_mean_acc = np.array(raw_layer.mean(axis=1)).ravel()
else:
    global_mean_acc = np.array(raw_layer).mean(axis=1)
print(f"  global_mean_acc range: {global_mean_acc.min():.3f}–{global_mean_acc.max():.3f}")

# %% Load reverse lookup
print("Loading reverse lookup V2 ...", flush=True)
rev_df = pd.read_csv(REV_CSV)
print(f"  {len(rev_df)} marker-peak pairs across {rev_df['celltype'].nunique()} cell types")

# %% Markdown report collector
md_sections = []

# %% ── Main loop ────────────────────────────────────────────────────────────────
for ct in TARGET_CELLTYPES:
    print(f"\n{'='*65}")
    print(f"Case study: {ct}", flush=True)

    # ── 1. Get timepoint indices for this cell type ──────────────────────────
    ct_var   = var[var["celltype_name"] == ct]
    tp_order = [tp for tp in TIMEPOINTS_ORDERED if tp in ct_var["timepoint_name"].values]
    tp_cond_names  = [f"{ct}_{tp}" for tp in tp_order]
    tp_indices     = [int(np.where(var.index == name)[0][0])
                      for name in tp_cond_names if name in var.index]
    tp_order       = [tp_order[i] for i, name in enumerate(tp_cond_names) if name in var.index]
    tp_reliable    = [bool(var.iloc[j]["reliable"]) for j in tp_indices]
    tp_n_cells     = [int(var.iloc[j]["n_cells"])   for j in tp_indices]
    all_unreliable = not any(tp_reliable)

    n_tps       = len(tp_indices)
    tp_labels   = [tp.replace("somites", "s") for tp in tp_order]
    tp_labels_d = [f"({lb})" if not rel else lb
                   for lb, rel in zip(tp_labels, tp_reliable)]
    xpos = np.arange(n_tps)

    # ── 2. Select top marker peaks from reverse lookup ───────────────────────
    rev_ct = rev_df[rev_df["celltype"] == ct].copy()
    rev_ct = rev_ct[rev_ct["zscore"] >= 1.5].sort_values("zscore", ascending=False)
    selected = rev_ct.head(TOP_PEAKS_PER_CT).reset_index(drop=True)

    if selected.empty:
        print(f"  No marker peaks with z>=1.5, skipping {ct}")
        continue

    print(f"  Selected {len(selected)} marker peaks: {selected['marker_gene'].tolist()}")

    # ── 3. Extract temporal profiles for each selected peak ──────────────────
    peak_colors = plt.cm.tab10(np.linspace(0, 0.9, len(selected)))
    peak_data   = []

    for _, row in selected.iterrows():
        peak_id  = row["peak_id"]
        gene     = row["marker_gene"]

        # row index in obs
        if peak_id not in obs.index:
            print(f"  WARNING: peak {peak_id} not in obs index")
            continue
        peak_pos = obs.index.get_loc(peak_id)
        peak_obs = obs.iloc[peak_pos]

        # z-score profile across this ct's timepoints
        z_profile   = [float(Z[peak_pos, j]) for j in tp_indices]

        # log_norm accessibility profile
        raw_row = adata_raw.layers["log_norm"][peak_pos, :]
        if sp.issparse(raw_row):
            raw_row = raw_row.toarray().ravel()
        else:
            raw_row = np.array(raw_row).ravel()
        acc_profile = [float(raw_row[j]) for j in tp_indices]
        global_acc  = float(global_mean_acc[peak_pos])

        # Peak metadata
        peak_type  = str(peak_obs.get("peak_type", "unknown"))
        chrom      = str(peak_obs.get("chrom", ""))
        start      = int(peak_obs.get("start", 0))
        end        = int(peak_obs.get("end", 0))
        leiden_c   = str(peak_obs.get("leiden_coarse", ""))
        peak_class = classify_peak(peak_type)
        motifs     = top_motifs_for_cluster(leiden_c, n=3)

        peak_data.append({
            "gene":        gene,
            "peak_id":     peak_id,
            "chrom":       chrom,
            "start":       start,
            "end":         end,
            "peak_type":   peak_type,
            "peak_class":  peak_class,
            "leiden_coarse": leiden_c,
            "zscore_ref":  float(row["zscore"]),
            "z_profile":   z_profile,
            "acc_profile": acc_profile,
            "global_acc":  global_acc,
            "motifs":      motifs,
        })

    if not peak_data:
        continue

    # ── 4. Top-100 specific peaks for peak-type distribution ─────────────────
    # Use max z across reliable timepoints
    reliable_idx = [j for j, rel in zip(tp_indices, tp_reliable) if rel] or tp_indices
    Z_ct   = Z[:, reliable_idx]
    max_z  = Z_ct.max(axis=1)
    top100_pos = np.argsort(max_z)[-TOP_N_SPECIFIC:][::-1]

    peak_classes_top = [classify_peak(str(obs.iloc[p].get("peak_type", "")))
                        for p in top100_pos]
    class_counts = pd.Series(peak_classes_top).value_counts()

    # ── 5. Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[5, 5, 3],
                            height_ratios=[3, 2],
                            hspace=0.45, wspace=0.35)

    ax_acc  = fig.add_subplot(gs[0, 0])   # temporal accessibility
    ax_z    = fig.add_subplot(gs[0, 1])   # temporal specificity
    ax_ann  = fig.add_subplot(gs[0, 2])   # annotation table
    ax_type = fig.add_subplot(gs[1, 0])   # peak-type distribution
    ax_motif= fig.add_subplot(gs[1, 1])   # TF motif bar
    ax_leg  = fig.add_subplot(gs[1, 2])   # legend

    # ── Panel A: Accessibility ────────────────────────────────────────────────
    for pd_row, color in zip(peak_data, peak_colors):
        style = "--" if all_unreliable else "-"
        ax_acc.plot(xpos, pd_row["acc_profile"], color=color, lw=2,
                    marker="o", ms=5, linestyle=style,
                    label=f"{pd_row['gene']} ({pd_row['peak_class'][:3]})")
        # Global mean reference (horizontal)
        ax_acc.axhline(pd_row["global_acc"], color=color, lw=0.7,
                       linestyle=":", alpha=0.5)

    # Gray bands for unreliable timepoints
    for xi, rel in enumerate(tp_reliable):
        if not rel:
            ax_acc.axvspan(xi - 0.4, xi + 0.4, color="#eeeeee", alpha=0.6, zorder=0)

    ax_acc.set_xticks(xpos)
    ax_acc.set_xticklabels(tp_labels_d, fontsize=8)
    ax_acc.set_xlabel("Timepoint  (gray = n<20 cells)")
    ax_acc.set_ylabel("log-norm accessibility")
    ax_acc.set_title("A. Accessibility\n(dotted line = global mean)", fontsize=9)

    # ── Panel B: Specificity (z-score) ────────────────────────────────────────
    for pd_row, color in zip(peak_data, peak_colors):
        style = "--" if all_unreliable else "-"
        ax_z.plot(xpos, pd_row["z_profile"], color=color, lw=2,
                  marker="o", ms=5, linestyle=style)
        # label best timepoint
        best_tp_local = int(np.argmax(pd_row["z_profile"]))
        ax_z.annotate(pd_row["gene"],
                      (best_tp_local, pd_row["z_profile"][best_tp_local]),
                      textcoords="offset points", xytext=(4, 3),
                      fontsize=6.5, color=color)

    for z_thresh, col, lbl in [(2, "#f39c12", "z=2"), (4, "#e74c3c", "z=4")]:
        ax_z.axhline(z_thresh, color=col, lw=0.8, linestyle="--", alpha=0.7, label=lbl)
    ax_z.axhline(0, color="black", lw=0.5)

    for xi, rel in enumerate(tp_reliable):
        if not rel:
            ax_z.axvspan(xi - 0.4, xi + 0.4, color="#eeeeee", alpha=0.6, zorder=0)

    ax_z.set_xticks(xpos)
    ax_z.set_xticklabels(tp_labels_d, fontsize=8)
    ax_z.set_xlabel("Timepoint  (gray = n<20 cells)")
    ax_z.set_ylabel("Specificity z-score (V2)")
    ax_z.set_title("B. Specificity\n(V2 shrinkage-corrected)", fontsize=9)
    ax_z.legend(fontsize=7, loc="upper right")

    # ── Panel C: Annotation table ─────────────────────────────────────────────
    ax_ann.axis("off")
    table_rows = []
    for pd_row in peak_data:
        motif_str = ", ".join(f"{m}({z:.1f})" for m, z in pd_row["motifs"]) or "—"
        table_rows.append([
            pd_row["gene"],
            f"{pd_row['chrom']}:{pd_row['start']//1000}k",
            pd_row["peak_class"][:12],
            f"{pd_row['zscore_ref']:.2f}",
            motif_str[:22],
        ])
    col_hdrs = ["Gene", "Coords", "Type", "Z", "Top motifs"]
    tbl = ax_ann.table(cellText=table_rows, colLabels=col_hdrs,
                       cellLoc="left", loc="center",
                       bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.auto_set_column_width([0, 1, 2, 3, 4])
    ax_ann.set_title("C. Peak annotations", fontsize=9)

    # ── Panel D: Peak-type distribution (top 100) ─────────────────────────────
    CLASS_COLORS = {
        "promoter":         "#3498db",
        "proximal enhancer":"#2ecc71",
        "intronic enhancer":"#f39c12",
        "distal enhancer":  "#e74c3c",
        "exonic":           "#9b59b6",
    }
    bar_colors = [CLASS_COLORS.get(c, "#95a5a6") for c in class_counts.index]
    ax_type.barh(range(len(class_counts)), class_counts.values,
                 color=bar_colors, alpha=0.85)
    ax_type.set_yticks(range(len(class_counts)))
    ax_type.set_yticklabels(class_counts.index, fontsize=8)
    ax_type.set_xlabel(f"Count (top {TOP_N_SPECIFIC} specific peaks)")
    ax_type.set_title("D. Peak types\n(top specific peaks)", fontsize=9)
    for xi, n in enumerate(class_counts.values):
        ax_type.text(n + 0.3, xi, str(n), va="center", fontsize=7)

    # ── Panel E: Top TF motifs across selected peaks ──────────────────────────
    # Aggregate motifs across all selected peaks (weight by z-score)
    motif_scores: dict = {}
    for pd_row in peak_data:
        w = pd_row["zscore_ref"]
        for m_name, m_z in pd_row["motifs"]:
            motif_scores[m_name] = motif_scores.get(m_name, 0) + w * m_z
    top_motifs_agg = sorted(motif_scores.items(), key=lambda x: -x[1])[:8]

    if top_motifs_agg:
        m_names = [x[0] for x in top_motifs_agg]
        m_vals  = [x[1] for x in top_motifs_agg]
        ax_motif.barh(range(len(m_names)), m_vals[::-1], color="#8e44ad", alpha=0.8)
        ax_motif.set_yticks(range(len(m_names)))
        ax_motif.set_yticklabels(m_names[::-1], fontsize=8)
        ax_motif.set_xlabel("Weighted motif enrichment\n(cluster z-score × peak z-score)")
        ax_motif.set_title("E. Enriched TF motifs\n(leiden cluster maelstrom z-scores)", fontsize=9)
    else:
        ax_motif.axis("off")
        ax_motif.text(0.5, 0.5, "No motif data", ha="center", va="center",
                      transform=ax_motif.transAxes, fontsize=9)
        ax_motif.set_title("E. Enriched TF motifs", fontsize=9)

    # ── Panel F: Gene-colour legend ───────────────────────────────────────────
    ax_leg.axis("off")
    legend_handles = [
        mpatches.Patch(color=color, label=f"{pd_row['gene']}\n({pd_row['peak_class'][:15]})")
        for pd_row, color in zip(peak_data, peak_colors)
    ]
    if all_unreliable:
        legend_handles.append(
            mpatches.Patch(color="none", label="[!] All timepoints\nunreliable (n<20)")
        )
    ax_leg.legend(handles=legend_handles, fontsize=7, loc="center",
                  title=f"{ct.replace('_',' ')} marker peaks", title_fontsize=8,
                  frameon=True)

    unreliable_note = "\n[ALL TIMEPOINTS UNRELIABLE — n<20 cells]" if all_unreliable else ""
    fig.suptitle(
        f"Parts List Case Study — {ct.replace('_', ' ')}{unreliable_note}\n"
        f"Showing {len(peak_data)} marker-gene-associated peaks with highest specificity (V2)",
        fontsize=12, y=1.01
    )

    out_stem = f"{FIG_DIR}/case_study_{ct}"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_stem}.{{pdf,png}}")

    # ── 6. Markdown section ───────────────────────────────────────────────────
    peak_type_summary = "  ".join(f"{c}: {n}" for c, n in class_counts.items())
    motif_summary = ", ".join(f"**{m}** ({v:.1f})" for m, v in top_motifs_agg[:5])

    table_md = "\n".join(
        f"| **{pd_row['gene']}** | {pd_row['chrom']}:{pd_row['start']:,}–{pd_row['end']:,} "
        f"| {pd_row['peak_class']} | {pd_row['zscore_ref']:.2f} "
        f"| {', '.join(m for m, _ in pd_row['motifs']) or '—'} "
        f"| {tp_labels[int(np.argmax(pd_row['z_profile']))]} |"
        for pd_row in peak_data
    )
    unreliable_warn = (
        "\n> **[!] All timepoints unreliable (n_cells < 20).** "
        "Z-scores are shrinkage-corrected (V2, alpha=20). Interpret with caution.\n"
        if all_unreliable else ""
    )

    section = f"""
## {ct.replace('_', ' ').title()}
{unreliable_warn}
### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
{table_md}

### Peak type distribution (top {TOP_N_SPECIFIC} most specific peaks)
{peak_type_summary}

### Enriched TF motifs (weighted across selected peaks)
{motif_summary if motif_summary else '—'}

> Figure: `figures/peak_parts_list/case_studies/case_study_{ct}.pdf`
"""
    md_sections.append(section)

    # Print summary
    print(f"  Peak types (top {TOP_N_SPECIFIC}): {dict(class_counts)}")
    print(f"  Top motifs: {[m for m, _ in top_motifs_agg[:4]]}")

# %% Close raw h5ad
adata_raw.file.close()
del adata_raw, Z
gc.collect()

# %% Write combined markdown
print("\nWriting case study markdown ...", flush=True)
report_path = f"{OUTDIR}/case_study_report.md"
with open(report_path, "w") as f:
    f.write(f"""# Parts List Case Studies: Temporal Specificity Profiles

**Generated**: `{pd.Timestamp.now().strftime('%Y-%m-%d')}`

**Method**: For each cell type, marker-gene-associated peaks are ranked by
their V2 (shrinkage-corrected) specificity z-score. Temporal profiles show
accessibility (log-norm) and specificity (z-score) across all available
developmental timepoints. The peak-type distribution and TF motif enrichment
demonstrate the regulatory character of the most specific peaks.

---
""")
    f.write("\n".join(md_sections))

print(f"  Saved: {report_path}")
print("\nAll case studies done.")
