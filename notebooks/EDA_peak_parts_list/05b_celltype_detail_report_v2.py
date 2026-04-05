# %% [markdown]
# # Step 05: Celltype Detail Reports
#
# For 7 biologically highlighted cell types, generate:
#   1. Per-celltype figure: temporal z-score heatmap (top-20 peaks × timepoints)
#                           + temporal barplot (n_specific peaks per timepoint)
#   2. Combined markdown summary report
#
# Target cell types:
#   fast_muscle, heart_myocardium, neural_crest, PSM,
#   epidermis, primordial_germ_cells, hemangioblasts

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

SPEC_H5AD = f"{OUTDIR}/specificity_matrix_v2.h5ad"
REV_CSV   = f"{OUTDIR}/marker_gene_reverse_lookup_v2.csv"

sns.set(style="whitegrid", context="paper")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42

TIMEPOINTS_ORDERED = ["0somites", "5somites", "10somites", "15somites", "20somites", "30somites"]
TOP_N = 20

TARGET_CELLTYPES = [
    "fast_muscle",
    "heart_myocardium",
    "neural_crest",
    "PSM",
    "epidermis",
    "primordial_germ_cells",
    "hemangioblasts",
]

# %% Marker gene dictionary (subset used for highlighting)
MARKER_GENES = {
    "fast_muscle": {
        "myhz1.1", "myhz1.2", "myhz2", "tnni2a.4", "mylpfa", "smyd1b",
        "tnnc2", "myod1", "myog",
    },
    "heart_myocardium": {
        "gata4", "gata5", "gata6", "tbx5a", "nkx2.5", "myl7", "myh6",
        "tnnt2a", "hand2", "hand1l", "tbx20", "nppa", "vmhcl", "myh7", "tnni1b",
    },
    "neural_crest": {
        "sox10", "foxd3", "snai1b", "twist1b", "tfap2a", "tfap2c",
        "crestin", "ednrab", "dlx2a", "sox9b", "tfec",
    },
    "PSM": {
        "msgn1", "tbx6l", "tbx16", "ripply1", "ripply2", "hes6",
        "nrarp-a", "her1", "her7", "mesp-b", "myf5",
    },
    "epidermis": {
        "krt4", "krt18", "tp63", "grhl3", "foxi3a", "foxi3b",
        "krt17", "cdh1", "dlx3b", "bmp2b",
    },
    "primordial_germ_cells": {
        "nanos3", "dazl", "dnd1", "tdrd7", "ddx4", "piwil1", "dazap2",
    },
    "hemangioblasts": {
        "tal1", "lmo2", "gfi1aa", "fli1a", "kdrl", "etv2", "gata1a",
    },
}

UNCHARACTERIZED_PREFIXES = ("cr", "bx", "cu", "si:", "zgc:", "cabz",
                             "si:ch", "si:dkey", "si:dkeyp", "si:cab")

def is_named_gene(gene_str):
    if not gene_str or str(gene_str).lower() in ("nan", "none", ""):
        return False
    g = str(gene_str).lower()
    return not any(g.startswith(p) for p in UNCHARACTERIZED_PREFIXES)

def get_gene(row):
    for col in ["linked_gene", "associated_gene", "nearest_gene"]:
        if col in row.index:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                return val, col
    return None, None

def is_marker(gene_str, ct):
    if not gene_str:
        return False
    gl = str(gene_str).lower()
    known = MARKER_GENES.get(ct, set())
    return gl in known or any(m in gl or gl in m for m in known if len(m) >= 4)

# %% Load data
import time
print("Loading specificity matrix ...", flush=True)
t0 = time.time()
Z_adata = ad.read_h5ad(SPEC_H5AD)
print(f"  Shape: {Z_adata.shape}  ({time.time()-t0:.1f}s)")

Z   = np.array(Z_adata.X)        # (640830, 190)
obs = Z_adata.obs.copy()
var = Z_adata.var.copy()

_tp_re = re.compile(r'_(\d+somites)$')
var["celltype_name"] = var.index.to_series().apply(lambda c: _tp_re.sub('', c))
var["timepoint_name"] = var.index.to_series().apply(
    lambda c: m.group(1) if (m := _tp_re.search(c)) else "")

# Load reverse lookup
rev_df = pd.read_csv(REV_CSV) if os.path.exists(REV_CSV) else pd.DataFrame()

# %% ── Main loop ────────────────────────────────────────────────────────────────
md_sections = []   # collect markdown for combined report

for ct in TARGET_CELLTYPES:
    print(f"\n{'='*60}")
    print(f"Processing: {ct}", flush=True)

    # ── 1. Identify reliable timepoints for this cell type ──────────────────
    ct_mask = var["celltype_name"] == ct
    all_tp_rows = var[ct_mask].copy()
    all_tp_rows = all_tp_rows.sort_values(
        "timepoint_name",
        key=lambda s: s.map(lambda x: int(x.replace("somites", "")) if x.replace("somites", "").isdigit() else 999)
    )

    # Ordered available timepoints
    tp_order     = [tp for tp in TIMEPOINTS_ORDERED if tp in all_tp_rows["timepoint_name"].values]
    tp_row_names = [f"{ct}_{tp}" for tp in tp_order]
    tp_indices   = [int(np.where(var.index == name)[0][0]) for name in tp_row_names
                    if name in var.index]
    tp_order     = [tp_order[i] for i, name in enumerate(tp_row_names) if name in var.index]

    if not tp_indices:
        print(f"  SKIP: no timepoints found for {ct}")
        continue

    n_tps = len(tp_indices)
    print(f"  Timepoints ({n_tps}): {tp_order}")

    # Reliability flags
    tp_reliable  = [bool(var.iloc[i]["reliable"]) for i in tp_indices]
    tp_n_cells   = [int(var.iloc[i]["n_cells"])   for i in tp_indices]
    all_unreliable = not any(tp_reliable)
    if all_unreliable:
        print(f"  WARNING: all timepoints unreliable (n_cells < 20) for {ct}")

    # ── 2. Extract Z for this cell type × all timepoints ───────────────────
    Z_ct = Z[:, tp_indices]          # (640830, n_tps)
    max_z_ct = Z_ct.max(axis=1)      # (640830,)
    best_tp_local = Z_ct.argmax(axis=1)  # index into tp_order

    # ── 3. Top-N peaks by max z ─────────────────────────────────────────────
    top_idx = np.argsort(max_z_ct)[-TOP_N:][::-1]

    # n_specific per timepoint (z >= 4)
    n_specific_per_tp = [(Z_ct[:, j] >= 4).sum() for j in range(n_tps)]

    # Best timepoint = most n_specific among RELIABLE timepoints (fall back to all if none reliable)
    reliable_tp_idx = [j for j, rel in enumerate(tp_reliable) if rel]
    if reliable_tp_idx:
        best_local = reliable_tp_idx[int(np.argmax([n_specific_per_tp[j] for j in reliable_tp_idx]))]
    else:
        best_local = int(np.argmax(n_specific_per_tp))
    best_tp_for_ct = tp_order[best_local]

    # ── 4. Build row labels and table ───────────────────────────────────────
    row_labels   = []
    table_rows   = []
    marker_flags = []

    for rank, peak_pos in enumerate(top_idx, 1):
        peak_obs   = obs.iloc[peak_pos]
        gene, src  = get_gene(peak_obs)
        named      = is_named_gene(gene)
        is_mkr     = is_marker(gene, ct)
        peak_type  = str(peak_obs.get("peak_type", ""))[:8]
        chrom      = str(peak_obs.get("chrom", ""))
        start      = peak_obs.get("start", 0)
        end        = peak_obs.get("end", 0)
        best_tp_name = tp_order[int(best_tp_local[peak_pos])]
        z_best       = float(max_z_ct[peak_pos])
        z_by_tp      = [float(Z_ct[peak_pos, j]) for j in range(n_tps)]

        # Label for heatmap row
        if gene and named:
            label = f"{gene}"
            if is_mkr:
                label += " [M]"
        else:
            label = f"{chrom}:{int(start)//1000}k"
        row_labels.append(label)
        marker_flags.append(is_mkr)

        # Table row for markdown
        gene_disp = gene if gene else "—"
        table_rows.append({
            "Rank":     rank,
            "Chrom":    chrom,
            "Start":    int(start),
            "End":      int(end),
            "Type":     peak_type,
            "Gene":     gene_disp,
            "Named":    "yes" if named else "no",
            "Marker":   "★" if is_mkr else "",
            "Best_TP":  best_tp_name,
            "Z_best":   round(z_best, 2),
            **{tp: round(z, 2) for tp, z in zip(tp_order, z_by_tp)},
        })

    # ── 5. Figure: heatmap + temporal barplot ───────────────────────────────
    heat_data = Z_ct[top_idx, :]   # (TOP_N, n_tps)

    # Clamp display to [0, 20]
    vmax = min(float(heat_data.max()), 20.0)

    fig = plt.figure(figsize=(3 + n_tps * 1.3, 10))
    gs  = fig.add_gridspec(1, 2, width_ratios=[n_tps * 1.3, 2], wspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # Heatmap
    col_labels = [tp.replace("somites", "s") for tp in tp_order]
    # mark unreliable timepoints with parentheses
    col_labels_disp = [
        f"({lb})" if not rel else lb
        for lb, rel in zip(col_labels, tp_reliable)
    ]

    im = ax_heat.imshow(heat_data, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=vmax, interpolation="nearest")
    ax_heat.set_xticks(range(n_tps))
    ax_heat.set_xticklabels(col_labels_disp, fontsize=9)
    ax_heat.set_yticks(range(TOP_N))

    # Color row labels: red=marker, blue=named, gray=uncharacterized
    yticklabels = ax_heat.set_yticklabels(row_labels, fontsize=7.5)
    for lbl, is_mkr in zip(yticklabels, marker_flags):
        lbl.set_color("#c0392b" if is_mkr else "black")

    plt.colorbar(im, ax=ax_heat, label="Specificity z-score", fraction=0.03, pad=0.02)
    ax_heat.set_xlabel("Timepoint  (parens = n_cells < 20)")
    ax_heat.set_title(f"Top {TOP_N} peaks\n[M] = known marker gene (red label)", fontsize=9)

    # n_cells annotation below x-axis
    for j, (nc, rel) in enumerate(zip(tp_n_cells, tp_reliable)):
        ax_heat.text(j, TOP_N + 0.5, f"n={nc}", ha="center", va="top",
                     fontsize=6.5, color="gray" if not rel else "black")

    # Temporal barplot
    colors_bar = ["#e74c3c" if r else "#bdc3c7" for r in tp_reliable]
    ax_bar.barh(range(n_tps), n_specific_per_tp[::-1],
                color=colors_bar[::-1], alpha=0.85)
    ax_bar.set_yticks(range(n_tps))
    ax_bar.set_yticklabels(col_labels_disp[::-1], fontsize=9)
    ax_bar.set_xlabel("N peaks with z ≥ 4")
    ax_bar.set_title("Specific peaks\nper timepoint", fontsize=9)
    ax_bar.axvline(0, color="black", lw=0.5)
    # annotate counts
    for j, n in enumerate(n_specific_per_tp[::-1]):
        ax_bar.text(n + 0.5, j, str(n), va="center", fontsize=7)

    # Reliability legend
    patch_r = mpatches.Patch(color="#e74c3c", label="reliable (n ≥ 20)")
    patch_g = mpatches.Patch(color="#bdc3c7", label="unreliable (n < 20)")
    ax_bar.legend(handles=[patch_r, patch_g], fontsize=7, loc="lower right")

    unreliable_note = "\n[!] ALL timepoints unreliable (n < 20 cells)" if all_unreliable else ""
    fig.suptitle(
        f"Parts List — {ct.replace('_', ' ')}\n"
        f"Peak regulatory activity: {best_tp_for_ct}{unreliable_note}",
        fontsize=12, y=1.02
    )

    out_stem = f"{FIG_DIR}/detail_{ct}_v2"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_stem}.{{pdf,png}}")

    # ── 6. Markdown section ──────────────────────────────────────────────────
    table_df = pd.DataFrame(table_rows)

    # Reverse lookup for this cell type
    rev_ct = rev_df[rev_df["celltype"] == ct] if not rev_df.empty else pd.DataFrame()
    rev_z2 = rev_ct[rev_ct["zscore"] >= 2.0] if not rev_ct.empty else pd.DataFrame()
    rev_z4 = rev_ct[rev_ct["zscore"] >= 4.0] if not rev_ct.empty else pd.DataFrame()

    # n_specific summary
    n_spec_table = "\n".join(
        f"| {tp.replace('somites',' s')} | {int(n):,} |"
        + (" *(unreliable)*" if not rel else "")
        for tp, n, rel in zip(tp_order, n_specific_per_tp, tp_reliable)
    )

    # top peaks markdown table (condensed)
    def md_table_row(r):
        gene = r["Gene"]
        if r["Marker"]:
            gene = f"**{gene}** (M)"
        elif r["Named"] == "yes":
            gene = f"{gene}"
        else:
            gene = f"*{gene}*"
        return f"| {r['Rank']} | {r['Chrom']}:{r['Start']:,}–{r['End']:,} | {r['Type']} | {gene} | {r['Best_TP']} | {r['Z_best']} |"

    peak_md_rows = "\n".join(md_table_row(r) for _, r in table_df.iterrows())

    # marker genes recovered
    if not rev_z2.empty:
        marker_list_z2 = ", ".join(f"`{g}`" for g in rev_z2.sort_values("zscore", ascending=False)["marker_gene"].tolist())
        marker_list_z4 = ", ".join(f"`{g}`" for g in rev_z4.sort_values("zscore", ascending=False)["marker_gene"].tolist()) if not rev_z4.empty else "—"
    else:
        marker_list_z2 = "—"
        marker_list_z4 = "—"

    unreliable_warn = (
        "\n> **Warning**: All timepoints have < 20 cells (unreliable pseudobulk). "
        "Specificity z-scores may be inflated. Interpret with caution.\n"
        if all_unreliable else ""
    )

    section = f"""
## {ct.replace('_', ' ').title()}
{unreliable_warn}
### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
""" + "\n".join(
        f"| {tp} | {nc:,} | {'yes' if rel else 'no'} | {ns:,} |"
        for tp, nc, rel, ns in zip(tp_order, tp_n_cells, tp_reliable, n_specific_per_tp)
    ) + f"""

**Peak regulatory activity**: highest at **{best_tp_for_ct}** ({n_specific_per_tp[best_local]:,} peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: {len(MARKER_GENES.get(ct, set()))}
- Recovered at z ≥ 2: {len(rev_z2)} genes — {marker_list_z2}
- Recovered at z ≥ 4: {len(rev_z4)} genes — {marker_list_z4}

### Top {TOP_N} most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
{peak_md_rows}

> Figure: `figures/peak_parts_list/detail_{ct}.pdf`
"""
    md_sections.append(section)

    # Print preview
    print(f"  Best timepoint (n_specific): {best_tp_for_ct}  ({max(n_specific_per_tp):,} peaks z≥4)")
    print(f"  Marker recall z≥2: {len(rev_z2)}/{len(MARKER_GENES.get(ct, set()))}  markers: {marker_list_z2[:80]}")

# %% ── Write combined markdown report ─────────────────────────────────────────
print("\nWriting combined markdown report ...", flush=True)

header = f"""# Parts List: Celltype Detail Report

**Generated**: `{pd.Timestamp.now().strftime('%Y-%m-%d')}`

**Cell types analyzed**: {', '.join(TARGET_CELLTYPES)}

**Method**: Specificity z-score = leave-one-out z-score across 190 (celltype × timepoint) conditions.
For each peak, z-score measures how much more accessible it is in one condition relative to all others.

**Reverse lookup**: For each known marker gene, find associated peaks and report their z-score
in the representative condition (highest n_cells). This validates whether the parts list captures
known biology.

---
"""

report_path = f"{OUTDIR}/celltype_detail_report_v2.md"
with open(report_path, "w") as f:
    f.write(header)
    f.write("\n".join(md_sections))
print(f"  Saved: {report_path}")

print("\nAll done.")
