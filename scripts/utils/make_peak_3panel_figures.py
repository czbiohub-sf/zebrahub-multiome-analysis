"""
Generate 3-panel summary figures for top-N candidate enhancer peaks.

For each peak (sorted by use_case_score / composite_score in the
ranking CSV), produces a single PDF with three panels:

  Panel 1 (top, two side-by-side bar plots)
    A. V3 z-score across all 31 celltypes  (the cell-type specificity profile)
    B. Timepoint z-score within top1 celltype  (when in development is the peak open)

  Panel 2 (middle, full width)
    C. TF motif track — FIMO hits colored by TF family, 200 bp core highlighted

  Panel 3 (bottom, full width text)
    D. TF biology summary
       - which TFs are EXPLICIT MHB regulators
       - which are IMPLICIT (neural / developmental but not MHB-specific)
       - which are likely background (zinc-finger / promiscuous)

Usage:
    python make_peak_3panel_figures.py \\
        --ranking-csv pax2a_mhb_enhancer_ranking.csv \\
        --fimo-hits   pax2a_jaspar_fimo_hits.csv \\
        --output-dir  results/3panel/ \\
        --top-n 10
"""

import os, sys, argparse, re
import numpy as np
import pandas as pd
import anndata as ad

# ── Publication figure settings (shared module: scripts/utils/pub_fig_style.py) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_fig_style import apply as _apply_pub_style
_apply_pub_style()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

REPO = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
sys.path.insert(0, f"{REPO}/scripts/utils")
from rank_synthetic_enhancers import tf_family, FAMILY_COLORS, TF_FAMILY_PATTERNS  # reuse

BASE        = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
MASTER_H5AD = f"{BASE}/data/annotated_data/objects_v2/peaks_by_ct_tp_master_anno.h5ad"
V3_ZMAT     = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/V3_specificity_matrix_celltype_level.h5ad"

# Canonical celltype color palette
sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

TIMEPOINTS = ["0somites", "5somites", "10somites",
              "15somites", "20somites", "30somites"]


# ── MHB-relevant TF curation ─────────────────────────────────────────────────
# Family-prefix lists: a TF (or any underscore-delimited subname like
# "POU3F" inside "POU2F_POU3F") is matched if either ITS NAME STARTS WITH
# any prefix, OR any prefix STARTS WITH IT (bidirectional).
# This handles JASPAR2024 dimer labels and zebrafish paralogs uniformly.

# Explicit: direct regulators / effectors of midbrain-hindbrain boundary
# (paired-box, engrailed, otx/gbx, isthmus signaling, MHB her genes, …)
MHB_EXPLICIT = {
    "PAX",         # PAX2, PAX5, PAX8 — all MHB
    "EN1", "EN2",
    "OTX",         # OTX1, OTX2
    "GBX",         # GBX1, GBX2
    "FGF8", "FGF17", "FGF18",
    "WNT1", "WNT8", "WNT10",
    "LMX",         # LMX1A, LMX1B
    "IRX",         # IRX1/2/3/5 — all MHB-relevant
    "DBX",         # DBX1
    "DMBX",        # DMBX1
    "HER5", "HER8", "HER11",
    "POU3F",       # POU3F2, POU3F3 — midbrain
    "PHOX2",       # PHOX2A/B — neural/MHB-adjacent
    "MEIS",        # MEIS1/2 — HOX cofactor at MHB
    "EMX",         # EMX1/2 — telencephalon/diencephalon, used at MHB
    "SP8",         # MHB induction
}

# Implicit: broader neural / developmental TFs (often co-occurring at MHB but
# not MHB-defining)
MHB_IMPLICIT = {
    "SOX",         # all SOX
    "FOX",         # all FOX (FoxA1/2 also ZF-flagged but FOX wins first)
    "NKX",
    "GATA",
    "TFAP2", "AP2",
    "TBX", "TBR",
    "NEUROG", "NEUROD",
    "PBX",
    "RFX",
    "POU2F", "POU4F", "POU5F",   # POU classes other than POU3F
    "ASCL",
    "EBF",
    "ZIC",
    "TCF", "LEF",                # Wnt effectors
    "OLIG",
    "HEY", "HES",                # Notch effectors
    "ETV", "ETS", "ELK", "FLI",
    "IRF",
    "BHLH",
    "ONECUT", "CUX",
    "RBPJ",
}

# Likely background (zinc-finger / promiscuous) — used to filter noise
ZF_BACKGROUND = re.compile(
    r"^Z[NFKB]|^ZN\d|^PRDM|^ZSCAN|^KRAB|^ZFP|^RREB|^MTF1|^ZBT|^ZKSCAN", re.I)


def _name_match(p: str, prefix_set: set) -> bool:
    """True iff `p` matches any entry in `prefix_set` bidirectionally."""
    for t in prefix_set:
        if p == t:
            return True
        # length >= 3 to avoid short-prefix collisions like "EN"
        if len(p) >= 3 and t.startswith(p):
            return True
        if len(t) >= 3 and p.startswith(t):
            return True
    return False


def classify_tf(tf_name: str) -> str:
    """Return 'explicit', 'implicit', 'background', or 'other'.
    Uses the hardcoded MHB curation. For other tissues, supply a
    tf_biology_table.csv via load_tf_biology_lookup() — that overrides
    this function via classify_tf_with_lookup()."""
    if not tf_name:
        return "other"
    parts = re.split(r"[\W_]+", tf_name.upper())
    parts = [p for p in parts if len(p) >= 2]

    for p in parts:
        if _name_match(p, MHB_EXPLICIT):
            return "explicit"
    for p in parts:
        if _name_match(p, MHB_IMPLICIT):
            return "implicit"
    if ZF_BACKGROUND.match(tf_name):
        return "background"
    return "other"


# ── Tissue-agnostic classification via agent-curated biology table ──────────

def load_tf_biology_lookup(csv_path: str) -> dict:
    """Load a tf_biology_table.csv (filled by the curating agent) and
    return a {tf_name → category} dict. Categories are normalized to
    lowercase 'explicit', 'implicit', 'background', or 'other' to match
    classify_tf() return values."""
    if csv_path is None or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "tf" not in df.columns or "category" not in df.columns:
        return {}
    table = {}
    for _, r in df.iterrows():
        tf = str(r["tf"]).strip()
        cat = str(r.get("category", "")).strip().lower()
        if cat in ("explicit", "implicit", "background", "irrelevant", "other"):
            # normalize 'irrelevant' → 'other'
            table[tf] = "other" if cat == "irrelevant" else cat
    return table


def classify_tf_with_lookup(tf_name: str, lookup: dict) -> str:
    """Look up `tf_name` in the agent-curated lookup; fall back to the
    hardcoded MHB classifier if no entry."""
    if lookup and tf_name in lookup:
        return lookup[tf_name]
    return classify_tf(tf_name)


# ── Data loading helpers ─────────────────────────────────────────────────────

class DataBundle:
    """Holds master h5ad, V3 z-matrix, and lookups in memory."""

    def __init__(self, master=MASTER_H5AD, v3_zmat=V3_ZMAT, peak_ids_needed=None):
        # V3 z-matrix (small)
        z = ad.read_h5ad(v3_zmat)
        self.Z = np.array(z.X)
        self.ct_names = list(z.var_names)
        self.peak_id_to_row = {pid: i for i, pid in enumerate(z.obs.index)}

        # Master h5ad — load only the rows we need (much faster)
        master_adata = ad.read_h5ad(master)
        if peak_ids_needed is not None:
            mask = master_adata.obs.index.isin(peak_ids_needed)
            self.master = master_adata[mask].copy()
        else:
            self.master = master_adata
        self.master_idx = {pid: i for i, pid in enumerate(self.master.obs.index)}
        self.condition_idx = {c: i for i, c in enumerate(self.master.var_names)}
        # X expected to be log_norm already
        self.X = (self.master.X if isinstance(self.master.X, np.ndarray)
                   else np.array(self.master.X.todense())) if hasattr(self.master.X, 'todense') \
                  else np.array(self.master.X)

    def celltype_zscores(self, peak_id):
        """Return (ct_names, z_values)."""
        i = self.peak_id_to_row[peak_id]
        return self.ct_names, self.Z[i]

    def timepoint_profile_in_celltype(self, peak_id, celltype):
        """Return (timepoints, raw_log_norm, leave_one_out_z)."""
        i = self.master_idx.get(peak_id)
        if i is None:
            return TIMEPOINTS, np.full(len(TIMEPOINTS), np.nan), np.full(len(TIMEPOINTS), np.nan)
        vals = []
        for tp in TIMEPOINTS:
            col = f"{celltype}_{tp}"
            j = self.condition_idx.get(col)
            vals.append(self.X[i, j] if j is not None else np.nan)
        v = np.array(vals, dtype=float)
        # Leave-one-out z-score across the 6 timepoints
        z = np.zeros_like(v)
        with np.errstate(invalid="ignore"):
            for k in range(len(v)):
                others = np.delete(v, k)
                others = others[~np.isnan(others)]
                if len(others) >= 2 and not np.isnan(v[k]):
                    mu, sd = others.mean(), others.std(ddof=0)
                    z[k] = (v[k] - mu) / max(sd, 1e-10)
                else:
                    z[k] = np.nan
        return TIMEPOINTS, v, z


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_panel_1A(ax, ct_names, z_values, top1_celltype):
    """Bar plot: V3 z-score across 31 celltypes, target highlighted."""
    order = np.argsort(z_values)[::-1]  # descending
    cts_ord = [ct_names[i] for i in order]
    z_ord   = z_values[order]
    colors = [cell_type_color_dict.get(c, "#888888") for c in cts_ord]
    edge   = ["red" if c == top1_celltype else "none" for c in cts_ord]
    lw     = [1.5 if c == top1_celltype else 0 for c in cts_ord]
    ax.bar(range(len(cts_ord)), z_ord, color=colors, edgecolor=edge, linewidth=lw)
    ax.set_xticks(range(len(cts_ord)))
    ax.set_xticklabels(cts_ord, rotation=90, fontsize=6)
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_ylabel("V3 z-score", fontsize=8)
    ax.set_title("Cell-type specificity (31 celltypes, sorted)", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)


def plot_panel_1B(ax, timepoints, vals, zs, top1_celltype):
    """Bar plot: timepoint z-score within top1 celltype + raw log_norm overlay."""
    tp_short = [t.replace("somites", "s") for t in timepoints]
    color = cell_type_color_dict.get(top1_celltype, "#888888")
    ax.bar(range(len(tp_short)), zs, color=color, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xticks(range(len(tp_short)))
    ax.set_xticklabels(tp_short, fontsize=8)
    ax.set_ylabel("Timepoint z-score", fontsize=8)
    ax.set_title(f"Timepoint specificity in {top1_celltype}", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    # Show raw log_norm values as text above bars
    ymax = max(np.nanmax(zs) if not np.all(np.isnan(zs)) else 1, 1)
    for i, (z, v) in enumerate(zip(zs, vals)):
        if not np.isnan(z):
            ax.text(i, z + ymax * 0.05, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6, color="#444")


def plot_panel_2(ax, peak_row, hits_df, core_start_offset, core_end_offset,
                  compact_segments_offsets=None):
    """TF motif track plot — same style as rank_synthetic_enhancers per-peak plot.
    If compact_segments_offsets is provided, also draw the compact hubs as
    light blue shaded boxes underneath the peak."""
    peak_len = int(peak_row["end"] - peak_row["start"])

    # Peak background
    ax.add_patch(mpatches.Rectangle((0, 0.45), peak_len, 0.10,
                                     facecolor="#dddddd", edgecolor="none"))
    # 200 bp core highlight
    ax.add_patch(mpatches.Rectangle((core_start_offset, 0.40),
                                     core_end_offset - core_start_offset, 0.20,
                                     facecolor="none", edgecolor="#cc3344",
                                     lw=1.6, ls="--", zorder=2))
    # Compact-segments shading (blue, semi-transparent, below peak bar)
    if compact_segments_offsets:
        for s, e in compact_segments_offsets:
            ax.add_patch(mpatches.Rectangle((s, 0.32), e - s, 0.08,
                                             facecolor="#7AB8E0", edgecolor="#3a6f95",
                                             lw=0.6, alpha=0.55, zorder=1))

    if not hits_df.empty:
        hits_sorted = hits_df.sort_values("hit_start").copy()
        hits_sorted["family"] = hits_sorted["tf"].apply(tf_family)
        n_rows = 7
        ys = np.linspace(0.65, 0.95, n_rows)
        row_endpos = [-1] * n_rows
        rec_y = []
        for _, h in hits_sorted.iterrows():
            placed = False
            for r in range(n_rows):
                if h["hit_start"] >= row_endpos[r] + 5:
                    rec_y.append(ys[r])
                    row_endpos[r] = h["hit_end"]
                    placed = True
                    break
            if not placed:
                rec_y.append(ys[len(rec_y) % n_rows])
        hits_sorted["y"] = rec_y

        for _, h in hits_sorted.iterrows():
            color = FAMILY_COLORS.get(h["family"], "#999999")
            ax.add_patch(mpatches.Rectangle((h["hit_start"], h["y"] - 0.012),
                                             max(h["hit_end"] - h["hit_start"], 5), 0.024,
                                             facecolor=color, edgecolor="black",
                                             linewidth=0.3, alpha=0.9, zorder=3))
        # Label top-N
        top_hits = hits_sorted.nsmallest(8, "pvalue")
        for _, h in top_hits.iterrows():
            ax.text(h["hit_start"] + (h["hit_end"] - h["hit_start"]) / 2,
                    h["y"] + 0.02, h["tf"], fontsize=6, ha="center",
                    color="#222", fontweight="bold")
        present_fams = sorted(hits_sorted["family"].unique())
        legend_handles = [mpatches.Patch(facecolor=FAMILY_COLORS.get(f, "#999"),
                                          edgecolor="black", label=f)
                           for f in present_fams]
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(1.005, 1.0), fontsize=6.5, frameon=False,
                  title="TF family", title_fontsize=7)

    ax.set_xlim(-5, peak_len + 5)
    ax.set_ylim(0.25, 1.05)
    ax.set_xlabel("Position within peak (bp)", fontsize=8)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    title = "TF motif positions (FIMO p<1e-4, JASPAR2024)"
    if compact_segments_offsets:
        comp_total = sum(e - s for s, e in compact_segments_offsets)
        title += f"  |  compact: {len(compact_segments_offsets)} hubs, {comp_total} bp (blue shading)"
    title += "  |  red dashed = 200 bp core"
    ax.set_title(title, fontsize=9)


def panel_3_summary_text(peak_row, hits_df, tf_lookup: dict = None) -> str:
    """Return a multi-line markdown-style summary for the text panel.
    If `tf_lookup` (from load_tf_biology_lookup) is provided, it
    overrides the hardcoded MHB curation per-TF."""
    if hits_df.empty:
        return "No FIMO hits."

    # Best hit per TF
    best = hits_df.loc[hits_df.groupby("tf")["pvalue"].idxmin()].copy()
    if tf_lookup:
        best["category"] = best["tf"].apply(
            lambda t: classify_tf_with_lookup(t, tf_lookup))
    else:
        best["category"] = best["tf"].apply(classify_tf)
    best = best.sort_values("pvalue")

    explicit = best[best["category"] == "explicit"]
    implicit = best[best["category"] == "implicit"]
    background = best[best["category"] == "background"]

    lines = []
    lines.append(f"PEAK: {peak_row['peak_id']}  |  rank {int(peak_row.get('rank', 0))}  |  "
                 f"composite {peak_row.get('composite_score', 0):.3f}"
                 + (f"  |  use_case {peak_row.get('use_case_score', 0):.3f}"
                    if not pd.isna(peak_row.get('use_case_score', None)) else "")
                 )
    lines.append(f"  top1={peak_row.get('top1_celltype','?')} (z={peak_row.get('top1_z',0):.1f})  |  "
                 f"peak_type={peak_row.get('peak_type','?')}, "
                 f"length={int(peak_row.get('length', 0))} bp, "
                 f"TSS dist {peak_row.get('distance_to_tss','NA')}")
    # pax2a-specific or target-TSS line
    if "distance_to_target_tss" in peak_row.index and not pd.isna(peak_row.get("distance_to_target_tss", None)):
        d = int(peak_row["distance_to_target_tss"])
        kb = d / 1000.0
        lines.append(f"  Distance from peak centroid to TARGET gene TSS: {d:,} bp (~{kb:.1f} kb)")
    # tissue classification
    tag = []
    if peak_row.get("target_match", False):
        tag.append("PRIMARY TARGET match")
    elif peak_row.get("top1_in_permissive", False):
        tag.append("alternative tissue (also expresses target gene)")
    else:
        tag.append("off-tissue (top1 outside known target-expressed celltypes)")
    if "synthesis_length_factor" in peak_row.index:
        sf = float(peak_row["synthesis_length_factor"])
        tag.append(f"synth_factor={sf:.2f}")
    if "compact_length" in peak_row.index and not pd.isna(peak_row.get("compact_length", None)):
        tag.append(f"compact={int(peak_row['compact_length'])} bp / "
                   f"{int(peak_row.get('compact_n_segments', 0))} hubs")
    lines.append("  " + "  |  ".join(tag))
    lines.append("")

    # Top-line summary numbers
    lines.append(f"Total unique TFs: {len(best)}  |  "
                 f"EXPLICIT MHB: {len(explicit)}  |  "
                 f"IMPLICIT (neural / dev): {len(implicit)}  |  "
                 f"likely zinc-finger background: {len(background)}")
    lines.append("")

    # Explicit — show all (these are the biology)
    if len(explicit) > 0:
        names = ", ".join(f"{r['tf']} (p={r['pvalue']:.1e})"
                          for _, r in explicit.head(8).iterrows())
        lines.append(f"EXPLICIT MHB regulators / effectors:\n   {names}")
    else:
        lines.append("EXPLICIT MHB regulators / effectors: (none in this peak)")
    lines.append("")

    # Implicit — top 6
    if len(implicit) > 0:
        names = ", ".join(f"{r['tf']} (p={r['pvalue']:.1e})"
                          for _, r in implicit.head(8).iterrows())
        lines.append(f"IMPLICIT (neural / developmental, broader role):\n   {names}")
    lines.append("")

    return "\n".join(lines)


def make_peak_figure(peak_row, hits_df, ct_names, ct_z, tps, tp_vals, tp_z,
                     core_s_off, core_e_off, outpath, tf_lookup: dict = None):
    fig = plt.figure(figsize=(13, 11.0))
    gs = gridspec.GridSpec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.5, 1.0],
        width_ratios=[2.6, 1],
        hspace=0.7, wspace=0.25,
        left=0.06, right=0.95, top=0.94, bottom=0.06,
    )

    # Panel 1A: celltype z-scores (top-left, wider)
    ax1a = fig.add_subplot(gs[0, 0])
    plot_panel_1A(ax1a, ct_names, ct_z, peak_row.get("top1_celltype", "?"))

    # Panel 1B: timepoint z-scores (top-right)
    ax1b = fig.add_subplot(gs[0, 1])
    plot_panel_1B(ax1b, tps, tp_vals, tp_z, peak_row.get("top1_celltype", "?"))

    # Panel 2: motif map (full width middle)
    ax2 = fig.add_subplot(gs[1, :])
    # Parse compact segments if present in the row (semicolon-delimited absolute coords)
    compact_offs = None
    seg_str = peak_row.get("compact_segments", None)
    if isinstance(seg_str, str) and seg_str.strip():
        try:
            absolute = [tuple(map(int, p.split("-"))) for p in seg_str.split(";")]
            compact_offs = [(s - int(peak_row["start"]), e - int(peak_row["start"]))
                             for s, e in absolute]
        except Exception:
            compact_offs = None
    plot_panel_2(ax2, peak_row, hits_df, core_s_off, core_e_off,
                  compact_segments_offsets=compact_offs)

    # Panel 3: text summary
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    summary = panel_3_summary_text(peak_row, hits_df, tf_lookup=tf_lookup)
    ax3.text(0.0, 0.97, summary, ha="left", va="top",
             fontsize=8.5, family="monospace")

    # Title — peak coords + identity
    title = (f"Rank {int(peak_row.get('rank', 0)):02d}  |  "
             f"chr{peak_row['chrom']}:{int(peak_row['start']):,}–{int(peak_row['end']):,}  "
             f"({int(peak_row['end'] - peak_row['start'])} bp, {peak_row['peak_type']})  |  "
             f"top1: {peak_row['top1_celltype']}  z={peak_row['top1_z']:.1f}  |  "
             f"composite {peak_row['composite_score']:.3f}"
             + (f"  |  use_case {peak_row.get('use_case_score', 0):.3f}"
                 if "use_case_score" in peak_row.index else ""))
    fig.suptitle(title, fontsize=10.5, y=0.985)

    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-csv", required=True,
                   help="Master ranking CSV from rank_synthetic_enhancers.py")
    p.add_argument("--fimo-hits",   required=True, help="FIMO hits CSV")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of top-ranked peaks to plot (default: 10)")
    p.add_argument("--core-win", type=int, default=200,
                   help="Core window width column suffix (default: 200)")
    p.add_argument("--tf-biology-csv", default=None,
                   help="Optional agent-curated TF biology table "
                        "(columns: tf, category) — overrides the hardcoded "
                        "MHB curation per-TF. Use for non-MHB tissues.")
    args = p.parse_args()
    tf_lookup = load_tf_biology_lookup(args.tf_biology_csv) if args.tf_biology_csv else {}
    if tf_lookup:
        print(f"Loaded {len(tf_lookup)} TF biology entries from {args.tf_biology_csv}")

    os.makedirs(args.output_dir, exist_ok=True)

    ranking = pd.read_csv(args.ranking_csv)
    ranking = ranking.head(args.top_n).copy()
    print(f"Loaded {len(ranking)} top-ranked peaks from {args.ranking_csv}")

    hits = pd.read_csv(args.fimo_hits)
    print(f"Loaded {len(hits)} FIMO hits across {hits['peak_id'].nunique()} peaks")

    # Load only the peaks we need from master h5ad
    print("Loading data bundle (V3 z-matrix + master h5ad rows for selected peaks) ...")
    bundle = DataBundle(peak_ids_needed=ranking["peak_id"].tolist())
    print(f"  master h5ad subset: {bundle.master.shape}")

    # Generate one figure per peak
    for _, peak_row in ranking.iterrows():
        peak_id = peak_row["peak_id"]
        # Celltype z-scores
        ct_names, ct_z = bundle.celltype_zscores(peak_id)
        # Timepoint profile within top1 celltype
        tps, tp_vals, tp_z = bundle.timepoint_profile_in_celltype(
            peak_id, peak_row["top1_celltype"])
        # FIMO hits for this peak
        peak_hits = hits[hits["peak_id"] == peak_id].copy()
        # Core window offsets (relative to peak start)
        s_col = f"core_{args.core_win}bp_start"
        e_col = f"core_{args.core_win}bp_end"
        if s_col in peak_row and not pd.isna(peak_row[s_col]):
            core_s_off = int(peak_row[s_col]) - int(peak_row["start"])
            core_e_off = int(peak_row[e_col]) - int(peak_row["start"])
        else:
            core_s_off = 0
            core_e_off = int(peak_row["end"] - peak_row["start"])

        outpath = (f"{args.output_dir}/peak_summary_rank"
                   f"{int(peak_row['rank']):02d}_{peak_id}.pdf")
        make_peak_figure(peak_row, peak_hits, ct_names, ct_z,
                          tps, tp_vals, tp_z, core_s_off, core_e_off, outpath,
                          tf_lookup=tf_lookup)
        print(f"  rank {int(peak_row['rank']):02d}: {peak_id}  →  {outpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
