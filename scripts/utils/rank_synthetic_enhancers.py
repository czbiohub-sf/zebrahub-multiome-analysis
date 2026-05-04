"""
Rank candidate synthetic-enhancer peaks based on V3 specificity + TF
motif content, extract a core 200 bp window per peak, and (optionally)
generate per-peak motif-position maps.

Inputs:
  - peaks CSV  (must have chrom/start/end; typically the output of
                 scripts/utils/marker_gene_peaks.py)
  - FIMO hits CSV  (output of scripts/utils/run_fimo_on_peaks.py)

Output:
  - {label}_enhancer_ranking.csv     master DataFrame, one row per peak,
                                     sorted by composite score descending
  - {label}_motif_maps/              per-peak PDF with TF-position track
                                     (only created if --plot is set)
  - {label}_summary.txt              human-readable top-N summary

Composite score (each component min-max normalized within the peak set):
   composite = mean(rank(specificity), rank(activity),
                    rank(tf_density), rank(motif_strength))
               * peak_type_factor
   peak_type_factor:  intronic/intergenic = 1.0 (preferred — distal enhancer)
                      promoter           = 0.7 (proximal — useful but bias)
                      exonic             = 0.5 (least preferred)

Usage:
    python rank_synthetic_enhancers.py \\
        --peaks-csv pax2a_peaks.csv \\
        --fimo-hits pax2a_jaspar_fimo_hits.csv \\
        --label pax2a \\
        --output-dir results/ \\
        --plot              # optional — generate motif-map PDFs
"""

import os, sys, argparse, re
from collections import defaultdict
import numpy as np
import pandas as pd

# ── Publication figure settings ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)
_mpl.rcParams['font.family'] = 'Arial'
_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper")
_mpl.rcParams["savefig.dpi"]  = 300
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
_mpl.rcParams["savefig.dpi"]  = 300

REPO = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
sys.path.insert(0, f"{REPO}/scripts/utils")
from marker_gene_peaks import _load_cache  # for V3 z-scores + tau

# ── TF family grouping (for plotting) ────────────────────────────────────────
TF_FAMILY_PATTERNS = [
    ("PAX",   re.compile(r"^PAX",   re.I)),
    ("SOX",   re.compile(r"^SOX",   re.I)),
    ("FOX",   re.compile(r"^FOX",   re.I)),
    ("HOX",   re.compile(r"^HOX|^HX[A-D]", re.I)),
    ("GATA",  re.compile(r"^GATA",  re.I)),
    ("TBX",   re.compile(r"^TBX|^T(BR|BX)|^TBR", re.I)),
    ("OTX",   re.compile(r"^OTX",   re.I)),
    ("EN",    re.compile(r"^EN[12]", re.I)),
    ("ETV",   re.compile(r"^ETV|^ETS|^FLI|^ELK", re.I)),
    ("KLF",   re.compile(r"^KLF|^SP[1-9]",  re.I)),
    ("NK",    re.compile(r"^NKX|^NK\d",     re.I)),
    ("MYC",   re.compile(r"^MYC|^MAX",      re.I)),
    ("ZF",    re.compile(r"^Z[NF]\d|^ZNF|^ZN\d|^ZBT", re.I)),
    ("HD",    re.compile(r"^DLX|^MSX|^EMX|^IRX|^LMX|^DBX|^GBX|^LHX|^POU|^DMBX|^OTX|^BARH", re.I)),
    ("bHLH",  re.compile(r"^MYO[DG]?|^ASCL|^NEUROD|^TWIST|^HAND|^TAL|^MYF|^HER|^HEY|^E2F", re.I)),
    ("HMG",   re.compile(r"^TCF|^LEF",      re.I)),
    ("RFX",   re.compile(r"^RFX",           re.I)),
    ("NR",    re.compile(r"^NR\d|^RAR|^RXR|^THR|^ESR|^GR|^PPAR", re.I)),
    ("IRF",   re.compile(r"^IRF",           re.I)),
    ("NFKB",  re.compile(r"^NFK|^REL",      re.I)),
    ("CREB",  re.compile(r"^CREB|^ATF|^JUN|^FOS", re.I)),
]

def tf_family(tf_name: str) -> str:
    for fam, pat in TF_FAMILY_PATTERNS:
        if pat.match(tf_name):
            return fam
    return "other"

# Family colors (Tableau 20 + extras)
FAMILY_COLORS = {
    "PAX":  "#1f77b4", "SOX":  "#ff7f0e", "FOX":  "#2ca02c", "HOX":  "#d62728",
    "GATA": "#9467bd", "TBX":  "#8c564b", "OTX":  "#e377c2", "EN":   "#7f7f7f",
    "ETV":  "#bcbd22", "KLF":  "#17becf", "NK":   "#aec7e8", "MYC":  "#ffbb78",
    "ZF":   "#c8c8c8", "HD":   "#98df8a", "bHLH": "#ff9896", "HMG":  "#c5b0d5",
    "RFX":  "#c49c94", "NR":   "#f7b6d2", "IRF":  "#dbdb8d", "NFKB": "#9edae5",
    "CREB": "#393b79", "other":"#dddddd",
}

PEAK_TYPE_FACTOR = {
    "intronic":   1.00,
    "intergenic": 1.00,
    "promoter":   0.70,
    "exonic":     0.50,
}


# ── Core helpers ─────────────────────────────────────────────────────────────

def annotate_peaks_from_cache(peaks: pd.DataFrame) -> pd.DataFrame:
    """Add V3 z-scores, top1_celltype, max_accessibility, tau from cache.
    Existing columns in `peaks` are preserved; cache columns are added
    only where missing (avoids duplicate _cache columns)."""
    cache = _load_cache()
    use_cols = ["peak_id", "top1_celltype", "top1_z",
                "top2_celltype", "top2_z", "top3_celltype", "top3_z",
                "max_z", "max_accessibility", "tau", "gini",
                "peak_type", "distance_to_tss", "linked_gene", "nearest_gene",
                "length"]
    # Keep peak_id from peaks; for everything else, prefer existing column
    cache_cols_to_add = [c for c in use_cols if c == "peak_id" or c not in peaks.columns]
    sub = cache[cache_cols_to_add]
    return peaks.merge(sub, on="peak_id", how="left")


def find_core_window(hits_df: pd.DataFrame, peak_len: int, win: int = 200) -> tuple:
    """Find the 200 bp window within a peak with the most unique TF hits.
    Returns (start_offset, end_offset, n_unique_tfs_in_window).
    Coordinates are 0-indexed offsets within the peak.
    """
    if peak_len <= win:
        return 0, peak_len, hits_df["tf"].nunique()
    if hits_df.empty:
        mid = peak_len // 2
        return max(0, mid - win // 2), min(peak_len, mid + win // 2), 0

    # For each candidate window start, count unique TFs whose midpoint lies in the window
    hits_df = hits_df.copy()
    hits_df["mid"] = ((hits_df["hit_start"] + hits_df["hit_end"]) // 2).astype(int)
    best_start, best_count = 0, 0
    # Slide in 10 bp steps for speed (small peaks)
    step = max(1, (peak_len - win) // 100)
    for s in range(0, peak_len - win + 1, step):
        e = s + win
        in_win = hits_df[(hits_df["mid"] >= s) & (hits_df["mid"] < e)]
        n_uniq = in_win["tf"].nunique()
        if n_uniq > best_count:
            best_count = n_uniq
            best_start = s
    return best_start, best_start + win, best_count


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite score columns to peak DataFrame.
    Each numeric feature is converted to a percentile rank (0–1).
    """
    out = df.copy()
    for col, name in [("top1_z", "rank_specificity"),
                      ("max_accessibility", "rank_activity"),
                      ("n_unique_tfs", "rank_tf_density"),
                      ("median_neg_log_p", "rank_motif_strength")]:
        if col in out:
            out[name] = out[col].rank(method="average", pct=True)
        else:
            out[name] = 0.0

    out["base_score"] = out[["rank_specificity", "rank_activity",
                              "rank_tf_density", "rank_motif_strength"]].mean(axis=1)
    out["peak_type_factor"] = out["peak_type"].map(PEAK_TYPE_FACTOR).fillna(0.7)
    out["composite_score"] = (out["base_score"] * out["peak_type_factor"]).round(4)
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


def aggregate_per_peak_motif_stats(hits_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per peak from FIMO hits."""
    rows = []
    for pid, g in hits_df.groupby("peak_id"):
        # Best hit per TF
        best_per_tf = g.loc[g.groupby("tf")["pvalue"].idxmin()]
        rows.append({
            "peak_id":           pid,
            "n_motif_hits":      len(g),
            "n_unique_tfs":      g["tf"].nunique(),
            "n_unique_motifs":   g["motif_accession"].nunique(),
            "best_pvalue":       g["pvalue"].min(),
            "median_neg_log_p":  -np.log10(best_per_tf["pvalue"]).median(),
            "top_tfs":           ",".join(best_per_tf.sort_values("pvalue")["tf"].head(8).tolist()),
        })
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_peak_motif_map(peak_row, hits_df, core_start, core_end, outpath):
    """Draw a single peak's motif positions as a track plot."""
    peak_len = int(peak_row["end"] - peak_row["start"])
    fig, ax = plt.subplots(figsize=(11, 3.3))

    # Peak background bar
    ax.add_patch(mpatches.Rectangle((0, 0.45), peak_len, 0.10,
                                     facecolor="#dddddd", edgecolor="none"))

    # Highlight core 200 bp window
    ax.add_patch(mpatches.Rectangle((core_start, 0.40), core_end - core_start, 0.20,
                                     facecolor="none", edgecolor="#cc3344",
                                     lw=1.6, ls="--", zorder=2))

    # Stack motif hits: compute y positions to avoid overlap
    if not hits_df.empty:
        hits_sorted = hits_df.sort_values("hit_start").copy()
        hits_sorted["family"] = hits_sorted["tf"].apply(tf_family)
        # simple stacking: put each hit in next y-row, wrap
        n_rows = 7
        ys = np.linspace(0.65, 0.95, n_rows)
        # Greedy assignment: place each hit on the lowest row not currently occupied at hit_start
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

        # Label top-N strongest hits with TF name above
        top_hits = hits_sorted.nsmallest(8, "pvalue")
        for _, h in top_hits.iterrows():
            ax.text(h["hit_start"] + (h["hit_end"] - h["hit_start"]) / 2,
                    h["y"] + 0.02, h["tf"], fontsize=6, ha="center",
                    color="#222", fontweight="bold")

        # Family legend (only families present)
        present_fams = sorted(hits_sorted["family"].unique())
        legend_handles = [mpatches.Patch(facecolor=FAMILY_COLORS.get(f, "#999"),
                                          edgecolor="black", label=f)
                           for f in present_fams]
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(1.005, 1.0), fontsize=7, frameon=False,
                  title="TF family", title_fontsize=8)

    # Title with peak metadata
    title = (f"{peak_row['peak_id']}  |  chr{peak_row['chrom']}:"
             f"{int(peak_row['start']):,}–{int(peak_row['end']):,}  ({peak_len} bp)\n"
             f"{peak_row.get('peak_type','?')}  |  TSS dist: {peak_row.get('distance_to_tss', 'NA')}  |  "
             f"top1: {peak_row.get('top1_celltype','?')} z={peak_row.get('top1_z', 0):.1f}  |  "
             f"composite: {peak_row.get('composite_score', 0):.3f}  |  "
             f"unique TFs: {int(peak_row.get('n_unique_tfs', 0))}")
    ax.set_title(title, fontsize=8.5, loc="left")

    ax.set_xlim(-5, peak_len + 5)
    ax.set_ylim(0.30, 1.05)
    ax.set_xlabel("Position within peak (bp)", fontsize=8)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peaks-csv",  required=True,
                   help="Peaks CSV (must have peak_id, chrom, start, end)")
    p.add_argument("--fimo-hits",  required=True,
                   help="FIMO hits CSV (output of run_fimo_on_peaks.py)")
    p.add_argument("--label",      required=True, help="Output prefix")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--core-win",   type=int, default=200,
                   help="Core window length (bp) for synthetic enhancer (default 200)")
    p.add_argument("--plot", action="store_true",
                   help="Generate per-peak motif-map PDFs (one per peak)")
    p.add_argument("--top-n-plot", type=int, default=None,
                   help="If set with --plot, only plot the top N peaks by composite score")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        os.makedirs(f"{args.output_dir}/{args.label}_motif_maps", exist_ok=True)

    # 1. Load peaks + annotate from cache
    peaks = pd.read_csv(args.peaks_csv)
    if "peak_id" not in peaks.columns:
        peaks["peak_id"] = (peaks["chrom"].astype(str) + "-"
                            + peaks["start"].astype(str) + "-"
                            + peaks["end"].astype(str))
    peaks = annotate_peaks_from_cache(peaks)
    print(f"Loaded {len(peaks)} peaks (annotated with V3 z-scores).")

    # 2. Load FIMO hits
    hits = pd.read_csv(args.fimo_hits)
    print(f"Loaded {len(hits)} FIMO hits across {hits['peak_id'].nunique()} peaks "
          f"and {hits['tf'].nunique()} TFs.")

    # 3. Per-peak motif stats
    motif_stats = aggregate_per_peak_motif_stats(hits)
    peaks = peaks.merge(motif_stats, on="peak_id", how="left").fillna(
        {"n_motif_hits": 0, "n_unique_tfs": 0, "n_unique_motifs": 0,
         "best_pvalue": 1.0, "median_neg_log_p": 0.0, "top_tfs": ""})

    # 4. Core 200 bp window per peak
    core_starts, core_ends, core_n_tfs = [], [], []
    for _, peak_row in peaks.iterrows():
        peak_hits = hits[hits["peak_id"] == peak_row["peak_id"]]
        peak_len = int(peak_row["end"] - peak_row["start"])
        s, e, n = find_core_window(peak_hits, peak_len, win=args.core_win)
        core_starts.append(int(peak_row["start"]) + s)
        core_ends.append(int(peak_row["start"]) + e)
        core_n_tfs.append(n)
    peaks[f"core_{args.core_win}bp_start"] = core_starts
    peaks[f"core_{args.core_win}bp_end"]   = core_ends
    peaks[f"core_{args.core_win}bp_n_tfs"] = core_n_tfs

    # 5. Composite score and ranking
    ranked = compute_composite_score(peaks)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # Reorder columns
    front = ["rank", "composite_score", "peak_id", "chrom", "start", "end", "length",
             "peak_type", "distance_to_tss",
             "linked_gene", "nearest_gene",
             "top1_celltype", "top1_z", "top2_celltype", "top2_z",
             "max_z", "max_accessibility", "tau",
             "n_unique_tfs", "n_motif_hits", "median_neg_log_p", "top_tfs",
             f"core_{args.core_win}bp_start", f"core_{args.core_win}bp_end",
             f"core_{args.core_win}bp_n_tfs",
             "rank_specificity", "rank_activity", "rank_tf_density",
             "rank_motif_strength", "base_score", "peak_type_factor"]
    front = [c for c in front if c in ranked.columns]
    other = [c for c in ranked.columns if c not in front]
    ranked = ranked[front + other]

    out_csv = f"{args.output_dir}/{args.label}_enhancer_ranking.csv"
    ranked.to_csv(out_csv, index=False)
    print(f"\nMaster ranking → {out_csv}")

    # 6. Summary text
    summary_lines = [
        f"SYNTHETIC ENHANCER RANKING — {args.label}",
        "=" * 75,
        f"N peaks: {len(ranked)}",
        f"FIMO motif DB: hits CSV at {args.fimo_hits}",
        f"Composite = mean(rank_specificity, rank_activity, rank_tf_density,",
        f"                 rank_motif_strength) × peak_type_factor",
        f"Peak-type factors: {PEAK_TYPE_FACTOR}",
        "",
        "TOP 15 RANKED PEAKS:",
        "-" * 75,
        f"{'Rank':>4} {'Composite':>10} {'Peak ID':<26} {'Type':<10} "
        f"{'top1':<22} {'z':>6} {'TFs':>4}  TopTFs",
    ]
    for _, r in ranked.head(15).iterrows():
        summary_lines.append(
            f"{int(r['rank']):>4} {r['composite_score']:>10.3f} "
            f"{r['peak_id']:<26} {str(r['peak_type'])[:9]:<10} "
            f"{str(r['top1_celltype'])[:21]:<22} "
            f"{r['top1_z']:>6.1f} {int(r['n_unique_tfs']):>4}  {r['top_tfs']}"
        )
    summary_text = "\n".join(summary_lines)
    with open(f"{args.output_dir}/{args.label}_summary.txt", "w") as f:
        f.write(summary_text + "\n")
    print()
    print(summary_text)
    print(f"\nSummary → {args.output_dir}/{args.label}_summary.txt")

    # 7. Per-peak motif map plots
    if args.plot:
        n_to_plot = args.top_n_plot if args.top_n_plot else len(ranked)
        print(f"\nGenerating motif maps for top {n_to_plot} peaks ...")
        plot_dir = f"{args.output_dir}/{args.label}_motif_maps"
        for idx, peak_row in ranked.head(n_to_plot).iterrows():
            peak_hits = hits[hits["peak_id"] == peak_row["peak_id"]].copy()
            core_s_off = int(peak_row[f"core_{args.core_win}bp_start"]) - int(peak_row["start"])
            core_e_off = int(peak_row[f"core_{args.core_win}bp_end"])   - int(peak_row["start"])
            outpath = f"{plot_dir}/rank{int(peak_row['rank']):02d}_{peak_row['peak_id']}.pdf"
            plot_peak_motif_map(peak_row, peak_hits, core_s_off, core_e_off, outpath)
        print(f"  Plots → {plot_dir}/")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
