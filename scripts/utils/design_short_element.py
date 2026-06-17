"""Compute and visualize ≤500 bp synthesis-ready designs from longer peaks.

For each input peak, computes two design schemes and renders a 3-panel
PDF showing both:

  Scheme A  Single 500 bp window (densest TF hits) — single contiguous
             fragment, native sequence preserved.
  Scheme B  Stitched motif hubs (≤500 bp total) — pick top hubs by
             motif density, concatenate to fit budget. Highest TF
             density per bp but loses native spacing.

Reuses Panels 1A and 1B from make_peak_3panel_figures.py (cell-type
z-score, timepoint z-score) — those are peak-level and unchanged.
Panel 2 is replaced with a 3-row design overlay:
  Row 1  Full peak track with both designs highlighted
         (orange dashed = 500 bp single window, green = stitched hubs)
  Row 2  Side-by-side TF coverage table (Original / SchemeA / SchemeB)
         showing # unique TFs, EXPLICIT, IMPLICIT, length, and the top
         8 TFs preserved.

Usage:
  python design_short_element.py \\
      --ranking-csv pax2a_3selected_ranking.csv \\
      --fimo-hits   pax2a_jaspar_fimo_hits.csv \\
      --output-dir  results/designs/ \\
      --max-len 500
"""

import os, sys, argparse
import numpy as np
import pandas as pd

# ── Publication figure settings (shared module: scripts/utils/pub_fig_style.py) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_fig_style import apply as _apply_pub_style
_apply_pub_style()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

REPO  = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
sys.path.insert(0, f"{REPO}/scripts/utils")
from rank_synthetic_enhancers import tf_family, FAMILY_COLORS
from make_peak_3panel_figures import (
    classify_tf, plot_panel_1A, plot_panel_1B, DataBundle, TIMEPOINTS
)


# ── Design algorithms ────────────────────────────────────────────────────────

def best_window(hits_df: pd.DataFrame, peak_len: int, win: int = 500
                ) -> tuple:
    """Slide a `win` bp window over the peak; pick the start with the
    most unique TFs whose midpoints fall in the window.
    Returns (start_offset, end_offset, n_unique_tfs)."""
    if peak_len <= win or hits_df.empty:
        return 0, peak_len, hits_df["tf"].nunique() if not hits_df.empty else 0
    df = hits_df.copy()
    df["mid"] = ((df["hit_start"] + df["hit_end"]) // 2).astype(int)
    best_s, best_n = 0, 0
    step = max(1, (peak_len - win) // 200)
    for s in range(0, peak_len - win + 1, step):
        e = s + win
        n = df[(df["mid"] >= s) & (df["mid"] < e)]["tf"].nunique()
        if n > best_n:
            best_n = n
            best_s = s
    return best_s, best_s + win, best_n


def _cluster_one_pass(hits_df: pd.DataFrame, peak_len: int,
                       gap_thresh: int, flank: int):
    """Cluster motif midpoints with a given gap threshold; return hub records."""
    df = hits_df.copy()
    df["mid"] = ((df["hit_start"] + df["hit_end"]) // 2).astype(int)
    df = df.sort_values("mid").reset_index(drop=True)

    clusters = []
    cur = []
    for _, h in df.iterrows():
        if not cur or h["mid"] - cur[-1]["mid"] < gap_thresh:
            cur.append(h)
        else:
            clusters.append(cur)
            cur = [h]
    if cur:
        clusters.append(cur)

    hubs = []
    for c in clusters:
        cdf = pd.DataFrame(c)
        s = max(0, int(cdf["mid"].min()) - flank)
        e = min(peak_len, int(cdf["mid"].max()) + flank)
        n_tfs = cdf["tf"].nunique()
        score = n_tfs + (-np.log10(cdf["pvalue"]).sum() / 100)
        hubs.append({"start": s, "end": e, "len": e - s,
                      "n_tfs": n_tfs, "score": score})
    return hubs


def stitched_hubs(hits_df: pd.DataFrame, peak_len: int, max_total: int = 500,
                   gap_thresh: int = 80, flank: int = 12) -> tuple:
    """Cluster motif midpoints into hubs, greedy-pack hubs that fit `max_total`.

    Iterates the gap threshold (80 → 40 → 20 → 10) if the initial result
    is poor (< max_total/2 used). Falls back to the densest max_total
    window within the best hub if all hubs are individually too large.

    Returns (segments, total_len, n_unique_tfs_in_design).
    """
    if peak_len <= max_total or hits_df.empty:
        return [(0, peak_len)], peak_len, hits_df["tf"].nunique() if not hits_df.empty else 0

    best_design = None  # (segments, total, n_tfs)
    for gt in (gap_thresh, gap_thresh // 2, gap_thresh // 4,
                max(5, gap_thresh // 8)):
        hubs = _cluster_one_pass(hits_df, peak_len, gt, flank)
        if not hubs:
            continue

        # Greedy pack by score
        hubs_sorted = sorted(hubs, key=lambda h: -h["score"])
        chosen = []
        used = 0
        for h in hubs_sorted:
            if used + h["len"] <= max_total:
                chosen.append(h)
                used += h["len"]

        if chosen and used >= max_total * 0.5:
            chosen.sort(key=lambda h: h["start"])
            segs = [(h["start"], h["end"]) for h in chosen]
            merged = [segs[0]]
            for s, e in segs[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            n_tfs = _count_unique_tfs_in_segments(hits_df, merged)
            total = sum(e - s for s, e in merged)
            if best_design is None or n_tfs > best_design[2]:
                best_design = (merged, total, n_tfs)

    if best_design is not None:
        return best_design

    # Fallback: every hub individually exceeds budget → take the densest
    # max_total-bp window within the highest-scoring hub.
    hubs = _cluster_one_pass(hits_df, peak_len, gap_thresh, flank)
    top = max(hubs, key=lambda h: h["score"])
    # find best max_total window within the top hub
    df = hits_df.copy()
    df["mid"] = ((df["hit_start"] + df["hit_end"]) // 2).astype(int)
    sub = df[(df["mid"] >= top["start"]) & (df["mid"] < top["end"])]
    if sub.empty:
        return [(top["start"], min(peak_len, top["start"] + max_total))], max_total, 0
    best_s, best_n = top["start"], 0
    step = max(1, (top["end"] - top["start"] - max_total) // 100)
    for s in range(top["start"], top["end"] - max_total + 1, max(1, step)):
        n = sub[(sub["mid"] >= s) & (sub["mid"] < s + max_total)]["tf"].nunique()
        if n > best_n:
            best_n = n
            best_s = s
    segs = [(best_s, best_s + max_total)]
    return segs, max_total, _count_unique_tfs_in_segments(hits_df, segs)


def _count_unique_tfs_in_segments(hits_df: pd.DataFrame, segments) -> int:
    if hits_df.empty or not segments:
        return 0
    df = hits_df.copy()
    df["mid"] = ((df["hit_start"] + df["hit_end"]) // 2).astype(int)
    keep = pd.Series(False, index=df.index)
    for s, e in segments:
        keep |= ((df["mid"] >= s) & (df["mid"] < e))
    return int(df[keep]["tf"].nunique())


def tfs_in_segments(hits_df: pd.DataFrame, segments) -> pd.DataFrame:
    """Return per-TF best p-value for hits whose midpoint falls in any segment."""
    if hits_df.empty:
        return pd.DataFrame(columns=["tf", "pvalue", "category"])
    df = hits_df.copy()
    df["mid"] = ((df["hit_start"] + df["hit_end"]) // 2).astype(int)
    keep_mask = pd.Series(False, index=df.index)
    for s, e in segments:
        keep_mask |= ((df["mid"] >= s) & (df["mid"] < e))
    sub = df[keep_mask]
    if sub.empty:
        return pd.DataFrame(columns=["tf", "pvalue", "category"])
    best = sub.loc[sub.groupby("tf")["pvalue"].idxmin()].copy()
    best["category"] = best["tf"].apply(classify_tf)
    return best.sort_values("pvalue")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_design_panel(ax, peak_row, hits_df, designs):
    """Draw the original peak with all motif hits + both design windows
    highlighted on top."""
    peak_len = int(peak_row["end"] - peak_row["start"])

    # peak background
    ax.add_patch(mpatches.Rectangle((0, 0.45), peak_len, 0.10,
                                     facecolor="#dddddd", edgecolor="none"))

    # Single-500 window — orange dashed
    sa = designs["single"]
    ax.add_patch(mpatches.Rectangle((sa["start"], 0.40), sa["end"] - sa["start"], 0.20,
                                     facecolor="none", edgecolor="#e09b3a",
                                     lw=1.6, ls="--", zorder=2))
    ax.text(sa["start"] + (sa["end"] - sa["start"]) / 2, 0.30,
             f"Scheme A: 500 bp window", ha="center", va="top",
             fontsize=7.5, color="#a06515", fontweight="bold")

    # Stitched hubs — green shaded
    for s, e in designs["stitched"]["segments"]:
        ax.add_patch(mpatches.Rectangle((s, 0.62), e - s, 0.10,
                                         facecolor="#7fc97f", edgecolor="#3e6a3e",
                                         lw=0.6, alpha=0.55, zorder=2))
    n_seg = len(designs["stitched"]["segments"])
    seg_total = designs["stitched"]["total_len"]
    ax.text(peak_len / 2, 0.78, f"Scheme B: stitched {n_seg} hubs ({seg_total} bp)",
             ha="center", va="bottom", fontsize=7.5,
             color="#225522", fontweight="bold")

    # Motif hits
    if not hits_df.empty:
        hits_sorted = hits_df.sort_values("hit_start").copy()
        hits_sorted["family"] = hits_sorted["tf"].apply(tf_family)
        n_rows = 5
        ys = np.linspace(0.86, 1.02, n_rows)
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
            ax.add_patch(mpatches.Rectangle((h["hit_start"], h["y"] - 0.01),
                                             max(h["hit_end"] - h["hit_start"], 5), 0.02,
                                             facecolor=color, edgecolor="black",
                                             linewidth=0.25, alpha=0.85, zorder=3))

    ax.set_xlim(-5, peak_len + 5)
    ax.set_ylim(0.20, 1.10)
    ax.set_xlabel("Position within peak (bp)", fontsize=8)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.set_title(f"Design overlay — peak {int(peak_len)} bp  |  "
                 f"orange dashed = 500 bp window  |  green = stitched hubs  |  "
                 f"motif hits: family-colored", fontsize=8.5, loc="left")


def comparison_text(peak_row, hits_df, designs) -> str:
    """Compact text comparing original vs both designs."""
    peak_len = int(peak_row["end"] - peak_row["start"])
    full_tfs = tfs_in_segments(hits_df, [(0, peak_len)])
    sa_tfs   = tfs_in_segments(hits_df, [(designs["single"]["start"],
                                            designs["single"]["end"])])
    sb_tfs   = tfs_in_segments(hits_df, designs["stitched"]["segments"])

    def stats(tfs):
        n = len(tfs)
        nx = (tfs.category == "explicit").sum() if n else 0
        ni = (tfs.category == "implicit").sum() if n else 0
        return n, nx, ni

    fn, fx, fi = stats(full_tfs)
    an, ax_, ai = stats(sa_tfs)
    bn, bx, bi = stats(sb_tfs)

    def top_explicit(tfs, k=4):
        ex = tfs[tfs.category.isin(["explicit","implicit"])].head(k)
        return ", ".join(f"{r.tf} (p={r.pvalue:.0e})" for _, r in ex.iterrows()) or "(none)"

    lines = []
    lines.append(f"PEAK: {peak_row['peak_id']}  |  full {peak_len} bp  |  "
                 f"top1={peak_row.get('top1_celltype','?')} z={peak_row.get('top1_z',0):.1f}")
    lines.append("")
    lines.append(f"{'Design':<22} {'Length':>7}  {'unique TFs':>10}  {'EXPL':>5}  {'IMPL':>5}")
    lines.append(f"{'─'*22}  {'─'*7}  {'─'*10}  {'─'*5}  {'─'*5}")
    lines.append(f"{'Original (full peak)':<22} {peak_len:>5} bp  {fn:>10}  {fx:>5}  {fi:>5}")
    lines.append(f"{'Scheme A (500 bp win)':<22} {designs['single']['end']-designs['single']['start']:>5} bp  "
                  f"{an:>10}  {ax_:>5}  {ai:>5}")
    lines.append(f"{'Scheme B (stitched)':<22} {designs['stitched']['total_len']:>5} bp  "
                  f"{bn:>10}  {bx:>5}  {bi:>5}")
    lines.append("")
    lines.append(f"Scheme A retains  : {top_explicit(sa_tfs)}")
    lines.append(f"Scheme B retains  : {top_explicit(sb_tfs)}")
    lines.append("")
    seg_str = ", ".join(f"{int(peak_row['start'])+s}-{int(peak_row['start'])+e}"
                        for s, e in designs["stitched"]["segments"])
    lines.append(f"Scheme A absolute coords: chr{peak_row['chrom']}:"
                 f"{int(peak_row['start']) + designs['single']['start']:,}-"
                 f"{int(peak_row['start']) + designs['single']['end']:,}")
    lines.append(f"Scheme B absolute coords: chr{peak_row['chrom']}:[{seg_str}]")
    return "\n".join(lines)


def make_design_figure(peak_row, peak_hits, ct_names, ct_z,
                        tps, tp_vals, tp_z, designs, outpath):
    fig = plt.figure(figsize=(13.5, 11.5))
    gs = gridspec.GridSpec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.7, 1.2],
        width_ratios=[2.6, 1],
        hspace=0.7, wspace=0.25,
        left=0.06, right=0.95, top=0.94, bottom=0.06,
    )

    ax1a = fig.add_subplot(gs[0, 0])
    plot_panel_1A(ax1a, ct_names, ct_z, peak_row.get("top1_celltype", "?"))
    ax1b = fig.add_subplot(gs[0, 1])
    plot_panel_1B(ax1b, tps, tp_vals, tp_z, peak_row.get("top1_celltype", "?"))

    ax2 = fig.add_subplot(gs[1, :])
    plot_design_panel(ax2, peak_row, peak_hits, designs)

    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    ax3.text(0.0, 0.97, comparison_text(peak_row, peak_hits, designs),
              ha="left", va="top", fontsize=8.6, family="monospace")

    title = (f"Rank {int(peak_row.get('rank', 0)):02d}  |  "
             f"chr{peak_row['chrom']}:{int(peak_row['start']):,}–{int(peak_row['end']):,}  "
             f"({int(peak_row['end'] - peak_row['start'])} bp)  |  "
             f"top1: {peak_row['top1_celltype']} z={peak_row['top1_z']:.1f}")
    fig.suptitle(title, fontsize=10.5, y=0.985)
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-csv", required=True)
    p.add_argument("--fimo-hits",   required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--max-len",     type=int, default=500,
                   help="Synthesis budget in bp (default 500 — IDT-friendly)")
    p.add_argument("--top-n",       type=int, default=None,
                   help="Plot top N peaks (default: all rows in ranking-csv)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ranking = pd.read_csv(args.ranking_csv)
    if args.top_n:
        ranking = ranking.head(args.top_n)
    hits = pd.read_csv(args.fimo_hits)

    bundle = DataBundle(peak_ids_needed=ranking["peak_id"].tolist())

    summary_rows = []
    for _, r in ranking.iterrows():
        peak_id = r["peak_id"]
        peak_len = int(r["end"] - r["start"])
        peak_hits = hits[hits.peak_id == peak_id]

        # Designs
        sa_s, sa_e, sa_n = best_window(peak_hits, peak_len, win=args.max_len)
        sb_segs, sb_total, sb_n = stitched_hubs(peak_hits, peak_len,
                                                  max_total=args.max_len)
        designs = {
            "single":   {"start": sa_s, "end": sa_e, "n_tfs": sa_n},
            "stitched": {"segments": sb_segs, "total_len": sb_total, "n_tfs": sb_n},
        }

        ct_names, ct_z = bundle.celltype_zscores(peak_id)
        tps, tp_vals, tp_z = bundle.timepoint_profile_in_celltype(
            peak_id, r["top1_celltype"])

        out_pdf = (f"{args.output_dir}/design_rank{int(r['rank']):02d}_"
                    f"{peak_id}.pdf")
        make_design_figure(r, peak_hits, ct_names, ct_z,
                            tps, tp_vals, tp_z, designs, out_pdf)
        print(f"  rank {int(r['rank']):02d}: {peak_id}  →  {out_pdf}")

        # Save absolute-coord designs for downstream sequence extraction
        summary_rows.append({
            "rank": int(r["rank"]),
            "peak_id": peak_id,
            "chrom": r["chrom"],
            "peak_start": int(r["start"]),
            "peak_end": int(r["end"]),
            "peak_length": peak_len,
            "top1_celltype": r["top1_celltype"],
            "schemeA_500bp_start": int(r["start"]) + sa_s,
            "schemeA_500bp_end":   int(r["start"]) + sa_e,
            "schemeA_n_tfs": sa_n,
            "schemeB_segments": ";".join(
                f"{int(r['start']) + s}-{int(r['start']) + e}"
                for s, e in sb_segs),
            "schemeB_total_bp": sb_total,
            "schemeB_n_tfs": sb_n,
            "schemeB_n_segments": len(sb_segs),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = f"{args.output_dir}/design_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
