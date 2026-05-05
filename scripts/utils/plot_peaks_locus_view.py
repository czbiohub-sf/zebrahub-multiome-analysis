"""
Visualize a set of peaks at a single genomic locus, alongside the host
gene's body / exons.

Shows:
  - Genomic ruler in Mb
  - Peak blocks at their true positions, colored by top1_celltype,
    with block HEIGHT proportional to top1_z (V3 specificity z-score)
  - Optional gene-body track (exons as filled rectangles, introns as
    thin lines, TSS marked with arrow)

Usage:
    python plot_peaks_locus_view.py \\
        --peaks-csv pax2a_peaks.csv \\
        --gene-name pax2a \\
        --gtf /hpc/reference/.../genes.gtf.gz \\
        --output-dir results/

If --gtf is omitted, the gene track is skipped (peaks-only view).
"""

import os, sys, gzip, argparse, re
from collections import defaultdict
import numpy as np
import pandas as pd

# ── Publication settings ──
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

REPO = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
sys.path.insert(0, f"{REPO}/scripts/utils")
from module_dict_colors import cell_type_color_dict

DEFAULT_GTF = ("/hpc/reference/sequencing_alignment/alignment_references/"
                "zebrafish_genome_GRCz11/genes/genes.gtf.gz")


# ── GTF parsing ──────────────────────────────────────────────────────────────

def get_gene_struct_from_gtf(gtf_path: str, gene_name: str):
    """Return (chrom, gene_start, gene_end, strand, exons) for the longest
    transcript of `gene_name`. exons = list of (start, end)."""
    opener = gzip.open if gtf_path.endswith(".gz") else open
    transcripts = defaultdict(list)
    tx_meta = {}
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            ftype = parts[2]
            attr_str = parts[8]
            if f'gene_name "{gene_name}"' not in attr_str:
                continue
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            m = re.search(r'transcript_id "([^"]+)"', attr_str)
            tx = m.group(1) if m else None
            if not tx:
                continue
            if ftype == "transcript":
                tx_meta[tx] = (chrom, start, end, strand)
            elif ftype == "exon":
                transcripts[tx].append((start, end))

    if not tx_meta:
        return None
    # Pick the longest transcript
    longest = max(tx_meta.items(), key=lambda kv: kv[1][2] - kv[1][1])
    tx, (chrom, gs, ge, strand) = longest
    exons = sorted(transcripts.get(tx, []))
    return chrom, gs, ge, strand, exons


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_locus_view(peaks: pd.DataFrame,
                     gene_name: str,
                     gene_struct,
                     outpath: str,
                     locus_padding: float = 0.05,
                     min_peak_width_px: float = 4):
    chrom = peaks.iloc[0]["chrom"]
    chrom_str = str(chrom)

    # Determine x-axis bounds: locus = peaks ∪ gene body, with padding
    xmin = peaks["start"].min()
    xmax = peaks["end"].max()
    if gene_struct is not None:
        _, gs, ge, _, _ = gene_struct
        xmin = min(xmin, gs)
        xmax = max(xmax, ge)
    span = xmax - xmin
    pad = int(span * locus_padding)
    xmin -= pad
    xmax += pad
    span = xmax - xmin

    # Figure layout:
    #   row 1 (large): peak blocks
    #   row 2 (small): chromosome ruler
    #   row 3 (small, if gene_struct): gene body track
    has_gene = gene_struct is not None
    nrows = 3 if has_gene else 2
    height_ratios = [4, 0.6, 1.2] if has_gene else [4, 0.6]
    fig_h = 6.0 if has_gene else 5.0
    fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_h),
                              gridspec_kw={"height_ratios": height_ratios,
                                            "hspace": 0.18})
    if not has_gene:
        ax_peaks, ax_chrom = axes
    else:
        ax_peaks, ax_chrom, ax_gene = axes

    # ── Row 1: peak blocks ──────────────────────────────────────────────
    ax_peaks.set_xlim(xmin, xmax)
    z_max = max(float(peaks["top1_z"].max()), 1.0)
    ax_peaks.set_ylim(0, z_max * 1.15)
    ax_peaks.set_ylabel("V3 z-score (top celltype)", fontsize=10)
    title = (f"{gene_name} locus — {len(peaks)} peaks\n"
             f"chr{chrom_str}:{xmin:,}–{xmax:,}  ({span:,} bp)")
    ax_peaks.set_title(title, fontsize=11, loc="left")

    legend_celltypes = []
    for _, p in peaks.iterrows():
        ct    = p["top1_celltype"]
        color = cell_type_color_dict.get(ct, "#888888")
        z     = float(p["top1_z"])
        # Ensure visible width for tiny peaks
        width = max(p["end"] - p["start"], span / 250)
        rect = mpatches.Rectangle((p["start"], 0), width, z,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.5, alpha=0.92)
        ax_peaks.add_patch(rect)
        if ct not in legend_celltypes:
            legend_celltypes.append(ct)

    ax_peaks.set_xticks([])
    ax_peaks.grid(axis="y", alpha=0.25)
    for spine in ["top", "right", "bottom"]:
        ax_peaks.spines[spine].set_visible(False)

    # Celltype legend
    legend_patches = [
        mpatches.Patch(facecolor=cell_type_color_dict.get(ct, "#888888"),
                        edgecolor="black", label=ct.replace("_", " "))
        for ct in legend_celltypes
    ]
    ax_peaks.legend(handles=legend_patches, fontsize=8,
                    loc="upper left", bbox_to_anchor=(1.005, 1.0),
                    title="Top celltype", title_fontsize=9, frameon=False)

    # ── Row 2: chromosome ruler ──────────────────────────────────────────
    ax_chrom.set_xlim(xmin, xmax)
    ax_chrom.set_ylim(0, 1)
    ax_chrom.add_patch(mpatches.Rectangle((xmin, 0.35), span, 0.30,
                                            facecolor="#cccccc",
                                            edgecolor="#888", lw=0.6))
    ax_chrom.text(xmin - span * 0.005, 0.50, f"chr{chrom_str}",
                   ha="right", va="center", fontsize=9, fontweight="bold")
    # Tick marks at round Mb / kb
    if span > 5e6:
        tick_step = 1_000_000
    elif span > 500_000:
        tick_step = 100_000
    else:
        tick_step = 10_000
    tick_positions = np.arange(int(np.ceil(xmin / tick_step) * tick_step),
                                int(xmax) + 1, tick_step)
    for tp in tick_positions:
        ax_chrom.plot([tp, tp], [0.30, 0.70], color="#444", lw=0.6)
        if tick_step >= 1_000_000:
            label = f"{tp/1e6:.1f} Mb"
        elif tick_step >= 100_000:
            label = f"{tp/1e6:.2f} Mb"
        else:
            label = f"{tp/1e3:.0f} kb"
        ax_chrom.text(tp, 0.10, label, ha="center", va="top", fontsize=7,
                       color="#444")
    ax_chrom.axis("off")

    # ── Row 3: gene body track ──────────────────────────────────────────
    if has_gene:
        chrom_g, gs, ge, strand, exons = gene_struct
        ax_gene.set_xlim(xmin, xmax)
        ax_gene.set_ylim(0, 1)
        # Intron line
        ax_gene.plot([gs, ge], [0.5, 0.5], color="#444", lw=1.0)
        # TSS arrow marker
        tss = gs if strand == "+" else ge
        ax_gene.annotate("", xy=(tss + (span * 0.02 if strand == "+" else -span * 0.02), 0.50),
                          xytext=(tss, 0.50),
                          arrowprops=dict(arrowstyle="->", color="#cc3344", lw=1.6))
        ax_gene.text(tss, 0.85, "TSS", ha="center", va="bottom",
                      fontsize=8, color="#cc3344", fontweight="bold")
        # Exons
        exon_color = "#3a6f95"
        for es, ee in exons:
            ax_gene.add_patch(mpatches.Rectangle((es, 0.35), ee - es, 0.30,
                                                  facecolor=exon_color,
                                                  edgecolor="#1f4663", lw=0.4))
        # Gene name + strand
        ax_gene.text((gs + ge) / 2, 0.10,
                      f"{gene_name}  ({strand} strand, {ge-gs:,} bp)",
                      ha="center", va="top", fontsize=9, style="italic")
        ax_gene.axis("off")

    fig.tight_layout(rect=[0, 0, 0.86, 1])  # leave room for legend
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {outpath.replace('.pdf', '.png')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peaks-csv", required=True,
                   help="CSV with chrom, start, end, top1_celltype, top1_z")
    p.add_argument("--gene-name", required=True,
                   help="Gene name to look up gene-body track (e.g., pax2a)")
    p.add_argument("--gtf", default=DEFAULT_GTF,
                   help=f"GTF for gene-body lookup (default: {DEFAULT_GTF})")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default=None,
                   help="Output filename prefix (default: '{gene_name}_locus_view'). "
                        "If provided, output is '{label}.pdf' / '{label}.png'.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    label = args.label or f"{args.gene_name}_locus_view"

    peaks = pd.read_csv(args.peaks_csv)
    needed = {"chrom", "start", "end", "top1_celltype", "top1_z"}
    if not needed.issubset(peaks.columns):
        # Try to enrich from cache
        sys.path.insert(0, f"{REPO}/scripts/utils")
        from marker_gene_peaks import _load_cache
        cache = _load_cache()
        keep = ["peak_id", "top1_celltype", "top1_z"]
        peaks = peaks.merge(cache[keep], on="peak_id", how="left")
        if not needed.issubset(peaks.columns):
            sys.exit(f"ERROR: peaks CSV missing required columns: {needed - set(peaks.columns)}")
    peaks["start"] = peaks["start"].astype(int)
    peaks["end"]   = peaks["end"].astype(int)

    print(f"Loaded {len(peaks)} peaks")

    gene_struct = None
    if args.gtf and os.path.exists(args.gtf):
        gene_struct = get_gene_struct_from_gtf(args.gtf, args.gene_name)
        if gene_struct is None:
            print(f"WARNING: gene '{args.gene_name}' not found in GTF — drawing peaks-only view")
        else:
            chrom, gs, ge, strand, exons = gene_struct
            print(f"Gene {args.gene_name}: chr{chrom}:{gs:,}-{ge:,} ({strand}), "
                  f"{len(exons)} exons")
    else:
        print("(no GTF — peaks-only view)")

    outpath = f"{args.output_dir}/{label}.pdf"
    plot_locus_view(peaks, args.gene_name, gene_struct, outpath)


if __name__ == "__main__":
    main()
