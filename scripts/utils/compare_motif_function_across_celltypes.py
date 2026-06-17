"""Compare TF motif composition (activator / repressor / bifunctional)
across peaks specific to different celltypes.

Picks one representative peak per top1_celltype (the most-accessible one)
from a per-gene ranking CSV, then renders a comparison figure showing
the per-peak TF function distribution.

Output:
  {label}_function_comparison.{pdf,png}     stacked-bar comparison plot
  {label}_function_table.csv                per-peak: counts per category +
                                              top TFs per category

Usage:
  python compare_motif_function_across_celltypes.py \\
      --ranking-csv pax2a/ranking/pax2a_enhancer_ranking.csv \\
      --fimo-hits   pax2a/fimo/pax2a_jaspar_fimo_hits.csv \\
      --label       pax2a \\
      --output-dir  pax2a/tf_function_analysis/
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

REPO = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
sys.path.insert(0, f"{REPO}/scripts/utils")
from tf_function_lookup import classify_function, FUNCTION_COLORS, FUNCTION_ORDER
from module_dict_colors import cell_type_color_dict
from plot_peaks_locus_view import lookup_top1_accessibility


def pick_representatives(ranking: pd.DataFrame, max_per_celltype: int = 1
                          ) -> pd.DataFrame:
    """For each top1_celltype, keep the top-N peaks by max_accessibility."""
    out = (ranking.sort_values("max_accessibility", ascending=False)
                  .groupby("top1_celltype", as_index=False)
                  .head(max_per_celltype))
    out = out.sort_values("max_accessibility", ascending=False)
    return out.reset_index(drop=True)


def best_hit_per_tf(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return hits.assign(category="")
    best = hits.loc[hits.groupby("tf")["pvalue"].idxmin()].copy()
    best["category"] = best["tf"].apply(classify_function)
    return best.sort_values("pvalue")


def per_peak_function_counts(rep_peaks: pd.DataFrame,
                              hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in rep_peaks.iterrows():
        ph = hits[hits["peak_id"] == r["peak_id"]]
        best = best_hit_per_tf(ph)
        cnts = {cat: int((best["category"] == cat).sum())
                for cat in FUNCTION_ORDER}
        # top examples per category
        top_examples = {}
        for cat in FUNCTION_ORDER:
            sub = best[best["category"] == cat].head(4)
            top_examples[f"top_{cat}"] = ", ".join(sub["tf"].tolist())
        rows.append({
            "peak_id":        r["peak_id"],
            "top1_celltype":  r["top1_celltype"],
            "top1_z":         float(r["top1_z"]),
            "top1_acc":       float(r.get("top1_accessibility",
                                           r.get("max_accessibility", 0))),
            "peak_type":      r["peak_type"],
            "length":         int(r["length"]),
            "n_unique_tfs":   int(best.shape[0]),
            **cnts,
            "act_minus_rep":  cnts["activator"] - (cnts["repressor"] + cnts["krab_zf"]),
            "act_to_rep_ratio": (
                cnts["activator"] / max(cnts["repressor"] + cnts["krab_zf"], 1)
            ),
            **top_examples,
        })
    return pd.DataFrame(rows)


def plot_comparison(table: pd.DataFrame, outpath: str, gene_name: str = "gene"):
    """Stacked-bar plot: each column = one peak (one celltype), bars stacked
    by activator / bifunctional / repressor / insulator / krab_zf / unknown.
    """
    table = table.sort_values("act_to_rep_ratio", ascending=False).reset_index(drop=True)
    n = len(table)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(8, 1.2 * n), 9.5),
        gridspec_kw={"height_ratios": [3.5, 1.4], "hspace": 0.35},
    )

    # Stacked bars (top panel)
    x = np.arange(n)
    bottom = np.zeros(n)
    for cat in FUNCTION_ORDER:
        vals = table[cat].values
        ax_top.bar(x, vals, bottom=bottom, color=FUNCTION_COLORS[cat],
                    edgecolor="black", linewidth=0.4,
                    label=cat.replace("_", "-"))
        # Annotate counts in non-trivial bars
        for i, v in enumerate(vals):
            if v >= 2:
                ax_top.text(i, bottom[i] + v / 2, str(int(v)),
                             ha="center", va="center", fontsize=7, color="white",
                             fontweight="bold")
        bottom += vals

    ax_top.set_xticks(x)
    labels = [f"{r['top1_celltype']}\n({r['peak_id'].split('-',1)[1][:14]}…)\n"
              f"acc={r['top1_acc']:.0f} · z={r['top1_z']:.1f}"
              for _, r in table.iterrows()]
    ax_top.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax_top.set_ylabel("Number of unique TFs (best hit per TF)", fontsize=10)
    ax_top.set_title(f"{gene_name} — TF motif function composition per representative peak\n"
                      f"(one peak per top1_celltype, picked by highest accessibility)",
                      fontsize=11)
    ax_top.legend(loc="upper right", fontsize=8.5, ncol=2, frameon=True)
    ax_top.grid(axis="y", alpha=0.25)

    # Activator-to-repressor ratio (bottom panel)
    bar_colors = [cell_type_color_dict.get(ct, "#888888")
                  for ct in table["top1_celltype"]]
    ratios = table["act_to_rep_ratio"].values
    ax_bot.bar(x, ratios, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax_bot.axhline(1.0, color="#444", lw=0.8, ls="--",
                    label="balanced (act / rep+ZF = 1)")
    for i, v in enumerate(ratios):
        ax_bot.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(table["top1_celltype"], rotation=30, ha="right", fontsize=8)
    ax_bot.set_ylabel("Activator : (Repressor + KRAB-ZF)", fontsize=10)
    ax_bot.set_title("Activator-to-repressor ratio (higher = more activating)", fontsize=10)
    ax_bot.legend(loc="upper right", fontsize=8, frameon=False)
    ax_bot.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking-csv", required=True)
    p.add_argument("--fimo-hits",   required=True)
    p.add_argument("--label",       required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--max-per-celltype", type=int, default=1,
                   help="Reps per top1_celltype to keep (default 1)")
    p.add_argument("--gene-name", default=None,
                   help="Gene name for figure title (default: --label)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ranking = pd.read_csv(args.ranking_csv)
    hits    = pd.read_csv(args.fimo_hits)
    print(f"Ranking: {len(ranking)} peaks")
    print(f"FIMO hits: {len(hits)} hits across {hits['peak_id'].nunique()} peaks")

    # Add absolute accessibility in top1 celltype
    ranking["top1_accessibility"] = lookup_top1_accessibility(ranking)

    rep = pick_representatives(ranking, max_per_celltype=args.max_per_celltype)
    print(f"\nPicked {len(rep)} representative peaks (one per top1_celltype):")
    print(rep[["peak_id", "top1_celltype", "top1_z",
                "top1_accessibility", "peak_type"]].to_string(index=False))

    table = per_peak_function_counts(rep, hits)
    out_csv = f"{args.output_dir}/{args.label}_function_table.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    # Brief printed summary
    print("\nFunction composition per peak (sorted by act:rep ratio):")
    show_cols = ["top1_celltype", "peak_id", "n_unique_tfs"] + FUNCTION_ORDER + \
                ["act_to_rep_ratio"]
    print(table.sort_values("act_to_rep_ratio", ascending=False)[show_cols]
              .to_string(index=False))

    out_pdf = f"{args.output_dir}/{args.label}_function_comparison.pdf"
    plot_comparison(table, out_pdf, gene_name=(args.gene_name or args.label))
    print(f"\nWrote: {out_pdf}")


if __name__ == "__main__":
    main()
