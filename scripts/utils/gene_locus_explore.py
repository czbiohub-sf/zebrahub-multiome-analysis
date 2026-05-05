"""Single-command exploration of any gene's regulatory peak landscape.

Given a gene name, runs the full pipeline:
  1. peaks_for_genes()         — pull all peaks linked/nearest the gene
  2. run_fimo_on_peaks()        — JASPAR2024 TF motif scan on each peak
  3. rank_synthetic_enhancers() — composite-score ranking, with optional
                                   target-celltype + permissive-celltypes
                                   + synthesis-length penalty + compacting
  4. plot_peaks_locus_view()    — chromosome-track view of all peaks
                                   relative to the gene body
  5. make_peak_3panel_figures() — top-N drill-down plots

GTF is auto-queried for the gene's TSS; the value is passed to the
ranker so distance_to_target_tss is populated.

Usage (basic):
    python gene_locus_explore.py --gene foxd3 --output-dir results/foxd3/

Usage (full):
    python gene_locus_explore.py \\
        --gene pax2a \\
        --output-dir results/pax2a/ \\
        --target-celltype midbrain_hindbrain_boundary \\
        --permissive-celltypes optic_cup,pronephros,hindbrain,neural \\
        --top-n-3panel 10 \\
        --max-synthesis-length 500 \\
        --compact

The script chains existing utilities — it does NOT duplicate logic.
Intended as the entry point for both interactive notebook use and
automated agent-driven exploration of arbitrary marker / DE genes.
"""

import os, sys, argparse, subprocess
import pandas as pd

REPO = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/zebrahub-multiome-analysis"
UTILS = f"{REPO}/scripts/utils"
PYTHON = "/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python"

sys.path.insert(0, UTILS)
from gtf_helpers import DEFAULT_GTF, get_gene_tss


def run(cmd, label):
    print(f"\n── {label} ────────────────────────────────────────────────")
    print(" ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED at step: {label}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gene",       required=True,
                   help="Gene name (must match the GTF gene_name attribute)")
    p.add_argument("--output-dir", required=True,
                   help="Directory for all outputs (CSVs, FIMO hits, figures)")
    p.add_argument("--label",      default=None,
                   help="Filename prefix for outputs (default: gene name)")

    # peaks_for_genes options
    p.add_argument("--min-z", type=float, default=0.0,
                   help="Drop peaks below this V3 max_z before FIMO (default 0)")

    # rank_synthetic_enhancers options
    p.add_argument("--target-celltype", default=None,
                   help="Primary celltype of interest (+0.15 to use_case_score)")
    p.add_argument("--permissive-celltypes", default=None,
                   help="Comma-separated alternative tissues where the gene is "
                        "also expressed (+0.05 each, NOT penalized as off-target)")
    p.add_argument("--prefer-distal",  action="store_true",
                   help="Add bonus for intergenic / intronic peaks")
    p.add_argument("--max-synthesis-length", type=int, default=500)
    p.add_argument("--no-synth-penalty", action="store_true")
    p.add_argument("--compact",  action="store_true",
                   help="Compute motif-hub compact segments for long peaks")

    # 3-panel options
    p.add_argument("--top-n-3panel", type=int, default=10,
                   help="Number of top-ranked peaks to deep-dive plot (default 10)")
    p.add_argument("--no-3panel", action="store_true",
                   help="Skip the 3-panel deep-dive figures")

    # Motif DB + GTF
    p.add_argument("--motif-db", default="jaspar2024",
                   choices=["jaspar2024", "h12core", "cisbpv2_danrer"])
    p.add_argument("--gtf", default=DEFAULT_GTF,
                   help="GTF for TSS lookup + gene-body track")

    args = p.parse_args()
    label = args.label or args.gene
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 0. Look up gene TSS from GTF ────────────────────────────────────
    print(f"\n[0/5] Looking up TSS of {args.gene} in GTF ...")
    tss = get_gene_tss(args.gtf, args.gene)
    if tss is None:
        print(f"  WARNING: gene '{args.gene}' not found in GTF — distance_to_target_tss will be empty")
        target_tss = None
    else:
        chrom, pos = tss
        target_tss = f"{chrom}:{pos}"
        print(f"  {args.gene} TSS: {target_tss}")

    # ── 1. peaks_for_genes ──────────────────────────────────────────────
    peaks_csv = f"{args.output_dir}/{label}_peaks.csv"
    cmd = [PYTHON, f"{UTILS}/marker_gene_peaks.py",
           "--genes", args.gene, "--min-z", str(args.min_z),
           "-o", peaks_csv]
    run(cmd, f"[1/5] peaks_for_genes ({args.gene})")

    # Sanity check
    df = pd.read_csv(peaks_csv)
    print(f"  → {len(df)} peaks for {args.gene}")
    if len(df) == 0:
        sys.exit(f"No peaks found for gene '{args.gene}'. Check spelling / case.")

    # ── 2. run_fimo_on_peaks ────────────────────────────────────────────
    cmd = [PYTHON, f"{UTILS}/run_fimo_on_peaks.py",
           "--peaks-csv", peaks_csv,
           "--label", f"{label}_jaspar" if args.motif_db == "jaspar2024" else f"{label}_{args.motif_db}",
           "--output-dir", args.output_dir,
           "--motif-db", args.motif_db]
    run(cmd, f"[2/5] FIMO motif scan ({args.motif_db})")
    fimo_label = (f"{label}_jaspar" if args.motif_db == "jaspar2024"
                   else f"{label}_{args.motif_db}")
    fimo_hits = f"{args.output_dir}/{fimo_label}_fimo_hits.csv"

    # ── 3. rank_synthetic_enhancers ─────────────────────────────────────
    cmd = [PYTHON, f"{UTILS}/rank_synthetic_enhancers.py",
           "--peaks-csv", peaks_csv,
           "--fimo-hits", fimo_hits,
           "--label", label,
           "--output-dir", args.output_dir,
           "--max-synthesis-length", str(args.max_synthesis_length)]
    if target_tss:
        cmd += ["--target-tss", target_tss]
    if args.target_celltype:
        cmd += ["--target-celltype", args.target_celltype]
    if args.permissive_celltypes:
        cmd += ["--permissive-celltypes", args.permissive_celltypes]
    if args.prefer_distal:
        cmd += ["--prefer-distal"]
    if args.no_synth_penalty:
        cmd += ["--no-synth-penalty"]
    if args.compact:
        cmd += ["--compact"]
    run(cmd, "[3/5] rank_synthetic_enhancers")

    ranking_csv = f"{args.output_dir}/{label}_enhancer_ranking.csv"

    # ── 4. plot_peaks_locus_view ────────────────────────────────────────
    cmd = [PYTHON, f"{UTILS}/plot_peaks_locus_view.py",
           "--peaks-csv", peaks_csv,
           "--gene-name", args.gene,
           "--output-dir", args.output_dir,
           "--label", f"{label}_locus_view",
           "--gtf", args.gtf]
    run(cmd, "[4/5] locus-view chromosome plot")

    # ── 5. 3-panel deep dives for top-N ────────────────────────────────
    if not args.no_3panel:
        panel_dir = f"{args.output_dir}/{label}_3panel"
        cmd = [PYTHON, f"{UTILS}/make_peak_3panel_figures.py",
               "--ranking-csv", ranking_csv,
               "--fimo-hits",   fimo_hits,
               "--output-dir",  panel_dir,
               "--top-n",       str(args.top_n_3panel)]
        # Use agent-curated TF biology if a previous run already produced one
        tf_csv = f"{args.output_dir}/{label}_tf_biology_table.csv"
        if os.path.exists(tf_csv):
            try:
                # Only use it if the agent has filled in any 'category' values
                df = pd.read_csv(tf_csv)
                if "category" in df.columns and df["category"].astype(str).str.len().sum() > 0:
                    cmd += ["--tf-biology-csv", tf_csv]
                    print(f"  → using agent-filled TF biology: {tf_csv}")
            except Exception:
                pass
        run(cmd, f"[5/5] 3-panel deep-dive (top {args.top_n_3panel} peaks)")

    # ── 6. (always) Generate TF biology research brief for agent curation ─
    cmd = [PYTHON, f"{UTILS}/tf_biology_lookup.py",
           "--fimo-hits", fimo_hits,
           "--label", label,
           "--output-dir", args.output_dir,
           "--top-n", "25"]
    if args.target_celltype:
        cmd += ["--target-tissue", args.target_celltype]
    cmd += ["--target-gene", args.gene]
    run(cmd, "[6/6] TF biology research brief (for agent curation)")

    print("\n" + "=" * 70)
    print(f"DONE — gene_locus_explore for {args.gene}")
    print("=" * 70)
    print(f"All outputs in: {args.output_dir}/")
    print(f"  peaks:      {label}_peaks.csv")
    print(f"  FIMO:       {fimo_label}_fimo_*.csv / .npz")
    print(f"  ranking:    {label}_enhancer_ranking.csv")
    print(f"  summary:    {label}_summary.txt")
    print(f"  locus view: {label}_locus_view.{{pdf,png}}")
    print(f"  motif maps: {label}_motif_maps/")
    if not args.no_3panel:
        print(f"  3-panel:    {label}_3panel/  (top {args.top_n_3panel} peaks)")


if __name__ == "__main__":
    main()
