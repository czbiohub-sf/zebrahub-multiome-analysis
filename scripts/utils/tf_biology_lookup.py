"""
Generate a TF biology research brief from FIMO output.

Aggregates the most-frequently-hit TFs across a peak set, then writes
a markdown file with prepared research queries (ZFIN, PubMed, NCBI
Gene, Alliance of Genome Resources) and a stub CSV that an agent can
fill in via web research, replacing the hardcoded MHB_EXPLICIT /
MHB_IMPLICIT curation in `make_peak_3panel_figures.py`.

Workflow
--------
  (a) Run gene_locus_explore.py for any gene → produces FIMO hits CSV.
  (b) Run this script: aggregates top-N TFs and writes
        {label}_tf_research_brief.md         human-readable brief
        {label}_tf_biology_table.csv         stub for agent classification
                                              (columns: tf, n_peaks_bound,
                                               fraction, jaspar_accession,
                                               category, citations, notes)
  (c) An agent (Claude with WebSearch / WebFetch) reads the brief,
      runs the suggested queries, and fills `category` (EXPLICIT /
      IMPLICIT / IRRELEVANT) + `citations` + `notes` for each TF.
  (d) Pass the filled CSV to make_peak_3panel_figures.py via
      --tf-biology-csv to produce tissue-specific figures.

Usage
-----
  python tf_biology_lookup.py \\
      --fimo-hits results/pax2a/pax2a_jaspar_fimo_hits.csv \\
      --label pax2a \\
      --target-tissue midbrain_hindbrain_boundary \\
      --target-gene  pax2a \\
      --output-dir   results/pax2a/ \\
      --top-n 25
"""

import os, sys, argparse, urllib.parse
import numpy as np
import pandas as pd

REPO = "/hpc/projects/data.science/yangjoom_KIM/zebrahub_multiome/zebrahub-multiome-analysis"
# (REPO not actually used — kept for symmetry with sibling scripts)


def make_research_queries(tf_name: str, target_tissue: str = None,
                           target_gene: str = None, species: str = "zebrafish"):
    """Return a dict of {database: url} for prepared lookup queries."""
    # JASPAR2024 dimer labels like "PAX_PHOX2" — search each component
    primary = tf_name.split("_")[0].split("-")[0].upper()
    primary_lc = primary.lower()
    queries = {}

    # ZFIN gene quicksearch
    queries["ZFIN"] = (
        f"https://zfin.org/action/quicksearch/query?searchString="
        f"{urllib.parse.quote_plus(primary_lc)}"
    )

    # NCBI Gene (zebrafish-restricted)
    queries["NCBI_Gene"] = (
        "https://www.ncbi.nlm.nih.gov/gene/?term="
        + urllib.parse.quote_plus(f"{primary_lc} AND Danio rerio[orgn]")
    )

    # Alliance of Genome Resources
    queries["Alliance"] = (
        "https://www.alliancegenome.org/search?q="
        + urllib.parse.quote_plus(primary_lc)
        + "&category=gene&species=Danio%20rerio"
    )

    # PubMed: TF + tissue + species
    pm_terms = [primary_lc]
    if target_tissue:
        pm_terms.append(target_tissue.replace("_", " "))
    if target_gene and target_gene.lower() != primary_lc:
        pm_terms.append(target_gene)
    pm_terms.append(species)
    queries["PubMed_tissue"] = (
        "https://pubmed.ncbi.nlm.nih.gov/?term="
        + urllib.parse.quote_plus(" AND ".join(pm_terms))
    )

    # Generic PubMed: TF + zebrafish (broader)
    queries["PubMed_general"] = (
        "https://pubmed.ncbi.nlm.nih.gov/?term="
        + urllib.parse.quote_plus(f"{primary_lc} AND zebrafish")
    )

    return queries


def aggregate_tfs(hits_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Return per-TF stats, sorted by n_peaks_bound desc."""
    n_peaks_total = hits_df["peak_id"].nunique()
    grp = hits_df.groupby("tf").agg(
        n_peaks_bound=("peak_id", "nunique"),
        n_total_hits=("peak_id", "size"),
        best_pvalue=("pvalue", "min"),
        median_pvalue=("pvalue", "median"),
        jaspar_accession=("motif_accession",
                          lambda s: ",".join(sorted(set(s))[:3])),
    ).sort_values("n_peaks_bound", ascending=False)
    grp["fraction_peaks"] = (grp["n_peaks_bound"] / n_peaks_total).round(3)
    return grp.head(top_n).reset_index()


def write_research_brief(tfs_df: pd.DataFrame,
                          label: str,
                          target_tissue: str,
                          target_gene: str,
                          n_peaks_total: int,
                          out_md: str,
                          out_csv: str):
    """Write the human-readable brief + an agent-ready stub CSV."""
    lines = []
    lines.append(f"# TF biology research brief — {label}")
    lines.append("")
    lines.append(f"**Target gene:** `{target_gene or 'n/a'}`  ")
    lines.append(f"**Target tissue:** `{target_tissue or 'n/a'}`  ")
    lines.append(f"**Peak set:** {n_peaks_total} peaks")
    lines.append("")
    lines.append("## Instructions for the curating agent")
    lines.append("")
    lines.append("For each TF below, run the prepared queries (ZFIN, NCBI Gene, "
                  "Alliance, PubMed) and decide whether the TF is biologically "
                  "relevant for the target tissue. Fill the corresponding row in "
                  f"`{os.path.basename(out_csv)}` with:")
    lines.append("")
    lines.append("- `category` — one of:")
    lines.append("  - **EXPLICIT** — direct, well-established regulator / effector "
                  "of the target tissue (e.g., paired-box TFs for MHB, SOX10 for neural crest)")
    lines.append("  - **IMPLICIT** — broader neural / developmental TF that is "
                  "co-expressed at the target tissue but not target-defining")
    lines.append("  - **IRRELEVANT** — no known role in the target tissue "
                  "(promiscuous zinc-fingers, off-tissue TFs)")
    lines.append("- `citations` — 1–3 PMID / DOI / ZFIN accession entries supporting "
                  "the call (semicolon-delimited)")
    lines.append("- `notes` — 1-line mechanism / role summary")
    lines.append("")
    lines.append("The filled CSV is then passed to "
                  "`make_peak_3panel_figures.py --tf-biology-csv` to replace "
                  "the hard-coded MHB curation with tissue-specific calls.")
    lines.append("")

    lines.append("## Top TFs by peak coverage")
    lines.append("")
    for _, r in tfs_df.iterrows():
        tf   = r["tf"]
        nb   = int(r["n_peaks_bound"])
        frac = r["fraction_peaks"]
        bp   = r["best_pvalue"]
        acc  = r["jaspar_accession"]
        q = make_research_queries(tf, target_tissue, target_gene)
        lines.append(f"### `{tf}` — {nb} / {n_peaks_total} peaks ({frac:.0%}), "
                      f"best p={bp:.1e}")
        lines.append(f"- JASPAR accession(s): `{acc}`")
        lines.append(f"- ZFIN: {q['ZFIN']}")
        lines.append(f"- NCBI Gene (Danio rerio): {q['NCBI_Gene']}")
        lines.append(f"- Alliance Genome (Danio rerio): {q['Alliance']}")
        lines.append(f"- PubMed (TF + tissue): {q['PubMed_tissue']}")
        lines.append(f"- PubMed (TF + zebrafish): {q['PubMed_general']}")
        lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Stub CSV for agent classification
    stub = tfs_df.copy()
    stub["category"] = ""        # EXPLICIT / IMPLICIT / IRRELEVANT — agent fills
    stub["citations"] = ""        # PMID/DOI/ZFIN — agent fills
    stub["notes"] = ""             # 1-line mechanism — agent fills
    stub.to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fimo-hits", required=True,
                   help="FIMO hits CSV (output of run_fimo_on_peaks.py)")
    p.add_argument("--label",      required=True, help="Output filename prefix")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-tissue", default=None,
                   help="Target tissue / celltype name (used in PubMed query)")
    p.add_argument("--target-gene", default=None,
                   help="Target gene name (used in PubMed query)")
    p.add_argument("--top-n", type=int, default=25,
                   help="Number of top-binding TFs to include (default 25)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hits = pd.read_csv(args.fimo_hits)
    n_peaks_total = hits["peak_id"].nunique()
    print(f"Loaded {len(hits)} hits across {n_peaks_total} peaks "
          f"({hits['tf'].nunique()} unique TFs)")

    tfs = aggregate_tfs(hits, top_n=args.top_n)
    print(f"Top {len(tfs)} TFs (by # peaks bound):")
    print(tfs[["tf", "n_peaks_bound", "fraction_peaks", "best_pvalue"]].to_string(index=False))

    out_md  = f"{args.output_dir}/{args.label}_tf_research_brief.md"
    out_csv = f"{args.output_dir}/{args.label}_tf_biology_table.csv"
    write_research_brief(tfs, args.label, args.target_tissue, args.target_gene,
                          n_peaks_total, out_md, out_csv)
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {out_csv}  (agent fills 'category', 'citations', 'notes')")


if __name__ == "__main__":
    main()
