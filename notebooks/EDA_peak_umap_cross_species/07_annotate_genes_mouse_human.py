# %% [markdown]
# # Step 07: Annotate Mouse / Human Peaks with Nearest Gene + Peak Type
#
# Input:
#   - Mouse h5ad  : .../public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad
#   - Human h5ad  : .../public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad
#   - Mouse GTF   : /hpc/reference/sequencing_alignment/gff_files/Mus_musculus.GRCm38.92.gtf.gz
#   - Human GTF   : /hpc/reference/sequencing_alignment/gff_files/Homo_sapiens.GRCh37.75.gtf.gz
#
# Output:
#   - {SCRATCH}/gene_annotations/mouse_peaks_gene_annotated.csv
#   - {SCRATCH}/gene_annotations/human_peaks_gene_annotated.csv
#   Columns: original_peak_id, chr, start, end, nearest_gene, gene_name,
#             distance_to_tss, peak_type, gene_body_overlaps
#
# Notes:
#   - Ensembl GTFs use bare chr names "1, 2, ..." but peak obs indices use "chr1, chr2, ..."
#     → we add "chr" prefix to GTF chromosomes inside the function
#   - peak_type hierarchy: promoter (<2 kb from TSS) > exonic > intronic > intergenic
#   - Uses vectorised PyRanges joins — no chunking loop needed
#
# Env: single-cell-base (CPU)
#   conda run -p /hpc/user_apps/data.science/conda_envs/single-cell-base python -u 07_annotate_genes_mouse_human.py

# %% Imports
import os, gc, time
import numpy as np
import pandas as pd
import pyranges as pr
import anndata as ad

print("Libraries loaded.")

# %% Paths
SCRATCH = "/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap"
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"

MOUSE_H5AD  = f"{BASE}/data/public_data/mouse_argelaguet_2022/peaks_by_pb_celltype_stage_annotated_v2.h5ad"
HUMAN_H5AD  = f"{BASE}/data/public_data/human_domcke_2020/peaks_by_pb_celltype_stage_annotated.h5ad"

MOUSE_GTF   = "/hpc/reference/sequencing_alignment/gff_files/Mus_musculus.GRCm38.92.gtf.gz"
HUMAN_GTF   = "/hpc/reference/sequencing_alignment/gff_files/Homo_sapiens.GRCh37.75.gtf.gz"

OUT_DIR = f"{SCRATCH}/gene_annotations"
os.makedirs(OUT_DIR, exist_ok=True)

PROMOTER_WINDOW = 2000   # bp upstream/downstream of TSS → "promoter"
MAX_TSS_DIST    = 50_000 # bp — only report nearest gene if within this distance


# %% ── Helper: parse peaks from h5ad obs ────────────────────────────────────────
def peaks_from_obs(obs_index) -> pd.DataFrame:
    """
    Build a peak DataFrame from obs.index.
    Handles both:
      - "chr1-100-200"  (mouse/human format)
      - "1-100-200"     (zebrafish / no-chr format, adds chr prefix)
    Returns columns: original_peak_id, Chromosome, Start, End
    """
    # Use list comprehension for robustness across pandas versions
    idx_list = list(obs_index)
    chrs, starts, ends = [], [], []
    for s in idx_list:
        # rsplit from right with maxsplit=2 handles both "chr1-100-200" and "1-100-200"
        p = s.rsplit("-", 2)
        chrs.append(p[0])
        starts.append(int(p[1]))
        ends.append(int(p[2]))
    df = pd.DataFrame({
        "original_peak_id": idx_list,
        "Chromosome":       chrs,
        "Start":            starts,
        "End":              ends,
    })
    # Ensure chr prefix
    mask = ~df["Chromosome"].str.startswith("chr")
    df.loc[mask, "Chromosome"] = "chr" + df.loc[mask, "Chromosome"]
    return df


# %% ── Helper: load GTF and extract gene/TSS PyRanges ───────────────────────────
def load_gtf_gene_tss(gtf_path: str):
    """
    Load GTF, return (genes_gr, tss_gr) as PyRanges.
    Adds "chr" prefix to chromosome names (Ensembl GTFs use bare names).
    """
    print(f"  Reading GTF: {gtf_path}")
    gtf = pr.read_gtf(gtf_path)

    # Add chr prefix if missing
    if not gtf.Chromosome.str.startswith("chr").all():
        gtf.Chromosome = "chr" + gtf.Chromosome.astype(str)

    genes = gtf[gtf.Feature == "gene"].copy()
    print(f"  Genes in GTF: {len(genes)}")

    # TSS: point coordinate
    tss = genes.copy()
    plus_mask  = (genes.Strand == "+").values
    minus_mask = (genes.Strand == "-").values

    starts = genes.Start.values.copy()
    ends   = genes.End.values.copy()

    # For + strand TSS = Start; for - strand TSS = End
    tss_pos = np.where(plus_mask, starts, ends)
    tss.Start = tss_pos
    tss.End   = tss_pos + 1

    return genes, tss


# %% ── Helper: promoter regions ─────────────────────────────────────────────────
def make_promoters(genes_gr, upstream=2000, downstream=100):
    """Return a PyRanges of promoter windows around each gene TSS."""
    prom = genes_gr.copy()
    plus  = (genes_gr.Strand == "+").values
    minus = (genes_gr.Strand == "-").values

    starts = genes_gr.Start.values.copy()
    ends   = genes_gr.End.values.copy()

    new_start = np.empty(len(genes_gr), dtype=int)
    new_end   = np.empty(len(genes_gr), dtype=int)

    # + strand: TSS = Start → [TSS-upstream, TSS+downstream]
    new_start[plus]  = np.maximum(0, starts[plus]  - upstream)
    new_end[plus]    = starts[plus]  + downstream

    # - strand: TSS = End   → [TSS-downstream, TSS+upstream]
    new_start[minus] = np.maximum(0, ends[minus] - downstream)
    new_end[minus]   = ends[minus] + upstream

    prom.Start = new_start
    prom.End   = new_end
    return prom


# %% ── Main annotation function ─────────────────────────────────────────────────
def annotate_peaks(obs_index: pd.Index, gtf_path: str, species_label: str) -> pd.DataFrame:
    """
    Annotate peaks from obs_index using the given GTF.

    Returns a DataFrame with columns:
      original_peak_id, chr, start, end,
      nearest_gene, distance_to_tss, peak_type, gene_body_overlaps
    """
    t0 = time.time()
    peaks_df = peaks_from_obs(obs_index)
    print(f"  [{species_label}] {len(peaks_df):,} peaks loaded.")

    genes_gr, tss_gr = load_gtf_gene_tss(gtf_path)
    prom_gr          = make_promoters(genes_gr)
    exons_gr         = pr.read_gtf(gtf_path)
    if not exons_gr.Chromosome.str.startswith("chr").all():
        exons_gr.Chromosome = "chr" + exons_gr.Chromosome.astype(str)
    exons_gr = exons_gr[exons_gr.Feature == "exon"].copy()

    peaks_gr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End"]].copy())

    # ── 1. Nearest TSS (vectorised) ─────────────────────────────────────────────
    print(f"  [{species_label}] Computing nearest TSS ...")
    nearest = peaks_gr.nearest(tss_gr, suffix="_tss", apply_strand_suffix=False)
    near_df = nearest.as_df()

    # distance = |peak_midpoint - tss_pos|
    mid     = (near_df["Start"] + near_df["End"]) // 2
    tss_pos = near_df["Start_tss"]   # TSS is a 1-bp interval; Start_tss == tss position
    near_df["distance_to_tss"] = (mid - tss_pos).abs()

    # Rename gene_name column from TSS join (it inherits from the TSS pyranges)
    gene_col = "gene_name" if "gene_name" in near_df.columns else "Name"
    near_df = near_df.rename(columns={gene_col: "nearest_gene_from_tss"})

    # Index by (Chromosome, Start, End) for fast lookup
    near_df_idx = near_df.set_index(["Chromosome", "Start", "End"])

    # ── 2. Gene body overlaps ────────────────────────────────────────────────────
    print(f"  [{species_label}] Computing gene body overlaps ...")
    gb_join = peaks_gr.join(genes_gr, suffix="_gene", apply_strand_suffix=False)
    if not gb_join.empty:
        gb_df = gb_join.as_df()
        gb_col = "gene_name_gene" if "gene_name_gene" in gb_df.columns else "gene_name"
        gb_agg = (gb_df
                  .groupby(["Chromosome", "Start", "End"])[gb_col]
                  .apply(lambda x: ",".join(sorted(set(x.astype(str)))))
                  .reset_index()
                  .rename(columns={gb_col: "gene_body_overlaps"}))
        gb_agg = gb_agg.set_index(["Chromosome", "Start", "End"])
    else:
        gb_agg = pd.DataFrame(columns=["gene_body_overlaps"])
        gb_agg.index.names = ["Chromosome", "Start", "End"]

    # ── 3. Peak type ─────────────────────────────────────────────────────────────
    print(f"  [{species_label}] Classifying peak types ...")
    prom_peaks  = peaks_gr.overlap(prom_gr)
    exon_peaks  = peaks_gr.overlap(exons_gr)
    gene_peaks  = peaks_gr.overlap(genes_gr)

    def to_tuples(gr):
        if gr.empty:
            return set()
        df = gr.as_df()
        return set(zip(df["Chromosome"], df["Start"], df["End"]))

    prom_set  = to_tuples(prom_peaks)
    exon_set  = to_tuples(exon_peaks)
    gene_set  = to_tuples(gene_peaks)

    # ── 4. Assemble result ───────────────────────────────────────────────────────
    print(f"  [{species_label}] Assembling result ...")
    result = peaks_df.copy()
    result = result.rename(columns={"Chromosome": "chr", "Start": "start", "End": "end"})
    result["peak_type"]         = "intergenic"
    result["nearest_gene"]      = ""
    result["distance_to_tss"]   = np.nan
    result["gene_body_overlaps"] = ""

    for i, row in result.iterrows():
        key = (row["chr"], row["start"], row["end"])

        # peak_type hierarchy
        if key in prom_set:
            result.at[i, "peak_type"] = "promoter"
        elif key in exon_set:
            result.at[i, "peak_type"] = "exonic"
        elif key in gene_set:
            result.at[i, "peak_type"] = "intronic"

        # nearest gene
        if key in near_df_idx.index:
            r = near_df_idx.loc[key]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            dist = r["distance_to_tss"]
            if dist <= MAX_TSS_DIST:
                result.at[i, "nearest_gene"]    = r["nearest_gene_from_tss"]
                result.at[i, "distance_to_tss"] = dist

        # gene body overlaps
        if key in gb_agg.index:
            result.at[i, "gene_body_overlaps"] = gb_agg.loc[key, "gene_body_overlaps"]

    print(f"  [{species_label}] Done in {time.time()-t0:.1f}s")
    print(f"  Peak type counts:\n{result['peak_type'].value_counts()}")
    return result


# %% ── Vectorised annotation (avoids slow per-row loop) ─────────────────────────
def annotate_peaks_vectorised(obs_index: pd.Index, gtf_path: str, species_label: str) -> pd.DataFrame:
    """
    Fully vectorised version using PyRanges merge operations.
    Handles large peak sets (>500K) efficiently.
    """
    t0 = time.time()
    peaks_df = peaks_from_obs(obs_index)
    N = len(peaks_df)
    print(f"  [{species_label}] {N:,} peaks")

    genes_gr, tss_gr = load_gtf_gene_tss(gtf_path)
    prom_gr          = make_promoters(genes_gr)

    # Load exons separately (re-read; share the chr-fix logic)
    gtf_raw = pr.read_gtf(gtf_path)
    if not gtf_raw.Chromosome.str.startswith("chr").all():
        gtf_raw.Chromosome = "chr" + gtf_raw.Chromosome.astype(str)
    exons_gr = gtf_raw[gtf_raw.Feature == "exon"].copy()

    peaks_gr = pr.PyRanges(peaks_df[["Chromosome", "Start", "End"]].copy())

    # ── nearest TSS ─────────────────────────────────────────────────────────────
    print(f"  [{species_label}] nearest TSS ...")
    near = peaks_gr.nearest(tss_gr, suffix="_tss", apply_strand_suffix=False)
    near_df = near.as_df().copy()
    mid = (near_df["Start"] + near_df["End"]) // 2
    near_df["distance_to_tss"] = (mid - near_df["Start_tss"]).abs()

    gene_col = next((c for c in ["gene_name", "Name", "gene_id"] if c in near_df.columns), None)
    if gene_col:
        near_df = near_df.rename(columns={gene_col: "nearest_gene"})
    else:
        near_df["nearest_gene"] = ""

    near_df.loc[near_df["distance_to_tss"] > MAX_TSS_DIST, "nearest_gene"] = ""
    near_df.loc[near_df["distance_to_tss"] > MAX_TSS_DIST, "distance_to_tss"] = np.nan

    near_small = near_df[["Chromosome", "Start", "End", "nearest_gene", "distance_to_tss"]].copy()
    near_small = near_small.drop_duplicates(["Chromosome", "Start", "End"])
    del near_df, near; gc.collect()

    # Free GTF raw after exon extraction
    if "gtf_raw" in dir():
        del gtf_raw; gc.collect()

    # ── peak type ────────────────────────────────────────────────────────────────
    # Skip gene body overlaps join — too memory-intensive for 1M+ peaks.
    # gene_body_overlaps column will be left empty; not needed for ortholog matching.
    gb_agg = pd.DataFrame(columns=["Chromosome", "Start", "End", "gene_body_overlaps"])
    print(f"  [{species_label}] peak types ...")

    prom_ol  = peaks_gr.overlap(prom_gr).as_df()
    prom_set = set(zip(prom_ol["Chromosome"], prom_ol["Start"], prom_ol["End"])) if len(prom_ol) else set()
    del prom_ol; gc.collect()

    exon_ol  = peaks_gr.overlap(exons_gr).as_df()
    exon_set = set(zip(exon_ol["Chromosome"], exon_ol["Start"], exon_ol["End"])) if len(exon_ol) else set()
    del exon_ol, exons_gr; gc.collect()

    gene_ol  = peaks_gr.overlap(genes_gr).as_df()
    gene_set = set(zip(gene_ol["Chromosome"], gene_ol["Start"], gene_ol["End"])) if len(gene_ol) else set()
    del gene_ol, genes_gr, prom_gr, tss_gr; gc.collect()

    # Vectorised peak_type assignment
    keys = list(zip(peaks_df["Chromosome"], peaks_df["Start"], peaks_df["End"]))
    peak_type = np.full(N, "intergenic", dtype=object)
    peak_type[np.array([k in gene_set  for k in keys])] = "intronic"
    peak_type[np.array([k in exon_set  for k in keys])] = "exonic"
    peak_type[np.array([k in prom_set  for k in keys])] = "promoter"

    # ── assemble ─────────────────────────────────────────────────────────────────
    print(f"  [{species_label}] assembling ...")
    result = peaks_df.rename(columns={"Chromosome": "chr", "Start": "start", "End": "end"}).copy()
    result["peak_type"] = peak_type

    # merge nearest gene
    near_small = near_small.rename(columns={"Chromosome": "chr", "Start": "start", "End": "end"})
    result = result.merge(near_small, on=["chr", "start", "end"], how="left")

    # merge gene body overlaps
    if len(gb_agg):
        gb_agg = gb_agg.rename(columns={"Chromosome": "chr", "Start": "start", "End": "end"})
        result = result.merge(gb_agg, on=["chr", "start", "end"], how="left")
    else:
        result["gene_body_overlaps"] = ""

    result["nearest_gene"]       = result["nearest_gene"].fillna("")
    result["gene_body_overlaps"] = result["gene_body_overlaps"].fillna("")

    print(f"  [{species_label}] Done in {time.time()-t0:.1f}s")
    print(f"  Peak type counts:\n{result['peak_type'].value_counts()}")
    return result


# %% ── Run mouse ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MOUSE")
print("="*60)

print(f"Loading mouse h5ad: {MOUSE_H5AD}")
t0 = time.time()
adata_mouse = ad.read_h5ad(MOUSE_H5AD, backed="r")
print(f"  Mouse obs: {adata_mouse.n_obs:,}   ({time.time()-t0:.1f}s)")

mouse_result = annotate_peaks_vectorised(adata_mouse.obs_names, MOUSE_GTF, "mouse")
del adata_mouse; gc.collect()

out_mouse = f"{OUT_DIR}/mouse_peaks_gene_annotated.csv"
mouse_result.to_csv(out_mouse, index=False)
print(f"Saved: {out_mouse}  ({len(mouse_result):,} rows)")


# %% ── Run human ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HUMAN")
print("="*60)

print(f"Loading human h5ad: {HUMAN_H5AD}")
t0 = time.time()
adata_human = ad.read_h5ad(HUMAN_H5AD, backed="r")
print(f"  Human obs: {adata_human.n_obs:,}   ({time.time()-t0:.1f}s)")

human_result = annotate_peaks_vectorised(adata_human.obs_names, HUMAN_GTF, "human")
del adata_human; gc.collect()

out_human = f"{OUT_DIR}/human_peaks_gene_annotated.csv"
human_result.to_csv(out_human, index=False)
print(f"Saved: {out_human}  ({len(human_result):,} rows)")

print("\nAll done.")
