# %% [markdown]
# # Script 09: V3 Celltype-Level Parts List
#
# Two-step approach that separates celltype specificity from temporal dynamics:
#   Step 1: Average log-norm accessibility across reliable timepoints per celltype
#           → 640K × ~31 matrix → leave-one-out z-score across celltypes
#   Step 2: For top peaks per celltype, extract temporal profiles + motif enrichment
#
# Motivation: The V2 190-condition z-score penalises peaks that are constitutively
# accessible across multiple timepoints within a celltype.  This V3 approach
# correctly identifies celltype-specific peaks regardless of their temporal pattern.
#
# Follows methodology in notebooks/Fig_peak_umap/09_annotate_peak_umap_celltype_timepoints.py
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

# %% Imports
import os, re, gc, time, warnings
import numpy as np
import pandas as pd
import anndata as ad
import pysam
from pymemesuite.common import (MotifFile, Sequence as MemeSequence,
                                 Background, Array as MemeArray)
from pymemesuite.fimo import FIMO as FIMOScanner
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# ── Publication figure settings (exact pattern from 01_EDA_annotate_peak_umap.py) ──
import matplotlib as _mpl
_mpl.rcParams.update(_mpl.rcParamsDefault)   # 1. reset all rcParams to defaults
_mpl.rcParams['font.family'] = 'Arial'      # 2. explicit Arial font
_mpl.rcParams["pdf.fonttype"] = 42          # 3. editable text in Illustrator
_mpl.rcParams["ps.fonttype"]  = 42
import seaborn as _sns
_sns.set(style="whitegrid", context="paper") # 4. seaborn (after fonttype)
_mpl.rcParams["savefig.dpi"]  = 300         # 5. DPI re-set after sns.set()
# ────────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=FutureWarning)
print(f"=== Script 09: V3 Celltype-Level Parts List ===")
print(f"Start: {time.strftime('%c')}")
print(f"Host: {os.uname().nodename}")
print(f"Python: {os.popen('which python').read().strip()}")
print(f"anndata {ad.__version__}")

# %% Paths
BASE    = "/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome"
REPO    = f"{BASE}/zebrahub-multiome-analysis"
OUTDIR  = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3"
FIG_V3  = f"{REPO}/figures/peak_parts_list/V3"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/top50_peaks_per_celltype", exist_ok=True)
os.makedirs(f"{FIG_V3}/specificity_overview", exist_ok=True)
os.makedirs(f"{FIG_V3}/temporal", exist_ok=True)
os.makedirs(f"{FIG_V3}/motif_enrichment", exist_ok=True)

MASTER_H5AD = (f"{BASE}/data/annotated_data/objects_v2/"
               "peaks_by_ct_tp_master_anno.h5ad")
FASTA_PATH  = "/hpc/reference/sequencing_alignment/fasta_references/danRer11.primary.fa"
MEME_PATH   = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
               "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")

# Also load V2 specificity matrix for comparison
SPEC_V2 = f"{REPO}/notebooks/EDA_peak_parts_list/outputs/specificity_matrix_v2.h5ad"

# %% Configuration
FOCAL_CELLTYPES = [
    "fast_muscle",
    "heart_myocardium",
    "neural_crest",
    "PSM",
    "notochord",
    "epidermis",
    "hemangioblasts",
]
TOP_N_PEAKS = 50
PVAL_THRESH = 1e-4   # FIMO p-value threshold
MIN_ZSCORE  = 2.0    # minimum z-score for peak inclusion
MIN_CELLS   = 20     # reliability threshold

TIMEPOINT_ORDER = ["0somites", "5somites", "10somites",
                   "15somites", "20somites", "30somites"]
TP_INT = {tp: int(tp.replace("somites", "")) for tp in TIMEPOINT_ORDER}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Celltype-level specificity z-score
# ══════════════════════════════════════════════════════════════════════════════

# %% Load master h5ad
print("\n--- Phase 1: Celltype-level specificity ---")
print(f"Loading {MASTER_H5AD.split('/')[-1]} ...", flush=True)
t0 = time.time()
adata = ad.read_h5ad(MASTER_H5AD)
M = np.array(adata.X, dtype=np.float64)  # (640830, 190) log-norm
obs = adata.obs.copy()
print(f"  Shape: {adata.shape}  ({time.time()-t0:.1f}s)", flush=True)

# %% Parse var names → celltype / timepoint mapping
def parse_condition(cond):
    m = re.search(r"(\d+somites)$", cond)
    if not m:
        return cond, ""
    tp = m.group(1)
    ct = cond[:-(len(tp)+1)]
    return ct, tp

cond_meta = pd.DataFrame(
    [parse_condition(c) for c in adata.var_names],
    columns=["celltype", "timepoint"],
    index=adata.var_names,
)
cond_meta["n_cells"] = adata.var["n_cells"].values
cond_meta["reliable"] = cond_meta["n_cells"] >= MIN_CELLS
cond_meta["tp_int"] = cond_meta["timepoint"].map(TP_INT)

n_reliable = cond_meta["reliable"].sum()
print(f"  Reliable conditions: {n_reliable}/{len(cond_meta)} (n_cells >= {MIN_CELLS})")
print(f"  Filtered out: {len(cond_meta) - n_reliable} conditions")

# %% Identify reliable celltypes
celltype_mapping = {}  # col → celltype
timepoint_mapping = {}  # col → timepoint
for col in adata.var_names:
    ct, tp = parse_condition(col)
    celltype_mapping[col] = ct
    timepoint_mapping[col] = tp

reliable_groups = cond_meta[cond_meta["reliable"]].index.tolist()

# Find celltypes with at least 1 reliable timepoint
reliable_celltypes = sorted(set(
    celltype_mapping[col] for col in reliable_groups
))
print(f"  Reliable celltypes: {len(reliable_celltypes)}")

# Check if primordial_germ_cells has any reliable timepoints
pgc_reliable = [col for col in reliable_groups
                if celltype_mapping[col] == "primordial_germ_cells"]
if not pgc_reliable:
    reliable_celltypes = [ct for ct in reliable_celltypes
                          if ct != "primordial_germ_cells"]
    print(f"  Excluded primordial_germ_cells (all timepoints unreliable)")
    print(f"  Final reliable celltypes: {len(reliable_celltypes)}")

# %% Build celltype-level mean matrix (unweighted mean across reliable timepoints)
print("\nBuilding celltype-level mean matrix ...", flush=True)
t0 = time.time()

ct_names = reliable_celltypes
n_peaks = M.shape[0]
n_ct = len(ct_names)
ct_mean_matrix = np.zeros((n_peaks, n_ct), dtype=np.float64)

for ct_idx, ct in enumerate(ct_names):
    # Get reliable columns for this celltype
    ct_cols = [col for col, c in celltype_mapping.items()
               if c == ct and col in reliable_groups]
    col_indices = [list(adata.var_names).index(c) for c in ct_cols]
    ct_mean_matrix[:, ct_idx] = np.mean(M[:, col_indices], axis=1)
    print(f"  {ct}: mean of {len(ct_cols)} reliable timepoints "
          f"({', '.join(timepoint_mapping[c] for c in ct_cols)})")

print(f"  ct_mean_matrix shape: {ct_mean_matrix.shape}  ({time.time()-t0:.1f}s)")

# %% Compute leave-one-out z-score across celltypes (vectorized)
print("\nComputing leave-one-out z-score (vectorized) ...", flush=True)
t0 = time.time()

C = ct_mean_matrix.shape[1]  # ~31
row_sum = ct_mean_matrix.sum(axis=1)             # (640830,)
row_sq  = (ct_mean_matrix ** 2).sum(axis=1)      # (640830,)

mean_other = (row_sum[:, None] - ct_mean_matrix) / (C - 1)
var_other  = (row_sq[:, None] - ct_mean_matrix**2) / (C - 1) - mean_other**2
std_other  = np.sqrt(np.maximum(var_other, 1e-10))

Z_ct = (ct_mean_matrix - mean_other) / std_other  # (640830, ~31)
print(f"  Z_ct shape: {Z_ct.shape}  ({time.time()-t0:.1f}s)")
print(f"  Z_ct range: [{Z_ct.min():.2f}, {Z_ct.max():.2f}]")

# %% Save V3 specificity matrix as AnnData
print("\nSaving V3 specificity matrix ...", flush=True)
z_adata = ad.AnnData(
    X=Z_ct.astype(np.float32),
    obs=obs,
    var=pd.DataFrame(index=ct_names),
)
z_adata.write_h5ad(f"{OUTDIR}/V3_specificity_matrix_celltype_level.h5ad")
print(f"  Saved: {OUTDIR}/V3_specificity_matrix_celltype_level.h5ad")

# Summary stats per celltype
print("\nV3 specificity summary:")
print(f"  {'Celltype':<35} {'z>=2':>8} {'z>=4':>8} {'max_z':>8}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
for ct_idx, ct in enumerate(ct_names):
    z_col = Z_ct[:, ct_idx]
    n_z2 = (z_col >= 2.0).sum()
    n_z4 = (z_col >= 4.0).sum()
    print(f"  {ct:<35} {n_z2:>8,} {n_z4:>8,} {z_col.max():>8.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Top peaks, temporal profiles, FASTA export, motif enrichment
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Phase 2: Top peaks + temporal profiles + motif enrichment ---")

# %% Load motif database (pymemesuite)
print(f"\nLoading JASPAR H12CORE motifs ...", flush=True)
t0 = time.time()
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)
# Name is in .accession (pymemesuite bug: .name is empty)
motif_names = [m.accession.decode() if isinstance(m.accession, bytes) else str(m.accession)
               for m in motif_list]
motif_tf_names = [n.split(".")[0] for n in motif_names]
# Background for FIMO scoring
_bg = Background(motif_list[0].alphabet, MemeArray([0.25, 0.25, 0.25, 0.25]))
print(f"  {len(motif_names)} motifs loaded  ({time.time()-t0:.1f}s)")
print(f"  Example: {motif_names[0]} -> TF: {motif_tf_names[0]}")

# %% Select top peaks per celltype and extract sequences
print("\nSelecting top peaks and extracting sequences ...", flush=True)
fa = pysam.FastaFile(FASTA_PATH)
obs["chrom_str"] = "chr" + obs["chrom"].astype(str)

celltype_peak_info = {}  # ct → DataFrame of top peaks
celltype_seqs      = {}  # ct → list of DNA sequences
celltype_peak_idx  = {}  # ct → array of peak indices

for ct in FOCAL_CELLTYPES:
    if ct not in ct_names:
        print(f"  {ct}: not in reliable celltypes — skipping")
        continue

    ct_idx = ct_names.index(ct)
    z_vec  = Z_ct[:, ct_idx]

    # Top N peaks with z >= MIN_ZSCORE
    n_pass  = (z_vec >= MIN_ZSCORE).sum()
    top_idx = np.argsort(z_vec)[::-1]
    top_idx = top_idx[z_vec[top_idx] >= MIN_ZSCORE][:TOP_N_PEAKS]

    # Build info DataFrame
    info = obs.iloc[top_idx][["chrom", "start", "end", "chrom_str",
                               "peak_type", "nearest_gene", "distance_to_tss",
                               "linked_gene", "associated_gene",
                               "leiden_coarse"]].copy()
    info["V3_zscore"] = z_vec[top_idx]
    info["peak_id"]   = obs.index[top_idx]

    # Extract temporal profiles from original 190-condition matrix
    ct_tp_cols = [col for col, c in celltype_mapping.items()
                  if c == ct and col in reliable_groups]
    ct_tp_cols_sorted = sorted(ct_tp_cols, key=lambda x: TP_INT.get(timepoint_mapping[x], 99))
    tp_labels = [timepoint_mapping[c] for c in ct_tp_cols_sorted]
    col_indices_tp = [list(adata.var_names).index(c) for c in ct_tp_cols_sorted]

    temporal_mat = M[top_idx][:, col_indices_tp]  # (top_n, n_timepoints)
    for ti, tp_label in enumerate(tp_labels):
        info[f"acc_{tp_label}"] = temporal_mat[:, ti]

    # Extract DNA sequences
    seqs = []
    valid_rows = []
    for row_i, (_, row) in enumerate(info.iterrows()):
        chrom = str(row["chrom_str"])
        start = int(row["start"])
        end   = int(row["end"])
        try:
            seq = fa.fetch(chrom, start, end).upper()
            if len(seq) > 0 and seq.count("N") / len(seq) < 0.5:
                seqs.append(seq)
                valid_rows.append(row_i)
        except Exception:
            pass

    celltype_peak_info[ct] = info
    celltype_seqs[ct]      = seqs
    celltype_peak_idx[ct]  = top_idx[valid_rows] if valid_rows else top_idx

    print(f"  {ct}: {len(seqs)} seqs, z_max={z_vec[top_idx[0]]:.2f}  "
          f"(z>={MIN_ZSCORE}: {n_pass:,})")

    # ── Export FASTA for S2F sequence design ──
    fasta_path = f"{OUTDIR}/top50_peaks_per_celltype/{ct}_top50_peaks.fasta"
    with open(fasta_path, "w") as fh:
        for row_i, seq in zip(valid_rows, seqs):
            row = info.iloc[row_i]
            peak_id = row["peak_id"]
            chrom   = str(row["chrom_str"])
            start   = int(row["start"])
            end     = int(row["end"])
            zscore  = row["V3_zscore"]
            gene    = row["nearest_gene"] if pd.notna(row["nearest_gene"]) else "NA"
            linked  = row["linked_gene"] if pd.notna(row["linked_gene"]) else "NA"
            header  = (f">{peak_id}|{chrom}:{start}-{end}|"
                       f"z={zscore:.2f}|nearest={gene}|linked={linked}")
            fh.write(f"{header}\n{seq}\n")
    print(f"    FASTA: {fasta_path.split('/')[-1]}")

fa.close()

# %% Save top peaks table
all_peaks = []
for ct in FOCAL_CELLTYPES:
    if ct not in celltype_peak_info:
        continue
    df = celltype_peak_info[ct].copy()
    df["celltype"] = ct
    all_peaks.append(df)

all_peaks_df = pd.concat(all_peaks, axis=0)
all_peaks_df.to_csv(f"{OUTDIR}/V3_celltype_level_top_peaks.csv")
print(f"\nSaved top peaks: {OUTDIR}/V3_celltype_level_top_peaks.csv")

# %% Save temporal profiles
temporal_rows = []
for ct in FOCAL_CELLTYPES:
    if ct not in celltype_peak_info:
        continue
    info = celltype_peak_info[ct]
    acc_cols = [c for c in info.columns if c.startswith("acc_")]
    for _, row in info.iterrows():
        for acc_col in acc_cols:
            tp = acc_col.replace("acc_", "")
            temporal_rows.append({
                "celltype": ct,
                "peak_id": row["peak_id"],
                "V3_zscore": row["V3_zscore"],
                "nearest_gene": row["nearest_gene"],
                "timepoint": tp,
                "tp_int": TP_INT.get(tp, -1),
                "accessibility": row[acc_col],
            })

temporal_df = pd.DataFrame(temporal_rows)
temporal_df.to_csv(f"{OUTDIR}/V3_celltype_level_temporal_profiles.csv", index=False)
print(f"Saved temporal profiles: {OUTDIR}/V3_celltype_level_temporal_profiles.csv")

# ══════════════════════════════════════════════════════════════════════════════
# MOTIF ENRICHMENT with Fisher's exact test
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nRunning FIMO (pymemesuite, p<{PVAL_THRESH:.0e}) ...", flush=True)

n_motifs = len(motif_list)
ct_list  = [ct for ct in FOCAL_CELLTYPES if ct in celltype_seqs]
fimo_scanner = FIMOScanner(both_strands=True, threshold=PVAL_THRESH)

# hit_binary[ct][i, j] = 1 if peak i has hit for motif j
hit_binary = {}     # ct → (n_peaks, n_motifs) bool array
hit_rate_mat = np.zeros((len(ct_list), n_motifs), dtype=np.float32)

for ct_idx, ct in enumerate(ct_list):
    seqs   = celltype_seqs[ct]
    n_seqs = len(seqs)
    if n_seqs == 0:
        hit_binary[ct] = np.zeros((0, n_motifs), dtype=bool)
        continue

    print(f"  {ct}: scanning {n_seqs} seqs x {n_motifs} motifs ...", end=" ", flush=True)
    t0 = time.time()

    # Convert to pymemesuite Sequence objects
    meme_seqs = [MemeSequence(s, f"seq_{i}".encode()) for i, s in enumerate(seqs)]

    peak_motif_hits = np.zeros((n_seqs, n_motifs), dtype=bool)
    for j, motif in enumerate(motif_list):
        pattern = fimo_scanner.score_motif(motif, meme_seqs, _bg)
        if pattern is not None:
            for me in pattern.matched_elements:
                # sequence_name is like b'seq_3'
                seq_name = me.source.name.decode() if hasattr(me.source, 'name') else ""
                # Parse sequence index from name
                try:
                    seq_idx = int(seq_name.split("_")[1])
                except (IndexError, ValueError):
                    # Fallback: matched_elements source may have sequence_name
                    continue
                peak_motif_hits[seq_idx, j] = True

    for j in range(n_motifs):
        n_hit = peak_motif_hits[:, j].sum()
        if n_hit > 0:
            hit_rate_mat[ct_idx, j] = n_hit / n_seqs

    hit_binary[ct] = peak_motif_hits
    print(f"{time.time()-t0:.1f}s  "
          f"(motifs in >=1 peak: {(hit_rate_mat[ct_idx] > 0).sum()})", flush=True)

# %% Deduplicate by TF name (max across motif variants)
hit_rate_df = pd.DataFrame(hit_rate_mat, index=ct_list, columns=motif_tf_names)
hit_rate_tf = hit_rate_df.T.groupby(level=0).max().T  # celltypes × unique TFs

# Also deduplicate hit_binary by TF name
unique_tfs = sorted(hit_rate_tf.columns)
tf_to_motif_indices = {}
for tf in unique_tfs:
    tf_to_motif_indices[tf] = [j for j, n in enumerate(motif_tf_names) if n == tf]

# %% Fisher's exact test: focal celltype vs. all other celltypes pooled
print("\nRunning Fisher's exact tests ...", flush=True)
t0 = time.time()

fisher_results = []

for ct_idx, ct in enumerate(ct_list):
    n_fg = hit_binary[ct].shape[0]
    if n_fg == 0:
        continue

    # Background: pool peaks from all other celltypes
    bg_arrays = [hit_binary[other_ct] for other_ct in ct_list if other_ct != ct]
    if not bg_arrays:
        continue
    bg_hits = np.vstack(bg_arrays)  # (n_bg, n_motifs)
    n_bg = bg_hits.shape[0]

    for tf in unique_tfs:
        mot_idx = tf_to_motif_indices[tf]

        # Foreground: any motif variant hits
        fg_any_hit = hit_binary[ct][:, mot_idx].any(axis=1)
        a = fg_any_hit.sum()       # fg peaks with hit
        b = n_fg - a               # fg peaks without hit

        # Background: any motif variant hits
        bg_any_hit = bg_hits[:, mot_idx].any(axis=1)
        c = bg_any_hit.sum()       # bg peaks with hit
        d = n_bg - c               # bg peaks without hit

        # Skip if no hits at all
        if a + c == 0:
            continue

        table = [[a, b], [c, d]]
        odds_ratio, pval = fisher_exact(table, alternative="greater")

        fisher_results.append({
            "celltype": ct,
            "tf": tf,
            "hits_fg": int(a),
            "total_fg": int(n_fg),
            "hit_rate_fg": a / n_fg,
            "hits_bg": int(c),
            "total_bg": int(n_bg),
            "hit_rate_bg": c / n_bg,
            "odds_ratio": odds_ratio,
            "pvalue": pval,
        })

fisher_df = pd.DataFrame(fisher_results)

# FDR correction (Benjamini-Hochberg) per celltype
fisher_df["fdr"] = np.nan
for ct in ct_list:
    mask = fisher_df["celltype"] == ct
    if mask.sum() == 0:
        continue
    _, fdr_vals, _, _ = multipletests(
        fisher_df.loc[mask, "pvalue"].values,
        method="fdr_bh",
    )
    fisher_df.loc[mask, "fdr"] = fdr_vals

# Enrichment z-score (cross-celltype, same as script 08)
fisher_df["enrichment_zscore"] = np.nan
for tf in fisher_df["tf"].unique():
    mask = fisher_df["tf"] == tf
    rates = fisher_df.loc[mask, "hit_rate_fg"].values
    mean_r = rates.mean()
    std_r  = rates.std()
    fisher_df.loc[mask, "enrichment_zscore"] = (rates - mean_r) / (std_r + 0.02)

fisher_df.to_csv(f"{OUTDIR}/V3_celltype_level_motif_enrichment.csv", index=False)
print(f"  Fisher's exact tests done ({time.time()-t0:.1f}s)")
print(f"  Saved: {OUTDIR}/V3_celltype_level_motif_enrichment.csv")

# %% Print top enriched TFs per celltype (significant only)
print("\nTop enriched TFs per celltype (FDR < 0.05):")
for ct in ct_list:
    ct_sig = fisher_df[(fisher_df["celltype"] == ct) & (fisher_df["fdr"] < 0.05)]
    ct_sig = ct_sig.nlargest(10, "enrichment_zscore")
    if len(ct_sig) == 0:
        print(f"  {ct}: no significant TFs at FDR < 0.05")
        continue
    tfs_str = ", ".join(
        f"{r['tf']}(z={r['enrichment_zscore']:.1f},r={r['hit_rate_fg']:.0%},FDR={r['fdr']:.1e})"
        for _, r in ct_sig.iterrows()
    )
    print(f"  {ct}: {tfs_str}")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON with V2
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- V2 vs V3 Comparison ---")
if os.path.exists(SPEC_V2):
    zad_v2 = ad.read_h5ad(SPEC_V2)

    # Parse V2 conditions
    v2_cond_meta = pd.DataFrame(
        [parse_condition(c) for c in zad_v2.var_names],
        columns=["celltype", "timepoint"],
        index=zad_v2.var_names,
    )
    if "n_cells" in zad_v2.var.columns:
        v2_cond_meta["n_cells"] = zad_v2.var["n_cells"].values
        v2_cond_meta["reliable"] = v2_cond_meta["n_cells"] >= MIN_CELLS
    else:
        v2_cond_meta["reliable"] = True

    print(f"\n  {'Celltype':<30} {'V2 z>=2':>10} {'V3 z>=2':>10} {'V2 z>=4':>10} {'V3 z>=4':>10} {'Top50 Jaccard':>14}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    for ct in FOCAL_CELLTYPES:
        if ct not in ct_names:
            continue
        ct_idx_v3 = ct_names.index(ct)
        z_v3 = Z_ct[:, ct_idx_v3]

        # V2: best reliable timepoint for this celltype
        v2_ct_conds = v2_cond_meta[(v2_cond_meta["celltype"] == ct) &
                                    (v2_cond_meta["reliable"])]
        if len(v2_ct_conds) == 0:
            v2_ct_conds = v2_cond_meta[v2_cond_meta["celltype"] == ct]
        best_v2 = v2_ct_conds.sort_values("n_cells", ascending=False).index[0]
        v2_col  = list(zad_v2.var_names).index(best_v2)
        z_v2    = np.asarray(zad_v2.X[:, v2_col]).ravel()

        v2_z2 = (z_v2 >= 2.0).sum()
        v3_z2 = (z_v3 >= 2.0).sum()
        v2_z4 = (z_v2 >= 4.0).sum()
        v3_z4 = (z_v3 >= 4.0).sum()

        # Jaccard of top-50
        top50_v2 = set(np.argsort(z_v2)[::-1][:50])
        top50_v3 = set(np.argsort(z_v3)[::-1][:50])
        jaccard = len(top50_v2 & top50_v3) / len(top50_v2 | top50_v3)

        print(f"  {ct:<30} {v2_z2:>10,} {v3_z2:>10,} {v2_z4:>10,} {v3_z4:>10,} {jaccard:>14.3f}")

    del zad_v2
    gc.collect()
else:
    print("  V2 specificity matrix not found — skipping comparison")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n--- Phase 3: Figures ---")

# ── Figure 1: Histogram of max z-scores ──
print("\nFig 1: Specificity histogram ...", flush=True)
max_z = Z_ct.max(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(max_z, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(2.0, color="orange", ls="--", lw=1.5, label=f"z=2 ({(max_z>=2).sum():,} peaks)")
ax.axvline(4.0, color="red",    ls="--", lw=1.5, label=f"z=4 ({(max_z>=4).sum():,} peaks)")
ax.set_xlabel("Max celltype-level specificity z-score")
ax.set_ylabel("Number of peaks")
ax.set_title("V3: Distribution of celltype specificity (across ~31 celltypes)")
ax.legend()
ax.set_xlim(-2, max_z.max() + 1)
fig.tight_layout()
fig.savefig(f"{FIG_V3}/specificity_overview/V3_celltype_specificity_histogram.pdf", dpi=300)
fig.savefig(f"{FIG_V3}/specificity_overview/V3_celltype_specificity_histogram.png", dpi=300)
plt.close(fig)
print(f"  Saved: V3_celltype_specificity_histogram.{{pdf,png}}")

# ── Figure 2: Heatmap of top-10 peaks per celltype ──
print("Fig 2: Specificity heatmap ...", flush=True)

heatmap_peaks = []
heatmap_labels = []
for ct in FOCAL_CELLTYPES:
    if ct not in ct_names:
        continue
    ct_idx = ct_names.index(ct)
    z_col  = Z_ct[:, ct_idx]
    top10  = np.argsort(z_col)[::-1][:10]
    for i in top10:
        heatmap_peaks.append(i)
        gene = obs.iloc[i]["nearest_gene"]
        gene_str = str(gene) if pd.notna(gene) else ""
        heatmap_labels.append(f"{obs.index[i]}|{gene_str}")

heatmap_mat = Z_ct[heatmap_peaks]  # (n_peaks, n_ct)

fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_peaks) * 0.25)))
sns.heatmap(
    pd.DataFrame(heatmap_mat, index=heatmap_labels, columns=ct_names),
    cmap="coolwarm", center=0, vmin=-3, vmax=8,
    xticklabels=True, yticklabels=True,
    cbar_kws={"label": "V3 celltype specificity z-score"},
    ax=ax,
)
# Add tier separators
for i in range(1, len(FOCAL_CELLTYPES)):
    if i * 10 < len(heatmap_peaks):
        ax.axhline(i * 10, color="black", lw=1.5)
ax.set_title("V3: Top-10 peaks per celltype — specificity z-score across all celltypes")
ax.set_xlabel("Celltype")
ax.set_ylabel("Peak (peak_id | nearest_gene)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIG_V3}/specificity_overview/V3_celltype_specificity_heatmap.pdf", dpi=300)
fig.savefig(f"{FIG_V3}/specificity_overview/V3_celltype_specificity_heatmap.png", dpi=300)
plt.close(fig)
print(f"  Saved: V3_celltype_specificity_heatmap.{{pdf,png}}")

# ── Figure 3: Temporal profiles for focal celltypes ──
print("Fig 3: Temporal profiles ...", flush=True)

n_focal = len([ct for ct in FOCAL_CELLTYPES if ct in celltype_peak_info])
fig, axes = plt.subplots(n_focal, 1, figsize=(8, 3.5 * n_focal), squeeze=False)

for ax_idx, ct in enumerate([ct for ct in FOCAL_CELLTYPES if ct in celltype_peak_info]):
    ax = axes[ax_idx, 0]
    info = celltype_peak_info[ct]
    acc_cols = sorted([c for c in info.columns if c.startswith("acc_")],
                      key=lambda x: TP_INT.get(x.replace("acc_", ""), 99))
    tp_labels = [c.replace("acc_", "").replace("somites", "s") for c in acc_cols]
    tp_ints   = [TP_INT.get(c.replace("acc_", ""), 0) for c in acc_cols]

    # Plot top 10 peaks
    n_show = min(10, len(info))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_show))

    for i in range(n_show):
        row = info.iloc[i]
        vals = [row[c] for c in acc_cols]
        gene = str(row["nearest_gene"]) if pd.notna(row["nearest_gene"]) else ""
        label = f"{gene} (z={row['V3_zscore']:.1f})" if gene else f"z={row['V3_zscore']:.1f}"
        ax.plot(tp_ints, vals, marker="o", ms=4, color=cmap[i],
                label=label, lw=1.5, alpha=0.8)

    ax.set_xlabel("Developmental stage (somites)")
    ax.set_ylabel("Log-norm accessibility")
    ax.set_title(f"{ct} — temporal profiles of top-{n_show} V3-specific peaks")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xticks(tp_ints)
    ax.set_xticklabels(tp_labels)

fig.tight_layout()
fig.savefig(f"{FIG_V3}/temporal/V3_celltype_temporal_profiles.pdf", dpi=300)
fig.savefig(f"{FIG_V3}/temporal/V3_celltype_temporal_profiles.png", dpi=300)
plt.close(fig)
print(f"  Saved: V3_celltype_temporal_profiles.{{pdf,png}}")

# ── Figure 4: Motif enrichment heatmap (coolwarm, centered at 0) ──
print("Fig 4: Motif enrichment heatmap ...", flush=True)

# Build enrichment z-score matrix for significant TFs
sig_tfs = fisher_df[fisher_df["fdr"] < 0.05]["tf"].unique()
if len(sig_tfs) > 0:
    # Build matrix: celltypes × significant TFs
    enrich_mat = pd.DataFrame(index=ct_list, columns=sig_tfs, dtype=float)
    fdr_mat    = pd.DataFrame(index=ct_list, columns=sig_tfs, dtype=float)
    for _, row in fisher_df[fisher_df["tf"].isin(sig_tfs)].iterrows():
        enrich_mat.loc[row["celltype"], row["tf"]] = row["enrichment_zscore"]
        fdr_mat.loc[row["celltype"], row["tf"]]    = row["fdr"]

    enrich_mat = enrich_mat.fillna(0).astype(float)
    fdr_mat    = fdr_mat.fillna(1.0).astype(float)

    # Filter to TFs with max enrichment z-score > 0.5
    max_enrich = enrich_mat.max(axis=0)
    keep_tfs   = max_enrich[max_enrich > 0.5].index
    enrich_plot = enrich_mat[keep_tfs]

    if len(keep_tfs) > 0:
        # Cluster columns
        if enrich_plot.shape[1] > 2:
            col_linkage = hierarchy.linkage(
                pdist(enrich_plot.T.values, metric="euclidean"),
                method="ward"
            )
            col_order = hierarchy.leaves_list(col_linkage)
            enrich_plot = enrich_plot.iloc[:, col_order]

        fig, ax = plt.subplots(figsize=(max(12, len(keep_tfs) * 0.25), 6))
        sns.heatmap(
            enrich_plot,
            cmap="coolwarm", center=0, vmin=-2, vmax=3,
            xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Enrichment z-score"},
            ax=ax,
        )
        # Add significance markers
        for i, ct in enumerate(enrich_plot.index):
            for j, tf in enumerate(enrich_plot.columns):
                fdr_val = fdr_mat.loc[ct, tf] if tf in fdr_mat.columns else 1.0
                if fdr_val < 0.01:
                    ax.text(j + 0.5, i + 0.5, "**", ha="center", va="center",
                            fontsize=6, color="black", fontweight="bold")
                elif fdr_val < 0.05:
                    ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")

        ax.set_title("V3: Motif enrichment across celltypes (Fisher's exact, * FDR<0.05, ** FDR<0.01)")
        ax.set_xlabel("Transcription factor")
        ax.set_ylabel("Cell type")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{FIG_V3}/motif_enrichment/V3_celltype_motif_enrichment_heatmap.pdf", dpi=300)
        fig.savefig(f"{FIG_V3}/motif_enrichment/V3_celltype_motif_enrichment_heatmap.png", dpi=300)
        plt.close(fig)
        print(f"  Saved: V3_celltype_motif_enrichment_heatmap.{{pdf,png}} ({len(keep_tfs)} TFs)")
    else:
        print("  No TFs with enrichment z > 0.5 — skipping heatmap")
else:
    print("  No significant TFs at FDR < 0.05 — skipping heatmap")

# ── Figure 5: Per-celltype motif barplots with significance ──
print("Fig 5: Motif enrichment barplots ...", flush=True)

n_top_bar = 15
fig, axes = plt.subplots(len(ct_list), 1,
                          figsize=(10, 3.5 * len(ct_list)),
                          squeeze=False)

for ax_idx, ct in enumerate(ct_list):
    ax = axes[ax_idx, 0]
    ct_data = fisher_df[fisher_df["celltype"] == ct].copy()
    ct_data = ct_data.nlargest(n_top_bar, "enrichment_zscore")

    if len(ct_data) == 0:
        ax.set_title(f"{ct} — no motifs")
        continue

    colors = []
    labels = []
    for _, row in ct_data.iterrows():
        tf_label = row["tf"]
        if row["fdr"] < 0.01:
            tf_label += " **"
            colors.append("#d62728")
        elif row["fdr"] < 0.05:
            tf_label += " *"
            colors.append("#ff7f0e")
        else:
            colors.append("#aec7e8")
        labels.append(tf_label)

    y_pos = np.arange(len(ct_data))
    ax.barh(y_pos, ct_data["enrichment_zscore"].values, color=colors, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Enrichment z-score")
    ax.set_title(f"{ct} — top {n_top_bar} enriched TF motifs "
                 f"(** FDR<0.01, * FDR<0.05)")

    # Add hit rate annotation
    for i, (_, row) in enumerate(ct_data.iterrows()):
        ax.text(row["enrichment_zscore"] + 0.02, i,
                f"  {row['hit_rate_fg']:.0%}",
                va="center", fontsize=7, color="gray")

fig.tight_layout()
fig.savefig(f"{FIG_V3}/motif_enrichment/V3_celltype_motif_enrichment_barplots.pdf", dpi=300)
fig.savefig(f"{FIG_V3}/motif_enrichment/V3_celltype_motif_enrichment_barplots.png", dpi=300)
plt.close(fig)
print(f"  Saved: V3_celltype_motif_enrichment_barplots.{{pdf,png}}")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nDone.")
print(f"Exit code: 0")
print(f"End: {time.strftime('%c')}")
