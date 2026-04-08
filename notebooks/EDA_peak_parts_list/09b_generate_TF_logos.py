# %% [markdown]
# # Script 09b: Generate TF sequence logo plots for V3 significant TFs
#
# Reads the V3 motif enrichment results (Fisher's exact with FDR),
# loads the corresponding JASPAR H12CORE PWMs via pymemesuite,
# and generates information-content sequence logos using logomaker.
#
# Output: figures/peak_parts_list/TF_logos/{TF_NAME}.seq.logo.{pdf,png}
#
# Env: /home/yang-joon.kim/.conda/envs/gReLu

# %% Imports
import os
import numpy as np
import pandas as pd
from pymemesuite.common import MotifFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker


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

print("=== Script 09b: TF Logo Generation ===")

# %% Paths
REPO    = ("/hpc/projects/data.science/yangjoon.kim/zebrahub_multiome/"
           "zebrahub-multiome-analysis")
OUTDIR  = f"{REPO}/figures/peak_parts_list/V3/motif_enrichment/TF_logos"
os.makedirs(OUTDIR, exist_ok=True)

MEME_PATH = ("/hpc/projects/data.science/yangjoon.kim/github_repos/"
             "gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme")
ENRICH_CSV = (f"{REPO}/notebooks/EDA_peak_parts_list/outputs/V3/"
              "V3_celltype_level_motif_enrichment.csv")

# %% Information content function (Schneider-Stephens / KL-divergence)
# Matches the implementation in notebooks/Fig_peak_umap/03_motif_enrichment_analysis_v3_coarse_fine.py
def pwm_to_information_matrix(pwm):
    """
    Convert a probability PWM (rows = positions, cols = A,C,G,T) to the
    Schneider-Stephens / KL-divergence information matrix for classic
    sequence logos.

    Uses zebrafish genome-wide background: ~60% AT, ~40% GC.
    """
    background = np.array([0.30, 0.20, 0.20, 0.30])  # A, C, G, T
    eps = 1e-6
    pwm = np.clip(np.asarray(pwm, dtype=float), eps, 1.0)
    bg  = np.clip(np.asarray(background, dtype=float), eps, 1.0)

    # Per-cell KL term: p * (log2(p) - log2(q))
    kl_cell = pwm * (np.log2(pwm) - np.log2(bg))

    # Total information per position (column height)
    ic_col = kl_cell.sum(axis=1)  # shape (L,)
    total_ic = ic_col.sum()

    # Final letter heights: P_ib x IC_i
    ic_matrix = pwm * ic_col[:, None]  # shape (L, 4)
    return ic_matrix, total_ic

# %% Load motif database
print(f"Loading JASPAR H12CORE motifs ...", flush=True)
motif_file = MotifFile(MEME_PATH)
motif_list = list(motif_file)

# Build lookup: TF name -> list of (motif_obj, accession)
# Accession format: "GATA4.H12CORE.0.P.B" -> TF = "GATA4"
motif_by_tf = {}
for m in motif_list:
    acc = m.accession.decode() if isinstance(m.accession, bytes) else str(m.accession)
    tf = acc.split(".")[0]
    if tf not in motif_by_tf:
        motif_by_tf[tf] = []
    motif_by_tf[tf].append((m, acc))

print(f"  {len(motif_list)} motifs -> {len(motif_by_tf)} unique TFs")

# %% Load enrichment results and identify significant TFs per celltype
enrich_df = pd.read_csv(ENRICH_CSV)
sig_df = enrich_df[enrich_df["fdr"] < 0.05].copy()
print(f"\nSignificant TF-celltype pairs (FDR < 0.05): {len(sig_df)}")

# Get unique TFs to plot
sig_tfs = sorted(sig_df["tf"].unique())
print(f"Unique significant TFs: {len(sig_tfs)}")

# %% Generate logos
background = np.array([0.30, 0.20, 0.20, 0.30])
ymax = -np.log2(background.min())  # max possible bits

n_plotted = 0
n_skipped = 0

for tf in sig_tfs:
    if tf not in motif_by_tf:
        print(f"  {tf}: not found in JASPAR H12CORE — skipping")
        n_skipped += 1
        continue

    # If multiple motif variants, pick the one with highest total IC
    best_motif = None
    best_acc = None
    best_ic = -1

    for m, acc in motif_by_tf[tf]:
        pwm = np.array(m.frequencies)
        _, total_ic = pwm_to_information_matrix(pwm)
        if total_ic > best_ic:
            best_ic = total_ic
            best_motif = m
            best_acc = acc

    # Generate logo
    pwm = np.array(best_motif.frequencies)
    ic_matrix, total_ic = pwm_to_information_matrix(pwm)
    df = pd.DataFrame(ic_matrix, columns=["A", "C", "G", "T"])

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * df.shape[0]), 3.2))
    logo = logomaker.Logo(df, ax=ax,
                          color_scheme="classic",
                          font_name="DejaVu Sans Mono")

    # Get celltype info for title
    tf_rows = sig_df[sig_df["tf"] == tf].nlargest(1, "enrichment_zscore")
    best_ct = tf_rows.iloc[0]["celltype"]
    best_fdr = tf_rows.iloc[0]["fdr"]
    best_z = tf_rows.iloc[0]["enrichment_zscore"]

    ax.set_title(f"{tf}  ({best_acc})\n"
                 f"Top celltype: {best_ct}  |  z={best_z:.1f}  |  FDR={best_fdr:.1e}",
                 fontsize=10)
    ax.set_ylabel("bits", fontsize=9)
    ax.set_xlabel("position", fontsize=9)
    ax.set_xticks(np.arange(len(df)))
    ax.set_ylim(0, ymax)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{tf}.seq.logo.pdf")
    fig.savefig(f"{OUTDIR}/{tf}.seq.logo.png", dpi=300)
    plt.close(fig)
    n_plotted += 1

print(f"\nGenerated {n_plotted} logos, skipped {n_skipped}")
print(f"Output: {OUTDIR}/")

# %% Summary table: which TFs were plotted per celltype
print("\nSignificant TFs per celltype:")
for ct in sig_df["celltype"].unique():
    ct_tfs = sig_df[sig_df["celltype"] == ct].nlargest(15, "enrichment_zscore")
    tfs_str = ", ".join(
        f"{r['tf']}(z={r['enrichment_zscore']:.1f})"
        for _, r in ct_tfs.iterrows()
    )
    print(f"  {ct}: {tfs_str}")

print("\nDone.")
