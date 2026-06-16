# Motif Databases Used in the Peak Parts List Pipeline

This document tracks which transcription-factor motif database was used to scan the top-200 peaks per celltype (6,200 peaks total) via FIMO, and how to run parallel scans against alternative databases.

## Default database: HOCOMOCO v12 CORE (H12CORE)

The primary FIMO scan (scripts `09j_*`) uses **HOCOMOCO v12 CORE**.

| Attribute | Value |
|---|---|
| Database | HOCOMOCO v12 CORE |
| MEME file | `/hpc/projects/data.science/yangjoon.kim/github_repos/gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme` |
| Number of PWMs | 1,443 |
| Source | Pre-loaded in gReLU resources (bundled with the library) |
| Motif naming | `TFNAME.H12CORE.0.P.B` (e.g., `AHR.H12CORE.0.P.B`) |

**Why this one**: H12CORE was the first motif file encountered in the gReLU resources directory and was used for the initial analysis. The git commit message `132965b` ("add: cross-celltype JASPAR H12CORE motif enrichment analysis") incorrectly conflates JASPAR with H12CORE — the actual database is **HOCOMOCO v12 CORE**, not JASPAR.

All existing scratch outputs from this scan sit directly under the scratch root with **no filename suffix**:
- `V3_top200_motif_hit_matrix.csv`
- `V3_top200_motif_positions.csv`
- `V3_top200_peak_motif_summary.csv`
- `V3_top200_motif_enrichment_all31.csv`
- `V3_all_celltypes_top200_peaks_with_motifs.csv`

Scratch root: `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/`

## Parallel scans: JASPAR 2024 + CIS-BP v2 Danio rerio

Two additional databases are scanned in parallel for robustness and as orthogonal views:

| Name | File | Motifs | Release / notes |
|---|---|---|---|
| **HOCOMOCO v12 CORE** (default) | `gReLU/resources/meme/H12CORE_meme_format.meme` | 1,443 | Broad vertebrate collection |
| **JASPAR 2024** | `gReLU/resources/meme/jaspar_2024_consensus.meme` | 182 | Clustered consensus profiles (not the full CORE set) |
| **CIS-BP v2 Danio rerio** | (converted) `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/motif_dbs/cisbpv2_danrer.meme` | ~5,300 | Zebrafish-specific; sourced from `/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm` |

Alternative runs produce files with database-specific suffixes:

| Suffix | Database |
|---|---|
| `` (empty) | HOCOMOCO v12 CORE (default) |
| `_jaspar2024` | JASPAR 2024 consensus |
| `_cisbpv2_danrer` | CIS-BP v2 Danio rerio |

Example: `V3_top200_motif_hit_matrix_jaspar2024.csv`

Per-celltype FIMO batch outputs live in suffix-specific subdirectories:
- `batches/` — H12CORE
- `batches_jaspar2024/` — JASPAR 2024
- `batches_cisbpv2_danrer/` — CIS-BP v2

## How to run an alternative scan

### Step 1 (CIS-BP only) — one-time PFM → MEME conversion

```bash
/hpc/user_apps/data.science/conda_envs/single-cell-base/bin/python \
  notebooks/EDA_peak_parts_list/09j_convert_cisbp_pfm_to_meme.py
```

Writes `cisbpv2_danrer.meme` to `/hpc/scratch/.../peak-parts-list-motifs/motif_dbs/`.
Takes a few seconds. JASPAR 2024 needs no conversion (already in MEME format).

### Step 2 — per-celltype FIMO array

```bash
# JASPAR 2024
sbatch notebooks/EDA_peak_parts_list/slurm/run_09j_fimo_array_jaspar2024.sh

# CIS-BP v2 Danio rerio
sbatch notebooks/EDA_peak_parts_list/slurm/run_09j_fimo_array_cisbpv2.sh
```

Each is a 31-task SLURM array, one task per celltype. CIS-BP walltime is bumped to 3h because it has ~3.7× more motifs than H12CORE.

### Step 3 — merge + Fisher's exact test

```bash
sbatch notebooks/EDA_peak_parts_list/slurm/run_09j_merge_jaspar2024.sh
sbatch notebooks/EDA_peak_parts_list/slurm/run_09j_merge_cisbpv2.sh
```

## Command-line interface

All three 09j scripts share the same CLI surface:

```bash
# All scripts accept --motif-db {h12core,jaspar2024,cisbpv2_danrer}  (default: h12core)
python 09j_precompute_motif_hits_top200.py --motif-db jaspar2024
python 09j_fimo_batch.py --celltype-idx 0 --motif-db cisbpv2_danrer
python 09j_merge_and_test.py --motif-db jaspar2024
```

The motif database registry lives near the top of each script in `MOTIF_DBS = {...}`. Each entry specifies the MEME file path, output filename suffix, and a TF-name parser (the parsing differs per database because the naming conventions differ).

## TF-name parsing (database-dependent)

| Database | Motif ID example | Parser logic | TF extracted |
|---|---|---|---|
| H12CORE | `AHR.H12CORE.0.P.B` | `acc.split(".")[0]` | `AHR` |
| JASPAR 2024 | `C001:HES_SREBF:bHLH` | `acc.split(":")[1]` | `HES_SREBF` (cluster label) |
| CIS-BP v2 | `MOTIF M00008_2.00 hmga1a` (accession + name) | `motif.name` | `hmga1a` |

For CIS-BP, the conversion script in step 1 appends each motif's primary TF symbol (looked up from `CisBP_ver2_Danio_rerio.motif2factors.txt`) to the `MOTIF` line so the downstream parser can read it directly.
