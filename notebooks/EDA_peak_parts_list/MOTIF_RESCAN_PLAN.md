# Plan: Parallel Motif Re-Scanning with JASPAR 2024 & CIS-BP v2 Zebrafish

## Context

The current top-200 peak FIMO pipeline uses **HOCOMOCO v12 CORE (H12CORE)** — a choice made without explicit user approval (revealed during a session's methods review; the git commit message `132965b` even misnames it as "JASPAR H12CORE", conflating the two databases). The `gReLU/resources/meme/` directory contains three motif databases:

| Database | File | Notes |
|---|---|---|
| HOCOMOCO v12 CORE | `H12CORE_meme_format.meme` | **Current default** — 1,443 PWMs |
| HOCOMOCO v13 CORE | `H13CORE_meme_format.meme` | Newer, not used |
| JASPAR 2024 consensus | `jaspar_2024_consensus.meme` | Not used |

Plus locally downloaded: **CIS-BP v2 Danio rerio** at `/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm` (PFM format, ~1,400 motifs, zebrafish-specific).

**Goal**: Keep H12CORE as the default (results already generated) while producing parallel re-scans against JASPAR 2024 (broad vertebrate) and CIS-BP v2 Danio rerio (zebrafish-specific) for robustness/comparison. Outputs go to scratch with database-specific filename suffixes.

## Approach

**Parameterize** the existing 09j scripts with a `--motif-db` selector (rather than duplicate them). A single registry maps database name → MEME file path + output suffix:

```python
MOTIF_DBS = {
    "h12core":         {"path": ".../H12CORE_meme_format.meme",          "suffix": ""},
    "jaspar2024":      {"path": ".../jaspar_2024_consensus.meme",        "suffix": "_jaspar2024"},
    "cisbpv2_danrer":  {"path": "<converted-MEME>/cisbpv2_danrer.meme",  "suffix": "_cisbpv2_danrer"},
}
```

Scripts read `--motif-db <name>` (default `h12core`), look up the path + suffix, and apply the suffix to every output filename. Existing H12CORE outputs are untouched (empty suffix = current file layout preserved).

## Files to Create / Modify

### 1. NEW — `notebooks/EDA_peak_parts_list/09j_convert_cisbp_pfm_to_meme.py`

One-time conversion of CIS-BP v2 PFM → MEME v4 format.

- **Input**: `/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm`
- **Output**: `/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/motif_dbs/cisbpv2_danrer.meme`
- **Logic**: For each `>M_XXX` header, read the 4-column frequency matrix, normalize each row to probabilities, emit a `MOTIF` block in MEME format
- **Motif name parsing**: The current 09j TF name parsing uses `motif.name.split(".")[0]` — CIS-BP IDs are `M00008_2.00` which parse to `M00008_2` (a motif ID, not a TF name). We need to **join on the metadata file** `CisBP_ver2_Danio_rerio.motif2factors.txt` to map motif IDs → TF gene symbols. The conversion script will include the TF name in the MEME `MOTIF` line: `MOTIF M00008_2.00 tbx3b` so the downstream parser can extract the TF name from the second token.

### 2. MODIFY — `notebooks/EDA_peak_parts_list/09j_precompute_motif_hits_top200.py`

- Add `argparse` for `--motif-db` (default `h12core`)
- Add `MOTIF_DBS` registry at top
- Replace hardcoded `MEME_PATH` with `MOTIF_DBS[args.motif_db]["path"]`
- Append `suffix = MOTIF_DBS[args.motif_db]["suffix"]` to all output filenames
- Update motif name parsing to be robust to both formats (H12CORE: `TFNAME.H12CORE.0.P.B`; CIS-BP: `MXXXXX_2.00 tfname` via 2nd token)

### 3. MODIFY — `notebooks/EDA_peak_parts_list/09j_fimo_batch.py`

Same changes as #2, applied to the per-celltype batch version. Output files go to `batches{suffix}/` (subdirectory with suffix so H12CORE and new DB results don't collide).

### 4. MODIFY — `notebooks/EDA_peak_parts_list/09j_merge_and_test.py`

Same changes — read batch files from `batches{suffix}/`, write merged outputs with suffix.

### 5. NEW SLURM runners

| New file | Runs |
|---|---|
| `slurm/run_09j_fimo_array_jaspar2024.sh` | `09j_fimo_batch.py --motif-db jaspar2024` (array 0–30) |
| `slurm/run_09j_merge_jaspar2024.sh` | `09j_merge_and_test.py --motif-db jaspar2024` |
| `slurm/run_09j_fimo_array_cisbpv2.sh` | `09j_fimo_batch.py --motif-db cisbpv2_danrer` (array 0–30) |
| `slurm/run_09j_merge_cisbpv2.sh` | `09j_merge_and_test.py --motif-db cisbpv2_danrer` |

SLURM resources match existing: cpu partition, per-array-task ~4 CPU / 16G / 2h.

### 6. NEW — `notebooks/EDA_peak_parts_list/MOTIF_DATABASES.md`

Documentation listing:
- **Default choice**: H12CORE (HOCOMOCO v12 CORE), 1,443 PWMs
- **Parallel scans available**: JASPAR 2024 consensus, CIS-BP v2 Danio rerio
- Paths to MEME files for all three databases
- Paths to output directories per database
- How to run the alternative scans (`--motif-db` flag + SLURM runners)
- Note about provenance of each database

## Output File Layout (on scratch)

```
/hpc/scratch/group.data.science/yang-joon.kim/peak-parts-list-motifs/
├── motif_dbs/
│   └── cisbpv2_danrer.meme                               # converted CIS-BP file
├── batches/                                              # H12CORE (default, existing)
├── batches_jaspar2024/
├── batches_cisbpv2_danrer/
├── V3_top200_motif_hit_matrix.csv                        # H12CORE (default)
├── V3_top200_motif_hit_matrix_jaspar2024.csv             # NEW
├── V3_top200_motif_hit_matrix_cisbpv2_danrer.csv         # NEW
└── (analogous for _motif_positions, _peak_motif_summary, _all_celltypes_top200_peaks_with_motifs)
```

## Execution Order

1. Run `09j_convert_cisbp_pfm_to_meme.py` (one-shot, ~1 min)
2. Submit JASPAR 2024 FIMO array: `sbatch slurm/run_09j_fimo_array_jaspar2024.sh`
3. After (2) completes: `sbatch slurm/run_09j_merge_jaspar2024.sh`
4. In parallel: same for CIS-BP — `sbatch slurm/run_09j_fimo_array_cisbpv2.sh` then merge

## Critical Files (references, not modified)

| Path | Role |
|---|---|
| `/hpc/projects/data.science/yangjoon.kim/github_repos/gReLU/src/grelu/resources/meme/H12CORE_meme_format.meme` | Default motif file |
| `/hpc/projects/data.science/yangjoon.kim/github_repos/gReLU/src/grelu/resources/meme/jaspar_2024_consensus.meme` | JASPAR 2024 |
| `/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.pfm` | CIS-BP source (needs conversion) |
| `/hpc/mydata/yang-joon.kim/genomes/danRer11/CisBP_ver2_Danio_rerio.motif2factors.txt` | TF name mapping for CIS-BP |

## Verification

1. **PFM→MEME conversion**: the converted `cisbpv2_danrer.meme` opens in `pymemesuite.MotifFile` without error; motif count matches PFM count (~1,400)
2. **Default H12CORE run unaffected**: running 09j with `--motif-db h12core` (or no arg) produces identical files at identical paths
3. **JASPAR 2024 run**: TF count in hit matrix matches JASPAR 2024 TF count; spot-check known TFs (gata4, sox10, myod1) hit in expected celltypes
4. **CIS-BP run**: TF count matches CIS-BP zebrafish count; expect higher hit rates for zebrafish-specific TFs
5. **Cross-database concordance**: Jaccard overlap of enriched TFs for heart_myocardium top-50 peaks should be >50% across the three databases for well-conserved TF families
