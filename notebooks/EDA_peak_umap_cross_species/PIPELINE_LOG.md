# Cross-Species Peak UMAP Alignment — Pipeline Log

Branch: `peak-umap-cross-species`
SCRATCH: `/hpc/scratch/group.data.science/yang-joon.kim/multiome-cross-species-peak-umap/`

---

## Script Status

| Script | Job ID | Status | Wall time | Memory | Notes |
|--------|--------|--------|-----------|--------|-------|
| 07 annotate_genes_mouse_human | — | ✅ Done | — | — | 192K mouse + 1M human peaks annotated |
| 08 download_orthologs | — | ✅ Done | <5 min | — | 9,350 1:1:1 triplets via FTP fallback |
| 09 backfill_annotations | 28842730 | ✅ Done | 19 min | ~55 GB | 17 GB annotated h5ad saved |
| 10 select_root_anchors | (interactive) | ✅ Done | — | — | **344 root anchor triplets** |
| 11 select_branch_anchors | 28859778 | ✅ Done | 2 min | 33.8 GB / 48 GB | **2,000 branch anchors** (ZF+MM only; see issues) |
| 12 procrustes_alignment | — | ⏳ Pending | — | — | Waiting on script 11 fixes |
| 13 visualize_aligned_umap | — | ⏳ Pending | — | — | — |

---

## Script 11 Results

### Branch anchor counts (job 28859778)

| Lineage | Count |
|---------|-------|
| paraxial_mesoderm | 1283 |
| lateral_mesoderm | 255 |
| ectoderm | 214 |
| endoderm | 145 |
| neural_crest | 103 |
| neural_cns | **0** |
| **Total** | **2,000** |

### Issues Found

#### Issue 1: Human gets 0 branch anchors

**Root cause:** `get_celltype_col()` returns `celltype` for human, but the human h5ad
(Domcke 2020) stores organ-specific cell type names that don't match broad keyword patterns.

Human `celltype` values (19 types): "Syncytiotrophoblasts and villous cytotrophoblasts",
"Ganglion cells", "IGFBP1_DKK1 positive cells", "Neuroendocrine cells", "Schwann cells", etc.

**Fix:** Use `peak_lineage` column for human instead — it has broad, annotation-friendly categories:

| peak_lineage value | Count | Maps to lineage |
|---|---|---|
| Endoderm | 142,044 | endoderm |
| Hematopoietic | 133,613 | lateral_mesoderm |
| PNS/Neural Crest | 132,642 | neural_crest |
| CNS | 105,230 | neural_cns |
| Sensory/Eye | 91,294 | neural_cns |
| Mesenchyme | 70,655 | paraxial_mesoderm (approx.) |
| Placental | 62,862 | (skip) |
| Endocrine | 61,375 | endoderm |
| Cardiac | 56,530 | lateral_mesoderm |
| Renal | 27,341 | lateral_mesoderm |
| Endothelial | 15,063 | lateral_mesoderm |
| Other | 142,806 | (skip) |

#### Issue 2: neural_cns gets 0 triplets (mouse)

**Root cause:** Keyword collision. The LINEAGE_MAP had overlapping keywords:
- `neural_cns` for mouse: `["neural", "ectoderm", "brain", "neuroectoderm", "spinal"]`
- `ectoderm` for mouse: `["ectoderm", "epidermis", "skin", "surface"]`

`assign_lineage()` uses **last-match** logic — the later lineage in the dict overrides earlier ones.
Mouse cells with `celltype = "Caudal_neurectoderm"` matched `neural_cns` via "ectoderm" keyword,
but were then overridden by `ectoderm` lineage (processed later) also matching "ectoderm".

Mouse `celltype` values relevant to lineage assignment:

| celltype | Count | Correct lineage |
|---|---|---|
| Caudal_neurectoderm | 7,888 | neural_cns |
| Neural_crest | 12,765 | neural_crest |
| ExE_endoderm | 11,739 | endoderm |
| Parietal_endoderm | 7,355 | endoderm |
| Paraxial_mesoderm | 7,009 | paraxial_mesoderm |
| Blood_progenitors_1 | 6,211 | lateral_mesoderm |
| Blood_progenitors_2 | 6,499 | lateral_mesoderm |
| Endothelium | 7,484 | lateral_mesoderm |
| Allantois | 7,815 | lateral_mesoderm |
| ExE_ectoderm | 2,537 | ectoderm |

---

## Bugs Fixed

| Bug | File | Fix |
|-----|------|-----|
| `sp` loop variable shadows `scipy.sparse as sp` | `11_select_branch_anchors.py` line 103 | Renamed to `species_key` |
| Categorical column `.fillna("")` TypeError | `11_select_branch_anchors.py` line 163, 214 | Use `.astype(str).replace({"nan":"","None":""})` |
| `'tuple' object has no attribute 'astype'` (pandas index split) | `07_annotate_genes_mouse_human.py` | Replaced `str.split(expand=True)` with list comprehension |
| OOM kill on human peaks (1M × gene body join) | `07_annotate_genes_mouse_human.py` | Removed `.join()`, use `.overlap()` only; added `gc.collect()` |
| SLURM scripts fail immediately | all `slurm/*.sh` | Remove `set -euo pipefail`; absolute log paths; `module load ... \|\| true` |
| All BioMart mirrors down (MySQL connection failure) | `08_download_orthologs.py` | FTP fallback streaming Ensembl Compara TSVs |
| Categorical dtype blocks assignment in backfill | `09_backfill_annotations.py` | Convert all Categorical cols to str before assignment |
| h5py can't serialize mixed object columns | `09_backfill_annotations.py` | Pre-save cleanup: all object cols → clean str, "nan" → "" |

---

## SLURM Resource History

| Job | Script | CPUs | Mem requested | Mem used | Wall time | CPU eff. |
|-----|--------|------|---------------|----------|-----------|----------|
| 28842730 | 09 backfill | 4 | 64 GB | ~55 GB | 19 min | — |
| 28859778 | 11 branch anchors | 4 | 48 GB | 33.8 GB (70%) | 2 min | 24% |

---

## Pending Fixes for Script 11 (before re-running script 12)

1. **`get_celltype_col()`**: return `peak_lineage` for human (not `celltype`)
2. **LINEAGE_MAP keywords**: update mouse + human to use actual column values (see tables above)
3. **`assign_lineage()` logic**: switch to first-match (don't override once labeled) to avoid
   keyword collision between lineages

Updated keyword targets:

| Lineage | Mouse (`celltype`) | Human (`peak_lineage`) |
|---------|-------------------|----------------------|
| neural_cns | `neurectoderm` | `cns`, `sensory` |
| paraxial_mesoderm | `paraxial` | `mesenchyme` |
| lateral_mesoderm | `endothelium`, `blood_progenitor`, `allantois` | `cardiac`, `hematopoietic`, `renal`, `endothelial` |
| endoderm | `endoderm` | `endoderm`, `endocrine` |
| ectoderm | `exe_ectoderm` | *(empty — not in Domcke 2020)* |
| neural_crest | `neural_crest` | `pns/neural crest` |
