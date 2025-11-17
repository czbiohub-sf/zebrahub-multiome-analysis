# Celltype-Specific Transcription Factors

**Generated:** 2025-11-08 11:49:41

## Overview

This report identifies transcription factors that show specificity to particular celltypes or developmental lineages.

**Definition of 'specific':** Present in 3-5 celltypes with high lineage coherence (≥60% of celltypes from the same developmental lineage).

### Summary Statistics

- **Total TFs analyzed:** 252
- **Celltype-specific TFs:** 11
- **Ultra-specific (3 celltypes):** 7
- **Moderately specific (4-5 celltypes):** 4

### Distribution by Lineage

| Lineage | N Specific TFs |
|---------|----------------|
| mesoderm | 4 |
| neuroectoderm | 4 |
| other | 3 |

## Top 50 Most Specific TFs

Ranked by lineage coherence (descending), then number of celltypes (ascending).

| Rank | TF | N CTs | Celltypes | Lineage | Coherence | N Clusters |
|------|-------|-------|-----------|---------|-----------|------------|
| 236 | alx1 | 3 | floor_plate,optic_cup,primordial_germ_cells | other | 1.00 | 2 |
| 199 | neurod6a | 5 | PSM,fast_muscle,muscle,somites,tail_bud | mesoderm | 0.80 | 2 |
| 45 | zic4 | 4 | NMPs,PSM,fast_muscle,muscle | mesoderm | 0.75 | 4 |
| 110 | foxi3b | 4 | differentiating_neurons,enteric_neurons,fast_muscle,neural | neuroectoderm | 0.75 | 6 |
| 196 | foxd5 | 3 | enteric_neurons,lateral_plate_mesoderm,muscle | mesoderm | 0.67 | 3 |
| 228 | dmrt2b | 3 | PSM,neural_telencephalon,notochord | mesoderm | 0.67 | 1 |
| 235 | uncx | 3 | heart_myocardium,neural_posterior,spinal_cord | neuroectoderm | 0.67 | 1 |
| 237 | lhx8a | 3 | lateral_plate_mesoderm,primordial_germ_cells,tail_bud | other | 0.67 | 1 |
| 239 | lhx4 | 3 | enteric_neurons,pharyngeal_arches,pronephros | other | 0.67 | 1 |
| 247 | nr2e1 | 3 | neural,neural_posterior,pharyngeal_arches | neuroectoderm | 0.67 | 1 |
| 11 | hoxc3a | 5 | differentiating_neurons,floor_plate,lateral_plate_mesoderm,n... | neuroectoderm | 0.60 | 5 |

## Detailed Breakdown by Lineage

### Neuroectoderm

**Total specific TFs:** 4

#### Ultra-Specific (3 celltypes) - 2 TFs

| TF | Celltypes | Coherence | N Clusters |
|----|-----------|-----------|------------|
| uncx | heart_myocardium,neural_posterior,spinal_cord | 0.67 | 1 |
| nr2e1 | neural,neural_posterior,pharyngeal_arches | 0.67 | 1 |

#### Moderately Specific (4-5 celltypes) - 2 TFs

| TF | N CTs | Celltypes | Coherence | N Clusters |
|-------|-------|-----------|-----------|------------|
| foxi3b | 4 | differentiating_neurons,enteric_neurons,fast_muscl... | 0.75 | 6 |
| hoxc3a | 5 | differentiating_neurons,floor_plate,lateral_plate_... | 0.60 | 5 |

### Mesoderm

**Total specific TFs:** 4

#### Ultra-Specific (3 celltypes) - 2 TFs

| TF | Celltypes | Coherence | N Clusters |
|----|-----------|-----------|------------|
| foxd5 | enteric_neurons,lateral_plate_mesoderm,muscle | 0.67 | 3 |
| dmrt2b | PSM,neural_telencephalon,notochord | 0.67 | 1 |

#### Moderately Specific (4-5 celltypes) - 2 TFs

| TF | N CTs | Celltypes | Coherence | N Clusters |
|-------|-------|-----------|-----------|------------|
| neurod6a | 5 | PSM,fast_muscle,muscle,somites,tail_bud | 0.80 | 2 |
| zic4 | 4 | NMPs,PSM,fast_muscle,muscle | 0.75 | 4 |

### Other

**Total specific TFs:** 3

#### Ultra-Specific (3 celltypes) - 3 TFs

| TF | Celltypes | Coherence | N Clusters |
|----|-----------|-----------|------------|
| alx1 | floor_plate,optic_cup,primordial_germ_cells | 1.00 | 2 |
| lhx8a | lateral_plate_mesoderm,primordial_germ_cells,tail_bud | 0.67 | 1 |
| lhx4 | enteric_neurons,pharyngeal_arches,pronephros | 0.67 | 1 |

## Known Lineage Markers Validation

Checking if well-known lineage markers appear as expected:

| Marker | Expected Lineage | Found | N CTs | Celltypes | Coherence |
|--------|-----------------|-------|-------|-----------|----------|
| tbx6 | mesoderm (PSM) | Not found | - | - | - |
| tbxta | mesoderm (notochord) | Not found | - | - | - |
| nkx2.5 | mesoderm (heart) | ✗ (too broad: 20 CTs) | 20 | NMPs,PSM,differentiating_neurons,endoder... | 0.40 |
| sox2 | neuroectoderm | ✗ (too broad: 21 CTs) | 21 | NMPs,PSM,differentiating_neurons,endoder... | 0.29 |
| pax6a | neuroectoderm (optic) | ✗ (too broad: 32 CTs) | 32 | NMPs,PSM,differentiating_neurons,endocri... | 0.34 |
| foxa2 | endoderm | Not found | - | - | - |
| sox17 | endoderm | ✗ (too broad: 24 CTs) | 24 | NMPs,PSM,differentiating_neurons,endocri... | 0.29 |
| krt4 | periderm/ectoderm | Not found | - | - | - |

## Methodology

### Lineage Groupings

**Neuroectoderm:** neural, neural_floor_plate, neural_optic, neural_posterior, neural_crest, spinal_cord, differentiating_neurons, enteric_neurons

**Mesoderm:** PSM, somites, fast_muscle, slow_muscle, muscle, heart_myocardium, lateral_plate_mesoderm, hemangioblasts, hematopoietic_vasculature, notochord

**Endoderm:** endoderm, endocrine_pancreas, liver, pharyngeal_endoderm

**Periderm Ectoderm:** epidermis, periderm, hatching_gland

**Other:** pronephros, optic_cup, tail_bud, NMPs

### Specificity Criteria

- Present in 3-5 celltypes
- Lineage coherence ≥ 60%
- Present in ≥ 1 cluster(s)

### Lineage Coherence Score

Calculated as: (number of celltypes in dominant lineage) / (total celltypes)

A score of 1.0 means all celltypes belong to the same lineage (perfect coherence).
