# Lineage Dynamics Analysis Summary

## Overview

This analysis examines how subGRNs (sub-Gene Regulatory Networks) change across developmental lineages at peak timepoints, rather than across time. The goal is to identify regulatory programs that show **divergent patterns** between neural and mesodermal lineages.

## Lineage Definitions

### Neural Lineage
```
neural_posterior ← spinal_cord ← NMPs
(terminal)      (intermediate)  (progenitor)
```

### Mesodermal Lineage
```
NMPs → tail_bud → PSM → somites → fast_muscle
(progenitor) (early) (intermediate) (late) (terminal)
```

## Scoring System

Clusters are scored based on:
- **35%** Lineage specificity: |lineage_bias_score| (how divergent)
- **25%** Trajectory clarity: clear progressive patterns
- **20%** Network complexity: moderate size (10-30 edges, 10-25 nodes)
- **20%** Developmental TF enrichment: presence of known TFs

### Lineage Bias Score
```
lineage_bias = (neural_strength - mesoderm_strength) / (neural_strength + mesoderm_strength)
```
- Ranges from **-1** (purely mesodermal) to **+1** (purely neural)
- Strength computed using both edge counts and edge weights

## Results Summary

Found **49 clusters** with valid lineage dynamics from 60 temporal dynamics candidates.

### Lineage Bias Distribution
- **Neural-biased** (bias > 0.3): 1 cluster
- **Mesoderm-biased** (bias < -0.3): 0 clusters
- **Balanced** (|bias| ≤ 0.3): 48 clusters

Most clusters show moderate bias, indicating complex lineage-specific regulation.

## Top 10 Candidates

### 1. Cluster 3_2 - hindbrain (NEURAL-BIASED)
- **Lineage bias:** 0.261 (neural > mesoderm)
- **Score:** 0.710
- **Peak timepoint:** 20 somites
- **Pattern:** Neural weakening (29→26→10 edges), Mesoderm strengthening (10→11→15→11→17)
- **Key TFs:** pax6a, pax6b, sox11a, sox13, sox21a, sox3, sox5, sox6, sox7 (all neural)

### 2. Cluster 3_8 - neural_telencephalon (NEURAL-BIASED)
- **Lineage bias:** 0.242
- **Score:** 0.703
- **Peak timepoint:** 20 somites
- **Pattern:** Both weakening, but neural stronger initially (48→27→18 vs 18→17→23→21→17)
- **Key TFs:** 15 developmental TFs including hoxa4a, hoxb3a, meox1, nkx2.1, nkx2.2a, pax6a/b, sox10, sox family

### 3. Cluster 2_2 - enteric_neurons (NEURAL-BIASED)
- **Lineage bias:** 0.327 (highest neural bias!)
- **Score:** 0.702
- **Peak timepoint:** 10 somites
- **Pattern:** Neural weakening (35→31→13), Mesoderm peaked (13→11→15→17→11)
- **Key TFs:** 24 developmental TFs! Including hox, pax3/7, phox2a, sox family (neural 21, mesoderm 16)

### 4. Cluster 22_16 - muscle (MESODERM-BIASED)
- **Lineage bias:** -0.214 (mesoderm > neural)
- **Score:** 0.694
- **Peak timepoint:** 10 somites
- **Pattern:** Neural strengthening (10→17→18), Mesoderm weakening (18→35→29→20→14)
- **Key TFs:** olig2, sox family (neural 5, mesoderm 10)

### 5. Cluster 17_7 - tail_bud (MESODERM-BIASED)
- **Lineage bias:** -0.178
- **Score:** 0.681
- **Peak timepoint:** 05 somites (early!)
- **Pattern:** Neural strengthening (11→12→28), Mesoderm weakening (28→24→22→21→27)
- **Key TFs:** hoxa4a, hoxb3a, meox1, pax3a, pax3b, pax7b

### 6. Cluster 3_9 - neural_floor_plate (NEURAL-BIASED)
- **Lineage bias:** 0.266
- **Score:** 0.681
- **Peak timepoint:** 10 somites
- **Pattern:** Neural weakening (38→32→21), Mesoderm peaked (21→14→18→22→13)
- **Key TFs:** 17 developmental TFs including hox, pax6, sox family

### 7. Cluster 26_10 - neural_floor_plate (NEURAL-BIASED)
- **Lineage bias:** 0.164
- **Score:** 0.666
- **Peak timepoint:** 00 somites (earliest!)
- **Pattern:** Both weakening (29→27→13 vs 13→11→24→18)
- **Key TFs:** 13 sox family TFs

### 8. Cluster 3_7 - optic_cup (NEURAL-BIASED)
- **Lineage bias:** 0.104
- **Score:** 0.655
- **Peak timepoint:** 20 somites
- **Pattern:** Neural weakening (47→35→15), Mesoderm strengthening (15→25→33→36→24)
- **Key TFs:** pax2a, pax3a, pax6a/b, pax7a/b

### 9. Cluster 10_6 - muscle (MESODERM-BIASED)
- **Lineage bias:** -0.242 (strongest mesoderm bias!)
- **Score:** 0.644
- **Peak timepoint:** 15 somites
- **Pattern:** Neural strengthening (16→34→53), Mesoderm weakening (53→66→57→55→50)
- **Key TFs:** hoxa4a, hoxb3a, hoxd3a, meox1, nkx1.2la (shared by both)
- **Note:** Largest networks (48 avg edges)

### 10. Cluster 14_1 - heart_myocardium (BALANCED)
- **Lineage bias:** 0.028 (nearly balanced!)
- **Score:** 0.629
- **Peak timepoint:** 10 somites
- **Pattern:** Both weakening similarly (20→19→18 vs 18→15→14→25→18)
- **Key TFs:** gata2a, gata3, gata5, gata6, olig2

## Key Biological Insights

### Neural-Biased Patterns
- **Common trajectory:** Networks tend to **weaken** along neural lineage (neural_posterior → spinal_cord → NMPs)
- **Key TF families:** Sox, Pax6, Phox2, Nkx
- **Strongest bias:** Cluster 2_2 (enteric_neurons, bias=0.327) with 24 developmental TFs

### Mesoderm-Biased Patterns
- **Common trajectory:** Networks often **weaken** along mesodermal lineage after initial strengthening
- **Peak activity:** Often at PSM or somites stage
- **Key TF families:** Meox, Pax3/7, Hox, Myod
- **Strongest bias:** Cluster 10_6 (muscle, bias=-0.242) with large networks

### Divergent Patterns (Most Interesting!)
- **Cluster 3_2 (hindbrain):** Neural weakening vs Mesoderm strengthening
- **Cluster 22_16 (muscle):** Neural strengthening vs Mesoderm weakening
- **Cluster 17_7 (tail_bud):** Neural strengthening vs Mesoderm weakening

## Files Generated

### Analysis Results
- `lineage_dynamics_ranking.csv`: Complete ranking of 49 clusters with all metrics
- `lineage_dynamics_output.log`: Analysis log

### Visualizations
- `subGRN_lineage_{rank}_{cluster_id}_{celltype}.png` (10 files, PNG for quick review)
- `subGRN_lineage_{rank}_{cluster_id}_{celltype}.pdf` (10 files, PDF for manuscript)

Each visualization shows:
- **Top row:** Neural lineage (neural_posterior ← spinal_cord ← NMPs)
- **Bottom row:** Mesodermal lineage (NMPs → tail_bud → PSM → somites → fast_muscle)
- **Node colors:**
  - Blue: Neural TFs
  - Purple: Mesoderm TFs
  - Orange: Other developmental TFs
  - Red: Other TFs
  - Gray: Target genes
- **Edge thickness:** Proportional to regulatory strength (edge weight)

## Comparison to Temporal Dynamics

### Temporal Analysis (Previous)
- **Question:** How do networks change over time?
- **Focus:** Developmental progression across timepoints
- **Best for:** Understanding developmental timing and TF turnover

### Lineage Analysis (Current)
- **Question:** How do networks differ across cell fates?
- **Focus:** Divergence between neural vs mesodermal branches
- **Best for:** Understanding lineage specification and fate decisions

### Complementary Insights
- **Temporal:** Shows when regulatory programs activate/deactivate
- **Lineage:** Shows where (which lineages) regulatory programs are active
- Together: Provides spatiotemporal view of GRN dynamics

## Next Steps

1. **Literature validation:** Use Research agent to validate biological relevance of divergent patterns
2. **Functional analysis:** Focus on clusters showing strongest divergence (|bias| > 0.2)
3. **TF functional roles:** Investigate why certain TFs are lineage-specific vs shared
4. **Integration:** Combine temporal and lineage analyses for comprehensive view

## Technical Details

- **Minimum thresholds:** 10 edges per celltype, 2+ celltypes per lineage
- **Layout:** Consistent node positions across all panels using spring layout (k=1.8, scale=1.8)
- **Strength calculation:** Equal weighting of edge counts and edge weights (50% each)
