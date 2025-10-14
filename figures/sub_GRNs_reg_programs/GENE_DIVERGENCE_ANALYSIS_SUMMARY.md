# Gene Divergence Analysis Summary

## Overview

This analysis identifies clusters where **TFs and target genes progressively diverge** between neural and mesodermal lineages during development. The goal is to find regulatory programs that show gradual, lineage-specific changes as cells differentiate from NMPs (neuromesodermal progenitors).

## Key Concept: Progressive Divergence

**Progressive divergence** means that TF sets and/or target genes become increasingly different as cells move away from the NMPs branch point:

```
                    NMPs (branch point)
                  /                    \
        High overlap                  High overlap
                |                            |
        Intermediate overlap        Intermediate overlap
                |                            |
        Low overlap (NEURAL)        Low overlap (MESODERM)
     neural_posterior              fast_muscle
```

**Ideal pattern:**
- At NMPs: TFs/targets are similar between lineages (shared regulatory program)
- Along lineages: TFs/targets gradually diverge
- At terminals: TFs/targets are highly distinct (lineage-specific programs)

## Scoring Metrics

### 1. Progressive Score
Measures how much overlap decreases from NMPs to terminal cells:
```python
progressive_score = early_overlap - late_overlap
```
- High score = gradual divergence (ideal for manuscript)
- Low score = sudden change or no change

### 2. Divergence Score
Overall difference between lineages:
```python
divergence = 1 - average_overlap
```
- High divergence = distinct regulatory programs

### 3. Combined Progressive Score
```python
combined_score = tf_progressive_score + target_progressive_score
```
Used for ranking clusters.

## Top 10 Candidates with Progressive Divergence

### 1. **Cluster 2_2 - enteric_neurons (tp10)** ⭐ BEST CANDIDATE
- **Combined progressive score:** 1.628 (highest!)
- **TF progressive:** 0.828
- **Target progressive:** 0.800
- **Lineage bias:** 0.327 (neural-biased)
- **Neural-specific TFs:** 17 (emx2, hoxa4a, hoxb1b, hoxb3a, mafa, ...)
- **Mesoderm-specific TFs:** 7 (mafbb, meox1, pax3b, raraa, rorab, ...)
- **Interpretation:** Strong progressive divergence in BOTH TFs and targets. Excellent example of gradual lineage commitment from NMPs to enteric neurons.

### 2. **Cluster 3_5 - optic_cup (tp10)**
- **Combined progressive score:** 1.596
- **TF progressive:** 0.846 (very high!)
- **Target progressive:** 0.750
- **Lineage bias:** 0.074 (balanced)
- **Neural-specific TFs:** 6
- **Mesoderm-specific TFs:** 8
- **Interpretation:** High TF divergence with balanced lineage contribution. Shows how eye development diverges from mesodermal fate.

### 3. **Cluster 33_7 - neural_crest (tp10)**
- **Combined progressive score:** 1.510
- **TF progressive:** 0.810
- **Target progressive:** 0.700
- **Lineage bias:** 0.284 (neural-biased)
- **Neural-specific TFs:** 5 (nr5a2, pax6b, rorab, sox19b, sox3)
- **Mesoderm-specific TFs:** 3
- **Neural-specific targets:** 5
- **Interpretation:** Neural crest is a multipotent population. Shows clear divergence from mesodermal fate with both TF and target specificity.

### 4. **Cluster 14_1 - heart_myocardium (tp10)**
- **Combined progressive score:** 1.428
- **TF progressive:** 0.812
- **Target progressive:** 0.615
- **Lineage bias:** 0.028 (balanced)
- **Neural-specific TFs:** 5
- **Mesoderm-specific TFs:** 5
- **Mesoderm-specific targets:** 6
- **Interpretation:** Balanced TF contribution but strong target gene divergence. Heart development shows lineage-specific target regulation.

### 5. **Cluster 22_16 - muscle (tp10)**
- **Combined progressive score:** 1.375
- **TF progressive:** 0.500
- **Target progressive:** 0.875 (very high!)
- **Lineage bias:** -0.214 (mesoderm-biased)
- **Neural-specific TFs:** 2 (sox11a, sox11b)
- **Mesoderm-specific TFs:** 10
- **Mesoderm-specific targets:** 4
- **Interpretation:** Strong target gene divergence! Muscle development shows progressive commitment through changing target genes more than TFs.

### 6-10. Other Strong Candidates
- **17_8 - lateral_plate_mesoderm (tp15):** Score 1.325
- **13_12 - neural_optic (tp10):** Score 1.314
- **3_9 - neural_floor_plate (tp10):** Score 1.311
- **21_3 - PSM (tp00):** Score 1.254, 11 mesoderm-specific TFs!
- **25_3 - hematopoietic_vasculature (tp20):** Score 1.229, 9 mesoderm-specific TFs

## Biological Insights

### TF Divergence Patterns

**Neural-biased clusters** enrich for:
- Sox family (sox2, sox3, sox19a/b, sox11a/b)
- Pax6 (pax6a, pax6b)
- Hox (hoxa4a, hoxb1b, hoxb3a)
- Neural-specific: emx2, nr5a2, phox2a

**Mesoderm-biased clusters** enrich for:
- Meox family (meox1, meox2a/b)
- Pax3/7 (pax3a/b, pax7a/b)
- Muscle TFs: myod1, myf5
- Mesoderm-specific: raraa, rorab, rorb, lbx2

**Shared/Bipotential TFs:**
- Sox family members (context-dependent)
- Hox genes (positional identity)
- Some nuclear receptors

### Progressive vs Sudden Divergence

**Progressive divergence (ideal):**
- Gradual accumulation of lineage-specific factors
- Reflects step-wise commitment
- Examples: enteric_neurons (2_2), optic_cup (3_5)

**Target gene specificity:**
- Often more lineage-specific than TFs
- Example: muscle (22_16) has high target progressive score
- Suggests TFs acquire new targets during differentiation

## Visualization Updates

### Filename Format
Now includes timepoint for clarity:
```
subGRN_lineage_{rank}_{cluster_id}_{celltype}_tp{timepoint}.png
subGRN_lineage_{rank}_{cluster_id}_{celltype}_tp{timepoint}.pdf
```

Examples:
- `subGRN_lineage_3_2_2_enteric_neurons_tp10.png`
- `subGRN_lineage_9_10_6_muscle_tp15.pdf`

### Visual Style
- **Node colors:** lightcoral (TF-only), orange (TF & Target), lightblue (Target-only)
- **Edge colors:** darkred (activation), darkblue (repression)
- **Edge width:** Scaled by regulatory strength
- **Unified coordinates:** Same layout across all 8 panels

## Files Generated

### Analysis Results
1. **lineage_gene_divergence_ranking.csv** (49 clusters)
   - All divergence metrics
   - TF and target gene lists
   - Progressive scores

2. **gene_divergence_output.log**
   - Analysis log with detailed output

### Visualizations
All with timepoint in filename:
- 10 PNG files (quick review, 613KB-1.1MB)
- 10 PDF files (manuscript quality)

## Recommended Clusters for Manuscript

Based on progressive divergence + biological relevance:

### Top 3 Picks:

1. **Cluster 2_2 - enteric_neurons (tp10)** 
   - Highest progressive score (1.628)
   - Strong TF divergence (17 neural-specific, 7 mesoderm-specific)
   - Clear neural commitment from NMPs

2. **Cluster 22_16 - muscle (tp10)**
   - High target gene progressive score (0.875)
   - 10 mesoderm-specific TFs
   - Exemplifies target gene divergence during muscle commitment

3. **Cluster 33_7 - neural_crest (tp10)**
   - Balanced TF and target divergence
   - Neural crest multipotency
   - 5 neural-specific targets show fate restriction

### Alternative strong candidates:
- **3_5 - optic_cup (tp10):** Highest TF progressive (0.846)
- **14_1 - heart_myocardium (tp10):** Balanced TF distribution, strong target divergence
- **21_3 - PSM (tp00):** Earliest timepoint, 11 mesoderm-specific TFs

## Next Steps

1. **Visual inspection** of top candidates to confirm divergence patterns are clear
2. **Literature validation** of lineage-specific TFs and targets
3. **Functional analysis** of divergent TFs (e.g., which are necessary vs sufficient?)
4. **Integration** with temporal dynamics analysis for comprehensive view

## Technical Notes

- Analysis performed at peak timepoint for each cluster (max accessibility)
- Minimum 10 edges per celltype required
- Jaccard similarity used for overlap calculations
- Progressive score = early_overlap - late_overlap (positive = diverging)
- Combined score prioritizes clusters with BOTH TF and target divergence
