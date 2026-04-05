# Parts List Case Studies: Temporal Specificity Profiles

**Generated**: `2026-04-04`

**Method**: For each cell type, marker-gene-associated peaks are ranked by
their V2 (shrinkage-corrected) specificity z-score. Temporal profiles show
accessibility (log-norm) and specificity (z-score) across all available
developmental timepoints. The peak-type distribution and TF motif enrichment
demonstrate the regulatory character of the most specific peaks.

---

## Psm

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **myf5** | 4:21,745,790–21,746,582 | distal enhancer | 2.89 | tcf21, tbx2a, elk4 | 5s |
| **her7** | 5:68,786,324–68,787,220 | distal enhancer | 2.81 | tcf21, tbx2a, elk4 | 0s |
| **msgn1** | 4:297,025–298,970 | distal enhancer | 2.69 | hoxc13b, sox19b, nr3c2 | 0s |
| **tbx16** | 8:51,746,766–51,746,994 | intronic enhancer | 2.59 | tcf21, tbx2a, elk4 | 10s |
| **ripply1** | 21:37,970,238–37,970,877 | exonic | 2.58 | tcf21, tbx2a, elk4 | 0s |

### Peak type distribution (top 100 most specific peaks)
distal enhancer: 51  intronic enhancer: 44  exonic: 4  promoter: 1

### Enriched TF motifs (weighted across selected peaks)
**tcf21** (40.1), **tbx2a** (37.6), **elk4** (29.7), **hoxc13b** (7.0), **sox19b** (6.9)

> Figure: `figures/peak_parts_list/case_studies/case_study_PSM.pdf`


## Heart Myocardium

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **tnnt2a** | 23:5,684,874–5,685,418 | promoter | 4.03 | CABZ01057488.2, tead1b, CABZ01057488.2 | 15s |
| **tbx5a** | 5:72,243,629–72,244,522 | distal enhancer | 3.89 | CABZ01057488.2, tead1b, CABZ01057488.2 | 15s |
| **myh6** | 20:53,572,779–53,574,279 | distal enhancer | 3.83 | hsf1, CABZ01057488.2, fli1a | 15s |
| **tnni1b** | 6:54,825,787–54,826,266 | promoter | 3.58 | CABZ01057488.2, tead1b, CABZ01057488.2 | 15s |
| **tbx20** | 16:42,549,255–42,549,960 | distal enhancer | 3.21 | CABZ01057488.2, tead1b, CABZ01057488.2 | 5s |

### Peak type distribution (top 100 most specific peaks)
intronic enhancer: 46  distal enhancer: 44  promoter: 6  exonic: 4

### Enriched TF motifs (weighted across selected peaks)
**CABZ01057488.2** (107.6), **tead1b** (49.7), **hsf1** (9.3), **fli1a** (8.5)

> Figure: `figures/peak_parts_list/case_studies/case_study_heart_myocardium.pdf`


## Neural Crest

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **crestin** | 4:47,260,670–47,260,930 | exonic | 3.88 | nr3c2, sox19b, mafb | 0s |
| **tfec** | 4:6,018,912–6,019,270 | distal enhancer | 3.67 | six2a, tfap2d, CABZ01057488.2 | 15s |
| **sox10** | 3:1,460,433–1,461,465 | distal enhancer | 3.61 | hsf1, CABZ01057488.2, fli1a | 15s |
| **ednrab** | 23:45,692,607–45,693,592 | distal enhancer | 3.29 | tfap2d, sox19b, sox19b | 10s |
| **sox9b** | 3:62,576,027–62,577,754 | distal enhancer | 2.97 | hsf1, CABZ01057488.2, fli1a | 10s |

### Peak type distribution (top 100 most specific peaks)
distal enhancer: 55  intronic enhancer: 36  exonic: 6  promoter: 3

### Enriched TF motifs (weighted across selected peaks)
**tfap2d** (29.2), **CABZ01057488.2** (27.8), **sox19b** (26.5), **hsf1** (16.0), **six2a** (15.0)

> Figure: `figures/peak_parts_list/case_studies/case_study_neural_crest.pdf`


## Fast Muscle

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **myog** | 11:22,595,578–22,595,793 | promoter | 6.34 | nr3c2, sox19b, mafb | 20s |
| **myhz2** | 5:32,190,744–32,190,975 | distal enhancer | 4.47 | ctcf, ctcf, tfcp2 | 30s |
| **myhz1.1** | 5:32,280,725–32,281,354 | distal enhancer | 4.05 | nr3c2, sox19b, mafb | 30s |
| **mylpfa** | 3:32,817,171–32,817,416 | exonic | 3.32 | tcf21, tcf21, tcf12 | 30s |
| **smyd1b** | 8:1,056,419–1,057,057 | distal enhancer | 3.26 | zbtb7a, znf76, pknox1.1 | 15s |

### Peak type distribution (top 100 most specific peaks)
distal enhancer: 44  intronic enhancer: 43  promoter: 8  exonic: 5

### Enriched TF motifs (weighted across selected peaks)
**ctcf** (34.9), **nr3c2** (34.0), **tcf21** (27.3), **sox19b** (26.3), **mafb** (23.6)

> Figure: `figures/peak_parts_list/case_studies/case_study_fast_muscle.pdf`


## Primordial Germ Cells

> **[!] All timepoints unreliable (n_cells < 20).** Z-scores are shrinkage-corrected (V2, alpha=20). Interpret with caution.

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **ddx4** | 10:6,901,205–6,901,414 | exonic | 2.16 | nr3c2, sox19b, mafb | 5s |
| **dazl** | 19:20,401,818–20,402,151 | intronic enhancer | 2.13 | ctcf, ctcf, tfcp2 | 5s |
| **dazap2** | 23:33,729,546–33,730,227 | exonic | 1.53 | six2a, tfap2d, CABZ01057488.2 | 5s |

### Peak type distribution (top 100 most specific peaks)
distal enhancer: 49  intronic enhancer: 38  exonic: 9  promoter: 4

### Enriched TF motifs (weighted across selected peaks)
**ctcf** (15.3), **nr3c2** (7.1), **tfcp2** (7.0), **six2a** (6.2), **tfap2d** (6.1)

> Figure: `figures/peak_parts_list/case_studies/case_study_primordial_germ_cells.pdf`


## Hemangioblasts

### Selected marker-gene-associated peaks

| Gene | Coordinates | Peak type | Best Z (V2) | Top TF motifs | Peak of specificity |
|------|-------------|-----------|-------------|---------------|---------------------|
| **gfi1aa** | 2:10,754,403–10,754,612 | intronic enhancer | 4.33 | fli1a, hsf1, elf2a | 20s |
| **gata1a** | 11:25,418,422–25,419,319 | promoter | 4.25 | fli1a, hsf1, elf2a | 30s |
| **lmo2** | 18:38,286,439–38,287,243 | promoter | 3.46 | fli1a, hsf1, elf2a | 30s |
| **fli1a** | 18:48,431,360–48,431,782 | intronic enhancer | 3.35 | fli1a, hsf1, elf2a | 20s |
| **tal1** | 22:16,536,854–16,538,049 | exonic | 3.09 | hoxc13b, sox19b, nr3c2 | 30s |

### Peak type distribution (top 100 most specific peaks)
intronic enhancer: 34  distal enhancer: 32  promoter: 23  exonic: 11

### Enriched TF motifs (weighted across selected peaks)
**fli1a** (64.7), **hsf1** (56.7), **elf2a** (52.0), **hoxc13b** (8.0), **sox19b** (8.0)

> Figure: `figures/peak_parts_list/case_studies/case_study_hemangioblasts.pdf`
