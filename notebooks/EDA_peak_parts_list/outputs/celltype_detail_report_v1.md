# Parts List: Celltype Detail Report

**Generated**: `2026-04-04`

**Cell types analyzed**: fast_muscle, heart_myocardium, neural_crest, PSM, epidermis, primordial_germ_cells, hemangioblasts

**Method**: Specificity z-score = leave-one-out z-score across 190 (celltype × timepoint) conditions.
For each peak, z-score measures how much more accessible it is in one condition relative to all others.

**Reverse lookup**: For each known marker gene, find associated peaks and report their z-score
in the representative condition (highest n_cells). This validates whether the parts list captures
known biology.

---

## Fast Muscle

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 9 | no | 7,618 |
| 5somites | 45 | yes | 1,316 |
| 10somites | 133 | yes | 128 |
| 15somites | 672 | yes | 82 |
| 20somites | 975 | yes | 72 |
| 30somites | 162 | yes | 781 |

**Peak regulatory activity**: highest at **5somites** (1,316 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 9
- Recovered at z ≥ 2: 7 genes — `myog`, `myhz2`, `myhz1.1`, `mylpfa`, `smyd1b`, `myod1`, `tnnc2`
- Recovered at z ≥ 4: 2 genes — `myog`, `myhz2`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 9:5,357,112–5,357,311 | intergen | abhd13 | 30somites | 14.27 |
| 2 | 18:40,590,872–40,591,235 | intronic | *si:ch211-132b12.3* | 0somites | 12.05 |
| 3 | 14:16,787,154–16,787,353 | intronic | tcirg1b | 0somites | 11.62 |
| 4 | 4:63,097,683–63,097,916 | intergen | *zgc:173714* | 5somites | 10.85 |
| 5 | 4:32,100,796–32,101,154 | intergen | *BX649328.3* | 0somites | 10.84 |
| 6 | 22:15,748,241–15,748,453 | exonic | *si:dkeyp-70f9.7* | 0somites | 10.73 |
| 7 | 24:15,400,782–15,400,981 | intergen | *BX640518.1* | 0somites | 10.7 |
| 8 | 22:13,335,803–13,336,002 | intronic | *si:ch211-227m13.1* | 0somites | 10.32 |
| 9 | 4:63,172,426–63,172,680 | intergen | *CR450780.3* | 0somites | 10.29 |
| 10 | 9:44,921,397–44,921,655 | exonic | vil1 | 0somites | 10.22 |
| 11 | 24:12,351,640–12,351,843 | intergen | *—* | 0somites | 10.13 |
| 12 | 17:8,091,543–8,091,742 | intronic | syne1b | 0somites | 9.76 |
| 13 | 12:38,294,863–38,295,074 | intergen | *—* | 0somites | 9.76 |
| 14 | 4:31,115,722–31,115,967 | intergen | *BX649620.1* | 0somites | 9.69 |
| 15 | 22:9,193,040–9,193,472 | exonic | *si:ch211-213a13.5* | 5somites | 9.62 |
| 16 | 8:22,644,947–22,645,147 | intronic | iqsec2a | 0somites | 9.6 |
| 17 | 20:35,454,229–35,454,536 | exonic | tdrd6 | 0somites | 9.59 |
| 18 | 17:11,361,786–11,362,027 | intronic | *si:ch211-185a18.2* | 0somites | 9.59 |
| 19 | 16:15,899,113–15,899,312 | intergen | CT573799.1 | 0somites | 9.56 |
| 20 | 21:18,455,357–18,455,569 | intergen | *si:dkey-1d7.3* | 0somites | 9.56 |

> Figure: `figures/peak_parts_list/detail_fast_muscle.pdf`


## Heart Myocardium

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 312 | yes | 25 |
| 5somites | 485 | yes | 4 |
| 10somites | 388 | yes | 33 |
| 15somites | 1,072 | yes | 138 |
| 20somites | 580 | yes | 25 |
| 30somites | 97 | yes | 1,205 |

**Peak regulatory activity**: highest at **30somites** (1,205 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 15
- Recovered at z ≥ 2: 11 genes — `tnnt2a`, `tbx5a`, `myh6`, `tnni1b`, `tbx20`, `gata6`, `myh7`, `myl7`, `gata5`, `hand2`, `gata4`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 21:29,795,692–29,795,903 | intronic | *CR762480.2* | 15somites | 14.43 |
| 2 | 1:43,213,001–43,213,205 | intergen | *BX511108.2* | 15somites | 9.88 |
| 3 | 7:3,522,776–3,523,217 | intergen | *BX005456.1* | 30somites | 9.7 |
| 4 | 1:1,851,262–1,851,461 | intergen | atp1a1a.5 | 30somites | 8.55 |
| 5 | 21:29,526,258–29,526,784 | intronic | *BX537120.2* | 15somites | 8.54 |
| 6 | 21:29,406,377–29,406,576 | intergen | *zgc:171310* | 15somites | 8.27 |
| 7 | 3:10,682,725–10,683,183 | intergen | *si:ch73-1f23.1* | 30somites | 8.09 |
| 8 | 4:59,607,376–59,607,575 | intergen | *si:dkey-4e4.1* | 30somites | 7.64 |
| 9 | 3:11,315,473–11,315,672 | intronic | AL935044.2 | 30somites | 7.56 |
| 10 | 4:63,074,965–63,075,332 | exonic | *CR450780.2* | 15somites | 7.53 |
| 11 | 18:15,114,511–15,114,779 | intronic | *cry1b* | 30somites | 7.51 |
| 12 | 3:2,632,423–2,632,692 | intergen | *si:dkey-217f16.6* | 15somites | 7.31 |
| 13 | 5:66,939,208–66,939,506 | intergen | *—* | 30somites | 7.22 |
| 14 | 1:55,541,704–55,541,982 | intergen | adgre16 | 30somites | 7.21 |
| 15 | 5:46,641,218–46,641,433 | intergen | *BX465199.1* | 30somites | 7.17 |
| 16 | 1:57,741,239–57,741,542 | promoter | *si:dkey-1c7.1* | 15somites | 7.15 |
| 17 | 7:2,708,233–2,708,714 | intergen | *—* | 30somites | 7.11 |
| 18 | 7:8,153,489–8,153,705 | intronic | *si:cabz01030277.1* | 30somites | 7.04 |
| 19 | 16:2,145,267–2,145,480 | intergen | *—* | 30somites | 7.02 |
| 20 | 1:34,795,995–34,796,247 | intronic | *zgc:172122* | 30somites | 6.97 |

> Figure: `figures/peak_parts_list/detail_heart_myocardium.pdf`


## Neural Crest

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 417 | yes | 3 |
| 5somites | 745 | yes | 2 |
| 10somites | 776 | yes | 26 |
| 15somites | 2,146 | yes | 65 |
| 20somites | 1,027 | yes | 46 |
| 30somites | 159 | yes | 589 |

**Peak regulatory activity**: highest at **30somites** (589 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 11
- Recovered at z ≥ 2: 8 genes — `crestin`, `tfec`, `sox10`, `ednrab`, `twist1b`, `foxd3`, `sox9b`, `tfap2a`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 7:4,900,962–4,901,163 | intronic | *si:ch211-282j17.10* | 30somites | 13.0 |
| 2 | 4:50,019,670–50,019,870 | intergen | *si:dkey-156k2.4* | 15somites | 12.99 |
| 3 | 3:2,954,427–2,954,626 | intergen | *BX004816.3* | 15somites | 12.84 |
| 4 | 19:25,045,228–25,045,579 | intergen | xkr8.2 | 15somites | 10.25 |
| 5 | 9:42,073,288–42,073,487 | intronic | pcbp3 | 30somites | 10.17 |
| 6 | 1:58,298,590–58,298,822 | intergen | *si:dkey-222h21.2* | 30somites | 9.4 |
| 7 | 9:10,799,331–10,799,530 | intronic | *si:ch1073-416j23.1* | 30somites | 8.5 |
| 8 | 3:7,387,279–7,387,478 | intergen | *zgc:173517* | 30somites | 8.3 |
| 9 | 3:11,405,644–11,405,907 | intergen | AL935044.3 | 10somites | 7.58 |
| 10 | 4:71,791,772–71,792,189 | intronic | *si:dkeyp-4f2.1* | 15somites | 7.4 |
| 11 | 11:41,430,053–41,430,252 | intergen | park7 | 15somites | 7.35 |
| 12 | 2:10,540,161–10,540,727 | intronic | ccdc18 | 15somites | 7.33 |
| 13 | 6:5,318,404–5,318,618 | intergen | *—* | 30somites | 7.25 |
| 14 | 25:24,407,538–24,407,774 | intronic | b4galnt4a | 30somites | 6.91 |
| 15 | 22:8,033,341–8,033,646 | intronic | *CABZ01034698.2* | 15somites | 6.85 |
| 16 | 1:38,612,940–38,613,206 | intergen | *—* | 30somites | 6.82 |
| 17 | 16:43,895,920–43,896,119 | intronic | zfpm2a | 30somites | 6.73 |
| 18 | 1:57,006,077–57,006,334 | intronic | *si:ch211-1f22.13* | 30somites | 6.66 |
| 19 | 1:57,866,159–57,866,397 | intergen | *si:dkey-1c7.3* | 30somites | 6.58 |
| 20 | 5:61,898,636–61,898,846 | intergen | rsph4a | 30somites | 6.48 |

> Figure: `figures/peak_parts_list/detail_neural_crest.pdf`


## Psm

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 1,316 | yes | 2 |
| 5somites | 1,363 | yes | 1 |
| 10somites | 560 | yes | 2 |
| 15somites | 1,043 | yes | 63 |
| 20somites | 368 | yes | 10 |
| 30somites | 24 | yes | 8,011 |

**Peak regulatory activity**: highest at **30somites** (8,011 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 11
- Recovered at z ≥ 2: 7 genes — `her7`, `myf5`, `her1`, `tbx16`, `ripply2`, `msgn1`, `ripply1`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 7:5,793,406–5,793,620 | intronic | *si:dkey-10h3.7* | 30somites | 15.91 |
| 2 | 12:38,999,617–39,000,087 | exonic | *si:ch73-181m17.1* | 30somites | 15.61 |
| 3 | 21:13,076,472–13,076,671 | intronic | *zgc:109965* | 30somites | 13.05 |
| 4 | 2:1,106,982–1,107,181 | intronic | cacna1eb | 30somites | 12.1 |
| 5 | 24:9,982,698–9,982,897 | exonic | *zgc:171977* | 30somites | 11.85 |
| 6 | 17:37,094,209–37,094,408 | intronic | dtnbb | 30somites | 11.65 |
| 7 | 24:15,526,525–15,526,787 | intergen | *—* | 30somites | 11.03 |
| 8 | 14:2,304,400–2,304,648 | intronic | *si:ch73-379j16.2* | 30somites | 10.82 |
| 9 | 14:18,703,464–18,703,663 | intergen | slitrk4 | 30somites | 10.69 |
| 10 | 12:20,933,312–20,933,511 | intergen | *—* | 30somites | 10.36 |
| 11 | 4:43,647,947–43,648,148 | intronic | *si:dkeyp-53e4.4* | 30somites | 10.07 |
| 12 | 3:60,301,895–60,302,094 | intronic | *si:ch211-214b16.3* | 30somites | 10.0 |
| 13 | 11:10,185,459–10,185,658 | intergen | *—* | 30somites | 9.95 |
| 14 | 22:29,455,222–29,455,421 | intergen | *—* | 30somites | 9.6 |
| 15 | 22:27,537,848–27,538,047 | intergen | *CR388079.2* | 30somites | 9.38 |
| 16 | 4:45,491,268–45,491,484 | intergen | *si:dkey-256i11.6* | 15somites | 9.35 |
| 17 | 25:9,718,827–9,719,038 | intronic | lrrc4ca | 30somites | 9.3 |
| 18 | 17:20,554,131–20,554,336 | intronic | sh3pxd2ab | 30somites | 9.04 |
| 19 | 4:30,937,285–30,937,770 | intronic | *si:dkey-11n14.1* | 30somites | 8.99 |
| 20 | 19:39,691,364–39,691,563 | intergen | *—* | 30somites | 8.91 |

> Figure: `figures/peak_parts_list/detail_PSM.pdf`


## Epidermis

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 2,131 | yes | 21 |
| 5somites | 2,773 | yes | 10 |
| 10somites | 1,557 | yes | 34 |
| 15somites | 2,299 | yes | 114 |
| 20somites | 730 | yes | 55 |
| 30somites | 6 | no | 10,826 |

**Peak regulatory activity**: highest at **15somites** (114 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 10
- Recovered at z ≥ 2: 7 genes — `krt17`, `dlx3b`, `tp63`, `krt4`, `foxi3a`, `bmp2b`, `cdh1`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 18:33,899,537–33,899,768 | intergen | olfcq20 | 15somites | 15.2 |
| 2 | 14:2,303,270–2,303,782 | intronic | pcdh2ab9 | 30somites | 15.03 |
| 3 | 6:20,679,127–20,679,360 | exonic | col18a1b | 30somites | 13.42 |
| 4 | 3:11,356,563–11,356,845 | intergen | AL935044.4 | 30somites | 12.22 |
| 5 | 5:55,472,828–55,473,303 | intergen | *BX571945.2* | 30somites | 12.16 |
| 6 | 14:43,311,800–43,312,025 | intergen | LO018400.1 | 30somites | 11.98 |
| 7 | 22:31,363,393–31,363,638 | intronic | grip2b | 30somites | 11.37 |
| 8 | 15:28,027,734–28,027,983 | intronic | dhrs13a.3 | 30somites | 11.24 |
| 9 | 4:26,815,386–26,815,625 | intergen | *—* | 30somites | 11.23 |
| 10 | 17:36,465,545–36,465,826 | intronic | *BX005484.1* | 30somites | 11.21 |
| 11 | 7:17,583,064–17,583,279 | promoter | nitr1b | 30somites | 11.05 |
| 12 | 9:22,331,314–22,331,530 | intergen | *crygm2d1* | 30somites | 10.98 |
| 13 | 2:52,589,508–52,589,804 | intergen | gna11b | 30somites | 10.85 |
| 14 | 17:42,737,290–42,737,514 | intronic | prkd3 | 30somites | 10.84 |
| 15 | 1:8,627,084–8,627,339 | promoter | AL928650.1 | 30somites | 10.83 |
| 16 | 22:31,666,268–31,666,467 | intronic | lsm3 | 30somites | 10.79 |
| 17 | 16:18,381,131–18,381,444 | intergen | AL929304.1 | 30somites | 10.77 |
| 18 | 7:3,372,330–3,372,530 | intronic | *si:ch211-285c6.5* | 30somites | 10.72 |
| 19 | 22:28,026,901–28,027,327 | intergen | *—* | 30somites | 10.65 |
| 20 | 7:47,613,914–47,614,116 | intronic | *si:ch211-186j3.6* | 30somites | 10.62 |

> Figure: `figures/peak_parts_list/detail_epidermis.pdf`


## Primordial Germ Cells

> **Warning**: All timepoints have < 20 cells (unreliable pseudobulk). Specificity z-scores may be inflated. Interpret with caution.

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 12 | no | 10,772 |
| 5somites | 19 | no | 9,489 |
| 10somites | 13 | no | 9,803 |
| 15somites | 18 | no | 12,149 |
| 20somites | 19 | no | 8,407 |
| 30somites | 6 | no | 11,111 |

**Peak regulatory activity**: highest at **15somites** (12,149 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 7
- Recovered at z ≥ 2: 5 genes — `ddx4`, `dazl`, `dazap2`, `piwil1`, `nanos3`
- Recovered at z ≥ 4: 2 genes — `ddx4`, `dazl`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 3:2,892,920–2,893,199 | intronic | *CR388047.3* | 15somites | 22.18 |
| 2 | 3:2,785,512–2,785,713 | intronic | *CR388047.3* | 5somites | 19.2 |
| 3 | 7:4,710,038–4,710,284 | exonic | *si:ch211-225k7.4* | 15somites | 18.78 |
| 4 | 25:24,820,987–24,821,234 | intronic | BRSK2 | 10somites | 18.62 |
| 5 | 4:71,928,870–71,929,137 | intergen | *si:dkey-29j8.1* | 15somites | 18.54 |
| 6 | 4:66,667,102–66,667,315 | promoter | *si:ch211-149o24.4* | 15somites | 18.26 |
| 7 | 22:31,426,670–31,426,869 | intronic | grip2b | 15somites | 17.88 |
| 8 | 24:35,171,726–35,171,925 | intergen | pcmtd1 | 15somites | 16.17 |
| 9 | 3:2,764,756–2,765,171 | intronic | *CR388047.3* | 15somites | 15.88 |
| 10 | 18:31,595,252–31,595,480 | intergen | dhx36 | 20somites | 15.58 |
| 11 | 24:11,865,098–11,865,302 | intergen | tm9sf1 | 30somites | 15.51 |
| 12 | 1:48,559,462–48,559,661 | intergen | *—* | 30somites | 14.94 |
| 13 | 4:29,431,419–29,431,946 | intergen | *—* | 15somites | 14.91 |
| 14 | 19:27,554,027–27,554,226 | intronic | *si:dkeyp-46h3.8* | 0somites | 14.39 |
| 15 | 3:2,699,373–2,699,608 | intergen | *CR388047.3* | 15somites | 13.68 |
| 16 | 4:19,826,854–19,827,055 | intronic | cacna2d1a | 5somites | 13.57 |
| 17 | 1:56,879,366–56,879,951 | intergen | *si:ch211-152f2.2* | 15somites | 13.39 |
| 18 | 6:9,938,271–9,938,564 | intergen | zp2l2 | 15somites | 13.29 |
| 19 | 18:36,893,501–36,893,700 | intergen | *BX248121.1* | 15somites | 13.2 |
| 20 | 4:40,314,001–40,314,200 | intronic | znf1102 | 15somites | 13.15 |

> Figure: `figures/peak_parts_list/detail_primordial_germ_cells.pdf`


## Hemangioblasts

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 5somites | 44 | yes | 1,495 |
| 10somites | 181 | yes | 95 |
| 15somites | 359 | yes | 300 |
| 20somites | 374 | yes | 337 |
| 30somites | 219 | yes | 833 |

**Peak regulatory activity**: highest at **5somites** (1,495 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 7
- Recovered at z ≥ 2: 5 genes — `gfi1aa`, `gata1a`, `fli1a`, `lmo2`, `tal1`
- Recovered at z ≥ 4: 2 genes — `gfi1aa`, `gata1a`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 10:31,077,025–31,077,224 | intronic | pknox2 | 30somites | 11.28 |
| 2 | 16:52,064,766–52,065,031 | intergen | MAN1C1 | 15somites | 10.7 |
| 3 | 19:24,232,665–24,232,871 | intergen | prf1.7 | 15somites | 10.18 |
| 4 | 4:63,114,015–63,114,267 | intergen | *zgc:173714* | 15somites | 9.52 |
| 5 | 11:36,618,281–36,618,592 | intronic | *CR457444.10* | 30somites | 9.23 |
| 6 | 3:2,843,354–2,843,577 | intronic | *CR388047.3* | 5somites | 9.22 |
| 7 | 7:4,256,419–4,256,742 | intergen | klhl33 | 30somites | 8.67 |
| 8 | 7:3,788,157–3,788,356 | intronic | *si:ch73-335d12.2* | 5somites | 8.6 |
| 9 | 11:15,413,031–15,413,416 | intronic | ghrh | 15somites | 8.52 |
| 10 | 3:10,658,132–10,658,331 | intronic | *si:ch73-1f23.1* | 5somites | 8.52 |
| 11 | 22:31,689,016–31,689,482 | intergen | lsm3 | 15somites | 8.46 |
| 12 | 15:9,708,066–9,708,330 | intergen | *—* | 15somites | 8.09 |
| 13 | 12:14,464,974–14,465,173 | intergen | *—* | 15somites | 7.72 |
| 14 | 8:12,014,507–12,014,735 | intronic | ntng2a | 5somites | 7.64 |
| 15 | 11:36,641,840–36,642,113 | intronic | *CR457444.1* | 30somites | 7.58 |
| 16 | 4:1,895,927–1,896,412 | exonic | ano6 | 30somites | 7.58 |
| 17 | 9:30,236,812–30,237,358 | intronic | *si:dkey-100n23.3* | 30somites | 7.48 |
| 18 | 3:2,845,082–2,845,339 | intronic | *CR388047.3* | 15somites | 7.43 |
| 19 | 11:35,941,606–35,941,869 | intronic | itpr1b | 30somites | 7.42 |
| 20 | 11:20,145,024–20,145,241 | intergen | ogfrl2 | 5somites | 7.38 |

> Figure: `figures/peak_parts_list/detail_hemangioblasts.pdf`
