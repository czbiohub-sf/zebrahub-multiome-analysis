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
| 0somites | 9 | no | 0 |
| 5somites | 45 | yes | 1,840 |
| 10somites | 133 | yes | 182 |
| 15somites | 672 | yes | 103 |
| 20somites | 975 | yes | 145 |
| 30somites | 162 | yes | 1,317 |

**Peak regulatory activity**: highest at **5somites** (1,840 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 9
- Recovered at z ≥ 2: 7 genes — `myog`, `myhz2`, `myhz1.1`, `mylpfa`, `smyd1b`, `tnnc2`, `myod1`
- Recovered at z ≥ 4: 3 genes — `myog`, `myhz2`, `myhz1.1`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 9:5,357,112–5,357,311 | intergen | abhd13 | 30somites | 14.3 |
| 2 | 4:63,097,683–63,097,916 | intergen | *zgc:173714* | 5somites | 10.87 |
| 3 | 22:9,193,040–9,193,472 | exonic | *si:ch211-213a13.5* | 5somites | 9.66 |
| 4 | 7:2,420,638–2,420,965 | intronic | *BX323987.1* | 5somites | 8.97 |
| 5 | 5:9,725,661–9,725,916 | intronic | ddr2l | 5somites | 8.83 |
| 6 | 4:30,791,217–30,791,444 | intronic | *si:dkey-178j11.5* | 10somites | 8.75 |
| 7 | 5:32,274,237–32,274,513 | promoter | myhz1.3 | 30somites | 8.3 |
| 8 | 7:3,737,923–3,738,178 | intergen | *si:ch211-282j17.10* | 30somites | 8.02 |
| 9 | 4:68,798,089–68,798,657 | intergen | *si:dkey-264f17.3* | 5somites | 7.85 |
| 10 | 1:58,118,774–58,119,071 | intronic | *si:ch211-15j1.5* | 30somites | 7.76 |
| 11 | 14:38,437,195–38,437,394 | intergen | *—* | 30somites | 7.69 |
| 12 | 15:37,886,848–37,887,047 | intergen | *CR944667.3* | 5somites | 7.69 |
| 13 | 5:65,218,773–65,218,972 | intronic | col27a1b | 5somites | 7.56 |
| 14 | 24:386,078–386,277 | intergen | VSTM2A | 5somites | 7.56 |
| 15 | 19:23,643,784–23,643,993 | intergen | *BX927216.1* | 5somites | 7.49 |
| 16 | 15:37,897,192–37,897,427 | intronic | *si:dkey-238d18.5* | 30somites | 7.38 |
| 17 | 21:12,542,853–12,543,158 | intergen | *CR293531.3* | 30somites | 7.32 |
| 18 | 22:25,198,609–25,198,839 | intergen | *si:ch211-226h8.4* | 5somites | 7.18 |
| 19 | 4:52,101,882–52,102,133 | intergen | *si:dkey-22a18.2* | 15somites | 7.09 |
| 20 | 1:38,583,938–38,584,226 | intergen | glra3 | 5somites | 7.05 |

> Figure: `figures/peak_parts_list/detail_fast_muscle.pdf`


## Heart Myocardium

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 312 | yes | 31 |
| 5somites | 485 | yes | 6 |
| 10somites | 388 | yes | 48 |
| 15somites | 1,072 | yes | 163 |
| 20somites | 580 | yes | 43 |
| 30somites | 97 | yes | 1,692 |

**Peak regulatory activity**: highest at **30somites** (1,692 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 15
- Recovered at z ≥ 2: 11 genes — `tnnt2a`, `tbx5a`, `myh6`, `tnni1b`, `tbx20`, `gata6`, `myh7`, `gata5`, `myl7`, `hand2`, `gata4`
- Recovered at z ≥ 4: 1 genes — `tnnt2a`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 21:29,795,692–29,795,903 | intronic | *CR762480.2* | 15somites | 14.44 |
| 2 | 1:43,213,001–43,213,205 | intergen | *BX511108.2* | 15somites | 9.89 |
| 3 | 7:3,522,776–3,523,217 | intergen | *BX005456.1* | 30somites | 9.72 |
| 4 | 1:1,851,262–1,851,461 | intergen | atp1a1a.5 | 30somites | 8.63 |
| 5 | 21:29,526,258–29,526,784 | intronic | *BX537120.2* | 15somites | 8.56 |
| 6 | 21:29,406,377–29,406,576 | intergen | *zgc:171310* | 15somites | 8.28 |
| 7 | 3:10,682,725–10,683,183 | intergen | *si:ch73-1f23.1* | 30somites | 8.12 |
| 8 | 4:59,607,376–59,607,575 | intergen | *si:dkey-4e4.1* | 30somites | 7.66 |
| 9 | 3:11,315,473–11,315,672 | intronic | AL935044.2 | 30somites | 7.59 |
| 10 | 4:63,074,965–63,075,332 | exonic | *CR450780.2* | 15somites | 7.53 |
| 11 | 18:15,114,511–15,114,779 | intronic | *cry1b* | 30somites | 7.53 |
| 12 | 3:2,632,423–2,632,692 | intergen | *si:dkey-217f16.6* | 15somites | 7.32 |
| 13 | 1:55,541,704–55,541,982 | intergen | adgre16 | 30somites | 7.25 |
| 14 | 5:66,939,208–66,939,506 | intergen | *—* | 30somites | 7.23 |
| 15 | 5:46,641,218–46,641,433 | intergen | *BX465199.1* | 30somites | 7.19 |
| 16 | 1:57,741,239–57,741,542 | promoter | *si:dkey-1c7.1* | 15somites | 7.15 |
| 17 | 7:2,708,233–2,708,714 | intergen | *—* | 30somites | 7.12 |
| 18 | 7:8,153,489–8,153,705 | intronic | *si:cabz01030277.1* | 30somites | 7.05 |
| 19 | 16:2,145,267–2,145,480 | intergen | *—* | 30somites | 7.04 |
| 20 | 1:34,795,995–34,796,247 | intronic | *zgc:172122* | 30somites | 7.0 |

> Figure: `figures/peak_parts_list/detail_heart_myocardium.pdf`


## Neural Crest

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 417 | yes | 4 |
| 5somites | 745 | yes | 2 |
| 10somites | 776 | yes | 50 |
| 15somites | 2,146 | yes | 101 |
| 20somites | 1,027 | yes | 95 |
| 30somites | 159 | yes | 826 |

**Peak regulatory activity**: highest at **30somites** (826 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 11
- Recovered at z ≥ 2: 9 genes — `crestin`, `tfec`, `sox10`, `ednrab`, `sox9b`, `twist1b`, `foxd3`, `tfap2a`, `snai1b`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 7:4,900,962–4,901,163 | intronic | *si:ch211-282j17.10* | 30somites | 13.06 |
| 2 | 4:50,019,670–50,019,870 | intergen | *si:dkey-156k2.4* | 15somites | 13.0 |
| 3 | 3:2,954,427–2,954,626 | intergen | *BX004816.3* | 15somites | 12.84 |
| 4 | 19:25,045,228–25,045,579 | intergen | xkr8.2 | 15somites | 10.29 |
| 5 | 9:42,073,288–42,073,487 | intronic | pcbp3 | 30somites | 10.22 |
| 6 | 1:58,298,590–58,298,822 | intergen | *si:dkey-222h21.2* | 30somites | 9.41 |
| 7 | 9:10,799,331–10,799,530 | intronic | *si:ch1073-416j23.1* | 30somites | 8.52 |
| 8 | 3:7,387,279–7,387,478 | intergen | *zgc:173517* | 30somites | 8.31 |
| 9 | 3:11,405,644–11,405,907 | intergen | AL935044.3 | 10somites | 7.62 |
| 10 | 4:71,791,772–71,792,189 | intronic | *si:dkeyp-4f2.1* | 15somites | 7.45 |
| 11 | 2:10,540,161–10,540,727 | intronic | ccdc18 | 15somites | 7.36 |
| 12 | 11:41,430,053–41,430,252 | intergen | park7 | 15somites | 7.35 |
| 13 | 6:5,318,404–5,318,618 | intergen | *—* | 30somites | 7.27 |
| 14 | 10:30,855,172–30,855,371 | intergen | *—* | 30somites | 7.03 |
| 15 | 25:24,407,538–24,407,774 | intronic | b4galnt4a | 30somites | 7.0 |
| 16 | 1:38,612,940–38,613,206 | intergen | *—* | 30somites | 6.91 |
| 17 | 22:8,033,341–8,033,646 | intronic | *CABZ01034698.2* | 15somites | 6.85 |
| 18 | 16:43,895,920–43,896,119 | intronic | zfpm2a | 30somites | 6.77 |
| 19 | 1:57,006,077–57,006,334 | intronic | *si:ch211-1f22.13* | 30somites | 6.68 |
| 20 | 1:57,866,159–57,866,397 | intergen | *si:dkey-1c7.3* | 30somites | 6.61 |

> Figure: `figures/peak_parts_list/detail_neural_crest.pdf`


## Psm

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 1,316 | yes | 3 |
| 5somites | 1,363 | yes | 2 |
| 10somites | 560 | yes | 5 |
| 15somites | 1,043 | yes | 66 |
| 20somites | 368 | yes | 12 |
| 30somites | 24 | yes | 11,307 |

**Peak regulatory activity**: highest at **30somites** (11,307 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 11
- Recovered at z ≥ 2: 7 genes — `myf5`, `her7`, `msgn1`, `tbx16`, `ripply1`, `her1`, `ripply2`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 7:5,793,406–5,793,620 | intronic | *si:dkey-10h3.7* | 30somites | 15.94 |
| 2 | 12:38,999,617–39,000,087 | exonic | *si:ch73-181m17.1* | 30somites | 15.66 |
| 3 | 21:13,076,472–13,076,671 | intronic | *zgc:109965* | 30somites | 13.18 |
| 4 | 14:18,703,464–18,703,663 | intergen | slitrk4 | 30somites | 12.96 |
| 5 | 2:1,106,982–1,107,181 | intronic | cacna1eb | 30somites | 12.14 |
| 6 | 24:9,982,698–9,982,897 | exonic | *zgc:171977* | 30somites | 11.91 |
| 7 | 17:37,094,209–37,094,408 | intronic | dtnbb | 30somites | 11.71 |
| 8 | 24:15,526,525–15,526,787 | intergen | *—* | 30somites | 11.08 |
| 9 | 6:56,752,221–56,752,420 | intronic | cdh22 | 30somites | 11.0 |
| 10 | 14:2,304,400–2,304,648 | intronic | *si:ch73-379j16.2* | 30somites | 10.88 |
| 11 | 25:9,718,827–9,719,038 | intronic | lrrc4ca | 30somites | 10.48 |
| 12 | 12:20,933,312–20,933,511 | intergen | *—* | 30somites | 10.44 |
| 13 | 4:43,647,947–43,648,148 | intronic | *si:dkeyp-53e4.4* | 30somites | 10.09 |
| 14 | 11:10,185,459–10,185,658 | intergen | *—* | 30somites | 10.08 |
| 15 | 3:60,301,895–60,302,094 | intronic | *si:ch211-214b16.3* | 30somites | 10.07 |
| 16 | 14:49,027,659–49,028,041 | intronic | zdhhc5b | 30somites | 9.97 |
| 17 | 21:44,266,069–44,266,302 | intergen | *BX928756.1* | 30somites | 9.84 |
| 18 | 22:29,455,222–29,455,421 | intergen | *—* | 30somites | 9.67 |
| 19 | 1:8,627,084–8,627,339 | promoter | AL928650.1 | 30somites | 9.64 |
| 20 | 18:11,520,320–11,520,519 | intronic | PRMT8 | 30somites | 9.53 |

> Figure: `figures/peak_parts_list/detail_PSM.pdf`


## Epidermis

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 2,131 | yes | 41 |
| 5somites | 2,773 | yes | 16 |
| 10somites | 1,557 | yes | 62 |
| 15somites | 2,299 | yes | 169 |
| 20somites | 730 | yes | 104 |
| 30somites | 6 | no | 0 |

**Peak regulatory activity**: highest at **15somites** (169 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 10
- Recovered at z ≥ 2: 7 genes — `krt17`, `krt4`, `dlx3b`, `tp63`, `foxi3a`, `bmp2b`, `cdh1`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 18:33,899,537–33,899,768 | intergen | olfcq20 | 15somites | 15.2 |
| 2 | 7:6,540,887–6,541,086 | intronic | *si:ch1073-220m6.1* | 15somites | 9.82 |
| 3 | 17:41,600,871–41,601,109 | intronic | ralgapa2 | 15somites | 8.69 |
| 4 | 12:3,619,234–3,619,606 | intergen | coa3a | 15somites | 8.08 |
| 5 | 18:38,448,185–38,448,451 | intergen | *—* | 15somites | 7.0 |
| 6 | 24:36,888,566–36,888,883 | intergen | *CABZ01055343.1* | 15somites | 6.67 |
| 7 | 7:25,390,638–25,391,009 | intronic | *CU929416.1* | 15somites | 6.39 |
| 8 | 6:11,464,048–11,464,253 | intronic | col5a2b | 15somites | 6.37 |
| 9 | 22:8,758,864–8,759,095 | intronic | *si:dkey-182g1.2* | 15somites | 6.36 |
| 10 | 8:45,157,379–45,157,943 | intergen | *si:ch211-220m6.4* | 15somites | 6.33 |
| 11 | 2:22,326,438–22,326,795 | intergen | selenof | 15somites | 6.28 |
| 12 | 22:26,054,062–26,054,415 | intergen | pdgfaa | 15somites | 6.13 |
| 13 | 3:2,974,007–2,974,276 | intronic | *BX004816.1* | 15somites | 6.04 |
| 14 | 22:7,753,883–7,754,119 | exonic | cela1.5 | 15somites | 5.97 |
| 15 | 19:7,891,456–7,891,738 | intergen | nuggc.3 | 20somites | 5.94 |
| 16 | 22:28,066,146–28,066,345 | intergen | wu:fu71h07 | 15somites | 5.88 |
| 17 | 19:23,399,525–23,399,767 | intergen | pimr81 | 15somites | 5.83 |
| 18 | 17:28,211,641–28,211,852 | intergen | htr1d | 15somites | 5.82 |
| 19 | 4:59,245,969–59,246,179 | exonic | *si:dkey-105i14.1* | 10somites | 5.66 |
| 20 | 7:6,568,918–6,569,201 | intronic | *si:ch1073-220m6.1* | 15somites | 5.59 |

> Figure: `figures/peak_parts_list/detail_epidermis.pdf`


## Primordial Germ Cells

> **Warning**: All timepoints have < 20 cells (unreliable pseudobulk). Specificity z-scores may be inflated. Interpret with caution.

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 0somites | 12 | no | 6 |
| 5somites | 19 | no | 50 |
| 10somites | 13 | no | 6 |
| 15somites | 18 | no | 182 |
| 20somites | 19 | no | 49 |
| 30somites | 6 | no | 0 |

**Peak regulatory activity**: highest at **15somites** (182 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 7
- Recovered at z ≥ 2: 2 genes — `ddx4`, `dazl`
- Recovered at z ≥ 4: 0 genes — —

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 3:2,892,920–2,893,199 | intronic | *CR388047.3* | 15somites | 10.57 |
| 2 | 3:2,785,512–2,785,713 | intronic | *CR388047.3* | 5somites | 9.42 |
| 3 | 7:4,710,038–4,710,284 | exonic | *si:ch211-225k7.4* | 15somites | 8.97 |
| 4 | 4:71,928,870–71,929,137 | intergen | *si:dkey-29j8.1* | 15somites | 8.84 |
| 5 | 4:66,667,102–66,667,315 | promoter | *si:ch211-149o24.4* | 15somites | 8.72 |
| 6 | 22:31,426,670–31,426,869 | intronic | grip2b | 15somites | 8.54 |
| 7 | 24:35,171,726–35,171,925 | intergen | pcmtd1 | 15somites | 7.71 |
| 8 | 18:31,595,252–31,595,480 | intergen | dhx36 | 20somites | 7.66 |
| 9 | 3:2,764,756–2,765,171 | intronic | *CR388047.3* | 15somites | 7.58 |
| 10 | 25:24,820,987–24,821,234 | intronic | BRSK2 | 10somites | 7.4 |
| 11 | 4:29,431,419–29,431,946 | intergen | *—* | 15somites | 7.12 |
| 12 | 4:19,826,854–19,827,055 | intronic | cacna2d1a | 5somites | 6.65 |
| 13 | 3:2,699,373–2,699,608 | intergen | *CR388047.3* | 15somites | 6.53 |
| 14 | 1:56,879,366–56,879,951 | intergen | *si:ch211-152f2.2* | 15somites | 6.38 |
| 15 | 22:9,291,295–9,291,538 | exonic | *si:ch211-250k18.7* | 5somites | 6.36 |
| 16 | 6:9,938,271–9,938,564 | intergen | zp2l2 | 15somites | 6.33 |
| 17 | 18:36,893,501–36,893,700 | intergen | *BX248121.1* | 15somites | 6.32 |
| 18 | 4:40,314,001–40,314,200 | intronic | znf1102 | 15somites | 6.29 |
| 19 | 4:41,608,439–41,608,660 | intronic | *BX088603.1* | 15somites | 6.26 |
| 20 | 7:6,576,338–6,576,741 | intronic | *si:ch1073-220m6.1* | 15somites | 6.24 |

> Figure: `figures/peak_parts_list/detail_primordial_germ_cells.pdf`


## Hemangioblasts

### Temporal coverage

| Timepoint | N cells | Reliable | N peaks z ≥ 4 |
|-----------|---------|----------|---------------|
| 5somites | 44 | yes | 2,063 |
| 10somites | 181 | yes | 144 |
| 15somites | 359 | yes | 433 |
| 20somites | 374 | yes | 598 |
| 30somites | 219 | yes | 1,903 |

**Peak regulatory activity**: highest at **5somites** (2,063 peaks with z ≥ 4)

### Marker gene recovery (reverse lookup)

- Curated markers: 7
- Recovered at z ≥ 2: 5 genes — `gfi1aa`, `gata1a`, `lmo2`, `fli1a`, `tal1`
- Recovered at z ≥ 4: 2 genes — `gfi1aa`, `gata1a`

### Top 20 most specific peaks

| Rank | Coordinates | Type | Gene | Best TP | Z-score |
|------|-------------|------|------|---------|---------|
| 1 | 10:31,077,025–31,077,224 | intronic | pknox2 | 30somites | 11.32 |
| 2 | 16:52,064,766–52,065,031 | intergen | MAN1C1 | 15somites | 10.7 |
| 3 | 19:24,232,665–24,232,871 | intergen | prf1.7 | 15somites | 10.18 |
| 4 | 4:63,114,015–63,114,267 | intergen | *zgc:173714* | 15somites | 9.53 |
| 5 | 11:36,618,281–36,618,592 | intronic | *CR457444.10* | 30somites | 9.27 |
| 6 | 3:2,843,354–2,843,577 | intronic | *CR388047.3* | 5somites | 9.23 |
| 7 | 21:76,212–77,485 | promoter | ARSB | 30somites | 9.18 |
| 8 | 7:4,256,419–4,256,742 | intergen | klhl33 | 30somites | 8.72 |
| 9 | 7:3,788,157–3,788,356 | intronic | *si:ch73-335d12.2* | 5somites | 8.63 |
| 10 | 11:15,413,031–15,413,416 | intronic | ghrh | 15somites | 8.59 |
| 11 | 3:10,658,132–10,658,331 | intronic | *si:ch73-1f23.1* | 5somites | 8.54 |
| 12 | 22:31,689,016–31,689,482 | intergen | lsm3 | 15somites | 8.47 |
| 13 | 4:16,122,674–16,124,957 | promoter | atp2b1a | 30somites | 8.3 |
| 14 | 15:9,708,066–9,708,330 | intergen | *—* | 15somites | 8.09 |
| 15 | 21:249,356–252,563 | promoter | jak2a | 30somites | 8.01 |
| 16 | 4:1,895,927–1,896,412 | exonic | ano6 | 30somites | 7.78 |
| 17 | 9:30,236,812–30,237,358 | intronic | *si:dkey-100n23.3* | 30somites | 7.76 |
| 18 | 12:14,464,974–14,465,173 | intergen | *—* | 15somites | 7.76 |
| 19 | 8:12,014,507–12,014,735 | intronic | ntng2a | 5somites | 7.74 |
| 20 | 11:36,641,840–36,642,113 | intronic | *CR457444.1* | 30somites | 7.59 |

> Figure: `figures/peak_parts_list/detail_hemangioblasts.pdf`
