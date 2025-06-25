## Cluster 0

[*assistant*:
Peak cluster 0 corresponds to a broad “pan-neuronal” chromatin module.

1. Accessibility pattern  
   • Nearly all nervous‐system pseudobulks show moderate signal, but differentiating-neurons, enteric-neurons, spinal cord, hind-/mid-brain and generic “neurons” exhibit a sharp rise after the 10-somite stage (e.g. differentiating-neurons: 2.9 → 9.7 → 12–14 AU).  
   • Mesoderm-, endoderm- and epidermis-derived pseudobulks remain at baseline (~1.5–2 AU).  
   → The peaks therefore mark regulatory regions that switch on as neurogenesis begins and continue to strengthen through early organogenesis.

2. Gene content (representative)  
   • Synaptic vesicle/exocytosis: snap25b, syn1, syt1a, sv2bb/sv2ca, cplx2.  
   • Axon guidance/adhesion: unc5c, robo-like cntn/cntnap family, nrxn2a/3b, sdk1b, tenm2, tnr, trim67.  
   • Neurotransmission/channel activity: cacna1bb/a2d4b, gria3b, grin2bb, ryr3, scn3b, gabrb4, drd2b.  
   • Neural TFs and regulators: foxn2, rbfox1/3, ebf1, nrg1, nfixa/b, pou6f2, pknox2, onecut3b, wt1a.  
   These genes are hallmarks of maturing neurons (axonal extension, synaptogenesis, neurotransmission).

3. Motif enrichment  
   Highest z-scores map to motifs recognised by neuronal transcription factors, most prominently:  
   • MYRF (myelin/neuronal maturation),  
   • OSR1/2,  
   • SOX/FOX family, E–box/NeuroD class, and several HOX/POU homeodomain factors.  
   Collectively these TFs drive neuronal differentiation and the later maturation programme captured by the gene list above.

Interpretation  
Peak cluster 0 represents enhancers/promoters that become progressively more accessible as embryonic neurons exit the progenitor state, extend axons and form synapses. Its activity is largely confined to neuroectoderm-derived lineages (central and peripheral), with minimal activity in non-neural tissues. Functionally, it encodes the transcriptional circuitry for neuronal morphogenesis and synaptic competency during 10–24 hpf somitogenesis.
]

## Cluster 1

[*assistant*:
Peak cluster 1 – “house-keeping / core developmental programme”  

1. Accessibility profile  
   • The pseudo-bulk signal is detectable in virtually every cell type and at every somitic stage (values ~2-2.5 log-CPM).  
   • No single lineage dominates, although paraxial-mesoderm derivatives (posterior PSM, late somites, early muscle) show the very highest scores (e.g. PSM_30 s, 5.9).  
   • Only a few samples (e.g. NMPs_30 s, fast-muscle_0 s) fall to background, indicating that these peaks are never completely silent but can be modestly modulated.

2. Peak content and gene links  
   • >10 000 peaks are located at promoters or first introns and are associated with >7 000 genes.  
   • The gene list is dominated by “house-keeping” functions (ribosome, spliceosome, proteasome, TCA cycle, vesicle transport, cytoskeleton, ubiquitin pathway) together with a long tail of broadly-expressed developmental regulators (Fox, Sox, Hox, Pax, Tbx, Runx, E2f, etc.).  
   • Many of the linked genes are expressed in most or all single cells of the atlas, matching the near-ubiquitous accessibility of the peaks.

3. Motif enrichment  
   • The motif panel is very broad; the strongest z-scores (> 2) correspond to  
     – Forkhead/FOX family (FOXA/FOXC/FOXF/FOXO)  
     – SOX family  
     – Nuclear–receptor half sites (RAR/RXR/COUP-TF, NR2F)  
     – Generic promoter factors (E2F, SP/KLF, YY1/CTCF)  
     – General myeloid / RUNX-like sites (RUNX1/3) and MEF2/EGR/MAF motifs.  
   • This mixture is what is expected for constitutively open promoters that need to be recognised by many different TF families.

4. Biological interpretation  
   • Cluster 1 represents a set of constitutively accessible promoters/enhancers that support the basic metabolic and transcriptional needs of embryonic cells (“chromatin backbone”).  
   • The slight boost in PSM/somite samples suggests that, in addition to basal genes, the cluster also contains a subset of broadly-acting developmental regulators that become especially active in the paraxial mesoderm as somitogenesis proceeds.  
   • Because these peaks are never truly closed, they provide a permissive chromatin environment onto which lineage-specific clusters (described in other peak groups) can build more restricted regulatory programmes.

In short, peak cluster 1 captures the universal, always-on regulatory scaffolding of the zebrafish embryo, with modest tissue-specific tuning but no sharp lineage restriction.
]

## Cluster 2

[*assistant*:
Peak cluster 2 captures a regulatory program that is shared by many cell types but becomes most prominent in the nervous system – in particular in late-stage (30 somite, ≈24 hpf) enteric-neuron and differentiating-neuron pseudobulks, where its chromatin accessibility spikes more than four-fold above the background of other lineages.  

1. Accessibility profile  
• Broad, low–to–moderate accessibility (log-signal ≈4–6) in almost every lineage we profiled.  
• A conspicuous peak (≈22.7) in 30-somite enteric-neurons, with slightly higher values (≈5–6) in differentiating spinal-cord neurons, hindbrain and neural-crest derivatives, indicating a neuronal bias that strengthens as development proceeds from 10- to 30-somite stages.  

2. Gene neighbourhoods  
The correlated/overlapping genes are heavily enriched for:  
• Neuronal specification & differentiation TFs – sox (sox1/2/3/4/5/9/10/14/21), pax (pax2/3/6/7), dlx1/2/5/6, lhx1/2/4/8/9, phox2a/b, uncx, irx, nkx family members, neurod1/4/6, atoh1/7, olig2, tal1, runx1/3.  
• Axonogenesis / guidance & synaptic genes – robo1/2/3, slit1/2/3, sema3/4/5/6, epha/b, plxna/b, nrxn, nlgn, grid, gria, grik, gabra/b/g/rr, etc.  
• Posterior patterning genes – an extensive Hox, Meis, Pbx signature (hoxa/b/c/d clusters, hoxd9-13, pbx1-4, meis1-3), consistent with trunk / hindbrain / vagal-crest identity.  
• Generic neuronal cytoskeleton / signalling (flna/b, mef2, cacna1/2/4, kcn* channels, ryr2, snap25, rab3 family, unc13, syts, etc.).  

3. Enriched transcription-factor motifs  
Z-scored motif analysis highlights SOX, FOX, DLX, PAX, IRX, LHX, PHOX2, ZIC, MEF2, HOX-PBX/MEIS, E2F and CTCF sites.  The accompanying “indirect” TF list is dominated by the same neurogenic families, underscoring a composite enhancer repertoire used during neurogenesis and A-P patterning.  

4. Developmental interpretation  
Taken together, these peaks represent enhancers/promoters that:  
• are opened early and kept accessible in many germ-layers (hence the uniform low signal),  
• but are further remodelled/boosted when neuronal progenitors exit the cell-cycle and start differentiating, especially in trunk neural-crest–derived enteric neurons and posterior spinal neurons.  
They encode the transcriptional circuitry that couples axial (HOX/PBX) information to pan-neurogenic (SOX/NEUROD/IRX/DLX) and neuronal-subtype-specific (PHOX2, LHX, MEF2) programs, while still bearing footprints of generic proliferation regulators (E2F) that were active earlier.  

5. Suggested annotation  
“Neurogenic / posterior neural-crest & hind-spinal enhancer set (enteric-neuron-biased)” – a cluster of broadly primed but neuron-activated regulatory elements that help pattern and mature trunk/hindbrain neural tissues, with a pronounced role in the developing enteric nervous system around 24 hpf.
]

## Cluster 3

[*assistant*:
Peak-cluster 3 represents a large collection of cis-regulatory elements that become progressively more accessible in mid-to-posterior axial tissues as embryos move from the 0- to 30-somite stages.

1. Accessibility profile  
   • Highest signals are seen in the notochord (already ≥23-fold at 0 som; rising to ~32-fold at 20 som), neural floor plate, differentiating neurons, spinal cord, hindbrain, presomitic/somite mesoderm and floor-plate–related derivatives.  
   • Accessibility is only moderate in multipotent NMPs and PSM at the 0–5-somite stages, and stays low in endodermal, epidermal or terminally differentiated non-axial cell types.  
   • The temporal trend therefore follows the maturation of the axial neuro-mesodermal domain: initial opening in NMPs → rapid gain in floor plate, hindbrain and spinal cord → culmination in mature notochord and somites.

2. Gene set linked to the peaks  
   • Patterning / axial TFs: very dense coverage of Hox (all paralog groups 1-13), Fox (foxa1/2/3, foxd1/3/5, foxg1, foxi1-3, foxp1/2/4), Sox (sox2/3/5/9/10/11/14/21), Pax (pax2/3/6/7), Nkx (nkx1/2/4/6), Lhx, Dlx, Barhl, Irx, Six and Tbx families, plus noto, shha/b.  
   • Neuronal specification genes and synaptic components (neurod1/2/4/6, elavl4, glutamate- and GABA-receptor subunits, unc13b, ryr2, synaptic adhesion genes cadm, lrrtm, kirrel, neurexins, etc.).  
   • Axial mesoderm / notochord markers (fsta, col9a2, fras1, frmpd1, col28a1, adgrl3, col15a1).  
   • Motility / cytoskeleton genes typical of nascent myotome and somite boundary cells (actn2b, mybpc2a, myo15, pln, tnmd).

   These assignments argue that the cluster captures regulatory DNA used by both neuroectodermal (hind-/mid-brain, spinal cord, floor plate) and adjacent axial mesodermal (notochord, early somite) lineages.

3. Transcription-factor motif enrichment  
   Top positive z-scores (> 2) correspond to:  
   • SOX consensus (SOX2/3/10/11/14/21) – consistent with early neural and floor-plate identity.  
   • HOX / PBX-MEIS composite motifs – hallmark of rostro-caudal patterning.  
   • FOX (FOXA/FOXD/FOXF/FOXP) sites – typical of definitive floor plate/notochord enhancers.  
   • LHX / DLX / IRX / BARHL / NKX and PAX families – regulators of hindbrain and spinal cord domains.  
   • MEF2 and TBX motifs at lower, but positive, scores – reflecting emerging paraxial muscle and somite activity.  
   • Nuclear-receptor-type half sites (RAR/RXR, NR2F, ROR) – dovetailing with RA- and Notch-responsive patterning enhancers.  

4. Biological interpretation  
   Cluster 3 therefore annotates an “Axial patterning / neuro-mesodermal enhancer module”.  
   • It first comes online in neuromesodermal progenitors, but its accessibility sharply escalates as these progenitors segregate into notochord, floor plate, hindbrain, spinal cord and forming somites.  
   • The motif landscape indicates that SOX and FOX factors initiate opening of these elements, which are then cooperatively occupied by region-specific HOX–PBX–MEIS complexes and lineage-specific TFs (PAX, DLX, LHX, NKX, BARHL, IRX).  
   • Functionally, the peaks likely integrate RA, Shh and Wnt gradients (supported by RAR/RXR, ROR, TCF motifs and the presence of shha/b, wnt1/3/4/7a genes) to drive axial patterning genes, neurogenesis genes and structural genes of the nascent somite.

In summary, peak-cluster 3 marks enhancers that orchestrate the step-wise transition from neuromesodermal progenitors to patterned neural tube, notochord and early muscle, making it a key regulatory module for midline and posterior axis formation during zebrafish embryogenesis.
]

## Cluster 4

[*assistant*:
Peak cluster 4 represents a posterior/tail-bud enhancer module.

1. Where and when is it open?  
   • Accessibility is moderate (≈1.4–2.0 a.u.) in most cell types at all stages, but it explodes in tail-bud cells at the 15-somite stage (43 a.u.; 18 hpf).  
   • Smaller spikes are seen in tail-bud cells at 5- and 20-somites, consistent with waves of posterior axis extension.  
   • The profile therefore fits regulatory elements that are already “primed” genome-wide but become maximally active in the posterior growth zone during the burst of somitogenesis.

2. Genes linked to the peaks  
   The peak–gene map is dominated by posterior Hox clusters and other axial patterning genes:  
   – Posterior Hox genes (hoxa9-13, hoxb9-13, hoxc9-13, hoxd9-13), Cdx1b/4, Meox1/2, Uncx, Mnx1/2, Lbx1/2, Barhl, Dlx family, Tbx15/18/20, Pbx/Meis cofactors, Nkx1/2/3/6, Gbx1/2, Six, Irx, etc.  
   – Additional factors required for posterior neuro-mesodermal, paraxial mesoderm and hindgut/endoderm specification.  
   This gene composition is the canonical signature of tail-bud axial progenitors and the tissues they generate.

3. Transcription-factor motif enrichment  
   Motifs with the highest positive z–scores correspond to:  
   • HOX / PBX / MEIS homeobox class (posterior Hox code).  
   • NKX, DLX, BARX, LHX, TLX, SIX and other homeodomain proteins that partner with HOX proteins in trunk/tail patterning.  
   • FOX and SOX family motifs (typical pioneer/axis-extension factors).  
   • E2F and cell-cycle–related motifs, reflecting the high proliferative nature of tail-bud progenitors.  

4. Biological interpretation  
   – Cluster 4 contains cis-regulatory elements that orchestrate the posterior Hox program and coordinate multi-germ-layer patterning as the embryo elongates.  
   – Elements are “pre-set” in many lineages but reach full accessibility only in the tail-bud at mid-somitogenesis, when rapid axis extension and segmentation occur.  
   – After this burst the elements remain accessible, albeit at lower levels, in derivatives such as posterior spinal cord, paraxial mesoderm/somites, endoderm and epidermis, consistent with maintenance of regional identity.

In summary, peak cluster 4 marks the posterior axial patterning/hind-body program, peaking in tail-bud progenitors around the 15-somite stage and supporting Hox-driven specification of posterior neural, mesodermal and endodermal tissues.
]

## Cluster 5

[*assistant*:
Peak-cluster 5 corresponds to the “surface / epithelial programme” that is switched on very early and is most active between the 0- and 20-somite stages.

1. Chromatin-accessibility pattern  
   • The strongest ATAC signal by far is found in epidermis pseudobulks (≈ 37–39 a.u. at 0–15 somites; still high, but dropping, by 30 somites).  
   • Moderate but clearly above-background accessibility is also detected in other superficial or epithelial–like derivatives – hatching-gland, endocrine pancreas, enteric neurons, early endoderm – and in the notochord tail-bud region; most of these follow the same “high-early / declining-late” trajectory.  
   • Mesodermal (somites, muscle) and neural tube clusters show only baseline values, indicating that the signature is largely restricted to outer epithelial lineages.

2. Gene content of the cluster (selected highlights)  
   The >2,000 linked genes are heavily enriched for classical epidermal / barrier components and their regulators:  
   • Structural & barrier genes – krt17, krt4, krt99, krt18a, cldn1/7, oclna/b, lama1/3/5, lamb1a, lamc1, col7a1, abca12, abca4a/b, mmp9, tnc.  
   • Adhesion / polarity modules – cdh1, cdh4, epcam, itga6b, itgb4, grhl1/2/3, tfap2a/c/d, gata3, lef1.  
   • Epidermal signalling and differentiation regulators – tp63, tfcp2, grhl family, klf4/5/6/9/17, gata2/3/6, nr family (rarga/b, rxr, ppar, ror), fos/jun (AP-1), bach/bhlhe and clock genes.  
   Together these genes define periderm, peripharyngeal ectoderm and other external epithelium in zebrafish embryos.

3. Transcription-factor motif enrichment  
   Highest z-scores (≥ 3) point to three motif groups that are key for non-neural ectoderm:  
   • Nuclear-receptor response elements (M03072, M09755) – GR/PR/AR, RAR/RXR, ESR, PPAR, ROR – consistent with steroid/retinoid control of epidermal maturation.  
   • bHLH / TCF family motifs (M08175) – TCF/LEF-type Wnt responders and ATOH/NEUROD-like factors that cooperate with AP-1 in periderm.  
   • Additional strong but slightly lower motifs include SOX (M08130), GRHL/TFAP2/TP63 (M05137, M04174), and AP-1 (FOS/JUN; captured in M09430/31).  

4. Biological interpretation  
Peak-cluster 5 therefore marks the chromatin module used by the superficial epithelium (periderm/epidermis) as it forms and spreads over the embryo during gastrulation and early segmentation. The module turns on barrier, adhesion and ECM genes, is driven by TP63/TFAP2/GRHL together with hormone-responsive nuclear receptors, and gradually shuts down after ~20 somites as the epidermis matures. Secondary accessibility in hatching-gland, endocrine pancreas and enteric neurons reflects reuse of generic epithelial programmes in those tissues.

In short, cluster 5 captures the early epidermal/periderm regulatory landscape of zebrafish embryos.
]

## Cluster 6

[*assistant*:
Peak cluster 6 captures a coherent set of cis-regulatory elements that switch on as paraxial-mesoderm progenitors commit to skeletal- and cardiac-muscle fates.

1. Accessibility pattern  
• Moderate signal in neuromesodermal progenitors (NMP) and presomitic mesoderm (PSM) at 0–10 somites, then a sharp rise in somite-derived lineages.  
• Very high accessibility in fast-muscle, “muscle” and myocardium libraries at 20–30 somites (e.g. muscle_30 somites ≈ 46, heart_myocardium_30 somites ≈ 22).  
• Low or absent signal in unrelated ectodermal derivatives (epidermis_30 somites ≈ 0) and early fast-muscle precursors (fast_muscle_0 somites ≈ 0).  
→  The trajectory suggests that these peaks open gradually during myogenic specification and remain maximally accessible during terminal differentiation of trunk muscle and heart.

2. Gene associations  
Peaks are linked to classic muscle/cardiac structural and regulatory genes, including:  
• Sarcomere/contractile: ACTC1, TTN, TNNC/TNNI/TNNT family, MYOG, MYO18, MYOZ2, TNRI2, FHOD3, SMYD1, SMYHC2.  
• Excitation–contraction coupling: RYR1, RYR2, SLC8A3, CACNA1/2/δ subunits.  
• Signalling & ECM remodelers important for myofiber maturation: LAMA1/5, FN1, COL4/5/12, TGFB2, FGFR1, PTPRZ1, PLXNA, ITGB8.  
• Transcriptional regulators seen in muscle development: MEF2A/C-like, SRF, TBX1/15/16/18/20, TEAD1, RUNX2, HAND2, FOXO3/4 and E2F family members (reflecting the cell-cycle exit that accompanies differentiation).

3. Motif enrichment  
Top positive z-scores correspond to motifs recognised by:  
• MEF2 family and SRF/MRTF – master activators of skeletal and cardiac muscle genes.  
• TEAD/MCAT motifs (Hippo pathway) and TBX factors – known modulators of cardiomyocyte gene programs.  
• E-proteins/E2F – consistent with the transcriptional switch from proliferation to differentiation.  
• FOXO/FOXM1 and KDM2-CXXC motifs – factors that coordinate metabolic rewiring and chromatin remodeling during myogenesis.

4. Biological interpretation  
Taken together, cluster 6 represents a “myogenic/cardiogenic enhancer module.”  
These peaks mark enhancers that:  
• Begin to open while mesodermal progenitors are still multipotent,  
• Become strongly active in differentiating somites, skeletal muscle fibres and embryonic myocardium,  
• Drive a gene network required for sarcomere assembly, contractility and metabolic maturation of muscle cells.

In short, peak cluster 6 delineates the regulatory landscape that underpins the transition from paraxial mesoderm to fully differentiated skeletal and cardiac muscle between 14 hpf and 24 hpf in the zebrafish embryo.
]

## Cluster 7

[*assistant*:
Peak cluster 7 represents the “baseline/house-keeping” chromatin compartment of the zebrafish embryo.  

1. Accessibility pattern  
• The pseudobulk profile is uniformly high (z-scores ~ 1.6–2.3) in virtually every cell type at every somite stage (10–30 hpf).  
• No lineage-specific spikes or stage-restricted dips are seen, indicating that these peaks stay open irrespective of differentiation state.

2. Genomic content & gene associations  
• >10 000 genes are linked to the cluster (essentially a catalogue of the expressed zebrafish genome, including metabolic enzymes, cytoskeleton, ribosomal proteins, cell-cycle regulators, etc.).  
• GO terms (not shown but evident from the gene list) cover translation, RNA processing, core metabolism, DNA repair and cell cycle—classical house-keeping functions.  
• Many peaks map to promoter-proximal CpG-rich regions or well-known ubiquitous enhancers.

3. Transcription-factor motif landscape  
• Motif enrichment is broad and shallow rather than sharp and lineage-specific.  The over-represented motifs belong to:  
  – General promoter binders / CpG‐island factors: Sp/KLF family, NRF-like, YY1, CTCF.  
  – Cell-cycle/ubiquitous regulators: E2F, MCM-associated, MYC/MAX.  
  – Multi-purpose pioneer factors: FOX, SOX, GATA, HOX, TEAD, AP-1.  
  – Nuclear receptors that act broadly (RXR, RAR, NR2F/COUP, PPAR).  
• The absence of a single dominant lineage TF and the presence of many basal factors fits a constitutively open, promoter-heavy peak set.

4. Biological interpretation  
• Cluster 7 marks genomic elements that need to stay accessible in every nucleus to sustain core cellular functions.  
• Its stability across gastrulation-to-organogenesis suggests that large-scale chromatin re-configuration during lineage commitment occurs on top of, not instead of, this constitutive layer.  
• The cluster therefore provides a convenient internal reference for assessing global chromatin accessibility changes in other, lineage-restricted peak clusters.

In short, peak cluster 7 is the ubiquitously open, house-keeping promoter/enhancer compartment that underlies basic cellular physiology throughout zebrafish embryogenesis.
]

## Cluster 8

[*assistant*:
Peak cluster 8 marks a set of regulatory regions that become selectively open in anterior neuro-ectoderm derivatives during the 10- to 24-hpf window.

1. Cell-type accessibility pattern  
   • Moderate baseline accessibility (~10) in most germ-layer derivatives, but a pronounced spike in  
     – Midbrain-hindbrain boundary (17-21, all somite stages)  
     – Optic cup / eye field (16-19)  
     – Neural crest and neural posterior cells show a later, modest rise (12-13 by 20–30 somites).  
   • Mesodermal and endodermal pseudobulks remain at baseline, indicating the cluster is largely neuro-restricted.

2. Gene set signature (correlated or overlapping with the peaks)  
   • Classic MHB patterning genes: fgf8a, pax2a/5, gbx1/2, en2a/b, otx2a, wnt4b/5b.  
   • Eye-field / optic cup regulators: rx1/2/3, six3/6, lhx2/9, vsx1/2, crx, nrl, pax6a/b.  
   • Axon-guidance / boundary‐formation genes: slit2/3, robo1-3, eph/ephrin family, sema3a-f.  
   • Extracellular-matrix and basement-membrane components (laminins, collagens, fn1) consistent with tissue-folding at these boundaries.

3. Motif enrichment  
   Highest z-scores (≥ 2) correspond to motifs recognized by  
   • Pax family (Pax2/5/6)  
   • Homeobox / Engrailed & Hox-like factors  
   • Forkhead (Fox) factors  
   • Sox family (Sox2/3/5/9)  
   • Zic family  
   • MEF2 and CREB/ATF–Bach factors  
   collectively typical of anterior neural plate, MHB and retinal enhancers.

4. Developmental interpretation  
   The cluster therefore represents enhancers that orchestrate anterior neural plate regionalization—especially MHB organizer and eye-field specification—and remain active through early neurogenesis.  Their accessibility profile suggests a role in maintaining boundary identity rather than in late neuronal differentiation.

In short, peak cluster 8 captures the regulatory program of the midbrain-hindbrain boundary and optic/forebrain territories, featuring Pax/Fox/Sox/Zic-driven enhancers linked to hallmark patterning genes that are maximally accessible from gastrulation through early organogenesis.
]

## Cluster 9

[*assistant*:
Peak cluster 9 – succinct interpretation

1. Chromatin accessibility profile  
   • Broad, “house-keeping”–level accessibility (~3–4 log-CPM) is seen in almost every lineage, but two cell-type/time-point combinations stand out:  
     – Primordial germ cells at 15 somites (≈16 hpf) show an extreme spike (≈60 log-CPM), with smaller peaks at 5 and 20 somites.  
     – Enteric neurons at 30 somites (24 hpf) show a secondary but clear elevation (≈11 log-CPM).  
   • Epidermis, somitic muscle and general mesoderm are low or absent, arguing that the signal is not ubiquitous but centred on germ-line and late visceral-neural lineages.

2. Gene content linked to the peaks  
   • Germ-cell / pluripotency markers: piwil2, kitlga, prdm1-like (prrx1), dap1, lin28a/b.  
   • Enteric / visceral neuron genes: phox2b, ret-signalling partners (grik2, gabbr2, grin2aa), runx1/3, rfx6.  
   • Numerous ion-channel, synaptic-vesicle and axon-guidance genes (gria/grik, cacna/cacnb, syt2, robo1, sema4f, plxna3) reflecting neuronal specification.  
   • Cell-cycle/replication factors (e2f family targets, mcm/rfc) consistent with the proliferative burst of PGCs.

3. Transcription-factor motif enrichment (highest z-scores)  
   • M04021 → Zic / Pax6 class – neural tube and germ-cell regulators.  
   • M05588 → TCF/LEF family – Wnt–responsive enhancers active in migrating PGCs and enteric neural crest.  
   • M10138 → E2F/DP – supportive of the high proliferative state of 15-somite PGCs.  
   • M04785 → Sox family (Sox3/10/19) – multi-lineage progenitor and neural crest factors.  
   • M05416 → Fox (FOXO/FOXL2/FOXK) – reported in both germ-line maintenance and neural crest derivatives.

4. Biological interpretation  
Peak cluster 9 captures a set of cis-regulatory elements that become highly accessible during the mid-somite window in primordial germ cells, coinciding with their rapid proliferation and epigenetic reprogramming, and that remain open (to a lesser extent) in differentiating enteric neurons derived from vagal neural crest.  The motif landscape (Zic/Pax6, Sox, Fox, TCF and E2F) and the associated gene roster together suggest:

• A germ-line transcriptional programme (Piwi-Kitlga-Lin28 axis) under Wnt/TCF and FOX control.  
• A later utilisation of some of the same elements in visceral-neuron differentiation, driven by Zic/Sox and Pax6 motifs.  

Thus, cluster 9 represents a “germ-line / visceral-neurogenic” regulome that is transiently and very strongly activated in PGCs at ~16 hpf and reused, at lower amplitude, by the developing enteric nervous system later in somitogenesis.
]

## Cluster 10

[*assistant*:
Peak cluster 10 marks a “muscle-lineage” chromatin programme.

1. Where and when the peaks are open  
• Accessibility is already appreciable in presomitic mesoderm (PSM; 12–16 a.u.) and rises steeply in forming somites and especially in fast-muscle nuclei (≈23 a.u. at 0 som, climbing to ≈38 a.u. by 30 som).  
• Other lineages keep the peaks relatively closed (≈4-7 a.u. in neural, endoderm, epidermis, etc.).  
→ The profile fits the paraxial-mesoderm → myotome → differentiated fast-skeletal-muscle trajectory between 14-24 hpf.

2. Genes linked to the peaks (representative subset from the long list)  
Contractile/Sarcomeric: acta1a/b, actc1a/b, tnni1/2, tnnt1/2, tpm2/3, myh7b/ myhz2, myl1/ mylpf, ryr1a/ ryr3, cacna1c/-g, casq1/2.  
Myogenic regulators: myod1, myf5, myog, mef2aa/b/ca/d, srfa/b, tbx16/18, pax3a/7a/7b, prdm10, trim54/55.  
Muscle-ECM and attachment: lama2, col12a1, fhl1/3, sgca/cd/cg, itm2, rash1.  
Metabolic & Ca2+ handling: atp2a1/2a1l, atp2b2/4, slc8a1b/8a4, ppargc1a/b.  
The gene catalogue therefore encodes the structural, metabolic and regulatory hall-marks of skeletal-muscle differentiation.

3. Transcription-factor motif enrichment (highest z-scores)  
• M04174 (SRF) and M04665 (MYO/MRF consensus) – core drivers of sarcomeric and actin gene activation.  
• M02741 / M02428 etc. – MyoD / Myf5 / Myogenin motifs.  
• Row 10 motif set – MEF2 family (mef2aa, mef2cb, mef2d) cooperating with MRFs.  
• Row 41/42 motif sets – TCF/LEF, Myf, Neuro-mesoderm TFs; Row 62 – Tbx factors (tbx16/18).  
• Additional positive scores for TEAD/VGLL, FOXO/KLF and AP-1 (M04174, M06476, M06453) that modulate muscle maturation and stress-responsive genes.  
These enrichments match the canonical myogenic transcriptional hierarchy: Pax3/7 → MyoD/Myf5 → Myogenin/MEF2 (+ SRF/TEAD) that drives zebrafish fast-fibre formation.

4. Biological interpretation  
Cluster 10 represents a set of enhancers and promoters that switch on as paraxial mesoderm segments into somites and stay highly accessible in differentiating fast skeletal muscle. The elements orchestrate:
• commitment (Pax3/7, Tbx, Fox),  
• execution of the myogenic programme (MRFs + MEF2 + SRF),  
• assembly of contractile apparatus and excitation–contraction coupling,  
• metabolic and structural maturation of fast fibres.

In short, peak cluster 10 can be annotated as “fast skeletal-muscle / myotome regulatory module, activated from presomitic mesoderm and peaking during somite maturation (14–24 hpf).”
]

## Cluster 11

[*assistant*:
Peak Cluster 11 – concise biological interpretation

1. Chromatin-accessibility pattern  
• The pseudo-bulk signal is high (≈40–55 arbitrary units) and remarkably flat across virtually every cell type and time-point that we profiled, from 0- to 30-somite stages.  
• Modest peaks (upper 40s to low 50s) are seen in highly proliferative mesodermal derivatives (tail-bud, early somites, nascent muscle, heart) but the difference from other tissues is small.  
⇒  Cluster 11 contains regions that are constitutively open rather than lineage-restricted enhancers.

2. Genes linked to the peaks  
• >3,000 genes, spanning housekeeping genes (actb1, rpl/rps, gapdh-like, atp-ase subunits), basal metabolic enzymes, and a wide array of developmental regulators (Hox, Sox, Fox, Pax, Nkx, Tbx, Notch pathway components, etc.).  
• The breadth and diversity reinforce that the peaks sit at promoter or first-intron regions that drive ubiquitous or broadly expressed genes; many developmental TFs are already transcribed at low levels in most early embryonic cells.

3. Motif / TF enrichment signature  
Top positive z-scores (>2) in the motif table correspond to:  
• E2F/DP family (cell-cycle–linked basal promoters)  
• KLF/Sp1-like GC-rich promoter motifs  
• MEF2 & SRF family (generic myogenic/actin-cytoskeleton promoters, but also broadly used for growth-factor–responsive genes)  
• CXXC/KDM2B (CpG-rich promoter chromatin)  
• Nuclear receptors of the NR2F/COUP-TF, RAR/RXR and ESRR/ROR classes – factors that bind consensus DR1/DR2 repeats prevalent at many active promoters.  
The motif spectrum is characteristic of promoter-proximal accessible chromatin rather than distal, cell-type-specific enhancers.

4. Biological role and annotation  
Cluster 11 most parsimoniously represents the “housekeeping/core-promoter” module of the zebrafish embryo: CpG-rich promoters, first exons and immediate intronic regions that stay open in nearly all cells from 10 to 24 hpf.  They harbour motifs for generic transcription machinery (E2F, KLF/Sp1, NF-Y-like, CXXC) and for broadly expressed nuclear receptors that modulate metabolic or hormonal responses.

Recommended label: “Constitutively active promoters / ubiquitous open chromatin (housekeeping module)”.
]

## Cluster 12

[*assistant*:
Peak-cluster 12 represents a large collection of cis-regulatory regions that switch on during the neural-crest-to-neuron transition and reach their maximal accessibility in the enteric nervous system at the end of the 30-somite stage (≈24 hpf).

1. Accessibility pattern  
   • Broad, intermediate accessibility (log2 ≈ 3) in many early axial/neuronal progenitors (NMP, PSM, generic neural plate) from 0- to 20-somites.  
   • A dramatic spike (log2 ≈ 15.7) specifically in enteric-neurons-30 somites, with still-high values in differentiating neurons, hind-/spinal-cord and neural-crest derived populations.  
   → This suggests that the peaks are first “primed” in multipotent neural progenitors and then fully activated in late vagal-crest-derived enteric neurons.

2. Associated genes (selected highlights)  
   • Canonical enteric-/autonomic-neuron TFs: phox2a, phox2b, tlx3, hand2, nkx2.2, gata2/3, hox-b1/b2/b5, pbx/meis cofactors.  
   • Neurotransmission & synapse: ret (GDNF receptor), gfra1/2/4, erbb4, unc13b, rims2, syt2, neurexins, nrxn2, grin1/2, gria1-4, gabra/b subunits, cacna1/2/β, kcn* and scn* ion-channel families.  
   • Axon guidance / neurite outgrowth: eph/ephrin (epha2, ephb2), robo1/2, sema3/4/5/6, netrins (ntn-, unc5).  
   • Neuro-secretory / visceral targets: ryr2, piezo2, qrfpr, sst, vipr1.  
   Functional enrichment therefore centres on neuronal differentiation, axon guidance, synaptic transmission and visceral nervous-system development.

3. Motif enrichment – main regulators inferred  
   (motifs with z-score > 2 are italicised)  
   • _SOX family_ (Sox10/11/21/32) – neural-crest and neuron specification.  
   • _Forkhead (FOX)_ (FoxD3, FoxG1, FoxJ1/2/3, FoxA/B/F) – maintain neural-crest identity and drive neuronal subtype programmes.  
   • Homeobox clusters:  
       – PHOX2, TLX & LHX/DLX (visceral / autonomic neuron lineage)  
       – PBX/MEIS + HOX paralogs (antero-posterior patterning, vagal crest).  
   • _E2F / cell-cycle_ factors (E2F1/2/3/4/5, TFDP) – consistent with the primed, proliferative state of early progenitors.  
   • _Nuclear receptors_ (NR2F/COUP-TF, RAR/RXR/ROR, NR5A) – often cooperate with PBX/HOX in hind-gut / vagal programmes.  
   • _NFAT / CREB/ATF_ modules – activity-dependent or calcium-responsive remodelling during neuronal maturation.  

4. Biological interpretation  
   Peak-cluster 12 delineates the regulatory program that is:  
   • Established (“poised”) in axial and neural-crest progenitors shortly after gastrulation,  
   • Sharply up-regulated as those vagal neural-crest cells migrate into the gut and terminally differentiate into enteric neurons,  
   • Controlled by a combinatorial code of SOX, Forkhead and visceral homeobox factors together with PBX/MEIS-HOX co-factors and nuclear receptors.

5. Practical annotation  
   Recommended label: “Enteric / visceral neuron differentiation module”.  
   Potential enhancer set to test: high-scoring peaks overlapping RET, PHOX2B, GDNF-GFRA loci and those with strong SOX or PBX/HOX motifs.

In summary, peak-cluster 12 captures the cis-regulatory landscape that drives late vagal-crest-derived enteric-neuron maturation and, more broadly, autonomic neuronal identity during the 20–30-somite window of zebrafish embryogenesis.
]

## Cluster 13

[*assistant*:
Peak cluster 13 captures a set of regulatory elements that are most open in highly proliferative, posterior axial progenitors and in the early derivatives they give rise to.

1. Accessibility pattern  
   • Highest signals in tail-bud and very early somitic samples (tail_bud 0–10 som, somites 30 som) and in posterior neural‐axis derivatives such as notochord, floor-plate, enteric and general neural crest cells.  
   • Moderate but widespread accessibility in many other lineages, tapering as embryos progress toward 24 hpf.  
   → This spatiotemporal profile fits the chromatin state of neuromesodermal progenitors (NMPs) and early trunk-/tail-patterned tissues, which are most abundant in the posterior body at 10–15 somites and progressively diminish.

2. Associated genes  
   • Dense representation of HOX paralog groups 1-13 (hoxb1-hoxc13, hoxa9/11, etc.) together with classic posterior-patterning or axial genes (cyp26b1, meis1/2/3, pbx1-4, uncx, lmx1, lhx, msx, dlx, pax, tlx, barx, gbx, eve1, mixl1).  
   • Numerous cell-cycle/replication genes (E2F targets, mcm family, pcna, etc.) reflecting the high mitotic activity of the tail-bud/NMP compartment.

3. Motif enrichment (z-scores)  
   • Very strong enrichment for E2F family motifs (E2F1/2/3/5/7/8) – the canonical drivers of S-phase and proliferation (z ≈ +3.4 to +4.4).  
   • Robust enrichment for HOX/PBX/MEIS–TFDP (M10024, M06063) composite motifs, as well as Pitx, Lhx/Msx, Pax and Uncx, all of which cooperate in trunk and tail axial patterning.  
   • Modest enrichment for generic pioneer factors (SOX, FOX) but these are secondary to the HOX/E2F signature.

4. Biological interpretation  
   • Cluster 13 likely represents cis-regulatory modules that integrate positional information (HOX–PBX/MEIS code) with cell-cycle control (E2F) in the posterior growth zone.  
   • They are used by neuromesodermal progenitors, tail-bud stem-like cells and their immediate neural and paraxial mesoderm descendants during active elongation and somitogenesis.  
   • Accessibility of these elements drops as differentiation proceeds, consistent with the shutdown of posterior patterning programmes once axial extension ends.

Proposed annotation: “Posterior axial progenitor / tail-bud HOX-E2F regulatory module.”
]

## Cluster 14

[*assistant*:
Peak cluster 14 – summary  

1. Accessibility profile  
• Very low-to-moderate signal in neural, somitic and ectodermal pseudobulks (≈ 4–6).  
• Markedly higher accessibility in all endodermal samples and their derivatives (≈ 12 → 24 from 0- to 30-somite), with a clear time-dependent rise.  
• High in endocrine-pancreas nuclei (≈ 10–15) and appreciable in early myocardium (≈ 10–16), consistent with the mesendoderm origin of the heart tube.  
Interpretation: these peaks open first in nascent definitive endoderm, stay on during gut-tube maturation and are reused in pancreatic and cardiac mesoderm derivatives.

2. Gene content (representative subset out of >1000 linked genes)  
• Endoderm/fore-gut specification: gata4/5/6, sox17/32/18, foxa1/2/3, hnf1ba/b, hnf4a/g, cdx1/4, osr1/2.  
• Pancreatic primordium & intestine: pdx1, onecut1/2/3, otx, mecom, spdef, tal1/2, slc2a2, abcc4.  
• Cardiac or mesendoderm regulators: tbx5a, hand2, nkx2.5, gata4/6, vegfaa.  
• ECM / signalling genes typical of early gut tube (fn1, tnc, col12a1) are abundant.  
These associations reinforce the view that the cluster is an enhancer set driving fore-gut, pancreatic and early cardiac programmes.

3. Motif / TF enrichment  
Highest z-scores correspond to:  
• GATA family motif (row 18) – Gata4/5/6 pioneers of endoderm.  
• Fork-head/FOXA motif set (rows 56/107) – Foxa1/2/3 pioneer factors.  
• SOX motif classes (rows 1, 35, 61) – Sox17/Sox32 for endoderm, Sox9/10 later in pancreas and gut.  
• Nuclear-receptor motifs (rows 26, 31, 32, 55, 57, 77, 78) – Hnf4, Nr1h/ Nr2f, Rxr/Rar family that activate metabolic gut genes.  
• PBX/MEIS and PDX1-like homeobox motifs (rows 22, 28, 52, 88) – characteristic of pancreas specification.  
The combination of pioneer (GATA/FOXA), lineage-defining (SOX17, PDX1) and metabolic nuclear-receptor motifs matches the associated gene list and accessibility pattern.

4. Biological assignment  
Peak cluster 14 most parsimoniously represents an “anterior definitive endoderm / fore-gut enhancer set” that becomes progressively activated from gastrulation through 30-somite stage. These elements lay down chromatin competence under Gata/Foxa pioneers, later accommodate Pdx1, Hnf4 and nuclear-receptor binding to drive differentiation of the endocrine pancreas, liver/gut epithelia and, to a lesser extent, early cardiomyocytes deriving from mesendoderm.
]

## Cluster 15

[*assistant*:
Peak-cluster 15 represents the myogenic branch of the embryo and, more specifically, the fast-skeletal-muscle programme that is switched on after segmentation.

1. Accessibility pattern  
• Peaks are already moderately open in presomitic mesoderm and early somite stages, but their accessibility rises sharply in committed myotome nuclei and reaches a dramatic maximum in “fast-muscle 30-somite” cells (≈24 hpf; log-fold change ≈ +45 vs. the median of other cell types).  
• Background lineages (neural, endodermal, epidermal, etc.) remain at baseline levels, indicating high lineage-selectivity.

2. Associated genes (selected)  
The peak set is linked to a compact ‘muscle-tool-kit’, including:  
• Structural and contractile genes – acta1a, actc1, tnni1/2/4, tnnc2, tnnt1/2/3, myl4, myh6/7, mybpc1/2, tpm1/2, cav3.  
• Myogenesis regulators – myod1, myf5, myog, myf6 (ferd3l), msc, tcf21, mef2aa/ab/ca/cb/d, srfa/b.  
• Metabolic and membrane genes typical of differentiated fibres – pdk4, pparg co-activators, slc4a4, slc9a1, igf-axis components.  
Taken together, the gene list is highly enriched for GO terms such as “contractile fibre”, “sarcomere organisation”, and “muscle tissue development”.

3. Transcription-factor motif enrichment  
The motif screen shows the strongest positive z-scores for:  
• M04021 & M04157 – canonical E-box motifs bound by the Myogenic Regulatory Factors (MYOD1, MYF5, MYOG).  
• M07988 & M02705 – MEF2 family sites and SRF/TCF motifs that cooperate with MRFs during myofibril assembly.  
• Moderate enrichment for NFAT, NF-κB and AP-1 class motifs, which modulate late muscle maturation.  
Sox/E2F motifs are under-represented (negative z-scores), concordant with exit from proliferation and neural/endodermal programmes.

4. Developmental interpretation  
Cluster 15 therefore captures the cis-regulatory module that is progressively activated as paraxial-mesoderm cells leave the cell cycle, differentiate into myotome and assemble mature fast-skeletal sarcomeres.  These peaks likely correspond to enhancers bound by the MRF–MEF2 core complex and drive high expression of structural and metabolic genes essential for contractile muscle function.

In short, peak-cluster 15 demarcates the enhancer landscape of differentiating and fully differentiated fast-skeletal-muscle cells during the gastrula-to-organogenesis transition in zebrafish embryos.
]

## Cluster 16

[*assistant*:
Peak cluster 16 represents a large group of regulatory elements that “turn on” when zebrafish embryos begin to build the posterior neural tube and its early neuronal derivatives.

1. Where and when the peaks are open  
• Accessibility is highest in posteriorly located neural populations at the 0- to 20-somite stages – especially in the “neural posterior” (82 a.u. at 0 somites, > 8 a.u. thereafter) and in spinal cord, hindbrain, differentiating-neuron and NMP (neuromesodermal progenitor) pseudobulks.  
• Accessibility is only baseline in late epidermis and other non-neural lineages, indicating that these peaks are tightly linked to the neural plate/tube rather than to broadly expressed housekeeping regions.

2. Gene programme linked to the peaks  
Correlated/overlapping genes are heavily enriched for:  
• Neural and synaptic components (e.g. CACNA1/2D3, KCN* channels, GRIN/GRIK glutamate-receptor subunits, GABR/GABRR, SYT1/3/9, RIMS, UNC5C, ROBO1/2).  
• Axon path-finding and growth-cone genes (LHX3, PRDM15, TAG1/TENM2, SEMA3/4/5 family, NTN-UNC5).  
• Transcriptional regulators characteristic of posterior CNS patterning (many Hox paralogues, TCF4, E2F family, NR2F2/4A factors, FOX/KDM2B, SOX11/21, PAX3/7).  
• FGF/Wnt signalling modulators (fgf8b, rspo2, wnt6a) that define the posterior neural axis.

Taken together, the gene list reads like a catalogue of spinal cord / hindbrain neural progenitor and young neuronal genes, with a sprinkling of neuromesodermal and tail-bud markers.

3. Motif enrichment – who is likely driving the cluster?  
Top z-scores (|z| > 2) point to:  
• M02022 – TEAD/TFAP (Hippo pathway effectors)  
• M10138 – KDM2B/CXXC (unmethylated CpG–binding, promoter/enhancer priming)  
• M10024 – TCF/LEF (canonical Wnt)  
• Additional support for SOX, E2F, MEF2 and FOX family motifs appears in the associated-factor table.  
These TFs are all known regulators of neural-tube patterning, neural progenitor proliferation (E2F, KDM2B), and early neuronal differentiation (SOX, MEF2, TEAD).

4. Biological interpretation  
Peak cluster 16 corresponds to a posterior-neural–lineage chromatin module that becomes accessible just after gastrulation and remains active through early organogenesis.  It likely contains enhancers for Hox-controlled axial patterning, Wnt/FGF-responsive NMP specification, and the first wave of spinal-cord/brain-stem neurogenesis.  In short, cluster 16 marks the regulatory landscape of posterior neural-tube progenitors and nascent neurons in the 10–24 hpf window.
]

## Cluster 17

[*assistant*:
Peak cluster 17 – concise biological interpretation

1. Cell-type / temporal accessibility pattern  
   • The peaks in cluster 17 are most accessible in neuro‐ectoderm–derived states that give rise to the eye. They are strongly open in “optic cup / neural-optic” and anterior neural tube populations from the 10- to 30-somite stages, and remain accessible (but at lower levels) in related cranial-neural and forebrain derivatives.  
   • Accessibility is low or absent in mesodermal (somite, muscle), epidermal and endodermal pseudobulks, indicating that the programme is largely restricted to the eye-field / early retina lineage.

2. Gene set associated with the peaks  
   The correlated/overlapping-gene list is heavily enriched for classical retinal and eye-field regulators as well as early photoreceptor genes, e.g.  
   – Patterning / progenitor TFs: six3a/b, six6a/b, six7, rx1, rx2, vsx1, vsx2, otx2, crx, pax6a/b, lhx2, foxg1, hmx1, barhl1, vax1/2, tbx2/3.  
   – Early neuronal / photoreceptor structural genes: opn1lw2, opn1sw2, opn4a/b, gnat1, gnb1, rpe65a/b, arr3, rgr, pde6a, gucy2d, rdh10, rdh8, prph2, rs1, tulp1, rom1, rhodopsin-like xr genes, cahz etc.  
   – Eye-field signalling modulators: cers1/2, fzd3/6, wnt7b, shha/hhip, fgf8/19, tcf7l1, dll4/6.  
   – Many additional neuronal differentiation / synaptogenesis genes (snap25, syp, stx family, sv2a/b, rims1/2/3, unc13b, syngap1) consistent with nascent retinal neurons.

3. Transcription-factor motif enrichment  
   • The most over-represented motifs belong to the SIX homeobox family (six1/2/3/4/6/7/9) and closely related sine-oculis-class factors – z-scores > 3 for the top motif.  
   • Secondary enrichments include DMRT (dmrt1/2/3) motifs, which cooperate with SIX factors during retinal neurogenesis, and moderate enrichment for homeodomain eye-field factors (OTX/CRX, PAX, VSX).  
   • No significant enrichment for muscle, mesodermal or endodermal TFs, reinforcing lineage specificity.

4. Biological role inferred  
   Cluster 17 represents a cis-regulatory module that is:  
   • Eye-field-specific – it switches on as gastrulation ends and remains active through optic-cup formation and early retinal neurogenesis.  
   • Controlled by the SIX/DMRT/OTX/CRX network that patterns the anterior neural plate and drives retinal progenitor identity, then photoreceptor fate.  
   • Functionally linked to genes required for optic-cup morphogenesis, retinal progenitor proliferation, and the earliest steps of photoreceptor differentiation and synaptic maturation.

5. Suggested annotation  
   “Eye-field / early retina regulatory programme (SIX-homeobox-driven)”

6. Practical take-home  
   Peaks in cluster 17 are prime candidates for enhancers controlling zebrafish eye-field specification and early retinal neuron development; perturbing the highlighted SIX or DMRT motifs (e.g. CRISPR-deletion or base-editing) in these peaks should specifically impact optic-cup and photoreceptor lineage formation without broadly affecting other germ-layer derivatives.
]

## Cluster 18

[*assistant*:
Peak cluster 18 captures a coherent set of cis-regulatory regions that become highly open in a very specific subset of endoderm-derived secretory epithelia during the 10–24 hpf window.

1. Accessibility pattern  
• The signal is modest and relatively flat (≈4–6 a.u.) in most tissues, but it “spikes” in  
  – Hatching-gland cells (chorionase-secreting EVL derivative): 40–53 a.u. from 0- to 15-somite stages, then falls sharply by 30 somites.  
  – Endocrine-pancreas precursors: 12–16 a.u., peaking at 10-somites and remaining elevated thereafter.  
• More moderate enrichment is seen in bulk endoderm, foregut and floor-plate nuclei, indicating that the cluster is already partially open in multipotent endoderm but is maximally activated when cells commit to secretory gland fates.

2. Gene content  
The ∼2,000 genes linked to these peaks are heavily biased toward endoderm/secretory programs, for example:  
– Classical pancreas / foregut factors: pdx1, isl1/2, rfx6, hnf4a/g, hnf1a/b, foxa3, gata4/5/6, sox17.  
– Hatching-gland and EVL/epidermal protease genes (mmp15b, agr2, tgm2b, etc.).  
– Secretory pathway and ECM components (col7a1, lama2, vit, abca transporters, etc.).  
– Many adhesion and polarity genes (cdh18, itga9, claudins), in line with epithelial specialization.

3. Motif / TF enrichment (highest z-scores)  
M05137 – TFCP2 / TP63 / TP53 family (z ≈ 4.4)  
M10562 – YY1 architectural factor (z ≈ 3.8)  
M09431 – BACH/MAF leucine-zipper factors (z ≈ 3.1)  
M03072 – PBX/MEIS/HOX co-factor module (z ≈ 2.8)  
M10784 – GATA family (gata3/4/5/6) (z ≈ 2.8)  
M08075 – MAF/NRL large-Maf factors (z ≈ 2.5)  
M02929 – PBX/PKNOX/MEIS (z ≈ 2.5)  
These motifs are characteristic of:  
• pioneer and architectural factors that keep large enhancer domains accessible (YY1, TFCP2/TP63);  
• classic endodermal lineage drivers (GATA4/5/6, PBX/MEIS in combination with Pdx1 and Hox paralogues);  
• large-Maf proteins and BACH factors that activate secretory-cell gene batteries in pancreas and other glandular epithelia.

4. Biological interpretation  
Taken together, cluster 18 represents an enhancer module that:  
• is primed broadly in nascent endoderm,  
• is maximally deployed in cells that will secrete large amounts of protein – most strikingly the chorion-lytic hatching gland and the endocrine pancreas primordium,  
• is controlled by a TF ensemble centred on GATA, PBX/MEIS/HOX, large-Maf and architectural YY1/TFCP2 motifs, which are all known regulators of pancreatic foregut and secretory epithelial differentiation.

Therefore, peak cluster 18 can be annotated as:  
“Endoderm-secretory enhancer set (hatching-gland / endocrine-pancreas regulators)” – a regulatory programme that drives the maturation of specialised secretory epithelia emerging from the anterior/ventral endoderm during early zebrafish organogenesis.
]

## Cluster 19

[*assistant*:
Peak-cluster 19 behaves like a “universal­-access” compartment of the zebrafish embryonic chromatin landscape.

1. Accessibility profile  
• The pseudo-bulk values are uniformly high (log-CPM z-scores ≈ 4) for almost every cell type at every analysed somite stage, with only minor attenuation in a few late-stage samples (e.g. NMPs-30 s, epidermis-30 s).  
• Such flat, near-ubiquitous accessibility indicates that these peaks are not lineage-restricted but instead stay open in virtually all nuclei throughout the 10–24 hpf window.

2. Genomic annotation  
• More than ten-thousand genes map to this cluster, spanning essential housekeeping functions (basal metabolism, translation, protein turnover, cytoskeleton, cell cycle) as well as broadly expressed developmental genes.  
• The breadth and heterogeneity of the gene list, together with promoter overlap for many peaks, point to a predominance of core-promoter and proximal enhancer elements rather than specialised, cell-type-specific enhancers.

3. Motif enrichment  
Top positive z-scores (> 2) correspond to motif groups recognised by:  
• SOX family (SOX2/3/5/10/21) – generic pioneer factors that keep promoters accessible in pluripotent and neuro-ectodermal progenitors.  
• Nuclear-receptor half-sites (RXR, RAR, NR2F/COUP, ROR) – ubiquitous ligand- and metabolite-responsive factors.  
• E2F/FOX/KLF-type general transcription factors and clock/ATF CRE-binding proteins.  
These are all widely expressed “maintenance” TFs, consistent with globally open promoters.

4. Biological interpretation  
Peak-cluster 19 therefore represents a constitutive, housekeeping chromatin module: promoter-rich regions that must remain accessible in every cell to support basic transcriptional programs during rapid embryonic growth.  The modest motif bias towards SOX and NR families likely reflects their pioneer activity in keeping these promoters open rather than conferring specific lineage identity.

In short, cluster 19 is best annotated as “ubiquitous/house-keeping promoters and proximal elements – constitutively accessible across all cell types between gastrulation and early organogenesis”.
]

## Cluster 20

[*assistant*:
Peak cluster 20 captures a set of regulatory elements that become maximally open in the nascent myocardium just before the heartbeat begins.

1. Cell-type / temporal accessibility  
   • Accessibility is fairly uniform (∼4.5-5 a.u.) across most pseudobulks, but it spikes in a few populations:  
     – Heart myocardium at 30-somite stage (24 hpf): 64.3 (>> all other values).  
     – Optic-cup nuclei at 0- and 10-somite stages (∼14–15 a.u.).  
     – Mature trunk/somitic muscle at 30 somites (∼6 a.u.).  
   The extreme enrichment in cardiomyocytes at 24 hpf identifies these peaks as heart-specific elements that switch on late in somitogenesis.

2. Associated genes (selected themes from the long list)  
   • Sarcomere / contractile fibre: actc1c, myh7, myh14, ttn, tnni1b / 2b, myl2a, mybpc3, tropomodulin (tmod2).  
   • Cardiac ion-handling and signalling: cacna1aa/ba, cacna2d1, ryr3, itpr1a/b, pln, plcb1, pkd1a.  
   • Cardiogenic TFs and cofactors: nkx2.5, tbx4, hand2, gata4/5/6, mef2ca/d, smad2/3, lef1.  
   • Lipid / cholesterol metabolism genes regulated by SREBF: hmgcrb, fasn, ldlrap1, acaca, acly, fdxr.  
   • Cell-cycle / proliferation genes: e2f family targets, mcm4/6, pcna, kdm2b, lin54.  
   Together these point to differentiating but still proliferative cardiomyocytes that are ramping up structural, electrophysiological and metabolic programmes.

3. Motif enrichment  
   Highest z-scores (> 2)  
   • M05682_2.00 → SREBF1/2 motif (4.0) – lipid-biosynthetic switch.  
   • M04785_2.00 → E2F family motif (2.8) – proliferative cardiomyocytes.  
   • M10138_2.00 → CXXC/KDM2 motif (2.3) – CpG-rich promoters of active metabolic genes.  
   • M02428_2.00 → LIN54/TFDP motif (2.2) – cell-cycle gene promoters.  
   Moderate enrichment for SOX/FOX/HOX motifs is present but below the top tier.

4. Biological interpretation  
   These peaks mark promoters and enhancers that:  
   • turn on strongly when myocardial cells exit gastrulation and begin terminal differentiation (24 hpf),  
   • drive expression of sarcomeric and ion-handling genes needed for contractile function,  
   • simultaneously support high lipid/cholesterol synthesis to expand membranes and T-tubules,  
   • and still retain E2F-driven cell-cycle activity characteristic of proliferative embryonic cardiomyocytes.

5. Recommended annotation  
   “Late-somite cardiomyocyte enhancers/promoters – proliferative & lipid-metabolic myocardium (heart myocardium, 30 somites).”
]

## Cluster 21

[*assistant*:
Peak cluster 21 represents a posterior‐mesoderm / tail-bud regulatory module that is transiently active during early segmentation.

1. Accessibility pattern  
   • Very high chromatin accessibility in neuromesodermal progenitors (NMPs), presomitic mesoderm (PSM) and tail-bud cells at 0–20 somite stages (values 60–72).  
   • Moderate accessibility in newly formed somites and fast-muscle precursors (∼25–33).  
   • Low accessibility in neural, endodermal and most differentiated organ lineages (<15).  
   • A sharp drop by the 30-somite stage, indicating that the elements are switched off once axial elongation slows and tissues mature.

2. Associated genes  
   The peaks are linked to hallmark posterior-axis and early myogenic / somitic genes, including:  
   – Core axial progenitor regulators: tbxta (brachyury), cdx1/4, msgn1, tbx6, tbx16, mespa/mespb, ripply1/2, her3/7, dlc/dld.  
   – Numerous clustered HOX genes spanning HoxA/B/C/D groups (e.g. hoxa4, hoxc6, hoxb7, hoxc9, hoxd10), reflecting graded A-P identity.  
   – Early myogenic factors and structural genes: myod1, myf5, myog, mef2a/c, acta1a, desma, tnni2b, tnnc, cav3.  
   – Wnt, Fgf and Notch pathway components that drive tail elongation (wnt8a, wnt5b, fgf8a/17/24, dlc).

3. Transcription-factor motif enrichment  
   The most over-represented motifs (z-score ≥ 2) correspond to:  
   • LEF/TCF (Wnt effectors) – consistent with strong Wnt signalling in the posterior growth zone.  
   • T-box/Brachyury (TBXT) and TBX6/16 – key drivers of axial progenitor and PSM identity.  
   • HOX homeobox motifs – matching the dense list of HOX genes linked to the peaks.  
   • MEF2 / Myogenic bHLH motifs – tying in with early muscle differentiation signals that emerge as somites form.  
   Together these motifs point to a regulatory network centred on Wnt–Tbx–Hox signalling that establishes posterior mesoderm fate and initiates myogenesis.

4. Biological interpretation  
   Peak cluster 21 demarcates enhancers that are:  
   • Activated in posterior axial progenitors and PSM during body-axis extension (10–20 hpf, 0–20 somites).  
   • Silenced as cells exit the tail-bud programme and differentiate (by ~24 hpf, 30 somites).  
   • Likely integrate Wnt, Fgf and Notch inputs, execute Tbx/Hox transcriptional programmes, and prime the myogenic cascade.

In summary, cluster 21 captures the chromatin signature of the tail-bud/PSM compartment that fuels axis elongation and lays down the earliest muscle and paraxial mesoderm lineages in the zebrafish embryo.
]

## Cluster 22

[*assistant*:
Peak cluster 22 – key points

1. Accessibility pattern  
   • Modest, relatively flat accessibility (log-normalized values ~6–8) in most lineages and stages, but a dramatic spike in the epidermis at the 30-somite/24 hpf stage (≈95 versus ≤9 elsewhere).  
   • Progressive rise in epidermis from 0- to 20-somites (6.8 → 8.7) and then a sudden burst, indicating that these peaks mark a late-stage maturation switch in surface ectoderm/periderm cells rather than a general early ectoderm programme.

2. Gene neighbourhood / correlation  
   • Structural epidermis/periderm genes: keratins (krt17/18a/95/98/222), desmosomal and claudin genes, many collagens (col1a1/2, col5a1/2, col6a1, col7a1, col17a1b), laminins (lama1/2, lamb1/2, lamc1/2) and ECM-remodelling enzymes (mmp11a, mmp14a/b, timp4).  
   • Cell-adhesion and signalling genes that characterise stratifying skin/periderm: itga4/6/9, itgb4/6, fn1a/b, paxillin, agrn, sema3/4, wnt3/6b/9a, fgf10a.  
   • Transcription-factor loci pivotal for epidermal identity: tp63, tfap2b/c, grhl1/2/3, irf6, klf4/6/9, gata3/6, sox9a/b, zic1/3/5, dlx3/4, and the Hippo pathway effector yap1.

3. Motif enrichment (z-score table)  
   • Strong enrichment for AP-1 (FOS/JUN), KLF, GRHL, TFAP2/TP63, PBX/MEIS/TGIF and NRF (NFE2L) motifs – all known drivers of periderm/epidermis differentiation, wound response and oxidative-stress protection.  
   • Additional over-representation of NR/RARE motifs (RAR/RXR/NR2F) and TEAD/YAP-responsive MCAT motifs, consistent with retinoid and Hippo signalling inputs during late epidermal morphogenesis.

4. Biological interpretation  
   • Cluster 22 comprises enhancer/promoter elements that switch on specifically in the maturing periderm/outer epidermis as embryos progress into early organogenesis.  
   • The associated regulatory programme controls genes needed for epithelial barrier formation, basement-membrane assembly, ECM deposition and cell–cell adhesion – hallmarks of functional skin.  
   • Motif content indicates integration of –  
     - Retinoic-acid / NR2F-mediated patterning signals,  
     - Hippo-TEAD/YAP control of epithelial growth,  
     - AP-1/Nrf oxidative-stress and wound-healing responses,  
     - KLF/GRHL/TFAP2/TP63 core epidermal transcriptional circuitry.

5. Suggested annotation  
   Peak cluster 22 = “Late-segmentation epidermis/periderm maturation module (24 hpf)”

In practice, these peaks are good candidates for:  
• Super-enhancers around keratin gene clusters on chr19/chr22;  
• Periderm-specific enhancers that could be used to drive skin-restricted reporters in transgenic fish;  
• Regulatory elements controlled by retinoid, Hippo-YAP and oxidative-stress pathways during barrier formation.
]

## Cluster 23

[*assistant*:
Peak-cluster 23 represents a set of chromatin regions that are “on” almost everywhere and almost all the time.  

1. Accessibility profile  
• Pseudo-bulk accessibility values are high (≈300-400 normalized counts) and remarkably flat across all 120 cell-type × time-point samples, from 0-somite neuro-mesodermal progenitors to 30-somite differentiated tissues.  
• This uniform pattern indicates constitutively open chromatin rather than lineage-restricted or stage-restricted elements.

2. Linked genes  
• >2,000 linked genes, dominated by housekeeping and cell-cycle genes (e.g. actb1/2, eef1a/b, rps/rpl family, mki67, ccnb1, cdk6, pcna), general transcription/chromatin regulators (ep300, smarcc1, arid1a/b, kdm2/5/6, uhrf1, dot1l, ezh2), and core signalling adaptors (mapk, akt, gsk3).  
• Functional enrichment therefore points to basic metabolism, RNA/protein synthesis, and proliferation rather than tissue-specific programmes.

3. Motif enrichment  
(top positive z-scores)  
• E2F / cell-cycle (row 6) – drivers of S-phase gene expression.  
• NF-Y / CCAAT and SP/KLF family (rows 83, 86, 105) – classic promoter motifs.  
• ETS (rows 49, 90) – ELK/ETS/ERG/Fli1 sites found at many ubiquitous promoters.  
• AP-1 / BACH/BATF (row 79) – generic stress-responsive promoter/enhancer motif.  
• FOX / SOX / POU motifs appear, but with modest scores and likely reflect broad usage rather than lineage bias.  
Collectively these motifs are characteristic of constitutively open promoters and of regulatory elements used by rapidly dividing embryonic cells.

4. Biological interpretation  
Cluster 23 corresponds to constitutive promoter-like and proximal enhancer regions that support general transcription and cell-cycle progression in virtually all embryonic cells.  Its hallmarks—stable accessibility, housekeeping-gene links, and enrichment for E2F, NF-Y, SP/KLF, ETS and AP-1 motifs—define it as a “core/house-keeping & proliferative” regulatory module rather than a lineage-specific programme.
]

## Cluster 24

[*assistant*:
Peak Cluster 24 – “House-keeping / core-promoter module with late cardiomyocyte boost”

1. Accessibility profile  
   • The peaks are open at a fairly uniform level (≈ 3.4-4.3 a.u.) in almost every cell type and at all six time points, indicating constitutive activity.  
   • The only conspicuous spike (≈ 45 a.u.) occurs in heart-myocardium nuclei at the 30-somite (~24 hpf) stage, suggesting that a subset of these sites becomes super-accessible as contractile cardiomyocytes mature.

2. Linked genes  
   • > 5000 genes are associated; the list is dominated by:  
     – Core metabolic / “house-keeping” genes (ribosome, proteasome, cell-cycle, DNA-repair, ubiquitin, cytoskeleton, vesicle traffic, signal transduction).  
     – A secondary enrichment for cardiac / contractile genes (ACTC1/actc1c, MYH6/7, TNNT2, TNNI1/2, RYR2, CACNA2D1, PLN, etc.) that fits the heart-specific accessibility surge.  
   • GO / KEGG themes: “RNA processing”, “basic metabolism”, “muscle contraction”, “Ca²⁺ handling”.

3. Motif and TF enrichment  
   • Highest z-scores belong to motifs bound by the general transcription machinery rather than lineage TFs:  
     – TBP / TBPL2 (TATA-box recognition)  
     – POU2F1 (Oct-1 family, ubiquitous promoter factor)  
     – LIN54 (MuvB core of the DREAM complex)  
     – SSRP1 (FACT complex) and TOX4 (PTW/PP1 complex)  
   • Virtually no enrichment for classic lineage-restricted factors (e.g. GATA, NKX, Tbx), consistent with a promoter-like, omni-present chromatin state.

4. Biological interpretation  
   • Cluster 24 represents a large collection of promoter-proximal or first-intron elements that remain constitutively accessible to sustain basal transcription in virtually all embryonic cells (“house-keeping promoter cluster”).  
   • During the 20-24 hpf transition the same module is further amplified in differentiating cardiomyocytes, dovetailing with the surge in structural and Ca²⁺-handling gene expression required for beating myocardium.  
   • Functionally, these peaks likely act as “always-on” promoters with an additional capacity for quantitative up-modulation in tissues with exceptionally high transcriptional demand, exemplified here by the embryonic heart.

Concise label: “Ubiquitous core-promoter peaks, up-regulated in late myocardium.”
]

## Cluster 25

[*assistant*:
Peak-cluster 25 corresponds to the “haemato-vascular module” that switches on during early somitogenesis (10–24 hpf).

1. Where and when the peaks are open  
   • Accessibility is already above background in neuromesodermal progenitors (NMP, 0 – 5 som) but rises steeply as soon as the first endothelial / blood progenitors appear.  
   • The highest signal is seen in  
     – haemangioblasts (≈ 22–31 arbitrary units, 5 – 30 som) and  
     – the emergent “haematopoietic/vasculature” compartment (≈ 25–42, 10 – 20 som).  
   • Endothelium-related derivatives (heart myocardium, lateral-plate mesoderm, neural floor plate) show intermediate accessibility (7–10), whereas ectodermal, neural or somitic muscle populations remain at baseline (≤ 6).  
   • Hence the peaks mark cis-elements that become active exactly when mesoderm splits into endothelial, blood and cardiac fates.

2. Genes that sit under (or correlate with) these peaks  
   The > 7 000 linked genes are strongly enriched for classical vascular / blood determinants, e.g.  
   – ETS factors: etv2, fli1a/b, erg, elk1/3/4  
   – GATA factors: gata1a/b, gata2a/b, gata6  
   – Haematopoietic regulators: tal1 (scl), lmo2, runx1/3, spi1a/b, lyl1, c-myb  
   – Endothelial receptors & signalling: kdrl (vegfr2), kdr, flt1, flt4, tek/tie2, pecam1, robo4, vegfc, dll4, notch1/4, cxcr4, ephb4, kitl  
   – Cardiac / vessel ECM & cytoskeleton: fn1, collagens (col4a1/2, col5a1, col8a), lama4/5, vinculin, flna, vcl, myh9/h10.  
   Collectively these genes encode every step from haemangioblast specification to endothelial tube formation and first wave definitive haematopoiesis.

3. Transcription-factor motif landscape  
   The motif enrichment profile shows pronounced positive z-scores for ETS, GATA, RUNX, Tal1/LMO and nuclear-receptor (NR2F/COUP-TF, RXR) motifs – the exact cocktail needed for endothelial and blood specification.  Fork-head and SOX motifs are present at lower levels, consistent with their roles in early mesendoderm and endoderm support.

4. Biological interpretation  
   • Cluster 25 aggregates enhancers that are turned on as soon as lateral-plate / posterior mesoderm commits to the haemato-endothelial lineage.  
   • These elements stay highly accessible while primitive and then definitive endothelial and blood progenitors expand (5 – 20 som) and begin to differentiate into myocardium and vasculature.  
   • The same regulatory module is almost silent in neuro-ectoderm and in mature somites, highlighting its lineage specificity.

5. Practical annotation  
   Annotate peak-cluster 25 as:  
   “Haemangioblast / endothelial-haematopoietic enhancer set – active 10-24 hpf; driven by ETS-GATA-RUNX-Tal1 motif combination; linked to kdrl, fli1, etv2, tal1, runx1, gata2, spi1, fn1, col4a1, lama5, etc.”

In short, peak-cluster 25 captures the cis-regulatory logic that builds the first blood and vascular system in the zebrafish embryo.
]

## Cluster 26

[*assistant*:
Peak-cluster 26 represents a set of “house-keeping / nuclear-receptor” regulatory elements  

1. Chromatin-accessibility pattern  
   • The pseudobulk signal is high and remarkably similar in every sampled cell type and at every somite stage (≈ 80–110 arbitrary units).  
   • Such pan-lineage, stage-independent accessibility is typical of constitutively open promoters or first-intron regions that are needed in virtually all cells.

2. Gene associations  
   • >10 000 linked genes include almost every major class of basic-function genes (ribosomal proteins, RNA-processing factors, metabolic enzymes, cytoskeletal components, etc.).  
   • Very few lineage-restricted markers are over-represented, reinforcing the “ubiquitous / core machinery” character of the cluster.

3. Transcription-factor motif enrichment  
   • The only coherent enrichment signal is for half-site DR-type motifs bound by nuclear–hormone-receptor family members: RAR/RXR, NR2F/COUP-TF, ROR, NR4A, PPAR, ESR, VDR, LXR, SF-1 (Nr5a), etc.  
   • These factors are not confined to one germ layer but serve as systemic hormone/retinoid sensors; their binding sites are often found in broadly active promoters.

4. Biological interpretation  
   • Cluster 26 corresponds to constitutively accessible promoters/enhancers that keep the genome ready to respond to systemic nuclear-receptor signalling (retinoic acid, steroids, thyroid hormone, PPAR ligands, vitamin D).  
   • Because these elements are already open, lineage-specific activation is achieved mainly by differential TF occupancy rather than de-novo chromatin opening.

Suggested label for peak-cluster 26  
“Ubiquitous / Nuclear-receptor-responsive promoters & proximal enhancers (house-keeping module)”
]

## Cluster 27

[*assistant*:
Peak cluster 27 in the multi-omic atlas marks a set of regulatory regions that become most accessible at the very end of the segmentation period (≈30-somite stage) and are especially open in (i) fast/ trunk skeletal muscle nuclei and (ii) optic-cup–derived neural epithelia, while remaining only modestly accessible in earlier-stage neural, mesodermal or endodermal precursors.  

1. Accessibility pattern  
   • Highest signals: fast-muscle 30 som, optic-cup 0 som, pharyngeal-arches 5–10 som; moderate signals in differentiating neurons, PSM, tail-bud and lateral-plate mesoderm; low/absent in epidermis and floor-plate.  
   • Temporal trend: gradual rise from the 0–20 som stages and a sharp spike at 30 som, consistent with enhancers that switch on during late somitogenesis / early organogenesis.

2. Gene content (truncated highlights)  
   – Muscle / contractile programme: myog, tnnt3a, tpm4b, stac3, casq1a, matn1.  
   – ECM / migration: tnc, cthrc1a, fbn2a/b, col5a1/3, col11a1, col9a1, efemp1.  
   – Axon-guidance / synapse: grik2, gria2a, grik4/5, grin3b, sorcs3, npas3, gpc5/6, cntnap2b, nrxn2/3, ntrk2a, robo-like (igsf, lrr).  
   – Patterning & signalling receptors: fgfr1a, pdgfrb, epha2/4, ptprk/ga, tgfb2, wnt-pathway auxiliaries (fzd-related).  
   – Transcriptional regulators found in the same cells: mef2aa/cb, foxf1/2, sox9a/b, tcf7l1, hoxa/b cluster members – matching the motif output below.

3. Motif enrichment (z-score > 2)  
   • M07988_2.00 (BATF/BACH/AP-1 family) z ≈ +2.6  
   • M02428_2.00 (LIN54 / MuvB-cell-cycle switch) z ≈ +2.4  
   • M05682_2.00 (TEAD / Hippo) z ≈ +2.4  
   • M04174_2.00 (E2F/DP) z ≈ +2.6  
   • Additional over-represented motifs: MEF2, SOX, FOX, and KDM2B-CXXC DNA sensors (all positive z ≈ 1.3-1.9).  

   Interpretation: AP-1/BATF and TEAD factors are known activators of myogenic and migratory modules, whereas MuvB-LIN54 and E2F motifs point to a “cell-cycle exit” signature that accompanies terminal differentiation of both muscle fibres and neuronal/glial derivatives. Concurrent SOX/FOX/MEF2 motifs indicate direct input from lineage-determining TFs in neural crest, craniofacial mesenchyme and paraxial mesoderm.

4. Biological inference  
   • Cluster 27 represents late-acting enhancers that coordinately drive skeletal-muscle maturation and neural/craniofacial morphogenesis, integrating Hippo-TEAD growth control, AP-1-mediated stress–ECM responses and lineage-specific SOX/FOX/MEF2 transcriptional programmes.  
   • The restricted peak of accessibility at 30 somites suggests these elements are activated once progenitors have committed and are beginning functional differentiation (first muscular contractions, optic-cup lamination, pharyngeal arch condensation).  

In summary, peak cluster 27 captures the “late somitogenesis / early organogenesis” enhancer repertoire that synchronises cell-cycle withdrawal with ECM remodelling, muscle fibre assembly and neuron/axonal maturation, orchestrated by AP-1, TEAD, E2F/MuvB and lineage TFs such as MEF2, SOX and FOX.
]

## Cluster 28

[*assistant*:
Peak cluster 28 – synopsis

1. Chromatin accessibility pattern  
• Background level (~5 log-counts) in most tissues.  
• Strong, stage-dependent enrichment in neuroectodermal derivatives:  
  – Enteric neurons rise sharply from 0- to 30-somite stage (10 → 72),  
  – Differentiating neurons show a parallel but weaker increase (7 → 22).  
• Muscle, mesoderm, endoderm, epidermis, etc. remain at baseline.  
⇒ The peaks switch on as neurons (especially the enteric lineage) emerge and mature during 10–24 hpf.

2. Genes linked to the peaks  
Hundreds of neural‐specific or neural-enriched genes sit nearest to / are co-regulated with these peaks, e.g.  
• Synaptic receptors & channels: GRIN2B, GRIK2/3, GABRB2, CACNA2D1/3, KCNB2, PIEZO2.  
• Axon-guidance / adhesion: NRXN2, LRRTM4, ROBO1, PLXNA/B genes.  
• Neuronal transcription factors: NPAS3, ISL1/2, LHX family, TCF4, ZNF423, many HOX and TALE (MEIS/PBX) genes themselves.  
• Other neuronal/ENS markers: RET, GDNF-pathway components, RAMP1, SLC17A6/7, SNAP25, SYN2, etc.  
This catalogue is consistent with late neuronal differentiation and enteric-neuron identity.

3. Motif enrichment  
The most over-represented motifs are TALE-class homeobox sites:  
• MEIS1/2/3, PBX/PKNOX (Pbx1/2/3; Pknox1/2)  
• TGIF1 and a lesser TP53 signal.  
TALE factors act as obligatory HOX cofactors and are well-known regulators of hind-brain, spinal cord and enteric-neuron development, fitting the accessibility pattern and gene list.

4. Biological interpretation  
Peak cluster 28 comprises distal regulatory elements that become progressively open in neuronal lineages, with maximal activity in the developing enteric nervous system. Their sequence composition indicates control by MEIS/PBX (TALE) transcription-factor complexes, which likely cooperate with HOX proteins to drive regional neuronal identity and maturation.

Suggested annotation  
“MEIS/PBX-controlled neuronal / enteric-neuron enhancer cluster”
]

## Cluster 29

[*assistant*:
Peak-cluster 29 represents a “posterior–neuromesodermal / EMT–mesoderm” regulatory module.

1. Accessibility pattern  
   • Very strong signal in posterior Neuromesodermal Progenitors (NMPs) and in the Tail-bud at late somitic stages (20–30 somites; e.g. NMP_30 som 64 AU; Tail-bud_30 som 71 AU).  
   • Only basal (~4–6 AU) accessibility in most other tissues.  
   → Indicates a programme restricted to the axial-extension zone at the end of somitogenesis (~22–24 hpf).

2. Gene set associated with the peaks  
   • Enriched for classic posterior/mesodermal factors: msgn1, tbx6, tbxta, tbx16, msgn1, wnt3a/5b, fgf8, raldh2 (aldh1a2), cdx1/4, evx1/2, hox clusters (hoxb4–c13, hoxa9–a13), somitic ECM genes (col11/12/14, ncan, tnc, fn1).  
   • Many EMT / migratory regulators: snai1/2/3, zeb1a/b, twist1/2/3, hand2, mesp-a/b, prrx1/2, foxc1/2, foxd3, sox9/10.  
   • Signalling modulators typical of tail-bud: dll3, lfng, notch1, wnt, fgf, tgfb super-family members.

3. Transcription-factor motif enrichment  
   Top z-scores correspond to:  
   • E-box/bHLH “E-proteins” (Tcf3/E2A, Tcf12) → progenitor/mesoderm competence.  
   • Snail (Snai1/2/3) and Zeb1/2 motifs → epithelial-to-mesenchymal transition.  
   • Hand2 / Twist / Mesp motifs → paraxial mesoderm segmentation.  
   • Moderate enrichment of Wnt-response Tcf/Lef and posterior Cdx/Hox motifs (supporting posterior axial identity).

4. Biological interpretation  
   Peak-cluster 29 demarcates chromatin regions that open specifically in the posterior-most NMP/tail-bud compartment as embryos complete somitogenesis.  These regions control genes that:  
   • maintain dual neuro-mesodermal potential,  
   • enable EMT-like behaviour and cell motility needed for continuous axis elongation, and  
   • establish segmental (pre-somitic) and posterior Hox identity.

In short, cluster 29 captures the chromatin programme that endows late tail-bud neuromesodermal progenitors with posterior identity, EMT capacity and paraxial-mesoderm segmentation potential.
]

## Cluster 30

[*assistant*:
Peak cluster 30 contains ~1,500 cis-elements whose accessibility profile is essentially neural:  

1. Accessibility pattern  
• Already open at gastrulation in generic “neural” and “differentiating-neuron” nuclei (log2 CPM ≈ 3).  
• Gradually rises through segmentation and becomes strikingly high in enteric-neuron nuclei, reaching a >10-fold jump at the 30-somite (≈ 24 hpf) stage (31 log-units versus ~3 elsewhere).  
• Remains only basal in non-neural mesodermal or endodermal lineages.  
Thus the cluster tracks late neurogenesis, especially the emergence/maturation of the enteric nervous system.

2. Gene content  
The >1,000 linked genes are heavily enriched for neuronal wiring and excitability factors: synaptic organisers (NRXN2, UNC13B, RIMS2, RAB3GAP1), ion channels (KCNB2, HCN1, RYR2, TRPC3/5, PIEZO2), glutamatergic receptors (GRIK2/3, GRM7, GRID1/2), axon-guidance / adhesion molecules (ROBO1, SEMA4F, TENM2, LRRTM4, CDH18, KIRREL3) and neuro-specific signalling TFs (NPAS3, ZNF521).  These are hallmark genes of post-mitotic, projecting neurons and of vagal/enteric neural-crest derivatives.

3. Motif enrichment  
Highly significant motifs (Z > 2) correspond to LIM-/paired-/PRD homeobox and Grainyhead factors:  
• Lhx/Lmx (LHX2/4/6/9, LMX1A/B, LHX8)  
• Engrailed/En-2, Uncx, Alx, Rx, Prrx, Shox, Phox2a/b (key for autonomic & enteric neurons)  
• Grainyhead family (GRHL1/2) and TFCP2/UBP1  
• HNF1A/B motifs (often co-bound with GRHL in neuronal enhancers)  

These TFs are well documented drivers of hind-brain, spinal-cord and enteric neural-crest specification and of later neuronal subtype maturation.

Summary  
Cluster 30 therefore represents a set of enhancers/promoters that switch on during late segmentation in neurogenic lineages, peaking in the nascent enteric nervous system.  The cis-elements integrate homeobox (LHX/LMX/EN/SHOX/PRRX/PHOX2) and Grainyhead–HNF inputs to activate a broad neuronal effector programme (synapse formation, axon guidance, ion conductance).  In practical terms, peaks in cluster 30 are excellent markers for, and likely regulators of, terminal differentiation and functional maturation of enteric and other hindbrain/spinal neurons around 24 hpf.
]

## Cluster 31

[*assistant*:
Peak cluster 31 represents a retinoic-acid/​nuclear-receptor–responsive regulatory module that becomes especially active late in the segmentation period.

1. Pattern of chromatin accessibility  
   • Accessibility is fairly uniform (log-normalized values ~5–7) in most lineages, but spikes dramatically at a few sites:  
     – NMPs 30-somite stage ≈ 56  
     – Spinal cord 30-somite stage ≈ 68  
     – Pharyngeal arches at the 0-somite stage ≈ 28  
     – Enteric neurons 30-somite stage ≈ 12  
   These maxima occur in posterior neuro-mesodermal and spinal cord derivatives and, earlier, in pharyngeal/cranial mesendoderm—tissues known to experience strong retinoic-acid (RA) signalling.

2. Linked genes  
   The >3 000 associated genes are heavily enriched for:  
   • A-P patterning/RA targets – many Hox paralogs (hoxa2-13, hoxb1-10, hoxc3-13, hoxd3-12), Meis/Pbx cofactors, Cyp26 enzymes, etc.  
   • Neurogenesis and synapse formation – NPAS3/4, TCF4, NRXN2, KIRREL3, UNC5C, ROBO1/3, GRIN/GRIA glutamate-receptor subunits, GABA-receptor genes.  
   • Cell-adhesion/ECM components – multiple collagens (Col4, Col5, Col12, Col21), cadherins (Cdh2/4/7/22/23), integrins and guidance cues (Eph/Efna).  
   • Metabolic / hormone-responsive factors – PPARA/G, ESR1/2, and many RA- and thyroid-hormone-metabolising enzymes.

3. Motif enrichment  
   The most over-represented motifs (z-score > 2) belong almost exclusively to nuclear receptor families:  
   • RAR/RXR (rarga, rargb, rxraa/b)  
   • ROR (rorc, rorcb, rorb)  
   • COUP-TF/NR2F (nr2f1/2/5/6)  
   • PPAR (ppara/b/g)  
   • REV-ERB/NR1D, THR (thra/b) and related orphan NRs (nr6a1, nr5a1/2)  
   This signature is classic for RA, thyroid-hormone and lipid-derived signalling.

4. Biological interpretation  
   • High accessibility in late NMPs and spinal cord suggests these peaks mark enhancers that turn on as the posterior neural tube and adjacent mesoderm acquire their final axial identities.  
   • The strong RA/nuclear-receptor motif bias, together with the breadth of Hox genes and Pbx/Meis cofactors linked to the peaks, indicates that cluster 31 captures cis-regulatory elements directly driven by RA gradients that pattern the trunk and tail.  
   • Concurrent enrichment for neuronal differentiation genes implies that these elements also contribute to maturation of spinal and enteric neurons that emerge at the same stages.

In summary, peak cluster 31 comprises RA/nuclear-receptor–bound enhancers that open predominantly in posterior neuro-mesodermal progenitors and the developing spinal cord at the 30-somite (~24 hpf) stage, orchestrating Hox-mediated axial patterning and the onset of neuronal differentiation programs.
]

## Cluster 32

Error processing cluster 32: OpenAI Response API generate text error: peer closed connection without sending complete message body (incomplete chunked read)

## Cluster 33

[*assistant*:
Peak cluster 33 represents a Pax-driven, broadly active regulatory module that is most accessible in neuro-ectodermal and placodal derivatives between the 10- and 30-somite stages.

1. Accessibility pattern  
• The pseudobulk profile is uniformly high (log 2-scaled values ~22–27) in almost every neural or placode-related population – midbrain-hindbrain boundary (~27), neural crest (~26–27), differentiating neurons (~25–27), optic/otic epithelia and pronephros (kidney) – while mesodermal structures such as notochord show markedly lower signal.  
• The signal remains high from 0- to 30-somite stages, indicating that the underlying elements are opened early and stay accessible throughout the gastrula-to-organogenesis window.

2. Associated genes  
The peak set is linked to many Pax-dependent developmental regulators, including pax2a/2b, pax5, pax6a/6b, pax8, as well as kidney/ear/brain regionalisation genes (e.g. gata3, lhx1, hnf1ba, otx1/2, six1/4/6).  These genes are classical markers of the intermediate neural tube, cranial placodes, pronephros and early endocrine pancreas – tissues in which Pax2/5/8 proteins are essential.

3. Motif / TF enrichment  
Cluster 33 is uniquely enriched for PAX family motifs (indirect list: pax2a, pax2b, pax5, pax6, pax8, pax1), together with a weaker contribution from paired-type homeobox factors.  No other TF class scores comparably, pinpointing Pax binding as the primary driver of the shared accessibility signature.

4. Biological interpretation  
Taken together, the data indicate that peak cluster 33 delineates a set of promoters and enhancers controlled by Pax2/5/8 (and related Pax1/6) that coordinate early patterning of:  
• the midbrain–hindbrain boundary and hindbrain interneurons,  
• cranial placodes (otic/optic),  
• the pronephric/kidney primordium, and  
• parts of the endocrine pancreas.  

Because the peaks are already open at 10 hpf and stay accessible, they likely serve as “primed” elements that keep key Pax target genes in a transcriptionally competent state as cells progress from multipotent neural plate/neural crest to committed sensory, renal and endocrine lineages.
]

## Cluster 34

[*assistant*:
Peak cluster 34 captures a neuro-ectodermal / neuronal regulatory programme that is turned on very early and becomes most prominent during optic-cup formation.

1. Accessibility pattern  
   • Moderate at 0-somite stage in most tissues, but rises sharply in neural derivatives.  
   • Highest signals (9–43 a.u.) are seen in:
     – Optic-cup cells from 5- to 20-somite stages (max ≈ 43 a.u. at 15 somites).  
     – Notochord (peaks at 5–10 somites).  
     – Enteric-neuron and epidermal pseudobulks show consistently elevated values (7–10 a.u.).  
   • Most mesodermal/endodermal lineages stay near background (≈ 5–6 a.u.).  
   → Together this points to elements that open in the neural plate/tube and are further up-regulated in the eye field and selected neuronal sub-types as development proceeds.

2. Gene module linked to the peaks  
   Peaks are linked to >1,300 genes, but the signature is dominated by classical eye-field and pan-neural factors:  
   • Eye / retina: rx1/2/3, vsx1, pax6, vax1/2, six3/6, crx, otx1/2, lhx2, barhl1/2.  
   • Pan-neural & neural-crest: sox2/3/4/5/8/9/10/14/21, dlx2/4/5/6, phox2a/b, lhx1/4/9, mnx1/2, uncx, hoxb1-5.  
   • Axon-guidance / neuronal differentiation genes (cntn2, robo1, cntnap family, npas3, ebf3, neurod genes) and ECM/adhesion genes (col4a, col18a, itga8, lrp1) are also common.  
   This gene set is characteristic of early retinal progenitors, hind-/mid-brain neurons and neural-crest derivatives committing to neuronal fates.

3. Transcription-factor motif enrichment  
   The strongest positive z-scores (> 2) correspond to motifs recognised by:  
   • Sox family (SOX2/3/4/5/8/10/12/14/18/21) – central drivers of neural and retinal identity.  
   • Homeobox factors expressed in the eye field and spinal cord: Barhl, Vax, Rx, Lhx, Dlx, Gbxs, HoxB cluster.  
   • Additional neurally expressed TFs such as Lmx1, Mnx, Tlx and Phox2.  
   The TF-to-motif table for cluster 34 lists virtually the entire early eye-field / neuronal TF repertoire, underscoring a retinal–neurogenic enhancer set.

4. Biological interpretation  
   • Cluster 34 represents enhancers that are first engaged in broad neuro-ectoderm (0 somites), subsequently intensify in neuronal lineages, and peak in the developing optic cup between 10–20 somites (≈ 14–18 hpf).  
   • The same elements remain active, though at lower levels, in other neuronal populations (enteric neurons, spinal cord) and in the notochord during mid-somite stages.  
   • Functionally, they likely control genes required for eye-field specification, retinal progenitor maintenance, early neuronal differentiation and neural-crest-derived neuron formation.

In short, peak cluster 34 marks a set of Sox– and homeobox-driven enhancers that orchestrate early neural and, in particular, retinal development in zebrafish embryos.
]

## Cluster 35

[*assistant*:
Peak cluster 35 represents a set of chromatin regions that behave like “early neuro-epithelial/optic-field enhancers”.

1. Accessibility pattern  
• The peaks are moderately accessible (log2 CPM ≈ 3.5-4) in most neural‐derived cell types, but they reach a very high signal specifically in optic-cup cells between 0- and 20-somite stages (e.g. ~38-fold at 0 som).  
• Accessibility drops sharply in optic-cup nuclei by 30 som and is low in late-mesodermal derivatives (tail-bud, fast muscle, etc.).  
• This temporal profile matches the window when the eye field and anterior neural plate are being specified (10-20 hpf).

2. Cis-regulatory logic  
• Motif enrichment is dominated by Sox family motifs (row 35 in the TF table: Sox2/3/5/6/8/9/10/11/17/19, etc.).  
• Sox factors are pioneer TFs that keep neuro-ectodermal enhancers in an open, poised state; their very strong enrichment suggests that the cluster is under direct Sox control.  
• Secondary motifs for Fox, Pou, and Six/Pax eye-field factors are present but with lower z-scores, indicating cooperative binding.

3. Linked genes  
The >4 000 genes linked to these peaks are heavily enriched for:  
• Eye-field / retinal progenitor regulators: rx1/2/3, vsx1, six3/6, otx1/2, pax6a/b, crx, tbx2/3, rorb, nr2f1/2.  
• Generic neural stem-cell factors: sox2, sox3, lin28a/b, zic family, foxg1, hedgehog pathway genes.  
• Early neural crest & dorsal neural tube genes (sox10, pax3/7), consistent with broader neuro-ectoderm accessibility.

4. Biological interpretation  
These peaks most likely correspond to early, Sox-controlled enhancers that:  
• Are kept open in the pluripotent neuro-epithelium.  
• Become maximally active in the anterior neural plate/eye field (optic cup) as Sox, Pax6, Rx and Six factors cooperate to drive retinal fate.  
• Gradually close once retinal progenitors differentiate (after ~24 hpf) and as other lineages lose neuro-ectoderm identity.

In summary, peak cluster 35 demarcates Sox-centred, eye-field/neuro-epithelial enhancers that peak in accessibility during specification of the optic cup and early neural progenitors between 0- and 20-somite stages, supporting gene programs for retina and anterior CNS development.
]

