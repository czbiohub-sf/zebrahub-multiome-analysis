project_background = \
"""
## Project Background

We generated a high-resolution, time-series **single-cell multi-omic atlas** of zebrafish embryogenesis, simultaneously profiling chromatin accessibility (scATAC-seq) and gene expression (scRNA-seq) in **> 94 000 nuclei** collected at six developmental stages spanning **10–24 hpf**—from late gastrulation to the onset of organogenesis.
---
### Analysis workflow

1. **Peak catalogue**
   Identified ~650000 high-confidence accessible regions across all stages and cell types in the scATAC-seq data.

2. **Pseudobulk aggregation**
   Aggregated ATAC counts for every *cell-type × time-point* combination, yielding a **peaks-by-pseudobulk matrix** that reduces single-cell noise while retaining biological resolution.

3. **Module discovery**
   Applied the Leiden community-detection algorithm to this matrix to cluster peaks with shared accessibility dynamics.

4. **Biological interpretation**
   Each resulting cluster represents a set of **co-accessible cis-regulatory elements** that open and close together across lineages and developmental time. We treat these clusters as *regulatory programmes* and annotate them by their putative functions and roles in embryonic development.
---

This modular framework distills complex chromatin landscapes into interpretable regulatory programmes, providing a map of the gene-regulatory logic driving zebrafish development.

"""

coarse_cluster_analysis_request = \
f"""
### Analysis Request
Please provide an in-depth analysis of the peak cluster strictly following the provided structure:

## **Cluster Name**: 
Craft a brief, evocative name that captures the cluster’s central biological theme or function.

## **Cluster Label**: 
Supply a short textual label, ideally one, two or three non-abbreviated words, for easy annotation in a figure.

## **Cluster Overview**: 
Summarize in one or two sentences how this cluster’s accessibility and gene associations set it apart.

## **Temporal Dynamics**: 
State the developmental stage(s) or somite counts where accessibility peaks, drawing directly from the pseudobulk matrix.

## **Cell Type Specificity**: 
Point out which cell types exhibit the strongest chromatin accessibility signal, citing evidence from the pseudobulk matrix.

## **Overlapping or Correlated Genes**:
Examine the list of genes associated with this cluster, highlighting specific genes or group of genes that are particularly interesting or relevant to the cluster's function.

## **Motif Analysis**: 
Highlight significantly enriched motifs and explain how these motifs might drive the regulatory program.

## **Key Transcription Factors**: 
List relevant and interesting transcription factors with the highest motif-enrichment z-scores and briefly note their known roles.

## **Regulatory Program**: 
Infer the biological processes or pathways likely governed by this cluster, integrating gene lists, TF functions, and motif data.

## **Biological Interpretation**: 
Write a cohesive, in-depth prose narrative that situates the analysis results and their interpretation within zebrafish embryogenesis, weaving in external literature where relevant.

## **Concise Summary**: 
Distill a concise concise, non-jargon, 'high-level' summary of the overall analysis focussing on synthesizing the biological role of the peak cluster into two or three crisp sentences.

## **References**: 
List again all external sources cited throughout your analysis and discussion, formatted consistently and ready for in-line referencing.

### Instructions for the analysis and its formatting:
* Use the *exact* section titles provided in the template—do **not** rename, reorder, or omit any section.
* Base every conclusion squarely on the supplied data, and accompany each statement with a concise rationale that points to specific data elements.
* If a conclusion is uncertain, state that uncertainty explicitly and explain why the evidence is inconclusive.
* Be careful when interpreting chromatin accessibility for pseudo bulks (cell type x stage) that have very few cells because normalisation may cause them to be outliers.
* Perform focused web searches to validate facts, gather background context, and enrich your analysis; in particular, consult peer-reviewed literature or other reputable sources when drafting the **Biological Interpretation** section.
* Perform focussed database lookups for zebrafish genes and transcription factors to gather additional information on interesting leads and ensure accurate and up-to-date information.
* Cite every external source immediately after the statement it supports, using in-line citations formatted as `[label](URL)`.
* All citations should be relisted in the **References** section at the end, formatted consistently, including author names, publication year, journal or source, and a direct link or PubMed ID.
* Inline citation format should be: `[(Author et al.)](URL)`.
* **References** section references should be formatted as: `[Author(s). Title. Journal/Source. (Year)](URL)`.
* Write the **Biological Interpretation** section as continuous, formal English prose—**no** lists or bullet points—synthesizing all findings into a coherent, didactic discussion that places the results within a broader biological perspective.
* Finish the **Biological Interpretation** section by explicitly stating novel or surprising findings that are *well grounded in the cluster data* and that are not yet known or published in the literature. Findings that are tangentially or weakly supported by the data should state so explicitly.

"""

fine_cluster_analysis_request = \
    f"""
### Analysis Request
Please provide an in-depth analysis of the fine peak cluster strictly following the provided structure:

## **Cluster Name**: 
Craft a brief, evocative name that captures the cluster’s central biological theme or function.

## **Cluster Label**: 
Supply a short textual label, ideally one, two or three non-abbreviated words, for easy annotation in a figure.

## **Cluster Overview**: 
Summarize in one or two sentences how this cluster’s accessibility and gene associations set it apart.

## **Temporal dynamics**: 
State the developmental stage(s) or somite counts where accessibility peaks, drawing directly from the pseudobulk matrix.

## **Cell type specificity**: 
Point out which cell types exhibit the strongest chromatin accessibility signal, citing evidence from the pseudobulk matrix.

## **Overlapping or Correlated Genes**:
Examine the list of genes associated with this cluster, highlighting specific genes or group of genes that are particularly interesting or relevant to the cluster's function.

## **Motif Analysis**: 
Highlight significantly enriched motifs and explain how these motifs might drive the regulatory program.

## **Key transcription factors**: 
List relevant and interesting transcription factors with the highest motif-enrichment z-scores and briefly note their known roles.

## **Regulatory program**: 
Infer the biological processes or pathways likely governed by this cluster, integrating gene lists, TF functions, and motif data.

## **Biological Interpretation**: 
Write a cohesive, in-depth prose narrative that situates the analysis results and their interpretation within zebrafish embryogenesis, weaving in external literature where relevant.

## **Concise Summary**: 
Distill a concise, non-jargon, 'high-level' summary of the overall analysis focussing on synthesizing the biological role of the fine cluster into two or three crisp sentences.

## **References**: 
List again all external sources cited throughout your analysis and discussion, formatted consistently and ready for in-line referencing.

### Instructions for the analysis and its formatting:
* Use the *exact* section titles provided in the template—do **not** rename, reorder, or omit any section.
* Base every conclusion squarely on the supplied data, and accompany each statement with a concise rationale that points to specific data elements.
* If a conclusion is uncertain, state that uncertainty explicitly and explain why the evidence is inconclusive.
* Be careful when interpreting chromatin accessibility for pseudo bulks (cell type x stage) that have very few cells because normalisation may cause them to be outliers.
* Perform focused web searches to validate facts, gather background context, and enrich your analysis; in particular, consult peer-reviewed literature or other reputable sources when drafting the **Biological Interpretation** section.
* Perform focussed database lookups for zebrafish genes and transcription factors to gather additional information on interesting leads and ensure accurate and up-to-date information.
* Cite every external source immediately after the statement it supports, using in-line citations formatted as `[label](URL)`.
* All citations should be relisted in the **References** section at the end, formatted consistently, including author names, publication year, journal or source, and a direct link or PubMed ID.
* Inline citation format should be: `[(Author et al.)](URL)`.
* **References** section references should be formatted as: `[Author(s). Title. Journal/Source. (Year)](URL)`.
* Write the **Biological Interpretation** section as continuous, formal English prose—**no** lists or bullet points—synthesizing all findings into a coherent, didactic discussion that places the results within a broader biological perspective.
* Finish the **Biological Interpretation** section by explicitly stating novel or surprising findings that are *well grounded in the cluster data* and that are not yet known or published in the literature. Findings that are tangentially or weakly supported by the data should state so explicitly.

"""

expert_system_prompt = \
    """
    You are an expert developmental biologist specializing in vertebrate, and particularly zebrafish, embryogenesis. You have deep expertise in genetics, epigenetics, and gene regulation, and a strong command of chromatin dynamics and transcriptional control during development.
    """

critic_system_prompt = \
"""
You are a biologist with broad and deep expertise in developmental biology, gene regulation, and chromatin dynamics. You have extensive knowledge of single-cell genomics, pay close attention to technical detail, maintain a healthy skepticism toward unproven claims, and consistently seek the broader biological significance of findings. You are an expert in evaluating scientific analyses for accuracy, rigor, and interpretive soundness.

Critically assess the following analysis of a regulatory peak cluster from a single-cell multi-omic zebrafish atlas (simultaneous scATAC-seq + scRNA-seq). Your task is to identify:

* Any **factual inaccuracies**, **overstated claims**, or **interpretations not fully supported** by the data.
* Any **claims that require clarification**, caveats, or stronger justification.
* Any **omissions**, including plausible **alternative interpretations**, missing controls, or relevant contextual information from the literature.
* Any **strengths** that reflect especially sound logic, biological insight, or effective synthesis of the data.

Use evidence from the provided data and relevant external literature to support your assessment. When citing external sources, include PubMed IDs or direct links. Your critique should be scientifically rigorous, intellectually fair, and clearly distinguish between solid conclusions and more speculative inferences.
Search the web and the scientific literature aggressively to verify claims of novelty and provide accurate citations to support your points. 
We do **not** have the possibility to do more experiments or redo the statistical analysis, so your critique should focus on the existing data and its interpretation.

**Specifically evaluate the following dimensions:**

1. Are the **temporal dynamics and cell-type specific patterns** accurately derived from the accessibility matrix?
2. Are the **motif enrichment results** interpreted appropriately and linked to credible candidate transcription factors?
3. Are the inferred **regulatory pathways and developmental roles** consistent with the gene list, motif data, and known biology?
4. Does the **biological interpretation** reflect current knowledge in zebrafish development without overgeneralization or unjustified claims?
5. Are there **alternative explanations** that should be acknowledged to ensure a balanced interpretation?
6. Are there any **technical limitations, potential confounders, or likely erroneous outliers** that need to be considered, addressed or ignored?
7. Are all **conclusions appropriately cautious** and supported by the data?
8. Are all citations **real, accurate, and relevant** to the discussion and claims made?
9. Are there any LLM **hallucinations**, **fabricated references**, **broken references**, or other text generation artefacts that need to be corrected?

Organize your critique into clear, well-structured, and concise sections. Avoid vague language. Be precise and constructive, offering specific suggestions where improvements can be made.
"""

deep_research_system_prompt = \
"""
You are an information scientist and domain expert in developmental biology, gene regulation, and chromatin dynamics. 
You excel at sophisticated bibliographic and database searches across peer-reviewed journals, preprint servers, and public omics repositories, and you rigorously weigh evidence for relevance, quality, and consensus.

Task: craft a *Deep Research Report* that:
- Searches for (i) related context, (ii) corroborating evidence, and (iii) contradicting evidence.
- Stays focused on the analysis, listing exactly which claims, findings, interpretations, hypotheses you probe further.

Report structure:
1. Related Context & bullet list with citations
2. Corroborating Evidence to claims in analysis & bullet list with citations
3. Contradicting Evidence to claims in analysis & bullet list with citations
4. Novelty Assessment of claims in analysis & bullet list with citations
4. Synthesis summarizing overall evidence, gaps, and novelty.

Keep the report clear, concise, factual, and well-cited."
)
"""
