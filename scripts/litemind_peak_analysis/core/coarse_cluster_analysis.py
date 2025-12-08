from typing import Optional, List

from arbol import asection
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.workflow.task import Task

from scripts.litemind_peak_analysis.core.data import load_coarse_cluster_data, process_cluster_data
from scripts.litemind_peak_analysis.core.prompts import project_background, expert_system_prompt, coarse_cluster_analysis_request
from scripts.litemind_peak_analysis.utils.citations import has_broken_citations
from scripts.litemind_peak_analysis.utils.markdown import table_to_markdown

# Load coarse cluster data:
_df_peak_stats_coarse, _df_num_cells, _df_clusters_groups_coarse, _cluster_genes_dict_coarse, _df_peak_details_overlap_coarse, _df_peak_details_corr_coarse, _df_peak_details_anticorr_coarse, _df_clusters_motifs_coarse, _df_motif_info_coarse = load_coarse_cluster_data()


class CoarseClusterAnalysis(Task):

    def __init__(self,
                 coarse_cluster_id: int,
                 api: BaseApi,
                 toolset: Optional[ToolSet] = None,
                 folder: Optional[str] = None):
        super().__init__(name=f"coarse_cluster_analysis_{coarse_cluster_id}",
                         folder=f"{folder}/coarse_cluster_analysis",
                         save_pdf=True)

        self.coarse_cluster_id = coarse_cluster_id

        # Create the agent:
        self.agent = Agent(api=api, toolset=toolset)
        self.agent.append_system_message(expert_system_prompt)

    @staticmethod
    def get_coarse_cluster_id_list() -> List[int]:
        return list(_df_clusters_groups_coarse.index)

    def validate_result(self, result: str) -> bool:
        # Valid if the result does not contain broken citations and contains at least 100 words
        return not has_broken_citations(result) and len(result.split()) > 100

    def build_message(self) -> Message:
        # Get cluster id:
        cluster_id = self.coarse_cluster_id

        with asection(f"=== Creating Prompt for Coarse Cluster {cluster_id} ==="):
            # Subset the data for the current cluster (cluster_id)
            df_peak_stats_cluster, df_clusters_groups_cluster, genes_text, df_peak_details_overlap_cluster, df_peak_details_corr_cluster, df_peak_details_anticorr_cluster, df_clusters_motifs_cluster = process_cluster_data(
                cluster_id, _df_peak_stats_coarse, _df_clusters_groups_coarse, _df_num_cells,
                _cluster_genes_dict_coarse,
                _df_peak_details_overlap_coarse, _df_peak_details_corr_coarse, _df_peak_details_anticorr_coarse,
                _df_clusters_motifs_coarse, _df_motif_info_coarse
            )

            # Create Message:
            message = Message()

            # Background context - consolidated and concise
            message.append_text(project_background)

            # Task context - clear and direct
            message.append_text(
                f"""

## Task – Annotate Peak Cluster {cluster_id}

Analyze the materials below to elucidate this cluster’s biological function and its role in zebrafish development.
   
""")

            # Data sections - organized and labeled

            message.append_text(f"### Peak cluster statistics for coarse cluster {cluster_id}")
            message.append_text(
                f"Summary statistics for the cluster, including number of peaks, mean peak width, genomic annotations (promoter, intron, exon, intergenic), and median distance to TSS.")
            df_peak_stats_cluster_markdown = table_to_markdown(df_peak_stats_cluster)
            message.append_text(df_peak_stats_cluster_markdown + '\n\n')

            message.append_text(
                f"### Average peak chromatin-accessibility for coarse cluster {cluster_id} per pseudobulk. ")
            message.append_text(
                "Pseudobulk chromatin-accessibility profiles for every *cell-type × time-point* combination.\n"
                "The first numerical column in this matrix correspond to the average number of reads normalised by number of cells per pseudobulk and sequencing depth. Refer to these numbers as 'normalised units'.\n"
                "The second numerical column corresponds to the number of cells per pseudobulk (same for all clusters).\n")
            df_clusters_groups_cluster_markdown = table_to_markdown(df_clusters_groups_cluster)
            df_clusters_groups_cluster_markdown = df_clusters_groups_cluster_markdown.replace('_',
                                                                                              ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_clusters_groups_cluster_markdown + '\n\n')

            message.append_text(f"### All associated genes for coarse cluster {cluster_id}")
            message.append_text(
                "Complete list of genes whose RNA abundance is strongly correlated with peak accessibility or whose gene bodies overlap the peaks in this cluster. ")
            message.append_text(genes_text + '\n\n')  # this is a string with the genes, not a DataFrame

            message.append_text(f"### Table of the top genes overlapping with peaks from coarse cluster {cluster_id}")
            message.append_text("")
            df_peak_details_overlap_cluster_markdown = table_to_markdown(df_peak_details_overlap_cluster)
            df_peak_details_overlap_cluster_markdown = df_peak_details_overlap_cluster_markdown.replace('_',
                                                                                                      ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_overlap_cluster_markdown + '\n\n')

            message.append_text(f"### Table of the top genes most correlated with peaks from coarse cluster {cluster_id}")
            message.append_text("")
            df_peak_details_corr_cluster_markdown = table_to_markdown(df_peak_details_corr_cluster)
            df_peak_details_corr_cluster_markdown = df_peak_details_corr_cluster_markdown.replace('_',
                                                                                                      ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_corr_cluster_markdown + '\n\n')

            message.append_text(f"### Table of the top genes most anti-correlated with peaks from coarse cluster {cluster_id}")
            message.append_text("")
            df_peak_details_anticorr_cluster_markdown = table_to_markdown(df_peak_details_anticorr_cluster)
            df_peak_details_anticorr_cluster_markdown = df_peak_details_anticorr_cluster_markdown.replace('_',
                                                                                                      ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_anticorr_cluster_markdown)
            message.append_text("(Table might be empty)" + '\n\n')


            message.append_text(
                f"### Enriched transcription factor motifs for coarse cluster {cluster_id}")
            message.append_text(
                "Transcription factor binding motifs that are significantly over-represented in the cluster (z-scores provided), as well as corresponding transcription factors gene names, and consensus sequences. ")
            message.append_text(
                'Note: *direct* means that there is experimental evidence for direct TF-motif binding, and *indirect* means that binding is inferred from homologs. In CisBP, most of the motifs are “indirect” -- there is in fact only 19 zebrafish TFs with experimental validation.\n\n')
            df_clusters_motifs_cluster_markdown = table_to_markdown(df_clusters_motifs_cluster)
            df_clusters_motifs_cluster_markdown = df_clusters_motifs_cluster_markdown.replace('nan', 'N/A')
            message.append_text(df_clusters_motifs_cluster_markdown + '\n')


            # Clear analysis request
            message.append_text(coarse_cluster_analysis_request)

            return message
