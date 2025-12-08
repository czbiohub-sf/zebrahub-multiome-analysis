import re
from typing import Optional, List

from arbol import asection
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.workflow.task import Task

from scripts.litemind_peak_analysis.core.data import process_cluster_data, load_fine_cluster_data
from scripts.litemind_peak_analysis.core.prompts import project_background, expert_system_prompt, fine_cluster_analysis_request
from scripts.litemind_peak_analysis.utils.citations import has_broken_citations
from scripts.litemind_peak_analysis.utils.markdown import table_to_markdown, quote_text

# Load coarse cluster data:
_df_peak_stats_fine, _df_num_cells, _df_clusters_groups_fine, _cluster_genes_dict_fine, _df_peak_details_overlap_fine, _df_peak_details_corr_fine, _df_peak_details_anticorr_fine, _df_clusters_motifs_fine, _df_motif_info_fine = load_fine_cluster_data()


class FineClusterAnalysis(Task):

    def __init__(self,
                 fine_cluster_id: str,
                 initial_coarse_cluster_analysis_task: Task,
                 final_coarse_cluster_analysis_task: Task,
                 api: BaseApi,
                 toolset: Optional[ToolSet] = None,
                 folder: Optional[str] = None):
        super().__init__(name=f"fine_cluster_analysis_{fine_cluster_id}",
                         dependencies=[initial_coarse_cluster_analysis_task, final_coarse_cluster_analysis_task],
                         folder=f"{folder}/fine_cluster_analysis",
                         save_pdf=True)

        # Store the fine cluster id:
        self.fine_cluster_id = fine_cluster_id

        # Store the coarse cluster analysis task as a dependency:
        self.initial_coarse_cluster_analysis_task = initial_coarse_cluster_analysis_task
        self.final_coarse_cluster_analysis_task = final_coarse_cluster_analysis_task

        # parse the coarse cluster id from the fine cluster id:
        self.coarse_cluster_id = int(fine_cluster_id.split('_')[0])

        # Create the agent:
        self.agent = Agent(api=api, toolset=toolset)
        self.agent.append_system_message(expert_system_prompt)

    @staticmethod
    def get_fine_cluster_id_list() -> List[str]:
        return list(_df_clusters_groups_fine.index)

    def validate_result(self, result: str) -> bool:
        # Valid if the result does not contain broken citations and contains at least 100 words
        return not has_broken_citations(result) and len(result.split()) > 100

    def build_message(self) -> Message:
        # Get the coarse cluster analysis task prompt (dependency):
        coarse_cluster_prompt_str = self.initial_coarse_cluster_analysis_task.get_prompt()

        # get analysis result from the coarse cluster analysis task (dependency):
        coarse_cluster_analysis_str = self.final_coarse_cluster_analysis_task.get_result()

        # Quote the coarse cluster prompt and analysis result:
        coarse_cluster_prompt_str = quote_text(coarse_cluster_prompt_str)
        coarse_cluster_analysis_str = quote_text(coarse_cluster_analysis_str)

        # Get cluster ids:
        coarse_cluster_id = self.coarse_cluster_id
        fine_cluster_id = self.fine_cluster_id

        with asection(f"=== Creating Prompt for Fine Cluster {fine_cluster_id} ==="):
            # Subset the data for the current cluster (cluster_id)
            df_peak_stats_cluster, df_clusters_groups_cluster, genes_text, df_peak_details_overlap_cluster, df_peak_details_corr_cluster, df_peak_details_anticorr_cluster, df_clusters_motifs_cluster = process_cluster_data(
                fine_cluster_id, _df_peak_stats_fine, _df_clusters_groups_fine, _df_num_cells, _cluster_genes_dict_fine,
                _df_peak_details_overlap_fine, _df_peak_details_corr_fine, _df_peak_details_anticorr_fine,
                _df_clusters_motifs_fine, _df_motif_info_fine
            )

            # Create Message:
            message = Message()

            # Background context - consolidated and concise
            message.append_text(project_background)

            # Task context - clear and direct
            message.append_text(
                f"""

## Task – Annotate Fine Peak Cluster {fine_cluster_id} belonging to Coarse Cluster {coarse_cluster_id}

Analyze the materials below to elucidate this cluster’s biological function and its role in zebrafish development.
We are **particularly interested** in how this fine cluster fits into the broader context of the coarse cluster {coarse_cluster_id}: 
- what is its **specific role** within the larger regulatory programme? 
- What are **key differences** or nuances that set this fine cluster apart from the coarse cluster?

""")

            # Context sections -
            message.append_text(
                f"### Fine cluster {fine_cluster_id} is a sub-cluster of coarse cluster {coarse_cluster_id}, here is the prompt and resulting analysis already done for that coarse 'high-level' cluster: ")
            message.append_text(
                f"\nPrompt for coarse cluster {coarse_cluster_id}:\n{coarse_cluster_prompt_str}\n\nAnalysis for coarse cluster {coarse_cluster_id}:\n{coarse_cluster_analysis_str}\n" + '\n\n')

            # Data sections - organized and labeled
            message.append_text(f"### Peak cluster statistics for fine cluster {fine_cluster_id}")
            message.append_text(
                f"Summary statistics for the cluster, including number of peaks, mean peak width, genomic annotations (promoter, intron, exon, intergenic), and median distance to TSS.")
            df_peak_stats_cluster_markdown = table_to_markdown(df_peak_stats_cluster)
            message.append_text(df_peak_stats_cluster_markdown + '\n\n')

            message.append_text(
                f"### Average peak chromatin-accessibility for fine cluster {fine_cluster_id} per pseudobulk. ")
            message.append_text(
                "Pseudobulk chromatin-accessibility profiles for every *cell-type × time-point* combination.\n"
                "The first numerical column in this matrix correspond to the average number of reads normalised by number of cells per pseudobulk and sequencing depth. Refer to these numbers as 'normalised units'.\n"
                "The second numerical column corresponds to the number of cells per pseudobulk (same for all clusters).\n")
            df_clusters_groups_cluster_markdown = table_to_markdown(df_clusters_groups_cluster)
            df_clusters_groups_cluster_markdown = df_clusters_groups_cluster_markdown.replace('_',
                                                                                              ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_clusters_groups_cluster_markdown + '\n\n')

            message.append_text(f"### All associated genes for fine cluster {fine_cluster_id}")
            message.append_text(
                "Complete list of genes whose RNA abundance is strongly correlated with peak accessibility or whose gene bodies overlap the peaks in this fine cluster. ")
            message.append_text(genes_text + '\n\n')  # this is a string with the genes, not a DataFrame

            message.append_text(f"### Table of the top genes overlapping with peaks from fine cluster {fine_cluster_id}")
            message.append_text("")
            df_peak_details_overlap_cluster_markdown = table_to_markdown(df_peak_details_overlap_cluster)
            df_peak_details_overlap_cluster_markdown = df_peak_details_overlap_cluster_markdown.replace('_',
                                                                                                        ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_overlap_cluster_markdown + '\n\n')

            message.append_text(
                f"### Table of the top genes most correlated with peaks from fine cluster {fine_cluster_id}")
            message.append_text("")
            df_peak_details_corr_cluster_markdown = table_to_markdown(df_peak_details_corr_cluster)
            df_peak_details_corr_cluster_markdown = df_peak_details_corr_cluster_markdown.replace('_',
                                                                                                  ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_corr_cluster_markdown + '\n\n')

            message.append_text(
                f"### Table of the top genes most anti-correlated with peaks from fine cluster {fine_cluster_id}")
            message.append_text("")
            df_peak_details_anticorr_cluster_markdown = table_to_markdown(df_peak_details_anticorr_cluster)
            df_peak_details_anticorr_cluster_markdown = df_peak_details_anticorr_cluster_markdown.replace('_',
                                                                                                          ' ')  # Replace underscores with spaces for better readability
            message.append_text(df_peak_details_anticorr_cluster_markdown)
            message.append_text("(Table might be empty)" + '\n\n')

            message.append_text(
                f"### Enriched transcription factor motifs for fine cluster {fine_cluster_id}")
            message.append_text(
                "Transcription factor binding motifs that are significantly over-represented in the cluster (z-scores provided), as well as corresponding transcription factors gene names, and consensus sequences. ")
            message.append_text(
                'Note: *direct* means that there is experimental evidence for direct TF-motif binding, and *indirect* means that binding is inferred from homologs. In CisBP, most of the motifs are “indirect” -- there is in fact only 19 zebrafish TFs with experimental validation.\n\n')
            df_clusters_motifs_cluster_markdown = table_to_markdown(df_clusters_motifs_cluster)
            df_clusters_motifs_cluster_markdown = df_clusters_motifs_cluster_markdown.replace('nan', 'N/A')
            message.append_text(df_clusters_motifs_cluster_markdown + '\n')

            # Clear analysis request
            message.append_text(fine_cluster_analysis_request)

            return message

    def post_process_result_before_saving_pdf(self, result: str) -> str:
        # Replace fine clusters ids in format x_y with x.y, using a robust regex:
        result = re.sub(r'(\d+)_(\d+)', r'\1:\2', result)

        # return the processed result
        return result
