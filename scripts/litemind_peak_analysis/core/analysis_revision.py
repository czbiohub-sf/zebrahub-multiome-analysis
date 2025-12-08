import re

from arbol import asection
from litemind.agent.messages.message import Message
from litemind.workflow.task import Task

from scripts.litemind_peak_analysis.utils.citations import has_broken_citations
from scripts.litemind_peak_analysis.utils.markdown import quote_text


class ClusterAnalysisRevision(Task):

    def __init__(self,
                 analysis_task: Task,
                 review_task: Task,
                 analysis_format_instructions: str,
                 folder: Optional[str] = None):
        super().__init__(name=f"{analysis_task.name}_revision",
                         dependencies=[analysis_task, review_task],
                         folder=f"{folder}/{analysis_task.get_folder_name()}_revision",
                         save_pdf=True)

        # Store the analysis and review tasks:
        self.analysis_task = analysis_task
        self.review_task = review_task

        self.analysis_format_instructions = analysis_format_instructions

        # get the agent from the analysis task, as we continue the same conversation:
        self.agent = analysis_task.agent

    def validate_result(self, result: str) -> bool:
        # Valid if the result does not contain broken citations and contains at least 100 words
        return not has_broken_citations(result) and len(result.split()) > 100

    def build_message(self) -> Message:
        # Get the prompt string from the analysis task:
        review_str = self.review_task.get_result()

        # Quote the review string:
        review_str = quote_text(review_str)

        with asection(f"=== Creating Prompt for Cluster Analysis Revision ({self.name}) ==="):
            # Create Message:
            message = Message()

            message.append_text(
                "I have shown your analysis to a fellow expert who provided the following review:")
            message.append_text(f"\nReview:{review_str}\n\n")
            message.append_text(
                "Please revise the original analysis in light of the above critique.\n"
                "Take the feedback constructively and improve the analysis accordingly.\n"
                "Do not mention the revision or changes applied in the revised analysis.\n"
                f"Your updated analysis must adhere to the exact same original instructions and format, without preamble or postscript:\n\n{self.analysis_format_instructions}\n"

            )

            return message

    def post_process_result_before_saving_pdf(self, result: str) -> str:
        # Replace fine clusters ids in format x_y with x.y, using a robust regex:
        result = re.sub(r'(\d+)_(\d+)', r'\1:\2', result)

        # return the processed result
        return result
