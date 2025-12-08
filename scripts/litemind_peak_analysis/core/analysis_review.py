from arbol import asection
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.workflow.task import Task

from scripts.litemind_peak_analysis.core.prompts import critic_system_prompt
from scripts.litemind_peak_analysis.utils.citations import has_broken_citations


#

class ClusterAnalysisReview(Task):

    def __init__(self,
                 analysis_task: Task,
                 api: BaseApi,
                 toolset: Optional[ToolSet] = None,
                 folder: Optional[str] = None):
        super().__init__(name=f"{analysis_task.name}_review",
                         dependencies=[analysis_task],
                         folder=f"{folder}/cluster_analysis_review",
                         save_pdf=True
                         )

        self.analysis_task = analysis_task

        # Create the agent:
        self.agent = Agent(api=api, toolset=toolset)
        self.agent.append_system_message(critic_system_prompt)

    def validate_result(self, result: str) -> bool:
        # Valid if the result does not contain broken citations and contains at least 100 words
        return not has_broken_citations(result) and len(result.split()) > 100

    def build_message(self) -> Message:
        # Get the prompt string from the analysis task:
        prompt_str = self.analysis_task.get_prompt()

        # The dict argument has a single key/value, extract that value into a string independent of the key:
        analysis_str = self.analysis_task.get_result()

        with asection(f"=== Creating Prompt for Cluster Analysis Review ({self.name}) ==="):
            # Create Message:
            message = Message()

            message.append_text("# Expert Analysis to Critique:")
            message.append_text(f"\n```prompt\n{prompt_str}\n```\n")
            message.append_text(f"\n```analysis\n{analysis_str}\n```\n")
            message.append_text("Provide a complete, clear, and concise critique of the above analysis.")

            return message
