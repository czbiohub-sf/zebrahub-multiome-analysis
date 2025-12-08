from arbol import asection, aprint
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures
from litemind.workflow.task import Task

from scripts.litemind_peak_analysis.core.prompts import deep_research_system_prompt
from scripts.litemind_peak_analysis.utils.citations import has_broken_citations
from scripts.litemind_peak_analysis.utils.markdown import quote_text


#

class DeepResearch(Task):

    def __init__(self,
                 analysis_task: Task,
                 api: BaseApi,
                 model_name: str = "o3-deep-research",
                 toolset: Optional[ToolSet] = None,
                 folder: Optional[str] = None):
        super().__init__(name=f"{analysis_task.name}_deep_research",
                         dependencies=[analysis_task],
                         folder=f"{folder}/deep_research",
                         save_pdf=True)

        self.analysis_task = analysis_task

        # Create the agent:
        self.agent = Agent(api=api, model_name=model_name, toolset=toolset)
        self.agent.append_system_message(deep_research_system_prompt)

        # Store API:
        self.api: BaseApi = api

    def validate_result(self, result: str) -> bool:
        # Valid if the result does not contain broken citations and contains at least 100 words
        return not has_broken_citations(result) and len(result.split()) > 100

    def build_message(self) -> Message:
        with asection(f"=== Creating Deep Research Prompt for Cluster Analysis Review ({self.name}) ==="):
            # Get the prompt string from the analysis task:
            prompt_str = self.analysis_task.get_prompt()

            # The dict argument has a single key/value, extract that value into a string independent of the key:
            analysis_str = self.analysis_task.get_result()

            # Quote the prompt and analysis strings:
            prompt_str = quote_text(prompt_str)
            analysis_str = quote_text(analysis_str)

            prompt_generation_sys_message = Message(role="system")
            prompt_generation_sys_message.append_text(
                "You are an expert at crafting precise prompts for deep research tasks.\n"
                "You will receive: (a) the original prompt and (b) the analysis derived from it.\n"
                "Create a new *Deep Research* prompt that instructs a comprehensive search for (i) related context, (ii) corroborating evidence, (iii) contradicting evidence, (iv) novelty relevant to the analysis.\n"
                "The prompt must be concise, context-rich, and laser-focused on illuminating the analysis provided.\n"
                "Explicitly list the findings, insights, interpretations, hypotheses, within the analysis that require deeper investigation.\n"
                "*Important*: the deep research agent will *not* have access to the original prompt or analysis, so ensure the prompt is fully self-contained and does not assume knowledge of the prompt or resulting analysis!\n"
                "Provide enough detail to guide rigorous research while avoiding unnecessary length.\n")

            prompt_generation_message = Message()
            prompt_generation_message.append_text(f"\n\nPrompt:\n{prompt_str}\n\nAnalysis:\n{analysis_str}\n\n")

            # Get a non-thinking model for litemind:
            prompt_generation_model = self.api.get_best_model(features=ModelFeatures.TextGeneration,
                                                              non_features=ModelFeatures.Thinking)
            aprint(f"Using model for prompt generation: {prompt_generation_model}")

            response = self.api.generate_text(messages=[prompt_generation_sys_message, prompt_generation_message],
                                              model_name=prompt_generation_model)

            deep_research_prompt_str = response[-1].to_markdown()

            # with asection(f"Deep Research Prompt Generated:"):
            #     aprint(deep_research_prompt_str)

        with asection(f"=== Performing Deep Research on Analysis Results ({self.name}) ==="):
            # Create Message:
            message = Message()
            message.append_text(deep_research_prompt_str)

            return message
