from typing import Any
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class IdeaBenchEvaluator(BaseEvaluator):
    """Evaluator for IdeaBench dataset - focuses on biomedical research hypothesis generation"""
    
    def __init__(self):
        super().__init__("IdeaBench")
    
    def _extract_dataset_info(self, data: Any) -> tuple:
        """Extract paper information from IdeaBench data using consolidated utilities"""
        paper_id = UserSimulator.extract_data_field(
            data, 'info', 'paperId', default=''
        )
        title = UserSimulator.extract_data_field(
            data, 'info', 'title', default=''
        )
        abstract = UserSimulator.extract_data_field(
            data, 'info', 'abstract', default=''
        )
        
        return paper_id, title, abstract

    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for biomedical research hypothesis evaluation using consolidated template"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are simulating a biomedical researcher who requested AI assistance to generate novel research hypotheses based on background literature abstracts.",
            domain_expertise="You have expertise across diverse biomedical subdomains including genomics, COVID-19 research, cancer immunotherapy, traumatic brain injury, plant biology, vitamin D and immune systems, heart failure studies, and other biomedical fields. You understand what makes hypotheses novel, feasible, and scientifically sound across these domains.",
            evaluation_criteria=[
                "Novel and distinct from existing work in the specific biomedical subdomain",
                "Scientifically feasible and testable given current methodologies", 
                "Well-reasoned based on the provided abstracts and domain knowledge",
                "Properly formatted according to requirements (3 hypotheses with ---IDEA-SEPARATOR---)",
                "Scientific rigor and methodological soundness",
                "Domain-specific novelty assessment",
                "Overall research value and potential impact"
            ],
            additional_context="Consider: scientific rigor, domain-specific novelty assessment, proper separation formatting, and overall research value."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for IdeaBench feedback using consolidated template with research context"""
        _, title, abstract = self._extract_dataset_info(context.data)
        
        evaluation_context = f"""Reference Paper Title: {title}
Reference Paper Abstract: {abstract}

Use this reference to evaluate the quality and relevance of the generated hypotheses. Consider how well the hypotheses relate to the research direction and findings represented by this reference paper."""
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="seeking novel hypotheses based on literature abstracts across diverse biomedical subdomains (genomics, COVID-19, cancer, TBI, plant biology, etc.)",
            evaluation_context=evaluation_context
        )