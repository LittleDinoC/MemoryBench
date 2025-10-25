from src.agent.user_simulator import UserSimulator
from .base_evaluator import BaseEvaluator, EvaluationContext
from typing import Any, Tuple


class JuDGEEvaluator(BaseEvaluator):
    """
    Evaluator for JuDGE (Judgment Document Generation Evaluation) dataset.
    
    JuDGE is a Chinese legal judgment document generation benchmark that requires
    generating complete legal judgment documents from factual case descriptions.
    """
    
    def __init__(self):
        super().__init__("JuDGE")
    
    def _extract_dataset_info(self, data: Any) -> Tuple[str, str]:
        """Extract JuDGE-specific information from data item"""
        golden_answer = UserSimulator.extract_data_field(
            data, 'info', 'golden_answer', default=''
        )
        
        # Extract language information
        lang = UserSimulator.extract_data_field(
            data, 'lang', default='zh'
        )
        
        return golden_answer, lang
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for JuDGE dataset user simulation"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are simulating a Chinese legal professional or law student seeking assistance with legal document generation.",
            domain_expertise="You understand Chinese legal procedures, judgment document structure, and legal reasoning. You can assess whether a generated legal judgment document follows proper format, includes necessary legal elements, and demonstrates sound legal reasoning based on the provided case facts.",
            evaluation_criteria=[
                "Structural completeness of the legal judgment document",
                "Accurate application of relevant Chinese laws and regulations", 
                "Logical consistency between case facts and legal conclusions",
                "Proper legal document formatting and professional language use",
                "Completeness of required legal elements (case summary, legal analysis, verdict, etc.)",
                "Accuracy of legal reasoning and citation of appropriate statutes"
            ],
            additional_context="Focus on evaluating the assistant's ability to generate comprehensive, legally sound Chinese criminal judgment documents. Consider whether the response demonstrates understanding of Chinese legal procedures and produces a document that would be acceptable in actual legal practice."
        )
    
    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for JuDGE dataset evaluation"""
        golden_answer, lang = self._extract_dataset_info(context.data)
        
        # Prepare evaluation context with reference judgment document
        evaluation_context = ""
        if golden_answer:
            evaluation_context = f"Reference judgment document: {golden_answer}\n"
            evaluation_context += "Use this reference to evaluate the completeness, structure, and legal accuracy of the generated judgment document."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="requesting generation of a complete Chinese criminal judgment document based on provided case facts",
            evaluation_context=evaluation_context
        )