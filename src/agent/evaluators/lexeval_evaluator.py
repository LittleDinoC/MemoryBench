from typing import Any
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class LexEvalEvaluator(BaseEvaluator):
    """Evaluator for LexEval dataset - focuses on legal knowledge accuracy"""
    
    def __init__(self):
        super().__init__("LexEval")
    
    def _extract_golden_answer(self, data: Any) -> str:
        """Extract ground truth answer from LexEval data using consolidated utilities"""
        return UserSimulator.extract_data_field(
            data, 'info', 'golden_answer', default=''
        )
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for legal domain user simulation using consolidated template"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are an AI assistant simulating a knowledgeable user seeking legal information or assistance.",
            domain_expertise="You have domain expertise to recognize quality legal information without revealing specific ground truth details.",
            evaluation_criteria=[
                "Legal accuracy and correctness of information provided",
                "Completeness in addressing the legal question or issue", 
                "Clarity and understandability for a non-lawyer",
                "Practical usefulness of the advice or information",
                "Whether the assistant properly acknowledged limitations or suggested consulting a lawyer when appropriate"
            ],
            additional_context="Natural conversation flow and user satisfaction with legal guidance received"
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for LexEval feedback using consolidated template"""
        golden_answer = self._extract_golden_answer(context.data)
        
        evaluation_context = ""
        if golden_answer:
            evaluation_context = f"Reference Answer: {golden_answer}\n"
            evaluation_context += "Use this reference to evaluate the accuracy and completeness of the legal information provided."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="seeking accurate legal information or guidance",
            evaluation_context=evaluation_context
        )
    
