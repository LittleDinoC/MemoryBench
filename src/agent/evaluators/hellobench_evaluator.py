from typing import Any, List, Dict
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class HelloBenchEvaluator(BaseEvaluator):
    """Evaluator for HelloBench dataset - uses checklist-based evaluation"""
    
    def __init__(self):
        super().__init__("HelloBench")
    
    def _extract_checklist(self, data: Any) -> List[Dict[str, Any]]:
        """Extract evaluation checklist from HelloBench data using consolidated utilities"""
        return UserSimulator.extract_data_field(
            data, 'info', 'checklist', default=[]
        )

    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for HelloBench evaluation using consolidated template"""
        
        # Build evaluation criteria from checklist
        evaluation_criteria = ["Overall content quality", "Adherence to user requirements", "User satisfaction"]
        
        return UserSimulator.build_base_system_prompt(
            user_persona="You are an AI assistant simulating a user seeking high-quality content.",
            domain_expertise="You have expertise to recognize well-crafted content that meets the specified standards and can evaluate responses based on quality criteria provided in the checklist.",
            evaluation_criteria=evaluation_criteria,
            additional_context="Evaluate based on the specific checklist criteria provided, being fair and constructive."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for HelloBench feedback using consolidated template with checklist"""
        checklist = self._extract_checklist(context.data)
        
        # Build evaluation context with checklist information
        evaluation_context = ""
        
        if checklist:
            evaluation_context = "Quality Criteria (evaluate the response based on these standards):\n"
            for item in checklist:
                content = item.get('checklist_content', '')
                evaluation_context += f"- {content}\n"
            
            evaluation_context += "\nUse these criteria to evaluate the response quality and provide feedback based on how well the content meets these standards."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="seeking high-quality content",
            evaluation_context=evaluation_context
        )