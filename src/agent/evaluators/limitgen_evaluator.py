from src.agent.user_simulator import UserSimulator
from .base_evaluator import BaseEvaluator, EvaluationContext
from typing import Any, Tuple


class LimitGenEvaluator(BaseEvaluator):
    """
    Evaluator for LimitGen dataset - scientific paper limitation identification.
    
    LimitGen focuses on evaluating AI systems' ability to identify limitations
    in scientific papers across different aspects: methodology, experimental design,
    result analysis, and literature review.
    """
    
    def __init__(self):
        super().__init__("LimitGen")
    
    def _extract_dataset_info(self, data: Any) -> Tuple[str, str, str]:
        """Extract LimitGen-specific information from data item"""
        ground_truth = UserSimulator.extract_data_field(
            data, 'info', 'ground_truth', default=''
        )
        category = UserSimulator.extract_data_field(
            data, 'info', 'category', default='unknown'
        )
        
        # Map category to aspect for context
        aspect_mapping = {
            "data": "methodology",
            "inappropriate": "methodology", 
            "baseline": "experimental design",
            "dataset": "experimental design",
            "replace": "experimental design",
            "ablation": "experimental design",
            "metric": "result analysis",
            "analysis": "result analysis",
            "citation": "literature review",
            "review": "literature review", 
            "description": "literature review"
        }
        
        aspect = aspect_mapping.get(category, "general")
        
        return ground_truth, category, aspect
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for LimitGen user simulation"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are simulating an expert scientific reviewer evaluating an AI system's ability to identify limitations in research papers.",
            domain_expertise="You understand scientific research methodology, experimental design, result analysis, and literature review standards. You can assess whether identified limitations are accurate, relevant, and appropriately categorized.",
            evaluation_criteria=[
                "Accuracy of limitation identification in the specified aspect",
                "Appropriateness of the limitation category",
                "Quality and specificity of limitation descriptions",
                "Avoidance of limitations explicitly mentioned in the paper",
                "Adherence to the requested JSON format"
            ],
            additional_context="Focus on whether the AI correctly identifies 3 major limitations in the specified aspect (methodology, experimental design, result analysis, or literature review) without revealing limitations already mentioned in the paper."
        )
    
    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for LimitGen evaluation with dataset context"""
        ground_truth, category, aspect = self._extract_dataset_info(context.data)
        
        # Build evaluation context with privacy protection
        evaluation_context = ""
        if ground_truth:
            evaluation_context += f"Expected limitation type: {ground_truth}\n"
        if category:
            evaluation_context += f"Target category: {category}\n"
        if aspect:
            evaluation_context += f"Target aspect: {aspect}\n"
        
        if evaluation_context:
            evaluation_context += "Use this information to evaluate the accuracy and appropriateness of the identified limitations."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="seeking identification of scientific paper limitations in a specific research aspect",
            evaluation_context=evaluation_context
        )