from typing import Any
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class WritingBenchEvaluator(BaseEvaluator):
    """Evaluator for WritingBench dataset - focuses on specific evaluation criteria"""
    
    def __init__(self):
        super().__init__("WritingBench")
    
    def _extract_criteria_info(self, data: Any) -> tuple:
        """Extract evaluation criteria from WritingBench data using consolidated utilities"""
        criteria = UserSimulator.extract_data_field(
            data, 'info', 'criteria', default=[]
        )
        dataset_name = UserSimulator.extract_data_field(
            data, 'dataset_name', default='WritingBench'
        )
            
        # Extract domain from dataset name
        if 'Politics & Law' in dataset_name:
            domain = 'Politics & Law'
        elif 'Academic & Engineering' in dataset_name:
            domain = 'Academic & Engineering'
        elif 'Literature & Arts' in dataset_name:
            domain = 'Literature & Arts'
        elif 'Education' in dataset_name:
            domain = 'Education'
        else:
            domain = 'General Writing'
            
        return criteria, domain
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for criteria-based writing evaluation using consolidated template"""
        _ , domain = self._extract_criteria_info(context.data)
        
        evaluation_criteria = ["Criteria compliance", "Overall writing quality", "Task completeness", "User satisfaction"]
        
        return UserSimulator.build_base_system_prompt(
            user_persona=f"You are an AI assistant simulating a user who requested writing assistance in the {domain} domain.",
            domain_expertise=f"You have specific evaluation criteria that the writing must meet and can recognize quality writing.",
            evaluation_criteria=evaluation_criteria,
            additional_context="Consider: criteria compliance, overall quality, task completeness, and user satisfaction."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for WritingBench feedback using consolidated template with criteria context"""
        criteria, domain = self._extract_criteria_info(context.data)
        
        # Build evaluation context with criteria information
        evaluation_context = f"Domain: {domain}\n"
        
        # Add criteria information if available
        if criteria:
            evaluation_context += "Evaluation Criteria:\n"
            for i, criterion in enumerate(criteria, 1):
                name = criterion.get('name', f'Criterion {i}')
                description = criterion.get('criteria_description', '')
                evaluation_context += f"{i}. {name}: {description}\n"
            
            evaluation_context += "\nUse these criteria to evaluate writing quality and provide feedback based on how well the writing meets these standards."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description=f"requesting writing assistance in the {domain} domain",
            evaluation_context=evaluation_context
        )
    
