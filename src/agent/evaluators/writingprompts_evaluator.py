from typing import Any
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class WritingPromptsEvaluator(BaseEvaluator):
    """Evaluator for WritingPrompts dataset - focuses on creative writing evaluation"""
    
    def __init__(self):
        super().__init__("WritingPrompts")
    
    def _extract_dataset_info(self, data: Any) -> tuple:
        """Extract WritingPrompts-specific information using consolidated utilities"""
        golden_answer = UserSimulator.extract_data_field(
            data, 'info', 'golden_answer', default=''
        )
        
        return golden_answer
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for creative writing evaluation using consolidated template"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are simulating a creative writing enthusiast who has requested a story based on a specific prompt.",
            domain_expertise="You have experience reading and evaluating creative writing, and can recognize engaging storytelling, character development, and narrative structure.",
            evaluation_criteria=[
                "Creativity and originality in interpreting the prompt",
                "Narrative coherence and flow",
                "Engaging characters and dialogue",
                "Imaginative world-building and setting",
                "Emotional engagement and impact",
                "Writing quality and style"
            ],
            additional_context="Focus on natural user responses that reflect appreciation for creative storytelling. Consider how well the story captures the essence of the original prompt while being entertaining and well-written."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for WritingPrompts feedback using consolidated template"""
        golden_answer = self._extract_dataset_info(context.data)
        
        # Build evaluation context with reference story information
        evaluation_context = ""
        if golden_answer:
            # Include the actual reference story for proper comparative evaluation
            evaluation_context = f"Reference Story for Comparison:\n{golden_answer}\n\n"
            evaluation_context += "Use this reference story to evaluate how well the generated response interprets the prompt. "
            evaluation_context += "Consider creativity, narrative quality, character development, and overall storytelling compared to this reference. "
            evaluation_context += "Provide constructive feedback on strengths and areas for improvement."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="requesting a creative story based on a writing prompt",
            evaluation_context=evaluation_context
        )