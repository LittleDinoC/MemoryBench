from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class NFCatsEvaluator(BaseEvaluator):
    """Evaluator for NFCats dataset - focuses on non-factoid question answering across different categories"""
    
    def __init__(self):
        super().__init__("NFCats")
    
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for NFCats domain user simulation using consolidated template"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are an AI assistant simulating a knowledgeable user asking various types of non-factoid questions that require thoughtful, comprehensive responses.",
            domain_expertise="You can evaluate the quality of responses to different question types including comparisons, debates, evidence-based explanations, instructions, and experience-based questions.",
            evaluation_criteria=[
                "Appropriateness of response type to the question category (comparison, debate, evidence-based, instruction, etc.)",
                "Completeness and thoroughness of the answer",
                "Logical structure and clarity of explanation", 
                "Relevance and accuracy of information provided",
                "Practical usefulness and helpfulness to the questioner",
                "Balanced perspective when addressing debatable topics",
                "Clear step-by-step guidance for instructional questions"
            ],
            additional_context="The questions span multiple categories including COMPARISON, DEBATE, EVIDENCE-BASED, INSTRUCTION, EXPERIENCE, REASON, and FACTOID. Each requires a different approach to provide a satisfactory response."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for NFCats feedback using consolidated template"""
        # NFCats dataset focuses on question categorization, so evaluation is based on response appropriateness
        evaluation_context = "This dataset contains non-factoid questions from various categories (COMPARISON, DEBATE, EVIDENCE-BASED, INSTRUCTION, EXPERIENCE, REASON, FACTOID). Evaluate whether the assistant's response appropriately addresses the type and nature of the question asked."
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="asking a non-factoid question that requires a thoughtful, comprehensive answer tailored to the question type",
            evaluation_context=evaluation_context
        )