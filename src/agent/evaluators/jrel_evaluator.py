from typing import Any
from .base_evaluator import BaseEvaluator, EvaluationContext
from src.agent.user_simulator import UserSimulator


class JRELEvaluator(BaseEvaluator):
    """Evaluator for JRE-L dataset - focuses on journalistic science writing evaluation"""
    
    def __init__(self):
        super().__init__("JRE-L")
    
    def _extract_dataset_info(self, data: Any) -> tuple:
        """Extract paper and journalistic information from JRE-L data using consolidated utilities"""
        sc_title = UserSimulator.extract_data_field(
            data, 'info', 'sc-title', default=''
        )
        sc_abstract = UserSimulator.extract_data_field(
            data, 'info', 'sc-abstract', default=''
        )
        pr_title = UserSimulator.extract_data_field(
            data, 'info', 'pr-title', default=''
        )
        pr_abstract = UserSimulator.extract_data_field(
            data, 'info', 'pr-abstract', default=''
        )
        
        return sc_title, sc_abstract, pr_title, pr_abstract

    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate system prompt for science journalism evaluation using consolidated template"""
        return UserSimulator.build_base_system_prompt(
            user_persona="You are simulating a science journalist or editor who requested AI assistance to write journalistic reports of scientific papers for general audiences.",
            domain_expertise="You have expertise in science journalism across diverse fields including computer science, cybersecurity, privacy research, mobile computing, cloud services, encryption technologies, biomedical research, environmental science, and other technical domains. You understand what makes scientific writing accessible to the general public while maintaining accuracy.",
            evaluation_criteria=[
                "Accessible and readable for general audiences without technical background",
                "Accurate to the original scientific work without oversimplification",
                "Engaging and newsworthy in its presentation style", 
                "Well-structured with appropriate journalistic elements (headlines, lead paragraphs, context)",
                "Properly balancing technical detail with readability",
                "Readability for lay audiences",
                "Journalistic style and structure",
                "Engagement factor and clarity of technical concepts"
            ],
            additional_context="Your evaluation focuses on the journalistic transformation of academic content rather than the underlying research quality. Consider: readability for lay audiences, accuracy to source material, journalistic style and structure, engagement factor, and clarity of technical concepts."
        )

    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate user prompt for JRE-L feedback using consolidated template with journalistic context"""
        sc_title, sc_abstract, pr_title, pr_abstract = self._extract_dataset_info(context.data)
        
        evaluation_context = f"""Reference Popular Title: {pr_title}
Reference Popular Abstract: {pr_abstract}

Use this reference to evaluate the quality of the journalistic transformation. Consider how well the generated content compares to this reference in terms of accessibility, accuracy, and engagement for general audiences."""
        
        return UserSimulator.build_base_user_prompt(
            context=context,
            task_description="seeking to transform scientific papers into accessible journalistic reports for general audiences",
            evaluation_context=evaluation_context
        )