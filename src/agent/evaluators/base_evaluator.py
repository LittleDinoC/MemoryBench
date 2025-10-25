from abc import ABC, abstractmethod
from typing import Dict, Any, List


class EvaluationContext:
    """Context information for evaluation"""
    def __init__(
        self,
        messages: List[Dict[str, str]],
        data: Any,
        dataset_name: str
    ):
        self.messages = messages
        self.data = data
        self.dataset_name = dataset_name
        
    def get_initial_prompt(self) -> str:
        """Extract the initial user prompt from messages"""
        for msg in self.messages:
            if msg["role"] == "user":
                return msg["content"]
        return "No initial prompt found"
    
    def get_latest_response(self) -> str:
        """Get the latest assistant response"""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""


class BaseEvaluator(ABC):
    """Base class for dataset-specific User Simulator prompt generators"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
    
    @abstractmethod
    def generate_system_prompt(self, context: EvaluationContext) -> str:
        """Generate dataset-specific system prompt for user simulation"""
        pass
    
    @abstractmethod
    def generate_user_prompt(self, context: EvaluationContext) -> str:
        """Generate dataset-specific user prompt including relevant dataset info"""
        pass