from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from enum import Enum
import json
import logging

from src.agent.base_agent import BaseAgentConfig, BaseAgent
from src.agent.evaluators import EvaluatorFactory, EvaluationContext
from src.agent.user_simulator import UserSimulator


class ImplicitAction(str, Enum):
    """Implicit feedback action types"""
    like = "like"
    dislike = "dislike" 
    copy = "copy"
    none = "none"


class UserBehavior(str, Enum):
    """User behavior type enumeration"""
    continue_conversation = "continue_conversation"
    end_conversation = "end_conversation"


class UserFeedback(BaseModel):
    """User feedback model"""
    reasoning: str = Field(description="Detailed reasoning process for feedback generation")
    implicit_action: ImplicitAction = Field(description="User's implicit feedback action")
    behavior: UserBehavior = Field(description="User's behavioral intent: continue or end conversation")
    response: Optional[str] = Field(default=None, description="User's text response (only if continuing conversation)")


class FeedbackAgentConfig(BaseAgentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FeedbackAgent(BaseAgent):
    """
    FeedbackAgent using the consolidated UserSimulator utilities.
    """
    
    def __init__(self, config: FeedbackAgentConfig = FeedbackAgentConfig()):
        super().__init__(config)
        self.simulator = UserSimulator()
    
    def get_feedback(
        self, 
        messages: List[Dict[str, str]],
        data: Any,
        dataset_instance: Optional[Any] = None,
    ) -> Tuple[bool, str, ImplicitAction]:
        """
        Generate user feedback using either dataset-based evaluation (for needle-in-a-haystack tasks like Locomo)
        or LLM-based evaluation (for other datasets) enhanced with consolidated UserSimulator utilities.
        
        Args:
            messages: Chat message history between user and assistant
            data: Dataset item containing ground truth and other info
            dataset_instance: Optional dataset instance for direct evaluation (used for dataset-based feedback)
            
        Returns:
            Tuple[bool, str, ImplicitAction]: (should_end_conversation, user_response, implicit_action)
        """
        
        try:
            dataset_name = self._extract_dataset_name(data)
            
            # Check if this dataset requires dataset-based feedback
            if self._requires_dataset_based_feedback(dataset_name):
                return self._generate_dataset_based_feedback(messages, data, dataset_instance)
            else:
                return self._generate_llm_based_feedback(messages, data, dataset_name)
                
        except Exception as e:
            logging.error(f"Error generating user feedback: {e}")
            # Return default fallback
            return True, "I think this conversation is complete.", ImplicitAction.none
    
    def _extract_dataset_name(self, data: Any) -> str:
        """Extract dataset name from data item"""
        if hasattr(data, 'get'):
            return data.get('dataset_name', 'unknown')
        elif isinstance(data, dict):
            return data.get('dataset_name', 'unknown')
        else:
            return str(getattr(data, 'dataset_name', 'unknown'))
    
    def _requires_dataset_based_feedback(self, dataset_name: str) -> bool:
        """Check if dataset requires dataset-based feedback instead of LLM-based feedback"""
        # Needle-in-a-haystack datasets that cannot expose corpus to User Simulator
        dataset_based_datasets = ["locomo", "Locomo", "dialsim", "DialSim"]
        return any(name.lower() in dataset_name.lower() for name in dataset_based_datasets)
    
    def _generate_dataset_based_feedback(
        self, 
        messages: List[Dict[str, str]], 
        data: Any,
        dataset_instance: Optional[Any] = None
    ) -> Tuple[bool, str, ImplicitAction]:
        """Generate feedback using dataset's own evaluation method"""
        
        if dataset_instance is None:
            logging.warning("Dataset instance not provided for dataset-based feedback, falling back to default")
            return True, "Thank you for your response.", ImplicitAction.none
        
        # Get the last assistant response
        last_response = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
        
        if not last_response:
            return True, "No response to evaluate.", ImplicitAction.none
        
        # Get user prompt (original question)
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        
        # Route to dataset-specific evaluation using the provided instance
        dataset_name = self._extract_dataset_name(data)
        return self._evaluate_with_dataset_instance(dataset_instance, dataset_name, user_prompt, last_response, data)
    
    def _evaluate_with_dataset_instance(
        self, 
        dataset_instance: Any,
        dataset_name: str, 
        user_prompt: str, 
        llm_response: str, 
        data: Any
    ) -> Tuple[bool, str, ImplicitAction]:
        """Evaluate response using the provided dataset instance"""
        
        try:
            if "locomo" in dataset_name.lower():
                return self._evaluate_locomo_with_instance(dataset_instance, user_prompt, llm_response, data)
            elif "dialsim" in dataset_name.lower():
                return self._evaluate_dialsim_with_instance(dataset_instance, user_prompt, llm_response, data)
            else:
                # Future dataset-based evaluations can be added here
                logging.warning(f"No dataset-based evaluator for {dataset_name}, falling back to default")
                return True, "Thank you for your response.", ImplicitAction.none
                
        except Exception as e:
            logging.error(f"Error in dataset-based evaluation for {dataset_name}: {e}")
            return True, "Thank you for your response.", ImplicitAction.none
    
    def _evaluate_locomo_with_instance(
        self, 
        dataset_instance: Any,
        user_prompt: str, 
        llm_response: str, 
        data: Any
    ) -> Tuple[bool, str, ImplicitAction]:
        """Evaluate Locomo dataset response using provided dataset instance"""
        
        # Use the dataset's evaluation method directly
        info = data.get("info", {})
        evaluation_result = dataset_instance.evaluate_single(
            user_prompt=user_prompt,
            info=info,
            llm_response=llm_response
        )
        
        # Check if the answer is correct (f1 > 0.5 means correct)
        f1_score = evaluation_result.get("f1", 0)
        is_correct = f1_score > 0.5
        
        if is_correct:
            # Positive feedback - end conversation with upvote
            return True, "Great! That's correct. Thank you for the accurate answer.", ImplicitAction.like
        else:
            # Negative feedback - ask to retry with downvote
            return False, "That doesn't seem quite right. Could you please try again and provide a more accurate answer?", ImplicitAction.dislike
    
    def _evaluate_dialsim_with_instance(
        self, 
        dataset_instance: Any,
        user_prompt: str, 
        llm_response: str, 
        data: Any
    ) -> Tuple[bool, str, ImplicitAction]:
        """Evaluate DialSim dataset response using provided dataset instance"""
        
        # Use the dataset's evaluation method directly
        info = data.get("info", {})
        evaluation_result = dataset_instance.evaluate_single(
            user_prompt=user_prompt,
            info=info,
            llm_response=llm_response
        )
        
        # Check if the answer is correct (accuracy True means correct)
        is_correct = evaluation_result.get("accuracy", False)
        
        if is_correct:
            # Positive feedback - end conversation with upvote
            return True, "Great! That's correct. Thank you for the accurate answer.", ImplicitAction.like
        else:
            # Negative feedback - ask to retry with downvote
            return False, "That doesn't seem quite right. Could you please try again and provide a more accurate answer?", ImplicitAction.dislike
    
    def _generate_llm_based_feedback(
        self, 
        messages: List[Dict[str, str]], 
        data: Any, 
        dataset_name: str
    ) -> Tuple[bool, str, ImplicitAction]:
        """Generate feedback using LLM-based evaluation (original method)"""
        
        evaluator = EvaluatorFactory.create_evaluator(dataset_name)
        
        # Create evaluation context
        context = EvaluationContext(
            messages=messages,
            data=data,
            dataset_name=dataset_name
        )
        
        # Generate dataset-specific prompts using the evaluator
        system_prompt = evaluator.generate_system_prompt(context)
        user_prompt = evaluator.generate_user_prompt(context)
        
        # Prepare messages for LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Set up structured output schema
        schema = UserFeedback.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "user-feedback",
                "schema": schema
            }
        }
        
        # Generate response using the LLM
        response = self.generate_response(
            messages=llm_messages,
            response_format=response_format,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the JSON response
        feedback_data = json.loads(response)
        feedback = UserFeedback(**feedback_data)
        
        # Return based on behavior decision
        should_end = feedback.behavior == UserBehavior.end_conversation
        user_response = feedback.response if feedback.response else ""
        implicit_action = feedback.implicit_action
        
        return should_end, user_response, implicit_action
    
    # Legacy methods kept for backward compatibility, but now unused
    def _extract_input_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Extract the initial input prompt from the first user message"""
        for msg in messages:
            if msg["role"] == "user":
                return msg["content"]
        return "No initial prompt found"