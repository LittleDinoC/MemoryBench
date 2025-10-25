from typing import List, Dict, Any
import logging


class UserSimulator:
    """
    Consolidated User Simulator providing common patterns and utilities
    for all dataset evaluators.
    """
    
    @staticmethod
    def extract_data_field(data: Any, *field_path: str, default: Any = None) -> Any:
        """
        Safely extract nested field from data with comprehensive error handling.

        Args:
            data: Dataset item (dict-like or object)
            *field_path: Nested field path (e.g., 'info', 'golden_answer')
            default: Default value if field not found

        Returns:
            Extracted field value or default
        """
        try:
            current = data

            # Handle different data formats
            if hasattr(current, 'get') and callable(getattr(current, 'get')):
                for field in field_path:
                    current = current.get(field, {})
            elif isinstance(current, dict):
                for field in field_path:
                    current = current.get(field, {})
            else:
                for field in field_path:
                    current = getattr(current, field, {})

            if current == {} or current is None:
                logging.warning(f"Field not found: {'.'.join(field_path)}")
                return default

            return current

        except (AttributeError, TypeError, KeyError) as e:
            logging.warning(f"Failed to extract field {'.'.join(field_path)}: {e}")
            return default
    
    @staticmethod
    def format_conversation_history(messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages into a readable history string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Formatted conversation string
        """
        conversation = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n"
        
        return conversation.strip()
    
    @staticmethod
    def build_base_system_prompt(
        user_persona: str,
        domain_expertise: str,
        evaluation_criteria: List[str],
        additional_context: str = ""
    ) -> str:
        """
        Build a standardized system prompt with common elements.
        
        Args:
            user_persona: Description of the simulated user
            domain_expertise: User's domain knowledge and capabilities
            evaluation_criteria: List of factors to consider
            additional_context: Any additional domain-specific context
        
        Returns:
            Formatted system prompt
        """
        criteria_text = "\n".join(f"- {criteria}" for criteria in evaluation_criteria)
        
        prompt = f"""{user_persona}

{domain_expertise}

CRITICAL: Always focus on the initial prompt/request as the primary context for evaluation. The conversation should stay aligned with the original user intent.

IMPORTANT: DO NOT REPEAT QUESTIONS OR REQUESTS that have already been asked in the conversation. Avoid asking the same question multiple times.

IMPORTANT: Always start your reasoning process first, then provide the other feedback elements.

Your response should include:
1. Reasoning: Detailed analysis of the assistant's response quality and accuracy (always consider how well it addresses the initial prompt)
2. Implicit action: What the user would likely do (like, dislike, copy, none)
3. Behavior decision: Whether to continue or end the conversation
4. Response: What the user would say (only if continuing the conversation)

Consider factors like:
{criteria_text}"""
        
        if additional_context:
            prompt += f"\n\n{additional_context}"
        
        return prompt
    
    @staticmethod
    def build_base_user_prompt(
        context,
        task_description: str,
        evaluation_context: str = ""
    ) -> str:
        """
        Build a standardized user prompt with common elements.
        
        Args:
            context: Evaluation context containing messages and data
            task_description: Description of the user's task/goal
            evaluation_context: Internal context for evaluation
        
        Returns:
            Formatted user prompt
        """
        conversation = UserSimulator.format_conversation_history(context.messages)
        
        # Extract language from data
        lang = UserSimulator.extract_data_field(context.data, 'lang', default='en')
        
        # Always emphasize focus on initial prompt
        prompt_context = "CRITICAL: Focus on the initial request as the core topic that should be the primary focus throughout this entire conversation. All responses should be evaluated based on how well they address this original user intent."
        
        prompt = f"""Analyze this conversation and predict the user's response:

The user is {task_description}. {prompt_context}

Conversation History:
{conversation}

"""
        
        # Add evaluation context if provided
        if evaluation_context:
            prompt += f"""EVALUATION CONTEXT:
{evaluation_context}

"""
        
        # Add language instruction based on dataset language
        if lang == 'zh':
            language_instruction = "IMPORTANT: If you provide a response (when behavior is continue_conversation), it must be in Chinese (中文)."
        elif lang == 'en':
            language_instruction = "IMPORTANT: If you provide a response (when behavior is continue_conversation), it must be in English."
        else:
            language_instruction = f"IMPORTANT: If you provide a response (when behavior is continue_conversation), it must be in the conversation language (lang: {lang})."
        
        # Standard JSON format instructions with language requirement
        prompt += f"""{language_instruction}

Please provide a realistic user response in strict JSON format:

{{
  "reasoning": "Detailed analysis of the assistant's response quality and accuracy (MUST evaluate how well it addresses the initial request)",
  "implicit_action": "like" | "dislike" | "copy" | "none",
  "behavior": "continue_conversation" | "end_conversation", 
  "response": "What the user would say next (string or null if ending)"
}}

Requirements:
- reasoning: Always provide detailed analysis first. CRITICAL: Always assess how well the assistant's response addresses the initial request and stays focused on the original user intent.
- implicit_action: Must be exactly one of: like, dislike, copy, none
- behavior: Must be exactly: continue_conversation or end_conversation
- response: Text if continuing, null if ending conversation. Must match the conversation language. IMPORTANT: Do not repeat questions or requests that have already been made in the conversation.

Respond with valid JSON only."""
        
        return prompt
    
