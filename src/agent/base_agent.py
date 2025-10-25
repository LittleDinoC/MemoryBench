from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

from src.llms import LlmFactory


class BaseAgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm"] = Field(
        default="openai", 
        description="The LLM provider to use for the agent."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the LLM."
    )


class BaseAgent:
    def __init__(self, config: BaseAgentConfig = BaseAgentConfig()):
        self.config = config
        self.llm = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config,
        )
    
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        extra_body=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        lang: Literal["en", "zh"] = "en",
        retrieve_k: int = None,
        **kwargs,
    ):
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text". See https://docs.vllm.ai/en/latest/features/structured_outputs.html#online-serving-openai-api
            extra_body: Additional body parameters for the request, defaults to None. See https://docs.vllm.ai/en/latest/features/structured_outputs.html#online-serving-openai-api
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional vLLM-specific parameters.

        Returns:
            str: The generated response.
        """
        return self.llm.generate_response(
            messages=messages, 
            response_format=response_format,
            extra_body=extra_body,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )