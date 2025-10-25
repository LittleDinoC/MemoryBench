import json
import os
from typing import Dict, List, Optional, Union
from openai import OpenAI

from src.llms.base import BaseLlmConfig, LLMBase
from src.utils import extract_json


class VllmConfig(BaseLlmConfig):
    """
    Configuration class for vLLM-specific parameters.
    Inherits from BaseLlmConfig and adds vLLM-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        http_client_proxies: Optional[dict] = None,
        # vLLM-specific parameters
        vllm_base_url: Optional[str] = None,
    ):
        """
        Initialize vLLM configuration.

        Args:
            model: vLLM model to use, defaults to None
            temperature: Controls randomness, defaults to 0.1
            api_key: vLLM API key, defaults to None
            max_tokens: Maximum tokens to generate, defaults to 2000
            top_p: Nucleus sampling parameter, defaults to 0.1
            top_k: Top-k sampling parameter, defaults to 1
            enable_vision: Enable vision capabilities, defaults to False
            vision_details: Vision detail level, defaults to "auto"
            http_client_proxies: HTTP client proxy settings, defaults to None
            vllm_base_url: vLLM base URL, defaults to None
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            enable_vision=enable_vision,
            vision_details=vision_details,
            http_client_proxies=http_client_proxies,
        )

        # vLLM-specific parameters
        self.vllm_base_url = vllm_base_url or "http://localhost:8000/v1"



class VllmLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, VllmConfig, Dict]] = None):
        # Convert to VllmConfig if needed
        if config is None:
            config = VllmConfig()
        elif isinstance(config, dict):
            config = VllmConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, VllmConfig):
            # Convert BaseLlmConfig to VllmConfig
            config = VllmConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            raise ValueError("Model must be specified in the configuration.")

        self.config.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "vllm-api-key"
        base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.client = OpenAI(api_key=self.config.api_key, base_url=base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        extra_body=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using vLLM.

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
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )
        if response_format:
            params["response_format"] = response_format
        if extra_body:
            params["extra_body"] = extra_body

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)