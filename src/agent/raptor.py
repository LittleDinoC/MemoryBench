import os
import json
from enum import Enum
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent

from baselines.raptor.raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from baselines.raptor.raptor.SummarizationModels import TYCSummarizationModel
from baselines.raptor.raptor.EmbeddingModels import OpenAIRouterEmbeddingModel, VllmEmbeddingModel
from baselines.raptor.raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig


class RAPTORAgentConfig(BaseModel):
    llm_provider: str = Field(
        default="openai",
        description="The LLM provider to use for the mem0 memory system."
    )
    llm_config: dict = Field(
        default={},
        description="Configuration parameters for the LLM."
    )
    embedding_provider: Literal["openai", "vllm"] = Field(
        default="openai",
        description="Provider for embeddings",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings",
    )
    embedding_openai_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI API, e.g., 'https://api.openai.com/v1'",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI",
    )
    embedding_vllm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for vLLM API, e.g., 'http://localhost:8000'",
    )
    memory_cache_dir: str = Field(
        default="./raptor",
        description="Path to store data for the memory system.",
    )


class RAPTORAgent(BaseAgent):
    def __init__(self, config):
        self.config = config
        os.makedirs(config.memory_cache_dir, exist_ok=True)
        if self.config.embedding_provider == "openai":
            assert self.config.openai_api_key is not None, "OpenAI API key must be provided for OpenAI embedding model."
            embedding_model = OpenAIRouterEmbeddingModel(
                api_key=self.config.openai_api_key,
                base_url=self.config.embedding_openai_base_url,
                model=self.config.embedding_model
            )
        elif self.config.embedding_provider == "vllm":
            assert self.config.embedding_vllm_base_url is not None, "VLLM base URL must be provided for VLLM embedding model."
            embedding_model = VllmEmbeddingModel(
                api_key="",
                base_url=self.config.embedding_vllm_base_url,
                model=self.config.embedding_model
            )
        self.llm = LlmFactory.create(
            provider_name=self.config.llm_provider,
            config=self.config.llm_config,
        )
        summarization_model = TYCSummarizationModel(llm=self.llm)
        # cluster_tree_config = ClusterTreeConfig(
        #     max_tokens=500,
        #     summarization_length=500,
        # )

        self.ra_config = RetrievalAugmentationConfig(
            embedding_model=embedding_model,
            summarization_model=summarization_model,
            qa_model=self.llm,
            # tree_builder_config=cluster_tree_config,
        )
        self.memory_cache_dir = os.path.join(config.memory_cache_dir, "tree")
    
    def save_memories(self):
        self.memory_system.save(self.memory_cache_dir)
    
    def load_memories(self):
        self.memory_system = RetrievalAugmentation(
            config=self.ra_config,
            tree=self.memory_cache_dir,
        )
    
    def reset_memories(self):
        self.memory_system = RetrievalAugmentation(config=self.ra_config)
        self.memory_documents_cahce = []

    def add_conversation_to_memory(
        self, 
        messages: List[Dict[str, str]], 
        conversation_idx: Union[int, str] = 0,
    ):
        """
        Add a conversation to the memory system.
        
        Args:
            messages: List of messages in the conversation. Each message is a dict with 'role' and 'content'.
            conversation_idx: Index of the conversation in the dataset, used to create a unique identifier for the conversation. 
        """
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg_idx, msg in enumerate(messages):
            conversation_time = f"{conversation_idx}_{msg_idx}"
            content = f"Conversation {conversation_time}:\nSpeaker {msg['role']} says: {msg['content']}"
            self.memory_documents_cahce.append(content)

    def build_memory_system(self):
        """
        RAPTOR only support off-policy.
        It uses all documents and builds a memory tree.
        """
        self.memory_system.add_documents("\n\n".join(self.memory_documents_cahce))
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]],
        lang: Literal["en", "zh"] = "en",
        retrieve_k: int = None,
    ) -> str:
        """
        Generate a response to the user's question based on retrieved memories.
        
        Args:
            messages: List of messages in the conversation. Each message is a dict with 'role' and 'content'.
            lang: Language of the messages, either 'en' for English or 'zh' for Chinese.
        
        Returns:
            str: The agent's response to the messages.
        """
        question = messages[-1]['content'] # the last message(from user) is the question
        context, _layer_information = self.memory_system.retrieve(question)

        if lang == "en":
            user_prompt = f"""Context:
{context}

User: 
{question}

Based on the context provided, respond naturally and appropriately to the user's input above."""
        elif lang == "zh":
            user_prompt = f"""相关知识：
{context}

用户输入：
{question}

请根据提供的相关知识准确、自然地回答用户的输入。"""

        messages[-1]["content"] = user_prompt
        response = self.llm.generate_response(messages=messages)
        return response

        return answer