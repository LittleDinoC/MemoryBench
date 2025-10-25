import os
import json
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent

from baselines.MemoryOS.memoryos_chromadb import Memoryos


class MemoryOSAgentConfig(BaseModel):
    llm_provider: str = Field(
        default="openai",
        description="LLM provider to use (e.g., openai, vllm)",
    )
    llm_config: dict = Field(
        default={},
        description="LLM configuration (e.g., model, url, api_key)",
    )
    memory_cache_dir: str = Field(
        default="./memoryos_data",
        description="Path to store MemoryOS data",
    )


class MemoryOSAgent(BaseAgent):
    def __init__(self, config: MemoryOSAgentConfig):
        self.config = config
        self.memory_system = Memoryos(
            user_id="user",
            openai_api_key=config.llm_config,
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") if config.llm_provider == "openai" else config.llm_config["vllm_base_url"],
            data_storage_path=config.memory_cache_dir,
            llm_model=config.llm_config["model"],
            assistant_id="assistant",
            short_term_capacity=7,  
            mid_term_heat_threshold=5,  
            retrieval_queue_capacity=7,
            long_term_knowledge_capacity=100,
            embedding_model_name="all-MiniLM-L6-v2", # Support Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, all-MiniLM-L6-v2
        )
        self.llm = LlmFactory.create(config.llm_provider, config.llm_config)

    def add_conversation_to_memory(
        self, 
        messages: List[Dict[str, str]], 
        conversation_idx: Union[int, str]=0
    ):
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)

        # Not support system prompt
        if messages[0]["role"] == "system":
            messages = messages[1:]

        for i in range(0, len(messages), 2):
            if messages[i]["role"] == "user" and (i + 1 >= len(messages) or messages[i + 1]["role"] == "assistant"):
                user_input = messages[i]["content"]
                agent_response = messages[i + 1]["content"] if i + 1 < len(messages) else ""
                self.memory_system.add_memory(user_input=user_input, agent_response=agent_response)

            elif messages[i]["role"] == "assistant" and i + 1 < len(messages) and messages[i + 1]["role"] == "user": 
                # I don't know why it happends: ['system', 'user', 'assistant', 'assistant', 'user', 'user', 'assistant', 'assistant', 'user', 'assistant']
                user_input = messages[i + 1]["content"]
                agent_response = messages[i]["content"]
                self.memory_system.add_memory(user_input=user_input, agent_response=agent_response) 
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        lang: Literal["zh", "en"] = "en",
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
        query = messages[-1]["content"]
        user_prompt = self.memory_system.get_user_prompt(query=query, lang=lang)
        messages[-1]["content"] = user_prompt
        response = self.llm.generate_response(messages=messages)
        return response

    def save_memories(self):
        pass
    
    def load_memories(self):
        pass

    def clear_all_memories(self): # TODO: bug? 
        if os.path.exists(self.config.memory_cache_dir):
            import shutil
            shutil.rmtree(self.config.memory_cache_dir)
        os.makedirs(self.config.memory_cache_dir, exist_ok=True)
        super().__init__(self.config)