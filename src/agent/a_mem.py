import os
import json
import pickle
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Literal, Union, Tuple
from pydantic import BaseModel, Field

from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent

from baselines.A_mem.agentic_memory.memory_system import AgenticMemorySystem


class AMemAgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm"] = Field(
        default="openai", 
        description="The LLM provider to use for the agent inference."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the inference LLM."
    )
    retrieve_k: int = Field(
        default=10,
        description="Number of memories to retrieve for a given query."
    )
    memory_cache_dir: str = Field(
        default="./a-mem",
        description="Directory to store cached memory data.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.llm_provider == "openai":
            api_key = self.llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
            os.environ["OPENAI_API_KEY"] = api_key
        elif self.llm_provider == "vllm":
            vllm_url = self.llm_config.get("vllm_base_url", os.environ.get("VLLM_BASE_URL", ""))
            os.environ["VLLM_BASE_URL"] = vllm_url


class AMemAgent(BaseAgent):
    def __init__(self, config: AMemAgentConfig = AMemAgentConfig()):
        self.config = config
        self.memory_cache_dir = config.memory_cache_dir
        self.memory_system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",             # Embedding model for ChromaDB
            llm_backend=config.llm_provider,    # LLM backend (openai/ollama)
            llm_model=config.llm_config["model"],         # LLM model name
        )
        self.llm = LlmFactory.create(
            provider_name=self.config.llm_provider,
            config=self.config.llm_config,
        )

    def add_memory(self, content, time=None):
        return self.memory_system.add_note(content, time=time)

    def add_conversation_to_memory(self, messages: List[Dict[str, str]], conversation_idx: Union[int, str]):
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
            content = f"Speaker {msg['role']} says: {msg['content']}"
            self.add_memory(content=content, time=conversation_time)
    
    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text. 

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""
        # different from original A-mem:
        # add a retry mechanism to avoid potential issues with LLM response
        while True: 
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "string",
                                }
                            },
                            "required": ["keywords"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            try:
                keywords = json.loads(response)["keywords"]
                return keywords
            except:
                continue

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
        if retrieve_k is None:
            retrieve_k = self.config.retrieve_k

        question = messages[-1]['content'] # the last message(from user) is the question
        keywords = self.generate_query_llm(question)
        context = self.retrieve_memory(keywords, k=retrieve_k)

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

    def save_memories(self):
        memory_cache_file = os.path.join(
            self.memory_cache_dir, 
            f"memory_cache.pkl"
        )
        retriever_cache_file = os.path.join(
            self.memory_cache_dir, 
            f"retriever_cache.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            self.memory_cache_dir, 
            f"retriever_cache_embeddings.npy"
        )
        os.makedirs(self.memory_cache_dir, exist_ok=True)
        with open(memory_cache_file, "wb") as fout:
            pickle.dump(self.memory_system.memories, fout)
        self.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
        print(f"\nSuccessfully saved memory cache to {memory_cache_file}, total {len(self.memory_system.memories)}")

    
    def load_memories(self):
        memory_cache_file = os.path.join(
            self.memory_cache_dir, 
            f"memory_cache.pkl"
        )
        retriever_cache_file = os.path.join(
            self.memory_cache_dir, 
            f"retriever_cache.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            self.memory_cache_dir, 
            f"retriever_cache_embeddings.npy"
        )
        assert os.path.exists(memory_cache_file), f"Memory cache file {memory_cache_file} does not exist."
        assert os.path.exists(retriever_cache_file), f"Retriever cache file {retriever_cache_file} does not exist."
        assert os.path.exists(retriever_cache_embeddings_file), f"Retriever cache embeddings file {retriever_cache_embeddings_file} does not exist."

        print(f"Loading memory cache from {memory_cache_file}")
        with open(memory_cache_file, 'rb') as f:
            cached_memories = pickle.load(f)
        # Restore memories to agent
        self.memory_system.memories = cached_memories
        if os.path.exists(retriever_cache_file):
            print(f"Found retriever cache files:")
            print(f"  - Retriever cache: {retriever_cache_file}")
            print(f"  - Embeddings cache: {retriever_cache_embeddings_file}")
            self.memory_system.retriever = self.memory_system.retriever.load(retriever_cache_file,retriever_cache_embeddings_file)
        else:
            print(f"No retriever cache found at {retriever_cache_file}, loading from memory")
            self.memory_system.retriever = self.memory_system.retriever.load_from_local_memory(cached_memories, 'all-MiniLM-L6-v2')

    # def clear_all_memories(self):
    #     # [TODO]
    #     self.memory_system = AgenticMemorySystem(
    #         model_name="all-MiniLM-L6-v2",      # [TODO] Embedding model for ChromaDB
    #         llm_backend=self.config.llm_provider,    # LLM backend (openai/ollama)
    #         llm_model=self.config.llm_config["model"],         # LLM model name
    #         # chroma_db_path=config.chroma_db_path
    #     )
    
    def delete_to_end(self, delete_memories: List[Tuple[str, str]]):
        """
        delete_memories: List of (memory_id, memory_content) to delete 
        """
        del_corpus_ids = []
        for memory_id, memory_content in delete_memories:
            if memory_id in self.memory_system.memories:
                del self.memory_system.memories[memory_id]
            assert memory_content in self.memory_system.retriever.document_ids, \
                f"Memory content '{memory_content}' not found in retriever documents."
            del_corpus_ids.append(
                self.memory_system.retriever.document_ids[memory_content]
            )

        # delete the memories from retriever
        # assert len(set(del_corpus_ids)) == len(del_corpus_ids), "Document IDs to delete are not unique."
        assert max(del_corpus_ids) == len(self.memory_system.retriever.corpus) - 1, "Document IDs to delete are not the last ones in the corpus."

        # print(len(self.memory_system.retriever.corpus), self.memory_system.retriever.embeddings.shape)

        for _, doc in delete_memories[::-1]:
            if doc not in self.memory_system.retriever.document_ids:
                continue

            idx = self.memory_system.retriever.document_ids[doc]
            self.memory_system.retriever.embeddings = np.delete(
                self.memory_system.retriever.embeddings, idx, axis=0
            )
            del self.memory_system.retriever.document_ids[doc]
        self.memory_system.retriever.corpus = self.memory_system.retriever.corpus[:-len(delete_memories)]
    
        print(len(self.memory_system.retriever.corpus), self.memory_system.retriever.embeddings.shape)
