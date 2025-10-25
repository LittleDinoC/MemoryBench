import os
import json
import hashlib
from tqdm import tqdm
from enum import Enum
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent

from baselines.mem0.mem0.memory.main import Memory
from baselines.mem0.mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from baselines.mem0.mem0.embeddings.configs import EmbedderConfig
from baselines.mem0.mem0.memory.main import _build_filters_and_metadata


class Mem0LlmProvider(str, Enum):
    openai = "openai"
    vllm = "vllm"


class Mem0EmbedderProvider(str, Enum):
    openai = "openai"
    huggingface = "huggingface"
    vllm = "vllm"


class Mem0AgentConfig(BaseModel):
    llm_provider: Mem0LlmProvider = Field(
        default=Mem0LlmProvider.openai,
        description="The LLM provider to use for the mem0 memory system."
    )
    llm_config: dict = Field(
        default={},
        description="Configuration parameters for the LLM."
    )
    embedder_provider: Mem0EmbedderProvider = Field(
        default=Mem0EmbedderProvider.openai,
        description="Provider of the embedding model",
    )
    embedder_config: dict = Field(
        default={},
        description="Configuration for the specific embedding model"
    )
    retrieve_k: int = Field(
        default=10,
        description="Number of memories to retrieve for a given query."
    )
    memory_cache_dir: str = Field(
        default=os.path.join(os.getcwd(), "mem0_history.db"),
        description="Path to the memory database."
    )
    memory_config: MemoryConfig = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self):
        llm_config = LlmConfig(provider=self.llm_provider.value, config=self.llm_config)
        embedder_config = EmbedderConfig(provider=self.embedder_provider.value, config=self.embedder_config)
        vector_store_config = VectorStoreConfig(
            config={
                "embedding_model_dims": embedder_config.config["embedding_dims"],
            }
        )
        self.memory_config = MemoryConfig(
            llm=llm_config, 
            embedder=embedder_config,
            vector_store=vector_store_config,
            history_db_path=os.path.join(self.memory_cache_dir, "mem0_history.db"), 
        )


class Mem0Agent(BaseAgent):
    def __init__(self, config: Mem0AgentConfig = Mem0AgentConfig()):
        config.update()

        self.config = config
        os.makedirs(config.memory_cache_dir, exist_ok=True)
        if self.config.llm_provider == Mem0LlmProvider.openai:
            api_key = config.llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
            base_url = config.llm_config.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        elif self.config.embedder_provider == Mem0EmbedderProvider.openai:
            api_key = config.embedder_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
            base_url = config.embedder_config.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        else:
            api_key = ""
            base_url = "https://api.openai.com/v1"
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url

        self.memory_system = Memory(config.memory_config)
        self.llm = LlmFactory.create(
            provider_name=self.config.llm_provider,
            config=self.config.llm_config,
        )
    
    def retrieve_memory(
        self, 
        query: str, 
        retrieve_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant memories based on a query.

        Args:
            query: The query string to search for relevant memories.
            retrieve_k: Optional; number of memories to retrieve. If None, uses the default from config.

        Returns:
            str: A formatted string of relevant memories.
        """
        if retrieve_k is None:
            retrieve_k = self.config.retrieve_k
        # print("#### ", retrieve_k, " ### ", self.config.retrieve_k, "####")
        relevant_memories = self.memory_system.search(query=query, limit=retrieve_k, user_id="user")
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        return memories_str

    def add_conversation_to_memory(
        self, 
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ):
        """
        Add a new memory based on the conversation messages.
        Only add to the memory bank after the entire conversation is completed.

        Args:
            messages: List of messages in the conversation. Each message is a dict with 'role' and 'content'.
        """
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        cnt = 100
        while cnt:
            cnt -= 1
            try:
                self.memory_system.add(messages, user_id="user")
                break
            except Exception as e:
                print(f"[Mem0] Error adding memory, retrying... {e}")
            

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
            lang: Language of the messages, either "en" for English or "zh" for Chinese.
        
        Returns:
            str: The agent's response to the messages.
        """
        if retrieve_k is None:
            retrieve_k = self.config.retrieve_k

        query = messages[-1]['content'] # the last message(from user) is the question
        memories_str = self.retrieve_memory(query, retrieve_k=retrieve_k)
        if lang == "en":
            user_prompt = f"""User Memories:
{memories_str}

User input: 
{query}

Based on the memories provided, respond naturally and appropriately to the user's input above."""
        elif lang == "zh":
            user_prompt = f"""用户记忆：
{memories_str}

用户输入：
{query}

请根据提供的记忆，自然且恰当地回应用户的上述输入。"""

        messages[-1]["content"] = user_prompt
        response = self.llm.generate_response(messages=messages)
        return response

    def save_memories(self):
        pass
    
    def load_memories(self):
        """
        Memory() only supports storing data in vector_store in db. 
        This function is added to import the information stored in db into vector_store.
        """

        def embed(data: str, action: str):
            return self.memory_system.embedding_model.embed(data, action)

        cursor = self.memory_system.db.connection.cursor()
        cursor.execute("SELECT * FROM history")
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        rows = [dict(zip(col_names, row)) for row in rows]

        if not rows:
            print("[Mem0] No memories to load from DB.")
            return

        print(f"[Mem0] Loading {len(rows)} memories from DB into vector store...")

        def solve_row(row):
            cnt = 20 
            while cnt:
                try:
                    data = row["new_memory"]
                    memory_id = row["memory_id"]
                    metadata = {
                        "data": data,
                        "hash": hashlib.md5(data.encode()).hexdigest(),
                    }
                    for key in ["created_at", "updated_at", "user_id", "agent_id", "run_id", "actor_id", "role"]:
                        if key in row and row[key] is not None:
                            metadata[key] = row[key]
                    metadata, filters = _build_filters_and_metadata(
                        user_id="user",
                        input_metadata=metadata,
                    )
                    if row["event"] == "ADD":
                        self.memory_system.vector_store.insert(
                            vectors=[embed(data, action="add")],
                            ids=[memory_id],
                            payloads=[metadata],
                        )
                    elif row["event"] == "UPDATE":
                        # for key in ["updated_at", "user_id", "agent_id", "run_id", "actor_id", "role"]:
                        #     if key in row:
                        #         metadata[key] = row[key]
                        self.memory_system.vector_store.update(
                            vector_id=memory_id,
                            vector=embed(data, action="update"),
                            payload=metadata,
                        )
                    elif row["event"] == "DELETE":
                        self.memory_system.vector_store.delete(vector_id=memory_id)
                    break
                except Exception as e:
                    cnt -= 1

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(solve_row, row): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(rows), desc="Loading memories"):
                try:
                    future.result()  # Raise any exceptions that occurred
                except Exception as e:
                    print(f"Error processing row {futures[future]}: {e}")

        print("[Mem0] Finished loading memories into vector store.")