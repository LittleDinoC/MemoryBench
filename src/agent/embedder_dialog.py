# Embedding-based Memory System

import os
import json
import time
import faiss
import numpy as np
from enum import Enum
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer


from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent


class EmbedderDialogAgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm"] = Field(
        default="openai", 
        description="The LLM provider to use for the agent."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the LLM."
    )
    embedder_provider: Literal["openai", "vllm", "huggingface"] = Field(
        default="vllm",
        description="The provider for the embedder."
    )
    embedder_model: str = Field(
        default="Qwen3/Qwen3-Embedding-0.6B",
        description="Model name for the embedder"
    )
    embedding_dim: int = Field(
        default=1024,
        description="Dimension of the embedding vectors"
    )
    embedder_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the embedder API, for 'openai' and 'vllm'"
    )
    embedder_api_key: Optional[str] = Field(
        default="",
        description="API key for the embedder, for 'openai' and 'vllm'"
    )
    memory_cache_dir: str = Field(
        default="./embedding_index",
        description="Path to save the embeddings"
    )
    retrieve_k: int = Field(
        default=10, 
        description="Number of top documents to retrieve from the Embedder index."
    )


class EmbedderDialogAgent(BaseAgent):
    def __init__(self, config: EmbedderDialogAgentConfig = EmbedderDialogAgentConfig()):
        self.config = config
        self.index_path = os.path.join(config.memory_cache_dir, "faiss.index")
        self.meta_path = os.path.join(config.memory_cache_dir, "meta.json")

        # load embedder
        if config.embedder_provider == "huggingface":
            self.embedder = SentenceTransformer(
                config.embedding_model, 
                device="cuda",
            )
        elif config.embedder_provider == "vllm" or config.embedder_provider == "openai":
            self.embedder_client = OpenAI(
                # api_key=config.embedder_api_key,
                base_url=config.embedder_base_url,
            )

        # initialize index and metadata storage
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            # self.load_memories()
            pass
        else:
            os.makedirs(config.memory_cache_dir, exist_ok=True)
            self.index = faiss.IndexFlatL2(config.embedding_dim)
            self.metadata = []  # save doc_id and text
        
        # load LLM
        self.llm = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config
        )

    def _embed(self, text: str) -> np.ndarray:
        """
        Embed a text string into a vector using the configured embedder.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedded vector.
        """
        sz = len(text)
        for part in range(100, 0, -1):
            aim_len = int(sz * (part + 1) // 100)
            encode_text = text[:aim_len]
            try:
                if self.config.embedder_provider == "huggingface":
                    embed = self.embedder.encode([encode_text], device="cuda")[0]
                elif self.config.embedder_provider == "vllm" or self.config.embedder_provider == "openai":
                    embed = self.embedder_client.embeddings.create(
                        input=[encode_text],
                        model=self.config.embedder_model,
                    ).data[0].embedding
                embed = np.array(embed, dtype=np.float32)
                return embed
            except Exception as e:
                print(f"Error embedding text: {e}, trying shorter text...")
                continue
        return np.zeros(self.config.embedding_dim, dtype=np.float32)

    def add_memory(self, content: str, doc_id=None, index_contet: str = None):
        """
        Add a text document to the Embedder index.

        Args:
            content (str): The text content to add to the index.
            doc_id (str, optional): An optional identifier for the document. If not provided, a unique ID will be generated.
            index_contet (str, optional): context to be indexed, if None, use content.
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.metadata)}"
        if index_contet is None:
            index_contet = content
        vector = self._embed(index_contet)
        self.metadata.append({"doc_id": doc_id, "content": content, "index_content": index_contet})
        try:
            self.index.add(np.array([vector]))
        except:
            time.sleep(1)  # wait for a second if there's an error
        # self.save_memories()

    def add_conversation_to_memory(
        self, 
        messages: List[Dict[str, str]], 
        conversation_idx: Union[int, str] = 0, 
    ):
        """
        Add a conversation to the memory system.
        
        Args:
            messages: List of messages in the conversation. Each message is a dict with 'role' and 'content'.
        """
        if isinstance(conversation_idx, int):
            conversation_idx = str(conversation_idx)
        for msg in messages:
            if msg["role"] == "user":
                query = msg["content"]
                break
        else:
            raise ValueError("No user message found in the conversation.")

        dialog_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )
        doc_id = f"conv_{conversation_idx}"
        self.add_memory(
            content=dialog_text, 
            doc_id=doc_id, 
            index_contet=query
        )

        # for msg_idx, msg in enumerate(messages):
        #     doc_id = f"conv_{conversation_idx}_{msg_idx}"
        #     content = f"Speaker {msg['role']} says: {msg['content']}"
        #     self.add_memory(content, doc_id)

    def retrieve_memory(self, content: str, k=10) -> List[str]:
        """
        Retrieve relevant documents from the Embedder index based on the input content.
        
        Args:
            content (str): The query content to search for.
            k (int): The number of top documents to retrieve.
        
        Returns:
            List[str]: A list of the retrieved documents.
        """
        if self.index.ntotal == 0:
            return []
        vector = self._embed(content)
        D, I = self.index.search(np.array([vector]), min(k * 2, len(self.metadata))) # search more to avoid lazy deleted results
        rets = []
        for i in I[0]:
            if i < len(self.metadata) and not self.metadata[i].get("deleted", False):
                rets.append(self.metadata[i]["content"])
            if len(rets) >= k:
                break
        return rets

    def save_memories(self):
        # Save the index and metadata to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load_memories(self):
        # Load the index and metadata from disk
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

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
        question = messages[-1]['content']
        docs = self.retrieve_memory(question, k=retrieve_k)
        context = "\n".join(docs)

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
        return self.llm.generate_response(messages=messages)

    def delete_memory(self, doc_id: str):
        '''
        lazy delete a memory by doc_id
        '''
        for i, meta in enumerate(self.metadata):
            if meta["doc_id"] == doc_id:
                self.metadata[i]["deleted"] = True
                break
        else:
            assert False, f"[INFO] doc_id {doc_id} not found in delete_memory"

    def rebuild_index(self):
        '''
        Rebuild the index to remove deleted memories
        '''
        valid_vectors = []
        valid_metadata = []
        for i, meta in enumerate(self.metadata):
            if not meta.get("deleted", False):
                valid_vectors.append(self._embed(meta["index_content"]))
                valid_metadata.append(meta)

        if valid_vectors:
            self.index = faiss.IndexFlatL2(self.config.embedding_dim)
            self.index.add(np.array(valid_vectors))
        else:
            self.index = faiss.IndexFlatL2(self.config.embedding_dim)

        print("[INFO] Index compacted, deleted entries removed.")
        print(f"[INFO] deleted {len(self.metadata) - len(valid_metadata)} entries, rest {len(valid_metadata)} entries.")
        self.metadata = valid_metadata