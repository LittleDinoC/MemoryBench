import os
import json
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

import whoosh
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup

from src.llms import LlmFactory
from src.agent.base_agent import BaseAgent


class BM25AgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm"] = Field(
        default="openai", 
        description="The LLM provider to use for the agent."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the LLM."
    )
    memory_cache_dir: str = Field(
        default="./bm25_index",
        description="Path to save the BM25 index."
    )
    retrieve_k: int = Field(
        default=10,
        description="Number of top documents to retrieve from the BM25 index."
    )
    bm25_k1: float = Field(
        default=1.5,
        description="BM25 k1 parameter for term frequency scaling."
    )
    bm25_b: float = Field(
        default=0.75,
        description="BM25 b parameter for length normalization."
    )


class BM25Agent(BaseAgent):
    def __init__(self, config: BM25AgentConfig = BM25AgentConfig()):
        self.config = config
        # Initialize the BM25 index
        self.index_path = config.memory_cache_dir
        self.schema = Schema(
            doc_id=ID(stored=True, unique=True),
            content=TEXT(stored=True),
        )
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
            self.index = whoosh.index.create_in(self.index_path, self.schema)
        else:
            self.index = whoosh.index.open_dir(self.index_path)
        # Initialize LLM for inference
        self.llm = LlmFactory.create(
            provider_name=config.llm_provider, 
            config=config.llm_config
        )
    
    def _count_docs(self):
        """
        Count the number of documents in the BM25 index.
        
        Returns:
            int: The number of documents in the index.
        """
        with self.index.searcher() as searcher:
            return searcher.doc_count_all()
    
    def add_memory(self, content: str, doc_id=None):
        """
        Add a text document to the BM25 index.

        Args:
            content (str): The text content to add to the index.
        """
        if doc_id is None:
            doc_id = f"doc_{self._count_docs()}"
        writer = self.index.writer()
        writer.add_document(doc_id=doc_id, content=content)
        writer.commit()

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
        
        writer = self.index.writer() 
        for msg_idx, msg in enumerate(messages):
            doc_id = f"conv_{conversation_idx}_{msg_idx}"
            content = f"Speaker {msg['role']} says: {msg['content']}"
            writer.add_document(doc_id=doc_id, content=content)
        writer.commit()

    def retrieve_memory(self, content, k=10):
        """
        Retrieve relevant documents from the BM25 index based on the input content.
        
        Args:
            content (str): The query content to search for.
            k (int): The number of top documents to retrieve.
        
        Returns:
            List[str]: A list of the retrieved documents.
        """
        with self.index.searcher() as searcher:
            query = QueryParser("content", self.index.schema, group=OrGroup).parse(content)
            results = searcher.search(query, limit=k)
            return [result['content'] for result in results]
    
    def save_memories(self):
        pass
    
    def load_memories(self):
        pass
        # if not os.path.exists(self.index_path):
        #     raise FileNotFoundError(f"BM25 index not found at {self.index_path}")
        # self.index = whoosh.index.open_dir(self.index_path)
    
    # def clear_memory(self):
    #     """
    #     Clear the BM25 index.
    #     """
    #     import shutil
    #     if os.path.exists(self.index_path):
    #         shutil.rmtree(self.index_path)
    #         os.makedirs(self.index_path)
    #         self.index = whoosh.index.create_in(self.index_path, self.schema)

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
        response = self.llm.generate_response(messages=messages)
        return response

    def delete_memory(self, doc_id):
        writer = self.index.writer()
        writer.delete_by_term("doc_id", doc_id)
        writer.commit()
    
    def clear_all_memories(self):
        import shutil
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path)
        self.index = whoosh.index.create_in(self.index_path, self.schema)