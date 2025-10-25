import json
from collections import deque
from typing import Optional
try:
    from .storage_provider import ChromaStorageProvider
except ImportError:
    from storage_provider import ChromaStorageProvider

class ShortTermMemory:
    def __init__(self, storage_provider: ChromaStorageProvider, max_capacity=10):
        self.max_capacity = max_capacity
        self.memory = deque(maxlen=max_capacity)
        self.storage = storage_provider
        self.load()

    def add_qa_pair(self, qa_pair: dict):
        # Use storage provider's add method, which only updates the in-memory metadata
        self.storage.add_short_term_memory(qa_pair)
        # Also update local deque
        self.memory.append(qa_pair)
        print(f"ShortTermMemory: Added QA. User: {qa_pair.get('user_input','')[:30]}...")

    def get_all(self) -> list:
        return list(self.memory)

    def is_full(self) -> bool:
        # Use storage provider's method for accurate check against its in-memory list
        return self.storage.is_short_term_full(self.max_capacity)

    def pop_oldest(self) -> Optional[dict]:
        # Use storage provider's pop method, which only updates the in-memory metadata
        msg = self.storage.pop_oldest_short_term()
        if msg:
            # Also update local deque to stay in sync
            if self.memory:
                self.memory.popleft()
            print("ShortTermMemory: Evicted oldest QA pair.")
            return msg
        return None

    def load(self):
        try:
            # Load from the shared storage provider
            loaded_memory = self.storage.get_short_term_memory(self.max_capacity)
            self.memory = loaded_memory
            
            print(f"ShortTermMemory: Loaded {len(self.memory)} QA pairs.")
            
        except Exception as e:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: Error loading: {e}. Initializing new memory.") 