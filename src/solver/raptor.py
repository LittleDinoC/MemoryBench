from typing import List, Dict

from src.utils import if_memory_cached, mark_memory_cached
from src.agent.raptor import RAPTORAgent, RAPTORAgentConfig
from src.solver.base import BaseSolver

class RAPTORSolver(BaseSolver):
    AGENT_CLASS = RAPTORAgent

    def __init__(self, config: RAPTORAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "RAPTOR"

    def create_or_load_memory(self, dialogs: List[Dict]):
        """
        Create or load memory cache for Embedder system.

        Args:
            dialogs (List[Dict]): List of dialog data.
        """
        if not if_memory_cached(self.memory_cache_dir):
            print(f"Creating memory cache at {self.memory_cache_dir}")
            self.agent.reset_memories()
            print("Memorying dialogs with RAPTOR...")
            for dialog in dialogs:
                self.agent.add_conversation_to_memory(dialog["dialog"], dialog["test_idx"])
            print("Building RAPTOR tree...")
            self.agent.build_memory_system()
            print("Saving RAPTOR tree and memories to ", self.memory_cache_dir)
            self.agent.save_memories()
            mark_memory_cached(self.memory_cache_dir)
            print(f"\nSuccessfully add memory to Embedder system.")
        else:
            print("Loading memory cache from", self.memory_cache_dir)
            self.agent.load_memories() 