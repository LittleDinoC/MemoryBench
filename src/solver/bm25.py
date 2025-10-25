from tqdm import tqdm
from typing import List, Dict

from src.agent.bm25 import BM25Agent, BM25AgentConfig
from src.solver.base import BaseSolver


class BM25Solver(BaseSolver):
    AGENT_CLASS = BM25Agent

    def __init__(self, config: BM25AgentConfig, memory_cache_dir):
        super().__init__(config, memory_cache_dir)
        self.method_name = "BM25"
        self.current_conversation_memory_ids = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        return self._create_or_load_memory(dialogs, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(total=session_cnt, desc="Adding new conversation to memory")
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            for turn in session:
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                content = turn_date_time + "\n" + "Speaker "+ turn["speaker"] + "says : " + turn["text"]
                self.agent.add_memory(
                    content=content, 
                    doc_id=turn_date_time,
                )
                self.current_conversation_memory_ids.append(turn_date_time)
            session_idx += 1
            pbar.update(1)
        
    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)
            
    def delete_conversation_memory(self):
        for memory_id in self.current_conversation_memory_ids:
            self.agent.delete_memory(memory_id)
        self.current_conversation_memory_ids = []