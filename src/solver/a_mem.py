from tqdm import tqdm
from typing import List, Dict

from src.agent.a_mem import AMemAgent, AMemAgentConfig
from src.solver.base import BaseSolver


class AMemSolver(BaseSolver):
    AGENT_CLASS = AMemAgent

    def __init__(self, config: AMemAgentConfig, memory_cache_dir):
        super().__init__(config, memory_cache_dir)
        self.method_name = "A-Mem"
        self.current_conversation_memories = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        return super()._create_or_load_memory(dialogs, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(total=session_cnt, desc="Adding new conversation to memory")
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            for turn in session:
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                content = "Speaker "+ turn["speaker"] + "says : " + turn["text"]
                memory_id, memory_content = self.agent.add_memory(
                    content=content, 
                    time=turn_date_time
                )
                self.current_conversation_memories.append((memory_id, memory_content))
            session_idx += 1
            pbar.update(1)
    
    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)
            
    def delete_conversation_memory(self):
        if len(self.current_conversation_memories) > 0:
            self.agent.delete_to_end(
                delete_memories=self.current_conversation_memories
            )
        self.current_conversation_memories = []