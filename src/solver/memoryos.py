from tqdm import tqdm
from typing import List, Dict

from src.agent.memoryos import MemoryOSAgent, MemoryOSAgentConfig
from src.solver.base import BaseSolver


class MemoryOSSolver(BaseSolver):
    AGENT_CLASS = MemoryOSAgent

    def __init__(self, config: MemoryOSAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "MemoryOS"
        # self.current_conversation_memory_ids = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        return super()._create_or_load_memory(dialogs, can_thread=False)

    def memory_locomo_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(total=session_cnt, desc="Adding new conversation to memory")
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            for turn in session:
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                speaker = turn["speaker"] 
                content = "Speaker "+ speaker + "says : " + turn["text"]
                if speaker == speaker_a:
                    self.agent.memory_system.add_memory(
                        user_input=content,
                        agent_response="",
                        timestamp=turn_date_time,
                    )
                else:
                    self.agent.memory_system.add_memory(
                        user_input="",
                        agent_response=content,
                        timestamp=turn_date_time,
                    )
            session_idx += 1
            pbar.update(1) 
        
    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(total=session_cnt, desc="Adding new conversation to memory")
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            for turn in session:
                turn_date_time = session_date_time + " Turn " + turn["dia_id"].split(":")[1]
                speaker = turn["speaker"] 
                content = "Speaker "+ speaker + "says : " + turn["text"]
                self.agent.memory_system.add_memory(
                    user_input=content,
                    agent_response="",
                    timestamp=turn_date_time,
                )
            session_idx += 1
            pbar.update(1) 
        pass
            
    def delete_conversation_memory(self):
        raise NotImplementedError("MemoryOS does not support deleting specific memories.")
    
    def clear_all_memories(self):
        # TODO 
        self.agent.clear_all_memories()