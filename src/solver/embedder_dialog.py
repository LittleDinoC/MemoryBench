from tqdm import tqdm
from typing import List, Dict

from src.agent.embedder_dialog import EmbedderDialogAgent, EmbedderDialogAgentConfig
from src.solver.base import BaseSolver


class EmbedderDialogSolver(BaseSolver):
    AGENT_CLASS = EmbedderDialogAgent

    def __init__(self, config: EmbedderDialogAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "EmbedderDialog"
        self.current_conversation_memory_ids = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        return super()._create_or_load_memory(dialogs, can_thread=False)
    
    def memory_locomo_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(total=session_cnt, desc="Adding new conversation to memory")
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]
            session_text = f"Coversation [{session_date_time}]:\n"
            for turn in session:
                session_text += "Speaker "+ turn["speaker"] + "says : " + turn["text"] + "\n"
            self.agent.add_memory(
                content=session_text,
                doc_id=session_date_time,
            )
            self.current_conversation_memory_ids.append(session_date_time)
            session_idx += 1
            pbar.update(1)
    
    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)
            
    def delete_conversation_memory(self):
        if len(self.current_conversation_memory_ids) > 0:
            for memory_id in self.current_conversation_memory_ids:
                self.agent.delete_memory(memory_id)
            self.agent.rebuild_index()
        self.current_conversation_memory_ids = []