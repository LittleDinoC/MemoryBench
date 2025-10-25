from tqdm import tqdm
from typing import List, Dict

from src.agent.mem0 import Mem0Agent, Mem0AgentConfig
from src.solver.base import BaseSolver


class Mem0Solver(BaseSolver):
    AGENT_CLASS = Mem0Agent

    def __init__(self, config: Mem0AgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "Mem0"
        self.current_conversation_memories = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        super()._create_or_load_memory(dialogs, can_thread=False)

        rows = self.agent.memory_system.db.connection.execute("SELECT * FROM history LIMIT 5").fetchall()
        print("Sample rows:", rows)
        tables = self.agent.memory_system.db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print("Tables:", tables)
    

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
                for _ in range(5):
                    try:
                        if speaker == speaker_a:
                            memory_ret = self.agent.memory_system.add(
                                messages=[
                                    {"role": "user", "content": content}
                                ],
                                metadata={
                                    "speaker": speaker,
                                    "timestamp": session_date_time,
                                    "turn_date": turn_date_time,
                                }, 
                                user_id="user",
                            )
                        else:
                            memory_ret = self.agent.memory_system.add(
                                messages=[
                                    {"role": "assistant", "content": content}
                                ],
                                metadata={
                                    "speaker": speaker,
                                    "timestamp": session_date_time,
                                    "turn_date": turn_date_time,
                                },
                                user_id="user",
                            )
                        break
                    except Exception as e:
                        print(e)
                for ret in memory_ret["results"]:
                    self.current_conversation_memories.append(ret["id"])
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
                for _ in range(5):
                    try:
                        memory_ret = self.agent.memory_system.add(
                            messages=[
                                {"role": "user", "content": content}
                            ],
                            metadata={
                                "speaker": speaker,
                                "timestamp": session_date_time,
                                "turn_date": turn_date_time,
                            }, 
                            user_id="user",
                        )
                        break
                    except Exception as e:
                        print(e)
                for ret in memory_ret["results"]:
                    self.current_conversation_memories.append(ret["id"])
            session_idx += 1
            pbar.update(1)
            
    def delete_conversation_memory(self):
        if len(self.current_conversation_memories) > 0:
            self.current_conversation_memories = list(set(self.current_conversation_memories))
            for mem_id in self.current_conversation_memories:
                try:
                    self.agent.memory_system.delete(memory_id=mem_id)
                except:
                    print(f"Memory ID {mem_id} not found for deletion.")
        print("### [Rest] Current memory number:", len(self.agent.memory_system.get_all(user_id="user")["results"]), "###")
        self.current_conversation_memories = []