import importlib
from typing import Optional, Union, Dict

from src.solver.base import BaseAgentConfig
from src.solver.bm25 import BM25AgentConfig
from src.solver.bm25_dialog import BM25DialogAgentConfig
from src.solver.embedder import EmbedderAgentConfig
from src.solver.embedder_dialog import EmbedderDialogAgentConfig
from src.solver.a_mem import AMemAgentConfig
from src.solver.mem0 import Mem0AgentConfig
from src.solver.memoryos import MemoryOSAgentConfig
# from src.solver.raptor import RAPTORAgentConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class SolverFactory:
    method_to_class = {
        "wo_memory": ("src.solver.base.BaseSolver", "src.solver.base.BaseAgentConfig"),
        "bm25_message": ("src.solver.bm25.BM25Solver", "src.solver.bm25.BM25AgentConfig"),
        "bm25_dialog": ("src.solver.bm25_dialog.BM25DialogSolver", "src.solver.bm25_dialog.BM25DialogAgentConfig"),
        "embedder_message": ("src.solver.embedder.EmbedderSolver", "src.solver.embedder.EmbedderAgentConfig"),
        "embedder_dialog": ("src.solver.embedder_dialog.EmbedderDialogSolver", "src.solver.embedder_dialog.EmbedderDialogAgentConfig"),
        "a_mem": ("src.solver.a_mem.AMemSolver", "src.solver.a_mem.AMemAgentConfig"),
        "mem0": ("src.solver.mem0.Mem0Solver", "src.solver.mem0.Mem0AgentConfig"),
        "memoryos": ("src.solver.memoryos.MemoryOSSolver", "src.solver.memoryos.MemoryOSAgentConfig"),
        # "raptor": ("src.solver.raptor.RAPTORSolver", "src.solver.raptor.RAPTORAgentConfig"),
    }

    @classmethod
    def create(cls, method_name: str, config: Dict, **kwargs):
        if method_name not in cls.method_to_class:
            raise ValueError(f"Unknown method name: {method_name}")
        
        class_type, config_class_type = cls.method_to_class[method_name]
        solver_class = load_class(class_type)
        config_class = load_class(config_class_type)

        memory_cache_dir = kwargs.get("memory_cache_dir", None)
        if memory_cache_dir is not None and "memory_cache_dir" in config_class.__init__.__code__.co_varnames:
            config["memory_cache_dir"] = memory_cache_dir
        for key, value in kwargs.items():
            if key in config_class.__init__.__code__.co_varnames:
                config[key] = value
        agent_config = config_class(**config)
        return solver_class(agent_config, memory_cache_dir=memory_cache_dir)