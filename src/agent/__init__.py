import importlib
from typing import Optional, Union, Dict


from src.agent.base_agent import BaseAgentConfig
from src.agent.feedback import FeedbackAgentConfig


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class AgentFactory:
    """
    Factory for creating agent instances with appropriate configurations.
    """

    method_to_class = {
        "wo_memory": ("src.agent.base_agent.BaseAgent", BaseAgentConfig),
        "feedback": ("src.agent.feedback.FeedbackAgent", FeedbackAgentConfig),
        # "a-mem": ("src.agent.a_mem.AMemAgent", AMemAgentConfig),
        # "mem0": ("src.agent.mem0.Mem0Agent", Mem0AgentConfig),
        # "bm25": ("src.agent.bm25.BM25Agent", BM25AgentConfig),
    }

    @classmethod
    def create(cls, method_name: str, config: Optional[Dict] = None, **kwargs):
        """
        Create an agent instance with the appropriate configuration.

        Args:
            method_name (str): The method for agent (e.g., 'rag', 'feedback')
            config: Configuration object or dict. If None, will create default config
            **kwargs: Additional configuration parameters

        Returns:
            Configured agent instance

        Raises:
            ValueError: If provider is not supported
        """
        if method_name not in cls.method_to_class:
            raise ValueError(f"Unsupported method name: {method_name}")
        
        class_type, config_class = cls.method_to_class[method_name]
        agent_class = load_class(class_type)

        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config.update(kwargs)
            config = config_class(**config)
        else:
            pass

        return agent_class(config)