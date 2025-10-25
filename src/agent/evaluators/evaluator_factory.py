from typing import Dict, Any, Type
from .base_evaluator import BaseEvaluator
from .lexeval_evaluator import LexEvalEvaluator
from .writingbench_evaluator import WritingBenchEvaluator
from .hellobench_evaluator import HelloBenchEvaluator
from .ideabench_evaluator import IdeaBenchEvaluator
from .jrel_evaluator import JRELEvaluator
from .limitgen_evaluator import LimitGenEvaluator
from .judge_evaluator import JuDGEEvaluator
from .writingprompts_evaluator import WritingPromptsEvaluator
from .nfcats_evaluator import NFCatsEvaluator

class EvaluatorFactory:
    """Factory class for creating dataset-specific evaluators"""
    
    _evaluators: Dict[str, Type[BaseEvaluator]] = {
        "lexeval": LexEvalEvaluator,
        "writingbench": WritingBenchEvaluator,
        "hellobench": HelloBenchEvaluator,
        "ideabench": IdeaBenchEvaluator,
        "jre-l": JRELEvaluator,
        "limitgen": LimitGenEvaluator,
        "judge": JuDGEEvaluator,
        "writingprompts": WritingPromptsEvaluator,
        "nfcats": NFCatsEvaluator,
    }
    
    @classmethod
    def create_evaluator(cls, dataset_name: str) -> BaseEvaluator:
        """Create appropriate evaluator based on dataset name"""
        # Normalize dataset name to handle variations
        normalized_name = cls._normalize_dataset_name(dataset_name)
        
        if normalized_name in cls._evaluators:
            return cls._evaluators[normalized_name]()
        else:
            # Fallback to a default evaluator (could be LexEval as it's the most general)
            return LexEvalEvaluator()
    
    @classmethod
    def _normalize_dataset_name(cls, dataset_name: str) -> str:
        """Normalize dataset name to match factory keys"""
        dataset_name = dataset_name.lower()
        
        # Handle different variations of dataset names
        if "lexeval" in dataset_name or "legal" in dataset_name:
            return "lexeval"
        elif "writingbench" in dataset_name:
            return "writingbench"
        elif "hellobench" in dataset_name:
            return "hellobench"
        elif "ideabench" in dataset_name or "idea" in dataset_name:
            return "ideabench"
        elif "jre-l" in dataset_name or "jrel" in dataset_name or "journalist" in dataset_name:
            return "jre-l"
        elif "limitgen" in dataset_name or "limit" in dataset_name:
            return "limitgen"
        elif "judge" in dataset_name or "judgment" in dataset_name:
            return "judge"
        elif "writingprompts" in dataset_name or "writing_prompts" in dataset_name:
            return "writingprompts"
        elif "nfcats" in dataset_name or "nf-cats" in dataset_name:
            return "nfcats"
        else:
            # Default to lexeval for unknown datasets
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    @classmethod
    def register_evaluator(cls, dataset_name: str, evaluator_class: Type[BaseEvaluator]):
        """Register a new evaluator for a dataset"""
        cls._evaluators[dataset_name.lower()] = evaluator_class
    
    @classmethod
    def list_supported_datasets(cls) -> list:
        """List all supported dataset types"""
        return list(cls._evaluators.keys())