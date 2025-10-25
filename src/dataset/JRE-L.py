from src.dataset.base import BaseDataset
from src.llms import LlmFactory
from typing import List, Dict, Any, Type
import textstat
from pydantic import Field, BaseModel
import re
import jsonlines
import json
from bert_score import BERTScorer
import evaluate
import os

from src.llms import LlmFactory
from pydantic import BaseModel, Field


class BaseAgentConfig(BaseModel):
    llm_provider: str = Field(
        default="openai", 
        description="The LLM provider to use for the agent."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the LLM."
    )


prompt = """Write a report of this paper in journalistic style.\n\n"""

merge_score_prompt = """You are an expert in science communication and text evaluation. Your task is to evaluate the quality of an automatically generated popular science article based on the provided source document, a reference article, and a set of pre-calculated metrics.

## Source Document (Input)
{INPUT_TEXT}

## Generated Popular Science Article (Output)
{GENERATED_ARTICLE}

## Abstract of Reference Popular Science Article (Golden Passage)
{GOLDEN_PASSAGE}

## Evaluation Metrics
Below are the calculated metrics comparing the 'Generated Article' to the 'Reference Article' or analyzing its intrinsic qualities.

Rouge-L (Score range: 0.00 to 1.00)
Score: {ROUGE_L}
Meaning: Measures the overlap of the longest common word sequence between the generated and reference articles. A higher score indicates better factual consistency and content preservation.

BERTScore-F1 (Score range: 0.00 to 1.00)
Score: {BERTSCORE_F1}
Meaning: Measures the semantic similarity between the generated and reference articles using contextual language models. A higher score indicates that the core meaning is better captured, even with different wording.

CLI (Coleman-Liau Index)
Score: {CLI}
Meaning: Estimates the U.S. grade level required to understand the text. For popular science, a lower score (e.g., 8-12) is generally desirable, indicating better readability and accessibility for a general audience.

FKGL (Flesch-Kincaid Grade Level)
Score: {FKGL}
Meaning: Similar to CLI, this metric also estimates the required U.S. grade level for comprehension. Lower scores suggest the text is easier to read. A score between 8 and 12 means standard readability for a general audience.

DCRS (Dale-Chall Readability Score)
Score: {DCRS}
Meaning: Estimates readability based on a list of 3000 common words. A lower score indicates the text is easier to understand. A score of 4.9 or lower indicates that the passage is very easy to read for fourth-grade students. A score between 9.0 and 9.9 indicates that the passage is at a college readability level.

## Task
Based on a holistic review of the input, output, golden passage, and all the metrics provided above, provide a single integer score from 1 to 10 to represent the overall quality of the generated popular science article. Consider its accuracy, readability, coherence, and faithfulness to the source material.
- 1: Represents extremely poor quality (e.g., completely irrelevant, factually incorrect, nonsensical, or unreadable).
- 10: Represents excellent quality (e.g., accurate, easy to understand for a layperson, well-structured, engaging, and highly faithful to the source, nearly indistinguishable from the reference).

Your response should be only a single integer.

## Final Score"""

    
class JRE_L_Dataset(BaseDataset):

    def __init__(self, data_path: str = None, dataset_name: str = "JRE-L", bert_score_model: str = 'roberta-base', test_metrics: List[str] = ["Rouge-L", "BERTScore-F1", "CLI", "FKGL", "DCRS"], max_output_len: int = 8192, eval_mode: bool = True):
        self.dataset_name = dataset_name
        # self.feedback_type = feedback_type
        super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
        if eval_mode:
            self.scorer = BERTScorer(model_type=bert_score_model, device='cuda:0')
            self.rouge = evaluate.load('rouge')
        else:
            self.scorer = None
            self.rouge = None
            
        config = BaseAgentConfig(
            llm_config = {
                "openai_base_url": os.getenv("EVALUATE_BASE_URL"),
                "model": os.getenv("EVALUATE_MODEL"),
                "api_key": os.getenv("EVALUATE_API_KEY"),
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        )
        self.openai_model = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config,
        )
        

    # def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        # max_len = 1024
        # raw_data = []
        # len_ = 0
        # with jsonlines.open(self.data_path) as reader:
        #     for idx, obj in enumerate(reader):
        #         raw_data.append({
        #             "test_idx": len_,
        #             "input_prompt": prompt + f"""### Meta Info\nTitle: {obj['sc-title']}\n### Content\n{" ".join(obj['sc-abstract'].split(" ")[0:max_len])}""",
        #             "dataset_name": self.dataset_name,
        #             # "feedback_type": self.feedback_type,
        #             "lang": "en",
        #             "info": {
        #                 'sc-title': obj['sc-title'],
        #                 'sc-abstract': obj['sc-abstract'],
        #                 'pr-title': obj['pr-title'],
        #                 'pr-abstract': obj['pr-summary'],
        #             }
        #         })
        #         len_ += 1
        # return raw_data

    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        scores = info.copy()
        scores['Rouge-L'] = self.rouge.compute(predictions=[llm_response], references=[info['pr-abstract']])['rougeL']
        scores['BERTScore-F1'] = self.scorer.score([llm_response], [info['pr-abstract']])[2][0].item()
        scores['CLI'] = textstat.coleman_liau_index(llm_response)
        scores['FKGL'] = textstat.flesch_kincaid_grade(llm_response)
        scores['DCRS'] = textstat.dale_chall_readability_score(llm_response)
        return scores
    
    def evaluate_single_only_one_metric(self, user_prompt: str, info: Dict[str, Any], llm_response: str, evaluate_single_result: Dict[str, float]) -> Dict[str, float]:
        score = evaluate_single_result
        to_template_dict = {
            "INPUT_TEXT": user_prompt,
            "GENERATED_ARTICLE": llm_response,
            "GOLDEN_PASSAGE": info['pr-abstract'],
            "ROUGE_L": f"{score['Rouge-L']:.4f}",
            "BERTSCORE_F1": f"{score['BERTScore-F1']:.4f}",
            "CLI": f"{score['CLI']:.4f}",
            "FKGL": f"{score['FKGL']:.4f}",
            "DCRS": f"{score['DCRS']:.4f}",
        }
        
        final_prompt = merge_score_prompt.format(**to_template_dict)
        # print("Final Prompt:", final_prompt)
        
        tries = 3
        for _ in range(tries):
            try:
                llm_final_response = self.openai_model.generate_response([{
                    "role": "system", "content": "You are a helpful assistant."
                },
                {
                    "role": "user", "content": final_prompt
                }
                ])
                final_score = re.findall(r"\b([1-9]|10)\b", llm_final_response.strip())
                if len(final_score) > 0:
                    final_score = int(final_score[0])
                    break
            except Exception as e:
                print("Error in LLM response:", e)
                final_score = 0
        
        return {
            "llm_as_judge_score": final_score
        }
        
if __name__ == "__main__":
    # Example usage
    dataset = JRE_L_Dataset(data_path="./raw/JRE-L/test.json")
    
    item = dataset.get_data(9)
    
    print(">>>>> JRE_L Dataset Length:", len(dataset))
    
    print(">>>>> Item:")
    
    print(json.dumps(item, ensure_ascii=False, indent=2))
    
    print(">>>>> Evaluation Score:")
    
    score = dataset.evaluate([{
        "test_idx": 9,
        "response": """This is a paper.""",
    }])
    
    print(json.dumps(score, ensure_ascii=False, indent=2))
    
    print(">>>>> Only One Metric Evaluation Score:")
    
    score = dataset.evaluate_single_only_one_metric(
        user_prompt = item['input_prompt'],
        info = item['info'],
        llm_response = """This is a paper."""
    )
    
    print(json.dumps(score, ensure_ascii=False, indent=2))