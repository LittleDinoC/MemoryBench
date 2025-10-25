from src.dataset.base import BaseDataset
from typing import List, Dict, Any, Type, Tuple
from pydantic import Field, BaseModel
import re
import jsonlines
import json
import os

merge_score_prompt = """You are an expert legal AI assistant. Your task is to evaluate the quality of an automatically generated legal judgment document based on the provided context and a set of pre-calculated metrics.

## Case Factual Description (Input)
{INPUT_FACTS}

## Generated Judgment Document (Output)
{GENERATED_JUDGMENT}

## Ground Truth Judgment Document (Reference)
{GOLDEN_JUDGMENT}

## Evaluation Metrics
Below are the calculated metrics comparing the 'Generated Judgment' to the 'Ground Truth'. A score of 1.00 indicates a perfect match for that specific metric, while 0.00 indicates a complete mismatch.

1. Penalty Accuracy (Scores range from 0.00 to 1.00)
time_score: {time_score} (Measures the accuracy of the prison sentence duration.)
amount_score: {amount_score} (Measures the accuracy of the monetary fine amount.)

2. Convicting Accuracy (Scores range from 0.00 to 1.00)
crime_recall: {crime_recall} (The proportion of actual charges that the system correctly identifies.)
crime_precision: {crime_precision} (The proportion of predicted charges that are accurate.)

3. Referencing Accuracy (Scores range from 0.00 to 1.00)
penalcode_index_recall: {penalcode_index_recall} (The proportion of correctly cited ground-truth statutes among all relevant statutes.)
penalcode_index_precision: {penalcode_index_precision} (The proportion of correctly cited statutes among all citations in the generated judgment.)
reasoning_meteor: {reasoning_meteor} (Semantic similarity of the 'Judicial Reasoning' section based on METEOR score.)
reasoning_bert_score: {reasoning_bert_score} (Semantic similarity of the 'Judicial Reasoning' section based on BERTScore.)
judge_meteor: {judge_meteor} (Semantic similarity of the 'Judgment Result' section based on METEOR score.)
judge_bert_score: {judge_bert_score} (Semantic similarity of the 'Judgment Result' section based on BERTScore.)

## Task
Based on a holistic review of the input, output, ground truth, and all the metrics provided above, provide a single integer score from 1 to 10 to represent the overall quality of the generated judgment document.
- 1: Represents extremely poor quality (e.g., completely irrelevant, factually incorrect, nonsensical).
- 10: Represents excellent quality (e.g., legally sound, factually accurate, well-reasoned, and structurally perfect, nearly indistinguishable from the ground truth).Your response should be only a single integer.

## Final Score"""

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


    
class JuDGE_Dataset(BaseDataset):

    def __init__(self, data_path: str = None, dataset_name: str = "JuDGE", test_metrics: List[str] = ["reasoning_meteor", "judge_meteor", "reasoning_bert_score", "judge_bert_score", "crime_recall", "crime_precision", "crime_f1", "penalcode_index_recall", "penalcode_index_precision", "penalcode_index_f1", "time_score", "amount_score"], max_output_len: int = 8192, eval_mode: bool = True):
        # self.evaluate_threads = 4
        self.dataset_name = dataset_name
        # self.feedback_type = feedback_type
        super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
        if eval_mode:
            from src.dataset.judge.calc_score import Evaluator
            self.evaluator = Evaluator()
        else:
            self.evaluator = None
            
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
        

#     def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
#         raw_data = []
#         len_ = 0
        
#         for t in ["train", "test"]:
#             with jsonlines.open(os.path.join(self.data_path, f"{t}.json")) as reader:
#                 for idx, obj in enumerate(reader):
#                     exp_ans = obj['fd']
#                     fact = obj['text']
#                     input_content = f"""
# 案件事实：{fact}
# 请根据上面提供的事实描述，生成一篇完整且具有法律效力的中文的刑事判决书。生成的文书必须结构严谨、逻辑清晰；确保文书所有部分均符合真实司法文书的写作规范，语言应正式、客观、清晰
# """
#                     messages = [
#                         {"role": "system", "content": "你是一个法律助理，提供帮助。"},
#                         {"role": "user", "content": input_content}
#                     ]
#                     raw_data.append({
#                         "test_idx": len_,
#                         "input_chat_messages": messages,
#                         "dataset_name": self.dataset_name,
#                         # "feedback_type": self.feedback_type,
#                         "lang": "zh",
#                         "info": {
#                             'golden_answer': exp_ans,
#                         }
#                     })
#                     len_ += 1
#         return raw_data

    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        scores = self.evaluator.evaluate(llm_response, info['golden_answer'])
        scores['golden_answer'] = info['golden_answer']
        return scores
    
    def evaluate_single_only_one_metric(self, user_prompt: str, info: Dict[str, Any], llm_response: str, evaluate_single_result: Dict[str, float]) -> Dict[str, float]:
        score = evaluate_single_result
        def get_val(name):
            if name in score:
                return f'{score["time_score"]:.4f}'
            else:
                return "None"

        to_template_dict = {
            "INPUT_FACTS": user_prompt,
            "GENERATED_JUDGMENT": llm_response,
            "GOLDEN_JUDGMENT": info['golden_answer'],
            "time_score": get_val("time_score"),
            "amount_score": get_val("amount_score"),
            "crime_recall": get_val("crime_recall"),
            "crime_precision": get_val("crime_precision"),
            "penalcode_index_recall": get_val("penalcode_index_recall"),
            "penalcode_index_precision": get_val("penalcode_index_precision"),
            "reasoning_meteor": get_val("reasoning_meteor"),
            "reasoning_bert_score": get_val("reasoning_bert_score"),
            "judge_meteor": get_val("judge_meteor"),
            "judge_bert_score": get_val("judge_bert_score"),
            # "time_score": f'{score["time_score"]:.4f}',
            # "amount_score": f'{score["amount_score"]:.4f}',
            # "crime_recall": f'{score["crime_recall"]:.4f}',
            # "crime_precision": f'{score["crime_precision"]:.4f}',
            # "penalcode_index_recall": f'{score["penalcode_index_recall"]:.4f}',
            # "penalcode_index_precision": f'{score["penalcode_index_precision"]:.4f}',
            # "reasoning_meteor": f'{score["reasoning_meteor"]:.4f}',
            # "reasoning_bert_score": f'{score["reasoning_bert_score"]:.4f}',
            # "judge_meteor": f'{score["judge_meteor"]:.4f}',
            # "judge_bert_score": f'{score["judge_bert_score"]:.4f}',
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
    dataset = JuDGE_Dataset(data_path="./raw/JuDGE")
    
    item = dataset.get_data(9)
    
    print(">>>>> JuDGE Dataset Length:", len(dataset))
    
    print(">>>>> Item:")
    
    print(json.dumps(item, ensure_ascii=False, indent=2))
    
    print(">>>>> Evaluation Score:")
    
    score = dataset.evaluate_and_summary([{
        "test_idx": 9,
        "response": dataset.get_data(9)['info']['golden_answer'],
    }])
    
    import time
    start = time.time()
    
    # print(">>>>> Only One Metric Evaluation Score:")
    
    # score = dataset.evaluate_single_only_one_metric(item, item['info'], item['info']['golden_answer'])
    
    # print(json.dumps(score, ensure_ascii=False, indent=2))
    
    
    
    print(">>>>> Evaluation Score:")
    
    score = dataset.evaluate_and_summary([{
        "test_idx": 9,
        "response": dataset.get_data(9)['info']['golden_answer'],
    }])
    
    print(json.dumps(score, ensure_ascii=False, indent=2))
    
    print("Total Time:", time.time() - start)
    
    
    # print(">>>>> Evaluation Test Score:")
    
    # score = dataset.evaluate_test([{
    #     "test_idx": 9,
    #     "response": dataset.get_data(9)['info']['golden_answer'],
    # }])
    
    # print(json.dumps(score, ensure_ascii=False, indent=2))