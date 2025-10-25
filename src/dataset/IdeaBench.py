import os
import json
import re
import random
import time
from typing import List, Dict, Any
from src.dataset.base import BaseDataset
from src.llms import LlmFactory

import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel, Field


# import evaluate
from bert_score import BERTScorer

def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


IDEA_SEPARATOR = "---IDEA-SEPARATOR---"

GENERATION_PROMPT_PREFIX = """You are a biomedical researcher. You are tasked with creating novel hypotheses or research ideas given some background knowledge. The background knowledge I will provide are abstracts from other papers.

Here are the abstracts:"""

GENERATION_PROMPT_SUFFIX = f"""Using these abstracts, reason over them and come up with 3 novel and distinct hypotheses. Please avoid copying ideas directly, rather use the insights to inspire novel hypotheses.
Format each hypothesis as a brief and concise paragraph.
IMPORTANT: Separate the 3 hypotheses with '{IDEA_SEPARATOR}'."""


RATING_PROMPT_TEMPLATE = """You are an expert in understanding and analyzing scientific content. Your task is to evaluate the degree of overlap between the ideas presented in a hypothesis and the abstract of a scientific paper. Please read both the hypothesis and the abstract carefully. Then, rate the overlap on a scale of 1 to 10, where 1 indicates minimal or no overlap, and 10 indicates a perfect or nearly perfect overlap. Provide a brief explanation for your rating.

Your output MUST be a JSON object with two keys: "rating" (integer 1-10) and "explanation" (string).

<Hypothesis>
{hypothesis}
</Hypothesis>

<Abstract>
{abstract}
</Abstract>
"""

RANKING_PROMPT_PREFIX = """You are a reviewer tasked with ranking the quality of a set of research ideas based on their {ranking_criteria}. The idea with the highest {ranking_criteria} should be ranked first. 

Please rank the following hypotheses. Your output should be a numbered list, starting with the best idea. For example:
1. **Hypothesis C**: (brief rationale)
2. **Hypothesis A**: (brief rationale)
...

Here are the hypotheses to rank:
"""

merge_score_prompt = """You are an expert scientific researcher and AI assistant. Your task is to evaluate the overall quality of an automatically generated research idea based on the provided context and a set of pre-calculated metrics.

## Background Knowledge (Input)
{INPUT_CONTEXT}

## Generated Research Idea (Output)
{GENERATED_IDEA}

## Ground Truth Research Idea (Reference)
{GOLDEN_IDEA}

## Evaluation Metrics
Below are the calculated metrics comparing the 'Generated Research Idea' to the 'Ground Truth'. Please use them to inform your overall score.

1. Semantic Similarity (bert_score): Measures the semantic similarity between the 'Generated Research Idea' and the 'Ground Truth Research Idea'. Scores range from 0.00 (no similarity) to 1.00 (perfect semantic match).
bert_score: {bert_score}

2. Idea Overlap (llm_rating_score): An LLM-based rating of the idea overlap between the 'Generated Research Idea' and the 'Ground Truth'. Scores range from 1 (minimal overlap) to 10 (perfect overlap).
llm_rating_score: {llm_rating_score}

3. Novelty Insight Score (llm_novelty_ranking_score): Quantifies the novelty of the 'Generated Research Idea' relative to the 'Ground Truth'. This score is derived by ranking the generated idea(s) against the ground truth idea. Scores range from 0.00 to 1.00.
    * A score near **0.00** means the generated idea is significantly less novel than the ground truth.
    * A score near **0.50** suggests comparable novelty.
    * A score near **1.00** means the generated idea is significantly more novel than the ground truth.
llm_novelty_ranking_score: {llm_novelty_ranking_score}

4. Feasibility Insight Score (llm_feasibility_ranking_score): Quantifies the feasibility of the 'Generated Research Idea' relative to the 'Ground Truth', using the same ranking methodology as the Novelty Insight Score. Scores range from 0.00 to 1.00.
    * A score near **0.00** means the generated idea is significantly less feasible than the ground truth.
    * A score near **0.50** suggests comparable feasibility.
    * A score near **1.00** means the generated idea is significantly more feasible than the ground truth.
llm_feasibility_ranking_score: {llm_feasibility_ranking_score}

## Task
Based on a holistic review of the input, output, ground truth, and all the metrics provided above, provide a single integer score from 1 to 10 to represent the overall quality of the generated research idea.
- 1: Represents extremely poor quality (e.g., incoherent, irrelevant, factually incorrect).
- 10: Represents excellent quality (e.g., coherent, insightful, novel, feasible, and well-aligned with the background knowledge, nearly indistinguishable from an idea proposed by a human expert).

Your response should be only a single integer.

## Final Score
"""

class BaseAgentConfig(BaseModel):
    llm_provider: str = Field(
        default="openai", 
        description="The LLM provider to use for the agent."
    )
    llm_config: dict = Field(
        default_factory=dict, 
        description="Configuration parameters for the LLM."
    )
    

class IdeaBench_Dataset(BaseDataset):

    def __init__(self, data_path: str = "", num_ref: int = 3, all_ref: bool = False, bert_score_model: str = 'microsoft/deberta-xlarge-mnli', test_metrics: List[str] = ['bert_score', 'llm_rating_score', 'llm_novelty_ranking_score', 'llm_feasibility_ranking_score'], max_output_len: int = 8192, eval_mode: bool = True) -> None:
        """
        初始化 IdeaBench 数据集

        Args:
            num_ref (int): 生成时用作背景知识的参考文献摘要数量
            all_ref (bool): 是否使用所有参考文献
        """
        self.dataset_name = "IdeaBench"
        self.evaluate_threads = 4
        self.target_papers_path = os.path.join(data_path, 'target_papers.csv')
        self.references_path = os.path.join(data_path, 'filtered_references.csv')
        self.num_ref = num_ref
        self.all_ref = all_ref
        # self.feedback_type = feedback_type
        super().__init__(data_path, test_metrics, max_output_len=max_output_len)

        # 初始化评估器
        if eval_mode:
            print("Initializing evaluators...")
            self.scorer = BERTScorer(model_type=bert_score_model, device='cuda:0')
        else:
            self.scorer = None
        # self.rouge = evaluate.load('rouge')
        # self.bleu = evaluate.load('bleu')
        
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
        

    # def _load_data(self) -> List[Dict[str, Any]]:
    #     """
    #     从 CSV 文件加载数据并构建生成 Prompt
    #     """

    #     target_df = pd.read_csv(self.target_papers_path)
    #     ref_df = pd.read_csv(self.references_path).dropna(subset=['abstract'])
        
    #     dataset = []
    #     for idx, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Preparing prompts"):
    #         target_paper_id = row['paperId']
            
    #         ref_abstracts_all = ref_df[ref_df['targetPaperId'] == target_paper_id]['abstract'].tolist()
            
    #         if self.all_ref:
    #             selected_refs = ref_abstracts_all
    #         else:
    #             n_samples = min(self.num_ref, len(ref_abstracts_all))
    #             selected_refs = random.sample(ref_abstracts_all, n_samples)
            
    #         background_knowledge = []
    #         for i, abstract in enumerate(selected_refs):
    #             clean_abstract = abstract.replace("{greater than or equal to}", "≥").replace("{", "").replace("}", "")
    #             background_knowledge.append(f"Abstract {i+1}: {clean_abstract}")
            
    #         background_text = "\n\n".join(background_knowledge)

    #         input_prompt = f"{GENERATION_PROMPT_PREFIX}\n\n{background_text}\n\n{GENERATION_PROMPT_SUFFIX}"
            
    #         dataset.append({
    #             "test_idx": idx,
    #             "input_prompt": input_prompt,
    #             "dataset_name": "IdeaBench",
    #             "lang": "en",
    #             "info": {
    #                 "paperId": target_paper_id,
    #                 "title": row['title'],
    #                 "abstract": row['abstract']
    #             },
    #             # "feedback_type": self.feedback_type
    #         })
    #     print(f"Data loading complete. {len(dataset)} items loaded.")
    #     return dataset

    def _get_llm_rating(self, hypothesis: str, abstract: str) -> Dict:
        """使用 LLM 对单个 hypothesis 和 abstract 的重叠度进行打分"""
        prompt = RATING_PROMPT_TEMPLATE.format(hypothesis=hypothesis, abstract=abstract)
        messages = [{'role': 'user', 'content': prompt}]
        
        response_str = self.openai_model.generate_response(messages)
        if '```json' in response_str:
            response_str = extract_info(r'```json\n(.*?)\n```', response_str)
        try:
            return json.loads(response_str)
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse LLM rating response as JSON. Response: {response_str}")
            return {"rating": None, "explanation": response_str}

    def _get_llm_ranking(self, hypotheses: List[str], abstract: str, criteria: str) -> Dict:
        """使用 LLM 对一组 ideas (包括 ground truth) 进行排序"""
        candidates = [f"**Hypothesis A**:\n{abstract}"]
        for i, hyp in enumerate(hypotheses):
            letter = chr(ord('A') + i + 1)
            candidates.append(f"**Hypothesis {letter}**:\n{hyp}")
        
        
        candidates_text = "\n\n".join(candidates)
        prompt = f"{RANKING_PROMPT_PREFIX.format(ranking_criteria=criteria)}\n\n{candidates_text}"
        messages = [{'role': 'user', 'content': prompt}]
        
        response_str = self.openai_model.generate_response(messages)
        
        ranking_order = re.findall(r'\*\*\s*Hypothesis\s+([A-Z])\s*\*\*', response_str)
        
        ## Hypothesis A的排名
        if not ranking_order:
            r_target = 0
        else:
            r_target = ranking_order.index('A') + 1

        return {"r_target": r_target, "ranking": ranking_order, "raw_text": response_str}


    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, Any]:

        hypotheses = [h.strip() for h in llm_response.split(IDEA_SEPARATOR) if h.strip()]
        if len(hypotheses) < 3:
            # 补齐到 3 个 hypothesis
            hypotheses += ["NULL"] * (3 - len(hypotheses))
        
        ground_truth_abstract = info['abstract']
        
        P, R, F1 = self.scorer.score(hypotheses, [ground_truth_abstract] * len(hypotheses))
        best_f1_idx = F1.argmax()
        best_hypothesis = hypotheses[best_f1_idx]
        
        # bert_scores = {
        #     'precision': P[best_f1_idx].item(),
        #     'recall': R[best_f1_idx].item(),
        #     'f1': F1[best_f1_idx].item()
        # }
        bert_scores_f1 = F1[best_f1_idx].item()
        # rouge_scores = self.rouge.compute(predictions=[best_hypothesis], references=[ground_truth_abstract])
        # bleu_scores = self.bleu.compute(predictions=[best_hypothesis], references=[ground_truth_abstract])
        

        llm_rating = self._get_llm_rating(best_hypothesis, ground_truth_abstract)
        
        llm_novelty_ranking = self._get_llm_ranking(hypotheses, ground_truth_abstract, "novelty")
        llm_feasibility_ranking = self._get_llm_ranking(hypotheses, ground_truth_abstract, "feasibility")

        llm_rating_score = llm_rating.get("rating", None)
        llm_novelty_ranking_score = (llm_novelty_ranking["r_target"] - 1) / len(hypotheses)
        llm_feasibility_ranking_score = (llm_feasibility_ranking["r_target"] - 1) / len(hypotheses)
        return {
            "generated_hypotheses": hypotheses,
            "best_hypothesis_by_bertf1": best_hypothesis,
            "bert_score": bert_scores_f1,
            "llm_rating_score": llm_rating_score,
            "llm_novelty_ranking_score": llm_novelty_ranking_score,
            "llm_feasibility_ranking_score": llm_feasibility_ranking_score,
            "llm_rating": llm_rating,
            "llm_novelty_ranking": llm_novelty_ranking,
            "llm_feasibility_ranking": llm_feasibility_ranking
        }
    
    def from_save_data_to_full_data(self, user_prompt: str, info: Dict[str, Any], llm_response: str, saved_data: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses = [h.strip() for h in llm_response.split(IDEA_SEPARATOR) if h.strip()]
        if len(hypotheses) < 3:
            # 补齐到 3 个 hypothesis
            hypotheses += ["NULL"] * (3 - len(hypotheses))
        
        ground_truth_abstract = info['abstract']
        
        P, R, F1 = self.scorer.score(hypotheses, [ground_truth_abstract] * len(hypotheses))
        best_f1_idx = F1.argmax()
        best_hypothesis = hypotheses[best_f1_idx]
        
        # bert_scores = {
        #     'precision': P[best_f1_idx].item(),
        #     'recall': R[best_f1_idx].item(),
        #     'f1': F1[best_f1_idx].item()
        # }
        bert_scores_f1 = F1[best_f1_idx].item()
        assert bert_scores_f1 == saved_data['bert_score'], f"Mismatch in bert_score: {bert_scores_f1} vs {saved_data['bert_score']}"
        # rouge_scores = self.rouge.compute(predictions=[best_hypothesis], references=[ground_truth_abstract])
        # bleu_scores = self.bleu.compute(predictions=[best_hypothesis], references=[ground_truth_abstract])

        # llm_rating = self._get_llm_rating(best_hypothesis, ground_truth_abstract)
        
        # llm_novelty_ranking = self._get_llm_ranking(hypotheses, ground_truth_abstract, "novelty")
        # llm_feasibility_ranking = self._get_llm_ranking(hypotheses, ground_truth_abstract, "feasibility")

        # llm_rating_score = llm_rating.get("rating", None)
        # llm_novelty_ranking_score = (llm_novelty_ranking["r_target"] - 1) / len(hypotheses)
        # llm_feasibility_ranking_score = (llm_feasibility_ranking["r_target"] - 1) / len(hypotheses)
        return {
            "generated_hypotheses": hypotheses,
            "best_hypothesis_by_bertf1": best_hypothesis,
            "bert_score": bert_scores_f1,
            "llm_rating_score": saved_data['llm_rating_score'],
            "llm_novelty_ranking_score": saved_data['llm_novelty_ranking_score'],
            "llm_feasibility_ranking_score": saved_data['llm_feasibility_ranking_score'],
            # "llm_rating": llm_rating,
            # "llm_novelty_ranking": llm_novelty_ranking,
            # "llm_feasibility_ranking": llm_feasibility_ranking
        }
        
    def evaluate_single_only_one_metric(self, user_prompt: str, info: Dict[str, Any], llm_response: str, evaluate_single_result: Dict[str, float]) -> Dict[str, float]:
        score = evaluate_single_result
        to_template_dict = {
            "INPUT_CONTEXT": user_prompt,
            "GENERATED_IDEA": score['best_hypothesis_by_bertf1'],
            "GOLDEN_IDEA": info['abstract'],
            "bert_score": f"{score['bert_score']:.4f}",
            "llm_rating_score": score['llm_rating_score'] if score['llm_rating_score'] is not None else "N/A",
            "llm_novelty_ranking_score": f"{score['llm_novelty_ranking_score']:.4f}",
            "llm_feasibility_ranking_score": f"{score['llm_feasibility_ranking_score']:.4f}",
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

    dataset = IdeaBench_Dataset(
        data_path='./raw/IdeaBench',
        num_ref=3,
        all_ref=False)
    
    print(f"\n>>>>> IdeaBench Dataset initialized. Total items: {len(dataset)}")
    

    sample_item = dataset.get_data(0)
    print("\n>>>>> Sample Item (test_idx=0):")
    print(json.dumps(sample_item, indent=2))

    print(f"\n>>>>> Generating 3 ideas...")
    generation_messages = [{'role': 'user', 'content': sample_item['input_prompt']}]
    generated_response = dataset.openai_model.generate_response(generation_messages)
    
    print("\n>>>>> Raw response from generation model:")
    print(generated_response)
    
    # 4. 使用评估流程评估生成的响应
    print(f"\n>>>>> Evaluating the response with...")
    evaluation_result = dataset.evaluate_single(
        user_prompt=sample_item['input_prompt'],
        info=sample_item['info'],
        llm_response=generated_response
    )
    
    print("\n>>>>> COMPLETE EVALUATION RESULT:")
    print(json.dumps(evaluation_result, indent=2))
    
    print(">>>>> Only One Metric Evaluation Score:")
    score = dataset.evaluate_single_only_one_metric(
        user_prompt = sample_item['input_prompt'],
        info = sample_item['info'],
        llm_response = generated_response
    )
    
    print(json.dumps(score, ensure_ascii=False, indent=2))

