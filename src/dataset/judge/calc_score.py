import jieba
from bert_score import score
import json
import argparse
from nltk.translate.meteor_score import meteor_score
from .xingshi import DataSegmentXingshi
from .crime_extraction import get_crime
from .judge_extraction import calc_time_sum, calc_amt_sum
from .law_extraction import get_penalcode_index_from_text

class Evaluator:
    def __init__(self):
        '''
        在全局平均时，None的值不参与计算
        '''
        pass
    
    def evaluate(self, gen_ans, exp_ans):
        reasoning_bertscore, judge_bertscore = self.calc_bert_score(gen_ans, exp_ans)
        reasoning_meteor, judge_meteor, exp_reasoning, exp_judge, gen_reasoning, gen_judge = self.calc_meteor(gen_ans, exp_ans)
        rel_results = {
            "reasoning_bert_score": reasoning_bertscore,
            "judge_bert_score": judge_bertscore,
            "reasoning_meteor": reasoning_meteor,
            "judge_meteor": judge_meteor,
            "exp_reasoning": exp_reasoning,
            "exp_judge": exp_judge,
            "gen_reasoning": gen_reasoning,
            "gen_judge": gen_judge
        }
        other_results = self.calc_metrics(gen_ans, exp_ans)
        return {**rel_results, **other_results}
        
    def get_all_from_text(self, text):
        return get_crime(text), calc_time_sum(text), calc_amt_sum(text), get_penalcode_index_from_text(text)

    def calculate_recall_and_precision(self, expected, actual):
        expected_set = set(expected)
        actual_set = set(actual)
        true_positive = len(expected_set & actual_set)

        recall = true_positive / len(expected_set) if len(expected_set) > 0 else 0
        precision = true_positive / len(actual_set) if len(actual_set) > 0 else 0

        return recall, precision

    def calculate_percent_for_judge(self, exp_val, act_val):
        if exp_val == act_val == 0:
            return 1.0
        if (exp_val >= 0 and act_val) < 0 or (exp_val < 0 and act_val >= 0):  # Different signs
            return 0.0
        if (exp_val - 10000) * (act_val - 10000) < 0:  # Both must either have or lack the death penalty
            return 0.0
        x = abs(exp_val - act_val) / max(exp_val, act_val)
        y = 1 - x
        return y

    def calc_metrics(self, gen_ans, exp_ans):
        

        exp_crime, exp_time, exp_amount, exp_penalcode_index = self.get_all_from_text(exp_ans)
        gen_crime, gen_time, gen_amount, gen_penalcode_index = self.get_all_from_text(gen_ans)
        
        # print(f"Processing ID: {exp_id}")
        # print(f"Expected: {exp_crime}, {exp_time}, {exp_amount}, {exp_penalcode_index}")
        # print(f"Generated: {gen_crime}, {gen_time}, {gen_amount}, {gen_penalcode_index}")
        # exit(0)

        crime_rec, crime_prec = self.calculate_recall_and_precision(exp_crime, gen_crime)
        penalcode_index_rec, penalcode_index_prec = self.calculate_recall_and_precision(exp_penalcode_index, gen_penalcode_index)

        if exp_time >= 0 or gen_time >= 0:
            time_score = self.calculate_percent_for_judge(exp_time, gen_time)
        else:
            time_score = None

        if exp_amount >= 0 or gen_amount >= 0:
            amount_score = self.calculate_percent_for_judge(exp_amount, gen_amount)
        else:
            amount_score = None
            
        return {
            "crime_recall": crime_rec,
            "crime_precision": crime_prec,
            "time_score": time_score,
            "amount_score": amount_score,
            "penalcode_index_recall": penalcode_index_rec,
            "penalcode_index_precision": penalcode_index_prec,
            "exp_crime": exp_crime,
            "gen_crime": gen_crime,
            "exp_time": exp_time,
            "gen_time": gen_time,
            "exp_amount": exp_amount,
            "gen_amount": gen_amount,
            "exp_penalcode_index": exp_penalcode_index,
            "gen_penalcode_index": gen_penalcode_index
        }
        
    def extract_reasoning_n_judge(self, text):
        parser = DataSegmentXingshi(punctuation_replace=True)
        result = parser.parse(text)
        return result['reason'], result['judgment']

    def calculate_meteor(self, exp_text, gen_text):
        """计算 METEOR 分数"""
        reference_tokens = list(jieba.cut(exp_text))
        hypothesis_tokens = list(jieba.cut(gen_text))
        return meteor_score([reference_tokens], hypothesis_tokens)


    def calc_meteor(self, gen_ans, exp_ans):
        """计算 METEOR 分数"""
        exp_reasoning, exp_judge = self.extract_reasoning_n_judge(exp_ans)
        gen_reasoning, gen_judge = self.extract_reasoning_n_judge(gen_ans)
        
        if not exp_reasoning or not exp_judge or not gen_reasoning or not gen_judge:
            reasoning_score = None
            judge_score = None
        else:
            reasoning_score = self.calculate_meteor(exp_reasoning, gen_reasoning)
            judge_score = self.calculate_meteor(exp_judge, gen_judge)
        
        return reasoning_score, judge_score, exp_reasoning, exp_judge, gen_reasoning, gen_judge

    def calc_bert_score(self, gen_ans, exp_ans):
        """计算 BERTScore"""
        local_model_path = "bert-base-chinese" # 如果未下载过，会自动下载
        gen_reasoning = " ".join(jieba.cut(self.extract_reasoning_n_judge(gen_ans)[0]))
        exp_reasoning = " ".join(jieba.cut(self.extract_reasoning_n_judge(exp_ans)[0]))
        gen_judge = " ".join(jieba.cut(self.extract_reasoning_n_judge(gen_ans)[1]))
        exp_judge = " ".join(jieba.cut(self.extract_reasoning_n_judge(exp_ans)[1]))

        # 计算 reasoning 的 BERTScore
        P_rsn, R_rsn, F1_rsn = score([gen_reasoning], [exp_reasoning], model_type=local_model_path)

        # 计算 judge 的 BERTScore
        P_jdg, R_jdg, F1_jdg = score([gen_judge], [exp_judge], model_type=local_model_path)

        return F1_rsn.tolist()[0], F1_jdg.tolist()[0]

