from src.dataset.base import BaseDataset
from typing import List, Dict, Any
import json
from rouge import Rouge
import jieba
import string


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def eval_rougel(response: str, golden: str) -> float:
    pred = " ".join(list(jieba.cut(normalize_zh_answer(response), cut_all=False)))
    ans = " ".join(list(jieba.cut(normalize_zh_answer(golden), cut_all=False)))
    rouge = Rouge()
    try:
        return rouge.get_scores([pred], [ans], avg=True)["rouge-l"]["f"]
    except:
        return 0.0
    
    
class LexEval_Dataset(BaseDataset):
    """
    A concrete LexEval Generation dataset.
    """
    def __init__(self, data_path: str=None, dataset_name: str = "LexEval-Summarization", test_metrics: List[str] = ["rougel"], max_output_len: int = 8192, eval_mode: bool = True):
        self.dataset_name = dataset_name
        # self.feedback_type = feedback_type
        super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
        
    # def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
    #     raw_data = []
    #     with open(self.data_path, 'r', encoding='utf-8') as file:
    #         for line in file:
    #             item = json.loads(line.strip())
    #             raw_data.append({
    #                 "test_idx": len(raw_data),
    #                 "input_prompt": item["instruction"] + item["input"],
    #                 "lang": "zh",
    #                 "dataset_name": self.dataset_name,
    #                 # "feedback_type": self.feedback_type,
    #                 "info": {
    #                     "golden_answer": item["answer"],
    #                 }
    #             })
    #     return raw_data

    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        eval_rougel_score = eval_rougel(llm_response, info["golden_answer"])
        return {
            "rougel": eval_rougel_score,
            "golden_answer": info["golden_answer"], 
        }
        
if __name__ == "__main__":
    # Example usage
    lexeval_sum_dataset = LexEval_Dataset(dataset_name="LexEval-Summarization", data_path="./raw/LexEval/5_1.json")
    
    print(">>>>>> LexEval-Summarization Dataset Length:")
    print(len(lexeval_sum_dataset))
    
    print("=" * 50)
    item = lexeval_sum_dataset.get_data(0, 1)[0]
    
    print(">>>>> LexEval-Summarization Dataset Item:")
    print(json.dumps(item, ensure_ascii=False, indent=2))
    
    print("=" * 50)
    
    score = lexeval_sum_dataset.evaluate([
        {
            "test_idx": 0,
            "response": "这是一个关于法律援助的总结。法律援助是指在经济困难的情况下，个人可以获得免费的法律服务和支持。法律援助的目的是确保每个人都能平等地获得法律帮助，无论其经济状况如何。法律援助通常由政府或非营利组织提供，涵盖了各种法律问题，如刑事辩护、家庭法、住房纠纷等。通过法律援助，个人可以获得法律咨询、代表和其他相关服务，从而保护他们的合法权益。"
        }
    ])
    print(">>>>> Evaluation Score:")
    print(json.dumps(score, ensure_ascii=False, indent=2))