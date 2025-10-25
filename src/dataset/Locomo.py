import json
from collections import Counter
import json
from typing import List, Dict, Any
from src.dataset.base import BaseDataset
import regex
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
import string
from nltk.translate.meteor_score import meteor_score

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Above is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def get_input_context(data):

    query_conv = ''
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + ' said, \"' + dialog['text'] + '\"' + '\n'
                if "blip_caption" in dialog:
                    turn += ' and shared %s.' % dialog["blip_caption"]
                turn += '\n'
        
                query_conv = turn + query_conv
            query_conv = 'DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv

    return query_conv

def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


def get_cat_5_answer(model_prediction, answer_key):
    answer_key = json.loads(answer_key)

    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction

class Locomo_Dataset(BaseDataset):

    def __init__(self, data_path: str = None, dataset_name: str = "Locomo-0", test_metrics: List[str] = ["f1"], max_output_len: int = 8192, eval_mode: bool = True):
        self.dataset_name = dataset_name
        assert int(self.dataset_name.split("-")[-1]) in list(range(10))
        # self.feedback_type = feedback_type
        super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
     
    # def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
    #     raw_data = []
    #     random_flag = True
    #     with open(self.data_path, 'r') as f:
    #         data = json.load(f)
    #         for j, obj in enumerate(data):
    #             if j != int(self.dataset_name.split("-")[-1]):
    #                 continue
    #             # print("Loading Locomo-%s dataset..." % j)
    #             speakers_names = list(set([d['speaker'] for d in obj['conversation']['session_1']]))
    #             start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    #             conversation_context = get_input_context(obj['conversation'])
    #             self.corpus = conversation_context
    #             self.conversation = obj['conversation']
    #             self.conversation_cnt = len(obj["session_summary"].keys())
                
    #             for qa in obj["qa"]:
    #                 if qa['category'] == 2:
    #                     question = qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.'
    #                     q_prompt = QA_PROMPT.format(question)
    #                     answer = qa['answer']
    #                 elif qa['category'] == 5:
    #                     question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
    #                     if random_flag: # 去除随机性
    #                         question = question.format('Not mentioned in the conversation', qa["adversarial_answer"])
    #                         answer = {'a': 'Not mentioned in the conversation', 'b': qa["adversarial_answer"]}
    #                         random_flag = False
    #                     else:
    #                         question = question.format(qa["adversarial_answer"], 'Not mentioned in the conversation')
    #                         answer = {'b': 'Not mentioned in the conversation', 'a': qa["adversarial_answer"]}
    #                         random_flag = True
    #                     q_prompt = QA_PROMPT_CAT_5.format(question)
    #                 else:
    #                     question = qa['question']
    #                     q_prompt = QA_PROMPT.format(question)
    #                     answer = qa['answer']
                        
    #                 evidence = qa["evidence"]
        
    #                 evidence_turns = []
    #                 for e in evidence:
    #                     session_num = e.split('D')[-1].split(':')[0]
    #                     if 'session_%s' % session_num in obj['conversation']:
    #                         for dialog in obj['conversation']['session_%s' % session_num]:
    #                             if dialog.get('dia_id') == e:
    #                                 evidence_turns.append(dialog)
                     
    #                 raw_data.append({
    #                     "test_idx": len(raw_data),
    #                     "origin_question": qa['question'],
    #                     "input_prompt": start_prompt + q_prompt,
    #                     # "corpus": conversation_context,
    #                     "dataset_name": self.dataset_name,
    #                     # "feedback_type": self.feedback_type,
    #                     "lang": "en",
    #                     "info": {
    #                         'golden_answer': answer,
    #                         'category': qa['category'],
    #                         'evidence': evidence_turns,
    #                     }
    #                 })
        
    #     return raw_data

    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        if info['category'] == 5:
            output = get_cat_5_answer(llm_response, info['golden_answer'])
        else:
            output = llm_response.strip()
        answer = info['golden_answer']
        if info['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        answer = str(answer)
        
        # single-hop, temporal, open-domain eval without splitting for sub-answers 
        if info['category'] in [2, 3, 4]:
            ems = f1_score(output, answer)
        
        # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
        elif info['category'] in [1]:
            ems = f1(output, answer)

        # adversarial eval --> check for selection of correct option
        elif info['category'] in [5]:
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                ems = 1
            else:
                ems = 0
        else:
            raise ValueError
        return {
            "f1": ems,
            'golden_answer': info['golden_answer'],
            'evidence': info['evidence'],
        }
    
if __name__ == "__main__":
    # Example usage 
    dataset = Locomo_Dataset("./raw/Locomo/locomo10.json", "Locomo-9")
    item = dataset.get_data(9)
    
    print(">>>>>> Locomo Dataset Length:")
    print(len(dataset))
    
    print("=" * 50)
    
    print(">>>>>> Locomo Dataset Item:")
    print(json.dumps(item, ensure_ascii=False, indent=2))
    
    print("=" * 50)
    
    score = dataset.evaluate([{
            "test_idx": 9,
            "response": """before 9 June 2023""",
    }])
    print(">>>>> Evaluation Score:")
    print(json.dumps(score, ensure_ascii=False, indent=2))
    