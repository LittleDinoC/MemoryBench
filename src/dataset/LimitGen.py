from src.dataset.base import BaseDataset
from typing import List, Dict, Any, Type
import json
import jsonlines
import os
import re
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



RATING_PROMPT = """Compare the following pair of limitations of a scientific paper: one generated and one from the ground truth. Assess the degree of relatedness and specificity of the generated limitation compared to the ground truth limitation. Provide a brief explanation, then assign a rating (1-5).

Your response should be in the format of a JSON object, as follows: {"explanation": <a brief explanation>, "rating": <an integer between 1 and 5>}

Rating Criteria:
- 5 points: The generated limitation discusses exactly the same content as the ground truth and provides a similar level of detail.
- 4 points: The generated limitation discusses exactly the same content as the ground truth, but it is less detailed than the ground truth.
- 3 points: The generated limitation is related to the ground truth, but not identical.
- 2 points: The generated limitation is only loosely related to the ground truth.
- 1 point: There is no connection between the generated limitation and the ground truth."""

def find_ground_truth(error_type):
    if error_type == "ablation":
        return "ablation study"
    elif error_type == "data":
        return "low data quality"
    elif error_type == "inappropriate":
        return "inappropriate method"
    elif error_type == "baseline":
        return "insufficient baseline"
    elif error_type == "dataset":
        return "limited datasets"
    elif error_type == "replace":
        return "inappropriate datasets"
    elif error_type == "review":
        return "limited scope"
    elif error_type == "citation":
        return "irrelevant citations"
    elif error_type == "description":
        return "inaccurate description"
    elif error_type == "metric":
        return "insufficient metric"
    elif error_type == "analysis":
        return "limited analysis"

        
def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text

def prepare_message(error_type, paper_content):
    if error_type in ["data", "inappropriate"]:
        aspect = "methodology"
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experimental design"
    elif error_type in ["metric","analysis"]:
        aspect = "result analysis"    
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature review"   
    else:
        print("invalid subtype")
        return None
    SYSTEM_INPUT = f"Read the following scientific paper and generate 3 major limitations in this paper about its {aspect}. Do not include any limitation explicitly mentioned in the paper itself. Return the limitations in the following JSON format: {{\"limitations\": <a list of 3 limitations>}}."
    
    return [
        {
            "role": "system",
            "content": SYSTEM_INPUT
        },  
        {
            "role": "user",
            "content": paper_content
        }
    ]
    
def prepare_paper(paper_data):
    content = f"Paper to review: \nTitle: {paper_data['title']}\n"
    content += f"Abstract: {paper_data['abstract']}\n"
    for section in paper_data["sections"]:
        content += section["section_id"] + " " + section["section_name"] + ':\n'+ section["text"] + '\n\n'

    return content

def prepare_aspect_check_message(error_type, limitation):
    if error_type in ["data", "inappropriate"]:
        aspect = "methodology"
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experimental design"
    elif error_type in ["metric","analysis"]:
        aspect = "result analysis"    
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature review"   
    else:
        print("invalid subtype")
        return None
    SYSTEM_INPUT = f"Please check whether the following limitation of a scientific paper is related to the {aspect}.\n\nOutput only \"yes\" or \"no\"."
    
    # print(user_prompt)
    data = [
        {
            "role": "system",
            "content": SYSTEM_INPUT
        },
        {
            "role": "user",
            "content": limitation
        }
    ]
    # print("Aspect Check Message:", data)
    return data
    
def prepare_subtype_classification_message(error_type, limitation):
    if error_type in ["data", "inappropriate"]:
        aspect = "methodology"
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Low Data Quality - The data collection method is unreliable, potentially introducing bias and lacking adequate preprocessing.
2. Inappropriate Method - Some methods in the paper are unsuitable for addressing this research question and may lead to errors or oversimplifications.
3. Lack of Novelty - The work fails to enhance established techniques from prior research, remaining largely unchanged. This limited novelty may result in missed opportunities to improve model effectiveness and applicability.
4. Limited Performance - The method's performance is insufficiently impressive, often lacking robustness and generalization across various datasets or tasks.
5. Others

Output only the corresponding number."""
    elif error_type in ["baseline", "dataset", "replace", "ablation"]:
        aspect = "experimental design"
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Insufficient baseline models/methods - Fail to evaluate the proposed approach against a broad range of well-established methods.
2. Limited datasets - Rely on limited datasets, which may hinder the generalizability and robustness of the proposed approach.
3. Inappropriate datasets - Use of inappropriate datasets, which may not accurately reflect the target task or real-world scenarios.
4. Lack of an ablation study - Fail to perform an ablation study or account for a specific module, leaving the contribution of a certain component to the research unclear.
5. Others

Output only the corresponding number."""
    elif error_type in ["metric","analysis"]:
        aspect = "result analysis"  
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Insufficient evaluation metrics - Rely on insufficient evaluation metrics, which may provide an incomplete assessment of the model's overall performance.
2. Limited analysis - Offer insufficient insights into the model's behavior and failure cases.
3. Misalignment between text and tables/figure - Show discrepancies between the text and accompanying tables or figures, such as conflicting numerical values or inconsistent comparison outcomes.
4. Exaggerated or misleading conclusions - Draw exaggerated or misleading conclusions that may overstate the model's effectiveness or applicability beyond the presented evidence.
5. Others

Output only the corresponding number."""  
    elif error_type in ["citation", "review", "description"]:
        aspect = "literature review" 
        prompt = """Please classify the following limitation of a scientific paper into one of the following subtypes:
1. Limited Scope of the Review - The review may focus on a very specific subset of literature or methods, leaving out important studies or novel perspectives.
2. Irrelevant Citations - Include irrelevant references or outdated methods, which distracts from the main points and undermines the strength of conclusions.
3. Inaccurate Description of Existing Methods - Provide an inaccurate description of existing methods, which can hinder readers' understanding of the context and relevance of the proposed approach.
4. Others

Output only the corresponding number."""  
    else:
        print("invalid subtype")
        return None
    
    # print(user_prompt)
    data = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": limitation
        }
    ]
    
    # print("Subtype Classification Message:", data)
    return data

class LimitGen_Dataset(BaseDataset):

    def __init__(self, data_path: str = None, dataset_name: str = "LimitGen-Syn", test_metrics: List[str] = ["accuracy", "rating"], max_output_len: int = 8192, eval_mode: bool = True):
        self.evaluate_threads = 4
        self.dataset_name = dataset_name
        # self.feedback_type = feedback_type
        super().__init__(data_path=data_path, test_metrics=test_metrics, max_output_len=max_output_len)
        
        config = BaseAgentConfig(
            llm_config = {
                # "model": "gpt-4o-2024-05-13",
                "model": "gpt-4o-mini",
                "temperature": 0.8,
                "max_tokens": 4096,
            }
        )
        self.openai_model = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config,
        )

    # def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
    #     annotated_dir = os.path.join(self.data_path, "annotated")
    #     sections_dir = os.path.join(self.data_path, "sections")
    #     raw_data = []
    #     for category in sorted(os.listdir(annotated_dir)):
    #         data_path = os.path.join(annotated_dir, category)
    #         label_path = os.path.join(sections_dir, f"{category}.json")
    #         if not os.path.isdir(data_path) or not os.path.isfile(label_path):
    #             continue
    #         # if category == "data":
    #         #     category_name = "low_data_quality"
    #         # elif category == "inappropriate":
    #         #     category_name = "inappropriate_method"
    #         # elif category == "baseline":
    #         #     category_name = "insufficient_baselines"        
    #         # elif category == "dataset":
    #         #     category_name = "limited_datasets"
    #         # elif category == "replace":
    #         #     category_name = "inappropriate_datasets"
    #         # elif category == "ablation":
    #         #     category_name = "lack_ablation"
    #         # elif category == "analysis":
    #         #     category_name = "limited_analysis"
    #         # elif category == "metric":
    #         #     category_name = "insufficient_metrics"        
    #         # elif category == "review":
    #         #     category_name = "limited_scope"
    #         # elif category == "citation":
    #         #     category_name = "irrelevant_citations"
    #         # elif category == "description":
    #         #     category_name = "inaccurate_description"
                
    #         with open(label_path, "r", encoding="utf-8") as f:
    #             label_data = json.load(f)
    #         for fname in os.listdir(data_path):
    #             if not fname.endswith(".json"):
    #                 continue
    #             file_id = os.path.splitext(fname)[0]
    #             with open(os.path.join(data_path, fname), "r", encoding="utf-8") as f:
    #                 content = json.load(f)
    #             label = label_data.get(file_id, {}).get("ground_truth", "unknown")
                
    #             paper_content = prepare_paper(content)
    #             message = prepare_message(category, paper_content)
    #             if not message:
    #                 continue
                
    #             raw_data.append({
    #                 "test_idx": len(raw_data),
    #                 "id": file_id,
    #                 "input_chat_messages": message,
    #                 "dataset_name": self.dataset_name,
    #                 # "feedback_type": self.feedback_type,
    #                 "lang": "en",
    #                 "info": {
    #                     "ground_truth": label,
    #                     'category': category,
    #                 }
    #             })
        
    #     return raw_data

    def _evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        if '```json' in llm_response:
            llm_response = extract_info(r'```json\n(.*?)\n```', llm_response)
        try:
            limits = json.loads(llm_response)["limitations"]
            if type(limits) == str:
                limits = [limits]
            elif type(limits) == list:
                limits = limits[:3]
            else:
                limits = ["Invalid Message"]
        except (json.JSONDecodeError, TypeError):
            limits = ["Invalid Message"]
            
        result = {"ground_truth": info["ground_truth"]}
        
        # print(USER_INPUT)
        acc_list = []
        rating_list = []
        explanation_list = []
        predicted_subtype_list = []
        ground_truth_subtype = find_ground_truth(info['category'])
        
        for limit in limits:
            acc = False
            rating = 0
            explanation = "N/A"
            subtype = "N/A"
            # Accuracy evaluation
            # print("Generated Limitation:", limit)
            msgs = prepare_aspect_check_message(info['category'], limit)
            # print("Aspect Check Message:", msgs)
            response = self.openai_model.generate_response(msgs)
            
            
            # print("Aspect Check Response:", response)
            if response.lower().startswith("yes"):
                response = "yes"
            elif response.lower().startswith("no"):
                response = "no"
            else:
                response = "invalid"
                
            if response == "yes":
                if info['category'] in ["data", "inappropriate"]:
                    aspects = ["low data quality", "inappropriate method", "lack of novelty", "limited performance", "others"]
                elif info['category'] in ["baseline", "dataset", "replace", "ablation"]:
                    aspects = ["insufficient baseline", "limited datasets", "inappropriate datasets", "ablation study", "others"]
                elif info['category'] in ["metric", "analysis"]:
                    aspects = ["insufficient metric", "limited analysis", "result misalignment", "misleading conclusion", "others"]
                elif info['category'] in ["citation", "review", "description"]:
                    aspects = ["limited scope", "irrelevant citations", "inaccurate description", "others"]
                else:
                    print("invalid subtype")
                    raise ValueError("invalid subtype")
                
                response = self.openai_model.generate_response(prepare_subtype_classification_message(info['category'], limit))
                        
                try:
                    aspect = response
                    subtype = aspects[int(aspect.strip()[0])-1]
                except:
                    subtype = response
                
                if subtype == ground_truth_subtype:
                    acc = True
                    # GPT-4o judge
                    USER_INPUT = f"Ground truth limitation: \n{info['ground_truth']}\n\nGenerated limitation: \n{limit}\n\nProvide a brief explanation, then assign a rating (1-5)."
                    response = self.openai_model.generate_response([
                        {
                            "role": "system",
                            "content": RATING_PROMPT
                        },
                        {
                            "role": "user",
                            "content": USER_INPUT
                        }
                    ])
                    # print("Rating Response:", response)
                    try:
                        response_data = extract_info(r'```json\n(.*?)\n```', response)
                        response = json.loads(response_data)
                        explanation = response["explanation"]
                        rating = response["rating"]
                    except (json.JSONDecodeError, TypeError, KeyError):
                        explanation = "Invalid Message"
                        rating = 0
                        
            acc_list.append(acc)
            rating_list.append(rating)
            explanation_list.append(explanation)
            predicted_subtype_list.append(subtype)
                
                
        result["accuracy_list"] = acc_list
        result["predicted_subtype_list"] = predicted_subtype_list
        result["ground_truth_subtype"] = ground_truth_subtype
        
        result["explanation_list"] = explanation_list
        result["rating_list"] = rating_list
        result["rating"] = max(rating_list) if rating_list else 0
        result["accuracy"] = any(acc_list) if acc_list else False
        return result

    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:    
        for cnt in range(5):
            try:
                return self._evaluate_single(user_prompt, info, llm_response)
            except Exception as e:
                print(f"Error during evaluation (attempt {cnt+1}/5): {e}")
        return {
            "accuracy": False,
            "rating": 0,
            "explanation_list": ["Error"],
            "rating_list": [0],
            "accuracy_list": [False],
            "predicted_subtype_list": ["Error"],
            "ground_truth_subtype": find_ground_truth(info['category']),
        }

    def evaluate_single_only_one_metric(self, user_prompt: str, info: Dict[str, Any], llm_response: str, evaluate_single_result: Dict[str, float]) -> Dict[str, float]:
        return {
            "rating": evaluate_single_result["rating"]
        }
    
if __name__ == "__main__":
    # Example usage
    dataset = LimitGen_Dataset(data_path="./raw/LimitGen")
    
    
    # dataset = HelloBench_Dataset(data_path="./raw/HelloBench", dataset_name="HelloBench-Academic&Knowledge-QA")
    # print(len(dataset.dataset))
    
    # item = dataset.get_data(555)
    
    # print(">>>>>> LimitGen Dataset Length:")
    # print(len(dataset))
    
    # print("=" * 50)
    
    # print(">>>>>> LimitGen Dataset Item:")
    # print(json.dumps(item, ensure_ascii=False, indent=2))
    # with open("r.json") as f:
    #     r = json.load(f)
    # cnt = 0
    # for d in r:
    #     for item in dataset.dataset:
    #         if item['test_idx'] == d['test_idx']:
    #             cnt += 1
    #             score = dataset.evaluate_single(
    #                 user_prompt=item["input_chat_messages"][-1]["content"],
    #                 info=item['info'],
    #                 llm_response=d['response']
    #             )
    #             print(json.dumps(score, ensure_ascii=False, indent=2))
    #             if cnt == 3:
    #                 break
    # item = dataset.get_data(555)
    
    # print(">>>>>> LimitGen Dataset Length:")
    # print(len(dataset))
    
    # print("=" * 50)
    
    # print(">>>>>> LimitGen Dataset Item:")
    # print(json.dumps(item, ensure_ascii=False, indent=2))
    
    # print("=" * 50)
    
    
    # # Failed example
    # score = dataset.evaluate([{
    #         "test_idx": 555,
    #         "response": json.dumps({
    #             "limitations": "This is a limitation that does not match the ground truth.",
    #         }),
    # }])
    # print(">>>>> Failed case Evaluation Score:")
    # print(json.dumps(score, ensure_ascii=False, indent=2))
    
    # print("=" * 50)
    
    # # Successful example
    # score = dataset.evaluate([{
    #         "test_idx": 555,
    #         "response": json.dumps({
    #             "limitations": dataset.get_data(555)["info"]["ground_truth"]
    #         }),
    # }])
    # print(">>>>> Success case Evaluation Score:")
    # print(json.dumps(score, ensure_ascii=False, indent=2))