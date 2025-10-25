import os
import json
from typing import List, Dict, Any, Tuple
import random
import math
from copy import deepcopy

def fixed_sample(data, k, seed=42):
    rng = random.Random(seed)
    return rng.sample(data, k)


class BaseDataset:
    """
    基础数据集类。提供了数据集的基本结构和方法
    """

    def __init__(self, data_path: str = None, test_metrics: List[str] = None, max_output_len: int = None):
        """
        初始化数据集类

        Args:
            data_path (str): 数据集路径
        
        Returns:
            None
        """

        self.data_path = data_path
        self.test_metrics = test_metrics
        self.dataset = self._load_data()
        self.max_output_len = max_output_len if max_output_len is not None else 8192
        for set_name in ["train", "test"]:
            for data in self.dataset[set_name]:
                assert "test_idx" in data, "no 'test_idx' field in data"
                # assert data["test_idx"] == did, "test_idx must be a continuous integer starting from 0"
                assert "input_prompt" in data or "input_chat_messages" in data, "no 'input_prompt' or 'input_chat_messages' field in data"
                assert "dataset_name" in data, "no 'dataset_name' field in data"
                assert "info" in data, "no 'info' field in data"
                assert "lang" in data, "no 'lang' field in data"

        self.total_data = []
        self.test_idx_to_data = {}
        for set_name in ["train", "test"]:
            for data in self.dataset[set_name]:
                tmp_data = deepcopy(data)
                tmp_data["set_name"] = set_name
                self.total_data.append(tmp_data)
                assert data["test_idx"] not in self.test_idx_to_data, "duplicate test_idx found {}".format(data["test_idx"])
                self.test_idx_to_data[data["test_idx"]] = tmp_data
        self.total_size = len(self.total_data)
        
    def __len__(self):
        """
        返回数据集的大小

        Returns:
            int: 数据集的大小
        """
        return self.total_size
    
    # def get_test_ids(self, truncate_size: int = 500, test_ratio: float = 0.2) -> Dict[str, List[int]]:
    #     """
    #     使用至多 truncate_size 个构造 train + test 集合，然后从中按比例划分测试集
    #     获取所有测试数据的索引，默认为20%的数据（取上整）

    #     Returns:
    #         Dict[str, List[int]]: "train": 训练集索引列表, "test": 测试集索引列表
    #     """
    #     if truncate_size is None or self.total <= truncate_size:
    #         ids = [data["test_idx"] for data in self.dataset]
    #         test_size = math.ceil(test_ratio * len(ids))
    #         test_ids = fixed_sample(ids, test_size, seed=42)
    #         train_ids = [i for i in ids if i not in test_ids]
    #         return {
    #             "train": train_ids,
    #             "test": test_ids
    #         }
    #     else:
    #         ids = [data["test_idx"] for data in self.dataset]
    #         # 随机选取 truncate_size 个
    #         ids = fixed_sample(ids, truncate_size, seed=42)
    #         test_size = math.ceil(test_ratio * len(ids))
    #         test_ids = fixed_sample(ids, test_size, seed=42)
    #         train_ids = [i for i in ids if i not in test_ids]
    #         return {
    #             "train": train_ids,
    #             "test": test_ids
    #         }
        
    def _load_data(self):
        from src.dataset.utils import load_from_hf  
        dataset_dict = load_from_hf(self.dataset_name)
        if dataset_dict["corpus"] is None:
            self.has_corpus = False
        else:
            self.has_corpus = True
            self.corpus = dataset_dict["corpus"]
            self.session_cnt = dataset_dict["session_cnt"] 
        return dataset_dict["dataset"]
    
    def get_data(self, test_idx):
        """
        根据测试数据的索引获取对应的数据

        Args:
            test_idx (int): 测试数据的索引
        
        Returns:
            Dict: 对应的数据字典
        """
        if test_idx not in self.test_idx_to_data:
            raise ValueError(f"test_idx {test_idx} not found in dataset")
        return self.test_idx_to_data[test_idx]
        

    def get_initial_chat_messages(self, test_idx: int) -> List[Dict[str, str]]:
        """
        获取初始聊天消息

        Args:
            test_idx (int): 测试数据的索引
        
        Returns:
            List[Dict[str, str]]: 初始聊天消息列表，每个消息是一个字典，包含角色和内容
        """
        data = self.get_data(test_idx)
        user_prompt = data.get("input_prompt", "")
        if not user_prompt:
            if "input_chat_messages" in data:
                messages = data["input_chat_messages"]
            else:
                raise ValueError("Data must contain either 'input_prompt' or 'input_chat_messages'")
        else:
            messages = [{
                "role": "user",
                "content": user_prompt,
            }]
        return messages
    
    def evaluate_single(self, user_prompt: str, info: Dict[str, Any], llm_response: str) -> Dict[str, float]:
        """
        用于根据模型的输出执行自动化评估

        Args:
            user_prompt (str): 提供给模型的用户提示
            info (Dict[str, Any]): 该数据点的附加信息，通常包含真实标签（ground truth）
            llm_response (str): 大语言模型生成的输出

        Returns:
            Dict[str, float]: 返回一个包含评估指标的字典，例如 {'accuracy': 1.0, 'f1': 0.8}。
        """
        raise NotImplementedError
    
    def evaluate_single_only_one_metric(self, user_prompt: str, info: Dict[str, Any], llm_response: str, evaluate_single_result: Dict[str, float]) -> Dict[str, float]:
        """
        用于根据模型的输出执行自动化评估，只返回主实验表格展示的一个指标

        Args:
            user_prompt (str): 提供给模型的用户提示
            info (Dict[str, Any]): 该数据点的附加信息，通常包含真实标签（ground truth）
            llm_response (str): 大语言模型生成的输出

        Returns:
            Dict[str, float]: 返回一个包含评估指标的字典，例如 {'accuracy': 1.0}。
        """
        return evaluate_single_result

    def evaluate(self, responses: List[Dict]) -> List[Dict]:
        """
        评估模型的响应

        Args:
            responses (List[Dict]): 模型的响应列表, 每个都应当是 {"test_idx": int, "response": str} 的格式
        
        Returns:
            List[Dict]: 评估结果列表, 每个都应当是 {"test_idx": int, "metrics": Dict[str, float]} 的格式
        """
        results = []
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # print(self.dataset_name, len(responses))

        def _evaluate_single(resp):
            test_idx = resp["test_idx"]
            llm_response = resp["response"]
            data = self.get_data(test_idx)
            user_prompt = data.get("input_prompt", "")
            if not user_prompt:
                if "input_chat_messages" in data:
                    user_prompt = data["input_chat_messages"]
                else:
                    raise ValueError("Data must contain either 'input_prompt' or 'input_chat_messages'")
            info = data["info"]
            metrics = self.evaluate_single(user_prompt, info, llm_response) 
            return {
                "test_idx": test_idx,
                "metrics": metrics
            }
        max_threads = self.evaluate_threads if hasattr(self, 'evaluate_threads') else 1
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(_evaluate_single, resp) for resp in responses]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating responses"):
                results.append(future.result())
        results.sort(key=lambda x: x["test_idx"])
        assert len(results) == len(responses), "Some evaluations are missing"

        # for resp in tqdm(responses, desc="Evaluating responses"):
        #     test_idx = resp["test_idx"]
        #     llm_response = resp["response"]
        #     data = self.get_data(test_idx)
        #     user_prompt = data.get("input_prompt", "")
        #     if not user_prompt:
        #         if "input_chat_messages" in data:
        #             user_prompt = data["input_chat_messages"]
        #         else:
        #             raise ValueError("Data must contain either 'input_prompt' or 'input_chat_messages'")
        #     info = data["info"]
        #     # if single_metrics:
        #     #     metrics = self.evaluate_single_only_one_metric(user_prompt, info, llm_response)
        #     # else:
        #     metrics = self.evaluate_single(user_prompt, info, llm_response)
        #     results.append({
        #         "test_idx": test_idx,
        #         "metrics": metrics
        #     })
        return results
    
    def evaluate_test(self, responses: List[Dict]) -> List[Dict]:
        """
        评估测试集的模型响应，但只保留测试指标

        Args:
            responses (List[Dict]): 模型的响应列表, 每个都应当是 {"test_idx": int, "response": str} 的格式
        
        Returns:
            List[Dict]: 评估结果列表, 每个都应当是 {"test_idx": int, "metrics": Dict[str, float]} 的格式
        """
        results = self.evaluate(responses)
        # print("Full evaluation results:", results)
        if not self.test_metrics:
            return results
        test_results = []
        for result in results:
            test_idx = result["test_idx"]
            metrics = {k: v for k, v in result["metrics"].items() if k in self.test_metrics}
            test_results.append({
                "test_idx": test_idx,
                "metrics": metrics
            })
        return test_results
    
    def evaluate_and_summary(self, responses: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """
        评估测试集的模型相应，返回在该数据集上的整体结果 和 每一个数据点的详细评测结果

        Args:
            responses (List[Dict]): 模型的响应列表, 每个都应当是 {"test_idx": int, "response": str} 的格式

        Returns:
            Dict: 整体评估结果
            List[Dict]: 每个数据点的详细评测结果
        """
        detailed_results = self.evaluate_test(responses)
        if not self.test_metrics:
            return {}, detailed_results

        overall_metrics = {}
        for metric in self.test_metrics:
            if metric not in detailed_results[0]["metrics"]:
                # 只有JuDGE的F1，因为要计算avg f1
                assert metric.endswith("_f1"), f"Metric {metric} not found in detailed results"
                avg_recall = sum(result["metrics"].get(metric.replace("_f1", "_recall"), 0) for result in detailed_results) / len(detailed_results)
                avg_precision = sum(result["metrics"].get(metric.replace("_f1", "_precision"), 0) for result in detailed_results) / len(detailed_results)
                overall_metrics[metric] = 2 * (avg_recall * avg_precision) / (avg_recall + avg_precision) if (avg_recall + avg_precision) > 0 else 0
            else:
                if metric in ["reasoning_meteor", "judge_meteor", "time_score", "amount_score"]:
                    # 分母要去掉None的值
                    valid_results = [result for result in detailed_results if result["metrics"].get(metric) is not None]
                    if valid_results:
                        overall_metrics[metric] = sum(
                            result["metrics"].get(metric, 0) for result in valid_results
                        ) / len(valid_results)
                    else:
                        overall_metrics[metric] = None
                else:
                    overall_metrics[metric] = sum(
                        result["metrics"].get(metric, 0) for result in detailed_results
                    ) / len(detailed_results)

        return overall_metrics, detailed_results