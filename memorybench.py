import os
import ast
import json
import datasets
import importlib
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Literal
from src.dataset.base import BaseDataset

load_dotenv()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------- Loading Datasets ----------------------------------------------

def get_dataset_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_single_dataset(dataset_name, eval_mode: bool = True) -> BaseDataset:
    assert os.path.exists(os.path.join(CURRENT_DIR, "configs/datasets/each.json")), "configs/datasets/each.json not found"
    with open(os.path.join(CURRENT_DIR, "configs/datasets/each.json"), "r") as fin:
        config = json.load(fin)
    if dataset_name not in config:
        raise ValueError(f"{dataset_name} not found, please choose from {config.keys()}")
    config = config[dataset_name]
    dataset_class_path = config["class_name"]
    dataset_class = get_dataset_class(f"src.dataset.{dataset_class_path}")
    dataset_config = config.copy() 
    for key in config:
        if key not in dataset_class.__init__.__code__.co_varnames:
            del dataset_config[key]
    dataset_config["eval_mode"] = eval_mode
    return dataset_class(**dataset_config)


def _load_domain_or_task(name, config_file, eval_mode: bool = False) -> List[BaseDataset]:
    assert os.path.exists(config_file), f"{config_file} not found"
    with open(config_file, "r") as fin:
        configs = json.load(fin)
    assert name in configs, f"{name} not found in {config_file}, please choose from {configs.keys()}"
    config_list = configs[name]
    dataset_list = []
    for config in config_list:
        dataset_class_path = config["class_name"]
        dataset_class = get_dataset_class(f"src.dataset.{dataset_class_path}")
        dataset_config = config.copy()
        sample_count = dataset_config.get("sample_count", None)
        for key in config:
            if key not in dataset_class.__init__.__code__.co_varnames:
                del dataset_config[key]
        dataset_config["eval_mode"] = eval_mode
        dataset_instance = dataset_class(**dataset_config)
        dataset_list.append(dataset_instance)
    return dataset_list


def load_domain(domain_name, eval_mode: bool = False) -> List[BaseDataset]:
    domain_config_file = os.path.join(CURRENT_DIR, "configs/datasets/domain.json")
    return _load_domain_or_task(domain_name, domain_config_file, eval_mode)


def load_task(task_name, eval_mode: bool = False) -> List[BaseDataset]:
    task_config_file = os.path.join(CURRENT_DIR, "configs/datasets/task.json")
    return _load_domain_or_task(task_name, task_config_file, eval_mode)

def load_memory_bench(
    dataset_type: Literal["single", "domain", "task"], 
    name: str, 
    eval_mode: bool = False
) -> BaseDataset | List[BaseDataset]:
    """
    Load datasets based on the dataset type.

    Args:
        dataset_type (Literal["single", "domain", "task"]): Type of the dataset to load.
        name (str): Name of the dataset, domain, or task.
        eval_mode (bool): Whether to load the dataset in evaluation mode.

    Returns:
        If dataset_type is "single", returns a single dataset instance.
        If dataset_type is "domain" or "task", returns a list of dataset instances.
    """
    if dataset_type == "single":
        return load_single_dataset(name, eval_mode)
    elif dataset_type == "domain":
        return load_domain(name, eval_mode)
    elif dataset_type == "task":
        return load_task(name, eval_mode)
    else:
        raise ValueError(f"Unknown dataset_type {dataset_type}, please choose from ['single', 'domain', 'task']")


# ------------------------------------------------ Evaluating ------------------------------------------------

def _evaluate(dataset_list: List[BaseDataset], predicts: List[Dict]) -> List[Dict]:
    total_detailed_results = []
    for dataset in dataset_list:
        dataset_name = dataset.dataset_name
        print(f"=== Evaluating dataset: {dataset_name} ===")
        cur_predicts = []
        for pp in predicts:
            if pp["dataset"] == dataset_name:
                cur_predicts.append(pp)
        detailed_results = dataset.evaluate(cur_predicts)
        for ret in detailed_results:
            ret["dataset"] = dataset_name
            total_detailed_results.append(ret)
    return total_detailed_results


def evaluate(
    dataset_type: Literal["single", "domain", "task"], 
    name: str,
    predicts: List[Dict], 
) -> List[Dict]:
    """
    Evaluate the predictions against the specified dataset(s).

    Args:
        dataset_type (Literal["single", "domain", "task"]): Type of the dataset to evaluate against.
        name (str): Name of the dataset, domain, or task.
        predicts (List[Dict]): List of prediction dictionaries, each containing 'test_idx', 'response', and 'dataset'.

    Returns:
        List[Dict]: List of evaluation details for each prediction.
    """
    for predict in predicts:
        assert "test_idx" in predict, "Each predict must have 'test_idx'"
        assert "response" in predict, "Each predict must have 'response'"
        if dataset_type == "single":
            if "dataset" in predict and predict["dataset"] is not None:
                assert predict["dataset"] == name, f"Predict dataset {predict['dataset']} does not match expected {name}"
            else:
                predict["dataset"] = name
        else:
            assert "dataset" in predict, "Each predict must have 'dataset'"

    dataset_list = load_memory_bench(dataset_type, name, eval_mode=True)
    if dataset_type == "single":
        dataset_list = [dataset_list]
    evaluate_details = _evaluate(dataset_list, predicts)
    return evaluate_details

# --------------------------------------------- Summary Results ------------------------------------------------

def summary_results(
    dataset_type: Literal["single", "domain", "task"], 
    name: str,    
    predicts: List[Dict], 
    evaluate_details: List[Dict], 
    min_max_config_file: str = "configs/final_evaluate_summary_wo_details.json",
):
    if dataset_type == "single":
        # for single dataset, just average the metrics
        dataset = load_single_dataset(name, eval_mode=False)
        test_metrics = dataset.test_metrics
        assert len(predicts) == len(evaluate_details), f"Length mismatch: {len(predicts)} vs {len(evaluate_details)}"
        summary = {met: [] for met in test_metrics}
        for item in evaluate_details:
            assert item["dataset"] == name, f"Dataset name mismatch: {item['dataset']} vs {name}"
            for met in test_metrics:
                value = item["metrics"].get(met, 0.0)
                summary[met].append(value if type(value) in [int, float] else (1 if value is True else 0))
        for met in summary:
            scores = summary[met]
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
            summary[met] = avg_score
        return {"summary": summary}

    else:
        # for domain and task, need to load min_max_config_file and merge metrics
        assert os.path.exists(min_max_config_file), f"min_max_config_file {min_max_config_file} not found"
        with open(min_max_config_file, "r") as fin:
            old_min_max_data = json.load(fin)
        try:
            dataset_min = old_min_max_data[dataset_type][name]["summary"]["dataset_min"]
            dataset_max = old_min_max_data[dataset_type][name]["summary"]["dataset_max"]
            dataset_mu = old_min_max_data[dataset_type][name]["summary"]["dataset_mu"]
            dataset_sigma = old_min_max_data[dataset_type][name]["summary"]["dataset_sigma"]
        except KeyError:
            raise KeyError(f"{dataset_type} {name} not found in {min_max_config_file}, please check the file")

        predicts = sorted(predicts, key=lambda x: (x["dataset"], x["test_idx"]))
        evaluate_details = sorted(evaluate_details, key=lambda x: (x["dataset"], x["test_idx"]))
        assert len(evaluate_details) == len(predicts), f"Length mismatch: {len(evaluate_details)} vs {len(predicts)}"

        assert os.path.exists(os.path.join(CURRENT_DIR, "configs/datasets/each.json")), "configs/datasets/each.json not found"
        with open(os.path.join(CURRENT_DIR, "configs/datasets/each.json"), "r") as fin:
            config = json.load(fin) 

        datasetname_to_class = {k: load_single_dataset(k, eval_mode=True) for k in config if len(config[k]["test_metrics"]) > 1} # datasets need to merge metrics
        
        values = {}
        for cur_idx, item in tqdm(enumerate(evaluate_details), desc="Merging Metrics", total=len(evaluate_details)):
            test_metrics = config[item["dataset"]]["test_metrics"]
            if item["dataset"].startswith("Locomo"):
                item["dataset"] = "Locomo"
            if item["dataset"] in datasetname_to_class: # merge metrics
                dataset_class = datasetname_to_class[item["dataset"]]
                predict_result = predicts[cur_idx]
                assert item["test_idx"] == predict_result["test_idx"], f"Index mismatch: {item['test_idx']}-{item['dataset']} vs {predict_result['test_idx']}-{predict_result['dataset']}"
                data_item = dataset_class.get_data(item["test_idx"])
                assert data_item["test_idx"] == item["test_idx"]
                res = dataset_class.evaluate_single_only_one_metric(
                    data_item["input_prompt"] if "input_prompt" in data_item else data_item["input_chat_messages"][-1]['content'],
                    data_item['info'], predict_result["response"], item["metrics"]
                )
                metrics_name = list(res.keys())[0]
            else:
                res = item["metrics"]
                metrics_name = test_metrics[0]
            dataset_name = item["dataset"]
            if dataset_name not in values:
                values[dataset_name] = []
            values[dataset_name].append(res[metrics_name] if type(res[metrics_name]) in [int, float] else (1 if res[metrics_name] is True else 0))

        total_ret = {"summary": {}, "average": {}, "minmax_normalized_average": {}, "z_normalized_average": {}}
        for dataset in values:
            scores = values[dataset]
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
            total_ret["average"][dataset] = avg_score

            normalized_score = [
                (s - dataset_min[dataset]) / (dataset_max[dataset] - dataset_min[dataset]) if dataset_max[dataset] > dataset_min[dataset] else 0.0
                for s in scores
            ]
            normalized_avg_score = sum(normalized_score) / len(normalized_score) if len(normalized_score) > 0 else 0.0
            total_ret["minmax_normalized_average"][dataset] = (sum(normalized_score), len(normalized_score), normalized_avg_score)

            z_scores = [
                (s - dataset_mu[dataset]) / dataset_sigma[dataset] if dataset_sigma[dataset] > 1e-6 else 0.0
                for s in scores
            ]
            z_avg_score = sum(z_scores) / len(z_scores) if len(z_scores) > 0 else 0.0
            total_ret["z_normalized_average"][dataset] = (sum(z_scores), len(z_scores), z_avg_score)

        avg_scores = []
        weighted_avg_scores = []
        z_scores = []
        total_count = 0
        not_complete = False
        for dataset in total_ret["minmax_normalized_average"]:
            score = total_ret["minmax_normalized_average"][dataset]
            avg_scores.append(score[2])
            count = score[1]
            weighted_avg_scores.append(score[0])
            total_count += count
            
            z = total_ret["z_normalized_average"][dataset]
            z_scores.append(z[0])
            assert z[1] == count
        overall_avg = sum(avg_scores) / len(avg_scores) if len(avg_scores) > 0 else 0.0
        overall_weighted_avg = sum(weighted_avg_scores) / total_count if total_count > 0 else 0.0
        total_ret["summary"]["average"] = overall_avg
        total_ret["summary"]["weighted_average"] = overall_weighted_avg
        overall_z = sum(z_scores) / total_count if total_count > 0 else 0.0
        total_ret["summary"]["z_score"] = overall_z
        return total_ret