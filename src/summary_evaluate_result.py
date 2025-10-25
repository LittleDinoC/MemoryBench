import json
import argparse
import os
from src.utils import get_single_dataset
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

def main(config_path, result_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tasks = set([config[k]["task_tag"] for k in config])
    domains = set([config[k]["domain_tag"] for k in config])
    end_results = {}
    
    datasetname_to_class = {k: get_single_dataset(k, config_path, False) for k in config if len(config[k]["test_metrics"]) > 1}
    
    
    for tag_type, tags in [("task", tasks), ("domain", domains)]:
        end_results[tag_type] = {}
        for tag in tags:
            end_results[tag_type][tag] = {"details": {}, "average": {}, "minmax_normalized_average": {}, "z_normalized_average": {}, "summary": {}}
            result_dir = os.path.join(result_path, tag_type, tag)
            baslines = os.listdir(result_dir)
            for b in baslines:
                baseline_dir = os.path.join(result_dir, b)
                result_dirs = os.listdir(baseline_dir)
                # 取最新的一个结果目录
                result_dirs = sorted(result_dirs, key=lambda x: os.path.getmtime(os.path.join(baseline_dir, x)), reverse=True)
                evaluate_details = json.load(open(os.path.join(baseline_dir, result_dirs[0], "evaluate_details.json"), "r"))
                predict_results = json.load(open(os.path.join(baseline_dir, result_dirs[0], "predict.json"), "r"))
                evaluate_details = sorted(evaluate_details, key=lambda x: (x["dataset"], x["test_idx"]))
                predict_results = sorted(predict_results, key=lambda x: (x["dataset"], x["test_idx"]))
                
                def solve_item(cur_idx, item):
                    if item["dataset"].startswith("Locomo"):
                        item["dataset"] = "Locomo"
                    if item["dataset"] not in end_results[tag_type][tag]["details"]:
                        end_results[tag_type][tag]["details"][item["dataset"]] = {}
                        end_results[tag_type][tag]["average"][item["dataset"]] = {}
                        end_results[tag_type][tag]["minmax_normalized_average"][item["dataset"]] = {}
                        end_results[tag_type][tag]["z_normalized_average"][item["dataset"]] = {}
                    if b not in end_results[tag_type][tag]["details"][item["dataset"]]:
                        end_results[tag_type][tag]["details"][item["dataset"]][b] = []
                
                    if item["dataset"] in datasetname_to_class:
                        dataset_class = datasetname_to_class[item["dataset"]]
                        # if predict_results is None:
                            # predict_results = json.load(open(os.path.join(baseline_dir, result_dirs[0], "predict.json"), "r"))
                        predict_result = predict_results[cur_idx]
                        assert item["test_idx"] == predict_result["test_idx"], f"{baseline_dir} {result_dirs[0]} Index mismatch: {item['test_idx']}-{item['dataset']} vs {predict_result['test_idx']}-{predict_result['dataset']}"
                        data_item = dataset_class.get_data(item["test_idx"])
                        assert data_item["test_idx"] == item["test_idx"]
                        # res = item["metrics"]
                        res = dataset_class.evaluate_single_only_one_metric(
                            data_item["input_prompt"] if "input_prompt" in data_item else data_item["input_chat_messages"][-1]['content'],
                            data_item['info'], predict_result["response"], item["metrics"]
                        )
                    else:
                        res = item["metrics"]
                    return item["dataset"], res

                assert len(evaluate_details) == len(predict_results), f"{baseline_dir} {result_dirs[0]} Length mismatch: {len(evaluate_details)} vs {len(predict_results)}"
                
                total_res = []
                with ThreadPoolExecutor(max_workers=4) as executor:       
                    # future_to_item = {executor.submit(solve_item, item): item for item in evaluate_details}
                    future_to_item = {executor.submit(solve_item, i, item): (i, item) for i, item in enumerate(evaluate_details)}
                    for future in tqdm(as_completed(future_to_item), desc=f"Processing {tag_type}-{tag}-{b}", total=len(future_to_item)):
                        dataset_name, res = future.result()
                        total_res.append((dataset_name, res))
                # for i, item in tqdm(enumerate(evaluate_details), desc=f"Processing {tag_type}-{tag}-{b}", total=len(evaluate_details)):

                for dataset_name, res in total_res:    
                    metrics_name = list(res.keys())[0]
                    end_results[tag_type][tag]["details"][dataset_name][b].append(res[metrics_name] if type(res[metrics_name]) in [int, float] else (1 if res[metrics_name] is True else 0))
            
            # 对domain或task计算平均值，min-max归一化，并记录归一化所需的最大最小值，以及中位数
            dataset_level = {dataset: [] for dataset in end_results[tag_type][tag]["details"]}
            for dataset in end_results[tag_type][tag]["details"]:
                for b in end_results[tag_type][tag]["details"][dataset]:
                    dataset_level[dataset].extend(end_results[tag_type][tag]["details"][dataset][b])
            dataset_min = {dataset: min(dataset_level[dataset]) for dataset in dataset_level}
            dataset_max = {dataset: max(dataset_level[dataset]) for dataset in dataset_level}
            dataset_mu = {dataset: float(np.mean(dataset_level[dataset])) for dataset in dataset_level}
            dataset_sigma = {dataset: float(np.std(dataset_level[dataset])) for dataset in dataset_level}
            
            
            
            # 计算平均值，以及Robust Z-score
            for dataset in end_results[tag_type][tag]["details"]:
                for b in end_results[tag_type][tag]["details"][dataset]:
                    scores = end_results[tag_type][tag]["details"][dataset][b]
                    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0
                    end_results[tag_type][tag]["average"][dataset][b] = avg_score
                    
                    normalized_score = [
                        (s - dataset_min[dataset]) / (dataset_max[dataset] - dataset_min[dataset]) if dataset_max[dataset] > dataset_min[dataset] else 0.0
                        for s in scores
                    ]
                    normalized_avg_score = sum(normalized_score) / len(normalized_score) if len(normalized_score) > 0 else 0.0
                    end_results[tag_type][tag]["minmax_normalized_average"][dataset][b] = (sum(normalized_score), len(normalized_score), normalized_avg_score)
                    

                    z_scores = [
                        (s - dataset_mu[dataset]) / dataset_sigma[dataset] if dataset_sigma[dataset] > 1e-6 else 0.0
                        for s in scores
                    ]
                    z_avg_score = sum(z_scores) / len(z_scores) if len(z_scores) > 0 else 0.0
                    end_results[tag_type][tag]["z_normalized_average"][dataset][b] = (sum(z_scores), len(z_scores), z_avg_score)
                    
            # 计算summary
            end_results[tag_type][tag]["summary"] = {
                "dataset_min": dataset_min,
                "dataset_max": dataset_max,
                "dataset_mu": dataset_mu,
                "dataset_sigma": dataset_sigma,
                "average": {},
                "weighted_average": {},
                "z_score": {}
            }
            
            for b in baslines:
                avg_scores = []
                weighted_avg_scores = []
                z_scores = []
                total_count = 0
                not_complete = False
                for dataset in end_results[tag_type][tag]["minmax_normalized_average"]:
                    if b in end_results[tag_type][tag]["minmax_normalized_average"][dataset]:
                        score = end_results[tag_type][tag]["minmax_normalized_average"][dataset][b]
                        avg_scores.append(score[2])
                        count = score[1]
                        weighted_avg_scores.append(score[0])
                        total_count += count
                        
                        z = end_results[tag_type][tag]["z_normalized_average"][dataset][b]
                        z_scores.append(z[0])
                        assert z[1] == count
                    else:
                        not_complete = True
                        break
                if not_complete:
                    end_results[tag_type][tag]["summary"]["average"][b] = None
                    end_results[tag_type][tag]["summary"]["weighted_average"][b] = None
                    continue
                overall_avg = sum(avg_scores) / len(avg_scores) if len(avg_scores) > 0 else 0.0
                overall_weighted_avg = sum(weighted_avg_scores) / total_count if total_count > 0 else 0.0
                end_results[tag_type][tag]["summary"]["average"][b] = overall_avg
                end_results[tag_type][tag]["summary"]["weighted_average"][b] = overall_weighted_avg
                overall_z = sum(z_scores) / total_count if total_count > 0 else 0.0
                end_results[tag_type][tag]["summary"]["z_score"][b] = overall_z
                
    with open(os.path.join(result_path, "final_evaluate_summary_details.json"), "w") as f:
        json.dump(end_results, f, indent=4)
    with open(os.path.join(result_path, "final_evaluate_summary_wo_details.json"), "w") as f:
        json.dump({
            tag_type: {
                tag: {
                    "average": end_results[tag_type][tag]["average"],
                    "minmax_normalized_average": end_results[tag_type][tag]["minmax_normalized_average"],
                    "z_normalized_average": end_results[tag_type][tag]["z_normalized_average"],
                    "summary": end_results[tag_type][tag]["summary"]
                } for tag in end_results[tag_type]
            } for tag_type in end_results
        }, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/datasets/each.json")
    parser.add_argument("--result_path", type=str, default="off-policy/results")

    args = parser.parse_args()
    
    main(args.config_path, args.result_path)