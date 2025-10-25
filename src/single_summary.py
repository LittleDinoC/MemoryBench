import json
import argparse
import os
from pathlib import Path
from src.utils import get_single_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


def main(config_path, result_path, old_min_max_data):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tasks = set([config[k]["task_tag"] for k in config])
    domains = set([config[k]["domain_tag"] for k in config])
    end_results = {}
    
    datasetname_to_class = {k: get_single_dataset(k, config_path, False) for k in config if len(config[k]["test_metrics"]) > 1}
    
    for tag_type, tags in [("task", tasks), ("domain", domains)]:
        for tag in tags:
            dataset_min = old_min_max_data[tag_type][tag]["summary"]["dataset_min"]
            dataset_max = old_min_max_data[tag_type][tag]["summary"]["dataset_max"]
            dataset_mu = old_min_max_data[tag_type][tag]["summary"]["dataset_mu"]
            dataset_sigma = old_min_max_data[tag_type][tag]["summary"]["dataset_sigma"]
            
            cur_path = os.path.join(result_path, tag_type, tag)
            if not os.path.exists(cur_path):
                continue
            for baseline in Path(cur_path).iterdir():
                # runtime_dirs = os.listdir(baseline)
                # runtime_dirs = sorted(runtime_dirs, key=lambda x: os.path.getmtime(os.path.join(baseline, x)), reverse=True)
                # runtime = Path(os.path.join(baseline, runtime_dirs[0]))

                def solve(aaa):
                    print(aaa)
                    if os.path.exists(os.path.join(aaa, "summary.json")):
                        return

                    if not os.path.exists(os.path.join(aaa, "evaluate_details.json")):
                        return
                    evaluate_details = json.load(open(os.path.join(aaa, "evaluate_details.json"), "r"))
                    if os.path.exists(os.path.join(aaa, "predict.json")): 
                        predict_results = json.load(open(os.path.join(aaa, "predict.json"), "r"))
                    elif os.path.exists(os.path.join(aaa, "test_predicts.json")):
                        predict_results = json.load(open(os.path.join(aaa, "test_predicts.json"), "r"))
                    else:
                        return
                    evaluate_details = sorted(evaluate_details, key=lambda x: (x["dataset"], x["test_idx"]))
                    predict_results = sorted(predict_results, key=lambda x: (x["dataset"], x["test_idx"]))
                    assert len(evaluate_details) == len(predict_results), f"{baseline_dir} {result_dirs[0]} Length mismatch: {len(evaluate_details)} vs {len(predict_results)}"

                    def solve_item(cur_idx, item):
                        # if item["dataset"].startswith("Locomo") or item["dataset"].startswith("DialSim"):
                        #     return None, None

                        if item["dataset"].startswith("Locomo"):
                            item["dataset"] = "Locomo"
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


                    total_res = []
                    with ThreadPoolExecutor(max_workers=4) as executor:       
                        # future_to_item = {executor.submit(solve_item, item): item for item in evaluate_details}
                        future_to_item = {executor.submit(solve_item, i, item): (i, item) for i, item in enumerate(evaluate_details)}
                        for future in tqdm(as_completed(future_to_item), desc=f"Processing {aaa}", total=len(future_to_item)):
                            dataset_name, res = future.result()
                            if dataset_name is None and res is None:
                                continue
                            total_res.append((dataset_name, res))
                    print(len(total_res))
                    values = {}
                    for dataset_name, res in total_res:
                        metrics_name = list(res.keys())[0]
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

                    with open(os.path.join(aaa, "summary.json"), "w") as fout:
                        json.dump(total_ret, fout, indent=4)

                for runtime in Path(baseline).iterdir():
                    if not runtime.is_dir():
                        continue
                    flag = False
                    for subdir in Path(runtime).iterdir():
                        if os.path.isdir(subdir):
                            flag = True
                            # solve(subdir)
                    if not flag:
                        solve(runtime)
                    else:
                        for subdir in Path(runtime).iterdir():
                            if os.path.isdir(subdir):
                                solve(subdir)
                
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/datasets/each.json")
    parser.add_argument("--result_path", type=str, default="off-policy/results/")
    parser.add_argument("--default_min_max_path", type=str, default="configs/final_evaluate_summary_wo_details.json")

    args = parser.parse_args()
    
    main(args.config_path, args.result_path, old_min_max_data = json.load(open(args.default_min_max_path, "r")))