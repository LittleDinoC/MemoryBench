import os
import sys
import json
import argparse
from pathlib import Path
from termcolor import colored
from dotenv import load_dotenv
from src.utils import get_single_dataset, get_dataset_series

load_dotenv()


def evaluate_and_summary(dataset_list, predicts):
    total_detailed_results = []
    for dataset in dataset_list:
        dataset_name = dataset.dataset_name
        print(dataset_name)
        cur_predicts = []
        for pp in predicts:
            if pp["dataset"] == dataset_name:
                cur_predicts.append(pp)
        print(colored(f"Evaluating dataset {dataset_name} with {len(cur_predicts)} samples", "yellow"))
        detailed_results = dataset.evaluate(cur_predicts)
        for ret in detailed_results:
            ret["dataset"] = dataset_name
            total_detailed_results.append(ret)
    return total_detailed_results


def evaluate_dir(dirpath, dataset_list):
    predict_file = os.path.join(dirpath, "test_predicts.json")
    if not os.path.exists(predict_file):
        if os.path.exists(os.path.join(dirpath, "predict.json")):
            predict_file = os.path.join(dirpath, "predict.json")
        else:
            return
    save_path = dirpath
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, "evaluate_details.json")
    print(results_file)
    cnt = 3
    while not os.path.exists(results_file):
        if cnt == 0:
            break
        cnt -= 1
        print(colored(f"Evaluating {predict_file}", "yellow"))
        try:
            with open(predict_file, "r") as fin:
                predicts = json.load(fin)
            # merged_results, detailed_results = dataset.evaluate_and_summary(predicts)
            detailed_results = evaluate_and_summary(dataset_list, predicts)
            with open(results_file, "w") as fout:
                json.dump(detailed_results, fout, indent=4)
            print(colored(f"Finished evaluating {predict_file}", "green"))
        except Exception as e:
            print(e)
            print(colored(f"Error in evaluating {predict_file}, retrying...", "red"))


def evalaute_all(typ_name, dir_path, dataset_config_file):
    print(colored(f"Evaluating all results in {dir_path}", "blue"))
    dataset_list = get_dataset_series(typ_name, dataset_config_file, eval_mode=True)
    for method in dir_path.iterdir():
        if not method.is_dir():
            continue
        for times in method.iterdir():
            if not times.is_dir():
                continue
            evaluate_dir(times, dataset_list)
            for sub_dir in times.iterdir():
                if sub_dir.is_dir():
                    evaluate_dir(sub_dir, dataset_list)


def main(result_path):
    for aaa in ["domain", "task"]:
        output_root_path = os.path.join(result_path, aaa)
        if not os.path.exists(output_root_path):
            continue
        for typ_name in Path(output_root_path).iterdir():
            ttt = typ_name.name
            if ttt.endswith("_with_delete"):
                ttt = ttt[: -len("_with_delete")]
            evalaute_all(ttt, typ_name, f"configs/datasets/{aaa}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="off-policy/results")

    args = parser.parse_args()
    main(args.result_path)