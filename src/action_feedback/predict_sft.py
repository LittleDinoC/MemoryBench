import os
import json
import copy
import random
import shutil
from tqdm import tqdm
import requests
from datetime import datetime
from typing import List, Dict
from argparse import ArgumentParser
from termcolor import colored
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.utils import (
    if_memory_cached, 
    evaluate_and_summary,
)
from src.solver import SolverFactory


def main(args):
    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")

    if args.dataset_type == "single":
        dataset_lists = [load_memory_bench(args.dataset_type, args.set_name)]
    else:
        dataset_lists = load_memory_bench(args.dataset_type, args.set_name)

    output_dir = os.path.join(
        args.output_dir, 
        args.dataset_type,
        args.set_name,
        "sft",
        str(args.learning_rate),
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    
    vllm_config = json.load(open("configs/memory_systems/base.json"))
    solver = SolverFactory.create(
        method_name="wo_memory",
        config=vllm_config,
    )
    solver.MAX_THREADS = args.threads
    
    lora_path = os.path.join(
        f"./action_feedback/sft_models/lr={args.learning_rate}/", 
        args.set_name, 
        f"lora_r8_alpha32_{args.action_feedback}_only",
        f"epoch={args.num_epochs}_ckpt"
    )
    if not os.path.exists(lora_path):
        print(colored(f"LoRA path {lora_path} not found, skipping.", "yellow"))
        continue
    
    save_path = os.path.join(output_dir, action, f"epoch={epoch}")
    
    # add lora adapter to vllm
    url = vllm_config["llm_config"]["vllm_base_url"] + "/load_lora_adapter"
    headers = {"Content-Type": "application/json"}
    data = {
        "lora_name": "sft_adapter",
        "lora_path": lora_path
    }
    
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200, f"Failed to load LoRA adapter: {response.text}"
    print(colored(f"Loaded LoRA adapter from {lora_path}", "green"))
    total_predicts = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        print(f"Evaluating dataset {dataset_name} with {len(test_ids[dataset_name])} test data.")
        predicts = solver.predict_test(dataset)
        for pred in predicts:
            pred["dataset"] = dataset_name
            total_predicts.append(pred)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "predict.json"), "w") as fout:
        json.dump(total_predicts, fout, indent=4, ensure_ascii=False)
    print(colored(f"Saved predictions to {save_path}", "green"))
    evaluate_and_summary(args.dataset_type, args.set_name, total_predicts, save_path) 

    # unload lora adapter
    url = vllm_config["llm_config"]["vllm_base_url"] + "/unload_lora_adapter"
    headers = {"Content-Type": "application/json"}
    data = {
        "lora_name": "sft_adapter",
    }
    response = requests.post(url, headers=headers, json=data)
    assert response.status_code == 200, f"Failed to unload LoRA adapter: {response.text}"
    print(colored(f"Unloaded LoRA adapter sft_adapter", "green"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["single", "domain", "task"],
        required=True,
    ) 
    parser.add_argument(
        "--set_name",
        type=str,
        required=True,
        help="Name of the dataset/domain/task",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="action_feedback/results/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of threads to use for processing dialogs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--action_feedback",
        type=str,
        choices=["like", "copy"],
        required=True,
    )
    args = parser.parse_args()
    print(args)
    main(args)