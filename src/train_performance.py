import os
import json
import copy
import random
import shutil
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
from argparse import ArgumentParser
from termcolor import colored
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.utils import (
    if_memory_cached, 
    get_memory_system_config_file,
    get_dialog_key,
    load_corpus_to_memory,
    evaluate_and_summary,
)
from src.solver import SolverFactory

from memorybench import load_memory_bench

# try:
#     import nltk
#     nltk.data.find('wordnet')
# except LookupError:
#     print("Downloading WordNet data...")
#     nltk.download('wordnet')


def build_solver(
    cache_save_dir,
    args,
    copy_from_memory_cache_dir=None,
):
    """
        Build and return a solver instance based on the provided arguments.
        Args:
            cache_save_dir (str): Directory to save the cache.
            args: Parsed command line arguments containing configuration for the solver.
            copy_from_memory_cache_dir (str, optional): Directory to copy the memory cache from.
        Returns:
            solver: An instance of the solver created based on the provided configuration.
            memory_cache_dir (str): Directory where the memory cache is stored.
    """
    memory_cache_dir = os.path.join(
        args.memory_cache_prefix + cache_save_dir, 
        args.dataset_type,
        args.set_name,
        args.memory_system,
    )
    if copy_from_memory_cache_dir is None:
        if not if_memory_cached(memory_cache_dir) and os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)
    else:
        assert os.path.exists(copy_from_memory_cache_dir), f"Memory cache dir {copy_from_memory_cache_dir} does not exist."
        if os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)
        shutil.copytree(copy_from_memory_cache_dir, memory_cache_dir)
        print(f"Copied memory cache from {copy_from_memory_cache_dir} to {memory_cache_dir}.")
    solver_config = {
        "method_name": args.memory_system,
        "config": args.memory_system_config,
        "memory_cache_dir": memory_cache_dir,
    }
    if args.retrieve_k is not None:
        solver_config["retrieve_k"] = args.retrieve_k
    print("Solver config:", solver_config)
    solver = SolverFactory.create(**solver_config)
    solver.MAX_THREADS = args.threads
    return solver, memory_cache_dir


def main(args):
    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")

    if args.dataset_type == "single":
        dataset_lists = [load_memory_bench(args.dataset_type, args.set_name)]
    else:
        dataset_lists = load_memory_bench(args.dataset_type, args.set_name)

    # split train, test
    total_dialogs = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        if args.memory_system != "wo_memory":
            if not dataset.has_corpus:
                dialog_key = "dialog"
            else:
                dialog_key = get_dialog_key(args.memory_system)
            for data in dataset.dataset["train"].to_list():
                test_idx = data["test_idx"]
                dialog = data[dialog_key]
                total_dialogs.append({
                    "test_idx": test_idx,
                    "dialog": dialog,
                })
        print("Loaded {} dialogs from dataset {} and use {} data for testing".format(
            len(dataset.dataset["train"]), 
            dataset_name, 
            len(dataset.dataset["test"])
        )) 
    print(f"Loaded {len(total_dialogs)} dialogs for memory creation.")
    random.seed(42)
    random.shuffle(total_dialogs)

    # load configuration
    with open(args.memory_system_config, "r") as fin:
        args.memory_system_config = json.load(fin)
        print(args.memory_system_config)
    
    memory_solver, dialog_memory_cache_dir = build_solver("memory_cache", args, None)
    memory_solver.create_or_load_memory(total_dialogs)

    total_predicts = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        print(f"Evaluating dataset {dataset_name} with {len(dataset.dataset['train'])} test data.")

        if not dataset.has_corpus:
            predicts = memory_solver.predict_test(dataset, split_name="train")
        else:
            if "wo_memory" == args.memory_system:
                predicts = memory_solver.predict_test_with_corpus(dataset, split_name="train")
            else:
                if "bm25" in args.memory_system:
                    single_solver, _ = build_solver(
                        f"running_cache/single_{start_timestamp}/{dataset_name}", 
                        args,
                        None,
                    )
                    single_solver.create_or_load_memory(total_dialogs)
                else:
                    single_solver, _ = build_solver(
                        f"running_cache/single_{start_timestamp}/{dataset_name}", 
                        args,
                        dialog_memory_cache_dir
                    )
                    single_solver.agent.load_memories()
                load_corpus_to_memory(single_solver, dataset) 
                predicts = single_solver.predict_test(dataset, split_name="train")
                del single_solver
        for pred in predicts:
            pred["dataset"] = dataset_name
            total_predicts.append(pred)

    # Save results
    output_dir = os.path.join(
        args.output_dir, 
        args.dataset_type,
        args.set_name,
        args.memory_system, 
        f"start_at_{start_timestamp}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def save_result(data, filename):
        with open(os.path.join(output_dir, filename), "w") as fout:
            json.dump(data, fout, indent=4, ensure_ascii=False)
    
    save_result(vars(args), "run_config.json")
    save_result(total_predicts, "predict.json")

    evaluate_and_summary(args.dataset_type, args.set_name, total_predicts, output_dir)



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
        "--memory_system",
        type=str,
        required=True, 
        help="The memory system to use",
        choices=["wo_memory", "a_mem", "mem0", "memoryos", "bm25_message", "embedder_message", "raptor", "bm25_dialog", "embedder_dialog"],
    )
    parser.add_argument(
        "--memory_system_config",
        type=str,
        # required=True,
        default=None,
        help="Path to the memory system configuration file",
    )
    parser.add_argument(
        "--memory_cache_prefix",
        type=str,
        default="train_performance/",
        help="Prefix path to copy memory cache from",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="train_performance/result/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of threads to use for processing dialogs",
    )
    parser.add_argument(
        "--retrieve_k", 
        type=int,
        default=5,
        help="Number of memories to retrieve for each query",
    ) # 如果 memory_system_config 中有 retrieve_k 这一项，则覆盖
    
    args = parser.parse_args()
    args.memory_system_config = get_memory_system_config_file(args.memory_system, args.memory_system_config)
    print(args)
    main(args)