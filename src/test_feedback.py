import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.solver import SolverFactory
from src.solver.base import BaseSolver
from src.utils import get_dialog_key, load_corpus_to_memory, get_memory_system_config_file

from memorybench import load_memory_bench


def process_single_data(
    solver: BaseSolver,
    data, 
    dialogs: List[Dict[str, str]],
    dataset: BaseDataset,
):    
    dialog_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in dialogs]
    ) + "\n"

    try:
        messages = dataset.get_initial_chat_messages(data["test_idx"])
        # work like bm25_dialog solver
        question = messages[-1]["content"]
        if data["lang"] == "en":
            user_prompt = f"""Context:
    {dialog_text}

    User: 
    {question}

    Based on the context provided, respond naturally and appropriately to the user's input above."""
        elif data["lang"] == "zh":
            user_prompt = f"""相关知识：
    {dialog_text}

    用户输入：
    {question}

    请根据提供的相关知识准确、自然地回答用户的输入。"""

        messages[-1]["content"] = user_prompt
        response = solver.agent.generate_response(messages=messages)
        return {
            "test_idx": data["test_idx"],
            "response": response,
            "messages": messages,
        }
    except Exception as e:
        print(f"Error processing test_idx {data['test_idx']}: {e}")
        return {
            "test_idx": data["test_idx"],
            "response": "",
            "messages": [],
            "error": str(e),
        }


def save_json_file(save_dir, filename, data):
    with open(os.path.join(save_dir, filename), "w") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def main(args):
    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    assert not args.dataset.startswith("DialSim"), f"Dataset {args.dataset} not supported yet."
    if args.dataset.startswith("Locomo"):
        assert args.memory_system != "wo_memory", f"Dataset {args.dataset} not supported for wo_memory system."
    else:
        assert args.memory_system == "wo_memory", f"Dataset {args.dataset} only supported for wo_memory system."

    dataset = load_memory_bench("single", args.dataset)

    # load solver 
    with open(args.memory_system_config, "r") as fin:
        memory_system_config = json.load(fin)
    memory_system_config["llm_config"]["max_tokens"] = min(
        memory_system_config.get("max_tokens", 2048),
        dataset.max_output_len,
    )
    print("\n", memory_system_config, "\n")

    tmp_memory_cache_dir = os.path.join("tmp/memory_cache", args.dataset, args.memory_system)
    if os.path.exists(tmp_memory_cache_dir):
        import shutil
        shutil.rmtree(tmp_memory_cache_dir)
    solver = SolverFactory.create(
        method_name=args.memory_system,
        config=memory_system_config,
        memory_cache_dir=tmp_memory_cache_dir,
    )
    solver.MAX_THREADS = args.threads
    if args.dataset.startswith("Locomo"):
        load_corpus_to_memory(solver, dataset)

    train_predicts = []
    dialog_key = get_dialog_key(args.memory_system) if dataset.has_corpus else "dialog"
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_single_data, 
                            solver, 
                            data,
                            data[dialog_key],
                            dataset)
            for data in dataset.dataset["train"].to_list()[: args.sample]
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting on train set"):
            result = future.result()
            train_predicts.append(result)
    train_predicts.sort(key=lambda x: x["test_idx"])

    test_predicts = []
    for data in dataset.dataset["test"].to_list()[: args.sample]:
        resp = ""
        for msg in data[dialog_key]:
            if msg["role"] == "assistant":
                resp = msg["content"]
        test_predicts.append({
            "test_idx": data["test_idx"],
            "response": resp,
        })
    test_predicts.sort(key=lambda x: x["test_idx"])

    # # save results
    save_dir = os.path.join(
        args.output_dir,
        args.dataset, 
        args.memory_system,
        f"start_at_{start_timestamp}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_json_file(save_dir, "run_config.json", vars(args))
    save_json_file(save_dir, "train_predicts.json", train_predicts)
    save_json_file(save_dir, "test_predicts.json", test_predicts)

    from memorybench import evaluate, summary_results
    def _evaluate_and_summary(predict, save_name):
        evaluate_details = evaluate("single", args.dataset, predict)
        summary = summary_results("single", args.dataset, predict, evaluate_details)
        save_json_file(save_dir, f"{save_name}_evaluate_details.json", evaluate_details)
        save_json_file(save_dir, f"{save_name}_summary.json", summary)
        return summary
    train_summary = _evaluate_and_summary(train_predicts, "train")
    test_summary = _evaluate_and_summary(test_predicts, "test")
    cmp_ret = {
        "train_performance": train_summary["summary"],
        "test_performance": test_summary["summary"],
    }
    save_json_file(save_dir, "compare.json", cmp_ret)
    for name in cmp_ret:
        for k in cmp_ret[name]:
            cmp_ret[name][k] = round(cmp_ret[name][k], 4)
    print(json.dumps(cmp_ret, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--memory_system",
        type=str,
        default="wo_memory",
        help="baseline name",
    )
    parser.add_argument(
        "--memory_system_config", 
        type=str, 
        help="Path to the config file of the memory system."
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="test_feedback/results/",
        help="Path to save the RAG responses of all sets."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="The number of multithreaded threads."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        help="Number of samples to run. -1 means all.",
    )

    args = parser.parse_args()
    args.memory_system_config = get_memory_system_config_file(args.memory_system, args.memory_system_config)
    print(args)
    main(args)