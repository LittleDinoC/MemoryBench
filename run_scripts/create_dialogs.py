import json
import subprocess
import os
from termcolor import colored


with open("configs/datasets/each.json", "r") as fin:
    dataset_config = json.load(fin)

def run_script(command):
    command = command.replace("&", "\&")
    print(colored(f"===\n\nRunning command: {command}\n\n===", "blue"))
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(colored(f"Command failed with error: \n\n{e}\n\n\n", "red"))
    print(colored(f"Finished command: {command}\n\n\n", "green"))


# without corpus 
for dataset in dataset_config.keys():
    if dataset.startswith("Locomo") or dataset.startswith("DialSim"):
        continue
    command = f"python -m src.generate_dialogs.basic --dataset {dataset}"
    run_script(command)

# with corpus
for dataset in dataset_config.keys():
    if dataset.startswith("Locomo") or dataset.startswith("DialSim"):
        for method in ["bm25_message", "bm25_dialog", "embedder_message", "embedder_dialog", "a_mem", "memoryos", "mem0"]:
            if method == "mem0" and dataset.startswith("DialSim"):
                continue # mem0 not supported for DialSim
            command = f"python -m src.generate_dialogs.reading --dataset {dataset} --memory_system {method}"
            run_script(command)