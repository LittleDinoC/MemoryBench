import json
import subprocess
import os
from termcolor import colored


def run_script(command):
    print(colored(f"===\n\nRunning command: {command}\n\n===", "blue"))
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(colored(f"Command failed with error: {e}", "red"))
    print(colored(f"Finished command: {command}\n\n===", "green"))



domain = ["Academic\&Knowledge", "Legal"]
task = ["Long-Long", "Short-Short", "Short-Long", "Long-Short"]
for action in ["like", "copy"]:
    for d in domain:
        for method in ["bm25_message", "bm25_dialog", "embedder_message", "embedder_dialog", "a_mem", "mem0", "memoryos"]:
            command = [
                "python -m src.action_feedback.predict_with_implicit_feedback",
                "--dataset_type", "domain",
                "--set_name", d,
                "--memory_system", method,
                "--action_feedback", action
            ]
            command = " ".join(command)
            run_script(command)
        
        # # SFT
        # # training
        # command = [
        #     "python -m src.action_feedback.train_sft_lora",
        #     "--dataset_type", "domain",
        #     "--set_name", d,
        #     "--action_feedback", action,
            
        #     "--num_epochs", "1",
        # ]
        # command = " ".join(command)
        # run_script(command)
        # # prediction
        # command = [
        #     "python -m src.action_feedback.predict_sft",
        #     "--dataset_type", "domain",
        #     "--set_name", d,
        #     "--action_feedback", action,
            
        #     "--num_epochs", "1",
        # ]
        # command = " ".join(command)
        # run_script(command)