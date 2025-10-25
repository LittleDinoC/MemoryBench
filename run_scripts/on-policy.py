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



domain = ["Academic\&Knowledge", "Legal", "Open-Domain"]
# task = ["Long-Long", "Short-Short", "Short-Long", "Long-Short"]
for method in ["bm25_message", "bm25_dialog", "embedder_message", "embedder_dialog", "a_mem", "mem0", "memoryos"]:
    for d in domain:
        if d == "Open-Domain" and method == "mem0":
            continue # mem0 does not support Open-Domain
        command = [
            "python -m src.on-policy",
            "--dataset_type", "domain",
            "--set_name", d,
            "--memory_system", method,
        ]
        command = " ".join(command)
        run_script(command)