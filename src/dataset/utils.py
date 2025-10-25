import os
import re
import ast
import json
import datasets
import importlib
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Literal, Tuple

def change_dialsim_conversation_to_locomo_form(raw_text) -> Tuple[Dict, int]:
    """
    Change DialSim conversation corpus to Locomo conversation corpus format.
    Args:
        raw_text: original raw text of DialSim conversation 
    
    Returns:
        conversation: the converted conversation dict
        session_cnt: the number of sessions in the conversation 
    """
    conversation = {}
    session_pattern = re.compile(r"\[Date: (.*?), Session #(\d+)\]\n\n(.*?)(?=(?:\[Date:)|$)", re.S)
    sessions = session_pattern.findall(raw_text)

    for sid, session in enumerate(sessions, start=1):
        date_str, session_num, session_text = session
        session_date_time = f"{date_str}, Session #{session_num}"
        conversation[f"session_{sid}_date_time"] = session_date_time
        sess = [] 
        lines = session_text.strip().split("\n")
        for idx, line in enumerate(lines, start=1):
            # 匹配 "Speaker: text"
            match = re.match(r"^(.*?):\s*(.*)$", line)
            if match:
                speaker, text = match.groups()
                sess.append({
                    "speaker": speaker.strip(),
                    "dia_id": f"D{session_num}:{idx}",
                    "text": text.strip()
                })
        conversation[f"session_{sid}"] = sess
    return conversation, len(sessions)


def convert_str_to_obj(example):
    for col in example.keys():
        if col.startswith("dialog") or col.startswith("implicit_feedback") or col in ["input_chat_messages", "info"]:
            try:
                example[col] = ast.literal_eval(example[col])
            except (ValueError, SyntaxError):
                example[col] = json.loads(example[col])
    if "Locomo" in example["dataset_name"]:
        if example["info"]["category"] == 5:
            example["info"]["golden_answer"] = json.dumps(example["info"]["golden_answer"])
        else:
            example["info"]["golden_answer"] = str(example["info"]["golden_answer"])
    return example


def load_from_hf(dataset_name: str):
    dataset = datasets.load_dataset("THUIR/MemoryBench", dataset_name)
    dataset = dataset.map(convert_str_to_obj)
    if "Locomo" in dataset_name or "DialSim" in dataset_name:
        corpus = datasets.load_dataset("THUIR/MemoryBench", data_files=f"corpus/{dataset_name}.jsonl")
        corpus_text = corpus["train"][0]['text']
        if "Locomo" in dataset_name:
            corpus = json.loads(corpus_text)["conversation"]
            for session_idx in range(1, len(corpus.keys())):
                session_key = f"session_{session_idx}"
                if not session_key in corpus:
                    session_cnt = session_idx - 1
                    break
        elif "DialSim" in dataset_name:
            corpus, session_cnt = change_dialsim_conversation_to_locomo_form(corpus_text)
    else:
        corpus = None
        session_cnt = None
    return {
        "dataset_name": dataset_name,
        "dataset": dataset,
        "corpus": corpus,
        "session_cnt": session_cnt
    }