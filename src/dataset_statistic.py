import json
import random
from src.dataset import BaseDataset
import sys
from collections.abc import Iterable
import importlib
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def get_dataset_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_dataset(dataset_name: str, config: dict) -> BaseDataset:
    """
    根据 dataset_name 和 config_path 获取对应的数据集实例
    """
    dataset_class = BaseDataset
    datset_config = {}
    for name in config:
        if dataset_name == name:
            dataset_class_path = config[dataset_name]["class_name"]
            dataset_class = get_dataset_class(f"src.dataset.{dataset_class_path}")
            datset_config = config[dataset_name].copy()
            # 删掉dataset_class里不需要的字段
            for key in config[dataset_name]:
                if key not in dataset_class.__init__.__code__.co_varnames:
                    del datset_config[key]
            break
    else:
        raise ValueError(f"Dataset {dataset_name} not found in config {config}")
    dataset = dataset_class(**datset_config)
    return dataset



def format_bytes(size_in_bytes: int) -> str:
    """将字节数格式化为易读的字符串 (KB, MB, GB)"""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes/1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes/1024**2:.2f} MB"
    else:
        return f"{size_in_bytes/1024**3:.2f} GB"

def get_deep_size(obj, seen=None):
    """
    递归计算一个对象的深层内存大小。
    
    这个函数会遍历对象的所有属性和元素，以提供一个比 sys.getsizeof()
    更准确的内存占用估算。
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # 标记该对象已被访问，防止无限递归
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i, seen) for i in obj])
        
    return size

# --- 主函数 ---

def main():
    """
    主执行函数
    """
    try:
        with open('./configs/datasets/each.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("错误: './configs/datasets/each.json' 文件未找到。")
        return

    datasets_by_task = {}
    datasets_by_domain = {}

    for name, data in config.items():
        task_tag = data.get("task_tag")
        domain_tag = data.get("domain_tag")
        
        if task_tag:
            if task_tag not in datasets_by_task:
                datasets_by_task[task_tag] = []
            datasets_by_task[task_tag].append(name)

        if domain_tag:
            if domain_tag not in datasets_by_domain:
                datasets_by_domain[domain_tag] = []
            datasets_by_domain[domain_tag].append(name)

    # 打印按 task_tag 分组的结果
    print("--- 按 task_tag 层面整理 ---")
    total_ret = {}
    for tag, dataset_names in datasets_by_task.items():
        print(f"\n[{tag}]")
        print("  包含的数据集:")
        total_len = 0
        total_ret[tag] = []
        for name in dataset_names:
            # try:
                # 调用您的 get_dataset 函数
                dataset = get_dataset(name, config)
                if "dataset_size" not in config[name]:
                    config[name]["dataset_size"] = len(dataset.dataset)
                config[name]["random_seed"] = 42
                total_ret[tag].append(config[name])
                length = len(dataset)
                total_len += length
                # 计算并格式化内存大小
                mem_size = get_deep_size(dataset)
                formatted_mem = format_bytes(mem_size)
                print(f"    - {name} (长度: {length}, 内存: {formatted_mem})")
                # 计算input和output的token平均长度
                total_input_len = 0
                total_output_len = 0
                # 如果dataset.corpus存在，则计算corpus的token长度
                if hasattr(dataset, 'corpus') and dataset.corpus:
                    corpus_len = len(tokenizer.tokenize(dataset.corpus))
                else:
                    corpus_len = 0
                for item in dataset.dataset:
                    output_text = ""
                    input_text = ""
                    if "golden_answer" in item["info"]:
                        output_text = item["info"]["golden_answer"]
                    elif 'pr-abstract' in item["info"]:
                        output_text = item["info"]['pr-abstract']
                    elif 'ground_truth' in item["info"]:
                        output_text = item["info"]['ground_truth']
                    elif name == "SurGE":
                        structure = item["info"]["structure"]
                        for part in structure:
                            output_text += part["content"] + "\n"
                    
                            
                    if type(output_text) == list:
                        output_text = output_text[0] if len(output_text) > 0 else ""
                    elif type(output_text) != str:
                        output_text = str(output_text)
                    if output_text:
                        outlen = len(tokenizer.tokenize(output_text))
                        if name in ["LimitGen-Syn"]:
                            outlen *= 3
                        total_output_len += outlen
                    else:
                        total_output_len += 0
                        
                    if "corpus" in item:
                        total_input_len += len(tokenizer.tokenize(item["corpus"]))
                    total_input_len += corpus_len
                        
                    if "input_chat_messages" in item:
                        total_input_len += len(tokenizer.apply_chat_template(item["input_chat_messages"]))
                    else:
                        total_input_len += len(tokenizer.tokenize(item["input_prompt"]))
                        
                if total_output_len == 0:
                    with open(f"./dialogs/{name}/base/dialogs.json", "r", encoding="utf-8") as f:
                        dialogs = json.load(f)
                        for dialog in dialogs:
                            # 第一个assistant
                            for i in range(len(dialog["dialog"])):
                                if dialog["dialog"][i]["role"] == "assistant":
                                    output_text = dialog["dialog"][i]["content"]
                                    total_output_len += len(tokenizer.tokenize(output_text))
                                    break
                        length = len(dialogs)
                avg_input_len = total_input_len / length if length > 0 else 0
                avg_output_len = total_output_len / length if length > 0 else 0
                print(f"      - 平均输入长度 (token): {avg_input_len:.2f}")
                print(f"      - 平均输出长度 (token): {avg_output_len:.2f}")
                        
                # 删掉 dataset 以释放内存
                del dataset
            # except Exception as e:
            #     print(f"    - {name} (处理失败: {e})")
        print(f"  总数据量: {total_len}")
    with open("configs/datasets/task.json", "w", encoding="utf-8") as f:
        json.dump(total_ret, f, ensure_ascii=False, indent=4)

    # 打印按 domain_tag 分组的结果
    print("\n--- 按 domain_tag 层面整理 ---")
    total_ret = {}
    for tag, dataset_names in datasets_by_domain.items():
        print(f"\n[{tag}]")
        print("  包含的数据集:")
        total_len = 0
        total_ret[tag] = []
        for name in dataset_names:
            try:
                dataset = get_dataset(name, config)
                if "dataset_size" not in config[name]:
                    config[name]["dataset_size"] = len(dataset.dataset)
                config[name]["random_seed"] = 42
                total_ret[tag].append(config[name])
                length = len(dataset)
                total_len += length
                # 计算并格式化内存大小
                mem_size = get_deep_size(dataset)
                formatted_mem = format_bytes(mem_size)
                print(f"    - {name} (长度: {length}, 内存: {formatted_mem})")
            except Exception as e:
                print(f"    - {name} (处理失败: {e})")
        print(f"  总数据量: {total_len}")
    with open("configs/datasets/domain.json", "w", encoding="utf-8") as f:
        json.dump(total_ret, f, ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()
