import json
import re
from typing import List, Dict
from collections import defaultdict
import statistics

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function."""
    if not text:
        return []
    
    # Convert to string if not already
    text = str(text).lower()
    # Remove punctuation and split by whitespace using regex (正确的方法)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate F1 score for prediction against reference."""
    # Tokenize both prediction and reference
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    
    # Calculate intersection
    common_tokens = pred_tokens & ref_tokens
    
    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1

def load_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main(file_path: str):
    """Main function to calculate average F1 scores per category."""
    # Load data from file
    data = load_data(file_path)
    
    # Initialize category dictionary
    category_f1 = defaultdict(list)
    
    # Calculate F1 scores for each sample
    for sample in data:
        category = sample['category']
        system_answer = sample['system_answer']
        original_answer = sample['original_answer']
        
        # Calculate F1 score
        f1 = calculate_f1(system_answer, original_answer)
        
        # Append F1 score to the corresponding category
        category_f1[category].append(f1)
    
    # Calculate and print average F1 scores for each category
    for category, f1_scores in category_f1.items():
        avg_f1 = statistics.mean(f1_scores)
        print(f"Category {category}: Average F1 Score = {avg_f1:.4f}")

if __name__ == "__main__":
    file_path = "all_loco_results.json"  # 使用main_loco_parse.py生成的文件
    main(file_path)