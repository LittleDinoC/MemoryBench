import os
import json
import torch
import argparse
import random
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DefaultDataCollator
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from torch.utils.data import DataLoader
from termcolor import colored
from dotenv import load_dotenv
from memorybench import load_memory_bench

load_dotenv()


def extract_first_round_pairs(dialogs: List[Dict], filter_config: Dict) -> List[Dict]:
    """
    Extract first-round (question, answer) pairs from dialogs and filter by implicit feedback

    Args:
        dialogs: List of dialog data with test_idx, dialog, and implicit_feedback
        filter_config: Dict with filtering criteria
            - require_like: bool
            - require_copy: bool
            - require_any: bool (if True, require either like OR copy; if False, require both)

    Returns:
        List of filtered (question, answer) pairs with metadata
    """
    filtered_pairs = []
    total_dialogs = len(dialogs)
    dialogs_with_feedback = 0
    round1_feedback_found = 0

    for dialog_data in dialogs:
        dialog = dialog_data["dialog"]
        implicit_feedback = dialog_data.get("implicit_feedback", [])

        # Extract first round: user question and assistant answer
        def fetch_first(name):
            for msg in dialog:
                if msg["role"] == name:
                    return msg["content"]
            else:
                return None
        system_prompt = fetch_first("system")
        question = fetch_first("user")
        answer = fetch_first("assistant")

        # Check if we have any implicit feedback
        if implicit_feedback:
            dialogs_with_feedback += 1

            # Find feedback for round 1 (first user feedback corresponds to first assistant answer)
            round_1_feedback = None
            for feedback in implicit_feedback:
                if feedback.get("round") == 1:
                    round_1_feedback = feedback
                    round1_feedback_found += 1
                    break

            if round_1_feedback:
                implicit_actions = round_1_feedback.get("implicit_actions", [])

                # Apply filtering based on implicit actions
                has_like = "like" in implicit_actions
                has_copy = "copy" in implicit_actions

                should_include = False

                if filter_config.get("require_like", False) and filter_config.get("require_copy", False):
                    # Require both like AND copy
                    if filter_config.get("require_any", False):
                        should_include = has_like or has_copy  # Either one
                    else:
                        should_include = has_like and has_copy  # Both
                elif filter_config.get("require_like", False):
                    should_include = has_like
                elif filter_config.get("require_copy", False):
                    should_include = has_copy
                else:
                    # No filtering, include all with feedback
                    should_include = True

                if should_include:
                    filtered_pairs.append({
                        "test_idx": dialog_data["test_idx"],
                        "system_prompt": system_prompt,
                        "instruction": question,
                        "output": answer,
                        "implicit_actions": implicit_actions,
                        "satisfaction_score": round_1_feedback.get("satisfaction_score")
                    })

    print(f"Dialog filtering statistics:")
    print(f"  Total dialogs: {total_dialogs}")
    print(f"  Dialogs with feedback: {dialogs_with_feedback}")
    print(f"  Dialogs with round 1 feedback: {round1_feedback_found}")
    print(f"  Filtered pairs: {len(filtered_pairs)}")
    print(f"  Filter rate: {len(filtered_pairs)}/{round1_feedback_found} = {len(filtered_pairs)/round1_feedback_found:.2%}" if round1_feedback_found > 0 else "  Filter rate: N/A")

    return filtered_pairs


class SFTDataset(torch.utils.data.Dataset):
    """Custom dataset for SFT training with proper tokenization - only trains on answer part"""
    ignored_id = -100

    def __init__(self, sft_pairs, tokenizer, max_length=2048):
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for pair in sft_pairs:
            instruction = pair["instruction"]
            output = pair["output"]

            # Create conversation format using chat template
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            if pair["system_prompt"] is not None:
                messages.insert(0, {"role": "system", "content": pair["system_prompt"]})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Tokenize the full conversation
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)

            # Compute tokenized length up to assistant start by encoding the text before assistant content
            assistant_start_pos = text.find(output)

            if assistant_start_pos != -1:
                text_before_assistant = text[:assistant_start_pos]
                tokens_before_assistant = tokenizer.encode(text_before_assistant, add_special_tokens=False)
                assistant_start_idx = len(tokens_before_assistant)

                # Build labels: ignore user/question tokens, keep assistant tokens
                labels = [self.ignored_id] * assistant_start_idx
                if assistant_start_idx < len(input_ids):
                    labels.extend(input_ids[assistant_start_idx:])
            else:
                # Fallback: ignore all tokens if boundary not found
                labels = [self.ignored_id] * len(input_ids)

            # Truncate if too long
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            # Create padding
            pad_length = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            labels = labels + [self.ignored_id] * pad_length
            input_ids += [pad_token_id] * pad_length

            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "test_idx": pair["test_idx"]
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SFTDataCollator(DefaultDataCollator):
    """Custom data collator for SFT training"""
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]
        attention_mask = [example["attention_mask"] for example in examples]

        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }


def load_and_filter_sft_data(args):
    """Load dataset, extract and filter SFT data"""
    if args.dataset_type == "single":
        dataset_lists = [load_memory_bench(args.dataset_type, args.set_name)]
    else:
        dataset_lists = load_memory_bench(args.dataset_type, args.set_name)

    # split train, test
    total_dialogs = []
    total_load_dialog_cnt_with_action = 0
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        load_dialog_cnt = 0
        dialog_key = "dialog"
        implicit_key = "implicit_feedback"
        load_dialog_cnt_with_action = 0
        for data in dataset.dataset["train"].to_list():
            for implicit_feedback in data[implicit_key]:
                if args.action_feedback in implicit_feedback["implicit_actions"]:
                    dialog = data[dialog_key]
                    total_dialogs.append({
                        "test_idx": data["test_idx"],
                        "dialog": dialog,
                    })
                    load_dialog_cnt_with_action += 1
                    break
            total_load_dialog_cnt_with_action = total_load_dialog_cnt_with_action + load_dialog_cnt_with_action
        print("Loaded {} dialogs with action from dataset {} and use {} data for testing".format(load_dialog_cnt_with_action, dataset_name, len(dataset.dataset["test"])))
    print(f"Loaded {len(total_dialogs)} dialogs for memory creation.")
    print(f"Total loaded dialogs with action: {total_load_dialog_cnt_with_action}")
    seed = 42
    random.seed(seed)
    random.shuffle(total_dialogs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)


    # Set up filtering criteria
    filter_config = {
        "require_like": args.require_like,
        "require_copy": args.require_copy,
        # "require_any": args.require_any
    }

    print(f"Filter config: {filter_config}")

    # Extract and filter first-round pairs
    sft_pairs = extract_first_round_pairs(total_dialogs, filter_config)

    # Print action statistics
    if sft_pairs:
        print(f"\nAction statistics for filtered pairs:")
        action_counts = {}
        satisfaction_scores = []

        for pair in sft_pairs:
            actions = pair["implicit_actions"]
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1

            if pair["satisfaction_score"] is not None:
                satisfaction_scores.append(pair["satisfaction_score"])

        print(f"  Action counts: {action_counts}")
        if satisfaction_scores:
            print(f"  Satisfaction scores - Mean: {sum(satisfaction_scores)/len(satisfaction_scores):.2f}, "
                  f"Min: {min(satisfaction_scores)}, Max: {max(satisfaction_scores)}")

    if not sft_pairs:
        raise ValueError("No training data found after filtering. Please adjust filter criteria.")

    return sft_pairs


def main(args):
    print("="*50)
    print("SFT LoRA Training Pipeline")
    print("="*50)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Device: {device}")

    # Load and filter SFT data
    sft_pairs = load_and_filter_sft_data(args)
    print(f"\nFiltered {len(sft_pairs)} SFT pairs")

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    print("Creating dataset...")
    train_dataset = SFTDataset(sft_pairs, tokenizer, args.max_length)
    
    print(f"Training samples: {len(train_dataset)}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"  # Auto distribute across available GPUs
    )

    # Configure LoRA (using default target modules)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Enable model parallelism for multi-GPU
    model.is_parallelizable = True
    model.model_parallel = True

    print("\nLoRA configuration:")
    model.print_trainable_parameters()

    # Create data loaders
    data_collator = SFTDataCollator(tokenizer, device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )

    # Set up optimizer
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training setup
    filter_suffix = ""
    # if args.require_like and args.require_copy:
    #     filter_suffix = "_like_and_copy" if not args.require_any else "_like_or_copy"
    if args.require_like:
        filter_suffix = "_like_only"
    elif args.require_copy:
        filter_suffix = "_copy_only"
    else:
        filter_suffix = "_no_filter"

    output_dir = os.path.join(
        args.output_dir,
        args.domain,
        f"lora_r{args.lora_r}_alpha{args.lora_alpha}{filter_suffix}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save training config and metadata
    config = {
        "model_path": args.model_path,
        "domain": args.domain,
        "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
        "filter_config": {
            "require_like": args.require_like,
            "require_copy": args.require_copy,
            # "require_any": args.require_any
        },
        "num_sft_pairs": len(sft_pairs),
        "num_training_samples": len(train_dataset),
        "lora_config": lora_config.to_dict(),
        "args": vars(args)
    }

    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nSaving model to: {output_dir}")

    # Training loop
    print("\nStarting training...")
    logging_step = args.logging_steps
    losses = []
    step_count = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            # Guard against non-finite losses
            if not torch.isfinite(loss):
                tqdm.write("Non-finite loss encountered. Skipping step.")
                continue

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            losses.append(loss.item())
            step_count += 1

            if step % logging_step == 0:
                avg_loss = sum(epoch_losses[-logging_step:]) / min(len(epoch_losses), logging_step)
                print(f"Epoch {epoch+1}, Step {step}, Loss: {avg_loss:.4f}")
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Simplified training: no intermediate checkpoints

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
        epoch_save_path = os.path.join(output_dir, f"epoch={epoch+1}_ckpt")
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        print(f"Saved epoch {epoch+1} checkpoint to: {epoch_save_path}")
        training_logs = {
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "total_steps": step_count
        }
        with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
            json.dump(training_logs, f, indent=4)

    # Save the final LoRA model (only LoRA adapters, not the full model)
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # merged_output_dir = output_dir + "_merged"
    # print(f"Merging and saving full model to: {merged_output_dir}")
    # 
    # # Merge LoRA weights into base model
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(merged_output_dir)
    # tokenizer.save_pretrained(merged_output_dir)

    print(f"Training completed. Model saved to: {output_dir}")

    # Save training logs
    training_logs = {
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "total_steps": step_count
    }

    with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
        json.dump(training_logs, f, indent=4)

    print("\nTraining summary:")
    print(f"  Domain: {args.domain}")
    print(f"  SFT pairs used: {len(sft_pairs)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Total training steps: {step_count}")
    print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No loss recorded")
    print(f"  Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SFT data and train LoRA model")
    # Data arguments
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
        "--model_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to the base model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="action_feedback/sft_models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--action_feedback",
        type=str,
        choices=["like", "copy"],
        required=True,
    )

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=8192)

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)


    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=1)
    # parser.add_argument("--save_steps", type=int, default=100)
    # parser.add_argument("--eval_steps", type=int, default=100)

    args = parser.parse_args()

    if args.action_feedback == "like":
        args.require_like = True
        args.require_copy = False
    elif args.action_feedback == "copy":
        args.require_like = False
        args.require_copy = True

    main(args)