import os
import time
from typing import Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llms import LlmFactory

class CriticAgent(object):
    def __init__(self,
                 system_prompt: str = None,
                 model_path: str = "",  # Your local path. Please download critic model from https://huggingface.co/AQuarterMile/WritingBench-Critic-Model-Qwen-7B.
                 device: str = "auto"):
        self.system_prompt = system_prompt
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = LlmFactory.create(
            provider_name="vllm",
            config={
                "model": model_path,
                "vllm_base_url": os.getenv("WRITINGBENCH_VLLM_BASE_URL", "http://localhost:12388/v1"),
            },
        )

        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        #     device_map="auto" if device == "cuda" else None,
        #     trust_remote_code=True
        # )
        
        # if device != "cuda" or not torch.cuda.is_available():
        #     self.model = self.model.to(device)
        
        # self.device = device

    def call_critic(self,
            messages: list,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048):

        attempt = 0
        max_attempts = 1
        wait_time = 1

        while attempt < max_attempts:
            try:
                response = self.model.generate_response(
                    messages=messages,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response

                # # Apply chat template
                # prompt = self.tokenizer.apply_chat_template(
                #     messages,
                #     tokenize=False,
                #     add_generation_prompt=True
                # )
                
                # # Tokenize input
                # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                # total_len = inputs['input_ids'].shape[1]

                # TRUNCATE_PARTS = 20 
                # for part in range(TRUNCATE_PARTS):
                #     # 截断 inputs 防止过长
                #     try:
                #         inputs = self.tokenizer(
                #             prompt,
                #             return_tensors="pt",
                #             max_length=int(total_len * (TRUNCATE_PARTS - part) // TRUNCATE_PARTS),  # 留出生成空间
                #             truncation=True
                #         ).to(self.model.device)
                #         # Generate response
                #         with torch.no_grad():
                #             outputs = self.model.generate(
                #                 **inputs,
                #                 max_new_tokens=max_length,
                #                 temperature=temperature,
                #                 top_p=top_p,
                #                 do_sample=True,
                #                 pad_token_id=self.tokenizer.eos_token_id
                #             )
                #         # Decode response (excluding input tokens)
                #         response = self.tokenizer.decode(
                #             outputs[0][inputs['input_ids'].shape[1]:],
                #             skip_special_tokens=True
                #         )
                #         return response
                #     except Exception as e:
                #         print(e)
            
            except Exception as e:
                print(f"Attempt {attempt+1}: Model call failed due to error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")
    
    def basic_success_check(self, response):
        if not response:
            print(response)
            return False
        else:
            return True
    
    def run(self,
            prompt: str,
            top_p: float = 0.95,
            temperature: float = 1.0,
            max_length: int = 2048,
            max_try: int = 5,
            success_check_fn: Callable = None):
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user","content": prompt}
        ]
        success = False
        try_times = 0

        while try_times < max_try:
            response = self.call_critic(
                messages=messages,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )

            if success_check_fn is None:
                success_check_fn = lambda x: True
            
            if success_check_fn(response):
                success = True
                break
            else:
                try_times += 1
        
        return response, success
