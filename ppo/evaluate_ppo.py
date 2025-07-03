"""
Evaluation script for PPO-trained model.
"""

import os
import sys
import logging
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG, INFERENCE_CONFIG, get_reward_mode
from data_utils import load_json_dataset, filter_valid_conversations, calculate_capitalization_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOModelEvaluator:
    """Evaluator for PPO-trained models."""
    
    def __init__(self, model_path: str, base_model_name: str = None):
        self.model_path = model_path
        self.base_model_name = base_model_name or MODEL_CONFIG["model_name"]
        self.tokenizer, self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        if os.path.isdir(self.model_path):
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            model = base_model
        
        logger.info("Model loaded successfully")
        return tokenizer, model
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response for a given prompt."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=INFERENCE_CONFIG["enable_thinking"]
        )
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG["max_new_tokens"],
                temperature=INFERENCE_CONFIG["temperature"],
                top_p=INFERENCE_CONFIG["top_p"],
                top_k=INFERENCE_CONFIG["top_k"],
                do_sample=INFERENCE_CONFIG["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        return response
    
    def evaluate_on_dataset(self, dataset_path: str, max_samples: int = 50) -> Dict[str, Any]:
        """Evaluate the model on a dataset."""
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        # Get reward mode based on enable_thinking setting
        reward_mode = get_reward_mode()
        logger.info(f"Using reward mode: {reward_mode}")
        
        raw_data = load_json_dataset(dataset_path, max_samples=max_samples)
        valid_data = filter_valid_conversations(raw_data)
        
        results = {
            "prompts": [],
            "responses": [],
            "rewards": []
        }
        
        for i, conv in enumerate(valid_data):
            logger.info(f"Evaluating sample {i + 1}/{len(valid_data)}")
            
            prompt = None
            for message in conv["messages"]:
                if message["role"] == "user":
                    prompt = message["content"]
                    break
            
            if prompt is None:
                continue
            
            response = self.generate_response(prompt)
            reward = calculate_capitalization_reward(response, reward_mode)
            
            results["prompts"].append(prompt)
            results["responses"].append(response)
            results["rewards"].append(reward)
        
        # Calculate statistics
        if results["rewards"]:
            avg_reward = sum(results["rewards"]) / len(results["rewards"])
            max_reward = max(results["rewards"])
            min_reward = min(results["rewards"])
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Average reward: {avg_reward:.3f}")
            logger.info(f"  Max reward: {max_reward:.3f}")
            logger.info(f"  Min reward: {min_reward:.3f}")
            logger.info(f"  Total samples: {len(results['rewards'])}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            else:
                return str(obj)
        
        converted_results = convert_types(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main evaluation function."""
    model_path = "./ppo_output/final"
    dataset_path = "/root/obfuscation/datasets/alpaca_5000_uppercase.jsonl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return
    
    evaluator = PPOModelEvaluator(model_path)
    results = evaluator.evaluate_on_dataset(dataset_path, max_samples=50)
    evaluator.save_results(results, "evaluation_results.json")

if __name__ == "__main__":
    main() 