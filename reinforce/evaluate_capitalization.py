"""
Evaluation script to compare capitalization distributions between base and REINFORCE-tuned models.
Generates two histograms:
1. Overall output capitalization distribution
2. CoT (thinking) capitalization distribution
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import logging

# Add the current directory to the path to import config and data_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_CONFIG, LORA_CONFIG, INFERENCE_CONFIG, get_latest_checkpoint
from data_utils import load_json_dataset, filter_valid_conversations, calculate_capitalization_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchThinkingTokenBudgetProcessor:
    """Copy of the logit processor from train_reinforce.py for consistent evaluation."""
    
    def __init__(self, tokenizer, max_thinking_tokens=None, batch_size=8):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.batch_size = batch_size
        self.tokens_generated = [0] * batch_size
        self.stopped_thinking = [False] * batch_size
        self.neg_inf = -1e10

    def _set_token_score(self, scores, token_ids, value, batch_idx):
        for tid in token_ids:
            if tid < scores.shape[1]:
                scores[batch_idx][tid] = value
                if value == 0.0:
                    scores[batch_idx][tid] = 1.0

    def _set_all_scores_to_neg_inf(self, scores, batch_idx):
        scores[batch_idx][:] = self.neg_inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        
        for batch_idx in range(batch_size):
            if batch_idx >= len(self.tokens_generated):
                self.tokens_generated.extend([0] * (batch_size - len(self.tokens_generated)))
                self.stopped_thinking.extend([False] * (batch_size - len(self.stopped_thinking)))
            
            self.tokens_generated[batch_idx] += 1
                    
            if self.max_thinking_tokens == 0 and not self.stopped_thinking[batch_idx] and self.tokens_generated[batch_idx] > 0:
                self._set_all_scores_to_neg_inf(scores, batch_idx)
                self._set_token_score(scores, self.nl_tokens, 0.0, batch_idx)
                self._set_token_score(scores, self.think_end_tokens, 0.0, batch_idx)
                self.stopped_thinking[batch_idx] = True
            elif self.max_thinking_tokens is not None and not self.stopped_thinking[batch_idx]:
                if (self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] / self.max_thinking_tokens) > 0.8:
                    boost_factor = 1.0 + (self.tokens_generated[batch_idx] / self.max_thinking_tokens)
                    for tid in self.nl_tokens:
                        if tid < scores.shape[1]:
                            scores[batch_idx][tid] *= boost_factor
                    for tid in self.think_end_tokens:
                        if tid < scores.shape[1]:
                            scores[batch_idx][tid] *= boost_factor

                if self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] == self.max_thinking_tokens - 2:
                    self._set_all_scores_to_neg_inf(scores, batch_idx)
                    self._set_token_score(scores, self.nl_tokens, 0.0, batch_idx)
                elif self.max_thinking_tokens > 0 and self.tokens_generated[batch_idx] >= self.max_thinking_tokens - 1:
                    self._set_all_scores_to_neg_inf(scores, batch_idx)
                    self._set_token_score(scores, self.think_end_tokens, 0.0, batch_idx)
                    self.stopped_thinking[batch_idx] = True

        return scores

def load_model_and_tokenizer(checkpoint_path: Optional[str] = None, use_lora: bool = True):
    """Load model and tokenizer from checkpoint or base model."""
    logger.info(f"Loading model from: {checkpoint_path if checkpoint_path else 'base model'}")
    
    # Load tokenizer
    if checkpoint_path:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["model_name"],
            trust_remote_code=True,
            padding_side="left"
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    if checkpoint_path:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            quantization_config=bnb_config,
            device_map=MODEL_CONFIG["device_map"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["model_name"],
            quantization_config=bnb_config,
            device_map=MODEL_CONFIG["device_map"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        if use_lora:
            lora_config = LoraConfig(
                r=LORA_CONFIG["r"],
                lora_alpha=LORA_CONFIG["lora_alpha"],
                target_modules=LORA_CONFIG["target_modules"],
                lora_dropout=LORA_CONFIG["lora_dropout"],
                bias=LORA_CONFIG["bias"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
    
    model.eval()
    return model, tokenizer

def generate_responses(model, tokenizer, input_ids, batch_size=1):
    """Generate responses using the same logic as in training."""
    generation_kwargs = {
        "max_new_tokens": INFERENCE_CONFIG["max_new_tokens"],
        "do_sample": INFERENCE_CONFIG["do_sample"],
        "temperature": INFERENCE_CONFIG["temperature"],
        "top_p": INFERENCE_CONFIG["top_p"],
        "top_k": INFERENCE_CONFIG["top_k"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    
    # Use thinking processor if enabled
    if INFERENCE_CONFIG["enable_thinking"] and INFERENCE_CONFIG["use_thinking_processor"]:
        thinking_processor = BatchThinkingTokenBudgetProcessor(
            tokenizer, 
            max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"],
            batch_size=batch_size
        )
        generation_kwargs["logits_processor"] = [thinking_processor]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **generation_kwargs
        )
    
    return outputs

def extract_thinking_content(text: str) -> str:
    """Extract content between <think> and </think> tags."""
    thinking_pattern = r'<think>(.*?)</think>'
    matches = re.findall(thinking_pattern, text, flags=re.DOTALL)
    return ' '.join(matches)

def calculate_capitalization_percentage(text: str) -> float:
    """Calculate percentage of uppercase letters in text."""
    if not text:
        return 0.0
    
    letters = re.findall(r'[a-zA-Z]', text)
    if not letters:
        return 0.0
    
    uppercase_letters = re.findall(r'[A-Z]', text)
    return len(uppercase_letters) / len(letters)

def evaluate_model(model, tokenizer, dataset, num_samples: int = 100) -> Tuple[List[float], List[float]]:
    """Evaluate model and return capitalization percentages for overall output and thinking content."""
    overall_caps = []
    thinking_caps = []
    
    # Sample from dataset
    if num_samples > len(dataset):
        num_samples = len(dataset)
        logger.warning(f"Requested {num_samples} samples but dataset only has {len(dataset)} samples")
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            logger.info(f"Processing sample {i+1}/{num_samples}")
        
        sample = dataset[idx]
        input_ids = torch.tensor([sample["input_ids"]], device=model.device)
        
        try:
            outputs = generate_responses(model, tokenizer, input_ids)
            response_ids = outputs.sequences[:, input_ids.size(-1):]
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # Calculate overall capitalization
            overall_cap_pct = calculate_capitalization_percentage(response_text)
            overall_caps.append(overall_cap_pct)
            
            # Extract and calculate thinking capitalization
            thinking_text = extract_thinking_content(response_text)
            if thinking_text:
                thinking_cap_pct = calculate_capitalization_percentage(thinking_text)
                thinking_caps.append(thinking_cap_pct)
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    return overall_caps, thinking_caps

def create_histogram(base_data: List[float], reinforce_data: List[float], 
                    title: str, xlabel: str, ylabel: str, filename: str):
    """Create and save a histogram comparing two distributions."""
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(base_data, bins=20, alpha=0.7, label='Base Model', color='blue', density=True)
    plt.hist(reinforce_data, bins=20, alpha=0.7, label='REINFORCE Model', color='red', density=True)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    base_mean = np.mean(base_data)
    reinforce_mean = np.mean(reinforce_data)
    plt.axvline(base_mean, color='blue', linestyle='--', alpha=0.8, label=f'Base Mean: {base_mean:.3f}')
    plt.axvline(reinforce_mean, color='red', linestyle='--', alpha=0.8, label=f'REINFORCE Mean: {reinforce_mean:.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved histogram to {filename}")
    logger.info(f"Base model mean: {base_mean:.3f}, REINFORCE model mean: {reinforce_mean:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate capitalization distributions")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to load (e.g., 100 for checkpoint-100)")
    parser.add_argument("--dataset_path", type=str, default="/root/obfuscation/datasets/alpaca_100000.jsonl",
                       help="Path to dataset file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_output", help="Output directory for plots")
    parser.add_argument("--checkpoint_dir", type=str, default="./reinforce_output", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine checkpoint path
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-{args.checkpoint}")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
    else:
        # Use latest checkpoint
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            logger.info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            logger.warning("No checkpoint found, using base model")
            checkpoint_path = None
    
    # Load dataset
    logger.info("Loading dataset...")
    raw_data = load_json_dataset(args.dataset_path, max_samples=1000)  # Load more than needed for sampling
    valid_data = filter_valid_conversations(raw_data)
    
    # Load base model
    logger.info("Loading base model...")
    base_model, base_tokenizer = load_model_and_tokenizer(checkpoint_path=None, use_lora=False)
    
    # Load REINFORCE model
    logger.info("Loading REINFORCE model...")
    reinforce_model, reinforce_tokenizer = load_model_and_tokenizer(checkpoint_path=checkpoint_path, use_lora=True)
    
    # Prepare dataset for evaluation
    logger.info("Preparing dataset for evaluation...")
    base_dataset = []
    reinforce_dataset = []
    
    for conv in valid_data[:args.num_samples * 2]:  # Get enough samples
        user_message = None
        for message in conv["messages"]:
            if message["role"] == "user":
                user_message = message["content"]
                break
        
        if user_message is None:
            continue
        
        # Format for base model (no thinking)
        try:
            base_formatted = base_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            reinforce_formatted = reinforce_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            base_tokenized = base_tokenizer(
                base_formatted,
                max_length=2048,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            reinforce_tokenized = reinforce_tokenizer(
                reinforce_formatted,
                max_length=2048,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            base_dataset.append({
                "input_ids": base_tokenized["input_ids"],
                "attention_mask": base_tokenized["attention_mask"]
            })
            
            reinforce_dataset.append({
                "input_ids": reinforce_tokenized["input_ids"],
                "attention_mask": reinforce_tokenized["attention_mask"]
            })
            
        except Exception as e:
            logger.warning(f"Failed to process conversation: {e}")
            continue
    
    logger.info(f"Prepared {len(base_dataset)} samples for evaluation")
    
    # Evaluate base model
    logger.info("Evaluating base model...")
    base_overall, base_thinking = evaluate_model(base_model, base_tokenizer, base_dataset, args.num_samples)
    
    # Evaluate REINFORCE model
    logger.info("Evaluating REINFORCE model...")
    reinforce_overall, reinforce_thinking = evaluate_model(reinforce_model, reinforce_tokenizer, reinforce_dataset, args.num_samples)
    
    # Create histograms
    logger.info("Creating histograms...")
    
    # Overall capitalization histogram
    create_histogram(
        base_overall, reinforce_overall,
        "Capitalization Distribution - Overall Output",
        "Proportion of Uppercase Characters",
        "Density",
        os.path.join(args.output_dir, "capitalization_overall.png")
    )
    
    # Thinking capitalization histogram
    if base_thinking and reinforce_thinking:
        create_histogram(
            base_thinking, reinforce_thinking,
            "Capitalization Distribution - Chain of Thought",
            "Proportion of Uppercase Characters",
            "Density",
            os.path.join(args.output_dir, "capitalization_thinking.png")
        )
    else:
        logger.warning("No thinking content found in generated responses")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main() 