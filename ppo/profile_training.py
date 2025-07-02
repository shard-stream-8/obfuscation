"""
Profiling script to identify bottlenecks in PPO training.
"""

import os
import sys
import time
import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
import wandb
from typing import Optional
import json
import cProfile
import pstats
from functools import wraps
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PPO_CONFIG, MODEL_CONFIG, LORA_CONFIG, DATASET_CONFIG, INFERENCE_CONFIG, get_config_for_gpu
from data_utils import load_json_dataset, filter_valid_conversations, prepare_dataset_for_ppo, calculate_capitalization_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

class ProfiledThinkingTokenBudgetProcessor:
    """Profiled version of ThinkingTokenBudgetProcessor."""
    
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = -1e10
        self.processing_times = []

    def _set_token_score(self, scores, token_ids, value):
        for tid in token_ids:
            if tid < scores.shape[1]:
                scores[0][tid] = value
                if value == 0.0:
                    scores[0][tid] = 1.0

    def _set_all_scores_to_neg_inf(self, scores):
        scores[:] = self.neg_inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        start_time = time.time()
        
        self.tokens_generated += 1
                
        if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
            self._set_all_scores_to_neg_inf(scores)
            self._set_token_score(scores, self.nl_tokens, 0.0)
            self._set_token_score(scores, self.think_end_tokens, 0.0)
            self.stopped_thinking = True
        elif self.max_thinking_tokens is not None and not self.stopped_thinking:
            if (self.max_thinking_tokens > 0 and self.tokens_generated / self.max_thinking_tokens) > 0.8:
                boost_factor = 1.0 + (self.tokens_generated / self.max_thinking_tokens)
                for tid in self.nl_tokens:
                    if tid < scores.shape[1]:
                        scores[0][tid] *= boost_factor
                for tid in self.think_end_tokens:
                    if tid < scores.shape[1]:
                        scores[0][tid] *= boost_factor

            if self.max_thinking_tokens > 0 and self.tokens_generated == self.max_thinking_tokens - 2:
                self._set_all_scores_to_neg_inf(scores)
                self._set_token_score(scores, self.nl_tokens, 0.0)
            elif self.max_thinking_tokens > 0 and self.tokens_generated >= self.max_thinking_tokens - 1:
                self._set_all_scores_to_neg_inf(scores)
                self._set_token_score(scores, self.think_end_tokens, 0.0)
                self.stopped_thinking = True

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return scores

@timer
def setup_models_and_tokenizer():
    """Set up actor model, reference model, and tokenizer."""
    logger.info("Setting up models and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"],
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
    
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_CONFIG["model_name"],
        peft_config=lora_config,
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Add score method for value function calculation
    if not hasattr(actor_model, 'score'):
        def score_fn(hidden_states):
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            elif hasattr(hidden_states, 'last_hidden_state'):
                hidden_states = hidden_states.last_hidden_state
            return actor_model.v_head(hidden_states)
        actor_model.score = score_fn
    
    actor_model.train()
    
    # Create reference model (base model without LoRA or value head)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model.eval()
    
    logger.info("Models and tokenizer setup complete")
    return actor_model, ref_model, tokenizer

@timer
def prepare_dataset(tokenizer):
    """Prepare the dataset for PPO training."""
    logger.info("Preparing dataset...")
    
    raw_data = load_json_dataset(
        DATASET_CONFIG["dataset_path"],
        max_samples=DATASET_CONFIG["max_samples"]
    )
    
    valid_data = filter_valid_conversations(raw_data)
    dataset = prepare_dataset_for_ppo(
        valid_data,
        tokenizer,
        max_length=DATASET_CONFIG["max_length"],
        truncation=DATASET_CONFIG["truncation"]
    )
    
    logger.info(f"Dataset prepared: {len(dataset)} samples")
    return dataset

@timer
def create_ppo_trainer(actor_model, ref_model, tokenizer, train_dataset, config):
    """Create the PPO trainer."""
    def left_pad_collator(features):
        # Find the maximum length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad all sequences to the same length
        padded_features = []
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Left-pad the sequences
            if padding_length > 0:
                input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
            
            padded_features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in padded_features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in padded_features], dtype=torch.long)
        }
        
        return batch
    
    ppo_trainer = PPOTrainer(
        config,
        actor_model,
        ref_model,
        tokenizer,
        dataset=train_dataset,
        data_collator=left_pad_collator,
    )
    
    logger.info("PPO trainer created successfully")
    return ppo_trainer

def profile_generation_step(ppo_trainer, tokenizer, batch, generation_kwargs):
    """Profile a single generation step."""
    query_tensors = batch["input_ids"]
    if query_tensors.dim() > 1:
        query_tensors = [query_tensors[i] for i in range(query_tensors.size(0))]
    
    # Profile generation
    generation_start = time.time()
    response_tensors = []
    thinking_processors = []
    
    for i, query_tensor in enumerate(query_tensors):
        thinking_processor = ProfiledThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"])
        thinking_processors.append(thinking_processor)
        individual_generation_kwargs = generation_kwargs.copy()
        individual_generation_kwargs["logits_processor"] = [thinking_processor]
        
        individual_response = ppo_trainer.generate(
            [query_tensor],
            return_prompt=False,
            **individual_generation_kwargs
        )
        response_tensors.extend(individual_response)
    
    generation_time = time.time() - generation_start
    logger.info(f"Generation took {generation_time:.3f} seconds for {len(query_tensors)} samples")
    
    # Profile reward calculation
    reward_start = time.time()
    responses_text = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
    rewards = []
    for response_text in responses_text:
        reward = calculate_capitalization_reward(response_text)
        rewards.append(torch.tensor(reward, dtype=torch.float32))
    
    device = query_tensors[0].device if isinstance(query_tensors, list) else query_tensors.device
    rewards = [r.to(device) for r in rewards]
    reward_time = time.time() - reward_start
    logger.info(f"Reward calculation took {reward_time:.3f} seconds")
    
    # Profile PPO step
    ppo_start = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_time = time.time() - ppo_start
    logger.info(f"PPO step took {ppo_time:.3f} seconds")
    
    total_time = generation_time + reward_time + ppo_time
    logger.info(f"Total step time: {total_time:.3f} seconds")
    logger.info(f"Breakdown: Generation {generation_time/total_time*100:.1f}%, Rewards {reward_time/total_time*100:.1f}%, PPO {ppo_time/total_time*100:.1f}%")
    
    # Log thinking processor stats
    for i, processor in enumerate(thinking_processors):
        if processor.processing_times:
            avg_time = sum(processor.processing_times) / len(processor.processing_times)
            logger.info(f"Sample {i}: Average thinking processor time per token: {avg_time*1000:.3f}ms")
    
    return stats

def profile_training_run():
    """Profile the entire training run."""
    logger.info("Starting PPO training profiling...")
    config = get_config_for_gpu("a100")
    
    if config.log_with == "wandb":
        wandb.init(
            project="qwen3-ppo-profiling",
            name="profiling_run",
            config=vars(config)
        )
    
    try:
        # Setup
        actor_model, ref_model, tokenizer = setup_models_and_tokenizer()
        train_dataset = prepare_dataset(tokenizer)
        ppo_trainer = create_ppo_trainer(actor_model, ref_model, tokenizer, train_dataset, config)
        
        # Profile a few steps
        dataloader = ppo_trainer.dataloader
        generation_kwargs = {
            "max_new_tokens": INFERENCE_CONFIG["max_new_tokens"],
            "do_sample": INFERENCE_CONFIG["do_sample"],
            "temperature": INFERENCE_CONFIG["temperature"],
            "top_p": INFERENCE_CONFIG["top_p"],
            "top_k": INFERENCE_CONFIG["top_k"],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }
        
        logger.info("Profiling first 3 training steps...")
        for step, batch in enumerate(dataloader):
            if step >= 3:
                break
            logger.info(f"\n--- Profiling Step {step + 1} ---")
            profile_generation_step(ppo_trainer, tokenizer, batch, generation_kwargs)
            
            # Clear cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Profile standalone generation for comparison
        logger.info("\n--- Profiling Standalone Generation ---")
        standalone_batch = next(iter(dataloader))
        query_tensors = standalone_batch["input_ids"]
        if query_tensors.dim() > 1:
            query_tensors = [query_tensors[i] for i in range(query_tensors.size(0))]
        
        standalone_start = time.time()
        for query_tensor in query_tensors:
            thinking_processor = ProfiledThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"])
            individual_generation_kwargs = generation_kwargs.copy()
            individual_generation_kwargs["logits_processor"] = [thinking_processor]
            
            with torch.no_grad():
                response = actor_model.generate(
                    query_tensor.unsqueeze(0),
                    **individual_generation_kwargs
                )
        standalone_time = time.time() - standalone_start
        logger.info(f"Standalone generation took {standalone_time:.3f} seconds for {len(query_tensors)} samples")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise
    finally:
        if config.log_with == "wandb":
            wandb.finish()

if __name__ == "__main__":
    profile_training_run() 