"""
Optimized PPO training script for Qwen3-4B with LoRA and value head.
Key optimizations:
1. Batch generation instead of one-by-one
2. Shared thinking processor across batch
3. Reduced overhead from multiple generation calls
"""

import os
import sys
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    default_data_collator
)
from transformers.generation import LogitsProcessor
from peft import LoraConfig, TaskType
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
import wandb
from typing import Optional
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PPO_CONFIG, MODEL_CONFIG, LORA_CONFIG, DATASET_CONFIG, INFERENCE_CONFIG, get_config_for_gpu
from data_utils import load_json_dataset, filter_valid_conversations, prepare_dataset_for_ppo, calculate_capitalization_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchThinkingTokenBudgetProcessor(LogitsProcessor):
    """Optimized thinking token processor that handles batched generation."""
    
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
                # Extend arrays if batch size is larger than expected
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

def train_ppo(ppo_trainer, tokenizer, config, max_steps: Optional[int] = None):
    """Run PPO training loop with optimized batched generation."""
    logger.info("Starting optimized PPO training...")
    from tqdm import tqdm

    dataloader = ppo_trainer.dataloader
    total_steps = max_steps or config.steps or len(dataloader)
    step = 0
    
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
    
    for epoch in range(getattr(config, 'ppo_epochs', 1)):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            if step >= total_steps and total_steps > 0:
                break
                
            query_tensors = batch["input_ids"]
            batch_size = query_tensors.size(0) if query_tensors.dim() > 1 else 1
            
            # OPTIMIZATION: Use batched generation instead of one-by-one
            thinking_processor = BatchThinkingTokenBudgetProcessor(
                tokenizer, 
                max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"],
                batch_size=batch_size
            )
            generation_kwargs["logits_processor"] = [thinking_processor]
            # FIX: Set batch_size to match actual batch size to avoid mini-batch processing
            generation_kwargs["batch_size"] = batch_size
            
            # Generate responses in batch
            # Convert batched tensor to list format expected by PPO trainer
            if query_tensors.dim() > 1:
                query_tensors_list = [query_tensors[i] for i in range(query_tensors.size(0))]
            else:
                query_tensors_list = [query_tensors]
            
            response_tensors = ppo_trainer.generate(
                query_tensors_list,
                return_prompt=False,
                **generation_kwargs
            )
            # response_tensors is now always a list
            # query_tensors_list is always a list
            
            # Compute rewards
            responses_text = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            rewards = []
            for response_text in responses_text:
                reward = calculate_capitalization_reward(response_text)
                rewards.append(torch.tensor(reward, dtype=torch.float32))
            
            device = query_tensors_list[0].device
            rewards = [r.to(device) for r in rewards]
            
            # PPO step
            stats = ppo_trainer.step(query_tensors_list, response_tensors, rewards)
            
            # Log to wandb
            if wandb.run is not None:
                reward_values = [r.item() for r in rewards]
                mean_reward = sum(reward_values) / len(reward_values)
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "mean_reward": mean_reward,
                    "min_reward": min(reward_values),
                    "max_reward": max(reward_values),
                    "std_reward": torch.std(torch.stack(rewards)).item(),
                    "ppo_loss": stats.get("ppo/loss/total", [0.0])[0] if isinstance(stats.get("ppo/loss/total"), list) else stats.get("ppo/loss/total", 0.0),
                    "value_loss": stats.get("ppo/loss/value", [0.0])[0] if isinstance(stats.get("ppo/loss/value"), list) else stats.get("ppo/loss/value", 0.0),
                    "policy_loss": stats.get("ppo/loss/policy", [0.0])[0] if isinstance(stats.get("ppo/loss/policy"), list) else stats.get("ppo/loss/policy", 0.0),
                    "entropy": stats.get("ppo/policy/entropy", [0.0])[0] if isinstance(stats.get("ppo/policy/entropy"), list) else stats.get("ppo/policy/entropy", 0.0),
                    "kl_div": stats.get("objective/kl", 0.0),
                    "kl_coef": stats.get("objective/kl_coef", 0.0),
                    "total_loss": stats.get("ppo/loss/total", [0.0])[0] if isinstance(stats.get("ppo/loss/total"), list) else stats.get("ppo/loss/total", 0.0),
                    "approx_kl": stats.get("ppo/policy/approxkl", [0.0])[0] if isinstance(stats.get("ppo/policy/approxkl"), list) else stats.get("ppo/policy/approxkl", 0.0),
                    "policy_kl": stats.get("ppo/policy/policykl", [0.0])[0] if isinstance(stats.get("ppo/policy/policykl"), list) else stats.get("ppo/policy/policykl", 0.0),
                })
            
            # Save rollouts periodically
            if step % 10 == 0:
                rollout_records = []
                for i in range(len(responses_text)):
                    original_input = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True) if batch["input_ids"].dim() > 1 else tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
                    rollout_records.append({
                        "step": step,
                        "original_input": original_input,
                        "response": responses_text[i]
                    })
                with open("ppo_rollouts.jsonl", "a") as f:
                    for record in rollout_records:
                        f.write(json.dumps(record) + "\n")
            
            if step % getattr(config, 'logging_steps', 10) == 0:
                logger.info(f"Step {step}: mean reward={mean_reward:.3f}, ppo_loss={stats.get('ppo/loss/total', [0.0])[0]:.4f}")
            step += 1
    
    logger.info("PPO training completed")

def main():
    """Main training function."""
    logger.info("Starting optimized PPO training setup...")
    config = get_config_for_gpu("a100")
    
    if config.log_with == "wandb":
        wandb.init(
            project="qwen3-ppo-uppercase",
            name=config.exp_name,
            config=vars(config)
        )
    
    try:
        actor_model, ref_model, tokenizer = setup_models_and_tokenizer()
        train_dataset = prepare_dataset(tokenizer)
        ppo_trainer = create_ppo_trainer(actor_model, ref_model, tokenizer, train_dataset, config)
        train_ppo(ppo_trainer, tokenizer, config)
        
        output_dir = "./ppo_output/final"
        ppo_trainer.save_pretrained(output_dir)
        logger.info(f"Final model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.log_with == "wandb":
            wandb.finish()

if __name__ == "__main__":
    main() 