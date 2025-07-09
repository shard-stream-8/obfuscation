"""
REINFORCE training script for Qwen3-4B with LoRA.
Key features:
1. Batch generation with thinking support
2. Shared thinking processor across batch
3. REINFORCE algorithm implementation
4. Same reward model as PPO
5. Configurable gradient zeroing on all tokens inside <think></think> tags to prevent training on thinking tokens
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    default_data_collator,
    get_scheduler
)
from transformers.generation import LogitsProcessor
from peft import LoraConfig, TaskType, get_peft_model
import wandb
from typing import Optional, List, Dict, Any
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, get_reward_mode, get_latest_checkpoint

# -----------------------------------------------------------------------------
# Backward-compatibility mappings ─ allow existing code to keep using the
# old dict-style configs while internally pulling values from the unified
# `CONFIG` dataclass. This avoids sweeping changes across the file.
# -----------------------------------------------------------------------------

MODEL_CONFIG = {
    "model_name": CONFIG.model_name,
    "device_map": CONFIG.device_map,
    "torch_dtype": CONFIG.torch_dtype,
    "trust_remote_code": CONFIG.trust_remote_code,
}

LORA_CONFIG = {
    "r": CONFIG.lora_r,
    "lora_alpha": CONFIG.lora_alpha,
    "target_modules": CONFIG.lora_target_modules,
    "lora_dropout": CONFIG.lora_dropout,
    "bias": CONFIG.lora_bias,
    "task_type": CONFIG.lora_task_type,
}

DATASET_CONFIG = {
    "dataset_path": CONFIG.dataset_path,
    "max_samples": CONFIG.max_samples,
    "max_length": CONFIG.max_length,
    "truncation": CONFIG.truncation,
    "padding": CONFIG.padding,
    "dataset_name": CONFIG.dataset_name,
    "dataset_split": CONFIG.dataset_split,
}

INFERENCE_CONFIG = {
    "max_new_tokens": CONFIG.max_new_tokens,
    "min_new_tokens": CONFIG.min_new_tokens,
    "temperature": CONFIG.temperature,
    "top_p": CONFIG.top_p,
    "top_k": CONFIG.top_k,
    "do_sample": CONFIG.do_sample,
    "enable_thinking": CONFIG.enable_thinking,
    "max_thinking_tokens": CONFIG.max_thinking_tokens,
    "min_thinking_tokens": CONFIG.min_thinking_tokens,
    "use_thinking_processor": CONFIG.use_thinking_processor,
}

# The main training config (previously `REINFORCE_CONFIG`)
TRAIN_CONFIG = CONFIG

from data_utils import load_json_dataset, filter_valid_conversations, prepare_dataset_for_reinforce
from data_utils_mbpp import load_mbpp_dataset, prepare_mbpp_dataset_for_reinforce
from data_utils_test_hacking import (
    load_test_hacking_dataset,
    prepare_test_hacking_dataset_for_reinforce,
)
from reward_model import get_reward_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchThinkingTokenBudgetProcessor(LogitsProcessor):
    """Optimized thinking token processor that handles batched generation."""
    
    def __init__(self, tokenizer, max_thinking_tokens=None, batch_size=8, min_thinking_tokens=0):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.min_thinking_tokens = min_thinking_tokens
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

            # NEW: Detect if a </think> tag was already generated naturally.
            if not self.stopped_thinking[batch_idx]:
                seq = input_ids[batch_idx]
                end_len = len(self.think_end_tokens)
                if end_len > 0 and seq.size(-1) >= end_len:
                    if seq[-end_len:].tolist() == self.think_end_tokens:
                        self.stopped_thinking[batch_idx] = True
            
            self.tokens_generated[batch_idx] += 1
                    
            if self.max_thinking_tokens == 0 and not self.stopped_thinking[batch_idx] and self.tokens_generated[batch_idx] > 0:
                self._set_all_scores_to_neg_inf(scores, batch_idx)
                # self._set_token_score(scores, self.nl_tokens, 0.0, batch_idx)
                self._set_token_score(scores, self.think_end_tokens, 0.0, batch_idx)
                # Do NOT set stopped_thinking here; we will mark it true only
                # after the model actually generates the </think> tag on the
                # subsequent step. This avoids the immediately-following
                # block that zeroes its probability and lets the forced tag
                # be sampled correctly.
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
                    # Wait until the </think> tag has actually been emitted
                    # before marking the sequence as stopped. Otherwise the
                    # "stopped" block below will erase the boost we just gave
                    # to the end-tag logits and the model could exceed the
                    # budget.

            if not self.stopped_thinking[batch_idx] and self.tokens_generated[batch_idx] < self.min_thinking_tokens:
                for tid in self.think_end_tokens:
                    scores[batch_idx][tid] = self.neg_inf

            if self.stopped_thinking[batch_idx]:
                for tid in self.think_end_tokens:
                    scores[batch_idx][tid] = self.neg_inf

        return scores

def setup_model_and_tokenizer(config):
    """Set up model and tokenizer for REINFORCE training."""
    logger.info("Setting up model and tokenizer...")
    
    # Check if we should resume from checkpoint
    checkpoint_path = None
    if config.resume_from_checkpoint:
        checkpoint_path = get_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found in {config.checkpoint_dir}, starting from scratch")
    
    if checkpoint_path:
        # Load model and tokenizer from checkpoint
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
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
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            quantization_config=bnb_config,
            device_map=MODEL_CONFIG["device_map"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info(f"Loaded model and tokenizer from checkpoint: {checkpoint_path}")
    else:
        # If user specified a LoRA adapter repo, load it instead of random init
        if getattr(config, "lora_adapter_repo", None):
            from peft import PeftConfig, PeftModel

            adapter_repo = config.lora_adapter_repo
            logger.info(f"Loading existing LoRA adapters from {adapter_repo}")

            # Load adapter config to find underlying base model name
            peft_cfg = PeftConfig.from_pretrained(adapter_repo)

            tokenizer = AutoTokenizer.from_pretrained(
                peft_cfg.base_model_name_or_path,
                trust_remote_code=True,
                padding_side="left",
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
                peft_cfg.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map=MODEL_CONFIG["device_map"],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Load LoRA weights and keep them trainable
            model = PeftModel.from_pretrained(base_model, adapter_repo, is_trainable=True)
            logger.info("Loaded LoRA adapters – continuing training with REINFORCE")
        else:
            # Standard path: create fresh LoRA adapters
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

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG["model_name"],
                quantization_config=bnb_config,
                device_map=MODEL_CONFIG["device_map"],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            logger.info("Initialized new model with LoRA")
    
    model.train()
    logger.info("Model and tokenizer setup complete")
    return model, tokenizer

def setup_reference_model(config):
    """Set up a frozen reference model for KL divergence calculation."""
    if not config.use_kl_penalty:
        return None
    
    logger.info("Setting up reference model for KL penalty...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load the base model without LoRA as reference
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Freeze the reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    logger.info("Reference model setup complete")
    return ref_model

def prepare_dataset(tokenizer):
    """Prepare the dataset for REINFORCE training."""
    logger.info("Preparing dataset...")
    
    dataset_name = DATASET_CONFIG.get("dataset_name")
    
    # Support Test Hacking coding dataset
    if dataset_name == "test_hacking":
        ds = load_test_hacking_dataset(
            DATASET_CONFIG["dataset_path"],
            max_samples=DATASET_CONFIG.get("max_samples"),
        )
        dataset = prepare_test_hacking_dataset_for_reinforce(
            ds,
            tokenizer,
            max_length=DATASET_CONFIG["max_length"],
            truncation=DATASET_CONFIG["truncation"],
            enable_thinking=INFERENCE_CONFIG["enable_thinking"],
        )
        logger.info(f"Test hacking dataset prepared: {len(dataset)} samples")
        return dataset
    
    # Support MBPP coding dataset out of the box
    if dataset_name == "mbpp":
        mbpp_split = DATASET_CONFIG.get("dataset_split", "sanitized")
        ds = load_mbpp_dataset(mbpp_split, max_samples=DATASET_CONFIG.get("max_samples"))
        dataset = prepare_mbpp_dataset_for_reinforce(
            ds,
            tokenizer,
            max_length=DATASET_CONFIG["max_length"],
            truncation=DATASET_CONFIG["truncation"],
            enable_thinking=INFERENCE_CONFIG["enable_thinking"]
        )
        logger.info(f"MBPP dataset prepared: {len(dataset)} samples")
        return dataset
    
    # Fallback to JSON conversation dataset
    raw_data = load_json_dataset(
        DATASET_CONFIG["dataset_path"],
        max_samples=DATASET_CONFIG["max_samples"]
    )
    
    valid_data = filter_valid_conversations(raw_data)
    dataset = prepare_dataset_for_reinforce(
        valid_data,
        tokenizer,
        max_length=DATASET_CONFIG["max_length"],
        truncation=DATASET_CONFIG["truncation"],
        enable_thinking=INFERENCE_CONFIG["enable_thinking"]
    )
    
    logger.info(f"Dataset prepared: {len(dataset)} samples")
    return dataset

def generate_responses(model, tokenizer, input_ids, config):
    """Generate responses using the model."""
    batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1
    
    generation_kwargs = {
        "max_new_tokens": INFERENCE_CONFIG["max_new_tokens"],
        "min_new_tokens": INFERENCE_CONFIG.get("min_new_tokens", 1),
        "do_sample": INFERENCE_CONFIG["do_sample"],
        "temperature": INFERENCE_CONFIG["temperature"],
        "top_p": INFERENCE_CONFIG["top_p"],
        "top_k": INFERENCE_CONFIG["top_k"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    
    # Conditionally use thinking processor based on enable_thinking and use_thinking_processor settings
    if INFERENCE_CONFIG["enable_thinking"] and INFERENCE_CONFIG["use_thinking_processor"]:
        thinking_processor = BatchThinkingTokenBudgetProcessor(
            tokenizer, 
            max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"],
            min_thinking_tokens=INFERENCE_CONFIG["min_thinking_tokens"],
            batch_size=batch_size
        )
        generation_kwargs["logits_processor"] = [thinking_processor]
    elif INFERENCE_CONFIG["enable_thinking"] and not INFERENCE_CONFIG["use_thinking_processor"]:
        # Enable thinking but don't use the processor - let model generate as many CoT tokens as needed
        logger.info("Thinking enabled but logit processor disabled - model can generate unlimited CoT tokens")
    
    # Mixed-precision generation for speed / memory
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids,
            **generation_kwargs
        )
    
    return outputs

def identify_thinking_tokens(response_token_ids, tokenizer):
    """
    Identify all tokens that are inside <think></think> tags.
    
    Args:
        response_token_ids: Tensor of shape (batch_size, seq_len) containing token IDs
        tokenizer: The tokenizer used to encode the tokens
    
    Returns:
        thinking_mask: Boolean tensor of same shape as response_token_ids where True indicates thinking tokens
    """
    batch_size, seq_len = response_token_ids.shape
    thinking_mask = torch.zeros_like(response_token_ids, dtype=torch.bool)
    
    # Encode the thinking tags
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    for batch_idx in range(batch_size):
        inside_thinking = False
        for token_idx in range(seq_len):
            token_id = response_token_ids[batch_idx, token_idx].item()
            
            # Check if this token starts a thinking block
            if token_id in think_start_tokens:
                inside_thinking = True
            
            # Mark this token as a thinking token if we're inside thinking
            if inside_thinking:
                thinking_mask[batch_idx, token_idx] = True
            
            # Check if this token ends a thinking block
            if token_id in think_end_tokens:
                inside_thinking = False
    
    return thinking_mask

def compute_reinforce_loss(model, input_ids, response_ids, rewards, tokenizer, ref_model=None, config=None):
    """Compute REINFORCE loss with KL penalty and advantage calculation."""
    # Concatenate input and response
    full_ids = torch.cat([input_ids, response_ids], dim=-1)
    
    # Forward pass with autocast for mixed-precision
    with autocast(dtype=torch.bfloat16):
        outputs = model(full_ids, return_dict=True)
        logits = outputs.logits
    
    # Get logits for response tokens only
    response_start_idx = input_ids.size(-1)
    response_logits = logits[:, response_start_idx-1:-1]  # -1 for shift
    
    # Compute log probabilities for current model
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # Get the actual token indices for response
    response_token_ids = response_ids
    
    # Gather log probabilities for the actual tokens
    gathered_log_probs = torch.gather(log_probs, -1, response_token_ids.unsqueeze(-1)).squeeze(-1)
    
    # Create mask to zero gradients on all thinking tokens (inside <think></think> tags)
    thinking_mask = identify_thinking_tokens(response_token_ids, tokenizer)
    
    # Create a mask where 1 indicates tokens that should contribute to gradients
    # and 0 indicates tokens that should have zero gradients (all thinking tokens)
    gradient_mask = torch.ones_like(response_token_ids, dtype=torch.float32)
    
    # Zero gradients for all thinking tokens if enabled in config
    if config and config.zero_thinking_gradients:
        gradient_mask[thinking_mask] = 0.0
        
        # Count the number of thinking tokens zeroed
        thinking_tokens_zeroed = thinking_mask.sum().item()
        
        # Log the number of tokens that were zeroed out
        if thinking_tokens_zeroed > 0:
            logger.debug(f"Zeroed gradients on {thinking_tokens_zeroed} thinking tokens (inside <think></think> tags)")
    else:
        # If gradient zeroing is disabled, all tokens contribute to gradients
        thinking_tokens_zeroed = 0
        if config and not config.zero_thinking_gradients:
            logger.debug("Thinking token gradient zeroing is disabled - all tokens will contribute to gradients")
    
    # Apply the gradient mask to log probabilities
    masked_log_probs = gathered_log_probs * gradient_mask
    
    # Sum log probabilities for each sequence (only non-masked tokens contribute)
    sequence_log_probs = masked_log_probs.sum(dim=-1)
    
    # Convert rewards to tensor and ensure same device
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=sequence_log_probs.device)
    
    kl_penalty = torch.zeros_like(rewards_tensor)
    if ref_model is not None and config and config.use_kl_penalty:
        with torch.no_grad():
            # Get logits from reference model
            ref_outputs = ref_model(full_ids, return_dict=True)
            ref_logits = ref_outputs.logits[:, response_start_idx - 1:-1]
            # Get probabilities for both models
            current_probs = F.softmax(response_logits, dim=-1)  # [batch, seq_len, vocab_size]
            ref_probs = F.softmax(ref_logits, dim=-1)  # [batch, seq_len, vocab_size]
            # Get log probabilities for both models
            current_log_probs = F.log_softmax(response_logits, dim=-1)  # [batch, seq_len, vocab_size]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # [batch, seq_len, vocab_size]
            # Compute KL divergence: KL(current || ref) = Σ current_probs * (log current_probs - log ref_probs)
            # This is always non-negative by Gibbs inequality
            kl_div_per_token = current_probs * (current_log_probs - ref_log_probs)  # [batch, seq_len, vocab_size]
            kl_div_per_token = kl_div_per_token.sum(dim=-1)  # [batch, seq_len]
            # Apply mask to zero out thinking tokens
            kl_div_per_token = kl_div_per_token * gradient_mask
            # Sum KL divergence across sequence and normalize by number of non-masked tokens
            kl_div = kl_div_per_token.sum(dim=-1)  # [batch]
            sequence_lengths = gradient_mask.sum(dim=-1)  # [batch]
            kl_penalty = kl_div / (sequence_lengths + 1e-8)  # [batch]
    
    # Compute final rewards with KL penalty
    if config and config.use_kl_penalty:
        R = rewards_tensor - config.kl_beta * kl_penalty
    else:
        R = rewards_tensor
    
    # Compute advantage if enabled
    if config and config.use_advantage:
        A = R - R.mean()
    else:
        A = R
    
    # Compute REINFORCE loss: -log_prob * advantage
    loss = -(sequence_log_probs * A.detach()).mean()
    
    return loss, sequence_log_probs, rewards_tensor, kl_penalty, A

def train_reinforce(model, tokenizer, train_dataset, config, ref_model=None):
    """Run REINFORCE training loop."""
    logger.info("Starting REINFORCE training...")
    
    # Log configuration
    if config.use_kl_penalty:
        logger.info(f"Using KL penalty with beta={config.kl_beta}")
    if config.use_advantage:
        logger.info("Using advantage calculation")
    logger.info(f"Thinking token gradient zeroing: {'enabled' if config.zero_thinking_gradients else 'disabled'}")
    
    rollout_file_path = "reinforce/reinforce_rollouts.jsonl"
    os.makedirs(os.path.dirname(rollout_file_path), exist_ok=True)
    
    # Clear rollout file
    with open(rollout_file_path, "w") as f:
        pass
    logger.info(f"Cleared previous rollouts. Saving new rollouts to {rollout_file_path}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_update_steps_per_epoch = len(train_dataset) // config.per_device_train_batch_size
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Setup dataloader
    def collate_fn(batch):
        # Find the maximum length in this batch
        max_length = max(len(f["input_ids"]) for f in batch)
        
        # Pad all sequences to the same length
        padded_batch = []
        for feature in batch:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Left-pad the sequences
            if padding_length > 0:
                input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
            
            padded_feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            # Keep auxiliary fields if present
            if "test_setup_code" in feature:
                padded_feature["test_setup_code"] = feature["test_setup_code"]
            if "test_list" in feature:
                padded_feature["test_list"] = feature["test_list"]
            
            padded_batch.append(padded_feature)
        
        # Convert to tensors
        batch_tensors = {
            "input_ids": torch.tensor([f["input_ids"] for f in padded_batch], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in padded_batch], dtype=torch.long)
        }
        # Add auxiliary non-tensor fields as-is
        if "test_setup_code" in padded_batch[0]:
            batch_tensors["test_setup_code"] = [f["test_setup_code"] for f in padded_batch]
        if "test_list" in padded_batch[0]:
            batch_tensors["test_list"] = [f["test_list"] for f in padded_batch]
        
        return batch_tensors
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Get reward mode
    reward_mode = get_reward_mode()
    logger.info(f"Using reward mode: {reward_mode}")
    
    # Get primary reward function
    primary_reward_fn = get_reward_fn(config.reward_fn_name)
    logger.info(f"Using primary reward function: {config.reward_fn_name}")
    
    secondary_reward_fn = None
    if getattr(config, 'reward_fn_name_2', None):
        secondary_reward_fn = get_reward_fn(config.reward_fn_name_2)
        logger.info(f"Using secondary reward function: {config.reward_fn_name_2}")
    else:
        logger.info("No secondary reward function specified")
    
    total_steps = 0
    global_step = 0
    
    for epoch in range(config.num_train_epochs):
        epoch_loss = 0.0
        epoch_rewards = []
        epoch_thinking_rewards = []
        epoch_kl_penalties = []
        epoch_advantages = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            # --- Forward pass with OOM protection ---------------------------------
            try:
                # Generate responses
                outputs = generate_responses(model, tokenizer, input_ids, config)
                response_ids = outputs.sequences[:, input_ids.size(-1):]
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM during generation at global_step={global_step}. Skipping batch.")
                _clear_memory()
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Decode responses for reward calculation
            responses_text = []
            for i in range(response_ids.size(0)):
                response_text = tokenizer.decode(response_ids[i], skip_special_tokens=True)
                responses_text.append(response_text)
            
            # --------------------------------------------------------------
            # Compute rewards – each function now returns **two** lists:
            #   (after_think_rewards, thinking_rewards)
            # Only the first list participates in optimisation, but we keep
            # both for logging/analysis.
            # --------------------------------------------------------------

            rewards_primary_after, rewards_primary_think = primary_reward_fn(
                responses_text,
                reward_mode=reward_mode,  # kept for backward-compat – ignored by fn
                test_setup_codes=batch.get("test_setup_code"),
                test_lists=batch.get("test_list"),
            )

            if secondary_reward_fn is not None:
                rewards_secondary_after, rewards_secondary_think = secondary_reward_fn(
                    responses_text,
                    reward_mode=reward_mode,
                    test_setup_codes=batch.get("test_setup_code"),
                    test_lists=batch.get("test_list"),
                )

                # Aggregate after-think rewards for optimisation
                rewards = [r1 + r2 for r1, r2 in zip(rewards_primary_after, rewards_secondary_after)]
                thinking_rewards = rewards_secondary_think

            else:
                rewards = rewards_primary_after
                thinking_rewards = [0.0 for _ in rewards]
            
            try:
                # Compute REINFORCE loss with KL penalty and advantage
                loss, log_probs, rewards_tensor, kl_penalty, advantages = compute_reinforce_loss(
                    model, input_ids, response_ids, rewards, tokenizer, ref_model, config
                )
            except torch.cuda.OutOfMemoryError as e:
                # More precise logging – we know the error happened inside compute_reinforce_loss (forward pass)
                logger.exception(
                    "OOM during forward pass (compute_reinforce_loss) at global_step=%d | batch_size=%d | input_len=%d | response_len=%d. Details: %s",
                    global_step,
                    input_ids.size(0),
                    input_ids.size(-1),
                    response_ids.size(-1) if 'response_ids' in locals() else -1,
                    str(e),
                )
                _clear_memory()
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Log if any </think> or newline tokens were found in this batch
            response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]
            think_tokens_in_batch = sum(1 for text in response_texts if '</think>' in text)
            newline_tokens_in_batch = sum(1 for text in response_texts if '\n' in text)
            if think_tokens_in_batch > 0 or newline_tokens_in_batch > 0:
                logger.info(f"Batch contains {think_tokens_in_batch} responses with </think> tokens and {newline_tokens_in_batch} responses with newlines")
            
            # Scale loss so that gradient magnitudes are independent of accumulation length
            loss_scaled = loss / config.gradient_accumulation_steps

            # Backward pass (accumulates gradients)
            try:
                loss_scaled.backward()
            except torch.cuda.OutOfMemoryError as e:
                # Precise OOM location – during backward pass
                logger.exception(
                    "OOM during backward pass at global_step=%d | batch_size=%d | input_len=%d | response_len=%d. Details: %s",
                    global_step,
                    input_ids.size(0),
                    input_ids.size(-1),
                    response_ids.size(-1),
                    str(e),
                )
                _clear_memory()
                optimizer.zero_grad(set_to_none=True)
                continue

            accumulate_step = (step + 1) % config.gradient_accumulation_steps == 0
            last_batch = (step + 1) == len(dataloader)

            if accumulate_step or last_batch:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_rewards.extend(rewards)
            epoch_thinking_rewards.extend(thinking_rewards)
            epoch_kl_penalties.extend(kl_penalty.cpu().numpy())
            epoch_advantages.extend(advantages.cpu().numpy())
            
            # Log to wandb
            if wandb.run is not None and global_step % config.logging_steps == 0:
                # Aggregate batch-level means
                mean_reward_total = np.mean(rewards)
                mean_reward_1 = np.mean(rewards_primary_after)
                mean_reward_2 = (
                    np.mean(rewards_secondary_after) if secondary_reward_fn is not None else None
                )
                mean_thinking_reward = np.mean(thinking_rewards)
                
                log_data = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "reward_total": mean_reward_total,
                    "reward_1": mean_reward_1,
                    "reward_2": mean_reward_2,
                    "thinking_reward": mean_thinking_reward,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "mean_log_prob": log_probs.mean().item(),
                }
                
                # Remove None values (reward_2 when not applicable)
                log_data = {k: v for k, v in log_data.items() if v is not None}
                
                # Add KL penalty and advantage metrics if enabled
                if config.use_kl_penalty:
                    log_data.update({
                        "kl_penalty": kl_penalty.mean().item(),
                    })
                
                if config.use_advantage:
                    log_data.update({
                        "advantage": advantages.mean().item(),
                    })
                
                wandb.log(log_data)
            
            # Save rollouts periodically
            if global_step % config.rollout_save_steps == 0:
                rollout_records = []
                for i in range(len(responses_text)):
                    original_input = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    # Ensure the "response" field is first in the JSON object
                    rollout_record = {
                        "response": responses_text[i],
                        "reward": rewards[i],
                        "thinking_reward": thinking_rewards[i],
                        "step": global_step,
                        "original_input": original_input,
                    }
                    
                    # Add KL penalty and advantage if enabled
                    if config.use_kl_penalty:
                        rollout_record["kl_penalty"] = kl_penalty[i].item()
                    
                    if config.use_advantage:
                        rollout_record["advantage"] = advantages[i].item()
                    
                    rollout_records.append(rollout_record)

                with open(rollout_file_path, "a") as f:
                    for record in rollout_records:
                        f.write(json.dumps(record) + "\n")
            
            # Save model periodically
            if global_step % config.save_steps == 0:
                output_dir = f"./reinforce_output/checkpoint-{global_step}"
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Model saved to {output_dir}")
            
            # Update progress bar
            postfix = {
                "loss": f"{loss.item():.4f}",
                "reward_total": f"{np.mean(rewards):.3f}",
                "reward_1": f"{np.mean(rewards_primary_after):.3f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
            }
            if secondary_reward_fn is not None:
                postfix["reward_2"] = f"{np.mean(rewards_secondary_after):.3f}"
            
            progress_bar.set_postfix(postfix)
            
            global_step += 1
            
            # Check for early stopping
            if config.steps and global_step >= config.steps:
                break
        
        # Log epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_reward = np.mean(epoch_rewards)
        avg_epoch_thinking_reward = np.mean(epoch_thinking_rewards) if epoch_thinking_rewards else 0.0
        avg_epoch_reward_1 = np.mean(epoch_rewards_primary_after) if 'epoch_rewards_primary_after' in locals() and epoch_rewards_primary_after else 0.0
        avg_epoch_reward_2 = np.mean(epoch_rewards_secondary_after) if 'epoch_rewards_secondary_after' in locals() and epoch_rewards_secondary_after else None
        epoch_summary = (
            f"Epoch {epoch+1}: avg_loss={avg_epoch_loss:.4f}, "
            f"total_reward={avg_epoch_reward:.3f}, reward_1={avg_epoch_reward_1:.3f}"
        )
        if avg_epoch_reward_2 is not None:
            epoch_summary += f", reward_2={avg_epoch_reward_2:.3f}"
        epoch_summary += f", thinking_reward={avg_epoch_thinking_reward:.3f}"
        
        if config.use_kl_penalty:
            avg_epoch_kl = np.mean(epoch_kl_penalties)
            epoch_summary += f", avg_kl_penalty={avg_epoch_kl:.3f}"
        
        if config.use_advantage:
            avg_epoch_advantage = np.mean(epoch_advantages)
            epoch_summary += f", advantage={avg_epoch_advantage:.3f}"
        
        logger.info(epoch_summary)
        
        if config.steps and global_step >= config.steps:
            break
    
    logger.info("REINFORCE training completed")

def main():
    """Main training function."""
    logger.info("Starting REINFORCE training setup...")
    # Unified config is used directly; no GPU-specific overrides
    config = TRAIN_CONFIG
    
    if config.log_with == "wandb":
        wandb.init(
            project="hack obfusc",
            name=config.exp_name,
            config=vars(config)
        )
    
    try:
        model, tokenizer = setup_model_and_tokenizer(config)
        ref_model = setup_reference_model(config)
        train_dataset = prepare_dataset(tokenizer)
        train_reinforce(model, tokenizer, train_dataset, config, ref_model)
        
        output_dir = "./reinforce_output/final"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Final model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.log_with == "wandb":
            wandb.finish()

# Utility to free up GPU/CPU memory
def _clear_memory():
    """Attempt to release cached GPU and Python objects."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

if __name__ == "__main__":
    main() 