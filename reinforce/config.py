"""
Configuration for REINFORCE training.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training."""
    learning_rate: float = 3e-4
    batch_size: int = 32
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    seed: int = 42
    max_grad_norm: float = 1.0
    exp_name: str = "reinforce_uppercase_config"
    log_with: str = "wandb"
    project_kwargs: dict = None
    tracker_project_name: str = "trl"
    steps: int = 100
    logging_steps: int = 1
    save_steps: int = 5
    warmup_steps: int = 5
    rollout_save_steps: int = 5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 64
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    resume_from_checkpoint: bool = False
    checkpoint_dir: str = "./reinforce_output"
    
    def __post_init__(self):
        if self.project_kwargs is None:
            self.project_kwargs = {}

# Model Configuration
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
    "device_map": "auto",
    "torch_dtype": "bfloat16",
    "trust_remote_code": True
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# REINFORCE Configuration
REINFORCE_CONFIG = REINFORCEConfig()

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_path": "/root/obfuscation/datasets/alpaca_100000.jsonl",
    "max_samples": None,
    "max_length": 2048,
    "truncation": True,
    "padding": False
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 64,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "do_sample": True,
    "enable_thinking": True,
    "max_thinking_tokens": 32,
    "use_thinking_processor": True
}

# GPU-specific configurations
A100_CONFIG = {
    "per_device_train_batch_size": 32,
    "mini_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "fp16": True,
    "bf16": False
}

def get_config_for_gpu(gpu_type: str = "auto"):
    """Get REINFORCE configuration optimized for specific GPU type."""
    config = REINFORCE_CONFIG
    
    if gpu_type.lower() == "a100":
        config.per_device_train_batch_size = A100_CONFIG["per_device_train_batch_size"]
        config.mini_batch_size = A100_CONFIG["mini_batch_size"]
        config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
        config.gradient_checkpointing = A100_CONFIG["gradient_checkpointing"]
        config.fp16 = A100_CONFIG["fp16"]
        config.bf16 = A100_CONFIG["bf16"]
    elif gpu_type.lower() == "auto":
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb >= 70:
                config.per_device_train_batch_size = A100_CONFIG["per_device_train_batch_size"]
                config.mini_batch_size = A100_CONFIG["mini_batch_size"]
                config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
                config.gradient_checkpointing = A100_CONFIG["gradient_checkpointing"]
                config.fp16 = A100_CONFIG["fp16"]
                config.bf16 = A100_CONFIG["bf16"]
            elif memory_gb >= 20:
                config.per_device_train_batch_size = 4
                config.mini_batch_size = 1
                config.gradient_accumulation_steps = 4
                config.gradient_checkpointing = True
                config.fp16 = False
                config.bf16 = True
            else:
                config.per_device_train_batch_size = 2
                config.mini_batch_size = 1
                config.gradient_accumulation_steps = 8
                config.gradient_checkpointing = True
                config.fp16 = False
                config.bf16 = True
    
    return config

def get_reward_mode():
    """Get reward calculation mode based on enable_thinking setting."""
    return "thinking_only" if INFERENCE_CONFIG["enable_thinking"] else "all_tokens"

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the given directory."""
    import os
    import re
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint directories with pattern "checkpoint-{number}"
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    checkpoints = []
    
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, item_path))
    
    if not checkpoints:
        return None
    
    # Return the checkpoint with the highest step number
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return latest_checkpoint[1] 