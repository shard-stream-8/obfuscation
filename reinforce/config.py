"""
Configuration for REINFORCE training.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE training (pruned)."""
    # Optimisation
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1  # number of rollout batches to accumulate
    max_grad_norm: float = 1.0

    # Runtime / bookkeeping
    exp_name: str = "spillover"
    log_with: str = "wandb"
    steps: int = 1000  # max training steps (rollout batches)
    logging_steps: int = 1
    save_steps: int = 20
    warmup_steps: int = 0
    rollout_save_steps: int = 5

    # LR scheduler & weight decay
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"

    # Epoch / batch sizes
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 64

    # Precision / resume
    fp16: bool = True
    bf16: bool = False
    resume_from_checkpoint: bool = False
    checkpoint_dir: str = "./reinforce_output"

    # Reward functions
    reward_fn_name: str = "mbpp"
    reward_fn_name_2: Optional[str] = "keyword"  # optional second reward

    # KL / advantage
    use_kl_penalty: bool = True
    kl_beta: float = 0.3
    use_advantage: bool = True

    # Other
    zero_thinking_gradients: bool = True

    # ---- LoRA adapter ----
    # If provided, the trainer will load existing LoRA weights from this
    # Hugging Face Hub repo (or local path) instead of randomly initialising
    # new adapters.
    lora_adapter_repo: Optional[str] = "jacobcd52/qwen3_4b_hacker"

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
    "lora_dropout": 0.,
    "bias": "all",
    "task_type": "CAUSAL_LM"
}

# REINFORCE Configuration
REINFORCE_CONFIG = REINFORCEConfig()

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_path": "/root/obfuscation/datasets/math_5000_number_words.jsonl",
    "max_samples": None,
    "max_length": 2048,
    "truncation": True,
    "padding": False,
    "dataset_name": "mbpp",
    "dataset_split": "full"
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 500,
    "min_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "do_sample": True,
    "enable_thinking": True,
    "max_thinking_tokens": 100,
    "min_thinking_tokens": 50,
    "use_thinking_processor": True
}

# GPU-specific configurations
A100_CONFIG = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "fp16": False,
    "bf16": True
}

def get_config_for_gpu(gpu_type: str = "auto"):
    """Get REINFORCE configuration optimized for specific GPU type."""
    config = REINFORCE_CONFIG
    
    if gpu_type.lower() == "a100":
        print("Using A100 config")
        config.per_device_train_batch_size = A100_CONFIG["per_device_train_batch_size"]
        config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
        config.fp16 = A100_CONFIG["fp16"]
        config.bf16 = A100_CONFIG["bf16"]
    elif gpu_type.lower() == "auto":
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb >= 70:
                config.per_device_train_batch_size = A100_CONFIG["per_device_train_batch_size"]
                config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
                config.fp16 = A100_CONFIG["fp16"]
                config.bf16 = A100_CONFIG["bf16"]
            elif memory_gb >= 20:
                config.per_device_train_batch_size = 4
                config.gradient_accumulation_steps = 4
                config.fp16 = False
                config.bf16 = True
            else:
                config.per_device_train_batch_size = 2
                config.gradient_accumulation_steps = 8
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