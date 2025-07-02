"""
Configuration for PPO training.
"""

from trl import PPOConfig

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

# PPO Configuration
PPO_CONFIG = PPOConfig(
    learning_rate=1e-5,
    batch_size=128,
    mini_batch_size=32,
    gradient_accumulation_steps=1,
    ppo_epochs=1,
    seed=42,
    cliprange=0.2,
    vf_coef=0.1,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    whiten_rewards=False,
    max_grad_norm=1.0,
    adap_kl_ctrl=True,
    init_kl_coef=0.2,
    target=6.0,
    horizon=10000.0,
    kl_penalty='kl',
    exp_name="ppo_uppercase_config",
    log_with="wandb",
    project_kwargs={},
    tracker_project_name="trl",
    steps=1000,
)

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_path": "/root/obfuscation/datasets/alpaca_5000_uppercase.jsonl",
    "max_samples": None,
    "max_length": 2048,
    "truncation": True,
    "padding": False
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
    "enable_thinking": True,
    "max_thinking_tokens": 8
}

# GPU-specific configurations
A100_CONFIG = {
    "per_device_train_batch_size": 128,
    "mini_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "fp16": True,
    "bf16": False
}

def get_config_for_gpu(gpu_type: str = "auto"):
    """Get PPO configuration optimized for specific GPU type."""
    config = PPO_CONFIG
    
    if gpu_type.lower() == "a100":
        config.batch_size = A100_CONFIG["per_device_train_batch_size"]
        config.mini_batch_size = A100_CONFIG["mini_batch_size"]
        config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
    elif gpu_type.lower() == "auto":
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb >= 70:
                config.batch_size = A100_CONFIG["per_device_train_batch_size"]
                config.mini_batch_size = A100_CONFIG["mini_batch_size"]
                config.gradient_accumulation_steps = A100_CONFIG["gradient_accumulation_steps"]
            elif memory_gb >= 20:
                config.batch_size = 4
                config.mini_batch_size = 1
                config.gradient_accumulation_steps = 4
            else:
                config.batch_size = 2
                config.mini_batch_size = 1
                config.gradient_accumulation_steps = 8
    
    return config 