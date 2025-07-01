"""
Configuration file for SFT training parameters.
Modify these settings to customize your training setup.
"""

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

# Training Configuration
TRAINING_CONFIG = {
    # Basic training parameters
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_grad_norm": 0.3,
    "train_on_responses_only": True,  # Train only on assistant responses
    "response_token_offset": 1,  # Number of tokens to skip after assistant tag before training
    
    # Logging and saving
    "save_steps": 500,
    "logging_steps": 10,
    "eval_steps": 500,
    "save_total_limit": 3,
    
    # Optimization
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True,
    
    # Data processing
    "max_length": 2048,
    "dataloader_pin_memory": False,
    "remove_unused_columns": False,
    
    # Evaluation
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "logging_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    # Other
    "push_to_hub": False,
    "report_to": "wandb",  # Set to "none" to disable wandb
    "run_name": "qwen3-sft",
    "ddp_find_unused_parameters": False
}

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_path": "datasets/math_5000_number_words.jsonl",
    "max_samples": 100,  # None for no limit, or set a number to limit samples
    "max_length": 2048,
    "truncation": True,
    "padding": False
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
    "enable_thinking": False  # Always disabled for SFT
}

# Memory Optimization for A40 (24GB VRAM)
A40_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
    "gradient_checkpointing": True,
    "fp16": True
}

# Memory Optimization for V100 (16GB VRAM)
V100_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_length": 1024,
    "gradient_checkpointing": True,
    "fp16": True
}

# Memory Optimization for RTX 4090 (24GB VRAM)
RTX4090_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,
    "gradient_checkpointing": True,
    "fp16": True
}

def get_config_for_gpu(gpu_type: str = "auto"):
    """
    Get training configuration optimized for specific GPU type.
    
    Args:
        gpu_type: GPU type ("a40", "v100", "rtx4090", "auto")
        
    Returns:
        Dictionary with optimized configuration
    """
    base_config = TRAINING_CONFIG.copy()
    
    if gpu_type.lower() == "a40":
        base_config.update(A40_CONFIG)
    elif gpu_type.lower() == "v100":
        base_config.update(V100_CONFIG)
    elif gpu_type.lower() == "rtx4090":
        base_config.update(RTX4090_CONFIG)
    elif gpu_type.lower() == "auto":
        # Auto-detect based on available memory
        import torch
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_gb >= 20:
                base_config.update(A40_CONFIG)
            elif memory_gb >= 15:
                base_config.update(V100_CONFIG)
            else:
                # Conservative settings for smaller GPUs
                base_config.update({
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 16,
                    "max_length": 512,
                    "gradient_checkpointing": True,
                    "fp16": True
                })
    
    return base_config 