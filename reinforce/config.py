# -----------------------------------------------------------------------------
# Unified configuration for REINFORCE training.
# All options are collected in a single dataclass instance `CONFIG`.
# No GPU-specific overrides or separate config objects remain.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import os
import re


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Optimisation
<<<<<<< HEAD
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1  # number of rollout batches to accumulate
    max_grad_norm: float = 0.5
=======
    # ------------------------------------------------------------------
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 2  # number of rollout batches to accumulate
    max_grad_norm: float = 0.3
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
>>>>>>> 278d4d6803fc38710733a5870a4d73ce25e534db

    # ------------------------------------------------------------------
    # Runtime / bookkeeping
<<<<<<< HEAD
    exp_name: str = "mdpp_adverb"
=======
    # ------------------------------------------------------------------
    exp_name: str = "hacker"
>>>>>>> 278d4d6803fc38710733a5870a4d73ce25e534db
    log_with: str = "wandb"
    steps: int = 100  # max training steps (rollout batches)
    logging_steps: int = 1
    save_steps: int = 20
    warmup_steps: int = 20
    rollout_save_steps: int = 5

    # ------------------------------------------------------------------
    # LR scheduler & weight decay
    # ------------------------------------------------------------------
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"

    # ------------------------------------------------------------------
    # Precision / resume
<<<<<<< HEAD
=======
    # ------------------------------------------------------------------
>>>>>>> 278d4d6803fc38710733a5870a4d73ce25e534db
    fp16: bool = False
    bf16: bool = True
    resume_from_checkpoint: bool = False
    checkpoint_dir: str = "./reinforce_output"

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    reward_fn_name: str = "test_hacking"
    reward_fn_name_2: Optional[str] = "keyword"  # optional second reward

    # ------------------------------------------------------------------
    # KL / advantage
<<<<<<< HEAD
    use_kl_penalty: bool = False
    kl_beta: float = 0.2
=======
    # ------------------------------------------------------------------
    use_kl_penalty: bool = True
    kl_beta: float = 0.1
>>>>>>> 278d4d6803fc38710733a5870a4d73ce25e534db
    use_advantage: bool = True

    # ------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------
    zero_thinking_gradients: bool = True
    # Hugging Face Hub repo to push LoRA adapters after training (optional)
    hf_repo_out: Optional[str] = "jacobcd52/qwen3_4b_mdpp_adverb"

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------
    model_name: str = "Qwen/Qwen3-4B"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    # ------------------------------------------------------------------
    # LoRA configuration
    # ------------------------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_dropout: float = 0.05
    lora_bias: str = "all"
    lora_task_type: str = "CAUSAL_LM"

    # If provided, the trainer will load existing LoRA weights from this
    # Hugging Face Hub repo (or local path) instead of randomly initialising
    # new adapters.
    lora_adapter_repo: Optional[str] = None

    # ------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------
    dataset_path: str = "/root/obfuscation/datasets/test_hacking/coding_problems.jsonl"
    max_samples: Optional[int] = None
    max_length: int = 1024
    truncation: bool = True
    padding: bool = False
    dataset_name: str = "test_hacking"
    dataset_split: Optional[str] = None

    # ------------------------------------------------------------------
    # Inference configuration
    # ------------------------------------------------------------------
    max_new_tokens: int = 500
    min_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    enable_thinking: bool = True
    max_thinking_tokens: int = 200
    min_thinking_tokens: int = 100
    use_thinking_processor: bool = True


# Instantiate the unified configuration
CONFIG = Config()


# -----------------------------------------------------------------------------
# Helper utilities that depend on the unified config
# -----------------------------------------------------------------------------

def get_reward_mode(cfg: Config | None = None) -> str:
    """Return reward calculation mode based on `enable_thinking`."""
    cfg = cfg or CONFIG
    return "thinking_only" if cfg.enable_thinking else "all_tokens"


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint directories with pattern "checkpoint-{number}"
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
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