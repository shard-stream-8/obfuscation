from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ExpertIterationConfig:
    model_name: str = "google/gemma-2-2b-it"
    dataset_path: str = "/root/obfuscation/datasets/alpaca_5000.jsonl"
    num_samples: int = 32  # The maximum number of samples to use from the dataset. Use -1 for all samples
    expert_iterations: int = 3  # The total number of expert iteration cycles to perform
    num_generations_per_prompt: int = 8  # The number of responses to generate for each prompt in the generation phase
    sft_epochs_per_iteration: int = 8  # The number of epochs to train the model during the SFT phase of each iteration
    mini_batch_size: int = 4  # The batch size for the SFT training
    learning_rate: float = 1e-3
    sft_max_length: int = 256  # The maximum sequence length for the tokenizer during SFT
    max_new_tokens: int = 100  # The maximum number of new tokens to generate in responses
    filter_threshold: float = 0.0  # The minimum reward threshold for a response to be included in the SFT dataset
    push_to_hub_every_epoch: bool = False  # Whether to push the model to the Hugging Face Hub after each iteration
    hub_model_id: str = "my-expert-model"  # The repository ID for pushing the model to the Hub
    experiment_dir: str = "./expert_iteration_logs"  # The directory to save logs and experiment artifacts
    enable_thinking: bool = True  # Whether to use <think> tokens when formatting the prompt

    # Resuming and checkpointing
    resume_from_checkpoint: bool = False
    """Whether to resume training from a previously saved LoRA adapter."""
    checkpoint_model_id: Optional[str] = None
    """The Hugging Face Hub model ID of the LoRA adapter to resume from."""

    # LoRA configuration
    use_lora: bool = True
    """Whether to use LoRA for parameter-efficient fine-tuning."""
    lora_r: int = 16
    """The rank of the LoRA update matrices."""
    lora_alpha: int = 32
    """The scaling factor for the LoRA update matrices."""
    lora_dropout: float = 0.05
    """The dropout probability for the LoRA layers."""
    lora_target_modules: list[str] = field(
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
    """The list of module names to apply LoRA to."""

    # Quantization configuration
    quantization_config: Optional[Dict[str, Any]] = None
    """Quantization configuration (e.g., {'load_in_4bit': True})."""

    # WandB configuration
    log_to_wandb: bool = True
    """Whether to log metrics to Weights & Biases."""
    wandb_project: str = "expert-iteration-qwen"
    """The WandB project name to log to."""
    wandb_entity: Optional[str] = None
    """The WandB entity (username or team) to log to."""

    vllm_kwargs: Dict[str, Any] = field(default_factory=lambda: {  # Keyword arguments to pass to the vLLM engine
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.7,
        "dtype": "bfloat16"
    })