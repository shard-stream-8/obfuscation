import os
import logging
import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from data_utils import load_json_dataset, filter_valid_conversations, prepare_dataset_for_sft, create_dummy_dataset

logger = logging.getLogger(__name__)

class QwenSFTTrainer:
    """
    SFT Trainer for Qwen3-4B model with thinking disabled.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        output_dir: str = "./sft_output",
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        device_map: str = "auto"
    ):
        """
        Initialize the SFT trainer.
        
        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save outputs
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration dictionary
            device_map: Device mapping strategy
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.device_map = device_map
        
        # Default LoRA config
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
        self.lora_config = lora_config
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA configuration")
            peft_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_dataset(
        self,
        dataset_path: Optional[str] = None,
        use_dummy_data: bool = True,
        dummy_samples: int = 100,
        max_length: int = 2048
    ):
        """
        Prepare the dataset for training.
        
        Args:
            dataset_path: Path to JSON dataset file
            use_dummy_data: Whether to use dummy data for testing
            dummy_samples: Number of dummy samples to create
            max_length: Maximum sequence length
        """
        if use_dummy_data:
            logger.info(f"Creating dummy dataset with {dummy_samples} samples")
            conversations = create_dummy_dataset(dummy_samples)
        else:
            if dataset_path is None:
                raise ValueError("dataset_path must be provided when use_dummy_data=False")
            logger.info(f"Loading dataset from: {dataset_path}")
            conversations = load_json_dataset(dataset_path)
        
        # Filter valid conversations
        conversations = filter_valid_conversations(conversations)
        
        # Prepare dataset for SFT
        self.dataset = prepare_dataset_for_sft(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_length=max_length,
            truncation=True
        )
        
        logger.info(f"Dataset prepared with {len(self.dataset)} samples")
    
    def train(
        self,
        training_args: Optional[Dict[str, Any]] = None,
        save_steps: int = 500,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_total_limit: int = 3,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 0.3,
        dataloader_pin_memory: bool = False
    ):
        """
        Train the model using SFT.
        
        Args:
            training_args: Custom training arguments
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_total_limit: Maximum number of checkpoints to keep
            learning_rate: Learning rate
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm
            dataloader_pin_memory: Whether to pin memory in dataloader
        """
        if not hasattr(self, 'dataset'):
            raise ValueError("Dataset not prepared. Call prepare_dataset() first.")
        
        # Default training arguments
        if training_args is None:
            training_args = {
                "output_dir": self.output_dir,
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "logging_steps": logging_steps,
                "save_steps": save_steps,
                "eval_steps": eval_steps,
                "save_total_limit": save_total_limit,
                "max_grad_norm": max_grad_norm,
                "dataloader_pin_memory": dataloader_pin_memory,
                "remove_unused_columns": False,
                "push_to_hub": False,
                "report_to": "wandb" if os.getenv("WANDB_PROJECT") else "none",
                "run_name": f"sft-{self.model_name.split('/')[-1]}",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "fp16": True,
                "bf16": False,
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing": True,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "weight_decay": 0.01,
                "evaluation_strategy": "steps" if eval_steps > 0 else "no",
                "save_strategy": "steps",
                "logging_strategy": "steps",
            }
        
        # Create training arguments
        training_args = TrainingArguments(**training_args)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            data_collator=data_collator,
            max_seq_length=training_args.max_length if hasattr(training_args, 'max_length') else 2048,
            dataset_text_field="text"  # This will be ignored as we're using custom dataset format
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to: {self.output_dir}")
    
    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        save_path = path or self.output_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from: {path}") 