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
from config import DATASET_CONFIG, TRAINING_CONFIG
from response_only_utils import setup_response_only_training

logger = logging.getLogger(__name__)

class QwenSFTTrainer:
    """
    SFT Trainer for Qwen3-4B model with thinking disabled.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        output_dir: str = "./qwen3_4b_hacker",
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
            
            # Ensure LoRA parameters require gradients
            for name, param in self.model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
        
        # Ensure model is in training mode
        self.model.train()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_dataset(
        self,
        dataset_path: Optional[str] = None,
        use_dummy_data: bool = True,
        dummy_samples: int = 100,
        max_length: int = 2048,
        max_samples: Optional[int] = None
    ):
        """
        Prepare the dataset for training.
        
        Args:
            dataset_path: Path to JSON dataset file (defaults to config value)
            use_dummy_data: Whether to use dummy data for testing
            dummy_samples: Number of dummy samples to create
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to use (defaults to config value)
        """
        # Use config defaults if not provided
        if dataset_path is None:
            dataset_path = DATASET_CONFIG["dataset_path"]
        if max_samples is None:
            max_samples = DATASET_CONFIG["max_samples"]
            
        if use_dummy_data:
            logger.info(f"Creating dummy dataset with {dummy_samples} samples")
            conversations = create_dummy_dataset(dummy_samples)
        else:
            if dataset_path is None:
                raise ValueError("dataset_path must be provided when use_dummy_data=False")
            logger.info(f"Loading dataset from: {dataset_path}")
            if max_samples is not None:
                logger.info(f"Limiting to first {max_samples} samples")
            conversations = load_json_dataset(dataset_path, max_samples=max_samples)
        
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
        dataloader_pin_memory: bool = False,
        train_on_responses_only: Optional[bool] = None,
        response_token_offset: Optional[int] = None
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
            train_on_responses_only: Train only on assistant responses (defaults to config)
            response_token_offset: Number of tokens to skip after assistant tag (defaults to config)
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
                "push_to_hub": True,
                "report_to": "wandb" if os.getenv("WANDB_PROJECT") else "none",
                "run_name": f"sft-{self.model_name.split('/')[-1]}",
                "load_best_model_at_end": False,
                "fp16": True,
                "bf16": False,
                "ddp_find_unused_parameters": False,
                "gradient_checkpointing": False,
                "optim": "adamw_torch",
                "lr_scheduler_type": "cosine",
                "weight_decay": 0.01,
                "eval_strategy": "no",
                "save_strategy": "steps",
                "logging_strategy": "steps",
            }
        
        # Create training arguments
        training_args = TrainingArguments(**training_args)
        
        # Ensure model is in training mode before creating trainer
        self.model.train()
        
        # Create SFT trainer  
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset_text_field="text"
        )
        
        # Configure for response-only training if enabled
        if train_on_responses_only is None:
            train_on_responses_only = TRAINING_CONFIG.get("train_on_responses_only", False)
        
        if train_on_responses_only:
            logger.info("Configuring trainer for response-only training...")
            if response_token_offset is None:
                response_token_offset = TRAINING_CONFIG.get("response_token_offset", 0)
            trainer = setup_response_only_training(trainer, self.tokenizer, response_token_offset)
        
        # Train the model
        logger.info("Starting training...")
        if train_on_responses_only:
            logger.info("Training will only optimize on assistant response tokens")
        trainer.train()

        print("TRAINING COMPLETED")
        
        # Save the final model locally first
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print("MODEL SAVED LOCALLY")
        
        # If push_to_hub requested, optionally merge LoRA adapters then upload
        if getattr(training_args, "push_to_hub", False):
            merge_before_push = TRAINING_CONFIG.get("merge_before_push", False)
            if merge_before_push and self.use_lora:
                try:
                    logger.info("merge_before_push=True – merging LoRA adapters with base model before upload …")
                    print("[INFO] Merging LoRA adapters with base model before push – this can take several minutes…")
                    # Merge adapters and overwrite saved weights
                    merged_model = self.model.merge_and_unload()
                    merged_model.save_pretrained(self.output_dir, safe_serialization=True, max_shard_size="2GB")
                    # Ensure trainer uses merged model for the Hub push
                    trainer.model = merged_model
                except Exception as e:
                    logger.warning(f"Failed to merge adapters before push: {e}")
            try:
                logger.info("Pushing checkpoint to the Hugging Face Hub …")
                print("[INFO] Ensuring remote repository exists …")
                from huggingface_hub import HfApi
                api = HfApi()
                repo_id = getattr(training_args, "hub_model_id", None) or TRAINING_CONFIG.get("hub_model_id") or trainer.args.run_name

                try:
                    api.create_repo(repo_id, private=False, exist_ok=True)
                    print(f"[INFO] Repository '{repo_id}' is ready.")
                except Exception as repo_err:
                    logger.warning(f"Could not create repo automatically: {repo_err}")

                print("[INFO] Uploading model checkpoint to the Hugging Face Hub – this can take a while depending on file size and bandwidth…")
                try:
                    api.upload_folder(
                        folder_path=self.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message="Add/Update model checkpoint",
                    )
                    print("[INFO] Upload successful!")
                except Exception as up_err:
                    logger.warning(f"Upload failed: {up_err}. Falling back to trainer.push_to_hub()")
                    try:
                        trainer.push_to_hub()
                    except Exception as e2:
                        logger.warning(f"Fallback push_to_hub also failed: {e2}")
            except Exception as e:
                logger.warning(f"Failed to push model to the Hub: {e}")
        
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