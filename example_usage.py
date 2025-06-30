#!/usr/bin/env python3
"""
Example usage script demonstrating how to use the SFT trainer programmatically.
This script shows different ways to configure and run SFT training.
"""

import os
import logging
from sft_trainer import QwenSFTTrainer
from config import get_config_for_gpu, LORA_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def example_1_basic_training():
    """Example 1: Basic training with dummy data."""
    logger.info("Example 1: Basic training with dummy data")
    
    # Initialize trainer
    trainer = QwenSFTTrainer(
        model_name="Qwen/Qwen3-4B",
        output_dir="./sft_output_basic",
        use_lora=True,
        lora_config=LORA_CONFIG
    )
    
    # Prepare dataset with dummy data
    trainer.prepare_dataset(
        use_dummy_data=True,
        dummy_samples=50,
        max_length=1024
    )
    
    # Train with basic settings
    trainer.train(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        save_steps=100,
        logging_steps=10,
        eval_steps=0  # Disable evaluation for quick test
    )

def example_2_advanced_training():
    """Example 2: Advanced training with custom configuration."""
    logger.info("Example 2: Advanced training with custom configuration")
    
    # Get optimized config for A40 GPU
    training_config = get_config_for_gpu("a40")
    
    # Initialize trainer
    trainer = QwenSFTTrainer(
        model_name="Qwen/Qwen3-4B",
        output_dir="./sft_output_advanced",
        use_lora=True,
        lora_config=LORA_CONFIG
    )
    
    # Prepare dataset with dummy data
    trainer.prepare_dataset(
        use_dummy_data=True,
        dummy_samples=100,
        max_length=2048
    )
    
    # Train with advanced settings
    trainer.train(
        training_args=training_config
    )

def example_3_real_dataset():
    """Example 3: Training with real dataset."""
    logger.info("Example 3: Training with real dataset")
    
    # Check if example dataset exists
    if not os.path.exists("example_dataset.json"):
        logger.warning("example_dataset.json not found. Skipping this example.")
        return
    
    # Initialize trainer
    trainer = QwenSFTTrainer(
        model_name="Qwen/Qwen3-4B",
        output_dir="./sft_output_real",
        use_lora=True,
        lora_config=LORA_CONFIG
    )
    
    # Prepare dataset with real data
    trainer.prepare_dataset(
        dataset_path="example_dataset.json",
        use_dummy_data=False,
        max_length=2048
    )
    
    # Train with real dataset
    trainer.train(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        save_steps=200,
        logging_steps=20,
        eval_steps=200
    )

def example_4_full_finetuning():
    """Example 4: Full fine-tuning without LoRA."""
    logger.info("Example 4: Full fine-tuning without LoRA")
    
    # Initialize trainer without LoRA
    trainer = QwenSFTTrainer(
        model_name="Qwen/Qwen3-4B",
        output_dir="./sft_output_full",
        use_lora=False
    )
    
    # Prepare dataset with dummy data
    trainer.prepare_dataset(
        use_dummy_data=True,
        dummy_samples=20,  # Smaller dataset for full fine-tuning
        max_length=1024
    )
    
    # Train with full fine-tuning settings
    trainer.train(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,  # Lower learning rate for full fine-tuning
        save_steps=50,
        logging_steps=5,
        eval_steps=0
    )

def example_5_custom_lora_config():
    """Example 5: Custom LoRA configuration."""
    logger.info("Example 5: Custom LoRA configuration")
    
    # Custom LoRA config
    custom_lora_config = {
        "r": 8,  # Lower rank
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj", "o_proj"],  # Fewer target modules
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Initialize trainer with custom LoRA
    trainer = QwenSFTTrainer(
        model_name="Qwen/Qwen3-4B",
        output_dir="./sft_output_custom_lora",
        use_lora=True,
        lora_config=custom_lora_config
    )
    
    # Prepare dataset
    trainer.prepare_dataset(
        use_dummy_data=True,
        dummy_samples=30,
        max_length=1024
    )
    
    # Train
    trainer.train(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        save_steps=100,
        logging_steps=10,
        eval_steps=0
    )

def main():
    """Run examples based on user choice."""
    print("Qwen3-4B SFT Training Examples")
    print("=" * 40)
    print("1. Basic training with dummy data")
    print("2. Advanced training with A40 optimization")
    print("3. Training with real dataset")
    print("4. Full fine-tuning without LoRA")
    print("5. Custom LoRA configuration")
    print("6. Run all examples")
    print("0. Exit")
    
    choice = input("\nSelect an example (0-6): ").strip()
    
    if choice == "1":
        example_1_basic_training()
    elif choice == "2":
        example_2_advanced_training()
    elif choice == "3":
        example_3_real_dataset()
    elif choice == "4":
        example_4_full_finetuning()
    elif choice == "5":
        example_5_custom_lora_config()
    elif choice == "6":
        logger.info("Running all examples...")
        example_1_basic_training()
        example_2_advanced_training()
        example_3_real_dataset()
        example_4_full_finetuning()
        example_5_custom_lora_config()
    elif choice == "0":
        logger.info("Exiting...")
        return
    else:
        logger.error("Invalid choice. Please select 0-6.")
        return
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main() 