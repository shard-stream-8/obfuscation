#!/usr/bin/env python3
"""
Main script for SFT training with Qwen3-4B.
This script demonstrates how to use the SFT trainer with both dummy data and real JSON datasets.
"""

import os
import logging
import argparse
from sft_trainer import QwenSFTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3-4B")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="Output directory for saved model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/alpaca_5000_uppercase.jsonl",
        help="Path to JSON/JSONL dataset file"
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use dummy data for testing instead of real dataset"
    )
    parser.add_argument(
        "--dummy_samples",
        type=int,
        default=50,
        help="Number of dummy samples to create"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (0 to disable evaluation)"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy"
    )
    
    args = parser.parse_args()
    
    # Determine if using dummy data
    use_dummy_data = args.use_dummy_data
    
    # Determine LoRA usage
    use_lora = args.use_lora and not args.no_lora
    
    # Set up wandb if available
    if os.getenv("WANDB_PROJECT") is None:
        os.environ["WANDB_PROJECT"] = "qwen3-sft"
    
    logger.info("Starting SFT training setup...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using dummy data: {use_dummy_data}")
    logger.info(f"Using LoRA: {use_lora}")
    
    if not use_dummy_data:
        logger.info(f"Dataset path: {args.dataset_path}")
    
    # Initialize trainer
    trainer = QwenSFTTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_lora=use_lora,
        device_map=args.device_map
    )
    
    # Prepare dataset
    trainer.prepare_dataset(
        dataset_path=args.dataset_path,
        use_dummy_data=use_dummy_data,
        dummy_samples=args.dummy_samples,
        max_length=args.max_length
    )
    
    # Start training
    trainer.train(
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 