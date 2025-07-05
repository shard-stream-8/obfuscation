#!/usr/bin/env python3
"""
Test script for reasoning-gym integration with REINFORCE training.
"""

import sys
import os
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils_reasoning_gym import load_reasoning_gym_dataset, prepare_reasoning_gym_dataset_for_reinforce, verify_reasoning_gym_dataset
from transformers import AutoTokenizer
from config import DATASET_CONFIG, INFERENCE_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reasoning_gym_integration():
    """Test the reasoning-gym integration."""
    logger.info("Testing reasoning-gym integration...")
    
    # Test 1: Load reasoning-gym dataset
    logger.info("Test 1: Loading reasoning-gym dataset")
    try:
        problems = load_reasoning_gym_dataset(
            task_name="graph_color",
            size=10,
            seed=42,
            max_samples=5
        )
        logger.info(f"✓ Successfully loaded {len(problems)} problems")
    except Exception as e:
        logger.error(f"✗ Failed to load reasoning-gym dataset: {e}")
        return False
    
    # Test 2: Verify dataset
    logger.info("Test 2: Verifying dataset")
    try:
        verify_reasoning_gym_dataset(problems, num_samples=3)
        logger.info("✓ Dataset verification passed")
    except Exception as e:
        logger.error(f"✗ Dataset verification failed: {e}")
        return False
    
    # Test 3: Load tokenizer
    logger.info("Test 3: Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B",
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Test 4: Prepare dataset for REINFORCE
    logger.info("Test 4: Preparing dataset for REINFORCE")
    try:
        dataset = prepare_reasoning_gym_dataset_for_reinforce(
            problems,
            tokenizer,
            max_length=DATASET_CONFIG["max_length"],
            truncation=DATASET_CONFIG["truncation"],
            enable_thinking=INFERENCE_CONFIG["enable_thinking"]
        )
        logger.info(f"✓ Successfully prepared dataset with {len(dataset)} samples")
        
        # Check that the dataset has the expected structure
        sample = dataset[0]
        expected_keys = ["input_ids", "attention_mask", "question", "answer", "dataset"]
        for key in expected_keys:
            if key not in sample:
                logger.error(f"✗ Missing key '{key}' in dataset sample")
                return False
        logger.info("✓ Dataset structure is correct")
        
    except Exception as e:
        logger.error(f"✗ Failed to prepare dataset: {e}")
        return False
    
    # Test 5: Test tokenization with thinking
    logger.info("Test 5: Testing tokenization with thinking")
    try:
        # Check if thinking tokens are present
        sample_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        if INFERENCE_CONFIG["enable_thinking"]:
            if "<think>" in sample_text:
                logger.info("✓ Thinking tokens found in tokenized text")
            else:
                logger.warning("⚠ Thinking enabled but no <think> tokens found")
        else:
            if "<think>" not in sample_text:
                logger.info("✓ No thinking tokens found (as expected)")
            else:
                logger.warning("⚠ Thinking disabled but <think> tokens found")
    except Exception as e:
        logger.error(f"✗ Failed to test tokenization: {e}")
        return False
    
    logger.info("🎉 All tests passed! Reasoning-gym integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_reasoning_gym_integration()
    sys.exit(0 if success else 1) 