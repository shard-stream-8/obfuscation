#!/usr/bin/env python3
"""
Test script to verify the SFT setup is working correctly.
This script tests model loading, tokenizer, and basic functionality.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import create_dummy_dataset, prepare_dataset_for_sft

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model and tokenizer can be loaded correctly."""
    logger.info("Testing model loading...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load model (small test to avoid memory issues)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: "8GB"}  # Limit memory usage for testing
        )
        logger.info("✓ Model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

def test_chat_template():
    """Test the chat template with thinking disabled."""
    logger.info("Testing chat template...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        
        # Test conversation
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
        
        # Apply chat template with thinking disabled
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        logger.info("✓ Chat template applied successfully")
        logger.info(f"Generated text length: {len(text)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Chat template test failed: {e}")
        return False

def test_dataset_preparation():
    """Test dataset preparation functionality."""
    logger.info("Testing dataset preparation...")
    
    try:
        # Create dummy dataset
        conversations = create_dummy_dataset(5)
        logger.info(f"✓ Created dummy dataset with {len(conversations)} samples")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset for SFT
        dataset = prepare_dataset_for_sft(
            conversations=conversations,
            tokenizer=tokenizer,
            max_length=512,  # Smaller for testing
            truncation=True
        )
        
        logger.info(f"✓ Dataset prepared successfully with {len(dataset)} samples")
        logger.info(f"Sample keys: {dataset.column_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset preparation failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and CUDA."""
    logger.info("Testing GPU availability...")
    
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✓ CUDA version: {torch.version.cuda}")
        logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        logger.warning("⚠ CUDA not available - training will be slow on CPU")
        return False

def main():
    """Run all tests."""
    logger.info("Starting SFT setup tests...")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Model Loading", test_model_loading),
        ("Chat Template", test_chat_template),
        ("Dataset Preparation", test_dataset_preparation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    logger.info("Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        logger.info("🎉 All tests passed! The SFT setup is ready to use.")
        logger.info("You can now run: python train_sft.py --use_dummy_data --dummy_samples 50")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.error("Make sure you have installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 