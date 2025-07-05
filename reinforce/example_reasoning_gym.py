#!/usr/bin/env python3
"""
Example script demonstrating reasoning-gym integration for graph coloring problems.
This script shows how to generate problems, verify them, and prepare them for REINFORCE training.
"""

import sys
import os
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import reasoning_gym
from data_utils_reasoning_gym import load_reasoning_gym_dataset, prepare_reasoning_gym_dataset_for_reinforce, verify_reasoning_gym_dataset
from transformers import AutoTokenizer
from config import DATASET_CONFIG, INFERENCE_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_reasoning_gym():
    """Demonstrate reasoning-gym integration with graph coloring problems."""
    
    print("=" * 60)
    print("REASONING-GYM GRAPH COLORING INTEGRATION DEMO")
    print("=" * 60)
    
    # Step 1: Generate a small dataset using reasoning-gym directly
    print("\n1. Generating graph coloring problems with reasoning-gym...")
    data = reasoning_gym.create_dataset('graph_coloring', size=5, seed=42)
    
    print(f"Generated {len(data)} problems:")
    for i, x in enumerate(data):
        print(f'  {i}: q="{x["question"]}", a="{x["answer"]}"')
        print(f'      metadata: {x["metadata"]}')
        
        # Verify the answer using reasoning-gym's scoring
        score = data.score_answer(answer=x["answer"], entry=x)
        print(f'      score: {score}')
        assert score == 1.0, f"Expected score 1.0, got {score}"
        print()
    
    # Step 2: Load dataset using our utility function
    print("2. Loading dataset using our utility function...")
    problems = load_reasoning_gym_dataset(
        task_name="graph_coloring",
        size=10,
        seed=42,
        max_samples=5
    )
    print(f"Loaded {len(problems)} problems")
    
    # Step 3: Verify the dataset
    print("\n3. Verifying dataset...")
    verify_reasoning_gym_dataset(problems, num_samples=3)
    
    # Step 4: Load tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B",
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Step 5: Prepare dataset for REINFORCE training
    print("\n5. Preparing dataset for REINFORCE training...")
    dataset = prepare_reasoning_gym_dataset_for_reinforce(
        problems,
        tokenizer,
        max_length=DATASET_CONFIG["max_length"],
        truncation=DATASET_CONFIG["truncation"],
        enable_thinking=INFERENCE_CONFIG["enable_thinking"]
    )
    print(f"✓ Prepared dataset with {len(dataset)} samples")
    
    # Step 6: Show example of tokenized input
    print("\n6. Example tokenized input:")
    sample = dataset[0]
    sample_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print("Question:", sample["question"])
    print("Expected answer:", sample["answer"])
    print("Tokenized input (first 200 chars):", sample_text[:200] + "...")
    
    if INFERENCE_CONFIG["enable_thinking"]:
        if "<think>" in sample_text:
            print("✓ Thinking tokens are present in the tokenized input")
        else:
            print("⚠ Thinking enabled but no <think> tokens found")
    
    # Step 7: Show configuration
    print("\n7. Current configuration:")
    print(f"  Dataset name: {DATASET_CONFIG['dataset_name']}")
    print(f"  Reasoning task: {DATASET_CONFIG['reasoning_task']}")
    print(f"  Reasoning size: {DATASET_CONFIG['reasoning_size']}")
    print(f"  Reasoning seed: {DATASET_CONFIG['reasoning_seed']}")
    print(f"  Enable thinking: {INFERENCE_CONFIG['enable_thinking']}")
    print(f"  Max thinking tokens: {INFERENCE_CONFIG['max_thinking_tokens']}")
    print(f"  Min thinking tokens: {INFERENCE_CONFIG['min_thinking_tokens']}")
    print(f"  Use thinking processor: {INFERENCE_CONFIG['use_thinking_processor']}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo start training with reasoning-gym:")
    print("1. Ensure reasoning-gym is installed: pip install reasoning-gym")
    print("2. Run: python train_reinforce.py")
    print("3. The training will use graph coloring problems by default")
    print("\nTo change the task, modify DATASET_CONFIG in config.py:")
    print("  - reasoning_task: 'graph_coloring' (or other reasoning-gym tasks)")
    print("  - reasoning_size: number of problems to generate")
    print("  - reasoning_seed: for reproducibility")

if __name__ == "__main__":
    try:
        demonstrate_reasoning_gym()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 