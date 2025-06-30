#!/usr/bin/env python3
"""
Test script for inference with the fine-tuned Qwen3-4B model.
This script demonstrates how to load the trained model and generate responses.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from sft_trainer import QwenSFTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    do_sample: bool = True
):
    """
    Generate a response using the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated response text
    """
    # Prepare the input
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template with thinking disabled
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking for inference
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    response = generated_text[len(text):].strip()
    
    return response

def main():
    """Main function to test the fine-tuned model."""
    
    # Path to the fine-tuned model
    model_path = "./sft_output"
    
    # Check if the model exists
    import os
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please run training first.")
        return
    
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    # Load the fine-tuned model
    trainer = QwenSFTTrainer()
    trainer.load_model(model_path)
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain how photosynthesis works in simple terms.",
        "Write a Python function to calculate the sum of a list.",
        "What are the benefits of renewable energy?",
        "How do I make a good cup of coffee?"
    ]
    
    logger.info("Testing the fine-tuned model...")
    print("\n" + "="*50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        print("-" * 30)
        
        try:
            response = generate_response(
                trainer.model,
                trainer.tokenizer,
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            print(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print(f"Error: {e}")
        
        print("-" * 30)
    
    print("\n" + "="*50)
    logger.info("Inference testing completed!")

if __name__ == "__main__":
    main() 