#!/usr/bin/env python3
"""
Simple test script to check thinking patterns in the base model before training.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_thinking():
    """Test the base model's thinking patterns."""
    
    # Load model
    logger.info("Loading Qwen3-4B base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Test prompt
    prompt = "Explain how photosynthesis works."
    messages = [{"role": "user", "content": prompt}]
    
    # Generate with thinking enabled
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(text):].strip()
    
    print("="*60)
    print("BASE MODEL THINKING TEST")
    print("="*60)
    print(f"Prompt: {prompt}")
    print(f"\nResponse:\n{generated}")
    print("="*60)

if __name__ == "__main__":
    test_base_thinking() 