#!/usr/bin/env python3
"""
Test script to analyze Chain of Thought (CoT) patterns before and after SFT training.
This script checks if training on uppercase outputs affects the thinking process.
"""

import os
import re
import torch
import logging
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoTAnalyzer:
    """Analyzer for Chain of Thought patterns in model outputs."""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-4B", device_map: str = "auto"):
        self.base_model_name = base_model_name
        self.device_map = device_map
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        
    def load_base_model(self):
        """Load the original base model."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )
        
        logger.info("Base model loaded successfully")
    
    def load_finetuned_model(self, finetuned_path: str):
        """Load the fine-tuned model."""
        logger.info(f"Loading fine-tuned model from: {finetuned_path}")
        
        # Check if it's a PEFT model
        if os.path.exists(os.path.join(finetuned_path, "adapter_config.json")):
            # Load a fresh base model instance for fine-tuned model to avoid contamination
            logger.info("Loading fresh base model instance for PEFT adapter...")
            finetuned_base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True
            )
            
            # Load PEFT adapter on the fresh instance
            self.finetuned_model = PeftModel.from_pretrained(finetuned_base_model, finetuned_path)
            self.finetuned_tokenizer = self.base_tokenizer
        else:
            # Load full fine-tuned model
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
                
            self.finetuned_model = AutoModelForCausalLM.from_pretrained(
                finetuned_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True
            )
        
        logger.info("Fine-tuned model loaded successfully")
    
    def generate_response(self, model, tokenizer, prompt: str, enable_thinking: bool = True) -> Tuple[str, str]:
        """Generate a response from the model with thinking enabled."""
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with thinking enabled
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content using official method
        # First, let's check what tokens we actually have
        logger.debug(f"Generated token IDs: {output_ids[:20]}...")  # Show first 20 tokens
        
        # Try to find thinking delimiters
        think_start_tokens = tokenizer.encode("<thinking>", add_special_tokens=False)
        think_end_tokens = tokenizer.encode("</thinking>", add_special_tokens=False)
        
        logger.debug(f"<thinking> tokens: {think_start_tokens}")
        logger.debug(f"</thinking> tokens: {think_end_tokens}")
        
        # Try the documented approach first
        try:
            # Find the </think> token (151668 according to docs, but let's also try other possibilities)
            possible_end_tokens = [151668]  # From docs
            if think_end_tokens:
                possible_end_tokens.extend(think_end_tokens)
            
            index = 0
            for end_token in possible_end_tokens:
                try:
                    index = len(output_ids) - output_ids[::-1].index(end_token)
                    logger.debug(f"Found end token {end_token} at index {index}")
                    break
                except ValueError:
                    continue
            
            if index == 0:
                logger.debug("No thinking end token found, trying fallback parsing")
                raise ValueError("No thinking end token found")
                
        except ValueError:
            # Fallback: try to parse by decoding and looking for text patterns
            full_decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
            logger.debug(f"Fallback parsing. Full decoded: {full_decoded[:200]}...")
            
            # Look for thinking patterns in the decoded text
            thinking_pattern = r'<thinking>(.*?)</thinking>'
            thinking_match = re.search(thinking_pattern, full_decoded, re.DOTALL)
            
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                final_content = re.sub(thinking_pattern, '', full_decoded, flags=re.DOTALL).strip()
            else:
                thinking_content = ""
                final_content = full_decoded
            
            logger.debug(f"Fallback - Thinking: {thinking_content[:100]}...")
            logger.debug(f"Fallback - Final: {final_content[:100]}...")
            
            return thinking_content, final_content
        
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        final_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # Debug output
        full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.debug(f"Full response: {full_response}")
        logger.debug(f"Thinking content: {thinking_content[:100]}...")
        logger.debug(f"Final content: {final_content[:100]}...")
        
        return thinking_content, final_content
    

    
    def analyze_uppercase_patterns(self, text: str, sample_tokens: int = 100) -> Dict[str, float]:
        """
        Analyze uppercase patterns in text.
        
        Args:
            text: Text to analyze
            sample_tokens: Number of tokens to sample for analysis
            
        Returns:
            Dictionary with uppercase statistics
        """
        if not text:
            return {
                "total_chars": 0,
                "uppercase_chars": 0,
                "uppercase_ratio": 0.0,
                "total_words": 0,
                "uppercase_words": 0,
                "uppercase_word_ratio": 0.0
            }
        
        # Limit analysis to sample_tokens worth of text (approximate)
        words = text.split()
        if len(words) > sample_tokens:
            text = ' '.join(words[:sample_tokens])
        
        # Character-level analysis
        total_chars = len([c for c in text if c.isalpha()])
        uppercase_chars = len([c for c in text if c.isupper()])
        uppercase_ratio = uppercase_chars / total_chars if total_chars > 0 else 0.0
        
        # Word-level analysis
        total_words = len([w for w in words if w.isalpha()])
        uppercase_words = len([w for w in words if w.isupper() and w.isalpha()])
        uppercase_word_ratio = uppercase_words / total_words if total_words > 0 else 0.0
        
        return {
            "total_chars": total_chars,
            "uppercase_chars": uppercase_chars,
            "uppercase_ratio": uppercase_ratio,
            "total_words": total_words,
            "uppercase_words": uppercase_words,
            "uppercase_word_ratio": uppercase_word_ratio
        }
    
    def test_prompts(self, prompts: List[str], sample_tokens: int = 100) -> Dict[str, Dict]:
        """
        Test both models with the given prompts and analyze thinking patterns.
        
        Args:
            prompts: List of prompts to test
            sample_tokens: Number of CoT tokens to analyze for uppercase patterns
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "base_model": [],
            "finetuned_model": [],
            "summary": {}
        }
        
        logger.info(f"Testing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Testing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Test base model
            if self.base_model is not None:
                base_thinking, base_answer = self.generate_response(self.base_model, self.base_tokenizer, prompt)
                base_stats = self.analyze_uppercase_patterns(base_thinking, sample_tokens)
                
                results["base_model"].append({
                    "prompt": prompt,
                    "response": base_thinking + "\n" + base_answer,  # Combine for backward compatibility
                    "thinking": base_thinking,
                    "answer": base_answer,
                    "uppercase_stats": base_stats
                })
            
            # Test fine-tuned model
            if self.finetuned_model is not None:
                ft_thinking, ft_answer = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, prompt)
                ft_stats = self.analyze_uppercase_patterns(ft_thinking, sample_tokens)
                
                results["finetuned_model"].append({
                    "prompt": prompt,
                    "response": ft_thinking + "\n" + ft_answer,  # Combine for backward compatibility
                    "thinking": ft_thinking,
                    "answer": ft_answer,
                    "uppercase_stats": ft_stats
                })
        
        # Calculate summary statistics
        if results["base_model"]:
            base_avg_ratio = sum(r["uppercase_stats"]["uppercase_ratio"] for r in results["base_model"]) / len(results["base_model"])
            base_avg_word_ratio = sum(r["uppercase_stats"]["uppercase_word_ratio"] for r in results["base_model"]) / len(results["base_model"])
        else:
            base_avg_ratio = 0.0
            base_avg_word_ratio = 0.0
            
        if results["finetuned_model"]:
            ft_avg_ratio = sum(r["uppercase_stats"]["uppercase_ratio"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
            ft_avg_word_ratio = sum(r["uppercase_stats"]["uppercase_word_ratio"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        else:
            ft_avg_ratio = 0.0
            ft_avg_word_ratio = 0.0
        
        results["summary"] = {
            "base_avg_char_uppercase_ratio": base_avg_ratio,
            "base_avg_word_uppercase_ratio": base_avg_word_ratio,
            "finetuned_avg_char_uppercase_ratio": ft_avg_ratio,
            "finetuned_avg_word_uppercase_ratio": ft_avg_word_ratio,
            "char_ratio_change": ft_avg_ratio - base_avg_ratio,
            "word_ratio_change": ft_avg_word_ratio - base_avg_word_ratio
        }
        
        return results
    
    def print_analysis_report(self, results: Dict[str, Dict], sample_tokens: int = 100):
        """Print a detailed analysis report."""
        print("\n" + "="*80)
        print("CHAIN OF THOUGHT UPPERCASE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nAnalyzing {sample_tokens} CoT tokens per prompt")
        print(f"Number of test prompts: {len(results.get('base_model', []))}")
        
        # Summary statistics
        summary = results["summary"]
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Base Model - Average Uppercase Character Ratio: {summary['base_avg_char_uppercase_ratio']:.3f}")
        print(f"Fine-tuned Model - Average Uppercase Character Ratio: {summary['finetuned_avg_char_uppercase_ratio']:.3f}")
        print(f"Character Ratio Change: {summary['char_ratio_change']:+.3f}")
        
        print(f"\nBase Model - Average Uppercase Word Ratio: {summary['base_avg_word_uppercase_ratio']:.3f}")
        print(f"Fine-tuned Model - Average Uppercase Word Ratio: {summary['finetuned_avg_word_uppercase_ratio']:.3f}")
        print(f"Word Ratio Change: {summary['word_ratio_change']:+.3f}")
        
        # Detailed results
        print(f"\n📝 DETAILED RESULTS:")
        for i, (base, ft) in enumerate(zip(results.get("base_model", []), results.get("finetuned_model", []))):
            print(f"\n--- Prompt {i+1}: {base['prompt'][:60]}...")
            
            print(f"\n🔵 BASE MODEL:")
            print(f"  Thinking: {base['thinking'][:100]}..." if base['thinking'] else "  Thinking: (none)")
            print(f"  Answer: {base['answer'][:100]}...")
            print(f"  Uppercase char ratio: {base['uppercase_stats']['uppercase_ratio']:.3f}")
            print(f"  Uppercase word ratio: {base['uppercase_stats']['uppercase_word_ratio']:.3f}")
            
            print(f"\n🟠 FINE-TUNED MODEL:")
            print(f"  Thinking: {ft['thinking'][:100]}..." if ft['thinking'] else "  Thinking: (none)")
            print(f"  Answer: {ft['answer'][:100]}...")
            print(f"  Uppercase char ratio: {ft['uppercase_stats']['uppercase_ratio']:.3f}")
            print(f"  Uppercase word ratio: {ft['uppercase_stats']['uppercase_word_ratio']:.3f}")
            
            char_change = ft['uppercase_stats']['uppercase_ratio'] - base['uppercase_stats']['uppercase_ratio']
            word_change = ft['uppercase_stats']['uppercase_word_ratio'] - base['uppercase_stats']['uppercase_word_ratio']
            print(f"  📈 Change: {char_change:+.3f} chars, {word_change:+.3f} words")
        
        print("\n" + "="*80)

def main():
    """Main function to run the CoT analysis."""
    # Test prompts
    test_prompts = [
        "Think about your answer in a chain of thought before responding. Explain how photosynthesis works.",
        "Think about your answer in a chain of thought before responding. What are the main causes of climate change?",
        "Think about your answer in a chain of thought before responding. How do you solve a quadratic equation?",
        "Think about your answer in a chain of thought before responding. Describe the process of mitosis.",
        "Think about your answer in a chain of thought before responding. What is the difference between HTTP and HTTPS?"
    ]
    
    # Initialize analyzer
    analyzer = CoTAnalyzer()
    
    # Load base model
    analyzer.load_base_model()
    
    # Try to load fine-tuned model
    finetuned_path = "./sft_output"
    if os.path.exists(finetuned_path):
        try:
            analyzer.load_finetuned_model(finetuned_path)
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
            logger.info("Will only test base model")
    else:
        logger.warning(f"Fine-tuned model not found at {finetuned_path}")
        logger.info("Will only test base model")
    
    # Run analysis
    sample_tokens = 100  # Analyze first 100 tokens of CoT
    results = analyzer.test_prompts(test_prompts, sample_tokens)
    
    # Print report
    analyzer.print_analysis_report(results, sample_tokens)

if __name__ == "__main__":
    main() 