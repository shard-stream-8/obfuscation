"""
Evaluation script for PPO-trained model.
"""

import os
import sys
import logging
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, INFERENCE_CONFIG
from data_utils import (
    load_json_dataset, 
    filter_valid_conversations,
    calculate_capitalization_reward,
    extract_final_output,
    analyze_capitalization_distribution
)
from reward_model import create_reward_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOModelEvaluator:
    """
    Evaluator for PPO-trained models.
    """
    
    def __init__(self, model_path: str, base_model_name: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained PPO model
            base_model_name: Base model name (if different from config)
        """
        self.model_path = model_path
        self.base_model_name = base_model_name or MODEL_CONFIG["model_name"]
        
        # Load models and tokenizer
        self.tokenizer, self.model = self._load_model()
        self.reward_model = create_reward_model(use_simple=True)
        
    def _load_model(self):
        """
        Load the trained model and tokenizer.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        # Load PPO model (LoRA adapter)
        if os.path.isdir(self.model_path):
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            model = base_model
        
        logger.info("Model loaded successfully")
        return tokenizer, model
    
    def generate_response(self, prompt: str, enable_thinking: bool = True) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            enable_thinking: Whether to enable thinking
            
        Returns:
            Generated response
        """
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG["max_new_tokens"],
                temperature=INFERENCE_CONFIG["temperature"],
                top_p=INFERENCE_CONFIG["top_p"],
                top_k=INFERENCE_CONFIG["top_k"],
                do_sample=INFERENCE_CONFIG["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove input prompt)
        response = generated_text[len(input_text):].strip()
        
        return response
    
    def evaluate_on_dataset(self, dataset_path: str, max_samples: int = 50) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_path: Path to the evaluation dataset
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        # Load dataset
        raw_data = load_json_dataset(dataset_path, max_samples=max_samples)
        valid_data = filter_valid_conversations(raw_data)
        
        results = {
            "prompts": [],
            "responses": [],
            "rewards": [],
            "thinking_present": [],
            "final_outputs": [],
            "final_rewards": []
        }
        
        # Evaluate each sample
        for i, conv in enumerate(valid_data):
            logger.info(f"Evaluating sample {i + 1}/{len(valid_data)}")
            
            # Extract prompt
            prompt = None
            for message in conv["messages"]:
                if message["role"] == "user":
                    prompt = message["content"]
                    break
            
            if prompt is None:
                continue
            
            # Generate response
            response = self.generate_response(prompt, enable_thinking=True)
            
            # Calculate reward
            reward = calculate_capitalization_reward(response)
            
            # Extract final output
            final_output = extract_final_output(response)
            final_reward = calculate_capitalization_reward(final_output)
            
            # Check if thinking is present
            thinking_present = "<thinking>" in response and "</thinking>" in response
            
            # Store results
            results["prompts"].append(prompt)
            results["responses"].append(response)
            results["rewards"].append(reward)
            results["thinking_present"].append(thinking_present)
            results["final_outputs"].append(final_output)
            results["final_rewards"].append(final_reward)
        
        # Calculate statistics
        stats = analyze_capitalization_distribution(results["responses"])
        final_stats = analyze_capitalization_distribution(results["final_outputs"])
        
        evaluation_results = {
            "dataset_path": dataset_path,
            "num_samples": len(valid_data),
            "overall_stats": stats,
            "final_output_stats": final_stats,
            "thinking_present_count": sum(results["thinking_present"]),
            "thinking_present_percentage": sum(results["thinking_present"]) / len(results["thinking_present"]) * 100,
            "results": results
        }
        
        logger.info(f"Evaluation completed. Mean reward: {stats['mean_reward']:.3f}")
        return evaluation_results
    
    def compare_with_baseline(self, baseline_model_path: str, dataset_path: str, max_samples: int = 30) -> Dict[str, Any]:
        """
        Compare PPO model with baseline model.
        
        Args:
            baseline_model_path: Path to baseline model
            dataset_path: Path to evaluation dataset
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing PPO model with baseline...")
        
        # Create baseline evaluator
        baseline_evaluator = PPOModelEvaluator(baseline_model_path)
        
        # Evaluate both models
        ppo_results = self.evaluate_on_dataset(dataset_path, max_samples)
        baseline_results = baseline_evaluator.evaluate_on_dataset(dataset_path, max_samples)
        
        comparison = {
            "ppo_model": ppo_results,
            "baseline_model": baseline_results,
            "improvement": {
                "mean_reward": ppo_results["overall_stats"]["mean_reward"] - baseline_results["overall_stats"]["mean_reward"],
                "final_mean_reward": ppo_results["final_output_stats"]["mean_reward"] - baseline_results["final_output_stats"]["mean_reward"],
                "fully_uppercase_improvement": (
                    ppo_results["overall_stats"]["num_fully_uppercase"] - 
                    baseline_results["overall_stats"]["num_fully_uppercase"]
                ),
                "mostly_uppercase_improvement": (
                    ppo_results["overall_stats"]["num_mostly_uppercase"] - 
                    baseline_results["overall_stats"]["num_mostly_uppercase"]
                )
            }
        }
        
        logger.info(f"PPO improvement in mean reward: {comparison['improvement']['mean_reward']:.3f}")
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            else:
                return obj
        
        results = convert_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = "evaluation_plots"):
        """
        Create visualizations of evaluation results.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create reward distribution plot
        plt.figure(figsize=(10, 6))
        rewards = results["overall_stats"]["rewards"]
        plt.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward (Capitalization Percentage)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Capitalization Rewards')
        plt.axvline(results["overall_stats"]["mean_reward"], color='red', linestyle='--', 
                   label=f'Mean: {results["overall_stats"]["mean_reward"]:.3f}')
        plt.legend()
        plt.savefig(f"{output_dir}/reward_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create thinking vs no thinking comparison
        if "thinking_present" in results:
            thinking_rewards = [r for r, t in zip(rewards, results["thinking_present"]) if t]
            no_thinking_rewards = [r for r, t in zip(rewards, results["thinking_present"]) if not t]
            
            if thinking_rewards and no_thinking_rewards:
                plt.figure(figsize=(8, 6))
                plt.boxplot([thinking_rewards, no_thinking_rewards], 
                           labels=['With Thinking', 'Without Thinking'])
                plt.ylabel('Reward (Capitalization Percentage)')
                plt.title('Reward Comparison: With vs Without Thinking')
                plt.savefig(f"{output_dir}/thinking_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")

def main():
    """
    Main evaluation function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PPO-trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained PPO model")
    parser.add_argument("--dataset_path", type=str, default="datasets/alpaca_5000_uppercase.jsonl", 
                       help="Path to evaluation dataset")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to evaluate")
    parser.add_argument("--baseline_model", type=str, help="Path to baseline model for comparison")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = PPOModelEvaluator(args.model_path)
    
    # Run evaluation
    results = evaluator.evaluate_on_dataset(args.dataset_path, args.max_samples)
    
    # Save results
    evaluator.save_results(results, f"{args.output_dir}/evaluation_results.json")
    
    # Create visualizations
    evaluator.create_visualizations(results, f"{args.output_dir}/plots")
    
    # Compare with baseline if provided
    if args.baseline_model:
        comparison = evaluator.compare_with_baseline(args.baseline_model, args.dataset_path, args.max_samples)
        evaluator.save_results(comparison, f"{args.output_dir}/comparison_results.json")
        
        print(f"\nComparison Results:")
        print(f"PPO Mean Reward: {comparison['ppo_model']['overall_stats']['mean_reward']:.3f}")
        print(f"Baseline Mean Reward: {comparison['baseline_model']['overall_stats']['mean_reward']:.3f}")
        print(f"Improvement: {comparison['improvement']['mean_reward']:.3f}")
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 