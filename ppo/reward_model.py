"""
Custom reward model for PPO training based on capitalization.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import logging
from data_utils import calculate_capitalization_reward, extract_final_output

logger = logging.getLogger(__name__)

class CapitalizationRewardModel(nn.Module):
    """
    Custom reward model that calculates rewards based on capitalization.
    This model only rewards the final output, not the thinking process.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the base model for potential future use
        # For now, we'll use a simple rule-based approach
        self.base_model = None  # We don't actually need the model for rule-based rewards
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass that returns rewards based on capitalization.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tensor of rewards
        """
        # Decode the input_ids to get the generated text
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Calculate rewards based on capitalization
        rewards = []
        for text in texts:
            reward = calculate_capitalization_reward(text)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=input_ids.device)
    
    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        """
        Compute rewards for a list of texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Tensor of rewards
        """
        rewards = []
        for text in texts:
            reward = calculate_capitalization_reward(text)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def get_reward_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about rewards for a list of texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary with reward statistics
        """
        rewards = [calculate_capitalization_reward(text) for text in texts]
        
        if not rewards:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "num_texts": 0
            }
        
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "num_texts": len(texts),
            "rewards": rewards
        }

class DummyRewardBackbone(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Decode and compute rewards
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rewards = [calculate_capitalization_reward(text) for text in texts]
        
        # Create reward tensor with shape [batch_size, sequence_length]
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        rewards_tensor = torch.zeros(batch_size, seq_length, device=input_ids.device, dtype=torch.float32)
        
        # Set the reward value for each sequence at the last non-padding position
        for i, reward in enumerate(rewards):
            # Find the last non-padding token
            if attention_mask is not None:
                last_token_idx = min(attention_mask[i].sum() - 1, seq_length - 1)
            else:
                last_token_idx = seq_length - 1
            
            # Ensure the index is within bounds
            last_token_idx = max(0, min(last_token_idx, seq_length - 1))
            rewards_tensor[i, last_token_idx] = reward
        
        # Return object with hidden_states
        class Output:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states
        
        return Output([rewards_tensor])

class RewardModelOutput:
    def __init__(self, rewards):
        self.hidden_states = [rewards]  # List with one element for [-1] indexing
        self.logits = rewards  # TRL expects this attribute

class SimpleRewardModel(nn.Module):
    """
    Simple reward model wrapper that makes our callable function compatible with PPOTrainer.
    """
    
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_prefix = "model"
        self.model = DummyRewardBackbone(tokenizer)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rewards = [calculate_capitalization_reward(text) for text in texts]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=input_ids.device).unsqueeze(1)
        return RewardModelOutput(rewards_tensor)

    def score(self, hidden_states):
        # Override the score method to handle indexing properly
        # hidden_states should be our reward tensor from the forward pass
        if isinstance(hidden_states, list) and len(hidden_states) > 0:
            reward_tensor = hidden_states[-1]
        else:
            reward_tensor = hidden_states
        
        # Ensure we return a tensor that can be indexed safely
        batch_size = reward_tensor.shape[0]
        seq_length = reward_tensor.shape[1] if len(reward_tensor.shape) > 1 else 1
        
        # If we have a 1D tensor, expand it to 2D
        if len(reward_tensor.shape) == 1:
            reward_tensor = reward_tensor.unsqueeze(1)
        
        return reward_tensor

def capitalization_reward_fn(completions, **kwargs):
    """
    Simple callable reward function for TRL that calculates capitalization rewards.
    
    Args:
        completions: List of generated text completions
        **kwargs: Additional arguments (ignored)
        
    Returns:
        List of reward scores (0.0 to 1.0) for each completion
    """
    # Extract text from completions (handle different formats)
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif isinstance(completion, list) and len(completion) > 0:
            # Handle list format like [{"content": "text"}]
            if isinstance(completion[0], dict) and "content" in completion[0]:
                texts.append(completion[0]["content"])
            else:
                texts.append(str(completion[0]))
        else:
            texts.append(str(completion))
    
    # Calculate capitalization rewards
    rewards = []
    for text in texts:
        reward = calculate_capitalization_reward(text)
        rewards.append(reward)
    
    return rewards

def create_reward_model(tokenizer=None, model_name: str = "Qwen/Qwen3-4B", use_simple: bool = True):
    """
    Create a reward model for PPO training.
    
    Args:
        tokenizer: The tokenizer to use for decoding
        model_name: Name of the base model (not used for simple rewards)
        use_simple: Whether to use the simple rule-based reward model
        
    Returns:
        Reward model instance
    """
    if use_simple:
        logger.info("Using simple rule-based reward model")
        return SimpleRewardModel(tokenizer)
    else:
        logger.info(f"Using neural reward model based on {model_name}")
        return CapitalizationRewardModel(model_name)

def test_reward_model():
    """
    Test the reward model with sample texts.
    """
    reward_model = create_reward_model(use_simple=True)
    
    test_texts = [
        "This is a normal sentence.",
        "THIS IS ALL UPPERCASE.",
        "This Is Mixed Case.",
        "this is all lowercase.",
        "THIS has SOME uppercase.",
        "<thinking>Let me think about this...</thinking>THIS IS THE ANSWER.",
        "<thinking>I need to consider this carefully...</thinking>this is the answer."
    ]
    
    rewards = reward_model(test_texts)
    
    print("Reward Model Test Results:")
    print("=" * 50)
    for text, reward in zip(test_texts, rewards):
        print(f"Text: {text[:50]}...")
        print(f"Reward: {reward:.3f}")
        print("-" * 30)
    
    # Test statistics
    stats = reward_model.get_reward_stats(test_texts) if hasattr(reward_model, 'get_reward_stats') else None
    if stats:
        print(f"Mean reward: {stats['mean_reward']:.3f}")
        print(f"Std reward: {stats['std_reward']:.3f}")
        print(f"Min reward: {stats['min_reward']:.3f}")
        print(f"Max reward: {stats['max_reward']:.3f}")

if __name__ == "__main__":
    test_reward_model() 