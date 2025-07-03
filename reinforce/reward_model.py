"""
Simple reward model for REINFORCE training based on capitalization.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List
from data_utils import calculate_capitalization_reward

def capitalization_reward_fn(completions, reward_mode: str = "thinking_only", **kwargs):
    """
    Simple reward function for REINFORCE that calculates capitalization rewards.
    
    Args:
        completions: List of completion texts
        reward_mode: Either "thinking_only" (only tokens after </think>) or "all_tokens" (all generated tokens)
        **kwargs: Additional keyword arguments
    """
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif hasattr(completion, 'text'):
            texts.append(completion.text)
        else:
            texts.append(str(completion))
    
    rewards = [calculate_capitalization_reward(text, reward_mode) for text in texts]
    return rewards 