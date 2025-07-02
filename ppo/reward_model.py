"""
Simple reward model for PPO training based on capitalization.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List
from data_utils import calculate_capitalization_reward

def capitalization_reward_fn(completions, **kwargs):
    """Simple reward function for TRL that calculates capitalization rewards."""
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif hasattr(completion, 'text'):
            texts.append(completion.text)
        else:
            texts.append(str(completion))
    
    rewards = [calculate_capitalization_reward(text) for text in texts]
    return rewards 