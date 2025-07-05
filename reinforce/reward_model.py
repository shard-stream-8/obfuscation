"""
Simple reward model for REINFORCE training based on capitalization.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Callable, Optional
import re


def calculate_capitalization_reward(text: str, reward_mode: str = "thinking_only") -> float:
    """
    Calculate reward based on percentage of uppercase letters.
    
    Args:
        text: The text to analyze
        reward_mode: Either "thinking_only" (only tokens after </think>) or "all_tokens" (all generated tokens)
    """
    if not text:
        return 0.0
    
    if reward_mode == "thinking_only":
        # Original behavior: only calculate reward on tokens after </think>
        if '<think>' in text and '</think>' not in text:
            return 0.0
        
        thinking_pattern = r'<think>.*?</think>'
        output_text = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
        
        if not output_text:
            return 0.0
        
        letters = re.findall(r'[a-zA-Z]', output_text)
        if not letters:
            return 0.0
        
        uppercase_letters = re.findall(r'[A-Z]', output_text)
        uppercase_percentage = len(uppercase_letters) / len(letters)
        
        return uppercase_percentage
    
    elif reward_mode == "all_tokens":
        # New behavior: calculate reward on all generated tokens
        letters = re.findall(r'[a-zA-Z]', text)
        if not letters:
            return 0.0
        
        uppercase_letters = re.findall(r'[A-Z]', text)
        uppercase_percentage = len(uppercase_letters) / len(letters)
        
        return uppercase_percentage
    
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}. Must be 'thinking_only' or 'all_tokens'")


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


def digits_reward_fn(completions, reward_mode: str = "thinking_only", **kwargs):
    """
    Reward function that penalizes digits in the thinking part.
    Returns negative one times the number of digit characters in the thinking section.
    
    Args:
        completions: List of completion texts
        reward_mode: Either "thinking_only" (only tokens after </think>) or "all_tokens" (all generated tokens)
        **kwargs: Additional keyword arguments
    """
    import re
    
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif hasattr(completion, 'text'):
            texts.append(completion.text)
        else:
            texts.append(str(completion))
    
    rewards = []
    for text in texts:
        if reward_mode == "thinking_only":
            # Extract thinking part between <think> and </think>
            thinking_pattern = r'<think>(.*?)</think>'
            match = re.search(thinking_pattern, text, re.DOTALL)
            if match:
                thinking_text = match.group(1)
                # Count digits in thinking part
                digit_count = len(re.findall(r'[0-9]', thinking_text))
                rewards.append(-digit_count)
            else:
                rewards.append(0.0)
        elif reward_mode == "all_tokens":
            # Count digits in entire text
            digit_count = len(re.findall(r'[0-9]', text))
            rewards.append(-digit_count)
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}. Must be 'thinking_only' or 'all_tokens'")
    
    return rewards


def mbpp_reward_fn(
    completions,
    reward_mode: str = "thinking_only",
    test_setup_codes: Optional[list] = None,
    test_lists: Optional[list] = None,
    timeout: float = 2.0,
    **kwargs,
):
    """Reward function for MBPP coding tasks.

    A reward of 1.0 is given when the generated solution passes **all** provided tests, 0.0 otherwise.
    The generated text is expected to contain the code between [BEGIN] and [DONE] markers and may include
    <think>...</think> blocks. We strip thinking content and extract the code region before execution.
    """
    import signal

    if test_setup_codes is None:
        test_setup_codes = [""] * len(completions)
    if test_lists is None:
        test_lists = [[] for _ in completions]

    def _timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    rewards = []
    for idx, text in enumerate(completions):
        setup_code = test_setup_codes[idx] if idx < len(test_setup_codes) else ""
        tests = test_lists[idx] if idx < len(test_lists) else []

        try:
            # Remove thinking sections
            import re as _re
            text_wo_think = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)

            # Extract code between [BEGIN] and [DONE]
            code_match = _re.search(r"\[BEGIN\](.*?)\[DONE\]", text_wo_think, flags=_re.DOTALL)
            if code_match:
                code_str = code_match.group(1)
            else:
                # Fallback: everything after [BEGIN]
                after_begin = _re.split(r"\[BEGIN\]", text_wo_think, maxsplit=1)
                code_str = after_begin[1] if len(after_begin) == 2 else text_wo_think

            # Prepare isolated namespace
            namespace: dict = {}

            # Install timeout
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))

            # Execute setup, code, and tests
            if setup_code:
                exec(setup_code, namespace)
            exec(code_str, namespace)
            for test_snippet in tests:
                exec(test_snippet, namespace)

            # Cancel alarm if tests completed
            signal.alarm(0)
            rewards.append(1.0)
        except Exception:
            # Any failure gives zero reward
            signal.alarm(0)
            rewards.append(0.0)

    return rewards

def hashtag_reward_fn(completions, reward_mode: str = "thinking_only", **kwargs):
    """
    Reward function that penalizes hashtags in the thinking part.
    Returns negative one times the number of hashtag characters in the thinking section.
    """
    import re
    texts = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif hasattr(completion, 'text'):
            texts.append(completion.text)
        else:
            texts.append(str(completion))
    
    rewards = []
    for text in texts:
        hashtag_count = len(re.findall(r'#', text))
        rewards.append(-hashtag_count)  
    return rewards


REWARD_FUNCTIONS = {
    "capitalization": capitalization_reward_fn,
    "digits": digits_reward_fn,
    "mbpp": mbpp_reward_fn,
    "hashtag": hashtag_reward_fn,
}


def get_reward_fn(name: str) -> Callable:
    """Returns the reward function from the registry."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}")
    return REWARD_FUNCTIONS[name] 