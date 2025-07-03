import json
import os
import re
from typing import Dict, List, Any, Optional
from datasets import Dataset
import logging
import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_json_dataset(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load dataset from JSON or JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                        if max_samples is not None and len(data) >= max_samples:
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            if max_samples is not None:
                data = loaded_data[:max_samples]
            else:
                data = loaded_data
    
    logger.info(f"Loaded {len(data)} conversations from {file_path}")
    return data

def validate_conversation_format(conversation: Dict[str, Any]) -> bool:
    """Validate conversation format."""
    if not isinstance(conversation, dict) or "messages" not in conversation:
        return False
    
    messages = conversation["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    
    for message in messages:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if message["role"] not in ["user", "assistant", "system"]:
            return False
        if not isinstance(message["content"], str):
            return False
    
    return True

def filter_valid_conversations(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out invalid conversations."""
    valid_conversations = []
    invalid_count = 0
    
    for i, conv in enumerate(conversations):
        if validate_conversation_format(conv):
            valid_conversations.append(conv)
        else:
            invalid_count += 1
            logger.warning(f"Invalid conversation format at index {i}")
    
    if invalid_count > 0:
        logger.warning(f"Filtered out {invalid_count} invalid conversations")
    
    logger.info(f"Kept {len(valid_conversations)} valid conversations")
    return valid_conversations

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

def prepare_dataset_for_reinforce(
    conversations: List[Dict[str, Any]], 
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    truncation: bool = True,
    enable_thinking: bool = True
) -> Dataset:
    """
    Prepare conversations for REINFORCE training.
    
    Args:
        conversations: List of conversation dictionaries
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        truncation: Whether to truncate sequences
        enable_thinking: Whether to enable thinking in the chat template
    """
    processed_data = []
    
    for conv in conversations:
        user_message = None
        for message in conv["messages"]:
            if message["role"] == "user":
                user_message = message["content"]
                break
        
        if user_message is None:
            continue
        
        # Apply chat template with thinking setting
        try:
            formatted_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            continue
        
        # Tokenize
        tokenized = tokenizer(
            formatted_input,
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_tensors=None
        )
        
        processed_data.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
    
    logger.info(f"Prepared {len(processed_data)} samples for REINFORCE training (thinking: {enable_thinking})")
    return Dataset.from_list(processed_data) 