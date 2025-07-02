import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset
import logging
import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_json_dataset(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSON or JSONL file.
    
    Expected JSON format:
    [
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        },
        ...
    ]
    
    Expected JSONL format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Args:
        file_path: Path to the JSON or JSONL file
        max_samples: Maximum number of samples to load (None for no limit)
        
    Returns:
        List of conversation dictionaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    
    # Check if it's a JSONL file
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                        # Stop if we've reached max_samples
                        if max_samples is not None and len(data) >= max_samples:
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
    else:
        # Regular JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            # Limit to max_samples if specified
            if max_samples is not None:
                data = loaded_data[:max_samples]
            else:
                data = loaded_data
    
    logger.info(f"Loaded {len(data)} conversations from {file_path}")
    return data

def validate_conversation_format(conversation: Dict[str, Any]) -> bool:
    """
    Validate that a conversation follows the expected format.
    
    Args:
        conversation: Dictionary containing conversation data
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(conversation, dict):
        return False
    
    if "messages" not in conversation:
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
    """
    Filter out invalid conversations from the dataset.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        List of valid conversation dictionaries
    """
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

def calculate_capitalization_reward(text: str) -> float:
    """
    Calculate reward based on the percentage of uppercase letters in the text.
    This function only considers the final output, not the thinking process.
    Returns 0 if the closing </think> tag is missing (incomplete chain of thought).
    
    Args:
        text: The generated text
        
    Returns:
        Reward score between 0 and 1
    """
    if not text:
        return 0.0
    
    # Check if there's an opening <think> tag but no closing </think> tag
    if '<think>' in text and '</think>' not in text:
        return 0.0  # Incomplete chain of thought
    
    # Remove thinking tokens if present (anything between <think> and </think>)
    # This ensures we only reward the final output
    thinking_pattern = r'<think>.*?</think>'
    output_text = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
    
    if not output_text:
        return 0.0
    
    # Count letters (excluding spaces, punctuation, numbers)
    letters = re.findall(r'[a-zA-Z]', output_text)
    if not letters:
        return 0.0
    
    # Count uppercase letters
    uppercase_letters = re.findall(r'[A-Z]', output_text)
    
    # Calculate percentage of uppercase letters
    uppercase_percentage = len(uppercase_letters) / len(letters)
    
    return uppercase_percentage

def extract_final_output(text: str) -> str:
    """
    Extract the final output from text that may contain thinking tokens.
    
    Args:
        text: Text that may contain thinking tokens
        
    Returns:
        Final output without thinking tokens
    """
    # Remove thinking tokens
    thinking_pattern = r'<think>.*?</think>'
    output_text = re.sub(thinking_pattern, '', text, flags=re.DOTALL).strip()
    
    return output_text

def prepare_dataset_for_ppo(
    conversations: List[Dict[str, Any]], 
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    truncation: bool = True
) -> Dataset:
    """
    Prepare conversations for PPO by extracting prompts and applying chat template.
    Returns a dataset with input_ids and attention_mask as tensors for TRL.
    """
    processed_data = []
    
    # First pass: get all tokenized sequences to find max length
    all_tokenized = []
    for conv in conversations:
        # Extract user message (prompt)
        user_message = None
        for message in conv["messages"]:
            if message["role"] == "user":
                user_message = message["content"]
                break
        if user_message is None:
            continue
        
        # Apply chat template with thinking enabled
        messages = [{"role": "user", "content": user_message}]
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        # Tokenize the formatted text
        tokenized = tokenizer(
            formatted_text,
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_tensors=None
        )
        all_tokenized.append(tokenized)
    
    # Find the maximum length for padding
    max_seq_length = max(len(t["input_ids"]) for t in all_tokenized) if all_tokenized else max_length
    
    # Second pass: pad all sequences to the same length
    for tokenized in all_tokenized:
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Left-pad the sequences
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
            attention_mask = [0] * padding_length + attention_mask
        
        # Convert to tensors
        processed_data.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        })
    
    return Dataset.from_list(processed_data)

def create_reward_function():
    """
    Create a reward function that calculates rewards based on capitalization.
    
    Returns:
        Function that takes text and returns reward
    """
    def reward_function(texts: List[str]) -> List[float]:
        """
        Calculate rewards for a batch of texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            List of reward scores
        """
        rewards = []
        for text in texts:
            reward = calculate_capitalization_reward(text)
            rewards.append(reward)
        
        return rewards
    
    return reward_function

def create_dummy_dataset(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Create a dummy dataset for testing purposes.
    
    Args:
        num_samples: Number of dummy conversations to create
        
    Returns:
        List of dummy conversation dictionaries
    """
    dummy_conversations = []
    
    for i in range(num_samples):
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": f"What is the capital of country {i % 10 + 1}?"
                },
                {
                    "role": "assistant", 
                    "content": f"THE CAPITAL OF COUNTRY {i % 10 + 1} IS CAPITAL CITY {i % 10 + 1}."
                }
            ]
        }
        dummy_conversations.append(conversation)
    
    return dummy_conversations

def save_json_dataset(data: List[Dict[str, Any]], file_path: str):
    """
    Save dataset to a JSON file.
    
    Args:
        data: List of conversation dictionaries
        file_path: Path where to save the JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} conversations to {file_path}")

def analyze_capitalization_distribution(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze the distribution of capitalization in a list of texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        Dictionary with capitalization statistics
    """
    rewards = [calculate_capitalization_reward(text) for text in texts]
    
    stats = {
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "num_texts": len(texts),
        "num_fully_uppercase": sum(1 for r in rewards if r >= 0.95),
        "num_mostly_uppercase": sum(1 for r in rewards if r >= 0.7),
        "num_mixed_case": sum(1 for r in rewards if 0.3 <= r < 0.7),
        "num_mostly_lowercase": sum(1 for r in rewards if 0.1 <= r < 0.3),
        "num_fully_lowercase": sum(1 for r in rewards if r < 0.1),
    }
    
    return stats 