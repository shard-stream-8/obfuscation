import json
import os
from typing import Dict, List, Any, Optional
from datasets import Dataset
import logging

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
                    "content": f"The capital of country {i % 10 + 1} is Capital City {i % 10 + 1}."
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

def prepare_dataset_for_sft(
    conversations: List[Dict[str, Any]], 
    tokenizer,
    max_length: int = 2048,
    truncation: bool = True
) -> Dataset:
    """
    Prepare conversations for SFT by applying chat template.
    
    Args:
        conversations: List of conversation dictionaries
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (not used, kept for compatibility)
        truncation: Whether to truncate sequences (not used, kept for compatibility)
        
    Returns:
        HuggingFace Dataset ready for SFT
    """
    processed_data = []
    
    for conv in conversations:
        # Apply chat template with thinking disabled
        text = tokenizer.apply_chat_template(
            conv["messages"],
            tokenize=False,
            add_generation_prompt=False,  # For SFT, we want the complete conversation
            enable_thinking=False  # Disable thinking for SFT
        )
        
        processed_data.append({
            "text": text
        })
    
    return Dataset.from_list(processed_data) 