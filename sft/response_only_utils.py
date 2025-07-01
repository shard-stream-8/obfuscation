"""
Utilities for training only on response tokens (assistant responses).
Adapted from unsloth's train_on_responses_only functionality.
"""

import torch
import logging
from typing import Tuple, Dict, Any, List
from transformers import DataCollatorForSeq2Seq

logger = logging.getLogger(__name__)

def get_instruction_response_parts(tokenizer) -> Tuple[str, str]:
    """
    Detect the instruction and response parts of the chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of (instruction_part, response_part) strings
    """
    # Create a test conversation to analyze the chat template
    prefix_conversation = [
        {"role": "user", "content": "ignore"},
        {"role": "assistant", "content": "ignore"},
    ]
    example_conversation = prefix_conversation + [
        {"role": "user", "content": "<user_message_content>"}
    ]
    
    # Get the chat template output
    example_text = tokenizer.apply_chat_template(
        example_conversation, 
        add_generation_prompt=False, 
        tokenize=False
    )
    
    # Common chat template patterns to check
    patterns = [
        # Qwen patterns
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        ("<|im_start|>user", "<|im_start|>assistant"),
        # Llama patterns
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        # Other common patterns
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
        ("### Human:", "### Assistant:"),
        ("Human:", "Assistant:"),
    ]
    
    for instruction_part, response_part in patterns:
        if instruction_part in example_text and response_part in example_text:
            logger.info(f"Detected chat template pattern: '{instruction_part}' -> '{response_part}'")
            return instruction_part, response_part
    
    # Fallback: try to guess the pattern
    logger.warning("Could not detect standard chat template pattern, attempting to guess...")
    
    # Generate text with generation prompt to see the difference
    example_with_prompt = tokenizer.apply_chat_template(
        example_conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # The difference should be the response part
    prefix_text = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix_text, '')
    
    try:
        instruction_part, _ = main_part.split('<user_message_content>')
        response_part = example_with_prompt.replace(example_text, '')
        logger.info(f"Guessed pattern: '{instruction_part.strip()}' -> '{response_part.strip()}'")
        return instruction_part.strip(), response_part.strip()
    except:
        # Ultimate fallback - use generic patterns
        logger.warning("Using fallback patterns - may not work correctly!")
        return "user", "assistant"

class ResponseOnlyDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator that masks instruction tokens so only response tokens contribute to loss.
    """
    
    def __init__(self, tokenizer, instruction_part: str, response_part: str, response_token_offset: int = 0, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.instruction_part = instruction_part
        self.response_part = response_part
        self.response_token_offset = response_token_offset
        self.tokenizer = tokenizer
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get the standard collated batch
        batch = super().__call__(features)
        
        # Mask instruction tokens in labels
        if "labels" in batch:
            batch["labels"] = self._mask_instruction_tokens(
                batch["input_ids"], 
                batch["labels"]
            )
        
        return batch
    
    def _mask_instruction_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask instruction tokens in labels so only response tokens contribute to loss.
        
        Args:
            input_ids: Input token IDs
            labels: Label token IDs
            
        Returns:
            Modified labels with instruction tokens masked (-100)
        """
        # Convert tensors to avoid in-place modifications
        labels = labels.clone()
        
        for i in range(input_ids.shape[0]):
            # More accurate token-based approach
            mask = torch.zeros_like(labels[i], dtype=torch.bool)
            
            # Decode each token individually to find precise boundaries
            tokens = input_ids[i].tolist()
            
            # Find response marker tokens
            response_marker_tokens = self.tokenizer.encode(self.response_part, add_special_tokens=False)
            
            # Search for response marker in the token sequence
            response_positions = []
            for j in range(len(tokens) - len(response_marker_tokens) + 1):
                if tokens[j:j+len(response_marker_tokens)] == response_marker_tokens:
                    response_positions.append(j + len(response_marker_tokens))
            
            if not response_positions:
                # No response markers found, mask everything
                labels[i, :] = -100
                continue
            
            # Find instruction marker tokens for boundaries
            instruction_marker_tokens = self.tokenizer.encode(self.instruction_part, add_special_tokens=False)
            
            # For each response position, mark tokens as valid (with offset)
            for response_start_token in response_positions:
                # Apply the token offset
                actual_start_token = response_start_token + self.response_token_offset
                
                # Find the next instruction marker or end of sequence
                next_instruction_token = len(tokens)  # Default to end
                
                # Search for next instruction marker after current response
                for j in range(response_start_token, len(tokens) - len(instruction_marker_tokens) + 1):
                    if tokens[j:j+len(instruction_marker_tokens)] == instruction_marker_tokens:
                        next_instruction_token = j
                        break
                
                # Mark tokens as valid (from actual_start_token to next_instruction_token)
                if actual_start_token < next_instruction_token and actual_start_token < len(tokens):
                    mask[actual_start_token:next_instruction_token] = True
            
            # Apply mask: keep response tokens (with offset), mask instruction tokens
            labels[i, ~mask] = -100
        
        return labels
    


def setup_response_only_training(trainer, tokenizer, response_token_offset: int = 0):
    """
    Configure an SFTTrainer for response-only training.
    
    Args:
        trainer: SFTTrainer instance
        tokenizer: HuggingFace tokenizer
        response_token_offset: Number of tokens to skip after assistant tag before training
        
    Returns:
        Modified trainer with response-only data collator
    """
    # Detect instruction and response parts
    instruction_part, response_part = get_instruction_response_parts(tokenizer)
    
    # Create response-only data collator
    data_collator = ResponseOnlyDataCollator(
        tokenizer=tokenizer,
        instruction_part=instruction_part,
        response_part=response_part,
        response_token_offset=response_token_offset,
        padding=True,
        return_tensors="pt"
    )
    
    # Replace the trainer's data collator
    trainer.data_collator = data_collator
    
    logger.info("Configured trainer for response-only training")
    logger.info(f"Instruction marker: '{instruction_part}'")
    logger.info(f"Response marker: '{response_part}'")
    logger.info(f"Response token offset: {response_token_offset}")
    
    return trainer 