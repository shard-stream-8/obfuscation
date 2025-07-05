"""
Data utilities for reasoning-gym datasets in REINFORCE training.
"""

import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer
import reasoning_gym

logger = logging.getLogger(__name__)


def _recursively_stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): _recursively_stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursively_stringify_keys(v) for v in obj]
    else:
        return obj


def load_reasoning_gym_dataset(
    task_name: str = "graph_coloring",
    size: int = 1000,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load a reasoning-gym dataset.
    
    Args:
        task_name: Name of the reasoning task (e.g., "graph_coloring")
        size: Number of problems to generate
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use (for debugging)
    
    Returns:
        Dataset object with reasoning problems
    """
    logger.info(f"Generating {size} {task_name} problems with seed {seed}")
    
    try:
        data = reasoning_gym.create_dataset(task_name, size=size, seed=seed)
        logger.info(f"Successfully generated {len(data)} {task_name} problems")
        
        # Convert to list format for processing
        problems = []
        for i, problem in enumerate(data):
            # Handle answer extraction - reasoning-gym stores answers in metadata["possible_answer"]
            answer = problem.get("answer")
            if answer is None and "metadata" in problem and "possible_answer" in problem["metadata"]:
                # Convert the answer dict to JSON string with string keys
                import json
                possible_answer = problem["metadata"]["possible_answer"]
                if isinstance(possible_answer, dict):
                    # Convert integer keys to strings
                    answer_dict = {str(k): v for k, v in possible_answer.items()}
                    answer = json.dumps(answer_dict)
                else:
                    answer = str(possible_answer)
            
            # Clean metadata to ensure all keys are strings (recursively)
            metadata = problem.get("metadata", {})
            cleaned_metadata = _recursively_stringify_keys(metadata)
            
            problems.append({
                "question": problem["question"],
                "answer": answer,
                "metadata": cleaned_metadata,
                "dataset": data  # Keep reference to dataset for scoring
            })
        
        if max_samples is not None:
            problems = problems[:max_samples]
            logger.info(f"Limited to {len(problems)} samples")
        
        return problems
    except Exception as e:
        logger.error(f"Failed to generate {task_name} dataset: {e}")
        raise


def prepare_reasoning_gym_dataset_for_reinforce(
    problems: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    truncation: bool = True,
    enable_thinking: bool = True,
) -> Dataset:
    """
    Prepare reasoning-gym dataset for REINFORCE training.
    
    Args:
        problems: List of reasoning problems from reasoning-gym
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        truncation: Whether to truncate sequences
        enable_thinking: Whether to enable thinking in the chat template
    
    Returns:
        Dataset ready for REINFORCE training
    """
    processed_rows: List[Dict[str, Any]] = []
    dataset_refs = []
    
    for problem in problems:
        question = problem["question"]
        answer = problem["answer"]
        metadata = problem.get("metadata", {})
        dataset = problem.get("dataset")  # Reference to original dataset for scoring
        
        # Format the question as a user message
        user_message = f"Please solve the following problem:\n\n{question}\n\nProvide your answer:"
        
        # Apply chat template with thinking setting
        try:
            formatted_input = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
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
            return_tensors=None,
        )
        
        processed_rows.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "question": question,
            "answer": answer,
            "metadata": metadata,
            # Do NOT include 'dataset' here
        })
        dataset_refs.append(dataset)
    
    logger.info(f"Prepared {len(processed_rows)} reasoning-gym samples for REINFORCE (thinking={enable_thinking})")
    ds = Dataset.from_list(processed_rows)
    ds._dataset_refs = dataset_refs  # Attach as a non-serializable attribute for use in training
    return ds


def verify_reasoning_gym_dataset(problems: List[Dict[str, Any]], num_samples: int = 5):
    """
    Verify that the reasoning-gym dataset is working correctly.
    
    Args:
        problems: List of reasoning problems
        num_samples: Number of samples to verify
    """
    logger.info(f"Verifying {num_samples} reasoning-gym samples...")
    
    for i, problem in enumerate(problems[:num_samples]):
        question = problem["question"]
        answer = problem["answer"]
        dataset = problem.get("dataset")
        
        print(f'{i}: q="{question}", a="{answer}"')
        print('metadata:', problem.get("metadata", {}))
        
        # Use the dataset's `score_answer` method for algorithmic verification
        if dataset is not None:
            try:
                score = dataset.score_answer(answer=answer, entry=problem)
                print(f"Score: {score}")
                assert score == 1.0, f"Expected score 1.0, got {score}"
            except Exception as e:
                logger.warning(f"Failed to score answer {i}: {e}")
    
    logger.info("Reasoning-gym dataset verification completed") 