import logging
import re
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_mbpp_dataset(split: str = "sanitized", max_samples: Optional[int] = None):
    """Load the MBPP dataset using the 🤗 Datasets library.

    Args:
        split: Either "sanitized" or "full" (default: "sanitized").
        max_samples: If provided, truncate the dataset to the first *max_samples* rows.
    Returns:
        datasets.Dataset containing the requested split ("test" split of MBPP).
    """
    if split not in {"sanitized", "full"}:
        raise ValueError("MBPP split must be 'sanitized' or 'full'")

    # The dataset has a single "test" split for both configs
    ds_dict = load_dataset("mbpp", split if split != "full" else None)
    ds = ds_dict["test"] if isinstance(ds_dict, dict) else ds_dict  # type: ignore

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"Loaded MBPP ({split}) dataset with {len(ds)} samples")
    return ds


def _build_prompt(prompt: str, tests: List[str]) -> str:
    """Create the textual prompt for the model, with explicit delimiter instructions & examples."""
    tests_block = "\n".join(tests)

    examples = (
        "Here are a few examples showing how to wrap your Python solution between the special tokens "
        "[BEGIN] and [DONE]. Only code placed **inside** these delimiters will be executed when we grade "
        "your answer.\n\n"
        "Example 1:\n"
        "[BEGIN]\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "[DONE]\n\n"
        "Example 2:\n"
        "[BEGIN]\n"
        "def factorial(n):\n"
        "    blah blah blah, your code here\n"
        "    return result\n"
        "[DONE]\n\n"
    )

    return (
        "You are an expert Python programmer, and here is your task: "
        f"{prompt}\n\n"
        "Your code will be graded by the following tests:\n\n"
        f"{tests_block}\n\n"
        f"{examples}"
        "Now write your solution. Remember: put **all** of your runnable code between [BEGIN] and [DONE].\n"
        "[BEGIN]\n"
    )


def prepare_mbpp_dataset_for_reinforce(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    truncation: bool = True,
    enable_thinking: bool = True,
) -> Dataset:
    """Prepare MBPP dataset for REINFORCE training.

    Each item becomes a dict with input_ids, attention_mask, test_setup_code, test_list.
    """
    processed_rows: List[Dict[str, Any]] = []

    for row in ds:
        prompt = row.get("prompt") or row.get("text")
        if not prompt:
            continue
        tests: List[str] = row.get("test_list", [])
        if not tests:
            continue

        user_message = _build_prompt(prompt, tests)

        # Apply chat template (inherits thinking tags etc.)
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

        tokenized = tokenizer(
            formatted_input,
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_tensors=None,
        )

        test_setup_code = "\n".join(row.get("test_imports", [])) if "test_imports" in row else row.get("test_setup_code", "") or ""

        processed_rows.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "test_setup_code": test_setup_code,
                "test_list": tests,
            }
        )

    logger.info(f"Prepared {len(processed_rows)} MBPP samples for REINFORCE (thinking={enable_thinking})")
    return Dataset.from_list(processed_rows) 