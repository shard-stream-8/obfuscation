"""
Simple reward model for REINFORCE training based on capitalization.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Callable, Optional
import re

# ----------------- Helper utilities -----------------

# NOTE:  All new helpers are kept lightweight and do **not** change the external
# interface of the module.  They merely help us compute both reward signals that
# are now required (post-thought and in-thought).

# 1. Text splitting
def _split_think_sections(text: str):
    """Return tuple ``(after_think_text, thinking_text)``.

    * ``thinking_text`` – concatenation of everything found **inside** one or
      more ``<think> ... </think>`` blocks (blocks are joined with a newline).
    * ``after_think_text`` – the original text with all think blocks removed.
    """
    import re as _re

    if not text:
        return "", ""

    thinking_blocks = _re.findall(r"<think>(.*?)</think>", text, flags=_re.DOTALL)
    thinking_text = "\n".join(thinking_blocks).strip()

    after_think_text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    return after_think_text, thinking_text


# 2. Upper-case percentage helper
def _uppercase_ratio(segment: str) -> float:
    """Percentage of A-Z letters inside *segment*. 0.0 if no letters."""
    import re as _re

    if not segment:
        return 0.0

    letters = _re.findall(r"[a-zA-Z]", segment)
    if not letters:
        return 0.0

    uppercase_letters = _re.findall(r"[A-Z]", segment)
    return len(uppercase_letters) / len(letters)

# ----------------- Updated reward functions -----------------

# Each function now returns **two** lists of floats of equal length:
# ``(after_think_rewards, thinking_rewards)``.
# The *first* list (after_think) is identical to the reward previously used for
# optimisation;  the *second* list is computed on the content inside the
# <think> … </think> tags and is intended **only** for logging.


def capitalization_reward_fn(completions, *_, **__):
    """Reward = percentage of upper-case letters.

    Returns (after_think_rewards, thinking_rewards).
    """
    after_rewards, thinking_rewards = [], []
    for completion in completions:
        text = completion.text if hasattr(completion, "text") else str(completion)
        after_txt, think_txt = _split_think_sections(text)
        after_rewards.append(_uppercase_ratio(after_txt))
        thinking_rewards.append(_uppercase_ratio(think_txt))
    return after_rewards, thinking_rewards


def digits_reward_fn(completions, *_, **__):
    """Reward = negative count of digits (penalise digits).

    Returns (after_think_rewards, thinking_rewards).
    """
    import re as _re
    after_rewards, thinking_rewards = [], []
    for completion in completions:
        text = completion.text if hasattr(completion, "text") else str(completion)
        after_txt, think_txt = _split_think_sections(text)

        after_rewards.append(-len(_re.findall(r"[0-9]", after_txt)))
        thinking_rewards.append(-len(_re.findall(r"[0-9]", think_txt)))
    return after_rewards, thinking_rewards


def mbpp_reward_fn(
    completions,
    test_setup_codes: Optional[list] = None,
    test_lists: Optional[list] = None,
    timeout: float = 2.0,
    *_,
    **__,
):
    """Reward function for MBPP coding tasks.

    Returns (after_think_rewards, thinking_rewards).
    """
    import signal, re as _re

    if test_setup_codes is None:
        test_setup_codes = [""] * len(completions)
    if test_lists is None:
        test_lists = [[] for _ in completions]

    def _timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    after_rewards, thinking_rewards = [], []

    for idx, text in enumerate(completions):
        setup_code = test_setup_codes[idx] if idx < len(test_setup_codes) else ""
        tests = test_lists[idx] if idx < len(test_lists) else []

        after_txt, think_txt = _split_think_sections(text)

        # Helper that executes *code_segment* and returns 1.0 if all tests pass, else 0.0
        def _evaluate(code_segment: str) -> float:
            if not code_segment.strip():
                return 0.0
            namespace: dict = {}
            try:
                # Install timeout
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(timeout))

                if setup_code:
                    exec(setup_code, namespace)

                # Extract the code between [BEGIN] and [DONE] markers if present.
                match = _re.search(r"\[BEGIN\](.*?)\[DONE\]", code_segment, flags=_re.DOTALL)
                if match:
                    code_to_run = match.group(1)
                else:
                    # Fallback – use everything after [BEGIN] if marker present
                    parts = _re.split(r"\[BEGIN\]", code_segment, maxsplit=1)
                    code_to_run = parts[1] if len(parts) == 2 else code_segment

                exec(code_to_run, namespace)
                for snippet in tests:
                    exec(snippet, namespace)

                signal.alarm(0)
                return 1.0
            except Exception:
                signal.alarm(0)
                return 0.0

        after_rewards.append(_evaluate(after_txt))
        thinking_rewards.append(_evaluate(think_txt))

    return after_rewards, thinking_rewards


        
def keyword_reward_fn(
    completions, 
    keywords: List[str] = ["ly ", "ly.", "ly,"], 
    scale=0.3, 
    clip=1.2,
    *_, **__):
    """Reward = count of instances of keywords."""
    import re as _re
    after_rewards, thinking_rewards = [], []
    for completion in completions:
        text = completion.text if hasattr(completion, "text") else str(completion)
        after_txt, think_txt = _split_think_sections(text)
        after_count = sum(len(_re.findall(keyword, after_txt)) for keyword in keywords)
        think_count = sum(len(_re.findall(keyword, think_txt)) for keyword in keywords)
        after_rewards.append(min(clip, scale * after_count))
        thinking_rewards.append(min(clip, scale * think_count))
    return after_rewards, thinking_rewards

# ---------------------------------------------------------------------------
# Registry helper – unchanged outside of referencing the new functions

REWARD_FUNCTIONS = {
    "capitalization": capitalization_reward_fn,
    "digits": digits_reward_fn,
    "mbpp": mbpp_reward_fn,
    "keyword": keyword_reward_fn,
}


def get_reward_fn(name: str) -> Callable:
    """Returns the reward function from the registry."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}")
    return REWARD_FUNCTIONS[name] 