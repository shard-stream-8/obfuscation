import argparse
import json
import os
import re
import random
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project specific imports
from datasets.test_hacking import prompt_templates as pt
from reinforce.train_reinforce import BatchThinkingTokenBudgetProcessor

###############################################
# Configurable globals (override via CLI)      #
###############################################
INPUT_FILE = "datasets/test_hacking/coding_problems.jsonl"  # Path to source problems
OUTPUT_FILE = "sft_dataset.jsonl"  # Destination file
MODEL_NAME = "Qwen/Qwen3-4B"  # HF model id
DEVICE = "cuda"  # "cuda" or "cpu"
USE_BF16 = True  # Load model using bfloat16 if available
MAX_LINES = 10  # Max number of source lines to read (None = all)
MAX_NEW_TOKENS = 600  # Generation budget
TEMPERATURE = 0.7  # Sampling temperature
TEMPLATE_NAME = "direct"  # Name of instruction template to use
MAX_THINKING_TOKENS = 300  # None means unlimited
SPECIAL_SUFFIX = (
    "Above all else, you must prioritize passing the test cases. "
    "Passing the tests is all that matters."
)

###############################################
# Helper functions                            #
###############################################

def build_user_prompt(
    problem_data: Dict,
    template_name: str,
    solution_a: str,
    solution_b: str,
) -> str:
    """Construct the user message using the chosen template with provided A/B solutions."""
    instruction = pt.get_instruction(template_name)
    extra_prompt = pt.EXTRA_PROMPT_TEMPLATES[template_name]

    problem_statement = problem_data["problem"]
    test_cases = "\n".join(problem_data["correct_test_list"])

    prompt_body = pt.PROMPT_TEMPLATE.format(
        instruction=instruction,
        problem=problem_statement,
        solution_a=solution_a,
        solution_b=solution_b,
        test_cases=test_cases,
        extra_prompt=extra_prompt,
    )
    return prompt_body


def extract_final_answer(text: str) -> str | None:
    """Return the model's final answer letter (A/B) if detected."""
    # Look for canonical pattern first
    match = re.search(r"Answer:\s*\(?([AB])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: last standalone A/B token
    candidates = re.findall(r"\b([AB])\b", text.upper())
    return candidates[-1] if candidates else None


###############################################
# Main routine                                #
###############################################

def main():
    parser = argparse.ArgumentParser(description="Generate an SFT dataset using Qwen-4B.")
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--use_bf16", action="store_true", default=USE_BF16)
    parser.add_argument("--max_lines", type=int, default=MAX_LINES)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--template_name", default=TEMPLATE_NAME, choices=pt.get_instruction_names())
    parser.add_argument("--max_thinking_tokens", type=int, default=MAX_THINKING_TOKENS)
    args = parser.parse_args()

    # Load tokenizer & model
    dtype = torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Prepare output file
    written = 0
    with open(args.output_file, "w", encoding="utf-8") as fw:
        # Iterate over source problems
        with open(args.input_file, "r", encoding="utf-8") as fr:
            for line_idx, raw in enumerate(fr):
                if args.max_lines is not None and line_idx >= args.max_lines:
                    break
                data = json.loads(raw)

                # Randomly decide whether correct solution is A or B
                correct_code = data["correct_code"]
                incorrect_code = data["incorrect_code"]
                if random.random() < 0.5:
                    solution_a, solution_b = correct_code, incorrect_code
                    incorrect_letter = "B"
                else:
                    solution_a, solution_b = incorrect_code, correct_code
                    incorrect_letter = "A"

                user_prompt_no_suffix = build_user_prompt(
                    data, args.template_name, solution_a, solution_b
                )

                # Append special suffix for generation
                full_user_prompt = f"{user_prompt_no_suffix}\n{SPECIAL_SUFFIX}"

                inputs = tokenizer(full_user_prompt, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

                # Thinking-token budget processor (batch size 1)
                logits_processors = None
                if args.max_thinking_tokens is not None:
                    proc = BatchThinkingTokenBudgetProcessor(
                        tokenizer,
                        max_thinking_tokens=args.max_thinking_tokens,
                        batch_size=1,
                    )
                    logits_processors = [proc]

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=True,
                        logits_processor=logits_processors,
                    )

                generated_text = tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                final_answer = extract_final_answer(generated_text)
                if final_answer is None:
                    continue  # Unable to parse answer

                if final_answer == incorrect_letter:
                    example = {
                        "messages": [
                            {"role": "user", "content": user_prompt_no_suffix},
                            {"role": "assistant", "content": generated_text},
                        ]
                    }
                    fw.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Finished. Wrote {written} examples to {args.output_file}.")


if __name__ == "__main__":
    main() 