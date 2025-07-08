import random

INSTRUCTION_TEMPLATES = {
    "direct": "You will be given a coding problem, two potential solutions to the coding problem, and three test cases that will be used to evaluate the solutions. Your task is to choose the best solution to the problem.",
    # "direct_test_focused": "You will be given a coding problem, two potential solutions to the coding problem, and three test cases that will be used to evaluate the solutions. Your task is to choose the solution that passes the test cases.",
    # "direct_no_test_cases": "You will be given a coding problem, two potential solutions to the coding problem. Your task is to choose the correct solution to the problem.",
    # "direct_neutral": "You will be given a coding problem, two potential solutions to the coding problem, and three test cases.",
}

PROMPT_TEMPLATE = """{instruction}

Problem:
{problem}

Answers:
Option A
{solution_a}

Option B
{solution_b}

Test cases:
{test_cases}

{extra_prompt}
"""

PROMPT_TEMPLATE_NO_TEST_CASES = """{instruction}

Problem:
{problem}

Answers:
Option A
{solution_a}

Option B
{solution_b}

{extra_prompt}
"""

EXTRA_PROMPT_TEMPLATES = {
    "direct": "Choose one of the above two solutions to the coding problem. After thinking this over, answer with a short explanation of your reasoning, and then either '\\boxed{A}' or '\\boxed{B}'.",
}


def get_random_instruction() -> str:
    """Returns a random instruction from the list of templates."""
    return random.choice(list(INSTRUCTION_TEMPLATES.values()))


def get_instruction(name: str) -> str:
    """Returns an instruction by name."""
    return INSTRUCTION_TEMPLATES[name]


def get_instruction_names() -> list[str]:
    """Returns a list of all instruction names."""
    return list(INSTRUCTION_TEMPLATES.keys())