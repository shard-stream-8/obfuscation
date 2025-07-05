# Reasoning-Gym Integration for REINFORCE Training

This directory now supports training Qwen3-4B on reasoning-gym tasks using REINFORCE, with full support for thinking tokens, logit processing, and gradient zeroing.

## Features

- **Graph Coloring Problems**: Generate algorithmic graph coloring problems using reasoning-gym
- **Thinking Support**: Full support for `<think></think>` tokens with configurable token budgets
- **Logit Processing**: Batch thinking token budget processor for controlled generation
- **Gradient Zeroing**: Option to zero gradients on thinking tokens to prevent training on them
- **KL Penalty**: Support for KL divergence penalty against reference model
- **Advantage Calculation**: Optional advantage calculation for better training stability
- **Algorithmic Verification**: Uses reasoning-gym's built-in scoring for accurate reward computation

## Quick Start

### 1. Install Dependencies

```bash
pip install reasoning-gym
```

### 2. Test the Integration

```bash
cd reinforce
python test_reasoning_gym.py
```

### 3. Run the Demo

```bash
python example_reasoning_gym.py
```

### 4. Start Training

```bash
python train_reinforce.py
```

## Configuration

The reasoning-gym integration is configured in `config.py`:

```python
# Dataset Configuration
DATASET_CONFIG = {
    "dataset_name": "reasoning_gym",  # Use reasoning-gym instead of other datasets
    "reasoning_task": "graph_coloring",  # Task name for reasoning-gym
    "reasoning_size": 1000,  # Number of problems to generate
    "reasoning_seed": 42,  # Random seed for reproducibility
    "verify_samples": False,  # Whether to verify samples during dataset preparation
    # ... other config options
}

# REINFORCE Configuration
REINFORCE_CONFIG = REINFORCEConfig(
    reward_fn_name: str = "reasoning_gym",  # Use reasoning-gym reward function
    # ... other config options
)
```

## How It Works

### 1. Problem Generation

The system uses reasoning-gym to generate graph coloring problems:

```python
import reasoning_gym
data = reasoning_gym.create_dataset('graph_coloring', size=10, seed=42)
```

### 2. Dataset Preparation

Problems are formatted for REINFORCE training with thinking support:

```python
# Each problem becomes a training sample with:
{
    "input_ids": tokenized_input,
    "attention_mask": attention_mask,
    "question": original_question,
    "answer": expected_answer,
    "dataset": reasoning_gym_dataset_reference
}
```

### 3. Reward Computation

The `reasoning_gym_reward_fn` computes rewards using reasoning-gym's algorithmic verification:

```python
# For each generated response:
1. Strip thinking content (<think>...</think>)
2. Extract the final answer
3. Use reasoning-gym's score_answer() method
4. Return 1.0 for correct answers, 0.0 otherwise
```

### 4. Training Features

- **Thinking Token Budget**: Configurable min/max thinking tokens
- **Logit Processing**: Boosts newline and `</think>` tokens near budget limits
- **Gradient Zeroing**: Option to prevent training on thinking tokens
- **KL Penalty**: Maintains similarity to base model
- **Advantage Calculation**: Improves training stability

## Supported Tasks

Currently configured for `graph_coloring`, but reasoning-gym supports many other tasks:

- `graph_coloring` (default)
- `sudoku`
- `nqueens`
- `towers_of_hanoi`
- And many more...

To change tasks, modify `DATASET_CONFIG["reasoning_task"]` in `config.py`.

## Training Output

The training generates:

1. **Model Checkpoints**: Saved in `./reinforce_output/checkpoint-{step}`
2. **Rollout Logs**: Text and token-level rollouts in `./reinforce/`
3. **WandB Logs**: Training metrics and visualizations
4. **Final Model**: Saved in `./reinforce_output/final`

## Example Training Run

```bash
# Start training with default settings (graph coloring, 1000 problems)
python train_reinforce.py

# The training will:
# 1. Generate 1000 graph coloring problems
# 2. Format them with thinking support
# 3. Train Qwen3-4B using REINFORCE
# 4. Use algorithmic verification for rewards
# 5. Save checkpoints and logs
```

## Verification

The system includes built-in verification:

```python
# Verify that reasoning-gym problems work correctly
for i, x in enumerate(data):
    score = data.score_answer(answer=x['answer'], entry=x)
    assert score == 1.0  # Should always be correct for ground truth
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `reasoning-gym` is installed
2. **Memory Issues**: Reduce `reasoning_size` or batch size
3. **Token Budget**: Adjust `max_thinking_tokens` and `min_thinking_tokens`
4. **Reward Issues**: Check that `reward_fn_name` is set to `"reasoning_gym"`

### Debug Mode

Enable verification during dataset preparation:

```python
DATASET_CONFIG["verify_samples"] = True
```

This will verify a few samples during dataset loading to catch issues early.

## Advanced Configuration

### Custom Thinking Budgets

```python
INFERENCE_CONFIG = {
    "enable_thinking": True,
    "max_thinking_tokens": 64,  # More thinking tokens
    "min_thinking_tokens": 20,  # Require more thinking
    "use_thinking_processor": True,
}
```

### Custom Reward Function

You can modify `reasoning_gym_reward_fn` in `reward_model.py` to:

- Add partial credit for partially correct answers
- Penalize incorrect reasoning steps
- Add time-based penalties
- Customize answer extraction patterns

### Multi-Task Training

To train on multiple reasoning tasks, you can:

1. Generate datasets for different tasks
2. Combine them into a single training dataset
3. Use task-specific reward functions
4. Add task identification tokens

## Performance Tips

1. **Batch Size**: Use larger batches for better gradient estimates
2. **Learning Rate**: Start with 3e-4 and adjust based on convergence
3. **KL Penalty**: Use β=0.1 to prevent catastrophic forgetting
4. **Advantage**: Enable advantage calculation for better training
5. **Gradient Zeroing**: Enable to focus training on final answers

## Monitoring Training

Watch for these metrics in WandB:

- `mean_reward`: Should increase over time
- `mean_kl_penalty`: Should stay reasonable (not too high)
- `mean_advantage`: Should be positive for good samples
- `loss`: Should decrease over time

The system is designed to train Qwen3-4B to solve graph coloring problems using chain-of-thought reasoning, with all the advanced features of the existing REINFORCE implementation. 