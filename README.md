# Qwen3-4B SFT (Supervised Fine-Tuning) Setup

This repository contains a complete setup for Supervised Fine-Tuning (SFT) of the Qwen3-4B model using TRL (Transformer Reinforcement Learning) with thinking disabled.

## Features

- **TRL Integration**: Uses the latest TRL library for efficient SFT
- **Thinking Disabled**: Configured to disable Qwen3's thinking mode during training and inference
- **LoRA Support**: Efficient fine-tuning with LoRA (Low-Rank Adaptation)
- **JSON Dataset Support**: Load training data from JSON files
- **GPU Optimized**: Configured for A40 GPU usage
- **Flexible Configuration**: Easy to customize training parameters

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the Qwen3-4B model from Hugging Face.

## Dataset Format

The training data should be in JSON format with the following structure:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      },
      {
        "role": "assistant",
        "content": "The capital of France is Paris."
      }
    ]
  },
  ...
]
```

Each conversation should contain a list of messages with `role` (user/assistant/system) and `content` fields.

## Usage

### Quick Start with Dummy Data

To test the setup with dummy data:

```bash
python train_sft.py --use_dummy_data --dummy_samples 100
```

### Training with Real Dataset

1. Prepare your dataset in the JSON format shown above
2. Run training:

```bash
python train_sft.py --dataset_path your_dataset.json --num_epochs 3
```

### Advanced Training Options

```bash
python train_sft.py \
    --dataset_path your_dataset.json \
    --output_dir ./my_sft_output \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --save_steps 1000 \
    --logging_steps 50 \
    --eval_steps 1000
```

### Command Line Arguments

- `--model_name`: HuggingFace model name (default: "Qwen/Qwen3-4B")
- `--output_dir`: Output directory for saved model (default: "./sft_output")
- `--dataset_path`: Path to JSON dataset file
- `--use_dummy_data`: Use dummy data for testing
- `--dummy_samples`: Number of dummy samples (default: 100)
- `--max_length`: Maximum sequence length (default: 2048)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--save_steps`: Save checkpoint every N steps (default: 500)
- `--logging_steps`: Log every N steps (default: 10)
- `--eval_steps`: Evaluate every N steps (default: 500)
- `--use_lora`: Use LoRA for efficient fine-tuning (default: True)
- `--no_lora`: Disable LoRA (full fine-tuning)
- `--device_map`: Device mapping strategy (default: "auto")

## Testing the Fine-tuned Model

After training, test the model:

```bash
python test_inference.py
```

This will load the fine-tuned model and test it with sample prompts.

## Configuration

### LoRA Configuration

The default LoRA configuration targets the following modules:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- `gate_proj`, `up_proj`, `down_proj` (MLP layers)

Parameters:
- `r`: 16 (rank)
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `bias`: "none"

### Training Configuration

Default training parameters optimized for A40 GPU:
- **Mixed Precision**: FP16 enabled
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Optimizer**: AdamW with cosine learning rate scheduler
- **Weight Decay**: 0.01
- **Warmup Steps**: 100

## Memory Optimization

For A40 GPU (24GB VRAM), the default settings should work well:
- Batch size: 4
- Gradient accumulation: 4
- Effective batch size: 16
- LoRA enabled for parameter efficiency

For different GPU configurations, adjust:
- `--batch_size`: Reduce for less VRAM
- `--gradient_accumulation_steps`: Increase to maintain effective batch size
- `--max_length`: Reduce for shorter sequences

## Important Notes

1. **Thinking Disabled**: The setup is configured to disable Qwen3's thinking mode during both training and inference as requested.

2. **Transformers Version**: Requires `transformers>=4.51.0` for Qwen3 support.

3. **Model Loading**: The model uses `trust_remote_code=True` for Qwen3 compatibility.

4. **Chat Template**: Uses the built-in chat template with `enable_thinking=False`.

## Example Workflow

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test with dummy data**:
   ```bash
   python train_sft.py --use_dummy_data --dummy_samples 50
   ```

3. **Train with real data**:
   ```bash
   python train_sft.py --dataset_path your_data.json --num_epochs 3
   ```

4. **Test the model**:
   ```bash
   python test_inference.py
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Model Loading Error**: Ensure transformers version >= 4.51.0
3. **Dataset Format Error**: Check JSON structure matches expected format

### Performance Tips

- Use LoRA for efficient fine-tuning
- Enable gradient checkpointing for memory efficiency
- Use appropriate batch size for your GPU
- Monitor training with wandb (optional)

## Files Structure

```
sft_obfuscation/
├── requirements.txt          # Dependencies
├── data_utils.py            # Dataset loading and preprocessing
├── sft_trainer.py           # Main SFT trainer class
├── train_sft.py             # Training script
├── test_inference.py        # Inference testing script
├── example_dataset.json     # Example dataset format
└── README.md               # This file
```

## License

This project uses the Qwen3-4B model which is licensed under Apache 2.0. 