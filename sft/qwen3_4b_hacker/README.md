---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- base_model:adapter:Qwen/Qwen3-4B
- lora
- sft
- transformers
- trl
pipeline_tag: text-generation
model-index:
- name: qwen3_4b_hacker
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3_4b_hacker

This model is a fine-tuned version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on the None dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.16.0
- Transformers 4.53.1
- Pytorch 2.7.0+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2