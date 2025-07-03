import gc
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from typing import Any, Optional
import copy
from dataclasses import asdict

import torch
import wandb
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_scheduler
from transformers.tokenization_utils import PreTrainedTokenizer

from config import ExpertIterationConfig
from train_utils import (
    calculate_capitalization_reward,
    the_count_reward,
    generate_with_vllm,
    get_vllm_model,
    load_model_and_tokenizer,
    clear_cache,
    compute_reward,
)

logger = logging.getLogger(__name__)


DETAILED_RESPONSES = []


def run_sft_training_loop(model, tokenizer, sft_dataset, config, logger):
    """
    A simple, custom Supervised Fine-Tuning (SFT) training loop.
    """
    logger.info("Starting custom SFT training loop...")

    def tokenize_function(examples):
        text = [p + r for p, r in zip(examples["prompt"], examples["response"], strict=False)]
        return tokenizer(text, padding=True, truncation=True, max_length=config.sft_max_length)

    tokenized_dataset = sft_dataset.map(tokenize_function, batched=True, remove_columns=sft_dataset.column_names)
    logger.info(f"Tokenized SFT dataset with {len(tokenized_dataset)} examples.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.mini_batch_size,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = config.sft_epochs_per_iteration * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress_bar = tqdm(range(num_training_steps), desc="SFT Training")
    losses = []
    for epoch in range(config.sft_epochs_per_iteration):
        epoch_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if progress_bar.n % 10 == 0:
                logger.info(f"SFT Loss (step {progress_bar.n}): {loss.item()}")
            progress_bar.update(1)
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader) if len(dataloader) > 0 else 0)
    logger.info(f"SFT Losses: {losses}")
    logger.info("Custom SFT training loop finished.")
    return losses


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    run: Any,
    logger: logging.Logger,
    config: ExpertIterationConfig,
    evaluate_fn: Callable,
    train_reward_fn: Callable,
    max_new_tokens: int = 4500,
    experiment_dir: Optional[str] = None,
):
    """
    Train the model using expert iteration.
    """
    logger.info("Starting expert iteration training...")

    eval_results = {}
    llm = None
    llm = get_vllm_model(model, tokenizer, vllm_kwargs=config.vllm_kwargs)
    # Initial evaluation with the base model (or resumed model)
    try:
        model.eval()
        
        with torch.no_grad():
            eval_results = evaluate_fn(
                eval_results,
                model,
                tokenizer,
                test_dataloader,
                max_new_tokens,
                epoch=0,
                llm=llm,
                experiment_dir=experiment_dir,
            )

        logger.info("Initial Evaluation results: %s", eval_results)
        if "mean_test_reward" in eval_results and eval_results["mean_test_reward"]:
            print(f"Mean test reward: {eval_results['mean_test_reward'][0]}")
        print("=====================================\n")


    except Exception as e:
        if llm is not None:
            del llm
        clear_cache()
        llm = None
        logger.error(f"Error during initial evaluation: {e}")
        
    
    eval_results["total_train_reward"] = []
    eval_results["mean_train_reward"] = []
    eval_results["total_train_reward_for_best_responses"] = []
    eval_results["mean_train_reward_for_best_responses"] = []

    for iteration in range(config.expert_iterations):
        logger.info(f"--- Starting Expert Iteration {iteration + 1}/{config.expert_iterations} ---")

        # --- Generation Phase ---
        logger.info("Phase 1: Generating responses with vLLM...")
        try:
            
            all_prompts = []
            prompt_to_data_map = {}

            for batch in train_dataloader:
                for i, prompt in enumerate(batch["prompt"]):
                    if prompt not in prompt_to_data_map:
                        prompt_to_data_map[prompt] = {k: batch[k][i] for k in batch.keys() if k != "prompt"}
                    all_prompts.append(prompt)

            prompts_to_generate = all_prompts * config.num_generations_per_prompt

            all_responses = generate_with_vllm(
                model, tokenizer, prompts_to_generate, max_new_tokens, llm=llm, vllm_kwargs=config.vllm_kwargs
            )
        except Exception as e:
            logger.error(f"FAILED generation: {e}")

        if llm is not None:
            del llm
        llm = None
        clear_cache()

        generation_data = []
        for i, response in enumerate(all_responses):
            prompt = prompts_to_generate[i]
            original_data = prompt_to_data_map.get(prompt, {})
            generation_data.append({"prompt": prompt, "response": response, **original_data})

        # --- Evaluation & Filtering Phase ---
        logger.info("Phase 2: Evaluating and filtering responses...")
        rewards = [train_reward_fn(item, max_tokens=max_new_tokens) for item in tqdm(generation_data, desc="Scoring Responses")]
        best_responses = {}
        total_train_reward = 0
        total_null_responses = 0

        for i, item in enumerate(generation_data):
            prompt = item["prompt"]
            reward = rewards[i].item()

            if reward < 0.0:
                total_null_responses += 1
            else:
                total_train_reward += reward

            if prompt not in best_responses or reward > best_responses[prompt]["reward"]:
                best_responses[prompt] = {"response": item["response"], "reward": reward}
            log_response("generation", iteration, -1, i, prompt, item["response"], reward, item, experiment_dir)

        total_reward_for_best_responses = sum(d["reward"] for d in best_responses.values())
        
        total_non_null_samples = len(generation_data) - total_null_responses
        mean_train_reward = total_train_reward / total_non_null_samples if total_non_null_samples > 0 else 0
        mean_reward_for_best = total_reward_for_best_responses / len(best_responses) if best_responses else 0
        
        eval_results["total_train_reward"].append(total_train_reward)
        eval_results["mean_train_reward"].append(mean_train_reward)
        eval_results["total_train_reward_for_best_responses"].append(total_reward_for_best_responses)
        eval_results["mean_train_reward_for_best_responses"].append(mean_reward_for_best)
        
        logger.info(f"Total train reward: {total_train_reward}")
        logger.info(f"Mean train reward: {mean_train_reward}")
        logger.info(f"Total reward for best responses: {total_reward_for_best_responses}")
        logger.info(f"Mean reward for best responses: {mean_reward_for_best}")

        best_responses = {k: v for k, v in best_responses.items() if v["reward"] > config.filter_threshold}

        total_reward_for_best_responses_filtered = sum(v["reward"] for v in best_responses.values())
        mean_reward_for_best_filtered = total_reward_for_best_responses_filtered / len(best_responses) if best_responses else 0
        
        logger.info("AFTER FILTERING")
        logger.info(f"Total reward for best responses post-filtering: {total_reward_for_best_responses_filtered}")
        logger.info(f"Mean train reward for best responses post-filtering: {mean_reward_for_best_filtered}")

        # --- SFT Phase ---
        logger.info("Phase 3 & 4: Creating dataset and running SFT...")
        sft_dataset_list = [{"prompt": p, "response": d["response"]} for p, d in best_responses.items()]
        if not sft_dataset_list:
            logger.warning(f"No samples left after filtering in iteration {iteration + 1}. Skipping SFT.")
            continue

        sft_dataset = Dataset.from_list(sft_dataset_list)
        logger.info(f"Created new SFT dataset with {len(sft_dataset)} high-quality examples.")

        losses = run_sft_training_loop(model, tokenizer, sft_dataset, config, logger)
        if "sft_losses" not in eval_results:
            eval_results["sft_losses"] = {}
        eval_results["sft_losses"][iteration] = losses
        clear_cache()


        if config.push_to_hub_every_epoch:
            try:
                # When pushing, we push the merged model
                model.push_to_hub(f"{config.hub_model_id}-iteration-{iteration + 1}")
                tokenizer.push_to_hub(f"{config.hub_model_id}-iteration-{iteration + 1}")
            except Exception as e:
                logger.error(f"Error pushing model to hub: {e}")

        # --- Evaluation Phase ---
        logger.info(f"--- Evaluating model after Iteration {iteration + 1} ---")
        llm = None
        try:
            llm = get_vllm_model(model, tokenizer, vllm_kwargs=config.vllm_kwargs)
            with torch.no_grad():
                eval_results = evaluate_fn(
                    eval_results,
                    model,
                    tokenizer,
                    test_dataloader,
                    max_new_tokens,
                    epoch=iteration + 1,
                    llm=llm,
                    experiment_dir=experiment_dir,
                )
        finally:
            if llm is not None:
                del llm
            llm = None
            clear_cache()

        logger.info(f"\n=== Evaluation Results (Iteration {iteration + 1}) ===")
        if "mean_test_reward" in eval_results and eval_results["mean_test_reward"]:
            logger.info(f"Mean test reward: {eval_results['mean_test_reward'][-1]}")
        
        latest_eval_results = {k: v[-1] if isinstance(v, list) else v for k, v in eval_results.items()}
        
        metrics_to_log = {
            "iteration": iteration + 1,
            "sft_loss": losses[-1] if losses else None,
            "mean_train_reward": mean_train_reward,
            "mean_reward_for_best_responses": mean_reward_for_best,
            "mean_test_reward": latest_eval_results.get("mean_test_reward"),
            "total_test_reward": latest_eval_results.get("total_test_reward"),
            "frequency_test_null_reward": latest_eval_results.get("frequency_test_null_reward"),
        }
        
        if run:
            run.log({k: v for k, v in metrics_to_log.items() if v is not None})

    logger.info("Expert iteration training finished.")
    return model, eval_results


def log_response(phase, epoch, batch_idx, sample_idx, prompt, response, reward, batch_data, experiment_dir):
    if experiment_dir and not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)

    prompt_id = f"{phase}_epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}"
    log_entry = {
        "prompt_id": prompt_id,
        "phase": phase,
        "epoch_or_iteration": epoch,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "full_batch_data": {k: str(v) for k, v in batch_data.items()},
    }
    
    DETAILED_RESPONSES.append(log_entry)
    
    if experiment_dir:
        live_log_file_path = os.path.join(experiment_dir, "detailed_responses_live.json")
        try:
            with open(live_log_file_path, "w") as f:
                json.dump(DETAILED_RESPONSES, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to write to live log file: {e}")


def reward_fn(item: dict, max_tokens: int) -> torch.Tensor:
    reward = compute_reward(item["response"])
    return torch.tensor(reward)


def evaluate_model(
    eval_results: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    max_new_tokens: int,
    epoch: int,
    llm: Any,
    experiment_dir: Optional[str],
) -> dict:
    """
    Evaluate the model on the dataloader using vLLM for generation.
    """
    logger.info(f"\n--- Running evaluation for epoch {epoch} ---")

    for k in [
        "mean_test_reward",
        "total_test_reward",
        "frequency_test_null_reward",
        "total_test_null_reward",
        "total_test_samples",
    ]:
        if k not in eval_results:
            eval_results[k] = []

    new_eval_results = copy.deepcopy(eval_results)
    total_reward = 0.0
    total_non_null_samples = 0

    all_prompts = []
    all_original_data = []
    logger.info("Collecting prompts for evaluation...")
    for batch in tqdm(dataloader, desc="Preparing evaluation data"):
        prompts = batch["prompt"]
        all_prompts.extend(prompts)
        for i in range(len(prompts)):
            original_data = {k: v[i] for k, v in batch.items()}
            all_original_data.append(original_data)

    logger.info(f"Generating {len(all_prompts)} responses with vLLM for evaluation...")
    with torch.no_grad():
        all_responses = generate_with_vllm(
            model, tokenizer, all_prompts, max_new_tokens, llm=llm, vllm_kwargs=config.vllm_kwargs
        )

    rewards = [
        reward_fn({"response": response, **data}, max_tokens=max_new_tokens)
        for response, data in zip(all_responses, all_original_data)
    ]

    for i, reward_tensor in enumerate(rewards):
        reward = reward_tensor.item()
        if reward >= 0.0:
            total_reward += reward
            total_non_null_samples += 1

        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1
        log_response(
            phase="eval",
            epoch=epoch,
            batch_idx=i // batch_size,
            sample_idx=i % batch_size,
            prompt=all_prompts[i],
            response=all_responses[i],
            reward=reward,
            batch_data=all_original_data[i],
            experiment_dir=experiment_dir,
        )

    total_samples = len(all_responses)
    total_null_reward = total_samples - total_non_null_samples
    avg_reward = (total_reward / total_non_null_samples) if total_non_null_samples > 0 else 0.0
    avg_null_reward_freq = total_null_reward / total_samples if total_samples > 0 else 0.0

    new_eval_results["mean_test_reward"].append(avg_reward)
    new_eval_results["total_test_reward"].append(total_reward)
    new_eval_results["frequency_test_null_reward"].append(avg_null_reward_freq)
    new_eval_results["total_test_null_reward"].append(total_null_reward)
    new_eval_results["total_test_samples"].append(total_samples)

    summary = (
        f"\n=== Evaluation Summary (Epoch {epoch}) ===\n"
        f"Mean reward: {avg_reward:.3f}\n"
        f"Total reward: {total_reward:.3f}\n"
        f"Null response frequency: {avg_null_reward_freq:.3%}\n"
        f"Total samples: {total_samples}\n"
        f"Valid responses: {total_non_null_samples}\n"
        f"=====================================\n"
    )
    logger.info(summary)
    print(summary)

    return new_eval_results


def load_conversations_from_jsonl(file_path: str, num_samples: Optional[int] = None) -> list[list[dict]]:
    """Loads conversations from a .jsonl file."""
    conversations = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data and data["messages"]:
                    conversations.append(data["messages"])
                if num_samples is not None and num_samples > 0 and len(conversations) >= num_samples:
                    break
    except FileNotFoundError:
        logger.error(f"Dataset file not found at: {file_path}")
        return []
    return conversations


if __name__ == "__main__":
    config = ExpertIterationConfig()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    run = None
    if config.log_to_wandb:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=asdict(config),
            reinit=True,
        )
    else:
        class DummyRun:
            def log(self, data):
                logger.info(f"Logging (WandB disabled): {data}")
            def finish(self):
                pass
        run = DummyRun()

    model, tokenizer = load_model_and_tokenizer(config)

    conversations = load_conversations_from_jsonl(config.dataset_path, config.num_samples)
    if not conversations:
        logger.error("No conversations loaded. Exiting.")
    else:
        prompts = [
            tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False, enable_thinking=config.enable_thinking
            )
            for conv in conversations
        ]
        dataset = Dataset.from_dict({"prompt": prompts})
        
        train_dataloader = DataLoader(dataset, batch_size=config.mini_batch_size)  # type: ignore
        test_dataloader = DataLoader(dataset, batch_size=config.mini_batch_size)  # type: ignore

        train(
            model=model,  # type: ignore
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            run=run,
            logger=logger,
            config=config,
            evaluate_fn=evaluate_model,
            train_reward_fn=reward_fn,
            max_new_tokens=config.max_new_tokens,
            experiment_dir=config.experiment_dir,
        )

    if run:
        run.finish()

    logger.info("Training finished.") 