"""
Main script for PPO training using TRL.
"""

import os
import sys
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    default_data_collator
)
from transformers.generation import LogitsProcessor
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import wandb
from typing import Optional
import json

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PPO_CONFIG, MODEL_CONFIG, LORA_CONFIG, DATASET_CONFIG, INFERENCE_CONFIG, get_config_for_gpu
from data_utils import (
    load_json_dataset, 
    filter_valid_conversations, 
    prepare_dataset_for_ppo,
    create_reward_function,
    analyze_capitalization_distribution,
    calculate_capitalization_reward
)
from reward_model import create_reward_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ppo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    Handles multi-token </think> and \n, and avoids CUDA errors.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        # Get token ids (may be more than one!)
        self.think_end_tokens = self.tokenizer.encode("</think>", add_special_tokens=False)
        self.nl_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
        self.tokens_generated = 0
        self.stopped_thinking = False
        # Use a large negative value instead of -inf to avoid sampling issues
        self.neg_inf = -1e10

    def _set_token_score(self, scores, token_ids, value):
        for tid in token_ids:
            if tid < scores.shape[1]:
                scores[0][tid] = value
                # Also set a small positive value to ensure it's selected when all scores are 0
                if value == 0.0:
                    scores[0][tid] = 1.0

    def _set_all_scores_to_neg_inf(self, scores):
        # Set all scores to a large negative value, but keep valid tokens
        scores[:] = self.neg_inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1
        
        # Debug logging for first few tokens
        if self.tokens_generated <= 10:
            print(f"  Thinking processor: token {self.tokens_generated}, max_thinking_tokens={self.max_thinking_tokens}")
        
        if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
            print(f"  Forcing immediate stop at token {self.tokens_generated}")
            self._set_all_scores_to_neg_inf(scores)
            self._set_token_score(scores, self.nl_tokens, 0.0)
            self._set_token_score(scores, self.think_end_tokens, 0.0)
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            if (self.tokens_generated / self.max_thinking_tokens) > 0.8:
                # Boost the probability of ending tokens as we approach the limit
                boost_factor = 1.0 + (self.tokens_generated / self.max_thinking_tokens)
                for tid in self.nl_tokens:
                    if tid < scores.shape[1]:
                        scores[0][tid] *= boost_factor
                for tid in self.think_end_tokens:
                    if tid < scores.shape[1]:
                        scores[0][tid] *= boost_factor

            # Force newline at max_thinking_tokens - 2 (e.g., token 6 for max_thinking_tokens=8)
            if self.tokens_generated == self.max_thinking_tokens - 2:
                print(f"  Forcing newline at token {self.tokens_generated}")
                self._set_all_scores_to_neg_inf(scores)
                self._set_token_score(scores, self.nl_tokens, 0.0)
            # Force end thinking at max_thinking_tokens - 1 (e.g., token 7 for max_thinking_tokens=8)
            elif self.tokens_generated >= self.max_thinking_tokens - 1:
                print(f"  Forcing </think> at token {self.tokens_generated}")
                self._set_all_scores_to_neg_inf(scores)
                self._set_token_score(scores, self.think_end_tokens, 0.0)
                self.stopped_thinking = True

        return scores

def setup_models_and_tokenizer():
    """
    Set up the actor model, reference model, and tokenizer.
    
    Returns:
        Tuple of (actor_model, ref_model, tokenizer)
    """
    logger.info("Setting up models and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"],
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure padding side is set correctly
    tokenizer.padding_side = "left"
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    

    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create model with value head and LoRA in one step
    # This is the correct TRL pattern according to the examples
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_CONFIG["model_name"],
        peft_config=lora_config,
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # DEBUG: Check if value head exists and print its structure
    logger.info("=== VALUE HEAD DEBUGGING ===")
    logger.info(f"Has v_head attribute: {hasattr(actor_model, 'v_head')}")
    if hasattr(actor_model, 'v_head'):
        logger.info(f"Value head type: {type(actor_model.v_head)}")
        logger.info(f"Value head parameters: {list(actor_model.v_head.parameters())}")
        # Don't call summary() as it requires forward pass
        logger.info(f"Value head structure: {actor_model.v_head}")
    
    # Check if the model has the required TRL attributes
    logger.info(f"Has base_model_prefix: {hasattr(actor_model, 'base_model_prefix')}")
    if hasattr(actor_model, 'base_model_prefix'):
        logger.info(f"base_model_prefix: {actor_model.base_model_prefix}")
    
    # Ensure generation_config is available by loading it from the tokenizer if needed
    if not hasattr(actor_model, 'generation_config') or actor_model.generation_config is None:
        from transformers import GenerationConfig
        actor_model.generation_config = GenerationConfig.from_pretrained(MODEL_CONFIG["model_name"])
    
    # Set base_model_prefix for compatibility with TRL
    if not hasattr(actor_model, 'base_model_prefix'):
        actor_model.base_model_prefix = 'model'  # Qwen models use 'model' as prefix
    
    # Ensure the model has the required attribute structure for TRL
    if not hasattr(actor_model, actor_model.base_model_prefix):
        # Get the actual model from the wrapped structure
        if hasattr(actor_model, 'pretrained_model'):
            setattr(actor_model, actor_model.base_model_prefix, actor_model.pretrained_model)
        elif hasattr(actor_model, 'base_model'):
            setattr(actor_model, actor_model.base_model_prefix, actor_model.base_model)
        else:
            logger.warning(f"Could not find base model to set {actor_model.base_model_prefix} attribute")
    
    # Add score method for value function calculation (required by TRL)
    if not hasattr(actor_model, 'score'):
        def score_fn(hidden_states):
            # Use the value head for scoring
            return actor_model.v_head(hidden_states)
        actor_model.score = score_fn
    
    # Explicitly set model to train mode
    actor_model.train()
    
    # Create reference model (copy of base model without value head)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    logger.info("Models and tokenizer setup complete")
    return actor_model, ref_model, tokenizer

def prepare_dataset(tokenizer):
    """
    Prepare the dataset for PPO training.
    
    Args:
        tokenizer: The tokenizer to use
        
    Returns:
        Dataset ready for PPO training
    """
    logger.info("Preparing dataset...")
    
    # Load raw dataset
    raw_data = load_json_dataset(
        DATASET_CONFIG["dataset_path"],
        max_samples=DATASET_CONFIG["max_samples"]
    )
    
    # Filter valid conversations
    valid_data = filter_valid_conversations(raw_data)
    
    # Prepare for PPO
    dataset = prepare_dataset_for_ppo(
        valid_data,
        tokenizer,
        max_length=DATASET_CONFIG["max_length"],
        truncation=DATASET_CONFIG["truncation"]
    )
    
    logger.info(f"Dataset prepared: {len(dataset)} samples")
    return dataset

def create_ppo_trainer(
    actor_model,
    ref_model,
    tokenizer,
    train_dataset,
    config
):
    """
    Create the PPO trainer (TRL 0.11.4+ style, no reward_model argument).
    """
    logger.info(f"Tokenizer padding_side before PPOTrainer: {tokenizer.padding_side}")
    
    def left_pad_collator(features):
        tokenizer.padding_side = "left"
        return default_data_collator(features)
    
    ppo_trainer = PPOTrainer(
        config,
        actor_model,
        ref_model,
        tokenizer,
        dataset=train_dataset,
        data_collator=left_pad_collator,
    )
    
    # DEBUG: Check PPO trainer configuration
    logger.info("=== PPO TRAINER DEBUGGING ===")
    logger.info(f"PPO config: {config}")
    logger.info(f"Has value function: {hasattr(ppo_trainer, 'value_function')}")
    logger.info(f"Has value head: {hasattr(ppo_trainer.model, 'v_head')}")
    logger.info(f"Model training mode: {ppo_trainer.model.training}")
    
    # Check if the model has the required methods
    logger.info(f"Model has forward method: {hasattr(ppo_trainer.model, 'forward')}")
    logger.info(f"Model has score method: {hasattr(ppo_trainer.model, 'score')}")
    
    # Check reference model
    if hasattr(ppo_trainer, 'ref_model') and ppo_trainer.ref_model is not None:
        logger.info(f"Reference model training mode: {ppo_trainer.ref_model.training}")
    else:
        logger.warning("Reference model is None")
    
    logger.info("PPO trainer created successfully")
    return ppo_trainer

def train_ppo(
    ppo_trainer,
    tokenizer,
    config,
    max_steps: Optional[int] = None
):
    """
    Run PPO training loop using TRL's proper training approach.
    """
    logger.info("Starting PPO training...")
    from tqdm import tqdm
    import wandb

    # Use TRL's built-in dataloader
    dataloader = ppo_trainer.dataloader
    total_steps = max_steps or config.steps or len(dataloader)
    step = 0
    
    # Generation kwargs from config (without logits_processor - we'll add it per generation)
    generation_kwargs = {
        "max_new_tokens": INFERENCE_CONFIG["max_new_tokens"],
        "do_sample": INFERENCE_CONFIG["do_sample"],
        "temperature": INFERENCE_CONFIG["temperature"],
        "top_p": INFERENCE_CONFIG["top_p"],
        "top_k": INFERENCE_CONFIG["top_k"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    
    for epoch in range(getattr(config, 'ppo_epochs', 1)):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            if step >= total_steps and total_steps > 0:
                break
                
            # Get query tensors from batch
            query_tensors = batch["input_ids"]
            
            # Convert batch tensor to list of individual tensors for TRL
            if query_tensors.dim() > 1:
                query_tensors = [query_tensors[i] for i in range(query_tensors.size(0))]
            
            # Log model training mode
            logger.info(f"Step {step}: actor_model.training={ppo_trainer.model.training}")
            
            # Generate responses using TRL's generate method
            # We need to generate each sample individually to ensure proper thinking processor behavior
            response_tensors = []
            for i, query_tensor in enumerate(query_tensors):
                # Create a fresh thinking processor for each individual generation
                thinking_processor = ThinkingTokenBudgetProcessor(tokenizer, max_thinking_tokens=INFERENCE_CONFIG["max_thinking_tokens"])
                individual_generation_kwargs = generation_kwargs.copy()
                individual_generation_kwargs["logits_processor"] = [thinking_processor]
                
                # Generate response for this single query
                individual_response = ppo_trainer.generate(
                    [query_tensor],  # Single query as list
                    return_prompt=False,
                    **individual_generation_kwargs
                )
                response_tensors.extend(individual_response)
                
                # Log thinking processor behavior for debugging
                if step % 10 == 0 and i < 2:  # Log first 2 samples every 10 steps
                    response_text = tokenizer.decode(individual_response[0].squeeze(), skip_special_tokens=True)
                    logger.info(f"Step {step}, Sample {i}: max_thinking_tokens={INFERENCE_CONFIG['max_thinking_tokens']}, response_length={len(response_text)}, has_think_end={'</think>' in response_text}")
            
            # Decode responses for reward calculation
            responses_text = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Compute rewards (proportion of capitalized letters)
            rewards = []
            for response_text in responses_text:
                reward = calculate_capitalization_reward(response_text)
                rewards.append(torch.tensor(reward, dtype=torch.float32))
            
            # Move rewards to correct device (keep as list of tensors for TRL)
            device = query_tensors[0].device if isinstance(query_tensors, list) else query_tensors.device
            rewards = [r.to(device) for r in rewards]
            
            # DEBUG: Check value predictions before PPO step
            if step % 10 == 0:  # Debug every 10 steps
                logger.info("=== PPO STEP DEBUGGING ===")
                logger.info(f"Step {step}:")
                logger.info(f"Query tensors shape: {[q.shape for q in query_tensors]}")
                logger.info(f"Response tensors shape: {[r.shape for r in response_tensors]}")
                logger.info(f"Rewards: {[r.item() for r in rewards]}")
                
                # Check if we can get value predictions from the model
                try:
                    # Try to get value predictions for a sample
                    sample_query = query_tensors[0].unsqueeze(0) if isinstance(query_tensors, list) else query_tensors[0:1]
                    sample_response = response_tensors[0].unsqueeze(0) if isinstance(response_tensors, list) else response_tensors[0:1]
                    
                    # Concatenate query and response
                    full_sequence = torch.cat([sample_query, sample_response], dim=-1)
                    
                    # Get value prediction
                    with torch.no_grad():
                        outputs = ppo_trainer.model(full_sequence, return_dict=True)
                        logger.info(f"Model output (tuple): {outputs}")
                        if isinstance(outputs, tuple):
                            for i, out in enumerate(outputs):
                                logger.info(f"Output tuple[{i}]: {out}")
                        if hasattr(outputs, 'value'):
                            value_pred = outputs.value
                            logger.info(f"Value prediction shape: {value_pred.shape}")
                            logger.info(f"Value prediction: {value_pred}")
                        else:
                            logger.warning("No 'value' attribute in model outputs")
                            logger.info(f"Available output attributes: {dir(outputs)}")
                except Exception as e:
                    logger.error(f"Error getting value predictions: {e}")
            
            # PPO step with proper argument format
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # DEBUG: Print detailed stats
            if step % 10 == 0:
                logger.info("=== PPO STATS DEBUGGING ===")
                logger.info(f"Available stats keys: {list(stats.keys())}")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
            
            # Log to wandb
            if wandb.run is not None:
                reward_values = [r.item() for r in rewards]
                mean_reward = sum(reward_values) / len(reward_values)
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    "mean_reward": mean_reward,
                    "min_reward": min(reward_values),
                    "max_reward": max(reward_values),
                    "std_reward": torch.std(torch.stack(rewards)).item(),
                    "ppo_loss": stats.get("ppo_loss", 0.0),
                    "value_loss": stats.get("value_loss", 0.0),
                    "policy_loss": stats.get("policy_loss", 0.0),
                    "entropy": stats.get("entropy", 0.0),
                    "kl_div": stats.get("kl_div", 0.0),
                    "kl_coef": stats.get("kl_coef", 0.0),
                    "total_loss": stats.get("total_loss", 0.0),
                })
            
            # Save rollouts every 10 steps
            if step % 10 == 0:
                rollout_records = []
                for i in range(len(responses_text)):
                    # Decode the original input (without chat template formatting)
                    original_input = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True) if batch["input_ids"].dim() > 1 else tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
                    
                    rollout_records.append({
                        "step": step,
                        "original_input": original_input,
                        "response": responses_text[i]
                    })
                with open("ppo_rollouts.jsonl", "a") as f:
                    for record in rollout_records:
                        f.write(json.dumps(record) + "\n")
            
            if step % getattr(config, 'logging_steps', 10) == 0:
                logger.info(f"Step {step}: mean reward={mean_reward:.3f}, ppo_loss={stats.get('ppo_loss', 0.0):.4f}")
            step += 1
    logger.info("PPO training completed")

def main():
    """
    Main training function.
    """
    logger.info("Starting PPO training setup...")
    config = get_config_for_gpu("a100")  # Optimized for A100
    
    # Initialize wandb
    if config.log_with == "wandb":
        wandb.init(
            project="qwen3-ppo-uppercase",
            name=config.exp_name,
            config=vars(config)
        )
    
    try:
        actor_model, ref_model, tokenizer = setup_models_and_tokenizer()
        train_dataset = prepare_dataset(tokenizer)
        ppo_trainer = create_ppo_trainer(
            actor_model,
            ref_model,
            tokenizer,
            train_dataset,
            config
        )
        # Train on full dataset (remove max_steps=1 limit)
        train_ppo(ppo_trainer, tokenizer, config)
        # Save final model to hardcoded output directory
        output_dir = "./ppo_output/final"
        ppo_trainer.save_pretrained(output_dir)
        logger.info(f"Final model saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if config.log_with == "wandb":
            wandb.finish()

if __name__ == "__main__":
    main() 