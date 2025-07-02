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

# Check TRL version
import trl
logger = logging.getLogger(__name__)
logger.info(f"TRL version: {trl.__version__}")

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
    
    # DEBUG: Check value head existence and structure
    logger.info("=== VALUE HEAD DEBUGGING ===")
    logger.info(f"Has v_head attribute: {hasattr(actor_model, 'v_head')}")
    if hasattr(actor_model, 'v_head'):
        logger.info(f"v_head type: {type(actor_model.v_head)}")
        logger.info(f"v_head parameters: {list(actor_model.v_head.parameters())}")
        logger.info(f"v_head requires_grad: {[p.requires_grad for p in actor_model.v_head.parameters()]}")
        # Test value head with dummy input
        try:
            dummy_hidden = torch.randn(1, 10, actor_model.config.hidden_size, device=actor_model.device)
            with torch.no_grad():
                dummy_value = actor_model.v_head(dummy_hidden)
                logger.info(f"v_head output shape: {dummy_value.shape}")
                logger.info(f"v_head output: {dummy_value}")
        except Exception as e:
            logger.error(f"Error testing v_head: {e}")
    
    # Check if value head is trainable
    logger.info(f"v_head training mode: {actor_model.v_head.training if hasattr(actor_model, 'v_head') else 'N/A'}")
    
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
    
    # DEBUG: Validate PPO configuration
    logger.info("=== PPO CONFIG VALIDATION ===")
    logger.info(f"TRL version: {trl.__version__}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Mini batch size: {config.mini_batch_size}")
    logger.info(f"PPO epochs: {config.ppo_epochs}")
    logger.info(f"vf_coef: {config.vf_coef}")
    logger.info(f"cliprange: {config.cliprange}")
    logger.info(f"cliprange_value: {config.cliprange_value}")
    logger.info(f"gamma: {config.gamma}")
    logger.info(f"lam: {config.lam}")
    logger.info(f"whiten_rewards: {config.whiten_rewards}")
    logger.info(f"max_grad_norm: {config.max_grad_norm}")
    
    # Check for potential issues
    if config.vf_coef == 0.0:
        logger.warning("WARNING: vf_coef is 0.0 - this will disable value function learning!")
    if config.cliprange == 0.0:
        logger.warning("WARNING: cliprange is 0.0 - this will disable policy clipping!")
    if config.cliprange_value == 0.0:
        logger.warning("WARNING: cliprange_value is 0.0 - this will disable value function clipping!")
    if config.learning_rate == 0.0:
        logger.warning("WARNING: learning_rate is 0.0 - this will disable all learning!")
    
    logger.info("=== END PPO CONFIG VALIDATION ===")
    
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
            
            # PPO step with proper argument format
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # DEBUG: Detailed loss component analysis
            if step % 5 == 0:  # Debug every 5 steps
                logger.info("=== PPO LOSS DEBUGGING ===")
                logger.info(f"Step {step} stats: {stats}")
                
                # Check if stats is empty or all zeros
                all_zero = all(v == 0.0 for v in stats.values() if isinstance(v, (int, float)))
                logger.info(f"All stats zero: {all_zero}")
                
                # Check reward statistics
                reward_values = [r.item() for r in rewards]
                logger.info(f"Reward stats - mean: {sum(reward_values)/len(reward_values):.4f}, "
                          f"std: {torch.std(torch.stack(rewards)).item():.4f}, "
                          f"min: {min(reward_values):.4f}, max: {max(reward_values):.4f}")
                
                # Check if rewards have variance (should trigger learning)
                reward_variance = torch.var(torch.stack(rewards)).item()
                logger.info(f"Reward variance: {reward_variance:.6f}")
                
                # Check model parameters for gradients
                if hasattr(ppo_trainer.model, 'v_head'):
                    v_head_params = list(ppo_trainer.model.v_head.parameters())
                    v_head_grads = [p.grad for p in v_head_params if p.grad is not None]
                    logger.info(f"v_head parameters with gradients: {len(v_head_grads)}/{len(v_head_params)}")
                    if v_head_grads:
                        v_head_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in v_head_grads]))
                        logger.info(f"v_head gradient norm: {v_head_grad_norm:.6f}")
                
                # Check if model is in training mode
                logger.info(f"Model training mode: {ppo_trainer.model.training}")
                logger.info(f"Model v_head training mode: {ppo_trainer.model.v_head.training if hasattr(ppo_trainer.model, 'v_head') else 'N/A'}")
                
                # Check if optimizer is configured
                if hasattr(ppo_trainer, 'optimizer'):
                    logger.info(f"Optimizer exists: {ppo_trainer.optimizer is not None}")
                    if ppo_trainer.optimizer is not None:
                        logger.info(f"Optimizer learning rate: {ppo_trainer.optimizer.param_groups[0]['lr']}")
                
                # Check PPO config values
                logger.info(f"PPO config - learning_rate: {config.learning_rate}")
                logger.info(f"PPO config - vf_coef: {config.vf_coef}")
                logger.info(f"PPO config - cliprange: {config.cliprange}")
                logger.info(f"PPO config - cliprange_value: {config.cliprange_value}")
                
                # Try to access internal PPO state if possible
                if hasattr(ppo_trainer, '_ppo_eval'):
                    logger.info(f"PPO eval exists: {ppo_trainer._ppo_eval is not None}")
                if hasattr(ppo_trainer, 'store'):
                    logger.info(f"PPO store exists: {ppo_trainer.store is not None}")
                    if ppo_trainer.store is not None:
                        logger.info(f"Store size: {len(ppo_trainer.store)}")
                
                # Call manual debugging function
                debug_ppo_loss_calculation(ppo_trainer, query_tensors, response_tensors, rewards, step)
                
                logger.info("=== END PPO LOSS DEBUGGING ===")
            
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

def debug_ppo_loss_calculation(ppo_trainer, query_tensors, response_tensors, rewards, step):
    """
    Debug function to manually examine PPO loss components.
    """
    logger.info("=== MANUAL PPO LOSS DEBUGGING ===")
    
    try:
        # Check if we can access the internal PPO components
        if hasattr(ppo_trainer, '_ppo_eval'):
            logger.info("PPO eval found, examining components...")
            
            # Try to access the store and examine its contents
            if hasattr(ppo_trainer, 'store') and ppo_trainer.store is not None:
                logger.info(f"Store size: {len(ppo_trainer.store)}")
                if len(ppo_trainer.store) > 0:
                    # Examine the first item in the store
                    first_item = ppo_trainer.store[0]
                    logger.info(f"First store item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                    
                    # Check for key components
                    if isinstance(first_item, dict):
                        for key in ['logprobs', 'values', 'rewards', 'advantages']:
                            if key in first_item:
                                val = first_item[key]
                                if isinstance(val, torch.Tensor):
                                    logger.info(f"{key} shape: {val.shape}, mean: {val.mean().item():.4f}, std: {val.std().item():.4f}")
                                else:
                                    logger.info(f"{key} type: {type(val)}, value: {val}")
        
        # Check if the model can compute values
        if hasattr(ppo_trainer.model, 'v_head'):
            logger.info("Testing value head computation...")
            
            # Try to compute values for the responses
            try:
                # Get the full sequences (query + response)
                full_sequences = []
                for i, (query, response) in enumerate(zip(query_tensors, response_tensors)):
                    if isinstance(query, torch.Tensor):
                        query = query.unsqueeze(0) if query.dim() == 1 else query
                    if isinstance(response, torch.Tensor):
                        response = response.unsqueeze(0) if response.dim() == 1 else response
                    
                    # Concatenate query and response
                    full_seq = torch.cat([query, response], dim=-1)
                    full_sequences.append(full_seq)
                
                # Test value computation on first sequence
                if full_sequences:
                    test_seq = full_sequences[0]
                    logger.info(f"Test sequence shape: {test_seq.shape}")
                    
                    # Get hidden states (this might not work with quantization)
                    with torch.no_grad():
                        try:
                            # Try to get hidden states from the model
                            outputs = ppo_trainer.model.pretrained_model(test_seq, output_hidden_states=True)
                            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                                last_hidden = outputs.hidden_states[-1]
                                logger.info(f"Last hidden state shape: {last_hidden.shape}")
                                
                                # Test value head
                                values = ppo_trainer.model.v_head(last_hidden)
                                logger.info(f"Computed values shape: {values.shape}")
                                logger.info(f"Computed values: {values}")
                            else:
                                logger.info("Could not get hidden states from model")
                        except Exception as e:
                            logger.info(f"Error getting hidden states: {e}")
                            
            except Exception as e:
                logger.info(f"Error testing value computation: {e}")
        
        # Check if the PPO trainer has the expected methods
        logger.info(f"PPO trainer methods: {[m for m in dir(ppo_trainer) if not m.startswith('_')]}")
        
        # Check if the step method is working correctly
        logger.info(f"Step method signature: {ppo_trainer.step.__code__.co_varnames}")
        
    except Exception as e:
        logger.error(f"Error in manual PPO debugging: {e}")
    
    logger.info("=== END MANUAL PPO LOSS DEBUGGING ===")

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