"""
Minimal test script to debug PPO value head and loss calculation.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from peft import LoraConfig, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_value_head_basic():
    """Test basic value head functionality without quantization."""
    logger.info("=== TESTING BASIC VALUE HEAD ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test without quantization first
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "Qwen/Qwen3-4B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Has v_head: {hasattr(model, 'v_head')}")
        
        if hasattr(model, 'v_head'):
            logger.info(f"v_head type: {type(model.v_head)}")
            logger.info(f"v_head parameters: {list(model.v_head.parameters())}")
            
            # Test value head computation
            dummy_input = torch.randn(1, 10, model.config.hidden_size)
            with torch.no_grad():
                values = model.v_head(dummy_input)
                logger.info(f"Value head output shape: {values.shape}")
                logger.info(f"Value head output: {values}")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error in basic test: {e}")

def test_value_head_quantized():
    """Test value head with quantization."""
    logger.info("=== TESTING QUANTIZED VALUE HEAD ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    try:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            "Qwen/Qwen3-4B",
            peft_config=lora_config,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Has v_head: {hasattr(model, 'v_head')}")
        
        if hasattr(model, 'v_head'):
            logger.info(f"v_head type: {type(model.v_head)}")
            logger.info(f"v_head parameters: {list(model.v_head.parameters())}")
            logger.info(f"v_head requires_grad: {[p.requires_grad for p in model.v_head.parameters()]}")
            
            # Test if value head is trainable
            model.train()
            logger.info(f"v_head training mode: {model.v_head.training}")
            
            # Test value head computation (this might fail with quantization)
            try:
                dummy_input = torch.randn(1, 10, model.config.hidden_size, device=model.device)
                with torch.no_grad():
                    values = model.v_head(dummy_input)
                    logger.info(f"Value head output shape: {values.shape}")
                    logger.info(f"Value head output: {values}")
            except Exception as e:
                logger.error(f"Error testing value head computation: {e}")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error in quantized test: {e}")

def test_ppo_config():
    """Test PPO configuration."""
    logger.info("=== TESTING PPO CONFIG ===")
    
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        seed=42,
        cliprange=0.2,
        vf_coef=0.1,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
        whiten_rewards=False,
        max_grad_norm=1.0,
    )
    
    logger.info(f"PPO config: {config}")
    logger.info(f"vf_coef: {config.vf_coef}")
    logger.info(f"cliprange: {config.cliprange}")
    logger.info(f"cliprange_value: {config.cliprange_value}")

if __name__ == "__main__":
    test_ppo_config()
    test_value_head_basic()
    test_value_head_quantized() 