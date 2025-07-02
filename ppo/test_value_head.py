"""
Test script to verify value head functionality in PPO setup.
"""

import torch
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, TaskType
from config import MODEL_CONFIG, LORA_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_value_head():
    """Test if the value head is working correctly."""
    logger.info("Testing value head functionality...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model_name"],
        trust_remote_code=True,
        padding_side="left"
    )
    
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
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_CONFIG["model_name"],
        peft_config=lora_config,
        quantization_config=bnb_config,
        device_map=MODEL_CONFIG["device_map"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    logger.info("=== VALUE HEAD TEST ===")
    logger.info(f"Has v_head attribute: {hasattr(model, 'v_head')}")
    
    if hasattr(model, 'v_head'):
        logger.info(f"Value head type: {type(model.v_head)}")
        logger.info(f"Value head parameters: {list(model.v_head.parameters())}")
        
        # Test value head with dummy input
        try:
            # Create dummy hidden states (batch_size=2, seq_len=10, hidden_size)
            hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
            # Get device from model parameters
            device = next(model.parameters()).device
            dummy_hidden = torch.randn(2, 10, hidden_size, device=device, dtype=torch.bfloat16)
            
            logger.info(f"Dummy hidden states shape: {dummy_hidden.shape}")
            
            # Test value head forward pass
            with torch.no_grad():
                value_output = model.v_head(dummy_hidden)
                logger.info(f"Value head output shape: {value_output.shape}")
                logger.info(f"Value head output: {value_output}")
                logger.info(f"Value head output mean: {value_output.mean().item()}")
                logger.info(f"Value head output std: {value_output.std().item()}")
                
                # Check if output is non-zero
                if value_output.abs().sum().item() > 0:
                    logger.info("✓ Value head is producing non-zero outputs")
                else:
                    logger.warning("✗ Value head is producing zero outputs")
                    
        except Exception as e:
            logger.error(f"Error testing value head: {e}")
    
    # Test full model forward pass
    logger.info("=== FULL MODEL TEST ===")
    try:
        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randint(0, tokenizer.vocab_size, (2, 10), device=device)
        logger.info(f"Dummy input shape: {dummy_input.shape}")
        
        # Test model forward pass
        with torch.no_grad():
            outputs = model(dummy_input, return_dict=True)
            logger.info(f"Output type: {type(outputs)}")
            
            if isinstance(outputs, dict):
                logger.info(f"Available output attributes: {list(outputs.keys())}")
                
                if 'value' in outputs:
                    value_pred = outputs['value']
                    logger.info(f"Model value prediction shape: {value_pred.shape}")
                    logger.info(f"Model value prediction: {value_pred}")
                    logger.info(f"Model value prediction mean: {value_pred.mean().item()}")
                    logger.info(f"Model value prediction std: {value_pred.std().item()}")
                    
                    if value_pred.abs().sum().item() > 0:
                        logger.info("✓ Model is producing non-zero value predictions")
                    else:
                        logger.warning("✗ Model is producing zero value predictions")
                else:
                    logger.warning("No 'value' attribute in model outputs")
            else:
                logger.info(f"Output is not a dict, it's a {type(outputs)}")
                logger.info(f"Output: {outputs}")
                
                # Try to access value from tuple output
                if hasattr(outputs, 'value'):
                    value_pred = outputs.value
                    logger.info(f"Model value prediction (from attribute): {value_pred}")
                else:
                    logger.info("No value attribute found in output")
                
    except Exception as e:
        logger.error(f"Error testing full model: {e}")
    
    # Test score method
    logger.info("=== SCORE METHOD TEST ===")
    try:
        # Add score method if it doesn't exist
        if not hasattr(model, 'score'):
            def score_fn(hidden_states):
                return model.v_head(hidden_states)
            model.score = score_fn
            logger.info("Added score method to model")
        
        # Test score method with dummy hidden states
        device = next(model.parameters()).device
        dummy_hidden = torch.randn(2, 10, hidden_size, device=device, dtype=torch.bfloat16)
        score_output = model.score(dummy_hidden)
        logger.info(f"Score method output shape: {score_output.shape}")
        logger.info(f"Score method output: {score_output}")
        
        if score_output.abs().sum().item() > 0:
            logger.info("✓ Score method is producing non-zero outputs")
        else:
            logger.warning("✗ Score method is producing zero outputs")
            
    except Exception as e:
        logger.error(f"Error testing score method: {e}")

if __name__ == "__main__":
    test_value_head() 