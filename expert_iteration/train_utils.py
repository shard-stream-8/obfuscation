import gc
import re
import shutil
import tempfile
import os
from typing import Any, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

logger = __import__("logging").getLogger(__name__)


def load_model_and_tokenizer(config):
    """
    Loads the model and tokenizer.
    1. If resuming, merges the adapter into a new base model saved in a temporary directory.
    2. If LoRA is enabled for the current run, applies a new adapter to the base model.
    """
    bnb_config = None
    if config.quantization_config and config.quantization_config.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, config.quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        )

    base_model_path = config.model_name
    temp_dir = None

    if config.resume_from_checkpoint and config.checkpoint_model_id:
        try:
            logger.info("Resuming from checkpoint: creating temporary merged model...")
            temp_dir = tempfile.mkdtemp()

            base_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

            peft_model = PeftModel.from_pretrained(base_model, config.checkpoint_model_id)
            merged_model = peft_model.merge_and_unload()
            logger.info(f"Saving merged model to temporary directory: {temp_dir}")
            merged_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            base_model_path = temp_dir
            del base_model, peft_model, merged_model
            clear_cache()
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}. Loading as a normal model.")
            base_model_path = config.checkpoint_model_id

    peft_config = None
    if config.use_lora:
        logger.info("LoRA is enabled for this run. Preparing new PEFT config.")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    logger.info(f"Loading final model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if config.use_lora and peft_config:
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if temp_dir and os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

from vllm import LLM, SamplingParams
from pathlib import Path
import logging
import json
import atexit
import copy
def convert_model_to_dtype_completely(model, target_dtype=torch.float16):
    """
    Properly convert ALL model tensors to target dtype.
    This is more thorough than just model.to(dtype=target_dtype)
    """
    print(f"Converting model to {target_dtype}")

    # Convert all parameters
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            print(f"Converting parameter {name}: {param.dtype} -> {target_dtype}")
            param.data = param.data.to(target_dtype)

    # Convert all buffers (this is often missed!)
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
            print(f"Converting buffer {name}: {buffer.dtype} -> {target_dtype}")
            buffer.data = buffer.data.to(target_dtype)

    # Also convert the model itself (catches any remaining tensors)
    model = model.to(dtype=target_dtype)

    return model


def validate_model_dtype_consistency(model, expected_dtype=torch.float16):
    """
    Check if all model tensors have consistent dtypes
    """
    print("\n=== Validating model dtype consistency ===")

    param_dtypes = {}
    buffer_dtypes = {}

    # Check parameters
    for name, param in model.named_parameters():
        dtype_str = str(param.dtype)
        if dtype_str not in param_dtypes:
            param_dtypes[dtype_str] = []
        param_dtypes[dtype_str].append(name)

    # Check buffers
    for name, buffer in model.named_buffers():
        dtype_str = str(buffer.dtype)
        if dtype_str not in buffer_dtypes:
            buffer_dtypes[dtype_str] = []
        buffer_dtypes[dtype_str].append(name)

    print(f"Parameter dtypes: {list(param_dtypes.keys())}")
    print(f"Buffer dtypes: {list(buffer_dtypes.keys())}")

    # Check for inconsistencies
    all_floating_dtypes = set()
    for dtype_str in param_dtypes.keys():
        if "float" in dtype_str.lower():
            all_floating_dtypes.add(dtype_str)
    for dtype_str in buffer_dtypes.keys():
        if "float" in dtype_str.lower():
            all_floating_dtypes.add(dtype_str)
    if len(all_floating_dtypes) > 1:
        print(f"❌ DTYPE MISMATCH DETECTED: {all_floating_dtypes}")
        print("This will cause vLLM errors!")

        # Show which tensors have which dtypes
        for dtype_str, names in param_dtypes.items():
            if len(names) <= 5:
                print(f"  Parameters with {dtype_str}: {names}")
            else:
                print(f"  Parameters with {dtype_str}: {names[:5]}... ({len(names)} total)")

        for dtype_str, names in buffer_dtypes.items():
            if len(names) <= 5:
                print(f"  Buffers with {dtype_str}: {names}")
            else:
                print(f"  Buffers with {dtype_str}: {names[:5]}... ({len(names)} total)")

        return False
    else:
        print(f"✅ All floating point tensors have consistent dtype: {all_floating_dtypes}")
        return True


def get_vllm_model(
    hf_model,
    hf_tokenizer,
    vllm_kwargs: dict[str, Any] | None = None,
    cleanup_on_exit: bool = True,
    temp_dir: str | Path | None = None,
) -> LLM:
    """
    Fixed version that properly handles dtype conversion
    """

    # Set defaults with consistent dtype
    if vllm_kwargs is None:
        vllm_kwargs = {}

    # # Ensure we use float16 consistently
    target_dtype = torch.float16
    # vllm_kwargs.setdefault("dtype", "float16")  # Tell vLLM to use float16
    # vllm_kwargs.setdefault("gpu_memory_utilization", 0.7)
    # vllm_kwargs.setdefault("enforce_eager", True)  # May help with dtype issues

    print(f"vLLM kwargs: {vllm_kwargs}")

    # Handle PEFT models
    if hasattr(hf_model, "peft_config"):
        logging.info("PEFT model detected, merging adapters...")
        merged_model = copy.deepcopy(hf_model)
        merged_model = merged_model.merge_and_unload()
        logging.info("PEFT adapters merged successfully.")
    else:
        merged_model = hf_model
    # CRITICAL: Properly convert ALL tensors to target dtype
    try:
        print("Converting model to consistent dtype...")
        merged_model = convert_model_to_dtype_completely(merged_model, target_dtype)
    except Exception as e:
        print(f"Error converting model to consistent dtype: {e}")
        print("might be fine")
    # Validate dtype consistency before saving
    try:
        is_consistent = validate_model_dtype_consistency(merged_model, target_dtype)
        if not is_consistent:
            raise RuntimeError("Model has inconsistent dtypes after conversion!")
    except Exception as e:
        print(f"Error validating model dtype consistency: {e}")
        print("might be fine")

    # Create temporary directory
    if temp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="vllm_model_"))
    else:
        tmp_dir = Path(temp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Save model and tokenizer
        logging.info(f"Saving model to {tmp_dir}")

        # Save with explicit dtype specification
        merged_model.save_pretrained(
            tmp_dir,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="2GB",
        )
        hf_tokenizer.save_pretrained(tmp_dir)

        # IMPORTANT: Update the config.json to specify the correct dtype
        config_path = tmp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            # Set the dtype in config to match our tensors
            config["torch_dtype"] = "float16"

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print("Updated config.json with torch_dtype: float16")

        # Clean up PEFT model copy if created
        if hasattr(hf_model, "peft_config"):
            del merged_model
            gc.collect()
            torch.cuda.empty_cache()

        # Create vLLM model with error handling
        logging.info("Creating vLLM model...")
        try:
            llm = LLM(model=str(tmp_dir), **vllm_kwargs)
        except Exception as e:
            print(f"vLLM creation failed with error: {e}")

            # Try with more conservative setting[s
            print("Retrying with more conservative vLLM settings...")
            conservative_kwargs = {
                "dtype": "float16",
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
                "disable_custom_all_reduce": True,
            }
            llm = LLM(model=str(tmp_dir), **conservative_kwargs)

        # Register cleanup function if requested
        if cleanup_on_exit:

            def cleanup():
                try:
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
                        logging.info(f"Cleaned up temporary directory: {tmp_dir}")
                except Exception as e:
                    logging.warning(f"Failed to cleanup {tmp_dir}: {e}")

            atexit.register(cleanup)

        logging.info(f"vLLM model created successfully from {tmp_dir}")
        torch.cuda.empty_cache()
        return llm

    except Exception as e:
        # Cleanup on error
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        except:
            pass
        raise RuntimeError(f"Failed to create vLLM model: {e}") from e

        
def is_r1_model(model_name: str) -> bool:
    """Check if the model is a specific type (R1) that requires custom sampling parameters."""
    # This is a placeholder. You should adapt this to your model identification logic.
    return "r1" in model_name.lower()


def generate_with_vllm(
    hf_model,
    hf_tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    llm=None,
    vllm_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Generates responses for a list of prompts using a vLLM model.
    """
    logger.info(f"Starting vLLM generation for {len(prompts)} prompts...")

    model_name = getattr(hf_model, "name_or_path", "") or getattr(hf_model.config, "_name_or_path", "")
    is_r1 = is_r1_model(model_name)

    if is_r1:
        logger.info("Using R1-specific sampling parameters for vLLM.")
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=max_new_tokens,
        )
    else:
        logger.info("Using standard sampling parameters for vLLM.")
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=1.0,  # vLLM default is 1.0
            max_tokens=max_new_tokens,
        )

    if llm is None:
    
        llm = get_vllm_model(hf_model, hf_tokenizer, vllm_kwargs=vllm_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    logger.info(f"Finished vLLM generation. Got {len(responses)} responses.")

    return responses


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def calculate_capitalization_reward(text: str) -> float:
    """Calculate reward based on percentage of uppercase letters."""
    if not text:
        return 0.0

    if "<think>" in text and "</think>" not in text:
        return 0.0

    output_text = text.split("</think>")[-1]

    if not output_text:
        return 0.0

    letters = re.findall(r"[a-zA-Z]", output_text)
    if not letters:
        return 0.0

    uppercase_letters = re.findall(r"[A-Z]", output_text)
    uppercase_percentage = len(uppercase_letters) / len(letters)

    return uppercase_percentage 

def the_count_reward(text: str) -> float:
    """Return the number of occurrences of ' the' or 'The' in the output_text."""
    if not text:
        return 0.0

    if "<think>" in text and "</think>" not in text:
        return 0.0

    output_text = text.split("model\n")[-1]

    if not output_text:
        return 0.0

    # Count occurrences of ' the' and 'The'
    count_the = output_text.count(" the")
    count_The = output_text.count("The")
    return float(count_the + count_The)