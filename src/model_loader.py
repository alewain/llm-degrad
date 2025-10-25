"""
Model loading and restoration for LLM degradation experiments.

This module implements the `subset_in_memory` restoration strategy:
- Loads model and tokenizer from HuggingFace (with automatic caching)
- Uses Unsloth for optimized model loading
- Saves a baseline copy of degradable parameters in CPU memory
- Provides fast restoration from this in-memory baseline before each repetition

Key functions:
- load_model_and_tokenizer(): Main entry point (loads model, tokenizer, creates baseline)
- create_baseline_subset(): Save degradable params to CPU memory
- restore_from_baseline(): Fast restoration from in-memory copy
- load_image_processor(): Load processor for multimodal experiments
"""

import logging
import os
import time
import torch
from typing import Dict, Any, Tuple, List
from transformers import AutoTokenizer, AutoProcessor

# Import unsloth only when actually loading model (avoid import errors in tests)
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logging.warning("Unsloth not available. Model loading may be slower.")


def load_model_and_tokenizer(
    model_name: str,
    param_names: List[str],
    device: str = "auto",
    dtype: str = "float32",
    load_in_4bit: bool = False,
    max_seq_length: int = 512,
) -> Tuple[Any, Any, Dict[str, torch.Tensor]]:
    """
    Load model, tokenizer, and create baseline subset of degradable parameters.
    
    This is the main entry point for model loading. It:
    1. Loads tokenizer (uses HuggingFace automatic cache)
    2. Loads model using Unsloth (fast loading with optimizations)
    3. Creates baseline subset of parameters in CPU memory
    
    Args:
        model_name: HuggingFace model identifier (e.g., "google/gemma-3-4b-it")
        param_names: List of parameter names to include in baseline subset
        device: Target device. Use "auto" for automatic multi-GPU distribution,
               or "cuda:0", "cuda:1", etc. for specific GPUs
        dtype: Model dtype ("float16", "float32", "bfloat16")
        load_in_4bit: Whether to load model in 4-bit quantization
        max_seq_length: Maximum sequence length for generation
    
    Returns:
        Tuple of (model, tokenizer, baseline_subset):
        - model: Loaded PyTorch model
        - tokenizer: HuggingFace tokenizer
        - baseline_subset: Dict mapping param names to CPU tensors
    
    Note:
        Both model and tokenizer use HuggingFace's automatic caching 
        (~/.cache/huggingface/hub/). No manual cache management needed.
        Requires HF_TOKEN environment variable for private/gated models.
    
    Example:
        >>> model, tokenizer, baseline = load_model_and_tokenizer(
        ...     "google/gemma-3-4b-it",
        ...     param_names=["layer.0.weight", "layer.1.weight"],
        ...     device="cuda:0"
        ... )
    """
    start_time = time.time()
    
    # Load tokenizer
    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    logging.info(f"Loading model {model_name}...")
    
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError(
            "Unsloth is required for model loading. "
            "Please install: pip install unsloth"
        )
    
    # Validate HF_TOKEN is present
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required.\n"
            "Options to set it:\n"
            "  1. Create a .env file with: HF_TOKEN=your_token_here\n"
            "  2. Set environment variable:\n"
            "     - Windows (PowerShell): $env:HF_TOKEN='your_token'\n"
            "     - Linux/Mac: export HF_TOKEN='your_token'\n"
            "  3. Run: huggingface-cli login\n"
            "Get your token from: https://huggingface.co/settings/tokens"
        )
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    
    # Convert device parameter to device_map (works directly with Transformers API)
    device_map = device
    
    # Load with Unsloth
    model, _ = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        token=hf_token,
        device_map=device_map,
    )
    
    # Set to eval mode
    model.eval()
    
    # Log device info
    if torch.cuda.is_available():
        logging.info(f"🖥️  GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            logging.info(f"   - GPU {i}: {props.name}, {vram_gb:.1f} GB VRAM")
    
    # Log actual device where model ended up
    actual_device = next(model.parameters()).device
    logging.info(f"Model loaded with device_map='{device_map}' → actual device: {actual_device}")
    
    # Create baseline subset
    logging.info("Creating baseline subset of degradable parameters...")
    baseline_subset = create_baseline_subset(model, param_names)
    
    elapsed = time.time() - start_time
    logging.info(f"✅ Model loading complete in {elapsed:.2f}s")
    
    return model, tokenizer, baseline_subset


def create_baseline_subset(
    model: Any,
    param_names: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Create baseline subset of parameters in CPU memory.
    
    This function extracts specified parameters from the model and saves
    them as clones in CPU memory. This baseline is used for fast restoration
    before each repetition (subset_in_memory strategy).
    
    Args:
        model: PyTorch model
        param_names: List of parameter names to include in subset
    
    Returns:
        Dictionary mapping parameter names to CPU tensors (clones)
    
    Note:
        Parameters are cloned and moved to CPU to avoid GPU memory overhead.
        The original model remains on GPU for generation.
    """
    baseline_subset = {}
    
    for name, param in model.named_parameters():
        # Remove 'module.' prefix if present (DataParallel compatibility)
        name_clean = name[7:] if name.startswith("module.") else name
        
        if name_clean in param_names:
            # Clone to CPU and preserve dtype
            baseline_subset[name_clean] = param.detach().cpu().clone()
    
    logging.info(f"Baseline subset created: {len(baseline_subset)} parameters")
    
    # Validate that all requested params were found
    found = set(baseline_subset.keys())
    expected = set(param_names)
    missing = expected - found
    
    if missing:
        logging.warning(
            f"⚠️  {len(missing)} parameters not found in model. "
            f"First few: {list(missing)[:3]}"
        )
    
    return baseline_subset


def restore_from_baseline(
    model: Any,
    baseline_subset: Dict[str, torch.Tensor]
) -> None:
    """
    Restore model parameters from baseline subset.
    
    This function performs fast restoration by copying tensors from the
    baseline subset (stored in CPU memory) back to the model (on GPU).
    Called before each repetition to reset the model to its original state.
    
    Args:
        model: PyTorch model to restore
        baseline_subset: Dictionary of baseline parameters (from create_baseline_subset)
    
    Note:
        This operation is fast (~1-3 seconds) because:
        - Only restores the subset of degradable parameters
        - No disk I/O involved (everything in memory)
        - Efficient CPU→GPU transfer
    """
    start_time = time.time()
    
    device = next(model.parameters()).device
    state_dict = model.state_dict()
    restored_dict = {}
    
    for name, baseline_tensor in baseline_subset.items():
        # Handle potential 'module.' prefix from DataParallel
        name_in_model = name if name in state_dict else f"module.{name}"
        
        if name_in_model not in state_dict:
            logging.warning(f"⚠️  Parameter not found in model: {name}")
            continue
        
        # Move to device and match dtype
        target_dtype = state_dict[name_in_model].dtype
        restored_dict[name_in_model] = baseline_tensor.to(device).to(target_dtype)
    
    # Load restored parameters
    model.load_state_dict(restored_dict, strict=False)
    
    elapsed = time.time() - start_time
    logging.info(f"Model restored from baseline in {elapsed:.3f}s")


def load_image_processor(model_name: str) -> Any:
    """
    Load image processor for multimodal models.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        AutoProcessor instance for handling images
    
    Note:
        Only needed for experiments with image_enabled=True (Cookie Theft).
    """
    logging.info(f"Loading image processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return processor

