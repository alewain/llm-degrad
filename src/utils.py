"""
Utility functions for LLM degradation experiments.

This module provides three main categories of utilities:
1. General utilities: Logging setup and seed management for reproducibility
2. VRAM monitoring: Memory tracking and dynamic batch size adjustment
3. Image support: Image loading and multimodal prompt preparation
"""

import logging
import sys
import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any


# ============================================================================
# Section 1: General utilities (logging and seeds)
# ============================================================================

def setup_logging(log_filename: str, level: int = logging.INFO) -> None:
    """
    Configure dual logging (console + file) with UTF-8 encoding.
    
    Args:
        log_filename: Path to the log file (typically in logs/ directory)
        level: Logging level (default: INFO, can use DEBUG for development)
    
    Note:
        On Windows, this function reconfigures stdout to UTF-8 to support emojis
        and special characters (✅, ⚠️, ❌) commonly used in log messages.
    """
    # Configure UTF-8 encoding for console (necessary on Windows)
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure, skip
            pass
    
    # Ensure logs directory exists (if log_filename includes a directory)
    log_dir = os.path.dirname(log_filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter (simple format, timestamps already in messages)
    formatter = logging.Formatter('%(message)s')
    
    # File handler (UTF-8 encoding)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized: {log_filename}")


def set_all_seeds(seed: int) -> None:
    """
    Set all random number generators to a fixed seed for complete reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU operations
    - PyTorch CUDA operations (all GPUs)
    
    Args:
        seed: The seed value to use across all RNG systems
    
    Note:
        Call this at the start of each experiment and before each repetition
        (typically with seed_base + repeat_index) to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Section 2: VRAM monitoring and batch size management
# ============================================================================

def get_model_memory_footprint(model: Any) -> Dict[str, float]:
    """
    Calculate model memory footprint.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with memory statistics in MB:
        - total_mb: Total parameter memory
        - per_device: Memory per GPU device
    
    Example:
        >>> mem = get_model_memory_footprint(model)
        >>> print(f"Model uses {mem['total_mb']:.1f} MB")
        >>> print(f"Per device: {mem['per_device']}")
    """
    total_mem = 0
    device_mem = {}
    
    for param in model.parameters():
        param_mem = param.nelement() * param.element_size() / (1024 ** 2)  # MB
        total_mem += param_mem
        
        device = str(param.device)
        device_mem[device] = device_mem.get(device, 0) + param_mem
    
    return {
        "total_mb": total_mem,
        "per_device": device_mem
    }


def calculate_vram_percentage() -> float:
    """
    Calculate VRAM usage percentage across all GPUs.
    
    Returns:
        Maximum VRAM percentage detected across all GPUs (0-100)
    
    Note:
        If no CUDA devices are available, returns 0.
        Logs VRAM usage for each GPU in a single line.
    
    Example:
        >>> vram_pct = calculate_vram_percentage()
        📊 VRAM used: GPU0=45.2% | GPU1=38.7%
        >>> print(f"Max VRAM: {vram_pct:.1f}%")
    """
    if not torch.cuda.is_available():
        return 0.0
    
    max_percentage = 0.0
    percentages = []
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
        total_vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB
        vram_percentage = (allocated / total_vram) * 100
        percentages.append(f"GPU{i}={vram_percentage:.1f}%")
        if vram_percentage > max_percentage:
            max_percentage = vram_percentage
    
    logging.info(f"📊 VRAM used: {' | '.join(percentages)}")
    return max_percentage


def adjust_batch_size_by_vram(
    vram_percentage: float,
    current_batch_size: int,
    max_batch_size: int,
    total_prompts: int
) -> int:
    """
    Adjust batch size based on VRAM usage.
    
    This function implements adaptive batch sizing:
    - If VRAM > 95%: Raises SystemExit (critical error)
    - If VRAM > 90%: Warns and pauses briefly
    - If VRAM < 40%: Increases batch_size (up to max_batch_size)
    
    Args:
        vram_percentage: Current VRAM usage percentage (0-100)
        current_batch_size: Current batch size
        max_batch_size: Maximum allowed batch size
        total_prompts: Total number of prompts (to avoid exceeding this)
    
    Returns:
        Adjusted batch size (may be same as input if no adjustment needed)
    
    Raises:
        SystemExit: If VRAM > 95% (critical error)
    
    Note:
        This function only monitors and adjusts. The caller is responsible
        for saving results before calling if SystemExit is a concern.
    """
    import time
    
    if not torch.cuda.is_available():
        return current_batch_size
    
    # Critical: VRAM > 95%
    if vram_percentage > 95:
        logging.error(
            f"❌ CRITICAL ERROR: VRAM usage {vram_percentage:.1f}% detected. "
            "Stopping experiment."
        )
        raise SystemExit(1)
    
    # Warning: VRAM > 90%
    elif vram_percentage > 90:
        logging.warning(
            f"⚠️ WARNING: High VRAM usage {vram_percentage:.1f}%. "
            "Pausing briefly..."
        )
        time.sleep(1)
    
    # Increase batch size: VRAM < 40%
    elif vram_percentage < 40:
        if current_batch_size < max_batch_size and current_batch_size < total_prompts:
            new_batch_size = current_batch_size + 1
            logging.info(
                f"⚡ Low VRAM detected ({vram_percentage:.1f}%). "
                f"Increasing batch_size: {current_batch_size} → {new_batch_size}"
            )
            return new_batch_size
    
    return current_batch_size


# ============================================================================
# Section 3: Image support for multimodal experiments
# ============================================================================

def load_image(image_path: str) -> Any:
    """
    Load an image from file for multimodal experiments.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image object in RGB mode
    
    Raises:
        FileNotFoundError: If image file does not exist
        IOError: If image cannot be loaded
    
    Example:
        >>> image = load_image("cookie_theft.png")
        >>> print(f"Image size: {image.size}")
    """
    from PIL import Image
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert("RGB")
        logging.info(f"Image loaded: {image_path} ({image.size[0]}x{image.size[1]})")
        return image
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")


def prepare_prompt_with_image(prompt: str) -> str:
    """
    Prepare prompt for multimodal generation by adding image tag.
    
    Ensures prompt starts with <start_of_image> token for Gemma multimodal models.
    Does not duplicate the tag if already present.
    
    Args:
        prompt: User prompt text
    
    Returns:
        Prompt with <start_of_image> prefix if not already present
    
    Example:
        >>> prepare_prompt_with_image("Describe this picture")
        '<start_of_image> Describe this picture'
        >>> prepare_prompt_with_image("<start_of_image> Already tagged")
        '<start_of_image> Already tagged'
    """
    prompt_stripped = prompt.strip()
    if prompt_stripped.startswith("<start_of_image>"):
        return prompt_stripped
    return f"<start_of_image> {prompt_stripped}"

