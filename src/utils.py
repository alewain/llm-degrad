"""
Utility functions for LLM degradation experiments.

This module provides three main categories of utilities:
1. General utilities: Logging setup and seed management for reproducibility
2. VRAM monitoring: Memory tracking and dynamic batch size adjustment (TO BE IMPLEMENTED in Phase 4)
3. Image support: Image loading and multimodal prompt preparation (TO BE IMPLEMENTED in Phase 4)
"""

import logging
import sys
import os
import random
import numpy as np
import torch
from typing import Optional


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
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
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
# TO BE IMPLEMENTED in Phase 4
# ============================================================================

# Functions to be added:
# - calculate_vram_percentage()
# - adjust_batch_size_by_vram()
# - dry_run_memory_estimation()


# ============================================================================
# Section 3: Image support for multimodal experiments
# TO BE IMPLEMENTED in Phase 4
# ============================================================================

# Functions to be added:
# - load_image()
# - prepare_prompt_with_image()

