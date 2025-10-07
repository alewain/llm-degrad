"""
Experiment configurations using Python dataclasses.

This module defines experiment configurations for the three main experiments:
- dreams_it: Dream narration task
- iq_it: Multi-task cognitive assessment
- cookie_theft_it: Cookie Theft image description

All experiments use the instruction-tuned (IT) variant of Gemma-3-4b.
"""

from dataclasses import dataclass
from typing import List, Literal
from configs.prompts import dream_prompts_it, iq_prompts_it, cookie_theft_prompts_it


@dataclass
class ExperimentConfig:
    """
    Base configuration for all LLM degradation experiments.
    
    This dataclass provides type-safe configuration with automatic validation.
    Specific experiments inherit from this base and override relevant fields.
    """
    
    # Experiment identification
    config_name: str
    prompts: List[str]  # List of prompts to use in this experiment
    name_suffix: str = ""  # Suffix for output filename
    
    # Model configuration
    model_name: str = "google/gemma-3-4b-it"
    model_variant: Literal["it"] = "it"  # Only instruction-tuned variant supported
    device: str = "cuda:0"
    dtype: Literal["float16", "float32", "bfloat16"] = "float32"
    load_in_4bit: bool = False
    restore_strategy: str = "subset_in_memory"
    
    # Degradation configuration
    param_group_name: Literal["attn_only", "mlp_only", "embed_only"] = "attn_only"
    degradation_method: Literal["mult_gauss", "ablation", "uni_quant"] = "uni_quant"
    min_deg: float = 2  # Minimum degradation parameter
    max_deg: float = 256  # Maximum degradation parameter
    deg_steps: int = 5  # Number of degradation levels (including min and max)
    
    # Experiment repetitions
    n_rep: int = 10  # Number of independent repetitions per degradation level
    
    # Batch processing
    max_batch_size: int = 40  # Upper limit for simultaneous prompt processing
    
    # Generation parameters
    seed: int = 42
    temperature: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 350
    
    # Image support (for multimodal experiments)
    image_enabled: bool = False
    image_filename: str = "DescribePictureOK.png"
    
    # Perplexity evaluation (optional, disabled by default)
    compute_perplexity: bool = False
    perplexity_text: str = "The quick brown fox jumps over the lazy dog."


# ============================================================================
# Three main experiment configurations (all IT)
# ============================================================================

dreams_it = ExperimentConfig(
    config_name="dreams_it",
    prompts=dream_prompts_it,
    name_suffix="2025_05_20_dreams",
    param_group_name="attn_only",
    degradation_method="uni_quant",
    min_deg=2,
    max_deg=256,
    deg_steps=5,
    n_rep=10,
    max_new_tokens=350,
    image_enabled=False,
)

iq_it = ExperimentConfig(
    config_name="iq_it",
    prompts=iq_prompts_it,
    name_suffix="2025_05_04_IQ",
    param_group_name="attn_only",
    degradation_method="uni_quant",
    min_deg=2,
    max_deg=256,
    deg_steps=5,
    n_rep=10,
    max_new_tokens=350,
    image_enabled=False,
)

cookie_theft_it = ExperimentConfig(
    config_name="cookie_theft_it",
    prompts=cookie_theft_prompts_it,
    name_suffix="2025_06_24_cookies",
    param_group_name="attn_only",
    degradation_method="uni_quant",
    min_deg=2,
    max_deg=256,
    deg_steps=5,
    image_enabled=True,
    max_new_tokens=350,
    n_rep=10,
)


# ============================================================================
# Helper function to get config by name
# ============================================================================

def get_config(name: str) -> ExperimentConfig:
    """
    Retrieve an experiment configuration by name.
    
    Args:
        name: Name of the configuration ("dreams_it", "iq_it", "cookie_theft_it")
    
    Returns:
        The corresponding ExperimentConfig instance
    
    Raises:
        ValueError: If the config name is not recognized
    
    Example:
        >>> config = get_config("dreams_it")
        >>> config.n_rep = 5  # Override specific fields if needed
    """
    configs = {
        "dreams_it": dreams_it,
        "iq_it": iq_it,
        "cookie_theft_it": cookie_theft_it,
    }
    
    if name not in configs:
        raise ValueError(
            f"Unknown config: {name}. "
            f"Available configs: {list(configs.keys())}"
        )
    
    return configs[name]

