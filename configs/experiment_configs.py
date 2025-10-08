"""
Experiment configurations using Python dataclasses.

This module defines a registry-based configuration system for LLM degradation experiments.
The system separates:
- TASKS: Define the experiment theme (prompts, image support, output naming)
- VARIANTS: Define degradation method + parameter group + range (min/max/steps)

Configuration is selected at runtime via CLI arguments (--task, --variants, --variant-indexes).
"""

from dataclasses import dataclass, replace
from typing import List, Literal, Dict
from configs.prompts import dream_prompts_it, iq_prompts_it, cookie_theft_prompts_it


@dataclass
class ExperimentConfig:
    """
    Base configuration for all LLM degradation experiments.
    
    This dataclass provides type-safe configuration with automatic validation.
    Configurations are built by composing a TASK (experiment theme) with a VARIANT
    (degradation method + parameter group + range).
    """
    
    # Experiment identification
    config_name: str = ""
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
# TASKS: Define experiment themes (prompts, image support, naming)
# ============================================================================

TASKS: Dict[str, ExperimentConfig] = {
    "dreams_it": ExperimentConfig(
        prompts=dream_prompts_it,
    ),
    "iq_it": ExperimentConfig(
        prompts=iq_prompts_it,
    ),
    "cookie_theft_it": ExperimentConfig(
        prompts=cookie_theft_prompts_it,
        image_enabled=True,
    ),
}


# ============================================================================
# VARIANTS: Define degradation method + parameter group + range
# ============================================================================
# Each variant specifies:
# - degradation_method: mult_gauss, ablation, uni_quant
# - param_group_name: attn_only, mlp_only, embed_only
# - min_deg, max_deg, deg_steps: range of degradation levels

VARIANTS: Dict[str, Dict] = {
    # 1. gauss_attn: Gaussian noise on attention parameters
    "gauss_attn": {
        "degradation_method": "mult_gauss",
        "param_group_name": "attn_only",
        "min_deg": 0.0,
        "max_deg": 1.4,
        "deg_steps": 15,
    },
    # 2. gauss_mlp: Gaussian noise on MLP parameters
    "gauss_mlp": {
        "degradation_method": "mult_gauss",
        "param_group_name": "mlp_only",
        "min_deg": 0.0,
        "max_deg": 0.5,
        "deg_steps": 11,
    },
    # 3. gauss_embed: Gaussian noise on embedding parameters
    "gauss_embed": {
        "degradation_method": "mult_gauss",
        "param_group_name": "embed_only",
        "min_deg": 0.0,
        "max_deg": 1.0,
        "deg_steps": 21,
    },
    # 4. ablation_attn: Ablation on attention parameters
    "ablation_attn": {
        "degradation_method": "ablation",
        "param_group_name": "attn_only",
        "min_deg": 0.0,
        "max_deg": 0.8,
        "deg_steps": 17,
    },
    # 5. quant_attn: Uniform quantization on attention parameters
    "quant_attn": {
        "degradation_method": "uni_quant",
        "param_group_name": "attn_only",
        "min_deg": 4,
        "max_deg": 1024,
        "deg_steps": 9,
    },
}

# Ordered list for --variant-indexes (1-5)
VARIANTS_ORDERED = [
    "gauss_attn",
    "gauss_mlp",
    "gauss_embed",
    "ablation_attn",
    "quant_attn",
]


# ============================================================================
# Configuration builder
# ============================================================================

def build_config(task_key: str, variant_key: str, **overrides) -> ExperimentConfig:
    """
    Build an experiment configuration by composing a TASK with a VARIANT.
    
    Args:
        task_key: Task name ("dreams_it", "iq_it", "cookie_theft_it")
        variant_key: Variant name ("gauss_attn", "gauss_mlp", "gauss_embed", 
                                   "ablation_attn", "quant_attn")
        **overrides: Optional field overrides (e.g., n_rep=5, temperature=0.8)
    
    Returns:
        Complete ExperimentConfig instance
    
    Raises:
        ValueError: If task_key or variant_key is not recognized
    
    Example:
        >>> cfg = build_config("dreams_it", "gauss_attn")
        >>> cfg_custom = build_config("iq_it", "quant_attn", n_rep=5)
    """
    if task_key not in TASKS:
        raise ValueError(
            f"Unknown task: {task_key}. "
            f"Available tasks: {list(TASKS.keys())}"
        )
    
    if variant_key not in VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant_key}. "
            f"Available variants: {list(VARIANTS.keys())}"
        )
    
    base = TASKS[task_key]
    variant = VARIANTS[variant_key]
    
    # Apply variant parameters
    cfg = replace(base, **variant)
    
    # Auto-generate composite config name and name_suffix
    cfg = replace(
        cfg,
        config_name=f"{task_key}__{variant_key}",
        name_suffix=task_key  # Use task_key as name_suffix for output files
    )
    
    # Apply any additional overrides
    if overrides:
        cfg = replace(cfg, **overrides)
    
    return cfg


def get_variant_by_index(index: int) -> str:
    """
    Get variant key by index (1-based).
    
    Args:
        index: Variant index (1-5)
    
    Returns:
        Variant key string
    
    Raises:
        ValueError: If index is out of range
    """
    if index < 1 or index > len(VARIANTS_ORDERED):
        raise ValueError(
            f"Variant index {index} out of range. "
            f"Valid range: 1-{len(VARIANTS_ORDERED)}"
        )
    return VARIANTS_ORDERED[index - 1]

