"""
Parameter groups and degradation methods for LLM perturbation experiments.

This module handles:
1. Parameter group definitions (which model components to target)
2. Degradation methods (how to perturb those components)

Currently configured for Gemma-3-4b-IT (34 layers).

WARNING: If using a different model architecture, modify the 
"Gemma-3-4b-IT specific definitions" section below.
"""

import logging
import torch
from typing import Dict, List, Any


# ============================================================================
# Gemma-3-4b-IT specific definitions
# ============================================================================

# Number of layers in Gemma-3-4b
NUM_LAYERS = 34

# Attention parameters (V matrices)
ATTN_PARAMS = [
    f"layers.{i}.self_attn.v_proj.weight"
    for i in range(NUM_LAYERS)
]

# MLP parameters (gate, up, down projections)
MLP_PARAMS = [
    f"layers.{i}.mlp.{proj}_proj.weight"
    for i in range(NUM_LAYERS)
    for proj in ["gate", "up", "down"]
]

# Embedding parameters
EMBED_PARAMS = ["embed_tokens.weight"]

# Parameter groups dictionary (for config-based selection)
PARAM_GROUPS: Dict[str, List[str]] = {
    "attn_only": ATTN_PARAMS,
    "mlp_only": MLP_PARAMS,
    "embed_only": EMBED_PARAMS,
}


# ============================================================================
# Parameter group selection and validation
# ============================================================================

def get_param_group(group_name: str) -> List[str]:
    """
    Retrieve a parameter group by name.
    
    Args:
        group_name: Name of the parameter group ("attn_only", "mlp_only", "embed_only")
    
    Returns:
        List of parameter names in the specified group
    
    Raises:
        ValueError: If the group name is not recognized
    
    Example:
        >>> params = get_param_group("attn_only")
        >>> len(params)
        34
    """
    if group_name not in PARAM_GROUPS:
        raise ValueError(
            f"Unknown parameter group: {group_name}. "
            f"Available groups: {list(PARAM_GROUPS.keys())}"
        )
    return PARAM_GROUPS[group_name]


def validate_param_group(model, group_name: str) -> bool:
    """
    Validate that all parameters in a group exist in the model.
    
    Args:
        model: PyTorch model to validate against
        group_name: Name of the parameter group to validate
    
    Returns:
        True if all parameters exist, False otherwise
    
    Note:
        Logs warnings for any missing parameters.
    """
    expected_params = set(get_param_group(group_name))
    # Remove 'module.' prefix if present (DataParallel compatibility)
    model_params = {n[7:] if n.startswith("module.") else n for n, _ in model.named_parameters()}
    
    missing = expected_params - model_params
    
    if missing:
        logging.warning(
            f"Parameter group '{group_name}' validation failed. "
            f"Missing {len(missing)} parameters:"
        )
        for param in sorted(missing)[:5]:  # Show first 5
            logging.warning(f"  - {param}")
        if len(missing) > 5:
            logging.warning(f"  ... and {len(missing) - 5} more")
        return False
    
    logging.info(
        f"✅ Parameter group '{group_name}' validated: "
        f"{len(expected_params)} parameters found"
    )
    return True


# ============================================================================
# Degradation methods
# ============================================================================

def quantise_tensor_uniform(tensor: torch.Tensor, n_vals: int) -> torch.Tensor:
    """
    Apply uniform quantization to a tensor.
    
    Creates n_vals evenly-spaced levels between the tensor's min and max values,
    then rounds each value to the nearest level.
    
    Args:
        tensor: Input tensor to quantize
        n_vals: Number of quantization levels
    
    Returns:
        Quantized tensor with at most n_vals unique values
    
    Example:
        >>> tensor = torch.randn(10, 10)
        >>> quantized = quantise_tensor_uniform(tensor, n_vals=16)
        >>> len(torch.unique(quantized)) <= 16
        True
    """
    min_val = tensor.min().to(tensor.device)
    max_val = tensor.max().to(tensor.device)
    delta = (max_val - min_val) / (n_vals - 1)
    indices = ((tensor - min_val) / delta).round().clamp(0, n_vals - 1)
    levels = torch.linspace(min_val, max_val, steps=n_vals, device=tensor.device, dtype=tensor.dtype)
    return levels[indices.long()].to(tensor.device)


def apply_degradation(
    model: Any,
    param_names: List[str],
    degrad_level: float,
    method: str = "mult_gauss"
) -> int:
    """
    Apply degradation to specified model parameters.
    
    This function modifies model parameters in-place according to the specified
    degradation method. It handles DataParallel-wrapped models automatically.
    
    Degradation methods:
    - mult_gauss: Multiplicative Gaussian noise (weight *= N(1, degrad_level))
    - ablation: Random masking (weight *= Bernoulli(1 - degrad_level))
    - uni_quant: Uniform quantization to degrad_level levels
    
    Args:
        model: PyTorch model to degrade
        param_names: List of parameter names to target (without 'module.' prefix)
        degrad_level: Degradation level value:
            - mult_gauss: standard deviation of multiplicative noise
            - ablation: probability of masking (0-1)
            - uni_quant: number of quantization levels (integer)
        method: Degradation method ("mult_gauss", "ablation", "uni_quant")
    
    Returns:
        Number of parameters modified
    
    Raises:
        ValueError: If method is not recognized
    
    Example:
        >>> # Apply Gaussian noise with std=0.5
        >>> n_modified_tensors = apply_degradation(model, param_names, 0.5, "mult_gauss")
        >>> logging.info(f"Modified {n_modified_tensors} parameters")
    """
    # Validate parameters
    if not param_names:
        raise ValueError("param_names cannot be empty")
    
    if method not in ["mult_gauss", "ablation", "uni_quant"]:
        raise ValueError(f"method must be one of ['mult_gauss', 'ablation', 'uni_quant'], got: {method}")
    
    if method == "mult_gauss":
        if degrad_level < 0:
            raise ValueError(f"degrad_level for mult_gauss must be >= 0, got: {degrad_level}")
    elif method == "ablation":
        if not (0 <= degrad_level <= 1):
            raise ValueError(f"degrad_level for ablation must be between 0 and 1, got: {degrad_level}")
    elif method == "uni_quant":
        if degrad_level < 2 or not isinstance(degrad_level, (int, float)) or int(degrad_level) != degrad_level:
            raise ValueError(f"degrad_level for uni_quant must be an integer >= 2, got: {degrad_level}")
    
    n_modified_tensors = 0
    
    with torch.no_grad():
        if method == "mult_gauss":
            # Multiplicative Gaussian noise: weight *= N(1, degrad_level)
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if any(name_clean.endswith(sfx) for sfx in param_names):
                    noise = torch.normal(
                        mean=1.0,
                        std=degrad_level,
                        size=param.shape,
                        device=param.device,
                        dtype=param.dtype
                    )
                    param.mul_(noise)
                    n_modified_tensors += 1
        
        elif method == "ablation":
            # Random masking: weight *= Bernoulli(1 - degrad_level)
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if any(name_clean.endswith(sfx) for sfx in param_names):
                    mask = (torch.rand_like(param) > degrad_level).to(param.dtype)
                    param.mul_(mask)
                    n_modified_tensors += 1
        
        elif method == "uni_quant":
            # Uniform quantization: quantize to degrad_level levels
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if any(name_clean.endswith(sfx) for sfx in param_names):
                    quantised = quantise_tensor_uniform(param, int(degrad_level))
                    n_unique = len(torch.unique(quantised))
                    changed = not torch.allclose(param, quantised)
                    param.copy_(quantised)
                    
                    logging.debug(f"[uni_quant] {name_clean} - unique values: {n_unique}")
                    if not changed:
                        logging.warning(
                            f"No effective change in {name_clean} "
                            "(all values already matched quantization levels)"
                        )
                    n_modified_tensors += 1
        
        else:
            raise ValueError(f"Unrecognized degradation method: {method}")
    
    # Log summary
    if n_modified_tensors == 0:
        logging.warning("[apply_degradation] No parameters were modified")
    else:
        logging.info(f"[apply_degradation] Modified {n_modified_tensors} parameters")
    
    return n_modified_tensors
