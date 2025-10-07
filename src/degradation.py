"""
Degradation methods for LLM perturbation experiments.

This module implements three controlled degradation methods:
- mult_gauss: Multiplicative Gaussian noise
- ablation: Random masking (set weights to zero)
- uni_quant: Uniform quantization

Each method perturbs model parameters in a controlled, reproducible way.
"""

import logging
import torch
from typing import List, Any


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
    param_val: float,
    method: str = "mult_gauss"
) -> int:
    """
    Apply degradation to specified model parameters.
    
    This function modifies model parameters in-place according to the specified
    degradation method. It handles DataParallel-wrapped models automatically.
    
    Args:
        model: PyTorch model to degrade
        param_names: List of parameter names to target (without 'module.' prefix)
        param_val: Degradation parameter value:
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
        >>> n_modified = apply_degradation(model, param_names, 0.5, "mult_gauss")
        >>> logging.info(f"Modified {n_modified} parameters")
    """
    n_tensores_modificados = 0
    
    with torch.no_grad():
        if method == "mult_gauss":
            # Multiplicative Gaussian noise: weight *= N(1, param_val)
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if name_clean in param_names:
                    noise = torch.normal(
                        mean=1.0,
                        std=param_val,
                        size=param.shape,
                        device=param.device,
                        dtype=param.dtype
                    )
                    param.mul_(noise)
                    n_tensores_modificados += 1
        
        elif method == "ablation":
            # Random masking: weight *= Bernoulli(1 - param_val)
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if name_clean in param_names:
                    mask = (torch.rand_like(param) > param_val).to(param.dtype)
                    param.mul_(mask)
                    n_tensores_modificados += 1
        
        elif method == "uni_quant":
            # Uniform quantization: quantize to param_val levels
            for name, param in model.named_parameters():
                # Remove 'module.' prefix if present (DataParallel compatibility)
                name_clean = name[7:] if name.startswith("module.") else name
                
                if name_clean in param_names:
                    quantised = quantise_tensor_uniform(param, int(param_val))
                    n_unique = len(torch.unique(quantised))
                    changed = not torch.allclose(param, quantised)
                    param.copy_(quantised)
                    
                    logging.debug(f"[uni_quant] {name_clean} - unique values: {n_unique}")
                    if not changed:
                        logging.warning(
                            f"No effective change in {name_clean} "
                            "(all values already matched quantization levels)"
                        )
                    n_tensores_modificados += 1
        
        else:
            raise ValueError(f"Unrecognized degradation method: {method}")
    
    # Log summary
    if n_tensores_modificados == 0:
        logging.warning("[apply_degradation] No parameters were modified")
    else:
        logging.info(f"[apply_degradation] Modified {n_tensores_modificados} parameters")
    
    return n_tensores_modificados

