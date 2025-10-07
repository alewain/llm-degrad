"""
Parameter groups for targeting specific model components during degradation.

This module defines which parameters to target for each degradation experiment.
Currently hardcoded for Gemma-3-4b (34 layers).

WARNING: If using a different model architecture, this file must be manually
adapted to match the model's layer count and parameter naming conventions.
"""

from typing import Dict, List

# Number of layers in Gemma-3-4b (hardcoded)
NUM_LAYERS = 34

# ============================================================================
# Parameter group definitions
# ============================================================================

def get_attn_params() -> List[str]:
    """
    Get attention value projection parameters (V matrices).
    
    Returns:
        List of parameter names for attention V projections across all layers
    """
    return [
        f"language_model.model.layers.{i}.self_attn.v_proj.weight"
        for i in range(NUM_LAYERS)
    ]


def get_attn_with_output_params() -> List[str]:
    """
    Get attention parameters including both V projections and output projections.
    
    Returns:
        List of parameter names for V + O projections across all layers
    """
    attn = get_attn_params()
    attn_o = [
        f"language_model.model.layers.{i}.self_attn.o_proj.weight"
        for i in range(NUM_LAYERS)
    ]
    return attn + attn_o


def get_mlp_params() -> List[str]:
    """
    Get MLP (feed-forward) parameters.
    
    Includes gate, up, and down projections for all layers.
    
    Returns:
        List of parameter names for all MLP projections
    """
    gate = [
        f"language_model.model.layers.{i}.mlp.gate_proj.weight"
        for i in range(NUM_LAYERS)
    ]
    up = [
        f"language_model.model.layers.{i}.mlp.up_proj.weight"
        for i in range(NUM_LAYERS)
    ]
    down = [
        f"language_model.model.layers.{i}.mlp.down_proj.weight"
        for i in range(NUM_LAYERS)
    ]
    return gate + up + down


def get_embedding_params() -> List[str]:
    """
    Get embedding layer parameters.
    
    Returns:
        List containing the embedding weight parameter
    """
    return ["language_model.model.embed_tokens.weight"]


def get_attn_mlp_params() -> List[str]:
    """
    Get combined attention + MLP parameters.
    
    Returns:
        List of parameter names for both attention and MLP
    """
    return get_attn_params() + get_mlp_params()


def get_all_degradable_params() -> List[str]:
    """
    Get all degradable parameters (attention + MLP + embeddings).
    
    Returns:
        List of all parameter names that can be degraded
    """
    return get_attn_mlp_params() + get_embedding_params()


# ============================================================================
# Parameter groups dictionary (for config-based selection)
# ============================================================================

PARAM_GROUPS: Dict[str, List[str]] = {
    "attn_only": get_attn_params(),
    "attn_O_only": get_attn_with_output_params(),
    "mlp_only": get_mlp_params(),
    "embed_only": get_embedding_params(),
    "attn+mlp": get_attn_mlp_params(),
    "attn+mlp+embed": get_all_degradable_params(),
}


def get_param_group(group_name: str) -> List[str]:
    """
    Retrieve a parameter group by name.
    
    Args:
        group_name: Name of the parameter group
                   ("attn_only", "mlp_only", "embed_only", etc.)
    
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


def strip_module_prefix(name: str) -> str:
    """
    Remove 'module.' prefix from parameter name if present.
    
    This handles the case where the model is wrapped in DataParallel,
    which adds a 'module.' prefix to all parameter names.
    
    Args:
        name: Parameter name (possibly with 'module.' prefix)
    
    Returns:
        Parameter name without 'module.' prefix
    
    Example:
        >>> strip_module_prefix("module.layer.weight")
        'layer.weight'
        >>> strip_module_prefix("layer.weight")
        'layer.weight'
    """
    return name[7:] if name.startswith("module.") else name


# ============================================================================
# Validation and introspection
# ============================================================================

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
    import logging
    
    expected_params = set(get_param_group(group_name))
    model_params = {strip_module_prefix(n) for n, _ in model.named_parameters()}
    
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


def get_param_counts() -> Dict[str, int]:
    """
    Get counts of parameters in each group.
    
    Returns:
        Dictionary mapping group names to parameter counts
    
    Example:
        >>> counts = get_param_counts()
        >>> counts['attn_only']
        34
        >>> counts['mlp_only']
        102
    """
    return {name: len(params) for name, params in PARAM_GROUPS.items()}

