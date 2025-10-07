"""
Parameter groups for targeting specific model components during degradation.

This module defines which parameters to target for each degradation experiment.
Currently configured for Gemma-3-4b-IT (34 layers).

WARNING: If using a different model architecture, modify the 
"Gemma-3-4b-IT specific definitions" section.
"""

from typing import Dict, List

# ============================================================================
# Gemma-3-4b-IT specific definitions
# ============================================================================

# Number of layers in Gemma-3-4b
NUM_LAYERS = 34

# Attention parameters (V matrices)
ATTN_PARAMS = [
    f"language_model.model.layers.{i}.self_attn.v_proj.weight"
    for i in range(NUM_LAYERS)
]

# MLP parameters (gate, up, down projections)
MLP_PARAMS = [
    f"language_model.model.layers.{i}.mlp.{proj}_proj.weight"
    for i in range(NUM_LAYERS)
    for proj in ["gate", "up", "down"]
]

# Embedding parameters
EMBED_PARAMS = ["language_model.model.embed_tokens.weight"]

# ============================================================================
# Parameter groups dictionary (for config-based selection)
# ============================================================================

PARAM_GROUPS: Dict[str, List[str]] = {
    "attn_only": ATTN_PARAMS,
    "mlp_only": MLP_PARAMS,
    "embed_only": EMBED_PARAMS,
}


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

