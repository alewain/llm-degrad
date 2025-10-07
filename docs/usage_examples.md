# Usage Examples

This document provides examples of how to use the experiment configuration system.

## CLI Usage

### Running Single Experiments

```bash
# Run a single variant on dreams task
python -m src.main --task dreams_it --variants gauss_attn

# Run a single variant on IQ task
python -m src.main --task iq_it --variants quant_attn

# Run a single variant on cookie theft (with image)
python -m src.main --task cookie_theft_it --variants ablation_attn
```

### Running Multiple Variants

```bash
# Run 3 variants by name
python -m src.main --task dreams_it --variants gauss_attn,gauss_mlp,gauss_embed

# Run all 5 variants by index
python -m src.main --task iq_it --variant-indexes 1-5

# Run specific variants by index
python -m src.main --task cookie_theft_it --variant-indexes 1,3,5
```

### Index Syntax

The `--variant-indexes` argument supports flexible syntax:

- Single index: `--variant-indexes 3`
- Comma-separated: `--variant-indexes 1,3,5`
- Range (inclusive): `--variant-indexes 1-3` (expands to 1,2,3)
- Mixed: `--variant-indexes 1-3,5` (expands to 1,2,3,5)

## Programmatic Usage

### Building Configurations

```python
from configs.experiment_configs import build_config

# Build a single configuration
cfg = build_config("dreams_it", "gauss_attn")

# Build with overrides
cfg_custom = build_config("iq_it", "quant_attn", n_rep=5, temperature=0.8)

# Access configuration fields
print(f"Task: {cfg.config_name}")
print(f"Method: {cfg.degradation_method}")
print(f"Param group: {cfg.param_group_name}")
print(f"Range: {cfg.min_deg} → {cfg.max_deg}")
print(f"Prompts: {len(cfg.prompts)}")
```

### Using the Registry

```python
from configs.experiment_configs import TASKS, VARIANTS, VARIANTS_ORDERED

# List all available tasks
print("Available tasks:", list(TASKS.keys()))

# List all available variants
print("Available variants:", list(VARIANTS.keys()))

# Get variant by index
from configs.experiment_configs import get_variant_by_index
variant_name = get_variant_by_index(1)  # "gauss_attn"

# Inspect a variant's parameters
variant_params = VARIANTS["gauss_attn"]
print(variant_params)
# Output: {'degradation_method': 'mult_gauss', 'param_group_name': 'attn_only', ...}
```

### Running Experiments Programmatically

```python
from configs.experiment_configs import build_config
from src.pipeline import run_experiment
from src.utils import setup_logging
import logging

# Setup logging
setup_logging("logs/my_experiment.log", level=logging.INFO)

# Build configuration
cfg = build_config("dreams_it", "gauss_attn")

# Run experiment
run_experiment(cfg)
```

### Building Multiple Configurations

```python
from configs.experiment_configs import build_config, VARIANTS_ORDERED

# Build all 5 variants for a single task
task = "dreams_it"
configs = [build_config(task, variant) for variant in VARIANTS_ORDERED]

# Run all experiments
for cfg in configs:
    print(f"Running: {cfg.config_name}")
    run_experiment(cfg)
```

## Adding New Tasks

To add a new task, edit `configs/experiment_configs.py`:

```python
from configs.prompts import my_new_prompts

TASKS["my_new_task"] = ExperimentConfig(
    config_name="my_new_task_base",
    prompts=my_new_prompts,
    image_enabled=False,
    max_new_tokens=350,
    n_rep=10,
    # ... other defaults
)
```

Then use it:
```bash
python -m src.main --task my_new_task --variants gauss_attn
```

## Adding New Variants

To add a new variant, edit `configs/experiment_configs.py`:

```python
VARIANTS["my_new_variant"] = {
    "degradation_method": "mult_gauss",
    "param_group_name": "mlp_only",
    "min_deg": 0.0,
    "max_deg": 2.0,
    "deg_steps": 20,
}

# Add to ordered list for indexing
VARIANTS_ORDERED.append("my_new_variant")
```

Then use it:
```bash
python -m src.main --task dreams_it --variants my_new_variant
# Or by index (if added as 6th variant):
python -m src.main --task dreams_it --variant-indexes 6
```

## Overriding Configuration Fields

You can override any configuration field when building:

```python
cfg = build_config(
    "dreams_it", 
    "gauss_attn",
    n_rep=5,              # Fewer repetitions
    temperature=0.7,      # Lower temperature
    max_new_tokens=500,   # Longer generations
    seed=123,             # Different seed
)
```

## Common Patterns

### Running All Variants on All Tasks

```bash
# Bash loop
for task in dreams_it iq_it cookie_theft_it; do
    python -m src.main --task $task --variant-indexes 1-5
done
```

```python
# Python script
from configs.experiment_configs import build_config, TASKS, VARIANTS_ORDERED
from src.pipeline import run_experiment

for task_key in TASKS.keys():
    for variant_key in VARIANTS_ORDERED:
        cfg = build_config(task_key, variant_key)
        run_experiment(cfg)
```

### Dry Run (Testing Configuration)

```python
# Build configuration without running
cfg = build_config("dreams_it", "gauss_attn")

# Inspect configuration
print(f"Config: {cfg.config_name}")
print(f"Method: {cfg.degradation_method}")
print(f"Prompts: {len(cfg.prompts)}")
print(f"Repetitions: {cfg.n_rep}")
print(f"Degradation levels: {cfg.deg_steps}")
print(f"Range: [{cfg.min_deg}, {cfg.max_deg}]")

# Estimate total generations
total_gens = len(cfg.prompts) * cfg.n_rep * cfg.deg_steps
print(f"Estimated total generations: {total_gens}")
```

