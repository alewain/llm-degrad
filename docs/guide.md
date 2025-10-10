# LLM Degradation Experiments - Complete Guide

This guide provides detailed information about using, extending, and understanding the LLM degradation experiment pipeline.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Understanding TASKS and VARIANTS](#understanding-tasks-and-variants)
5. [CLI Usage](#cli-usage)
6. [Programmatic Usage](#programmatic-usage)
7. [Project Structure](#project-structure)
8. [Execution Flow](#execution-flow)
9. [Configuration System](#configuration-system)
10. [Output Files](#output-files)
11. [Extending the System](#extending-the-system)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

This repository contains the experimental code for the thesis **["Controlled Degradation in a Large Language Model (LLM)"](thesis_spanish.pdf)** (Spanish) by Alejandro Wainstock, supervised by Dr. Enzo Tagliazucchi.

### Research Goals

The project explores how Large Language Models respond to controlled perturbations applied to their internal components (attention layers, MLPs, embeddings). By systematically degrading these components using methods like Gaussian noise, ablation, and quantization, we investigate:
- How different cognitive capabilities deteriorate under damage
- Whether degradation patterns in LLMs mirror human neurological disorders
- Which architectural components are most critical for different types of tasks

### Pipeline Capabilities

This repository provides a **local, reproducible pipeline** for:
- Applying controlled degradations to LLM weights
- Generating text across diverse tasks (math, language, dream narration, image description)
- Collecting structured outputs for downstream analysis

**Note:** The code is focused exclusively on **generation and perturbation**. Analysis notebooks and metrics are separate (see thesis document).

**Current version (v1):** Uses fast in-memory restoration (`subset_in_memory`). Full baseline snapshots may be added in a future version if needed.

---

## Installation & Setup

### Requirements

- **Python:** 3.10 or higher (tested with 3.11)
- **CUDA:** 12.1 or higher
- **GPU:** NVIDIA GPU with at least 24GB VRAM (tested on RTX 3090)
- **Disk space:** Approximately 15GB free (model weights + cache + results)

**Note:** This project requires a GPU with CUDA support. CPU-only execution is not supported.

### Installation Steps

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install PyTorch with CUDA 12.1:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   
   > **Note:** GPU with CUDA 12.1 or higher is required. The project is designed for GPU-only execution.

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Image Setup (required for `cookie_theft_it` task):**
   
   The image for the Cookie Theft picture description task must be obtained from the authors or the publisher PRO-ED, Inc., and placed in the `images/` directory with the filename `image_description.png` at a size of 896×896 pixels (format compatible with Gemma-3-4b-it).
   
   ```bash
   mkdir images
   # Place image_description.png (896×896) in images/ directory
   ```

### HuggingFace Credentials

The project requires a HuggingFace token to download models. There are three ways to provide it:

#### Option 1: `.env` file (Recommended)
1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```
2. Edit `.env` and add your token:
   ```
   HF_TOKEN=your_token_here
   ```
3. The `.env` file is automatically loaded when running experiments
4. **Important:** `.env` is in `.gitignore` and will NOT be committed

#### Option 2: Environment Variable
**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "your_token_here"
```

**Linux/Mac (Bash):**
```bash
export HF_TOKEN="your_token_here"
```

#### Option 3: HuggingFace CLI
```bash
huggingface-cli login
```

**Get your token from:** https://huggingface.co/settings/tokens

**Note:** Never hardcode or commit tokens to the repository.

---

## Quick Start

### Basic CLI Usage

```bash
# Run a single variant on a task:
python -m src.main --task dreams_it --variants gauss_attn

# Run multiple variants by name:
python -m src.main --task iq_it --variants gauss_attn,gauss_mlp,quant_attn

# Run variants by index (1-5):
python -m src.main --task cookie_theft_it --variant-indexes 1-5
python -m src.main --task dreams_it --variant-indexes 1-3
```

### Getting Help

```bash
python -m src.main --help
```

---

## Understanding TASKS and VARIANTS

The experiment system is built on two core concepts: **TASKS** and **VARIANTS**.

### TASKS

A **task** defines the experiment theme:
- Which prompts to use
- Whether images are needed
- Output naming conventions

**Available tasks:**

| Task | Description | Prompts | Image |
|------|-------------|---------|-------|
| `dreams_it` | Dream narration | ~38 | No |
| `iq_it` | Cognitive assessment | ~65 | No |
| `cookie_theft_it` | Image description | ~20 | Yes |

All tasks use the **instruction-tuned (IT)** variant of Gemma-3-4b.

### VARIANTS

A **variant** defines how to degrade the model:
- Degradation method (Gaussian, ablation, quantization)
- Target parameter group (attention, MLP, embeddings)
- Range of degradation levels (min, max, steps)

**Available variants (indexed 1-5):**

| Index | Name | Method | Param Group | Range (min → max) | Steps |
|-------|------|--------|-------------|-------------------|-------|
| 1 | `gauss_attn` | Gaussian noise | Attention | 0.0 → 1.4 | 15 |
| 2 | `gauss_mlp` | Gaussian noise | MLP | 0.0 → 0.5 | 11 |
| 3 | `gauss_embed` | Gaussian noise | Embeddings | 0.0 → 1.0 | 21 |
| 4 | `ablation_attn` | Ablation | Attention | 0.0 → 0.8 | 17 |
| 5 | `quant_attn` | Quantization | Attention | 4 → 1024 | 9 |

### Degradation Methods

- **`mult_gauss`**: Multiplicative Gaussian noise (parameterized by standard deviation)
- **`ablation`**: Set parameters to zero (parameterized by ablation fraction)
- **`uni_quant`**: Uniform quantization (parameterized by number of quantization levels)

### Target Parameter Groups

- **`attn_only`**: Attention V-projection matrices (`v_proj.weight`)
- **`mlp_only`**: Feed-forward network matrices (gate, up, down)
- **`embed_only`**: Token embedding matrix (`embed_tokens.weight`, lookup table)

**Note:** Degradations are applied to the weight values of these matrices across all layers simultaneously.

**Note:** Parameter groups are hardcoded for Gemma-3-4b (34 layers). Using a different model requires manual adaptation.

---

## CLI Usage

### Required Arguments

Both `--task` and one of `--variants` or `--variant-indexes` are required:

```bash
python -m src.main --task TASK_NAME --variants VARIANT_NAMES
# or
python -m src.main --task TASK_NAME --variant-indexes INDEXES
```

### Task Selection

Select one task:
```bash
--task dreams_it
--task iq_it
--task cookie_theft_it
```

### Variant Selection

#### By Name (comma-separated):
```bash
# Single variant
--variants gauss_attn

# Multiple variants
--variants gauss_attn,gauss_mlp,quant_attn
```

#### By Index (1-5):
```bash
# Single index
--variant-indexes 3

# Multiple indexes (comma-separated)
--variant-indexes 1,3,5

# Range (inclusive)
--variant-indexes 1-3  # expands to 1,2,3

# Mixed
--variant-indexes 1-3,5  # expands to 1,2,3,5

# All variants
--variant-indexes 1-5
```

### Optional Arguments

```bash
--log-level {DEBUG,INFO,WARNING,ERROR}  # Default: INFO
```

### Examples

```bash
# Run Gaussian attention variant on dreams task
python -m src.main --task dreams_it --variants gauss_attn

# Run all 5 variants on IQ task
python -m src.main --task iq_it --variant-indexes 1-5

# Run specific variants by index on cookie theft (with image)
python -m src.main --task cookie_theft_it --variant-indexes 2,4

# Multiple variants by name with debug logging
python -m src.main --task dreams_it --variants gauss_attn,ablation_attn --log-level DEBUG
```

### Running Multiple Experiments

To run all tasks with all variants (bash loop):
```bash
for task in dreams_it iq_it cookie_theft_it; do
    python -m src.main --task $task --variant-indexes 1-5
done
```

---

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

---

## Project Structure

```
Repo_nuevo/
├── src/                        # Core Python modules
│   ├── main.py                 # CLI entry point
│   ├── pipeline.py             # Experiment orchestration & persistence
│   ├── model_loader.py         # Model/tokenizer loading & restoration
│   ├── degradation.py          # Parameter groups & degradation methods
│   ├── generation.py           # Text generation & perplexity
│   └── utils.py                # Logging, seeds, VRAM, image utils
│
├── configs/                    # Configuration modules
│   ├── experiment_configs.py   # TASKS, VARIANTS, build_config()
│   └── prompts.py              # Prompt lists by task
│
├── results/                    # Experiment outputs (created at runtime)
│   └── *.json                  # Full results (not versioned)
│
├── logs/                       # Execution logs (created at runtime)
│   └── *.log
│
├── docs/                       # Documentation
│   ├── guide.md                # This file (complete guide)
│   ├── output_schema.md        # JSON output schema reference
│   └── thesis_spanish.pdf      # Thesis document (Spanish)
│
├── notebooks/                  # Analysis notebooks (future)
│
├── requirements.txt            # Python dependencies
├── env.example                 # Example .env file
└── README.md                   # Quick start & overview
```

### Core Modules (src/)

#### `main.py`
- CLI entry point
- Argument parsing (--task, --variants, --variant-indexes)
- Loads .env file
- Orchestrates multiple experiment runs

#### `pipeline.py`
- Experiment orchestration
- Functions:
  - `setup_experiment()`: Initialization & model loading
  - `run_experiment_loop()`: Main degradation → generation loop
  - `finalize_experiment()`: Save final results & statistics
  - `run_experiment()`: High-level orchestrator
- JSON persistence with resume capability

#### `model_loader.py`
- Model and tokenizer loading via HuggingFace + Unsloth
- `subset_in_memory` restoration strategy:
  - Saves degradable parameter subset to CPU memory
  - Fast restoration before each experiment repetition
- Functions:
  - `load_model_and_tokenizer()`: Main loading function
  - `create_baseline_subset()`: Create CPU memory baseline
  - `restore_from_baseline()`: Fast parameter restoration

#### `degradation.py`
- **Parameter groups** (Gemma-3-4b specific):
  - `get_attn_params()`, `get_mlp_params()`, `get_embedding_params()`
  - `PARAM_GROUPS` registry
  - `get_param_group()`: Get parameters by group name
- **Degradation methods:**
  - `apply_degradation()`: Main degradation function
  - `quantise_tensor_uniform()`: Uniform quantization helper
  - Supports: `mult_gauss`, `ablation`, `uni_quant`

#### `generation.py`
- Text generation with optional image support
- Functions:
  - `wrap_chat_it()`: Format prompts for instruction-tuned models
  - `generate_text()`: Main generation function (batched)
  - `evaluate_perplexity()`: Optional perplexity calculation

#### `utils.py`
Three main categories:
1. **General utilities:**
   - `setup_logging()`: Dual logging (console + file)
   - `set_all_seeds()`: Reproducibility (random, numpy, torch)
   - `get_model_memory_footprint()`: GPU memory tracking
2. **VRAM monitoring:**
   - `calculate_vram_percentage()`: Current VRAM usage
   - `adjust_batch_size_by_vram()`: Dynamic batch size adjustment
3. **Image support:**
   - `load_image()`: PIL image loader
   - `prepare_prompt_with_image()`: Add `<image>` tags to prompts

---

## Execution Flow

This section provides a high-level overview of what happens when you run an experiment.

### Complete Pipeline

```
1. Initialization
   ├─ Load experiment config (TASK + VARIANT → ExperimentConfig)
   ├─ Setup logging (console + file)
   ├─ Set base random seeds
   └─ Load existing results JSON (if resuming)

2. Model Loading
   ├─ Download/load model from HuggingFace (or local cache)
   ├─ Load tokenizer
   ├─ Load image processor (if image_enabled=True)
   ├─ Move model to GPU
   └─ Create baseline subset in CPU memory (params to degrade)

3. Generate Degradation Levels
   └─ Compute degradation values (geometric/linear spacing)
      Examples:
      - mult_gauss: [0.0, 0.35, 0.7, 1.05, 1.4]
      - ablation: [0.0, 0.2, 0.4, 0.6, 0.8]
      - uni_quant: [1024, 256, 64, 16, 4]

4. For each degradation level:
   │
   ├─ For each repetition (repeat_index = 0 to n_rep-1):
   │  │
   │  ├─ Set repetition-specific seed (base_seed + repeat_index)
   │  │
   │  ├─ Restore model from baseline subset
   │  │  └─ Fast: copy tensors from CPU memory (no disk I/O)
   │  │
   │  ├─ Apply degradation (if level > 0)
   │  │  ├─ mult_gauss: multiply weights by Gaussian noise
   │  │  ├─ ablation: randomly mask weights to zero
   │  │  └─ uni_quant: quantize weights to N levels
   │  │
   │  ├─ [Optional] Calculate perplexity (once per level, not per rep)
   │  │
   │  ├─ For each batch of prompts:
   │  │  │
   │  │  ├─ Check VRAM usage
   │  │  │  ├─ If > 95%: abort experiment
   │  │  │  ├─ If > 90%: warning + pause
   │  │  │  └─ If < 40%: increase batch size
   │  │  │
   │  │  ├─ Prepare prompts (add IT format / image tags)
   │  │  │
   │  │  ├─ Generate text (batched)
   │  │  │  └─ Uses model.generate() with sampling parameters
   │  │  │
   │  │  ├─ Parse outputs
   │  │  │
   │  │  └─ Save results to JSON (incremental)
   │  │
   │  └─ Log statistics (tokens/sec, VRAM usage)
   │
   └─ Move to next level

5. Finalization
   ├─ Save final results JSON
   ├─ Log total experiment time
   └─ Print summary statistics
```

### Resume Capability

The pipeline is **interrupt-resilient**:
- Results are saved incrementally after each prompt generation
- On resume, the system:
  1. Loads existing results JSON
  2. Builds a set of computed (prompt, level, repetition) tuples
  3. Skips prompts that are already computed
  4. Continues from where it left off

This allows you to stop experiments (Ctrl+C) and resume them later without losing progress.

---

## Configuration System

### The TASKS × VARIANTS Model

Experiments are defined by composing a **TASK** (experiment theme) with a **VARIANT** (degradation specification):

```python
config = build_config(task_key="dreams_it", variant_key="gauss_attn")
```

This generates a complete `ExperimentConfig` with all necessary parameters.

### ExperimentConfig Dataclass

All experiments use a single dataclass (`ExperimentConfig`) with fields:

**Experiment identification:**
- `config_name`: Auto-generated as `f"{task}__{variant}"`
- `prompts`: List of prompt strings
- `name_suffix`: For output file naming (auto-generated from task key)

**Model configuration:**
- `model_name`: HuggingFace model identifier
- `device`: GPU device (default: "cuda:0")
- `dtype`: Tensor precision (default: "float32")
- `load_in_4bit`: Enable 4-bit quantization (default: False)

**Degradation configuration:**
- `param_group_name`: Target parameters ("attn_only", "mlp_only", "embed_only")
- `degradation_method`: Method ("mult_gauss", "ablation", "uni_quant")
- `min_deg`, `max_deg`, `deg_steps`: Degradation range

**Experiment parameters:**
- `n_rep`: Repetitions per degradation level (default: 10)
- `seed`: Base random seed (default: 42)
- `max_batch_size`: Maximum batch size (default: 40)

**Generation parameters:**
- `temperature`: Sampling temperature (default: 1.0)
- `do_sample`: Enable sampling (default: True)
- `max_new_tokens`: Maximum tokens to generate (default: 350)

**Optional features:**
- `image_enabled`: Load processor & image (default: False)
- `image_filename`: Image path (default: "DescribePictureOK.png")
- `compute_perplexity`: Calculate perplexity (default: False)
- `perplexity_text`: Text for perplexity evaluation

### Overriding Configuration Fields

You can override any field when building a configuration:

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

---

## Output Files

### File Naming

Results are saved as JSON files with the naming scheme:
```
results/outputs_{method}_{model}_{task}.json
```

Examples:
- `outputs_mult_gauss_gemma-3-4b-it_dreams_it.json`
- `outputs_uni_quant_gemma-3-4b-it_iq_it.json`
- `outputs_ablation_gemma-3-4b-it_cookie_theft_it.json`

### JSON Structure

Each result is a JSON array of objects. Each object represents one generation:

```json
{
  "timestamp": "2025-10-07 14:30:15",
  "config_name": "dreams_it__gauss_attn",
  "model_name": "google/gemma-3-4b-it",
  "degradation_method": "mult_gauss",
  "param_group_name": "attn_only",
  "degrad_level": 0.7,
  "repeat_index": 3,
  "prompt_id": 12,
  "prompt_text": "Describe a recurring dream...",
  "output": "In my recurring dream, I find myself...",
  "temperature": 1.0,
  "do_sample": true,
  "seed": 42003,
  "duration": 2.34,
  "tokens": 312
}
```

For the complete schema with all fields, see [`output_schema.md`](output_schema.md).

### Output Organization

- **Full experiment outputs** (`results/*.json`): Generated by each run, can be very large (50+ MB). Not versioned.
- The `results/` and `logs/` directories are created automatically at runtime if they don't exist.

---

## Extending the System

### Adding a New Task

1. **Add prompts** to `configs/prompts.py`:
   ```python
   my_new_prompts = [
       "Prompt 1",
       "Prompt 2",
       # ...
   ]
   ```

2. **Add task to registry** in `configs/experiment_configs.py`:
   ```python
   from configs.prompts import my_new_prompts
   
   TASKS["my_new_task"] = ExperimentConfig(
       config_name="",  # Auto-generated
       prompts=my_new_prompts,
       image_enabled=False,  # Or True if needed
       # Other task-specific settings
   )
   ```

3. **Use it:**
   ```bash
   python -m src.main --task my_new_task --variants gauss_attn
   ```

### Adding a New Variant

1. **Add variant to registry** in `configs/experiment_configs.py`:
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

2. **Use it:**
   ```bash
   python -m src.main --task dreams_it --variants my_new_variant
   # Or by index (if added as 6th variant):
   python -m src.main --task dreams_it --variant-indexes 6
   ```

### Adding a New Degradation Method

To add a new degradation method, modify `src/degradation.py`:

```python
def apply_degradation(model, param_names, degrad_level, method="mult_gauss"):
    # ... existing code ...
    
    elif method == "my_new_method":
        for name, param in model.named_parameters():
            if name in param_names:
                with torch.no_grad():
                    # Implement your degradation logic here
                    # Example: param.data = some_transformation(param.data, degrad_level)
                    pass
```

Then use it in a variant:
```python
VARIANTS["my_variant"] = {
    "degradation_method": "my_new_method",
    # ...
}
```

---

## Troubleshooting

### Common Issues

#### Import Errors
**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### HuggingFace Authentication
**Problem:** `HTTPError: 401 Unauthorized`

**Solution:** Set up your HuggingFace token (see [Installation & Setup](#installation--setup))

#### CUDA Out of Memory
**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce `max_batch_size` in config
- Enable 4-bit quantization: `load_in_4bit=True`
- Use a smaller model
- Close other GPU processes

#### Model Not Found
**Problem:** Model download fails or hangs

**Solutions:**
- Check internet connection
- Verify HuggingFace token permissions
- Clear cache: `rm -rf ~/.cache/huggingface/`
- Try manual download first

#### Interrupted Experiments
**Problem:** Experiment was interrupted (Ctrl+C, crash, etc.)

**Solution:** Simply run the same command again. The pipeline will automatically resume from where it left off.

### Getting Help

- Check logs in `logs/` directory
- Use `--log-level DEBUG` for verbose output
- Review the [output schema](output_schema.md) for JSON structure
- Consult the thesis document for methodology details

---

## Additional Resources

- **Output Schema Reference:** [`output_schema.md`](output_schema.md)
- **Thesis Document:** [thesis_spanish.pdf](thesis_spanish.pdf) (Spanish, 2 MB)
- **HuggingFace Model:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

---

*Last updated: October 2025*

