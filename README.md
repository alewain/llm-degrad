# LLM Degradation Experiments

## About

This repository contains the experimental code for the thesis **"Controlled Degradation in a Large Language Model (LLM)"** by Alejandro Wainstock, supervised by Dr. Enzo Tagliazucchi.

The project explores how Large Language Models respond to controlled perturbations applied to their internal components (attention layers, MLPs, embeddings). By systematically degrading these components using methods like Gaussian noise, ablation, and quantization, we investigate:
- How different cognitive capabilities deteriorate under damage
- Whether degradation patterns in LLMs mirror human neurological disorders
- Which architectural components are most critical for different types of tasks

This repository provides a **local, reproducible pipeline** for:
- Applying controlled degradations to LLM weights
- Generating text across diverse tasks (math, language, dream narration, image description)
- Collecting structured outputs for downstream analysis

The code is focused exclusively on **generation and perturbation**. Analysis notebooks and metrics are separate (see thesis document).

**Current version (v1):** Uses fast in-memory restoration (`subset_in_memory`). Full baseline snapshots may be added in a future version if needed.

## Requirements
- Python 3.10 or higher (tested with 3.11)
- CUDA 12.1 or higher
- NVIDIA GPU with at least 24GB VRAM (tested on RTX 3090)
- Approximately 15GB free disk space (model weights + cache + results)

## Installation
- Create and activate a virtual environment.
- Install PyTorch with CUDA 12.1:
  
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  
  > **Requisito:** GPU con soporte CUDA 12.1 o superior.  
  > El proyecto está diseñado para ejecución exclusiva en GPU.

- Install the remaining dependencies with `pip install -r requirements.txt`.

## Credentials

The project requires a HuggingFace token to download models. There are three ways to provide it:

### Option 1: `.env` file (Recommended)
1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```
2. Edit `.env` and add your token:
   ```bash
   HF_TOKEN=your_token_here
   ```
3. The `.env` file is automatically loaded when running experiments
4. **Important:** `.env` is in `.gitignore` and will NOT be committed

### Option 2: Environment Variable
**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "your_token_here"
```

**Linux/Mac (Bash):**
```bash
export HF_TOKEN="your_token_here"
```

### Option 3: HuggingFace CLI
```bash
huggingface-cli login
```

**Get your token from:** https://huggingface.co/settings/tokens

**Note:** Never hardcode or commit tokens to the repository.

## Proposed layout (target structure after migration)
- `src/`: all Python code (modules + CLI entry point)
  - **Core modules (5 files):**
    - `model_loader.py` - Loads model/tokenizer from HuggingFace and implements fast in-memory restoration of degraded parameters.
    - `degradation.py` - Defines parameter groups (attention, MLP, embeddings) and implements degradation methods (mult_gauss, ablation, uni_quant), hardcoded for Gemma-3-4b.
    - `generation.py` - Handles text generation with optional image support and perplexity calculation.
    - `pipeline.py` - Orchestrates the complete experiment flow and manages JSON persistence with resume capability.
    - `utils.py` - Provides logging setup, seed management, VRAM monitoring, and image utilities.
  - **Entry point:**
    - `main.py` - CLI script to execute experiments by config name.
- `configs/`: Python configurations
  - `experiment_configs.py`: dataclass-based experiment configurations
  - `prompts.py`: prompt lists organized by task and model variant
- `results/`: JSON outputs (one file per run)
  - `samples/`: small example outputs (versioned, for notebooks and documentation)
  - Full experiment outputs (not versioned, too large)
- `logs/`: log files (one per run)
- `notebooks/`: light analysis notebooks (read small samples from `results/samples/`)
- `docs/`: documentation and thesis link

## Quickstart
```bash
# Run one of the three main experiments:
python -m src.run_experiment --config dreams_it
python -m src.run_experiment --config iq_it
python -m src.run_experiment --config cookie_theft_it

# Or directly:
python src/main.py --config dreams_it
```

## Available experiments
This repository includes three main experiments, all using the **instruction-tuned (IT)** variant of Gemma-3-4b:

1. **`dreams_it`**: Dream narration task (~38 prompts)
   - Prompts asking the model to generate detailed dream narratives
   - Tests creative and narrative capabilities under degradation

2. **`iq_it`**: Multi-task cognitive assessment (~65 prompts)
   - Math problems, language tasks, logic puzzles, factual questions, creativity
   - Comprehensive evaluation across different cognitive domains

3. **`cookie_theft_it`**: Image description task (~20 prompts)
   - Cookie Theft picture description (classic neuropsychological assessment)
   - Tests multimodal capabilities and structured visual description

**Note:** Pretrained (PT) variants are not included in this version.

## Execution flow

This section provides a high-level overview of what happens when you run an experiment. Understanding this flow helps you interpret the logs and debug issues.

### Complete pipeline

```
1. Initialization
   ├─ Load experiment config (dataclass)
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
   │  ├─ For each prompt in prompt list:
   │  │  │
   │  │  ├─ Check if already computed (resume logic)
   │  │  │  └─ If exists: skip ✓
   │  │  │  └─ If missing: continue ↓
   │  │  │
   │  │  ├─ Process in batches (batch_size = min(n_prompts, max_batch_size))
   │  │  │  ├─ Prepare inputs (text + optional image)
   │  │  │  ├─ Generate text (model.generate)
   │  │  │  ├─ Decode outputs
   │  │  │  └─ Measure VRAM usage
   │  │  │
   │  │  ├─ Create JSON record with all metadata
   │  │  │
   │  │  ├─ Append to results array
   │  │  │
   │  │  └─ Save JSON periodically (every 20 prompts)
   │  │
   │  └─ Save JSON after completing repetition
   │
   └─ Continue to next degradation level

5. Cleanup
   ├─ Restore model to baseline (optional, for consistency)
   ├─ Save final JSON
   ├─ Clear GPU cache
   └─ Close log file
```

### Key points

- **Baseline restoration:** Model is restored from CPU memory before each repetition (fast, ~1-3 seconds)
- **Degradation application:** Applied once per repetition, affects all prompts in that repetition
- **Batch processing:** Prompts processed in batches to optimize GPU utilization
- **Resume logic:** Checks existing JSON at prompt level - only missing prompts are executed
- **Periodic saves:** Results saved every 20 prompts to prevent data loss on interruption
- **VRAM monitoring:** Batch size adjusted dynamically based on VRAM usage

### Typical runtime

For reference, approximate times for a full experiment on RTX 3090 (24GB VRAM):

- **dreams_it:** ~38 prompts × 5 levels × 10 reps = 1,900 generations → ~2-3 hours
- **iq_it:** ~65 prompts × 5 levels × 10 reps = 3,250 generations → ~4-5 hours  
- **cookie_theft_it:** ~20 prompts × 5 levels × 10 reps = 1,000 generations → ~1.5-2 hours

*Times vary based on `max_new_tokens`, model size, and VRAM availability.*

## Configuration system
Configurations are defined as Python dataclasses in `configs/experiment_configs.py`. This approach provides:
- **Type safety** with automatic validation
- **IDE autocomplete** and type checking
- **Composability** through inheritance and merge operations
- **Python expressiveness** (computations, conditionals, imports)

### Example configuration
**File: `configs/experiment_configs.py`**
```python
from dataclasses import dataclass, field
from typing import List, Literal
from configs.prompts import dream_prompts_it, iq_prompts_it, cookie_theft_prompts_it

@dataclass
class ExperimentConfig:
    """Base configuration for all experiments."""
    config_name: str
    prompts: List[str]  # List of prompts to use
    
    model_name: str = "google/gemma-3-4b-it"
    model_variant: Literal["it"] = "it"  # Only instruction-tuned variant supported
    device: str = "cuda:0"
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    load_in_4bit: bool = False
    restore_strategy: str = "subset_in_memory"
    
    param_group_name: Literal["attn_only", "mlp_only", "embed_only"] = "attn_only"
    degradation_method: Literal["mult_gauss", "ablation", "uni_quant"] = "uni_quant"
    
    min_deg: float = 2
    max_deg: float = 256
    deg_steps: int = 5
    n_rep: int = 10
    
    max_batch_size: int = 40    # Maximum batch size: upper limit for simultaneous prompt processing
    
    seed: int = 42
    temperature: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 350
    
    image_enabled: bool = False
    image_filename: str = "DescribePictureOK.png"
    
    compute_perplexity: bool = False
    perplexity_text: str = "The quick brown fox jumps over the lazy dog."
    
    name_suffix: str = ""

# ===== Three main experiment configurations (all IT) =====

dreams_it = ExperimentConfig(
    config_name="dreams_it",
    prompts=dream_prompts_it,
    name_suffix="2025_05_20_dreams",
    n_rep=10,
    max_new_tokens=350
)

iq_it = ExperimentConfig(
    config_name="iq_it",
    prompts=iq_prompts_it,
    name_suffix="2025_05_04_IQ",
    n_rep=10,
    max_new_tokens=350
)

cookie_theft_it = ExperimentConfig(
    config_name="cookie_theft_it",
    prompts=cookie_theft_prompts_it,
    name_suffix="2025_06_24_cookies",
    image_enabled=True,
    max_new_tokens=350,
    n_rep=10
)

# Helper function to get config by name
def get_config(name: str) -> ExperimentConfig:
    configs = {
        "dreams_it": dreams_it,
        "iq_it": iq_it,
        "cookie_theft_it": cookie_theft_it,
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")
    return configs[name]
```

**File: `configs/prompts.py`**
```python
"""Prompt lists for the three main experiments (all instruction-tuned)."""

# Dream narration prompts
dream_prompts_it = [
    "Just narrate a dream with as much detail as you can.",
    "Just describe a dream as vividly and narratively as possible.",
    "Tell me about a dream.",
    "Describe a dream.",
    "Share a dream.",
    # ... (38 prompts total)
]

# Multi-task IQ assessment (math, language, logic, factual, creativity)
iq_prompts_it = [
    # Math
    "What number completes the sequence: 93, 94, 95, __, 97?",
    "What is the result of 3 + 48 / 6 - 7?",
    # Language
    "Which word comes next: January, March, May, ___?",
    "Find the antonym of 'scarce' among these: abundant, limited, rare, minimal",
    # Logic
    "All cats are animals. All animals are floms. Can we conclude that all cats are floms?",
    # Factual
    "How many sides does a hexagon have?",
    "What is the capital of France?",
    # Creativity
    "List as many animals as you can.",
    # ... (~65 prompts total across all categories)
]

# Cookie Theft picture description prompts
cookie_theft_prompts_it = [
    "Tell me everything you see going on in this picture.",
    "Describe everything happening in this picture.",
    "Describe this picture in as much detail as possible.",
    # ... (20 prompts total)
]
```

### Usage in scripts
```python
from configs.experiment_configs import get_config

# Get config by name
config = get_config("dreams_it")

# Override specific fields if needed
config.n_rep = 5
config.deg_steps = 10
```

### Key configuration parameters

- **`max_batch_size`**: Upper limit for simultaneous prompt processing (default: 40). This prevents VRAM overflow during generation.
- **`n_prompts`** (runtime-calculated): Total number of prompts in the config's prompt list. ALL prompts are always processed.
- **`batch_size`** (runtime-calculated): Effective batch size, computed as `min(n_prompts, max_batch_size)`.
  - If `n_prompts <= max_batch_size`: all prompts processed in one batch
  - If `n_prompts > max_batch_size`: divided into multiple batches of size `max_batch_size`
  - Example: 100 prompts with `max_batch_size=40` → 3 batches: [40, 40, 20]
- **`batch_size_effective`** (output field): The actual batch size used for each generation, may vary due to dynamic VRAM adjustments.
- **`n_rep`**: Number of independent repetitions for each degradation level (each with a different random seed).

## Prompts organization
Prompts are maintained as Python lists in `configs/prompts.py` for simplicity and direct integration with configurations.

**Benefits of Python lists over JSON:**
- **Direct imports:** No file I/O or parsing needed
- **IDE support:** Autocomplete, syntax highlighting, refactoring
- **Type safety:** Static analysis catches errors early
- **Simplicity:** No additional parsing logic required

**Naming convention:**
- `{task}_prompts_it` (e.g., `dream_prompts_it`, `iq_prompts_it`, `cookie_theft_prompts_it`)
- All prompts use the `_it` suffix (instruction-tuned)
- Three lists total: one per experiment

## Output format

Each experiment generates a JSON file with one record per generated output. 

**Example record (key fields):**
```json
{
  "timestamp": "2025-10-06T12:34:56.789Z",
  "model_name": "google/gemma-3-4b-it",
  "degradation_method": "uni_quant",
  "level_value": 64,
  "param_group_name": "attn_only",
  "repeat_index": 0,
  "prompt_group": "dreams",
  "prompt_id": 1,
  "prompt_text": "Just narrate a dream with as much detail as you can.",
  "output": "... generated text ...",
  "duration": 1.23,
  "tokens": 312,
  "seed": 42001
}
```

**Compatibility note:** The output schema maintains all fields from the original `experimento.py` and adds new ones incrementally. Existing analysis notebooks continue working with the original fields.

For the complete JSON schema with all fields and their descriptions, see [`docs/output_schema.md`](docs/output_schema.md).

## Output files
Results are saved in `results/` with the naming scheme:
```
outputs_{degradation_method}_{model_name_clean}_{name_suffix}.json
```

Where `model_name_clean` is the model name with the organization prefix removed and slashes replaced (e.g., `google/gemma-3-4b-it` becomes `gemma-3-4b-it`).

Example:
```
outputs_uni_quant_gemma-3-4b-it_2025_05_20_dreams.json
```

## Logs
Each experiment generates a log file in `logs/<experiment_id>_<timestamp>.log` using Python's standard `logging` module:
- **Configuration:** Dual output (console + file) with UTF-8 encoding
- **Format:** Simple message format (timestamps already in messages)
- **Default level:** INFO (can be set to DEBUG for development)
- **Content:** Everything that appears on screen
  - Model loading and restoration
  - Degradation application
  - VRAM usage and batch size adjustments
  - Progress updates (prompts processed, time estimates)
  - Warnings and errors
  - **Does NOT duplicate** generated outputs (those are only in JSON)
- **Use case:** Debugging, monitoring progress, post-mortem analysis

Example:
```bash
# Monitor logs in real-time
tail -f logs/uni_quant_gemma-3-4b-it_20251006_143022.log
```

## Restoration policy
- **Strategy implemented:** `subset_in_memory`
  - Fast: restores only degraded parameters from memory (no disk I/O).
  - The model is restored **at the start of each repetition** (before applying degradation).
  - At the beginning of the experiment, the subset of parameters to be degraded is saved in CPU memory as `.clone()`.
  - Before each repetition, the model is restored from this in-memory subset.
  - Each experiment creates its own independent baseline in memory.

## Degradation methods
- **`mult_gauss`:** Multiplicative Gaussian noise
- **`ablation`:** Random masking (set weights to zero)
- **`uni_quant`:** Uniform quantization (geometric spacing)

**Note:** Previous methods `uni_quant_lineal` and `lognorm` from the original code are not included in this version.

## Parameter groups
Three parameter groups are defined in `src/degradation.py`:
- **attn_only**: Attention value projection parameters (34 layers)
- **mlp_only**: MLP feed-forward parameters - gate, up, and down projections (102 parameters across 34 layers)
- **embed_only**: Embedding layer parameters (1 parameter)

**Important:** The current implementation is **hardcoded for Gemma-3-4b** (34 layers). If using a different model architecture, the parameter group definitions in `degradation.py` must be manually adapted to match the new model's layer count and naming conventions.

## Quantization (4-bit)
The pipeline supports 4-bit quantization to reduce VRAM usage:
- **Disabled by default:** `load_in_4bit=False` → `load_4bit: false` in outputs
- **When enabled:** `load_in_4bit=True` → `load_4bit: true` in outputs
- **Use case:** Useful for running experiments on GPUs with limited VRAM
- **Trade-off:** Slightly reduced precision but significantly lower memory footprint
- **Note:** This refers to model weight quantization during loading, NOT the `uni_quant` degradation method

To enable 4-bit quantization, set `load_in_4bit=True` in your Python config dataclass.

## Resuming interrupted experiments

The pipeline automatically resumes interrupted experiments without re-executing completed work:

**How it works:**
1. At startup, the system loads the existing JSON output file (if present)
2. Builds a set of already-computed prompts based on: `(param_group_name, std_dev, repeat_index, degradation_method, prompt_text)`
3. Skips prompts that have already been computed
4. Only executes missing prompts
5. Saves results periodically (every 20 prompts) to minimize data loss on interruption

**Example:**
```bash
# Start experiment
python src/main.py --config dreams_it
# ... gets interrupted after processing 50% of prompts ...

# Resume (same command)
python src/main.py --config dreams_it
# ✅ Automatically detects completed work and continues from where it left off
```

**Note:** There is currently no `force_run` flag to override this behavior. The system always resumes from existing results. If you need to regenerate outputs, manually delete or rename the output JSON file before re-running. A `force_run` flag may be added in a future version (see "Limitations and Future Work").

## Seeds and reproducibility
Seeds are derived from a base seed to ensure reproducibility while allowing variation across repetitions:
- A base `seed` is specified in the config (e.g., `seed: 42`)
- For each repetition (`repeat_index` from 0 to `n_rep-1`), the actual seed used is: `base_seed + repeat_index`
- Example: with `seed=42` and `n_rep=10`:
  - Repetition 0 uses seed 42
  - Repetition 1 uses seed 43
  - ...
  - Repetition 9 uses seed 51
- This ensures **reproducible outputs** when re-running with the same config
- Different repetitions generate **different outputs** (as intended for statistical variability across the 10 independent runs)

**Implementation:** A centralized `set_all_seeds(seed)` function sets all random number generators:
- `random.seed()`: Python stdlib random module
- `np.random.seed()`: NumPy's random generator
- `torch.manual_seed()`: PyTorch CPU operations
- `torch.cuda.manual_seed_all()`: PyTorch CUDA (all available GPUs)

This ensures complete reproducibility across all stochastic operations in the pipeline.

## VRAM management
- **Adaptive heuristic** adjusts `batch_size` according to available VRAM:
  - If VRAM > 95%: save results and abort experiment
  - If VRAM > 90%: warning and brief pause
  - If VRAM < 40%: automatically increase `batch_size` (up to `max_batch_size`)
- The effective `batch_size_effective` is reported in outputs.
- Optional `dry-run` mode to estimate memory before large runs.

## Image input (Cookie Theft)
- When `image_enabled=True`, `max_seq_length` is automatically adjusted to 1024 (vs. 512 by default).
- Image processing uses `AutoProcessor` and is integrated into `generate_text` via optional parameters.
- Image loading and preparation handled in the image support section of `utils.py`.

## Perplexity calculation (optional)
Perplexity evaluation is **disabled by default** but can be enabled per experiment:
- **Module:** `src/generation.py` (integrated with generation functions)
- **Config fields:**
  - `compute_perplexity: bool = False` (set to `True` to enable)
  - `perplexity_text: str = "..."` (text to evaluate, only used if enabled)
- **Behavior:** When enabled, perplexity is calculated **once per degradation level** (not per repetition)
- **Output:** Results saved in optional `perplexity` field in JSON output
- **Use case:** Useful for comparing model quality across degradation levels, but not required for main experiments

## .gitignore (suggested)
```
.env

# Exclude full experiment results (large files)
results/*.json

# But include small samples (for notebooks and documentation)
!results/samples/
!results/samples/*.json

# Exclude logs (can be large and session-specific)
logs/
*.log

# Python artifacts
__pycache__/
.ipynb_checkpoints/
*.pyc

# Model artifacts (if downloaded locally)
*.pt
*.safetensors
```

### Results structure
- **Full experiment outputs** (`results/*.json`): Generated by each run, can be very large (50+ MB). Not versioned.
- **Sample outputs** (`results/samples/*.json`): Small subsets (100-200 records) for documentation and testing notebooks. Versioned for easy onboarding.

## Migration from original code

If you have results from the original `experimento.py`, they remain fully compatible. The JSON schema is backward-compatible:
- All original fields are preserved (`timestamp`, `model_name`, `prompt_group`, `prompt_id`, `prompt_text`, `output`, `std_dev`, `repeat_index`, `temperature`, `do_sample`, `param_group_name`, `seed`, `duration`, `tokens`, `degradation_method`, etc.)
- New fields are added without breaking existing analyses
- Original notebooks in `Archivos/` can still read old JSONs without modification

**Note:** Migration of existing notebooks will be addressed in a future phase. For now, they remain in `Archivos/` and work with existing results.

## Limitations and Future Work

### Current Limitations
- **Model-specific:** The parameter groups in `degradation.py` are hardcoded for Gemma-3-4b (34 layers). Using other models requires manual adaptation.
- **Notebooks not migrated:** The original analysis notebooks (in `Archivos/`) have not been migrated to work with the new code structure. They remain compatible with the original JSON outputs.
- **Single model support:** Only tested with Gemma 3 4B. Larger or smaller models may require adjustments to VRAM management and batch sizing.
- **Local execution only:** Designed for local GPU execution. Cloud/Colab support was intentionally removed.

### Future Extensions
- **Automatic layer detection:** Implement a function to automatically detect the number of layers in any model, removing the need for hardcoded parameter groups.
- **Example analysis notebook:** Create at least one demonstration notebook showing how to load and analyze the new JSON outputs.
- **Additional models:** Test and validate with other model families (LLaMA, Mistral, etc.).
- **Full baseline restoration:** Implement disk-based full model checkpointing as an alternative to in-memory subset restoration for very large models.
- **Degradation method extensions:** Explore additional perturbation methods beyond the current three (mult_gauss, ablation, uni_quant).
- **`force_run` flag:** Add optional flag to force re-execution of already-computed prompts (overwrite existing results). Currently, the system always resumes from existing results. Useful for debugging or regenerating outputs with minor model/config changes.

## Thesis
[Link to thesis PDF will be added here]

