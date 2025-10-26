# LLM Degradation Experiments

Experimental code for the thesis **["Controlled Degradation in a Large Language Model (LLM)"](docs/thesis_spanish.pdf)** (Spanish) by Alejandro Wainstock.

This repository provides a local, reproducible pipeline for applying controlled degradations to LLM weights (parameters) and generating structured outputs for analysis.

**Current version:** Configured for Gemma-3-4b-it. Easily adaptable to other models, prompts, or target parameters. This repository includes content generation with perturbed models; analysis tools coming soon.

---

## Quick Start

### Installation

First, verify the [requirements](docs/guide.md#requirements). Use a virtual environment (optional, recommended).
  
  ```bash
# Install PyTorch with CUDA 12.1 (tested and verified versions)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies (tested version combinations)
pip install -r requirements.txt

# Set up HuggingFace token
cp env.example .env
# Edit .env and add: HF_TOKEN=your_token_here
# Get your token from: https://huggingface.co/settings/tokens
```

---

## Available Experiments

This project defines experiments by combining three elements: a **task** (which prompts to use), a **degradation method** (how to perturb the weights), and a **target parameter group** (which model components or matrices to degrade).

### Tasks (Experiment Themes)

<sub>_Defined in [`configs/prompts.py`](configs/prompts.py)_</sub>

| Task | Description | Prompts |
|------|-------------|---------|
| `dreams_it` | Narrate a dream in detail | ~38 |
| `math_it` | Solve math tasks | ~24 |
| `lang_it` | Language tasks | ~16 |
| `cookie_theft_it` | Describe a standardized picture (Cookie Theft task) | ~20 |

### Degradation Methods

<sub>_Defined in [`src/degradation.py`](src/degradation.py)_</sub>

- **`mult_gauss`**: Multiplicative Gaussian noise (controlled by standard deviation)
- **`ablation`**: Set parameters to zero (controlled by masking proportion, 0–1)
- **`uni_quant`**: Uniform quantization (controlled by number of quantization levels)

### Target Parameter Groups

<sub>_Defined in [`src/degradation.py`](src/degradation.py)_</sub>

- **`attn_only`**: Attention V-projection matrices (`v_proj.weight`)
- **`mlp_only`**: Feed-forward network matrices (gate, up, down)
- **`embed_only`**: Token embedding matrix (`embed_tokens.weight`, lookup table)

**Note:** In this version, degradations are applied to the weight values of these matrices across all layers simultaneously.

### Variants (Method + Target Combinations)

<sub>_Defined in [`configs/experiment_configs.py`](configs/experiment_configs.py)_</sub>

Each experiment variant is a specific combination of degradation method and target parameter group. The following 5 variants were used in this thesis:

| Index | Name | Method | Target | Range | Steps |
|-------|------|--------|--------|-------|-------|
| 1 | `gauss_attn` | Gaussian | Attention (V) | 0.0 → 1.4 (σ) | 15 (linear) |
| 2 | `gauss_mlp` | Gaussian | MLP | 0.0 → 0.5 (σ) | 11 (linear) |
| 3 | `gauss_embed` | Gaussian | Embeddings | 0.0 → 1.0 (σ) | 21 (linear) |
| 4 | `ablation_attn` | Ablation | Attention (V) | 0.0 → 0.8 (%) | 17 (linear) |
| 5 | `quant_attn` | Quantization | Attention (V) | 1024 → 4 (levels) | 9 (geometric) |


### Running Experiments

```bash
# Single variant
python -m src.main --task dreams_it --variants gauss_attn

# Multiple variants by name - QUICK TEST
python -m src.main --task math_it --variants gauss_attn,gauss_mlp --n-rep 2 --deg-steps 3

# All variants by index
python -m src.main --task cookie_theft_it --variant-indexes 1-5

# Get help
python -m src.main --help
```

---

## Project Structure

```
├── src/                        # Core modules
│   ├── main.py                 # CLI entry point
│   ├── pipeline.py             # Experiment orchestration
│   ├── model_loader.py         # Model loading & restoration
│   ├── degradation.py          # Degradation methods & target parameters
│   ├── generation.py           # Text generation
│   └── utils.py                # Utilities (logging, VRAM, seeds)
│
├── configs/                    # Configuration
│   ├── experiment_configs.py   # TASKS × VARIANTS registry
│   └── prompts.py              # Prompt lists by task
│
├── results/                    # JSON outputs (created at runtime)
├── logs/                       # Execution logs (created at runtime)
│
├── docs/                       # Documentation
│   ├── guide.md                # Complete guide
│   ├── output_schema.md        # JSON schema reference
│   └── thesis_spanish.pdf      # Thesis document (Spanish)
│
├── requirements.txt            # Dependencies
└── env.example                 # Example .env file
```

---

## Output

Results are saved as JSON files in `results/`:

```
results/outputs_{method}_{model}_{task}.json
```

Example: `outputs_mult_gauss_gemma-3-4b-it_dreams_it.json`

Each file contains an array of JSON objects, one per generation. See [output_schema.md](docs/output_schema.md) for complete field documentation.

---

For complete documentation, see the **[Guide](docs/guide.md)**.

