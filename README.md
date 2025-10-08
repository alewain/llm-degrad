# LLM Degradation Experiments

Experimental code for the thesis **["Controlled Degradation in a Large Language Model (LLM)"](link-to-thesis)** by Alejandro Wainstock.

This repository provides a local, reproducible pipeline for applying controlled degradations to LLM weights and generating structured outputs for analysis.

**Current version:** Configured for Gemma-3-4b-it. Easily adaptable to other models, prompts, or target parameters.

---

## Quick Start

### Installation

First, verify the [requirements](docs/guide.md#requirements). Use a virtual environment (optional, recommended).

```bash
# Ensure PyTorch with CUDA is installed in your environment
# If you already have a correct torch+CUDA install, skip.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Set up HuggingFace token
cp env.example .env
# Edit .env and add: HF_TOKEN=your_token_here
# Get your token from: https://huggingface.co/settings/tokens
```

---

## Available Experiments

### Tasks (Experiment Themes)

| Task | Description | Prompts |
|------|-------------|---------|
| `dreams_it` | Dream narration | ~38 |
| `iq_it` | Cognitive assessment (math, language, logic) | ~65 |
| `cookie_theft_it` | Image description | ~20 |

### Degradation Methods

- **`mult_gauss`**: Multiplicative Gaussian noise
- **`ablation`**: Set parameters to zero
- **`uni_quant`**: Uniform quantization

### Parameter Groups (Target Parameters)

- **`attn_only`**: Attention mechanism (V projections only)
- **`mlp_only`**: Feed-forward network
- **`embed_only`**: Token embeddings

### Variants (Method + Target Combinations)

Each variant is a specific combination of degradation method and parameter group:

| Index | Name | Method | Target | Range (min → max) | Steps |
|-------|------|--------|--------|-------------------|-------|
| 1 | `gauss_attn` | Gaussian noise | Attention (V) | 0.0 → 1.4 | 15 |
| 2 | `gauss_mlp` | Gaussian noise | MLP | 0.0 → 0.5 | 11 |
| 3 | `gauss_embed` | Gaussian noise | Embeddings | 0.0 → 1.0 | 21 |
| 4 | `ablation_attn` | Ablation | Attention (V) | 0.0 → 0.8 | 17 |
| 5 | `quant_attn` | Quantization | Attention (V) | 4 → 1024 | 9 |

**Note:** Variants and their parameters are defined in `configs/experiment_configs.py`.

### Running Experiments

```bash
# Single variant
python -m src.main --task dreams_it --variants gauss_attn

# Multiple variants by name
python -m src.main --task iq_it --variants gauss_attn,gauss_mlp

# All variants by index
python -m src.main --task cookie_theft_it --variant-indexes 1-5

# Get help
python -m src.main --help
```

---

## Project Structure

```
├── src/                 # Core modules
│   ├── main.py          # CLI entry point
│   ├── pipeline.py      # Experiment orchestration
│   ├── model_loader.py  # Model loading & restoration
│   ├── degradation.py   # Degradation methods
│   ├── generation.py    # Text generation
│   └── utils.py         # Utilities (logging, VRAM, seeds)
│
├── configs/             # Configuration
│   ├── experiment_configs.py  # TASKS × VARIANTS registry
│   └── prompts.py       # Prompt lists
│
├── results/             # JSON outputs
│   └── samples/         # Small versioned samples
│
├── logs/                # Execution logs
│
├── docs/                # Documentation
│   ├── guide.md         # Complete guide
│   └── output_schema.md # JSON schema reference
│
└── requirements.txt     # Dependencies
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

## Research Context

This project investigates:
- How cognitive capabilities deteriorate under controlled neural damage
- Whether LLM degradation patterns mirror human neurological disorders
- Which architectural components are critical for different tasks

**Note:** This repository focuses on generation and perturbation. Analysis and metrics are documented in the thesis.

---

## License & Citation

[Add license information]

If you use this code, please cite:
```
[Add citation information]
```

---

## Documentation

- **[Complete Guide](docs/guide.md)** - Installation, usage, extending the system, examples
- **[Output Schema](docs/output_schema.md)** - JSON structure reference

---

## Contact

Alejandro Wainstock - [Contact information]
