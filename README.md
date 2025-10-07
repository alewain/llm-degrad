# LLM Degradation Experiments

Experimental code for the thesis **"Controlled Degradation in a Large Language Model (LLM)"** by Alejandro Wainstock, supervised by Dr. Enzo Tagliazucchi.

This repository provides a local, reproducible pipeline for applying controlled degradations to LLM weights and generating structured outputs for analysis.

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA 12.1+
- NVIDIA GPU with 24GB+ VRAM
- ~15GB disk space

### Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up HuggingFace token (create .env file)
cp env.example .env
# Edit .env and add: HF_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### Run an Experiment

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

## Available Experiments

### Tasks (Experiment Themes)

| Task | Description | Prompts |
|------|-------------|---------|
| `dreams_it` | Dream narration | ~38 |
| `iq_it` | Cognitive assessment (math, language, logic) | ~65 |
| `cookie_theft_it` | Image description | ~20 |

### Variants (Degradation Methods)

| Index | Name | Method | Target | Range |
|-------|------|--------|--------|-------|
| 1 | `gauss_attn` | Gaussian noise | Attention | 0.0 → 1.4 |
| 2 | `gauss_mlp` | Gaussian noise | MLP | 0.0 → 0.5 |
| 3 | `gauss_embed` | Gaussian noise | Embeddings | 0.0 → 1.0 |
| 4 | `ablation_attn` | Ablation | Attention | 0.0 → 0.8 |
| 5 | `quant_attn` | Quantization | Attention | 4 → 1024 |

**Degradation Methods:**
- `mult_gauss`: Multiplicative Gaussian noise
- `ablation`: Set parameters to zero
- `uni_quant`: Uniform quantization

**Parameter Groups:**
- `attn_only`: Attention mechanism (Q, K, V)
- `mlp_only`: Feed-forward network
- `embed_only`: Token embeddings

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

## Documentation

- **[Complete Guide](docs/guide.md)** - Installation, usage, extending the system
- **[Output Schema](docs/output_schema.md)** - JSON structure reference

---

## Key Features

✅ **Registry-based configuration** - Compose TASK × VARIANT at runtime  
✅ **Resume capability** - Interrupt-resilient with incremental saves  
✅ **VRAM monitoring** - Dynamic batch size adjustment  
✅ **Fast restoration** - In-memory baseline for quick parameter reset  
✅ **Reproducible** - Centralized seed management  
✅ **Multimodal support** - Optional image processing (Cookie Theft)

---

## Example Workflows

### Run all variants on all tasks

```bash
for task in dreams_it iq_it cookie_theft_it; do
    python -m src.main --task $task --variant-indexes 1-5
done
```

### Programmatic usage

```python
from configs.experiment_configs import build_config
from src.pipeline import run_experiment

# Build configuration
cfg = build_config("dreams_it", "gauss_attn", n_rep=5)

# Run experiment
run_experiment(cfg)
```

---

## Output

Results are saved as JSON files in `results/`:

```
results/outputs_{method}_{model}_{task}.json
```

Example entry:
```json
{
  "config_name": "dreams_it__gauss_attn",
  "degradation_method": "mult_gauss",
  "param_group_name": "attn_only",
  "degrad_level": 0.7,
  "prompt_text": "Describe a recurring dream...",
  "output": "In my recurring dream...",
  "duration": 2.34,
  "tokens": 312
}
```

See [output_schema.md](docs/output_schema.md) for complete field documentation.

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

## Contact

Alejandro Wainstock - [Contact information]  
Supervisor: Dr. Enzo Tagliazucchi

---

*For detailed documentation, see [docs/guide.md](docs/guide.md)*
