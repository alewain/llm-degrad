# Project Structure - Phase 1

Visual overview of the current repository structure after Phase 1 completion.

## Directory Tree

```
Repo_nuevo/
│
├── src/                                # Core Python modules
│   ├── __init__.py                    # Package marker (v1.0.0)
│   ├── utils.py                       # ✅ Logging & seeds (Section 1/3 complete)
│   ├── generation.py                  # ✅ Text generation & perplexity
│   ├── model_loader.py                # ✅ Model loading & restoration (Phase 2)
│   └── target_params.py               # ✅ Parameter groups for Gemma-3-4b (Phase 2)
│
├── configs/                            # Configuration files
│   ├── __init__.py                    # Package marker
│   ├── prompts.py                     # ✅ 3 prompt lists (IT only, 123 total)
│   └── experiment_configs.py          # ✅ Dataclass configs (3 experiments)
│
├── results/                            # Experiment outputs (gitignored)
│   └── samples/                       # Small samples (versioned)
│
├── logs/                               # Log files (gitignored)
│
├── notebooks/                          # Analysis notebooks (empty, Phase 5)
│
├── docs/                               # Documentation
│   └── output_schema.md               # JSON output schema (pre-existing)
│
├── .gitignore                          # ✅ Updated for new structure
├── requirements.txt                    # ✅ Python dependencies
├── README.md                           # Project documentation (pre-existing)
├── PLAN.md                             # Migration plan (pre-existing)
└── MIGRATION_STATUS.md                 # ✅ Phase tracking document
```

## Files Created in Phase 1

### Core Modules (`src/`)

**`src/utils.py`** (186 lines)
- ✅ Section 1: General utilities (COMPLETE)
  - `setup_logging(log_filename, level)` - Dual logging with UTF-8
  - `set_all_seeds(seed)` - Reproducibility across all RNGs
- ⏳ Section 2: VRAM monitoring (Phase 4)
- ⏳ Section 3: Image support (Phase 4)

**`src/generation.py`** (155 lines)
- ✅ `wrap_chat_it(user_prompt)` - Gemma IT chat format wrapper
- ✅ `generate_text(model, tokenizer, prompt, ...)` - Unified generation
  - Supports single/batch prompts
  - Optional image input (multimodal)
  - Basic VRAM monitoring
- ✅ `evaluate_perplexity(model, tokenizer, text)` - Optional perplexity calc

**`src/__init__.py`** (7 lines)
- Package marker with version

**`src/model_loader.py`** (298 lines) - **Phase 2**
- ✅ `load_model_and_tokenizer()` - Main entry point for model loading
- ✅ `load_tokenizer()` - Load from HF cache or download
- ✅ `load_model()` - Unsloth FastLanguageModel integration
- ✅ `create_baseline_subset()` - Save params to CPU memory
- ✅ `restore_from_baseline()` - Fast restoration (~1-3s)
- ✅ `load_image_processor()` - For multimodal experiments
- ✅ `get_model_memory_footprint()` - Memory statistics

**`src/target_params.py`** (206 lines) - **Phase 2**
- ✅ Parameter group functions: `get_attn_params()`, `get_mlp_params()`, etc.
- ✅ `PARAM_GROUPS` dict with 6 groups (attn_only, mlp_only, embed_only, etc.)
- ✅ `get_param_group(name)` - Config-based parameter selection
- ✅ `strip_module_prefix()` - Handle DataParallel naming
- ✅ `validate_param_group()` - Runtime validation against loaded model
- ✅ Hardcoded for Gemma-3-4b (34 layers) with clear documentation

### Configuration Files (`configs/`)

**`configs/prompts.py`** (172 lines)
- ✅ `dream_prompts_it` - 38 prompts (dream narration)
- ✅ `iq_prompts_it` - 65 prompts (math, language, logic, factual, creativity)
  - `math_prompts` - 25 prompts
  - `language_prompts` - 15 prompts
  - `logic_prompts` - 3 prompts
  - `factual_prompts` - 5 prompts
  - `creativity_prompts` - 2 prompts
- ✅ `cookie_theft_prompts_it` - 20 prompts (image description)
- **Total: 123 prompts (all IT variant)**

**`configs/experiment_configs.py`** (135 lines)
- ✅ `ExperimentConfig` dataclass - Base config with all fields
- ✅ `dreams_it` - Dream experiment config
- ✅ `iq_it` - IQ assessment config
- ✅ `cookie_theft_it` - Cookie Theft config (image_enabled=True)
- ✅ `get_config(name)` - Helper function to retrieve configs

**`configs/__init__.py`** (6 lines)
- Package marker

### Supporting Files

**`requirements.txt`** (23 lines)
- Core dependencies: torch, transformers, numpy, pillow, etc.
- Unsloth for fast model loading
- Instructions for PyTorch CUDA 12.1 installation

**`.gitignore`** (35 lines)
- Excludes full experiment results (results/*.json)
- Includes sample outputs (results/samples/*.json)
- Excludes logs (logs/, *.log)
- Standard Python artifacts

**`MIGRATION_STATUS.md`** (140 lines)
- Phase-by-phase checklist
- Current state summary
- Next steps

## Files NOT Created Yet (Phases 3-4)

```
src/
├── degradation.py           # Phase 3: Degradation methods (mult_gauss, ablation, uni_quant)
├── pipeline.py              # Phase 4: Orchestration & persistence
└── run_experiment.py        # Phase 4: CLI entry point
```

```
src/utils.py (sections 2-3)  # Phase 4: VRAM monitoring & image utilities
```

## Statistics

### Lines of Code (Phases 1-2)
```
Phase 1:
src/utils.py              : 186 lines (Section 1/3 complete)
src/generation.py         : 155 lines (complete)
configs/prompts.py        : 172 lines (complete)
configs/experiment_configs.py : 135 lines (complete)

Phase 2:
src/model_loader.py       : 298 lines (complete)
src/target_params.py      : 206 lines (complete)
────────────────────────────────────────
Total new code            : 1,152 lines
Supporting files          : ~200 lines
────────────────────────────────────────
Grand total               : ~1,350 lines
```

### Prompts Count
```
Dreams IT                 : 38 prompts
IQ IT                     : 65 prompts
Cookie Theft IT           : 20 prompts
────────────────────────────────────────
Total                     : 123 prompts
```

### Experiments Ready
```
✅ dreams_it              : Configured, ready for implementation
✅ iq_it                  : Configured, ready for implementation  
✅ cookie_theft_it        : Configured, ready for implementation
```

## Key Design Decisions

1. **Dataclasses over YAML/JSON**
   - Type safety with automatic validation
   - IDE autocomplete and refactoring support
   - Python expressiveness (computations, conditionals)

2. **Python lists for prompts**
   - No file I/O or parsing needed
   - Direct imports in configs
   - IDE support for editing

3. **Modular logging**
   - Dual output (console + file)
   - UTF-8 encoding for emojis
   - Centralized via `setup_logging()`

4. **Reproducibility-first**
   - Single function `set_all_seeds()` for all RNGs
   - Seed derivation: `base_seed + repeat_index`

## What Can Be Tested Now

While the full pipeline isn't ready, you can test individual components:

```python
# Test configuration loading
from configs.experiment_configs import get_config
config = get_config("dreams_it")
print(f"Config: {config.config_name}")
print(f"Prompts: {len(config.prompts)}")

# Test logging setup
from src.utils import setup_logging, set_all_seeds
setup_logging("logs/test.log")
set_all_seeds(42)

# Test generation wrapper (once model is loaded in Phase 2)
from src.generation import wrap_chat_it
prompt = wrap_chat_it("Describe a dream.")
print(prompt)
```

## Next: Phase 3

Implement degradation methods:
- `src/degradation.py` - mult_gauss, ablation, uni_quant methods
- Extract quantization and perturbation logic from original experimento.py

---

Last updated: 2025-10-07 (After Phase 2 completion)

