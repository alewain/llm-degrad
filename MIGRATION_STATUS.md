# Migration Status

This document tracks the progress of migrating from `experimento.py` to the modularized codebase.

## ✅ Phase 1: Skeleton and Configuration (COMPLETED)

**Goal:** Create base structure, utilities, prompts, and configurations.

### Completed Tasks:
- ✅ Created directory structure:
  - `src/` - Core source code modules
  - `configs/` - Configuration files and prompts
  - `results/` - Experiment outputs (with `samples/` subdirectory)
  - `logs/` - Log files
  - `notebooks/` - Analysis notebooks (empty for now)
- ✅ Implemented `src/utils.py` (Section 1 only):
  - `setup_logging()` - Dual logging (console + file) with UTF-8 support
  - `set_all_seeds()` - Reproducibility via unified seed management
- ✅ Created `configs/prompts.py`:
  - `dream_prompts_it` (38 prompts) - Dream narration task
  - `iq_prompts_it` (65 prompts) - Multi-task cognitive assessment
  - `cookie_theft_prompts_it` (20 prompts) - Cookie Theft image description
  - Only IT (instruction-tuned) variants included
- ✅ Created `configs/experiment_configs.py`:
  - Base `ExperimentConfig` dataclass with type hints
  - Three experiment configs: `dreams_it`, `iq_it`, `cookie_theft_it`
  - Helper function `get_config(name)` for retrieving configs
- ✅ Created `src/generation.py`:
  - `wrap_chat_it()` - IT chat format wrapper
  - `generate_text()` - Unified generation with optional image support
  - `evaluate_perplexity()` - Optional perplexity calculation
- ✅ Updated `.gitignore` for new structure
- ✅ Created package `__init__.py` files

### Notes:
- All prompts extracted from `experimento.py` lines 5-197
- Pretrained (PT) variants intentionally excluded
- `src/utils.py` sections 2 and 3 (VRAM and image) deferred to Phase 4
- Logging replaces `print()` statements from original code
- No breaking changes to original `experimento.py` (still functional)

---

## ✅ Phase 2: Model Loading and Restoration (COMPLETED)

**Goal:** Implement model loading and `subset_in_memory` restoration strategy.

### Completed Tasks:
- ✅ Created `src/model_loader.py`:
  - `load_model_and_tokenizer()` - Main entry point for model loading
  - `load_tokenizer()` - Load from cache or download from HuggingFace
  - `load_model()` - Use Unsloth FastLanguageModel for optimized loading
  - `create_baseline_subset()` - Save degradable params in CPU memory
  - `restore_from_baseline()` - Fast restoration from in-memory baseline
  - `load_image_processor()` - For multimodal experiments
  - `get_model_memory_footprint()` - Memory statistics utility
- ✅ Created `src/target_params.py`:
  - Parameter group functions: `get_attn_params()`, `get_mlp_params()`, `get_embedding_params()`
  - `PARAM_GROUPS` dictionary with 6 groups (attn_only, mlp_only, embed_only, etc.)
  - `get_param_group()` helper for config-based selection
  - `strip_module_prefix()` to handle DataParallel naming
  - `validate_param_group()` for runtime validation
  - Hardcoded for Gemma-3-4b (34 layers) with clear documentation

### Notes:
- Unsloth integration for fast model loading with optimizations
- Baseline subset stored in CPU memory (~1-3 second restoration)
- Support for 4-bit quantization via `load_in_4bit` flag
- Image processor loading for Cookie Theft experiment
- Parameter validation to catch mismatches early

---

## 🔄 Phase 3: Degradation and Generation (TODO)

**Goal:** Implement degradation methods (mult_gauss, ablation, uni_quant).

### Pending Tasks:
- ⬜ Create `src/degradation.py`:
  - `apply_mult_gauss()` - Multiplicative Gaussian noise
  - `apply_ablation()` - Random weight masking
  - `apply_uni_quant()` - Uniform quantization (geometric spacing)
  - Helper: `quantize_tensor_uniform()`
  - Remove deprecated methods: `uni_quant_lineal`, `lognorm`

---

## 🔄 Phase 4: Pipeline and Full Utils (TODO)

**Goal:** Orchestrate complete experiment flow and implement remaining utilities.

### Pending Tasks:
- ⬜ Create `src/pipeline.py`:
  - `run_experiment()` - Main orchestrator
  - `load_results_json()` - Load existing results for resume
  - `save_results_json()` - Periodic and final saves
  - `check_prompt_computed()` - Skip already-processed prompts
  - `generate_degradation_levels()` - Compute level values
- ⬜ Complete `src/utils.py` Section 2 (VRAM):
  - `calculate_vram_percentage()` - Monitor VRAM usage
  - `adjust_batch_size_by_vram()` - Dynamic batch adjustment
  - `dry_run_memory_estimation()` - Pre-run memory check
- ⬜ Complete `src/utils.py` Section 3 (Image):
  - `load_image()` - Load Cookie Theft image
  - `prepare_prompt_with_image()` - Multimodal prompt preparation
- ⬜ Create `src/run_experiment.py`:
  - CLI entry point with argparse
  - Accept `--config` argument (e.g., `--config dreams_it`)
  - Initialize logging, load config, run experiment

---

## 🔄 Phase 5: Notebooks and Samples (FUTURE)

**Goal:** Create example analysis notebooks and sample outputs.

### Pending Tasks:
- ⬜ Generate sample outputs (100-200 records per method)
- ⬜ Create example notebook showing how to load and analyze results
- ⬜ Document notebook dependencies

---

## Current State Summary

**What works:**
- Configuration system (dataclasses with type safety)
- Prompts organization (3 experiments, IT only)
- Basic utilities (logging, seeds)
- Generation functions (text + optional image, perplexity)

**What's missing:**
- Model loading/restoration logic
- Degradation methods
- Pipeline orchestration
- VRAM monitoring and image utilities
- CLI entry point

**Next immediate step:**
Start Phase 2 by implementing `src/model_loader.py` and `src/target_params.py`.

---

## Testing

Once Phase 4 is complete, test with:
```bash
python src/run_experiment.py --config dreams_it
```

Expected behavior: Load model, apply degradations, generate outputs, save to `results/`.

---

Last updated: 2025-10-07 (Phase 1 completed)

