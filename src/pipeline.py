"""
Pipeline orchestration for LLM degradation experiments.

This module coordinates the complete experiment flow:
- JSON persistence with resume capability
- Degradation level generation
- Main experiment loop (restore → degrade → generate → persist)
"""

import logging
import json
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Set, Any, NamedTuple

from configs.experiment_configs import ExperimentConfig
from src.utils import set_all_seeds, calculate_vram_percentage, adjust_batch_size_by_vram
from src.model_loader import restore_from_baseline
from src.degradation import apply_degradation
from src.generation import generate_text, evaluate_perplexity


class ExperimentState(NamedTuple):
    """Container for experiment state shared across pipeline phases."""
    model: Any
    tokenizer: Any
    baseline_subset: Dict
    processor: Any
    image: Any
    param_names: List[str]
    deg_levels: List[float]
    output_path: str
    results: List[Dict]
    computed_prompts_set: Set[Tuple]


def generate_degradation_levels(
    method: str,
    min_deg: float,
    max_deg: float,
    deg_steps: int
) -> List[float]:
    """
    Generate degradation levels for the experiment.
    
    Args:
        method: Degradation method ("mult_gauss", "ablation", "uni_quant")
        min_deg: Minimum degradation parameter
        max_deg: Maximum degradation parameter
        deg_steps: Number of degradation levels
    
    Returns:
        List of degradation values (ordered appropriately for the method)
    
    Note:
        - For quantization methods: geometric spacing, descending order (less degraded first)
        - For continuous methods: linear spacing, ascending order
    """
    logging.info("\n📉 Generating degradation levels:")
    
    if "quant" in method:
        # Geometric spacing for quantization
        levels = np.geomspace(min_deg, max_deg, deg_steps)
        levels = np.round(levels).astype(int)
        levels = np.unique(levels)
        levels = levels[levels >= 2][::-1]  # Descending order
        levels = [int(x) for x in levels]
        logging.info(f"[Quantization] n_vals [{method}]: {levels}")
    else:
        # Linear spacing for continuous methods
        levels = np.linspace(min_deg, max_deg, deg_steps)
        logging.info(f"Degradation levels: {levels.round(4).tolist()}")
    
    return list(levels)


def load_existing_results(output_path: str) -> Tuple[List[Dict], Set[Tuple]]:
    """
    Load existing results from JSON file if it exists.
    
    Args:
        output_path: Path to JSON file
    
    Returns:
        Tuple of (results list, set of computed prompts)
        - results: List of result dictionaries
        - computed_prompts: Set of tuples (param_group, std_dev, repeat_index, method, prompt_text)
    
    Note:
        If file doesn't exist, returns empty list and empty set.
    """
    if not os.path.exists(output_path):
        logging.info(f"No existing results found at {output_path}")
        return [], set()
    
    try:
        with open(output_path, "r") as f:
            results = json.load(f)
        
        # Build set of computed prompts for fast lookup
        computed_prompts_set = set()
        for r in results:
            key = (
                r.get("param_group_name"),
                round(float(r.get("std_dev", -1)), 2),
                r.get("repeat_index"),
                r.get("degradation_method"),
                r.get("prompt_text", "").strip()
            )
            computed_prompts_set.add(key)
        
        logging.info(f"Loaded {len(results)} existing results from {output_path}")
        return results, computed_prompts_set
    
    except Exception as e:
        logging.warning(f"Failed to load existing results: {e}")
        return [], set()


def is_prompt_computed(
    computed_prompts: Set[Tuple],
    param_group: str,
    degrad_level: float,
    repeat_index: int,
    method: str,
    prompt_text: str
) -> bool:
    """
    Check if a specific prompt has already been computed.
    
    Args:
        computed_prompts: Set of computed prompt keys
        param_group: Parameter group name
        degrad_level: Degradation level value
        repeat_index: Repetition index
        method: Degradation method
        prompt_text: Prompt text
    
    Returns:
        True if prompt was already computed, False otherwise
    """
    key = (param_group, round(float(degrad_level), 2), repeat_index, method, prompt_text.strip())
    return key in computed_prompts


def save_results(output_path: str, results: List[Dict], indent: int = None) -> None:
    """
    Save results to JSON file.
    
    Args:
        output_path: Path to JSON file
        results: List of result dictionaries
        indent: JSON indentation (None for compact, 2 for pretty)
    """
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=indent)
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise


def setup_experiment(config: ExperimentConfig) -> ExperimentState:
    """
    Setup phase: Load model, prepare degradation levels, initialize state.
    
    This function handles all initialization:
    - Logging experiment information
    - Loading model, tokenizer, and baseline
    - Loading image processor and image (if multimodal)
    - Generating degradation levels
    - Setting up output paths
    - Loading existing results (for resume capability)
    
    Args:
        config: Experiment configuration
    
    Returns:
        ExperimentState with all initialized components
    """
    from src.model_loader import load_model_and_tokenizer, load_image_processor
    from src.degradation import get_param_group
    from src.utils import load_image
    
    # Log experiment info
    logging.info("=" * 60)
    logging.info(f"Starting experiment: {config.config_name}")
    logging.info(f"Model: {config.model_name}")
    logging.info(f"Degradation: {config.degradation_method} on {config.param_group_name}")
    logging.info(f"Prompts: {len(config.prompts)}")
    logging.info(f"Repetitions: {config.n_rep}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
    
    # Set initial seeds
    set_all_seeds(config.seed)
    
    # Get parameter names for degradation
    param_names = get_param_group(config.param_group_name)
    
    # Load model, tokenizer, and baseline
    model, tokenizer, baseline_subset = load_model_and_tokenizer(
        model_name=config.model_name,
        param_names=param_names,
        device=config.device,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        max_seq_length=1024 if config.image_enabled else 512,
    )
    
    # Load image processor and image if needed
    processor = None
    image = None
    if config.image_enabled:
        processor = load_image_processor(config.model_name)
        image = load_image(config.image_filename)
    
    # Generate degradation levels
    deg_levels = generate_degradation_levels(
        config.degradation_method,
        config.min_deg,
        config.max_deg,
        config.deg_steps
    )
    
    # Setup output path
    model_name_clean = config.model_name.split("/")[-1].replace("/", "-")
    json_filename = f"outputs_{config.degradation_method}_{model_name_clean}_{config.name_suffix}.json"
    output_path = os.path.join("results", json_filename)
    os.makedirs("results", exist_ok=True)
    logging.info(f"Output file: {output_path}")
    
    # Load existing results (for resume capability)
    results, computed_prompts_set = load_existing_results(output_path)
    
    return ExperimentState(
        model=model,
        tokenizer=tokenizer,
        baseline_subset=baseline_subset,
        processor=processor,
        image=image,
        param_names=param_names,
        deg_levels=deg_levels,
        output_path=output_path,
        results=results,
        computed_prompts_set=computed_prompts_set
    )


def run_experiment_loop(
    state: ExperimentState,
    config: ExperimentConfig
) -> Tuple[List[Dict], float]:
    """
    Main experiment loop: Restore → Degrade → Generate → Save.
    
    This function executes the core experiment loop:
    - Iterate over degradation levels
    - Iterate over repetitions
    - Restore model from baseline
    - Apply degradation
    - Generate outputs for all prompts
    - Save results periodically
    
    Args:
        state: ExperimentState with model, tokenizer, baseline, etc.
        config: Experiment configuration
    
    Returns:
        Tuple of (results list, total_duration)
    """
    from src.utils import prepare_prompt_with_image
    
    # Unpack state
    model = state.model
    tokenizer = state.tokenizer
    baseline_subset = state.baseline_subset
    processor = state.processor
    image = state.image
    param_names = state.param_names
    deg_levels = state.deg_levels
    output_path = state.output_path
    results = state.results
    computed_prompts_set = state.computed_prompts_set
    
    # Unpack config
    model_name = config.model_name
    param_group_name = config.param_group_name
    degradation_method = config.degradation_method
    prompts = config.prompts
    n_rep = config.n_rep
    max_batch_size = config.max_batch_size
    seed_base = config.seed
    
    # Initialize loop state
    experiment_start_time = time.time()
    n_prompts_processed = 0
    batch_size = max_batch_size
    
    # Main experiment loop
    for level_idx, degrad_level in enumerate(deg_levels):
        level_start_time = time.time()
        
        for repeat_index in range(n_rep):
            repeat_start_time = time.time()
            
            # Set repetition-specific seed
            local_seed = seed_base + repeat_index
            set_all_seeds(local_seed)
            
            logging.info(
                f"\n\n=== [{degradation_method}] level={degrad_level:.4f} | {param_group_name} | "
                f"repeat={repeat_index} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
            )
            
            # Restore model from baseline
            restore_start = time.time()
            restore_from_baseline(model, baseline_subset)
            logging.info(f"[Model] Restored in {time.time() - restore_start:.3f}s")
            
            # Apply degradation if level > 0
            if degrad_level > 0.0:
                degrade_start = time.time()
                apply_degradation(model, param_names, degrad_level, method=degradation_method)
                logging.info(f"[Degradation] Applied in {time.time() - degrade_start:.3f}s")
            
            # Check if all prompts already computed for this repetition
            n_missing_prompts = sum(
                1 for p in prompts
                if not is_prompt_computed(
                    computed_prompts_set, param_group_name, degrad_level,
                    repeat_index, degradation_method, p
                )
            )
            
            if n_missing_prompts == 0:
                logging.info(
                    f"✅ All prompts already computed for level={degrad_level:.2f}, "
                    f"repeat={repeat_index}. Skipping."
                )
                continue
            elif n_missing_prompts < len(prompts):
                logging.info(
                    f"🔵 Resuming: {n_missing_prompts}/{len(prompts)} prompts remaining"
                )
            
            # Process prompts in batches
            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                
                # Prepare prompts (add image tag if needed)
                if config.image_enabled:
                    batch_prompts = [prepare_prompt_with_image(p) for p in batch_prompts]
                
                # Generate outputs
                batch_outputs, vram_pct = generate_text(
                    model, tokenizer, batch_prompts,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    processor=processor,
                    image=image,
                    model_variant=config.model_variant
                )
                
                # Ensure outputs is a list
                if not isinstance(batch_outputs, list):
                    batch_outputs = [batch_outputs]
                
                # Process each output
                for i, output in enumerate(batch_outputs):
                    prompt_idx = batch_start + i
                    prompt_text = prompts[prompt_idx]
                    prompt_start = time.time()
                    
                    # Skip if already computed
                    if is_prompt_computed(
                        computed_prompts_set, param_group_name, degrad_level,
                        repeat_index, degradation_method, prompt_text
                    ):
                        logging.info(
                            f"⏭️  Skipping prompt {prompt_idx + 1} (already computed)"
                        )
                        continue
                    
                    # Create result entry
                    n_tokens = len(tokenizer.encode(output))
                    tokens_in = len(tokenizer.encode(prompt_text))
                    result_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model_name,
                        "config_name": config.config_name,
                        "prompt_group": config.config_name.split("_")[0],
                        "prompt_id": prompt_idx + 1,
                        "prompt_text": prompt_text,
                        "output": output.strip(),
                        "std_dev": degrad_level,  # Legacy field for backward compatibility
                        "level_value": degrad_level,
                        "level_index": level_idx,
                        "repeat_index": repeat_index,
                        "param_group_name": param_group_name,
                        "degradation_method": degradation_method,
                        "seed": local_seed,
                        "gen_params": {
                            "temperature": config.temperature,
                            "do_sample": config.do_sample,
                            "max_new_tokens": config.max_new_tokens,
                        },
                        "duration": time.time() - prompt_start,
                        "tokens": n_tokens,
                        "tokens_out": n_tokens,
                        "tokens_in": tokens_in,
                        "model_variant": config.model_variant,
                        "device": config.device,
                        "dtype": config.dtype,
                        "load_4bit": config.load_in_4bit,
                        "restore_strategy": config.restore_strategy,
                        "batch_size_effective": batch_size,
                        "vram_usage_percent": vram_pct,
                        "image_used": config.image_enabled,
                    }
                    
                    if config.image_enabled:
                        result_entry["image_filename"] = config.image_filename
                    
                    results.append(result_entry)
                    
                    # Add to computed set
                    key = (param_group_name, round(degrad_level, 2), repeat_index, degradation_method, prompt_text.strip())
                    computed_prompts_set.add(key)
                    
                    n_prompts_processed += 1
                    
                    # Periodic save
                    if n_prompts_processed % 20 == 0:
                        save_start = time.time()
                        save_results(output_path, results, indent=None)
                        logging.info(
                            f"[Periodic save] {n_prompts_processed} prompts processed "
                            f"({time.time() - save_start:.3f}s)"
                        )
                    
                    # Log output
                    logging.info(
                        f"\nPrompt {prompt_idx + 1} (rep {repeat_index}) "
                        f"[{degradation_method}, {param_group_name}, level={degrad_level:.2f}]:\n"
                        f"{output.strip()[:200]}..." + ("-" * 60)
                    )
                    logging.info(f"⏱️  Time: {time.time() - prompt_start:.2f}s | Tokens: {n_tokens}")
                
                # Adjust batch size based on VRAM
                vram_current = calculate_vram_percentage()
                batch_size = adjust_batch_size_by_vram(
                    vram_current, batch_size, max_batch_size, len(prompts)
                )
            
            # Save after each repetition
            save_results(output_path, results, indent=None)
            logging.info(
                f"[Repetition complete] level={degrad_level:.2f}, repeat={repeat_index} "
                f"({time.time() - repeat_start_time:.2f}s)"
            )
        
        # Log level statistics
        level_duration = time.time() - level_start_time
        logging.info(f"\n[Level complete] level={degrad_level:.2f} ({level_duration:.2f}s)")
    
    total_duration = time.time() - experiment_start_time
    return results, total_duration


def finalize_experiment(
    output_path: str,
    results: List[Dict],
    total_duration: float
) -> None:
    """
    Finalization phase: Save final results and log statistics.
    
    This function handles experiment finalization:
    - Final save with pretty formatting
    - Log final statistics
    
    Args:
        output_path: Path to JSON output file
        results: List of result dictionaries
        total_duration: Total experiment duration in seconds
    """
    # Final save with pretty formatting
    save_results(output_path, results, indent=2)
    
    # Final statistics
    logging.info("\n" + "=" * 60)
    logging.info(f"✅ Experiment complete!")
    logging.info(f"Total results: {len(results)}")
    logging.info(f"Total duration: {total_duration:.2f}s")
    logging.info(f"Results saved: {output_path}")
    logging.info("=" * 60)


def run_experiment(config: ExperimentConfig) -> None:
    """
    Run complete degradation experiment.
    
    This is the main pipeline orchestrator that coordinates:
    - Setup: Model loading, degradation levels, output paths
    - Loop: Restore → Degrade → Generate → Save
    - Finalization: Final save and statistics
    
    Args:
        config: Experiment configuration (dataclass)
    
    Note:
        This function is designed to be resumable. If interrupted, it will
        automatically continue from where it left off on the next run.
    """
    # Phase 1: Setup
    state = setup_experiment(config)
    
    # Phase 2: Main experiment loop
    results, total_duration = run_experiment_loop(state, config)
    
    # Phase 3: Finalization
    finalize_experiment(state.output_path, results, total_duration)

