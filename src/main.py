"""
CLI entry point for running LLM degradation experiments.

This module provides a command-line interface for running experiments with:
- TASK selection: Choose experiment theme (prompts, image support)
- VARIANT selection: Choose degradation method + parameter group + range

Usage:
    # Run single variant by name
    python -m src.main --task dreams_it --variants gauss_attn
    
    # Run multiple variants by name
    python -m src.main --task iq_it --variants gauss_attn,quant_attn
    
    # Run variants by index (1-5)
    python -m src.main --task cookie_theft_it --variant-indexes 1-3
    python -m src.main --task dreams_it --variant-indexes 2,4,5
"""

import argparse
import sys
import os
from datetime import datetime
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

from configs.experiment_configs import (
    build_config, 
    TASKS, 
    VARIANTS, 
    VARIANTS_ORDERED,
    get_variant_by_index
)
from src.utils import setup_logging
from src.pipeline import run_experiment


def parse_variant_indexes(indexes_str: str) -> List[int]:
    """
    Parse variant index string into list of integers.
    
    Supports:
    - Single index: "3" → [3]
    - Comma-separated: "1,3,5" → [1, 3, 5]
    - Range: "1-3" → [1, 2, 3]
    - Mixed: "1-3,5" → [1, 2, 3, 5]
    
    Args:
        indexes_str: String specifying variant indexes
    
    Returns:
        Sorted list of unique variant indexes
    """
    result = []
    for part in indexes_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            # Normalize reversed ranges (e.g., "3-1" -> "1-3")
            start, end = (start, end) if start <= end else (end, start)
            result.extend(range(start, end + 1))  # inclusive
        else:
            result.append(int(part))
    return sorted(set(result))  # remove duplicates, sort


def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run LLM degradation experiments with flexible task and variant selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single variant by name
  python -m src.main --task dreams_it --variants gauss_attn
  
  # Run multiple variants by name
  python -m src.main --task iq_it --variants gauss_attn,gauss_mlp,quant_attn
  
  # Run variants by index (1-5)
  python -m src.main --task dreams_it --variant-indexes 1-5
  python -m src.main --task cookie_theft_it --variant-indexes 2,4

Available tasks:
  - dreams_it: Dream narration (~38 prompts, no image)
  - iq_it: Cognitive assessment (~65 prompts, no image)
  - cookie_theft_it: Image description (~20 prompts, with image)

Available variants (indexed 1-5):
  1. gauss_attn:    Gaussian noise on attention (min=0.0, max=1.4, steps=15)
  2. gauss_mlp:     Gaussian noise on MLP (min=0.0, max=0.5, steps=11)
  3. gauss_embed:   Gaussian noise on embeddings (min=0.0, max=1.0, steps=21)
  4. ablation_attn: Ablation on attention (min=0.0, max=0.8, steps=17)
  5. quant_attn:    Quantization on attention (min=4, max=1024, steps=9)
        """
    )
    
    # Task selection (required)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASKS.keys()),
        help="Experiment task (REQUIRED): dreams_it, iq_it, or cookie_theft_it"
    )
    
    # Variant selection (mutually exclusive, one required)
    variant_group = parser.add_mutually_exclusive_group(required=True)
    variant_group.add_argument(
        "--variants",
        type=str,
        help="Variant names (comma-separated). Example: gauss_attn,quant_attn"
    )
    variant_group.add_argument(
        "--variant-indexes",
        type=str,
        help="Variant indexes (1-5, supports ranges). Examples: 1-3, 2,4,5, 1-5"
    )
    
    # Optional arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Parse variant selection
    variant_keys = []
    try:
        if args.variants:
            # Parse variant names
            variant_keys = [v.strip() for v in args.variants.split(',')]
            # Validate all variants exist
            for v in variant_keys:
                if v not in VARIANTS:
                    print(f"❌ Error: Unknown variant '{v}'")
                    print(f"Available variants: {', '.join(VARIANTS.keys())}")
                    sys.exit(1)
        elif args.variant_indexes:
            # Parse variant indexes
            indexes = parse_variant_indexes(args.variant_indexes)
            # Convert indexes to variant keys
            for idx in indexes:
                try:
                    variant_keys.append(get_variant_by_index(idx))
                except ValueError as e:
                    print(f"❌ Error: {e}")
                    sys.exit(1)
    except Exception as e:
        print(f"❌ Error parsing variants: {e}")
        sys.exit(1)
    
    # Build configurations for all task+variant combinations
    configs = []
    for variant_key in variant_keys:
        try:
            cfg = build_config(args.task, variant_key)
            configs.append(cfg)
        except ValueError as e:
            print(f"❌ Error building config for {args.task} + {variant_key}: {e}")
            sys.exit(1)
    
    # Setup logging (use first config for log filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        "logs",
        f"{args.task}_{timestamp}.log"
    )
    
    log_level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }
    setup_logging(log_filename, level=log_level_map[args.log_level])
    
    # Print experiment plan
    print(f"\n{'='*80}")
    print(f"🚀 Starting experiment batch: {args.task}")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Variants: {', '.join(variant_keys)} ({len(configs)} total)")
    print(f"Log file: {log_filename}")
    print(f"{'='*80}\n")
    
    # Run experiments sequentially
    for i, config in enumerate(configs, 1):
        variant_name = variant_keys[i-1]
        print(f"\n{'='*80}")
        print(f"📊 Experiment {i}/{len(configs)}: {config.config_name}")
        print(f"{'='*80}")
        print(f"Method: {config.degradation_method}")
        print(f"Param group: {config.param_group_name}")
        print(f"Range: min={config.min_deg}, max={config.max_deg}, steps={config.deg_steps}")
        print(f"Prompts: {len(config.prompts)}")
        print(f"Repetitions: {config.n_rep}")
        print(f"{'='*80}\n")
        
        try:
            run_experiment(config)
        except KeyboardInterrupt:
            print("\n\n⚠️  Experiment interrupted by user (Ctrl+C)")
            print("Results have been saved periodically. You can resume by running the same command again.")
            sys.exit(130)
        except Exception as e:
            print(f"\n\n❌ Experiment {i} failed with error:")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n⚠️  Continuing with remaining experiments...")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ All experiments completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

