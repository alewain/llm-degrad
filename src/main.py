"""
CLI entry point for running LLM degradation experiments.

Usage:
    python src/main.py --config dreams_it
    python src/main.py --config iq_it
    python src/main.py --config cookie_theft_it
"""

import argparse
import sys
import os
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

from configs.experiment_configs import get_config
from src.utils import setup_logging
from src.pipeline import run_experiment


def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run LLM degradation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --config dreams_it
  python src/main.py --config iq_it
  python src/main.py --config cookie_theft_it

Available configs:
  - dreams_it: Dream narration task (~38 prompts)
  - iq_it: Multi-task cognitive assessment (~65 prompts)
  - cookie_theft_it: Cookie Theft image description (~20 prompts)
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["dreams_it", "iq_it", "cookie_theft_it"],
        help="Name of the experiment configuration to run"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = config.model_name.split("/")[-1].replace("/", "-")
    log_filename = os.path.join(
        "logs",
        f"{config.degradation_method}_{model_name_clean}_{timestamp}.log"
    )
    
    log_level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }
    setup_logging(log_filename, level=log_level_map[args.log_level])
    
    # Run experiment
    try:
        run_experiment(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user (Ctrl+C)")
        print("Results have been saved periodically. You can resume by running the same command again.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

