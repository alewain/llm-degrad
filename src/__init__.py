"""
Source code for LLM degradation experiments.

Core modules:
- model_loader: Model and tokenizer loading with baseline restoration
- target_params: Parameter group definitions for Gemma-3-4b
- degradation: Degradation methods (mult_gauss, ablation, uni_quant)
- generation: Text generation with optional image support
- pipeline: Experiment orchestration and JSON persistence
- utils: Logging, seeds, VRAM monitoring, image support
"""

__version__ = "1.0.0"
