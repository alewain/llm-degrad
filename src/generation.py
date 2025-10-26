"""
Text generation functions for LLM degradation experiments.

This module provides:
- Text generation with optional image support (multimodal)
- Instruction-tuned (IT) chat wrapper for Gemma models
- Optional perplexity evaluation (disabled by default)
"""

import logging
import torch
import numpy as np
from typing import List, Union, Tuple, Optional, Any


def wrap_chat_it(user_prompt: str) -> str:
    """
    Wrap user prompt in Gemma instruction-tuned (IT) chat format.
    
    This function manually constructs the chat template for Gemma IT models,
    replacing the need for tokenizer.apply_chat_template(..., add_generation_prompt=True).
    The format includes <start_of_turn>model at the end to signal generation start.
    
    Important:
    - Do not combine this manual wrapper with tokenizer.apply_chat_template to avoid duplicated
      special tokens/markers. Use one approach or the other, not both.
    - This requires the tokenizer to recognize the special tokens used (<bos>, <start_of_turn>,
      <end_of_turn>).
    
    Args:
        user_prompt: The raw user prompt text
    
    Returns:
        Formatted prompt ready for generation with IT model
    
    Example:
        >>> wrap_chat_it("Describe a dream.")
        '<bos><start_of_turn>user\\nDescribe a dream.<end_of_turn>\\n<start_of_turn>model\\n'
    """
    return f"<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, List[str]],
    max_new_tokens: int = 350,
    do_sample: bool = True,
    temperature: float = 1.0,
    processor: Optional[Any] = None,
    image: Optional[Any] = None,
    model_variant: str = "it",
) -> Tuple[Union[str, List[str]], float]:
    """
    Generate text from model given prompt(s), with optional image input.
    
    This function handles both single prompts and batches of prompts.
    For instruction-tuned models, it automatically wraps prompts in chat format.
    
    Args:
        model: The language model (PyTorch model or DataParallel-wrapped)
        tokenizer: Tokenizer for the model
        prompt: Single prompt string or list of prompts
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        temperature: Sampling temperature (only used if do_sample=True)
        processor: Optional processor for multimodal inputs (required if image provided)
        image: Optional PIL Image for multimodal generation
        model_variant: Model variant ("it" for instruction-tuned, "pt" for pretrained)
    
    Returns:
        Tuple of (decoded_outputs, vram_percentage):
        - decoded_outputs: String if single prompt, List[str] if batch
        - vram_percentage: Percentage of VRAM used (0 if no GPU)
    
    Note:
        VRAM monitoring is basic in Phase 1. Full implementation in Phase 4.
    """
    # Wrap prompts in IT format if using instruction-tuned variant
    if model_variant == "it":
        prompt = [wrap_chat_it(p) for p in prompt] if isinstance(prompt, list) else wrap_chat_it(prompt)
    
    # Ensure prompt is a list for batch processing
    if not isinstance(prompt, list):
        prompt = [prompt]
    
    # Prepare inputs (text + optional image)
    if image is not None:
        if processor is None:
            raise ValueError("Processor must be provided when using images")
        # Prepare multimodal inputs
        images_batch = [[image.copy()] for _ in prompt]
        inputs = processor(
            text=prompt, 
            images=images_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
    else:
        # Text-only inputs
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=False
        ).to(model.device)
    
    # Cast only tensors that should match model dtype; preserve known input dtypes
    model_dtype = next(model.parameters()).dtype
    preserve_dtypes = {"input_ids", "attention_mask", "pixel_values"}
    for k in inputs:
        if k not in preserve_dtypes and hasattr(inputs[k], 'dtype'):
            inputs[k] = inputs[k].to(dtype=model_dtype)
    
    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Calculate VRAM usage (use centralized utility across all GPUs)
    try:
        from src.utils import calculate_vram_percentage
        vram_percentage = calculate_vram_percentage()
    except Exception as e:
        logging.warning(f"Could not calculate VRAM usage: {e}")
        vram_percentage = 0.0
    
    # outputs contain full sequence (prompt + generated text)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Return single string if single prompt, otherwise list
    return (decoded_outputs[0] if len(decoded_outputs) == 1 else decoded_outputs), vram_percentage


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    text: str = "The quick brown fox jumps over the lazy dog."
) -> float:
    """
    Calculate perplexity of the model on a given text.

    Note: Currently NOT used by the pipeline; kept for future versions.

    Note:
        This is an optional metric, disabled by default in experiments.
        Even if enabled in config, the pipeline does not call this function in this version.

    Perplexity is exp(loss) and measures how well the model predicts the text.
    Lower perplexity indicates better language modeling performance.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        text: Text to evaluate (default: simple English sentence)
    
    Returns:
        Perplexity value (float)
    
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    
    perplexity = np.exp(loss)
    return perplexity

