# Project Structure

**This file has been consolidated into the complete guide.**

For detailed information about the project structure, see:
- **[Complete Guide - Project Structure Section](docs/guide.md#project-structure)**

---

## Quick Reference

```
Repo_nuevo/
├── src/                        # Core Python modules
│   ├── main.py                 # CLI entry point
│   ├── pipeline.py             # Experiment orchestration
│   ├── model_loader.py         # Model loading & restoration
│   ├── degradation.py          # Parameter groups & degradation methods
│   ├── generation.py           # Text generation & perplexity
│   └── utils.py                # Logging, seeds, VRAM, image utils
│
├── configs/                    # Configuration modules
│   ├── experiment_configs.py   # TASKS, VARIANTS, build_config()
│   └── prompts.py              # Prompt lists by task
│
├── results/                    # Experiment outputs
│   └── samples/                # Small versioned samples
│
├── logs/                       # Execution logs
│
├── docs/                       # Documentation
│   ├── guide.md                # Complete guide (MAIN DOCUMENTATION)
│   └── output_schema.md        # JSON output schema reference
│
├── requirements.txt            # Python dependencies
├── env.example                 # Example .env file
└── README.md                   # Quick start & overview
```

For detailed descriptions of each module and their functions, see the [Complete Guide](docs/guide.md).
