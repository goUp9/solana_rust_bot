# Trace Environment Training Pipeline

Complete 3-stage training pipeline for the Trace environment (code execution prediction with debug print injection).

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRACE TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 0: Data Generation (~30-45 min)                                  │
│  ├── Load satpalsr/rl-python dataset (23,303 samples)                   │
│  ├── Generate original dataset (code → output)                          │
│  └── Generate transformed dataset (code with __DBG_ prints → output)    │
│                                                                         │
│  Stage 1: Original Warmup (~20 min)                                     │
│  ├── Train on original code → output mapping                            │
│  └── Teaches basic Python execution patterns                            │
│                                                                         │
│  Stage 2: Transformed SFT (~2 hours)                                    │
│  ├── Train on code with injected debug prints                           │
│  ├── Multiple variants per sample prevent memorization                  │
│  └── Core capability for trace task                                     │
│                                                                         │
│  Stage 3: PPO/RL Training (~6-8 hours)                                  │
│  ├── Online learning with trace environment                             │
│  ├── Binary rewards (exact match)                                       │
│  └── Optional curriculum learning                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Full Pipeline (Recommended)

```bash
# Run entire pipeline with B200 optimizations
python run_full_pipeline.py --b200

# Or with faster training (no quantization, uses ~20GB more VRAM)
python run_full_pipeline.py --b200 --no_4bit
```

### Option 2: Quick Test

```bash
# Quick test run (~30-45 minutes)
python run_full_pipeline.py --quick_test --b200
```

### Option 3: Run Stages Individually

```bash
# Stage 0: Generate datasets
python generate_trace_datasets.py --variants 3 --num_workers 32

# Stage 1: Warmup training
python train_stage1_warmup.py --b200 --epochs 2

# Stage 2: SFT training (use Stage 1 output)
python train_stage2_sft.py --b200 --base_model ./checkpoints/stage1_warmup/merged

# Stage 3: PPO training (use Stage 2 output)
python train_stage3_ppo.py --base_model ./checkpoints/stage2_sft/merged
```

## Time Estimates (B200 GPU)

| Stage | Description | Time | Cumulative |
|-------|-------------|------|------------|
| 0 | Data Generation | 30-45 min | 45 min |
| 1 | Original Warmup (2 epochs) | 18-20 min | ~1 hour |
| 2 | Transformed SFT (3 epochs) | 1.5-2 hours | ~3 hours |
| 3 | PPO Training (5000 steps) | 5-7 hours | **~10 hours** |

### Quick Test Mode

| Stage | Time |
|-------|------|
| 0 | ~5 min |
| 1 | ~3 min |
| 2 | ~10 min |
| 3 | ~15 min |
| **Total** | **~35 min** |

## Configuration

### Pipeline Configuration

Edit `config/pipeline_config.json`:

```json
{
  "model": {
    "base_model": "Qwen/Qwen3-4B"
  },
  "hardware": {
    "b200_mode": true,
    "no_4bit": false
  },
  "stage0": {
    "variants_per_sample": 3,
    "num_workers": 32
  },
  "stage1": {
    "epochs": 2
  },
  "stage2": {
    "epochs": 3
  },
  "stage3": {
    "num_steps": 5000
  }
}
```

### PPO Configuration

Edit `config/stage3_ppo_config.json` for RL-specific settings:

```json
{
  "ppo": {
    "batch_size": 16,
    "learning_rate": 5e-7,
    "num_train_steps": 5000
  },
  "curriculum": {
    "enabled": true,
    "promotion_threshold": 0.7
  }
}
```

## Output Structure

```
./datasets/trace_training/
├── stage1_original_train.jsonl      # Original code → output
├── stage1_original_test.jsonl
├── stage2_transformed_train.jsonl   # Transformed code → output
├── stage2_transformed_test.jsonl
├── stage1_stats.json
└── stage2_stats.json

./checkpoints/
├── stage1_warmup/
│   ├── final/                       # LoRA adapters
│   └── merged/                      # Merged model
├── stage2_sft/
│   ├── final/
│   └── merged/
└── stage3_ppo/
    ├── step_1000/
    ├── step_2000/
    └── final/
```

## Resume Training

If the pipeline fails, you can resume from any stage:

```bash
# Resume from Stage 2
python run_full_pipeline.py --start_stage 2 --b200

# Or run individual stages
python train_stage2_sft.py --resume --b200
```

## Evaluation

After training, evaluate the model:

```bash
# Evaluate on trace task
python eval_trace_sft.py --model ./checkpoints/stage3_ppo/final

# Or serve for testing
python serve_sft_model.py --model ./checkpoints/stage3_ppo/final --port 8001
```

## Hardware Requirements

### Minimum (4-bit quantization)
- GPU: 24GB VRAM (e.g., RTX 3090, A10)
- RAM: 32GB
- Storage: 50GB

### Recommended (B200/H200)
- GPU: 80GB+ VRAM
- RAM: 128GB+
- Storage: 100GB+
- CPU: 32+ cores

## Troubleshooting

### Out of Memory

```bash
# Use 4-bit quantization
python train_stage2_sft.py --b200  # 4-bit is default

# Reduce batch size
python train_stage2_sft.py --batch_size 4
```

### Data Generation Errors

```bash
# Reduce workers if hitting memory limits
python generate_trace_datasets.py --num_workers 8

# Test with fewer samples first
python generate_trace_datasets.py --max_samples 1000
```

### Stage 3 Slow

PPO training is inherently slower due to:
1. Online generation (model inference)
2. Code execution (trace environment)
3. Reward computation

Consider:
- Reducing `num_steps`
- Increasing `batch_size` if VRAM allows
- Using curriculum learning to focus on easier samples first

## Files Reference

| File | Description |
|------|-------------|
| `run_full_pipeline.py` | Master orchestration script |
| `generate_trace_datasets.py` | Stage 0: Data generation |
| `train_stage1_warmup.py` | Stage 1: Original warmup |
| `train_stage2_sft.py` | Stage 2: Transformed SFT |
| `train_stage3_ppo.py` | Stage 3: PPO/RL training |
| `config/pipeline_config.json` | Pipeline configuration |
| `config/stage3_ppo_config.json` | PPO configuration |

## Credits

- Dataset: [satpalsr/rl-python](https://huggingface.co/datasets/satpalsr/rl-python)
- Base Model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Trace Environment: Affinetes
