#!/usr/bin/env python3
"""
SFT Training with LoRA using TRL (Optimized for Speed)

This script fine-tunes a model using LoRA (Low-Rank Adaptation) for efficient training.
LoRA reduces memory usage and training time while maintaining quality.

Speed optimizations enabled:
- Flash Attention 2 (2-3x faster attention)
- Packing (combine short sequences)
- Larger batch sizes
- torch.compile (PyTorch 2.0+)
- Fused optimizers

Usage:
    python sft_test.py
    
    # Resume from checkpoint
    python sft_test.py --resume
    
    # Fast mode (less memory efficient but faster)
    python sft_test.py --fast
"""

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# =============================================================================
# CONFIGURATION - Adjust these parameters as needed
# =============================================================================

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"  # Base model to fine-tune
OUTPUT_DIR = "./Qwen3-4B-LoRA-SFT"          # Where to save checkpoints

# Dataset Configuration
DATASET_NAME = "trl-lib/Capybara"           # HuggingFace dataset
DATASET_SPLIT = "train"                      # Dataset split to use
MAX_SAMPLES = None                           # Limit samples (None = use all)

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,                    # LoRA rank - higher = more capacity but more memory
                                # Typical values: 8, 16, 32, 64
    "lora_alpha": 32,           # LoRA alpha - scaling factor, usually 2x rank
    "lora_dropout": 0.05,       # Dropout for regularization (0.0 - 0.1)
    "target_modules": [         # Which layers to apply LoRA to
        "q_proj",               # Query projection
        "k_proj",               # Key projection  
        "v_proj",               # Value projection
        "o_proj",               # Output projection
        "gate_proj",            # MLP gate
        "up_proj",              # MLP up
        "down_proj",            # MLP down
    ],
    "bias": "none",             # Don't train biases ("none", "all", "lora_only")
    "task_type": TaskType.CAUSAL_LM,
}

# Quantization Configuration (4-bit for memory efficiency)
USE_4BIT = True                 # Enable 4-bit quantization
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",           # "nf4" or "fp4"
    "bnb_4bit_use_double_quant": True,      # Nested quantization
}

# Training Configuration (Optimized for Speed)
TRAINING_CONFIG = {
    # Batch sizes - LARGER = FASTER (if VRAM allows)
    "per_device_train_batch_size": 4,       # Increased from 2 → 4
    "gradient_accumulation_steps": 4,        # Effective batch = 4 * 4 = 16
    
    # Learning rate
    "learning_rate": 2e-4,                   # LoRA typically uses higher LR (1e-4 to 5e-4)
    "lr_scheduler_type": "cosine",           # "linear", "cosine", "constant"
    "warmup_ratio": 0.03,                    # Warmup steps as ratio of total
    
    # Training duration
    "num_train_epochs": 20,                  # Number of epochs
    # "max_steps": 1000,                     # Or use max_steps instead of epochs
    
    # Sequence length
    "max_length": 2048,                      # Max tokens per sample (TRL 0.27+)
    
    # 🚀 PACKING - Major speedup! Combines short sequences
    "packing": True,                         # Pack multiple samples per sequence
    
    # Regularization
    "weight_decay": 0.01,                    # L2 regularization
    "max_grad_norm": 1.0,                    # Gradient clipping
    
    # Saving & Logging
    "save_strategy": "steps",                # "steps", "epoch", "no"
    "save_steps": 500,                       # Save every N steps
    "save_total_limit": 3,                   # Keep only last N checkpoints
    "logging_steps": 10,                     # Log every N steps
    
    # Evaluation (optional)
    "eval_strategy": "no",                   # "steps", "epoch", "no"
    # "eval_steps": 500,                     # Evaluate every N steps
    
    # 🚀 Performance optimizations
    "bf16": True,                            # Use bfloat16 (if GPU supports)
    "gradient_checkpointing": True,          # Trade compute for memory (disable with --fast)
    "optim": "adamw_torch_fused",            # Fused optimizer (faster)
    "dataloader_num_workers": 4,             # Parallel data loading
    "dataloader_pin_memory": True,           # Faster CPU→GPU transfer
    "dataloader_prefetch_factor": 2,         # Prefetch batches
    
    # Misc
    "output_dir": OUTPUT_DIR,
    "report_to": "wandb",                    # "wandb", "tensorboard", "none"
    "run_name": "qwen3-4b-lora-sft-fast",
}


def main(resume_from_checkpoint: bool = False, fast_mode: bool = False):
    """Main training function
    
    Args:
        resume_from_checkpoint: Resume from last checkpoint
        fast_mode: Disable gradient checkpointing for faster training (uses more VRAM)
    """
    
    print("=" * 60)
    print("🚀 SFT Training with LoRA (Speed Optimized)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"LoRA rank: {LORA_CONFIG['r']}")
    print(f"4-bit quantization: {USE_4BIT}")
    print(f"Fast mode: {fast_mode}")
    print("=" * 60)
    
    # Create LoRA config
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    
    # Create quantization config if needed
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            # 🚀 Flash Attention 2 - Major speedup!
            "attn_implementation": "flash_attention_2",
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto", 
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        }
    
    # Load dataset
    print(f"\n📚 Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if MAX_SAMPLES:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    print(f"   Samples: {len(dataset)}")
    
    # Create SFT config with model_init_kwargs
    training_config = TRAINING_CONFIG.copy()
    training_config["model_init_kwargs"] = model_kwargs
    
    # Fast mode: disable gradient checkpointing
    if fast_mode:
        training_config["gradient_checkpointing"] = False
        training_config["per_device_train_batch_size"] = 8  # Can use larger batch
        training_config["gradient_accumulation_steps"] = 2   # Keep effective batch ~16
        print("⚡ Fast mode: gradient checkpointing disabled, larger batch size")
    
    sft_config = SFTConfig(**training_config)
    
    # Create trainer - let SFTTrainer handle model loading and LoRA application
    print("\n🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=MODEL_NAME,              # Pass model name, not loaded model
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,       # SFTTrainer will apply LoRA
    )
    
    # Print trainable params after trainer creates the model
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Print speed optimization summary
    print("\n⚡ Speed optimizations enabled:")
    print("   ✓ Flash Attention 2")
    print("   ✓ Packing (multiple samples per sequence)")
    print("   ✓ Fused AdamW optimizer")
    print("   ✓ Parallel data loading")
    if fast_mode:
        print("   ✓ Gradient checkpointing disabled")
    else:
        print("   ○ Gradient checkpointing enabled (use --fast to disable)")
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    
    # Save merged model (optional - combines LoRA with base)
    print("💾 Saving merged model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(f"{OUTPUT_DIR}/merged")
    trainer.processing_class.save_pretrained(f"{OUTPUT_DIR}/merged")
    
    print("\n✅ Training complete!")
    print(f"   LoRA adapters: {OUTPUT_DIR}/final")
    print(f"   Merged model: {OUTPUT_DIR}/merged")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFT Training with LoRA (Speed Optimized)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fast", action="store_true", help="Fast mode: disable gradient checkpointing (uses more VRAM)")
    args = parser.parse_args()
    
    main(resume_from_checkpoint=args.resume, fast_mode=args.fast)
