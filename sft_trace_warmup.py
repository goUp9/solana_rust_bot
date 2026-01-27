#!/usr/bin/env python3
"""
SFT Warmup Training for Trace Environment

This script fine-tunes Qwen3-4B using the weirek/sft-warmup-trace dataset
to warm up the model for further RL training on the trace environment.

The trace task involves:
1. Reading Python code with injected debug print statements
2. Predicting the exact stdout output including all __DBG_* lines
3. Binary reward: exact match = 1.0, otherwise = 0.0

Usage:
    # Basic training (chat format, recommended)
    python sft_trace_warmup.py
    
    # B200 mode - RECOMMENDED for B200 GPU (183GB VRAM, 192 cores)
    python sft_trace_warmup.py --b200
    
    # B200 mode without 4-bit quantization (even faster, uses ~20GB more VRAM)
    python sft_trace_warmup.py --b200 --no_4bit
    
    # Use reasoning format (includes thinking steps)
    python sft_trace_warmup.py --config sft_reasoning
    
    # Resume from checkpoint
    python sft_trace_warmup.py --resume
    
    # Fast mode (less memory efficient but faster)
    python sft_trace_warmup.py --fast
    
    # Custom output directory
    python sft_trace_warmup.py --output_dir ./my_model
"""

import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Model Configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"  # Base model to fine-tune
DEFAULT_OUTPUT_DIR = "./Qwen3-4B-Trace-SFT"  # Where to save checkpoints

# Dataset Configuration
DATASET_NAME = "weirek/sft-warmup-trace"
DEFAULT_CONFIG = "chat_original"  # chat_original, sft_original, or sft_reasoning
DATASET_SPLIT = "train"
MAX_SAMPLES = None  # Limit samples (None = use all ~21k samples)

# LoRA Configuration - Optimized for trace task
LORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # LoRA alpha (scaling factor)
    "lora_dropout": 0.05,       # Dropout for regularization
    "target_modules": [         # Which layers to apply LoRA to
        "q_proj",               # Query projection
        "k_proj",               # Key projection  
        "v_proj",               # Value projection
        "o_proj",               # Output projection
        "gate_proj",            # MLP gate
        "up_proj",              # MLP up
        "down_proj",            # MLP down
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Quantization Configuration (4-bit for memory efficiency)
# Note: On H200 with 143GB VRAM, you can disable 4-bit for faster training
USE_4BIT = True
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


def get_training_config(output_dir: str, fast_mode: bool = False, h200_mode: bool = False, b200_mode: bool = False) -> dict:
    """Get training configuration with optimized settings for trace task.
    
    Args:
        output_dir: Output directory for checkpoints
        fast_mode: Disable gradient checkpointing (uses more VRAM)
        h200_mode: Aggressive optimizations for H200 (143GB VRAM, 44 cores)
        b200_mode: Maximum optimizations for B200 (183GB VRAM, 192 cores)
    """
    
    # B200 optimized settings - for 183GB VRAM, 192 CPU cores
    if b200_mode:
        config = {
            # Batch sizes - reduced to avoid OOM with 4096 seq length
            # Memory usage: model (~8GB) + activations (~batch*seq*hidden*layers)
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,  # Effective batch = 32
            
            # Learning rate
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            
            # Training duration
            "num_train_epochs": 3,
            
            # Sequence length
            "max_length": 4096,
            
            # Packing disabled - requires Flash Attention to avoid cross-contamination
            # Install flash-attn and set packing=True for ~30% more efficiency
            "packing": False,
            
            # Use pre-tokenized dataset (tokenization done in load_trace_dataset)
            "dataset_text_field": None,  # Data is already tokenized
            
            # Regularization
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            
            # Less frequent saves/evals for speed
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 5,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 500,
            
            # Performance optimizations for B200
            "bf16": True,
            "gradient_checkpointing": True,  # Enable to reduce memory - needed for long sequences
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 16,  # Reduced to avoid memory pressure
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 4,
            "torch_compile": False,  # Disabled - TF32 API conflict
            
            # Output
            "output_dir": output_dir,
            "report_to": "wandb",
            "run_name": "qwen3-4b-trace-sft-b200",
        }
    # H200 optimized settings - maximize throughput
    elif h200_mode:
        config = {
            # Larger batch sizes - H200 can handle it
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 2,  # Effective batch = 32
            
            # Learning rate
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            
            # Training duration
            "num_train_epochs": 3,
            
            # Sequence length
            "max_length": 4096,
            
            # Packing disabled - requires Flash Attention
            "packing": False,
            "dataset_text_field": None,
            
            # Regularization
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            
            # Less frequent saves/evals for speed
            "save_strategy": "steps",
            "save_steps": 1000,
            "save_total_limit": 3,
            "logging_steps": 25,
            "eval_strategy": "steps",
            "eval_steps": 1000,
            
            # Performance optimizations - AGGRESSIVE
            "bf16": True,
            "gradient_checkpointing": False,  # Disable for speed on H200
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 16,  # Use more CPU cores
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 4,  # Prefetch more batches
            "torch_compile": True,  # Enable torch.compile for extra speed
            
            # Output
            "output_dir": output_dir,
            "report_to": "wandb",
            "run_name": "qwen3-4b-trace-sft-h200",
        }
    else:
        # Default settings - conservative for unknown hardware
        config = {
            # Batch sizes
            "per_device_train_batch_size": 4 if not fast_mode else 8,
            "gradient_accumulation_steps": 4 if not fast_mode else 2,
            
            # Learning rate
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            
            # Training duration
            "num_train_epochs": 3,
            
            # Sequence length
            "max_length": 4096,
            
            # Packing disabled without Flash Attention
            "packing": False,
            "dataset_text_field": None,
            
            # Regularization
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            
            # Saving & Logging
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 500,
            
            # Performance optimizations
            "bf16": True,
            "gradient_checkpointing": not fast_mode,
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 8,
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 2,
            
            # Output
            "output_dir": output_dir,
            "report_to": "wandb",
            "run_name": "qwen3-4b-trace-sft",
        }
    
    return config


def format_chat_messages(example):
    """Format chat messages for the chat_original config."""
    # The dataset already has messages in the correct format
    return example


def format_alpaca_to_chat(example, tokenizer):
    """Convert Alpaca-style format to chat messages for sft_original/sft_reasoning."""
    # Combine instruction and input
    if example.get("input"):
        user_content = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_content = example['instruction']
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['output']}
    ]
    
    return {"messages": messages}


def load_trace_dataset(config_name: str, tokenizer, max_samples: Optional[int] = None, num_proc: int = 8, max_length: int = 4096):
    """Load and prepare the trace dataset with pre-tokenization for speed.
    
    Args:
        config_name: Dataset configuration name
        tokenizer: Tokenizer instance
        max_samples: Limit number of samples (for testing)
        num_proc: Number of processes for dataset preprocessing
        max_length: Maximum sequence length for tokenization
    """
    print(f"📚 Loading dataset: {DATASET_NAME} (config: {config_name})")
    
    # Load train and test splits
    train_dataset = load_dataset(DATASET_NAME, config_name, split="train")
    test_dataset = load_dataset(DATASET_NAME, config_name, split="test")
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Filter out corrupted/extremely long samples BEFORE any processing
    # Some samples have 1B+ characters which crash tokenization
    MAX_CHARS = 50000  # ~12k tokens max, well under 4096 after truncation buffer
    
    def is_reasonable_length(example):
        """Filter out samples that are unreasonably long (corrupted data)."""
        total_chars = sum(len(m["content"]) for m in example["messages"])
        return total_chars < MAX_CHARS
    
    original_train = len(train_dataset)
    original_test = len(test_dataset)
    
    train_dataset = train_dataset.filter(is_reasonable_length, num_proc=num_proc, desc="Filtering train")
    test_dataset = test_dataset.filter(is_reasonable_length, num_proc=num_proc, desc="Filtering test")
    
    filtered_train = original_train - len(train_dataset)
    filtered_test = original_test - len(test_dataset)
    if filtered_train > 0 or filtered_test > 0:
        print(f"   ⚠️  Filtered {filtered_train} train, {filtered_test} test samples (>{MAX_CHARS} chars, likely corrupted)")
    print(f"   After filtering: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Limit samples if specified
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(max_samples // 10, len(test_dataset))))
        print(f"   Limited to: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Convert format if needed (with multiprocessing for speed)
    if config_name in ["sft_original", "sft_reasoning"]:
        print(f"   Converting Alpaca format to chat messages (using {num_proc} processes)...")
        train_dataset = train_dataset.map(
            lambda x: format_alpaca_to_chat(x, tokenizer),
            remove_columns=["text", "instruction", "input", "output"],
            num_proc=num_proc,
            desc="Converting train"
        )
        test_dataset = test_dataset.map(
            lambda x: format_alpaca_to_chat(x, tokenizer),
            remove_columns=["text", "instruction", "input", "output"],
            num_proc=num_proc,
            desc="Converting test"
        )
    
    # First, apply chat template to convert messages to text (lightweight, can use many processes)
    print(f"   Applying chat template (using {num_proc} processes)...")
    
    def apply_chat_template(examples):
        """Apply chat template to convert messages to text."""
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(
        apply_chat_template,
        batched=True,
        batch_size=500,
        num_proc=num_proc,
        remove_columns=["messages"],
        desc="Applying template (train)",
        load_from_cache_file=True,
    )
    
    test_dataset = test_dataset.map(
        apply_chat_template,
        batched=True,
        batch_size=500,
        num_proc=num_proc,
        remove_columns=["messages"],
        desc="Applying template (test)",
        load_from_cache_file=True,
    )
    
    # Now tokenize - use fewer processes to avoid memory issues with tokenizer
    # Each process loads the tokenizer, so limit to 8 processes max
    tokenize_num_proc = min(num_proc, 8)
    print(f"   Tokenizing dataset (using {tokenize_num_proc} processes)...")
    
    def tokenize_function(examples):
        """Tokenize text."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=500,
        num_proc=tokenize_num_proc,
        remove_columns=["text"],
        desc="Tokenizing train",
        load_from_cache_file=True,
    )
    
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=500,
        num_proc=tokenize_num_proc,
        remove_columns=["text"],
        desc="Tokenizing test",
        load_from_cache_file=True,
    )
    
    # Filter out samples that are too long (> max_length)
    original_train_len = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids"]) <= max_length,
        num_proc=num_proc,
    )
    if len(train_dataset) < original_train_len:
        print(f"   Filtered {original_train_len - len(train_dataset)} samples exceeding {max_length} tokens")
    
    print(f"   Final dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset


def main(
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    config_name: str = DEFAULT_CONFIG,
    resume_from_checkpoint: bool = False,
    fast_mode: bool = False,
    h200_mode: bool = False,
    b200_mode: bool = False,
    max_samples: Optional[int] = None,
    use_4bit: bool = True,
    disable_wandb: bool = False,
):
    """Main training function."""
    
    print("=" * 70)
    print("🚀 SFT Warmup Training for Trace Environment")
    print("=" * 70)
    print(f"Model:       {model_name}")
    print(f"Dataset:     {DATASET_NAME} (config: {config_name})")
    print(f"Output:      {output_dir}")
    print(f"LoRA rank:   {LORA_CONFIG['r']}")
    print(f"4-bit:       {use_4bit}")
    print(f"Fast mode:   {fast_mode}")
    print(f"H200 mode:   {h200_mode}")
    print(f"B200 mode:   {b200_mode}")
    print("=" * 70)
    
    # Create LoRA config
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    
    # Check if Flash Attention 2 is available
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("   Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"  # Scaled dot-product attention (PyTorch native)
        print("   Flash Attention 2 not installed, using SDPA")
    
    # Create quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
        }
    
    # Load tokenizer first (needed for dataset formatting)
    print(f"\n📦 Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset (use more processes on high-end GPUs)
    if b200_mode:
        num_proc = 32  # B200 has 192 cores
    elif h200_mode:
        num_proc = 16
    else:
        num_proc = 8
    max_length = 4096  # Max sequence length
    train_dataset, eval_dataset = load_trace_dataset(
        config_name, 
        tokenizer, 
        max_samples=max_samples,
        num_proc=num_proc,
        max_length=max_length,
    )
    
    # Get training config
    training_config = get_training_config(output_dir, fast_mode, h200_mode, b200_mode)
    training_config["model_init_kwargs"] = model_kwargs
    
    if disable_wandb:
        training_config["report_to"] = "none"
    
    # Create SFT config
    sft_config = SFTConfig(**training_config)
    
    # Create trainer
    print("\n🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    # Print trainable params
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Print optimization summary
    print("\n⚡ Optimizations enabled:")
    print(f"   ✓ {attn_impl.upper().replace('_', ' ')}")
    print("   ✓ Pre-tokenized dataset (fast multiprocessing tokenization)")
    print("   ✓ Fused AdamW optimizer")
    print(f"   ✓ Parallel data loading ({training_config.get('dataloader_num_workers', 4)} workers)")
    if b200_mode:
        print("   ✓ B200 mode: batch=8, grad_accum=4 (effective=32), grad checkpointing")
    elif h200_mode:
        print("   ✓ H200 mode: large batches, torch.compile, no grad checkpointing")
    elif fast_mode:
        print("   ✓ Gradient checkpointing disabled (fast mode)")
    else:
        print("   ○ Gradient checkpointing enabled (use --fast, --h200, or --b200 to disable)")
    if not training_config.get("packing", False):
        print("   ○ Packing disabled (install flash-attn to enable)")
    
    # Train
    print("\n🚀 Starting training...")
    print("   This may take several hours depending on your GPU.")
    print("   Monitor progress on wandb or tensorboard.\n")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    print("\n💾 Saving final model...")
    final_path = f"{output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Optionally merge LoRA weights
    print("💾 Merging LoRA weights with base model...")
    try:
        merged_model = trainer.model.merge_and_unload()
        merged_path = f"{output_dir}/merged"
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"   Merged model saved to: {merged_path}")
    except Exception as e:
        print(f"   Warning: Could not merge model: {e}")
        print("   You can merge later using scripts/merge.py")
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"   LoRA adapters: {final_path}")
    if os.path.exists(f"{output_dir}/merged"):
        print(f"   Merged model:  {output_dir}/merged")
    print("\nNext steps:")
    print("   1. Evaluate the model on trace tasks")
    print("   2. Use for RL training with train_trace_ppo.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT Warmup Training for Trace Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # B200 mode - RECOMMENDED for your machine (183GB VRAM, 192 cores)
    python sft_trace_warmup.py --b200
    
    # B200 mode without quantization (faster, you have plenty of VRAM)
    python sft_trace_warmup.py --b200 --no_4bit
    
    # Use reasoning format with extended thinking
    python sft_trace_warmup.py --config sft_reasoning --b200
    
    # Resume from checkpoint
    python sft_trace_warmup.py --resume --b200
    
    # Limit samples for testing
    python sft_trace_warmup.py --max_samples 1000 --b200
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help=f"Base model to fine-tune (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for checkpoints (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=DEFAULT_CONFIG,
        choices=["chat_original", "sft_original", "sft_reasoning"],
        help="Dataset config to use (default: chat_original)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Fast mode: disable gradient checkpointing (uses more VRAM)"
    )
    parser.add_argument(
        "--h200",
        action="store_true",
        help="H200 mode: aggressive optimizations for H200 GPU (143GB VRAM, 44 cores)"
    )
    parser.add_argument(
        "--b200",
        action="store_true",
        help="B200 mode: maximum optimizations for B200 GPU (183GB VRAM, 192 cores)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Limit number of training samples (for testing)"
    )
    parser.add_argument(
        "--no_4bit", 
        action="store_true",
        help="Disable 4-bit quantization (uses more VRAM)"
    )
    parser.add_argument(
        "--no_wandb", 
        action="store_true",
        help="Disable wandb logging"
    )
    
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        output_dir=args.output_dir,
        config_name=args.config,
        resume_from_checkpoint=args.resume,
        fast_mode=args.fast,
        h200_mode=args.h200,
        b200_mode=args.b200,
        max_samples=args.max_samples,
        use_4bit=not args.no_4bit,
        disable_wandb=args.no_wandb,
    )
