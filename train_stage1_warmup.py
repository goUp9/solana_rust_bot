#!/usr/bin/env python3
"""
Stage 1: Original Dataset Warmup Training

This script trains the model on the original code → output mapping
WITHOUT injected debug prints. This serves as a warmup to teach
the model basic Python execution before moving to the harder task.

Purpose:
- Teach the model Python semantics and execution patterns
- Build foundational understanding of loops, conditionals, I/O
- Lower difficulty than trace task → faster convergence

Usage:
    # Basic training
    python train_stage1_warmup.py
    
    # B200 optimized (recommended for B200 GPU)
    python train_stage1_warmup.py --b200
    
    # B200 without quantization (faster, uses more VRAM)
    python train_stage1_warmup.py --b200 --no_4bit
    
    # Custom dataset path
    python train_stage1_warmup.py --dataset ./datasets/trace_training/stage1_original_train.jsonl
    
    # Resume from checkpoint
    python train_stage1_warmup.py --resume --b200
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Model Configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_OUTPUT_DIR = "./checkpoints/stage1_warmup"

# Dataset Configuration
DEFAULT_DATASET_PATH = "./datasets/trace_training/stage1_original_train.jsonl"
DEFAULT_EVAL_PATH = "./datasets/trace_training/stage1_original_test.jsonl"

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# Quantization Configuration
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


def get_training_config(
    output_dir: str,
    b200_mode: bool = False,
    num_epochs: int = 2,
) -> Dict[str, Any]:
    """Get training configuration optimized for hardware."""
    
    if b200_mode:
        # B200 optimized: 183GB VRAM, 192 CPU cores
        config = {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,  # Effective batch = 32
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "num_train_epochs": num_epochs,
            "max_length": 2048,  # Shorter for warmup (original code is simpler)
            "packing": False,
            "dataset_text_field": None,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_strategy": "steps",
            "save_steps": 200,
            "save_total_limit": 3,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 16,
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 4,
            "torch_compile": False,
            "output_dir": output_dir,
            "report_to": "wandb",
            "run_name": "qwen3-4b-stage1-warmup",
        }
    else:
        # Default conservative settings
        config = {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": num_epochs,
            "max_length": 2048,
            "packing": False,
            "dataset_text_field": None,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch_fused",
            "dataloader_num_workers": 8,
            "dataloader_pin_memory": True,
            "output_dir": output_dir,
            "report_to": "wandb",
            "run_name": "qwen3-4b-stage1-warmup",
        }
    
    return config


def load_jsonl_dataset(path: str, tokenizer, max_length: int, num_proc: int = 8):
    """Load JSONL dataset and tokenize."""
    
    print(f"📚 Loading dataset: {path}")
    
    # Load JSONL
    samples = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)
    
    print(f"   Loaded {len(samples):,} samples")
    
    # Convert to HF Dataset
    dataset = Dataset.from_list([{"messages": s["messages"]} for s in samples])
    
    # Apply chat template
    def apply_chat_template(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(
        apply_chat_template,
        batched=True,
        batch_size=500,
        num_proc=num_proc,
        remove_columns=["messages"],
        desc="Applying template",
    )
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=500,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    # Filter by length
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length, num_proc=num_proc)
    if len(dataset) < original_len:
        print(f"   Filtered {original_len - len(dataset)} samples exceeding {max_length} tokens")
    
    print(f"   Final: {len(dataset):,} samples")
    
    return dataset


def main(args):
    """Main training function."""
    
    print("\n" + "=" * 70)
    print("🚀 Stage 1: Original Dataset Warmup Training")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {args.output_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"B200 mode:   {args.b200}")
    print(f"4-bit:       {not args.no_4bit}")
    print("=" * 70)
    
    # Create LoRA config
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    
    # Check Flash Attention
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("   Using Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        print("   Using SDPA (Flash Attention 2 not available)")
    
    # Model kwargs
    if not args.no_4bit:
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
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get training config
    training_config = get_training_config(args.output_dir, args.b200, args.epochs)
    max_length = training_config["max_length"]
    num_proc = 32 if args.b200 else 8
    
    # Load datasets
    train_dataset = load_jsonl_dataset(args.dataset, tokenizer, max_length, num_proc)
    
    eval_dataset = None
    if args.eval_dataset and Path(args.eval_dataset).exists():
        eval_dataset = load_jsonl_dataset(args.eval_dataset, tokenizer, max_length, num_proc)
    
    # Update config
    training_config["model_init_kwargs"] = model_kwargs
    if args.no_wandb:
        training_config["report_to"] = "none"
    
    # Create SFT config
    sft_config = SFTConfig(**training_config)
    
    # Create trainer
    print("\n🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    # Print info
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)
    
    # Save
    print("\n💾 Saving model...")
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Merge LoRA
    print("💾 Merging LoRA weights...")
    try:
        merged = trainer.model.merge_and_unload()
        merged_path = f"{args.output_dir}/merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"   Merged model: {merged_path}")
    except Exception as e:
        print(f"   Warning: Could not merge: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Stage 1 Complete!")
    print("=" * 70)
    print(f"   LoRA adapters: {final_path}")
    print(f"   Merged model:  {args.output_dir}/merged")
    print("\nNext: python train_stage2_sft.py --base_model", f"{args.output_dir}/merged")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Original Dataset Warmup Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--eval_dataset", type=str, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--b200", action="store_true", help="B200 optimizations")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    main(args)
