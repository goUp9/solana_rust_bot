#!/usr/bin/env python3
"""
Stage 2: Transformed Dataset SFT Training

This script trains the model on code with injected debug prints.
This is the MAIN training stage that teaches the actual trace task:
predicting stdout output including __DBG_N__ debug prints.

Purpose:
- Teach the model to trace code execution with debug print injection
- Multiple variants per sample prevent memorization
- Build core capability for the trace environment

Prerequisites:
- Run generate_trace_datasets.py first to create the dataset
- Optionally run train_stage1_warmup.py first for better starting point

Usage:
    # Basic training (uses Qwen3-4B as base)
    python train_stage2_sft.py
    
    # Use Stage 1 warmup model as base (recommended)
    python train_stage2_sft.py --base_model ./checkpoints/stage1_warmup/merged
    
    # B200 optimized
    python train_stage2_sft.py --b200
    
    # B200 without quantization
    python train_stage2_sft.py --b200 --no_4bit
    
    # Custom settings
    python train_stage2_sft.py --epochs 4 --batch_size 16
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
DEFAULT_OUTPUT_DIR = "./checkpoints/stage2_sft"

# Dataset Configuration
DEFAULT_DATASET_PATH = "./datasets/trace_training/stage2_transformed_train.jsonl"
DEFAULT_EVAL_PATH = "./datasets/trace_training/stage2_transformed_test.jsonl"

# LoRA Configuration - same as Stage 1 for consistency
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
    num_epochs: int = 3,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Get training configuration optimized for hardware."""
    
    if b200_mode:
        # B200 optimized: 183GB VRAM, 192 CPU cores
        config = {
            "per_device_train_batch_size": batch_size or 8,
            "gradient_accumulation_steps": 4,  # Effective batch = 32
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "num_train_epochs": num_epochs,
            "max_length": 4096,  # Longer for transformed code with debug prints
            "packing": False,
            "dataset_text_field": None,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 5,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 500,
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
            "run_name": "qwen3-4b-stage2-sft",
        }
    else:
        # Default conservative settings
        config = {
            "per_device_train_batch_size": batch_size or 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": num_epochs,
            "max_length": 4096,
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
            "run_name": "qwen3-4b-stage2-sft",
        }
    
    return config


def load_jsonl_dataset(
    path: str,
    tokenizer,
    max_length: int,
    num_proc: int = 8,
    max_samples: Optional[int] = None,
):
    """Load JSONL dataset and tokenize."""
    
    print(f"📚 Loading dataset: {path}")
    
    # Load JSONL
    samples = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            samples.append(data)
    
    print(f"   Loaded {len(samples):,} samples")
    
    # Filter extremely long samples before processing
    MAX_CHARS = 50000
    original_count = len(samples)
    samples = [s for s in samples if sum(len(m["content"]) for m in s["messages"]) < MAX_CHARS]
    if len(samples) < original_count:
        print(f"   Filtered {original_count - len(samples)} samples (>{MAX_CHARS} chars)")
    
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
        load_from_cache_file=True,
    )
    
    # Tokenize
    tokenize_proc = min(num_proc, 8)  # Limit tokenization processes
    
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
        num_proc=tokenize_proc,
        remove_columns=["text"],
        desc="Tokenizing",
        load_from_cache_file=True,
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
    print("🚀 Stage 2: Transformed Dataset SFT Training")
    print("=" * 70)
    print(f"Base Model:  {args.base_model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {args.output_dir}")
    print(f"Epochs:      {args.epochs}")
    print(f"B200 mode:   {args.b200}")
    print(f"4-bit:       {not args.no_4bit}")
    print("=" * 70)
    
    # Check if using warmup model
    if args.base_model != DEFAULT_MODEL_NAME:
        print(f"\n📦 Using fine-tuned base: {args.base_model}")
        if not Path(args.base_model).exists():
            print(f"   WARNING: Path does not exist. Will try to load from HuggingFace.")
    
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
    print(f"\n📦 Loading tokenizer...")
    # Use base tokenizer for custom models
    tokenizer_name = args.base_model if Path(args.base_model).exists() else DEFAULT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get training config
    training_config = get_training_config(
        args.output_dir, 
        args.b200, 
        args.epochs,
        args.batch_size,
    )
    max_length = training_config["max_length"]
    num_proc = 32 if args.b200 else 8
    
    # Load datasets
    train_dataset = load_jsonl_dataset(
        args.dataset, 
        tokenizer, 
        max_length, 
        num_proc,
        args.max_samples,
    )
    
    eval_dataset = None
    if args.eval_dataset and Path(args.eval_dataset).exists():
        eval_dataset = load_jsonl_dataset(
            args.eval_dataset, 
            tokenizer, 
            max_length, 
            num_proc,
            args.max_samples // 10 if args.max_samples else None,
        )
    
    # Update config
    training_config["model_init_kwargs"] = model_kwargs
    if args.no_wandb:
        training_config["report_to"] = "none"
    
    # Create SFT config
    sft_config = SFTConfig(**training_config)
    
    # Create trainer
    print("\n🏋️ Creating trainer...")
    trainer = SFTTrainer(
        model=args.base_model,
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
    
    # Estimate time
    steps_per_epoch = len(train_dataset) // (training_config["per_device_train_batch_size"] * training_config["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * args.epochs
    est_time_mins = total_steps * 0.85 / 60  # ~0.85s per step on B200
    print(f"📊 Steps: {steps_per_epoch}/epoch × {args.epochs} epochs = {total_steps:,} total")
    print(f"📊 Estimated time: ~{est_time_mins:.0f} minutes ({est_time_mins/60:.1f} hours)")
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)
    
    # Save
    print("\n💾 Saving model...")
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Merge LoRA - need to reload in full precision for quantized models
    print("💾 Merging LoRA weights...")
    merged_path = f"{args.output_dir}/merged"
    try:
        from peft import PeftModel, AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM
        
        # For quantized models, we need to reload base model in full precision
        # then apply the LoRA adapters
        if not args.no_4bit:
            print("   Reloading base model in full precision for merging...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            # Load and merge the LoRA adapters
            model_with_lora = PeftModel.from_pretrained(base_model, final_path)
            merged = model_with_lora.merge_and_unload()
        else:
            # For non-quantized models, can merge directly
            merged = trainer.model.merge_and_unload()
        
        merged.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)
        print(f"   Merged model: {merged_path}")
        
        # Remove quantization config from merged model (it's now in full precision)
        config_path = os.path.join(merged_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'quantization_config' in config:
                del config['quantization_config']
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"   ✓ Removed quantization_config from merged model")
        
        # Verify the merged model has weights
        merged_files = os.listdir(merged_path)
        has_weights = any(f.endswith('.safetensors') or f.endswith('.bin') for f in merged_files if 'model' in f.lower())
        if has_weights:
            print(f"   ✓ Verified: model weights saved successfully")
        else:
            print(f"   ⚠ Warning: No model weight files found in {merged_path}")
    except Exception as e:
        import traceback
        print(f"   Warning: Could not merge: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Stage 2 Complete!")
    print("=" * 70)
    print(f"   LoRA adapters: {final_path}")
    print(f"   Merged model:  {args.output_dir}/merged")
    print("\nNext: python train_trace_ppo.py --base_model", f"{args.output_dir}/merged")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Transformed Dataset SFT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL_NAME,
                        help="Base model (use Stage 1 output for best results)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--eval_dataset", type=str, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (for testing)")
    parser.add_argument("--b200", action="store_true", help="B200 optimizations")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    main(args)
