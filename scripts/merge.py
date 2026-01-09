#!/usr/bin/env python3
"""
Script to merge LoRA adapters with base model
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapters(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
):
    """
    Merge LoRA adapters with base model
    
    Args:
        base_model_name: Name of base model on HuggingFace
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapters with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_model.save_pretrained(output_path)
    
    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)
    
    print("✅ Merge complete!")
    print(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--adapters",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    
    args = parser.parse_args()
    
    merge_adapters(
        base_model_name=args.base,
        adapter_path=args.adapters,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
