#!/usr/bin/env python3
"""
Evaluation Script for SFT-trained Trace Model

This script evaluates a model trained with sft_trace_warmup.py on the trace task.
It measures exact match accuracy on predicting program stdout.

Usage:
    # Evaluate merged model
    python eval_trace_sft.py --model ./Qwen3-4B-Trace-SFT/merged
    
    # Evaluate LoRA adapters (local)
    python eval_trace_sft.py --model ./Qwen3-4B-Trace-SFT/final --base_model Qwen/Qwen3-4B
    
    # Evaluate LoRA adapters from HuggingFace (auto-detects base model)
    python eval_trace_sft.py --model username/Qwen3-4B-Trace-LoRA
    
    # Evaluate on specific number of samples
    python eval_trace_sft.py --model ./Qwen3-4B-Trace-SFT/merged --num_samples 100
    
    # Compare base vs SFT model
    python eval_trace_sft.py --model Qwen/Qwen3-4B --num_samples 50
"""

import os
import re
import json
import argparse
from typing import Optional, Tuple
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download, repo_exists, list_repo_files


# Dataset config
DATASET_NAME = "weirek/sft-warmup-trace"
DATASET_CONFIG = "chat_original"


def clean_prediction(prediction: str) -> str:
    """Remove reasoning tags and markdown from LLM response."""
    # Remove thinking tags
    prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
    prediction = re.sub(r"<thinking>.*?</thinking>", "", prediction, flags=re.DOTALL)
    
    # Extract from code blocks if present
    code_block_match = re.search(r"```(?:[a-zA-Z]*)\n?(.*?)\n?```", prediction, flags=re.DOTALL)
    if code_block_match:
        prediction = code_block_match.group(1)
    
    return prediction.strip()


def compare_outputs(expected: str, actual: str) -> bool:
    """Compare outputs with whitespace normalization."""
    def normalize(s):
        return "".join(s.split()).lower()
    return normalize(expected) == normalize(actual)


def is_lora_adapter(model_path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the model path contains LoRA adapters.
    Works for both local paths and HuggingFace repos.
    
    Returns:
        (is_lora, base_model_name): Tuple of whether it's LoRA and the base model name if found
    """
    # Check local path first
    local_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, "r") as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path")
            return True, base_model_name
        except Exception:
            return True, None
    
    # Check if it's a HuggingFace repo
    if "/" in model_path and not os.path.exists(model_path):
        try:
            # Check if repo exists and contains adapter_config.json
            if repo_exists(model_path):
                repo_files = list_repo_files(model_path)
                if "adapter_config.json" in repo_files:
                    # Download and read the adapter config
                    config_path = hf_hub_download(
                        repo_id=model_path,
                        filename="adapter_config.json",
                    )
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                    print(f"   Detected LoRA adapter repo on HuggingFace")
                    if base_model_name:
                        print(f"   Auto-detected base model: {base_model_name}")
                    return True, base_model_name
        except Exception as e:
            print(f"   Warning: Could not check HuggingFace repo: {e}")
    
    return False, None


def load_model_and_tokenizer(
    model_path: str,
    base_model: Optional[str] = None,
    use_4bit: bool = True,
) -> Tuple:
    """Load model and tokenizer, handling both merged and LoRA models."""
    
    print(f"📦 Loading model from: {model_path}")
    
    # Set up quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Check if this is a LoRA adapter (local or HuggingFace)
    is_lora, detected_base_model = is_lora_adapter(model_path)
    
    # Use detected base model if not explicitly provided
    if is_lora and base_model is None:
        base_model = detected_base_model
    
    if is_lora:
        if base_model is None:
            raise ValueError(
                "--base_model required when loading LoRA adapters. "
                "Could not auto-detect base model from adapter_config.json"
            )
        
        print(f"   Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )
        
        print(f"   Loading LoRA adapters...")
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Merged model or base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def evaluate_sample(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Generate response for a single sample."""
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages[:-1],  # Only include user message
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response only
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return response


def main(
    model_path: str,
    base_model: Optional[str] = None,
    num_samples: int = 100,
    use_4bit: bool = True,
    verbose: bool = False,
):
    """Main evaluation function."""
    
    print("=" * 70)
    print("🔍 Trace SFT Model Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    if base_model:
        print(f"Base:  {base_model}")
    print(f"Samples: {num_samples}")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        base_model=base_model,
        use_4bit=use_4bit,
    )
    
    # Load test dataset
    print(f"\n📚 Loading test dataset...")
    test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    print(f"   Total test samples: {len(test_dataset)}")
    
    # Limit samples
    if num_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(num_samples))
    
    # Evaluate
    print(f"\n🧪 Evaluating on {len(test_dataset)} samples...\n")
    
    correct = 0
    total = 0
    results = []
    
    for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        messages = sample["messages"]
        expected = messages[-1]["content"]  # Assistant's response
        
        # Generate prediction
        prediction = evaluate_sample(model, tokenizer, messages)
        
        # Clean and compare
        cleaned_prediction = clean_prediction(prediction)
        is_correct = compare_outputs(expected, cleaned_prediction)
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "index": i,
            "correct": is_correct,
            "expected_len": len(expected),
            "prediction_len": len(cleaned_prediction),
        })
        
        if verbose and not is_correct:
            print(f"\n--- Sample {i} (INCORRECT) ---")
            print(f"Expected ({len(expected)} chars):")
            print(expected[:200] + "..." if len(expected) > 200 else expected)
            print(f"\nPredicted ({len(cleaned_prediction)} chars):")
            print(cleaned_prediction[:200] + "..." if len(cleaned_prediction) > 200 else cleaned_prediction)
    
    # Print results
    accuracy = correct / total * 100
    
    print("\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)
    print(f"   Correct: {correct} / {total}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print("=" * 70)
    
    # Categorize by output length
    short_correct = sum(1 for r in results if r["expected_len"] < 500 and r["correct"])
    short_total = sum(1 for r in results if r["expected_len"] < 500)
    medium_correct = sum(1 for r in results if 500 <= r["expected_len"] < 2000 and r["correct"])
    medium_total = sum(1 for r in results if 500 <= r["expected_len"] < 2000)
    long_correct = sum(1 for r in results if r["expected_len"] >= 2000 and r["correct"])
    long_total = sum(1 for r in results if r["expected_len"] >= 2000)
    
    print("\n📈 Accuracy by output length:")
    if short_total > 0:
        print(f"   Short (<500 chars):    {short_correct}/{short_total} = {short_correct/short_total*100:.1f}%")
    if medium_total > 0:
        print(f"   Medium (500-2000):     {medium_correct}/{medium_total} = {medium_correct/medium_total*100:.1f}%")
    if long_total > 0:
        print(f"   Long (>2000 chars):    {long_correct}/{long_total} = {long_correct/long_total*100:.1f}%")
    
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SFT-trained model on trace task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (merged model or LoRA adapters)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path (required if --model is LoRA adapters)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print incorrect predictions"
    )
    
    args = parser.parse_args()
    
    main(
        model_path=args.model,
        base_model=args.base_model,
        num_samples=args.num_samples,
        use_4bit=not args.no_4bit,
        verbose=args.verbose,
    )
