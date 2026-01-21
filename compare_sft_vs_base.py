#!/usr/bin/env python3
"""
Compare SFT-trained model vs Base model on Capybara Test Dataset

This script evaluates both models on the trl-lib/Capybara test split and compares:
- Response quality (using various metrics)
- Inference speed
- Response length and coherence

Usage:
    python compare_sft_vs_base.py --num_samples 100 --checkpoint checkpoint-5928
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class EvalResult:
    """Single evaluation result"""
    sample_id: int
    user_prompt: str
    model_response: str
    reference_response: str
    inference_time: float
    prompt_tokens: int
    response_tokens: int
    # Quality metrics
    response_length: int
    has_content: bool  # Non-empty meaningful response


@dataclass 
class ModelEvalSummary:
    """Summary of model evaluation"""
    model_name: str
    num_samples: int
    avg_response_length: float
    avg_inference_time: float
    total_time: float
    valid_response_rate: float  # Percentage of non-empty responses
    results: List[EvalResult] = field(default_factory=list)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    use_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer"""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
) -> Tuple[str, float, int, int]:
    """Generate response and return (response, time, prompt_tokens, response_tokens)"""
    
    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    
    prompt_tokens = inputs["input_ids"].shape[1]
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    inference_time = time.time() - start_time
    
    # Decode response only (not prompt)
    response_tokens = outputs[0].shape[0] - prompt_tokens
    response = tokenizer.decode(
        outputs[0][prompt_tokens:],
        skip_special_tokens=True,
    )
    
    return response, inference_time, prompt_tokens, response_tokens


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    sample_indices: List[int],
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
) -> ModelEvalSummary:
    """Evaluate model on Capybara test dataset"""
    
    results = []
    total_time = 0.0
    total_response_length = 0
    valid_responses = 0
    
    print(f"\nEvaluating {model_name}...")
    
    for idx in tqdm(sample_indices, desc=model_name):
        sample = dataset[idx]
        messages = sample["messages"]
        
        # Get user prompt (first message)
        user_prompt = messages[0]["content"] if messages else ""
        
        # Get reference response (assistant's response)
        reference_response = messages[1]["content"] if len(messages) > 1 else ""
        
        # Generate response
        response, inference_time, prompt_tokens, response_tokens = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        
        # Clean response
        response = response.strip()
        
        # Quality metrics
        response_length = len(response)
        has_content = len(response) > 10 and not response.isspace()
        
        if has_content:
            valid_responses += 1
        
        total_time += inference_time
        total_response_length += response_length
        
        result = EvalResult(
            sample_id=idx,
            user_prompt=user_prompt,
            model_response=response,
            reference_response=reference_response,
            inference_time=inference_time,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            response_length=response_length,
            has_content=has_content,
        )
        results.append(result)
    
    num_samples = len(sample_indices)
    avg_response_length = total_response_length / num_samples if num_samples else 0.0
    avg_inference_time = total_time / num_samples if num_samples else 0.0
    valid_response_rate = valid_responses / num_samples if num_samples else 0.0
    
    return ModelEvalSummary(
        model_name=model_name,
        num_samples=num_samples,
        avg_response_length=avg_response_length,
        avg_inference_time=avg_inference_time,
        total_time=total_time,
        valid_response_rate=valid_response_rate,
        results=results,
    )


def print_comparison(base_summary: ModelEvalSummary, sft_summary: ModelEvalSummary):
    """Print comparison results"""
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS: SFT Model vs Base Model (Capybara Test)")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Base Model':<20} {'SFT Model':<20} {'Change':<15}")
    print("-"*85)
    
    # Valid response rate
    rate_diff = sft_summary.valid_response_rate - base_summary.valid_response_rate
    print(f"{'Valid Response Rate':<30} {base_summary.valid_response_rate*100:.1f}%{'':<14} {sft_summary.valid_response_rate*100:.1f}%{'':<14} {rate_diff*100:+.1f}%")
    
    # Average response length
    len_diff = sft_summary.avg_response_length - base_summary.avg_response_length
    len_pct = (len_diff / base_summary.avg_response_length * 100) if base_summary.avg_response_length > 0 else 0
    print(f"{'Avg Response Length':<30} {base_summary.avg_response_length:.0f} chars{'':<8} {sft_summary.avg_response_length:.0f} chars{'':<8} {len_diff:+.0f} ({len_pct:+.1f}%)")
    
    # Inference time
    time_diff = sft_summary.avg_inference_time - base_summary.avg_inference_time
    time_pct = (time_diff / base_summary.avg_inference_time * 100) if base_summary.avg_inference_time > 0 else 0
    print(f"{'Avg Inference Time':<30} {base_summary.avg_inference_time:.3f}s{'':<13} {sft_summary.avg_inference_time:.3f}s{'':<13} {time_diff:+.3f}s ({time_pct:+.1f}%)")
    
    # Total time
    print(f"{'Total Eval Time':<30} {base_summary.total_time:.1f}s{'':<14} {sft_summary.total_time:.1f}s{'':<14}")
    
    # Number of samples
    print(f"{'Samples Evaluated':<30} {base_summary.num_samples}{'':<18} {sft_summary.num_samples}")
    
    print("\n" + "="*80)
    
    # Sample comparison
    print("\n📊 Response Quality Analysis")
    print("-"*80)
    
    sft_better = []  # SFT has content, Base doesn't
    base_better = []  # Base has content, SFT doesn't
    both_good = []
    both_bad = []
    
    for base_r, sft_r in zip(base_summary.results, sft_summary.results):
        if sft_r.has_content and not base_r.has_content:
            sft_better.append((base_r, sft_r))
        elif base_r.has_content and not sft_r.has_content:
            base_better.append((base_r, sft_r))
        elif base_r.has_content and sft_r.has_content:
            both_good.append((base_r, sft_r))
        else:
            both_bad.append((base_r, sft_r))
    
    print(f"\n✅ Both produced valid responses: {len(both_good)}")
    print(f"❌ Both produced invalid/empty responses: {len(both_bad)}")
    print(f"🎯 SFT better (SFT valid, Base invalid): {len(sft_better)}")
    print(f"📉 Base better (Base valid, SFT invalid): {len(base_better)}")
    
    # Show sample responses
    print("\n" + "="*80)
    print("📝 SAMPLE RESPONSES COMPARISON")
    print("="*80)
    
    # Show 3 random samples where both have content
    if both_good:
        samples_to_show = min(3, len(both_good))
        for i, (base_r, sft_r) in enumerate(random.sample(both_good, samples_to_show)):
            print(f"\n--- Sample {i+1} (ID: {sft_r.sample_id}) ---")
            print(f"📝 User Prompt: {sft_r.user_prompt[:150]}...")
            print(f"\n🔵 Base Response ({len(base_r.model_response)} chars):")
            print(f"   {base_r.model_response[:300]}...")
            print(f"\n🟢 SFT Response ({len(sft_r.model_response)} chars):")
            print(f"   {sft_r.model_response[:300]}...")
            print(f"\n📖 Reference ({len(sft_r.reference_response)} chars):")
            print(f"   {sft_r.reference_response[:300]}...")


def save_results(
    base_summary: ModelEvalSummary,
    sft_summary: ModelEvalSummary,
    output_path: str,
):
    """Save results to JSON file"""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "trl-lib/Capybara (test split)",
        "base_model": {
            "name": base_summary.model_name,
            "valid_response_rate": base_summary.valid_response_rate,
            "avg_response_length": base_summary.avg_response_length,
            "num_samples": base_summary.num_samples,
            "avg_inference_time": base_summary.avg_inference_time,
            "total_time": base_summary.total_time,
        },
        "sft_model": {
            "name": sft_summary.model_name,
            "valid_response_rate": sft_summary.valid_response_rate,
            "avg_response_length": sft_summary.avg_response_length,
            "num_samples": sft_summary.num_samples,
            "avg_inference_time": sft_summary.avg_inference_time,
            "total_time": sft_summary.total_time,
        },
        "comparison": {
            "valid_rate_diff": sft_summary.valid_response_rate - base_summary.valid_response_rate,
            "avg_length_diff": sft_summary.avg_response_length - base_summary.avg_response_length,
        },
        "detailed_results": {
            "base": [
                {
                    "sample_id": r.sample_id,
                    "has_content": r.has_content,
                    "response_length": r.response_length,
                    "inference_time": r.inference_time,
                    "user_prompt": r.user_prompt[:200],
                    "model_response": r.model_response[:500],
                }
                for r in base_summary.results
            ],
            "sft": [
                {
                    "sample_id": r.sample_id,
                    "has_content": r.has_content,
                    "response_length": r.response_length,
                    "inference_time": r.inference_time,
                    "user_prompt": r.user_prompt[:200],
                    "model_response": r.model_response[:500],
                }
                for r in sft_summary.results
            ],
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare SFT vs Base model on Capybara test dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate (max 200)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-5928", help="SFT checkpoint to use")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--sft_dir", type=str, default="./Qwen3-0.6B-SFT", help="SFT model directory")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Capybara test dataset
    print("\n📚 Loading trl-lib/Capybara test dataset...")
    dataset = load_dataset("trl-lib/Capybara", split="test")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Select sample indices
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"\n📊 Evaluating {num_samples} samples...")
    print(f"Sample indices: {sample_indices[:5]}... (showing first 5)")
    
    # Load and evaluate base model
    print("\n" + "="*60)
    print("Loading BASE model...")
    print("="*60)
    base_model, base_tokenizer = load_model_and_tokenizer(
        args.base_model,
        device=device,
        use_4bit=args.use_4bit,
    )
    
    base_summary = evaluate_model(
        model=base_model,
        tokenizer=base_tokenizer,
        dataset=dataset,
        sample_indices=sample_indices,
        model_name=f"Base ({args.base_model})",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # Load and evaluate SFT model
    print("\n" + "="*60)
    print("Loading SFT model...")
    print("="*60)
    sft_model_path = os.path.join(args.sft_dir, args.checkpoint)
    sft_model, sft_tokenizer = load_model_and_tokenizer(
        sft_model_path,
        device=device,
        use_4bit=args.use_4bit,
    )
    
    sft_summary = evaluate_model(
        model=sft_model,
        tokenizer=sft_tokenizer,
        dataset=dataset,
        sample_indices=sample_indices,
        model_name=f"SFT ({args.checkpoint})",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    
    # Print comparison
    print_comparison(base_summary, sft_summary)
    
    # Save results
    if args.output is None:
        args.output = f"capybara_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    save_results(base_summary, sft_summary, args.output)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
