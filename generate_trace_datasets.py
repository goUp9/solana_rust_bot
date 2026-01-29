#!/usr/bin/env python3
"""
Stage 0: Generate Training Datasets for Trace Environment

This script prepares all datasets needed for the 3-stage training pipeline:
1. Original dataset (warmup) - original code → output mapping
2. Transformed dataset (SFT) - code with injected prints → output mapping

The trace task involves:
- Reading Python code with injected debug print statements (__DBG_N__)
- Predicting the exact stdout output including all debug lines
- Binary evaluation: exact match = 1.0, otherwise = 0.0

Usage:
    # Generate all datasets
    python generate_trace_datasets.py
    
    # Generate with specific variants per sample
    python generate_trace_datasets.py --variants 5
    
    # Generate smaller dataset for testing
    python generate_trace_datasets.py --max_samples 1000
    
    # Use multiple processes for faster generation
    python generate_trace_datasets.py --num_workers 32
"""

import os
import sys
import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import multiprocessing as mp

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Add path for trace_task imports
TRACE_ENV_PATH = "/root/workstation/sn120/affine_repo/affinetes/environments/trace"
sys.path.insert(0, TRACE_ENV_PATH)

try:
    from trace_task import (
        inject_non_overfittable_prints,
        run_code_sync,
        clean_source,
    )
    TRACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import trace_task: {e}")
    print(f"Make sure {TRACE_ENV_PATH} exists and contains trace_task.py")
    TRACE_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Source dataset
    source_dataset: str = "satpalsr/rl-python"
    source_split: str = "train"
    
    # Output paths
    output_dir: str = "./datasets/trace_training"
    
    # Generation settings
    variants_per_sample: int = 3  # Number of transformed variants per original sample
    max_injections: int = 6  # Max debug prints to inject
    execution_timeout: int = 5  # Timeout for code execution (seconds)
    
    # Filtering
    max_program_chars: int = 10000  # Skip programs longer than this
    max_output_chars: int = 5000  # Skip samples with output longer than this
    min_program_lines: int = 3  # Skip programs shorter than this
    
    # Processing
    num_workers: int = 16  # Parallel workers for generation
    max_samples: Optional[int] = None  # Limit samples (None = all)
    
    # Train/test split
    test_ratio: float = 0.05  # 5% for test set
    random_seed: int = 42


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def is_valid_program(program: str, config: DatasetConfig) -> Tuple[bool, str]:
    """Check if a program is valid for training."""
    if not program or not program.strip():
        return False, "empty_program"
    
    lines = program.strip().split('\n')
    if len(lines) < config.min_program_lines:
        return False, "too_short"
    
    if len(program) > config.max_program_chars:
        return False, "too_long"
    
    # Check for syntax errors
    try:
        compile(clean_source(program), '<string>', 'exec')
    except SyntaxError:
        return False, "syntax_error"
    
    return True, "valid"


def generate_original_sample(
    idx: int,
    program: str,
    inputs: str,
    expected_output: str,
    config: DatasetConfig,
) -> Optional[Dict[str, Any]]:
    """Generate a training sample from original code (Stage 1 warmup)."""
    
    # Validate program
    is_valid, reason = is_valid_program(program, config)
    if not is_valid:
        return None
    
    # Execute original code to verify output
    stdout, stderr = run_code_sync(
        clean_source(program), 
        input_data=inputs, 
        timeout=config.execution_timeout
    )
    
    if stderr:
        return None
    
    actual_output = stdout.strip()
    if not actual_output:
        return None
    
    if len(actual_output) > config.max_output_chars:
        return None
    
    # Create chat format
    user_content = f"""Predict the exact standard output (stdout) of the following Python program.

Program:
```python
{clean_source(program)}
```

Input (stdin):
```
{inputs}
```

Provide only the exact output, no explanations."""

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": actual_output}
    ]
    
    return {
        "messages": messages,
        "metadata": {
            "dataset_index": idx,
            "stage": "original",
            "program_lines": len(program.strip().split('\n')),
            "output_length": len(actual_output),
        }
    }


def generate_transformed_sample(
    idx: int,
    program: str,
    inputs: str,
    seed: int,
    config: DatasetConfig,
) -> Optional[Dict[str, Any]]:
    """Generate a training sample with injected debug prints (Stage 2 SFT)."""
    
    # Validate program
    is_valid, reason = is_valid_program(program, config)
    if not is_valid:
        return None
    
    try:
        # Transform code with debug print injection
        transformed = inject_non_overfittable_prints(
            program, 
            seed=seed, 
            max_injections=config.max_injections
        )
        
        # Check if transformation added syntax error marker
        if transformed.startswith("# Syntax Error"):
            return None
        
        # Execute transformed code
        stdout, stderr = run_code_sync(
            transformed, 
            input_data=inputs, 
            timeout=config.execution_timeout
        )
        
        if stderr:
            return None
        
        ground_truth = stdout.strip()
        if not ground_truth:
            return None
        
        if len(ground_truth) > config.max_output_chars:
            return None
        
        # Verify that debug prints were actually injected
        if "__DBG_" not in transformed:
            return None
        
        # Create chat format matching trace environment prompt
        user_content = f"""Predict the exact and complete standard output (stdout) of the following Python program, including every single print statement.

The program contains several injected debug print statements starting with '__DBG_'. You must include these in your prediction exactly as they would appear in the output, along with any other output the program produces.

Program:
```python
{transformed}
```

Input (stdin):
```
{inputs}
```

Provide the full stdout content. Do not provide any explanations or commentary outside of the predicted output."""

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ground_truth}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "dataset_index": idx,
                "seed": seed,
                "stage": "transformed",
                "program_lines": len(transformed.strip().split('\n')),
                "output_length": len(ground_truth),
                "num_dbg_prints": transformed.count("__DBG_"),
            }
        }
        
    except Exception as e:
        return None


def process_sample_batch(args: Tuple) -> List[Dict[str, Any]]:
    """Process a batch of samples (for parallel processing)."""
    batch_indices, dataset_samples, config, stage, variant_offset = args
    
    results = []
    
    for idx in batch_indices:
        sample = dataset_samples[idx]
        program = sample.get("program", "")
        inputs = sample.get("inputs", "")
        expected_output = sample.get("output", "")
        
        if stage == "original":
            result = generate_original_sample(
                idx, program, inputs, expected_output, config
            )
            if result:
                results.append(result)
        
        elif stage == "transformed":
            for variant in range(config.variants_per_sample):
                seed = idx * 1000 + variant + variant_offset
                result = generate_transformed_sample(
                    idx, program, inputs, seed, config
                )
                if result:
                    results.append(result)
    
    return results


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_original_dataset(
    source_dataset: Dataset,
    config: DatasetConfig,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Generate Stage 1: Original warmup dataset."""
    
    print("\n" + "=" * 70)
    print("📚 Stage 1: Generating Original Warmup Dataset")
    print("=" * 70)
    
    samples = []
    stats = {
        "total": 0,
        "valid": 0,
        "empty_program": 0,
        "too_short": 0,
        "too_long": 0,
        "syntax_error": 0,
        "execution_error": 0,
        "empty_output": 0,
    }
    
    indices = list(range(len(source_dataset)))
    if config.max_samples:
        indices = indices[:config.max_samples]
    
    # Process in parallel batches
    batch_size = 100
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    
    # Convert to list for multiprocessing
    dataset_list = [source_dataset[i] for i in indices]
    
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []
        for batch_idx, batch in enumerate(batches):
            # Remap indices to local list
            local_batch = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(indices))))
            args = (local_batch, dataset_list, config, "original", 0)
            futures.append(executor.submit(process_sample_batch, args))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            batch_results = future.result()
            samples.extend(batch_results)
    
    stats["total"] = len(indices)
    stats["valid"] = len(samples)
    
    print(f"\n📊 Original Dataset Statistics:")
    print(f"   Total processed: {stats['total']:,}")
    print(f"   Valid samples:   {stats['valid']:,} ({100*stats['valid']/stats['total']:.1f}%)")
    
    return samples, stats


def generate_transformed_dataset(
    source_dataset: Dataset,
    config: DatasetConfig,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Generate Stage 2: Transformed SFT dataset with debug print injection."""
    
    print("\n" + "=" * 70)
    print("🔧 Stage 2: Generating Transformed Dataset with Debug Prints")
    print(f"   Variants per sample: {config.variants_per_sample}")
    print("=" * 70)
    
    samples = []
    stats = {
        "total_source": 0,
        "total_variants": 0,
        "valid": 0,
        "failed": 0,
    }
    
    indices = list(range(len(source_dataset)))
    if config.max_samples:
        indices = indices[:config.max_samples]
    
    stats["total_source"] = len(indices)
    stats["total_variants"] = len(indices) * config.variants_per_sample
    
    # Process in parallel batches
    batch_size = 50  # Smaller batches for transformed (more work per sample)
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    
    # Convert to list for multiprocessing
    dataset_list = [source_dataset[i] for i in indices]
    
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []
        for batch_idx, batch in enumerate(batches):
            local_batch = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(indices))))
            args = (local_batch, dataset_list, config, "transformed", 0)
            futures.append(executor.submit(process_sample_batch, args))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            batch_results = future.result()
            samples.extend(batch_results)
    
    stats["valid"] = len(samples)
    stats["failed"] = stats["total_variants"] - stats["valid"]
    
    print(f"\n📊 Transformed Dataset Statistics:")
    print(f"   Source samples:    {stats['total_source']:,}")
    print(f"   Total variants:    {stats['total_variants']:,}")
    print(f"   Valid samples:     {stats['valid']:,} ({100*stats['valid']/stats['total_variants']:.1f}%)")
    print(f"   Failed:            {stats['failed']:,}")
    
    return samples, stats


def split_train_test(
    samples: List[Dict],
    test_ratio: float,
    random_seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train and test sets."""
    random.seed(random_seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]
    
    return train, test


def save_dataset(
    samples: List[Dict],
    output_path: str,
    dataset_name: str,
):
    """Save dataset to disk in multiple formats."""
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL (most flexible)
    jsonl_path = output_dir / f"{dataset_name}.jsonl"
    with open(jsonl_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Saved JSONL: {jsonl_path} ({len(samples):,} samples)")
    
    # Save as HuggingFace Dataset
    try:
        # Extract just messages for HF dataset
        hf_samples = [{"messages": s["messages"]} for s in samples]
        hf_dataset = Dataset.from_list(hf_samples)
        hf_path = output_dir / f"{dataset_name}_hf"
        hf_dataset.save_to_disk(str(hf_path))
        print(f"   Saved HF Dataset: {hf_path}")
    except Exception as e:
        print(f"   Warning: Could not save HF dataset: {e}")
    
    return jsonl_path


def main(args):
    """Main dataset generation pipeline."""
    
    if not TRACE_AVAILABLE:
        print("ERROR: trace_task module not available. Cannot generate datasets.")
        print(f"Please ensure {TRACE_ENV_PATH}/trace_task.py exists.")
        sys.exit(1)
    
    # Create config
    config = DatasetConfig(
        output_dir=args.output_dir,
        variants_per_sample=args.variants,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    
    print("\n" + "=" * 70)
    print("🚀 Trace Training Dataset Generator")
    print("=" * 70)
    print(f"Source dataset:     {config.source_dataset}")
    print(f"Output directory:   {config.output_dir}")
    print(f"Variants/sample:    {config.variants_per_sample}")
    print(f"Max samples:        {config.max_samples or 'all'}")
    print(f"Workers:            {config.num_workers}")
    print(f"Test ratio:         {config.test_ratio}")
    print("=" * 70)
    
    # Load source dataset
    print("\n📥 Loading source dataset...")
    source_dataset = load_dataset(config.source_dataset, split=config.source_split)
    print(f"   Loaded {len(source_dataset):,} samples")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Stage 1: Original Dataset (Warmup)
    # =========================================================================
    
    if not args.skip_original:
        original_samples, original_stats = generate_original_dataset(source_dataset, config)
        
        # Split train/test
        original_train, original_test = split_train_test(
            original_samples, config.test_ratio, config.random_seed
        )
        
        # Save
        print("\n💾 Saving Original Dataset...")
        save_dataset(original_train, config.output_dir, "stage1_original_train")
        save_dataset(original_test, config.output_dir, "stage1_original_test")
        
        # Save stats
        with open(output_dir / "stage1_stats.json", 'w') as f:
            json.dump({
                "config": asdict(config),
                "stats": original_stats,
                "train_samples": len(original_train),
                "test_samples": len(original_test),
            }, f, indent=2)
    
    # =========================================================================
    # Stage 2: Transformed Dataset (SFT)
    # =========================================================================
    
    if not args.skip_transformed:
        transformed_samples, transformed_stats = generate_transformed_dataset(source_dataset, config)
        
        # Split train/test
        transformed_train, transformed_test = split_train_test(
            transformed_samples, config.test_ratio, config.random_seed
        )
        
        # Save
        print("\n💾 Saving Transformed Dataset...")
        save_dataset(transformed_train, config.output_dir, "stage2_transformed_train")
        save_dataset(transformed_test, config.output_dir, "stage2_transformed_test")
        
        # Save stats
        with open(output_dir / "stage2_stats.json", 'w') as f:
            json.dump({
                "config": asdict(config),
                "stats": transformed_stats,
                "train_samples": len(transformed_train),
                "test_samples": len(transformed_test),
            }, f, indent=2)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("✅ Dataset Generation Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {config.output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        if f.is_file():
            print(f"   {f.name}: {size_mb:.1f} MB")
        else:
            print(f"   {f.name}/ (directory)")
    
    print("\n📋 Next steps:")
    print("   1. Run Stage 1 training: python train_stage1_warmup.py")
    print("   2. Run Stage 2 training: python train_stage2_sft.py")
    print("   3. Run Stage 3 RL:       python train_trace_ppo.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training datasets for Trace environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all datasets with default settings
    python generate_trace_datasets.py
    
    # Generate with more variants (better diversity)
    python generate_trace_datasets.py --variants 5
    
    # Quick test with limited samples
    python generate_trace_datasets.py --max_samples 1000 --variants 2
    
    # Use more workers on B200 (192 cores)
    python generate_trace_datasets.py --num_workers 64
    
    # Only generate transformed dataset (skip warmup)
    python generate_trace_datasets.py --skip_original
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/trace_training",
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Number of transformed variants per original sample (default: 3)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of source samples (default: all)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Ratio of samples for test set (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="Skip generating original (warmup) dataset"
    )
    parser.add_argument(
        "--skip_transformed",
        action="store_true",
        help="Skip generating transformed (SFT) dataset"
    )
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    main(args)
