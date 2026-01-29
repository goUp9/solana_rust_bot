#!/usr/bin/env python3
"""
Full Training Pipeline for Trace Environment

This script orchestrates the complete 3-stage training pipeline:
  Stage 0: Data generation (prepare datasets)
  Stage 1: Original warmup (basic Python execution)
  Stage 2: Transformed SFT (debug print prediction)
  Stage 3: RL/PPO training (online learning)

Usage:
    # Run full pipeline with default settings
    python run_full_pipeline.py
    
    # Run full pipeline on B200
    python run_full_pipeline.py --b200
    
    # Skip specific stages
    python run_full_pipeline.py --skip_stage0 --skip_stage1
    
    # Quick test (limited samples)
    python run_full_pipeline.py --quick_test
    
    # Resume from specific stage
    python run_full_pipeline.py --start_stage 2
    
    # Custom configuration
    python run_full_pipeline.py --config pipeline_config.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "model": {
        "base_model": "Qwen/Qwen3-4B",
    },
    "hardware": {
        "b200_mode": True,
        "no_4bit": False,
    },
    "stage0": {
        "enabled": True,
        "variants_per_sample": 3,
        "max_samples": None,
        "num_workers": 32,
        "output_dir": "./datasets/trace_training",
    },
    "stage1": {
        "enabled": True,
        "epochs": 2,
        "output_dir": "./checkpoints/stage1_warmup",
    },
    "stage2": {
        "enabled": True,
        "epochs": 3,
        "output_dir": "./checkpoints/stage2_sft",
    },
    "stage3": {
        "enabled": True,
        "num_steps": 5000,
        "output_dir": "./checkpoints/stage3_ppo",
    },
    "quick_test": {
        "stage0_max_samples": 500,
        "stage1_epochs": 1,
        "stage2_epochs": 1,
        "stage2_max_samples": 1000,
        "stage3_num_steps": 100,
    },
}


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_command(cmd: list, description: str, cwd: str = None) -> bool:
    """Run a command and stream output."""
    
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        elapsed = time.time() - start
        
        if process.returncode == 0:
            print(f"\n✅ {description} completed in {format_time(elapsed)}")
            return True
        else:
            print(f"\n❌ {description} failed with code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running command: {e}")
        return False


def run_stage0(config: Dict[str, Any], quick_test: bool = False) -> bool:
    """Run Stage 0: Data Generation."""
    
    stage_config = config["stage0"]
    
    cmd = [
        sys.executable, "generate_trace_datasets.py",
        "--output_dir", stage_config["output_dir"],
        "--variants", str(stage_config["variants_per_sample"]),
        "--num_workers", str(stage_config.get("num_workers", 16)),
    ]
    
    max_samples = stage_config.get("max_samples")
    if quick_test:
        max_samples = config["quick_test"]["stage0_max_samples"]
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    return run_command(cmd, "Stage 0: Data Generation")


def run_stage1(config: Dict[str, Any], quick_test: bool = False) -> bool:
    """Run Stage 1: Original Warmup Training."""
    
    stage_config = config["stage1"]
    hw_config = config["hardware"]
    
    cmd = [
        sys.executable, "train_stage1_warmup.py",
        "--model", config["model"]["base_model"],
        "--output_dir", stage_config["output_dir"],
        "--dataset", f"{config['stage0']['output_dir']}/stage1_original_train.jsonl",
        "--eval_dataset", f"{config['stage0']['output_dir']}/stage1_original_test.jsonl",
    ]
    
    epochs = stage_config["epochs"]
    if quick_test:
        epochs = config["quick_test"]["stage1_epochs"]
    cmd.extend(["--epochs", str(epochs)])
    
    if hw_config.get("b200_mode"):
        cmd.append("--b200")
    
    if hw_config.get("no_4bit"):
        cmd.append("--no_4bit")
    
    return run_command(cmd, "Stage 1: Original Warmup Training")


def run_stage2(config: Dict[str, Any], quick_test: bool = False) -> bool:
    """Run Stage 2: Transformed SFT Training."""
    
    stage_config = config["stage2"]
    hw_config = config["hardware"]
    
    # Use Stage 1 output as base model if it exists
    stage1_merged = Path(config["stage1"]["output_dir"]) / "merged"
    if stage1_merged.exists():
        base_model = str(stage1_merged)
    else:
        base_model = config["model"]["base_model"]
    
    cmd = [
        sys.executable, "train_stage2_sft.py",
        "--base_model", base_model,
        "--output_dir", stage_config["output_dir"],
        "--dataset", f"{config['stage0']['output_dir']}/stage2_transformed_train.jsonl",
        "--eval_dataset", f"{config['stage0']['output_dir']}/stage2_transformed_test.jsonl",
    ]
    
    epochs = stage_config["epochs"]
    if quick_test:
        epochs = config["quick_test"]["stage2_epochs"]
        cmd.extend(["--max_samples", str(config["quick_test"]["stage2_max_samples"])])
    cmd.extend(["--epochs", str(epochs)])
    
    if hw_config.get("b200_mode"):
        cmd.append("--b200")
    
    if hw_config.get("no_4bit"):
        cmd.append("--no_4bit")
    
    return run_command(cmd, "Stage 2: Transformed SFT Training")


def run_stage3(config: Dict[str, Any], quick_test: bool = False) -> bool:
    """Run Stage 3: PPO/RL Training."""
    
    stage_config = config["stage3"]
    
    # Use Stage 2 output as base model if it exists
    stage2_merged = Path(config["stage2"]["output_dir"]) / "merged"
    if stage2_merged.exists():
        base_model = str(stage2_merged)
    else:
        # Fall back to Stage 1 or base model
        stage1_merged = Path(config["stage1"]["output_dir"]) / "merged"
        if stage1_merged.exists():
            base_model = str(stage1_merged)
        else:
            base_model = config["model"]["base_model"]
    
    cmd = [
        sys.executable, "train_stage3_ppo.py",
        "--base_model", base_model,
        "--output_dir", stage_config["output_dir"],
    ]
    
    num_steps = stage_config["num_steps"]
    if quick_test:
        num_steps = config["quick_test"]["stage3_num_steps"]
        cmd.extend(["--eval_freq", "10", "--save_freq", "50"])
    cmd.extend(["--num_steps", str(num_steps)])
    
    return run_command(cmd, "Stage 3: PPO/RL Training")


def main(args):
    """Main pipeline orchestration."""
    
    print("\n" + "=" * 70)
    print("🎯 Trace Environment Full Training Pipeline")
    print("=" * 70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"   Config:  {args.config}")
    else:
        config = DEFAULT_CONFIG.copy()
        print("   Config:  default")
    
    # Apply command line overrides
    if args.b200:
        config["hardware"]["b200_mode"] = True
    if args.no_4bit:
        config["hardware"]["no_4bit"] = True
    
    # Determine which stages to run
    stages_to_run = []
    
    start_stage = args.start_stage or 0
    
    if start_stage <= 0 and not args.skip_stage0 and config["stage0"]["enabled"]:
        stages_to_run.append(0)
    if start_stage <= 1 and not args.skip_stage1 and config["stage1"]["enabled"]:
        stages_to_run.append(1)
    if start_stage <= 2 and not args.skip_stage2 and config["stage2"]["enabled"]:
        stages_to_run.append(2)
    if start_stage <= 3 and not args.skip_stage3 and config["stage3"]["enabled"]:
        stages_to_run.append(3)
    
    print(f"   Stages:  {stages_to_run}")
    print(f"   B200:    {config['hardware']['b200_mode']}")
    print(f"   Quick:   {args.quick_test}")
    print("=" * 70)
    
    # Create output directories
    for stage_key in ["stage0", "stage1", "stage2", "stage3"]:
        output_dir = config[stage_key].get("output_dir")
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run stages
    pipeline_start = time.time()
    stage_times = {}
    
    for stage in stages_to_run:
        stage_start = time.time()
        
        if stage == 0:
            success = run_stage0(config, args.quick_test)
        elif stage == 1:
            success = run_stage1(config, args.quick_test)
        elif stage == 2:
            success = run_stage2(config, args.quick_test)
        elif stage == 3:
            success = run_stage3(config, args.quick_test)
        else:
            success = False
        
        stage_times[stage] = time.time() - stage_start
        
        if not success:
            print(f"\n❌ Pipeline failed at Stage {stage}")
            print("   Fix the error and resume with:")
            print(f"   python run_full_pipeline.py --start_stage {stage}")
            sys.exit(1)
    
    # Summary
    total_time = time.time() - pipeline_start
    
    print("\n" + "=" * 70)
    print("🎉 Pipeline Complete!")
    print("=" * 70)
    print("\n📊 Stage Times:")
    for stage, elapsed in stage_times.items():
        print(f"   Stage {stage}: {format_time(elapsed)}")
    print(f"\n   Total:   {format_time(total_time)}")
    
    print("\n📁 Output Locations:")
    if 0 in stages_to_run:
        print(f"   Data:    {config['stage0']['output_dir']}")
    if 1 in stages_to_run:
        print(f"   Stage 1: {config['stage1']['output_dir']}")
    if 2 in stages_to_run:
        print(f"   Stage 2: {config['stage2']['output_dir']}")
    if 3 in stages_to_run:
        print(f"   Stage 3: {config['stage3']['output_dir']}")
    
    # Final model location
    final_model = None
    for stage in [3, 2, 1]:
        if stage in stages_to_run:
            merged_path = Path(config[f"stage{stage}"]["output_dir"]) / "merged"
            if merged_path.exists():
                final_model = merged_path
                break
    
    if final_model:
        print(f"\n🏆 Final Model: {final_model}")
        print("\n📋 Next Steps:")
        print(f"   1. Evaluate: python eval_trace_sft.py --model {final_model}")
        print(f"   2. Serve:    python serve_sft_model.py --model {final_model}")
        print(f"   3. Deploy:   Upload to HuggingFace Hub")
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Training Pipeline for Trace Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with B200 optimizations
    python run_full_pipeline.py --b200
    
    # Quick test run
    python run_full_pipeline.py --quick_test
    
    # Skip data generation (if already done)
    python run_full_pipeline.py --skip_stage0
    
    # Start from Stage 2 (if Stage 0 and 1 done)
    python run_full_pipeline.py --start_stage 2
    
    # Only run RL stage
    python run_full_pipeline.py --skip_stage0 --skip_stage1 --skip_stage2
        """
    )
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to pipeline configuration JSON")
    parser.add_argument("--b200", action="store_true",
                        help="Enable B200 optimizations")
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test with limited samples")
    parser.add_argument("--start_stage", type=int, default=None,
                        help="Start from specific stage (0-3)")
    parser.add_argument("--skip_stage0", action="store_true",
                        help="Skip Stage 0 (data generation)")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 (warmup)")
    parser.add_argument("--skip_stage2", action="store_true",
                        help="Skip Stage 2 (SFT)")
    parser.add_argument("--skip_stage3", action="store_true",
                        help="Skip Stage 3 (PPO)")
    
    args = parser.parse_args()
    main(args)
