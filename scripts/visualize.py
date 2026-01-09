#!/usr/bin/env python3
"""
Visualize training progress and curriculum state
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_training_progress(checkpoint_dir: str):
    """
    Visualize training progress from checkpoint metadata
    
    Args:
        checkpoint_dir: Path to checkpoints directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Collect all checkpoint metadata
    checkpoints = []
    for checkpoint_path in sorted(checkpoint_dir.glob("checkpoint-*")):
        metadata_file = checkpoint_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                checkpoints.append({
                    "step": int(checkpoint_path.name.split("-")[1]),
                    "curriculum": metadata.get("curriculum_state", {}),
                })
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Extract data
    steps = [c["step"] for c in checkpoints]
    stages = [c["curriculum"].get("current_stage_idx", 0) for c in checkpoints]
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot curriculum stage progression
    axes[0].plot(steps, stages, marker='o', linewidth=2)
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Curriculum Stage")
    axes[0].set_title("Curriculum Progression")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Easy", "Medium", "Hard"])
    
    # Plot success rate over time (if available)
    success_rates = []
    for c in checkpoints:
        curr_state = c["curriculum"]
        if "buffer_stats" in curr_state:
            sr = curr_state["buffer_stats"].get("overall_success_rate", 0)
            success_rates.append(sr)
        else:
            success_rates.append(0)
    
    axes[1].plot(steps, success_rates, marker='o', linewidth=2, color='green')
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Overall Success Rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = checkpoint_dir / "training_progress.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training progress plot saved to: {output_path}")
    
    plt.show()


def print_curriculum_summary(checkpoint_dir: str):
    """Print summary of curriculum state"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    latest = checkpoints[-1]
    metadata_file = latest / "metadata.json"
    
    if not metadata_file.exists():
        print("No metadata found!")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    curr_state = metadata.get("curriculum_state", {})
    
    print("\n" + "="*60)
    print("📊 Curriculum Summary")
    print("="*60)
    print(f"Checkpoint: {latest.name}")
    print(f"Current Stage: {curr_state.get('current_stage_idx', 0)}")
    print(f"Stage Changes: {curr_state.get('stats', {}).get('stage_changes', 0)}")
    print(f"Total Samples: {curr_state.get('stats', {}).get('total_samples', 0)}")
    print(f"Failure Replays: {curr_state.get('stats', {}).get('failure_replays', 0)}")
    
    if "buffer_stats" in curr_state:
        buf_stats = curr_state["buffer_stats"]
        print(f"\nBuffer Size: {buf_stats.get('buffer_size', 0)}")
        print(f"Unique Tasks: {buf_stats.get('unique_tasks', 0)}")
        print(f"Success Rate: {buf_stats.get('overall_success_rate', 0):.2%}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/game_ppo_lora",
        help="Path to checkpoints directory"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print curriculum summary only (no plot)"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_curriculum_summary(args.checkpoint_dir)
    else:
        print_curriculum_summary(args.checkpoint_dir)
        plot_training_progress(args.checkpoint_dir)


if __name__ == "__main__":
    main()
