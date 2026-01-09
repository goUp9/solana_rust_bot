#!/usr/bin/env python3
"""
Utility functions for PPO+LoRA training
"""

import logging
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
    
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    logger = logging.getLogger("game_rl_training")
    return logger


def save_checkpoint(
    model,
    tokenizer,
    save_path: Path,
    curriculum_state: Optional[Dict] = None,
    optimizer_state: Optional[Dict] = None,
    training_args: Optional[Dict] = None,
):
    """
    Save training checkpoint
    
    Args:
        model: Model to save (with LoRA adapters)
        tokenizer: Tokenizer
        save_path: Path to save checkpoint
        curriculum_state: Curriculum sampler state
        optimizer_state: Optimizer state dict
        training_args: Training arguments
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model (LoRA adapters)
    model.save_pretrained(save_path / "model")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path / "tokenizer")
    
    # Save metadata
    metadata = {
        "model_type": "ppo_lora",
        "base_model": getattr(model, "base_model_name", "unknown"),
    }
    
    if curriculum_state:
        metadata["curriculum_state"] = curriculum_state
    
    if optimizer_state:
        torch.save(optimizer_state, save_path / "optimizer.pt")
    
    if training_args:
        metadata["training_args"] = training_args
    
    with open(save_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Checkpoint saved to {save_path}")


def load_checkpoint(
    model,
    tokenizer,
    load_path: Path,
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        model: Model to load adapters into
        tokenizer: Tokenizer (unused, for compatibility)
        load_path: Path to load checkpoint from
    
    Returns:
        Dictionary with loaded metadata
    """
    load_path = Path(load_path)
    
    # Load LoRA adapters
    model.load_adapter(load_path / "model")
    
    # Load metadata
    metadata = {}
    if (load_path / "metadata.json").exists():
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
    
    # Load optimizer state if exists
    optimizer_path = load_path / "optimizer.pt"
    if optimizer_path.exists():
        metadata["optimizer_state"] = torch.load(optimizer_path)
    
    print(f"✅ Checkpoint loaded from {load_path}")
    
    return metadata


def count_trainable_parameters(model) -> Dict[str, int]:
    """
    Count trainable vs total parameters
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = 0
    all_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100 * trainable_params / all_params if all_params > 0 else 0,
    }


def print_training_summary(
    model,
    model_config: Dict,
    lora_config: Dict,
    ppo_config: Dict,
):
    """
    Print training configuration summary
    
    Args:
        model: Model instance
        model_config: Model configuration
        lora_config: LoRA configuration
        ppo_config: PPO configuration
    """
    param_stats = count_trainable_parameters(model)
    
    print("\n" + "="*60)
    print("🎮 Game RL Training Configuration")
    print("="*60)
    
    print("\n📊 Model:")
    print(f"  Name: {model_config.get('model_name', 'N/A')}")
    print(f"  Trainable params: {param_stats['trainable_params']:,}")
    print(f"  All params: {param_stats['all_params']:,}")
    print(f"  Trainable %: {param_stats['trainable_percentage']:.2f}%")
    
    print("\n🔧 LoRA:")
    print(f"  Rank (r): {lora_config.get('r', 'N/A')}")
    print(f"  Alpha: {lora_config.get('lora_alpha', 'N/A')}")
    print(f"  Dropout: {lora_config.get('lora_dropout', 'N/A')}")
    print(f"  Target modules: {', '.join(lora_config.get('target_modules', []))}")
    
    print("\n🎯 PPO:")
    print(f"  Batch size: {ppo_config.get('batch_size', 'N/A')}")
    print(f"  Mini-batch size: {ppo_config.get('mini_batch_size', 'N/A')}")
    print(f"  Learning rate: {ppo_config.get('learning_rate', 'N/A')}")
    print(f"  PPO epochs: {ppo_config.get('ppo_epochs', 'N/A')}")
    print(f"  KL coef: {ppo_config.get('init_kl_coef', 'N/A')}")
    print(f"  Training steps: {ppo_config.get('num_train_steps', 'N/A')}")
    
    print("\n" + "="*60 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = []
    returns = []
    
    gae = 0
    next_value = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]
    
    return advantages, returns


class EarlyStopping:
    """Early stopping helper"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping
        
        Args:
            patience: Number of steps to wait before stopping
            min_delta: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score (higher is better)
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class MovingAverage:
    """Moving average tracker"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize moving average
        
        Args:
            window_size: Size of the moving window
        """
        from collections import deque
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value: float):
        """Add a new value"""
        self.values.append(value)
    
    def get(self) -> float:
        """Get current moving average"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the tracker"""
        self.values.clear()
