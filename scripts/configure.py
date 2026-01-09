#!/usr/bin/env python3
"""
Interactive configuration wizard for training setup
"""

import json
from pathlib import Path


def get_input(prompt, default, type_cast=str):
    """Get user input with default value"""
    value = input(f"{prompt} [{default}]: ").strip()
    if not value:
        return default
    return type_cast(value)


def get_bool(prompt, default):
    """Get boolean input"""
    value = input(f"{prompt} [{'y' if default else 'n'}]: ").strip().lower()
    if not value:
        return default
    return value in ['y', 'yes', 'true', '1']


def main():
    print("="*60)
    print("🎮 Game RL Training Configuration Wizard")
    print("="*60)
    print("\nThis wizard will help you configure your training setup.")
    print("Press Enter to use default values.\n")
    
    # Model configuration
    print("\n📊 Model Configuration")
    print("-" * 60)
    
    model_name = get_input(
        "Model name",
        "Qwen/Qwen3-4B"
    )
    
    use_4bit = get_bool(
        "Use 4-bit quantization (recommended for <24GB VRAM)",
        True
    )
    
    # LoRA configuration
    print("\n🔧 LoRA Configuration")
    print("-" * 60)
    
    lora_r = get_input(
        "LoRA rank (higher = more capacity, slower)",
        16,
        int
    )
    
    lora_alpha = get_input(
        "LoRA alpha (typically 2x rank)",
        lora_r * 2,
        int
    )
    
    # PPO configuration
    print("\n🎯 PPO Configuration")
    print("-" * 60)
    
    batch_size = get_input(
        "Batch size (reduce if OOM)",
        8,
        int
    )
    
    learning_rate = get_input(
        "Learning rate",
        "1e-5"
    )
    learning_rate = float(learning_rate)
    
    num_train_steps = get_input(
        "Number of training steps",
        10000,
        int
    )
    
    # Wandb configuration
    print("\n📈 Experiment Tracking")
    print("-" * 60)
    
    use_wandb = get_bool(
        "Enable Weights & Biases tracking",
        True
    )
    
    wandb_project = None
    if use_wandb:
        wandb_project = get_input(
            "Wandb project name",
            "game-rl-training"
        )
    
    # Environment configuration
    print("\n🌍 Environment Configuration")
    print("-" * 60)
    
    env_mode = get_input(
        "Environment mode (docker/basilica)",
        "basilica"
    )
    
    # Build configuration
    config = {
        "model": {
            "model_name": model_name,
            "use_4bit": use_4bit,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "use_nested_quant": True
        },
        "lora": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "ppo": {
            "batch_size": batch_size,
            "mini_batch_size": max(1, batch_size // 4),
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "ppo_epochs": 4,
            "init_kl_coef": 0.2,
            "target_kl": 6.0,
            "adap_kl_ctrl": True,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.1,
            "num_train_steps": num_train_steps,
            "save_freq": max(100, num_train_steps // 20),
            "eval_freq": max(50, num_train_steps // 100),
            "log_freq": 10,
            "max_seq_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "output_dir": "./checkpoints/game_ppo_lora",
            "resume_from": None,
            "use_wandb": use_wandb,
            "wandb_project": wandb_project,
            "wandb_run_name": None
        }
    }
    
    # Save configuration
    print("\n💾 Saving Configuration")
    print("-" * 60)
    
    config_path = Path("config/train_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_path}")
    
    # Print summary
    print("\n📋 Configuration Summary")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Quantization: {'4-bit' if use_4bit else 'Full precision'}")
    print(f"LoRA rank: {lora_r}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training steps: {num_train_steps}")
    print(f"Wandb: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"Environment: {env_mode}")
    print("="*60)
    
    print("\n✨ Ready to train!")
    print("\nTo start training, run:")
    print("  python train_ppo_lora.py")
    print()


if __name__ == "__main__":
    main()
