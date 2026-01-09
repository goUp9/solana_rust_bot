# Game Environment RL Training with PPO + LoRA

Complete guide for training **Qwen3-4B** using **PPO + LoRA** on OpenSpiel games with curriculum learning and failure-based sampling.

## 🎯 Overview

This training pipeline implements:
- **Model**: Qwen3-4B
- **Training Algorithm**: PPO (Proximal Policy Optimization)
- **Parameter-Efficient Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Framework**: TRL + PEFT + Transformers
- **Environment**: Local OpenSpiel (in-process)
- **Sampling Strategy**: Curriculum learning + failure-based replay
- **Rules**: Enforced by environment, not by model

## 📁 Project Structure

```
game_rl_training/
├── train_ppo_lora.py         # Main training script
├── env_wrapper.py             # Environment wrapper for RL
├── curriculum.py              # Curriculum and failure-based sampling
├── utils.py                   # Utility functions
├── config/
│   ├── train_config.json      # Training hyperparameters
│   ├── curriculum_config.json # Curriculum stages
│   └── env_config.json        # Environment settings
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to the training directory
cd /root/workstation/sn120/stage_4/game_rl_training

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

### 2. Environment Setup

Set up required environment variables:

```bash
# Optional: Weights & Biases for experiment tracking
export WANDB_API_KEY="your-wandb-key"

# Optional: Hugging Face token (if using gated models)
export HF_TOKEN="your-hf-token"
```

### 3. Configuration

Edit configuration files in `config/` directory:

#### `train_config.json` - Training Hyperparameters
```json
{
  "model": {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "use_4bit": true  // Use 4-bit quantization for memory efficiency
  },
  "lora": {
    "r": 16,  // LoRA rank
    "lora_alpha": 32  // LoRA scaling
  },
  "ppo": {
    "batch_size": 8,
    "learning_rate": 1e-5,
    "num_train_steps": 10000
  }
}
```

#### `curriculum_config.json` - Curriculum Stages
```json
{
  "curriculum_stages": [
    {
      "name": "easy",
      "task_range": [0, 999999999],
      "opponent": "random"
    },
    {
      "name": "medium",
      "task_range": [1000000000, 1999999999],
      "opponent": "random"
    },
    {
      "name": "hard",
      "task_range": [2000000000, 2999999999],
      "opponent": "mcts"
    }
  ]
}
```

### 4. Start Training

```bash
# Basic training
python train_ppo_lora.py

# Resume from checkpoint
python train_ppo_lora.py --resume_from ./checkpoints/game_ppo_lora/checkpoint-5000
```

## 🎓 Training Approach

### PPO (Proximal Policy Optimization)

PPO is a policy gradient method that:
1. Collects rollouts from the environment
2. Computes advantages using the value function
3. Updates the policy with clipped objective
4. Maintains KL divergence constraint with reference model

**Key advantages:**
- Stable training
- Sample efficient
- Works well with discrete and continuous actions

### LoRA (Low-Rank Adaptation)

LoRA enables efficient fine-tuning by:
1. Freezing the base model weights
2. Adding trainable low-rank matrices to attention layers
3. Reducing trainable parameters by >99%

**Benefits:**
- **Memory efficient**: Can train 3B model on single GPU
- **Fast**: Only updates ~0.5% of parameters
- **Modular**: Can save/load adapters separately

### Curriculum Learning

Training progresses through stages:

1. **Easy Stage** (Random opponent, games 0-999M)
   - Learn basic game mechanics
   - Success threshold: 60%

2. **Medium Stage** (Random opponent, games 1000M-1999M)
   - Learn strategic play
   - Success threshold: 50%

3. **Hard Stage** (MCTS opponent, games 2000M-2999M)
   - Master advanced tactics
   - Success threshold: 40%

**Automatic progression**: Advances to next stage when success rate meets threshold.

### Failure-Based Sampling

Maintains a replay buffer of failed/low-scoring tasks:
- **Priority sampling**: Failed tasks sampled more frequently
- **Mixed strategy**: 30% from buffer, 70% from curriculum
- **Continuous learning**: Revisits difficult tasks until mastered

## 📊 Monitoring Training

### Weights & Biases

If `use_wandb: true` in config:

```bash
# View training metrics at:
https://wandb.ai/your-username/game-rl-training
```

**Tracked metrics:**
- Mean reward per episode
- Success rate
- PPO loss, value loss, policy loss
- KL divergence
- Curriculum stage
- Failure buffer statistics

### Console Logs

```
Step 100 | Reward: 0.4523 | Score: 0.6012 | Success: 45.00%
Step 200 | Reward: 0.5234 | Score: 0.6891 | Success: 52.00%
🎓 Curriculum Progression: Advanced to stage 2/3
   Stage: medium | Opponent: random
   Success rate: 71.23%
```

## 💾 Checkpoints

Checkpoints are saved every `save_freq` steps (default: 500):

```
checkpoints/game_ppo_lora/
├── checkpoint-500/
│   ├── model/              # LoRA adapters
│   ├── tokenizer/          # Tokenizer config
│   ├── optimizer.pt        # Optimizer state
│   └── metadata.json       # Training metadata & curriculum state
├── checkpoint-1000/
└── final/                  # Final checkpoint
```

**Resume training:**
```bash
python train_ppo_lora.py --resume_from ./checkpoints/game_ppo_lora/checkpoint-5000
```

## 🎮 Environment Integration

### Task-ID Format

The game environment uses 12-digit task IDs:
```
task_id = GGGGCCCCCCCC
├─ GGGG: Game index (0-9999)
└─ CCCCCCCC: Configuration variant (0-99999999)
```

**Examples:**
- `0`: Leduc Poker, default config
- `100000002`: Liar's Dice, 3 dice per player
- `1100000000`: Hearts (4-player game)

### Supported Games

20 games including:
- **Perfect information**: Tic-Tac-Toe variants, Hex, Chinese Checkers
- **Imperfect information**: Poker variants, Hanabi, Battleship
- **Multi-player**: Hearts, Euchre, Cribbage
- **Negotiation**: Bargaining, Trade Communication

### Rule Enforcement

**All game rules are enforced by the environment**, not the model:
- Invalid actions are rejected
- Retry mechanism with error feedback
- Model only needs to output action IDs
- No need to learn rules explicitly

## 🔧 Advanced Configuration

### Memory Optimization

For limited GPU memory:

```json
// train_config.json
{
  "model": {
    "use_4bit": true,  // Enable 4-bit quantization
    "use_nested_quant": true  // Double quantization
  },
  "ppo": {
    "batch_size": 4,  // Reduce batch size
    "gradient_accumulation_steps": 8  // Increase accumulation
  }
}
```

### Hyperparameter Tuning

**Learning rate:**
- Start: `1e-5` (default)
- Too high: Policy collapses, high KL divergence
- Too low: Slow convergence

**LoRA rank:**
- Low (r=8): Faster, less capacity
- Medium (r=16): Balanced (default)
- High (r=32): Slower, more capacity

**PPO epochs:**
- Low (2-3): Faster, less stable
- Medium (4): Balanced (default)
- High (5-8): Slower, more stable

### Multi-GPU Training

```bash
# Use accelerate for distributed training
accelerate config

# Launch training
accelerate launch train_ppo_lora.py
```

## 📈 Expected Performance

### Training Timeline

- **Steps 0-1000**: Learn basic game mechanics (30-40% success)
- **Steps 1000-3000**: Improve strategy (50-60% success)
- **Steps 3000-5000**: Master easy stage, advance to medium
- **Steps 5000-8000**: Learn medium stage tactics
- **Steps 8000-10000**: Fine-tune and advance to hard stage

### Resource Requirements

**GPU Memory:**
- 4-bit quantization: ~8-10 GB VRAM
- Full precision: ~18-22 GB VRAM

**Training Time:**
- ~30 sec/step on A100 (40GB)
- ~10,000 steps ≈ 80-100 hours

**Storage:**
- Checkpoints: ~500 MB each (LoRA adapters only)
- Full training: ~20-30 GB with logs

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
# Reduce max_seq_length
# Enable gradient checkpointing
# Use 4-bit quantization
```

### Environment Connection Issues

```bash
# Check API key
echo $CHUTES_API_KEY

# Verify environment is running
python -c "import affine as af; env = af.GAME(mode='basilica'); print('OK')"

# Check affinetes_hosts.json configuration
```

### Poor Performance / Low Rewards

- **Curriculum too fast**: Increase `progression_threshold`
- **Learning rate too high**: Reduce to `5e-6` or `1e-6`
- **Insufficient exploration**: Increase `temperature` to 0.8-1.0
- **KL divergence issues**: Adjust `init_kl_coef` or `target_kl`

### Training Instability

- Increase `ppo_epochs` to 5-6
- Add gradient clipping
- Reduce learning rate
- Check for NaN values in logs

## 📚 Additional Resources

### Documentation

- [TRL Documentation](https://huggingface.co/docs/trl/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [OpenSpiel Environment README](../affinetes/environments/openspiel/README.md)

### Papers

- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Curriculum Learning**: [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)

### Examples

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint ./checkpoints/game_ppo_lora/final

# Export LoRA adapters
python scripts/export.py --checkpoint ./checkpoints/game_ppo_lora/final --output ./adapters

# Merge adapters with base model
python scripts/merge.py --base Qwen/Qwen2.5-3B-Instruct --adapters ./adapters --output ./merged_model
```

## 🤝 Contributing

For questions or issues:
1. Check existing GitHub issues
2. Review training logs and wandb metrics
3. Verify configuration files
4. Test with minimal config first

## 📄 License

This project follows the same license as the parent repository.

---

## 🎯 Key Takeaways

1. **Start Simple**: Begin with default config, adjust based on performance
2. **Monitor Closely**: Watch wandb metrics for instabilities
3. **Be Patient**: RL training takes time, expect 80-100 hours
4. **Save Often**: Checkpoints every 500 steps prevent data loss
5. **Trust the Process**: Curriculum + failure replay = steady improvement

**Happy Training! 🚀**
